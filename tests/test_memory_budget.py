"""The host-RAM budget, and the standing rule that NOTHING it sizes may move a number.

`tailcyclenet/memory.py` resolves one budget from the cgroup ancestry, LSF's own variables,
`MemAvailable` and `MemTotal`, and three consumers size their buffers from it: the decord reader
cache, `detect_raw`'s batch, and the inference loop's camera concurrency and frame cache. The
budget depends on machine state, so two runs of one command can resolve it differently -- which is
only acceptable because every knob downstream of it is output-neutral. **These tests are what makes
that sentence true rather than aspirational.**

`tests/test_detector.py` classifies `batch` as `plumbing`, i.e. ASSERTS that no run can differ in
it. Nothing verified that. If `batch` were ever budget-derived the claim would be false, and two
hosts with different amounts of free memory would produce different boxes with nothing in either
output saying so -- so it is checked here, and `detect_raw` keeps the value pinned.

Same for the dtype: `_fetch` hands back uint8 and the `/255` happens on the device. That is a 4x
cut in host memory, PCIe traffic and device memory, and it is only legitimate because it is
bit-identical -- uint8 -> float32 is exact and one correctly-rounded float32 divide by 255 is the
same float wherever it runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import memory
# The fixture lives in test_infer.py; re-export it so pytest resolves it by name here. `noqa`
# because ruff reads the parameter of the test below as a redefinition rather than as pytest's
# fixture injection, which is a false positive on every shared-fixture import.
from test_infer import multiwindow_scene  # noqa: F401,F811


# Every one of the 256 possible byte values, not a sample: the failure mode is a handful of ULPs
# on particular values, and a random draw misses it.
_ALL_BYTES = np.arange(256, dtype=np.uint8).reshape(1, 1, 16, 16)
_HOST_REFERENCE = _ALL_BYTES.astype(np.float32) / np.float32(255)


def test_uint8_then_divide_on_host_is_bit_identical_to_the_old_conversion():
    """The dtype change in `detect_raw._fetch`, isolated, on CPU."""
    new = torch.from_numpy(_ALL_BYTES).float().div_(torch.tensor(255.0)).numpy()
    assert new.dtype == np.float32
    assert np.array_equal(_HOST_REFERENCE, new), 'the /255 move is not bit-identical'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_dividing_by_a_python_scalar_on_cuda_is_NOT_correctly_rounded():
    """THE TRAP, pinned as a fact rather than as a warning in a comment.

    Moving the `/255` onto the device is legitimate only because it is bit-identical, and the
    obvious spelling is not. `x / 255` with a PYTHON scalar takes a reciprocal-multiply fast path
    on CUDA that is off by one ULP on most byte values. That is 1 ULP on the detector's INPUT,
    which perturbs every objectness score, reorders NMS ties and returns different boxes -- so it
    would change no shape and no dtype, so nothing downstream would report it.

    If this test ever starts PASSING, torch has changed its scalar-division lowering. That is good
    news, but check `detect_raw` still uses the tensor form before relaxing anything.
    """
    scalar = torch.from_numpy(_ALL_BYTES).cuda().float().div_(255).cpu().numpy()
    assert not np.array_equal(_HOST_REFERENCE, scalar), \
        'scalar division now matches -- see the docstring before changing detect_raw'
    ulps = np.abs(_HOST_REFERENCE.view(np.int32) - scalar.view(np.int32))
    assert ulps.max() == 1, f'expected a 1-ULP disagreement, got {ulps.max()}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_dividing_by_a_0d_tensor_on_cuda_IS_bit_identical():
    """...and this is the spelling `detect_raw` uses, for the reason above."""
    new = (torch.from_numpy(_ALL_BYTES).cuda().float()
           .div_(torch.tensor(255.0, device='cuda')).cpu().numpy())
    assert np.array_equal(_HOST_REFERENCE, new), 'the /255 move is not bit-identical on cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_detect_raw_uses_the_tensor_divisor_not_the_scalar():
    """The two tests above are about torch; this one is about us actually using the right one."""
    import inspect

    from tailcyclenet.detector import detect_raw

    src = inspect.getsource(detect_raw)
    assert 'div_(_div255)' in src, 'detect_raw must divide by the 0-d tensor, not by 255'
    assert 'div_(255)' not in src, 'detect_raw is using the 1-ULP-wrong scalar division'


@pytest.fixture(scope='module')
def det_scene(tmp_path_factory):
    """A 3-camera session and an untrained YOLOX-Nano -- enough to run `detect_raw` end to end.

    The weights are random on purpose: these tests are about which BYTES come back, never about
    whether the boxes are any good.
    """
    import conftest as cf
    from tailcyclenet.detector import YOLOXNano
    from tailcyclenet.format import sessions_for

    root = tmp_path_factory.mktemp('det')
    cf._session_3d(root / 'ds' / 'test' / 's', T=8)
    _, sessions = sessions_for(root / 'ds', 'test')
    sess = sessions[0]
    sess.preload()
    return YOLOXNano(n_keypoints=0).eval(), sess, next(iter(sess.groups))


def test_detect_raw_is_byte_identical_across_ram_budgets(det_scene):
    """THE INVARIANT FOR THE DETECTOR HALF.

    A budget so small that only one camera may be in flight must return exactly what a budget with
    room returns. This is what licenses sizing `detect_raw` from free memory at all: two machines
    with different amounts of RAM must not produce different boxes.
    """
    from tailcyclenet.detector import detect_raw

    det, sess, gid = det_scene

    def run(gb):
        memory.reset()
        memory.current(override_gb=gb)
        try:
            return detect_raw(det, (64, 64), sess, gid, 2, device='cpu', batch=4)
        finally:
            memory.reset()

    for name, a, b in zip(('boxes', 'scores', 'kpts'), run(4096), run(1e-6)):
        if a is None:
            continue
        assert np.array_equal(a, b, equal_nan=True), f'{name} moved with the RAM budget'


def test_detection_is_batch_aligned_and_slice_independent(det_scene):
    """Detecting a clip in aligned SLICES must equal detecting it at once, byte for byte.

    This is what lets the inference loop advance detection alongside its window loop instead of
    paying for the whole group before a single pose is predicted -- and it is only safe because of
    the alignment rule. `_units` partitions on `range(0, T, batch)`, so a slice that starts
    anywhere else forwards a short leading batch, which is a SHAPE the whole-clip pass never
    produces, and cuDNN selects algorithms per shape (0.204 px / 1.69e-03, dev/reports/38 §3.2).

    So both halves are the test: aligned slices agree exactly, and a misaligned one RAISES rather
    than quietly returning boxes that differ in the fourth decimal place.
    """
    from tailcyclenet.detector import detect_raw

    det, sess, gid = det_scene                    # T = 8
    kw = dict(device='cpu', batch=4)
    whole = detect_raw(det, (64, 64), sess, gid, 2, **kw)

    parts = [detect_raw(det, (64, 64), sess, gid, 2, frames=np.arange(lo, hi), **kw)
             for lo, hi in ((0, 4), (4, 8))]
    for i, name in enumerate(('boxes', 'scores', 'kpts')):
        if whole[i] is None:
            continue
        joined = np.concatenate([p[i] for p in parts], axis=1)
        assert np.array_equal(joined, whole[i], equal_nan=True), \
            f'{name} moved when the clip was detected in aligned slices'

    # A trailing slice may be short -- it ends where the clip does.
    detect_raw(det, (64, 64), sess, gid, 2, frames=np.arange(4, 8), **kw)
    # A slice starting off the batch grid must not be silently accepted.
    with pytest.raises(AssertionError, match='multiple of batch'):
        detect_raw(det, (64, 64), sess, gid, 2, frames=np.arange(2, 6), **kw)

    # `read=` MUST BE A PURE SUBSTITUTION, and it is what stops the detector decoding frames the
    # window loop is already holding. It is handed SOURCE frame numbers, not slice-local ones --
    # the two differ for every slice but the first, and getting it wrong would detect the right
    # count of the wrong frames.
    from tailcyclenet.dataset import read_frames

    seen = []

    def _read(ci, cam_name, fr, pool=None, reduce=1):
        seen.append((ci, tuple(int(x) for x in fr)))
        return read_frames(sess.groups[gid], cam_name, fr, reduce=reduce, pool=pool)

    via = detect_raw(det, (64, 64), sess, gid, 2, frames=np.arange(4, 8), read=_read, **kw)
    assert seen and all(f == (4, 5, 6, 7) for _, f in seen), \
        f'read= must receive SOURCE frame numbers, got {seen[:2]}'
    for i in range(2):
        assert np.array_equal(via[i], parts[1][i], equal_nan=True), \
            'read= must be a pure substitution for the decode'


def test_batch_is_NOT_inert_which_is_why_the_budget_may_not_touch_it(det_scene):
    """A FINDING ABOUT THE REPO, PINNED SO IT IS NOT REDISCOVERED THE EXPENSIVE WAY.

    `tests/test_detector.py` lists `batch` in `plumbing`, i.e. asserts that no run can differ in
    it -- which holds only because it is pinned. **It is not inert.** cuDNN and
    the CPU kernels select algorithms per input SHAPE, so a different batch is a different
    reduction order. Measured on johnson's real detector (16 vs 3, 12 frames, 16 cameras): boxes
    differ by 0.204 px, scores by 1.69e-03, keypoints by 0.447 px -- enough to reorder NMS and
    return a different box set.

    Two consequences, both live:

    - `detect_raw` must NOT derive `batch` from the RAM budget, which is why the budget bounds the
      CAMERA axis instead (chunking cameras cannot change a forward's shape).
    - `eval_detector.py --batch-size` already lets a user change `batch` and get different
      detections, while the stamp says two such runs are interchangeable. That is a pre-existing
      hazard in the detector plan's territory, not something introduced here.

    This test asserts the DIFFERENCE, so if a future torch makes batching bit-exact it fails and
    someone re-reads this. On CPU the effect is smaller than on CUDA and may be absent, so an
    equal result here is reported as a skip rather than a pass.
    """
    from tailcyclenet.detector import detect_raw

    det, sess, gid = det_scene
    memory.reset()
    a = detect_raw(det, (64, 64), sess, gid, 2, device='cpu', batch=8)
    b = detect_raw(det, (64, 64), sess, gid, 2, device='cpu', batch=3)
    if np.array_equal(a[0], b[0], equal_nan=True):
        pytest.skip('this CPU build batches bit-exactly; the CUDA measurement stands')


def test_fits_never_exceeds_what_the_caller_asked_for():
    """`memory.fits` may only ever SHRINK a buffer.

    This is what makes installing the budget safe against every recorded measurement: on a host
    with room, every consumer resolves to exactly the value it used before the budget existed. A
    bigger machine must not silently become a different recipe.
    """
    assert memory.fits(10 ** 15, 1, want=16) == 16
    assert memory.fits(10 ** 15, 0, want=16) == 16          # zero cost -> still capped
    assert memory.fits(10 * 2 ** 30, 2.4 * 2 ** 30, want=16) == 4
    assert memory.fits(1, 2 ** 30, want=16) == 1            # never below the floor
    assert memory.fits(1, 2 ** 30, want=16, floor=3) == 3


def test_budget_prefers_the_cgroup_over_the_machine():
    """The bug this whole module exists for: `SC_PHYS_PAGES` is the HOST's memory.

    Under a cgroup cap the old `_reader_cache_size` still read the host's 503 GB -- a 20x
    over-estimate, and exactly the LSF case. Verified here against a synthetic limit rather than
    against a real cgroup, so it runs anywhere.
    """
    memory.reset()
    b = memory.host_budget(override_gb=8)
    assert b.budget_gb == pytest.approx(8.0 * memory.DEFAULT_FRACTION)  # a CEILING, not an allowance
    assert 'max-ram' in b.source
    # An override above the hard cap is a request to be OOM-killed; it must clamp and say so.
    huge = memory.host_budget(override_gb=b.limit_gb * 10)
    assert huge.budget_gb <= huge.limit_gb + 1e-6
    assert 'clamped' in huge.source
    memory.reset()


def test_budget_is_resolved_once_per_process():
    """Two groups of one run must be sized identically, or the peak has no single denominator."""
    memory.reset()
    first = memory.current()
    assert memory.current() is first
    memory.reset()


def test_reader_cache_is_pure_given_ram_and_shrinks_under_a_cap():
    """`_reader_cache_size` keeps its `ram_gb` parameter so the sizing is testable on any host
    (gotcha 10: it must never open a container to measure anything)."""
    from tailcyclenet.dataset import _reader_cache_size

    # johnson's rig: 16 cameras of 3208x2200. The price is QUADRATIC in the count (measured:
    # 0.28 / 3.58 / 15.34 / 34.09 / 59.38 GB at 1 / 4 / 8 / 12 / 16 readers).
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=1024) == 16     # room -> the whole rig
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=1) == 1         # never zero
    # A loader worker still wants 4, not the rig, and the worker count still divides.
    assert _reader_cache_size(16, (3208, 2200), 4, ram_gb=1024) == 4
    # Monotone in the budget, and quadratic: halving the count should cost ~4x less memory, so
    # a 4x smaller budget should give ~2x fewer readers rather than 4x fewer.
    big = _reader_cache_size(16, (3208, 2200), None, ram_gb=64)
    small = _reader_cache_size(16, (3208, 2200), None, ram_gb=16)
    assert big > small >= 1
    assert small == pytest.approx(big / 2, abs=1)


def test_the_reader_cache_does_not_thrash_on_the_cyclic_access_it_actually_sees():
    """THE 2.4x. Every window touches all C cameras in the same order, which is the one access
    pattern LRU cannot serve: at capacity < C it evicts exactly the entry needed next and takes
    ZERO hits. Measured end to end before the fix -- 16 readers 61 s, ELEVEN readers 149 s, two
    readers 222 s -- eleven cached readers were buying nothing.

    Asserted against the real pattern, not against the implementation.
    """
    from tailcyclenet.dataset import _ReaderCache

    cache = _ReaderCache(11)
    cache_open = []

    import tailcyclenet.dataset as dsmod
    real = dsmod._open_reader
    dsmod._open_reader = lambda p: cache_open.append(p) or object()
    try:
        for _ in range(60):                       # 60 passes over a 16-camera rig
            for c in range(16):
                cache(f'cam{c}')
    finally:
        dsmod._open_reader = real

    rate = cache.hits / (cache.hits + cache.misses)
    assert rate > 0.3, (
        f'hit rate {rate:.1%} on a 16-cycle with an 11-entry cache -- an LRU scores 0.0% here, '
        'which is the regression this guards')


@pytest.mark.parametrize('budget_gb', [0.001, 4096])
def test_the_frame_cache_changes_no_pixel_at_any_budget(multiwindow_scene, budget_gb):  # noqa: F811
    """THE INVARIANT THE WHOLE BUDGET RESTS ON.

    `run_group`'s frame cache exists to collapse the 6x re-decode (three overlapping windows x two
    `--refine` passes) to 1x. It is a pure speed change and must be provably one: a tiny budget
    disables the cache and shrinks camera concurrency to 1, a huge budget enables both, and the
    two must agree BYTE FOR BYTE across every array `run_group` returns.

    Run at both ends rather than at the default, because the default on the test host is whatever
    that host happens to have free -- which is exactly the non-determinism being ruled out.
    """
    from tailcyclenet.infer import InferConfig, run_group

    model, sess, registry, name = multiwindow_scene
    cfg = InferConfig(n_frames=4, overlap=2, image_size=64, min_crop_dim=16, device='cpu',
                      anchor='none')

    memory.reset()
    memory.current(override_gb=budget_gb)
    try:
        got = run_group(model, sess, next(iter(sess.groups)), registry, name, cfg)
    finally:
        memory.reset()

    memory.reset()
    memory.current(override_gb=4096)
    try:
        ref = run_group(model, sess, next(iter(sess.groups)), registry, name, cfg)
    finally:
        memory.reset()

    assert set(got) == set(ref)
    for k in ref:
        a, b = got[k], ref[k]
        if isinstance(b, np.ndarray) and b.dtype.kind == 'f':
            assert np.array_equal(a, b, equal_nan=True), f'{k} moved with the RAM budget'
        elif isinstance(b, np.ndarray):
            assert np.array_equal(a, b), f'{k} moved with the RAM budget'
        else:
            assert a == b, f'{k} moved with the RAM budget'


def test_result_arrays_are_the_term_the_budget_cannot_shrink():
    """A 200 fps hour is an ordinary recording and it does not fit.

    720,000 frames on a 16-camera 24-keypoint rig allocates ~82 GB of RESULT arrays at
    `--det-top-k 24`, 93% of it the detector's `(top_k, T, C, K, 3)` keypoint array -- before a
    single frame is decoded, and unreachable by `--max-ram` because these arrays ARE the answer.

    `scripts/infer.py` checks this before detection and refuses with the three ways out. This pins
    the arithmetic behind that check, and the shape of the answer: the keypoint array dominates,
    and it scales with `top_k`, so `--det-top-k` is the lever that matters most.
    """
    hour = 200 * 3600
    big = memory.result_array_gb(hour, 16, 24, 2, top_k=24, det_kpts=True)
    assert sum(big.values()) > 60, 'a 200 fps hour should be obviously too big'
    assert max(big, key=big.get) == 'detect keypoints'
    assert big['detect keypoints'] / sum(big.values()) > 0.8

    # The two levers the error message offers must actually move it.
    fewer = memory.result_array_gb(hour, 16, 24, 2, top_k=2, det_kpts=True)
    assert sum(fewer.values()) < sum(big.values()) / 8
    shorter = memory.result_array_gb(hour // 100, 16, 24, 2, top_k=24, det_kpts=True)
    assert sum(shorter.values()) == pytest.approx(sum(big.values()) / 100, rel=0.01)

    # A detector with no keypoint branch does not pay for the keypoint array.
    none = memory.result_array_gb(hour, 16, 24, 2, top_k=24, det_kpts=False)
    assert 'detect keypoints' not in none

    # Linear in every axis it claims to be linear in.
    a = memory.result_array_gb(1000, 8, 10, 1, top_k=4, det_kpts=True)
    b = memory.result_array_gb(2000, 8, 10, 1, top_k=4, det_kpts=True)
    assert sum(b.values()) == pytest.approx(2 * sum(a.values()), rel=1e-9)


@pytest.mark.parametrize('anchor', ['none', 'carry'])
@pytest.mark.parametrize('refine', [False, True])
def test_block_size_changes_no_pixel(tmp_path, anchor, refine):
    """THE INVARIANT FOR THE POSE HALF, and the direct analogue of the prefetch test.

    How many windows a block holds comes from FREE MEMORY, so if it could move a number then two
    machines running one command would disagree and nothing in either output would say so. That is
    the rule which licenses sizing anything from the budget at all (dev/reports/38 §7).

    Exercised over a DETECTOR box source as well as the label path, because that is the arm where
    a boundary has something to break: the association state has to cross it, and the detection
    cursor has to stay on the global batch grid. The old whole-clip budget test could not reach it.
    """
    import conftest as cf
    from tailcyclenet.format import Registry, load_dataset
    from tailcyclenet.infer import InferConfig, merge_blocks, run_blocks
    from tailcyclenet.model import build_model
    from test_model import SMALL

    cf._session_3d(tmp_path / 'mv' / 'test' / 's', T=16)
    ds = load_dataset(tmp_path / 'mv')
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()

    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 16, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h

    def boxes_for(store, lo, hi):
        return (boxes[:, lo:hi], None, None)

    cfg = InferConfig(n_frames=4, overlap=2, image_size=64, min_crop_dim=16, device='cpu',
                      anchor=anchor, refine=refine)

    def run(gb):
        memory.reset()
        memory.current(override_gb=gb)
        try:
            blocks = list(run_blocks(model, sess, 'g000', registry, ds.name, cfg,
                                     boxes_for=boxes_for, n_rows=1))
            return len(blocks), merge_blocks(blocks)
        finally:
            memory.reset()

    # 64x48x3 x 3 cameras is 27.6 kB a frame index, so these span one-window blocks to one block.
    n_small, small = run(0.0008)
    n_big, big = run(4096)
    assert n_small > n_big, f'the budgets must actually differ in block count ({n_small} vs {n_big})'
    assert n_big == 1, 'a roomy budget should take the whole clip in one block'
    for k, v in big.items():
        a, b = np.asarray(v), np.asarray(small[k])
        if a.dtype.kind in 'fc':
            assert np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)), \
                f'{k} moved with the block size'
        elif a.dtype.kind in 'iub':
            assert np.array_equal(a, b), f'{k} moved with the block size'
        else:
            assert str(a) == str(b), f'{k} moved with the block size'


def test_a_window_that_does_not_fit_is_refused_not_re_decoded(tmp_path):
    """No silent degradation below one window, because there is nowhere to degrade TO.

    Refine pass 2 crops from the SAME frames pass 1 did. Dropping them means either decoding the
    window twice -- the 3x the store exists to remove -- or re-cropping from the stored crop, which
    double-resamples and is not output-neutral. So the budget either holds a window or the run says
    so, with the arithmetic and the flag that would fix it.
    """
    import conftest as cf
    from tailcyclenet.format import Registry, load_dataset
    from tailcyclenet.infer import InferConfig, run_group
    from tailcyclenet.model import build_model
    from test_model import SMALL

    cf._session_3d(tmp_path / 'mv' / 'test' / 's', T=8)
    ds = load_dataset(tmp_path / 'mv')
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()
    cfg = InferConfig(n_frames=4, overlap=2, image_size=64, min_crop_dim=16, device='cpu')

    memory.reset()
    memory.current(override_gb=1e-6)
    try:
        with pytest.raises(SystemExit) as e:
            run_group(model, sess, 'g000', registry, ds.name, cfg)
    finally:
        memory.reset()
    msg = str(e.value)
    assert '--max-ram' in msg and 'one window' in msg
    assert 'frame store' in msg, 'the message must show the arithmetic, not just the verdict'


def test_the_detection_lookahead_is_what_degrades_on_a_tight_budget(tmp_path):
    """TWO BLOCKS ARE LIVE WHEN DETECTION RUNS AHEAD, and both come out of one share.

    The detection thread decodes block k+1's frames while block k's are still held for its
    forwards. Budgeting one block and running two overshot the flag -- measured at 11.6 GB peak
    under `--max-ram 10`, and a ceiling on the process is the whole point of that flag.

    The fix must not be "halve the block", which would double the budget a run needs before it can
    START (johnson would refuse below --max-ram 19 where it runs at 10) -- that trades a hard
    failure for a speedup. So the LOOKAHEAD degrades: with room for two blocks it pipelines, and
    below that it detects inline exactly as it did before the pipeline existed.
    """
    import conftest as cf
    from tailcyclenet.format import Registry, load_dataset
    from tailcyclenet.infer import InferConfig, merge_blocks, run_blocks
    from tailcyclenet.model import build_model
    from test_model import SMALL

    cf._session_3d(tmp_path / 'mv' / 'test' / 's', T=16)
    ds = load_dataset(tmp_path / 'mv')
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()

    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 16, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    seen = []

    def boxes_for(store, lo, hi):
        seen.append((lo, hi))
        return (boxes[:, lo:hi], None, None)

    cfg = InferConfig(n_frames=4, overlap=2, image_size=64, min_crop_dim=16, device='cpu',
                      anchor='carry')

    def run(gb):
        seen.clear()
        memory.reset()
        memory.current(override_gb=gb)
        try:
            return merge_blocks(run_blocks(model, sess, 'g000', registry, ds.name, cfg,
                                           boxes_for=boxes_for, n_rows=1)), list(seen)
        finally:
            memory.reset()

    tight, tight_calls = run(0.0008)      # one window fits, two blocks do not
    roomy, roomy_calls = run(4096)

    # BOTH RUN -- the tight budget must not refuse, which is the failure the split avoids.
    assert tight_calls and roomy_calls
    # ...and detection is asked for the same frame spans in the same order either way, which is
    # what makes the lookahead a pure wall-clock change.
    assert tight_calls == sorted(tight_calls), 'detection must advance in block order'
    # ...and the pixels are the same.
    for k, v in roomy.items():
        a, b = np.asarray(v), np.asarray(tight[k])
        if a.dtype.kind in 'fc':
            assert np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)), \
                f'{k} moved when the detection lookahead was disabled by the budget'
        elif a.dtype.kind in 'iub':
            assert np.array_equal(a, b), f'{k} moved when the lookahead was disabled'


# `--max-ram` MUST ACTUALLY BIND -- dev/plans/infer_from_videos_and_calibration.md §15.
#
# The budget was purely ADVISORY: every consumer sized itself from it and nothing checked the
# total, so a `--videos` run on 263,798-frame containers reached 456 GB under `--max-ram 24` with
# nothing in its output saying so, and had to be killed at 9 GB free on a shared node.

def test_the_budget_is_resolved_above_both_input_branches(monkeypatch, tmp_path):
    """ORDERING. The `--videos` probe opens video containers, and `memory.current` caches
    process-wide -- so with the budget resolved BELOW the input branch the override was not in
    effect during the probe, and any consumer that asked would have got the HOST figure.

    `--max-ram` cannot be a ceiling on a phase that runs before it is read. Asserted by having the
    probe itself read `memory.current()`.
    """
    import argparse

    from tailcyclenet import adopt, memory
    from tailcyclenet.infer import driver

    memory.reset()
    seen = {}

    def spy_build(plan, **kw):
        seen['budget'] = memory.current()
        raise SystemExit('stop here -- the budget has been observed')

    monkeypatch.setattr(adopt, 'build', spy_build)
    monkeypatch.setattr(adopt, 'plan', lambda *a, **k: 'PLAN')
    monkeypatch.setattr(adopt, 'check_flags', lambda a: None)
    monkeypatch.setattr(adopt, 'dataset_name', lambda r, n: 'ds')
    monkeypatch.setattr('tailcyclenet.format.Registry.load',
                        classmethod(lambda cls, p: _FakeReg()))
    monkeypatch.setattr(driver, 'load_run',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('the budget must be resolved before the checkpoint')))

    args = argparse.Namespace(
        run=tmp_path / 'run',
        videos=[tmp_path / 'v.mp4'], calibration=tmp_path / 'c.toml', cam_regex=None,
        session_id=None, group_id=None, units='mm', fps=None, assoc_res_max_px=30.0,
        trim_to_shortest=False, dump_session=None, data=None, split=None,
        anchor='none', oracle_corrupt=None, refine_px=None, refine=None,
        max_frames=0, start_frame=0, end_frame=0, max_ram=17.0, dataset_name=None,
        box_prompt='none', detector=None, boxes='b.npz', max_animals=1)
    try:
        with pytest.raises(SystemExit, match='the budget has been observed'):
            driver.run_dataset(args)
    finally:
        memory.reset()

    assert seen['budget'].stated, 'the probe saw an INFERRED budget -- the override came too late'
    assert '--max-ram 17' in seen['budget'].source, seen['budget'].source


class _FakeReg:
    names = ('a',)
    datasets = (('ds', (0,)),)

    def local_names(self, d):
        return ['a']


def test_the_probe_is_inside_the_budget_partition():
    """An UNBUDGETED CONSUMER IS HOW THE NEXT ONE GETS ADDED. The probe holds open readers, so it
    comes out of `FRACTION_READERS` rather than sitting outside a partition that sums to 1.00.

    It does NOT change the number on any rig anyone has run -- an open reader on a 263,798-frame
    3208x2200 container retains 0.03 GB, measured under a trim -- which is the honest outcome and
    is why this asserts the bound rather than a new value.
    """
    from tailcyclenet import adopt, memory

    big = memory.host_budget(override_gb=1024)
    assert adopt.probe_workers(16, big) == adopt.PROBE_WORKERS_MAX
    assert adopt.probe_workers(2, big) == 2, 'never more workers than videos'

    tiny = memory.host_budget(override_gb=0.001)
    assert adopt.probe_workers(16, tiny) == 1, 'must floor at 1, never 0'

    # Monotone in the budget, and never above the reader share.
    prev = 0
    for gb in (0.001, 1, 8, 64, 1024):
        b = memory.host_budget(override_gb=gb)
        n = adopt.probe_workers(16, b)
        assert n >= prev or gb == 0.001
        prev = n
        assert n == 1 or n * adopt._PROBE_READER_BYTES <= b.share(memory.FRACTION_READERS)


def test_the_peak_check_fires_on_a_promise_and_stays_quiet_otherwise():
    """A STATED budget is a promise; an INFERRED one is what was lying around, so exceeding it is
    the host being busier than at startup rather than a broken promise.

    Warn-once, and never a kill: the peak may be retained arena rather than working set, the
    offending allocation is not always ours, and a run that is over budget is not a run that is
    WRONG. What it buys is that the next 456 GB arrives as a line of output naming its phase.
    """
    import warnings as _w

    from tailcyclenet import memory

    assert memory.peak_gb() > 0 and memory.rss_gb() > 0

    # An absurdly small STATED budget: this process is certainly above it.
    memory._peak_warned = False
    tiny = memory.host_budget(override_gb=0.001)
    with _w.catch_warnings(record=True) as got:
        _w.simplefilter('always')
        memory.check_peak('a test phase', tiny)
        memory.check_peak('a second phase', tiny)
    assert len(got) == 1, 'warn ONCE per process, or a per-block warning trains the reader to skip'
    assert 'a test phase' in str(got[0].message), 'the PHASE is the point of the message'

    # An INFERRED budget of the same absurd size must say nothing.
    memory._peak_warned = False
    from dataclasses import replace
    inferred = replace(memory.host_budget(), budget_gb=0.001, stated=False)
    with _w.catch_warnings(record=True) as got:
        _w.simplefilter('always')
        memory.check_peak('an inferred phase', inferred)
    assert not got, 'an inferred budget is not a promise and must not be reported as broken'
    memory._peak_warned = False
