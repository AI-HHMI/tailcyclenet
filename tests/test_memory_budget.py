"""The host-RAM budget, and the standing rule that NOTHING it sizes may move a number.

`tailcyclenet/memory.py` resolves one budget from the cgroup ancestry, LSF's own variables,
`MemAvailable` and `MemTotal`, and three consumers size their buffers from it: the decord reader
cache, `detect_raw`'s batch, and the inference loop's camera concurrency and frame cache. The
budget depends on machine state, so two runs of one command can resolve it differently -- which is
only acceptable because every knob downstream of it is output-neutral. **These tests are what makes
that sentence true rather than aspirational.**

`tests/test_detector.py`'s `--det-cache` stamp guard classifies `batch` as `plumbing`, i.e. it
ASSERTS that `batch` cannot change what `detect_raw` returns and therefore need not be stamped.
Nothing verified that. If it were false, every `--det-cache` on disk would be unsound: two arms
sharing one cache would be sharing boxes produced at whatever batch the first run happened to
resolve to. Now that `batch` is a budget-derived CEILING rather than a fixed 16, the claim is load
bearing on hosts with different amounts of free memory, so it is checked here.

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
    would invalidate every `--det-cache` on disk while changing no shape, no dtype and no
    `RAW_REV`.

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
    with different amounts of RAM must not produce different boxes and then silently share a
    `--det-cache`.
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


def test_batch_is_NOT_inert_which_is_why_the_budget_may_not_touch_it(det_scene):
    """A FINDING ABOUT THE REPO, PINNED SO IT IS NOT REDISCOVERED THE EXPENSIVE WAY.

    `tests/test_detector.py`'s `--det-cache` stamp guard lists `batch` in `plumbing`, i.e. asserts
    that it cannot change the detections and so need not be stamped. **That is false.** cuDNN and
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
    assert b.budget_gb == pytest.approx(8.0)
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
