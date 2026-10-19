"""The host-RAM budget (`tailcyclenet/memory.py`), and the standing rule that NOTHING it sizes
may move a number: the budget depends on machine state, which is acceptable only because every
knob downstream of it is output-neutral. These tests are what makes that true.
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


@pytest.fixture(scope='module')
def mv16(tmp_path_factory):
    """(model, session, registry, name): a static 3-camera 3D session, T=16 (seven windows)."""
    import conftest as cf
    from tailcyclenet.format import Registry, load_dataset
    from tailcyclenet.pose.model import build_model
    from test_model import SMALL

    root = tmp_path_factory.mktemp('mv16')
    cf._session_3d(root / 'mv' / 'test' / 's', T=16)
    ds = load_dataset(root / 'mv')
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    return build_model(SMALL, n_keypoints=registry.n_keypoints).eval(), sess, registry, ds.name


def _whole_frame_boxes(sess, T=16):
    """One detector-style row whose box is the whole frame in every camera."""
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, T, len(sess.rig), 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    return boxes


def _assert_same(a_out, b_out, what):
    """Every array of two merged outputs is identical (NaN-aware)."""
    assert set(a_out) == set(b_out)
    for k, v in b_out.items():
        a, b = np.asarray(a_out[k]), np.asarray(v)
        if b.dtype.kind in 'fc':
            assert np.array_equal(a, b, equal_nan=True), f'{k} moved with {what}'
        elif b.dtype.kind in 'iub':
            assert np.array_equal(a, b), f'{k} moved with {what}'
        else:
            assert str(a) == str(b), f'{k} moved with {what}'


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
def test_cuda_divides_bytes_exactly_only_by_a_0d_tensor():
    """`x / 255` with a PYTHON scalar takes a reciprocal-multiply path on CUDA that is 1 ULP off on
    most bytes (enough to move detector boxes); dividing by a 0-d tensor, the spelling
    `detect_raw` uses, is bit-identical to the host. If the first half fails, torch changed."""
    x = torch.from_numpy(_ALL_BYTES).cuda().float()
    scalar = x.clone().div_(255).cpu().numpy()
    ulps = np.abs(_HOST_REFERENCE.view(np.int32) - scalar.view(np.int32))
    assert ulps.max() == 1, f'expected a 1-ULP disagreement, got {ulps.max()}'
    exact = x.clone().div_(torch.tensor(255.0, device='cuda')).cpu().numpy()
    assert np.array_equal(_HOST_REFERENCE, exact), 'the /255 move is not bit-identical on cuda'


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
    return YOLOXNano(version='hybrid').eval(), sess, next(iter(sess.groups))


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

    for name, a, b in zip(('boxes', 'scores'), run(4096), run(1e-6)):
        if a is None:
            continue
        assert np.array_equal(a, b, equal_nan=True), f'{name} moved with the RAM budget'


def test_detection_is_batch_aligned_and_slice_independent(det_scene):
    """Detecting a clip in aligned SLICES must equal detecting it at once, byte for byte -- what
    lets the loop advance detection alongside its window loop. A misaligned slice RAISES rather
    than quietly returning boxes that differ (cuDNN picks algorithms per shape).
    """
    from tailcyclenet.detector import detect_raw

    det, sess, gid = det_scene                    # T = 8
    kw = dict(device='cpu', batch=4)
    whole = detect_raw(det, (64, 64), sess, gid, 2, **kw)

    parts = [detect_raw(det, (64, 64), sess, gid, 2, frames=np.arange(lo, hi), **kw)
             for lo, hi in ((0, 4), (4, 8))]
    for i, name in enumerate(('boxes', 'scores')):
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
    """The bug this whole module exists for: `SC_PHYS_PAGES` is the HOST's memory, so under a
    cgroup cap the old sizing read the host's 503 GB -- a 20x over-estimate, and exactly the LSF
    case. Verified against a synthetic limit so it runs anywhere.
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
    (it must never open a container to measure anything).
    """
    from tailcyclenet.dataset import _reader_cache_size

    # johnson's rig: 16 cameras of 3208x2200. The price is LINEAR in the count on PyAV; the
    # quadratic law this test used to assert came from the old decoder, measured without a trim on an
    # unconstrained host (i.e. it measured allocator arena).
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=1024) == 16     # room -> the whole rig
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=1) == 1         # never zero
    # A loader worker still wants 4, not the rig, and the worker count still divides.
    assert _reader_cache_size(16, (3208, 2200), 4, ram_gb=1024) == 4
    # Monotone in the budget, and LINEAR: a 4x smaller budget gives ~4x fewer readers. Asserted
    # below the rig at both ends, or the `want` cap hides the shape.
    big = _reader_cache_size(64, (3208, 2200), None, ram_gb=64)
    small = _reader_cache_size(64, (3208, 2200), None, ram_gb=16)
    assert big > small >= 1
    assert small == pytest.approx(big / 4, rel=0.25)


def test_the_reader_cache_does_not_thrash_on_the_cyclic_access_it_actually_sees():
    """THE 2.4x. Every window touches all C cameras in the same order, which is the one access
    pattern LRU cannot serve: at capacity < C it evicts exactly the entry needed next and takes
    ZERO hits. Asserted against the real pattern, not against the implementation.
    """
    from tailcyclenet.dataset import _ReaderCache

    cache = _ReaderCache(11)
    cache_open = []

    import tailcyclenet.video as vmod
    real = vmod.open_reader
    vmod.open_reader = lambda p: cache_open.append(p) or object()
    try:
        for _ in range(60):                       # 60 passes over a 16-camera rig
            for c in range(16):
                cache(f'cam{c}')
    finally:
        vmod.open_reader = real

    rate = cache.hits / (cache.hits + cache.misses)
    assert rate > 0.3, (
        f'hit rate {rate:.1%} on a 16-cycle with an 11-entry cache -- an LRU scores 0.0% here, '
        'which is the regression this guards')


def test_the_frame_cache_and_block_size_change_no_pixel(mv16):
    """THE INVARIANT THE WHOLE BUDGET RESTS ON: a tiny budget (no frame cache, one camera in
    flight, one-window blocks, no detection lookahead) and a huge one agree BYTE FOR BYTE, on the
    label path and over a detector box source with carry + refine (where a block boundary has
    association state and a detection cursor to break)."""
    from tailcyclenet.infer.window import InferConfig, merge_blocks, run_blocks

    model, sess, registry, name = mv16
    boxes = _whole_frame_boxes(sess)

    seen = []

    def boxes_for(store, lo, hi):
        seen.append((lo, hi))
        return (boxes[:, lo:hi], None)

    def run(gb, **kw):
        cfg = InferConfig(n_frames=4, overlap=2, image_size=64, min_crop_dim=16, device='cpu',
                          **kw)
        memory.reset()
        memory.current(override_gb=gb)
        try:
            blocks = list(run_blocks(model, sess, 'g000', registry, name, cfg,
                                     **({'boxes_for': boxes_for, 'n_rows': 1} if kw.get('refine')
                                        else {})))
            return len(blocks), merge_blocks(blocks)
        finally:
            memory.reset()

    _assert_same(run(0.001, anchor='none')[1], run(4096, anchor='none')[1], 'the RAM budget')
    n_small, small = run(0.0008, anchor='carry', refine=True)
    tight_calls = list(seen)
    seen.clear()
    n_big, big = run(4096, anchor='carry', refine=True)
    assert n_small > n_big == 1, f'the budgets must differ in block count ({n_small} vs {n_big})'
    assert tight_calls and seen, 'the tight budget disables the detection lookahead, not the run'
    assert tight_calls == sorted(tight_calls), 'detection must advance in block order'
    _assert_same(small, big, 'the block size')


def test_a_window_that_does_not_fit_is_refused_not_re_decoded(mv16):
    """Below one window there is nowhere to degrade to (pass 2 crops pass 1's frames; re-decoding
    or re-cropping is not output-neutral), so the run refuses, naming the arithmetic and the flag."""
    from tailcyclenet.infer.window import InferConfig, run_group

    model, sess, registry, name = mv16
    cfg = InferConfig(n_frames=4, overlap=2, image_size=64, min_crop_dim=16, device='cpu')
    memory.reset()
    memory.current(override_gb=1e-6)
    try:
        with pytest.raises(SystemExit) as e:
            run_group(model, sess, 'g000', registry, name, cfg)
    finally:
        memory.reset()
    msg = str(e.value)
    assert '--max-ram' in msg and 'one window' in msg
    assert 'frame store' in msg, 'the message must show the arithmetic, not just the verdict'


# `--max-ram` MUST ACTUALLY BIND.
#
# The budget was purely ADVISORY: every consumer sized itself from it and nothing checked the
# total, so a `--videos` run on 263,798-frame containers reached 456 GB under `--max-ram 24` with
# nothing in its output saying so.


def test_the_budget_is_resolved_above_both_input_branches(monkeypatch, tmp_path):
    """ORDERING. The `--videos` probe opens video containers, and `memory.current` caches
    process-wide -- so with the budget resolved BELOW the input branch the override was not in
    effect during the probe, and any consumer that asked would have got the HOST figure.
    """
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

    from tailcyclenet.infer.cli import build_parser

    args = build_parser().parse_args([
        '--run', str(tmp_path / 'run'), '--videos', str(tmp_path / 'v.mp4'),
        '--calibration', str(tmp_path / 'c.toml'), '--units', 'mm', '--anchor', 'none',
        '--max-ram', '17.0', '--box-prompt', 'none', '--boxes', 'b.npz',
        '--max-animals', '1', '--out', str(tmp_path / 'out')])
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
    """An UNBUDGETED CONSUMER IS HOW THE NEXT ONE GETS ADDED: the probe holds open readers, so it
    comes out of `FRACTION_READERS` rather than sitting outside a partition that sums to 1.00.
    It does NOT change the number on any rig anyone has run -- an open reader retains ~0.03 GB.
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
    the host being busier than at startup. Warn-once, never a kill: the peak may be retained
    arena, the offending allocation is not always ours, and over budget is not WRONG -- what it
    buys is that the next 456 GB arrives as a line of output naming its phase.
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


def test_plan_windows_refuses_or_plans_from_an_injected_budget():
    """Too small for one window: `SystemExit` naming the arithmetic. Roomy: the starts are exactly
    `window_starts`' and the blocks cover them."""
    from tailcyclenet.infer.window import _plan_windows, window_starts
    from tailcyclenet.memory import Budget

    tiny = Budget(limit_gb=0.01, available_gb=0.01, budget_gb=0.01, source='test', stated=True)
    with pytest.raises(SystemExit, match='does not fit, and there is no'):
        _plan_windows('sess', 'g000', [(3840, 2160)], n_frames=12, overlap=4,
                      T_total=100, frame_start=0, budget=tiny)

    roomy = Budget(limit_gb=64.0, available_gb=64.0, budget_gb=64.0, source='test', stated=True)
    starts, blocks, cam_decode, pipeline_det = _plan_windows(
        'sess', 'g000', [(64, 64)], n_frames=12, overlap=4, T_total=100, frame_start=0,
        budget=roomy)
    assert starts == window_starts(100, 12, 4, start=0)
    assert blocks and blocks[0][0] == 0 and blocks[-1][1] == len(starts)
    assert cam_decode >= 1 and isinstance(pipeline_det, bool)


def test_a_stated_budget_above_the_cap_gets_the_fraction_of_the_cap(monkeypatch):
    """Asking for MORE than the machine has must not hand buffers more than asking for exactly
    the cap: both clamp the PROCESS ceiling to the cap and give buffers the same fraction of it.
    """
    gb = memory.GB
    monkeypatch.setattr(memory, '_meminfo', lambda: {'MemTotal': 512 * gb, 'MemAvailable': 400 * gb})
    monkeypatch.setattr(memory, '_cgroup', lambda: (64 * gb, 0.0))
    monkeypatch.setattr(memory, '_lsf_limit', lambda: None)
    exact = memory.host_budget(64.0)
    over = memory.host_budget(100.0)
    assert over.budget_gb == pytest.approx(exact.budget_gb)
    assert over.budget_gb == pytest.approx(64.0 * memory.DEFAULT_FRACTION)
    assert over.ceiling == pytest.approx(64.0) and 'clamped' in over.source
    memory.reset()


def test_the_peak_check_measures_against_the_stated_figure_after_rebudget(monkeypatch):
    """`rebudget` subtracts the process floor from the buffers, not from the user's ceiling, so
    a peak below `--max-ram` must stay quiet and one above it must warn.
    """
    import warnings as _w

    gb = memory.GB
    monkeypatch.setattr(memory, '_meminfo', lambda: {'MemTotal': 512 * gb, 'MemAvailable': 400 * gb})
    monkeypatch.setattr(memory, '_cgroup', lambda: (None, 0.0))
    monkeypatch.setattr(memory, '_lsf_limit', lambda: None)
    monkeypatch.setattr(memory, 'rss_gb', lambda: 5.0)
    b = memory.rebudget(24.0)
    assert b.ceiling == pytest.approx(24.0)
    for peak, n_warn in ((23.0, 0), (25.0, 1)):
        memory._peak_warned = False
        monkeypatch.setattr(memory, 'peak_gb', lambda peak=peak: peak)
        with _w.catch_warnings(record=True) as got:
            _w.simplefilter('always')
            memory.check_peak('a phase', b)
        assert len(got) == n_warn, peak
    memory._peak_warned = False
    memory.reset()


def test_cgroup_v1_limit_is_read_from_the_memory_hierarchy(tmp_path):
    """On cgroup v1 the memory controller is mounted at `/sys/fs/cgroup/memory/<rel>` and its
    line is usually not the last one in `/proc/self/cgroup`.
    """
    proc = tmp_path / 'cgroup'
    proc.write_text('12:pids:/job\n4:memory:/lsf/job7\n1:name=systemd:/job\n')
    leaf = tmp_path / 'sys' / 'memory' / 'lsf' / 'job7'
    leaf.mkdir(parents=True)
    (leaf / 'memory.limit_in_bytes').write_text(str(8 << 30))
    (leaf / 'memory.usage_in_bytes').write_text(str(1 << 30))
    (tmp_path / 'sys' / 'memory' / 'memory.limit_in_bytes').write_text(str(1 << 63))
    assert memory._cgroup(proc, tmp_path / 'sys') == (float(8 << 30), float(1 << 30))

    proc.write_text('0::/user.slice/s.scope\n')
    v2 = tmp_path / 'sys' / 'user.slice'
    (v2 / 's.scope').mkdir(parents=True)
    (v2 / 'memory.max').write_text(str(4 << 30))
    (v2 / 's.scope' / 'memory.max').write_text('max')
    (v2 / 's.scope' / 'memory.current').write_text('7')
    assert memory._cgroup(proc, tmp_path / 'sys') == (float(4 << 30), 7.0)
