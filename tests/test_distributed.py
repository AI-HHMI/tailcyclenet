"""The world-size axis: what a config key means on N gpus, and the guards around it.

None of this needs a process group; the parts that genuinely need NCCL are guarded at RUNTIME by
`check_ranks_agree`. THE ONE RULE WORTH REMEMBERING: DDP registers parameters when the module is
WRAPPED and never re-checks, so a tensor frozen at wrap time is never all-reduced afterwards --
this repo ships a staged encoder unfreeze, which is exactly that case.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from tailcyclenet import distributed as dist_utils
from tailcyclenet.dataset import _reader_cache_size
from tailcyclenet.optim import group_lr
from tailcyclenet.unfreeze import apply_staged_unfreeze

REPO = Path(__file__).resolve().parent.parent


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_module():
    return _module(REPO / 'scripts' / 'train.py', 'tcn_train')


def _optim_tests():
    """`Tiny` reproduces upstream's staged-unfreeze contract verbatim; reuse it rather than
    keeping a second copy of a gate whose whole point is that it must not drift."""
    return _module(Path(__file__).parent / 'test_optim.py', 'tcn_test_optim')


# -- what a count means on N gpus ----------------------------------------------------------

def test_per_rank_is_a_ceiling_with_a_floor_of_one():
    """`n_iterations` and every frequency are TOTALS across ranks: ceiling (an 8-gpu run does at
    least the requested number of samples), floor 1 (a frequency rounded to 0 would fire every step).
    """
    assert dist_utils.per_rank(60000, 4) == 15000
    assert dist_utils.per_rank(60001, 4) == 15001
    assert dist_utils.per_rank(3, 8) == 1
    assert dist_utils.per_rank(0, 8) == 1
    assert dist_utils.ceil_div(0, 8) == 0, 'the resume position may legitimately be 0'
    assert dist_utils.ceil_div(9, 4) == 3


def test_world_one_leaves_every_schedule_exactly_as_configured():
    """The claim that `--devices 1` is the loop it always was, as arithmetic."""
    for x in (60000, 200, 1000, 20, 1, 0):
        assert dist_utils.per_rank(x, 1) == max(1, x)
    assert dist_utils.ceil_div(37, 1) == 37


def test_the_unfreeze_iteration_cannot_be_stepped_over():
    """The global iteration advances by `world`, so it lands ON 10,000 only when world divides it;
    the gate must be `>=`, or a 3-gpu run sails past and the encoder never trains, silently.
    """
    Tiny = _optim_tests().Tiny
    for world in range(1, 9):
        m, fired = Tiny(unfreeze_at=10000, n_last=3), []
        for step in range(0, 20000 // world + 2):
            if m.unfreeze_video_encoder(step * world):
                fired.append(step * world)
        assert len(fired) == 1, f'world {world}: expected exactly one unfreeze, got {fired}'
        assert 10000 <= fired[0] < 10000 + world, f'world {world}: fired at {fired[0]}'


# -- the learning rate ---------------------------------------------------------------------

def test_scaling_touches_only_the_absolute_rates():
    """`encoder_lr_scale` and `muon_lr_scale` are MULTIPLIERS on the absolute rates, so scaling
    them too would move the recipe's ratios and make a multi-gpu arm several levers off its control.
    """
    cfg = {'learning_rate': 1e-4, 'kpt_lr': 5e-4, 'encoder_lr_scale': 0.1, 'muon_lr_scale': 2.0,
           'weight_decay': 0.002, 'optimizer': 'muon'}
    s = dist_utils.scale_optimizer_cfg(cfg, 4)
    assert s['learning_rate'] == pytest.approx(2e-4)
    assert s['kpt_lr'] == pytest.approx(1e-3)
    for k in ('encoder_lr_scale', 'muon_lr_scale', 'weight_decay', 'optimizer'):
        assert s[k] == cfg[k]
    assert cfg['learning_rate'] == 1e-4, 'the caller\'s dict must not be mutated: the run folder '\
                                         'records the CONFIGURED rate, or a resume re-scales it'


def test_world_one_scaling_is_the_identity():
    cfg = {'learning_rate': 1e-4, 'kpt_lr': 5e-4}
    assert dist_utils.scale_optimizer_cfg(cfg, 1) == cfg


def test_an_absent_kpt_lr_stays_absent_and_still_follows_the_scaled_rate():
    """`build_muon` reads `cfg.get('kpt_lr', lr)`, so an absent key must not be materialised at the
    UNSCALED value -- that would silently pin the fresh 2D matrices to the 1-gpu rate."""
    s = dist_utils.scale_optimizer_cfg({'learning_rate': 1e-4}, 9)
    assert 'kpt_lr' not in s
    assert group_lr('muon_fresh', s) == pytest.approx(3e-4)


def test_the_staged_unfreeze_reads_the_scaled_rates():
    """The unfreeze adds the encoder's groups thousands of iterations after the build, reading the
    config dict again -- handing the scaled copy to the build alone would give the encoder a
    1-gpu rate half a run later.
    """
    tests = _optim_tests()
    m = tests.Tiny(unfreeze_at=10, n_last=3)
    cfg = dist_utils.scale_optimizer_cfg(tests._cfg(learning_rate=1e-4, encoder_lr_scale=0.1), 4)
    from tailcyclenet.optim import build_muon
    opt = build_muon(m, fresh=set(), cfg=cfg)
    assert apply_staged_unfreeze(m, opt, cfg, 10, fresh=set())
    lrs = [g['lr'] for g in opt.opt_muon.param_groups]
    assert pytest.approx(1e-4 * 2 * 0.1) in lrs, f'no encoder group at the scaled rate: {lrs}'


def test_train_hands_the_scaled_config_to_both_unfreeze_entry_points():
    """The build, the live unfreeze and the resume replay must all read ONE dict. Checked on the
    source because the alternative -- noticing at iteration 10,000 of a 60,000-iteration job -- is
    the failure this test exists to prevent."""
    src = (REPO / 'scripts' / 'train.py').read_text()
    for call in ('build_optimizer(model, fresh, opt_cfg_scaled)',
                 'apply_staged_unfreeze(raw, opt, opt_cfg_scaled',
                 'replay_staged_unfreeze(model, opt, opt_cfg_scaled'):
        assert call in src, f'{call!r} is not what scripts/train.py calls'


# -- host resources ------------------------------------------------------------------------

def test_reader_cache_divides_by_ranks_as_well_as_workers(monkeypatch):
    """Four ranks of eight workers is 32 decoding processes on one host, not 8 -- sizing the cache
    as if it were one rank is how a job that fits on one gpu gets OOM-killed on four.
    """
    monkeypatch.delenv('TAILCYCLENET_READER_CACHE', raising=False)
    monkeypatch.delenv('TAILCYCLENET_LOCAL_WORLD_SIZE', raising=False)
    kw = dict(n_cams=16, wh=(1024, 570), workers=8, ram_gb=500.0)
    one = _reader_cache_size(**kw, procs=1)
    assert _reader_cache_size(**kw, procs=4) <= one
    assert _reader_cache_size(n_cams=16, wh=(3208, 2200), workers=8, ram_gb=64.0, procs=8) >= 1, \
        'the clamp must never return 0: a cache of 0 misses on every call'


def test_absent_local_world_size_reproduces_the_single_process_numbers(monkeypatch):
    """A one-rank run, an inference pass and every test that predates this must be unchanged."""
    monkeypatch.delenv('TAILCYCLENET_READER_CACHE', raising=False)
    monkeypatch.delenv('TAILCYCLENET_LOCAL_WORLD_SIZE', raising=False)
    for cams, wh, workers, ram in ((16, (1024, 570), 8, 500.0), (4, (4696, 2048), None, 500.0),
                                   (16, (3208, 2200), 12, 250.0)):
        assert (_reader_cache_size(cams, wh, workers, ram)
                == _reader_cache_size(cams, wh, workers, ram, procs=1))


def test_local_world_size_is_read_from_the_environment(monkeypatch):
    """`scripts/train.py` sets it from `fabric.world_size` right after launch. Read, never probed."""
    monkeypatch.delenv('TAILCYCLENET_READER_CACHE', raising=False)
    monkeypatch.setenv('TAILCYCLENET_LOCAL_WORLD_SIZE', '8')
    kw = dict(n_cams=16, wh=(1024, 570), workers=8, ram_gb=500.0)
    assert _reader_cache_size(**kw) == _reader_cache_size(**kw, procs=8)


def test_ranks_do_not_replay_one_anothers_sampler_stream():
    """Every rank runs the same `torch.manual_seed(seed)`, so the sampler and workers' seeds must
    be decorrelated explicitly or all N ranks draw the same windows.
    """
    ds = list(range(100))

    def draw(rank):
        g = torch.Generator().manual_seed(23 + rank)
        return list(torch.utils.data.RandomSampler(ds, replacement=True, num_samples=16,
                                                   generator=g))

    assert draw(0) != draw(1)
    assert draw(3) == draw(3), 'the same rank must still be reproducible'


# -- the straggler fix: equal COST per rank, per step ---------------------------------------

CGROUP = 4          # `_item`'s field order: views, coords, vis, frames, cgroup, ...


def _ds(root, **over):
    """`prob_2d_only = 0` unless a test says otherwise: it defaults to 0.25, which collapses a 3D
    item to ONE camera and would otherwise supply the variation these tests are attributing to
    `cams_to_sample`."""
    from tailcyclenet.dataset import LoaderConfig, PoseDataset
    return PoseDataset(root, 'train', LoaderConfig(**{'prob_2d_only': 0.0, **over}))


def test_the_sampler_yields_an_ordinal_that_is_the_step_number():
    """The ordinal must be the position in the STREAM, not the index: it is the only thing every
    rank agrees on, and it is what the shape draw is keyed to."""
    from tailcyclenet.dataset import StepSampler
    g0 = torch.Generator().manual_seed(23)
    g1 = torch.Generator().manual_seed(24)
    a = list(StepSampler(100, num_samples=8, generator=g0))
    b = list(StepSampler(100, num_samples=8, generator=g1))
    assert [k for k, _ in a] == [k for k, _ in b] == list(range(8)), 'ordinals must agree'
    assert [i for _, i in a] != [i for _, i in b], 'the INDEX stream must stay per-rank'
    assert len(StepSampler(100, num_samples=8, generator=g0)) == 8


def test_two_ranks_draw_the_same_camera_count_for_the_same_step(tiny_root):
    """THE FIX. A DDP step costs the slowest rank's item, so a per-rank camera-count draw made
    almost every step an 8-camera step. Two datasets stand in for two ranks -- separate objects,
    same ordinals.
    """
    r0, r1 = _ds(tiny_root / 'mouselike', cams_to_sample=[1, 3]), \
        _ds(tiny_root / 'mouselike', cams_to_sample=[1, 3])
    seen = set()
    for ordinal in range(12):
        a = r0[(ordinal, ordinal % len(r0))]
        b = r1[(ordinal, (ordinal + 1) % len(r1))]       # a DIFFERENT window on the other rank
        assert len(a[CGROUP]) == len(b[CGROUP]), \
            f'step {ordinal}: {len(a[CGROUP])} cameras vs {len(b[CGROUP])}'
        seen.add(len(a[CGROUP]))
    assert len(seen) > 1, 'the draw must still VARY across steps, only not across ranks'


def test_a_retry_does_not_desynchronise_the_shape(tiny_root):
    """An item that fails to build re-picks its index inside `__getitem__`; if the shape were
    redrawn on the retry, one rank would consume a draw its peers did not and they'd skew for the
    rest of the run.
    """
    ds = _ds(tiny_root / 'mouselike', cams_to_sample=[1, 3])
    calls = []
    real_shape = ds._shape
    ds._shape = lambda rng: (calls.append(1), real_shape(rng))[1]
    ds[(7, 0)]
    assert len(calls) == 1, f'_shape drawn {len(calls)} times for one item'


def test_camera_count_distribution_is_unchanged(tiny_root):
    """Synchronising the draw must not RESHAPE it. The marginal over steps is what the recipe
    means by `cams_to_sample`, and it has to survive, or this is a training change wearing a
    performance fix's clothes."""
    ds = _ds(tiny_root / 'mouselike', cams_to_sample=[1, 3])
    counts = [ds._shape(np.random.default_rng((23, 0x5AFE, k)))['n_cams'] for k in range(3000)]
    share = {c: counts.count(c) / len(counts) for c in (1, 2, 3)}
    assert all(abs(v - 1 / 3) < 0.03 for v in share.values()), share


def test_the_single_view_coin_is_also_synchronised(tiny_root):
    """`prob_2d_only` collapses a 3D item to ONE camera, so it is a cost draw too -- a rank that
    took the single-view branch while its peers encoded 8 cameras is the same straggler in
    reverse. Its default is 0.25, so this fires on a quarter of steps in any 3D recipe that leaves
    it alone."""
    r0 = _ds(tiny_root / 'mouselike', cams_to_sample=3, prob_2d_only=0.5)
    r1 = _ds(tiny_root / 'mouselike', cams_to_sample=3, prob_2d_only=0.5)
    seen = set()
    for ordinal in range(16):
        a, b = r0[(ordinal, 0)], r1[(ordinal, 0)]
        assert len(a[CGROUP]) == len(b[CGROUP]), f'step {ordinal}: single-view draw disagreed'
        seen.add(len(a[CGROUP]))
    assert seen == {1, 3}, f'both branches must still occur across steps, saw {seen}'


def test_a_bare_index_still_works(tiny_root):
    """Val and test address the index directly (`Subset`, `linspace`), and every test written
    before the ordinal existed passes a plain int. That path must keep drawing its own shape."""
    ds = _ds(tiny_root / 'mouselike', cams_to_sample=2)
    assert len(ds[0][CGROUP]) == 2


# -- the drift guard -----------------------------------------------------------------------

def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


# -- save_checkpoint's eval/train toggle must run on EVERY rank ----------------------------
#
# Found on the FIRST real 4-gpu job: drift 0.0e+00 at the checkpoint boundary right after
# resuming, 2.9e-07 -- and rising -- at the very next one. `save_checkpoint` was called
# `if is0 else None`, so its internal schedule-free eval/train toggle ran only on rank 0 -- and
# that round trip is not float-exact, so rank 0's raw training parameters came out of every
# boundary perturbed relative to the untouched ranks.


def test_the_schedulefree_eval_train_round_trip_is_not_float_exact():
    """THE MECHANISM. `eval()` and `train()` are two DIFFERENT lerps relying on algebra to land
    back on the original `p`, and float32 does not guarantee that.
    """
    from schedulefree import AdamWScheduleFree
    torch.manual_seed(0)
    model = nn.Linear(64, 64)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
    opt.train()
    # 50 steps, not 3: `z` and `y` need to have genuinely diverged for the round trip to show any
    # difference at all -- at a handful of steps `p.lerp_` on nearly-equal `p`/`z` rounds to
    # exactly 0 diff, which would make this test pass for the wrong reason.
    for _ in range(50):
        opt.zero_grad()
        model(torch.randn(8, 64)).sum().backward()
        opt.step()
    before = model.weight.clone()
    opt.eval()
    opt.train()
    assert not torch.equal(model.weight, before), \
        'if this round trip were exact the asymmetry below could not matter'


@pytest.mark.filterwarnings('ignore')
def test_save_checkpoint_write_false_still_tolls_the_toggle_but_touches_no_disk(tmp_path):
    """`write=False` is not an optimisation -- it is what keeps a non-writing rank's raw
    parameters in step with rank 0's. It must run the toggle and must not create the checkpoints
    directory, matching the call `scripts/train.py` makes on every non-zero rank.
    """
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.checkpoints import save_checkpoint

    torch.manual_seed(0)
    model = nn.Linear(64, 64)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
    opt.train()
    for _ in range(50):                # see the divergence note above
        opt.zero_grad()
        model(torch.randn(4, 64)).sum().backward()
        opt.step()
    before = model.weight.clone()
    run = tmp_path / 'run'
    p = save_checkpoint(run, 10, model, opt, {'model': {}}, write=False)
    assert p is None
    assert not run.exists(), 'write=False must create nothing on disk'
    assert not torch.equal(model.weight, before), \
        'the toggle must still have run -- this is the whole point of write=False'


@pytest.mark.filterwarnings('ignore')
def test_two_ranks_writing_asymmetrically_is_exactly_the_bug_this_fixes(tmp_path):
    """Reproduces the real failure on two REAL `AdamWScheduleFree` copies with IDENTICAL state --
    two synthetic ranks -- then shows the old call pattern (`if is0: save_checkpoint(...)`)
    diverges them and the new one (`save_checkpoint(..., write=is0)`, called by both) does not.
    """
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.checkpoints import save_checkpoint

    def _rank():
        torch.manual_seed(0)
        model = nn.Linear(64, 64)
        opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
        opt.train()
        g = torch.Generator().manual_seed(0)
        for _ in range(50):             # see the divergence note in the test above
            opt.zero_grad()
            model(torch.randn(4, 64, generator=g)).sum().backward()
            opt.step()
        return model, opt

    m0_old, o0_old = _rank()
    m1_old, o1_old = _rank()
    assert dist_utils.signature_drift(dist_utils.param_signature(m0_old),
                                      dist_utils.param_signature(m1_old)) == 0.0

    # THE OLD PATTERN: only "rank 0" calls save_checkpoint at all.
    save_checkpoint(tmp_path / 'old', 10, m0_old, o0_old, {'model': {}})
    old_drift = dist_utils.signature_drift(dist_utils.param_signature(m0_old),
                                           dist_utils.param_signature(m1_old))
    assert old_drift > 1e-9, 'the bug must reproduce, or this test proves nothing'

    m0_new, o0_new = _rank()
    m1_new, o1_new = _rank()
    # THE FIX: both call it, only "rank 0" writes.
    save_checkpoint(tmp_path / 'new', 10, m0_new, o0_new, {'model': {}}, write=True)
    save_checkpoint(tmp_path / 'new', 10, m1_new, o1_new, {'model': {}}, write=False)
    new_drift = dist_utils.signature_drift(dist_utils.param_signature(m0_new),
                                           dist_utils.param_signature(m1_new))
    assert new_drift == 0.0, f'the fix must leave both ranks bit-identical, got {new_drift:.3e}'


def test_a_non_averaged_optimizer_is_unaffected_either_way(tmp_path):
    """No `eval()`/`train()` means no toggle and no mechanism for this bug -- a plain optimizer
    (or `muon_schedulefree = false`) must behave identically under both call patterns."""
    from tailcyclenet.checkpoints import save_checkpoint

    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
    before = model.weight.clone()
    save_checkpoint(tmp_path / 'run', 5, model, opt, {'model': {}}, write=False)
    assert torch.equal(model.weight, before)


def test_the_signature_sees_a_single_changed_tensor():
    """This is the whole guard: two ranks that stopped all-reducing one tensor drift, and nothing
    else in the run notices. Two float64 scalars per tensor is enough to catch it and small enough
    to broadcast at every checkpoint boundary."""
    a, b = _tiny(), _tiny()
    sa, sb = dist_utils.param_signature(a), dist_utils.param_signature(b)
    assert dist_utils.signature_drift(sa, sb) == 0.0, 'identical weights must read exactly 0'
    with torch.no_grad():
        b[0].weight[0, 0] += 1e-3
    assert dist_utils.signature_drift(dist_utils.param_signature(b), sa) > 1e-9


def test_the_drift_is_relative_PER_ENTRY_not_to_the_largest_tensor():
    """THE BUG THIS TEST EXISTS FOR, found by running the negative control: normalising by the
    largest signature entry lets a real divergence in a small tensor read 5.6e-10 and pass. One
    tensor's sum of squares can dwarf another's by orders of magnitude, so the comparison has to be
    per entry or it only ever watches the biggest tensor."""
    ref = torch.tensor([1e6, 1e-3], dtype=torch.float64)
    sig = torch.tensor([1e6, 1.1e-3], dtype=torch.float64)     # 10% off, in the small entry
    assert dist_utils.signature_drift(sig, ref) == pytest.approx(0.1, rel=1e-6)


def test_the_tolerance_sits_between_the_two_measured_populations():
    """THE TOLERANCE IS AN EMPIRICAL BOUNDARY: a correct run reads exactly 0.0, the one real bug
    reads 2.9e-07 to 4.9e-07. `tol` must pass the first and catch the second; 1e-3 is ~2,000x
    ABOVE the real bug's signature and would sail past it.
    """
    import inspect
    tol = inspect.signature(dist_utils.check_ranks_agree).parameters['tol'].default
    assert tol > 0.0, 'exact equality would be brittle against an unobserved noise floor'
    assert tol < 2.9e-07 / 100, (
        f'tol={tol:g} is too close to the smallest REAL divergence ever measured (2.9e-07); the '
        'guard needs at least two orders of margin below it to catch the bug early')


def test_a_non_finite_parameter_is_a_failure_not_agreement():
    """`NaN > tol` is False, so a poisoned weight would otherwise pass the comparison as
    agreement -- the same class of trap as a NaN loss read as a small one."""
    ref = torch.tensor([1.0], dtype=torch.float64)
    assert not (dist_utils.signature_drift(torch.tensor([float('nan')], dtype=torch.float64),
                                           ref) <= 1e-12)


def test_the_signature_covers_the_trainable_params_only():
    """A frozen tensor is identical on every rank by construction -- and including it would make
    the guard pass for exactly the tensors it exists to watch, since the failure it detects is a
    tensor that WAS frozen when the module was wrapped."""
    m = _tiny()
    full = dist_utils.param_signature(m).numel()
    m[0].weight.requires_grad_(False)
    assert dist_utils.param_signature(m).numel() == full - 2


def test_the_reductions_are_no_ops_without_a_world():
    """Every collective helper has to be callable at world 1, or the single-gpu path forks."""
    assert dist_utils.all_ranks_finite(None, True) is True
    assert dist_utils.all_ranks_finite(None, False) is False
    assert dist_utils.all_ranks_mean(None, 2.5) == 2.5
    assert dist_utils.gather_metrics(None, [{'mpjpe': 1.0}]) == [{'mpjpe': 1.0}]
    assert dist_utils.check_ranks_agree(None, _tiny()) == 0.0
    dist_utils.check_registry(None, ('nose', 'tail'))


# -- the CLI refusals ----------------------------------------------------------------------

def _args(**over):
    base = dict(device='cuda:0', devices=1, strategy=None, precision='32-true')
    base.update(over)
    return SimpleNamespace(**base)


def test_16_mixed_is_refused_by_name():
    """It needs a GradScaler on the optimizer, and this loop steps its own optimizer rather than
    handing it to Fabric. Refused loudly instead of silently training without the scaler.
    """
    with pytest.raises(SystemExit, match='GradScaler'):
        _train_module().launch(_args(precision='16-mixed'))


def test_device_cpu_still_means_cpu():
    """`--device cpu` is how a smoke test runs on a box whose gpus are busy, and it predates
    `--devices`. An accelerator chosen from `cuda.is_available()` alone would overrule it."""
    fabric = _train_module().launch(_args(device='cpu'))
    assert str(fabric.device) == 'cpu'


def test_one_device_and_many_devices_together_are_refused():
    """`--device cuda:1 --devices 4` would silently ignore the first: at more than one device the
    launcher places every rank."""
    with pytest.raises(SystemExit, match='--devices'):
        _train_module().launch(_args(device='cuda:1', devices=4))


def test_devices_minus_one_resolves_to_the_visible_count(monkeypatch):
    """`--devices -1` (the default) means "every visible GPU", so it must resolve to the visible
    count BEFORE the strategy is chosen: with ONE visible GPU it takes the exact `--devices 1`
    path (strategy auto, no DDP wrapper), and `--device cpu` resolves to the one-process CPU
    path rather than tripping the multi-device refusal."""
    import lightning.fabric as lf

    seen = {}

    def spy(self, *a, **k):
        seen.update(k)
        raise SystemExit('stop-before-fabric-launch')

    monkeypatch.setattr(lf.Fabric, '__init__', spy)
    tr = _train_module()

    monkeypatch.setattr('torch.cuda.device_count', lambda: 1)
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    with pytest.raises(SystemExit, match='stop-before-fabric-launch'):
        tr.launch(_args(devices=-1))
    assert seen['devices'] == [0] and seen['strategy'] == 'auto' \
        and seen['accelerator'] == 'gpu', 'one visible GPU must be the --devices 1 path'

    seen.clear()
    monkeypatch.setattr('torch.cuda.device_count', lambda: 4)
    with pytest.raises(SystemExit, match='stop-before-fabric-launch'):
        tr.launch(_args(devices=-1))
    assert seen['devices'] == 4 and seen['strategy'] == 'ddp_find_unused_parameters_true', \
        'many visible GPUs must stay on the DDP path'

    seen.clear()
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    with pytest.raises(SystemExit, match='stop-before-fabric-launch'):
        tr.launch(_args(devices=-1))
    assert seen['devices'] == 1 and seen['strategy'] == 'auto' \
        and seen['accelerator'] == 'cpu', 'no CUDA must resolve to the one-process CPU path'

    seen.clear()
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    with pytest.raises(SystemExit, match='stop-before-fabric-launch'):
        tr.launch(_args(devices=-1, device='cpu'))
    assert seen['devices'] == 1 and seen['strategy'] == 'auto' \
        and seen['accelerator'] == 'cpu', '--device cpu must not trip the multi-device refusal'


# -- the two handles -----------------------------------------------------------------------

def test_run_batch_gives_the_loss_the_unwrapped_module():
    """`TotalLoss` reads `model.training` and `model.stride_overlap`; the forward goes through the
    WRAPPED module (that all-reduces gradients) and the loss through the raw one.
    """
    tr = _train_module()
    seen = {}
    wrapped = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}    # noqa: E731
    raw = object()

    def spy(model, out, **kw):
        seen['model'] = model
        return torch.zeros((), requires_grad=False)

    batch = SimpleNamespace(
        views=[torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)],
        cgroup=[{'size': torch.tensor([4, 4])}], sample_info={'mode': '3d'},
        kpt_ids=torch.zeros(1, 2, dtype=torch.long), kpt_prior=torch.zeros(1, 2, 3),
        prompt_t=torch.zeros(1, 2, dtype=torch.int32), coords=torch.zeros(1, 2, 2, 3),
        vis=None, vis_2d=None, p2d=None)
    tr.run_batch(wrapped, spy, batch, 'cpu', raw=raw)
    assert seen['model'] is raw
    tr.run_batch(wrapped, spy, batch, 'cpu')
    assert seen['model'] is wrapped, 'raw=None must leave the single-gpu path unchanged'
