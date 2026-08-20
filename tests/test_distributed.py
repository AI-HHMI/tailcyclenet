"""The world-size axis: what a config key means on N gpus, and the guards around it.

None of this needs a process group. Every rule that decides how a run behaves on 4 gpus is a pure
function of ints or of tensors, and that is deliberate -- the parts that genuinely need NCCL (the
gradient all-reduce, the unfreeze re-wrap) are guarded at RUNTIME by `check_ranks_agree`, which
raises inside the real job rather than being approximated in a test.

THE ONE RULE WORTH REMEMBERING: DDP registers parameters when the module is WRAPPED and never
re-checks (`torch/nn/parallel/distributed.py:1338`), so a tensor frozen at wrap time is never
all-reduced afterwards. This repo ships a staged encoder unfreeze, which is exactly that case.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    """`n_iterations` and every frequency are TOTALS across ranks. Ceiling, so an 8-gpu run does
    at least the requested number of samples rather than 7 fewer; floor 1, because a frequency
    that rounded to 0 would fire on every step (or divide by zero)."""
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
    """The global iteration advances by `world`, so it lands ON 10,000 only when world divides it.

    Upstream's gate is `iteration < unfreeze_iter -> False`, i.e. `>=`, which is what makes a
    counter moving in strides of 8 safe. If it were `==`, a 3-gpu run would sail past 10,000 and
    the encoder would never train -- silently, and only on some gpu counts.
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
    them too would move the recipe's ratios and make a multi-gpu arm several levers off its
    control rather than one."""
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
    """The unfreeze adds the encoder's groups THOUSANDS of iterations after the build, reading the
    config dict again through `group_lr`. Handing the scaled copy to the build alone would give the
    encoder a 1-gpu rate half a run later, with nothing printed about it."""
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
    """Four ranks of eight workers is 32 decoding processes on one host, not 8. Sizing the cache
    as if it were one rank is how a job that fits on one gpu gets OOM-killed on four."""
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
    """`scripts/train.py` sets it from `fabric.world_size` right after launch. Read, never probed
    -- opening anything to measure this is gotcha 11."""
    monkeypatch.delenv('TAILCYCLENET_READER_CACHE', raising=False)
    monkeypatch.setenv('TAILCYCLENET_LOCAL_WORLD_SIZE', '8')
    kw = dict(n_cams=16, wh=(1024, 570), workers=8, ram_gb=500.0)
    assert _reader_cache_size(**kw) == _reader_cache_size(**kw, procs=8)


def test_ranks_do_not_replay_one_anothers_sampler_stream():
    """Every rank runs the same `torch.manual_seed(seed)` so the model inits identically; the
    sampler and the workers' base seed therefore have to be decorrelated explicitly, or all N ranks
    draw the same windows and the extra gpus buy variance reduction of zero."""
    ds = list(range(100))

    def draw(rank):
        g = torch.Generator().manual_seed(23 + rank)
        return list(torch.utils.data.RandomSampler(ds, replacement=True, num_samples=16,
                                                   generator=g))

    assert draw(0) != draw(1)
    assert draw(3) == draw(3), 'the same rank must still be reproducible'


# -- the drift guard -----------------------------------------------------------------------

def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


def test_the_signature_sees_a_single_changed_tensor():
    """This is the whole guard: two ranks that stopped all-reducing one tensor drift, and nothing
    else in the run notices. Two float64 scalars per tensor is enough to catch it and small enough
    to broadcast at every checkpoint boundary."""
    a, b = _tiny(), _tiny()
    sa, sb = dist_utils.param_signature(a), dist_utils.param_signature(b)
    assert dist_utils.signature_drift(sa, sb) == 0.0, 'identical weights must read exactly 0'
    with torch.no_grad():
        b[0].weight[0, 0] += 1e-3
    assert dist_utils.signature_drift(dist_utils.param_signature(b), sa) > 1e-12


def test_the_drift_is_relative_PER_ENTRY_not_to_the_largest_tensor():
    """THE BUG THIS TEST EXISTS FOR, found by running the negative control: normalising by the
    largest signature entry lets a real divergence in a small tensor read 5.6e-10 and pass. One
    tensor's sum of squares can dwarf another's by orders of magnitude, so the comparison has to be
    per entry or it only ever watches the biggest tensor."""
    ref = torch.tensor([1e6, 1e-3], dtype=torch.float64)
    sig = torch.tensor([1e6, 1.1e-3], dtype=torch.float64)     # 10% off, in the small entry
    assert dist_utils.signature_drift(sig, ref) == pytest.approx(0.1, rel=1e-6)


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
    handing it to Fabric. Refused loudly instead of silently training without the scaler."""
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


# -- the two handles -----------------------------------------------------------------------

def test_run_batch_gives_the_loss_the_unwrapped_module():
    """`TotalLoss` reads `model.training` and `model.stride_overlap`; the forward has to go through
    the WRAPPED module (that is what all-reduces the gradients) and the loss through the raw one."""
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
