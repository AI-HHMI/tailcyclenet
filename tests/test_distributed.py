"""The world-size axis: what a config key means on N gpus, and the guards around it.

No process group needed; what genuinely needs NCCL is guarded at runtime by `check_ranks_agree`.
DDP registers parameters at WRAP time and never re-checks, so a tensor frozen at wrap time is never
all-reduced afterwards -- exactly the staged encoder unfreeze's case.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import tailcyclenet.train as tr
from tailcyclenet import distributed as dist_utils
from tailcyclenet import runtime
from tailcyclenet.dataset import _reader_cache_size
from tailcyclenet.optim import build_muon, group_lr
from tailcyclenet.unfreeze import apply_staged_unfreeze

from tests.test_optim import Tiny, _cfg


# -- what a count means on N gpus ----------------------------------------------------------

def test_per_rank_is_a_ceiling_with_a_floor_of_one():
    """Every count is a TOTAL across ranks: ceiling (do at least the requested samples), floor 1
    (a frequency rounded to 0 fires every step); world 1 leaves every schedule as configured."""
    assert dist_utils.per_rank(60000, 4) == 15000
    assert dist_utils.per_rank(60001, 4) == 15001
    assert dist_utils.per_rank(3, 8) == 1
    assert dist_utils.per_rank(0, 8) == 1
    assert dist_utils.ceil_div(0, 8) == 0, 'the resume position may legitimately be 0'
    assert dist_utils.ceil_div(9, 4) == 3
    for x in (60000, 20, 1, 0):
        assert dist_utils.per_rank(x, 1) == max(1, x)


def test_the_unfreeze_iteration_cannot_be_stepped_over():
    """The global iteration advances by `world`; the gate must be `>=`, or a 3-gpu run sails past
    10,000 and the encoder never trains."""
    for world in (1, 3, 7):
        m, fired = Tiny(unfreeze_at=10000, n_last=3), []
        for step in range(0, 20000 // world + 2):
            if m.unfreeze_video_encoder(step * world):
                fired.append(step * world)
        assert len(fired) == 1, f'world {world}: expected exactly one unfreeze, got {fired}'
        assert 10000 <= fired[0] < 10000 + world, f'world {world}: fired at {fired[0]}'


def test_scaling_touches_only_the_absolute_rates():
    """`learning_rate`/`kpt_lr` scale by sqrt(world); multipliers do not; the caller's dict is not
    mutated (the run folder records the CONFIGURED rate); an absent `kpt_lr` stays absent but
    follows the scaled rate; world 1 is the identity."""
    cfg = {'learning_rate': 1e-4, 'kpt_lr': 5e-4, 'encoder_lr_scale': 0.1, 'muon_lr_scale': 2.0,
           'weight_decay': 0.002, 'optimizer': 'muon'}
    s = dist_utils.scale_optimizer_cfg(cfg, 4)
    assert s['learning_rate'] == pytest.approx(2e-4)
    assert s['kpt_lr'] == pytest.approx(1e-3)
    for k in ('encoder_lr_scale', 'muon_lr_scale', 'weight_decay', 'optimizer'):
        assert s[k] == cfg[k]
    assert cfg['learning_rate'] == 1e-4
    assert dist_utils.scale_optimizer_cfg(cfg, 1) == cfg

    s = dist_utils.scale_optimizer_cfg({'learning_rate': 1e-4}, 9)
    assert 'kpt_lr' not in s
    assert group_lr('muon_fresh', s) == pytest.approx(3e-4)


def test_the_live_unfreeze_and_the_resume_replay_read_the_scaled_config(tmp_path):
    """The build, the live unfreeze (thousands of steps later) and the resume replay all read ONE
    scaled dict, so the encoder group lands at the scaled rate on both paths."""
    from tailcyclenet import start
    from tailcyclenet.checkpoints import save_checkpoint
    from tailcyclenet.format import Registry

    cfg = dist_utils.scale_optimizer_cfg(_cfg(learning_rate=1e-4, encoder_lr_scale=0.1), 4)
    want = pytest.approx(1e-4 * 2 * 0.1)
    m = Tiny(unfreeze_at=10, n_last=3)
    opt = build_muon(m, fresh=set(), cfg=cfg)
    assert apply_staged_unfreeze(m, opt, cfg, 12, fresh=set())
    assert want in [g['lr'] for g in opt.opt_muon.param_groups]
    save_checkpoint(tmp_path / 'run', 12, m, opt, {'model': {}, 'data': {}},
                    registry=Registry(names=('a',), datasets=(('d', (0,)),)), world_size=4)
    got = start.build_start(start.plan_start(tmp_path / 'run', world=4),
                            lambda pretrained: Tiny(unfreeze_at=10, n_last=3), cfg)
    lrs = [g['lr'] for g in got.optimizer.opt_muon.param_groups]
    assert want in lrs, f'no encoder group at the scaled rate: {lrs}'


# -- host resources ------------------------------------------------------------------------

def test_reader_cache_divides_by_ranks_as_well_as_workers(monkeypatch):
    """Four ranks of eight workers is 32 decoding processes on one host. The local world size is
    READ from the environment (never probed); absent, it is one process; the clamp never hits 0."""
    monkeypatch.delenv('TAILCYCLENET_LOCAL_WORLD_SIZE', raising=False)
    kw = dict(n_cams=16, wh=(1024, 570), workers=8, ram_gb=500.0)
    one = _reader_cache_size(**kw, procs=1)
    assert _reader_cache_size(**kw) == one
    assert _reader_cache_size(**kw, procs=4) <= one
    assert _reader_cache_size(n_cams=16, wh=(3208, 2200), workers=8, ram_gb=64.0, procs=8) >= 1
    monkeypatch.setenv('TAILCYCLENET_LOCAL_WORLD_SIZE', '8')
    assert _reader_cache_size(**kw) == _reader_cache_size(**kw, procs=8)


# -- the straggler fix: equal COST per rank, per step ---------------------------------------

CGROUP = 4          # `_item`'s field order: views, coords, vis, frames, cgroup, ...


def _ds(root, **over):
    """`prob_2d_only = 0` unless stated, so camera-count variation comes from `cams_to_sample`."""
    from tailcyclenet.dataset import LoaderConfig, PoseDataset
    return PoseDataset(root, 'train', LoaderConfig(**{'prob_2d_only': 0.0, **over}))


def test_the_sampler_yields_an_ordinal_that_is_the_step_number():
    """The ordinal is the STREAM position (what every rank agrees on); the index stream stays
    per-rank."""
    from tailcyclenet.dataset import StepSampler
    a = list(StepSampler(100, num_samples=8, generator=torch.Generator().manual_seed(23)))
    b = list(StepSampler(100, num_samples=8, generator=torch.Generator().manual_seed(24)))
    assert [k for k, _ in a] == [k for k, _ in b] == list(range(8)), 'ordinals must agree'
    assert [i for _, i in a] != [i for _, i in b], 'the INDEX stream must stay per-rank'


@pytest.mark.parametrize('over, want', [
    (dict(cams_to_sample=[1, 3]), None),
    (dict(cams_to_sample=3, prob_2d_only=0.5), {1, 3}),
], ids=['cams_to_sample', 'prob_2d_only'])
def test_two_ranks_draw_the_same_cost_for_the_same_step(tiny_root, over, want):
    """A DDP step costs the slowest rank's item, so the camera count and the single-view coin are
    keyed to the ordinal: two datasets (two ranks) on DIFFERENT windows agree, yet still vary."""
    r0, r1 = _ds(tiny_root / 'mouselike', **over), _ds(tiny_root / 'mouselike', **over)
    seen = set()
    for ordinal in range(8):
        a = r0[(ordinal, ordinal % len(r0))]
        b = r1[(ordinal, (ordinal + 1) % len(r1))]
        assert len(a[CGROUP]) == len(b[CGROUP]), f'step {ordinal} disagreed'
        seen.add(len(a[CGROUP]))
    assert len(seen) > 1 if want is None else seen == want, seen


def test_a_retry_does_not_redraw_the_shape_and_a_bare_index_still_works(tiny_root):
    """A retried item must not consume an extra shape draw (ranks would skew for the rest of the
    run); val/test address the index directly and draw their own shape."""
    ds = _ds(tiny_root / 'mouselike', cams_to_sample=[1, 3])
    calls = []
    real_shape = ds.shape
    ds.shape = lambda rng: (calls.append(1), real_shape(rng))[1]
    ds[(7, 0)]
    assert len(calls) == 1, f'shape drawn {len(calls)} times for one item'
    assert len(_ds(tiny_root / 'mouselike', cams_to_sample=2)[0][CGROUP]) == 2


def test_camera_count_distribution_is_unchanged(tiny_root):
    """Synchronising the draw must not reshape its marginal over steps."""
    ds = _ds(tiny_root / 'mouselike', cams_to_sample=[1, 3])
    counts = [ds.shape(np.random.default_rng((23, 0x5AFE, k)))['n_cams'] for k in range(3000)]
    share = {c: counts.count(c) / len(counts) for c in (1, 2, 3)}
    assert all(abs(v - 1 / 3) < 0.03 for v in share.values()), share


# -- save_checkpoint's eval/train toggle must run on EVERY rank ----------------------------

@pytest.mark.filterwarnings('ignore')
def test_two_ranks_writing_asymmetrically_is_exactly_the_bug_this_fixes(tmp_path):
    """Found on the first real 4-gpu job: the schedule-free eval/train toggle inside
    `save_checkpoint` is not float-exact, so calling it on rank 0 only drifted the ranks.
    `save_checkpoint(..., write=is0)` on every rank keeps them bit-identical and writes nothing
    where `write=False`; a non-averaged optimizer is untouched.
    """
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.checkpoints import save_checkpoint

    def _rank():
        torch.manual_seed(0)
        model = nn.Linear(64, 64)
        opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
        opt.train()
        g = torch.Generator().manual_seed(0)
        for _ in range(50):        # at a handful of steps the round trip rounds to exactly 0
            opt.zero_grad()
            model(torch.randn(4, 64, generator=g)).sum().backward()
            opt.step()
        return model, opt

    def drift(a, b):
        return dist_utils.signature_drift(dist_utils.param_signature(a),
                                          dist_utils.param_signature(b))

    m0, o0 = _rank()
    m1, _ = _rank()
    assert drift(m0, m1) == 0.0
    save_checkpoint(tmp_path / 'old', 10, m0, o0, {'model': {}})       # THE OLD PATTERN
    assert drift(m0, m1) > 1e-9, 'the bug must reproduce, or this test proves nothing'

    (m0, o0), (m1, o1) = _rank(), _rank()
    save_checkpoint(tmp_path / 'new', 10, m0, o0, {'model': {}}, write=True)
    assert save_checkpoint(tmp_path / 'r1', 10, m1, o1, {'model': {}}, write=False) is None
    assert not (tmp_path / 'r1').exists(), 'write=False must create nothing on disk'
    assert drift(m0, m1) == 0.0, 'the fix must leave both ranks bit-identical'

    sgd_model = nn.Linear(8, 8)
    before = sgd_model.weight.clone()
    save_checkpoint(tmp_path / 'sgd', 5, sgd_model,
                    torch.optim.SGD(sgd_model.parameters(), lr=1e-3), {'model': {}}, write=False)
    assert torch.equal(sgd_model.weight, before)


# -- the drift guard -----------------------------------------------------------------------

def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


def test_the_signature_sees_one_changed_trainable_tensor():
    """Two ranks that stopped all-reducing ONE tensor drift; a frozen tensor is excluded (it is
    identical by construction and would hide the very tensor the guard watches)."""
    a, b = _tiny(), _tiny()
    sa = dist_utils.param_signature(a)
    assert dist_utils.signature_drift(dist_utils.param_signature(b), sa) == 0.0
    with torch.no_grad():
        b[0].weight[0, 0] += 1e-3
    assert dist_utils.signature_drift(dist_utils.param_signature(b), sa) > 1e-9
    full = sa.numel()
    a[0].weight.requires_grad_(False)
    assert dist_utils.param_signature(a).numel() == full - 2


def test_the_drift_is_per_entry_nan_fails_and_tol_sits_below_the_real_bug():
    """Normalising by the largest entry let a real small-tensor divergence read 5.6e-10 and pass;
    `NaN > tol` is False so NaN must not read as agreement; the one real bug read 2.9e-07, so
    `tol` must sit two orders below it (and above 0)."""
    import inspect

    ref = torch.tensor([1e6, 1e-3], dtype=torch.float64)
    sig = torch.tensor([1e6, 1.1e-3], dtype=torch.float64)
    assert dist_utils.signature_drift(sig, ref) == pytest.approx(0.1, rel=1e-6)
    nan = torch.tensor([float('nan')], dtype=torch.float64)
    assert not (dist_utils.signature_drift(nan, torch.tensor([1.0], dtype=torch.float64)) <= 1e-12)
    tol = inspect.signature(dist_utils.check_ranks_agree).parameters['tol'].default
    assert 0.0 < tol < 2.9e-07 / 100


def test_the_reductions_are_no_ops_without_a_world():
    """Every collective helper is callable at world 1, or the single-gpu path forks."""
    assert dist_utils.all_ranks_finite(None, True) is True
    assert dist_utils.all_ranks_finite(None, False) is False
    assert dist_utils.all_ranks_mean(None, 2.5) == 2.5
    assert dist_utils.gather_metrics(None, [{'mpjpe': 1.0}]) == [{'mpjpe': 1.0}]
    assert dist_utils.check_ranks_agree(None, _tiny()) == 0.0
    dist_utils.check_registry(None, ('nose', 'tail'))


# -- the launcher --------------------------------------------------------------------------

def _args(**over):
    return SimpleNamespace(**{'device': 'cuda:0', 'devices': 1, 'strategy': None,
                              'precision': '32-true', **over})


@pytest.mark.parametrize('over, match', [
    (dict(precision='16-mixed'), 'GradScaler'),          # this loop steps its own optimizer
    (dict(device='cuda:1', devices=4), '--devices'),     # would silently ignore --device
])
def test_launch_refusals(over, match):
    with pytest.raises(SystemExit, match=match):
        runtime.launch(_args(**over))


LAUNCH_CASES = [   # (visible gpus, cuda available, args, the Fabric kwargs they must resolve to)
    (1, True, dict(devices=-1), dict(devices=[0], strategy='auto', accelerator='gpu')),
    (4, True, dict(devices=-1), dict(devices=4, strategy='ddp_find_unused_parameters_true')),
    (4, False, dict(devices=-1), dict(devices=1, strategy='auto', accelerator='cpu')),
    (4, True, dict(devices=-1, device='cpu'), dict(devices=1, strategy='auto', accelerator='cpu')),
    (4, True, dict(device='cuda:2', devices=1), dict(devices=[2], accelerator='gpu')),
]


def test_launch_resolves_devices_before_choosing_the_strategy(monkeypatch):
    """`--devices -1` resolves to the visible count first (one GPU = the `--devices 1` path, no
    DDP); `--device cpu` and `--device cuda:N` are honoured -- for every trainer (the scorer's
    launcher used to hand Fabric gpu 0)."""
    import lightning.fabric as lf

    from tailcyclenet.scorer import train as scorer_train

    assert scorer_train.launch is runtime.launch
    seen = {}

    def spy(self, *a, **k):
        seen.update(k)
        raise SystemExit('stop-before-fabric-launch')

    monkeypatch.setattr(lf.Fabric, '__init__', spy)
    for n_gpus, cuda, over, want in LAUNCH_CASES:
        seen.clear()
        monkeypatch.setattr('torch.cuda.device_count', lambda: n_gpus)
        monkeypatch.setattr('torch.cuda.is_available', lambda: cuda)
        with pytest.raises(SystemExit, match='stop-before-fabric-launch'):
            runtime.launch(_args(**over))
        assert {k: seen[k] for k in want} == want, (n_gpus, cuda, over)


def test_run_batch_gives_the_loss_the_unwrapped_module():
    """The forward goes through the WRAPPED module (it all-reduces gradients); the loss reads
    `model.training`/`stride_overlap` off the raw one."""
    seen = {}
    wrapped = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}    # noqa: E731
    raw = object()

    def spy(model, out, **kw):
        seen['model'] = model
        return torch.zeros(())

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
