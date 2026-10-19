"""How a pose run starts: `start.plan_start` decides, `start.build_start` builds.

Behavioural, on the tiny router fixture from `test_optim` -- no Fabric, no tracktail model. Each
case is a resume/warm-start path that used to crash or silently change the recipe.
"""
import pytest
import torch

from tailcyclenet import start
from tailcyclenet.checkpoints import save_checkpoint
from tailcyclenet.format import Registry
from tailcyclenet.optim import PoseDualOptimizer, build_muon
from tailcyclenet.train import (KNOWN_TRAINING_KEYS, check_training_keys, has_val_split,
                                sampler_generator)
from tailcyclenet.loader.sampling import StepSampler
from tailcyclenet.unfreeze import apply_staged_unfreeze

from tests.test_optim import N_LAST, UNFREEZE_AT, Tiny, _cfg

REG = Registry(names=('a', 'b', 'c', 'd', 'e'), datasets=(('ds', (0, 1, 2, 3, 4)),))
CONFIG = {'model': {}, 'data': {}}


def _trained(run, iteration=UNFREEZE_AT + 2, cfg=None, world=1, registry=REG):
    """A run folder holding a real Muon checkpoint past the staged unfreeze."""
    cfg = cfg or _cfg()
    torch.manual_seed(0)
    model = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    opt = build_muon(model, set(), cfg)
    opt.train()
    apply_staged_unfreeze(model, opt, cfg, iteration)
    model.decoder.mlps[0](torch.randn(2, 8)).pow(2).sum().backward()
    opt.step()
    return save_checkpoint(run, iteration, model, opt, CONFIG, registry=registry,
                           world_size=world)


class _Factory:
    """`make_model` for `build_start`, recording each call's `pretrained` flag."""

    def __init__(self):
        self.calls = []

    def __call__(self, pretrained):
        self.calls.append(pretrained)
        return Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)


@pytest.fixture
def no_warm_weights(monkeypatch):
    """Record warm starts instead of running tracktail's migrations on the fixture."""
    seen = []

    def fake(model, path, base_names=None, **_):
        seen.append((path, base_names))
        return set()

    monkeypatch.setattr(start, 'warm_start', fake)
    return seen


def test_no_checkpoint_is_a_fresh_start_with_the_pretrained_backbone(tmp_path):
    """Also `no_warm_start` (never resolves `checkpoint_path`) and `no_resume` (a
    `checkpoint_path` warm-starts, nothing else is fresh)."""
    plan = start.plan_start(tmp_path / 'run')
    assert plan.mode == 'fresh' and plan.start_it == 0 and plan.registry is None
    make = _Factory()
    got = start.build_start(plan, make, _cfg())
    assert make.calls == [True], 'only a fresh run needs the VJEPA2 download'
    assert isinstance(got.optimizer, PoseDualOptimizer)
    assert start.plan_start(tmp_path / 'x', 'org/never-fetched', no_warm_start=True).mode == 'fresh'

    run = tmp_path / 'run'
    path = _trained(run)
    assert start.plan_start(run, str(path), no_resume=True).mode == 'warm'
    assert start.plan_start(run, no_resume=True).mode == 'fresh'


def test_own_folder_resumes_without_resolving_checkpoint_path(tmp_path, monkeypatch):
    """B13: the base checkpoint is not downloaded when the run folder resumes itself."""
    run = tmp_path / 'run'
    _trained(run)

    def boom(*a, **k):
        raise AssertionError('the warm-start base was fetched for an own-folder resume')

    monkeypatch.setattr(start, 'resolve_hf_checkpoint', boom)
    plan = start.plan_start(run, 'org/base-model')
    assert plan.mode == 'resume_own' and plan.start_it == UNFREEZE_AT + 2
    assert plan.registry == REG


def test_own_folder_resume_round_trips_and_replays_the_unfreeze_at_the_given_rates(tmp_path):
    """The build, the replay and the live unfreeze read ONE (world-scaled) dict."""
    run = tmp_path / 'run'
    _trained(run)
    make = _Factory()
    got = start.build_start(start.plan_start(run), make, _cfg())
    assert make.calls == [False], 'a resume overwrites the backbone; no download'
    assert got.plan.mode == 'resume_own' and got.plan.state is None
    enc = got.model.scene_encoder.encoder
    assert any(p.requires_grad for p in enc.blocks[-1].parameters()), 'unfreeze not replayed'
    lrs = [g['lr'] for g in got.optimizer.opt_muon.param_groups]
    assert pytest.approx(1e-4 * 0.1) in lrs, f'no encoder group at the configured rate: {lrs}'
    assert got.optimizer.opt_adam.param_groups[0]['k'] == 1, 'the optimizer state was not loaded'


def test_a_run_folder_as_checkpoint_path_resolves_to_its_checkpoints(tmp_path):
    """B1: `checkpoint_path = <run folder>` (or its `checkpoints/`) is a supported value."""
    other = tmp_path / 'other'
    _trained(other)
    for ref in (other, other / 'checkpoints'):
        plan = start.plan_start(tmp_path / 'run', str(ref))
        assert plan.mode == 'resume_ckpt'
        assert plan.path == other / 'checkpoints' / 'checkpoint_last.pth'
        assert plan.registry == REG


def test_weights_only_checkpoint_is_a_warm_start(tmp_path, no_warm_weights):
    """B2: a packaged pose `.pth` (no optimizer_state) used to KeyError; it warm-starts."""
    ck = torch.load(_trained(tmp_path / 'src'), weights_only=False)
    packaged = tmp_path / 'pose.pth'
    torch.save({'kind': 'pose', 'model_state': ck['model_state_eval'], 'iteration': 9,
                'keypoint_registry': REG.to_dict()}, packaged)
    plan = start.plan_start(tmp_path / 'run', str(packaged))
    assert plan.mode == 'warm' and 'weights only' in plan.reason
    make = _Factory()
    got = start.build_start(plan, make, _cfg())
    assert make.calls == [False]
    assert no_warm_weights == [(packaged, REG.names)], 'warm start keyed by the base registry'
    assert got.plan.start_it == 0


def test_a_layout_mismatch_falls_back_to_a_clean_warm_start(tmp_path, no_warm_weights, capsys):
    """B3: the fallback used to keep the REPLAYED unfreeze, training the encoder from step 0."""
    path = _trained(tmp_path / 'src')
    plan = start.plan_start(tmp_path / 'run', str(path))
    assert plan.mode == 'resume_ckpt'
    make = _Factory()
    got = start.build_start(plan, make, _cfg(learning_rate=2e-4))
    assert 'does not resume' in capsys.readouterr().out
    assert got.plan.mode == 'warm' and got.plan.start_it == 0
    assert make.calls == [False, False], 'the fallback builds a NEW model'
    enc = got.model.scene_encoder.encoder
    assert not any(p.requires_grad for p in enc.parameters()), 'encoder trainable at step 0'
    assert apply_staged_unfreeze(got.model, got.optimizer, _cfg(learning_rate=2e-4),
                                 UNFREEZE_AT), 'the configured unfreeze must still fire'


def test_a_layout_mismatch_in_the_own_folder_is_refused(tmp_path):
    run = tmp_path / 'run'
    _trained(run)
    with pytest.raises(SystemExit, match='does not resume.*learning rates'):
        start.build_start(start.plan_start(run), _Factory(), _cfg(learning_rate=2e-4))


def test_weights_that_do_not_load_fall_back_to_a_warm_start(tmp_path, no_warm_weights):
    path = _trained(tmp_path / 'src')
    plan = start.plan_start(tmp_path / 'run', str(path))

    def wider(pretrained):
        m = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
        m.query_encoder.kpt_embed = torch.nn.Embedding(7, 8)
        return m

    got = start.build_start(plan, wider, _cfg())
    assert got.plan.mode == 'warm' and 'do not load' in got.plan.reason


def test_own_folder_on_another_world_size_is_refused_naming_its_devices(tmp_path):
    """B4 (decided: refuse). The rates are sqrt(world)-scaled."""
    run = tmp_path / 'run'
    _trained(run, world=4)
    (run / 'provenance.toml').write_text('world_size = 4\ndevices = "4"\n')
    with pytest.raises(SystemExit, match='--devices 4'):
        start.plan_start(run, world=2)
    assert start.plan_start(run, world=4).mode == 'resume_own'


def test_checkpoint_path_from_another_world_size_warm_starts(tmp_path):
    path = _trained(tmp_path / 'src', world=4)
    plan = start.plan_start(tmp_path / 'run', str(path), world=2)
    assert plan.mode == 'warm' and '4 rank(s)' in plan.reason


def test_a_checkpoint_without_a_world_size_resumes_with_a_note(tmp_path, capsys):
    run = tmp_path / 'run'
    path = _trained(run)
    ck = torch.load(path, weights_only=False)
    del ck['optimizer_metadata']['world_size']
    torch.save(ck, path)
    assert start.plan_start(run, world=3).mode == 'resume_own'
    assert 'records no world size' in capsys.readouterr().out


def test_pre_metadata_checkpoints_no_longer_resume(tmp_path):
    run = tmp_path / 'run'
    path = _trained(run)
    ck = torch.load(path, weights_only=False)
    del ck['optimizer_metadata']
    torch.save(ck, path)
    with pytest.raises(SystemExit, match='v0.3.4'):
        start.plan_start(run)
    assert start.plan_start(tmp_path / 'other', str(path)).mode == 'warm'


def test_optimizer_kind_mismatch(tmp_path):
    run = tmp_path / 'run'
    path = _trained(run)
    with pytest.raises(SystemExit, match='schedulefree'):
        start.plan_start(run, opt_kind='schedulefree')
    plan = start.plan_start(tmp_path / 'other', str(path), opt_kind='schedulefree')
    assert plan.mode == 'warm' and 'muon' in plan.reason


def test_the_registry_comes_from_the_checkpoint_not_the_run_folder(tmp_path):
    """B10: a lone `.pth`'s own registry keys its identity rows; without one it is refused."""
    path = _trained(tmp_path / 'src')
    ck = torch.load(path, weights_only=False)
    lone = tmp_path / 'lone.pth'
    torch.save({'kind': 'pose', 'model_state': ck['model_state'],
                'keypoint_registry': REG.to_dict()}, lone)
    assert start.plan_start(tmp_path / 'run', str(lone)).registry == REG
    torch.save({'kind': 'pose', 'model_state': ck['model_state']}, lone)
    with pytest.raises(SystemExit, match='no keypoint registry'):
        start.plan_start(tmp_path / 'run', str(lone))
    torch.save({'model_state': {'decoder.mlps.0.weight': torch.zeros(8, 8)}}, lone)
    plan = start.plan_start(tmp_path / 'run', str(lone))
    assert plan.mode == 'warm' and plan.registry is None, 'a base with no keypoint table is fine'


def test_training_keys_are_checked_and_freeze_encoder_raises_by_name(tmp_path):
    check_training_keys({k: None for k in KNOWN_TRAINING_KEYS})
    with pytest.raises(SystemExit, match='n_iteration'):
        check_training_keys({'n_iteration': 5})
    with pytest.raises(SystemExit, match='freeze_encoder was removed'):
        check_training_keys({'freeze_encoder': True})
    # and only a MISSING val split skips validation (a folder of roots counts any root's)
    root = tmp_path / 'root'
    (root / 'train').mkdir(parents=True)
    assert not has_val_split(root)
    (root / 'val').mkdir()
    assert has_val_split(root)
    (tmp_path / 'multi' / 'a' / 'val').mkdir(parents=True)
    assert has_val_split(tmp_path / 'multi')


def test_a_resumed_sampler_continues_its_ordinal_on_a_new_stream():
    """B6: the ordinal keys the loader's cost draws, so a resume must not restart it at 0."""
    assert sampler_generator(23, 0, 1, 0) is None, 'a fresh 1-gpu run keeps its stream'
    assert sampler_generator(23, 1, 2, 0).initial_seed() == 24
    sampler = StepSampler(10, 3, generator=sampler_generator(23, 0, 1, 50), start=50)
    assert [k for k, _ in sampler] == [50, 51, 52]
    a = [i for _, i in StepSampler(1000, 20, sampler_generator(23, 0, 1, 50), start=50)]
    b = [i for _, i in StepSampler(1000, 20, sampler_generator(23, 0, 1, 60), start=60)]
    assert a != b
