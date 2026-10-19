"""A lone checkpoint file is a complete model specification for pose and scorer.

`save_checkpoint` embeds the kind-stamped config, the registry and the provenance, so the `.pth`
alone builds the same model as its run folder; an older file falls back to its folder.
"""
import shutil

import pytest
import torch

from tailcyclenet import checkpoints as ck
from tailcyclenet import config as cfgmod
from tailcyclenet.format import Registry
from tailcyclenet.pose import checkpoint as pck
from tailcyclenet.pose.model import build_model
from tailcyclenet.scorer.checkpoint import load_scorer_run, save_scorer_checkpoint
from tests.helpers import SMALL

REGISTRY = Registry(names=('nose', 'tail'), datasets=(('ds', (0, 1)),))


def _pose_config():
    return {'model': dict(SMALL), 'data': {'image_size': 64, 'n_frames': 4, 'min_crop_dim': 16,
                                           'box_source': 'keypoints'},
            'training': {'seed': 7}}


def _scorer_config():
    config = cfgmod.load_config(cfgmod.SCORER_CONFIG, base=cfgmod.SCORER_CONFIG)
    model = {**SMALL, 'box_prompt': 'film'}
    model.pop('query_encoder', None)
    config['model'] = model
    config['data'] = {**config['data'], 'image_size': 64, 'n_frames': 4}
    config['scorer'] = {**config['scorer'], 'pool_num_heads': 2, 'score_hidden': 16}
    return config


def _scorer_model(config):
    from tailcyclenet.scorer.losses import build_scorer_loss
    from tailcyclenet.scorer.model import build_scorer
    sc = config['scorer']
    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         REGISTRY.n_keypoints, pool_num_heads=sc['pool_num_heads'],
                         score_hidden=sc['score_hidden'], use_precision=sc['use_precision'],
                         output_granularity=sc['output_granularity'])
    model.add_module('frame_loss', build_scorer_loss(config, sc['output_granularity']))
    return model


class _Averaged(torch.optim.AdamW):
    """AdamW with a (no-op) averaged iterate, so `model_state_eval` is written like a real run's."""

    def eval(self):
        pass

    def train(self):
        pass


def _train_run(tmp_path, kind):
    """A run folder written by the real writers: `save_run_meta` then `save_checkpoint`."""
    config = _pose_config() if kind == 'pose' else _scorer_config()
    model = (build_model({**config['model'], 'video_encoder_pretrained': False},
                         n_keypoints=REGISTRY.n_keypoints) if kind == 'pose'
             else _scorer_model(config))
    opt = _Averaged([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    run = tmp_path / f'{kind}-run'
    ck.save_run_meta(run, config, REGISTRY, kind=kind, extra={'world_size': 1})
    if kind == 'pose':
        path = ck.save_checkpoint(run, 5, model, opt, config, registry=REGISTRY)
    else:
        path = save_scorer_checkpoint(run, 5, model, opt, config, registry=REGISTRY)
    return run, path, model


def _load(kind, where, **kw):
    return (pck.load_run if kind == 'pose' else load_scorer_run)(where, **kw)


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@pytest.fixture
def no_download(monkeypatch):
    """Loading a finetuned checkpoint must never fetch the VJEPA2 backbone (offline nodes)."""
    def boom(*a, **k):
        raise AssertionError('a pretrained VJEPA2 checkpoint was fetched from the network')

    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', boom)


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_a_lone_checkpoint_loads_the_same_model_as_its_run_folder(tmp_path, kind, no_download):
    """The embedded config IS `config.toml` (a snapshot, not a reference), and the `.pth` alone
    loads the run folder's weights, config, registry and provenance."""
    import tomllib
    run, path, _ = _train_run(tmp_path, kind)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    with open(run / 'config.toml', 'rb') as f:
        assert ckpt['config'] == tomllib.load(f)
    assert ckpt['config']['run']['kind'] == kind
    assert ckpt['keypoint_registry'] == REGISTRY.to_dict()
    assert ckpt['provenance'].get('world_size') == 1
    assert ck.peek_registry(path) == REGISTRY

    ref, ref_config, _, _ = _load(kind, run)
    lone = tmp_path / 'elsewhere' / 'model.pth'
    lone.parent.mkdir()
    shutil.copy(path, lone)
    shutil.rmtree(run)
    model, config, registry, got = _load(kind, lone)
    assert got == lone and registry == REGISTRY
    assert config['run']['kind'] == kind and config['model'] == ref_config['model']
    assert model.run_provenance.get('world_size') == 1
    _same_weights(model, ref)


def test_the_embedded_config_is_a_snapshot_not_a_reference(tmp_path):
    config = {'model': {'latent_dim': 64}, 'data': {'image_size': 64}}
    model = torch.nn.Linear(2, 2)
    path = ck.save_checkpoint(tmp_path, 0, model, torch.optim.AdamW(model.parameters()), config,
                              registry=REGISTRY)
    config['model']['latent_dim'] = 999
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    assert ckpt['config']['model']['latent_dim'] == 64
    assert 'run' not in config, 'stamping the kind must not mutate the training config'


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_an_old_checkpoint_falls_back_to_its_run_folder(tmp_path, kind):
    """No embedded config/registry/provenance: read the folder; a lone one is refused by name. A
    checkpoint selector beside a file path is refused too."""
    run, path, _ = _train_run(tmp_path, kind)
    with pytest.raises(ValueError, match='already names a checkpoint file'):
        _load(kind, path, checkpoint='checkpoint_best.pth')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('config', 'keypoint_registry', 'provenance'):
        ckpt.pop(key)
    torch.save(ckpt, path)
    model, _, registry, _ = _load(kind, path)
    ref, _, _, _ = _load(kind, run)
    _same_weights(model, ref)
    assert registry == REGISTRY
    assert model.run_provenance.get('world_size') == 1

    lone = tmp_path / 'lone.pth'
    shutil.copy(path, lone)
    with pytest.raises(ValueError, match='no embedded config'):
        _load(kind, lone)


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_a_packaged_checkpoint_loads_alone_like_its_run(tmp_path, kind, no_download):
    from tailcyclenet.packaging import package_pose
    run, _, _ = _train_run(tmp_path, kind)
    packaged = package_pose(run, tmp_path / f'{kind}.pth', kind=kind)
    ckpt = torch.load(packaged, map_location='cpu', weights_only=False)
    assert ckpt['kind'] == kind
    assert 'optimizer_state' not in ckpt
    assert ckpt['provenance'].get('world_size') == 1
    ref, ref_config, _, _ = _load(kind, run)
    shutil.rmtree(run)
    model, config, registry, _ = _load(kind, packaged)
    _same_weights(model, ref)
    assert config == ref_config
    assert registry == REGISTRY


def test_a_checkpoint_of_the_wrong_kind_is_refused(tmp_path):
    """A scorer IS a pose encoder plus heads, so loading one as the other must raise, not build."""
    from tailcyclenet.packaging import package_pose
    _, pose, _ = _train_run(tmp_path, 'pose')
    scorer_run, scorer, _ = _train_run(tmp_path, 'scorer')
    with pytest.raises(ValueError, match="'pose' checkpoint, not a 'scorer'"):
        load_scorer_run(pose)
    with pytest.raises(ValueError, match="'scorer' checkpoint, not a 'pose'"):
        pck.load_run(scorer)
    with pytest.raises(ValueError, match="'scorer' checkpoint, not a 'pose'"):
        package_pose(scorer_run, tmp_path / 'x.pth', kind='pose')
