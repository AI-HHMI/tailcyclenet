"""A lone checkpoint file is a complete model specification for pose and scorer.

`save_checkpoint` embeds the config (with `[run] kind` stamped, exactly as `config.toml`), the
keypoint registry and the provenance, so `load_run` / `load_scorer_run` accept the `.pth` alone
and build the same model as the run folder. A file written before any of those were embedded
falls back to the run folder it sits in; a lone one missing them is refused by name.
"""
import shutil

import pytest
import torch

from tailcyclenet import checkpoints as ck
from tailcyclenet.format import Registry
from tailcyclenet.model import build_model

from .test_model import SMALL

REGISTRY = Registry(names=('nose', 'tail'), datasets=(('ds', (0, 1)),))


def _pose_config():
    return {'model': dict(SMALL), 'data': {'image_size': 64, 'n_frames': 4, 'min_crop_dim': 16,
                                           'box_source': 'keypoints'},
            'training': {'seed': 7}}


def _scorer_config():
    config = ck.load_config(ck._SCORER_CONFIG, base=ck._SCORER_CONFIG)
    model = {**SMALL, 'box_prompt': 'film'}
    model.pop('query_encoder', None)
    config['model'] = model
    config['data'] = {**config['data'], 'image_size': 64, 'n_frames': 4}
    config['scorer'] = {**config['scorer'], 'pool_num_heads': 2, 'score_hidden': 16}
    return config


def _scorer_model(config):
    from tailcyclenet.scorer.model import build_scorer
    from tailcyclenet.scorer.train import build_scorer_loss
    sc = config['scorer']
    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         REGISTRY.n_keypoints, pool_num_heads=sc['pool_num_heads'],
                         score_hidden=sc['score_hidden'], use_precision=sc['use_precision'],
                         output_granularity=sc['output_granularity'])
    model.add_module('frame_loss', build_scorer_loss(config, sc['output_granularity']))
    return model


def _train_run(tmp_path, kind):
    """A run folder written by the real writers: `save_run_meta` then `save_checkpoint`."""
    config = _pose_config() if kind == 'pose' else _scorer_config()
    model = (build_model({**config['model'], 'video_encoder_pretrained': False},
                         n_keypoints=REGISTRY.n_keypoints) if kind == 'pose'
             else _scorer_model(config))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    run = tmp_path / f'{kind}-run'
    ck.save_run_meta(run, config, REGISTRY, kind=kind, extra={'world_size': 1})
    path = ck.save_checkpoint(run, 5, model, opt, config, registry=REGISTRY, kind=kind)
    return run, path, model


def _load(kind, where, **kw):
    return (ck.load_run if kind == 'pose' else ck.load_scorer_run)(where, **kw)


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_embedded_config_is_the_run_folder_config(tmp_path, kind):
    import tomllib
    run, path, _ = _train_run(tmp_path, kind)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    with open(run / 'config.toml', 'rb') as f:
        on_disk = tomllib.load(f)
    assert ckpt['config'] == on_disk
    assert ckpt['config']['run']['kind'] == kind
    assert ckpt['keypoint_registry'] == REGISTRY.to_dict()
    assert ckpt['provenance'].get('world_size') == 1


def test_the_embedded_config_is_a_snapshot_not_a_reference(tmp_path):
    config = _pose_config()
    model = build_model({**config['model'], 'video_encoder_pretrained': False}, n_keypoints=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = ck.save_checkpoint(tmp_path, 0, model, opt, config, registry=REGISTRY)
    config['model']['latent_dim'] = 999
    config['data']['image_size'] = 999
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    assert ckpt['config']['model']['latent_dim'] == SMALL['latent_dim']
    assert ckpt['model_config']['latent_dim'] == SMALL['latent_dim']
    assert 'run' not in config, 'stamping the kind must not mutate the training config'


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_a_lone_checkpoint_loads_the_same_model_as_its_run_folder(tmp_path, kind):
    run, path, _ = _train_run(tmp_path, kind)
    lone = tmp_path / 'elsewhere' / 'model.pth'
    lone.parent.mkdir()
    shutil.copy(path, lone)
    shutil.rmtree(run)  # nothing to fall back on

    model, config, registry, got = _load(kind, lone)
    assert got == lone
    assert registry == REGISTRY
    assert config['run']['kind'] == kind
    assert model.run_provenance.get('world_size') == 1

    run2, path2, _ = _train_run(tmp_path, kind)
    ref, ref_config, _, _ = _load(kind, run2)
    torch.save(torch.load(path2, map_location='cpu', weights_only=False), lone)
    model, config, _, _ = _load(kind, lone)
    _same_weights(model, ref)
    assert config['model'] == ref_config['model']


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_an_old_checkpoint_falls_back_to_its_run_folder(tmp_path, kind):
    run, path, _ = _train_run(tmp_path, kind)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('config', 'model_config', 'keypoint_registry', 'provenance'):
        ckpt.pop(key)
    torch.save(ckpt, path)
    model, config, registry, _ = _load(kind, path)
    ref, _, _, _ = _load(kind, run)
    _same_weights(model, ref)
    assert registry == REGISTRY
    assert model.run_provenance.get('world_size') == 1

    lone = tmp_path / 'lone.pth'
    shutil.copy(path, lone)
    with pytest.raises(ValueError, match='no embedded config'):
        _load(kind, lone)


@pytest.mark.parametrize('kind', ['pose', 'scorer'])
def test_a_checkpoint_selector_is_refused_beside_a_file(tmp_path, kind):
    _, path, _ = _train_run(tmp_path, kind)
    with pytest.raises(ValueError, match='already names a checkpoint file'):
        _load(kind, path, checkpoint='checkpoint_best.pth')


def test_a_pose_checkpoint_is_refused_as_a_scorer_and_vice_versa(tmp_path):
    _, pose, _ = _train_run(tmp_path, 'pose')
    _, scorer, _ = _train_run(tmp_path, 'scorer')
    with pytest.raises(ValueError, match="'pose' checkpoint, not a 'scorer'"):
        ck.load_scorer_run(pose)
    with pytest.raises(ValueError, match="'scorer' checkpoint, not a 'pose'"):
        ck.load_run(scorer)


def test_peek_registry_reads_a_lone_checkpoint(tmp_path):
    _, path, _ = _train_run(tmp_path, 'pose')
    assert ck.peek_registry(path) == REGISTRY
