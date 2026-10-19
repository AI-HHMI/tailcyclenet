"""Cross-rig detector levers (`tailcyclenet/detector/transfer_aug.py`) and the report-73 config."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tailcyclenet.data.pixels import read_frames
from tailcyclenet.detector import BoxDataset, letterbox
from tailcyclenet.detector.config import load_detector_config
from tailcyclenet.detector.transfer_aug import PAD, composite, invert, normalize_input


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    # near-grey like every rig in scope; luma equalisation leaves chroma alone by design
    img = np.repeat(rng.integers(20, 200, (60, 80, 1)), 3, -1).astype(np.uint8)
    img[:, :10] = PAD                                     # letterbox border
    return img


def test_normalize_input_and_invert():
    """`normalize_input` is deterministic, keeps the letterbox grey, removes a global gain, and
    'none' is the identity; `invert` flips content and keeps the padding."""
    img = _scene()
    content = ~(img == PAD).all(-1)
    dark = img.copy()
    dark[content] = (img[content] * 0.1).astype(np.uint8)
    assert np.abs(img[content].astype(int) - dark[content].astype(int)).mean() > 60
    for mode in ('equalize', 'percentile'):
        a = normalize_input(img, mode)
        assert np.array_equal(a, normalize_input(img, mode))
        assert (a[:, :10] == PAD).all()
        b = normalize_input(dark, mode)
        assert np.abs(a[content].astype(int) - b[content].astype(int)).mean() < 12
    assert normalize_input(img, 'none') is img
    out = invert(img)
    assert (out[~content] == PAD).all()
    assert np.array_equal(out[content], 255 - img[content])


def test_dataset_item_matches_deployment_normalisation(tiny_root):
    """`detect_raw` letterboxes then normalises; the eval item must be that exact image."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=2, input_norm='equalize', antialias=True)
    sess, gid, f, ci = ds.index[0]
    raw = read_frames(sess.groups[gid], sess.cam_names[ci], [f])[0]
    lb, _, _ = letterbox(raw, ds.input_wh, src_wh=sess.rig.size(sess.cam_names[ci]),
                         antialias=True)
    want = torch.as_tensor(normalize_input(lb, 'equalize'), dtype=torch.float32)
    got = ds[0]['x'].permute(1, 2, 0) * 255.0
    torch.testing.assert_close(got, want, atol=1e-3, rtol=0)


@pytest.fixture
def flat_canvas(monkeypatch):
    """Make `composite`'s synthetic canvas a flat grey, so pasted pixels are checkable."""
    from tailcyclenet.detector import transfer_aug
    monkeypatch.setattr(transfer_aug, 'synthetic_canvas',
                        lambda wh, rng, blob_side=40.0: np.full((wh[1], wh[0], 3), 90, np.uint8))


def test_composite_moves_the_box_with_the_animal(flat_canvas):
    """A pasted animal's box moves with its pixels; an empty canvas carries no box."""
    img = np.zeros((96, 128, 3), np.uint8)
    img[40:56, 60:84] = 255                              # the 'animal'
    boxes = torch.tensor([[60.0, 40.0, 84.0, 56.0], [float('nan')] * 4])
    for seed in range(5):
        out, new = composite(img, boxes, np.random.default_rng(seed), (128, 96),
                             empty_prob=0.0, decoys_ok=False)
        assert new.shape == (1, 4)
        x0, y0, x1, y1 = new[0].round().int().tolist()
        assert (out[y0 + 2:y1 - 2, x0 + 2:x1 - 2] == 255).all()
        assert (x1 - x0, y1 - y0) == (24, 16)
    out, new = composite(img, boxes[:1], np.random.default_rng(0), (128, 96),
                         empty_prob=1.0, decoys_ok=False)
    assert new.shape == (0, 4) and (out == 90).all()


def _cfg(tmp_path, extra='', model='yolox = "hybrid"'):
    p = tmp_path / 'c.toml'
    p.write_text(f'[data]\npath = "/tmp/ds"\n{extra}\n[model]\n{model}\n'
                 '[training]\nout = "/tmp/run"\n')
    return p


def test_shipped_defaults_are_the_report_73_recipe(tmp_path):
    """The shipped recipe is report 73's; an overlay changing `yolox` never inherits a
    `pretrained` source that cannot load into it."""
    cfg = load_detector_config(_cfg(tmp_path, '', model=''))
    d, m = cfg['data'], cfg['model']
    assert (m['yolox'], m['pretrained']) == ('convnext-t', 'imagenet')
    assert d['input_norm'] == 'equalize' and d['invert_prob'] == 0.15
    assert d['scale_range'] == [0.55, 1.4] and d['background_prob'] == 0.25
    assert cfg['training']['iters'] == 8000 and d['boxes'] == 'instances'
    for model, want in (('yolox = "convnext-t"\npretrained = ""', ''), ('yolox = "hybrid"', '')):
        assert load_detector_config(_cfg(tmp_path, '', model=model))['model']['pretrained'] == want


@pytest.mark.parametrize('extra, model, match', [
    ('exposure_prob = 0.5', '', 'exposure_prob'),
    ('keypoints = false', '', "deleted key.*'keypoints'"),
    ('', 'yolox = "dinov2-s"', 'deleted'),
    ('', 'yolox = "tiny"', 'deleted.*v0.3.4'),
    ('', 'yolox = "hybrid"\npretrained = "coco"', "pretrained='coco' is deleted"),
    ('', 'yolox = "hybrid"\npretrained = "imagenet"', 'imagenet'),
    ('input_norm = "clahe"', '', 'input_norm'),
    ('scale_range = [1.2, 0.5]', '', 'scale_range')])
def test_deleted_and_mismatched_keys_raise_by_name(tmp_path, extra, model, match):
    """One case per family: deleted key, deleted architecture, deleted source, a source the
    architecture cannot load, and a nonsense value."""
    with pytest.raises(SystemExit, match=match):
        load_detector_config(_cfg(tmp_path, extra, model=model))


@pytest.mark.parametrize('meta, match', [
    ({'yolox_version': 'tiny'}, 'deleted architecture.*v0.3.4'),
    ({}, "yolox='trimmed' is a deleted architecture"),          # absent key = legacy default
    ({'yolox_version': 'hybrid', 'n_keypoints': 17}, 'keypoint branch is deleted'),
    ({'yolox_version': 'hybrid', 'in_channels': 6}, 'in_channels=6')])
def test_load_detector_refuses_retired_checkpoints_by_name(tmp_path, meta, match):
    from tailcyclenet.detector import load_detector
    p = tmp_path / 'detector_it000010.pth'
    torch.save({'input_wh': [64, 64], 'norm': 'gn', 'model_state': {}, **meta}, p)
    with pytest.raises(ValueError, match=match):
        load_detector(p)


def test_convnext_meets_the_backbone_contract():
    """Strides 4/8/16/32 at exactly H//s x W//s; the ImageNet trunk is what gets the backbone LR
    scale. Architecture only -- no weight download in CI."""
    from tailcyclenet.detector.yolox import YOLOXNano
    from tailcyclenet.train_detector import _pretrained_params
    m = YOLOXNano(version='convnext-t', p2=True)
    feats = m.backbone(torch.rand(1, 3, 96, 160))
    assert [tuple(f.shape[2:]) for f in feats] == [(96 // s, 160 // s) for s in (4, 8, 16, 32)]
    assert [f.shape[1] for f in feats] == list(m.backbone.out_channels)
    trunk = {id(p) for p in _pretrained_params(m)}
    assert trunk == {id(p) for p in m.backbone.features.parameters()}

