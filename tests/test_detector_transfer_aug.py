"""Cross-rig detector levers (`tailcyclenet/detector/transfer_aug.py`).

Each test pins one claim:
- `normalize_input` is deterministic, keeps the letterbox grey, and removes a global gain;
- the training item and the deployment input see the SAME normalisation (train == deploy);
- `composite` (synthetic canvases only) moves a box with its animal's pixels;
- the shipped defaults are the report-73 recipe, nonsense values and DELETED keys / `yolox` values
  raise by name, and an absent `pretrained` follows the architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.dataset import read_frames
from tailcyclenet.detector import BoxDataset, letterbox
from tailcyclenet.detector.config import load_detector_config
from tailcyclenet.detector.transfer_aug import PAD, composite, invert, normalize_input


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    # near-grey like every rig in scope; luma equalisation leaves chroma alone by design
    img = np.repeat(rng.integers(20, 200, (60, 80, 1)), 3, -1).astype(np.uint8)
    img[:, :10] = PAD                                     # letterbox border
    return img


@pytest.mark.parametrize('mode', ['equalize', 'percentile'])
def test_normalize_input_is_deterministic_keeps_padding_and_removes_gain(mode):
    img = _scene()
    a = normalize_input(img, mode)
    assert np.array_equal(a, normalize_input(img, mode))
    assert (a[:, :10] == PAD).all()
    dark = img.copy()
    content = ~(img == PAD).all(-1)
    dark[content] = (img[content] * 0.1).astype(np.uint8)
    b = normalize_input(dark, mode)
    # a 10x darker exposure maps to (nearly) the same input; without normalisation it is ~10x off
    assert np.abs(a[content].astype(int) - b[content].astype(int)).mean() < 12
    assert np.abs(img[content].astype(int) - dark[content].astype(int)).mean() > 60


def test_normalize_input_none_is_identity():
    img = _scene()
    assert normalize_input(img, 'none') is img


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
    img = np.zeros((96, 128, 3), np.uint8)
    img[40:56, 60:84] = 255                              # the 'animal'
    boxes = torch.tensor([[60.0, 40.0, 84.0, 56.0], [float('nan')] * 4])
    for seed in range(10):
        out, new = composite(img, boxes, np.random.default_rng(seed), (128, 96),
                             empty_prob=0.0, decoys_ok=False)
        assert new.shape == (1, 4)
        x0, y0, x1, y1 = new[0].round().int().tolist()
        assert (out[y0 + 2:y1 - 2, x0 + 2:x1 - 2] == 255).all()
        assert (x1 - x0, y1 - y0) == (24, 16)


def test_composite_empty_canvas_has_no_boxes(flat_canvas):
    img = np.zeros((96, 128, 3), np.uint8)
    boxes = torch.tensor([[60.0, 40.0, 84.0, 56.0]])
    out, new = composite(img, boxes, np.random.default_rng(0), (128, 96),
                         empty_prob=1.0, decoys_ok=False)
    assert new.shape == (0, 4) and (out == 90).all()


def _cfg(tmp_path, extra='', model='yolox = "tiny"'):
    p = tmp_path / 'c.toml'
    p.write_text(f'[data]\npath = "/tmp/ds"\n{extra}\n[model]\n{model}\n'
                 '[training]\nout = "/tmp/run"\n')
    return p


def test_shipped_defaults_are_the_report_73_recipe(tmp_path):
    cfg = load_detector_config(_cfg(tmp_path, '', model=''))
    d, m = cfg['data'], cfg['model']
    assert (m['yolox'], m['pretrained']) == ('convnext-t', 'imagenet')
    assert d['input_norm'] == 'equalize' and d['invert_prob'] == 0.15
    assert d['scale_range'] == [0.55, 1.4] and d['background_prob'] == 0.25
    assert cfg['training']['iters'] == 8000 and d['boxes'] == 'instances'


@pytest.mark.parametrize('model, want', [
    ('yolox = "convnext-t"', 'imagenet'), ('yolox = "convnext-t"\npretrained = ""', ''),
    ('yolox = "tiny"', ''), ('yolox = "hybrid"', ''),
    ('yolox = "tiny"\npretrained = "coco"\nbottleneck_expansion = 1.0', 'coco')])
def test_absent_pretrained_follows_the_architecture(tmp_path, model, want):
    """An overlay that changes `yolox` never inherits a source that cannot load into it."""
    assert load_detector_config(_cfg(tmp_path, '', model=model))['model']['pretrained'] == want


@pytest.mark.parametrize('extra, model, match', [
    ('exposure_prob = 0.5', '', 'exposure_prob'),
    ('background_own = "x"', '', 'background_own'),
    ('background_generic = "x"', '', 'background_generic'),
    ('background_weights = [0, 0, 1]', '', 'background_weights'),
    ('', 'yolox = "dinov2-s"', 'deleted'), ('', 'yolox = "hybrid-tiny"', 'deleted'),
    ('', 'yolox = "cspvit-m"', 'deleted'), ('', 'yolox = "tiny"\npretrained = "imagenet"', 'imagenet')])
def test_deleted_and_mismatched_keys_raise_by_name(tmp_path, extra, model, match):
    with pytest.raises(SystemExit, match=match):
        load_detector_config(_cfg(tmp_path, extra, model=model))


def test_load_detector_refuses_a_deleted_architecture(tmp_path):
    from tailcyclenet.detector import load_detector
    p = tmp_path / 'detector_it000010.pth'
    torch.save({'yolox_version': 'hybrid-tiny', 'input_wh': [64, 64], 'norm': 'gn',
                'model_state': {}}, p)
    with pytest.raises(ValueError, match='deleted architecture'):
        load_detector(p)


@pytest.mark.parametrize('extra, match', [
    ('input_norm = "clahe"', 'input_norm'),
    ('invert_prob = 1.5', 'invert_prob'),
    ('scale_range = [1.2, 0.5]', 'scale_range'),
    ('background_prob = 0.5\nkeypoints = true', 'box-only'),
    ('keypoints = true', 'background_prob = 0 alongside keypoints'),
])
def test_transfer_keys_refuse_nonsense(tmp_path, extra, match):
    with pytest.raises(SystemExit, match=match):
        load_detector_config(_cfg(tmp_path, extra))


def test_invert_flips_content_and_keeps_padding():
    img = _scene()
    out = invert(img)
    content = ~(img == PAD).all(-1)
    assert (out[~content] == PAD).all()
    assert np.array_equal(out[content], 255 - img[content])


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


def test_synthetic_canvas_is_deterministic_given_the_rng():
    from tailcyclenet.detector.transfer_aug import synthetic_canvas
    a = synthetic_canvas((96, 64), np.random.default_rng(3))
    b = synthetic_canvas((96, 64), np.random.default_rng(3))
    assert a.shape == (64, 96, 3) and a.dtype == np.uint8 and np.array_equal(a, b)
