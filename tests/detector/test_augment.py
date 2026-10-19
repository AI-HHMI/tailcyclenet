"""Detector augmentation: affine/rotation, photometric draws, input normalisation, compositing."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tailcyclenet.detector.augment import (
    PAD,
    composite,
    invert,
    normalize_input,
    photometric,
    random_affine,
)


def test_rotation_off_is_byte_identical_and_on_turns_about_its_centre():
    """`rotate_deg=0` reproduces the pre-rotation matrix draw for draw (drawing an angle and
    zeroing it would reseat every later draw); rotation turns about `centre`."""
    def before(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5):
        w, h = float(size[0]), float(size[1])
        s = rng.uniform(*scale)
        sx = -s if rng.random() < hflip else s
        cx, cy = w / 2, h / 2
        return np.array([[sx, 0.0, cx - sx * cx + rng.uniform(-translate, translate) * w],
                         [0.0, s, cy - s * cy + rng.uniform(-translate, translate) * h]],
                        np.float32)

    for seed in range(5):
        a = before((4696, 2048), np.random.default_rng([0, seed]))
        b = random_affine((4696, 2048), np.random.default_rng([0, seed]))
        assert np.array_equal(a, b), f'seed {seed} moved'

    # And the rotation is about `centre`, which is the whole reason tiling and rotation compose.
    M = random_affine((1000, 1000), np.random.default_rng(1), scale=(1.0, 1.0), translate=0.0,
                      hflip=0.0, rotate_deg=180.0, centre=(300.0, 700.0))
    fixed = M @ np.array([300.0, 700.0, 1.0])
    np.testing.assert_allclose(fixed, [300.0, 700.0], atol=1e-3)


def test_photometric_is_one_gain_draw():
    """`photometric(img, rng)` consumes exactly ONE `uniform(0.7, 1.3)` and applies it as a
    clipped gain -- the augmentation stream's draw order rests on this."""
    img = (np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3) % 200).astype(np.uint8)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    out1 = photometric(img, rng1)
    gain = rng2.uniform(0.7, 1.3)
    np.testing.assert_array_equal(out1, np.clip(img * gain, 0, 255).astype(np.uint8))
    assert rng1.uniform() == rng2.uniform()


def test_random_affine_vflip_and_full_circle_rotation(tmp_path):
    """vflip=0 draws nothing (every augmentation stream on record reproduces), vflip=1 mirrors;
    the shipped recipe rotates over +-180 deg about the centre (owner decision 2026-10-02)."""
    from tailcyclenet.detector.config import load_detector_config

    for seed in range(5):
        a = random_affine((640, 480), np.random.default_rng(seed), rotate_deg=30.0)
        b = random_affine((640, 480), np.random.default_rng(seed), rotate_deg=30.0, vflip=0.0)
        assert np.array_equal(a, b)
    neutral = dict(scale=(1.0, 1.0), translate=0.0, hflip=0.0)
    M = random_affine((100, 100), np.random.default_rng(0), vflip=1.0, **neutral)
    np.testing.assert_allclose(M[:, :2], [[1, 0], [0, -1]])
    np.testing.assert_allclose(M @ [50, 50, 1], [50, 50], atol=1e-5)

    cfg = tmp_path / 'c.toml'
    cfg.write_text(f'[data]\npath = "{tmp_path}"\n[model]\n[training]\nout = "{tmp_path}"\n')
    assert load_detector_config(cfg)['data']['rotate_deg'] == 180.0
    angles = []
    for seed in range(400):
        M = random_affine((100, 60), np.random.default_rng(seed), rotate_deg=180.0, **neutral)
        np.testing.assert_allclose(M @ [50, 30, 1], [50, 30], atol=1e-4)
        angles.append(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    assert min(angles) < -170 and max(angles) > 170


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


@pytest.fixture
def flat_canvas(monkeypatch):
    """Make `composite`'s synthetic canvas a flat grey, so pasted pixels are checkable."""
    from tailcyclenet.detector import augment as transfer_aug
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
