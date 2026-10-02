"""The cross-rig detector changes: scored weights, tight-extent target, anti-aliasing, new augs.

Each test pins one claim the change rests on:
- `model_state` is the weight the training loop SCORED (schedule-free's averaged `x`), not the
  gradient point `y` the optimizer swaps back in after scoring;
- the crop rule is a function of the extent alone, so regressing the tight extent and applying
  `crop_boxes_from_extents` after detection gives the crop `crop_box_for_points` would;
- `antialias` changes the filter, never the geometry;
- `vflip` / `grayscale_prob` draw nothing at 0, and do what they say at 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import crop_box_for_points, crop_boxes_from_extents
from tailcyclenet.detector import BoxDataset, letterbox, unletterbox_boxes
from tailcyclenet.detector.data import MIN_EXTENT_PX, random_affine, warp_image

REPO = Path(__file__).resolve().parent.parent


def test_crop_boxes_from_extents_is_the_crop_rule_int32_exact():
    """Vectorised over extents, it must return `crop_box_for_points` on the extent's corners,
    bit for bit -- including clamping, the floor, the square and the frame cap."""
    g = torch.Generator().manual_seed(0)
    for size in ((4696, 2048), (400, 400), (63, 50)):
        W, H = size
        for mcd, pad in ((64, 20), (64, 0), (8, 20), (200, 5)):
            n = 400
            x0 = torch.rand(n, generator=g) * 1.4 * W - 0.2 * W
            y0 = torch.rand(n, generator=g) * 1.4 * H - 0.2 * H
            w = torch.rand(n, generator=g) ** 3 * W
            h = torch.rand(n, generator=g) ** 3 * H
            boxes = torch.stack([x0, y0, x0 + w, y0 + h], -1)
            boxes[::17] = float('nan')
            got = crop_boxes_from_extents(boxes, size, mcd, pad)
            for i in range(n):
                if not torch.isfinite(boxes[i]).all():
                    assert torch.isnan(got[i]).all()
                    continue
                want = crop_box_for_points(boxes[i].view(2, 2), torch.tensor(size), mcd, pad)
                assert torch.equal(want.float(), got[i]), (size, mcd, pad, boxes[i])


def test_extent_target_is_tight_and_gives_back_the_crop_rule(tiny_root):
    """The `extent` target is the points' own min/max (letterboxed), and the crop rule applied to
    it after unletterboxing -- what `detect_raw` does -- is the crop the `crop` target trains on."""
    kw = dict(input_wh=(128, 128), min_crop_dim=8, max_frames_per_group=2)
    ext = BoxDataset(tiny_root / 'ratlike', 'train', box_target='extent', **kw)
    crop = BoxDataset(tiny_root / 'ratlike', 'train', **kw)
    checked = 0
    for i in range(len(ext)):
        sess, gid, f, ci = ext.index[i]
        cam = sess.rig.posetail()[ci]
        img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
        _, scale, pad = letterbox(img, ext.input_wh)
        got = ext.boxes_for(i)
        lab = sess.labels(gid)
        for s in range(got.shape[0]):
            pts = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
            fin = pts[torch.isfinite(pts).all(-1)]
            if not fin.shape[0]:
                assert torch.isnan(got[s]).all()
                continue
            side = got[s, 2:] - got[s, :2]
            assert (side >= MIN_EXTENT_PX - 1e-4).all()
            back = unletterbox_boxes(got[s][None], scale, pad)[0]
            if ((fin.max(0).values - fin.min(0).values) * scale > MIN_EXTENT_PX).all():
                lo = fin.min(0).values.clamp(min=0)
                hi = torch.minimum(fin.max(0).values, cam['size'].float())
                torch.testing.assert_close(back, torch.cat([lo, hi]), atol=1e-3, rtol=0)
                rule = crop_boxes_from_extents(back[None], cam['size'], ext.min_crop_dim, 20)[0]
                want = crop_box_for_points(pts, cam['size'], ext.min_crop_dim).float()
                torch.testing.assert_close(rule, want, atol=1.0, rtol=0)
                legacy = unletterbox_boxes(crop.boxes_for(i)[s][None], scale, pad)[0]
                torch.testing.assert_close(legacy, want, atol=0.51, rtol=0)
                checked += 1
    assert checked, 'no box was big enough to escape the floor -- the test checked nothing'


def test_random_affine_new_levers_draw_nothing_when_off():
    """At vflip=0 the draw sequence is the legacy one, so every augmentation stream on record is
    reproduced; at 1 it is an exact mirror."""
    for seed in range(20):
        a = random_affine((640, 480), np.random.default_rng(seed), rotate_deg=30.0)
        b = random_affine((640, 480), np.random.default_rng(seed), rotate_deg=30.0, vflip=0.0)
        assert np.array_equal(a, b)
    neutral = dict(scale=(1.0, 1.0), translate=0.0, hflip=0.0)
    M = random_affine((100, 100), np.random.default_rng(0), vflip=1.0, **neutral)
    np.testing.assert_allclose(M[:, :2], [[1, 0], [0, -1]])
    np.testing.assert_allclose(M @ [50, 50, 1], [50, 50], atol=1e-5)


def test_shipped_detector_rotation_is_the_full_circle(tmp_path):
    """The shipped recipe rotates uniformly over +-180 degrees (owner decision 2026-10-02), and
    `random_affine` at 180 reaches both ends of the circle about the frame centre."""
    from tailcyclenet.detector.config import load_detector_config

    cfg = tmp_path / 'c.toml'
    cfg.write_text(f'[data]\npath = "{tmp_path}"\n[model]\n[training]\nout = "{tmp_path}"\n')
    assert load_detector_config(cfg)['data']['rotate_deg'] == 180.0
    neutral = dict(scale=(1.0, 1.0), translate=0.0, hflip=0.0)
    angles = []
    for seed in range(400):
        M = random_affine((100, 60), np.random.default_rng(seed), rotate_deg=180.0, **neutral)
        np.testing.assert_allclose(M @ [50, 30, 1], [50, 30], atol=1e-4)
        angles.append(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    assert min(angles) < -170 and max(angles) > 170


def test_vflip_is_off_under_keypoints(tiny_root):
    """A vertical mirror swaps left/right keypoints exactly as a horizontal one does."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), keypoints=True, vflip=0.5)
    assert ds.vflip == 0.0 and ds.hflip == 0.0


def _checker(n=256):
    """A 1-px checkerboard: the worst case for a bilinear shrink."""
    yy, xx = np.mgrid[:n, :n]
    img = (((yy + xx) % 2) * 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, -1)


def test_antialias_changes_the_filter_not_the_geometry():
    """Off = `cv2.warpAffine` byte-identical; on = a 1-px checkerboard shrinks to flat grey (no
    aliasing), and a blob's centroid lands where the plain warp puts it."""
    import cv2

    M = np.array([[0.2, 0.0, 7.3], [0.0, 0.2, 3.1]], np.float32)
    img = _checker()
    plain = warp_image(img, M, (64, 64), antialias=False)
    assert np.array_equal(plain, cv2.warpAffine(img, M, (64, 64), borderValue=(114, 114, 114)))
    inner = (slice(10, 50), slice(10, 50))
    assert warp_image(img, M, (64, 64), antialias=True)[inner].std() < 2.0

    blob = np.full((400, 400, 3), 114, np.uint8)
    cv2.circle(blob, (230, 170), 40, (255, 255, 255), -1)

    def centroid(a):
        """Centre of the blob: weight = departure from the grey that is both border and field."""
        w = np.abs(a[..., 0].astype(np.float64) - 114)
        yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
        return np.array([(xx * w).sum(), (yy * w).sum()]) / w.sum()

    want = np.array([230.0, 170.0]) * 0.2 + [7.3, 3.1]
    np.testing.assert_allclose(centroid(warp_image(blob, M, (96, 96), antialias=True)), want,
                               atol=0.15)
    np.testing.assert_allclose(centroid(warp_image(blob, M, (96, 96), antialias=False)), want,
                               atol=0.15)


def test_letterbox_antialias_is_an_area_shrink():
    """The deployment path (`detect_raw` -> `letterbox`) shrinks with INTER_AREA under
    `antialias`, and keeps its geometry (scale, pad) either way."""
    import cv2

    img = _checker(400)
    a, sa, pa = letterbox(img, (100, 80), antialias=True)
    b, sb, pb = letterbox(img, (100, 80))
    assert (sa, pa) == (sb, pb)
    want = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
    assert np.array_equal(a[:, pa[0]:pa[0] + 80], want)
    assert a[:, pa[0]:pa[0] + 80].std() < b[:, pb[0]:pb[0] + 80].std()


def test_grayscale_prob_one_makes_every_train_view_monochrome(tiny_root):
    """`grayscale_prob = 1` must reach `__getitem__`'s pixels, after every colour op."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), max_frames_per_group=2,
                    augment=True, strong=True, grayscale_prob=1.0)
    for i in range(min(4, len(ds))):
        x = ds[i]['x']
        assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])


@pytest.fixture(scope='module')
def det_scene(tmp_path_factory):
    """A 3-camera session and an untrained YOLOX-Nano (random weights: only the bytes matter)."""
    import conftest as cf
    from tailcyclenet.detector import YOLOXNano
    from tailcyclenet.format import sessions_for

    root = tmp_path_factory.mktemp('det')
    cf._session_3d(root / 'ds' / 'test' / 's', T=4)
    _, sessions = sessions_for(root / 'ds', 'test')
    sess = sessions[0]
    sess.preload()
    return YOLOXNano(n_keypoints=0).eval(), sess, next(iter(sess.groups))


def test_detect_raw_applies_the_crop_rule_to_an_extent_head(det_scene):
    """An `extent` checkpoint's boxes leave `detect_raw` as crop-rule boxes in SOURCE pixels,
    padded by the session's own box source; a `crop` checkpoint's are untouched."""
    from tailcyclenet.detector import detect_raw

    det, sess, gid = det_scene
    det.box_target = 'crop'
    raw, sc, _ = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4,
                            score_thresh=0.0)
    det.box_target, det.min_crop_dim = 'extent', 16
    try:
        for source, pad in (('keypoints', 20), ('instances', 0)):
            got, sc2, _ = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4,
                                     score_thresh=0.0, box_source=source)
            assert np.array_equal(sc, sc2, equal_nan=True)
            assert np.isfinite(raw).any()
            for ci, cam in enumerate(sess.cam_names):
                size = sess.rig.size(cam)
                want = crop_boxes_from_extents(torch.as_tensor(raw[:, :, ci].reshape(-1, 4)),
                                               size, 16, pad).numpy()
                assert np.array_equal(got[:, :, ci].reshape(-1, 4), want, equal_nan=True)
    finally:
        det.box_target = 'crop'


def test_checkpoint_holds_the_weights_that_were_scored(tmp_path, dense_root, monkeypatch):
    """Under schedule-free, `opt.train()` swaps the scored `x` back to `y` IN PLACE right after
    scoring; the checkpoint used to be written after that swap. Record the state the scorer
    actually saw and require `model_state` to equal it, and `model_state_train` to differ."""
    import importlib.util

    spec = importlib.util.spec_from_file_location('tcn_train_detector_scored',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []
    real = mod.score_dataset

    def spy(model, *a, **k):
        """Snapshot the weights at scoring time, then score as usual."""
        seen.append({n: t.detach().clone() for n, t in model.state_dict().items()})
        return real(model, *a, **k)

    monkeypatch.setattr(mod, 'score_dataset', spy)
    out = tmp_path / 'run'
    cfg = tmp_path / 'config.toml'
    cfg.write_text(f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 4
batch_size = 2
num_workers = 0
seed = 0
device = "cpu"
eval_every = 4
eval_batches = 1
warmup_steps = 0
""")
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    ckpt = torch.load(out / 'detector_it000004.pth', map_location='cpu', weights_only=False)
    assert ckpt['model_state_is'] == 'eval' and 'model_state_train' in ckpt
    assert ckpt['box_target'] == 'extent' and ckpt['antialias'] is True
    scored = seen[-1]
    for n, t in ckpt['model_state'].items():
        assert torch.equal(t, scored[n]), f'{n}: the checkpoint is not the scored weight'
    assert any(not torch.equal(t, ckpt['model_state_train'][n])
               for n, t in ckpt['model_state'].items()), 'x == y: the test checked nothing'

    from tailcyclenet.detector import load_detector
    det = load_detector(out, checkpoint='detector_it000004.pth')[0]
    assert det.box_target == 'extent' and det.antialias and det.min_crop_dim == 16
