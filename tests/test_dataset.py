"""The loader, and the one thing in it that must be bit-exact.

The crop rule is shared between the pose model and the detector -- the detector is trained to
reproduce the pose crop -- and `test_crop_rule_is_int32_exact` is what licenses that.
"""
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet.data import crop as cropmod
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.format import MISSING, UNLABELED, VISIBLE, Registry, load_dataset

from .conftest import KPTS_2D


# ----------------------------------------------------------------------------------------------
# the crop rule
# ----------------------------------------------------------------------------------------------

def test_crop_rule_is_int32_exact():
    """`crop_box_for_points` equals the library's `crop_cgroup_to_points` exactly (gotcha 7), and is
    None (not a raise) when nothing is finite so the detector can emit a NaN box."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    shim = SimpleNamespace(min_crop_dim=64)
    for trial in range(200):
        w, h = int(rng.integers(80, 2000)), int(rng.integers(80, 2000))
        n = int(rng.integers(2, 30))
        pts = torch.as_tensor(rng.uniform(-200, max(w, h) + 200, size=(n, 2)), dtype=torch.float32)
        if trial % 7 == 0:                        # some non-finite points, the normal case
            pts[rng.integers(n)] = float('nan')
        size = torch.tensor([w, h], dtype=torch.int32)
        mine = cropmod.crop_box_for_points(pts, size, 64)
        good = torch.all(torch.isfinite(pts), dim=1)
        theirs = _library_box(shim, pts[good], size)
        assert mine is not None and mine.dtype == torch.int32
        assert torch.equal(mine, theirs), f'{w}x{h}: {mine.tolist()} != {theirs.tolist()}'
    pts = torch.full((5, 2), float('nan'))
    assert cropmod.crop_box_for_points(pts, torch.tensor([100, 100], dtype=torch.int32)) is None


def _library_box(self, pflat, size):
    """The library's crop arithmetic, transcribed independently so the two can drift apart."""
    low = torch.clamp(torch.min(pflat, dim=0).values - 20, torch.tensor([0, 0]), size).to(torch.int32)
    high = torch.clamp(torch.max(pflat, dim=0).values + 20, torch.tensor([0, 0]), size).to(torch.int32)
    cw, ch = high[0] - low[0], high[1] - low[1]
    base = max(self.min_crop_dim, int(cw), int(ch))
    min_dim_x, min_dim_y = min(base, int(size[0])), min(base, int(size[1]))
    if cw < min_dim_x:
        cx = (low[0] + high[0]) // 2
        low[0] = torch.clamp(cx - min_dim_x // 2, 0, size[0] - min_dim_x)
        high[0] = low[0] + min_dim_x
    if ch < min_dim_y:
        cy = (low[1] + high[1]) // 2
        low[1] = torch.clamp(cy - min_dim_y // 2, 0, size[1] - min_dim_y)
        high[1] = low[1] + min_dim_y
    return torch.cat([low, high])


def test_jitter_stays_inside_the_image():
    rng = np.random.default_rng(0)
    jit = cropmod.jitter_box(rng, 0.3, 0.3)
    size = torch.tensor([200, 150], dtype=torch.int32)
    for _ in range(200):
        box = torch.tensor([40, 30, 120, 100], dtype=torch.int32)
        out = jit(box, size)
        assert out[0] >= 0 and out[1] >= 0 and out[2] <= 200 and out[3] <= 150
        assert out[2] > out[0] and out[3] > out[1]


# ----------------------------------------------------------------------------------------------
# reading pixels
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('angle', [0.0, 150.0])
def test_the_fused_warp_agrees_with_the_camera(angle):
    """The composed rotate->crop->resize affine lands a painted marker where the camera says it
    goes -- the only check on the `A @ M3` composition."""
    import cv2

    from tailcyclenet import format as fmt
    from tailcyclenet.data.pixels import crop_affine
    from tailcyclenet.data.camera import resize_camera
    from tracktail.datasets.tracktail_dataset import rotate_camera_image_plane_3d
    from tracktail.model.cube import project_points_torch

    W, H = 320, 240
    from aniposelib.cameras import CameraGroup
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False},
                  calibrated={'cam0': True})
    cam = rig.tracktail()[0]

    # a point that projects well inside the frame: f = max(W,H) = 320, principal point (160,120)
    pt = torch.tensor([[[0.4, 0.2, 4.0]]])                      # (T=1, K=1, 3)
    src = project_points_torch([cam], pt)[0][0, 0]
    img = np.zeros((H, W, 3), np.uint8)
    img[int(round(float(src[1]))), int(round(float(src[0])))] = 255

    rotation = None
    if angle:
        cam, rotation = rotate_camera_image_plane_3d(cam, angle)
    box = cropmod.crop_box_for_points(project_points_torch([cam], pt)[0], cam['size'], 96)
    cam = cropmod.apply_crop(cam, box)
    cam, _ = resize_camera(cam, 64)

    want = project_points_torch([cam], pt)[0][0, 0]             # where the camera says it is now
    M, size = crop_affine((W, H), box, cam['size'].tolist(), rotation)
    out = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR)

    # bilinear spreads the marker over up to 4 pixels; its intensity centroid is the position
    ys, xs = np.nonzero(out[..., 0])
    assert xs.size, 'the marker fell outside the crop -- the test setup is wrong, not the code'
    wgt = out[ys, xs, 0].astype(np.float64)
    got = np.array([(xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()])
    assert np.allclose(got, np.asarray(want, np.float64), atol=1.0), f'{got} vs {want}'


@pytest.mark.parametrize('angle', [-128.0])
def test_the_2d_rotation_keeps_every_pixel_and_agrees_with_its_camera(angle):
    """`rotate_2d` loses no source pixel (unlike the library's inscribed crop) and its 2x3 matches
    the camera it returns."""
    from aniposelib.cameras import CameraGroup

    from tailcyclenet import format as fmt
    from tailcyclenet.data.camera import rotate_2d

    W, H = 4696, 2048
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False}, calibrated={'cam0': True})
    cam = rig.tracktail()[0]
    # the four frame corners plus its centre -- if any of these leaves the canvas, an animal can
    corners = torch.tensor([[[0.0, 0.0], [W, 0.0], [W, H], [0.0, H], [W / 2, H / 2]]])

    out, moved, (M, size) = rotate_2d(cam, corners, angle)

    assert tuple(out['size'].tolist()) == tuple(size)
    # TIGHT: the rotated frame's bounding box is the canvas EXACTLY -- nothing lost off an edge and
    # nothing wasted. The inscribed crop fails the first half; a lazily oversized canvas the second.
    box = moved[0, :4]
    np.testing.assert_allclose(box.amin(0).numpy(), [0.0, 0.0], atol=0.51)
    np.testing.assert_allclose(box.amax(0).numpy(), list(size), atol=1.01)
    # The camera's principal point moved by the canvas expansion, which is what keeps the pixels
    # `crop_affine` produces agreed with the camera `apply_crop` hands downstream: the frame
    # centre must still be the canvas centre.
    centre = moved[0, 4].numpy()
    np.testing.assert_allclose(centre, [size[0] / 2, size[1] / 2], atol=1.01)
    shift = (out['mat'][:2, 2] - cam['mat'][:2, 2]).numpy()
    np.testing.assert_allclose(centre - shift, [W / 2, H / 2], atol=1.01)


def test_computed_frame_paths_match_the_listing(dataset_3d, monkeypatch):
    """`read_frames` computes `%06d.<ext>` paths (fixture frames differ per (cam, frame)); a
    repeated index decodes once but is not aliased (cutout writes in place)."""
    import cv2

    from tailcyclenet.data.pixels import load_warps, read_frames

    sess = dataset_3d.sessions['train'][0]
    group = sess.groups['g000']
    cam = sess.cam_names[0]
    kind, src, ext = group.source(cam)
    assert (kind, ext) == ('frames', '.png')
    listed = sorted(f for f in src.iterdir() if f.suffix in ('.png', '.jpg'))
    frames = [3, 0, 2, 0]                       # out of order and repeated, as windows are
    want = [load_warps(str(listed[i]), [(None, None)], None, 1)[0] for i in frames]
    calls, real = [], cv2.imread
    monkeypatch.setattr(cv2, 'imread', lambda p, *a, **k: calls.append(p) or real(p, *a, **k))
    got = read_frames(group, cam, frames)
    for a, b in zip(want, got):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(got[0], got[2])
    assert got[1] is not got[3]
    assert len(calls) == 3, f'decoded {len(calls)} times for three distinct frames'


# ----------------------------------------------------------------------------------------------
# the loader
# ----------------------------------------------------------------------------------------------

def test_pose_dataset_does_not_discover_other_splits(tmp_path):
    """Train/val construction must not parse a malformed test session."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path / 'ds'
    C._session_2d(root / 'train' / 'train_session')
    C._session_2d(root / 'val' / 'val_session')
    test_session = root / 'test' / 'test_session'
    C._session_2d(test_session)
    (test_session / 'calibration.toml').unlink()

    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    train = PoseDataset(root, 'train', cfg)
    val = PoseDataset(root, 'val', cfg, registry=train.registry, train=False)

    assert {s.split for ds in train.datasets for s in ds.sessions['train']} == {'train'}
    assert {s.split for ds in val.datasets for s in ds.sessions['val']} == {'val'}
    assert len(train) > 0 and len(val) > 0


CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)


def _batch(ds, i=0):
    return pose_collate([ds[i]])


def test_item_shapes_and_the_vis_wiring(tiny_root):
    """Item shapes per dimensionality; `vis`/`vis_2d` are both-or-neither in 3D (gotcha 6), `vis` is
    None at R == 2, and keypoints are never filtered (gotcha 4)."""
    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 1
    assert b.views[0].shape[:2] == (1, 4)                 # (B, T, H, W, 3)
    assert b.views[0].dtype == torch.uint8                # `model.forward` divides by 255
    assert b.coords.shape == (1, 4, 4, 2)                 # R=2 for a true-2D session
    assert b.p2d.shape == (1, 1, 4, 4, 2)
    assert b.kpt_ids.shape == (1, 4)
    assert b.kpt_prior.shape == (1, 4, 2)
    assert b.prompt_t.shape == (1, 4)
    assert b.vis_2d is not None and b.vis_2d.shape == (1, 4, 4, 1, 1)
    for i in range(len(ds)):
        assert pose_collate([ds[i]]).vis is None

    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 3
    assert b.coords.shape == (1, 4, 3, 3)                 # R=3 world
    assert b.p2d is None
    assert b.vis.shape == (1, 4, 3, 1)                    # trailing dim get_eval_metrics wants
    assert b.vis_2d.shape == (1, 4, 3, 3, 1)
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        assert (b.vis is None) == (b.vis_2d is None)
        assert b.coords.shape[2] == 3 and b.kpt_ids.shape[1] == 3


def test_missing_2d_point_supervises_occlusion_despite_nan_coords(tiny_root):
    """A `missing` point's target must come from `status`, not from coordinate finiteness, or no
    `missing` row would ever supervise occlusion. Checks the vis_2d VALUE at the two known slots.
    """
    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)
    kpt_missing = KPTS_2D.index('left_ear')
    kpt_unlabeled = KPTS_2D.index('tail_base')

    found_missing = found_unlabeled = False
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        row, T = b.sample_info, b.coords.shape[1]
        start = row['start']
        if row['animal'] == 'a01' and start <= 0 < start + T:
            local = 0 - start
            coord = b.coords[0, local, kpt_missing]
            assert torch.isnan(coord).all(), \
                'the fixture wrote this point missing -> coords must be NaN by spec'
            assert b.vis_2d is not None, 'a missing point is an ASSESSMENT; must not be withheld'
            v = b.vis_2d[0, local, kpt_missing, 0, 0]
            assert v.item() == 0.0, (
                'a MISSING point must supervise occlusion (target 0.0) even though its own '
                'coordinates are NaN -- the target comes from `status`, never from `coords`')
            found_missing = True
        if row['animal'] == 'a02' and start <= 2 < start + T and b.vis_2d is not None:
            local = 2 - start
            v = b.vis_2d[0, local, kpt_unlabeled, 0, 0]
            assert torch.isnan(v), 'an UNASSESSED point must stay NaN, never a fabricated target'
            found_unlabeled = True
    assert found_missing, 'never saw a01 frame 0 -- fixture or window logic changed'
    assert found_unlabeled, 'never saw a02 frame 2 -- fixture or window logic changed'


def test_prob_2d_only_uses_true_2d_targets_on_3d_session(tiny_root):
    """The train-only 2D path on a 3D session: one camera, R=2 targets; stored per-camera labels win,
    projected 3D is the fallback, and a `missing` slot stays NaN."""
    from types import SimpleNamespace

    from tracktail.model.cube import project_points_torch

    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=1.0, aug_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    b = _batch(PoseDataset(tiny_root / 'mouselike', 'train', cfg))
    assert b.coords.shape[-1] == 2 and len(b.views) == 1 and b.p2d is not None
    assert b.sample_info['mode'] == '2d' and b.sample_info['single_view'] is False

    ds = PoseDataset(tiny_root / 'mouselike', 'train',
                     LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0,
                                  aug_prob=0.0, crop_jitter=0.0))
    sess = ds.datasets[0].sessions['train'][0]
    lab = sess.labels(next(iter(sess.groups)))
    frames = np.array([0])
    cgroup = sess.cgroup(next(iter(sess.groups)), frames)
    xyz = torch.as_tensor(lab.points3d[0, 0])
    projected = project_points_torch([cgroup[2]], xyz)[0]
    finite = torch.isfinite(projected).all(-1).nonzero(as_tuple=True)[0]
    assert len(finite) >= 1
    xyz = xyz[int(finite[0])].numpy()
    points3d = np.broadcast_to(xyz, (1, 1, 4, 3)).copy()
    points2d = np.full((1, 1, 4, 3, 2), np.nan, np.float32)
    vis2d = np.full((1, 1, 4, 3), UNLABELED, np.int8)
    vis2d[0, 0, 0, 2] = MISSING
    vis2d[0, 0, 1, 2] = UNLABELED
    vis2d[0, 0, 2, 2] = VISIBLE
    vis2d[0, 0, 3, 2] = VISIBLE
    points2d[0, 0, 3, 2] = [123.0, 124.0]
    hybrid = ds._hybrid_2d_coords(
        SimpleNamespace(points3d=points3d, points2d=points2d, vis2d=vis2d),
        0, frames, cgroup[2], 2)
    assert torch.isnan(hybrid[0, 0]).all()
    assert torch.isfinite(hybrid[0, 1]).all()
    assert torch.isfinite(hybrid[0, 2]).all()
    assert hybrid[0, 3].tolist() == [123.0, 124.0]


def test_cams_to_sample_takes_a_range(tiny_root):
    """`[low, high]` draws a per-item camera count, clamped to what the session has."""
    def counts(spec, n=30):
        cfg = LoaderConfig(n_frames=4, image_size=64, cams_to_sample=spec, prob_2d_only=0.0,
                           aug_prob=0.0, crop_jitter=0.0, prompt_dropout=0.0)
        ds = PoseDataset(tiny_root / 'mouselike', 'train', cfg)
        return [len(ds[i % len(ds)][0]) for i in range(n)]

    assert set(counts([1, 8])) == {1, 2, 3}      # every value occurs, clamped to the session
    assert set(counts([2, 2], 6)) == {2}         # a degenerate range is a fixed count
    assert set(counts(0, 3)) == {3}              # 0 still means "all cameras"


def test_mixed_2d_and_3d_in_one_run(tiny_root):
    """One `train/` may hold both; the registry spans them and ids are disjoint."""
    ds = PoseDataset(tiny_root, 'train', CFG)
    assert {d.name for d in ds.datasets} == {'ratlike', 'mouselike'}
    modes = set()
    # balance_datasets samples the dataset per item, so iterating the index once is not a
    # coverage guarantee -- draw enough that missing one is 2^-30.
    for i in range(30):
        b = pose_collate([ds[i % len(ds)]])
        modes.add(b.sample_info['mode'])
        ids = b.kpt_ids[0].tolist()
        assert ids == list(ds.registry.ids_for_dataset(b.sample_info['dataset']))
    assert modes == {'2d', '3d'}


def test_registry_ids_are_append_only(tiny_root):
    """Ids are APPEND-ONLY against an existing registry (discovery order is not stable), or warm
    start remaps the embedding."""
    from tailcyclenet.format import load_datasets

    ds = load_datasets(tiny_root)
    first = Registry.build(ds)
    assert dict(Registry.build(list(reversed(ds))).datasets) != dict(first.datasets), \
        'the fixture must be order-sensitive, or this test proves nothing'
    again = Registry.build(list(reversed(ds)), first)
    assert dict(again.datasets) == dict(first.datasets)
    assert again.names == first.names

    grown = Registry.build(ds + [_FakeDataset('zzz', ['a', 'b'])], first)
    assert grown.names[:len(first.names)] == first.names
    assert list(grown.ids_for_dataset('zzz')) == [len(first.names), len(first.names) + 1]


class _FakeDataset:
    """The two attributes Registry.build reads. Cheaper than writing a session to disk."""

    def __init__(self, name, names):
        self.name, self.names = name, names


def test_per_camera_augmentation_is_constant_down_a_clip(tiny_root):
    """A camera's appearance holds steady down a clip (an identity cue) while per-image noise varies;
    val pixels are never augmented."""
    from tailcyclenet.dataset import _build_augmenters

    per_camera, per_image = _build_augmenters(
        LoaderConfig(aug_prob=1.0, per_image_aug_prob=1.0))
    img = (np.mgrid[0:64, 0:64][0][..., None] * np.ones((1, 1, 3))).astype(np.uint8)

    det = per_camera.to_deterministic()
    same = [det(image=img.copy()) for _ in range(4)]
    for s in same[1:]:
        np.testing.assert_array_equal(same[0], s)
    assert not np.array_equal(same[0], img), 'per-camera pipeline did nothing at aug_prob=1'

    varied = [per_image(image=img.copy()) for _ in range(4)]
    assert any(not np.array_equal(varied[0], v) for v in varied[1:])

    aug = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=1.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    train_ds = PoseDataset(tiny_root / 'ratlike', 'train', aug)
    assert PoseDataset(tiny_root / 'ratlike', 'val', aug)._aug is None
    plain = _batch(PoseDataset(tiny_root / 'ratlike', 'train', CFG))
    assert not torch.equal(plain.views[0], _batch(train_ds).views[0])


def test_invert_augmentation_complements_pixels(tiny_root):
    """Color inversion maps every uint8 channel to its black/white complement."""
    class _Identity:
        def to_deterministic(self):
            return self

        def __call__(self, *, image):
            return image

    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                       per_image_aug_prob=0.0, grayscale_prob=0.0, invert_prob=1.0,
                       crop_jitter=0.0)
    ds = PoseDataset(tiny_root / 'ratlike', 'train', cfg)
    # Keep this unit test focused on inversion rather than imgaug's independent appearance draws.
    ds._aug = (_Identity(), _Identity())
    imgs = [np.array([[[0, 1, 127], [128, 254, 255]]], dtype=np.uint8)]
    got = ds._augment([imgs[0].copy()], 0, torch.tensor([2, 1]), None, None, False,
                      np.random.default_rng(0), invert=True)
    np.testing.assert_array_equal(got[0], 255 - imgs[0])
    assert got[0].dtype == np.uint8


def test_cutout_marks_covered_keypoints_not_visible():
    """A keypoint under a cutout rect must be labelled not-visible, including where it was NaN."""
    from tailcyclenet.dataset import _cutout_rects

    rng = np.random.default_rng(0)
    T, K, C = 4, 6, 2
    size = torch.tensor([200, 160], dtype=torch.int32)
    p2d = torch.stack([torch.as_tensor(rng.uniform([0, 0], [200, 160], (T, K, 2)),
                                       dtype=torch.float32) for _ in range(C)])
    vis_2d = torch.ones((T, K, C))
    vis_2d[0, 0, 0] = float('nan')                 # never assessed by this camera
    covered_nan = False
    for _ in range(40):                            # rects are random; drive it until NaN is hit
        v = vis_2d.clone()
        rects = _cutout_rects(rng, size, p2d, v, 0)
        assert rects, 'cutout produced no rectangles'
        for x1, y1, x2, y2, fill in rects:
            assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 160
            assert len(fill) == 3 and all(0 <= c < 256 for c in fill)
            inside = ((p2d[0, ..., 0] >= x1) & (p2d[0, ..., 0] <= x2) &
                      (p2d[0, ..., 1] >= y1) & (p2d[0, ..., 1] <= y2))
            assert (v[..., 0][inside] == 0).all()
            if inside[0, 0]:
                covered_nan = True
        assert torch.isnan(v[..., 1]).sum() == 0 and (v[..., 1] == 1).all()   # other cam untouched
    assert covered_nan, 'never covered the NaN keypoint -- the NaN -> 0 path went untested'


@pytest.mark.timeout(60)
def test_workers_do_not_share_a_random_stream(tiny_root):
    """Two workers must not produce identically-augmented items: torch reseeds per worker, but
    imgaug's own global RNG is fork-copied and needs `worker_init`.
    """
    from tailcyclenet.dataset import worker_init

    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=1.0,
                       per_image_aug_prob=0.0, grayscale_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    ds = PoseDataset(tiny_root / 'mouselike', 'train', cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=2, collate_fn=pose_collate,
        sampler=[0, 0, 0, 0], worker_init_fn=worker_init)
    got = [(float(b.cgroup[0]['ext'].sum()), b.views[0].float().mean().item())
           for b in loader]
    # items 0 and 1 land on different workers; equality would mean one shared stream
    assert got[0][0] != pytest.approx(got[1][0]), 'np.random is shared across workers'
    assert got[0][1] != pytest.approx(got[1][1]), 'imgaug RNG is shared across workers'


def test_prompt_is_the_first_labelled_frame(tiny_root, centred_root):
    """prompt_t is the first LABELLED frame, not 0 (> 0 on 19.5% of rat-city windows)."""
    b = _batch(PoseDataset(tiny_root / 'ratlike', 'val', CFG))
    finite = torch.isfinite(b.coords[0]).all(-1)       # (T,K)
    for k in range(finite.shape[1]):
        if finite[:, k].any():
            assert b.prompt_t[0, k] == int(finite[:, k].float().argmax())
            torch.testing.assert_close(b.kpt_prior[0, k],
                                       b.coords[0, b.prompt_t[0, k].long(), k])
    cfg = LoaderConfig(n_frames=24, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    b = _batch(PoseDataset(centred_root, 'train', cfg, train=False))
    assert (b.prompt_t[0] == 11).all() and torch.isfinite(b.kpt_prior[0]).all()


def test_prompt_dropout_is_per_item_not_per_keypoint(tiny_root):
    """`prompt_dropout` is the fraction of STEPS run query-free: every window is all-NaN or no-NaN.
    Per-keypoint draws would make the query-free forward that val scores exponentially rare."""
    def cfg(p):
        return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=p, prob_2d_only=0.0)

    ds5 = PoseDataset(tiny_root / 'ratlike', 'train', cfg(0.5))
    seen = set()
    for _ in range(24):
        nan = torch.isnan(_batch(ds5).kpt_prior)
        assert nan.all() or not nan.any(), 'dropout mixed within one window -> it is per-keypoint'
        seen.add(bool(nan.all()))
    assert seen == {True, False}, f'p=0.5 never produced both outcomes in 24 draws: {seen}'


def test_window_is_sized_to_the_labelled_span(tiny_root, centred_root):
    """T is derived from the labels (`n_frames` is its ceiling) and covers them; never 1 (gotcha 1):
    `n_frames = 1` is refused and a short group yields its own length."""
    with pytest.raises(AssertionError, match='n_frames'):
        PoseDataset(tiny_root / 'ratlike', 'train',
                    LoaderConfig(n_frames=1, image_size=64, aug_prob=0.0, crop_jitter=0.0))
    cfg = LoaderConfig(n_frames=8, image_size=64, aug_prob=0.0, crop_jitter=0.0)
    assert _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg)).views[0].shape[1] == 4

    cfg = LoaderConfig(n_frames=24, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg)
    for _ in range(10):
        b = _batch(ds)
        assert b.views[0].shape[1] == 4
        assert set(range(11, 14)) <= set(b.fnums[0].tolist()), b.fnums[0].tolist()


def test_frame_strides_widen_the_window_on_the_lattice(dense_root, centred_root):
    """A strided train window must stay ON its lattice, in the group, and over its anchor label."""
    def cfg(strides, n=8):
        return LoaderConfig(n_frames=n, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=0.0, frame_strides=strides)

    # 3 as well as 4: an odd stride does not divide the group, so the anchor's lattice offset eats
    # into the room at the end of it.
    for s in (4, 3):
        ds = PoseDataset(dense_root, 'train', cfg([s]))
        for _ in range(10):
            f = pose_collate([ds[0]]).fnums[0].numpy()
            assert list(np.diff(f)) == [s] * (len(f) - 1), f  # no clamping, no off-lattice start
            assert f.max() < 32 and f.min() >= 0, f

    # A one-label lattice: labels sit at 11-13, so at stride 4 only the anchor is reachable and T
    # falls to the floor of 2 -- more motion for the same two encodes. This is the allen shape.
    mid = PoseDataset(centred_root, 'train', cfg([4], n=24))
    for _ in range(5):
        f = pose_collate([mid[0]]).fnums[0].numpy()
        assert len(f) == 2 and f[1] - f[0] == 4, f
        assert set(f) & {11, 12, 13}, f

    val = PoseDataset(dense_root, 'train', cfg([4]), train=False)
    assert list(np.diff(pose_collate([val[0]]).fnums[0].numpy())) == [1] * 7


def test_visibility_stays_three_state(tiny_root):
    """"Not assessed" reaches the loss as NaN, never as 0, and the installed tracktail keeps the
    loss and gradient finite on a NaN target (>= 0.3.2)."""
    from tracktail.model.losses import BCELossVis

    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    saw_unassessed = False
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        if b.vis_2d is None:
            continue
        finite = b.vis_2d[torch.isfinite(b.vis_2d)]
        assert set(finite.unique().tolist()) <= {0.0, 1.0}, 'assessed entries are 0 or 1'
        saw_unassessed |= bool(torch.isnan(b.vis_2d).any())
    assert saw_unassessed, 'the 3D fixture carries unassessed entries; they must survive as NaN'

    pred = torch.zeros(2, 4, 3, 1, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 3, 1)).float()
    target[0, 1, 2, 0] = float('nan')
    loss = BCELossVis(weight=1.0)(pred, target, device='cpu')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_no_visibility_supervision_without_ground_truth(tracked_no_assessment_root):
    """A session with no `missing` row anywhere withholds the whole vis target (session-level gate)."""
    b = _batch(PoseDataset(tracked_no_assessment_root, 'train', CFG))
    assert b.vis is None and b.vis_2d is None


# ----------------------------------------------------------------------------------------------
# a label in the middle of a group must be usable -- on BOTH paths
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('train,n_frames', [(True, 8), (False, 24)])
def test_centred_labels_are_usable(centred_root, train, n_frames):
    """A window need not START on a label: every window contains one, not always at frame 0."""
    cfg = LoaderConfig(n_frames=n_frames, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=train)
    assert len(ds) > 0, 'a group whose labels are centred must still produce windows'
    n = 20 if train else len(ds)
    at_frame_zero = 0
    for i in range(n):
        finite = torch.isfinite(pose_collate([ds[i % len(ds)]]).coords[0]).all(-1)   # (T,K)
        assert finite.any(), 'every window must contain at least one labelled frame'
        at_frame_zero += int(finite[0].any())
    assert at_frame_zero < n, 'the label must not always land on frame 0'


def test_val_windows_are_deterministic_and_unpadded(tiny_root, centred_root):
    """A val item is reproducible, and a val start never clamp-pads past `n_frames - T`."""
    ds = PoseDataset(tiny_root / 'ratlike', 'val', CFG)
    a, b = _batch(ds), _batch(ds)
    torch.testing.assert_close(a.coords, b.coords, equal_nan=True)
    torch.testing.assert_close(a.views[0], b.views[0])

    cfg = LoaderConfig(n_frames=24, image_size=32, aug_prob=0.0, crop_jitter=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=False)
    for item in ds.index:
        assert 0 <= item.start <= max(0, item.session.groups[item.gid].n_frames - 24)
    b = _batch(ds)
    assert b.sample_info['start'] == 0
    assert b.fnums[0].tolist() == list(range(24)), 'no duplicated frames'


def test_projected_carries_position_but_trains_no_visibility(projected_root):
    """`projected` is a position with no visibility claim: coords arrive, no vis target is built."""
    from tailcyclenet import format as fmt

    sess = fmt.Session.load(projected_root / 'train' / 's')
    assert not [e for e in fmt.validate_session(sess) if 'WARNING' not in e]
    lab = sess.labels('g000')
    assert (lab.vis2d == fmt.PROJECTED).all()
    assert np.isfinite(lab.points2d).all(), 'a projected row carries its position'
    cfg = LoaderConfig(n_frames=2, image_size=32, aug_prob=0.0, crop_jitter=0.0)
    b = pose_collate([PoseDataset(projected_root, 'train', cfg, train=False)[0]])
    assert b.vis is None and b.vis_2d is None, 'no visibility target may be built from projected'
    assert torch.isfinite(b.coords).any(), 'the 3D targets still arrive'


# ----------------------------------------------------------------------------------------------
# the training mix
# ----------------------------------------------------------------------------------------------

def test_sampling_mix_is_two_level_and_skips_absent_levels(mixed_source_root):
    """`annot_frac` then `mode_3d_frac` WITHIN each source (absent cells skipped), and the realised
    draw matches -- the old index-count weighting starved short sources."""
    cfg = LoaderConfig(n_frames=2, image_size=32, annot_frac=0.4, mode_3d_frac=0.7)
    ds = PoseDataset(mixed_source_root, 'train', cfg)
    want = {'2d-annotated': 0.4 * 0.3, '3d-annotated': 0.4 * 0.7, '3d-tracked': 0.6}
    assert ds.mix() == pytest.approx(want, abs=1e-9)

    rng = np.random.default_rng(0)
    n = 5000
    drawn = Counter()
    for i in range(n):
        s = ds._pick(i % len(ds.index), rng).session
        drawn[f'{s.mode}-{s.label_source}'] += 1
    for k, p in want.items():
        assert drawn[k] / n == pytest.approx(p, abs=0.03), f'{k}: {drawn[k] / n:.3f} != {p}'


def test_sampling_mix_defaults_to_the_uniform_draw(mixed_source_root):
    """Unset fractions reproduce the uniform draw exactly; one level set leaves the other natural;
    a fraction outside [0, 1] is refused."""
    cfg = LoaderConfig(n_frames=2, image_size=32)
    ds = PoseDataset(mixed_source_root, 'train', cfg)
    assert ds._pools[0][1] is None, 'no weights should be built when neither fraction is set'
    rng = np.random.default_rng(0)
    assert [ds._pick(i, rng) for i in range(len(ds.index))] == ds.index

    only_src = PoseDataset(mixed_source_root, 'train',
                           LoaderConfig(n_frames=2, image_size=32, annot_frac=0.4)).mix()
    assert only_src['3d-tracked'] == pytest.approx(0.6)
    assert only_src['2d-annotated'] + only_src['3d-annotated'] == pytest.approx(0.4)
    assert only_src['2d-annotated'] > only_src['3d-annotated']    # 2 animals vs 1

    with pytest.raises(ValueError, match=r'annot_frac must be in \[0, 1\]'):
        PoseDataset(mixed_source_root, 'train',
                    LoaderConfig(n_frames=2, image_size=32, annot_frac=1.4))


def test_prompt_noise_is_in_pixels_on_both_modes(tiny_root):
    """One sigma is the same VISUAL displacement on a px session and a mm one (3D sigma is scaled by
    `cube_scale`); NaN + noise stays NaN, so a withheld prior stays withheld."""
    from tracktail.model.cube import get_camera_scale

    SIG = 3.0

    def mk(s, drop=0.0):
        return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=drop, prompt_noise_px=s, prob_2d_only=0.0)

    def spread(root, sigma):
        clean = _batch(PoseDataset(root, 'train', mk(0.0))).kpt_prior
        ds = PoseDataset(root, 'train', mk(sigma))
        d = torch.stack([(_batch(ds).kpt_prior - clean).abs().median() for _ in range(15)])
        return float(d.median())

    px = spread(tiny_root / 'ratlike', SIG)          # mode='2d', units='px'
    mm = spread(tiny_root / 'mouselike', SIG)        # mode='3d', units='mm'
    assert 0.3 * SIG < px < 3 * SIG, px
    b = _batch(PoseDataset(tiny_root / 'mouselike', 'train', mk(0.0)))
    scale = float(torch.nanmedian(get_camera_scale(b.cgroup, b.kpt_prior)))
    assert scale > 0
    assert 0.3 * SIG < mm / scale < 3 * SIG, (mm, scale, mm / scale)
    dropped = _batch(PoseDataset(tiny_root / 'ratlike', 'train', mk(5.0, 1.0))).kpt_prior
    assert torch.isnan(dropped).all()


# ----------------------------------------------------------------------------------------------
# box_source: cropping on instances.pq instead of the labels
# ----------------------------------------------------------------------------------------------

# min_crop_dim 8, not the default 64: the fixture image is 64x48, so a 64 px floor would force
# every crop to the whole frame and the test would pass whatever the box source did.
BOXCFG = dict(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
              crop_jitter=0.0, prompt_dropout=0.0, min_crop_dim=8)


def _boxed_animal(root, source):
    """The val window for a02 -- the one animal the 2D fixture gives a stored box."""
    ds = PoseDataset(root / 'ratlike', 'val', LoaderConfig(box_source=source, **BOXCFG))
    a = next(i for i, it in enumerate(ds.index) if it.animal == 1)
    return _batch(ds, a)


def test_box_source_instances_crops_on_the_stored_box(tiny_root):
    """The stored extent (a02: [10,10,30,30], keypoints over [5,43]) decides the crop, at pad=0."""
    b = _boxed_animal(tiny_root, 'instances')
    box = cropmod.crop_box_for_points(torch.tensor([[10.0, 10.0], [30.0, 30.0]]),
                                      torch.tensor([64, 48], dtype=torch.int32), 8, pad=0)
    assert box.tolist() == [10, 10, 30, 30], 'a padded extent must not be padded a second time'
    torch.testing.assert_close(b.cgroup[0]['offset'], torch.tensor([10.0, 10.0]) * (64 / 20))
    assert not torch.allclose(b.cgroup[0]['offset'],
                              _boxed_animal(tiny_root, 'keypoints').cgroup[0]['offset'])


def test_box_source_falls_back_per_view(tiny_root):
    """An animal with no stored box trains EXACTLY as under `keypoints` (what makes `instances` safe
    to set across mixed roots); a typo'd source is refused."""
    def a01(source):
        ds = PoseDataset(tiny_root / 'ratlike', 'val',
                         LoaderConfig(box_source=source, **BOXCFG))
        a = next(i for i, it in enumerate(ds.index) if it.animal == 0)
        return _batch(ds, a)

    got, want = a01('instances'), a01('keypoints')
    torch.testing.assert_close(got.coords, want.coords, equal_nan=True)
    torch.testing.assert_close(got.cgroup[0]['offset'], want.cgroup[0]['offset'])
    np.testing.assert_array_equal(got.views[0].numpy(), want.views[0].numpy())
    with pytest.raises(AssertionError, match='box_source'):
        PoseDataset(tiny_root / 'ratlike', 'val', LoaderConfig(box_source='keypoint', **BOXCFG))


def test_crop_inflate_default_is_inert(tiny_root):
    """`crop_inflate = 1.0` (the default) is BYTE-IDENTICAL to a config without the key."""
    for root, split in ((tiny_root / 'ratlike', 'val'), (tiny_root / 'mouselike', 'train')):
        base = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.25,
                            crop_jitter=0.3, prompt_dropout=0.4)
        b = _batch(PoseDataset(root, split, base, train=False))
        e = _batch(PoseDataset(root, split, replace(base, crop_inflate=1.0), train=False))
        torch.testing.assert_close(b.coords, e.coords, equal_nan=True)
        np.testing.assert_array_equal(b.views[0].numpy(), e.views[0].numpy())
        torch.testing.assert_close(b.cgroup[0]['offset'], e.cgroup[0]['offset'])
        torch.testing.assert_close(b.cgroup[0]['size'].float(), e.cgroup[0]['size'].float())


def test_crop_inflate_widens_the_crop(tiny_root):
    """Above 1.0 both crop paths widen the box about its own centre (3D: on the real fixture rig)."""
    from tailcyclenet.format import load_dataset

    cam = {'size': torch.tensor([2000, 2000], dtype=torch.int32), 'offset': torch.zeros(2)}
    coords = torch.tensor([[900.0, 900.0], [1100.0, 1100.0]])
    _, tight_box, _ = cropmod.crop_to_points_2d(cam, coords, min_crop_dim=16, inflate=1.0)
    _, wide_box, _ = cropmod.crop_to_points_2d(cam, coords, min_crop_dim=16, inflate=1.5)
    tw, ww = tight_box[2] - tight_box[0], wide_box[2] - wide_box[0]
    assert float(ww) > float(tw) * 1.2
    tc = (float(tight_box[0]) + float(tight_box[2])) / 2
    wc = (float(wide_box[0]) + float(wide_box[2])) / 2
    assert abs(tc - wc) < float(tw), 'inflation is about the SAME centre'

    sess = load_dataset(tiny_root / 'mouselike').sessions['train'][0]
    sess.preload()
    frames = np.arange(4)
    cgroup = sess.cgroup('g000', frames)
    coords = torch.as_tensor(sess.labels('g000').points3d[0][frames], dtype=torch.float32)
    _, tight = cropmod.crop_to_points_3d(cgroup, coords, min_crop_dim=8, inflate=1.0)
    _, wide = cropmod.crop_to_points_3d(cgroup, coords, min_crop_dim=8, inflate=1.5)
    grew = False
    for c in range(len(cgroup)):
        tw, ww = tight[c][2] - tight[c][0], wide[c][2] - wide[c][0]
        th, wh = tight[c][3] - tight[c][1], wide[c][3] - wide[c][1]
        assert float(ww) >= float(tw) and float(wh) >= float(th)
        # the 64x48 fixture frame clamps the short axis, so require growth somewhere
        grew = grew or float(ww) > float(tw) * 1.1 or float(wh) > float(th) * 1.1
    assert grew, 'crop_inflate = 1.5 produced no wider box on any camera/axis'


def test_crop_inflate_draw(tiny_root):
    """A scalar is returned as-is; `[low, high]` draws in range on train and is the exact midpoint on
    val (so `checkpoint_best` reads one geometry) -- also end to end through a real item."""
    from tailcyclenet.dataset import _crop_inflate

    rng = np.random.default_rng(0)
    cfg = LoaderConfig(crop_inflate=1.3)
    assert _crop_inflate(cfg, rng, True) == _crop_inflate(cfg, rng, False) == 1.3
    cfg = LoaderConfig(crop_inflate=[0.9, 1.5])
    train_draws = [_crop_inflate(cfg, rng, True) for _ in range(50)]
    assert all(0.9 <= v <= 1.5 for v in train_draws) and len(set(train_draws)) > 1
    assert _crop_inflate(cfg, rng, False) == pytest.approx(1.2)

    range_cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                             crop_jitter=0.0, prompt_dropout=0.0, min_crop_dim=8,
                             crop_inflate=[0.9, 1.8])

    class _FixedRng:
        """`uniform` returns a range endpoint (the small frame would otherwise often clamp)."""
        def __init__(self, value, inner):
            self._value, self._inner = value, inner

        def uniform(self, lo, hi):
            return self._value

        def __getattr__(self, name):
            return getattr(self._inner, name)

    train_ds = PoseDataset(tiny_root / 'ratlike', 'train', range_cfg, train=True)
    scales = []          # `mat[0,0]`: smaller for a wider source box (size is always image_size)
    for end in (0.9, 1.8):
        item = train_ds._item(0, _FixedRng(end, np.random.default_rng(0)))
        scales.append(float(pose_collate([item]).cgroup[0]['mat'][0, 0]))
    assert scales[1] < scales[0], scales
    range_val = _batch(PoseDataset(tiny_root / 'ratlike', 'val', range_cfg, train=False))
    midpoint_val = _batch(PoseDataset(tiny_root / 'ratlike', 'val',
                                      replace(range_cfg, crop_inflate=1.35), train=False))
    torch.testing.assert_close(range_val.cgroup[0]['mat'][0, 0], midpoint_val.cgroup[0]['mat'][0, 0])


# ----------------------------------------------------------------------------------------------
# the reader cache size
# ----------------------------------------------------------------------------------------------

# `reader_cache_size` is pure given `ram_gb`, which is the whole point: the sizing rule is
# asserted here on any host, with no video, no decode and no GPU.

def test_reader_cache_size(monkeypatch):
    """One process caches the whole rig; a worker wants ChunkShuffle.mix (4), not the rig; RAM clamps
    before the OOM killer, never below 1; and the rule opens nothing (gotcha 10)."""
    import builtins

    from tailcyclenet.data.reader_cache import reader_cache_size

    def boom(*a, **k):
        raise AssertionError('the sizing rule must not open anything')

    monkeypatch.setattr(builtins, 'open', boom)
    monkeypatch.setattr('os.stat', boom)
    assert reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 16
    assert reader_cache_size(1, (1024, 570), None, ram_gb=503) == 4   # narrow rig still gets mix
    assert reader_cache_size(16, (3208, 2200), 12, ram_gb=503) == 4   # not x worker count
    assert reader_cache_size(16, (3208, 2200), 12, ram_gb=32) < 4
    assert reader_cache_size(16, (3208, 2200), None, ram_gb=8) < 16
    assert reader_cache_size(16, (3208, 2200), None, ram_gb=64) == 16
    assert reader_cache_size(16, (3208, 2200), 64, ram_gb=1) == 1


def test_the_advertised_ram_actually_buys_every_reader():
    """The `--max-ram` the warning advertises buys a reader per camera, and is not padded."""
    from tailcyclenet import memory
    from tailcyclenet.data.reader_cache import reader_cache_size, reader_cache_ram_gb

    n_cams, wh = 16, (3208, 2200)                                     # johnson
    need = reader_cache_ram_gb(n_cams, wh)
    at = reader_cache_size(n_cams, wh, None, ram_gb=memory.host_budget(override_gb=need).budget_gb)
    assert at >= n_cams, f'--max-ram {need:.1f} advertised for {n_cams} cameras but bought {at}'
    below = reader_cache_size(
        n_cams, wh, None, ram_gb=memory.host_budget(override_gb=need - 1).budget_gb)
    assert below < n_cams, f'{need:.1f} GB is padded: {need - 1:.1f} already buys {below}'


def test_the_reader_price_is_linear_not_quadratic():
    """Doubling the budget doubles the count; the old quadratic law capped a 16-camera rig at 10."""
    from tailcyclenet.data.reader_cache import reader_cache_size

    a = reader_cache_size(64, (3208, 2200), None, ram_gb=16)
    b = reader_cache_size(64, (3208, 2200), None, ram_gb=32)
    assert b / a == pytest.approx(2.0, rel=0.2), f'{a} -> {b} is not linear in the budget'
    assert reader_cache_size(16, (3208, 2200), None, ram_gb=32) == 16


def test_reader_cache_env_var_is_deleted_and_raises_by_name(monkeypatch):
    """The override is gone; `--max-ram` is the only knob, and a stale export must not be ignored."""
    from tailcyclenet.data.reader_cache import reader_cache_size

    monkeypatch.setenv('TAILCYCLENET_READER_CACHE', '1')
    with pytest.raises(SystemExit, match='TAILCYCLENET_READER_CACHE is deleted'):
        reader_cache_size(16, (3208, 2200), None, ram_gb=503)


def test_a_whole_body_offset_moves_every_keypoint_the_same_way(tiny_root):
    """`prompt_offset_px` is ONE vector for the whole pose (a deployment lag), unlike i.i.d. jitter."""
    def cfg(**kw):
        return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=0.0, prob_2d_only=0.0, **kw)

    clean = _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg())).kpt_prior[0]
    off = _batch(PoseDataset(tiny_root / 'ratlike', 'train',
                             cfg(prompt_offset_px=8.0))).kpt_prior[0]
    ok = torch.isfinite(clean).all(-1) & torch.isfinite(off).all(-1)
    assert ok.sum() >= 2, 'need at least two real priors to compare displacements'
    d = (off - clean)[ok]
    assert d.abs().max() > 1e-4, 'the offset must actually move the prior'
    # ONE vector for the whole pose: every keypoint's displacement is identical.
    torch.testing.assert_close(d, d[:1].expand_as(d))
    # ...where the i.i.d. jitter is not.
    jit = _batch(PoseDataset(tiny_root / 'ratlike', 'train',
                             cfg(prompt_noise_px=8.0))).kpt_prior[0]
    dj = (jit - clean)[torch.isfinite(clean).all(-1) & torch.isfinite(jit).all(-1)]
    assert not torch.allclose(dj, dj[:1].expand_as(dj))


def test_a_stale_prior_is_a_wrong_position_not_a_withdrawn_one(tiny_root):
    """A carried prior degrades into an earlier frame's pose when the box source loses an animal;
    swapping in NaN instead would just be `prompt_dropout` again.
    """
    def cfg(**kw):
        return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=0.0, prob_2d_only=0.0, **kw)

    base = _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg()))
    moved = False
    for _ in range(10):
        b = _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg(prompt_stale_frames=3)))
        assert torch.isnan(b.kpt_prior).sum() == torch.isnan(base.kpt_prior).sum(), \
            'staleness must not withhold a prior'
        # `prompt_t` still claims the first labelled frame; only the POSE came from elsewhere.
        assert (b.prompt_t == base.prompt_t).all()
        moved |= not torch.allclose(b.kpt_prior, base.kpt_prior, equal_nan=True)
    assert moved, 'prompt_stale_frames=3 never moved the prior in 10 draws'


def test_an_extreme_aspect_crop_never_resizes_to_a_zero_side():
    """`cv2.warpAffine` with a 0 in `dsize` returns the FULL SOURCE FRAME and does not raise, so
    a camera rounded to [256, 0] would hand the model silent garbage instead of a clean failure.
    """
    import cv2

    from tailcyclenet.data.camera import resize_camera

    cam = {'size': torch.tensor([2000, 1], dtype=torch.int32),
           'mat': torch.eye(3), 'offset': torch.zeros(2)}
    out, _ = resize_camera(cam, 256)
    assert out['size'].tolist() == [256, 1], f'a zero side survived: {out["size"].tolist()}'

    # The reason it matters, pinned on OpenCV itself so it stays true if the behaviour ever moves.
    src = np.zeros((480, 640, 3), np.uint8)
    aff = np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    assert cv2.warpAffine(src, aff, (256, 0)).shape == src.shape, \
        'if OpenCV ever raises on a zero dsize, the clamp above is belt-and-braces'
    assert cv2.warpAffine(src, aff, (256, 1)).shape == (1, 256, 3)


def test_a_3d_session_without_points3d_is_triangulated_at_load(tmp_path):
    """Stored 2D positions recover the 3D window targets when points3d.pq is absent."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf
    from tailcyclenet import format as fmt

    root = tmp_path / 'ds'
    original = cf._session_3d(root / 'train' / 's_3d')
    known = original.points3d.copy()
    path = root / 'train' / 's_3d'
    rig = fmt.load_calibration(path / 'calibration.toml')
    original.points2d = np.full((1, 4, len(cf.KPTS_3D), 3, 2), np.nan, np.float32)
    for c, cam in enumerate(rig.cameras):
        projected = cam.project(original.points3d[0].reshape(-1, 3)).detach().cpu().numpy()
        projected = projected.reshape(4, -1, 2)
        positioned = np.isin(original.vis2d[0, :, :, c], (fmt.VISIBLE, fmt.PROJECTED))
        original.points2d[0, :, :, c] = np.where(
            positioned[..., None], projected, np.nan)
    original.points3d = original.vis3d = None
    fmt.write_session(path, mode='3d', units='mm', label_source='tracked',
                      names=cf.KPTS_3D, rig=rig,
                      groups={'g000': fmt.Group('g000', 4)}, labels={'g000': original})
    # Re-write emits keypoints.pq only; explicitly pin the input's missing 3D table.
    (path / 'points3d.pq').unlink(missing_ok=True)

    ds = PoseDataset(root, 'train', LoaderConfig(n_frames=2, image_size=32, aug_prob=0.0,
                                                 crop_jitter=0.0), train=False)
    lab = ds.index[0].session.labels('g000')
    assert lab.points3d is not None and np.isfinite(lab.points3d).any()
    valid = np.isfinite(known).all(-1) & (lab.vis3d == fmt.VISIBLE)
    assert np.allclose(lab.points3d[valid], known[valid], atol=1e-2)
    batch = pose_collate([ds[0]])
    assert torch.isfinite(batch.coords).any(), 'triangulated 3D targets must reach windows'


def test_a_session_with_no_pose_table_is_skipped_not_refused(tmp_path):
    """No label table is required (format §3): a boxes-only or unlabelled session in the split
    contributes no windows, and the labelled sessions beside it still train.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    root = tmp_path / 'ds'
    cf._session_2d(root / 'train' / 's_lab')
    cf._session_2d(root / 'train' / 's_box')
    (root / 'train' / 's_box' / 'keypoints.pq').unlink()
    cf._session_2d(root / 'train' / 's_none')
    for f in ('keypoints.pq', 'instances.pq'):
        (root / 'train' / 's_none' / f).unlink()

    ds = PoseDataset(root, 'train', CFG, train=False)
    assert ds.index
    assert {it.session.path.name for it in ds.index} == {'s_lab'}


# the two prior corruptions: a swapped keypoint pair, and a jump to a nearby animal's pose.
# Both are train-only, off by default, and gated so a default config never draws an extra `rng`
# value.


def _train_item(ds, idx=0, seed=0):
    """One deterministic TRAIN item via `_item` (3D also draws from the global `np.random`)."""
    np.random.seed(seed)
    return ds._item(idx, np.random.default_rng(seed))


def test_prompt_swaps_default_off_is_byte_identical(tiny_root):
    """Swap keys at 0.0 (the defaults) are BYTE-IDENTICAL to a config without them, and a seeded
    train item reproduces itself on both dimensionalities."""
    for root, split in ((tiny_root / 'ratlike', 'val'), (tiny_root / 'mouselike', 'train')):
        base = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.25,
                            crop_jitter=0.3, prompt_dropout=0.4)
        explicit = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.25,
                                crop_jitter=0.3, prompt_dropout=0.4, prompt_swap_kpt_pairs=0.0,
                                prompt_swap_animal=0.0)
        ds_base = PoseDataset(root, split, base, train=False)
        ds_explicit = PoseDataset(root, split, explicit, train=False)
        b = _batch(ds_base)
        e = _batch(ds_explicit)
        torch.testing.assert_close(b.coords, e.coords, equal_nan=True)
        torch.testing.assert_close(b.kpt_prior, e.kpt_prior, equal_nan=True)
        torch.testing.assert_close(b.prompt_t, e.prompt_t)
        np.testing.assert_array_equal(b.views[0].numpy(), e.views[0].numpy())

    # TRAIN mode too, where the new coin draws are actually gated (`self.train`-guarded): a
    # config that never mentions the keys must still draw exactly what it always drew, on both
    # dimensionalities, so a run that predates this change replays identically.
    for root in (tiny_root / 'ratlike', tiny_root / 'mouselike'):
        base_ds = PoseDataset(root, 'train', CFG, train=True)
        b = pose_collate([_train_item(base_ds)])
        e = pose_collate([_train_item(base_ds)])          # same seed twice: must reproduce itself
        torch.testing.assert_close(b.kpt_prior, e.kpt_prior, equal_nan=True)
        torch.testing.assert_close(b.coords, e.coords, equal_nan=True)


def test_animal_swap_is_a_no_op_when_no_neighbour_is_eligible(tmp_path):
    """One animal (structural, consumes no rng) or a neighbour outside every camera: the prior and
    crop stay bit-identical rather than a partially-NaN'd lie."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    cf._session_3d(tmp_path / 'one' / 'train' / 's', T=6)
    cf._session_3d_multi(tmp_path / 'far' / 'train' / 's', T=6, sep=1e6)
    for root in (tmp_path / 'one', tmp_path / 'far'):
        b = pose_collate([_train_item(PoseDataset(root, 'train', CFG, train=True))])
        s = pose_collate([_train_item(PoseDataset(root, 'train',
                                                  replace(CFG, prompt_swap_animal=1.0),
                                                  train=True))])
        torch.testing.assert_close(b.kpt_prior, s.kpt_prior, equal_nan=True)
        torch.testing.assert_close(b.coords, s.coords, equal_nan=True)
        torch.testing.assert_close(b.cgroup[0]['mat'], s.cgroup[0]['mat'])


def test_animal_swap_changes_only_the_prior(tiny_root):
    """`prompt_swap_animal = 1.0` touches `kpt_prior` and nothing else (no neighbour leaks into the crop)."""
    root = tiny_root / 'ratlike'
    base_ds = PoseDataset(root, 'train', CFG, train=True)
    swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_animal=1.0), train=True)

    b = pose_collate([_train_item(base_ds)])
    s = pose_collate([_train_item(swap_ds)])

    torch.testing.assert_close(b.coords, s.coords, equal_nan=True)
    torch.testing.assert_close(b.cgroup[0]['mat'], s.cgroup[0]['mat'])
    torch.testing.assert_close(b.cgroup[0]['offset'], s.cgroup[0]['offset'])
    torch.testing.assert_close(b.cgroup[0]['size'].float(), s.cgroup[0]['size'].float())
    if b.vis_2d is not None:
        torch.testing.assert_close(b.vis_2d, s.vis_2d, equal_nan=True)
    assert not torch.allclose(b.kpt_prior, s.kpt_prior, equal_nan=True), \
        'prompt_swap_animal = 1.0 on a 2-animal session must actually move the prior'


def test_animal_swap_prior_is_in_the_model_frame(tmp_path):
    """The swapped prior is the neighbour's pose through the target's own transform: with a rigid pair
    under a world rotation, corrupted - base is one vector of norm `sep`."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    sep = 5.0
    root = tmp_path / 'mv'
    cf._session_3d_multi(root / 'train' / 's', T=6, sep=sep)
    base_ds = PoseDataset(root, 'train', CFG, train=True)
    swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_animal=1.0), train=True)

    b = pose_collate([_train_item(base_ds)])
    s = pose_collate([_train_item(swap_ds)])
    assert not torch.allclose(b.kpt_prior, s.kpt_prior, equal_nan=True)

    diff = (s.kpt_prior - b.kpt_prior)[0]                    # (K,3)
    ok = torch.isfinite(diff).all(-1)
    assert int(ok.sum()) >= 2, 'need at least two keypoints to check the diff is one vector'
    torch.testing.assert_close(diff[ok], diff[ok][:1].expand_as(diff[ok]), atol=1e-3, rtol=1e-3)
    assert float(diff[ok][0].norm()) == pytest.approx(sep, abs=1e-2), \
        'a world ROTATION preserves the norm of a rigid offset'


def test_animal_swap_2d_prior_is_the_neighbours_own_labelled_pixel(tiny_root):
    """2D half of the same claim, inverted through the item's OWN recorded crop geometry rather
    than a second copy of the crop rule. `scale` is solved from the target's own correspondence.
    """
    root = tiny_root / 'ratlike'
    cfg = replace(CFG, aug_prob=0.0, crop_jitter=0.0)
    base_ds = PoseDataset(root, 'train', cfg, train=True)
    swap_ds = PoseDataset(root, 'train', replace(cfg, prompt_swap_animal=1.0), train=True)
    b = pose_collate([_train_item(base_ds)])
    s = pose_collate([_train_item(swap_ds)])
    torch.testing.assert_close(b.prompt_t, s.prompt_t)

    sess = load_dataset(root).sessions['train'][0]
    sess.preload()
    lab = sess.labels('g000')
    target = torch.as_tensor(lab.points2d[0][:, :, 0], dtype=torch.float32)  # (T_group,K,2)
    other = torch.as_tensor(lab.points2d[1][:, :, 0], dtype=torch.float32)  # (T_group,K,2)

    offset = s.cgroup[0]['offset'].float()
    start, stride = int(b.sample_info['start']), int(b.sample_info['stride'])
    pt = b.prompt_t[0]
    finite_t = torch.isfinite(b.kpt_prior[0]).all(-1)
    assert bool(finite_t.any()), 'need at least one target keypoint to solve `scale` from'
    k0 = int(finite_t.nonzero(as_tuple=True)[0][0])
    raw0 = target[start + int(pt[k0]) * stride, k0]
    scale = float(((b.kpt_prior[0, k0] + offset) / raw0).mean())

    recovered = (s.kpt_prior[0] + offset) / scale              # (K, 2), back in SOURCE pixels
    finite = torch.isfinite(s.kpt_prior[0]).all(-1)
    assert bool(finite.any()), 'need at least one corrupted keypoint to check'
    for k in finite.nonzero(as_tuple=True)[0].tolist():
        d = (other[:, k] - recovered[k]).norm(dim=-1)
        d = d[torch.isfinite(d)]                    # a02 carries an UNLABELED slot (`_session_2d`)
        assert len(d) and float(d.min()) < 0.5, \
            f'keypoint {k}: recovered source pixel matches no frame of the other animal'


def test_kpt_swap_is_a_per_keypoint_rate_onto_another_point(tiny_root):
    """At p = 1.0 EVERY finite keypoint moves (a rate, not a pair count), each onto ANOTHER
    keypoint's original position, never its own."""
    for root in (tiny_root / 'ratlike', tiny_root / 'mouselike'):
        b = pose_collate([_train_item(PoseDataset(root, 'train', CFG, train=True))])
        s = pose_collate([_train_item(PoseDataset(root, 'train',
                                                  replace(CFG, prompt_swap_kpt_pairs=1.0),
                                                  train=True))])
        original, corrupted = b.kpt_prior[0], s.kpt_prior[0]
        finite = torch.isfinite(original).all(-1) & torch.isfinite(corrupted).all(-1)
        moved = ~torch.isclose(original, corrupted).all(-1) & finite
        assert int(finite.sum()) >= 2 and int(moved.sum()) == int(finite.sum()), root.name
        for k in moved.nonzero(as_tuple=True)[0].tolist():
            mask = finite.clone()
            mask[k] = False
            assert float((original[mask] - corrupted[k]).norm(dim=-1).min()) < 1e-4, (root.name, k)
            assert float((corrupted[k] - original[k]).norm()) > 1e-4, (root.name, k)


def test_pose_only_prob_default_is_byte_identical_to_pre_change_code(tiny_root):
    """`pose_only_prob = 0.0` reproduces the pre-change loader (`57b5350^`, read from git) rng draw
    for rng draw: a hoisted unused coin would shift every later draw invisibly."""
    import importlib.util
    import subprocess
    import sys

    root = tiny_root / 'mouselike'
    src = subprocess.run(['git', 'show', '57b5350^:tailcyclenet/dataset.py'],
                         capture_output=True, text=True, check=True, cwd=Path(__file__).parent.parent).stdout
    assert 'pose_only_prob' not in src, \
        'the pre-landing source already has this key -- point this test at an earlier commit'
    # The historical source predates the posetail -> tracktail rename (0.5.0); translate its
    # library imports and this repo's `Rig.posetail()` so it runs against the installed library.
    for before, after in (('posetail.datasets.posetail_dataset', 'tracktail.datasets.tracktail_dataset'),
                          ('posetail.posetail', 'tracktail.model'), ('PosetailDataset', 'TracktailDataset'),
                          ('.posetail(', '.tracktail(')):
        src = src.replace(before, after)

    spec = importlib.util.spec_from_loader('tailcyclenet._dataset_pre_change', loader=None)
    old_mod = importlib.util.module_from_spec(spec)
    old_mod.__package__ = 'tailcyclenet'
    sys.modules['tailcyclenet._dataset_pre_change'] = old_mod
    try:
        exec(compile(src, 'dataset.py (57b5350^)', 'exec'), old_mod.__dict__)

        cfg_kwargs = dict(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                          crop_jitter=0.0, prompt_dropout=0.3, prompt_offset_px=2.0,
                          prompt_noise_px=1.5, box_prompt='film', box_prompt_dropout=0.5)

        old_ds = old_mod.PoseDataset(root, 'train', old_mod.LoaderConfig(**cfg_kwargs), train=True)
        new_ds = PoseDataset(root, 'train', LoaderConfig(**cfg_kwargs, pose_only_prob=0.0),
                             train=True)

        for seed in range(2):
            np.random.seed(seed)
            old_item = old_ds._item(0, np.random.default_rng(seed))
            np.random.seed(seed)
            new_item = new_ds._item(0, np.random.default_rng(seed))
            assert (old_item is None) == (new_item is None), f'seed {seed}: presence mismatch'
            if old_item is None:
                continue
            assert len(old_item) == len(new_item), f'seed {seed}: field count mismatch'
            for k, (o, n) in enumerate(zip(old_item, new_item)):
                if isinstance(o, torch.Tensor):
                    assert torch.equal(torch.nan_to_num(o, nan=-9e9),
                                       torch.nan_to_num(n, nan=-9e9)), \
                        f'seed {seed} field {k}: pose_only_prob=0.0 diverged from pre-change code'
    finally:
        del sys.modules['tailcyclenet._dataset_pre_change']


def test_pose_only_prob_gives_box_dropped_items_a_clean_prior(tiny_root):
    """At `pose_only_prob = 1.0` a box-dropped item gets the clean prior, none of the prompt corruptions."""
    root = tiny_root / 'mouselike'
    clean_cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                             crop_jitter=0.0, prompt_dropout=0.0, prompt_offset_px=0.0,
                             prompt_noise_px=0.0)
    clean_ds = PoseDataset(root, 'train', clean_cfg, train=True)

    corrupt_cfg = replace(clean_cfg, prompt_dropout=1.0, prompt_offset_px=5.0,
                         prompt_noise_px=5.0, box_prompt='film', box_prompt_dropout=1.0,
                         pose_only_prob=1.0)
    pose_only_ds = PoseDataset(root, 'train', corrupt_cfg, train=True)

    np.random.seed(0)
    clean_item = clean_ds._item(0, np.random.default_rng(0))
    np.random.seed(0)
    pose_only_item = pose_only_ds._item(0, np.random.default_rng(0))
    assert clean_item is not None and pose_only_item is not None

    # kpt_prior is field index 11 in the `_item` output tuple (see its own docstring's ordering)
    clean_prior, pose_only_prior = clean_item[11], pose_only_item[11]
    assert torch.equal(torch.nan_to_num(clean_prior, nan=-9e9),
                       torch.nan_to_num(pose_only_prior, nan=-9e9)), \
        'box_prompt_dropout=1.0, pose_only_prob=1.0 must give the SAME clean prior prompt_' \
        'dropout=0 would, not the corrupted one prompt_dropout=1.0 alone would produce'


# ----------------------------------------------------------------------------------------------
# A1 (dev/plans/multianimal_system_improvements.md): a whole-item dropout stays query-free
# ----------------------------------------------------------------------------------------------

def test_whole_item_dropout_survives_the_neighbour_swap(tmp_path, tiny_root):
    """A `prompt_dropout` item stays query-free: a later `prompt_swap_animal` draw cannot repopulate
    it from the neighbour (3D and 2D), while the swap still fires on a kept prior."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    root = tmp_path / 'mv'
    cf._session_3d_multi(root / 'train' / 's', T=6)
    drop = replace(CFG, prompt_dropout=1.0, prompt_swap_animal=0.9)
    for r in (root, tiny_root / 'ratlike'):
        item = pose_collate([_train_item(PoseDataset(r, 'train', drop, train=True))])
        assert int(torch.isfinite(item.kpt_prior).sum()) == 0, r
    swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_animal=0.9), train=True)
    assert int(torch.isfinite(pose_collate([_train_item(swap_ds)]).kpt_prior).sum()) > 0


class _Squares(torch.utils.data.Dataset):
    """A module-level (so spawn-picklable) dataset carrying a big array, for `SpawnHandoff`."""

    def __init__(self):
        self.big = np.arange(300_000, dtype=np.float64)

    def __len__(self):
        return 6

    def __getitem__(self, i):
        return torch.tensor(self.big[i] ** 2)


def test_spawn_handoff_sends_a_handle_and_the_same_dataset():
    """`SpawnHandoff` pickles to a small payload (shared bytes travel as a handle) and unpickles to
    exactly the wrapped dataset; the in-process path forwards untouched."""
    import pickle
    from multiprocessing.reduction import ForkingPickler

    import torch.multiprocessing  # noqa: F401  (registers torch's shared-memory reducers)

    from tailcyclenet.dataset import SpawnHandoff

    ds = _Squares()
    wrapped = SpawnHandoff(ds)
    assert len(wrapped) == 6 and wrapped[3] == ds[3]
    payload = bytes(ForkingPickler.dumps(wrapped))
    assert len(payload) < 64 * 1024 < len(pickle.dumps(ds))
    back = pickle.loads(payload)
    assert type(back) is _Squares and np.array_equal(back.big, ds.big)


@pytest.mark.slow
def test_a_spawn_worker_sees_the_wrapped_dataset():
    """A real spawn DataLoader worker (~5 s of interpreter start-up) reads the wrapped dataset."""
    from tailcyclenet.dataset import SpawnHandoff

    ds = _Squares()
    loader = torch.utils.data.DataLoader(SpawnHandoff(ds), batch_size=None, num_workers=2,
                                         multiprocessing_context='spawn')
    assert [float(x) for x in loader] == [float(ds[i]) for i in range(6)]


def test_an_undecodable_video_frame_skips_the_item_instead_of_killing_the_worker(tmp_path):
    """The image path returns None for an unreadable frame and the loader draws another item;
    the video path used to RAISE out of `realise`, killing the DataLoader worker."""
    import shutil

    from .conftest import _session_2d, _write_video

    items = []
    for n in (4, 2):
        s = tmp_path / f'ds{n}' / 'train' / 'a'
        _session_2d(s, T=4)
        shutil.rmtree(s / 'groups' / 'g000' / 'cam0')
        _write_video(s / 'groups' / 'g000' / 'cam0.mp4', 0, n, (64, 48))
        ds = PoseDataset(tmp_path / f'ds{n}', 'train', CFG)
        rng = np.random.default_rng(0)
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter('always')
            items.append((ds._item(0, rng, ds.shape(rng)), got))
    assert items[0][0] is not None and not items[0][1]
    assert items[1][0] is None
    assert any('did not decode' in str(w.message) for w in items[1][1])


@pytest.mark.parametrize('bad', [{'two_d_sampling': 'true-2d'}, {'box_prompt_frames': 'frist'},
                                 {'box_prompt': 'flim'}, {'cams_to_sample': [4, 2]},
                                 {'val_cams_to_sample': [3, 1]}, {'crop_inflate': [2.0, 1.0]},
                                 {'aug_rotation_prob': 1.5}])
def test_loader_config_refuses_a_bad_value_at_construction(bad):
    """These used to surface only at the first item, inside a worker -- or, for a typo'd
    `box_prompt`, turned the box channel ON."""
    with pytest.raises(ValueError, match=next(iter(bad))):
        LoaderConfig(**bad)
    with pytest.raises(ValueError, match=next(iter(bad))):
        replace(LoaderConfig(), **bad)


def test_loader_config_accepts_every_legal_value():
    for ok in ({'two_d_sampling': 'legacy-3d-single-view'}, {'box_prompt_frames': 'first'},
               {'box_prompt': 'film'}, {'cams_to_sample': [1, 8]}, {'cams_to_sample': 3},
               {'crop_inflate': [1.0, 1.0]}, {'crop_inflate': 1.5}, {'aug_rotation_prob': 0.0},
               {'aug_rotation_prob': None}):
        LoaderConfig(**ok)


def test_loader_config_from_refuses_unknown_keys_and_allows_the_callers_plumbing():
    """The one `[data]` -> `LoaderConfig` path: a typo is refused by name, plumbing is not passed."""
    from tailcyclenet.dataset import SCORER_PLUMBING, loader_config_from

    lc = loader_config_from({'path': 'x', 'num_workers': 2, 'n_frames': 6})
    assert lc.n_frames == 6
    with pytest.raises(SystemExit, match=r"unknown key\(s\) \['n_frame'\]"):
        loader_config_from({'n_frame': 6})
    with pytest.raises(SystemExit, match='prefetch_factor'):
        loader_config_from({'prefetch_factor': 2})
    assert loader_config_from({'prefetch_factor': 2}, extra=SCORER_PLUMBING).n_frames == LoaderConfig().n_frames
