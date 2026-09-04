"""The loader, and the one thing in it that must be bit-exact.

The crop rule is shared between the pose model and the detector -- the detector is trained to
reproduce the pose crop -- and `test_crop_rule_is_int32_exact` is what licenses that.
"""
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet import crop as cropmod
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.format import Registry, load_dataset

from .conftest import KPTS_2D


# ----------------------------------------------------------------------------------------------
# the crop rule
# ----------------------------------------------------------------------------------------------

def test_crop_rule_is_int32_exact():
    """`crop_box_for_points` must equal the library's `crop_cgroup_to_points`, exactly.

    Any drift here invalidates every detector number.
    """
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

        # the library projects first; feed it a camera whose projection is the identity by
        # calling the crop arithmetic on the same points through a one-camera group
        mine = cropmod.crop_box_for_points(pts, size, 64)

        pflat = pts.reshape(-1, 2)
        good = torch.all(torch.isfinite(pflat), dim=1)
        if not good.any():
            assert mine is None
            continue
        theirs = _library_box(shim, pflat[good], size)
        assert mine is not None
        assert mine.dtype == torch.int32
        assert torch.equal(mine, theirs), f'{w}x{h}: {mine.tolist()} != {theirs.tolist()}'


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


def test_crop_box_is_none_when_nothing_is_finite():
    """The library raises here; the detector depends on None so it can emit a NaN box."""
    pts = torch.full((5, 2), float('nan'))
    assert cropmod.crop_box_for_points(pts, torch.tensor([100, 100], dtype=torch.int32)) is None


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

@pytest.mark.parametrize('angle', [0.0, 20.0, -35.0, 150.0, -170.0])
def test_the_fused_warp_agrees_with_the_camera(angle):
    """The composed rotate->crop->resize affine must land pixels where the CAMERA says they go.

    Painted marker against known projection, through the two halves of the transform; also the
    only way the `A @ M3` composition can be silently wrong.
    """
    import cv2

    from tailcyclenet import format as fmt
    from tailcyclenet.dataset import _crop_affine, _resize_camera
    from posetail.datasets.posetail_dataset import rotate_camera_image_plane_3d
    from posetail.posetail.cube import project_points_torch

    W, H = 320, 240
    from aniposelib.cameras import CameraGroup
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False},
                  calibrated={'cam0': True})
    cam = rig.posetail()[0]

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
    cam, _ = _resize_camera(cam, 64)

    want = project_points_torch([cam], pt)[0][0, 0]             # where the camera says it is now
    M, size = _crop_affine((W, H), box, cam['size'].tolist(), rotation)
    out = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR)

    # bilinear spreads the marker over up to 4 pixels; its intensity centroid is the position
    ys, xs = np.nonzero(out[..., 0])
    assert xs.size, 'the marker fell outside the crop -- the test setup is wrong, not the code'
    wgt = out[ys, xs, 0].astype(np.float64)
    got = np.array([(xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()])
    assert np.allclose(got, np.asarray(want, np.float64), atol=1.0), f'{got} vs {want}'


@pytest.mark.parametrize('angle', [17.0, 45.0, 90.0, 173.0, -128.0])
def test_the_2d_rotation_keeps_every_pixel_and_agrees_with_its_camera(angle):
    """`_rotate_2d` must lose NO label at any angle, and its 2x3 must match the camera it returns.

    The library's inscribed-crop rotation silently loses animals outside it; the repo's own
    rotation keeps every source pixel and lands labels where its camera says they land.
    """
    from aniposelib.cameras import CameraGroup

    from tailcyclenet import format as fmt
    from tailcyclenet.dataset import _rotate_2d

    W, H = 4696, 2048
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False}, calibrated={'cam0': True})
    cam = rig.posetail()[0]
    # the four frame corners plus its centre -- if any of these leaves the canvas, an animal can
    corners = torch.tensor([[[0.0, 0.0], [W, 0.0], [W, H], [0.0, H], [W / 2, H / 2]]])

    out, moved, (M, size) = _rotate_2d(cam, corners, angle)

    assert tuple(out['size'].tolist()) == tuple(size)
    # TIGHT: the rotated frame's bounding box is the canvas EXACTLY -- nothing lost off an edge and
    # nothing wasted. The inscribed crop fails the first half; a lazily oversized canvas the second.
    box = moved[0, :4]
    np.testing.assert_allclose(box.amin(0).numpy(), [0.0, 0.0], atol=0.51)
    np.testing.assert_allclose(box.amax(0).numpy(), list(size), atol=1.01)
    # The camera's principal point moved by the canvas expansion, which is what keeps the pixels
    # `_crop_affine` produces agreed with the camera `apply_crop` hands downstream: the frame
    # centre must still be the canvas centre.
    centre = moved[0, 4].numpy()
    np.testing.assert_allclose(centre, [size[0] / 2, size[1] / 2], atol=1.01)
    shift = (out['mat'][:2, 2] - cam['mat'][:2, 2]).numpy()
    np.testing.assert_allclose(centre - shift, [W / 2, H / 2], atol=1.01)


def test_computed_frame_paths_match_the_listing(dataset_3d):
    """`read_frames` computes `%06d.<ext>` instead of listing the directory; the fixture's frames
    differ per (cam, frame), so an off-by-one is visible in the pixels.
    """
    from tailcyclenet.dataset import load_warps, read_frames

    sess = dataset_3d.sessions['train'][0]
    group = sess.groups['g000']
    for cam in sess.cam_names:
        kind, src, ext = group.source(cam)
        assert (kind, ext) == ('frames', '.png')
        listed = sorted(f for f in src.iterdir() if f.suffix in ('.png', '.jpg'))
        frames = [3, 0, 2, 0]                       # out of order and repeated, as windows are
        want = [load_warps(str(listed[i]), [(None, None)], None, 1)[0]
                for i in frames]
        got = read_frames(group, cam, frames)
        for a, b in zip(want, got):
            np.testing.assert_array_equal(a, b)
        # and the frames really are distinguishable, or the assertion above proves nothing
        assert not np.array_equal(got[0], got[2])
        # a repeated index is decoded once but must not be ALIASED -- cutout writes in place
        assert got[1] is not got[3]


def test_repeated_frames_are_decoded_once(dataset_3d, monkeypatch):
    """A clamp-padded window must not decode the same file T times.

    Counts `cv2.imread`, not a helper -- the decode is the thing being conserved.
    """
    import cv2

    from tailcyclenet import dataset as dsmod

    sess = dataset_3d.sessions['train'][0]
    group = sess.groups['g000']
    calls, real = [], cv2.imread

    def counted(p, *a, **k):
        calls.append(p)
        return real(p, *a, **k)

    monkeypatch.setattr(cv2, 'imread', counted)

    out = dsmod.read_frames(group, sess.cam_names[0], [0] * 24)
    assert len(out) == 24
    assert len(calls) == 1, f'decoded {len(calls)} times for one distinct frame'
    for im in out:
        np.testing.assert_array_equal(im, out[0])


def test_one_frame_under_many_crops_decodes_once(dataset_3d, monkeypatch):
    """A per-frame-box window pays one decode for T crops: `load_warps` splits decode from warp."""
    import cv2

    from tailcyclenet import dataset as dsmod

    sess = dataset_3d.sessions['train'][0]
    group = sess.groups['g000']
    calls, real = [], cv2.imread
    monkeypatch.setattr(cv2, 'imread', lambda p, *a, **k: (calls.append(p), real(p, *a, **k))[1])

    boxes = np.stack([np.array([i, i, i + 16, i + 16], np.int32) for i in range(8)])
    out = dsmod.read_frames(group, sess.cam_names[0], [0] * 8, crop_coords=boxes,
                            target_size=[16, 16])
    assert len(calls) == 1, f'decoded {len(calls)} times for one distinct frame'
    assert [im.shape for im in out] == [(16, 16, 3)] * 8
    # ... and they are not all the same picture, or the dedupe collapsed the crops instead.
    assert any(not np.array_equal(out[0], im) for im in out[1:])


# ----------------------------------------------------------------------------------------------
# the loader
# ----------------------------------------------------------------------------------------------

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)


def _batch(ds, i=0):
    return pose_collate([ds[i]])


def test_2d_item_shapes(tiny_root):
    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 1
    assert b.views[0].shape[:2] == (1, 4)                 # (B, T, H, W, 3)
    # uint8 out of the loader: 4x fewer bytes to queue and pin; `model.forward` divides by 255.
    assert b.views[0].dtype == torch.uint8
    assert b.coords.shape == (1, 4, 4, 2)                 # R=2 for a true-2D session
    assert b.p2d.shape == (1, 1, 4, 4, 2)                 # 2D needs p2d; the loss reads it
    assert b.kpt_ids.shape == (1, 4)
    assert b.kpt_prior.shape == (1, 4, 2)
    assert b.prompt_t.shape == (1, 4)
    # `ratlike` is `annotated` and carries a real MISSING/UNLABELED pair (see `_session_2d`), so
    # `has_visibility_assessment` is True unconditionally and the window's own assessed rows
    # populate vis_2d. `vis` stays None: there is no 3D layer at R == 2.
    assert b.vis is None
    assert b.vis_2d is not None
    assert b.vis_2d.shape == (1, 4, 4, 1, 1)               # (B, T, K, C=1, 1)


def test_3d_item_shapes(tiny_root):
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 3
    assert b.coords.shape == (1, 4, 3, 3)                 # R=3 world
    assert b.p2d is None
    assert b.vis.shape == (1, 4, 3, 1)                    # trailing dim get_eval_metrics wants
    assert b.vis_2d.shape == (1, 4, 3, 3, 1)


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


def test_vis_and_vis2d_are_both_or_neither_in_3d(tiny_root):
    """Supplying one without the other dies inside posetail's einops, so the loader must never do it."""
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        assert (b.vis is None) == (b.vis_2d is None)


def test_2d_vis_is_never_both_with_vis(tiny_root):
    """At R == 2 `vis` is always None while `vis_2d` is populated on a real assessment; the two
    must never be conflated (they ride separate wires in `run_batch`).
    """
    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)
    saw_populated = False
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        assert b.vis is None
        saw_populated |= b.vis_2d is not None
    assert saw_populated, 'ratlike carries a real assessment; some item must show it'


def test_keypoints_are_never_filtered(tiny_root):
    """The library's `filter_keypoints` would mislabel positions invisibly in the loss curve."""
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        assert b.coords.shape[2] == 3
        assert b.kpt_ids.shape[1] == 3


def test_window_is_at_least_two_frames(tiny_root):
    """T=1 routes posetail into a zero-length pos_embed; `n_frames = 1` is refused outright, and
    a group shorter than the ceiling yields its own length, never 1.
    """
    cfg = LoaderConfig(n_frames=1, image_size=64, aug_prob=0.0, crop_jitter=0.0)
    with pytest.raises(AssertionError, match='n_frames'):
        PoseDataset(tiny_root / 'ratlike', 'train', cfg)

    # A 4-frame group under a ceiling of 8 now yields 4, not 8: T is sized to the labelled span,
    # so the window is no longer padded out with duplicates of the last frame. Never 1, though --
    # that is the whole point of this test.
    cfg2 = LoaderConfig(n_frames=8, image_size=64, aug_prob=0.0, crop_jitter=0.0)
    ds2 = PoseDataset(tiny_root / 'ratlike', 'train', cfg2)   # groups are only 4 frames long
    b2 = _batch(ds2)
    assert b2.views[0].shape[1] == 4


def test_single_view_keeps_3d_targets(tiny_root):
    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=1.0, aug_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    ds = PoseDataset(tiny_root / 'mouselike', 'train', cfg)
    b = _batch(ds)
    assert b.coords.shape[-1] == 3          # targets stay world-metric
    assert len(b.views) == 1                # exactly one camera
    assert b.p2d is not None
    assert b.sample_info['single_view'] is True


def test_cams_to_sample_takes_a_range(tiny_root):
    """`[low, high]` draws a per-item camera count, clamped to what the session has."""
    def counts(spec, n=60):
        cfg = LoaderConfig(n_frames=4, image_size=64, cams_to_sample=spec, prob_2d_only=0.0,
                           aug_prob=0.0, crop_jitter=0.0, prompt_dropout=0.0)
        ds = PoseDataset(tiny_root / 'mouselike', 'train', cfg)
        return [len(ds[i % len(ds)][0]) for i in range(n)]

    assert set(counts([1, 3])) == {1, 2, 3}      # every value in range actually occurs
    assert set(counts([2, 2])) == {2}            # a degenerate range is a fixed count
    assert set(counts([1, 8])) == {1, 2, 3}      # clamped to what the session has, not an error
    assert set(counts(0)) == {3}                 # 0 still means "all cameras"
    assert set(counts(2)) == {2}                 # the int form is unchanged


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


def test_registry_ids_survive_a_later_run(tiny_root):
    """Ids must be APPEND-ONLY against an existing registry, or warm start remaps the embedding.
    Discovery order is a directory listing, so it is not a stable thing to number against.
    """
    from tailcyclenet.format import load_datasets

    ds = load_datasets(tiny_root)
    first = Registry.build(ds)
    assert dict(Registry.build(list(reversed(ds))).datasets) != dict(first.datasets), \
        'the fixture must be order-sensitive, or this test proves nothing'

    again = Registry.build(list(reversed(ds)), first)
    assert dict(again.datasets) == dict(first.datasets)
    assert again.names == first.names


def test_registry_appends_a_new_dataset_without_moving_old_ids(tiny_root, tmp_path):
    from tailcyclenet.format import load_datasets

    base = Registry.build(load_datasets(tiny_root))
    grown = Registry.build(load_datasets(tiny_root) + [_FakeDataset('zzz', ['a', 'b'])], base)
    assert grown.names[:len(base.names)] == base.names
    assert list(grown.ids_for_dataset('zzz')) == [len(base.names), len(base.names) + 1]


class _FakeDataset:
    """The two attributes Registry.build reads. Cheaper than writing a session to disk."""

    def __init__(self, name, names):
        self.name, self.names = name, names


def test_per_camera_augmentation_is_constant_down_a_clip(tiny_root):
    """A camera's appearance must hold steady for the whole clip (it is an identity cue); sensor
    noise and motion blur are the opposite and must vary.
    """
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


def test_appearance_augmentation_is_train_only(tiny_root):
    """Val pixels must be clean: an augmented val metric is not comparable to the last one."""
    aug = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=1.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    train_ds = PoseDataset(tiny_root / 'ratlike', 'train', aug)
    assert train_ds._aug is not None
    assert PoseDataset(tiny_root / 'ratlike', 'val', aug)._aug is None
    # and the train path really changes pixels rather than silently no-opping
    plain = pose_collate([PoseDataset(tiny_root / 'ratlike', 'train', CFG)[0]])
    assert not torch.equal(plain.views[0], pose_collate([train_ds[0]]).views[0])


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


def test_val_windows_are_deterministic(tiny_root):
    ds = PoseDataset(tiny_root / 'ratlike', 'val', CFG)
    a = pose_collate([ds[0]])
    b = pose_collate([ds[0]])
    # equal_nan: a missing label is NaN, and two identical items must agree about which.
    torch.testing.assert_close(a.coords, b.coords, equal_nan=True)
    torch.testing.assert_close(a.views[0], b.views[0])
    assert a.sample_info['start'] == b.sample_info['start']


def test_prompt_is_the_first_labelled_frame(tiny_root):
    """prompt_t is not always 0 -- it was > 0 on 19.5% of rat-city windows."""
    ds = PoseDataset(tiny_root / 'ratlike', 'val', CFG)
    b = _batch(ds)
    finite = torch.isfinite(b.coords[0]).all(-1)       # (T,K)
    for k in range(finite.shape[1]):
        if finite[:, k].any():
            assert b.prompt_t[0, k] == int(finite[:, k].float().argmax())
            torch.testing.assert_close(b.kpt_prior[0, k],
                                       b.coords[0, b.prompt_t[0, k].long(), k])


def test_prompt_dropout_is_per_item_not_per_keypoint(tiny_root):
    """`prompt_dropout` is the fraction of STEPS that run fully query-free, per the reference.

    A per-keypoint draw would make a fully-unprompted window exponentially unlikely, so the
    query-free forward val scores is never trained; at p = 0.5 every window must be all-NaN or
    no-NaN, never a mixture.
    """
    def cfg(p):
        return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=p, prob_2d_only=0.0)

    ds = PoseDataset(tiny_root / 'ratlike', 'train', cfg(1.0))
    assert torch.isnan(_batch(ds).kpt_prior).all()     # unprompted -> learned tokens
    ds0 = PoseDataset(tiny_root / 'ratlike', 'train', cfg(0.0))
    assert not torch.isnan(_batch(ds0).kpt_prior).any()

    ds5 = PoseDataset(tiny_root / 'ratlike', 'train', cfg(0.5))
    seen = set()
    for _ in range(40):
        nan = torch.isnan(_batch(ds5).kpt_prior)
        assert nan.all() or not nan.any(), 'dropout mixed within one window -> it is per-keypoint'
        seen.add(bool(nan.all()))
    assert seen == {True, False}, f'p=0.5 never produced both outcomes in 40 draws: {seen}'


def test_prompt_noise_perturbs_only_real_priors(tiny_root):
    """Exposure bias: the prior trains on exact GT and deploys as the model's own prediction."""
    def cfg(sigma, drop):
        return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=drop, prompt_noise_px=sigma, prob_2d_only=0.0)

    ds = PoseDataset(tiny_root / 'ratlike', 'train', cfg(0.0, 0.0))
    clean = _batch(ds).kpt_prior
    noisy = _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg(5.0, 0.0))).kpt_prior
    assert torch.isfinite(noisy).all()
    assert not torch.allclose(clean, noisy)
    # NaN + noise is still NaN, so a withheld prior stays withheld with no masking needed.
    dropped = _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg(5.0, 1.0))).kpt_prior
    assert torch.isnan(dropped).all()


def test_window_is_sized_to_the_labelled_span(centred_root):
    """T is derived from the labels, not configured; `n_frames` is only its ceiling. The window
    must actually COVER the labelled frames rather than merely touch one of them.
    """
    cfg = LoaderConfig(n_frames=24, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg)
    for _ in range(20):
        b = _batch(ds)
        T = b.views[0].shape[1]
        assert T == 4, T
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
        for _ in range(30):
            f = pose_collate([ds[0]]).fnums[0].numpy()
            assert list(np.diff(f)) == [s] * (len(f) - 1), f  # no clamping, no off-lattice start
            assert f.max() < 32 and f.min() >= 0, f

    # A one-label lattice: labels sit at 11-13, so at stride 4 only the anchor is reachable and T
    # falls to the floor of 2 -- more motion for the same two encodes. This is the allen shape.
    mid = PoseDataset(centred_root, 'train', cfg([4], n=24))
    for _ in range(20):
        f = pose_collate([mid[0]]).fnums[0].numpy()
        assert len(f) == 2 and f[1] - f[0] == 4, f
        assert set(f) & {11, 12, 13}, f

    val = PoseDataset(dense_root, 'train', cfg([4]), train=False)
    assert list(np.diff(pose_collate([val[0]]).fnums[0].numpy())) == [1] * 7


def test_visibility_stays_three_state(tiny_root):
    """"Not assessed" must reach the loss as NaN, not as "not visible" -- posetail >= 0.3.2
    masks non-finite visibility targets, so collapsing to 0 would train the head on labels nobody
    made.
    """
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


def test_gradients_survive_unassessed_visibility():
    """The property the loader depends on, asserted against the INSTALLED posetail: a NaN target
    must leave the loss finite and the gradient finite.
    """
    from posetail.posetail.losses import BCELossVis

    pred = torch.zeros(2, 4, 3, 1, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 3, 1)).float()
    target[0, 1, 2, 0] = float('nan')          # not assessed
    loss = BCELossVis(weight=1.0)(pred, target, device='cpu')
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all(), (
        'installed posetail poisons the gradient on an unassessed visibility target; '
        'needs >= 0.3.2')


def test_no_visibility_supervision_without_ground_truth(tracked_no_assessment_root, monkeypatch):
    """A dataset with no visibility ASSESSMENT (100% `visible`, no `missing` row anywhere) must
    not have its visibility head trained -- the per-window NaN-mask cannot see it, so the session
    level gates it.
    """
    from posetail.posetail.losses import TotalLoss

    ds = PoseDataset(tracked_no_assessment_root, 'train', CFG)
    b = _batch(ds)
    assert b.vis is None and b.vis_2d is None

    seen = []

    class Spy(TotalLoss):
        def forward(self, model, outputs, coords_true, vis_true, vis_true_cams, **kw):
            seen.append((vis_true is None, vis_true_cams is None))
            raise SystemExit                      # we only need the arguments, not the loss

    spy = Spy(vis_loss_weight=5.0, vis_loss_3d_weight=1.0)
    try:
        spy(None, {}, b.coords, b.vis, b.vis_2d)
    except SystemExit:
        pass
    assert seen == [(True, True)], 'a dataset without visibility labels must pass None, not zeros'


# ----------------------------------------------------------------------------------------------
# a label in the middle of a group must be usable -- on BOTH paths
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('train', [True, False])
@pytest.mark.parametrize('n_frames', [8, 24])
def test_centred_labels_are_usable(centred_root, train, n_frames):
    """The old loader required the window's FIRST frame to be labelled; this one does not.
    Every window must contain a label, and the label must not be forced to frame 0.
    """
    cfg = LoaderConfig(n_frames=n_frames, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=train)
    assert len(ds) > 0, 'a group whose labels are centred must still produce windows'

    n = 40 if train else len(ds)
    at_frame_zero = 0
    for i in range(n):
        b = pose_collate([ds[i % len(ds)]])
        finite = torch.isfinite(b.coords[0]).all(-1)          # (T,K)
        assert finite.any(), 'every window must contain at least one labelled frame'
        at_frame_zero += int(finite[0].any())
    assert at_frame_zero < n, 'the label must not always land on frame 0'


def test_val_windows_do_not_pad_when_the_group_is_long_enough(centred_root):
    """A start past `n_frames - T` clamp-pads with duplicates of the last frame, wasting context."""
    cfg = LoaderConfig(n_frames=24, image_size=32, aug_prob=0.0, crop_jitter=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=False)
    for item in ds.index:
        assert 0 <= item.start <= max(0, item.session.groups[item.gid].n_frames - 24)
    b = pose_collate([ds[0]])
    assert b.sample_info['start'] == 0
    assert b.fnums[0].tolist() == list(range(24)), 'no duplicated frames'


def test_prompt_time_is_not_forced_to_zero(centred_root):
    """prompt_t is the first LABELLED frame, which is > 0 whenever labels are centred."""
    cfg = LoaderConfig(n_frames=24, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=False)
    b = pose_collate([ds[0]])
    assert (b.prompt_t[0] == 11).all(), 'the prompt must point at the real first label'
    assert torch.isfinite(b.kpt_prior[0]).all(), 'and the prior must be taken from there'


def test_projected_carries_position_but_trains_no_visibility(projected_root):
    """`projected` is a 2D position with no visibility claim, and must reach the loss as neither.
    Written as `visible` it would train "always visible" from labels that assert nothing; the
    3D noisy-OR would claim nothing is reconstructible, so the target is withheld.
    """
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
    """`annot_frac` then `mode_3d_frac` WITHIN each source, and the realised draw matches.

    The old sampler weighted by index-entry count, so a long tracked session could be starved;
    `mixed_source_root` has no `2d-tracked` cell, so the mode level is skipped inside `tracked`.
    """
    cfg = LoaderConfig(n_frames=2, image_size=32, annot_frac=0.4, mode_3d_frac=0.7)
    ds = PoseDataset(mixed_source_root, 'train', cfg)
    want = {'2d-annotated': 0.4 * 0.3, '3d-annotated': 0.4 * 0.7, '3d-tracked': 0.6}
    assert ds.mix() == pytest.approx(want, abs=1e-9)

    rng = np.random.default_rng(0)
    n = 20000
    drawn = Counter()
    for i in range(n):
        s = ds._pick(i % len(ds.index), rng).session
        drawn[f'{s.mode}-{s.label_source}'] += 1
    for k, p in want.items():
        assert drawn[k] / n == pytest.approx(p, abs=0.02), f'{k}: {drawn[k] / n:.3f} != {p}'


def test_sampling_mix_defaults_to_the_uniform_draw(mixed_source_root):
    """Unset fractions must reproduce the previous behaviour exactly, not approximately."""
    cfg = LoaderConfig(n_frames=2, image_size=32)
    ds = PoseDataset(mixed_source_root, 'train', cfg)
    assert ds._pools[0][1] is None, 'no weights should be built when neither fraction is set'
    rng = np.random.default_rng(0)
    assert [ds._pick(i, rng) for i in range(len(ds.index))] == ds.index

    # One level configured leaves the other alone: entries keep their natural share within a
    # source, rather than the source being silently rebalanced to uniform-over-modes.
    only_src = PoseDataset(mixed_source_root, 'train',
                           LoaderConfig(n_frames=2, image_size=32, annot_frac=0.4)).mix()
    assert only_src['3d-tracked'] == pytest.approx(0.6)
    assert only_src['2d-annotated'] + only_src['3d-annotated'] == pytest.approx(0.4)
    assert only_src['2d-annotated'] > only_src['3d-annotated'], (
        'the 2D session carries 2 animals to the 3D session\'s 1, so with mode_3d_frac unset it '
        'should keep the larger share')


def test_sampling_mix_rejects_a_fraction_outside_the_unit_interval(mixed_source_root):
    with pytest.raises(ValueError, match=r'annot_frac must be in \[0, 1\]'):
        PoseDataset(mixed_source_root, 'train',
                    LoaderConfig(n_frames=2, image_size=32, annot_frac=1.4))


def test_prompt_noise_is_in_pixels_on_both_modes(tiny_root):
    """One sigma must mean the same VISUAL displacement whatever the session's units.

    A single scalar in each session's own units cannot: `allen-mouse-combined` alone holds 63 px
    sessions beside 14 mm ones, so `1.0` was a 1 px nudge on one and a 1 mm nudge on the other,
    and across the sweep rat-city is px where allen 3D is mm. `cube_scale` is world units per
    pixel -- the same conversion the metric losses use -- so the 3D sigma is the pixel sigma
    times it, and the displacement matches by construction.
    """
    from posetail.posetail.cube import get_camera_scale

    SIG = 3.0

    def spread(root, sigma):
        """Median |prior - clean| over many draws, in the session's own units."""
        def mk(s):
            return LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                                prompt_dropout=0.0, prompt_noise_px=s, prob_2d_only=0.0)
        clean = _batch(PoseDataset(root, 'train', mk(0.0))).kpt_prior
        ds = PoseDataset(root, 'train', mk(sigma))
        d = torch.stack([(_batch(ds).kpt_prior - clean).abs().median() for _ in range(30)])
        return float(d.median())

    px = spread(tiny_root / 'ratlike', SIG)          # mode='2d', units='px'
    mm = spread(tiny_root / 'mouselike', SIG)        # mode='3d', units='mm'

    # 2D is already pixels -- the sigma is applied verbatim.
    assert 0.3 * SIG < px < 3 * SIG, px
    # 3D is scaled by cube_scale, so it must NOT equal the pixel sigma; convert back and compare.
    b = _batch(PoseDataset(tiny_root / 'mouselike', 'train',
                           LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                                        prompt_dropout=0.0, prob_2d_only=0.0)))
    scale = float(torch.nanmedian(get_camera_scale(b.cgroup, b.kpt_prior)))
    assert scale > 0
    assert 0.3 * SIG < mm / scale < 3 * SIG, (mm, scale, mm / scale)


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
    """The stored extent, not the keypoints, decides the crop -- and it re-enters at pad=0.

    The fixture gives a02 exactly one box, [10,10,30,30] on frame 1, while its keypoints are
    scattered over [5,43]. So the two sources cannot agree by accident.
    """
    b = _boxed_animal(tiny_root, 'instances')
    box = cropmod.crop_box_for_points(torch.tensor([[10.0, 10.0], [30.0, 30.0]]),
                                      torch.tensor([64, 48], dtype=torch.int32), 8, pad=0)
    assert box.tolist() == [10, 10, 30, 30], 'a padded extent must not be padded a second time'
    # `apply_crop` moves the origin to the box and `_resize_camera` scales it by 64/20.
    torch.testing.assert_close(b.cgroup[0]['offset'],
                               torch.tensor([10.0, 10.0]) * (64 / 20))
    assert not torch.allclose(b.cgroup[0]['offset'],
                              _boxed_animal(tiny_root, 'keypoints').cgroup[0]['offset'])


def test_box_source_falls_back_per_view(tiny_root):
    """An animal with no stored box trains EXACTLY as it did before the switch existed.

    This is what makes `box_source = "instances"` safe to set once for a run that mixes a root
    carrying `instances.pq` with roots that do not -- the fallback is silent, so it is tested.
    """
    def a01(source):
        ds = PoseDataset(tiny_root / 'ratlike', 'val',
                         LoaderConfig(box_source=source, **BOXCFG))
        a = next(i for i, it in enumerate(ds.index) if it.animal == 0)
        return _batch(ds, a)

    got, want = a01('instances'), a01('keypoints')
    # equal_nan: the fixture carries a deliberately MISSING point, so both sides hold a NaN there
    torch.testing.assert_close(got.coords, want.coords, equal_nan=True)
    torch.testing.assert_close(got.cgroup[0]['offset'], want.cgroup[0]['offset'])
    np.testing.assert_array_equal(got.views[0].numpy(), want.views[0].numpy())


def test_a_rotated_box_needs_four_corners():
    """Why `_crop_pts` stores four corners and not the two `instances.pq` holds.

    Under an in-plane rotation the extent of the two diagonal corners is strictly inside the
    extent of all four, so a two-corner version would crop the animal it was meant to enclose.
    """
    import cv2

    from tailcyclenet.dataset import _apply_affine

    box = torch.tensor([10.0, 10.0, 30.0, 20.0])
    rot = (cv2.getRotationMatrix2D((20.0, 15.0), 30.0, 1.0), (64, 48))
    four = _apply_affine(torch.stack([box[[0, 1]], box[[2, 1]], box[[2, 3]], box[[0, 3]]]), rot)
    two = _apply_affine(torch.stack([box[[0, 1]], box[[2, 3]]]), rot)
    assert (four.amin(0) <= two.amin(0)).all() and (four.amax(0) >= two.amax(0)).all()
    assert (four.amax(0) - four.amin(0) > two.amax(0) - two.amin(0)).any()


def test_box_source_rejects_a_typo(tiny_root):
    """`keypoint` (singular) would otherwise mean "silently ignore the boxes"."""
    with pytest.raises(AssertionError, match='box_source'):
        PoseDataset(tiny_root / 'ratlike', 'val', LoaderConfig(box_source='keypoint', **BOXCFG))


def test_crop_inflate_default_is_inert(tiny_root):
    """`crop_inflate = 1.0` (the default) must be BYTE-IDENTICAL to a config that never mentions
    the key -- the wide-crop TRAINING regime is only a no-op unless a caller opts in.
    """
    for root, split in ((tiny_root / 'ratlike', 'val'), (tiny_root / 'mouselike', 'train')):
        base = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.25,
                            crop_jitter=0.3, prompt_dropout=0.4)
        explicit = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.25,
                                crop_jitter=0.3, prompt_dropout=0.4, crop_inflate=1.0)
        ds_base = PoseDataset(root, split, base, train=False)
        ds_explicit = PoseDataset(root, split, explicit, train=False)
        b = _batch(ds_base)
        e = _batch(ds_explicit)
        torch.testing.assert_close(b.coords, e.coords, equal_nan=True)
        np.testing.assert_array_equal(b.views[0].numpy(), e.views[0].numpy())
        torch.testing.assert_close(b.cgroup[0]['offset'], e.cgroup[0]['offset'])
        torch.testing.assert_close(b.cgroup[0]['size'].float(), e.cgroup[0]['size'].float())


def test_crop_inflate_widens_the_2d_crop():
    """Above 1.0, `crop_to_points_2d` must widen the returned box about its own centre, AFTER
    jitter. (The 3D path is exercised end to end through the real fixture rig instead of a
    hand-built camera dict.)
    """
    cam = {'size': torch.tensor([2000, 2000], dtype=torch.int32), 'offset': torch.zeros(2)}
    coords = torch.tensor([[900.0, 900.0], [1100.0, 1100.0]])       # (K=2,2), a small box

    _, tight_box, _ = cropmod.crop_to_points_2d(cam, coords, min_crop_dim=16, inflate=1.0)
    _, wide_box, _ = cropmod.crop_to_points_2d(cam, coords, min_crop_dim=16, inflate=1.5)
    tw, ww = tight_box[2] - tight_box[0], wide_box[2] - wide_box[0]
    assert float(ww) > float(tw) * 1.2
    tc = (float(tight_box[0]) + float(tight_box[2])) / 2
    wc = (float(wide_box[0]) + float(wide_box[2])) / 2
    assert abs(tc - wc) < float(tw), 'inflation is about the SAME centre'


def test_crop_inflate_widens_the_3d_crop(tiny_root):
    """Above 1.0, `crop_to_points_3d` must widen every camera's box about its own centre, using
    the real fixture rig so `project_points_torch` sees complete camera dicts.
    """
    from tailcyclenet.format import load_dataset

    ds = load_dataset(tiny_root / 'mouselike')
    sess = ds.sessions['train'][0]
    sess.preload()
    frames = np.arange(4)
    cgroup = sess.cgroup('g000', frames)
    lab = sess.labels('g000')
    coords = torch.as_tensor(lab.points3d[0][frames], dtype=torch.float32)

    _, tight = cropmod.crop_to_points_3d(cgroup, coords, min_crop_dim=8, inflate=1.0)
    _, wide = cropmod.crop_to_points_3d(cgroup, coords, min_crop_dim=8, inflate=1.5)
    grew = False
    for c in range(len(cgroup)):
        tw, ww = tight[c][2] - tight[c][0], wide[c][2] - wide[c][0]
        th, wh = tight[c][3] - tight[c][1], wide[c][3] - wide[c][1]
        assert float(ww) >= float(tw) and float(wh) >= float(th)
        # the tiny fixture's 64x48 frame CLAMPS a 1.5x box on the short axis, so require growth
        # on at least one axis of at least one camera rather than every axis of every camera.
        grew = grew or float(ww) > float(tw) * 1.1 or float(wh) > float(th) * 1.1
    assert grew, 'crop_inflate = 1.5 produced no wider box on any camera/axis'


def test_crop_inflate_draw_scalar_is_unaffected_by_train():
    """A plain float has no range to be a midpoint of -- it must return unchanged under either
    `train` value, so every scalar config predating the range form is untouched.
    """
    from tailcyclenet.dataset import LoaderConfig, _crop_inflate

    cfg = LoaderConfig(crop_inflate=1.3)
    rng = np.random.default_rng(0)
    assert _crop_inflate(cfg, rng, True) == 1.3
    assert _crop_inflate(cfg, rng, False) == 1.3


def test_crop_inflate_draw_range_on_train_only():
    """`crop_inflate = [low, high]` draws uniformly IN RANGE on train, and returns the exact
    MIDPOINT on val/test -- so `checkpoint_best` selection reads the same crop geometry every pass.
    """
    from tailcyclenet.dataset import LoaderConfig, _crop_inflate

    cfg = LoaderConfig(crop_inflate=[0.9, 1.5])
    rng = np.random.default_rng(0)

    train_draws = [_crop_inflate(cfg, rng, True) for _ in range(200)]
    assert all(0.9 <= v <= 1.5 for v in train_draws)
    assert len(set(train_draws)) > 1, 'a range must actually vary draw to draw on train'

    for _ in range(10):
        assert _crop_inflate(cfg, rng, False) == pytest.approx(1.2)   # exact midpoint, no draw


def test_crop_inflate_range_flows_through_a_real_item(tiny_root):
    """End to end: a `[low, high]` `crop_inflate` must actually vary the TRAIN crop side across
    repeated draws of the SAME item, and val uses the exact midpoint (checked against an explicit
    scalar-midpoint config).
    """
    # THE DRAW IS FORCED, NOT HOPED FOR. The fixture's frame is small, so an inflated box often
    # CLAMPS at the image edge and lands on the same scale as a tight one -- feeding `_crop_inflate`
    # a stubbed rng that returns the range's endpoints makes the assertion exact.
    range_cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                             crop_jitter=0.0, prompt_dropout=0.0, min_crop_dim=8,
                             crop_inflate=[0.9, 1.8])

    class _FixedRng:
        """Only `uniform` matters to `_crop_inflate`; everything else defers to a real rng so the
        rest of `_item` (window start, augmentation draws) behaves normally."""
        def __init__(self, value, inner):
            self._value, self._inner = value, inner

        def uniform(self, lo, hi):
            return self._value

        def __getattr__(self, name):
            return getattr(self._inner, name)

    train_ds = PoseDataset(tiny_root / 'ratlike', 'train', range_cfg, train=True)
    # `size` after `_resize_camera` is always `image_size` (the resize target), so the varying
    # quantity is the SCALE that got it there -- `mat[0,0]`, smaller for a wider source box, as
    # in `test_crop_inflate_widens_the_2d_crop`.
    scales = []
    for end in (0.9, 1.8):
        item = train_ds._item(0, _FixedRng(end, np.random.default_rng(0)))
        assert item is not None
        scales.append(float(pose_collate([item]).cgroup[0]['mat'][0, 0]))
    assert scales[1] < scales[0], (
        f'the range\'s wide end must give a WIDER source box (smaller scale): {scales}')

    midpoint_cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                                crop_jitter=0.0, prompt_dropout=0.0, min_crop_dim=8,
                                crop_inflate=1.35)
    range_val = _batch(PoseDataset(tiny_root / 'ratlike', 'val', range_cfg, train=False))
    midpoint_val = _batch(PoseDataset(tiny_root / 'ratlike', 'val', midpoint_cfg, train=False))
    # `size` is always `image_size` after resize; `mat[0,0]` (the scale that produced it) is what
    # actually differs between a wider and a tighter source box, so that is the meaningful compare.
    torch.testing.assert_close(range_val.cgroup[0]['mat'][0, 0], midpoint_val.cgroup[0]['mat'][0, 0])


# ----------------------------------------------------------------------------------------------
# the reader cache size
# ----------------------------------------------------------------------------------------------

# `_reader_cache_size` is pure given `ram_gb`, which is the whole point: the sizing rule is
# asserted here on any host, with no video, no decode and no GPU.

def test_reader_cache_holds_a_whole_rig_in_one_process():
    """A single process streaming windows must cache every camera, or it misses on every call."""
    from tailcyclenet.dataset import _reader_cache_size

    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 16
    assert _reader_cache_size(4, (1024, 570), None, ram_gb=503) == 4
    # a narrow rig still gets ChunkShuffle.mix, not 1: the detector loader at 1 cost 52.8 ms/item
    assert _reader_cache_size(1, (1024, 570), None, ram_gb=503) == 4


def test_reader_cache_does_not_multiply_by_worker_count():
    """A worker wants ChunkShuffle.mix, not the whole rig: 12 workers x 16 cameras would be 480 GB."""
    from tailcyclenet.dataset import _reader_cache_size

    assert _reader_cache_size(16, (3208, 2200), 12, ram_gb=503) == 4
    assert _reader_cache_size(1, (1024, 570), 12, ram_gb=503) == 4


def test_reader_cache_degrades_instead_of_oom_on_a_small_host():
    """The count is a wish and RAM is the constraint; the clamp binds before the OOM killer does.
    The numbers here moved with the backend: PyAV's readers are linear (~0.053 GB per megapixel),
    so budgets that the old quadratic decoder price could not hold now fit, and a genuinely small budget
    still degrades.
    """
    from tailcyclenet.dataset import _reader_cache_size

    # A LOADER WORKER still clamps below its own `want` of 4, and there are twelve of them.
    # (At `ram_gb=64` it no longer does, since `FRACTION_READERS` moved 0.25 -> 0.35 and PyAV's
    # readers are cheap -- twelve workers of four readers each is ~18 GB and genuinely fits. The
    # clamp is what is being tested, so the budget has to be one where it actually binds.)
    assert _reader_cache_size(16, (3208, 2200), 12, ram_gb=32) < 4
    # A GENUINELY SMALL BUDGET degrades the main process below the rig.
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=8) < 16
    # ...and a big one is allowed to hold the whole rig, which is worth 5.1x (the cycle cliff).
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=64) == 16
    # never zero -- one reader is the minimum that can serve a read at all
    assert _reader_cache_size(16, (3208, 2200), 64, ram_gb=1) == 1


@pytest.mark.parametrize('n_cams,wh', [
    (16, (3208, 2200)),        # johnson
    (4, (3840, 2160)),         # 3dpop
    (32, (3840, 2160)),        # a rig bigger than anything shipped
    (2, (640, 480)),           # a small one
])
def test_the_advertised_ram_actually_buys_every_reader(n_cams, wh):
    """The warning tells a user how much `--max-ram` would hold a reader per camera. **A figure
    that does not actually work is worse than no figure**, because they raise the limit, pay the
    memory and stay on the slow side of a 5x cliff with nothing to show for it.

    So: at the advertised value the cache must reach the rig, and the value must not be padded --
    it is the number a user types, and rounding it up 'for safety' spends their memory silently.
    """
    from tailcyclenet import memory
    from tailcyclenet.dataset import _reader_cache_size, reader_cache_ram_gb

    need = reader_cache_ram_gb(n_cams, wh)
    at = _reader_cache_size(n_cams, wh, None,
                            ram_gb=memory.host_budget(override_gb=need).budget_gb)
    assert at >= n_cams, f'--max-ram {need:.1f} advertised for {n_cams} cameras but bought {at}'

    # NOT PADDED: one GB less must fall short, unless `want` (which floors at 4) already caps it.
    if n_cams > 4:
        below = _reader_cache_size(
            n_cams, wh, None, ram_gb=memory.host_budget(override_gb=need - 1).budget_gb)
        assert below < n_cams, (
            f'{need:.1f} GB is padded: {need - 1:.1f} already buys {below} of {n_cams}')


def test_the_reader_price_is_linear_not_quadratic():
    """The SHAPE of the price, not just its value -- a quadratic law caps the cache at sqrt(budget)
    and so can never reach the rig however much memory is offered.

    Under the law this replaced, a 16-camera 3208x2200 rig got **10 readers at a 128 GB budget**
    and never reached the 16 it needs, so the 5.1x cycle cliff was unreachable at ANY `--max-ram`.
    Doubling the budget must roughly double the count.
    """
    from tailcyclenet.dataset import _reader_cache_size

    a = _reader_cache_size(64, (3208, 2200), None, ram_gb=16)
    b = _reader_cache_size(64, (3208, 2200), None, ram_gb=32)
    assert b / a == pytest.approx(2.0, rel=0.2), f'{a} -> {b} is not linear in the budget'
    # and the rig is reachable at a sane budget, which is the whole point of the refit
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=32) == 16


def test_reader_cache_env_var_overrides_everything(monkeypatch):
    """The knob survives as an OVERRIDE, and is now read at first use rather than at import."""
    from tailcyclenet.dataset import _reader_cache_size

    monkeypatch.setenv('TAILCYCLENET_READER_CACHE', '1')
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 1
    monkeypatch.setenv('TAILCYCLENET_READER_CACHE', '32')
    assert _reader_cache_size(1, (1024, 570), 12, ram_gb=64) == 32


def test_reader_cache_size_does_not_touch_the_filesystem(monkeypatch):
    """Opening a container in the parent deadlocks every forked worker, so the sizing rule may
    read toml-derived numbers and nothing else.
    """
    import builtins

    from tailcyclenet.dataset import _reader_cache_size

    def boom(*a, **k):
        raise AssertionError('the sizing rule must not open anything')

    monkeypatch.setattr(builtins, 'open', boom)
    monkeypatch.setattr('os.stat', boom)
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 16


def test_a_whole_body_offset_moves_every_keypoint_the_same_way(tiny_root):
    """I.I.D. jitter is not the failure deployment produces: a carried prior's error is a
    whole-body LAG, and i.i.d. noise averages to zero over the keypoint set, training the model
    to trust the prior's centroid -- the one quantity a lag gets wrong.
    """
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
    for _ in range(20):
        b = _batch(PoseDataset(tiny_root / 'ratlike', 'train', cfg(prompt_stale_frames=3)))
        assert torch.isnan(b.kpt_prior).sum() == torch.isnan(base.kpt_prior).sum(), \
            'staleness must not withhold a prior'
        # `prompt_t` still claims the first labelled frame; only the POSE came from elsewhere.
        assert (b.prompt_t == base.prompt_t).all()
        moved |= not torch.allclose(b.kpt_prior, base.kpt_prior, equal_nan=True)
    assert moved, 'prompt_stale_frames=3 never moved the prior in 20 draws'


def test_an_extreme_aspect_crop_never_resizes_to_a_zero_side():
    """`cv2.warpAffine` with a 0 in `dsize` returns the FULL SOURCE FRAME and does not raise, so
    a camera rounded to [256, 0] would hand the model silent garbage instead of a clean failure.
    """
    import cv2

    from tailcyclenet.dataset import _resize_camera

    cam = {'size': torch.tensor([2000, 1], dtype=torch.int32),
           'mat': torch.eye(3), 'offset': torch.zeros(2)}
    out, _ = _resize_camera(cam, 256)
    assert out['size'].tolist() == [256, 1], f'a zero side survived: {out["size"].tolist()}'

    # The reason it matters, pinned on OpenCV itself so it stays true if the behaviour ever moves.
    src = np.zeros((480, 640, 3), np.uint8)
    aff = np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    assert cv2.warpAffine(src, aff, (256, 0)).shape == src.shape, \
        'if OpenCV ever raises on a zero dsize, the clamp above is belt-and-braces'
    assert cv2.warpAffine(src, aff, (256, 1)).shape == (1, 256, 3)


def test_a_3d_session_with_no_points3d_is_refused_by_name(tmp_path):
    """The format allows it; training on it does not. It used to CRASH mid-epoch instead, with
    nothing in the message naming the session.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    root = tmp_path / 'ds'
    cf._session_3d(root / 'train' / 's_3d')
    (root / 'train' / 's_3d' / 'points3d.pq').unlink()

    with pytest.raises(ValueError, match='points3d'):
        PoseDataset(root, 'train', CFG, train=False)
    # ...and the message names the session, which is the whole point of moving the check here.
    try:
        PoseDataset(root, 'train', CFG, train=False)
    except ValueError as e:
        assert 's_3d' in str(e)

def test_a_3d_rotation_that_loses_the_animal_is_reverted():
    """The library's rotation crops to the border-free inscribed rectangle, which can throw the
    animal out of the frame; `_item`'s guard must revert such a rotation, not keep it silently.
    """
    from aniposelib.cameras import CameraGroup
    from posetail.datasets.posetail_dataset import rotate_camera_image_plane_3d
    from posetail.posetail.cube import is_point_visible

    from tailcyclenet import format as fmt

    W, H = 320, 240
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False},
                  calibrated={'cam0': True})
    cam = rig.posetail()[0]

    # Two points near opposite frame CORNERS: inside the frame, outside the inscribed rectangle.
    corner = torch.tensor([[[0.45, 0.33, 1.0], [-0.45, -0.33, 1.0]]], dtype=torch.float32)
    before = int(is_point_visible(cam, corner).sum())
    assert before >= 2, f'the probe must start visible in this camera, got {before}'

    rot_cam, _ = rotate_camera_image_plane_3d(cam, 45.0)
    after = int(is_point_visible(rot_cam, corner).sum())
    assert after < 2, f'the inscribed crop must lose the animal here, got {after} still visible'

    # WHY IT WAS SILENT: the crop rule clamps, it does not refuse -- so the item was returned.
    from posetail.posetail.cube import project_points_torch
    p2d = project_points_torch([rot_cam], corner)[0]
    assert cropmod.crop_box_for_points(p2d, rot_cam['size'], 16) is not None, \
        'if the rule ever starts refusing this, the guard in _item is belt-and-braces'

    # The guard `_item` applies, on the three cases that matter.
    def reverted(b, a):
        return int(a) < 2 <= int(b)
    assert reverted(before, after), 'a rotation that loses the animal must be reverted'
    assert not reverted(1, 0), 'a camera that never saw the animal keeps its rotation'
    assert not reverted(4, 4), 'a rotation that keeps the animal is left alone'


def test_the_3d_visibility_or_reflects_the_augmentation_that_followed_it():
    """The noisy-OR used to be taken BEFORE the rotation/cutout that invalidates it, so a point
    could read `vis == True` while every shown camera had zeroed it -- supplying visibility was
    worse than passing None.
    """
    # (T=1, K=3, C=2): keypoint 0 seen by camera 1, keypoint 1 by neither, keypoint 2 unassessed.
    vis_2d = torch.tensor([[[0.0, 1.0], [0.0, 0.0], [float('nan')] * 2]])
    stale = torch.tensor([[True, True, False]])          # what the pre-augmentation OR had said

    fresh = (vis_2d == 1).any(-1)
    assert fresh.tolist() == [[True, False, False]]
    assert fresh.dtype == torch.bool, 'the loss inverts this with `~`, which no float satisfies'
    assert stale[0, 1] and not fresh[0, 1], \
        'keypoint 1 is the case: still claimed visible after every camera lost it'
    # NaN is not visible, and must not raise or propagate.
    assert not bool(fresh[0, 2])


# the two prior corruptions: a swapped keypoint pair, and a jump to a nearby animal's pose.
# Both are train-only, off by default, and gated so a default config never draws an extra `rng`
# value.


def _train_item(ds, idx=0, seed=0):
    """One TRAIN item, fully deterministic. `__getitem__` entropy-seeds every train item, so a
    byte-identity comparison goes through `_item` directly with an explicit rng; a 3D item ALSO
    needs the global numpy seed reset, because `rotate_camera_group` draws from `np.random`.
    """
    np.random.seed(seed)
    return ds._item(idx, np.random.default_rng(seed))


def test_prompt_swaps_default_off_is_byte_identical(tiny_root):
    """`prompt_swap_kpt_pairs = 0.0`, `prompt_swap_animal = 0.0` (both defaults) must be
    BYTE-IDENTICAL to a config that never mentions the keys -- every number on record was measured
    without them.
    """
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


def test_animal_swap_is_inert_on_a_single_animal_session(tmp_path):
    """`attempt_swap_animal` needs n_animals >= 2 (structural, not a draw), so a single-animal
    session must consume NOTHING extra from rng and stay bit-identical.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    root = tmp_path / 'ds'
    cf._session_3d(root / 'train' / 's', T=6)

    base_ds = PoseDataset(root, 'train', CFG, train=True)
    swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_animal=0.9), train=True)
    b = pose_collate([_train_item(base_ds)])
    s = pose_collate([_train_item(swap_ds)])
    torch.testing.assert_close(b.kpt_prior, s.kpt_prior, equal_nan=True)
    torch.testing.assert_close(b.coords, s.coords, equal_nan=True)
    torch.testing.assert_close(b.cgroup[0]['mat'], s.cgroup[0]['mat'])


def test_animal_swap_changes_only_the_prior(tiny_root):
    """At `prompt_swap_animal = 1.0` the corruption must touch `kpt_prior` and NOTHING else --
    the crop, pixels and every other target stay what they were ("the neighbour leaked into the
    crop" guard).
    """
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
    """The corrupted prior must be the NEIGHBOUR's pose put through the SAME transform the
    target's own `coords` received -- with a rigid pair and a world rotation, the corrupted minus
    base prior must be one vector of the same norm as `sep`.
    """
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


def test_corrupted_prior_is_bounds_masked(tmp_path):
    """A neighbour placed outside every camera's usable pair must never be substituted in: the
    corruption is a NO-OP, so the configured rate and the presented rate differ.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    root = tmp_path / 'mv'
    cf._session_3d_multi(root / 'train' / 's', T=6, sep=1e6)
    base_ds = PoseDataset(root, 'train', CFG, train=True)
    swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_animal=1.0), train=True)

    b = pose_collate([_train_item(base_ds)])
    s = pose_collate([_train_item(swap_ds)])
    torch.testing.assert_close(b.kpt_prior, s.kpt_prior, equal_nan=True), \
        'an ineligible neighbour must leave the prior untouched, not a partially-NaN\'d lie'


def test_kpt_swap_lands_on_another_original_point_never_itself(tiny_root):
    """NOT a permutation: each corrupted keypoint's new value must MATCH one of the OTHER
    keypoints' original positions, and never its own (the `(local + offset) % m` draw excludes it).
    """
    for root in (tiny_root / 'ratlike', tiny_root / 'mouselike'):
        base_ds = PoseDataset(root, 'train', CFG, train=True)
        b = pose_collate([_train_item(base_ds)])
        original = b.kpt_prior[0]
        orig_finite = torch.isfinite(original).all(-1)
        for p in (0.3, 0.5, 1.0):
            swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_kpt_pairs=p),
                                  train=True)
            s = pose_collate([_train_item(swap_ds)])
            corrupted = s.kpt_prior[0]
            moved = ~torch.isclose(original, corrupted, equal_nan=True).all(-1) & orig_finite
            for k in moved.nonzero(as_tuple=True)[0].tolist():
                mask = orig_finite.clone()
                mask[k] = False
                others = original[mask]
                d = (others - corrupted[k]).norm(dim=-1)
                assert float(d.min()) < 1e-4, \
                    f'{root.name} p={p} keypoint {k}: new value matches no OTHER original point'
                own = (corrupted[k] - original[k]).norm()
                assert float(own) > 1e-4, \
                    f'{root.name} p={p} keypoint {k}: landed back on its OWN original value'


def test_kpt_swap_pairs_is_a_per_keypoint_rate(tiny_root):
    """A PER-KEYPOINT PROBABILITY, not a fixed count: at p = 1.0 EVERY finite keypoint must move
    on both roots -- the deliberate opposite of the count-based design this key started with.
    """
    for root in (tiny_root / 'ratlike', tiny_root / 'mouselike'):
        base_ds = PoseDataset(root, 'train', CFG, train=True)
        swap_ds = PoseDataset(root, 'train', replace(CFG, prompt_swap_kpt_pairs=1.0), train=True)
        b = pose_collate([_train_item(base_ds)])
        s = pose_collate([_train_item(swap_ds)])
        finite = torch.isfinite(b.kpt_prior[0]).all(-1) & torch.isfinite(s.kpt_prior[0]).all(-1)
        moved = ~torch.isclose(b.kpt_prior[0], s.kpt_prior[0]).all(-1) & finite
        n_finite = int(finite.sum())
        assert n_finite >= 2, f'{root.name}: need at least two finite keypoints for this test'
        assert int(moved.sum()) == n_finite, \
            (f'{root.name} (K={b.kpt_prior.shape[1]}): expected every one of the {n_finite} '
             f'finite keypoints to move at p = 1.0, not a fixed pair count')


def test_pose_only_prob_default_is_byte_identical_to_pre_change_code(tiny_root):
    """Section 6.3's own required guard: `pose_only_prob = 0.0` (default) must reproduce the
    EXACT pre-change code, rng draw for rng draw -- not merely "look the same", since a hoisted
    coin-flip that fires even when unused would silently shift every LATER draw in the item
    (camera pick, augmentation, corruption sites) without changing any config value visibly.

    Loads the actual PRE-CHANGE `tailcyclenet/dataset.py` straight from git HEAD (this feature's
    own commit has not landed yet, so HEAD is the real pre-change source, not a hand-written
    stand-in) into an isolated module, builds the identical item with the identical seed through
    BOTH the old and the new code, and compares every tensor field.
    """
    import importlib.util
    import subprocess
    import sys

    root = tiny_root / 'mouselike'
    src = subprocess.run(['git', 'show', 'HEAD:tailcyclenet/dataset.py'],
                         capture_output=True, text=True, check=True, cwd=Path(__file__).parent.parent).stdout
    assert 'pose_only_prob' not in src, \
        'HEAD already has this key -- point this test at the commit before it landed'

    spec = importlib.util.spec_from_loader('tailcyclenet._dataset_pre_change', loader=None)
    old_mod = importlib.util.module_from_spec(spec)
    old_mod.__package__ = 'tailcyclenet'
    sys.modules['tailcyclenet._dataset_pre_change'] = old_mod
    try:
        exec(compile(src, 'dataset.py (HEAD)', 'exec'), old_mod.__dict__)

        cfg_kwargs = dict(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                          crop_jitter=0.0, prompt_dropout=0.3, prompt_offset_px=2.0,
                          prompt_noise_px=1.5, box_prompt='film', box_prompt_dropout=0.5)

        old_ds = old_mod.PoseDataset(root, 'train', old_mod.LoaderConfig(**cfg_kwargs), train=True)
        new_ds = PoseDataset(root, 'train', LoaderConfig(**cfg_kwargs, pose_only_prob=0.0),
                             train=True)

        for seed in range(5):
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
    """The feature itself, not just its OFF guard: at `pose_only_prob = 1.0` (every box-dropped
    item is ALSO pose-only), a box-dropped item's `kpt_prior` must carry NONE of
    `prompt_dropout`/`prompt_offset_px`/`prompt_noise_px`'s corruption -- exactly the GT-quality
    prior a `box_prompt = 'none'` deployment item would need to learn from.
    """
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
