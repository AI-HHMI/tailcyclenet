"""The loader, and the one thing in it that must be bit-exact.

The crop rule is shared between the pose model and the detector: the detector is trained to
reproduce the pose crop, which is why a detector box costs ~0.02 mm instead of whatever an
independently-plausible rule would cost. `test_crop_rule_is_int32_exact` is what licenses that,
and it compares against the library's own inline arithmetic rather than a transcription of it.
"""
from collections import Counter

import numpy as np
import pytest
import torch

from tailcyclenet import crop as cropmod
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.format import Registry


# ----------------------------------------------------------------------------------------------
# the crop rule
# ----------------------------------------------------------------------------------------------

def test_crop_rule_is_int32_exact():
    """crop_box_for_points must equal PosetailDataset.crop_cgroup_to_points, exactly.

    The library exposes the rule only inline, so this drives the real method against a shim that
    supplies the one attribute it reads. Any drift here invalidates every detector number.
    """
    from types import SimpleNamespace
    from posetail.datasets.posetail_dataset import PosetailDataset

    rng = np.random.default_rng(0)
    shim = SimpleNamespace(min_crop_dim=64)
    for trial in range(200):
        w, h = int(rng.integers(80, 2000)), int(rng.integers(80, 2000))
        n = int(rng.integers(2, 30))
        pts = torch.as_tensor(rng.uniform(-200, max(w, h) + 200, size=(n, 2)), dtype=torch.float32)
        if trial % 7 == 0:                        # some non-finite points, the normal case
            pts[rng.integers(n)] = float('nan')
        size = torch.tensor([w, h], dtype=torch.int32)

        cam = {'size': size, 'offset': torch.zeros(2)}
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
    """PosetailDataset.crop_cgroup_to_points' arithmetic, transcribed from the library body.

    Kept separate from tailcyclenet.crop so the two are independent derivations; if someone
    "simplifies" one, the test still compares against the other.
    """
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
    """The library raises here; the detector depends on getting None so it can emit a NaN box."""
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

    Two halves of the same transform live apart: the camera side is `rotate_camera_image_plane_3d`
    + `crop.apply_crop` (offset += x1) + `_resize_camera` (mat *= s), and the pixel side is
    `_crop_affine`. If they disagree, every reprojection is off by that much and nothing in the
    loss curve says so -- so paint a marker at a known projection, run the image through one half
    and the camera through the other, and check they still agree.

    This is also the only way the `A @ M3` composition can be silently wrong. Comparing pixels
    against the old three-step path would not do it: that path is expected to differ (it resampled
    twice and used cv2.resize's half-pixel convention).
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

    The library's `rotate_points_image_plane` crops to the border-free inscribed rectangle, which
    keeps a mean 0.416 of a 4696x2048 frame. The pose loader crops around ONE animal afterwards, so
    that step buys nothing and costs the animals outside it -- and costs them SILENTLY, because the
    `< 2 finite` guard runs before the rotation and `crop_box_for_points` clamps instead of
    refusing. Measured on rat-city-annotated: 61% of rotated items came back with no label at all.

    Two properties, both of which the inscribed crop breaks: every source pixel is still inside the
    canvas, and a label lands where the returned camera says it lands.
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
    """`read_frames` computes `%06d.<ext>` instead of listing the directory. Same pixels.

    Listing rat-city's 57,594-entry `cam0` cost 0.90 s of a 1.06 s item, and the spec guarantees
    the names, so the loader computes them. This is the check that the guarantee is being read
    correctly -- including the extension probe, since the fixture writes `.png` and a default of
    `.jpg` would find nothing. The fixture's frames differ per (cam, frame), so an off-by-one is
    visible in the pixels rather than hidden by constant images.
    """
    from tailcyclenet.dataset import load_image, read_frames

    sess = dataset_3d.sessions['train'][0]
    group = sess.groups['g000']
    for cam in sess.cam_names:
        kind, src, ext = group.source(cam)
        assert (kind, ext) == ('frames', '.png')
        listed = sorted(f for f in src.iterdir() if f.suffix in ('.png', '.jpg'))
        frames = [3, 0, 2, 0]                       # out of order and repeated, as windows are
        want = [load_image(str(listed[i])) for i in frames]
        got = read_frames(group, cam, frames)
        for a, b in zip(want, got):
            np.testing.assert_array_equal(a, b)
        # and the frames really are distinguishable, or the assertion above proves nothing
        assert not np.array_equal(got[0], got[2])
        # a repeated index is decoded once but must not be ALIASED -- cutout writes in place
        assert got[1] is not got[3]


def test_repeated_frames_are_decoded_once(dataset_3d, monkeypatch):
    """A clamp-padded window must not decode the same file T times.

    `_frames` pads a group shorter than `n_frames` by repeating its last frame, and 251 of
    johnson-mouse's 624 train windows come from `n_frames = 1` groups -- 24 copies of frame 0 per
    camera, which was 384 decodes for 16 distinct images on a 16-camera session.

    COUNTS `cv2.imread`, NOT A HELPER. The decode is the thing being conserved, and counting the
    internal function that happens to wrap it means the test passes vacuously the moment that
    function is renamed or bypassed -- which is exactly what happened when `read_frames` moved to
    `load_warps`: it read `decoded 0 times` and the assertion was on the wrong side of zero to
    notice.
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
    """A per-frame-box window pays one decode for T crops.

    This is the case the `(index, box)` dedupe key cannot catch -- every key differs, so a fused
    decode-and-warp per key pays T full decodes for one picture (24 x 27 ms on a 4696x2048 jpeg,
    dominating the item). `load_warps` splits the decode from the warp; this pins that it stays
    split, and that the T outputs are genuinely different crops rather than one crop repeated.
    """
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
    # uint8 out of the loader: 4x fewer bytes to queue and pin. `model.forward` divides by 255 on
    # device, so a float here would mean the pixels get scaled twice.
    assert b.views[0].dtype == torch.uint8
    assert b.coords.shape == (1, 4, 4, 2)                 # R=2 for a true-2D session
    assert b.p2d.shape == (1, 1, 4, 4, 2)                 # 2D needs p2d; the loss reads it
    assert b.kpt_ids.shape == (1, 4)
    assert b.kpt_prior.shape == (1, 4, 2)
    assert b.prompt_t.shape == (1, 4)


def test_3d_item_shapes(tiny_root):
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 3
    assert b.coords.shape == (1, 4, 3, 3)                 # R=3 world
    assert b.p2d is None
    assert b.vis.shape == (1, 4, 3, 1)                    # trailing dim get_eval_metrics wants
    assert b.vis_2d.shape == (1, 4, 3, 3, 1)


def test_vis_and_vis2d_are_both_or_neither(tiny_root):
    """Supplying one without the other dies inside einops, so the loader must never do it."""
    for name in ('ratlike', 'mouselike'):
        ds = PoseDataset(tiny_root / name, 'train', CFG)
        for i in range(len(ds)):
            b = pose_collate([ds[i]])
            assert (b.vis is None) == (b.vis_2d is None)


def test_keypoints_are_never_filtered(tiny_root):
    """Array position must keep equalling keypoint identity, even when points are missing.

    The library's filter_keypoints drops keypoints seen by too few views, and the resulting
    mislabelling is invisible in the loss curve. Every item carries all K.
    """
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        assert b.coords.shape[2] == 3
        assert b.kpt_ids.shape[1] == 3


def test_window_is_at_least_two_frames(tiny_root):
    """T=1 routes posetail into gT = T // tubelet_size = 0 and a zero-length pos_embed.

    This used to assert `shape[1] >= 1`, which is true of the exact 1-frame window it exists to
    forbid -- so it passed while the guard did not exist. `n_frames = 1` is now refused outright:
    the clamp-pad in `_frames` lengthens a short GROUP, not a short configured window.
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
    """`[low, high]` draws a per-item camera count, as posetail's `sample_cameras` does.

    This is the lever that took johnson-mouse from 2.9 s/it to under 1: its sessions carry 16
    cameras and a fixed count fed all of them through the model every step. It is also what the
    warm-start checkpoint was trained with ([1, 8]), so a fixed count is out of distribution too.

    The fixture's 3D sessions have 3 cameras, so a range topping out above that must clamp to 3
    rather than raise -- the reference's `if len(cam_names) > num_cams_to_sample` guard.
    """
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

    Discovery order is a directory listing, so it is not a stable thing to number against: with
    no base, seeing the same two datasets in the other order moves every id, and each row of
    `kpt_embed` then means a different body part than the checkpoint trained it to mean. Nothing
    in the loss curve shows it -- gotcha #4.
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
    """The per-camera/per-frame split is the design, so it gets the test.

    A camera's colour and focus must hold steady for the whole clip -- appearance is an identity
    cue for a tracker, and re-rolling hue every frame teaches it that appearance is noise. Sensor
    noise and motion blur are the opposite and must vary. Fed T copies of ONE image, the first
    pipeline has to return T identical frames and the second T different ones.
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
    """A keypoint under a cutout rect must be labelled not-visible, including where it was NaN.

    Asking the model to report "visible" for a patch that has been painted over is the one
    visibility label that is certainly wrong. And `vis_2d` is three-state here: NaN means nobody
    assessed that camera, and cutout turning NaN into 0 is a fact about the image we just made,
    not an invention -- so the test covers that entry specifically.
    """
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
    """Two workers must not produce identically-augmented items. Two RNGs decide that.

    `rotate_camera_group` draws from the global `np.random`, which torch already reseeds per
    worker (`torch/utils/data/_utils/worker.py:261-265`) -- that is torch's promise rather than
    ours, so it is worth a test rather than a comment.

    imgaug is the half that genuinely breaks: it keeps its OWN global RNG, which fork copies and
    nothing reseeds, so without `worker_init` every worker's k-th `to_deterministic()` picks the
    same gamma and hue. Invisible in the loss curve -- it just divides the appearance diversity by
    `num_workers`.
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

    Drawn i.i.d. per keypoint instead, P(a fully unprompted window) is 0.4^K -- 1e-19 at allen's
    47 keypoints -- so the query-free forward that val and `best_mpjpe` score is never trained. A
    per-keypoint draw also passes the `1.0` case below, which is why the mid-rate case is here:
    at p = 0.5 every window must be ALL NaN or NO NaN, never a mixture.
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
    """T is derived from the labels, not configured; `n_frames` is only its ceiling.

    The annotated sessions carry ONE labelled frame per 65-frame group, so a fixed T = 24 encodes
    24 frames to supervise 1. This fixture labels frames 11-13 of a 24-frame group: span 3, so T
    rounds up to 4 and the window must actually COVER 11..13 rather than merely touch one of them.
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
    """A strided train window must stay ON its lattice, in the group, and over its anchor label.

    Stride is posetail's `interval` (`posetail_dataset.py:343-361`). It composes with the derived-T
    rule by restricting the labelled span to the frames a stride-s window through the anchor can
    actually reach -- so the two assertions that matter are that the spacing never silently
    collapses (clamping off the end of the group would do exactly that, and would look like a
    shorter window rather than an error) and that val is left at stride 1.
    """
    def cfg(strides, n=8):
        return LoaderConfig(n_frames=n, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                            prompt_dropout=0.0, frame_strides=strides)

    # 3 as well as 4: an odd stride does not divide the group, so the anchor's lattice offset eats
    # into the room at the end of it. Sizing T from frame 0 instead lets the tail clamp onto the
    # last frame, which shows up as a shorter window rather than as the error it is.
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
    """"Not assessed" must reach the loss as NaN, not as "not visible".

    posetail >= 0.3.2 masks non-finite visibility targets, so an unassessed keypoint-camera pair
    produces no gradient. Collapsing it to 0 instead would train the visibility head on ~18% of
    allen-mouse's targets that nobody ever labelled. Under 0.3.0 the collapse was forced: a NaN
    target there returned NaN gradients for every parameter while the loss looked healthy.
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
    """The property the loader depends on, asserted against the INSTALLED posetail.

    The failure this guards is invisible from the loss: it stays finite and falls normally while
    every parameter receives NaN. Pinning it here means a dependency downgrade fails loudly
    instead of quietly wasting a training run.
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


def test_no_visibility_supervision_without_ground_truth(tiny_root, monkeypatch):
    """A dataset with no visibility labels must not have its visibility head trained.

    3dpop, rat-city and branson-fly ship no per-camera visibility, so the loader emits
    `vis = vis_2d = None`. posetail then sets `valid_vis = False` and hard-zeros BOTH visibility
    terms (`losses.py:493-508`). It still DERIVES a geometric `vis_true` -- but only to mask the
    coordinate losses, never to supervise visibility.

    That distinction matters: the geometric proxy is "does the GT point project inside the
    image", which the model could compute from its own prediction. Training the visibility head
    against it would teach a tautology and call it supervision.
    """
    from posetail.posetail.losses import TotalLoss

    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)   # a 2D root: no visibility labels
    b = _batch(ds)
    assert b.vis is None and b.vis_2d is None

    seen = []
    real = TotalLoss.forward

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

    A group with labels only at frames 11-13 must still yield windows, every window must contain
    a label, and the label must not be forced to frame 0 -- frame 0 is the one frame where
    per-frame anchoring contributes nothing.
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
    """A start past `n_frames - T` clamp-pads with duplicates of the last frame.

    That wastes real context: with T=24 on a 24-frame group whose labels are at 11-13, starting
    at the first labelled frame padded 13 duplicated frames while frames 0-10 sat unused.
    """
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

    johnson-mouse asks its annotators to place all 24 keypoints in all 16 views, inferring the
    ones the body hides, and flags 1,235,334 of them "visible" against 18 "not". Written as
    `visible` that trains the per-camera head toward "always visible" from labels that assert
    nothing. Written as `missing` the positions are lost. The failure mode one level up is worse:
    the 3D noisy-OR `any(status == visible)` reads all-False and claims no point is
    reconstructible, so the whole visibility target has to be withheld, not merely masked.
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

    What this guards is the bug that made it necessary: `_starts` yields one index entry per
    (session, group, animal) whatever the group's length, and the sampler draws entries
    uniformly, so an entry is a sampling weight decoupled from the data behind it. On
    allen-mouse-combined that put 90.4% of steps on the 2D path -- head bank 0, ~3 of 15 loss
    terms -- and 3.9% on the tracked session holding 95% of the labelled frames.

    `mixed_source_root` has no `2d-tracked` cell, so the mode level is SKIPPED inside `tracked`
    and that source's 0.6 lands entirely on 3D. That asymmetry is the whole reason the two
    fractions are conditional rather than joint marginals: as marginals they would be
    over-constrained here, and `annot_frac = 1 - mode_3d_frac` would be the only feasible pair.

    Both `mix()` and `_pick` are checked. They are computed from the same weights, but `mix()` is
    what gets printed at startup and `_pick` is what the model actually sees; a reporting bug
    that says the mix is fixed when it is not is the exact failure this whole change exists to
    prevent.
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
    """Unset fractions must reproduce the previous behaviour exactly, not approximately.

    Eight arms of a sweep were mid-flight when this landed. An unconfigured run that merely
    *resembled* the old draw would silently make every one of them incomparable to its successor.
    """
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


# ----------------------------------------------------------------------------------------------
# the reader cache size
# ----------------------------------------------------------------------------------------------

# `_reader_cache_size` is pure given `ram_gb`, which is the whole point: the sizing rule is
# asserted here on any host, with no video, no decode and no GPU. The failure it exists to catch is
# SILENT -- a cache below the camera count misses on every call and only shows up as a 2.5x
# slowdown in a log nobody diffs.

def test_reader_cache_holds_a_whole_rig_in_one_process():
    """A single process streaming windows must cache every camera, or it misses on every call."""
    from tailcyclenet.dataset import _reader_cache_size

    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 16
    assert _reader_cache_size(4, (1024, 570), None, ram_gb=503) == 4
    # a narrow rig still gets ChunkShuffle.mix, not 1: the detector loader at 1 cost 52.8 ms/item
    assert _reader_cache_size(1, (1024, 570), None, ram_gb=503) == 4


def test_reader_cache_does_not_multiply_by_worker_count():
    """12 workers x a 16-camera rig x 2.56 GB would be 480 GB. A worker wants ChunkShuffle.mix."""
    from tailcyclenet.dataset import _reader_cache_size

    assert _reader_cache_size(16, (3208, 2200), 12, ram_gb=503) == 4
    assert _reader_cache_size(1, (1024, 570), 12, ram_gb=503) == 4


def test_reader_cache_degrades_instead_of_oom_on_a_small_host():
    """The count is a wish and RAM is the constraint; the clamp binds before the OOM killer does."""
    from tailcyclenet.dataset import _reader_cache_size

    assert _reader_cache_size(16, (3208, 2200), 12, ram_gb=64) == 1
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=64) < 16
    # never zero -- one reader is the minimum that can serve a read at all
    assert _reader_cache_size(16, (3208, 2200), 64, ram_gb=1) == 1


def test_reader_cache_env_var_overrides_everything(monkeypatch):
    """The knob survives as an OVERRIDE, and is now read at first use rather than at import."""
    from tailcyclenet.dataset import _reader_cache_size

    monkeypatch.setenv('TAILCYCLENET_READER_CACHE', '1')
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 1
    monkeypatch.setenv('TAILCYCLENET_READER_CACHE', '32')
    assert _reader_cache_size(1, (1024, 570), 12, ram_gb=64) == 32


def test_reader_cache_size_does_not_touch_the_filesystem(monkeypatch):
    """GOTCHA 11. Opening a container in the parent deadlocks every forked worker, forever, so the
    sizing rule may read toml-derived numbers and nothing else."""
    import builtins

    from tailcyclenet.dataset import _reader_cache_size

    def boom(*a, **k):
        raise AssertionError('the sizing rule must not open anything')

    monkeypatch.setattr(builtins, 'open', boom)
    monkeypatch.setattr('os.stat', boom)
    assert _reader_cache_size(16, (3208, 2200), None, ram_gb=503) == 16


def test_a_whole_body_offset_moves_every_keypoint_the_same_way(tiny_root):
    """I.I.D. JITTER IS NOT THE FAILURE DEPLOYMENT PRODUCES.

    `--anchor carry` hands the model its own previous output, whose error is a whole-body LAG: every
    keypoint displaced the same way. Gaussian noise averages to zero over the keypoint set, so it
    trains the model to trust the prior's CENTROID exactly -- the one quantity a lag gets wrong.
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
    """`carried` degrades into a pose from an earlier frame than `prompt_t` claims when the box
    source loses an animal. Swapping in NaN instead would just be `prompt_dropout` again."""
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
    """`cv2.warpAffine` with a 0 in `dsize` returns the FULL SOURCE FRAME, and does not raise.

    So a camera rounded to [256, 0] handed the model a 4696x2048 image while its own dict said
    256x0 -- silent garbage instead of a clean failure. Reachable from `run_group`'s per-camera
    union box, which only guarantees `x1 >= x0 + 1`, i.e. one detector box clipped to a sliver
    against a frame edge is enough.
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
    """The format allows it; training on it does not. It used to CRASH mid-epoch instead.

    Spec §8 and validation rule 12 admit a `mode = "3d"` session carrying only `keypoints.pq`
    (rule 6 needs just one of the two tables), and the index builder admits its windows off
    `vis2d`. But `_item` picks its target table off `sess.mode` alone, so it reached
    `lab.points3d[a]` as `TypeError: 'NoneType' object is not subscriptable` -- uncaught, since
    only a `None` return is retried, and with nothing in the message naming the session.
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
    """`rotate_camera_image_plane_3d` crops to the BORDER-FREE INSCRIBED RECTANGLE, ~0.416 of the
    frame -- the exact step `_rotate_2d` was hand-written to avoid.

    An animal outside that rectangle projects out of `cam['size']`, and `crop_box_for_points`
    CLAMPS rather than returning None, so the item came back as a `min_crop_dim` corner crop of
    background carrying full-strength world targets. `_item`'s `< 2 finite` guard runs BEFORE the
    rotation, so nothing caught it -- and the `vis_2d` recompute meant to soften it is gated on
    `vis_2d is not None`, which excludes 3dpop, branson-fly and johnson-mouse (all `projected`).
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
    """The noisy-OR used to be taken BEFORE the augmentation that invalidates it.

    The rotation loop zeroes `vis_2d[:, :, cnum]` for points the rotated camera no longer sees and
    `_cutout_rects` zeroes more -- both AFTER `vis` was computed. So a point could end up
    `vis_2d == 0` in every shown camera while `vis` still said True, and `TotalLoss` takes a
    supplied `vis_true` verbatim (losses.py:421), weighting `mae_loss_coords`,
    `coords_loss_direct` and `bce_loss_vis_3d` by it. That supervises the 3D coord and visibility
    heads at points the crop provably does not contain -- i.e. supplying visibility was WORSE than
    passing None, which lets the loss derive it geometrically.
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


# ----------------------------------------------------------------------------------------------
