"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tailcyclenet.data.crop import crop_box_for_points, crop_boxes_from_extents
from tailcyclenet.data.pixels import read_frames
from tailcyclenet.detector.augment import (
    normalize_input,
    random_affine,
)
from tailcyclenet.detector.dataset import MIN_EXTENT_PX, BoxDataset
from tailcyclenet.detector.letterbox import letterbox, unletterbox_boxes


def test_targets_are_the_crop_rule(tiny_root):
    """The regression target must BE crop_box_for_points, not something similar to it."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2)
    item = ds[0]
    x, boxes = item['x'], item['boxes']
    assert x.shape == (3, 128, 128)
    sess, gid, f, ci = ds.index[0]
    lab = sess.labels(gid)
    cam = sess.rig.tracktail()[ci]
    for s in range(boxes.shape[0]):
        want = crop_box_for_points(
            torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32),
            cam['size'], ds.min_crop_dim, 0.0, ds.crop_pad_frac)
        got = boxes[s]
        if want is None:
            assert torch.isnan(got).all()
            continue
        # the stored box is the crop rule's box, letterboxed; undo that and it must come back
        img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
        _, scale, pad = letterbox(img, ds.input_wh)
        back = unletterbox_boxes(got[None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_augmented_targets_are_still_the_crop_rule(tiny_root):
    """Augmentation warps the GEOMETRY and re-derives the box; scaling the box would scale the
    pad but not the `min_crop_dim` floor -- a box the crop rule can never emit."""
    from tailcyclenet.detector.augment import random_affine

    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                       max_frames_per_group=2)
    sess, gid, f, ci = ds.index[0]
    cam = sess.rig.tracktail()[ci]
    warp = random_affine(cam['size'], np.random.default_rng([ds.seed, 0]))
    got = ds.boxes_for(0, warp)

    _, scale, pad = letterbox(np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8),
                              ds.input_wh)
    lab = sess.labels(gid)
    for s in range(got.shape[0]):
        p = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
        p = p @ torch.as_tensor(warp[:, :2]).T + torch.as_tensor(warp[:, 2])
        w, h = float(cam['size'][0]), float(cam['size'][1])
        outside = (p[:, 0] < 0) | (p[:, 0] > w) | (p[:, 1] < 0) | (p[:, 1] > h)
        want = crop_box_for_points(torch.where(outside[:, None], torch.nan, p),
                                   cam['size'], ds.min_crop_dim, 0.0, ds.crop_pad_frac)
        if want is None:
            assert torch.isnan(got[s]).all()
            continue
        back = unletterbox_boxes(got[s][None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)

    # And with no warp it is byte-identical to the unaugmented loader -- `augment` must be a key
    # you turn on, not a thing that leaks into a run that did not ask for it.
    torch.testing.assert_close(ds.boxes_for(0), plain.boxes_for(0))


def test_augment_decodes_real_pixels_through_getitem(tiny_root):
    """`augment=True` must survive a real `__getitem__` (the only caller of `photometric`): a
    cleanup once left a `NameError` on every augmented item."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    for i in range(min(2, len(ds))):
        item = ds[i]
        x = item['x']
        _boxes = item['boxes']
        assert x.shape == (3, 48, 64)
        assert torch.isfinite(x).all()
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, \
            'the photometric gain must still land in the normalised [0, 1] range'


def test_a_tiled_item_turns_about_its_own_tile(tiny_root):
    """A tile is cut AFTER the warp, so the warp turns about the tile's centre (about the frame
    centre only 0.075 of rat-city tiles kept their animal)."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(32, 32), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, rotate_deg=180.0,
                    tile_wh=(32, 32), tile_scale=1.0)
    i = next(j for j in range(len(ds)) if ds.origins[j] is not None)
    ox, oy = ds.origins[i]
    tw, th = ds._tile_extent()
    assert ds._warp_centre(i) == (ox + tw / 2, oy + th / 2)

    sess, _, _, ci = ds.index[i]
    size = tuple(sess.rig.size(sess.cam_names[ci]))
    M = random_affine(size, np.random.default_rng([ds.seed, i]), hflip=0.0, rotate_deg=180.0,
                      translate=0.0, scale=(1.0, 1.0), centre=ds._warp_centre(i))
    c = np.array([ox + tw / 2, oy + th / 2, 1.0])
    np.testing.assert_allclose(M @ c, c[:2], atol=1e-3)


def test_chunk_is_one_containers_worth_of_index(tiny_root):
    """`ChunkShuffle`'s block must be one video: a hardcoded 512 spanned 13 calms21 videos and
    ran the reader cache at a 16% hit rate."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), max_frames_per_group=2)
    n_src = len({(s.session_id, g, c) for s, g, _, c in ds.index})
    assert ds.chunk == len(ds.index) // n_src
    assert ds.chunk < len(ds.index) or n_src == 1


def test_tiled_targets_are_still_the_crop_rule(tiny_root):
    """The tiled box is RE-DERIVED in source px, never scaled by tile_scale."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2,
                    tile_wh=(32, 32), tile_scale=1.0)
    from tailcyclenet.detector.letterbox import tile_transform
    for i in range(min(3, len(ds))):
        sess, gid, f, ci = ds.index[i]
        boxes = ds.boxes_for(i)
        lab = sess.labels(gid)
        cam = sess.rig.tracktail()[ci]
        ox, oy = ds.origins[i]
        tw, th = ds._tile_extent()
        for s in range(boxes.shape[0]):
            pts = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
            keep = ((pts[:, 0] >= ox) & (pts[:, 0] <= ox + tw) &
                    (pts[:, 1] >= oy) & (pts[:, 1] <= oy + th))
            want = crop_box_for_points(torch.where(keep[:, None], pts, torch.nan),
                                       cam['size'], ds.min_crop_dim, 0.0, ds.crop_pad_frac)
            if want is None:
                assert torch.isnan(boxes[s]).all()
                continue
            scale, pad = tile_transform((ox, oy), ds.tile_scale)
            back = unletterbox_boxes(boxes[s][None], scale, pad)[0]
            torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_an_off_frame_tile_drops_its_points_and_is_grey(tiny_root):
    """Out-of-tile behaves like out-of-frame (no box), and off-frame pixels are grey, not wrapped."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    tile_wh=(16, 16), tile_scale=1.0)
    ds.origins[0] = (1000.0, 1000.0)        # the fixture's points are in [5, 43]
    assert torch.isnan(ds.boxes_for(0)).all()
    ds.origins[0] = (-16.0, -16.0)          # wholly outside, up and left
    x = ds[0]['x']
    assert x.shape == (3, 16, 16)
    torch.testing.assert_close(x, torch.full_like(x, 114 / 255.0))


def test_strong_augment_never_drops_a_box(tiny_root):
    """Appearance ops and cutout never move a box -- only mosaic-lite may ADD one, fully interior."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    for i in range(min(2, len(ds))):
        base = ds.boxes_for(i)
        for _ in range(4):
            assert ds[i]['boxes'].shape[0] >= base.shape[0]


def test_mosaic_paste_is_fully_interior_and_reencodes_the_crop_rule(tiny_root):
    """The appended box is `src_box + translation`, entirely inside `input_wh`. The item's own
    boxes are dropped so the tiny canvas has empty space (the overlap test is
    `test_mosaic_paste_never_covers_the_items_own_boxes`)."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    rng = np.random.default_rng(1)
    for i in range(min(3, len(ds))):
        base = ds.boxes_for(i)[:0]
        img, _ = ds._load_letterbox(i)
        boxes, img2 = ds._mosaic_paste(i, base.clone(), img.copy(), rng)
        if boxes.shape[0] == base.shape[0]:
            continue    # no finite source box was found in a few tries; not this fixture's job
        new = boxes[base.shape[0]:]
        assert (new[:, 0] >= 0).all() and (new[:, 1] >= 0).all()
        assert (new[:, 2] <= ds.input_wh[0]).all() and (new[:, 3] <= ds.input_wh[1]).all()
        return
    pytest.skip('no fixture item produced a finite mosaic source box in the tries allotted')


def test_extent_target_is_tight_and_gives_back_the_crop_rule(tiny_root):
    """The `extent` target is the points' own min/max (letterboxed), and the crop rule applied to
    it after unletterboxing -- what `detect_raw` does -- is the crop the `crop` target trains on."""
    kw = dict(input_wh=(128, 128), min_crop_dim=8, max_frames_per_group=2)
    ext = BoxDataset(tiny_root / 'ratlike', 'train', box_target='extent', **kw)
    crop = BoxDataset(tiny_root / 'ratlike', 'train', **kw)
    checked = 0
    for i in range(len(ext)):
        sess, gid, f, ci = ext.index[i]
        cam = sess.rig.tracktail()[ci]
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
                rule = crop_boxes_from_extents(back[None], cam['size'], ext.min_crop_dim, 0.0,
                                               ext.crop_pad_frac)[0]
                want = crop_box_for_points(pts, cam['size'], ext.min_crop_dim, 0.0,
                                           ext.crop_pad_frac).float()
                torch.testing.assert_close(rule, want, atol=1.0, rtol=0)
                legacy = unletterbox_boxes(crop.boxes_for(i)[s][None], scale, pad)[0]
                torch.testing.assert_close(legacy, want, atol=0.51, rtol=0)
                checked += 1
    assert checked, 'no box was big enough to escape the floor -- the test checked nothing'


def test_grayscale_prob_one_makes_every_train_view_monochrome(tiny_root):
    """`grayscale_prob = 1` must reach `__getitem__`'s pixels, after every colour op."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), max_frames_per_group=2,
                    augment=True, strong=True, grayscale_prob=1.0)
    for i in range(min(4, len(ds))):
        x = ds[i]['x']
        assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])


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


def test_mosaic_paste_never_covers_the_items_own_boxes():
    """detector B4: the paste lands in EMPTY space, never over a real animal's box."""
    from tailcyclenet.detector.dataset import _overlaps_any

    class Stub:
        input_wh, box_target = (100, 100), 'crop'
        _mosaic_paste = BoxDataset._mosaic_paste

        def __len__(self):
            return 2

        def _load_letterbox(self, j):
            return np.full((100, 100, 3), 255, np.uint8), torch.tensor([[10., 10., 50., 50.]])

    own = torch.tensor([[0., 0., 55., 100.]])
    rng = np.random.default_rng(0)
    pasted = 0
    for _ in range(200):
        out, img = Stub()._mosaic_paste(0, own.clone(), np.zeros((100, 100, 3), np.uint8), rng)
        if len(out) > 1:
            pasted += 1
            assert not _overlaps_any(tuple(out[1].tolist()), own)
            assert not img[:, :55].any(), 'the paste covered the real animal'
    assert pasted > 100


def test_training_seed_fixes_the_augmentation_stream(tmp_path):
    """detector B5: `[training].seed` seeds augmentation; another seed draws another stream."""
    from tests.helpers import _session_3d

    root = tmp_path / 'ds'
    _session_3d(root / 'train' / 's')

    def first_views(seed):
        ds = BoxDataset(root, 'train', input_wh=(64, 64), augment=True, strong=True, seed=seed,
                        max_frames_per_group=0)
        return [ds[i % len(ds)]['x'] for i in range(4)]

    a, b, c = first_views(5), first_views(5), first_views(6)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))
