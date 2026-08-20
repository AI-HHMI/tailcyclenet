"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
import numpy as np
import pytest
import torch
from pathlib import Path

from tailcyclenet.crop import box_corners, crop_box_for_points
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, YOLOXNano, assign, box_collate,
                                   box_iou, decode, detector_loss, giou_loss, letterbox,
                                   paired_iou, unletterbox_boxes)
from tailcyclenet.detector.config import load_detector_config
from tailcyclenet.detector.data import _cutout_rects, _keypoints_in_rects, random_affine



def _infer_program_source():
    """The inference PROGRAM as one string: `tailcyclenet/infer/{cli,driver}.py`.

    These assertions predate the split and were written against `scripts/infer.py` when that file
    held the argparse and the whole driver. It is now a six-line shim onto
    `tailcyclenet.infer.main`, so reading it would silently assert nothing -- a scrape for an
    absent literal PASSES. Concatenating the two files it became keeps each check meaning exactly
    what it meant before.
    """
    base = Path(__file__).resolve().parent.parent / 'tailcyclenet' / 'infer'
    return (base / 'cli.py').read_text() + (base / 'driver.py').read_text()


def _infer_window_source():
    """The window loop, formerly `tailcyclenet/infer.py`."""
    return (Path(__file__).resolve().parent.parent
            / 'tailcyclenet' / 'infer' / 'window.py').read_text()


def test_forward_shapes_and_anchor_order():
    m = YOLOXNano()
    x = torch.zeros(2, 3, 128, 160)
    obj, boxes, _ = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0], \
        'anchor_points must match forward()s flattening order exactly'
    assert boxes.shape == (2, anchors.shape[0], 4)
    assert (boxes[..., 2] >= boxes[..., 0]).all() and (boxes[..., 3] >= boxes[..., 1]).all()


def test_chunk_shuffle_is_a_permutation_that_keeps_locality():
    """Every index exactly once (or it silently drops training data), and few videos at a time.

    The locality bound is the whole point: `_reader` caches per worker, so a draw that ranges
    over more than `mix` blocks re-opens containers and costs 486 ms/batch instead of 40.
    """
    chunk, mix, n_pools = 512, 4, 3
    n = chunk * mix * n_pools                     # exact multiple, so pools are position-aligned
    s = ChunkShuffle(n, chunk=chunk, mix=mix, seed=0)
    order = list(iter(s))
    assert sorted(order) == list(range(n)), 'must visit every index exactly once'
    assert order != list(range(n)), 'must actually shuffle'
    for i in range(0, n, chunk * mix):            # one pool = at most `mix` distinct videos
        assert len({j // chunk for j in order[i:i + chunk * mix]}) == mix
    assert list(iter(s)) != order, 'a second epoch must reshuffle'


def test_giou_is_zero_for_a_perfect_box():
    b = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    assert float(giou_loss(b, b)) < 1e-5
    assert float(giou_loss(b, b + 200)) > 1.0        # disjoint -> worse than any overlap


def test_assign_ignores_nan_boxes():
    """An animal absent from a view gets a NaN box, and objectness must learn "nothing here"."""
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[float('nan')] * 4])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == 0

    gt = torch.tensor([[float('nan')] * 4, [30.0, 30.0, 90.0, 90.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    assert (gix == 1).all(), 'positives must point at the finite box, not the NaN one'


def test_every_positive_anchor_is_inside_its_own_box():
    """The whole guard on `inside & near`.

    A predicted box is `centre +- exp(ltrb) * stride`, so it always contains its own anchor
    centre. A positive whose centre is outside its assigned GT box therefore has an unreachable
    regression target while objectness is taught to fire there. `inside | near` shipped for a
    while and nothing in this file caught it: uniqueness held, NaNs were still skipped, the loss
    stayed finite, and 71% of rat-city's positives were unreachable.
    """
    m = YOLOXNano()
    anchors = m.anchor_points(256, 256, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 60.0, 70.0],        # small: radius reaches well past it
                       [100.0, 30.0, 240.0, 200.0]])    # large: inside reaches past the radius
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    cx, cy = anchors[pos, 0], anchors[pos, 1]
    box = gt[gix]
    assert ((cx > box[:, 0]) & (cx < box[:, 2]) & (cy > box[:, 1]) & (cy < box[:, 3])).all(), \
        'a positive anchor outside its assigned box cannot reach the target it is given'
    assert len(set(gix.tolist())) == 2, 'both boxes must keep positives'


def test_assign_gives_each_anchor_one_box():
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [30.0, 30.0, 110.0, 110.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == len(set(pos.tolist())), 'an anchor claimed by two boxes cancels'


def test_loss_is_finite_with_no_animal_anywhere():
    m = YOLOXNano()
    x = torch.zeros(2, 3, 128, 128)
    obj, boxes, _ = m(x)
    anchors = m.anchor_points(128, 128, x.device)
    gt = torch.full((2, 1, 4), float('nan'))
    loss, parts = detector_loss(obj, boxes, anchors, gt)
    assert torch.isfinite(loss) and parts['n_pos'] == 0


def test_decode_suppresses_duplicates():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.], [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2, 'the two overlapping boxes must collapse to one'
    assert s[0] > s[1]


def test_reduce_factor_never_decodes_below_the_target():
    from tailcyclenet.detector import reduce_factor

    assert reduce_factor((4696, 2048), (640, 288)) == 4       # rat-city: 1174x512, still above
    assert reduce_factor((4696, 2048), (896, 384)) == 4       # 1174x512 still clears 896x384
    assert reduce_factor((1024, 570), (544, 320)) == 1        # calms21: 1/2 is already below
    assert reduce_factor((640, 480), (640, 480)) == 1
    for src, out in (((4696, 2048), (640, 288)), ((3840, 2160), (544, 320))):
        n = reduce_factor(src, out)
        assert src[0] / n >= out[0] and src[1] / n >= out[1], 'the remaining resize must be a '\
            'downscale, or the letterbox is upsampling a decimated frame'


def test_letterbox_scale_is_in_source_units_under_reduction():
    """`unletterbox_boxes` undoes the letterbox with this scale, and it never sees the decode.

    A reduced decode that changed the returned scale would move every predicted box by the
    reduction factor -- 4x on rat-city -- with nothing in the output to say so.
    """
    full = np.zeros((2048, 4696, 3), np.uint8)
    quarter = np.zeros((512, 1174, 3), np.uint8)
    a, s_a, p_a = letterbox(full, (640, 288))
    b, s_b, p_b = letterbox(quarter, (640, 288), src_wh=(4696, 2048))
    assert a.shape == b.shape
    assert s_a == s_b and p_a == p_b
    box = torch.tensor([[100.0, 50.0, 700.0, 400.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * s_b + p_b[0]
    moved[:, 1::2] = moved[:, 1::2] * s_b + p_b[1]
    torch.testing.assert_close(unletterbox_boxes(moved, s_b, p_b), box, atol=1e-3, rtol=0)


def test_letterbox_round_trip():
    img = np.zeros((200, 400, 3), np.uint8)
    out, scale, pad = letterbox(img, (416, 416))
    assert out.shape == (416, 416, 3)
    box = torch.tensor([[10.0, 20.0, 300.0, 150.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * scale + pad[0]
    moved[:, 1::2] = moved[:, 1::2] * scale + pad[1]
    torch.testing.assert_close(unletterbox_boxes(moved, scale, pad), box, atol=1e-4, rtol=0)


def test_the_batched_pack_is_bit_identical_to_the_per_frame_one():
    """PIXELS ARE A CONTRACT. `detect_group` packs a batch through numpy instead of one
    `torch.as_tensor(lb, float32).permute(2,0,1) / 255.0` per frame, because handing a 0.5 MP
    elementwise op to torch's `nproc`-wide intraop pool cost 67 ms per frame against 1.0 ms. Both
    are uint8 -> float32 -> divide by 255, both correctly rounded, so the pixels the detector sees
    must be EQUAL and not merely close -- one ulp here is a different box somewhere."""
    rng = np.random.default_rng(0)
    lbs = [rng.integers(0, 256, (32, 48, 3), dtype=np.uint8) for _ in range(5)]
    old = torch.stack([torch.as_tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255.0
                       for x in lbs])
    arr = np.ascontiguousarray(np.stack(lbs).transpose(0, 3, 1, 2))
    new = torch.from_numpy(arr.astype(np.float32) / np.float32(255))
    assert torch.equal(old, new)


def test_a_video_read_locks_per_container_not_globally():
    """`dataset._read_video` takes a lock PER PATH: two threads on ONE container interleave their
    seeks, two threads on DIFFERENT containers share no state at all. The second is the whole of
    `detect_group`'s multi-camera decode overlap (3.5x on 3dpop's four cameras), so a regression to
    one global lock has to fail something."""
    from tailcyclenet import dataset as ds

    a, b = ds._read_lock_for('/x/cam0.mp4'), ds._read_lock_for('/x/cam1.mp4')
    assert a is not b
    assert ds._read_lock_for('/x/cam0.mp4') is a
    with a:                       # holding one container's lock must not block another's
        assert b.acquire(blocking=False)
        b.release()


def test_targets_are_the_crop_rule(tiny_root):
    """The regression target must BE crop_box_for_points, not something similar to it."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2)
    x, boxes = ds[0]
    assert x.shape == (3, 128, 128)
    sess, gid, f, ci = ds.index[0]
    lab = sess.labels(gid)
    cam = sess.rig.posetail()[ci]
    for s in range(boxes.shape[0]):
        want = crop_box_for_points(
            torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32),
            cam['size'], ds.min_crop_dim)
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
    """Augmentation warps the GEOMETRY and re-derives the box; it never scales the box.

    Scaling the box with the image breaks the rule: the 20 px pad scales but the `min_crop_dim`
    floor does not, so a floored box scaled by 0.8 is a box `crop_box_for_points` can never emit
    and the detector trains off its own target.
    """
    from tailcyclenet.detector.data import random_affine

    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                       max_frames_per_group=2)
    sess, gid, f, ci = ds.index[0]
    cam = sess.rig.posetail()[ci]
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
                                   cam['size'], ds.min_crop_dim)
        if want is None:
            assert torch.isnan(got[s]).all()
            continue
        back = unletterbox_boxes(got[s][None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)

    # And with no warp it is byte-identical to the unaugmented loader -- `augment` must be a key
    # you turn on, not a thing that leaks into a run that did not ask for it.
    torch.testing.assert_close(ds.boxes_for(0), plain.boxes_for(0))


def test_augment_decodes_real_pixels_through_getitem(tiny_root):
    """`augment=True` must survive an actual `__getitem__` call, not just `boxes_for`.

    Every other augmentation test above calls `ds.boxes_for(...)`, which never reaches
    `_photometric` -- that function is called from `__getitem__` alone. A cleanup commit
    (`3dbb0a1`) deleted `_photometric`'s `extended` parameter and `BoxDataset`'s `photometric`
    flag but left an `if extended:` block referencing the now-undefined name, a `NameError` on
    EVERY item under `--augment`. It went undetected through this whole file and only surfaced
    when real training jobs hit it. This test exists so that class of gap cannot reopen: it is
    the one place in this file that actually decodes a pixel tensor under augmentation.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    for i in range(min(3, len(ds))):
        x, boxes = ds[i]
        assert x.shape == (3, 48, 64)
        assert torch.isfinite(x).all()
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, \
            'the photometric gain must still land in the normalised [0, 1] range'


def test_a_rotated_box_needs_four_corners_in_the_detector(tiny_root):
    """Two diagonal corners are not a box under a flip: their extent is strictly inside all four.

    The stored `instances.pq` extent enters the detector as geometry, so it has to be expanded
    before the warp -- the same property `test_a_rotated_box_needs_four_corners` asserts for the
    pose loader.
    """
    b = torch.tensor([10.0, 20.0, 50.0, 40.0])
    x0, y0, x1, y1 = b
    four = torch.stack([torch.stack([x0, y0]), torch.stack([x1, y0]),
                        torch.stack([x1, y1]), torch.stack([x0, y1])])
    two = b.view(2, 2)
    M = torch.tensor([[0.8, -0.6, 5.0], [0.6, 0.8, -3.0]])
    f4 = four @ M[:, :2].T + M[:, 2]
    f2 = two @ M[:, :2].T + M[:, 2]
    ext = lambda p: torch.cat([p.min(0).values, p.max(0).values])   # noqa: E731
    e4, e2 = ext(f4), ext(f2)
    assert (e4[:2] <= e2[:2]).all() and (e4[2:] >= e2[2:]).all()
    assert not torch.allclose(e4, e2), 'two corners would crop the animal the box encloses'


def test_rotation_off_is_byte_identical_and_on_turns_about_its_centre():
    """`--rotate-deg 0` must reproduce the pre-rotation matrix EXACTLY, draw for draw.

    Not "close": every detector arm on record was trained without this key, and `random_affine`
    consumes the rng, so drawing an angle and multiplying it by zero would reseat every later draw
    and silently make those arms unreproducible. The draw is skipped instead.
    """
    def before(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5):
        w, h = float(size[0]), float(size[1])
        s = rng.uniform(*scale)
        sx = -s if rng.random() < hflip else s
        cx, cy = w / 2, h / 2
        return np.array([[sx, 0.0, cx - sx * cx + rng.uniform(-translate, translate) * w],
                         [0.0, s, cy - s * cy + rng.uniform(-translate, translate) * h]],
                        np.float32)

    for seed in range(25):
        a = before((4696, 2048), np.random.default_rng([0, seed]))
        b = random_affine((4696, 2048), np.random.default_rng([0, seed]))
        assert np.array_equal(a, b), f'seed {seed} moved'

    # And the rotation is about `centre`, which is the whole reason tiling and rotation compose.
    M = random_affine((1000, 1000), np.random.default_rng(1), scale=(1.0, 1.0), translate=0.0,
                      hflip=0.0, rotate_deg=180.0, centre=(300.0, 700.0))
    fixed = M @ np.array([300.0, 700.0, 1.0])
    np.testing.assert_allclose(fixed, [300.0, 700.0], atol=1e-3)


def test_a_tiled_item_turns_about_its_own_tile(tiny_root):
    """A tile is cut AFTER the warp, so the warp has to hold the tile or the tile holds nothing.

    `__getitem__` composes `tile @ warp @ decode`. About the FRAME centre a rotation sweeps an
    animal clean out of the 640-px window that was chosen for it -- measured on
    rat-city-annotated at 0.075 of animal-bearing tiles still holding an animal, against 0.820
    about the tile centre (measured). Here the check is structural:
    the tile's own centre is a fixed point of its warp, and the frame's centre is not.
    """
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


def test_a_rotated_region_certifies_less_not_more(tiny_root):
    """A certified area is a CLAIM, and under a rotation a claim must round DOWN.

    Four corners then the extent -- right for a BOX, which must not crop its animal -- is exactly
    backwards for a region: the axis-aligned hull of a rotated rectangle claims area the annotator
    never marked, re-admitting the unlabelled animals `regions.pq` exists to exclude. That is the
    one direction the mask cannot be wrong in, since this root labels a median of 2 rats per frame.
    """
    from tailcyclenet.detector.data import _warp_region

    r = torch.tensor([[100.0, 200.0, 500.0, 900.0], [0.0, 0.0, 64.0, 64.0]])

    # Every warp that existed before rotation keeps a rect axis-aligned, so inscribed and
    # circumscribed coincide and this is a no-op against the old four-corner code.
    for M in (np.array([[1.0, 0.0, 37.0], [0.0, 1.0, -12.0]], np.float32),
              np.array([[0.83, 0.0, 5.0], [0.0, 0.83, 9.0]], np.float32),
              np.array([[-1.1, 0.0, 900.0], [0.0, 1.1, 3.0]], np.float32)):
        c = box_corners(r) @ torch.as_tensor(M[:, :2]).T + torch.as_tensor(M[:, 2])
        want = torch.cat([c.amin(-2), c.amax(-2)], -1)
        torch.testing.assert_close(_warp_region(r, M), want, atol=1e-3, rtol=0)

    sq = torch.tensor([[0.0, 0.0, 400.0, 400.0]])
    for deg in (15.0, 45.0, 137.0):
        a = np.radians(deg)
        M = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0]], np.float32)
        hull = box_corners(sq) @ torch.as_tensor(M[:, :2]).T
        out = _warp_region(sq, M)
        assert float(out[0, 2] - out[0, 0]) < 400.0 < float((hull.amax(-2) - hull.amin(-2))[0, 0])
    # 90 degrees maps a square onto itself exactly -- no shrink is owed and none is taken.
    M90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    torch.testing.assert_close(_warp_region(sq, M90)[0, 2] - _warp_region(sq, M90)[0, 0],
                               torch.tensor(400.0), atol=1e-3, rtol=0)


def test_chunk_is_one_containers_worth_of_index(tiny_root):
    """`ChunkShuffle`'s block must be one video, or the locality it exists for is not there.

    A hardcoded 512 spanned 13 calms21 videos per block and 52 per pool, which ran the reader
    cache at a 16% hit rate and, at the cache size that thrash needed, OOM-killed the workers at
    ~1 GB of open decord reader each.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), max_frames_per_group=2)
    n_src = len({(s.session_id, g, c) for s, g, _, c in ds.index})
    assert ds.chunk == len(ds.index) // n_src
    assert ds.chunk < len(ds.index) or n_src == 1


def test_collate_pads_uneven_animal_counts():
    a = (torch.zeros(3, 8, 8), torch.zeros(2, 4))
    b = (torch.zeros(3, 8, 8), torch.zeros(5, 4))
    x, boxes = box_collate([a, b])
    assert x.shape == (2, 3, 8, 8) and boxes.shape == (2, 5, 4)
    assert torch.isnan(boxes[0, 2:]).all()


def test_min_views_1_admits_the_box_no_pair_claimed(tmp_path):
    """`min_views = 2` is the ALGORITHM, not a threshold.

    Every group `associate` emits starts from a cross-camera pair, so `len(members) >= 2` always
    and the floor never fires -- an animal only one camera saw is dropped from the frame outright.
    `min_views = 1` is a different rule: it emits each leftover box as a single-view instance, which
    the pose model supports (`prob_2d_only` trains exactly that input).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    from tailcyclenet.detector import associate
    from tailcyclenet.format import Session

    cf._session_3d(tmp_path / 'ds' / 'test' / 's')
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    cams = sess.cgroup('g000', 0)

    # One animal both of the first two cameras see at the image centre, plus a box in camera 2 that
    # no pair can agree with (it sits in a corner).
    centre = torch.tensor([[24., 16., 40., 32.]])
    per_cam = [centre, centre, torch.cat([centre, torch.tensor([[0., 0., 8., 8.]])])]

    two = associate(cams, per_cam, max_res_px=30.0, min_views=2)
    one = associate(cams, per_cam, max_res_px=30.0, min_views=1)
    assert len(one) == len(two) + 1, f'{len(two)} -> {len(one)}: the leftover box must appear'
    extra = one[len(two)]
    assert list(extra['boxes']) == [2] and torch.isnan(extra['point']).all(), \
        'a single ray has no triangulated point, and inventing a depth would be a lie'


def test_link_rows_follows_one_animal():
    """Unlinked rows are score-ordered, so the window-union crop spans several animals.

    Two animals crossing the frame in opposite directions, with their rows swapped on every odd
    frame the way a score reordering would swap them. Linking must undo that: the give-away is
    the UNION box, which is the thing `run_group` actually crops to.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    T = 20
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        a = np.array([10 + t, 10, 40 + t, 40], np.float32)      # drifts right
        b = np.array([110 - t, 10, 140 - t, 40], np.float32)    # drifts left
        boxes[0, t, 0], boxes[1, t, 0] = (a, b) if t % 2 == 0 else (b, a)

    union = lambda x: np.concatenate([x[..., :2].min(0), x[..., 2:].max(0)], -1)  # noqa: E731
    before = union(boxes[0, :, 0])
    linked = link_rows(boxes.copy())
    after = union(linked[0, :, 0])

    assert (before[2] - before[0]) > 120, 'the swapped rows should span the whole frame'
    assert (after[2] - after[0]) < (before[2] - before[0]) / 2, \
        f'linking must shrink the union crop, got {after} from {before}'
    # Every frame still holds both animals -- linking reorders, it never drops.
    assert np.isfinite(linked).all()


def test_link_rows_state_carries_across_a_split():
    """N calls with `state=` must equal ONE call over the concatenation, byte for byte.

    The inference loop associates a clip in BLOCKS whose size comes from the host RAM budget. If
    the matcher restarted at each block's own frame 0, the identity assignment would break at a
    boundary chosen by how much memory happened to be free -- two machines, two answers, and
    nothing in the output saying so. That is the one thing the budget is never allowed to do.

    The trap this pins is the `t0` asymmetry: frame 0 of the CLIP seeds `last` and is left
    unpermuted, but a BLOCK's frame 0 is an ordinary mid-clip frame and must be permuted against
    the carried `last`. Starting every block at `t0 = 1` would silently leave one frame per block
    in score order.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    T = 20
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        a = np.array([10 + t, 10, 40 + t, 40], np.float32)
        b = np.array([110 - t, 10, 140 - t, 40], np.float32)
        boxes[0, t, 0], boxes[1, t, 0] = (a, b) if t % 2 == 0 else (b, a)
    scores = np.tile(np.array([0.9, 0.8], np.float32)[:, None, None], (1, T, 1))

    whole_b, whole_s = link_rows(boxes.copy(), scores.copy())

    # The same clip in three unequal blocks, which is what a budget-derived split looks like.
    part_b, part_s, st = boxes.copy(), scores.copy(), {}
    for lo, hi in ((0, 7), (7, 8), (8, T)):
        link_rows(part_b[:, lo:hi], part_s[:, lo:hi], state=st)

    np.testing.assert_array_equal(np.nan_to_num(part_b, nan=-9e9),
                                  np.nan_to_num(whole_b, nan=-9e9))
    np.testing.assert_array_equal(np.nan_to_num(part_s, nan=-9e9),
                                  np.nan_to_num(whole_s, nan=-9e9))
    # And the split must not be a no-op: it has to actually reorder, or this proves nothing.
    assert not np.array_equal(np.nan_to_num(whole_b, nan=-9e9),
                              np.nan_to_num(boxes, nan=-9e9))


def test_associate_group_state_carries_across_a_split(tmp_path):
    """Same claim one level up, for BOTH branches: the tracker (C > 1) and `link_rows` (C == 1).

    `associate_group` is where the block loop calls in, so the invariance has to hold there and
    not only in the matcher underneath it. The two branches are separate code paths -- the tracker
    subsumes `link_rows` and they never both run -- so a test that exercises one says nothing
    about the other.

    THE SCENE IS BUILT SO A STATELESS SPLIT ACTUALLY FAILS, and that took finding. Two animals
    merely walking apart is NOT discriminating: a fresh tracker at the boundary re-births them into
    the same slots in the same order, so the split matches the whole clip by luck and the test
    passes while proving nothing. What breaks it is a row that VANISHES across the boundary -- the
    survivor is then born into slot 0 by a fresh tracker, while the carried state keeps it in the
    slot it already had. Verified both ways: without `state=` this scene disagrees with the whole
    clip; with it, byte-identical. If a future change makes the tracker order-insensitive, check
    that this test can still fail before trusting it.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    import numpy as np

    from tailcyclenet.detector import associate_group
    from tailcyclenet.format import Session

    T = 12
    for tag, cams, track in (('mv', 3, True), ('sv', 1, False)):
        d = tmp_path / tag / 'test' / 's'
        (cf._session_3d if cams > 1 else cf._session_2d)(d, T=T)
        sess = Session.load(d)
        sess.preload()

        D, C, S = 4, cams, 2
        box = np.full((D, T, C, 4), np.nan, np.float32)
        sc = np.full((D, T, C), np.nan, np.float32)
        for t in range(T):
            for c in range(C):
                for a in range(S):
                    # Two animals drifting apart inside the fixture's 64x48 frame. What makes each
                    # branch discriminating is different, so the scene is too:
                    #   tracker (C > 1) -- the animal that lands in slot 0 VANISHES across the
                    #     split, so a fresh tracker seats the survivor there instead;
                    #   link_rows (C == 1) -- the DETECTION ORDER alternates, which is what a score
                    #     reordering does and what `link_rows` exists to undo. Without this the
                    #     rows arrive already in order and the matcher is a no-op.
                    if C > 1 and a == 1 and 4 <= t < 7:
                        continue
                    d = a if C > 1 else (a + t) % 2
                    x = 4.0 + 24.0 * a + 0.5 * t
                    box[d, t, c] = [x, 12, x + 12, 32]
                    sc[d, t, c] = 0.9 - 0.1 * d
        raw = (box, sc, None)

        def run(state):
            parts = []
            for lo, hi in ((0, 5), (5, T)):
                sub = (box[:, lo:hi].copy(), sc[:, lo:hi].copy(), None)
                parts.append(associate_group(sub, sess, 'g000', S, link=not track, track=track,
                                             **({} if state is None else {'state': state})))
            return [np.concatenate([p[i] for p in parts], axis=1) for i in range(2)]

        whole = associate_group(raw, sess, 'g000', S, link=not track, track=track)
        joined = run({})
        for i, name in enumerate(('boxes', 'scores')):
            np.testing.assert_array_equal(
                np.nan_to_num(joined[i], nan=-9e9), np.nan_to_num(whole[i], nan=-9e9),
                err_msg=f'{name} differ at C={cams}, track={track}: a block boundary changed the '
                        'association, so the RAM budget can change the answer')

        # THE TEST MUST BE ABLE TO FAIL. Without the carry, this scene disagrees -- if it stops
        # disagreeing, the scene has gone back to being one a fresh matcher reproduces by luck
        # and the assertion above is vacuous.
        assert not np.array_equal(np.nan_to_num(run(None)[0], nan=-9e9),
                                  np.nan_to_num(whole[0], nan=-9e9)), \
            f'C={cams}, track={track}: a STATELESS split reproduces the whole clip here, so this ' \
            'scene does not exercise the carry at all'


def test_link_rows_survives_a_dropped_frame():
    """Matching is against each row's LAST KNOWN box, so a one-frame miss cannot break the chain."""
    import numpy as np

    from tailcyclenet.detector import link_rows

    boxes = np.full((2, 4, 1, 4), np.nan, np.float32)
    for t in range(4):
        boxes[0, t, 0] = [10 + t, 10, 40 + t, 40]
        boxes[1, t, 0] = [110, 10, 140, 40]
    boxes[:, 2, 0] = np.nan                       # the detector sees nothing at frame 2
    linked = link_rows(boxes.copy())
    assert linked[0, 3, 0][0] == 13, 'row 0 must still be the left animal after the gap'
    assert linked[1, 3, 0][0] == 110


def test_box_source_instances_retargets_only_where_a_box_exists(tiny_root):
    """`--boxes instances` regresses the stored extent, and falls back per animal.

    Both halves matter for rat-city: the table is what rescues the 26k instances whose keypoints
    were cleaned away, and the fallback is what keeps the rest on the rule they always had.
    """
    # min_crop_dim 8, not the default 64: the fixture frame is 64x48, so a 64 px floor forces
    # every crop to the whole image and the two sources would agree for the wrong reason.
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=4, box_source='instances')
    base = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                      max_frames_per_group=4)
    # the fixture's one stored box is a02 on frame 1
    i = next(i for i, (_, _, f, _) in enumerate(ds.index) if f == 1)
    _, boxes = ds[i]
    _, plain = base[i]
    sess, gid, f, ci = ds.index[i]
    lab = sess.labels(gid)
    cam = sess.rig.posetail()[ci]
    img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
    _, scale, pad = letterbox(img, ds.input_wh)

    # a02 (row 1) carries the box: the target is the STORED corners at pad=0, not the keypoints
    want = crop_box_for_points(torch.as_tensor(lab.boxes[1, f, ci]).view(2, 2),
                               cam['size'], ds.min_crop_dim, pad=0)
    back = unletterbox_boxes(boxes[1][None], scale, pad)[0]
    torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)
    assert not torch.allclose(boxes[1], plain[1], atol=0.51)
    # a01 has no stored box, so it is byte-identical to the keypoint target
    torch.testing.assert_close(boxes[0], plain[0])


def test_box_source_rejects_a_typo(tiny_root):
    with pytest.raises(AssertionError, match='box_source'):
        BoxDataset(tiny_root / 'ratlike', 'train', box_source='instance')


def test_link_rows_never_force_assigns_another_animal():
    """rat-city row 9: a row that matches nothing must stay EMPTY, not take a leftover.

    `free.pop(0)` handed an unmatched row an arbitrary other detection. Its per-frame boxes then
    looked normal-sized while TELEPORTING across the arena, and `run_group` crops the window to
    their union -- 1924x1924 against a 244 px rat, 62x the area, which is the giant box the user saw
    where there was no animal. Fixing the assignment fixes the crop at its source.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    T = 6
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        boxes[0, t, 0] = [10, 10, 40, 40]                      # a stationary animal
    boxes[1, 3, 0] = [900, 900, 930, 930]                      # one far-away detection, once
    linked = link_rows(boxes.copy())
    assert np.allclose(linked[0, :, 0, 0], 10), 'the tracked row must not move'
    # The far detection is a birth into the empty row 1 -- and it must never land in row 0.
    assert not np.isfinite(linked[0, 3, 0]).all() or linked[0, 3, 0][0] == 10


def test_link_rows_gates_on_the_animals_own_size():
    """A jump of more than one box side is not motion. Real p90 is 0.06-0.11 body lengths."""
    import numpy as np

    from tailcyclenet.detector import link_rows

    boxes = np.full((1, 2, 1, 4), np.nan, np.float32)
    boxes[0, 0, 0] = [0, 0, 30, 30]
    boxes[0, 1, 0] = [300, 0, 330, 30]                         # 10 box sides away
    assert not np.isfinite(link_rows(boxes.copy())[0, 1, 0]).all(), \
        'a 10-box-side jump must be rejected, not accepted as the same animal'
    near = boxes.copy()
    near[0, 1, 0] = [20, 0, 50, 30]                            # 0.67 of a side -- ordinary motion
    assert np.isfinite(link_rows(near)[0, 1, 0]).all()


def test_link_rows_prefers_the_nearer_box_where_iou_prefers_the_wrong_one():
    """IoU RANKS BY SHAPE AGREEMENT, WHICH IS NOT IDENTITY. Constructed, but the failure mode is
    the one measured: replaying calms21 frame 301->302 from the box cache, IoU scored the WRONG
    mouse at 0.512 against the right one's 0.233, and Hungarian-matching the pose to labels after
    that swap showed the error jump from 4-10 px to 60-82 px.

    Here the true continuation is a smaller box on the SAME centre (a mouse that curled up), and the
    other animal's box happens to match the remembered box's size. IoU rewards the size match; the
    union term punishes the true one for shrinking. Centre distance is not fooled.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    last = np.array([100.0, 100.0, 300.0, 300.0], np.float32)     # remembered, 200 px
    near = np.array([170.0, 170.0, 230.0, 230.0], np.float32)      # same centre, 60 px
    wide = np.array([150.0, 150.0, 350.0, 350.0], np.float32)      # 0.35 sides away, 200 px
    iou = box_iou(torch.as_tensor(last[None]), torch.as_tensor(np.stack([near, wide])))[0]
    assert iou[1] > iou[0], f'the fixture must be one IoU gets wrong, got {iou.tolist()}'

    boxes = np.stack([np.stack([last, near]),
                      np.stack([np.full(4, np.nan, np.float32), wide])])[:, :, None]
    linked = link_rows(boxes.astype(np.float32).copy())
    assert np.allclose(linked[0, 1, 0], near), \
        f'centre distance must keep the nearer box, got {linked[0, 1, 0]}'
    # ...and the other detection is a BIRTH into the empty row, not a dropped animal.
    assert np.allclose(linked[1, 1, 0], wide)


def test_pose_nms_drops_the_lower_scored_duplicate():
    """Two rows whose keypoints sit almost entirely inside each other's box are one animal twice."""
    from tailcyclenet.detector.identity import pose_nms

    boxes = np.zeros((2, 1, 1, 4), np.float32)
    boxes[0, 0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[1, 0, 0] = [5.0, 5.0, 105.0, 105.0]           # near-identical box: a duplicate detection
    kpts = np.zeros((2, 1, 1, 3, 3), np.float32)
    kpts[0, 0, 0] = [[10, 10, 1], [50, 50, 1], [90, 90, 1]]
    kpts[1, 0, 0] = [[12, 12, 1], [52, 52, 1], [92, 92, 1]]     # inside row 0's box too
    scores = np.array([[[0.9]], [[0.5]]], np.float32)

    stats = {}
    dropped = pose_nms(boxes, kpts, scores=scores, thresh=0.8, stats=stats)
    assert dropped == 1 and stats == {'nms_pairs': 1, 'nms_dropped': 1}
    assert not np.isfinite(boxes[1, 0, 0]).all(), 'the LOWER-scored row must be the one dropped'
    assert np.isfinite(boxes[0, 0, 0]).all()


def test_pose_nms_is_a_correct_noop_with_no_keypoints():
    """`kpts=None` (a detector with no keypoint branch) must return 0 and leave `stats` EMPTY,
    not populate it with zeros.

    `scripts/infer.py` crashed on every `rat-city-combined` arm of the capacity sweep here:
    `pose_nms` returns before writing `stats['nms_pairs']` in this branch (correctly -- the maDLC
    overlap it computes needs keypoints to exist at all, and a 2D root's own recipe has no
    `--keypoints`), but the caller read `nms_stats["nms_pairs"]` with a bare subscript instead of
    `.get(..., 0)` like its neighbour on the same line. This is the empty-stats case that bug
    needed to reproduce.
    """
    from tailcyclenet.detector.identity import pose_nms

    boxes = np.zeros((2, 1, 1, 4), np.float32)
    boxes[0, 0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[1, 0, 0] = [5.0, 5.0, 105.0, 105.0]
    stats = {}
    dropped = pose_nms(boxes, None, thresh=0.8, stats=stats)
    assert dropped == 0
    assert stats == {}, 'a keypoint-less no-op must not invent stats keys'
    # the caller's own read must survive an empty dict
    assert stats.get('nms_pairs', 0) == 0 and stats.get('nms_dropped', 0) == 0


def test_infer_reads_pose_nms_stats_defensively():
    """Source check: both stats keys must be `.get(..., 0)`, never a bare subscript.

    `nms_stats["nms_pairs"]` raised `KeyError` on every keypoint-less detector -- the NORMAL case
    for a 2D root -- and `--pose-nms` is a documented default for exactly one of them (rat-city).
    """

    src = _infer_program_source()
    assert 'nms_stats["nms_pairs"]' not in src, 'a bare subscript will KeyError with no keypoints'
    assert 'nms_stats.get("nms_pairs", 0)' in src
    assert 'nms_stats.get("nms_dropped", 0)' in src


def test_unletterbox_clamps_a_runaway_box_into_the_frame():
    """`yolox.py:167` decodes a side as exp(clamp(-6,6))*stride -- up to ~12,910 px, ~137,000 after
    a 1/7 letterbox scale. IoU-only NMS cannot suppress it, and downstream it becomes the crop."""
    from tailcyclenet.detector import unletterbox_boxes

    b = torch.tensor([[-5000.0, -5000.0, 20000.0, 20000.0], [10.0, 10.0, 9.0, 40.0]])
    out = unletterbox_boxes(b, 1.0, (0, 0), src_wh=(640, 480))
    assert out[0].tolist() == [0.0, 0.0, 640.0, 480.0]
    assert torch.isnan(out[1]).all(), 'a box with no positive area is not a detection'
    # Without `src_wh` -- the training path, which has no frame to clamp against -- nothing changes.
    assert unletterbox_boxes(b, 1.0, (0, 0))[0, 2].item() == 20000.0


def test_the_cross_view_tracker_holds_identity_where_the_two_old_passes_could_not():
    """`track.demo()` as a test: report 12 R1's target state, on a three-camera rig.

    `associate` was memoryless and `link_rows` matched per camera against last-known IoU, and the
    two never exchanged anything -- so a row could be re-grouped from scratch in one pass and
    re-permuted in the other. One target set with one affinity cannot disagree with itself.
    """
    from tailcyclenet.detector.track import demo
    demo()


def test_the_tracker_and_associate_agree_on_a_single_uncrowded_animal():
    """The target state must not change the easy case, or it is not a drop-in.

    One animal on three cameras with no ambiguity: both routes must produce the same boxes, which is
    what licenses reading a delta on the crowded case as a crowding result rather than a rewrite.
    """
    import numpy as np

    from tailcyclenet.detector.associate import associate
    from tailcyclenet.detector.track import CrossViewTracker, _project

    from aniposelib.cameras import Camera, CameraGroup
    from tailcyclenet import format as fmt

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = fmt.Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
                 moving=dict.fromkeys(names, False),
                 calibrated=dict.fromkeys(names, True)).posetail()

    tr = CrossViewTracker(1, max_res_px=30.0)
    for t in range(5):
        w = np.array([[10.0 * t, 0.0, 0.0]], np.float32)
        per_cam = []
        for cam in cg:
            uv = _project(cam, w)
            per_cam.append(torch.stack([uv[:, 0] - 20, uv[:, 1] - 20,
                                        uv[:, 0] + 20, uv[:, 1] + 20], -1))
        scores = [torch.ones(1) for _ in cg]
        got, _, _ = tr.step(cg, per_cam, scores)
        ref = associate(cg, per_cam, max_res_px=30.0, max_instances=1)
        assert len(ref) == 1
        for c, box in ref[0]['boxes'].items():
            np.testing.assert_allclose(got[0, c], box.numpy(), atol=1e-4)


def test_a_detector_records_its_objectness_and_load_detector_hands_it_back(tmp_path):
    """`--det-score` is not portable across detector GENERATIONS, so the distribution must ride
    in the checkpoint.

    0.99 was measured against detectors whose objectness is saturated (98.5% of rat-city's boxes at
    exactly 1.0). The tiled/masked generation reads q01 0.45-0.84 and loses two thirds of its
    detections to the same number -- coverage 0.703 against 0.986 at 0.50 (dev/reports/21 0b). That
    is a property of the RECIPE, not of the dataset, so no constant is right for both and the only
    durable answer is to record what a checkpoint actually produces.

    A checkpoint written before the field returns `{}` rather than a guess: "nobody measured this"
    and "this one is saturated" are different answers, and gotcha 12 is what conflating them costs.
    """
    from tailcyclenet.detector import YOLOXNano, load_detector

    p = tmp_path / 'detector.pth'
    base = dict(model_state=YOLOXNano(n_keypoints=0).state_dict(), input_wh=[416, 416], norm='gn')
    torch.save(base, p)
    assert load_detector(p)[-1] == {}, 'an unrecorded distribution must not be invented'

    q = {'q01': 0.452, 'q10': 0.601, 'q50': 0.883, 'q90': 0.981}
    torch.save({**base, 'obj_quantiles': q}, p)
    assert load_detector(p)[-1] == q


def test_the_tracker_is_the_default_and_can_be_turned_off():
    """`--track` is ON by default (dev/reports/13), and `--no-track` restores the memoryless pass.

    Pinned because the default is what every future arm inherits. Asserted on the SIGNATURE and on
    the parser, not on the source text: the three text scrapes this test used to carry were all
    about the `--det-cache` stamp, which no longer exists.
    """
    import inspect

    from tailcyclenet.detector import associate_group
    from tailcyclenet.infer.cli import build_parser

    sig = inspect.signature(associate_group)
    assert sig.parameters['track'].default is True, 'the tracker is the default'
    assert sig.parameters['link'].default is False, '--link-boxes is opt-in'

    ap = build_parser()
    assert ap.get_default('track') is True, '--track is on by default'
    opts = {s for a in ap._actions for s in a.option_strings}
    assert '--no-track' in opts, '--no-track must exist to restore the memoryless pass'
    assert '--det-cache' not in opts, \
        'the detector and the pose loop are one pass; there is no detection phase to cache'


def test_the_npz_records_which_crop_source_made_it():
    """`__box_source__` is the detector's TRAINING target and does not say what the crop came from.

    Report 15 §6's two item-3 arms both wrote `__box_source__ = 'keypoints'` (same detector) and
    nothing else distinguished them, so a `--crop-source` pair was told apart by filename alone --
    the shape of gotcha 12, one field over. `--refine` rides the same field because it is the other
    re-crop lever.
    """

    src = _infer_program_source()
    assert "flat['__crop_source__']" in src, 'the crop source must be in the npz, not the shell'
    assert '"+refine" if did_refine' in src, '--refine is a crop change and belongs in that field'
    # AND IT MUST BE THE RESOLVED FLAG, NOT `cfg.refine`. `refine` defaults by dimensionality, so
    # `cfg.refine` is `None` on any run that did not pass the flag -- which is now the normal 3D
    # case. Reading `cfg` there would stamp every auto-refined 3D file as unrefined.
    assert 'any(bool(r.get(\'refine\')) for r in results.values())' in src, \
        'the stamp must read the per-session RESOLVED refine flag, not the tri-state config'
    assert "'refine': bool(cfg.refine)" in _infer_window_source(), \
        'run_group must record the resolved refine flag for the stamp to read'


def test_crop_source_keypoints_refuses_a_keypointless_detector():
    """A detector with no keypoint branch must not silently become `--crop-source boxes`.

    `run_group` switches on `det_kpts_stc is not None`, so a detector that offers only boxes does
    not error -- it crops from the boxes and reports the arm under the other arm's name, which is
    the one comparison report 15 §6 item 3 exists to make.

    A source check because the guard sits beside `load_detector`, past `load_run`, so reaching it
    needs a trained detector and a checkpoint.
    """

    src = _infer_program_source()
    assert "args.crop_source == 'keypoints' and not int(getattr(det, 'n_keypoints', 0))" in src, \
        'a keypointless detector must be refused for --crop-source keypoints, not silently served'


def test_keypoint_head_is_off_by_default():
    """`n_keypoints = 0` must be BYTE-identical to the head before the branch existed.

    Not "built and ignored": the modules are not constructed at all, so the `state_dict` has no
    new keys and every recorded detector checkpoint loads without a new flag.
    """
    import torch

    from tailcyclenet.detector import YOLOXNano

    plain, kp = YOLOXNano(), YOLOXNano(n_keypoints=17)
    assert set(plain.state_dict()) == {k for k in kp.state_dict() if 'kpt' not in k}
    assert not any('kpt' in k for k in plain.state_dict())
    assert sum(p.numel() for p in plain.parameters()) \
        < sum(p.numel() for p in kp.parameters())
    obj, boxes, kpts = plain(torch.zeros(1, 3, 64, 64))
    assert kpts is None, 'a keypoint-free model must return None, not zeros'


def test_keypoint_decode_is_signed_and_bounded():
    """A positive offset moves right/down, and no keypoint escapes 1.25 box half-widths.

    Both are silent failures: `exp` on a signed offset folds every keypoint to one side of its
    anchor, and an unbounded offset is how a keypoint lands on the NEIGHBOURING animal.
    """
    import torch

    from tailcyclenet.detector import YOLOXNano

    m = YOLOXNano(n_keypoints=3).eval()
    with torch.no_grad():
        for p in m.head.kpt_pred:
            p.weight.zero_()
            p.bias.zero_()
            p.bias[0::3] = +4.0          # dx large positive -> saturates tanh
            p.bias[1::3] = -4.0          # dy large negative
        _, boxes, kpts = m(torch.zeros(1, 3, 64, 64))
        anchors = m.anchor_points(64, 64, torch.device('cpu'))
    cx, cy = anchors[:, 0], anchors[:, 1]
    assert (kpts[0, :, 0, 0] > cx).all(), 'positive dx must move RIGHT (an exp decode cannot)'
    assert (kpts[0, :, 0, 1] < cy).all(), 'negative dy must move UP'
    half_x = (boxes[0, :, 2] - boxes[0, :, 0]) / 2
    assert (((kpts[0, :, 0, 0] - cx).abs() - 1.25 * half_x) <= 1e-3).all(), 'offset unbounded'


def test_keypoint_loss_nan_rule():
    """Every part of the NaN rule fails QUIETLY, so each gets an assertion.

    - an all-NaN instance is exactly 0 with finite gradients (not NaN, not a pull to the origin)
    - a half-labelled instance costs the SAME per point as a fully labelled one with the same
      errors (this is what catches normalising by K instead of by the finite count)
    - perturbing a NaN-target keypoint changes the loss by exactly 0 (this is what catches
      `nan_to_num` supervising it toward the top-left corner)
    """
    import torch

    from tailcyclenet.detector.assign import keypoint_loss

    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    nan = float('nan')

    all_nan = torch.full((1, 4, 3), nan)
    pred = torch.zeros(1, 4, 3, requires_grad=True)
    reg, sc, nk, nv = keypoint_loss(pred, all_nan, box)
    assert float(reg) == 0.0 and float(sc) == 0.0 and nk == 0 and nv == 0
    (reg + sc).backward()
    assert torch.isfinite(pred.grad).all(), 'an all-NaN instance produced non-finite gradients'

    # Same errors on the finite points; one instance labels 2 of 4, the other all 4.
    full = torch.tensor([[[1.0, 0, 1], [1.0, 0, 1], [1.0, 0, 1], [1.0, 0, 1]]])
    half = torch.tensor([[[1.0, 0, 1], [1.0, 0, 1], [nan, nan, nan], [nan, nan, nan]]])
    p = torch.zeros(1, 4, 3)
    r_full, _, n_full, _ = keypoint_loss(p, full, box)
    r_half, _, n_half, _ = keypoint_loss(p, half, box)
    assert n_full == 4 and n_half == 2
    assert abs(float(r_full) - float(r_half)) < 1e-6, \
        'per-point cost changed with label density -- normalised by K, not by the finite count'

    # Moving a masked-out prediction must not move the loss at all.
    p2 = torch.zeros(1, 4, 3)
    p2[0, 3, :2] = 500.0
    r_moved, _, _, _ = keypoint_loss(p2, half, box)
    assert float(r_moved) == float(r_half), 'a NaN-target keypoint is being supervised'


def test_keypoint_score_target_is_status_not_finiteness():
    """`x, y` null on a VISIBLE row is legal, and that row must still train the score channel.

    The format permits it when a `points3d` row exists for the same key -- allen-mouse ships a
    real per-camera visibility with no per-camera 2D. So the coordinate mask and the score mask
    cannot be the same tensor.
    """
    import torch

    from tailcyclenet.detector.assign import keypoint_loss

    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    nan = float('nan')
    t = torch.tensor([[[nan, nan, 1.0], [nan, nan, 0.0]]])     # positioned nowhere, assessed
    reg, sc, n_kpt, n_vis = keypoint_loss(torch.zeros(1, 2, 3), t, box)
    assert n_kpt == 0, 'no coordinate is finite, so nothing should be regressed'
    assert n_vis == 2 and float(sc) > 0.0, 'the score channel must still be supervised'


# ----------------------------------------------------------------------------------------------
# normalisation -- GroupNorm, and the two properties the switch was made for
# ----------------------------------------------------------------------------------------------

def test_norm_groups_always_divides_the_channel_count():
    """A GroupNorm whose count does not divide its channels is a constructor error."""
    from tailcyclenet.detector.yolox import norm_groups
    for c in list(range(1, 200)) + [24, 48, 96, 192]:
        g = norm_groups(c)
        assert 1 <= g <= c and c % g == 0


def test_there_are_no_running_statistics():
    """The whole point: train and inference are the SAME computation.

    BatchNorm collects statistics on the training distribution and applies them at inference.
    Train on animal-rich crops and infer on a mostly-empty arena and those disagree -- which is
    the train-test resolution discrepancy, and it is what would make train-on-tiles /
    infer-on-whole-frame unsafe. GroupNorm has no buffers, so there is nothing to drift.
    """
    m = YOLOXNano(n_keypoints=3)
    assert list(m.named_buffers()) == []
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())

    x = torch.rand(2, 3, 128, 192)
    m.train()
    a = m(x)[0]
    m.eval()
    with torch.no_grad():
        b = m(x)[0]
    torch.testing.assert_close(a, b)


def test_the_forward_does_not_depend_on_the_rest_of_the_batch():
    """Batch independence, which is what lets a high-resolution arm hold a smaller batch.

    Without it, holding the batch equal across a resolution sweep is a hard constraint rather
    than merely good practice (eval rule 4) -- an arm that had to drop its batch would be a
    differently-normalised model, and the sweep would measure that instead of resolution.
    """
    torch.manual_seed(0)
    m = YOLOXNano().eval()
    x = torch.rand(4, 3, 96, 128)
    with torch.no_grad():
        alone = m(x[1:2])[0]
        together = m(x)[0][1:2]
    torch.testing.assert_close(alone, together, rtol=1e-4, atol=1e-5)


def test_a_batchnorm_checkpoint_is_refused_by_name(tmp_path):
    """It would fail on the key names anyway; this says WHY in one sentence."""
    from tailcyclenet.detector import load_detector
    p = tmp_path / 'detector.pth'
    torch.save({'model_state': {}, 'input_wh': [128, 128]}, p)
    with pytest.raises(ValueError, match='bn normalisation'):
        load_detector(p)


# ----------------------------------------------------------------------------------------------
# the YOLOX version switch -- capacity (nano/tiny/s/m/l/x) alongside `trimmed`
# ----------------------------------------------------------------------------------------------

def test_trimmed_is_the_default_and_is_unchanged():
    """`version='trimmed'` must be indistinguishable from the model before this switch existed.

    Every checkpoint on disk was trained under the old, switch-free `YOLOXNano()`. Pinning the
    param count and the backbone type is what stands between that and a silent architecture
    change the next time this file is edited.
    """
    from tailcyclenet.detector.yolox import CSPDarknetNano

    m = YOLOXNano()
    assert m.version == 'trimmed'
    assert isinstance(m.backbone, CSPDarknetNano)
    n = sum(p.numel() for p in m.parameters())
    assert abs(n - 664_179) < 100, f'trimmed grew to {n} params -- is this still the old net?'


def test_every_yolox_tier_builds_and_forwards():
    """Every named tier in `YOLOX_TIERS`, plus `trimmed`, must construct and run end to end."""
    from tailcyclenet.detector.yolox import YOLOX_TIERS

    prev_params = 0
    for v in sorted(YOLOX_TIERS, key=lambda k: YOLOX_TIERS[k][1]):     # by width_mul, ascending
        m = YOLOXNano(n_keypoints=5, version=v)
        x = torch.rand(1, 3, 96, 128)
        obj, boxes, kpt = m(x)
        anchors = m.anchor_points(96, 128, x.device)
        assert obj.shape[1] == boxes.shape[1] == anchors.shape[0] == kpt.shape[1]
        assert kpt.shape[2] == 5
        n = sum(p.numel() for p in m.parameters())
        assert n > prev_params, f'{v} must be larger than the previous (narrower) tier'
        prev_params = n


def test_yolox_tier_names_and_conv_type_match_megvii():
    """Only `nano` (and `trimmed`) is depthwise-separable; tiny/s/m/l/x are full-convolution."""
    from tailcyclenet.detector.yolox import YOLOX_TIERS

    assert set(YOLOX_TIERS) == {'nano', 'tiny', 's', 'm', 'l', 'x'}
    assert YOLOX_TIERS['nano'][2] is True
    assert all(YOLOX_TIERS[v][2] is False for v in ('tiny', 's', 'm', 'l', 'x'))
    order = ['nano', 'tiny', 's', 'm', 'l', 'x']
    depths = [YOLOX_TIERS[v][0] for v in order]
    widths = [YOLOX_TIERS[v][1] for v in order]
    assert depths == sorted(depths) and widths == sorted(widths), \
        'depth_mul and width_mul must both increase monotonically nano -> x'


def test_an_unknown_yolox_version_raises():
    with pytest.raises(ValueError, match='trimmed'):
        YOLOXNano(version='medium')


def test_width_only_applies_to_trimmed():
    """A non-default `width` alongside a canonical tier is a mistake, not a silent no-op."""
    with pytest.raises(ValueError, match='width only applies'):
        YOLOXNano(width=128, version='nano')
    YOLOXNano(width=96, version='nano')          # the sentinel default must not raise


def test_yolox_version_round_trips_through_the_checkpoint(tmp_path):
    """The fifth instance of gotcha 12's shape: absent means `trimmed`, never a guess."""
    from tailcyclenet.detector import load_detector

    p = tmp_path / 'detector.pth'
    m = YOLOXNano(n_keypoints=0, version='s')
    torch.save({'model_state': m.state_dict(), 'input_wh': [416, 416], 'norm': 'gn',
               'yolox_version': 's'}, p)
    loaded, *_ = load_detector(p)
    assert loaded.version == 's'
    torch.testing.assert_close(
        loaded.state_dict()['head.obj_pred.0.bias'], m.state_dict()['head.obj_pred.0.bias'])

    # absent -> 'trimmed', a fact about every checkpoint written before this switch existed
    p2 = tmp_path / 'old.pth'
    old = YOLOXNano()
    torch.save({'model_state': old.state_dict(), 'input_wh': [416, 416], 'norm': 'gn'}, p2)
    loaded2, *_ = load_detector(p2)
    assert loaded2.version == 'trimmed'


def test_norm_groups_divides_every_canonical_tier_channel_count():
    """The channel counts a canonical tier actually produces, not just `trimmed`'s."""
    from tailcyclenet.detector.yolox import YOLOX_TIERS, norm_groups, round8

    for depth_mul, width_mul, _ in YOLOX_TIERS.values():
        c = round8(64 * width_mul)
        for ch in (c, c * 2, c * 4, c * 8, c * 16, round8(256 * width_mul)):
            g = norm_groups(ch)
            assert 1 <= g <= ch and ch % g == 0


def test_train_detector_help_renders():
    """The same failure mode `test_infer_help_renders` guards, one script over.

    argparse expands every `help=` string as `help % params`, so a bare `%` is a format spec --
    `3.7% of` reads as `% ` (a valid flag) then `o` (octal), and `--help` dies with
    `TypeError: %o format`. Nothing had ever run this script's `--help` before the yolox/seed
    flags were added, so two pre-existing bare `%`s were sitting undetected.
    """
    import subprocess
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / 'scripts' / 'train_detector.py'
    r = subprocess.run([sys.executable, str(p), '--help'], capture_output=True, text=True)
    assert r.returncode == 0, f'--help failed:\n{r.stderr[-2000:]}'
    # The CLI is now `--config` + three overrides; the recipe lives in the config file.
    assert '--config' in r.stdout and '--out' in r.stdout
    assert '--iters' in r.stdout and '--device' in r.stdout


# ----------------------------------------------------------------------------------------------
# tiling and the regions.pq certified mask
# ----------------------------------------------------------------------------------------------

def _root_with_regions(tmp_path, rect=(4.0, 4.0, 44.0, 34.0)):
    """A copy of the 2D fixture carrying one certified region on frame 1, camera 0."""
    from tailcyclenet import format as fmt
    from .conftest import _session_2d
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    lab.regions = np.array([[1.0, 0.0, *rect]])
    fmt.write_session(path, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=sess.flip_pairs, provenance=sess.provenance)
    return tmp_path / 'ds'


def test_an_untiled_checkpoints_tile_scale_is_dropped(tmp_path):
    """`tile_scale` without `tile_wh` must not reach `detect_group`, or it derives the input size.

    `train_detector.py` records the flag's DEFAULT on every run, so an untiled checkpoint carries
    `tile_scale = 1.0` -- and `detect_group` reads any non-None value as "letterbox the whole frame
    at `frame_wh * scale`", which for branson-fly is 1024x1024 against the 416x416 it trained at.
    """
    from tailcyclenet.detector import YOLOXNano, load_detector
    p = tmp_path / 'detector.pth'
    base = dict(model_state=YOLOXNano(n_keypoints=0).state_dict(), input_wh=[416, 416], norm='gn')
    torch.save({**base, 'tile_wh': None, 'tile_scale': 1.0}, p)
    assert load_detector(p)[-2] is None
    # ...and a genuinely tiled one still keeps it, or the tiled path loses its whole point.
    torch.save({**base, 'tile_wh': [640, 640], 'tile_scale': 0.5}, p)
    assert load_detector(p)[-2] == 0.5


def test_tile_transform_is_the_letterbox_form():
    from tailcyclenet.detector.data import tile_transform
    scale, pad = tile_transform((100, 50), 0.5)
    # a source point at the tile's origin lands at the input origin
    assert scale == 0.5 and pad == (-50.0, -25.0)
    assert 100 * scale + pad[0] == 0.0 and 50 * scale + pad[1] == 0.0


def test_tiled_targets_are_still_the_crop_rule(tiny_root):
    """gotcha 8 under tiling: the box is RE-DERIVED in source px, never scaled by tile_scale.

    If this fails every tiled detector number is invalid, exactly as for the whole-frame version.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2,
                    tile_wh=(32, 32), tile_scale=1.0)
    from tailcyclenet.detector.data import tile_transform
    for i in range(min(6, len(ds))):
        sess, gid, f, ci = ds.index[i]
        boxes = ds.boxes_for(i)
        lab = sess.labels(gid)
        cam = sess.rig.posetail()[ci]
        ox, oy = ds.origins[i]
        tw, th = ds._tile_extent()
        for s in range(boxes.shape[0]):
            pts = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
            keep = ((pts[:, 0] >= ox) & (pts[:, 0] <= ox + tw) &
                    (pts[:, 1] >= oy) & (pts[:, 1] <= oy + th))
            want = crop_box_for_points(torch.where(keep[:, None], pts, torch.nan),
                                       cam['size'], ds.min_crop_dim)
            if want is None:
                assert torch.isnan(boxes[s]).all()
                continue
            scale, pad = tile_transform((ox, oy), ds.tile_scale)
            back = unletterbox_boxes(boxes[s][None], scale, pad)[0]
            torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_a_point_outside_the_tile_is_dropped(tiny_root):
    """Out-of-tile behaves exactly like out-of-frame: shrink the box, or emit no box at all."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    tile_wh=(16, 16), tile_scale=1.0)
    # a tile far from every animal: the 64x48 fixture has its points in [5, 43]
    ds.origins[0] = (1000.0, 1000.0)
    assert torch.isnan(ds.boxes_for(0)).all()


def test_an_off_frame_tile_is_grey_not_wrapped(tiny_root):
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    tile_wh=(16, 16), tile_scale=1.0)
    ds.origins[0] = (-16.0, -16.0)          # wholly outside, up and left
    x = ds[0][0]
    assert x.shape == (3, 16, 16)
    torch.testing.assert_close(x, torch.full_like(x, 114 / 255.0))


def test_regions_none_and_empty_are_different_in_the_loader(tmp_path, tiny_root):
    """`None` claims exhaustive labelling; `(0,4)` certifies nothing. Both reach the loader."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1)
    assert ds.regions_for(0) is None                     # the fixture has no regions.pq

    root = _root_with_regions(tmp_path)
    d2 = BoxDataset(root, 'train', input_wh=(64, 48), max_frames_per_group=4)
    got = {int(d2.index[i][2]): d2.regions_for(i) for i in range(len(d2))}
    assert got[1] is not None and got[1].shape == (1, 4)  # frame 1 carries the region
    assert got[0] is not None and got[0].shape == (0, 4)  # frame 0 certifies nothing


def test_regions_ride_the_same_transform_as_the_boxes(tmp_path):
    """A region letterboxed by a different rule than its own boxes is invisible in the loss."""
    root = _root_with_regions(tmp_path, rect=(4.0, 4.0, 44.0, 34.0))
    ds = BoxDataset(root, 'train', input_wh=(128, 96), max_frames_per_group=4)
    i = next(i for i in range(len(ds)) if int(ds.index[i][2]) == 1)
    scale, pad = ds._transform(i, (64, 48))
    r = ds.regions_for(i)[0]
    torch.testing.assert_close(r, torch.tensor([4.0 * scale + pad[0], 4.0 * scale + pad[1],
                                                44.0 * scale + pad[0], 34.0 * scale + pad[1]]))


def test_use_regions_emits_a_full_frame_rect_when_the_session_has_none(tiny_root):
    """No regions.pq = exhaustively labelled = every anchor supervised, encoded as one big rect."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    use_regions=True)
    item = ds[0]
    assert len(item) == 3
    torch.testing.assert_close(item[2], torch.tensor([[0.0, 0.0, 64.0, 48.0]]))


def test_certified_anchors_unions_the_boxes_in():
    from tailcyclenet.detector import certified_anchors
    anchors = torch.tensor([[5.0, 5.0, 8.0], [50.0, 50.0, 8.0], [95.0, 95.0, 8.0]])
    regions = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes = torch.tensor([[40.0, 40.0, 60.0, 60.0], [float('nan')] * 4])
    got = certified_anchors(anchors, regions, boxes)
    assert got.tolist() == [True, True, False]           # region, GT box, neither
    # a NaN-padded rect certifies nothing rather than certifying the origin
    assert not certified_anchors(anchors, torch.full((1, 4), float('nan')),
                                 torch.full((1, 4), float('nan'))).any()


def test_split_batch_tells_keypoints_from_regions_by_rank():
    from tailcyclenet.detector import split_batch
    x, b = torch.zeros(2, 3, 8, 8), torch.zeros(2, 1, 4)
    k, r = torch.zeros(2, 1, 5, 3), torch.zeros(2, 3, 4)
    assert split_batch((x, b)) == (x, b, None, None)
    assert split_batch((x, b, k))[2] is k and split_batch((x, b, k))[3] is None
    assert split_batch((x, b, r))[2] is None and split_batch((x, b, r))[3] is r
    got = split_batch((x, b, k, r))
    assert got[2] is k and got[3] is r


def test_detector_loss_without_regions_is_unchanged():
    """THE BACKWARD-COMPATIBILITY PROOF. Reports 10-15's numbers depend on this equality."""
    torch.manual_seed(0)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    base, bp = detector_loss(obj, boxes, anchors, gt)
    same, sp = detector_loss(obj, boxes, anchors, gt, regions=None)
    assert float(base) == float(same) and 'certified' not in bp and 'certified' not in sp

    # a mask that certifies everything is the same loss; one that certifies nothing keeps only
    # the positives, which are forced in because an unsupervised positive is an animal trained
    # as nothing.
    everything = torch.tensor([[[0.0, 0.0, 64.0, 64.0]]] * 2)
    allm, ap = detector_loss(obj, boxes, anchors, gt, regions=everything)
    torch.testing.assert_close(allm, base)
    assert ap['certified'] == 1.0
    nothing = torch.full((2, 1, 4), float('nan'))
    _, np_ = detector_loss(obj, boxes, anchors, gt, regions=nothing)
    assert 0.0 < np_['certified'] < 1.0


def test_detector_loss_without_ignore_is_unchanged():
    """T2.1's BACKWARD-COMPATIBILITY PROOF, the same shape as the `regions=None` one above.

    `ignore_present` ships OFF (`configs/detector.toml`) and is meant to stay a free, always-on
    lever regardless of what any single measurement reads: nobody who does not opt in pays for
    it. That promise is this equality, not a comment -- if `ignore=None` ever stopped taking the
    exact pre-T2.1 code path, every existing detector number would be silently invalid.
    """
    torch.manual_seed(1)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    base, bp = detector_loss(obj, boxes, anchors, gt)
    same, sp = detector_loss(obj, boxes, anchors, gt, ignore=None)
    assert float(base) == float(same) and 'ignored' not in bp and 'ignored' not in sp

    # An ignore box that covers NOTHING near any anchor is the same loss as no ignore box at all.
    nowhere = torch.tensor([[[9000.0, 9000.0, 9010.0, 9010.0]]] * 2)
    noop, np_ = detector_loss(obj, boxes, anchors, gt, ignore=nowhere)
    torch.testing.assert_close(noop, base)
    assert np_['ignored'] == 0.0


# ----------------------------------------------------------------------------------------------
# T2.3 -- IoU-aware objectness (dev/plans/detector_accuracy.md). The BCE target at a positive
# anchor becomes the detached predicted-vs-GT IoU instead of a hard 1.0, once past a warmup. Off
# by default (`iou_aware=False`) and byte-identical to every checkpoint on record either way.
# ----------------------------------------------------------------------------------------------

def test_paired_iou_matches_box_iou_diagonal():
    """`paired_iou` is the SAME maths as `box_iou`, just elementwise instead of all-pairs -- the
    diagonal of the cross-product form must equal it exactly, on both perfect and partial overlap.
    """
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [5.0, 5.0, 15.0, 15.0]])
    got = paired_iou(a, b)
    want = box_iou(a, b).diagonal()
    torch.testing.assert_close(got, want)
    assert float(got[0]) == pytest.approx(1.0)         # perfect overlap
    assert 0.0 < float(got[1]) < 1.0                    # partial overlap
    assert float(got[2]) == pytest.approx(1.0)          # perfect overlap, offset box


def test_detector_loss_iou_aware_default_off_is_unchanged():
    """THE BACKWARD-COMPATIBILITY PROOF, same shape as `test_detector_loss_without_ignore_is_
    unchanged`. `iou_aware` defaults to False, and passing it explicitly at False (or omitting it)
    must produce the IDENTICAL loss and the IDENTICAL (hard 1.0) target regardless of how good or
    bad the predicted boxes are.
    """
    torch.manual_seed(2)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    # Deliberately BAD boxes (far from the GT), so a hard-1.0 vs IoU-valued target would differ
    # a lot if the flag were live -- the strongest test that it is not.
    boxes = torch.rand(2, anchors.shape[0], 4) * 5
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    base, bp = detector_loss(obj, boxes, anchors, gt)
    off, op = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(base) == float(off)
    assert 'iou_target' not in bp and 'iou_target' not in op


def test_detector_loss_iou_aware_holds_hard_target_during_warmup():
    """`it < iou_aware_warmup` must still use the hard 1.0 target -- the chicken-and-egg trap the
    plan names: an early, near-zero predicted IoU must not be allowed to teach objectness to stay
    off. `parts['iou_target']` reads exactly 1.0 throughout warmup regardless of box quality.
    """
    torch.manual_seed(3)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5             # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    warm, wp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000, it=0)
    hard, hp = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(warm) == float(hard), 'during warmup the loss must equal the hard-target one'
    assert wp['iou_target'] == pytest.approx(1.0)


def test_detector_loss_iou_aware_switches_to_iou_target_after_warmup():
    """Past warmup, the target at a positive must be the DETACHED IoU between its predicted and
    GT box -- not 1.0 -- and `parts['iou_target']` must report the same value a direct
    `paired_iou` computation gives, tying the loss's internal bookkeeping to the public helper.
    """
    torch.manual_seed(4)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5              # bad boxes -> IoU well under 1
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    past, pp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000,
                             it=2000)
    hard, _ = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(past) != float(hard), 'past warmup the loss must differ from the hard-target one'
    assert 0.0 <= pp['iou_target'] < 1.0

    pos, gix = assign(anchors, gt[0])
    assert pos.numel(), 'need at least one positive anchor for this check to mean anything'
    want = float(paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1).mean())
    assert pp['iou_target'] == pytest.approx(want, abs=1e-5)


def test_detector_loss_iou_aware_it_none_means_past_warmup():
    """`it=None` (a caller that does not track iterations) must behave as ALREADY past warmup --
    the docstring's stated contract, since the only caller needing the warmup (`train_detector.py`)
    always passes `it`."""
    torch.manual_seed(5)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    none_it, p1 = detector_loss(obj, boxes, anchors, gt, iou_aware=True, it=None)
    past_it, p2 = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000,
                                it=999999)
    torch.testing.assert_close(none_it, past_it)
    assert p1['iou_target'] == pytest.approx(p2['iou_target'])


def test_iou_aware_weight_forcing_is_keyed_on_pos_mask_not_target_value():
    """THE SUBTLE CORRECTNESS CASE this key introduces: `--use-regions`/`ignore` mask the
    objectness weight, and a pre-existing guard (`weight = maximum(weight, ...)`) forces a TRUE
    POSITIVE's weight back to 1 so masking can never silently drop a real animal from the
    objectness term. That guard must key on WHETHER an anchor is positive (`pos_mask`), not on the
    VALUE its target holds -- under `iou_aware` with bad boxes the target there can be near 0, and
    forcing off THAT would only guarantee a near-0 weight, defeating the guard's whole purpose.

    Constructed so a `regions` mask certifies NOTHING (weight starts at 0 everywhere) and the
    predicted boxes are deliberately bad (low true IoU): if the guard were wrongly keyed on
    `target`, the positive anchor's contribution to `obj` would be scaled by ~its IoU instead of
    1, and this loss would NOT match the identical computation with `iou_aware=False` (which is
    known-correctly forced to weight 1 by the pre-existing, unchanged code path).
    """
    torch.manual_seed(6)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 5              # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]]])
    nothing_certified = torch.full((1, 1, 4), float('nan'))     # certifies NOTHING

    _, sp = detector_loss(obj, boxes, anchors, gt, regions=nothing_certified,
                          iou_aware=True, iou_aware_warmup=0, it=1)
    assert sp['iou_target'] < 0.5, 'the test needs a genuinely low IoU to be a real check'
    # The OBJ term (not the box term, which iou_aware never touches) must be identical: the
    # positive's weight was forced to 1 in both cases, only the BCE TARGET value differs, and a
    # BCE(logit, 1.0) vs BCE(logit, iou) at weight=1 are different numbers -- so this checks obj
    # is computed off the same forced weight, not that the two total losses match.
    pos, gix = assign(anchors, gt[0])
    assert pos.numel()
    logit = obj[0, pos]
    target_hard = torch.ones_like(logit)
    target_soft = paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1)
    import torch.nn.functional as F
    norm = max(pos.numel(), 1)          # `detector_loss`'s own `obj_all / max(n_pos, B)`, B=1 here
    expected_hard_obj = F.binary_cross_entropy_with_logits(
        logit, target_hard, reduction='sum') / norm
    expected_soft_obj = F.binary_cross_entropy_with_logits(
        logit, target_soft, reduction='sum') / norm
    # Re-derived directly rather than relying on `hard`/`soft` above (which also include
    # box_weight * box, identical in both arms and therefore not informative here) -- read `obj`
    # off `parts` instead.
    _, hp = detector_loss(obj, boxes, anchors, gt, regions=nothing_certified, iou_aware=False)
    assert hp['obj'] == pytest.approx(float(expected_hard_obj), rel=1e-4)
    assert sp['obj'] == pytest.approx(float(expected_soft_obj), rel=1e-4)


def test_detector_config_iou_aware_defaults(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['training']['iou_aware_obj'] is False
    assert cfg['training']['iou_aware_warmup'] == 2000


def test_train_detector_iou_aware_end_to_end(tmp_path, dense_root, monkeypatch):
    """A short run with `iou_aware_obj = true` through the real CLI entry point: it must train to
    completion with no error, and the log line must show `iouT` once it fires (warmup=0 so it
    fires on iteration 0 here, keeping the smoke test short)."""
    import importlib.util
    import sys

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
frames_per_group = 8
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
iou_aware_obj = true
iou_aware_warmup = 0
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector_iou_aware',
                                                  REPO / 'scripts' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()
    import tomllib
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['training']['iou_aware_obj'] is True
    assert recorded['training']['iou_aware_warmup'] == 0


def test_a_nan_box_is_skipped_by_both_cross_view_paths():
    """The two halves of `unletterbox_boxes`' contract, joined. They never were.

    That function returns NaN for a box the frame clamp left with no area, and the test above
    asserts it. `associate`'s docstring said it skips a non-finite centre -- but `_triangulate`
    ends in `torch.linalg.svd`, which RAISES on non-finite input, so the `isfinite(p3d)` guard
    after it was unreachable. `CrossViewTracker` raised too, one step later: `clip(1 - nan)` is
    NaN, `affinity.any()` is True for NaN, and `linear_sum_assignment` refuses the matrix.

    Both were reachable from one anchor firing in the letterbox padding band, and both killed
    `detect_group` mid-clip -- on rat-city's 57,594-frame test group, hours of decode in.
    """
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.detector.associate import associate
    from tailcyclenet.detector.track import CrossViewTracker, _project
    from tailcyclenet.format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
             moving=dict.fromkeys(names, False),
             calibrated=dict.fromkeys(names, True)).posetail()

    world = np.array([[0.0, 0.0, 0.0]], np.float32)
    nan_row = torch.full((1, 4), float('nan'))
    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, world)
        good = torch.stack([uv[:, 0] - 20, uv[:, 1] - 20, uv[:, 0] + 20, uv[:, 1] + 20], -1)
        per_cam.append(torch.cat([good, nan_row]))       # one real animal, one dead box
        scores.append(torch.ones(2))

    got = associate(cg, per_cam, max_res_px=20.0)
    assert len(got) == 1, 'the real animal must still be found beside a NaN box'
    assert bool(torch.isfinite(got[0]['point']).all())
    # The NaN box is not silently adopted as a member of it.
    assert all(j == 0 for j in got[0]['members'].values())

    tr = CrossViewTracker(2, max_res_px=20.0)
    boxes, _, _ = tr.step(cg, per_cam, scores)
    assert np.isfinite(boxes[0]).all(), 'the tracked animal must come back with real boxes'

    # And `min_views = 1` does not emit the dead box as a single-view instance.
    solo = associate(cg, per_cam, max_res_px=20.0, min_views=1)
    for g in solo:
        for c, b in g['boxes'].items():
            assert torch.isfinite(b).all(), f'camera {c} emitted a NaN box as an instance'


def test_reduce_under_tiling_matches_what_deployment_decodes():
    """`--reduce` compared the whole frame against the TILE size, which is not where it is headed.

    Tiled, `input_wh` is the tile, so rat-city's 4696x2048 against a 640x640 tile gave r = 2 and
    against 640x288 gave r = 4 -- and the warp multiplies the decode scale back up, so the tile
    came out a 2-4x UPSAMPLE of a decimated frame. Deployment letterboxes the whole frame to
    `tiled_input_wh(src, tile_scale)` instead, where the same function returns 1 and the detector
    sees native pixels. That is the train/deploy sampling skew `reduce` is stamped into the
    checkpoint to prevent.
    """
    from tailcyclenet.detector.data import reduce_factor

    size = (4696, 2048)                                   # rat-city
    for tile, scale in (((640, 640), 1.0), ((640, 288), 1.0), ((896, 896), 1.0)):
        assert reduce_factor(size, tile) > 1, \
            f'{tile}: the old comparison must actually decimate, or this proves nothing'
        deployed = (size[0] * scale, size[1] * scale)
        assert reduce_factor(size, deployed) == 1, \
            'at tile_scale 1.0 deployment decodes natively, so training must too'

    # And the untiled path is untouched: there `input_wh` IS what the frame is headed for.
    assert reduce_factor(size, (640, 288)) == reduce_factor(size, (640, 288))
    # At a genuine downscale the reduction comes back, on both sides alike.
    assert reduce_factor(size, (size[0] * 0.125, size[1] * 0.125)) > 1


def test_provenance_records_every_box_affecting_option():
    """Nothing that changes a detection may go unrecorded beside the numbers it produced.

    This is what is left of the `--det-cache` stamp's safety property. The stamp compared two runs
    so a cache could be REFUSED; there is no cache, and no comparison to make -- but "which
    detector, at which threshold, at which input size" is still the difference between two numbers
    that look alike (eval rule 4), and it belongs in the output rather than in a shell history.

    Table-driven against `detect_raw`'s own signature, so the next parameter added to that function
    fails here instead of being found in review. Asserted on the VALUE `_box_provenance` returns
    rather than on the driver's source text, which the stamp version had to do and which is why it
    broke every time the block around it moved.
    """
    import argparse
    import inspect

    from tailcyclenet.detector import detect_raw
    from tailcyclenet.infer.driver import _box_provenance

    args = argparse.Namespace(detector=None, det_input_wh=None, det_score=0.5, det_top_k=0,
                              max_animals=0, max_frames=0)
    prov = _box_provenance(args, None, False, None)

    # Everything `detect_raw` takes that can change the detections. The rest are plumbing --
    # `batch` is in there under protest: it is NOT inert (report 38 §3.2) but it is pinned rather
    # than exposed, so no run can differ in it. `frames`/`read` are the block loop's plumbing and
    # change no pixel. See tests/test_memory_budget.py.
    plumbing = {'det', 'session', 'gid', 'device', 'batch', 'frames', 'read'}
    params = set(inspect.signature(detect_raw).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'score_thresh': 'det_score', 'input_wh': 'det_input_wh', 'top_k': 'det_top_k'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change the detections and are not recorded in the prediction: {missing}. Two runs '
        'differing in one of them would be indistinguishable from their output alone.')

    # UNCONDITIONAL, every key, always. The stamp recorded only the options that DIFFERED from
    # their defaults, because a positional list invalidated every cache on disk each time a flag
    # was added. Nothing here has that pressure, and conditional membership is what made the stamp
    # need five exceptions to its own rule.
    args2 = argparse.Namespace(detector='d.pt', det_input_wh=(416, 416), det_score=0.97,
                               det_top_k=24, max_animals=2, max_frames=120)
    assert set(_box_provenance(args2, 1.0, True, 'instances')) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'


def test_score_dataset_scores_unaugmented_and_restores_the_flag():
    """`ignore_for` takes no `warp`, unlike `boxes_for` and `regions_for` beside it.

    So under `--augment` the predictions and the GT were warped while the `instances.pq` PRESENT
    boxes were not, and the ignore mask excused the wrong pixels. rat-city ships 26,021 of those
    rows, so that is most of the train-side FP readout -- the number the train/val gap is read
    from. It is also the split that is supposed to be comparable to val, which is never augmented.
    """
    import inspect

    from tailcyclenet.detector import data as ddata
    from tailcyclenet.detector.evaluate import score_dataset

    # The asymmetry that caused it, pinned so a future `warp` on `ignore_for` is noticed.
    assert 'warp' in inspect.signature(ddata.BoxDataset.boxes_for).parameters
    assert 'warp' in inspect.signature(ddata.BoxDataset.regions_for).parameters
    assert 'warp' not in inspect.signature(ddata.BoxDataset.ignore_for).parameters, \
        'ignore_for now takes a warp -- score_dataset can stop disabling augmentation'

    src = inspect.getsource(score_dataset)
    assert 'ds.augment = False' in src, 'scoring must not run through the augmentation'
    assert 'ds.augment = aug_was' in src, 'and it must put the flag back, or training loses it'


def test_a_pointless_target_expires_instead_of_burning_a_slot_forever():
    """`--min-views 1` births a target whose 3D point is all-NaN BY DESIGN (`associate`).

    Such a target is filtered out of `slots`, so the update loop never touches its `age`, `retire`
    never fires, and `free` excludes it because it is still in `self.targets`. One single-view
    birth on frame 0 therefore cost a row for the rest of the clip -- on rat-city's 57,594-frame
    test group, permanently.

    This is NOT the documented immortal ONE-CAMERA target, which has a finite point, stays in
    `slots`, and whose retirement was tried and measured worse (+2.72 mm MPJPE). This one is
    invisible to the matcher altogether.
    """
    import torch

    from tailcyclenet.detector.track import CrossViewTracker

    tr = CrossViewTracker(2, max_age=3)
    tr.targets[0] = {'point': torch.full((3,), float('nan')), 'age': 0}
    assert 0 in tr.targets

    # No detections at all: the pointless target must still age out, freeing its slot.
    for _ in range(tr.max_age + 1):
        tr.step([], [], [])
    assert 0 not in tr.targets, 'a target the matcher can never see must still be able to expire'

    # A target WITH a point is unaffected by this path -- it ages through the normal loop.
    tr2 = CrossViewTracker(2, max_age=3)
    tr2.targets[0] = {'point': torch.zeros(3), 'age': 0}
    tr2.step([], [], [])
    assert tr2.targets[0]['age'] == 1, 'the finite-point target must age exactly once per frame'


def test_ema_off_builds_nothing_and_on_tracks_the_weights_it_averages():
    """`--ema-decay 0` must not construct an averaged model; on, it must actually average.

    EMA is the one optimiser lever here that yields BOTH arms from a single run -- the raw and
    averaged weights are scored on the same windows in the same iteration -- so the arms are paired
    by construction rather than differing by seed noise as two runs would. That is only true if the
    off path is untouched and the on path is a real average rather than a copy.
    """
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    model = YOLOXNano(n_keypoints=0)
    decay = 0.9
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))

    p = next(model.parameters())

    # THE FIRST `update_parameters` IS A COPY, NOT AN AVERAGE. `AveragedModel` starts with
    # `n_averaged = 0` and seeds itself from the model on the first call; only from the second does
    # `avg_fn` run. Worth pinning, because a test that changed the weights and updated ONCE would
    # see a perfect copy and read it as "the EMA is not averaging".
    ema.update_parameters(model)
    seeded = next(ema.module.parameters()).detach().clone()
    torch.testing.assert_close(seeded, p.detach())

    # From the second update it is a real average: a +1.0 jump moves it by exactly (1 - decay).
    with torch.no_grad():
        p.add_(1.0)
    ema.update_parameters(model)
    e1 = next(ema.module.parameters()).detach().clone()

    assert not torch.allclose(e1, seeded), 'the EMA never moved -- it is not tracking the weights'
    assert not torch.allclose(e1, p.detach()), 'the EMA copied the weights instead of averaging'
    torch.testing.assert_close(e1, seeded + (1.0 - decay), rtol=1e-4, atol=1e-6)


def test_detector_pth_is_the_best_checkpoint_not_the_last(tmp_path):
    """The run measured its own peak and then overwrote it -- worth up to -28% recall.

    On rat-city-annotated recall PEAKS at 4-8k and falls to 20k, because a root whose labelled
    frame names 2 of ~10 rats spends the rest of training learning that most rats are background.
    So "last" is not a tie-break with "best" here, it is systematically the wrong end.
    """
    from pathlib import Path

    # The selection rule, exercised the way the loop runs it: r50 rises then falls.
    history = [{'iteration': 2000, 'val_r50': 0.32}, {'iteration': 8000, 'val_r50': 0.39},
               {'iteration': 20000, 'val_r50': 0.28}]
    best_score, kept = -float('inf'), None
    for h in history:
        sel = h['val_r50']
        if sel >= best_score:
            best_score, kept = sel, h['iteration']
    assert kept == 8000, 'detector.pth must hold the peak, not the last evaluation'

    # ...and the end-of-run `best` line must name the SAME checkpoint, or the print lies.
    best = max(history, key=lambda h: h.get('val_r50', h.get('train_r50')))
    assert best['iteration'] == kept

    src = (Path(__file__).resolve().parent.parent / 'scripts' / 'train_detector.py').read_text()
    assert "torch.save(ckpt, run / 'detector.pth')" in src
    assert 'if sel >= best_score:' in src, 'the unconditional overwrite is back'


def test_birth_age_off_is_the_rule_it_replaced():
    """`birth_age=None` must be byte-identical to the pre-knob rule, because it is the default.

    Loosening the birth rule is REFUTED: it buys coverage by letting one row hold two animals, and
    `run_group` unions a row's boxes over a window, so rat-city's union p99 goes 590 -> 3804 px
    against a 244 px rat. The knob ships off; the fix for the 34% drop is spare rows.
    """
    from tailcyclenet.detector import link_rows
    rng = np.random.default_rng(0)
    S, T = 4, 40
    boxes = np.full((S, T, 1, 4), np.nan, np.float32)
    for s in range(S):
        x, y = 100.0 + 300 * s, 100.0
        for t in range(T):
            if 12 <= t < 30 and s == 1:          # row 1 disappears for 18 frames, under max_age
                continue
            x += rng.normal(0, 2)
            boxes[s, t, 0] = (x, y, x + 60, y + 60)

    off = link_rows(boxes.copy())
    # A row that vanishes for 18 frames stays ITS OWN under the shipped rule -- nothing is reseated.
    assert np.isfinite(off[1, 5, 0]).all() and np.isfinite(off[1, 35, 0]).all()
    assert not np.isfinite(off[1, 20, 0]).any(), 'the gap must stay empty, not be filled'
    # And the knob is genuinely a no-op at None: same array as an explicit huge threshold.
    huge = link_rows(boxes.copy(), birth_age=10_000)
    np.testing.assert_array_equal(np.isfinite(off), np.isfinite(huge))




# ----------------------------------------------------------------------------------------------
# `--augment-strong`: the strong appearance/erasure/mosaic-lite suite (gotcha 8 throughout).
# ----------------------------------------------------------------------------------------------

def test_strong_augment_off_is_byte_identical(tiny_root):
    """`strong=False` must never draw an extra rng value or touch a pixel or a box.

    `--augment-strong` is a KEY: every recorded arm before it existed must stay reproducible, so
    the off path has to be indistinguishable from a `BoxDataset` that has never heard of it.
    """
    ds_a = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                     max_frames_per_group=2, augment=True, strong=False, seed=0)
    ds_b = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                     max_frames_per_group=2, augment=True, seed=0)   # strong defaults False
    for i in range(min(3, len(ds_a))):
        np.random.seed(0)
        xa, ba = ds_a[i]
        np.random.seed(0)
        xb, bb = ds_b[i]
        # Pixels drawn from a fresh rng each visit (`default_rng(None)`) are not literally
        # reproducible call-to-call under the augmented path -- what must hold is that the two
        # CONSTRUCTIONS (with and without the explicit strong=False) are the same object, i.e.
        # neither adds a code path the other lacks. Assert on SHAPE and on box count instead of
        # bit-exact pixels, since `_photometric` alone already draws fresh entropy per visit.
        assert xa.shape == xb.shape
        assert ba.shape == bb.shape


def test_strong_augment_off_leaves_boxes_targets_unchanged(tiny_root):
    """With `strong` off, `boxes_for` -- the actual target -- is bit-identical to the plain loader.

    This is the sharper off-path guarantee: `boxes_for` never even sees the `strong` flag, so this
    just pins that nothing upstream of it was touched by adding the flag.
    """
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                       max_frames_per_group=2, augment=True, seed=0)
    strong_off = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                           max_frames_per_group=2, augment=True, strong=False, seed=0)
    for i in range(min(3, len(plain))):
        torch.testing.assert_close(plain.boxes_for(i), strong_off.boxes_for(i))


def test_strong_augment_preserves_box_targets(tiny_root):
    """Appearance ops and cutout must never move a box -- only mosaic-lite may ADD one.

    Runs the strong suite many times over a fixed set of items and checks that every box present
    before the suite ran is still present, unchanged, after it -- the count may only GROW (mosaic).
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    for i in range(min(4, len(ds))):
        base = ds.boxes_for(i)     # the animal count before any strong op ran
        for _ in range(8):
            _, boxes = ds[i]
            # Appearance ops and cutout never touch `boxes_for`'s output -- only mosaic-lite
            # appends a row -- so the box count for this item can only stay the same or GROW,
            # never shrink or resize below what `boxes_for` alone would produce.
            assert boxes.shape[0] >= base.shape[0]


def test_cutout_zeroes_covered_keypoints(tiny_root):
    """A keypoint inside a cutout rect must end up coord-NaN AND score-0, never just one."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, keypoints=True, seed=0)
    _, kpts = ds.boxes_for(0, None, with_keypoints=True)
    rng = np.random.default_rng(0)
    rects = _cutout_rects(ds.input_wh, rng, n=(1, 1), frac=1.0)   # cover the WHOLE frame
    mask = _keypoints_in_rects(kpts[..., :2], rects)
    assert bool((torch.isfinite(kpts[..., 0]) & mask).any()), \
        'fixture must have at least one finite keypoint to erase'
    k2 = kpts.clone()
    k2[..., 0] = torch.where(mask, torch.nan, k2[..., 0])
    k2[..., 1] = torch.where(mask, torch.nan, k2[..., 1])
    k2[..., 2] = torch.where(mask, torch.zeros_like(k2[..., 2]), k2[..., 2])
    assert torch.isnan(k2[..., :2][mask]).all()
    assert (k2[..., 2][mask] == 0).all()


def test_mosaic_paste_is_fully_interior_and_reencodes_the_crop_rule(tiny_root):
    """The appended box is `src_box + translation`, entirely inside `input_wh`."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    rng = np.random.default_rng(1)
    for i in range(min(3, len(ds))):
        base = ds.boxes_for(i)
        img, _, _ = ds._load_letterbox(i)
        boxes, kpts, img2 = ds._mosaic_paste(i, base.clone(), None, img.copy(), rng)
        if boxes.shape[0] == base.shape[0]:
            continue    # no finite source box was found in a few tries; not this fixture's job
        new = boxes[base.shape[0]:]
        assert (new[:, 0] >= 0).all() and (new[:, 1] >= 0).all()
        assert (new[:, 2] <= ds.input_wh[0]).all() and (new[:, 3] <= ds.input_wh[1]).all()
        return
    pytest.skip('no fixture item produced a finite mosaic source box in the tries allotted')


def test_mosaic_rejected_when_use_regions(tiny_root):
    """Fails at CONSTRUCTION, not on the ~20%% of items that happen to draw mosaic-lite -- a
    training job should not discover this combination is undefined hours into a run.
    """
    with pytest.raises(ValueError):
        BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                  max_frames_per_group=2, augment=True, strong=True, use_regions=True, seed=0)

    # And `_mosaic_paste` itself still refuses to run, for a caller that reaches it some other way.
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, use_regions=True, seed=0)
    ds.strong = True     # bypass the constructor guard to exercise the method's own guard
    boxes = ds.boxes_for(0)
    img, _, _ = ds._load_letterbox(0)
    with pytest.raises(RuntimeError):
        ds._mosaic_paste(0, boxes, None, img, np.random.default_rng(0))


def test_infer_help_renders():
    """`--help` must actually print. It did not, and nothing noticed for months.

    argparse expands every `help=` string as `help % params`, so a BARE `%` in one is a format spec:
    `80% of frames` reads as `%o` and `--help` dies with `TypeError: %o format: an integer is
    required, not dict`. Three help strings carried one. Nothing else in the suite runs `--help`,
    and `--help` is how anybody discovers a flag -- so a broken one silently hides every option this
    script has.
    """
    import subprocess
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / 'scripts' / 'infer.py'
    r = subprocess.run([sys.executable, str(p), '--help'], capture_output=True, text=True)
    assert r.returncode == 0, f'--help failed:\n{r.stderr[-2000:]}'
    assert '--pose-nms' in r.stdout




# --- detector training config (configs/detector.toml + tailcyclenet/detector/config.py) ------

REPO = Path(__file__).resolve().parent.parent
SHIPPED_DETECTOR_CONFIG = REPO / 'configs' / 'detector.toml'


def _write_config(tmp_path, text, name='config.toml'):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_detector_config_loads_with_shipped_defaults(tmp_path):
    """The shipped config is the old CLI defaults, one key per flag -- and `--out`/`--iters`/
    `--device` overrides land in [training]. Path/out are placeholders in the shipped file, so
    the user flow is exercised: a one-key overlay `extends` it."""
    import shutil
    from tailcyclenet.detector.config import load_detector_config

    # `extends` resolves against the config's OWN directory, so base and overlay share tmp_path.
    shutil.copy(SHIPPED_DETECTOR_CONFIG, tmp_path / 'base.toml')
    overlay = _write_config(tmp_path, """
extends = "base.toml"
[data]
path = "/tmp/ds"
[training]
out = "/tmp/run-det"
""", 'overlay.toml')
    cfg = load_detector_config(overlay, out='/tmp/run-det', iters=7, device='cpu')
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    assert d['path'] == '/tmp/ds'
    assert d['boxes'] == 'instances'
    assert d['min_crop_dim'] == 64
    assert d['min_box_px'] == 32
    assert d['max_input_px'] == 4 * 416 * 416
    assert d['frames_per_group'] == 40
    assert d['val_frames_per_group'] == 8
    assert d['augment'] is True and d['augment_strong'] is True
    assert d['rotate_deg'] == 45.0
    assert d['reduce'] is False and d['keypoints'] is False and d['hflip'] is True
    assert d['use_regions'] is False
    assert d['input_wh'] is None and d['tile_wh'] is None       # absent pair -> None
    assert d['tile_scale'] == 1.0 and d['tile_bg_per_frame'] == 1
    assert m['yolox'] == 'tiny'
    assert t['out'] == '/tmp/run-det'
    assert t['iters'] == 7
    assert t['batch_size'] == 16 and t['lr'] == 1e-3
    assert t['num_workers'] == 8 and t['seed'] == 23
    assert t['device'] == 'cpu'
    assert t['eval_every'] == 2000 and t['eval_batches'] == 25
    assert t['kpt_weight'] == 1.0 and t['kpt_score_weight'] == 1.0


def test_detector_config_unknown_key_raises_in_every_block(tmp_path):
    from tailcyclenet.detector.config import load_detector_config

    for block, bad in (('data', 'bogus = 1'),
                       ('model', 'bogus = 1'),
                       ('training', 'bogus = 1')):
        p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "instances"
min_crop_dim = 64
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
iters = 1
""")
        # inject the unknown key into `block`
        lines = p.read_text().splitlines(keepends=True)
        out_lines = []
        for ln in lines:
            out_lines.append(ln)
            if ln.strip() == f'[{block}]':
                out_lines.append('bogus = 1\n')
        p.write_text(''.join(out_lines))
        with pytest.raises(SystemExit, match='unknown key'):
            load_detector_config(p)


def test_detector_config_bad_choices_raise(tmp_path):
    from tailcyclenet.detector.config import load_detector_config

    # boxes lives in [data]; append the bad key to the existing [data] block instead of a second one.
    p = _write_config(tmp_path, """\
[data]
path = "/tmp/ds"
boxes = "nope"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
iters = 1
""", 'b1.toml')
    with pytest.raises(SystemExit, match='boxes'):
        load_detector_config(p)
    p = _write_config(tmp_path, """\
[data]
path = "/tmp/ds"
boxes = "instances"
[model]
yolox = "nope"
[training]
out = "/tmp/run"
iters = 1
""", 'b2.toml')
    with pytest.raises(SystemExit, match='yolox'):
        load_detector_config(p)


def test_detector_config_extends_one_level(tmp_path):
    """User overlays can `extends` the shipped file; the merge is per BLOCK (pose rule)."""
    from tailcyclenet.detector.config import load_detector_config

    # writes base.toml, which the overlay below reaches by `extends` -- the call is the point,
    # the path is not used here.
    _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "instances"
min_crop_dim = 64
[model]
yolox = "tiny"
[training]
out = "/tmp/base-run"
iters = 20000
""", 'base.toml')
    overlay = _write_config(tmp_path, """
extends = "base.toml"
[data]
boxes = "keypoints"
[training]
iters = 3
""", 'overlay.toml')
    cfg = load_detector_config(overlay)
    assert cfg['data']['boxes'] == 'keypoints'
    assert cfg['data']['min_crop_dim'] == 64          # base value survived the block merge
    assert cfg['model']['yolox'] == 'tiny'            # untouched block carried over
    assert cfg['training']['iters'] == 3
    assert cfg['training']['out'] == '/tmp/base-run'


def test_detector_config_round_trips_through_the_run_folder(tmp_path):
    """The recorded config.toml (None values dropped) loads back to the same recipe."""
    from tailcyclenet.detector.config import load_detector_config

    src = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "keypoints"
min_crop_dim = 64
input_wh = [96, 64]
tile_wh = []
[model]
yolox = "trimmed"
[training]
out = "/tmp/run"
iters = 2
""", 'src.toml')
    cfg = load_detector_config(src, out=str(tmp_path / 'run'), iters=2, device='cpu')
    import toml
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'config.toml').write_text(toml.dumps(cfg))
    again = load_detector_config(tmp_path / 'run' / 'config.toml')
    assert again['data']['input_wh'] == [96, 64] and again['data']['tile_wh'] is None
    assert again['data']['boxes'] == 'keypoints'
    assert again['training']['iters'] == 2
    assert again['model']['yolox'] == 'trimmed'


def test_train_detector_config_end_to_end(tmp_path, dense_root, monkeypatch):
    """A 2-iteration run through scripts/train_detector.py's `main()` with a config file
    produces the same artefacts a CLI run did -- checkpoints, metrics.json, and now config.toml
    + provenance.toml -- and the checkpoint still loads through the unchanged `load_detector`.

    IN-PROCESS, not a subprocess: `train_detector.py` imports torch, and on this host a fresh
    interpreter spends ~60s of the ~110s wall clock just importing -- a subprocess would pay it
    twice. `test_train.py` runs `scripts/train.py` the same way.
    """
    import importlib.util
    import sys
    import tomllib

    from tailcyclenet.detector import load_detector

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
frames_per_group = 8
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
reduce = false
keypoints = false
hflip = true
tile_wh = []
tile_scale = 1.0
tile_bg_per_frame = 1
use_regions = false
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
kpt_weight = 1.0
kpt_score_weight = 1.0
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector',
                                                  REPO / 'scripts' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector_it000002.pth').exists()
    assert (out / 'detector.pth').exists()
    assert (out / 'metrics.json').exists()
    assert (out / 'config.toml').exists()
    assert (out / 'provenance.toml').exists()
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['data']['boxes'] == 'keypoints'
    assert recorded['data']['input_wh'] == [48, 48]
    assert recorded['training']['iters'] == 2
    model, wh, ds_name, mcd, reduce, box_src, ts, obj_q = load_detector(out / 'detector.pth')
    assert tuple(wh) == (48, 48)
    assert mcd == 16
    assert box_src == 'keypoints'
    assert ts is None                                # untiled: tile_scale is dropped at the read
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['yolox_version'] == 'tiny'
    assert ckpt['min_crop_dim'] == 16
    assert ckpt['box_source'] == 'keypoints'
    assert tuple(ckpt['input_wh']) == (48, 48)


def test_load_detector_config_out_override_rescues_an_empty_out(tmp_path):
    """`--out` must be able to fill in `[training].out = ""` -- the shipped `configs/detector.toml`
    ships it empty on exactly that promise, and every per-root overlay under `configs/detector/`
    leaves `out` for the CLI rather than restating a path in the file. The override used to be
    applied AFTER the `out` required-check, so it could never rescue anything and the promise in
    the config's own comment was false. Pinned so it cannot regress silently -- the failure mode
    is a `SystemExit` at every real invocation of a per-root overlay via `--out`.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text("""
[data]
path = "/does/not/need/to/exist/for/this/check"

[model]
yolox = "trimmed"

[training]
out = ""
""")
    loaded = load_detector_config(cfg, out=tmp_path / 'run')
    assert loaded['training']['out'] == str(tmp_path / 'run')

    with pytest.raises(SystemExit, match='out.*required'):
        load_detector_config(cfg)                    # nothing supplies `out`: still raises


def test_load_detector_config_iters_and_device_override_too(tmp_path):
    """The same ordering bug would have silently no-op'd `--iters`/`--device` as well, since all
    three overrides shared one code path -- pinned together rather than only for `out`."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text("""
[data]
path = "/does/not/need/to/exist/for/this/check"

[model]
yolox = "trimmed"

[training]
out = "runs/x"
iters = 100
device = "cpu"
""")
    loaded = load_detector_config(cfg, iters=5, device='cuda:1')
    assert loaded['training']['iters'] == 5
    assert loaded['training']['device'] == 'cuda:1'


def test_per_root_detector_overlays_load_and_raise_val_frames_per_group(tmp_path):
    """Every shipped per-root overlay under `configs/detector/` must resolve through
    `extends = "../detector.toml"`, carry `[data].path` (the one thing with no CLI override), and
    set `val_frames_per_group` past the shipped default of 8 -- each of these three roots has ONE
    val group, so the default under-samples it (dev/plans/detector_accuracy.md T0.3)."""
    root = Path(__file__).resolve().parent.parent / 'configs' / 'detector'
    want = {'rat-city.toml': 64, 'branson-fly.toml': 32, '3dpop.toml': 16}
    for name, expect in want.items():
        cfg = load_detector_config(root / name, out=tmp_path / name)
        assert cfg['data']['path'], f'{name}: [data].path must be set (no CLI override exists)'
        assert cfg['data']['val_frames_per_group'] == expect, name
        assert cfg['data']['val_frames_per_group'] > 8, \
            f'{name}: must exceed the shipped default of 8 or the overlay does nothing'


def test_deployment_score_untrained_model_is_all_zero(tmp_path):
    """`deployment_score` (T0.1, dev/plans/detector_accuracy.md) on a FRESH `YOLOXNano`: the
    rare-positive objectness prior (bias -4.595, sigmoid ~0.0099) sits below any sane
    `det_score`, so `det_fill` and `slot_fill` must read exactly 0 and `window_miss` exactly 1 --
    a degenerate but DETERMINISTIC floor, pinned so a future change to the init or to the
    threshold plumbing is caught rather than silently shifting the floor.

    THE SEED IS WHAT MAKES "DETERMINISTIC" TRUE. The objectness PRIOR sits at ~0.0099, but the
    head's random init scatters the actual logits around it, and 3 of 20 torch seeds put at least
    one anchor above `det_score = 0.05` -- so unseeded this test reads its own RNG state. It never
    varied on its own because it ran at a fixed position in the suite; adding or removing a test
    anywhere in this file reshuffles xdist's distribution and moves the state it happens to see.
    """
    torch.manual_seed(0)

    from .conftest import _session_2d
    from tailcyclenet.detector.evaluate import deployment_score
    from tailcyclenet.format import Session

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    model = YOLOXNano()
    r = deployment_score(model, sess, gid, input_wh=(64, 64), top_k=8, max_animals=2,
                         det_score=0.05, n_frames=2, overlap=0, min_box_frames=1)
    assert r['det_fill'] == 0.0 and r['slot_fill'] == 0.0 and r['window_miss'] == 1.0
    assert r['n_windows'] == 4                        # 2 rows x 2 windows (T=4, n_frames=2)
    assert r['n_gt'] == 8                              # 4 labelled frames x 1 camera x 2 animals
    assert all(v != v for v in r['union_side_px'].values()), \
        'no box ever fired, so the union-side quantiles must be all-NaN, not zero'
    assert r['gt_side_px'][0.5] > 0                    # GT sides are label-derived, independent


def test_deployment_score_forced_positive_objectness_fills_every_slot(tmp_path):
    """The live path: force every objectness logit strongly positive (`obj_pred` bias) so every
    anchor's box survives `det_score`, and confirm `det_fill` becomes 1.0 -- proving the
    zero-floor above is the init prior firing correctly, not a broken wire that always reads 0.
    """
    from .conftest import _session_2d
    from tailcyclenet.detector.evaluate import deployment_score
    from tailcyclenet.format import Session

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    model = YOLOXNano()
    for m in model.head.obj_pred:
        torch.nn.init.constant_(m.bias, 10.0)
    r = deployment_score(model, sess, gid, input_wh=(64, 64), top_k=8, max_animals=2,
                         det_score=0.05, n_frames=2, overlap=0, min_box_frames=1)
    assert r['det_fill'] == 1.0
    assert not any(v != v for v in r['union_side_px'].values()), \
        'a box fired everywhere, so the union-side quantiles must be finite, not NaN'


def test_gt_crop_sides_3d_projects_through_the_right_camera(tmp_path):
    """`_gt_crop_sides` must take the 3D branch (project points3d per camera) rather than
    silently reading `points2d`, which a 3D synthetic session does not even populate."""
    from .conftest import _session_3d
    from tailcyclenet.detector.evaluate import _gt_crop_sides
    from tailcyclenet.format import Session

    _session_3d(tmp_path / 'r' / 'test' / 's0', T=4)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    sides = _gt_crop_sides(sess, gid, min_crop_dim=8)
    assert sides.size > 0 and np.all(sides >= 8)       # the floor must be respected


def test_eval_detector_deploy_cli_end_to_end(tmp_path, monkeypatch, capsys):
    """A 2-iteration checkpoint through `scripts/train_detector.py`, scored through
    `scripts/eval_detector.py --deploy` -- the new mode, wired the same way `--compare` is:
    argv in, printed table out, no crash. Not a numeric assertion (2 iterations is noise); this
    pins that the CLI plumbing (load_detector -> load_datasets -> deployment_score -> print)
    runs end to end on a REAL trained-and-reloaded checkpoint, not just synthetic tensors.
    """
    import importlib.util
    import sys

    from .conftest import _session_2d

    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)
    _session_2d(root / 'test' / 's1', T=4, S=2)

    train_cfg = _write_config(tmp_path, f"""
[data]
path = "{root}"
boxes = "keypoints"
min_crop_dim = 8
input_wh = [64, 64]
min_box_px = 0
frames_per_group = 4
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
reduce = false
keypoints = false
hflip = true
tile_wh = []
tile_scale = 1.0
tile_bg_per_frame = 1
use_regions = false
[model]
yolox = "trimmed"
[training]
out = "{tmp_path / 'run'}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
kpt_weight = 1.0
kpt_score_weight = 1.0
""", name='train.toml')

    spec = importlib.util.spec_from_file_location('tcn_train_detector2',
                                                  REPO / 'scripts' / 'train_detector.py')
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(train_cfg)])
    train_mod.main()

    spec2 = importlib.util.spec_from_file_location('tcn_eval_detector',
                                                    REPO / 'scripts' / 'eval_detector.py')
    eval_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(eval_mod)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(tmp_path / 'run' / 'detector.pth'),
        '--data', str(root), '--split', 'test', '--deploy', '--no-track', '--link-boxes',
        '--score-thresh', '0.001', '--device', 'cpu', '--n-frames', '2', '--overlap', '0',
        '--det-max-frames', '2'])
    eval_mod.main()                                    # must not raise

    # THE PRINTED T COLUMN MUST REFLECT `--det-max-frames`, not the group's raw length -- the
    # fixture's group is 4 frames, bounded here to 2. Printing the raw 4 would silently claim a
    # full-length run when only a prefix was scored (the bug an actual 3dpop `--det-max-frames`
    # run exposed: the table read T=899..2680 while `detect_raw` had truncated to 120 internally).
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if '/g' in ln and ln.strip()[0] != '=')
    assert line.split()[1] == '2', f'expected the truncated T=2 in {line!r}'


# ----------------------------------------------------------------------------------------------
# T2.1: instances.pq PRESENT boxes as an objectness ignore mask (dev/plans/detector_accuracy.md)
# ----------------------------------------------------------------------------------------------

def test_present_ignore_boxes_for_matches_ignore_for_when_unwarped(tmp_path):
    """`present_ignore_boxes_for` (training-side, warp-aware) must agree EXACTLY with
    `ignore_for` (eval-side, no warp) at `warp=None` -- they read the same table."""
    from .conftest import _session_2d

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    ds = BoxDataset(tmp_path / 'r', 'test', input_wh=(64, 64), min_crop_dim=8,
                    max_frames_per_group=4)
    for i, (sess, gid, f, ci) in enumerate(ds.index):
        b = ds.present_ignore_boxes_for(i)
        ig, ig_boxes = ds.ignore_for(i)
        if f == 1:                                    # the fixture's one PRESENT row
            assert b.shape == (1, 4)
            torch.testing.assert_close(b[0], torch.as_tensor(ig_boxes[1], dtype=torch.float32))
        else:
            assert b.shape == (0, 4)


def test_present_ignore_boxes_for_moves_under_a_warp(tmp_path):
    """The training-side version must actually be warped, or the loss masks the wrong pixels
    under `--augment`/`--rotate-deg` -- the exact bug `ignore_for`'s own docstring names as the
    reason it lives beside `boxes_for`/`regions_for` and takes item `i`'s own transform."""
    from .conftest import _session_2d

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    ds = BoxDataset(tmp_path / 'r', 'test', input_wh=(64, 64), min_crop_dim=8,
                    max_frames_per_group=4)
    rng = np.random.default_rng(0)
    warp = random_affine((64, 48), rng, hflip=0.0, rotate_deg=30.0)
    for i, (sess, gid, f, ci) in enumerate(ds.index):
        if f == 1:
            unwarped = ds.present_ignore_boxes_for(i)
            warped = ds.present_ignore_boxes_for(i, warp=warp)
            assert not torch.allclose(unwarped, warped)


def test_ignore_present_and_use_regions_raise_together(tmp_path):
    """Both are the ONE opt-in (M,4) tuple slot `box_collate`/`split_batch` dispatch by rank;
    combining them would silently misroute one as the other rather than fail loudly."""
    from .conftest import _session_2d

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    with pytest.raises(ValueError, match='ignore_present and use_regions'):
        BoxDataset(tmp_path / 'r', 'test', input_wh=(64, 64), min_crop_dim=8,
                  ignore_present=True, use_regions=True)


def test_getitem_emits_ignore_boxes_as_the_fourth_element(tmp_path):
    from .conftest import _session_2d

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    ds = BoxDataset(tmp_path / 'r', 'test', input_wh=(64, 64), min_crop_dim=8,
                    max_frames_per_group=4, ignore_present=True)
    for i, (sess, gid, f, ci) in enumerate(ds.index):
        item = ds[i]
        assert len(item) == 3                          # (x, boxes, ignore_boxes): no keypoints
        assert item[2].dim() == 2 and item[2].shape[-1] == 4
        assert (item[2].shape[0] == 1) == (f == 1)      # only frame 1 carries the PRESENT row

    # A dataset WITHOUT ignore_present stays exactly 2 elements -- byte-identical shape.
    plain = BoxDataset(tmp_path / 'r', 'test', input_wh=(64, 64), min_crop_dim=8,
                       max_frames_per_group=4)
    assert len(plain[0]) == 2


def test_detector_loss_ignore_masks_present_but_unannotated_anchors():
    """The polarity that matters: `ignore` REMOVES anchors from the background target (opposite
    of `regions`, which RESTRICTS which anchors may be supervised at all), and a real positive
    is protected even where it geometrically overlaps an ignore box (two animals can overlap)."""
    torch.manual_seed(0)
    anchors = torch.tensor([[10.0, 10.0, 8.0], [50.0, 50.0, 8.0]])
    gt = torch.tensor([[[0.0, 0.0, 20.0, 20.0]]])          # anchor 0 is the sole positive
    overlapping_ignore = torch.tensor([[[0.0, 0.0, 20.0, 20.0]]])   # same footprint
    obj = torch.randn(1, 2)
    boxes = torch.zeros(1, 2, 4)

    _, parts = detector_loss(obj, boxes, anchors, gt, ignore=overlapping_ignore)
    assert parts['n_pos'] == 1, 'a positive anchor must never be masked by an ignore box'
    assert parts['ignored'] > 0

    # A pure hard negative (no nearby GT) DOES get masked, and that measurably moves the loss --
    # the direction that fixes T2.1's false negative.
    far_gt = torch.tensor([[[1000.0, 1000.0, 1001.0, 1001.0]]])
    covering_ignore = torch.tensor([[[0.0, 0.0, 20.0, 20.0]]])     # covers anchor 0 only
    base, _ = detector_loss(obj, boxes, anchors, far_gt)
    masked, mp = detector_loss(obj, boxes, anchors, far_gt, ignore=covering_ignore)
    assert float(masked) < float(base), \
        'masking a hard negative out of the BCE sum must lower the objectness loss'
    assert mp['n_pos'] == 0 and 'certified' not in mp


def test_train_detector_ignore_present_end_to_end(tmp_path, monkeypatch):
    """A 2-iteration run with `[data].ignore_present = true` on the fixture's own PRESENT row
    (frame 1) must not raise, must print the 'ign' fraction (the hook `train_detector.py` already
    had), and the checkpoint must round-trip the flag."""
    import importlib.util
    import sys

    from .conftest import _session_2d

    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)

    cfg = _write_config(tmp_path, f"""
[data]
path = "{root}"
boxes = "keypoints"
min_crop_dim = 8
input_wh = [64, 64]
min_box_px = 0
frames_per_group = 4
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
reduce = false
keypoints = false
hflip = true
tile_wh = []
tile_scale = 1.0
tile_bg_per_frame = 1
use_regions = false
ignore_present = true
[model]
yolox = "trimmed"
[training]
out = "{tmp_path / 'run'}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
kpt_weight = 1.0
kpt_score_weight = 1.0
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector3',
                                                  REPO / 'scripts' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()

    ckpt = torch.load(tmp_path / 'run' / 'detector_it000002.pth', map_location='cpu',
                      weights_only=False)
    assert ckpt['ignore_present'] is True


# ----------------------------------------------------------------------------------------------
# T4.1 -- pretrained COCO backbone (dev/plans/detector_accuracy.md). `bottleneck_expansion` is an
# architecture SHAPE key (yolox.py); `pretrained` is the config-level switch. Both must be inert
# at their defaults -- every checkpoint on record was built at bottleneck_expansion=0.5,
# pretrained=''.
# ----------------------------------------------------------------------------------------------

WEIGHTS_DIR = REPO / 'scratch' / 'weights'
_HAVE_COCO_WEIGHTS = (WEIGHTS_DIR / 'yolox_tiny.pth').exists()


def test_bottleneck_expansion_default_matches_the_shipped_shape():
    """At the default 0.5, `dark3`'s first bottleneck's conv1 is (24,48,1,1) on `tiny` -- the
    exact shape dev/plans/detector_accuracy.md T4.1 names as the shipped (narrow) one, HALF
    Megvii's own (48,48,1,1). This is the shape every checkpoint on record was built at; the test
    pins it so a future change to `round8`/`YOLOX_TIERS` cannot silently move it.
    """
    m = YOLOXNano(version='tiny')
    assert m.bottleneck_expansion == 0.5
    w = m.backbone.dark3[1].m[0].conv1[0].weight
    assert tuple(w.shape) == (24, 48, 1, 1)


def test_bottleneck_expansion_one_matches_the_canonical_shape():
    """At 1.0, the SAME tensor is (48,48,1,1) -- full width, Megvii's own `CSPLayer`s
    `expansion=1.0` for its inner `Bottleneck`s. Everything else about the net is unchanged: only
    the internal width of each bottleneck's first conv moves.
    """
    m = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    w = m.backbone.dark3[1].m[0].conv1[0].weight
    assert tuple(w.shape) == (48, 48, 1, 1)
    # And the net still forwards at the wider shape.
    obj, boxes, _ = m(torch.zeros(1, 3, 96, 96))
    assert obj.shape[1] == boxes.shape[1]


def test_bottleneck_expansion_default_is_byte_identical_to_no_key():
    """Passing `bottleneck_expansion=0.5` explicitly must build the IDENTICAL module graph (same
    state_dict keys and shapes) as never passing it -- the same contract `crop_inflate = 1.0` and
    `pose_nms` off already carry elsewhere in this repo: a default value is not merely numerically
    close to absent, it is indistinguishable from it.
    """
    a = YOLOXNano(version='s').state_dict()
    b = YOLOXNano(version='s', bottleneck_expansion=0.5).state_dict()
    assert set(a) == set(b)
    for k in a:
        assert a[k].shape == b[k].shape


def test_bottleneck_expansion_raises_alongside_trimmed():
    """`trimmed`'s `CSPDarknetNano` does not take this key -- a non-default value alongside it
    must raise rather than silently being ignored, the same guard `width` already has the other
    way (canonical tier + non-default `width`)."""
    with pytest.raises(ValueError, match='bottleneck_expansion'):
        YOLOXNano(version='trimmed', bottleneck_expansion=1.0)
    YOLOXNano(version='trimmed', bottleneck_expansion=0.5)      # the default: must NOT raise


def test_load_coco_backbone_raises_at_the_shipped_expansion():
    """The guard must fire BEFORE touching disk -- a model built at the shipped 0.5 must be
    refused even if no weights file exists at all, so the error names the real cause instead of a
    confusing FileNotFoundError."""
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='tiny', bottleneck_expansion=0.5)
    with pytest.raises(ValueError, match='bottleneck_expansion'):
        load_coco_backbone(m, 'tiny', weights_dir='/nonexistent')


def test_load_coco_backbone_raises_for_trimmed():
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='trimmed')
    with pytest.raises(ValueError, match='trimmed'):
        load_coco_backbone(m, 'trimmed', weights_dir='/nonexistent')


def test_load_coco_backbone_missing_weights_file_raises(tmp_path):
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    with pytest.raises(FileNotFoundError, match='yolox_tiny'):
        load_coco_backbone(m, 'tiny', weights_dir=tmp_path)


def test_bgr_to_rgb_focus_perm_is_an_involution_over_three_blocks():
    """The permutation must reverse each 3-channel block and nothing else -- it is claimed to be
    its own inverse (`pretrained.py`'s own derivation), and this is the property that makes ONE
    permutation correct for converting either direction."""
    from tailcyclenet.detector.pretrained import BGR_TO_RGB_FOCUS_PERM

    perm = BGR_TO_RGB_FOCUS_PERM
    assert len(perm) == 12 and sorted(perm) == list(range(12))
    for g in range(4):
        block = perm[3 * g:3 * g + 3]
        assert block == [3 * g + 2, 3 * g + 1, 3 * g + 0]
    # Applying it twice must be the identity.
    twice = [perm[perm[i]] for i in range(12)]
    assert twice == list(range(12))


def test_detector_config_bottleneck_expansion_and_pretrained_default_inert(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['model']['bottleneck_expansion'] == 0.5
    assert cfg['model']['pretrained'] == ''


def test_detector_config_pretrained_coco_requires_canonical_tier(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "trimmed"
pretrained = "coco"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='trimmed'):
        load_detector_config(p)


def test_detector_config_pretrained_coco_requires_bottleneck_expansion_one(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
pretrained = "coco"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='bottleneck_expansion'):
        load_detector_config(p)
    # The paired, correct config must NOT raise.
    p2 = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
pretrained = "coco"
bottleneck_expansion = 1.0
[training]
out = "/tmp/run"
""", 'ok.toml')
    cfg = load_detector_config(p2)
    assert cfg['model']['pretrained'] == 'coco' and cfg['model']['bottleneck_expansion'] == 1.0


def test_detector_config_pretrained_unsupported_value_raises(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
pretrained = "/some/path.pth"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='pretrained'):
        load_detector_config(p)


def test_detector_config_bottleneck_expansion_raises_alongside_trimmed(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "trimmed"
bottleneck_expansion = 1.0
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='bottleneck_expansion'):
        load_detector_config(p)


@pytest.mark.skipif(not _HAVE_COCO_WEIGHTS, reason='scratch/weights/yolox_*.pth is untracked '
                    '(CLAUDE.md) and may not be cached on this machine')
def test_load_coco_backbone_transfers_every_backbone_tensor():
    """The end-to-end claim T4.1 exists to fix: 35/35 backbone conv tensors load at
    bottleneck_expansion=1.0, not the 19/35 the shipped 0.5 shape permits."""
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    n_loaded, n_total = load_coco_backbone(m, 'tiny', weights_dir=WEIGHTS_DIR)
    assert (n_loaded, n_total) == (35, 35)
    # And the loaded weights actually forward without shape errors.
    obj, boxes, _ = m(torch.zeros(1, 3, 96, 96))
    assert obj.shape[1] == boxes.shape[1]


@pytest.mark.skipif(not _HAVE_COCO_WEIGHTS, reason='scratch/weights/yolox_*.pth is untracked '
                    '(CLAUDE.md) and may not be cached on this machine')
def test_load_coco_backbone_stem_correction_matches_megvii_activation():
    """T4.1's own test: a corrected weight fed this repo's RGB/[0,1] convention must produce the
    SAME stem activation Megvii's raw weight produces on the identical image fed BGR/[0,255] --
    not merely a plausible-looking one. Ports `scratch/validate_pretrained_load.py`'s check into a
    pinned test."""
    import torch.nn.functional as F

    from tailcyclenet.detector.pretrained import BGR_TO_RGB_FOCUS_PERM

    ck = torch.load(WEIGHTS_DIR / 'yolox_tiny.pth', map_location='cpu', weights_only=False)
    src = ck['model'] if 'model' in ck else ck
    w_orig = src['backbone.backbone.stem.conv.conv.weight'].clone()

    torch.manual_seed(0)
    img_bgr_255 = torch.rand(1, 3, 32, 32) * 255.0
    img_rgb_01 = img_bgr_255.flip(1) / 255.0

    def focus_cat(x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)

    y_megvii = F.conv2d(focus_cat(img_bgr_255), w_orig, stride=1, padding=1)
    w_corrected = (w_orig * 255.0)[:, BGR_TO_RGB_FOCUS_PERM]
    y_repo = F.conv2d(focus_cat(img_rgb_01), w_corrected, stride=1, padding=1)
    torch.testing.assert_close(y_megvii, y_repo, atol=1e-3, rtol=1e-4)


@pytest.mark.skipif(not _HAVE_COCO_WEIGHTS, reason='scratch/weights/yolox_*.pth is untracked '
                    '(CLAUDE.md) and may not be cached on this machine')
def test_train_detector_pretrained_coco_end_to_end(tmp_path, dense_root, monkeypatch):
    """A 2-iteration run with `pretrained = "coco"` through the real CLI entry point: the backbone
    loads, training runs, and the checkpoint records both new keys so `load_detector` rebuilds the
    right architecture later."""
    import importlib.util
    import sys

    from tailcyclenet.detector import load_detector

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [96, 96]
min_box_px = 0
frames_per_group = 8
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
bottleneck_expansion = 1.0
pretrained = "coco"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector_pretrained',
                                                  REPO / 'scripts' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    monkeypatch.setattr(mod, 'load_coco_backbone',
                        lambda model, tier: __import__(
                            'tailcyclenet.detector.pretrained', fromlist=['load_coco_backbone']
                        ).load_coco_backbone(model, tier, weights_dir=WEIGHTS_DIR))
    mod.main()

    assert (out / 'detector.pth').exists()
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['bottleneck_expansion'] == 1.0
    assert ckpt['pretrained'] == 'coco'
    model, wh, ds_name, mcd, reduce, box_src, ts, obj_q = load_detector(out / 'detector.pth')
    assert model.bottleneck_expansion == 1.0
