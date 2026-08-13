"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
import numpy as np
import pytest
import torch

from tailcyclenet.crop import crop_box_for_points
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, YOLOXNano, assign, box_collate,
                                   box_iou, decode, detector_loss, giou_loss, letterbox,
                                   unletterbox_boxes)


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

    # A SECOND DETECTION OF AN ANIMAL ALREADY IN THE OUTPUT IS NOT NEW COVERAGE. The same box
    # again, in a camera that already contributed to the triangulated instance, reprojects on top of
    # it -- `fp_dup`, not a found animal -- and `dup_res_px` is the gate on that.
    dup_cam = [centre, centre, torch.cat([centre, centre])]
    assert len(associate(cams, dup_cam, max_res_px=30.0, min_views=1)) == len(two) + 1
    assert len(associate(cams, dup_cam, max_res_px=30.0, min_views=1,
                         dup_res_px=30.0)) == len(two)
    # ...and the gate must not swallow the corner box, which is a different place.
    assert len(associate(cams, per_cam, max_res_px=30.0, min_views=1,
                         dup_res_px=30.0)) == len(two) + 1


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

    from tailcyclenet.detector import box_iou, link_rows

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


def test_the_tracker_is_the_default_and_can_be_turned_off():
    """`--track` is ON by default (dev/reports/13), so `detect_group` must default to it too.

    Pinned because the default is what every future arm inherits, and because the ONE thing that
    makes flipping it safe is that `track` became unconditional in `--det-cache`'s stamp: a cache
    written while it was off carries no `track` entry and must now be REFUSED rather than reused as
    if it had been tracked. That is the same guard `det_score` needed when its default moved.
    """
    import inspect
    from pathlib import Path

    from tailcyclenet.detector import detect_group

    sig = inspect.signature(detect_group)
    assert sig.parameters['track'].default is True, 'the tracker is the default'
    assert sig.parameters['link'].default is False, '--link-boxes is still opt-in'

    src = (Path(__file__).resolve().parent.parent / 'scripts' / 'infer.py').read_text()
    assert "('track', str(args.track))" in src, \
        'track must be UNCONDITIONAL in the cache stamp now that its default has moved'
    assert 'BooleanOptionalAction' in src, '--no-track must exist to restore the old behaviour'


def test_a_cache_without_keypoints_cannot_serve_the_keypoint_crop_source():
    """A cache hit must not silently turn `--crop-source keypoints` into `--crop-source boxes`.

    `run_group` switches on `det_kpts_stc is not None`, so a cache that holds only boxes does not
    error -- it crops from the boxes and reports the arm under the other arm's name, which is the
    one comparison item 3 exists to make. Two halves: keypoints are STORED under their own key, and
    a cache lacking that key is REFUSED for this crop source rather than served.

    A source check, like the `track` stamp test above and for the same reason: the guard is inside
    `main`'s group loop, past `load_run`, so reaching it needs a trained detector and a checkpoint.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / 'scripts' / 'infer.py').read_text()
    assert "det_cache[f'{key}|kpt'] = det_kpts" in src, \
        'detector keypoints must be cached, or the two crop sources cannot share one box set'
    assert "args.crop_source == 'keypoints' and det_kpts is None" in src, \
        'a keypoint-free cache must be refused for --crop-source keypoints, not silently accepted'
    assert '"|score", "|kpt"' in src, 'the cached-group count must not count the keypoint entries'


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
