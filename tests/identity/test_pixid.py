"""identity.crops / identity.pixid / identity.maskhead on synthetic sessions (CPU, no video, no checkpoints)."""
import numpy as np
import pandas as pd
import pytest
import torch

from tailcyclenet.identity import crops as CR
from tailcyclenet.identity import pixid as PX
from tailcyclenet.identity.maskhead import NPARAM, MaskHead, load_maskhead

H, W = 120, 160


def _image_source(tmp_path, frames, gid='g0'):
    """A source session whose group is an image directory of `frames` (T,H,W) uint8."""
    import cv2
    cam = tmp_path / 'src' / 'groups' / gid / 'cam0'
    cam.mkdir(parents=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(cam / f'{i:06d}.png'), f)
    return tmp_path / 'src'


def _fake_session(monkeypatch, tmp_path, pred, boxes, names, src, gid='g0'):
    """Route `crops._session` to in-memory predictions (the prediction-session reader is tested elsewhere)."""
    d = dict(group_id=gid, pred=pred, boxes=boxes, animal_ids=np.array([f'det{i:02d}' for i in range(len(pred))], object))
    monkeypatch.setattr(CR, '_session', lambda pred_dir, source, group, stop=None: (d, {'names': names}, src))


def _mouse_frame(cx, cy, ang, L=40):
    """A bright bar (body) from tail base to nose along `ang`, a bright blob at the nose."""
    import cv2
    img = np.zeros((H, W), np.uint8)
    u = np.array([np.cos(ang), np.sin(ang)])
    nose, tail = np.array([cx, cy]) + u * L / 2, np.array([cx, cy]) - u * L / 2
    cv2.line(img, tuple(int(v) for v in tail), tuple(int(v) for v in nose), 120, 6)
    cv2.circle(img, tuple(int(v) for v in nose), 6, 255, -1)
    return img, nose, tail


def test_keypoint_crops_put_the_head_up_and_carry_the_angle(tmp_path, monkeypatch):
    """The crop is centred on the keypoint mean and rotated head-up whatever the body angle; a frame
    whose axis is undefined keeps the previous angle; too few keypoints -> not ok."""
    T, angs = 4, [0.0, 1.2, -2.0, -2.0]
    frames, pred = [], np.full((1, T, 3, 2), np.nan, np.float32)
    for t, a in enumerate(angs):
        img, nose, tail = _mouse_frame(80, 60, a)
        frames.append(img)
        pred[0, t] = [nose, (nose + tail) / 2, tail]
    pred[0, 2, 0] = np.nan              # no head keypoint: angle carried from frame 1
    pred[0, 3, :2] = np.nan             # 1 of 3 keypoints: not ok
    src = _image_source(tmp_path, frames)
    _fake_session(monkeypatch, tmp_path, pred, None, ['nose', 'centre', 'tail_base'], src)
    c = CR.keypoint_crops('unused', out_px=32, side=1.6)
    assert c['ok'].tolist() == [[True, True, True, False]]
    assert c['bl'] == pytest.approx(40 * 1.25, rel=1e-4)
    np.testing.assert_allclose(c['ang'][0, :3], [0.0, 1.2, 1.2], atol=1e-5)
    for t in (0, 1):
        x = c['crops'][0, t].astype(float)
        assert x[:12].max() == 255 and x[-12:].max() < 255          # the nose blob is in the top band
        assert x[:, 16].sum() > 4 * x[:, 3].sum()                    # the body is vertical, centred
    assert not c['crops'][0, 3].any()


def test_box_crops_are_unrotated_and_sized_by_the_median_box(tmp_path, monkeypatch):
    """field = -1 -> side = median box long side; the crop is the box centre; ok needs a finite box
    and a finite centroid; body length is the instances.pq box median on the dense grid (K = 1)."""
    S, T = 2, 3
    frames = np.zeros((T, H, W), np.uint8)
    frames[:, 20:40, 30:50] = 200                                   # row 0's box content
    boxes = np.full((S, T, 1, 4), np.nan, np.float32)
    boxes[0, :, 0] = [30, 20, 50, 40]
    boxes[1, :2, 0] = [100, 60, 130, 90]
    pred = np.full((S, T, 1, 2), np.nan, np.float32)
    pred[0, :, 0] = [40, 30]
    pred[1, 0, 0] = [115, 75]
    pd_dir = tmp_path / 'pred'
    pd_dir.mkdir()
    rows = [dict(group_id='g0', frame=t, animal_id=f'det{s:02d}', x0=b[0], y0=b[1], x1=b[2], y1=b[3])
            for s in range(S) for t in range(T) for b in [boxes[s, t, 0]] if np.isfinite(b).all()]
    pd.DataFrame(rows).to_parquet(pd_dir / 'instances.pq')
    pd.DataFrame([dict(group_id='g0', n_frames=T)]).to_parquet(pd_dir / 'groups.pq')
    pd.DataFrame([dict(group_id='g0', frame=0, animal_id='det00', bodypart='centroid', x=40.0, y=30.0)]).to_parquet(pd_dir / 'points2d.pq')
    _fake_session(monkeypatch, tmp_path, pred, boxes, ['centroid'], _image_source(tmp_path, frames))
    c = CR.box_crops(pd_dir, field=-1.0, out_px=16)
    assert c['ok'].tolist() == [[True, True, True], [True, False, False]]
    assert c['bl'] == np.float32(20.0)                                # median of 20, 20, 20, 30, 30
    assert CR.box_body_length(pd_dir, 'g0', stop=1) == 25.0             # frame 0 only: 20, 30
    assert (c['crops'][0] > 150).mean() > 0.8                        # 20 px box in a 20 px field
    assert c['crops'][1, 1].max() == 0                               # box present, centroid missing: cropped, not ok


def test_crops_roundtrip(tmp_path):
    c = dict(gid='g0', crops=np.zeros((2, 3, 8, 8), np.uint8), ok=np.ones((2, 3), bool),
             ctr=np.zeros((2, 3, 2), np.float32), bl=np.float32(12.5), ids=np.array(['a', 'b']), names=np.array(['centroid']))
    CR.save_crops(tmp_path / 'c.npz', c)
    r = CR.load_crops(tmp_path / 'c.npz')
    assert r['gid'] == 'g0' and r['bl'] == 12.5 and r['crops'].shape == (2, 3, 8, 8) and list(r['ids']) == ['a', 'b']


def test_moment_geometry_points_along_the_heavy_end():
    """The axis angle is the mask's long axis, signed toward positive skew: away from the heavy end,
    along the long thin side."""
    m = np.zeros((60, 80), bool)
    m[28:33, 10:70] = True
    m[24:37, 55:70] = True                                           # heavy at +x
    cx, cy, th = CR._moment_geom(m)
    assert abs(np.cos(th) + 1) < 1e-3 and 40 < cx < 60 and abs(cy - 30) < 1
    cx2, cy2, th2 = CR._moment_geom(m[:, ::-1])
    assert abs(np.cos(th2) - 1) < 1e-3
    assert CR._moment_geom(np.zeros((5, 5), bool)) is None


def test_mask_geoms_take_the_exclusive_region_in_source_px():
    """Two overlapping soft masks on a stride grid: each geometry is its own argmax region, mapped
    through the inverse letterbox into source px."""
    qs = np.zeros((2, 40, 40), np.float32)
    qs[0, 10:20, 5:15] = 0.9
    qs[1, 10:20, 12:30] = 0.8
    Minv = np.array([[2.0, 0, 0], [0, 2.0, 0]])                      # input px -> source px (x2)
    g = CR._mask_geoms(list(qs), Minv, 200, 200)
    assert g[0][0] == pytest.approx(2 * 10, abs=1.5) and g[1][0] == pytest.approx(2 * 22.5, abs=1.5)
    assert g[0][1] == pytest.approx(2 * 15, abs=1.5)


def test_maskhead_roundtrip_and_dynamic_shapes(tmp_path):
    torch.manual_seed(0)
    mh = MaskHead(cin=16, mid=8, ctrl=16)
    torch.save({'state': mh.state_dict(), 'arch': dict(cin=16, mid=8, ctrl=16)}, tmp_path / 'mh.pth')
    mh2 = load_maskhead(tmp_path / 'mh.pth')
    feats = [torch.randn(1, 16, 16, 16), torch.randn(1, 16, 8, 8), torch.randn(1, 16, 4, 4)]
    mf, prm = mh2.mask_feats(feats), mh2.params(feats)
    assert mf.shape == (1, 8, 16, 16) and prm.shape == (1, 16 * 16 + 8 * 8 + 4 * 4, NPARAM)
    with torch.no_grad():
        logits = MaskHead.dynamic(mf[0], prm[0, :3], torch.tensor([[10., 10.], [30., 20.], [5., 50.]]), torch.tensor([4., 20., 9.]))
        assert logits.shape == (3, 16, 16)
        assert torch.equal(mh.mask_feats(feats), mf)


def test_tracklets_cut_at_gaps_jumps_and_contacts():
    ctr = np.zeros((2, 10, 2))
    ctr[0, :, 0] = np.arange(10)
    ctr[1, :, 0] = 100.0
    ctr[0, 6:, 0] += 50                                              # jump of 51 at t = 6
    ctr[1, 3, 0] = ctr[0, 3, 0] + 1                                  # contact at t = 3
    ok = np.ones((2, 10), bool)
    ok[1, 8] = False
    tid, info, d, sep = PX.tracklets(ctr, ok, bl=10.0)
    assert info == [(0, 0, 2), (0, 4, 5), (0, 6, 9), (1, 0, 2), (1, 4, 7), (1, 9, 9)]
    assert tid[0, 3] == -1 and tid[1, 3] == -1
    segs = PX.global_segments(tid, 2)
    assert segs[0] == (0, 2, (0, 3)) and (4, 5, (1, 4)) in segs


def _two_animal_crops(T=240, opx=16, seed=0):
    """Two textures (horizontal vs vertical stripes) on rows that swap animal at every contact."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:opx, :opx]
    tex = [(np.sin(yy * 1.3) > 0) * 160 + 40, (np.sin(xx * 1.3) > 0) * 160 + 40]
    S = 2
    animal = np.zeros((S, T), int)
    animal[1] = 1
    contacts = list(range(40, T, 50))
    for t0 in contacts:                                              # rows swap animals at each contact
        animal[:, t0:] = animal[::-1, t0:]
    crops = np.zeros((S, T, opx, opx), np.uint8)
    for s in range(S):
        for t in range(T):
            crops[s, t] = np.clip(tex[animal[s, t]] + rng.normal(0, 20, (opx, opx)), 0, 255)
    ctr = np.zeros((S, T, 2))
    ctr[0, :, 0], ctr[1, :, 0] = 0.0, 100.0
    for t0 in contacts:
        ctr[:, t0 - 1:t0 + 2, 0] = 50.0                              # rows meet: tracklets cut
    return dict(gid='g0', crops=crops, ok=np.ones((S, T), bool), ctr=ctr, bl=40.0), animal


def test_train_pixid_recovers_identity_across_tracklets_and_leaves_rng_alone():
    c, animal = _two_animal_crops()
    state = torch.get_rng_state()
    kw = dict(device='cpu', epochs=2, cap=200, log=lambda *a: None)
    r = PX.train_pixid(c, **kw)
    assert torch.equal(state, torch.get_rng_state())
    assert r['probs'].shape == (1, 2, 240, 2) and r['probs'].dtype == np.float16 and r['N'] == 2
    lab = r['probs'][0].astype(np.float32).argmax(-1)
    m = r['tid'] >= 0
    agree = (lab[m] == animal[m]).mean()
    assert max(agree, 1 - agree) > 0.95 and r['uniq'] > 0.9
    assert (r['lab'] >= 0).sum() > 2                                  # accumulation went past the first segment
    r2 = PX.train_pixid(c, **kw)
    assert np.array_equal(r['probs'], r2['probs'], equal_nan=True)   # CPU: a run is a function of seed


def test_train_pixid_without_a_global_segment_fails_by_name(tmp_path):
    c, _ = _two_animal_crops(T=40)
    c['ok'][1] = False
    c['ok'][1, :2] = True
    c['ok'][0, :2] = False                                            # never two rows at once
    r = PX.train_pixid(c, device='cpu', N=2, log=lambda *a: None)
    assert r == {'fail': True}
    PX.save_pixid(tmp_path / 'f.npz', r)
    assert bool(np.load(tmp_path / 'f.npz')['fail'])
