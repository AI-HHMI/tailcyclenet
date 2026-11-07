"""identity.crops / identity.pixid on synthetic sessions (CPU, no video, no checkpoints)."""
import numpy as np
import pandas as pd
import torch

from tailcyclenet.identity import crops as CR
from tailcyclenet.identity import pixid as PX

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
    """Two rotation-distinct textures (fine vs coarse concentric rings; the network is rotation-blind) on rows
    that swap animal at every contact."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:opx, :opx]
    rr = np.hypot(yy - (opx - 1) / 2, xx - (opx - 1) / 2)
    tex = [(np.sin(rr * 2.2) > 0) * 160 + 40, (np.sin(rr * 0.7) > 0) * 160 + 40]
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


def test_training_rotation_is_uniform_over_the_full_circle(monkeypatch):
    """`prep(aug=True)` draws one uniform angle in [0, 2 pi) per crop; inputs are masked to the inscribed circle."""
    seen = []
    real = PX.rotate
    monkeypatch.setattr(PX, 'rotate', lambda x, th: seen.append(th.clone()) or real(x, th))
    circ = PX.circle_mask(16, torch.device('cpu'))
    torch.manual_seed(0)
    x = PX.prep(torch.full((4096, 1, 16, 16), 200, dtype=torch.uint8), circ, aug=True)
    th = seen[0]
    assert th.shape == (4096,) and th.min() >= 0 and th.max() < 2 * np.pi
    assert th.min() < 0.05 and th.max() > 2 * np.pi - 0.05 and abs(th.mean().item() - np.pi) < 0.1
    assert circ[0, 0] == 0 and circ[8, 8] == 1 and x.shape == (4096, 1, 16, 16)
    seen.clear()
    PX.prep(torch.zeros((2, 1, 16, 16), dtype=torch.uint8), circ)          # no aug, th0 = 0: no rotation at all
    assert seen == []


def test_tta_softmax_is_the_mean_softmax_over_8_rotations():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(256, 3)).eval()
    circ = PX.circle_mask(16, torch.device('cpu'))
    xb = torch.randint(0, 255, (5, 1, 16, 16), dtype=torch.uint8)
    got = PX.tta_softmax(net, xb, circ)
    with torch.no_grad():
        want = torch.stack([torch.softmax(net(PX.prep(xb, circ, th0=2 * np.pi * r / 8)), -1) for r in range(8)]).mean(0)
        plain = torch.softmax(net(PX.prep(xb, circ)), -1)
    assert PX.TTA_ROT == 8 and torch.allclose(got, want, atol=1e-6) and got.shape == (5, 3)
    assert not torch.allclose(got, plain, atol=1e-4)                       # it is not a single forward
    assert torch.equal(got, PX.tta_softmax(net, xb, circ))                 # deterministic on CPU
