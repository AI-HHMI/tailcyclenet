"""tailcyclenet.identity: the gate's frozen constants, the empty-calibration fallback and the F2C0 solve on a
synthetic two-animal swap (synthetic only; equivalence to the research outputs is checked in dev/)."""

import hashlib

import numpy as np
import pytest

from tailcyclenet.identity import gate as GT, graph as G, witness as W

DRAWS_SHA256 = '8184087c2387cfae1adfa11f8b524d883870f001b34725e675e06c59c642d305'


def test_random_draws_frozen():
    assert hashlib.sha256(GT.DRAWS.read_bytes()).hexdigest() == DRAWS_SHA256


def test_bar_bounds_and_monotone():
    b = [GT.bar(n, 2) for n in (10, 30, 100)]
    assert all(0 < x <= 1 for x in b) and b[0] >= b[1] >= b[2]
    assert GT.bar(50, 4) != GT.bar(50, 2)  # the two animal-count groups use different draws


def _swap_video(S=2, T=200, cut=100, K=3):
    """two animals on parallel tracks; the base rows swap at ``cut`` (a row-assignment error), the pixel column is
    right everywhere."""
    t = np.arange(T, dtype=float)
    A = np.stack([t, np.full(T, 0.0)], -1)
    B = np.stack([t, np.full(T, 40.0)], -1)
    off = np.array([[0, 0], [4, 0], [0, 4]], float)
    rows = np.stack([np.where(t[:, None] < cut, A, B), np.where(t[:, None] < cut, B, A)])  # (S, T, 2)
    pred = rows[:, :, None, :] + off[None, None]
    cls = np.stack([np.where(t < cut, 0, 1), np.where(t < cut, 1, 0)])  # true animal per row-frame
    probs = np.where(np.arange(S)[None, None] == cls[..., None], 0.95, 0.05)[None].astype(np.float16)
    info = np.array([[0, 0, cut - 1], [0, cut, T - 1], [1, 0, cut - 1], [1, cut, T - 1]], np.int64)
    return pred, dict(probs=probs, info=info, N=S), cls


def test_solve_f2c0_undoes_a_row_swap():
    pred, pix, cls = _swap_video()
    S, T = pred.shape[:2]
    F, Wf = G.pix_frames(pix['probs'], S, T)
    valid = np.isfinite(pred[..., 0]).all(-1)
    lab, dec = G.solve_f2c0(pix['info'], F, Wf, G.row_positions(pred), 10.0, valid, 2, anchor=True)
    assert dec['anchored'] and np.array_equal(lab, cls)
    out = G.relabel(pred, lab)
    assert np.allclose(out[0, :, 0, 1], 0.0) and np.allclose(out[1, :, 0, 1], 40.0)  # each output row = one animal


def test_relabel_drops_invalid_and_refuses_conflicts():
    pred = np.zeros((2, 3, 1, 2))
    lab = np.array([[0, 0, -1], [1, 1, 1]])
    assert np.isnan(G.relabel(pred, lab)[0, 2]).all()
    with pytest.raises(AssertionError):
        G.relabel(pred, np.array([[0, 0, 0], [0, 1, 1]]))


def test_empty_link_table_is_C0_false_and_gate_off():
    """no node reaches NMIN pixel frames -> no candidate link, no self-calibration pair -> C0 False -> OFF (the
    research run crashed here; F60S1MGPBAGZJ448L7VS)."""
    pred, pix, _ = _swap_video()
    pix = dict(pix, probs=np.full_like(pix['probs'], np.nan))
    r = GT.run_identity(pred, pix, [], [], 10.0)
    assert r['inputs']['C0'] is False and r['inputs']['n_links'] == 0 and not r['on'] and r['labels'] is None


def test_empty_selfcal_is_C0_false():
    pred, pix, _ = _swap_video()
    S, T = pred.shape[:2]
    F, Wf = G.pix_frames(pix['probs'], S, T)
    info = pix['info']
    P, n = G.node_agg(F, Wf, info)
    Ec = G.build_edges(info, n >= G.NMIN)
    out = W.c0(F, Wf, info, np.zeros(len(info), bool), n, Ec, P, T)  # no informative node -> no self-cal pair
    assert out['n_links'] > 0 and out['n_selfcal'] == 0 and out['C0'] is False


def test_no_pixel_column_is_off():
    pred, pix, _ = _swap_video()
    assert not GT.run_identity(pred, None, [], [], 10.0)['on']
    assert GT.run_identity(pred, dict(pix, N=3), [], [], 10.0)['inputs']['note'] == 'no pixel column'


def _tt(info, same):
    """a tt edge table on the swap video's tracklets: every pair, tt says same iff ``same``."""
    E = W.tt_pairs(info)
    lo = np.where([same(int(a), int(b)) for a, b in E], 4.0, -4.0)
    return E, lo, np.ones(len(E), bool)


def test_gate_v7_arm_and_C0_veto():
    pred, pix, cls = _swap_video()
    S, T = pred.shape[:2]
    F, Wf = G.pix_frames(pix['probs'], S, T)
    info = pix['info']
    P, n = G.node_agg(F, Wf, info)
    inf = n >= G.NMIN
    node_cls = cls[info[:, 0], info[:, 1]]
    E, lo, ok = _tt(info, lambda a, b: node_cls[a] == node_cls[b])
    E = np.repeat(E, 10, 0)
    lo = np.repeat(lo, 10)
    ok = np.repeat(ok, 10)  # >= 10 tt-SAME links
    z = dict(info=info, E=E, lo=lo, ok=ok, P=P, n=n, Ec=G.build_edges(info, inf), inf=inf, N=2, T=T)
    g = GT.gate_v8(z, True)
    assert g['on'] and g['v7'] and g['aS'] == 1.0 and g['p_null'] is None
    assert not GT.gate_v8(z, False)['on']


def test_tt_edges_same_and_birth_abstain():
    info = np.array([[0, 0, 49], [0, 60, 99], [1, 60, 99]], np.int64)
    L = np.full((2, 100), -1, np.int32)
    W_ = np.ones((2, 100), np.float32)
    L[0, :50] = 7
    L[0, 60:] = 7
    L[1, 60:] = 8
    E, lo, ok = W.tt_edges(info, L, W_, [('birth', 8, 55)])
    d = {tuple(e): (v, bool(o)) for e, v, o in zip(E.tolist(), lo, ok)}
    assert d[(0, 1)] == (4.0, True)  # same carried id, full share -> +(2 + 2)
    assert d[(0, 2)] == (0.0, False)  # id 8 born after node 0 ended -> abstain


def test_carry_labels_reads_points_inside_a_row():
    pred = np.zeros((2, 1, 4, 2))
    pred[0, 0] = [[0, 0], [10, 0], [10, 10], [0, 10]]
    pred[1, 0] = pred[0, 0] + 100
    pts = np.array([[[2, 2], [5, 5], [8, 3]]], float)
    rec = dict(ws=0, nfr=1, ids={3: dict(tr=pts, vis=np.ones((1, 3)))})
    L, Wt = W.carry_labels([rec], pred, 10.0)
    assert L[0, 0] == 3 and L[1, 0] == -1 and Wt[0, 0] == 1.0


def test_platt_single_class_falls_back_to_sigmoid():
    f = W.platt(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert np.allclose(f(np.array([0.0, 20.0])), [0.5, 1 / (1 + np.exp(-12))])


def test_body_length_notes_win():
    pred = np.zeros((1, 2, 2, 2))
    pred[0, :, 1] = [3, 4]
    assert W.body_length(pred) == 5.0
    assert W.body_length(pred, 'body_length_px=146.00; crossing_frac=0.09') == 146.0
    assert W.body_length(pred, 'species=mouse; box_px=65; boxes=blob') == 65.0
