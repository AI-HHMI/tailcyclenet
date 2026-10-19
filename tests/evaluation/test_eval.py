"""The offline evaluator (`evaluation/eval.py`): chunking, ranges, aggregation, the CLI."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import tailcyclenet.evaluation.eval as ev
from tailcyclenet import format as fmt
from tailcyclenet.format import Labels, Session
from tests.helpers import _session_2d, _session_3d_multi


def test_chunk_frames_scores_a_trailing_one_frame_remainder():
    """`--chunk 4` on a 9-frame group scores all 9 frames: the one-frame tail joins the last
    chunk instead of being dropped (it used to vanish, labels and all)."""
    import dataclasses

    from tailcyclenet.evaluation.eval import chunk_frames

    @dataclasses.dataclass
    class Lab:
        """The two label fields `chunk_frames` reads."""
        points3d: np.ndarray
        points2d: np.ndarray

    T = 9
    true = np.random.default_rng(0).random((1, T, 3, 2)).astype(np.float32)
    lab = Lab(points3d=None, points2d=true[..., None, :])
    preds = {'s/g': {'mode': '2d', 'pred': true.copy()}}
    p, lb = chunk_frames(preds, {'s/g': (lab, None)}, 4)
    assert sorted(p) == ['s/g#0', 's/g#4']
    assert [p[k]['pred'].shape[1] for k in ('s/g#0', 's/g#4')] == [4, 5]
    assert lb['s/g#4'][0].points2d.shape[1] == 5
    p8, _ = chunk_frames({'s/g': {'mode': '2d', 'pred': true[:, :8].copy()}},
                         {'s/g': (Lab(None, true[:, :8, :, None]), None)}, 4)
    assert [p8[k]['pred'].shape[1] for k in sorted(p8)] == [4, 4]


def _ranged_prediction(tmp_path, provenance):
    """A 2D session (T=8) and a prediction equal to its labels on frames [2, 6), NaN elsewhere."""

    from tailcyclenet.format import load_dataset
    from tailcyclenet.infer.session_writer import SessionWriter

    root = tmp_path / 'data'
    _session_2d(root / 'test' / 's', T=8)
    sess = load_dataset(root).sessions['test'][0]
    gid = 'g000'
    lab = sess.labels(gid)
    S, T, K, C = lab.points2d.shape[0], 8, sess.n_keypoints, len(sess.cam_names)
    pred2d = np.full((S, T, C, K, 2), np.nan, np.float32)
    pred2d[:, 2:6] = np.moveaxis(lab.points2d[:, 2:6], 3, 2)
    blk = {
        'animal_ids': np.asarray(lab.animal_ids, object),
        'pred': pred2d[:, :, 0].copy(),
        'conf': np.ones((S, T, K), np.float32),
        'pred2d': pred2d,
        'conf2d': np.ones((S, T, C, K), np.float32),
        'model_conf2d': np.zeros((S, T, C, K), np.float32),
        'box_agree': np.full((S, T, C), np.nan, np.float32),
        'outcome': np.zeros((S, 1), np.int8),
        'crop': np.zeros((S, 1, C, 4), np.float32),
        'outcome_names': ['ok'],
        'window_start': np.array([2], np.int32),
        'window_stop': np.array([6], np.int32),
        'owner_window': np.zeros(T, np.int32),
    }
    out = tmp_path / 'prediction'
    writer = SessionWriter(out, sess, type('Registry', (), {'names': sess.names})(),
                           {'source_session_id': sess.session_id, 'n_frames': 4, **provenance},
                           [gid])
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)
    return root, out


def test_eval_scores_a_ranged_run_only_inside_its_range(tmp_path, capsys):
    """A `--start-frame 2 --end-frame 6` session is scored on [2, 6) only (out-of-range labels
    are not misses); without a recorded range it is scored whole; `--vs` refuses two ranges."""
    import re

    import tailcyclenet.evaluation.eval as eval_cli

    root, ranged = _ranged_prediction(tmp_path / 'a', {'frame_start': 2, 'frame_stop': 6})
    _, whole = _ranged_prediction(tmp_path / 'b', {})

    def coverage(out):
        eval_cli.main([str(out), '--data', str(root), '--split', 'test'])
        text = capsys.readouterr().out
        return text, float(re.search(r'\[2d\] coverage ([0-9.]+)', text).group(1))

    text, cov = coverage(ranged)
    assert 'frame range [2, 6)' in text and cov == 1.0
    assert coverage(whole)[1] < 0.75
    with pytest.raises(SystemExit, match='different frame ranges'):
        eval_cli.main([str(ranged), '--data', str(root), '--split', 'test', '--vs', str(whole)])
    eval_cli.main([str(ranged), '--data', str(root), '--split', 'test', '--vs', str(ranged)])
    assert 'PAIRED' in capsys.readouterr().out


def test_paired_deltas_use_only_what_both_arms_have():
    """An arm that declines the hard points/steps must not read better (eval rule 10)."""
    true = np.zeros((1, 4, 1, 2))
    good = np.zeros((1, 4, 1, 2))                             # exact on all four frames
    good[0, 2] = 3.0                                          # except one, where it is 3 off
    picky = np.full((1, 4, 1, 2), np.nan)
    picky[0, :2] = 0.0                                        # attempts only the two easy frames

    ea, eb, n, nlab, fa, fb, fn = ev._shared_error({'_pred': good, '_true': true},
                                                   {'_pred': picky, '_true': true})
    assert (n, nlab) == (2, 4)
    assert ea == eb == 0.0, 'over the shared points the two are identical, and the delta is 0'
    assert fn == 2, 'the per-frame weight counts the frames BOTH arms matched, not all four'
    assert fa == fb == 0.0, 'the per-frame delta is 0 for the same reason as the pointwise one'
    # ...where a whole-set comparison would have made the picky arm look better by 3/4 of a unit.
    assert np.nanmean(np.linalg.norm(good - true, axis=-1)) > 0

    true = np.zeros((1, 4, 1, 2))
    true[0, :, 0, 0] = np.arange(4)
    full = true.copy()
    picky = np.full((1, 4, 1, 2), np.nan)
    picky[0, :2] = true[0, :2]                                 # only the first step
    ra, rb, n = ev._shared_motion({'_pred': full, '_true': true},
                                  {'_pred': picky, '_true': true})
    assert n == 1 and ra == pytest.approx(1.0) and rb == pytest.approx(1.0)


def test_idsw_band_flag_reaches_the_cli_end_to_end(tmp_path, capsys):
    """`--idsw-band` through `eval.main()` on an exact-crossing tie, so the band has width."""
    _session_3d_multi(tmp_path / 'ds' / 'test' / 's', T=3, sep=20.0)
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    gid = next(iter(sess.groups))
    true = sess.labels(gid).points3d.copy()
    pred = true.copy()
    mid = 0.5 * (true[0, 1] + true[1, 1])
    pred[0, 1], pred[1, 1] = mid, mid            # an exact tie at frame 1

    npz = tmp_path / 'pred.npz'
    np.savez(npz, __keys__=np.array([f's/{gid}']), __run__=np.array('r'),
             __anchor__=np.array('carry'), __boxes__=np.array('none'),
             **{f's/{gid}|pred': pred, f's/{gid}|mode': np.array('3d')})

    sys.argv = ['eval.py', str(npz), '--data', str(tmp_path / 'ds'), '--split', 'test',
               '--idsw-band', '8', '--mota-dist', '5.0']
    ev.main()
    out = capsys.readouterr().out
    assert 'idsw band (n=8)' in out, out


def test_chunking_partitions_slices_labels_and_holds_the_radius(tmp_path):
    """`--chunk` partitions the frames, holds the whole group's match radius, and slices the labels
    by chunk even when the prediction is a `--max-frames` prefix (each chunk once scored frames 0..n-1)."""
    _session_2d(tmp_path / 'ds' / 'test' / 's', T=20)
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    labels = {f's/{gid}': (sess.labels(gid), sess) for gid in sess.groups}
    gid = next(iter(sess.groups))
    true = labels[f's/{gid}'][0].points2d[..., 0, :]
    preds = {f's/{gid}': {'pred': true.copy(), 'mode': np.array('2d')}}

    chunked_p, chunked_l = ev.chunk_frames(preds, labels, 5)
    assert len(chunked_p) == 4, f'20 frames in 5s is 4 chunks, got {sorted(chunked_p)}'
    # A PARTITION: every frame in exactly one chunk, and the concatenation is the original.
    rebuilt = np.concatenate([chunked_p[f's/{gid}#{t0}']['pred'] for t0 in (0, 5, 10, 15)], axis=1)
    np.testing.assert_array_equal(rebuilt, true)
    for t0 in (0, 5, 10, 15):
        assert chunked_p[f's/{gid}#{t0}']['pred'].shape[1] == 5
        # The labels are sliced to match, or a chunk scores its prediction against another's labels.
        np.testing.assert_array_equal(
            chunked_l[f's/{gid}#{t0}'][0].points2d[..., 0, :], true[:, t0:t0 + 5])

    # THE RADIUS IS THE WHOLE GROUP'S. Every chunk carries the same one, and it is what an unchunked
    # score would have used -- so chunking cannot move the number, only its uncertainty.
    ext = {float(chunked_p[f's/{gid}#{t0}']['__extent__']) for t0 in (0, 5, 10, 15)}
    assert len(ext) == 1, f'chunks must share one match radius, got {ext}'
    rows = ev.score(preds, labels, quiet=True)
    assert abs(ext.pop() - rows[0]['mpjpe_r']) < 1e-6, \
        'the shared radius must be the one the unchunked scoring used'

    _session_2d(tmp_path / 'ds' / 'test' / 's', T=20)
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    gid = next(iter(sess.groups))
    labels = {f's/{gid}': (sess.labels(gid), sess)}
    true = labels[f's/{gid}'][0].points2d[..., 0, :]
    assert true.shape[1] == 20

    # THE PREDICTION COVERS ONLY THE FIRST 10 FRAMES -- what `--max-frames 10` produces.
    preds = {f's/{gid}': {'pred': true[:, :10].copy(), 'mode': np.array('2d')}}
    cp, cl = ev.chunk_frames(preds, labels, 5)
    assert sorted(cp) == [f's/{gid}#0', f's/{gid}#5'], sorted(cp)
    for t0 in (0, 5):
        got = cl[f's/{gid}#{t0}'][0].points2d[..., 0, :]
        np.testing.assert_array_equal(got, true[:, t0:t0 + 5],
                                      err_msg=f'chunk {t0} was handed the wrong frames')
    # ...and the second chunk is a PERFECT prediction of its own frames, so it must score as one.
    # Unsliced labels made this chunk score frames 5-9 against frames 0-4 and read as a total miss.
    rows = ev.score(cp, cl, quiet=True)
    assert len(rows) == 2
    for r in rows:
        assert r['coverage'] == pytest.approx(1.0), f'{r["coverage"]} -- exact prediction, full labels'
        assert r['err'] == pytest.approx(0.0, abs=1e-6), r['err']


def test_chunk_seam_switch_is_counted_with_threaded_mota_state():
    """A switch straddling a `--chunk` seam was counted by neither chunk; `score()` threads one
    `mota(last=)` dict per group so chunked idsw == whole-clip idsw (3b94ca4)."""
    S, T, K = 2, 4, 1
    lab = fmt.empty_labels(S, T, K, 1, mode3d=False)
    lab.vis2d[:] = fmt.VISIBLE
    lab.points2d[0, :, 0, 0, :] = [0.0, 0.0]
    lab.points2d[1, :, 0, 0, :] = [100.0, 100.0]     # far enough apart that radius=1.0 cannot
                                                       # confuse them

    true = lab.points2d[..., 0, :].copy()
    pred = true.copy()
    pred[:, 2:] = true[::-1, 2:]                      # a real seam switch: rows swap at t=2

    preds = {'s/g0': {'pred': pred, 'mode': np.array('2d')}}
    labels = {'s/g0': (lab, None)}

    whole = ev.score(preds, labels, mota_dist=1.0, quiet=True)
    assert whole[0]['mota']['idsw'] == 2, 'the whole-clip reading is the ground truth here'

    cp, cl = ev.chunk_frames(preds, labels, 2)
    chunked = ev.score(cp, cl, mota_dist=1.0, quiet=True)
    assert len(chunked) == 2
    by_t0 = {r['group']: r['mota']['idsw'] for r in chunked}
    assert by_t0 == {'s/g0#0': 0, 's/g0#2': 2}, \
        'the switch must be attributed to the chunk it is first observed in, not lost at the seam'
    assert sum(by_t0.values()) == whole[0]['mota']['idsw'] == 2, \
        'a chunked idsw total must equal the whole-clip idsw, or the debt this test guards is back'


def test_vis_confusion_follows_the_match_and_skips_projected():
    """`_vis_confusion` scores a row against the GT row it was MATCHED to, and PROJECTED is not an
    assessed negative (A5)."""
    conf = np.array([[[5.0, -5.0]], [[-5.0, 5.0]]])       # (Sp=2, T=1, K=2)
    out = {'conf': conf}
    vis3d = np.array([[[fmt.VISIBLE, fmt.MISSING]], [[fmt.MISSING, fmt.VISIBLE]]])
    lab = SimpleNamespace(vis3d=vis3d, vis2d=None)

    identity = ev._vis_confusion(out, lab, '3d', 1, frame_pairs=[[(0, 0, 0.0), (1, 1, 0.0)]])
    assert identity['vis_precision'] == pytest.approx(1.0)
    assert identity['vis_recall'] == pytest.approx(1.0)

    swapped = ev._vis_confusion(out, lab, '3d', 1, frame_pairs=[[(0, 1, 0.0), (1, 0, 0.0)]])
    assert swapped['vis_precision'] == pytest.approx(0.0), \
        'the wrong correspondence must score as wrong, not as the row-position accident'
    assert swapped['vis_recall'] == pytest.approx(0.0)

    out = {'conf': np.array([[[5.0, 5.0]]])}                      # (Sp=1, T=1, K=2)
    vis3d = np.array([[[fmt.VISIBLE, fmt.PROJECTED]]])
    lab = SimpleNamespace(vis3d=vis3d, vis2d=None)

    m = ev._vis_confusion(out, lab, '3d', 1)
    assert m['vis_n'] == 1, 'the PROJECTED keypoint must not be counted as assessed'
    assert m['vis_precision'] == pytest.approx(1.0)


def test_score_skips_motion_ratio_when_a_group_has_zero_true_instances():
    """S_true=0 with a predicted row (facemap cam0_D*) used to crash `motion_ratio` on a shape
    mismatch; it is undefined there, so it is skipped."""
    T, K = 12, 4
    true = np.full((0, T, K, 1, 2), np.nan, np.float32)
    lab = Labels(animal_ids=[], points3d=None, vis3d=None, points2d=true, vis2d=None,
                boxes=None, instance=None)
    pred = np.random.default_rng(0).normal(size=(1, T, K, 2)).astype(np.float32)
    preds = {'s/g': {'pred': pred, 'mode': np.array('2d')}}
    labels = {'s/g': (lab, None)}

    rows = ev.score(preds, labels, quiet=True)
    assert len(rows) == 1
    assert rows[0]['S_pred'] == 1 and rows[0]['S_true'] == 0
    assert rows[0]['motion_ratio'] is None


def test_idf1_group_mapping_spans_chunks():
    """Per-unit IDF1 maps each chunk alone; `idf1_group` maps the whole group = unchunked."""
    S, T = 2, 8
    lab = fmt.empty_labels(S, T, 1, 1, mode3d=False)
    lab.vis2d[:] = fmt.VISIBLE
    lab.points2d[0, :, 0, 0, :] = [0.0, 0.0]
    lab.points2d[1, :, 0, 0, :] = [100.0, 100.0]
    true = lab.points2d[..., 0, :].copy()
    pred = true.copy()
    pred[:, 4:] = true[::-1, 4:]
    preds = {'s/g0': {'pred': pred, 'mode': np.array('2d')}}
    labels = {'s/g0': (lab, None)}
    whole = ev.score(preds, labels, mota_dist=1.0, quiet=True)[0]
    assert whole['idf1']['idf1'] == pytest.approx(0.5)
    cp, cl = ev.chunk_frames(preds, labels, 4)
    chunked = ev.score(cp, cl, mota_dist=1.0, quiet=True)
    assert [r['idf1']['idf1'] for r in chunked] == [1.0, 1.0], 'each unit maps on its own'
    assert all(r['idf1_group']['idf1'] == pytest.approx(0.5) for r in chunked)
    assert chunked[0]['idf1_group'] == whole['idf1_group']
