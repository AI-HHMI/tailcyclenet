import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))


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
    import conftest as cf

    from tailcyclenet.format import load_dataset
    from tailcyclenet.infer.session_writer import SessionWriter

    root = tmp_path / 'data'
    cf._session_2d(root / 'test' / 's', T=8)
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
