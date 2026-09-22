import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

def test_independent_window_eval_keeps_visibility_logits_and_refuses_mota(tmp_path):
    import conftest as cf

    from tailcyclenet.eval import _independent_slots, label_lookup, score
    from tailcyclenet.format import load_dataset
    from tailcyclenet.infer.predictions import SessionWriter, load_predictions
    import tailcyclenet.eval as eval_cli

    root = tmp_path / 'data'
    cf._session_2d(root / 'test' / 's', T=4)
    ds = load_dataset(root)
    sess = ds.sessions['test'][0]
    gid = 'g000'
    S, T, K, C = 2, 4, sess.n_keypoints, len(sess.cam_names)
    pred = np.zeros((S, T, K, 2), np.float32)
    pred2d = np.zeros((S, T, C, K, 2), np.float32)
    visibility = np.ones((S, T, C, K), np.float32)
    blk = {
        'animal_ids': np.array(['det0', 'det1'], object),
        'pred': pred,
        'conf': np.ones((S, T, K), np.float32),
        'pred2d': pred2d,
        'conf2d': visibility,
        'model_conf2d': np.zeros_like(visibility),
        'box_agree': np.full((S, T, C), np.nan, np.float32),
        'outcome': np.zeros((S, 1), np.int8),
        'crop': np.zeros((S, 1, C, 4), np.float32),
        'outcome_names': ['ok'],
        'window_start': np.array([0], np.int32),
        'window_stop': np.array([T], np.int32),
        'owner_window': np.zeros(T, np.int32),
    }
    out = root / 'test' / 'prediction'
    writer = SessionWriter(out, sess, type('Registry', (), {'names': sess.names})(), {
        'source_session_id': sess.session_id,
        'independent_windows': True,
        'n_frames': T,
    }, [gid])
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)

    from tailcyclenet.dataset import LoaderConfig, PoseDataset
    with pytest.raises(ValueError, match='not annotation inputs'):
        PoseDataset(root, 'test', LoaderConfig(n_frames=T, image_size=64))
    from tailcyclenet.detector.data import BoxDataset
    with pytest.raises(ValueError, match='not detector-training annotations'):
        BoxDataset(root, 'test')

    preds, _ = load_predictions(out)
    labels = label_lookup(root, 'test')
    assert _independent_slots(out, preds, labels)
    rows = score(preds, labels, identity_metrics=False, quiet=True)
    assert len(rows) == 1
    assert rows[0]['vis_n'] > 0
    assert rows[0]['vis_precision'] == 1.0
    assert 'mota' not in rows[0] and 'idsw' not in rows[0]

    with pytest.raises(SystemExit, match='MOTA/IDSW'):
        eval_cli.main([str(out), '--data', str(root), '--split', 'test', '--mota-dist', '20'])
