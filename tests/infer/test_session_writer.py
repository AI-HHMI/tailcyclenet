"""The prediction-session writer: tables, signals, sidecars and provenance."""
import numpy as np
import pytest

from tailcyclenet.format import Registry, load_dataset
from tests.helpers import _session_2d


def test_session_writer_preserves_named_2d_signals_and_window_metadata(rat, tmp_path):
    """The new prediction table keeps visibility and model confidence as separate signals."""
    import pyarrow.parquet as pq

    _, sess, registry, _ = rat
    from tailcyclenet.infer.session_writer import SessionWriter

    gid = next(iter(sess.groups))
    K = sess.n_keypoints
    C = len(sess.rig)
    pred2d = np.zeros((1, 2, C, K, 2), np.float32)
    visibility_logit = (np.arange(2 * C * K, dtype=np.float32).reshape(1, 2, C, K) / 10) - 1
    model_logit = (np.arange(2 * C * K, dtype=np.float32).reshape(1, 2, C, K) / 7) - 2
    blk = {
        'animal_ids': np.array(['a00'], object),
        'pred': np.zeros((1, 2, K, 2), np.float32),
        'conf': np.zeros((1, 2, K), np.float32),
        'pred2d': pred2d,
        'conf2d': visibility_logit,
        'model_conf2d': model_logit,
        'box_agree': np.full((1, 2, C), np.nan, np.float32),
        'outcome': np.zeros((1, 1), np.int8),
        'crop': np.zeros((1, 1, C, 4), np.float32),
        'outcome_names': ['ok'],
        'window_start': np.array([0], np.int32),
        'window_stop': np.array([2], np.int32),
        'owner_window': np.array([0, 0], np.int32),
    }
    out = tmp_path / 'pred'
    writer = SessionWriter(out, sess, registry, {'n_frames': 2}, [gid])
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)

    table = pq.read_table(out / 'points2d.pq')
    assert {'visibility_logit', 'visibility_probability', 'confidence_logit',
            'confidence_probability', 'slot', 'window', 'window_start', 'window_stop'} <= set(
                table.column_names)
    visibility = table['visibility_logit'].to_numpy()
    confidence = table['confidence_logit'].to_numpy()
    assert np.allclose(table['visibility_probability'].to_numpy(),
                       1 / (1 + np.exp(-visibility)))
    assert np.allclose(table['confidence_probability'].to_numpy(),
                       1 / (1 + np.exp(-confidence)))
    assert not np.allclose(visibility, confidence)

    no_model = dict(blk)
    no_model.pop('model_conf2d')
    out_no_model = tmp_path / 'pred-no-model-confidence'
    writer = SessionWriter(out_no_model, sess, registry, {'n_frames': 2}, [gid])
    writer.write_block(gid, no_model, 0, 0)
    writer.close(complete=True)
    missing = pq.read_table(out_no_model / 'points2d.pq')
    assert np.isnan(missing['confidence_probability'].to_numpy()).all()
    assert np.isnan(missing['confidence_logit'].to_numpy()).all()


def test_prediction_3d_missing_status_retains_best_guess_coordinates(mv, tmp_path):
    """A 3D `missing` prediction row keeps its best-guess coordinates and validates."""
    import pyarrow.parquet as pq

    _, sess, registry, _ = mv
    from tailcyclenet.format.prediction import load_predictions
    from tailcyclenet.infer.session_writer import SessionWriter

    gid = next(iter(sess.groups))
    K, C, T = sess.n_keypoints, len(sess.rig), 2
    pattern = np.resize(np.array([-1.0, 0.0, 1.0], np.float32), K)
    logits = np.tile(pattern[None, None], (1, T, 1))
    pred = np.arange(T * K * 3, dtype=np.float32).reshape(1, T, K, 3) + 1
    blk = {
        'animal_ids': np.array(['a00'], object), 'pred': pred, 'conf': logits,
        'pred2d': np.full((1, T, C, K, 2), np.nan, np.float32),
        'conf2d': np.full((1, T, C, K), np.nan, np.float32),
        'box_agree': np.full((1, T, C), np.nan, np.float32),
        'outcome': np.zeros((1, 1), np.int8), 'crop': np.zeros((1, 1, C, 4), np.float32),
        'outcome_names': ['ok'], 'window_start': np.array([0], np.int32),
        'window_stop': np.array([T], np.int32), 'owner_window': np.zeros(T, np.int32),
    }
    out = tmp_path / 'pred3d'
    writer = SessionWriter(out, sess, registry,
                           {'n_frames': T, 'source_session_id': sess.session_id}, [gid])
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)

    table = pq.read_table(out / 'points3d.pq')
    rows = table.to_pylist()
    assert len(rows) == T * K
    for row in rows:
        assert row['status'] == ('visible' if row['score'] > 0.5 else 'missing')
        assert np.isfinite([row['x'], row['y'], row['z']]).all()
    zero_logit = next(row for row in rows if row['score_logit'] == 0.0)
    assert zero_logit['status'] == 'missing' and zero_logit['x'] is not None
    preds, _ = load_predictions(out)
    assert np.isfinite(preds[f'{sess.session_id}/{gid}']['pred'][:, :T]).all()
    from tailcyclenet.format import Session, validate_session
    errs = [e for e in validate_session(Session.load(out)) if '[rule 7]' not in e]
    assert not errs, f'prediction-only 3D rows should validate apart from missing pixels: {errs}'


def test_quality_columns_are_optional_and_align_with_pose_rows(scene, tmp_path):
    import pyarrow.parquet as pq

    from tailcyclenet.infer.session_writer import SessionWriter

    _, sess, registry, _ = scene
    gid = next(iter(sess.groups))
    T, K, C = 2, sess.n_keypoints, len(sess.rig)
    R = 3 if sess.mode == '3d' else 2
    pred = np.arange(T * K * R, dtype=np.float32).reshape(1, T, K, R) + 1
    pred2d = (pred[:, :, None] if R == 2 else
              np.full((1, T, C, K, 2), np.nan, np.float32))
    quality = np.arange(T * K, dtype=np.float32).reshape(1, T, K) / 10
    precision = quality + 1
    blk = {
        'animal_ids': np.array(['a00'], object), 'pred': pred,
        'conf': np.ones((1, T, K), np.float32), 'pred2d': pred2d,
        'conf2d': np.ones((1, T, C, K), np.float32),
        'model_conf2d': np.ones((1, T, C, K), np.float32),
        'quality': quality, 'quality_precision': precision,
        'box_agree': np.full((1, T, C), np.nan, np.float32),
        'outcome': np.zeros((1, 1), np.int8), 'crop': np.zeros((1, 1, C, 4), np.float32),
        'outcome_names': ['ok'], 'window_start': np.array([0], np.int32),
        'window_stop': np.array([T], np.int32), 'owner_window': np.zeros(T, np.int32),
    }
    out = tmp_path / f'quality-{sess.mode}'
    writer = SessionWriter(out, sess, registry, {'n_frames': T}, [gid], quality_enabled=True)
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)
    table_name = 'points3d.pq' if R == 3 else 'points2d.pq'
    rows = pq.read_table(out / table_name).to_pylist()
    assert len(rows) == T * K
    got = {(row['frame'], row['bodypart']):
           (row['quality_score'], row['quality_precision']) for row in rows}
    for frame in range(T):
        for k, bodypart in enumerate(registry.names):
            assert got[(frame, bodypart)] == pytest.approx((quality[0, frame, k],
                                                             precision[0, frame, k]))


def test_window_prediction_sidecar_streams_typed_nullable_records(rat, tmp_path):
    import pyarrow.parquet as pq

    _, sess, registry, _ = rat
    from tailcyclenet.infer.session_writer import SessionWriter

    gid = next(iter(sess.groups))
    out = tmp_path / 'pred-with-sidecar'
    writer = SessionWriter(out, sess, registry,
                           {'n_frames': 4, 'window_predictions': True}, [gid],
                           quality_enabled=True)
    writer.write_window_records([
        {'record_type': 'window', 'group_id': gid, 'frame': None, 'window': 1,
         'window_start': 2, 'window_stop': 4, 'slot': 0, 'animal_id': 'a00',
         'camera': sess.cam_names[0], 'bodypart': None, 'outcome': 'ok',
         'crop_x0': 1.0, 'crop_y0': 2.0, 'crop_x1': 3.0, 'crop_y1': 4.0,
         'crop_refined_x0': 5.0},
        {'record_type': 'point3d', 'group_id': gid, 'frame': 2, 'window': 1,
         'window_start': 2, 'window_stop': 4, 'slot': 0, 'animal_id': 'a00',
         'camera': None, 'bodypart': sess.names[0], 'x': 1.5, 'y': 2.5, 'z': 3.5,
         'status': 'missing', 'score': 0.4, 'score_logit': -0.4,
         'quality_score': 0.7, 'quality_precision': 0.8, 'gated': True},
    ])
    writer.close(complete=True)

    table = pq.read_table(out / 'window_predictions.pq')
    rows = table.to_pylist()
    assert len(rows) == 2
    assert rows[0]['frame'] is None and rows[0]['bodypart'] is None
    assert rows[0]['refined_x0'] == 5.0
    assert rows[1]['status'] == 'missing' and rows[1]['gated'] is True
    assert (rows[1]['x'], rows[1]['y'], rows[1]['z']) == (1.5, 2.5, 3.5)
    assert (rows[1]['quality_score'], rows[1]['quality_precision']) == pytest.approx((0.7, 0.8))
    assert rows[0]['quality_score'] is None
    assert table.schema.field('frame').nullable
    assert table.schema.field('bodypart').nullable


def test_uncalibrated_singleview_prediction_session_uses_normalized_units(tmp_path):
    """The finite single-view fallback must not inherit a 2D source's pixel units."""
    import tomllib

    from tailcyclenet.format import Session
    from tailcyclenet.infer.session_writer import SessionWriter

    source_path = tmp_path / 'r' / 'train' / 's'
    _session_2d(source_path, T=4)
    source = Session.load(source_path)
    source.mode = '3d'
    registry = Registry.build([load_dataset(source_path.parent.parent)])
    out = tmp_path / 'prediction'
    writer = SessionWriter(out, source, registry, {}, list(source.groups))
    writer.close(complete=True)
    cfg = tomllib.loads((out / 'session.toml').read_text())
    assert cfg['mode'] == '3d'
    assert cfg['units'] == 'normalized'


def test_a_duplicate_provenance_key_raises_rather_than_silently_winning(tmp_path):
    """Two provenance values under one name raise (the run's `box_source` was once silently
    overwritten by the detector's); the same value twice is fine."""
    from tailcyclenet.format import Registry, Session
    from tailcyclenet.infer.session_writer import SessionWriter

    _session_2d(tmp_path / 'r' / 'test' / 's')
    sess = Session.load(tmp_path / 'r' / 'test' / 's')
    sess.preload()
    reg = Registry.build([load_dataset(tmp_path / 'r')])

    with pytest.raises(ValueError, match='given twice'):
        SessionWriter(tmp_path / 'out', sess, reg,
                      [('box_source', 'keypoints'), ('box_source', 'instances')],
                      list(sess.groups))
    # ...and the same value twice is not a conflict.
    SessionWriter(tmp_path / 'ok', sess, reg,
                  [('box_source', 'keypoints'), ('box_source', 'keypoints')],
                  list(sess.groups)).close()
