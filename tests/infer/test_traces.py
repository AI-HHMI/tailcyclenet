"""Diagnostic traces: overlap and detection traces round-trip."""
import numpy as np


def test_overlap_trace_round_trips_both_estimates_and_its_census(tmp_path):
    """`--overlap-trace` writes exactly what the hook recorded: both estimates, the census (incl. what
    was NOT seen), the prior-OOB and model-confidence probes; an empty capture keeps its census."""
    import json

    from tailcyclenet.infer.traces import write_overlap_trace

    rows = [{'animal': 0, 'frame': 12, 'pos': 0, 'dist': 1.5, 'n_shared': 3,
             'prev': np.arange(6, dtype=np.float32).reshape(3, 2),
             'new': np.arange(6, dtype=np.float32).reshape(3, 2) + 1.5},
            {'animal': 1, 'frame': 13, 'pos': 1, 'dist': 0.25, 'n_shared': 2,
             'prev': np.zeros((3, 2), np.float32),
             'new': np.full((3, 2), 0.25, np.float32)}]
    census = {'pairs_seen': 2, 'pairs_no_prev': 7, 'pairs_no_shared': 1,
              'block_boundary_seams': 3, 'block_boundary_frames': 12}
    out = tmp_path / 'sub' / 'ovl.npz'
    write_overlap_trace(out, {"s/g000": (rows, census, [(0, 8, 2, 17)], [(0, 8, -0.5, -0.9, 3)])})

    z = np.load(out, allow_pickle=True)
    assert list(z['animal']) == [0, 1] and list(z['frame']) == [12, 13]
    assert list(z['pos']) == [0, 1] and list(z['n_shared']) == [3, 2]
    np.testing.assert_allclose(z['dist'], [1.5, 0.25])
    assert z['prev'].shape == (2, 3, 2) and z['new'].shape == (2, 3, 2)
    np.testing.assert_allclose(z['new'][0] - z['prev'][0], 1.5)
    got = json.loads(str(z['census']))['s/g000']
    assert got == census, got
    assert list(z['oob_animal']) == [0] and list(z['oob_frame']) == [8]
    assert list(z['oob_nan']) == [2] and list(z['oob_k']) == [17]
    assert list(z['conf_animal']) == [0] and list(z['conf_frame']) == [8]
    np.testing.assert_allclose(z['conf_mean'], [-0.5])
    np.testing.assert_allclose(z['conf_min'], [-0.9])

    empty = tmp_path / 'empty.npz'
    write_overlap_trace(empty, {'s/g000': ([], {'pairs_seen': 0, 'pairs_no_prev': 4}, [], [])})
    ze = np.load(empty, allow_pickle=True)
    assert ze['dist'].size == 0 and ze['prev'].size == 0
    assert json.loads(str(ze['census']))['s/g000']['pairs_no_prev'] == 4


def test_det_trace_file_is_valid_json(tmp_path):
    """`--det-trace` writes JSON a reader can load: it used to end in a literal backslash-n."""
    import json

    from tailcyclenet.infer.traces import write_det_trace

    path = tmp_path / 'sub' / 'trace.json'
    write_det_trace(path, {'s/g000': [{'frame': 0, 'kept': 2}]})
    assert json.loads(path.read_text()) == {'s/g000': [{'frame': 0, 'kept': 2}]}
    assert path.read_text().endswith('}\n')
