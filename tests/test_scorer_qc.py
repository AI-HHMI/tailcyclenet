"""The QC path: device/batching contract, `--spans-csv` selection, and `score_root`'s coverage.

`score_root` once fed a CUDA model CPU tensors; batching `kpt_ids` early tripped `score`'s `(B, K)`
assertion. Both halves are asserted on CPU (via 'meta') so no GPU is needed.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import polars as pl
import torch

import tailcyclenet.infer.score as qc
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.infer.score import _span_indices, _to_device


def _window():
    """One loader-shaped window: a per-camera view list, coords, a camera dict and keypoint ids."""
    views = [torch.zeros(4, 8, 8, 3, dtype=torch.uint8), torch.zeros(4, 6, 6, 3, dtype=torch.uint8)]
    coords = torch.zeros(4, 5, 3)
    cgroup = [{'mat': torch.eye(3), 'offset': torch.zeros(2), 'name': 'cam0', 'n_frames': 4},
              {'mat': torch.eye(3), 'offset': torch.zeros(2), 'name': 'cam1', 'n_frames': 4}]
    return views, coords, cgroup, torch.arange(5)


def test_to_device_batches_views_and_coords_and_moves_everything():
    """Views/coords gain a batch axis, `kpt_ids` does not; camera tensors move, labels survive."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'meta')
    assert views[0].shape == (1, 4, 8, 8, 3) and views[1].shape == (1, 4, 6, 6, 3)
    assert coords.shape == (1, 4, 5, 3)
    assert kpt_ids.shape == (5,)
    assert all(t.device.type == 'meta' for t in [*views, coords, kpt_ids])
    assert all(t.device.type == 'meta' for cam in cgroup for t in cam.values()
               if torch.is_tensor(t))
    assert [(c['name'], c['n_frames']) for c in cgroup] == [('cam0', 4), ('cam1', 4)]


def _fake_dataset(starts, animals=('det00',), session='sess', gid='g'):
    """An index-only stand-in: `_span_indices` must never realise a window."""
    def labels(_gid):
        return SimpleNamespace(animal_ids=list(animals))

    sess = SimpleNamespace(session_id=session, labels=labels)
    index = []
    for a in range(len(animals)):
        for st in starts:
            index.append(SimpleNamespace(session=sess, gid=gid, animal=a, start=st))
    return SimpleNamespace(index=index)


def test_spans_select_half_open_by_window_start_and_row_key():
    """A span keeps windows STARTING in `[start, end)` (filtered on the index, never decoded),
    keyed on `(session_id, group, animal_id)`; a key naming anything else matches nothing."""
    ds = _fake_dataset([0, 12, 24, 36, 48])
    kept = _span_indices(ds, {('sess', 'g', 'det00'): (12, 36)})
    assert [ds.index[i].start for i in kept] == [12, 24], 'the span end must be exclusive'
    for key in (('other', 'g', 'det00'), ('sess', 'other', 'det00'), ('sess', 'g', 'det01')):
        assert _span_indices(ds, {key: (0, 100)}) == []
    ds = _fake_dataset([0, 12], animals=('det00', 'det01'))
    kept = _span_indices(ds, {('sess', 'g', 'det01'): (0, 100)})
    assert [ds.index[i].animal for i in kept] == [1, 1]


def test_get_once_does_not_retry_a_failed_index():
    """QC's single-attempt accessor cannot silently draw a replacement index."""
    ds = PoseDataset.__new__(PoseDataset)
    ds.train = False
    ds.seed = 23
    calls = []
    ds.shape = lambda rng: {'n_cams': 1, 'single_view_draw': 0.0}
    ds._item = lambda idx, rng, shape: calls.append(idx) or None

    assert ds.get_once(7) is None
    assert calls == [7]


def _patch_score_root(monkeypatch, index=(), config=None, load=None):
    """Stub out the dataset and run loader so `score_root` runs on an index alone."""
    class FakeDataset:
        registry = SimpleNamespace(names=['k'])
        seed = 23
        train = False

        def __init__(self, *_args, **_kwargs):
            self.index = list(index)

        def __len__(self):
            return len(self.index)

        def get_once(self, _idx):
            return None

    class FakeModel:
        def eval(self):
            return self

    def default_load(*_args, **_kwargs):
        return (FakeModel(), config or {}, SimpleNamespace(names=['k']),
                SimpleNamespace(name='checkpoint_last.pth'))

    monkeypatch.setattr(qc, 'PoseDataset', FakeDataset)
    monkeypatch.setattr(qc, '_loader_config', lambda _config: SimpleNamespace())
    monkeypatch.setattr(qc, 'load_scorer_run',
                        (lambda *a, **k: load(FakeModel(), *a, **k)) if load else default_load)


def _sess(name):
    sess = SimpleNamespace(session_id=name, names=['k'])
    sess.labels = lambda _gid: SimpleNamespace(animal_ids=['a0'])
    return sess


def test_score_root_applies_session_then_half_open_window_range(monkeypatch):
    """Session filtering happens before the zero-based half-open slice; out of range raises."""
    a, b = _sess('sess-a'), _sess('sess-b')
    index = ([SimpleNamespace(session=a, gid='g', animal=0, start=i) for i in range(4)]
             + [SimpleNamespace(session=b, gid='g', animal=0, start=i) for i in range(2)])
    _patch_score_root(monkeypatch, index)
    coverage = []
    table, _registry, _config = qc.score_root(
        Path('run'), 'data', 'test', session='sess-a', start=1, stop=3, coverage=coverage)
    assert table.is_empty()
    assert [row['index'] for row in coverage] == [1, 2]
    with pytest.raises(ValueError, match='invalid window range'):
        qc.score_root(Path('run'), 'data', 'test', session='sess-b', start=0, stop=3)


def test_score_root_records_failure_without_scoring_a_replacement(monkeypatch):
    """A failed requested window is coverage, never a random group's score."""
    _patch_score_root(monkeypatch, [SimpleNamespace(session=_sess('sess'), gid='target',
                                                    animal=0, start=0)])
    coverage = []
    table, _registry, _config = qc.score_root(
        Path('run'), 'data', 'test', spans={('sess', 'target', 'a0'): (0, 100)},
        coverage=coverage)
    assert table.is_empty()
    assert coverage == [{'index': 0, 'session': 'sess', 'group': 'target', 'animal': 'a0',
                         'start': 0, 'status': 'unscorable',
                         'reason': 'item_build_failed'}]


def test_empty_frame_qc_keeps_frame_schema_and_raw_output(monkeypatch, tmp_path):
    import toml

    _patch_score_root(monkeypatch, config={'scorer': {'output_granularity': 'frame'}})
    table, _registry, config = qc.score_root(Path('run'), 'data', 'test')
    assert {'frame', 'local_t', 'window_start'}.issubset(table.columns)
    qc.write_outputs(tmp_path, table, Path('run'), 'data', 'test', 'report',
                     output_granularity=qc.scorer_output_granularity(config))
    assert (tmp_path / 'window_scores.pq').exists()
    assert toml.load(tmp_path / 'provenance.toml')['output_granularity'] == 'frame'


def test_checkpoint_selection_is_explicit_and_recorded(monkeypatch, tmp_path):
    """No selection passes `None` (latest, never implicit best); a named file reaches the loader,
    its iteration lands in provenance, and a missing one raises."""
    import toml

    checkpoint = tmp_path / 'checkpoint_best.pth'
    torch.save({'iteration': 37}, checkpoint)
    calls = []

    def load(model, *_args, **kwargs):
        calls.append(kwargs)
        if kwargs['checkpoint'] == 'missing.pth':
            raise FileNotFoundError('missing.pth')
        return model, {}, SimpleNamespace(names=['k']), checkpoint

    _patch_score_root(monkeypatch, load=load)
    qc.score_root(Path('run'), 'data', 'test')
    info = {}
    qc.score_root(Path('run'), 'data', 'test', checkpoint='checkpoint_best.pth',
                  checkpoint_info=info)
    assert [c['checkpoint'] for c in calls] == [None, 'checkpoint_best.pth']
    assert info == {'checkpoint_file': str(checkpoint), 'checkpoint_iteration': 37}
    with pytest.raises(FileNotFoundError, match='missing.pth'):
        qc.score_root(Path('run'), 'data', 'test', checkpoint='missing.pth')

    monkeypatch.setattr(qc, 'provenance', lambda: {})
    out = tmp_path / 'out'
    qc.write_outputs(out, pl.DataFrame({'score': [1.0]}), Path('run'), 'data', 'test',
                     'report', **info)
    result = toml.load(out / 'provenance.toml')
    assert result['checkpoint_file'] == str(checkpoint)
    assert result['checkpoint_iteration'] == 37


def test_the_deleted_window_offset_levers_are_refused_by_name():
    """`--window-offset` / nonzero `[data].val_offset` were refuted (report 60); inert 0 loads."""
    from tailcyclenet.infer.score import main
    from tailcyclenet.scorer.train import loader_config

    with pytest.raises(SystemExit, match='--window-offset is deleted'):
        main(['--run', 'r', '--data', 'd', '--out', 'o', '--window-offset', '4'])
    assert loader_config({'val_offset': 0}, {}).val_stride == 0
    assert qc._loader_config({'data': {'val_offset': 0}}).val_stride == 0
    for call in (lambda: qc._loader_config({'data': {'val_offset': 3}}),
                 lambda: loader_config({'val_offset': 3}, {})):
        with pytest.raises(SystemExit, match=r'\[data\]\.val_offset is deleted'):
            call()
