"""The QC path's device contract: a window must be BATCHED and MOVED before the forward.

`score_root` took `--device` all the way to `load_scorer_run` and then fed the model CPU tensors
from the loader, so every non-CPU scoring run died in `conv3d` with "Input type (torch.FloatTensor)
and weight type (torch.cuda.FloatTensor) should be the same". Nothing caught it because the one
caller on record scored on CPU, where the missing transfer is a no-op.

The shapes matter as much as the device. `views` and `coords` gain a batch axis at the call site
while `kpt_ids` does not -- batching the ids early made them `[1, 1, K]` and tripped `score`'s own
`(B, K)` assertion. Both halves are asserted here, on CPU, so no GPU is needed to hold the line.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import polars as pl
import torch

import tailcyclenet.scorer.qc as qc
from tailcyclenet.dataset import PoseDataset
from tailcyclenet.scorer.qc import _span_indices, _to_device


def _window():
    """One loader-shaped window: a per-camera view list, coords, a camera dict and keypoint ids."""
    views = [torch.zeros(4, 8, 8, 3, dtype=torch.uint8), torch.zeros(4, 6, 6, 3, dtype=torch.uint8)]
    coords = torch.zeros(4, 5, 3)
    cgroup = [{'mat': torch.eye(3), 'offset': torch.zeros(2), 'name': 'cam0', 'n_frames': 4},
              {'mat': torch.eye(3), 'offset': torch.zeros(2), 'name': 'cam1', 'n_frames': 4}]
    return views, coords, cgroup, torch.arange(5)


def test_views_and_coords_get_a_batch_axis_and_kpt_ids_do_not():
    """The batching split is the contract `score` asserts against."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'cpu')
    assert views[0].shape == (1, 4, 8, 8, 3)
    assert views[1].shape == (1, 4, 6, 6, 3)
    assert coords.shape == (1, 4, 5, 3)
    assert kpt_ids.shape == (5,)


def test_everything_lands_on_the_requested_device():
    """The camera tensors travel too -- they feed the decoder's geometry, not just the pixels."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'cpu')
    assert all(v.device.type == 'cpu' for v in views)
    assert coords.device.type == 'cpu'
    assert kpt_ids.device.type == 'cpu'
    assert all(t.device.type == 'cpu' for cam in cgroup for t in cam.values()
               if torch.is_tensor(t))


def test_non_tensor_camera_entries_survive():
    """A camera dict carries labels and counts alongside its tensors; those must pass through."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'cpu')
    assert [c['name'] for c in cgroup] == ['cam0', 'cam1']
    assert [c['n_frames'] for c in cgroup] == [4, 4]
    assert len(cgroup) == len(views)


def test_the_move_is_real_and_not_a_noop_that_happens_to_type_check():
    """Targeting 'meta' proves the transfer happens: nothing was already on it."""
    views, coords, _cgroup, _kpt_ids = _to_device(*_window(), 'meta')
    assert views[0].device.type == 'meta'
    assert coords.device.type == 'meta'


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


def test_spans_select_by_window_start_not_by_decoded_item():
    """A span picks the windows that START inside it, and only those.

    The filter runs on `dataset.index`, so a 1000-frame span of a 122k-frame clip costs ~84
    windows to score instead of the ~10k the whole group would. Scoring the item first and
    dropping it after would spend exactly the cost `--spans-csv` exists to avoid.
    """
    ds = _fake_dataset([0, 12, 24, 36, 48])
    kept = _span_indices(ds, {('sess', 'g', 'det00'): (12, 36)})
    assert [ds.index[i].start for i in kept] == [12, 24, 36]


def test_spans_key_is_the_loader_row_key_and_a_miss_is_empty():
    """The key is `(session_id, group, animal_id)` -- what the loader's `row` carries. A key that
    names a different session, group or animal matches nothing rather than selecting by accident.
    """
    ds = _fake_dataset([0, 12])
    assert _span_indices(ds, {('other', 'g', 'det00'): (0, 100)}) == []
    assert _span_indices(ds, {('sess', 'other', 'det00'): (0, 100)}) == []
    assert _span_indices(ds, {('sess', 'g', 'det01'): (0, 100)}) == []


def test_every_animal_of_a_group_is_matched_by_its_own_key():
    """A multi-animal group has one index entry per animal; the span names one of them."""
    ds = _fake_dataset([0, 12], animals=('det00', 'det01'))
    kept = _span_indices(ds, {('sess', 'g', 'det01'): (0, 100)})
    assert [ds.index[i].animal for i in kept] == [1, 1]


def test_get_once_does_not_retry_a_failed_index():
    """QC's single-attempt accessor cannot silently draw a replacement index."""
    ds = PoseDataset.__new__(PoseDataset)
    ds.train = False
    ds.seed = 23
    calls = []
    ds._shape = lambda rng: {'n_cams': 1, 'single_view_draw': 0.0}
    ds._item = lambda idx, rng, shape: calls.append(idx) or None

    assert ds.get_once(7) is None
    assert calls == [7]


def test_score_root_records_failure_without_scoring_a_replacement(monkeypatch):
    """A failed requested window is coverage, never a random group's score."""

    sess = SimpleNamespace(session_id='sess', names=['k'])
    sess.labels = lambda _gid: SimpleNamespace(animal_ids=['a0'])

    class FakeDataset:
        registry = SimpleNamespace(names=['k'])
        seed = 23
        train = False
        index = [SimpleNamespace(session=sess, gid='target', animal=0, start=0)]

        def __init__(self, *_args, **_kwargs):
            self.calls = []

        def __len__(self):
            return len(self.index)

        def get_once(self, idx):
            self.calls.append(idx)
            return None

    class FakeModel:
        def eval(self):
            return self

    ckpt = SimpleNamespace(name='checkpoint_last.pth')
    monkeypatch.setattr(qc, 'PoseDataset', FakeDataset)
    monkeypatch.setattr(qc, '_loader_config', lambda _config: SimpleNamespace())
    monkeypatch.setattr(qc, 'load_scorer_run',
                        lambda *_args, **_kwargs: (FakeModel(), {},
                                                    SimpleNamespace(names=['k']), ckpt))
    coverage = []
    table, _registry, _config = qc.score_root(
        Path('run'), 'data', 'test', spans={('sess', 'target', 'a0'): (0, 100)},
        coverage=coverage)

    assert table.is_empty()
    assert coverage == [{'index': 0, 'session': 'sess', 'group': 'target', 'animal': 'a0',
                         'start': 0, 'status': 'unscorable',
                         'reason': 'item_build_failed'}]


def test_score_root_forwards_explicit_checkpoint_and_records_iteration(monkeypatch, tmp_path):
    """An explicitly selected checkpoint reaches the loader and output metadata sink."""
    checkpoint = tmp_path / 'checkpoint_best.pth'
    torch.save({'iteration': 37}, checkpoint)
    calls = {}

    class FakeDataset:
        registry = SimpleNamespace(names=['k'])
        index = []

        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 0

    class FakeModel:
        def eval(self):
            return self

    def fake_load(*args, **kwargs):
        calls.update(kwargs)
        return FakeModel(), {}, SimpleNamespace(names=['k']), checkpoint

    monkeypatch.setattr(qc, 'PoseDataset', FakeDataset)
    monkeypatch.setattr(qc, '_loader_config', lambda _config: SimpleNamespace())
    monkeypatch.setattr(qc, 'load_scorer_run', fake_load)
    info = {}
    table, _registry, _config = qc.score_root(
        Path('run'), 'data', 'test', checkpoint='checkpoint_best.pth', checkpoint_info=info)

    assert table.is_empty()
    assert calls == {'checkpoint': 'checkpoint_best.pth', 'device': 'cpu'}
    assert info == {'checkpoint_file': str(checkpoint), 'checkpoint_iteration': 37}


def test_score_root_omits_checkpoint_selection_by_default(monkeypatch):
    """No selection remains the loader's existing latest/last behavior, not implicit best."""
    calls = {}

    class FakeDataset:
        registry = SimpleNamespace(names=['k'])
        index = []

        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 0

    class FakeModel:
        def eval(self):
            return self

    def fake_load(*args, **kwargs):
        calls.update(kwargs)
        return FakeModel(), {}, SimpleNamespace(names=['k']), SimpleNamespace(name='checkpoint_last.pth')

    monkeypatch.setattr(qc, 'PoseDataset', FakeDataset)
    monkeypatch.setattr(qc, '_loader_config', lambda _config: SimpleNamespace())
    monkeypatch.setattr(qc, 'load_scorer_run', fake_load)
    qc.score_root(Path('run'), 'data', 'test')

    assert calls == {'checkpoint': None, 'device': 'cpu'}


def test_score_root_explicit_missing_checkpoint_raises(monkeypatch):
    """The loader's explicit-file refusal is not swallowed by QC."""
    def fake_load(*_args, **kwargs):
        raise FileNotFoundError(kwargs['checkpoint'])

    monkeypatch.setattr(qc, 'load_scorer_run', fake_load)
    with pytest.raises(FileNotFoundError, match='missing.pth'):
        qc.score_root(Path('run'), 'data', 'test', checkpoint='missing.pth')


def test_write_outputs_records_checkpoint_provenance(monkeypatch, tmp_path):
    """The output records the resolved file and the checkpoint's training iteration."""
    import toml

    monkeypatch.setattr(qc, 'provenance', lambda: {})
    qc.write_outputs(tmp_path, pl.DataFrame({'score': [1.0]}), Path('run'), 'data', 'test',
                     'report', checkpoint_file=Path('/runs/checkpoint_best.pth'),
                     checkpoint_iteration=37)
    result = toml.load(tmp_path / 'provenance.toml')
    assert result['checkpoint_file'] == '/runs/checkpoint_best.pth'
    assert result['checkpoint_iteration'] == 37


def test_score_session_help_requires_explicit_best_checkpoint(capsys):
    """The CLI documents that validation-selected best is never implicit."""
    import importlib.util

    spec = importlib.util.spec_from_file_location('score_session_cli', 'scripts/score_session.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main(['--help'])
    help_text = capsys.readouterr().out
    assert '--checkpoint' in help_text
    assert 'checkpoint_best.pth' in help_text
    assert 'validation-selected' in help_text
    assert 'explicitly named' in help_text
