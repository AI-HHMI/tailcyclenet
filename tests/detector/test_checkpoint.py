"""`load_detector`, run-folder resolution, packaging, and the retired-checkpoint refusals."""
from __future__ import annotations

import pytest
import torch

from tailcyclenet.detector.yolox import YOLOXNano
from tests.helpers import _detector_checkpoint


def test_an_untiled_checkpoints_tile_scale_is_dropped(tmp_path):
    """An untiled checkpoint records `tile_scale = 1.0`; reaching `detect_group` it would
    letterbox the whole frame at `frame_wh * scale`."""

    from tailcyclenet.detector.checkpoint import load_detector
    p = tmp_path / 'detector.pth'
    base = _detector_checkpoint()
    torch.save({**base, 'tile_wh': None, 'tile_scale': 1.0}, p)
    assert load_detector(p).tile_scale is None
    # ...and a genuinely tiled one still keeps it, or the tiled path loses its whole point.
    torch.save({**base, 'tile_wh': [640, 640], 'tile_scale': 0.5}, p)
    assert load_detector(p).tile_scale == 0.5


def test_detector_run_directory_resolves_latest_last_and_best(tmp_path):
    """A run folder resolves to its highest-numbered checkpoint by default (an incomplete run
    too); `last` still demands `detector_last.pth`; `best` is `detector.pth`."""
    from tailcyclenet.detector.checkpoint import load_detector, resolve_detector_checkpoint
    from tailcyclenet.infer.cli import build_parser

    run = tmp_path / 'run'
    run.mkdir()
    torch.save({'iteration': 2}, run / 'detector_it000002.pth')
    (run / 'config.toml').write_text('[training]\niters = 4\n')
    with pytest.raises(ValueError, match='detector_last.pth'):
        resolve_detector_checkpoint(run, checkpoint='last')
    (run / 'metrics.json').write_text('[{"iteration": 2}]')
    torch.save({'iteration': 2}, run / 'detector_last.pth')
    assert resolve_detector_checkpoint(run, checkpoint='latest') == run / 'detector_it000002.pth'
    assert resolve_detector_checkpoint(run, checkpoint='detector_it000002.pth') == \
        run / 'detector_it000002.pth'

    torch.save({'iteration': 4}, run / 'detector_it000004.pth')
    torch.save({'iteration': 2}, run / 'detector.pth')
    torch.save({'iteration': 4}, run / 'detector_last.pth')
    (run / 'metrics.json').write_text('[{"iteration": 2}, {"iteration": 4}]')
    assert resolve_detector_checkpoint(run) == run / 'detector_it000004.pth'
    assert resolve_detector_checkpoint(run, checkpoint='last') == run / 'detector_last.pth'
    assert resolve_detector_checkpoint(run, checkpoint='best') == run / 'detector.pth'

    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o'])
    assert args.detector_checkpoint == 'latest'
    assert load_detector.__defaults__[-1] == 'latest'


def test_load_detector_round_trips_and_refuses_by_name(tmp_path):
    """The checkpoint's own `yolox_version`/`p2`/`obj_quantiles` decide what loads (an unrecorded
    quantile set is `{}`, not a guess); a BatchNorm, a version-less (`trimmed`) or a pose
    checkpoint is refused by name."""
    from tailcyclenet.detector.checkpoint import load_detector

    with pytest.raises(ValueError, match='hybrid'):
        YOLOXNano(version='medium')
    p = tmp_path / 'detector.pth'
    m = YOLOXNano(version='hybrid', p2=True)
    base = _detector_checkpoint(m, input_wh=(96, 96))
    torch.save(base, p)
    ld = load_detector(p)
    assert ld.model.version == 'hybrid' and ld.model.p2 is True
    assert ld.model.STRIDES == (4, 8, 16, 32)
    torch.testing.assert_close(ld.model.state_dict()['head.obj_pred.0.bias'],
                               m.state_dict()['head.obj_pred.0.bias'])
    assert ld.obj_quantiles == {}
    q = {'q01': 0.452, 'q10': 0.601, 'q50': 0.883, 'q90': 0.981}
    torch.save({**base, 'obj_quantiles': q}, p)
    assert load_detector(p).obj_quantiles == q

    refused = {'bn normalisation': {'model_state': {}, 'input_wh': [128, 128],
                                    'yolox_version': 'hybrid'},
               "'trimmed' is a deleted architecture": {k: v for k, v in base.items()
                                                        if k != 'yolox_version'},
               "'pose' checkpoint, not a detector": {'kind': 'pose', 'model_state': {}}}
    for match, ckpt in refused.items():
        torch.save(ckpt, p)
        with pytest.raises(ValueError, match=match):
            load_detector(p)


def test_detector_packaging_preserves_config(tmp_path):
    from tailcyclenet.detector.checkpoint import load_detector
    from tailcyclenet.packaging import package_detector

    run = tmp_path / 'run'
    run.mkdir()
    config = {'data': {'input_wh': [96, 96]}, 'model': {'yolox': 'hybrid'},
              'training': {'iters': 2, 'seed': 7}}
    (run / 'config.toml').write_text('[training]\niters = 2\n')
    model = YOLOXNano(version='hybrid')
    source = {'iteration': 2, 'model_state': model.state_dict(), 'input_wh': [96, 96],
              'norm': 'gn', 'yolox_version': 'hybrid', 'p2': False, 'config': config}
    torch.save(source, run / 'detector_last.pth')
    torch.save(source, run / 'detector_it000002.pth')
    (run / 'metrics.json').write_text('[{"iteration": 2}]')
    out = tmp_path / 'detector.pth'
    package_detector(run, out)
    _ld = load_detector(out)
    loaded, wh = _ld.model, _ld.input_wh
    packaged = torch.load(out, map_location='cpu', weights_only=False)
    assert wh == (96, 96)
    assert packaged['config'] == config
    assert packaged['kind'] == 'detector'
    assert loaded.version == 'hybrid'


def test_load_detector_refuses_a_convnext_size_not_divisible_by_32(tmp_path):
    """B3: a convnext-t checkpoint with a side not divisible by 32 is refused at load, before any
    weights are built, not by an opaque RuntimeError in `detect_raw`'s first forward."""
    from tailcyclenet.detector.checkpoint import load_detector
    torch.save({'kind': 'detector', 'input_wh': [120, 96], 'norm': 'gn',
                'yolox_version': 'convnext-t', 'model_state': {}}, tmp_path / 'det.pth')
    with pytest.raises(ValueError, match='divisible by 32'):
        load_detector(tmp_path / 'det.pth')


@pytest.mark.parametrize('meta, match', [
    ({'yolox_version': 'tiny'}, 'deleted architecture.*v0.3.4'),
    ({}, "yolox='trimmed' is a deleted architecture"),          # absent key = legacy default
    ({'yolox_version': 'hybrid', 'n_keypoints': 17}, 'keypoint branch is deleted'),
    ({'yolox_version': 'hybrid', 'in_channels': 6}, 'in_channels=6')])
def test_load_detector_refuses_retired_checkpoints_by_name(tmp_path, meta, match):
    from tailcyclenet.detector.checkpoint import load_detector
    p = tmp_path / 'detector_it000010.pth'
    torch.save({'input_wh': [64, 64], 'norm': 'gn', 'model_state': {}, **meta}, p)
    with pytest.raises(ValueError, match=match):
        load_detector(p)
