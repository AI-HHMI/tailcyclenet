"""Resolving a pose run's config into the inference recipe (`infer/resolve.py`)."""
from types import SimpleNamespace

from tailcyclenet.infer.cli import build_parser
from tailcyclenet.infer.resolve import check_frame_range, infer_config, trained_window
from tailcyclenet.loader.config import LoaderConfig


def _resolve(data):
    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o'])
    check_frame_range(args)
    config = {'data': data, 'model': {'box_prompt': 'none'}}
    frames, px = trained_window(args, config)
    return infer_config(args, config, SimpleNamespace(mode='2d'), frames, px, 'cpu')


def test_the_run_configs_data_block_reaches_the_infer_config():
    """A non-default `[data]` value is what inference runs at; an absent one is `LoaderConfig`'s
    own default, so the driver restates no loader default a training change could leave behind."""
    cfg = _resolve({'n_frames': 8, 'image_size': 128, 'min_crop_dim': 32,
                    'box_source': 'keypoints'})
    assert (cfg.n_frames, cfg.image_size, cfg.min_crop_dim, cfg.box_source) == \
        (8, 128, 32, 'keypoints')
    d = LoaderConfig()
    assert (8, 128, 32, 'keypoints') != (d.n_frames, d.image_size, d.min_crop_dim, d.box_source)
    cfg = _resolve({})
    assert (cfg.n_frames, cfg.image_size, cfg.min_crop_dim, cfg.box_source) == \
        (d.n_frames, d.image_size, d.min_crop_dim, d.box_source)
