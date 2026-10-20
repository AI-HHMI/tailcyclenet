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


def test_box_corrupt_needs_a_detector_and_a_valid_spec():
    """`--box-corrupt` breaks detector boxes: refused without --detector or with a bad spec."""
    import pytest
    from tailcyclenet.infer.resolve import refuse_before_load
    for argv, needle in ((['--box-corrupt', 'drop:0.2'], 'without --detector'),
                         (['--box-corrupt', 'blur:0.2', '--detector', 'x'], 'kind must be')):
        args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o', *argv])
        with pytest.raises(SystemExit, match=needle):
            refuse_before_load(args)
    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o', '--detector',
                                      'x', '--box-corrupt', 'swap:0.25'])
    refuse_before_load(args)
    config = {'data': {}, 'model': {'box_prompt': 'none'}}
    frames, px = trained_window(args, config)
    assert infer_config(args, config, SimpleNamespace(mode='3d'), frames, px,
                        'cpu').box_corrupt == 'swap:0.25'


def test_refine_consensus_needs_3d_and_refine():
    """R1 compares pass 1's cameras: refused in 2D and under --no-refine, resolved in 3D."""
    import pytest
    config = {'data': {}, 'model': {'box_prompt': 'none'}}
    for mode, extra in (('2d', []), ('3d', ['--no-refine'])):
        args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o',
                                          '--refine-consensus', *extra])
        check_frame_range(args)
        with pytest.raises(SystemExit, match='refine-consensus'):
            infer_config(args, config, SimpleNamespace(mode=mode), 4, 64, 'cpu')
    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o',
                                      '--refine-consensus', '--refine-consensus-res', '0.3'])
    check_frame_range(args)
    cfg = infer_config(args, config, SimpleNamespace(mode='3d'), 4, 64, 'cpu')
    assert (cfg.refine_consensus, cfg.refine_consensus_res, cfg.refine_consensus_vis) == \
        (True, 0.3, 0.0)


def test_refine_carry_fallback_is_3d_only():
    """R4 reprojects a carried 3D pose: refused on a 2D run, resolved on a 3D one."""
    import pytest
    config = {'data': {}, 'model': {'box_prompt': 'none'}}
    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o',
                                      '--refine-carry-fallback'])
    check_frame_range(args)
    with pytest.raises(SystemExit, match='refine-carry-fallback'):
        infer_config(args, config, SimpleNamespace(mode='2d'), 4, 64, 'cpu')
    assert infer_config(args, config, SimpleNamespace(mode='3d'), 4, 64,
                        'cpu').refine_carry_fallback
