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
    cfg = _resolve({'n_frames': 8, 'image_size': 128, 'min_crop_dim': 48,
                    'crop_pad_frac': 0.1, 'box_source': 'keypoints'})
    assert (cfg.n_frames, cfg.image_size, cfg.min_crop_dim, cfg.crop_pad_frac,
            cfg.box_source) == (8, 128, 48, 0.1, 'keypoints')
    d = LoaderConfig()
    assert (8, 128, 'keypoints') != (d.n_frames, d.image_size, d.box_source)
    cfg = _resolve({})
    assert (cfg.n_frames, cfg.image_size, cfg.box_source) == (d.n_frames, d.image_size, d.box_source)
    assert (cfg.min_crop_dim, cfg.crop_pad_frac) == (d.min_crop_dim, d.crop_pad_frac) == (32, 0.05)


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
    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o', '--refine',
                                      '--refine-consensus', '--refine-consensus-res', '0.3'])
    check_frame_range(args)
    cfg = infer_config(args, config, SimpleNamespace(mode='3d'), 4, 64, 'cpu')
    assert (cfg.refine_consensus, cfg.refine_consensus_res, cfg.refine_consensus_vis) == \
        (True, 0.3, float('-inf'))


def test_refine_defaults_off_everywhere():
    """--refine defaults OFF in 2D and 3D (owner 2026-11-07); --refine turns it on."""
    base = ['--run', 'r', '--data', 'd', '--out', 'o', '--detector', 'det']
    box = {'data': {}, 'model': {'box_prompt': 'film'}}
    for mode, want in (('3d', False), ('2d', False)):
        args = build_parser().parse_args(base)
        check_frame_range(args)
        assert infer_config(args, box, SimpleNamespace(mode=mode), 4, 64, 'cpu').refine is want
    args = build_parser().parse_args(base + ['--refine'])
    check_frame_range(args)
    assert infer_config(args, box, SimpleNamespace(mode='3d'), 4, 64, 'cpu').refine is True


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


def test_refine_carry_dup_radius_resolves_and_refuses_negative():
    """R4's duplicate gate defaults to the tracker's 0.75 sides; 0 is accepted (ungated R4) and
    a negative radius is refused by name."""
    import pytest
    config = {'data': {}, 'model': {'box_prompt': 'none'}}
    base = ['--run', 'r', '--data', 'd', '--out', 'o', '--refine-carry-fallback']
    args = build_parser().parse_args(base)
    check_frame_range(args)
    assert infer_config(args, config, SimpleNamespace(mode='3d'), 4, 64,
                        'cpu').refine_carry_dup_radius == 0.75
    args = build_parser().parse_args(base + ['--refine-carry-dup-radius', '0'])
    check_frame_range(args)
    assert infer_config(args, config, SimpleNamespace(mode='3d'), 4, 64,
                        'cpu').refine_carry_dup_radius == 0.0
    args = build_parser().parse_args(base + ['--refine-carry-dup-radius', '-1'])
    check_frame_range(args)
    with pytest.raises(SystemExit, match='refine-carry-dup-radius'):
        infer_config(args, config, SimpleNamespace(mode='3d'), 4, 64, 'cpu')


def _defaults(mode, model, extra=()):
    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o', *extra])
    check_frame_range(args)
    return infer_config(args, {'data': {}, 'model': model}, SimpleNamespace(mode=mode), 4, 64,
                        'cpu')


def test_anchor_defaults_to_none_in_3d_and_carry_in_2d(capsys):
    """Report 86/90: a 3D session deploys without a carried prior by default (one printed line);
    2D keeps carry; an explicit --anchor carry still wins in 3D."""
    plain = {'box_prompt': 'none'}
    assert _defaults('3d', plain).anchor == 'none'
    assert 'no carried prior by default, report 86' in capsys.readouterr().out
    assert _defaults('2d', plain).anchor == 'carry'
    assert 'report 86' not in capsys.readouterr().out
    assert _defaults('3d', plain, ['--anchor', 'carry']).anchor == 'carry'
    assert _defaults('3d', {'box_prompt': 'film'}, ['--detector', 'det']).anchor == 'none'
    moe3 = {'box_prompt': 'film', 'set_decoder': True, 'set_queries': 4}
    assert _defaults('3d', moe3, ['--detector', 'det']).anchor == 'carry'   # MoE: as before


def test_set_only_checkpoint_still_refuses_explicit_carry():
    import pytest
    with pytest.raises(SystemExit, match='--anchor carry'):
        _defaults('3d', {'set_decoder': 'only', 'box_prompt': 'film'}, ['--anchor', 'carry'])


def test_box_prompt_defaults_to_detector_for_box_film_given_a_box_source():
    """A box-FiLM checkpoint with --detector / --boxes deploys the box every window (`detector`,
    as every report-86 number); explicit --box-prompt firstonly still wins; a non-box model
    ignores the box either way; without a box source a box model is refused as before."""
    import pytest
    film = {'box_prompt': 'film'}
    for mode in ('2d', '3d'):
        for src in (['--detector', 'det'], ['--boxes', 'b.npz']):
            cfg = _defaults(mode, film, src)
            assert (cfg.box_prompt, cfg.box_prompt_first_only) == ('detector', False)
        cfg = _defaults(mode, film, ['--detector', 'det', '--box-prompt', 'firstonly'])
        assert (cfg.box_prompt, cfg.box_prompt_first_only) == ('detector', True)
        assert _defaults(mode, {'box_prompt': 'none'}, ['--detector', 'det']).box_prompt == 'none'
        with pytest.raises(SystemExit, match='no --detector or --boxes'):
            _defaults(mode, film)
