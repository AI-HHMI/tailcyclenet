"""Opt-in 2D reflection augmentation tests."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tailcyclenet import format as fmt
from tailcyclenet.dataset import LoaderConfig, PoseDataset, _flip_2d
from tailcyclenet.scorer.triplet import _make_anchor


def _cfg(**over):
    values = dict(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                  crop_jitter=0.0, crop_inflate=1.0)
    values.update(over)
    return LoaderConfig(**values)


def _replace_flip_pairs(path, value):
    lines = path.read_text().splitlines()
    out = [line for line in lines if value is not None or not line.startswith('flip_pairs = ')]
    if value is not None:
        out = [f'flip_pairs = {value}' if line.startswith('flip_pairs = ') else line
               for line in out]
    path.write_text('\n'.join(out) + '\n')


def test_flip_config_validation():
    assert _cfg(flip_2d_prob=0.0).flip_2d_modes == ['horizontal']
    assert _cfg(flip_2d_prob=0.0, flip_2d_modes=[]).flip_2d_modes == []
    assert _cfg(flip_2d_prob=0.0, flip_2d_modes=('horizontal',)).flip_2d_modes == ['horizontal']
    with pytest.raises(AssertionError, match='flip_2d_modes.*non-empty'):
        _cfg(flip_2d_prob=0.5, flip_2d_modes=[])
    with pytest.raises(ValueError, match=r'flip_2d_prob.*\[0, 1\]'):
        _cfg(flip_2d_prob=1.1)
    with pytest.raises(ValueError, match='flip_2d_modes.*list'):
        _cfg(flip_2d_modes='horizontal')
    with pytest.raises(ValueError, match='only'):
        _cfg(flip_2d_modes=['diagonal'])
    with pytest.raises(ValueError, match='duplicate'):
        _cfg(flip_2d_modes=['horizontal', 'horizontal'])


def _camera(size=(64, 48), offset=(0.0, 0.0), dist=None):
    from aniposelib.cameras import CameraGroup

    cam = fmt.nominal_camera('cam0', size, dist=dist)
    rig = fmt.Rig(CameraGroup([cam]), {'cam0': offset}, {'cam0': False},
                  {'cam0': True})
    return rig.posetail()[0]


@pytest.mark.parametrize('mode', ['horizontal', 'vertical'])
def test_flip_affine_matches_pixels_and_projection(mode):
    import cv2
    from posetail.posetail.cube import project_points_torch
    from tailcyclenet.dataset import _crop_affine

    W, H = 64, 48
    cam = _camera(offset=(4.0, 3.0),
                  dist=np.array([0.01, -0.002, 0.003, 0.004, 0.0001]))
    points = torch.tensor([[[0.4, 0.2, 4.0]]])
    source = project_points_torch([cam], points)[0]
    perm = torch.arange(1)
    flipped_cam, flipped, rotation = _flip_2d(cam, source, mode, None, perm)
    expected = source.clone()
    if mode == 'horizontal':
        expected[..., 0] = W - 1 - expected[..., 0]
    else:
        expected[..., 1] = H - 1 - expected[..., 1]
    torch.testing.assert_close(flipped, expected)
    torch.testing.assert_close(project_points_torch([flipped_cam], points)[0], expected)

    y, x = np.mgrid[:H, :W]
    marker = np.stack([(x * 3 + y * 5) % 256, (x * 7 + y * 11) % 256,
                       (x + 2 * y) % 256], -1).astype(np.uint8)
    matrix, out_size = _crop_affine((W, H), None, (W, H), rotation)
    got = cv2.warpAffine(marker, matrix, out_size, flags=cv2.INTER_LINEAR)
    want = cv2.flip(marker, 1 if mode == 'horizontal' else 0)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('mode', ['horizontal', 'vertical'])
def test_flip_permutates_labels_visibility_and_view_pixels(tiny_root, mode):
    plain_cfg = _cfg(flip_2d_prob=0.0)
    flip_cfg = _cfg(flip_2d_prob=1.0, flip_2d_modes=[mode])
    plain_ds = PoseDataset(tiny_root / 'ratlike', 'train', plain_cfg, train=True, seed=3)
    flip_ds = PoseDataset(tiny_root / 'ratlike', 'train', flip_cfg, train=True, seed=3)

    sel_plain = plain_ds._select(0, np.random.default_rng(17),
                                 plain_ds._shape(np.random.default_rng(19)))
    sel_flip = flip_ds._select(0, np.random.default_rng(17),
                               flip_ds._shape(np.random.default_rng(19)))
    assert sel_plain is not None and sel_flip is not None
    source_before = sel_flip.coords.clone()
    view_plain = plain_ds._realise(sel_plain, np.random.default_rng(23))
    view_flip = flip_ds._realise(sel_flip, np.random.default_rng(23))
    assert view_plain is not None and view_flip is not None

    perm = torch.tensor([0, 2, 1, 3])
    expected = view_plain.coords.clone()
    expected[..., 0 if mode == 'horizontal' else 1] = (
        (63 if mode == 'horizontal' else 47)
        - expected[..., 0 if mode == 'horizontal' else 1])
    expected = expected[:, perm]
    torch.testing.assert_close(view_flip.coords, expected, equal_nan=True)
    assert view_flip.kpt_perm is not None
    assert torch.equal(view_flip.kpt_perm, perm)
    if view_plain.vis_2d is not None:
        torch.testing.assert_close(view_flip.vis_2d, view_plain.vis_2d[:, perm], equal_nan=True)
    np.testing.assert_array_equal(view_flip.views[0].numpy(),
                                  np.flip(view_plain.views[0].numpy(),
                                          axis=2 if mode == 'horizontal' else 1))
    torch.testing.assert_close(sel_flip.coords, source_before, equal_nan=True)
    assert sel_flip.sess.flip_pairs == [['left_ear', 'right_ear']]


def test_flip_preflight_requires_declaration_and_accepts_explicit_empty(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    absent = tmp_path / 'absent' / 'train' / 's'
    C._session_2d(absent)
    session_toml = absent / 'session.toml'
    _replace_flip_pairs(session_toml, None)
    with pytest.raises(ValueError, match='requires a declared flip_pairs key'):
        PoseDataset(absent.parent.parent, 'train', _cfg(flip_2d_prob=1.0), train=True)

    explicit = tmp_path / 'explicit' / 'train' / 's'
    C._session_2d(explicit)
    session_toml = explicit / 'session.toml'
    _replace_flip_pairs(session_toml, '[]')
    explicit_loaded = fmt.Session.load(explicit)
    assert explicit_loaded.flip_pairs_declared and explicit_loaded.flip_pairs == []
    ds = PoseDataset(explicit.parent.parent, 'train', _cfg(flip_2d_prob=1.0), train=True)
    assert ds._flip_perm


def test_forced_2d_flip_handles_moving_camera(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    session = tmp_path / 'paired' / 'train' / 'moving'
    C._session_3d(session, moving=True,
                   names=['nose', 'left_ear', 'right_ear'],
                   flip_pairs=[['left_ear', 'right_ear']])
    ds = PoseDataset(session.parent.parent, 'train',
                     _cfg(flip_2d_prob=1.0, flip_2d_modes=['vertical'], prob_2d_only=1.0),
                     train=True)
    sel = ds._select(0, np.random.default_rng(4), ds._shape(np.random.default_rng(5)))
    assert sel is not None and sel.true_2d
    view = ds._realise(sel, np.random.default_rng(6))
    assert view is not None and view.r == 2
    assert torch.equal(view.kpt_perm, torch.tensor([0, 2, 1]))
    assert view.coords.shape[-1] == 2
    assert view.cgroup[0]['ext'].ndim == 3


def test_flip_preflight_is_not_needed_for_3d_when_prob_2d_only_is_zero(tiny_root):
    ds = PoseDataset(tiny_root / 'mouselike', 'train',
                     _cfg(flip_2d_prob=1.0, prob_2d_only=0.0), train=True)
    assert ds._flip_perm == {}
    sel = ds._select(0, np.random.default_rng(1), ds._shape(np.random.default_rng(2)))
    assert sel is not None and not sel.true_2d


def test_flip_rng_draws_are_pinned(tiny_root):
    cfg = _cfg(flip_2d_prob=1.0, flip_2d_modes=['horizontal'])
    ds = PoseDataset(tiny_root / 'ratlike', 'train', cfg, train=True)
    sel = ds._select(0, np.random.default_rng(8), ds._shape(np.random.default_rng(9)))
    assert sel is not None
    rng = np.random.default_rng(10)
    assert ds._realise(sel, rng) is not None
    expected = np.random.default_rng(10)
    expected.random()                 # existing rotation coin
    expected.random()                 # flip coin
    expected.integers(1)              # mode index, even for one mode
    assert rng.integers(2**31) == expected.integers(2**31)

    cfg0 = _cfg(flip_2d_prob=0.0, flip_2d_modes=[])
    ds0 = PoseDataset(tiny_root / 'ratlike', 'train', cfg0, train=True)
    rng0 = np.random.default_rng(10)
    assert ds0._realise(sel, rng0) is not None
    expected0 = np.random.default_rng(10)
    expected0.random()                # existing rotation coin only
    assert rng0.integers(2**31) == expected0.integers(2**31)


def test_format_round_trip_preserves_flip_pair_declaration(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    source = tmp_path / 'source' / 'train' / 's'
    C._session_2d(source)
    session_toml = source / 'session.toml'
    _replace_flip_pairs(session_toml, None)
    absent = fmt.Session.load(source)
    assert not absent.flip_pairs_declared
    out = tmp_path / 'out'
    fmt.write_session(out, mode=absent.mode, units=absent.units,
                      label_source=absent.label_source, names=absent.names, rig=absent.rig,
                      groups=absent.groups, labels={g: absent.labels(g) for g in absent.groups},
                      flip_pairs=absent.flip_pairs if absent.flip_pairs_declared else None)
    assert not fmt.Session.load(out).flip_pairs_declared

    explicit_out = tmp_path / 'explicit_out'
    fmt.write_session(explicit_out, mode=absent.mode, units=absent.units,
                      label_source=absent.label_source, names=absent.names, rig=absent.rig,
                      groups=absent.groups, labels={g: absent.labels(g) for g in absent.groups},
                      flip_pairs=[])
    explicit_round_trip = fmt.Session.load(explicit_out)
    assert explicit_round_trip.flip_pairs_declared and explicit_round_trip.flip_pairs == []


def _fake_view(perm):
    return SimpleNamespace(
        cgroup=[{'size': torch.tensor([100, 100], dtype=torch.int32)}],
        scale=[torch.ones(2)], boxes=[torch.tensor([0, 0, 100, 100], dtype=torch.int32)],
        rotation=[None], kpt_perm=torch.as_tensor(perm, dtype=torch.long))


def test_scorer_anchor_transfer_handles_independent_semantic_flips():
    view_a = _fake_view([0, 2, 1])
    view_b = _fake_view([0, 1, 2])
    source_a = torch.tensor([[[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]])
    anchor = _make_anchor(source_a, 1.0, '2d', view_a, view_b,
                          source_slots=torch.tensor([0, 1, 2]))
    expected = torch.tensor([[[[10.0, 0.0], [30.0, 0.0], [20.0, 0.0]]]])
    torch.testing.assert_close(anchor, expected)
