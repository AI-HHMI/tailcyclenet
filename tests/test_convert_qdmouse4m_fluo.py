"""The camera-axis map of `scripts/convert_qdmouse4m_fluo.py`, end to end.

The bug this pins was silent in both directions: the derived dataset VALIDATED (every camera had
a calibration block and a video) and only the 2D labels were attached to the wrong physical view,
so the failure was a spatially offset overlay rather than an error. The converter expanded the
per-camera arrays as a block copy while the rig INTERLEAVES `<view>` and `<view>_fluo`, so
destination camera `j` received the labels of source camera `j % 6`.

The end-to-end test therefore does not read the converter's own output arrays -- it runs
`convert_session`, RELOADS the derived session from disk, and reprojects: each destination
camera's 2D labels must equal the projection of the 3D layer through THAT camera's own
calibration. A copy taken from the wrong view fails that by hundreds of pixels.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent

SRC = ['a', 'b', 'c']
DST = ['a', 'a_fluo', 'b', 'b_fluo', 'c', 'c_fluo']


@pytest.fixture(scope='module')
def conv():
    spec = importlib.util.spec_from_file_location(
        'tcn_convert_qdmouse4m_fluo', REPO / 'scripts' / 'convert_qdmouse4m_fluo.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# -- the map itself -------------------------------------------------------------------------

def test_interleaved_rig_maps_each_camera_to_its_own_view(conv):
    """The interleaved order must give [0, 0, 1, 1, 2, 2] -- not [0, 1, 2, 0, 1, 2]."""
    assert conv.camera_index_map(SRC, DST) == [0, 0, 1, 1, 2, 2]


def test_a_block_copy_would_have_been_the_bug(conv):
    """Guard the shape of the mistake: the map is NOT the identity, so a naive
    `concatenate([a, a], axis=camera)` cannot produce the correct array."""
    take = conv.camera_index_map(SRC, DST)
    assert take != list(range(len(DST)))


def test_unknown_view_is_loud(conv):
    """A camera with no source view must raise rather than silently pick a neighbour's labels."""
    with pytest.raises(RuntimeError, match='no source camera'):
        conv.camera_index_map(SRC, ['a', 'zzz_fluo'])


def test_taking_the_map_puts_each_camera_on_its_own_projection(conv):
    """End to end on arrays: after `take`, slot j's labels are view j//2's, and the two slots
    of one view are identical (a channel of the same camera sees the same animal)."""
    take = conv.camera_index_map(SRC, DST)
    src = np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3)     # camera axis last
    dst = np.take(src, take, axis=3)
    assert dst.tolist() == [[[[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]]]


# -- the converter, on a session, reloaded --------------------------------------------------

def _source_session(path: Path, T: int = 4) -> None:
    """A three-camera 3D source session whose per-camera 2D is each camera's OWN projection.

    Distinct views are the whole point: a block copy hands a camera another view's projection,
    which is hundreds of pixels away and cannot be confused with a rounding difference. One
    camera-specific MISSING slot and one region per odd camera ride along, so the masks and the
    `regions` camera column are checked too.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    from posetail.posetail.cube import project_cam
    from tailcyclenet import format as fmt

    W, H = 64, 48
    rig = C._rig([(name, W, H, True, False, i + 1) for i, name in enumerate(SRC)])
    names = C.KPTS_3D
    K = len(names)
    lab = fmt.empty_labels(1, T, K, 3, mode3d=True, animal_ids=['m1'])
    lab.points3d[:] = np.random.default_rng(0).uniform(-30, 30, (1, T, K, 3)).astype(np.float32)
    lab.vis3d[:] = fmt.VISIBLE
    xyz = torch.as_tensor(lab.points3d, dtype=torch.float64)
    lab.points2d = np.stack([project_cam(cam, xyz).numpy() for cam in rig.posetail()], axis=3)
    lab.vis2d = np.full(lab.points2d.shape[:-1], fmt.PROJECTED, np.int8)
    lab.vis2d[0, 0, 0, 1] = fmt.MISSING                     # camera 1, frame 0, kpt 0
    lab.points2d[0, 0, 0, 1] = np.nan
    lab.regions = np.array([[0.0, 0.0, 1.0, 1.0, 5.0, 5.0],
                            [0.0, 2.0, 2.0, 2.0, 6.0, 6.0]])
    fmt.write_session(path, mode='3d', units='mm', label_source='annotated', names=names,
                      rig=rig, groups={'g000': fmt.Group('g000', T, fps=100.0)},
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    group = path / 'groups' / 'g000'
    group.mkdir(parents=True, exist_ok=True)
    for name in SRC:                                        # the converter only links these
        (group / f'{name}.mp4').touch()
        (group / f'{name}_fluo.mp4').touch()


@pytest.fixture(scope='module')
def converted(tmp_path_factory, conv):
    """(source session, derived session) after a real `convert_session`."""
    from tailcyclenet import format as fmt

    root = tmp_path_factory.mktemp('fluo_convert')
    source, derived = root / 'src' / 'sess', root / 'out' / 'sess'
    _source_session(source)
    conv.convert_session(source, derived)
    return fmt.Session.load(source), fmt.Session.load(derived)


def test_derived_rig_interleaves_reference_and_fluorescence(converted):
    """The expanded axis is `<view>`, `<view>_fluo`, per view -- the order the map assumes."""
    _source, dst = converted
    assert dst.cam_names == DST


def test_each_reloaded_camera_reprojects_from_its_own_calibration(converted):
    """The independent check: RELOADED labels vs a fresh projection through each camera.

    This is the assertion a block copy cannot pass, and it is run on the written-and-reloaded
    session rather than on the converter's in-memory arrays.
    """
    from posetail.posetail.cube import project_cam

    _source, dst = converted
    lab = dst.labels('g000')
    xyz = torch.as_tensor(np.nan_to_num(lab.points3d), dtype=torch.float64)
    for j, name in enumerate(dst.cam_names):
        got = lab.points2d[0, :, :, j]
        want = project_cam(dst.rig.posetail()[j], xyz)[0].numpy()
        m = np.isfinite(got).all(-1) & np.isfinite(want).all(-1)
        assert m.any(), f'{name}: no finite slot to compare'
        worst = np.abs(got[m] - want[m]).max()
        assert worst < 1e-3, f'{name}: reloaded labels are {worst:.1f} px off its own projection'


def test_the_two_channels_of_a_view_carry_identical_labels(converted):
    """A reference camera and its fluorescence twin see the same animal: same points, same mask."""
    _source, dst = converted
    lab = dst.labels('g000')
    for p in range(len(SRC)):
        a, b = 2 * p, 2 * p + 1
        np.testing.assert_array_equal(lab.vis2d[0, :, :, a], lab.vis2d[0, :, :, b])
        np.testing.assert_array_equal(np.isnan(lab.points2d[0, :, :, a]),
                                      np.isnan(lab.points2d[0, :, :, b]))
        np.testing.assert_allclose(lab.points2d[0, :, :, a], lab.points2d[0, :, :, b],
                                   equal_nan=True)


def test_camera_specific_masks_and_regions_land_on_both_channels(converted):
    """The mask and the `regions` camera column are remapped by the same map as the points."""
    _source, dst = converted
    lab = dst.labels('g000')
    assert lab.vis2d[0, 0, 0, 2] == fmt_missing(), 'the MISSING slot must follow camera 1'
    assert lab.vis2d[0, 0, 0, 3] == fmt_missing()
    assert np.isnan(lab.points2d[0, 0, 0, 2]).all()
    assert lab.vis2d[0, 0, 0, 0] != fmt_missing(), 'camera 0 must stay labeled'
    cams = sorted({int(r[1]) for r in lab.regions})
    assert cams == [0, 1, 4, 5], f'regions must follow both channels of views 0 and 2: {cams}'


def fmt_missing():
    """`fmt.MISSING`, imported lazily so the module stays importable without the package."""
    from tailcyclenet import format as fmt
    return fmt.MISSING
