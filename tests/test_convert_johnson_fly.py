"""Focused geometry and parser checks for the Fly50 telecentric conversion."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet import format as fmt


@pytest.fixture(scope='module')
def converter():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'convert_johnson_fly.py'
    spec = importlib.util.spec_from_file_location('convert_johnson_fly_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _skew_rig(offset=(0.0, 0.0), distortion=None):
    from aniposelib.cameras import Camera, CameraGroup

    K = np.array([[800.0, 40.0, 320.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float)
    cams = []
    for name, tvec in [('c0', [0.0, 0.0, 0.0]),
                       ('c1', [-1.0, 0.0, 0.0]),
                       ('c2', [0.0, -1.0, 0.0])]:
        cam = Camera(matrix=K, dist=dist, rvec=np.zeros(3), tvec=np.asarray(tvec), name=name)
        cam.set_size((640, 480))
        cams.append(cam)
    return fmt.Rig(
        CameraGroup(cams),
        offset={name: tuple(offset) for name in ('c0', 'c1', 'c2')},
        moving={name: False for name in ('c0', 'c1', 'c2')},
        calibrated={name: True for name in ('c0', 'c1', 'c2')},
    )


def test_parse_projection_accepts_integer_decimal_and_scientific_values(converter, tmp_path):
    path = tmp_path / 'cam.yaml'
    path.write_text("""projectionMatrix:
  rows: 3
  cols: 4
  data: [8e2, 4.0e1, 3.2e2, -1, 0, 7.8E2, +2.4e2, 3.5, 0, 0, 0, 1]
""")
    got = converter.parse_projection(path)
    assert got.shape == (3, 4)
    np.testing.assert_array_equal(got, [[800, 40, 320, -1], [0, 780, 240, 3.5], [0, 0, 0, 1]])


def test_skew_triangulation_handles_view_counts_offsets_and_round_trip(converter, tmp_path):
    rig = _skew_rig(offset=(7.0, -4.0))
    from posetail.posetail.cube import project_points_torch

    world = np.array([[0.2, -0.1, 10.0], [1.1, 0.3, 12.0], [-0.7, 0.5, 8.0]], dtype=np.float64)
    observed = project_points_torch(
        rig.posetail(), torch.as_tensor(world, dtype=torch.float64)).detach().cpu().numpy()

    got = converter._triangulate_skew_aware(rig, observed)
    np.testing.assert_allclose(got, world, rtol=0.0, atol=1e-8)

    two_view = observed.copy()
    two_view[2, 1] = np.nan
    got_two = converter._triangulate_skew_aware(rig, two_view)
    np.testing.assert_allclose(got_two[1], world[1], rtol=0.0, atol=1e-8)

    one_view = observed.copy()
    one_view[1:, 2] = np.nan
    got_one = converter._triangulate_skew_aware(rig, one_view)
    assert np.isnan(got_one[2]).all()

    zero_view = observed.copy()
    zero_view[:, 2] = np.nan
    got_zero = converter._triangulate_skew_aware(rig, zero_view)
    assert np.isnan(got_zero[2]).all()

    calibration = tmp_path / 'calibration.toml'
    fmt.dump_calibration(calibration, rig)
    reread = fmt.load_calibration(calibration)
    np.testing.assert_array_equal(reread.by_name('c0').matrix.detach().cpu().numpy(),
                                  rig.by_name('c0').matrix.detach().cpu().numpy())
    assert reread.offset['c0'] == (7.0, -4.0)


def test_skew_triangulation_rejects_nonzero_distortion(converter):
    rig = _skew_rig(distortion=[0.01, 0.0, 0.0, 0.0, 0.0])
    points = np.ones((3, 1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match='requires zero distortion'):
        converter._triangulate_skew_aware(rig, points)
