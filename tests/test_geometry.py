"""Production camera geometry uses the complete intrinsic matrix, including skew."""
from __future__ import annotations

import numpy as np
import torch

from tailcyclenet import format as fmt
from tailcyclenet.geometry import undistort_points


def _rig(offset=(0.0, 0.0), distortion=None):
    from aniposelib.cameras import Camera, CameraGroup

    K = np.array([[800.0, 40.0, 300.0], [0.0, 780.0, 220.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float)
    cams = []
    for name, tvec in [('c0', [0.0, 0.0, 0.0]), ('c1', [-1.0, 0.0, 0.0]),
                       ('c2', [0.0, -1.0, 0.0])]:
        cam = Camera(matrix=K, dist=dist, rvec=np.zeros(3), tvec=np.asarray(tvec), name=name)
        cam.set_size((640, 480))
        cams.append(cam)
    names = ('c0', 'c1', 'c2')
    return fmt.Rig(CameraGroup(cams), offset={n: tuple(offset) for n in names},
                   moving={n: False for n in names}, calibrated={n: True for n in names})


def test_skew_projection_inverse_and_rays_round_trip():
    from posetail.posetail.cube import points_to_rays, project_points_torch

    rig = _rig(offset=(7.0, -4.0), distortion=[0.01, -0.001, 0.0002, -0.0001, 0.0003])
    world = torch.tensor([[0.2, -0.1, 10.0], [1.1, 0.3, 12.0]], dtype=torch.float64)
    cam = rig.posetail()[0]
    pixels = project_points_torch([cam], world)[0]
    normalized = undistort_points(cam, pixels)
    # c0 has identity rotation and zero translation, so the independent normalized oracle is
    # simply X/Z. The pixels above include both skew and radial/tangential distortion.
    expected = world[:, :2] / world[:, 2, None]
    torch.testing.assert_close(normalized, expected, rtol=0.0, atol=2e-6)

    rays = points_to_rays(cam, pixels, normalize_t=False)
    direction = world - torch.zeros(3, dtype=torch.float64)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(rays[:, 2, :3], direction, rtol=0.0, atol=2e-5)


def test_skew_projection_sensitivity_matches_autograd():
    from posetail.posetail.cube import project_cam
    from tailcyclenet.geometry import projection_sensitivity

    rig = _rig(offset=(7.0, -4.0), distortion=[0.01, -0.001, 0.0002, -0.0001, 0.0003])
    cam = rig.posetail()[0]
    points = torch.tensor([[0.2, -0.1, 10.0], [1.1, 0.3, 12.0]], dtype=torch.float64)
    got = projection_sensitivity(cam, points)
    for i, point in enumerate(points):
        oracle = torch.autograd.functional.jacobian(lambda x: project_cam(cam, x[None])[0], point)
        torch.testing.assert_close(got[i], oracle, rtol=1e-5, atol=1e-7)

    outside = torch.tensor([[100.0, 0.0, 10.0]], dtype=torch.float64)
    got = projection_sensitivity(cam, outside)
    oracle = torch.autograd.functional.jacobian(
        lambda x: project_cam(cam, x[None])[0], outside[0])
    torch.testing.assert_close(got[0], oracle, rtol=1e-5, atol=1e-7)


def test_skew_image_rotation_keeps_projection_in_warp_frame():
    from posetail.posetail.cube import project_cam
    from tailcyclenet.dataset import rotate_camera_image_plane_3d

    rig = _rig(offset=(7.0, -4.0))
    cam = rig.posetail()[0]
    world = torch.tensor([[0.2, -0.1, 10.0], [1.1, 0.3, 12.0]], dtype=torch.float32)
    before = project_cam(cam, world)
    cam_rot, (matrix, _) = rotate_camera_image_plane_3d(cam, 27.0)
    after = project_cam(cam_rot, world)
    affine = torch.as_tensor(matrix, dtype=before.dtype)
    expected = before @ affine[:, :2].T + affine[:, 2]
    torch.testing.assert_close(after, expected, rtol=0.0, atol=1e-4)
    torch.testing.assert_close(cam_rot['ext'], cam['ext'])


def test_skew_rotate_crop_resize_matches_pixel_affine():
    from posetail.posetail.cube import project_cam
    from tailcyclenet.crop import apply_crop
    from tailcyclenet.dataset import _crop_affine, _resize_camera, rotate_camera_image_plane_3d

    rig = _rig(offset=(7.0, -4.0))
    cam = rig.posetail()[0]
    world = torch.tensor([[0.2, -0.1, 10.0], [1.1, 0.3, 12.0]], dtype=torch.float32)
    rotated, rotation = rotate_camera_image_plane_3d(cam, 27.0)
    w, h = (int(x) for x in rotated['size'])
    box = torch.tensor([10, 8, w - 10, h - 8], dtype=torch.int32)
    cropped = apply_crop(rotated, box)
    resized, _ = _resize_camera(cropped, 128)
    affine, _ = _crop_affine(cam['size'].tolist(), box, resized['size'].tolist(), rotation)
    affine = torch.as_tensor(affine, dtype=world.dtype)
    expected = project_cam(cam, world) @ affine[:, :2].T + affine[:, 2]
    got = project_cam(resized, world)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=2e-4)


def test_image_rotation_supports_anisotropic_focal_and_tangential_distortion():
    from posetail.posetail.cube import project_cam
    from tailcyclenet.dataset import rotate_camera_image_plane_3d

    cam = _rig(offset=(7.0, -4.0)).posetail()[0]
    cam['mat'] = cam['mat'].clone()
    cam['mat'][0, 0], cam['mat'][0, 1] = 900.0, 0.0
    cam['dist'] = torch.tensor([0.01, -0.001, 0.0002, -0.0001, 0.0003])
    world = torch.tensor([[0.2, -0.1, 10.0]], dtype=torch.float32)
    before = project_cam(cam, world)
    cam_rot, (matrix, _) = rotate_camera_image_plane_3d(cam, 27.0)
    affine = torch.as_tensor(matrix, dtype=before.dtype)
    expected = before @ affine[:, :2].T + affine[:, 2]
    torch.testing.assert_close(project_cam(cam_rot, world), expected, rtol=0.0, atol=2e-4)


def test_skew_detector_association_triangulates_with_full_intrinsics():
    from posetail.posetail.cube import project_points_torch
    from tailcyclenet.detector.associate import _triangulate

    rig = _rig(offset=(7.0, -4.0))
    cameras = rig.posetail()
    world = torch.tensor([0.2, -0.1, 10.0], dtype=torch.float64)
    pixels = project_points_torch(cameras, world[None])[:, 0]
    got = _triangulate(cameras, (0, 1, 2), pixels)
    torch.testing.assert_close(got, world.float(), rtol=0.0, atol=2e-5)


def test_all_production_inverse_aliases_use_full_intrinsics():
    import posetail.posetail.tracker_encoder as tracker_encoder
    import posetail.posetail.cube as cube

    assert cube.undistort_points is undistort_points
    assert tracker_encoder.undistort_points is undistort_points
