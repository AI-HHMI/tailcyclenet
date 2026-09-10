"""Full-intrinsic camera geometry compatibility for the pinned posetail release.

The installed posetail forward projection already multiplies by the full upper-triangular
intrinsic matrix. Its inverse path historically extracted only fx/fy/cx/cy, however, which makes
nonzero-skew cameras inconsistent. This module preserves the old path for zero-skew cameras and
replaces only the inverse normalization for skewed cameras.
"""
from __future__ import annotations

import sys

import torch

from posetail.posetail import cube


if not hasattr(cube, '_tailcyclenet_scalar_undistort_points'):
    cube._tailcyclenet_scalar_undistort_points = cube.undistort_points
if not hasattr(cube, '_tailcyclenet_scalar_projection_sensitivity'):
    cube._tailcyclenet_scalar_projection_sensitivity = cube.projection_sensitivity


def _has_skew(cam) -> bool:
    """Whether the camera's 2x2 image transform has an off-diagonal term."""
    matrix = cam['mat']
    return bool(torch.any(matrix[:2, :2].abs().masked_fill(
        torch.eye(2, dtype=torch.bool, device=matrix.device), 0.0) != 0))


def undistort_points(cam, points):
    """Convert pixels to normalized coordinates using full K when skew is nonzero.

    `points` are stored-image pixels; camera offsets are added before normalization. Distortion is
    inverted in normalized coordinates using the same five-iteration model as posetail. Zero-skew
    cameras delegate to the installed implementation for exact compatibility.
    """
    matrix = cam['mat']
    if not _has_skew(cam):
        return cube._tailcyclenet_scalar_undistort_points(cam, points)
    if cam.get('type', 'pinhole') != 'pinhole':
        raise ValueError('full-skew inverse geometry supports pinhole cameras only')

    output_dtype = points.dtype
    offset = cam.get('offset')
    if offset is None:
        offset = points.new_zeros(2)
    if offset.ndim > 1:
        aligned = cube._align_offset(offset, points)
        points = points.to(torch.float64) + aligned.to(torch.float64)
        offset = points.new_zeros(2)

    shape = points.shape
    points = points.reshape(-1, 2)
    K = matrix.to(dtype=torch.float64)
    sensor = points.to(torch.float64) + offset.to(torch.float64)
    homogeneous = torch.cat([sensor, sensor.new_ones((len(sensor), 1))], dim=1)
    ray = homogeneous @ torch.linalg.inv(K).t()
    normalized = ray[:, :2] / ray[:, 2, None]
    x0, y0 = normalized[:, 0], normalized[:, 1]
    x, y = x0.clone(), y0.clone()
    dist = cam['dist'].to(torch.float64)
    for _ in range(5):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        k1, k2, p1, p2 = dist[:4]
        k3 = dist[4] if dist.shape[0] > 4 else dist.new_zeros(())
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial
    return torch.stack([x, y], dim=1).reshape(shape).to(output_dtype)


def projection_sensitivity(cam, points):
    """Jacobian of full-K Brown pinhole projection, preserving zero-skew compatibility."""
    if not _has_skew(cam):
        return cube._tailcyclenet_scalar_projection_sensitivity(cam, points)
    if cam.get('type', 'pinhole') != 'pinhole':
        raise ValueError('full-skew projection sensitivity supports pinhole cameras only')

    p = points.to(torch.float64)
    ext = cam['ext'].to(torch.float64)
    if ext.ndim == 3:
        ext = ext[0]
    p_cam = torch.matmul(torch.cat([p, torch.ones_like(p[:, :1])], dim=1), ext.t())[:, :3]
    X, Y, Z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    q = p_cam[:, :2] / Z[:, None]
    q_raw = q
    q = q.clamp(-3.0, 3.0)
    x, y = q[:, 0], q[:, 1]
    dist = cam['dist'].to(torch.float64)
    k1, k2, p1, p2 = dist[:4]
    k3 = dist[4] if dist.shape[0] > 4 else dist.new_zeros(())
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    dr = (k1 + 2 * k2 * r2 + 3 * k3 * r4) * 2
    drdx, drdy = dr * x, dr * y
    dtxdx, dtxdy = 2 * p1 * y + 6 * p2 * x, 2 * p1 * x + 2 * p2 * y
    dtydx, dtydy = 2 * p1 * x + 2 * p2 * y, 6 * p1 * y + 2 * p2 * x
    J_dist = torch.stack([
        torch.stack([radial + x * drdx + dtxdx, x * drdy + dtxdy], dim=-1),
        torch.stack([y * drdx + dtydx, radial + y * drdy + dtydy], dim=-1),
    ], dim=1)
    inside = (q_raw.abs() < 3.0).to(J_dist.dtype)
    # `inside` indexes the normalized-coordinate input columns (x and y), not output rows.
    J_dist = J_dist * inside[:, None, :]
    J_q = torch.zeros((len(p), 2, 3), dtype=torch.float64, device=p.device)
    J_q[:, 0, 0] = 1 / Z
    J_q[:, 0, 2] = -X / Z.square()
    J_q[:, 1, 1] = 1 / Z
    J_q[:, 1, 2] = -Y / Z.square()
    J_cam = torch.matmul(cam['mat'].to(torch.float64)[:2, :2],
                         torch.matmul(J_dist, J_q))
    return torch.matmul(J_cam, ext[:3, :3])


def install_full_intrinsics() -> None:
    """Install the skew-aware inverse at posetail's shared geometry boundary.

    `TrackerEncoder` imported `undistort_points` into its module namespace, so update that alias
    when the module is already loaded. The shared `points_to_rays` function resolves its inverse
    through the cube module and therefore sees the replacement automatically.
    """
    cube.undistort_points = undistort_points
    cube.projection_sensitivity = projection_sensitivity
    tracker = sys.modules.get('posetail.posetail.tracker_encoder')
    if tracker is not None:
        tracker.undistort_points = undistort_points
    tapnext = sys.modules.get('posetail.posetail.tracker_tapnext')
    if tapnext is not None:
        tapnext.undistort_points = undistort_points


install_full_intrinsics()
