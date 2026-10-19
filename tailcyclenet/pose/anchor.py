"""Where the pose model's 3D output is anchored: the query-free scene point and the residual offset.

`gridresid` reconstructs `world = query + R @ residual` with ONE anchor per window. `query_anchored`
keeps that anchor where a real prior anchored it; `reanchor_per_frame` re-adds the residual to each
frame's own (detached) triangulation (`[model].gridresid_offset`).
"""
import torch
from einops import einsum, repeat

from tracktail.model.cube import from_homogeneous, to_homogeneous, undistort_points


def scene_center(camera_group):
    """The 3D point every camera's crop centre looks at. (3,) float32.

    Deployment has no ground truth, so derive it from the CAMERAS: each crop centre back-projects
    to a ray and the point closest to all of them is the scene centre -- gauge-correct by
    construction. On a MOVING rig every frame contributes its own ray: a moving crop gives every
    frame its own ray, the one place a per-frame offset must not be collapsed.

    The solve is a least-squares point minimising distance to a set of lines,
    (sum_i P_i) x = sum_i P_i c_i, with a ridge term that keeps a near-degenerate rig from
    producing a wild centre -- ~1e-6 of the trace, so it never moves a good solve.
    """
    origins, dirs = [], []
    for cam in camera_group:
        px = (cam['size'].to(torch.float64) / 2.0).reshape(1, 2)
        off = cam['offset']
        und = [undistort_points(dict(cam, offset=o), px)
               for o in (off if off.ndim == 2 else off[None])]
        ext = cam['ext'] if cam['ext'].ndim == 3 else cam['ext'][None]
        centre = cam['center'] if cam['center'].ndim == 2 else cam['center'][None]
        for t in range(max(ext.shape[0], len(und))):
            u = und[min(t, len(und) - 1)]
            ray_cam = torch.cat([u[0], u.new_ones(1)])
            d = ext[min(t, ext.shape[0] - 1), :3, :3].to(torch.float64).t() @ ray_cam
            dirs.append(d / d.norm())
            origins.append(centre[min(t, centre.shape[0] - 1)].to(torch.float64))
    origins, dirs = torch.stack(origins), torch.stack(dirs)

    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    P = eye - dirs[:, :, None] * dirs[:, None, :]
    A = P.sum(0)
    b = torch.einsum('cij,cj->i', P, origins)
    A = A + eye * (1e-6 * torch.diagonal(A).abs().sum().clamp_min(1.0))
    return torch.linalg.solve(A, b).to(torch.float32)


def uncalibrated_singleview_anchor(cam):
    """Put a query-free nominal camera anchor one arbitrary unit along its crop-centre ray.

    A single camera's minimum-norm scene centre is its optical-ray point nearest the origin. For
    an uncalibrated identity camera that is the camera centre itself, where projection sensitivity
    is undefined and tracktail's ``cube_scale`` becomes NaN. The one-unit offset supplies a finite
    gauge for inference only; it cannot recover metric depth or world scale.
    """
    size = cam['size'].to(torch.float64)
    px = (size / 2.0).reshape(1, 2)
    offset = cam['offset']
    offsets = offset if offset.ndim == 2 else offset[None]
    und = [undistort_points(dict(cam, offset=o), px)[0] for o in offsets]
    ext = cam['ext'] if cam['ext'].ndim == 3 else cam['ext'][None]
    center = cam['center'] if cam['center'].ndim == 2 else cam['center'][None]
    ray_cam = torch.cat([und[0], und[0].new_ones(1)])
    direction = ext[0, :3, :3].to(torch.float64).t() @ ray_cam
    direction = direction / direction.norm().clamp_min(1e-12)
    return center[0].to(torch.float64) + direction


def _rays_fallback(out):
    """The conf-weighted MEAN of the per-camera ray points -- the honest version of `3d_pred_rays`.

    The library's is a weighted SUM with no division (an unnormalised sigmoid), so at one camera
    it lands about halfway from the world origin to the animal. Used wherever a triangulation is
    missing or degenerate.
    """
    w = torch.sigmoid(out['conf_pred_2d'])
    num = einsum(out['3d_pred_cams_rays'], w, 'c b t n r, c b t n -> b t n r')
    return num / w.sum(0)[..., None].clamp_min(1e-6)


def query_anchored(out, query_ok):
    """gridresid is an OFFSET FROM THE QUERY POINT, so honour it only where the query is real.

    The library reconstructs `world = query_world + R @ residual` with ONE anchor per window;
    that is right for a genuine prior and wrong for the derived scene centre (identical for
    every keypoint, carries no information). So the prediction is query-anchored WHERE a prior
    exists and each frame's own triangulation elsewhere -- the DETACHED substitution is also the
    loss gate: the direct terms then see a constant at those points and supervise query points
    only.

    In 3D SINGLE-VIEW no triangulation exists, so the back-projected rays are the only anchor
    there is (`cams_to_sample = [1, 8]` draws one camera). The substitution repairs FIRST so the loss
    and the metric see ONE tensor. A non-finite entry also poisons the solve's BACKWARD (the NaN
    is created inside `torch.linalg.solve`'s factorisation, downstream of any forward-value fix),
    so on a degenerate step the triangulation supervision goes gradient-free; the forward value
    is unchanged. The repaired triangulation is not written back on the single-view path: that
    key drives `coords_loss_triangulate_reproj` at weight 2.0, and pointing it at the rays would
    reweight that supervision by 40x. Degeneracy is recorded, not gated: a seed taken from
    `_rays_fallback` is a point no camera claimed, and the caller decides what a degenerate frame
    is worth.

    When no keypoint has a valid query there is nothing to supervise the 3D grid CE with --
    every point is triangulated. That term is killed and nothing else: popping `grid` would also
    drop `depth_softmax` (weight 1.5, the largest CE term) and `f_eff` behind it, so the anchor
    is NaN'd in place -- a non-finite anchor is the library's own off switch.
    """
    tri = out.get('3d_pred_triangulate')
    out = dict(out)
    if tri is None:
        sub = _rays_fallback(out)
    else:
        bad = ~torch.isfinite(tri).all(-1)
        if not torch.isfinite(tri).all():
            tri = tri.detach()
        sub = torch.where(torch.isfinite(tri), tri, _rays_fallback(out))
        out['3d_pred_triangulate'] = sub
        out['tri_degenerate'] = bad

    m = query_ok[None, :, None, :, None]
    direct = torch.where(m, out['3d_pred_cams_direct'], sub.detach()[None])
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', direct, conf)
    out['3d_pred_cams_direct'] = direct
    out['3d_pred_direct'] = coords_pred
    out['coords_pred'] = coords_pred

    if not bool(query_ok.any()) and out.get('grid') is not None:
        grid = dict(out['grid'])
        grid['anchor_local'] = torch.full_like(grid['anchor_local'], float('nan'))
        out['grid'] = grid
    return out


def reanchor_per_frame(out, anchor):
    """Re-anchor the 3D residual on EACH frame's own triangulation. `3d_pred_cams_direct =
    query + R @ residual`, so subtracting the query recovers the world-space residual, re-added
    to each frame's own (detached) triangulation.

    The substitution repairs FIRST so the loss and the metric see ONE tensor -- see
    `query_anchored` for the backward-poisoning reason the detach happens (the solve's backward
    NaNs).
    """
    src = out['3d_pred_triangulate']
    bad = ~torch.isfinite(src).all(-1)
    if not torch.isfinite(src).all():
        src = src.detach()
    src = torch.where(torch.isfinite(src), src, _rays_fallback(out))
    out = dict(out)
    out['tri_degenerate'] = bad

    residual = out['3d_pred_cams_direct'] - anchor[:, None, :, :][None]
    new_direct = src.detach()[None] + residual
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', new_direct, conf)

    out['3d_pred_cams_direct'] = new_direct
    out['3d_pred_direct'] = coords_pred
    out['coords_pred'] = coords_pred
    out['3d_pred_triangulate'] = src
    return _reanchor_ce_target(out, src)


def _reanchor_ce_target(out, src):
    """Move the 3D cross-entropy target onto the SAME anchor the outputs now use. Without this,
    `grid['anchor_local']` still describes the query-anchored task while every metric sees
    per-frame refinement. float64 is mandatory: the library's einsum cancels exactly only in
    float64, where the ray-local coordinates are large and nearly equal.
    """
    grid = out.get('grid')
    if grid is None or grid.get('anchor_local') is None or grid.get('rays_c') is None:
        return out
    n_cams = grid['anchor_local'].shape[0]
    aw = repeat(src, 'b t n r -> cams b t n r', cams=n_cams)
    anchor_local = from_homogeneous(einsum(
        grid['rays_c'].to(torch.float64), to_homogeneous(aw.to(torch.float64)),
        'cams x r, cams b t n r -> cams b t n x'))
    grid = dict(grid)
    grid['anchor_local'] = anchor_local.to(grid['anchor_local'].dtype).detach()
    out['grid'] = grid
    return out
