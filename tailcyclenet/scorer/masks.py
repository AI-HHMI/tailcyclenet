"""Which points a corruption actually moved, and which are visible: the triplet's distance masks."""
import torch

from tracktail.model.cube import is_point_visible, project_points_torch


def visible_points_mask(coords, cgroup, mode, cam_thresh):
    """Per-point mask: visible in >= `cam_thresh` cameras in at least one frame.

    Mirrors the reference's, and exists because an in-plane rotation can swing an off-centre track
    fully off-frame, where the crop (clamped to the image) cannot recover it. NaN counts as
    not-visible, so a point survives while it is seen in any un-dropped frame.

    Inputs: coords -- [t,k,R] for ONE member; cgroup -- its cameras; mode -- '2d' or '3d';
            cam_thresh -- how many cameras must see it.
    Outputs: [k] bool, the any-frame reduction of `_frame_visibility`.
    Side effects: none.
    """
    return _frame_visibility(coords[None], cgroup, mode, cam_thresh)[0][0].any(dim=0)


def _frame_visibility(coords, camera_group, mode, cam_thresh=1):
    """Return per-frame visibility and the per-camera visibility matrix.

    ``coords`` is ``[B,T,K,R]`` in the final view coordinate system.  The first result is true
    only when the coordinate is finite and visible in at least ``cam_thresh`` cameras.  Keeping
    the camera matrix as a second result is important for 3-D displacement: a ray-parallel shift
    must be measured only in cameras where the clean point is actually visible.
    """
    if coords.ndim != 4:
        raise ValueError(f'coords must be [B,T,K,R], got {tuple(coords.shape)}')
    finite = torch.isfinite(coords).all(-1)
    if mode == '2d':
        if len(camera_group) != 1:
            raise ValueError('2D scorer triplets require exactly one camera')
        w, h = (float(x) for x in camera_group[0]['size'][:2])
        inside = (finite & (coords[..., 0] >= 0) & (coords[..., 0] < w)
                  & (coords[..., 1] >= 0) & (coords[..., 1] < h))
        return inside, inside.unsqueeze(-1)
    if mode != '3d':
        raise ValueError(f'unknown scorer triplet mode {mode!r}')
    b, t, k, _ = coords.shape
    flat = coords.reshape(-1, coords.shape[-1])
    cams = []
    for cam in camera_group:
        vis = is_point_visible(cam, flat).reshape(b, t, k)
        cams.append(vis & finite)
    if not cams:
        per_cam = torch.zeros((*finite.shape, 0), dtype=torch.bool, device=coords.device)
    else:
        per_cam = torch.stack(cams, dim=-1)
    return per_cam.sum(-1) >= int(cam_thresh), per_cam


def _projected_displacement(good, bad, camera_group, good_per_camera=None):
    """Image-plane displacement of ``good`` to ``bad`` in final-crop pixels.

    The scalar is the maximum over cameras on which the clean point is visible.  The companion
    mean is useful for a census but is not used as the active gate.  No visible camera yields NaN;
    callers keep that slot out of all masks rather than silently treating missing geometry as a
    zero displacement.
    """
    if good.shape != bad.shape or good.ndim != 4:
        raise ValueError('good and bad must have identical [B,T,K,R] shapes')
    if good_per_camera is None:
        _, good_per_camera = _frame_visibility(good, camera_group, '3d')
    gp = project_points_torch(camera_group, good)
    bp = project_points_torch(camera_group, bad)
    finite = torch.isfinite(gp).all(-1) & torch.isfinite(bp).all(-1)
    d = torch.linalg.vector_norm(bp - gp, dim=-1)
    use = good_per_camera.permute(3, 0, 1, 2) & finite
    neg_inf = torch.full_like(d, -torch.inf)
    max_d = torch.where(use, d, neg_inf).amax(dim=0)
    sum_d = torch.where(use, d, torch.zeros_like(d)).sum(dim=0)
    n = use.sum(dim=0)
    max_d = torch.where(n > 0, max_d, torch.full_like(max_d, float('nan')))
    mean_d = torch.where(n > 0, sum_d / n.clamp_min(1), torch.full_like(sum_d, float('nan')))
    return max_d, mean_d, gp, bp, good_per_camera


def _distance_2d(a, b):
    """Pixel distance for two final-view 2-D coordinate tensors, preserving NaN slots."""
    finite = torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
    d = torch.linalg.vector_norm(b - a, dim=-1)
    return torch.where(finite, d, torch.full_like(d, float('nan')))


def displacement_masks(good, bad, anchor, camera_group, mode, cfg=None, *,
                       anchor_camera_group=None, moved_mask=None, cam_thresh=1):
    """Build framewise displacement, visibility, reference and supervision masks.

    This is the data-side contract consumed by :class:`FrameTripletScorerLoss`.  All returned
    masks have ``[B,T,K]`` shape (except the per-camera diagnostic tensors), and all thresholds are
    in final-crop pixels.  In 3-D the displacement is the maximum selected-camera reprojection
    distance over cameras where the clean point is visible; in 2-D it is the direct pixel norm.

    ``reference_gate='source_far'`` deliberately uses the stored clean point as a proxy and is
    reported in ``reference_source``; it is not a claim that the stored track is ground truth.
    ``reference_gate='none'`` leaves the base far state alone.

    Outputs include ``base_far_mask`` (before reference gating), ``reference_far_mask``,
    ``reference_rejected_mask``, ``far_mask``, ``near_mask``, ``ambiguous_mask``, ``active_mask``,
    ``observed_mask``, ``anchor_observed_mask``, ``in_view_mask``, ``moved_mask``,
    ``displacement_px`` and ``displacement_mean_px``.  The result also carries reference distance
    diagnostics and the gate/source names as scalar Python metadata.
    """
    cfg = {} if cfg is None else dict(cfg)
    if good.ndim != 4 or bad.shape != good.shape or anchor.ndim != good.ndim:
        raise ValueError('good, bad and anchor must be [B,T,K,R] tensors of matching shape')
    if anchor.shape != good.shape:
        raise ValueError(
            f'anchor shape {tuple(anchor.shape)} must match good {tuple(good.shape)}')
    anchor_camera_group = camera_group if anchor_camera_group is None else anchor_camera_group
    mode = str(mode)
    obs = torch.isfinite(good).all(-1) & torch.isfinite(bad).all(-1)
    anchor_obs = torch.isfinite(anchor).all(-1)
    good_view, good_per_cam = _frame_visibility(good, camera_group, mode, cam_thresh)
    bad_view, _ = _frame_visibility(bad, camera_group, mode, cam_thresh)
    anchor_view, _ = _frame_visibility(anchor, anchor_camera_group, mode, cam_thresh)
    in_view = good_view & bad_view & anchor_view

    if mode == '2d':
        displacement = _distance_2d(good, bad)
        displacement_mean = displacement.clone()
    elif mode == '3d':
        displacement, displacement_mean, _, _, good_per_cam = _projected_displacement(
            good, bad, camera_group, good_per_cam)
    else:
        raise ValueError(f'unknown scorer triplet mode {mode!r}')

    finite_d = torch.isfinite(displacement)
    if moved_mask is None:
        moved = finite_d & (displacement > 0)
    else:
        moved = moved_mask.to(device=good.device, dtype=torch.bool) & finite_d
    min_px = float(cfg.get('min_corrupt_px', 0.0))
    max_px = float(cfg.get('max_clean_px', 0.0))
    if (min_px < 0 or max_px < 0 or (min_px > 0 and min_px <= max_px)
            or (min_px == 0 and max_px > 0)):
        raise ValueError(
            f'distance thresholds must satisfy 0 <= max_clean_px < min_corrupt_px when a '
            f'far threshold is configured, or both zero when using moved-mask fallback; '
            f'got min={min_px}, max={max_px}')
    base_far = finite_d & moved & (displacement >= min_px if min_px > 0 else moved)
    near = finite_d & (displacement <= max_px)
    ambiguous = finite_d & ~base_far & ~near

    reference_gate = str(cfg.get('reference_gate', 'source_far'))
    reference_source = 'none'
    good_ref = bad_ref = None
    if reference_gate == 'none':
        reference_far = torch.ones_like(base_far)
    elif reference_gate == 'source_far':
        reference_source = 'stored_good_proxy'
        good_ref = torch.zeros_like(displacement)
        bad_ref = displacement.clone()
        reference_far = finite_d & moved & (bad_ref >= min_px if min_px > 0 else moved)
    else:
        raise ValueError(
            f'unknown reference_gate {reference_gate!r}; expected none or source_far')

    reference_rejected = base_far & ~reference_far
    far = base_far & reference_far
    active = far & obs & in_view & anchor_obs
    out = {
        'observed_mask': obs,
        'anchor_observed_mask': anchor_obs,
        'in_view_mask': in_view,
        'moved_mask': moved if moved_mask is None else moved_mask.to(torch.bool),
        'displacement_px': displacement,
        'displacement_mean_px': displacement_mean,
        'base_far_mask': base_far,
        'reference_far_mask': reference_far,
        'reference_rejected_mask': reference_rejected,
        'far_mask': far,
        'near_mask': near,
        'ambiguous_mask': ambiguous,
        'active_mask': active,
        'reference_gate': reference_gate,
        'reference_source': reference_source,
        'good_reference_distance_px': good_ref,
        'bad_reference_distance_px': bad_ref,
    }
    return out
