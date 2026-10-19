"""Moving points between two independent realisations of one selection (the anchor's view B)."""
import numpy as np
import torch


def view_affine_2d(view):
    """The source-pixels -> this-view-pixels affine, as a 3x3 float64 matrix.

    A 2D view's coordinates ARE pixels of its own final crop, so every step of the chain moves
    them. The chain is rotate/reflection -> crop -> resize, and the composed matrix is
    `S(scale) @ T(-box[:2]) @ (F_flip @ R(rot))`. It is built from the pieces the realisation
    ALREADY returned rather than re-derived from the camera dicts: a re-derivation is how two
    frames come to disagree about where the animal is. Semantic keypoint permutations are stored
    separately on `View.kpt_perm`; they cannot be represented by this geometric matrix.

    Inputs: view -- a `View` from a single-camera 2D realisation.
    Outputs: a (3,3) float64 tensor.
    Side effects: none.
    """
    assert len(view.cgroup) == 1, 'a 2D view is single-camera'
    scale = torch.as_tensor(view.scale[0], dtype=torch.float64)
    if scale.ndim == 0:
        scale = scale.repeat(2)
    box = view.boxes[0]
    x1, y1 = float(box[0]), float(box[1])

    H = torch.eye(3, dtype=torch.float64)
    rot = view.rotation[0]
    if rot is not None:
        M = np.asarray(rot[0], dtype=np.float64)
        R = torch.eye(3, dtype=torch.float64)
        R[:2, :2] = torch.as_tensor(M[:, :2])
        R[:2, 2] = torch.as_tensor(M[:, 2])
        H = R @ H
    T = torch.eye(3, dtype=torch.float64)
    T[0, 2] = -x1
    T[1, 2] = -y1
    H = T @ H
    S = torch.eye(3, dtype=torch.float64)
    S[0, 0], S[1, 1] = scale[0], scale[1]
    return S @ H


def transfer_points_2d(pts, H_from, H_to):
    """Move (...,2) pixel points from one 2D view's frame into another's.

    `x_to = H_to @ inv(H_from) @ x_from` -- undo the source view back to source pixels, then apply
    the target view's chain. Both matrices are source->view, so the composition is exact rather
    than approximate. NaN slots stay NaN, which is what keeps a dropped point dropped.

    Inputs: pts -- (...,2) tensor; H_from / H_to -- (3,3) source->view matrices.
    Outputs: (...,2) tensor in the target view's frame.
    Side effects: none.
    """
    M = (H_to @ torch.linalg.inv(H_from)).to(pts.dtype)
    flat = pts.reshape(-1, 2)
    ones = torch.ones((flat.shape[0], 1), dtype=flat.dtype, device=flat.device)
    out = torch.cat([flat, ones], dim=-1) @ M.t()
    return out[:, :2].reshape(pts.shape)


def _view_kpt_permutation(view, n, *, device):
    """Return a validated view-slot -> source-slot permutation.

    ``PoseDataset.View.kpt_perm`` is full-session length even when a triplet later drops keypoints
    that have no finite observation.  Keeping the validation here makes the scorer tolerant of old
    ``View`` objects (where the field is absent/``None``) while refusing a silently misaligned
    semantic axis.

    Inputs: view -- a realised dataset ``View``; n -- expected full keypoint count; device -- the
        device on which to place the returned index tensor.
    Outputs: a long ``[n]`` permutation, identity when the view carries no permutation.
    Side effects: none.
    """
    raw = getattr(view, 'kpt_perm', None)
    if raw is None:
        return torch.arange(n, dtype=torch.long, device=device)
    perm = torch.as_tensor(raw, dtype=torch.long, device=device).reshape(-1)
    if perm.numel() != int(n):
        raise ValueError(
            f'view keypoint permutation has length {perm.numel()}, expected {int(n)}')
    want = torch.arange(n, dtype=torch.long, device=device)
    if not torch.equal(torch.sort(perm).values, want):
        raise ValueError(f'view keypoint permutation must be a bijection, got {perm.tolist()}')
    return perm


def _raw_view_kpt_permutation(view, *, device):
    """Read an optional full-length ``View.kpt_perm`` without imposing a point count."""
    raw = getattr(view, 'kpt_perm', None)
    if raw is None:
        return None
    perm = torch.as_tensor(raw, dtype=torch.long, device=device).reshape(-1)
    n = int(perm.numel())
    _view_kpt_permutation(view, n, device=device)
    return perm


def make_anchor(source, anchor_label, mode, view_a, view_b, source_slots=None):
    """Carry a source coordinate tensor from view A into the independent anchor view.

    For 2-D, the compact ``source`` tensor is in view-A semantic slots.  The permutation path is
    view A slots -> source/session slots -> view B slots; ``source_slots`` names the ordered
    full-session slots retained by the scorer's alive mask, and the result is returned in that same
    semantic order on B.  A target whose corresponding A slot was removed remains NaN.  With no
    ``source_slots`` (the direct/full-axis helper case), the full result is returned in view-B order.
    """
    if mode == '3d':
        return source
    if source.ndim != 4 or source.shape[-1] != 2:
        raise ValueError(f'2D anchor source must be [B,T,K,2], got {tuple(source.shape)}')

    points = source[0]
    ha, hb = view_affine_2d(view_a), view_affine_2d(view_b)
    pa = _raw_view_kpt_permutation(view_a, device=points.device)
    pb = _raw_view_kpt_permutation(view_b, device=points.device)
    if pa is None and pb is None:
        return transfer_points_2d(points, ha, hb)[None]
    if pa is None:
        pa = torch.arange(pb.numel(), dtype=torch.long, device=points.device)
    if pb is None:
        pb = torch.arange(pa.numel(), dtype=torch.long, device=points.device)
    if pa.numel() != pb.numel():
        raise ValueError(
            f'view keypoint permutations disagree: A has {pa.numel()} slots, '
            f'B has {pb.numel()}')
    n = int(pa.numel())
    pa = _view_kpt_permutation(view_a, n, device=points.device)
    pb = _view_kpt_permutation(view_b, n, device=points.device)

    if source_slots is None:
        if points.shape[-2] != n:
            raise ValueError(
                f'full 2D anchor source has K={points.shape[-2]}, but views have K={n}; '
                'pass source_slots for a compact scorer axis')
        source_axis = points.index_select(-2, torch.argsort(pa))
        transferred = transfer_points_2d(source_axis, ha, hb)
        return transferred.index_select(-2, pb)[None]

    slots = torch.as_tensor(source_slots, dtype=torch.long, device=points.device).reshape(-1)
    if slots.numel() != points.shape[-2] or bool((slots < 0).any()) or bool((slots >= n).any()):
        raise ValueError(
            f'source_slots must contain {points.shape[-2]} valid view-A slots in [0,{n}), '
            f'got {slots.tolist()}')

    a_inverse = torch.argsort(pa)
    a_position = torch.full((n,), -1, dtype=torch.long, device=points.device)
    a_position[slots] = torch.arange(slots.numel(), dtype=torch.long, device=points.device)
    source_a_slots = a_inverse.index_select(0, pb.index_select(0, slots))
    compact_index = a_position.index_select(0, source_a_slots)
    valid = compact_index >= 0
    safe_index = compact_index.clamp_min(0)
    gathered = points.index_select(-2, safe_index)
    nan = torch.full_like(gathered, float('nan'))
    gathered = torch.where(valid.reshape(1, -1, 1), gathered, nan)
    return transfer_points_2d(gathered, ha, hb)[None]
