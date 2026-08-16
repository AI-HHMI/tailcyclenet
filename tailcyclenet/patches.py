"""Behaviour this repo needs from `posetail` that the pinned release does not provide.

**EVERY PATCH HERE IS A BUG THAT SHOULD BE FIXED UPSTREAM.** They live in one file, applied from
`tailcyclenet/__init__.py`, so the divergence from `posetail==0.3.2` is enumerable in one place --
and so that upgrading the pin means reading this file and deleting whatever landed, rather than
discovering a silently double-applied fix. Each entry says what to send upstream.

Patching is a last resort and the bar is: the behaviour is needed by the model AND the call sites
are inside the library, so a local reimplementation (the `crop_box_for_points` route) cannot reach
them. `project_cam` is called from `posetail/posetail/losses.py` (four reprojection terms) and
`posetail/posetail/encoder_decoder.py`, neither of which this repo owns.

---------------------------------------------------------------------------------------------
1. `cube.project_cam` -- PER-FRAME CAMERA OFFSET
---------------------------------------------------------------------------------------------

UPSTREAM FIX: in `posetail/posetail/cube.py::project_cam`, change

    p2d = p2d - offset[None, :]
to
    p2d = p2d - offset.reshape(*offset.shape[:-1], 1, offset.shape[-1]) if offset.ndim > 1 \
          else p2d - offset[None, :]

i.e. right-align the offset instead of prepending exactly one axis, so a per-frame `(T,2)` offset
aligns with the time axis the function ALREADY documents for `ext`. Please send this upstream; the
patch below is a stopgap and should be deleted when the pin moves.

WHY IT MATTERS. A moving crop -- one that follows the animal per frame instead of standing still
over the window (`crop.moving_boxes`) -- is expressible as a per-frame camera OFFSET and nothing
else. `crop.apply_crop` writes only `offset` and `size`, and the rule holds the side constant, so
`mat` and `ext` are untouched. The library already supports a time-varying camera on the extrinsic
side: `ext` may be `(T,4,4)` and `project_cam`'s own comment spells out that the caller must put
the time axis at position -3 of `p3d_t`. `offset` was simply never given the same treatment.

WHAT GOES WRONG WITHOUT IT, and it is a silent shape bug rather than an exception. `offset[None,:]`
prepends exactly one axis, so with a `(T,2)` offset it becomes `(1,T,2)` and collides with the
keypoint axis; with a pre-reshaped `(T,1,2)` it becomes `(1,T,1,2)`, whose VALUES are right at
every call site but whose RANK is wrong wherever `p3d` is rank 3. Measured:

    p3d (T,K,3)    -- the loader        -> (1,T,K,2)   spurious leading axis, values correct
    p3d (B,T,K,3)  -- losses.py reproj  -> (B,T,K,2)   correct

So no single offset shape is correct everywhere, which is why this is a patch and not a calling
convention.

THE WRAPPER DOES NOT REIMPLEMENT `project_cam`. It calls the original with `offset` withheld and
subtracts afterwards, so distortion, the depth clamp, the float64 promotion and every future
upstream change to the projection are inherited rather than copied. The only arithmetic here is
the subtraction the original would have done.

`downsample_factor` DIVIDES AFTER THE SUBTRACTION upstream -- `(raw - offset) / df` -- so the
wrapper subtracts `offset / df` from the already-divided result. Same value, and it is the kind of
detail a reimplementation gets wrong.

A 1-D offset takes the original code path untouched, so every run that does not use a moving crop
is bit-identical. `tests/test_patches.py` pins both halves.
"""
from __future__ import annotations

import torch

_APPLIED = False


def _patch_project_cam():
    """Right-align `project_cam`'s offset subtraction so a per-frame `(T,2)` offset works."""
    from posetail.posetail import cube

    original = cube.project_cam
    if getattr(original, '_tailcyclenet_patched', False):
        return

    def project_cam(cam, p3d_t, downsample_factor=1, max_normalized=3.0):
        offset = cam.get('offset')
        if offset is None or offset.ndim <= 1:
            # THE UNPATCHED PATH, byte for byte. Everything without a moving crop lands here.
            return original(cam, p3d_t, downsample_factor, max_normalized)
        if offset.ndim != 2:
            raise ValueError(f'camera offset must be (2,) or (T,2), got {tuple(offset.shape)}')
        # Withhold the offset so the original does the projection and the division, then apply the
        # subtraction it skipped -- right-aligned, so the time axis meets the time axis.
        bare = {k: v for k, v in cam.items() if k != 'offset'}
        p2d = original(bare, p3d_t, downsample_factor, max_normalized)
        T = offset.shape[0]
        # A TIME-LESS POINT SET THROUGH A MOVING CAMERA IS A BUG, NOT A BROADCAST. Without this the
        # subtraction happily GROWS a time axis -- `(1,n,2) - (T,1,2) -> (T,n,2)` -- and the caller
        # gets a plausible tensor of the wrong rank. That is what took out all three 3D jobs of the
        # first sweep4 submission, in `get_camera_scale`, which projects a single mean pose: the
        # failure surfaced as an IndexError deep in the library, three frames from its cause.
        # Anything projecting a pose that is not per-frame must collapse the offset first --
        # `crop.with_static_offset`, which is EXACT for offset-invariant quantities.
        if p2d.ndim < 3 or p2d.shape[-3] != T:
            raise ValueError(
                f'per-frame camera offset {tuple(offset.shape)} against points projecting to '
                f'{tuple(p2d.shape)}: axis -3 must be the time axis of length {T}, and is '
                f'{"absent" if p2d.ndim < 3 else p2d.shape[-3]}. Either give the points a time '
                'axis, or collapse the offset with crop.with_static_offset -- see that docstring '
                'for which of the two is right.')
        off = offset.to(p2d.dtype) / downsample_factor
        return p2d - off[:, None, :]

    project_cam._tailcyclenet_patched = True
    project_cam._tailcyclenet_original = original
    cube.project_cam = project_cam

    # `project_points_torch` closed over the name at def time in some versions; rebind defensively
    # so both entry points see the patch regardless.
    if hasattr(cube, 'project_points_torch'):
        import torch

        def project_points_torch(camera_group, coords_3d, downsample_factor=1):
            return torch.stack([cube.project_cam(cam, coords_3d, downsample_factor)
                                for cam in camera_group])

        project_points_torch._tailcyclenet_patched = True
        cube.project_points_torch = project_points_torch

    # The library re-exports both by VALUE into modules that imported them at their own import
    # time (`from ... import project_points_torch`), so rebinding `cube` alone misses them.
    for mod in ('posetail.posetail.losses', 'posetail.posetail.encoder_decoder',
                'posetail.datasets.posetail_dataset', 'posetail.datasets.inference_dataset',
                'posetail.inference.inference_utils'):
        try:
            import importlib
            m = importlib.import_module(mod)
        except Exception:                       # not every module is importable in every env
            continue
        for name in ('project_cam', 'project_points_torch'):
            if hasattr(m, name) and not getattr(getattr(m, name), '_tailcyclenet_patched', False):
                setattr(m, name, getattr(cube, name))


def _patch_get_camera_scale():
    """`get_camera_scale` is offset-INVARIANT, so give it a static offset rather than failing.

    UPSTREAM FIX: in `posetail/posetail/cube.py::get_camera_scale`, collapse a per-frame `offset`
    to a single one before use — the function returns a projection SENSITIVITY (world units per
    pixel), a Jacobian, and a constant image-plane translation has zero derivative. Please send
    this upstream with the `project_cam` change.

    WHY A PATCH AND NOT A CALLING CONVENTION. Unlike the other call sites, this one is inside the
    library and out of reach: `tracker_encoder.py:318` calls it as
    `get_camera_scale(camera_group, coords, times=query_times)` with the query anchor, which is one
    point per keypoint for the WHOLE window and so carries no time axis. Under a moving crop that
    hits `project_cam`'s guard and stops the run.

    WHAT IS EXACT AND WHAT IS NOT. The Jacobian is exact under the collapse. The function also
    gates on `is_point_visible`, which genuinely does depend on the offset — with a moving crop a
    point can be inside the crop on some frames and not others — so that gate becomes "visible in
    the MEAN crop". It only selects which points enter a median over keypoints, and there is no
    per-frame answer to give for a point set that has no time axis.
    """
    from posetail.posetail import cube

    original = cube.get_camera_scale
    if getattr(original, '_tailcyclenet_patched', False):
        return

    def get_camera_scale(camera_group, *args, **kwargs):
        from .crop import with_static_offset
        return original([with_static_offset(c) for c in camera_group], *args, **kwargs)

    get_camera_scale._tailcyclenet_patched = True
    get_camera_scale._tailcyclenet_original = original
    cube.get_camera_scale = get_camera_scale
    for mod in ('posetail.posetail.losses', 'posetail.posetail.encoder_decoder',
                'posetail.posetail.tracker_encoder', 'posetail.datasets.posetail_dataset'):
        try:
            import importlib
            m = importlib.import_module(mod)
        except Exception:
            continue
        if hasattr(m, 'get_camera_scale') and not getattr(
                m.get_camera_scale, '_tailcyclenet_patched', False):
            m.get_camera_scale = get_camera_scale


def _patch_undistort_points():
    """`undistort_points` reads `offset[0]`/`offset[1]` as SCALARS, so a per-frame offset dies.

    UPSTREAM FIX: in `posetail/posetail/cube.py::undistort_points`, accept a `(T,2)` offset the way
    `points_to_rays` already accepts a `(T,4,4)` extrinsic, i.e. expanded over the `(t n)` ray
    order. Send this with the other two.

    THE `(t n)` ORDER IS THE LIBRARY'S OWN, NOT AN INFERENCE. `tracker_encoder.py:489-500` builds
    the rays as `p2d_ib = rearrange(p2d_query[i, b], 't n r -> (t n) r')` and, for a MOVING RIG,
    expands the extrinsic to match with `repeat(cam['ext'], 't i j -> (t n) i j', n=N)` — its own
    comment says "ray order is (t n)". A per-frame offset is the same expansion over the same
    order, so this mirrors handling the library already has rather than guessing a memory layout.
    Getting it wrong would displace each ray by up to the crop's travel and mis-triangulate
    silently, which is why it is done by construction and asserted rather than assumed.

    Implemented as T calls to the ORIGINAL, one per frame with that frame's static offset, and
    concatenated back in `(t n)` order. Exact, and it inherits the distortion model rather than
    copying it. T <= the window length, so this is a handful of small calls.
    """
    from posetail.posetail import cube

    original = cube.undistort_points
    if getattr(original, '_tailcyclenet_patched', False):
        return

    def undistort_points(cam, points, *args, **kwargs):
        off = cam.get('offset')
        if off is None or off.ndim <= 1:
            return original(cam, points, *args, **kwargs)
        T = off.shape[0]
        one = lambda t, p: original(dict(cam, offset=off[t]), p, *args, **kwargs)  # noqa: E731
        # TWO LAYOUTS REACH THIS FUNCTION and they are not interchangeable.
        #   (..., T, N, 2)  -- `tracker_encoder.py:614`, the 2D head's own prediction. Time sits at
        #                      axis -3, the same convention `project_cam` documents for `ext`.
        #   (T*N, 2)        -- `points_to_rays` via `tracker_encoder.py:500`, flattened in the
        #                      `(t n)` order that file's own comment states and that it already
        #                      uses to expand a moving rig's extrinsic.
        # Assuming either one alone is wrong: the first version of this patch took the flattened
        # reading and rejected a (B,T,N,2) tensor as "1 point against 24 frames".
        if points.ndim >= 3 and points.shape[-3] == T:
            return torch.stack([one(t, points[..., t, :, :]) for t in range(T)], dim=-3)
        if points.ndim == 2 and points.shape[0] % T == 0:
            per = points.reshape(T, -1, points.shape[-1])
            return torch.cat([one(t, per[t]) for t in range(T)], dim=0)
        raise ValueError(
            f'undistort_points got points of {tuple(points.shape)} against a per-frame offset of '
            f'{T} frames: expected time at axis -3, or a flat (T*N, 2) in (t n) order.')

    undistort_points._tailcyclenet_patched = True
    undistort_points._tailcyclenet_original = original
    cube.undistort_points = undistort_points
    for mod in ('posetail.posetail.tracker_encoder', 'posetail.posetail.encoder_decoder',
                'posetail.posetail.losses'):
        try:
            import importlib
            m = importlib.import_module(mod)
        except Exception:
            continue
        if hasattr(m, 'undistort_points') and not getattr(
                m.undistort_points, '_tailcyclenet_patched', False):
            m.undistort_points = undistort_points


def apply_all():
    """Idempotent. Called once from `tailcyclenet/__init__.py`."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_project_cam()
    _patch_get_camera_scale()
    _patch_undistort_points()
    _APPLIED = True
