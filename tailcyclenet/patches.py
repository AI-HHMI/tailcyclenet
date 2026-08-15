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
        # Withhold the offset so the original does the projection and the division, then apply the
        # subtraction it skipped -- right-aligned, so the time axis meets the time axis.
        bare = {k: v for k, v in cam.items() if k != 'offset'}
        p2d = original(bare, p3d_t, downsample_factor, max_normalized)
        off = offset.to(p2d.dtype) / downsample_factor
        return p2d - off.reshape(*off.shape[:-1], 1, off.shape[-1])

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


def apply_all():
    """Idempotent. Called once from `tailcyclenet/__init__.py`."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_project_cam()
    _APPLIED = True
