"""`tailcyclenet.patches` -- the library behaviour this repo overrides, and its blast radius.

A monkeypatch on a pinned dependency is the highest-risk thing in this package: it changes code
four call sites inside `posetail` reach, and nothing in a loss curve would show it going wrong. So
the bar here is both halves, every time -- the UNPATCHED path is bit-identical, and the new path is
correct against arithmetic computed a different way.

Every patch should also be fixed UPSTREAM; `tailcyclenet/patches.py` says what to send.
"""
import numpy as np
import pytest
import torch

import tailcyclenet  # noqa: F401  -- importing the package is what applies the patches
from tailcyclenet import crop as cropmod
from tailcyclenet.dataset import _resize_camera


def _cam(**over):
    cam = dict(ext=torch.eye(4, dtype=torch.float64),
               mat=torch.tensor([[100., 0, 50], [0, 100., 50], [0, 0, 1]], dtype=torch.float64),
               dist=torch.tensor([0.1, -0.05, 0.01, 0.02, 0.001], dtype=torch.float64),
               type='pinhole', size=torch.tensor([200, 160], dtype=torch.int32),
               offset=torch.zeros(2, dtype=torch.float64))
    cam.update(over)
    return cam


def _points(*shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g).double() + torch.tensor([0., 0., 5.])


def test_the_patch_is_applied_to_every_call_site():
    """`posetail` binds these by VALUE at import time, so `cube` alone is not enough."""
    from posetail.posetail import cube, encoder_decoder, losses

    assert getattr(cube.project_cam, '_tailcyclenet_patched', False)
    assert getattr(losses.project_points_torch, '_tailcyclenet_patched', False)
    assert getattr(encoder_decoder.project_cam, '_tailcyclenet_patched', False)


@pytest.mark.parametrize('shape', [(5, 4, 3), (3, 5, 4, 3)])
@pytest.mark.parametrize('df', [1, 2])
def test_a_static_offset_takes_the_original_code_path_exactly(shape, df):
    """Every run without a moving crop must be bit-identical to the unpatched library."""
    from posetail.posetail.cube import project_cam

    original = project_cam._tailcyclenet_original
    cam = _cam(offset=torch.tensor([7., 3.], dtype=torch.float64))
    p3d = _points(*shape)
    assert torch.equal(project_cam(cam, p3d, df), original(cam, p3d, df))


@pytest.mark.parametrize('shape', [(5, 4, 3), (3, 5, 4, 3)])
@pytest.mark.parametrize('df', [1, 2])
def test_a_per_frame_offset_is_right_aligned_at_every_rank(shape, df):
    """The bug: `offset[None, :]` prepends ONE axis, so rank 3 came back rank 4.

    Values were correct and the SHAPE was not, which is a silent corruption rather than a crash --
    the loader projects at rank 3 and `losses.py` at rank 4, so no single offset shape worked for
    both and the failure landed downstream of where it was caused.
    """
    from posetail.posetail.cube import project_cam

    T = shape[-3]
    cam = _cam()
    p3d = _points(*shape)
    base = project_cam(cam, p3d, df)
    off = torch.arange(T * 2, dtype=torch.float64).reshape(T, 2)
    got = project_cam(_cam(offset=off), p3d, df)
    assert got.shape == base.shape
    # `downsample_factor` divides AFTER the subtraction upstream, so the offset is scaled too.
    torch.testing.assert_close(got, base - (off / df).reshape(T, 1, 2))


def test_a_constant_per_frame_offset_equals_the_static_one():
    """The two code paths must agree where they describe the same camera."""
    from posetail.posetail.cube import project_cam

    p3d = _points(5, 4, 3)
    off = torch.tensor([9., -4.], dtype=torch.float64)
    torch.testing.assert_close(project_cam(_cam(offset=off), p3d),
                               project_cam(_cam(offset=off.expand(5, 2).clone()), p3d))


# ----------------------------------------------------------------------------------------------
# what the patch exists for
# ----------------------------------------------------------------------------------------------

def test_the_3d_moving_crop_projects_where_the_pixels_were_cut():
    """END TO END: patch + `apply_crop_moving` + `_resize_camera` must compose to the right frame.

    Computed a DIFFERENT WAY on the other side -- project into the uncropped source camera, then
    subtract that frame's own box origin and apply the resize scale, which is what `read_frames`
    does to the pixels. If these disagree the skeleton sits off the animal, and in 3D that is
    invisible until a reprojection loss quietly stops converging.
    """
    from posetail.posetail.cube import project_points_torch

    T = 6
    cam = _cam()
    # A track that walks across the frame, so the crop actually has to move.
    coords = torch.stack([torch.stack([torch.tensor([0.4 * t - 1.0 + 0.1 * k, 0.2 * t - 0.5, 5.0],
                                                    dtype=torch.float64)
                                       for k in range(3)]) for t in range(T)])
    cg, boxes = cropmod.crop_to_points_3d_moving([cam], coords, min_crop_dim=16)
    assert cg is not None and boxes[0].shape == (T, 4)
    assert (boxes[0][:, 0] != boxes[0][0, 0]).any(), 'the crop must actually move'
    sides = np.unique(np.stack([boxes[0][:, 2] - boxes[0][:, 0], boxes[0][:, 3] - boxes[0][:, 1]]))
    assert len(sides) == 1, f'the side must be constant, got {sides}'

    src = project_points_torch([cam], coords)[0]                       # (T,K,2) source pixels
    origin = torch.as_tensor(boxes[0][:, :2].astype(np.float64))       # (T,2)
    cgr, scale = _resize_camera(dict(cg[0]), 32)
    want = (src - origin[:, None, :]) * scale
    got = project_points_torch([cgr], coords)[0]
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_the_moving_crop_holds_the_animal_still_and_the_static_one_does_not():
    """The property the whole rule is for, stated as a measurement rather than a shape."""
    from posetail.posetail.cube import project_points_torch

    T = 8
    cam = _cam()
    coords = torch.stack([torch.stack([torch.tensor([0.5 * t - 2.0 + 0.1 * k, 0.0, 5.0],
                                                    dtype=torch.float64)
                                       for k in range(3)]) for t in range(T)])
    mv, _ = cropmod.crop_to_points_3d_moving([cam], coords, min_crop_dim=16)
    st, _ = cropmod.crop_to_points_3d([cam], coords, min_crop_dim=16)

    def wander(cg):
        p = project_points_torch(cg, coords)[0]
        c = p.nanmean(dim=1)                                # (T,2) centroid inside the crop
        return float((c - c.mean(0)).abs().max())

    assert wander(mv) < wander(st) / 2, 'a moving crop must hold the animal roughly still'


def test_the_patch_must_be_applied_at_PACKAGE_import(monkeypatch):
    """WHY `__init__.py` and not a lazier hook, pinned so nobody "tidies" it into one.

    `tailcyclenet.query_encoder` does `from posetail.posetail.cube import project_points_torch` at
    module level -- it binds BY VALUE, exactly as `posetail`'s own modules do. Applying the patch
    from, say, `format.py` or `crop.py` would leave query_encoder holding the ORIGINAL function,
    because it imports neither of them. Importing the package first is what makes the bound name
    the patched one.

    The cost of that choice is that `import tailcyclenet` requires torch, which is why the sweep
    scripts' pre-warm imports `posetail` alone -- a login node cannot mmap `libtorch_cpu.so` and
    the whole submission died on it once.
    """
    import tailcyclenet.query_encoder as qe

    assert getattr(qe.project_points_torch, '_tailcyclenet_patched', False), \
        'query_encoder bound the projector by value before the patch was applied'


def test_applying_twice_does_not_stack():
    """`apply_all` is called from package import and is idempotent; a double wrap would subtract
    the offset twice and be invisible except as a constant pose shift."""
    import torch

    from posetail.posetail import cube
    from tailcyclenet import patches

    before = cube.project_cam
    patches._APPLIED = False          # force a re-entry, as a second import would
    patches.apply_all()
    assert cube.project_cam is before, 'apply_all re-wrapped an already-patched function'

    cam = _cam(offset=torch.arange(10, dtype=torch.float64).reshape(5, 2))
    p3d = _points(5, 4, 3)
    bare = {k: v for k, v in cam.items() if k != 'offset'}
    torch.testing.assert_close(cube.project_cam(cam, p3d),
                               cube.project_cam(bare, p3d) - cam['offset'].reshape(5, 1, 2))
