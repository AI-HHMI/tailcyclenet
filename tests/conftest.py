"""A tiny synthetic dataset on disk, built through the public write path.

Small enough to be fast, but it exercises every branch the real datasets use: a 2D
single-camera multi-animal session, a 3D multi-camera session whose per-camera visibility is
coordinate-free (allen-mouse's shape), a moving camera, and an ignore region.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from tailcyclenet import format as fmt

# CAP THE INTRAOP POOL, OR `-n` MAKES THE SUITE SLOWER. torch's pool is `nproc` wide (report 14's
# 67 ms/frame uint8 convert is the same cause), so on a 128-core box eight xdist workers ask for
# 1024 threads and the suite goes from 58 s serial to 269 s at `-n 8`. Capped: 33 s. Here rather
# than in the pixi task because a bare `pytest -n 8` has to be safe too, and a single process
# barely notices (41.6 s against 39.4 s at 8 threads, measured on tests/test_model.py).
torch.set_num_threads(4)

KPTS_2D = ['nose', 'left_ear', 'right_ear', 'tail_base']
KPTS_3D = ['nose', 'neck', 'tail_base']


def _rig(specs):
    """specs: [(name, w, h, calibrated, moving, idx)] -> a Rig of aniposelib cameras."""
    from aniposelib.cameras import Camera, CameraGroup

    cams, offset, moving_d, calib_d = [], {}, {}, {}
    for name, w, h, calibrated, moving, idx in specs:
        f = float(max(w, h))
        if calibrated:
            cam = Camera(matrix=np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]),
                         dist=np.zeros(5), rvec=np.array([0.0, 0.1 * idx, 0.0]),
                         tvec=np.array([10.0 * idx, 0.0, 300.0]), name=name)
            cam.set_size((w, h))
        else:
            cam = fmt.nominal_camera(name, (w, h))
        cams.append(cam)
        offset[name], moving_d[name], calib_d[name] = (0.0, 0.0), moving, calibrated
    return fmt.Rig(CameraGroup(cams), offset=offset, moving=moving_d, calibrated=calib_d)


def _write_frames(group_dir, cam, n_frames, size):
    """Frames whose CONTENT identifies (cam, frame), not a constant.

    A test that reads frame 3 and gets frame 4 cannot fail against all-zero images, and the
    loader now computes frame paths from the `%06d` rule instead of listing the directory -- so
    an off-by-one there has to be detectable in the pixels.
    """
    d = group_dir / cam
    d.mkdir(parents=True, exist_ok=True)
    W, H = size
    y, x = np.mgrid[0:H, 0:W]
    for i in range(n_frames):
        px = np.stack([(x * 3 + i * 11) % 256, (y * 5 + i * 7) % 256,
                       np.full((H, W), (i * 37 + len(cam) * 3) % 256)], -1).astype(np.uint8)
        Image.fromarray(px).save(d / f'{i:06d}.png')


def _session_2d(path, T=4, S=2, label_source='annotated'):
    """rat-city's shape: one uncalibrated camera, several animals, pixel labels.

    `label_source` is a parameter only so a test can drive an invalid value through
    `write_session`; every fixture below leaves it at the default.
    """
    W, H = 64, 48
    rig = _rig([('cam0', W, H, False, False, 0)])
    K = len(KPTS_2D)
    lab = fmt.empty_labels(S, T, K, 1, mode3d=False, animal_ids=['a01', 'a02'])
    rng = np.random.default_rng(0)
    lab.vis2d[:] = fmt.VISIBLE
    lab.points2d[..., 0, :] = rng.uniform(5, min(W, H) - 5, size=(S, T, K, 2)).astype(np.float32)
    # one explicitly-missing point and one never-assessed point, so both codes are exercised
    lab.vis2d[0, 0, 1, 0] = fmt.MISSING
    lab.points2d[0, 0, 1, 0] = np.nan
    lab.vis2d[1, 2, 3, 0] = fmt.UNLABELED
    lab.points2d[1, 2, 3, 0] = np.nan
    # a01 is labeled, a02 is a present-but-unannotated ignore region on frame 1
    lab.instance = np.full((S, T, 1), fmt.INST_NONE, np.int8)
    lab.boxes = np.full((S, T, 1, 4), np.nan, np.float32)
    lab.instance[:, :, 0] = fmt.INST_LABELED
    lab.instance[1, 1, 0] = fmt.INST_PRESENT
    lab.boxes[1, 1, 0] = [10.0, 10.0, 30.0, 30.0]

    groups = {'g000': fmt.Group('g000', T, fps=40.0, source_video='movie.avi')}
    fmt.write_session(path, mode='2d', units='px', label_source=label_source, names=KPTS_2D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, flip_pairs=[['left_ear', 'right_ear']],
                      provenance={'source': 'synthetic'})
    _write_frames(path / 'groups' / 'g000', 'cam0', T, (W, H))
    return lab


def _session_2d_tracked_dense(path, T=4, S=2):
    """calms21 / rat-city-tracked / branson-fly's shape: `tracked`, every row `visible`, no
    assessment ever recorded.

    Distinct from `_session_2d` above (which is `annotated` and carries a real `missing` /
    `unlabeled` pair): this is what `Session.has_visibility_assessment` must read as False, and
    what the loader must therefore withhold a visibility target for -- for the WHOLE session, not
    per window, because every row here reads `visible` (finite, not NaN), so the per-window
    NaN-masking that catches an all-`projected` session cannot see this case at all.
    """
    W, H = 64, 48
    rig = _rig([('cam0', W, H, False, False, 0)])
    K = len(KPTS_2D)
    lab = fmt.empty_labels(S, T, K, 1, mode3d=False, animal_ids=['a01', 'a02'])
    rng = np.random.default_rng(3)
    lab.vis2d[:] = fmt.VISIBLE
    lab.points2d[..., 0, :] = rng.uniform(5, min(W, H) - 5, size=(S, T, K, 2)).astype(np.float32)
    groups = {'g000': fmt.Group('g000', T, fps=40.0, source_video='movie.avi')}
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=KPTS_2D,
                      rig=rig, groups=groups, labels={'g000': lab},
                      provenance={'source': 'synthetic', 'visibility': 'none'})
    _write_frames(path / 'groups' / 'g000', 'cam0', T, (W, H))
    return lab


@pytest.fixture(scope='session')
def tracked_no_assessment_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('tracked_dense')
    _session_2d_tracked_dense(root / 'catlike' / 'train' / 's')
    return root / 'catlike'


def _session_3d(path, T=4, moving=False, label_source='tracked'):
    """allen-mouse's shape: native 3D plus coordinate-free per-camera visibility rows."""
    W, H = 64, 48
    rig = _rig([(f'cam{i}', W, H, True, moving and i == 0, i + 1) for i in range(3)])
    K = len(KPTS_3D)
    lab = fmt.empty_labels(1, T, K, 3, mode3d=True, animal_ids=['m1'])
    rng = np.random.default_rng(1)
    lab.vis3d[:] = fmt.VISIBLE
    lab.points3d[:] = rng.uniform(-50, 50, size=(1, T, K, 3)).astype(np.float32)
    lab.vis3d[0, 1, 2] = fmt.MISSING
    lab.points3d[0, 1, 2] = np.nan
    # per-camera visibility with NO x,y -- the rule 10 exemption.
    # Three states on purpose: visible, assessed-but-occluded, and never assessed. The last one
    # is what must survive to the loss as NaN rather than being collapsed into "not visible".
    lab.vis2d[:] = fmt.VISIBLE
    lab.vis2d[0, :, 0, 2] = fmt.MISSING
    lab.vis2d[0, 1, 2] = fmt.UNLABELED          # keypoint 2, frame 1: nobody looked
    if moving:
        lab.ext = np.tile(np.eye(4), (3, T, 1, 1))
        for t in range(T):
            lab.ext[0, t, 0, 3] = float(t)          # camera 0 slides along x
        for i, cam in enumerate(rig.cameras):
            if not rig.moving[cam.get_name()]:
                lab.ext[i] = cam.get_extrinsics_mat().detach().cpu().numpy()

    groups = {'g000': fmt.Group('g000', T, fps=200.0)}
    fmt.write_session(path, mode='3d', units='mm', label_source=label_source, names=KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
    return lab


def _session_3d_multi(path, T=4, sep=5.0, label_source='tracked'):
    """`_session_3d`'s shape with a SECOND animal, for the animal-swap prior corruption
    (dev/plans/prompt_prior_corruptions.md) -- `_session_3d` ships exactly one animal, which
    cannot exercise `prompt_swap_animal` at all (`want_swap_animal` requires `n_animals >= 2`).

    The second animal is the FIRST one's own pose rigidly shifted by `sep` world units along one
    axis, so a SMALL `sep` keeps both animals inside one camera's crop (ELIGIBLE, per
    `prior_out_of_bounds`) and a LARGE one puts the second animal well outside it (INELIGIBLE) --
    both cases a test needs, from one knob, rather than two separately-authored fixtures.
    """
    W, H = 64, 48
    rig = _rig([(f'cam{i}', W, H, True, False, i + 1) for i in range(3)])
    K = len(KPTS_3D)
    lab = fmt.empty_labels(2, T, K, 3, mode3d=True, animal_ids=['m1', 'm2'])
    rng = np.random.default_rng(2)
    lab.vis3d[:] = fmt.VISIBLE
    base = rng.uniform(-50, 50, size=(T, K, 3)).astype(np.float32)
    lab.points3d[0] = base
    lab.points3d[1] = base + np.array([sep, 0.0, 0.0], dtype=np.float32)
    lab.vis2d[:] = fmt.VISIBLE

    groups = {'g000': fmt.Group('g000', T, fps=200.0)}
    fmt.write_session(path, mode='3d', units='mm', label_source=label_source, names=KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
    return lab


@pytest.fixture(scope='session')
def mixed_source_root(tmp_path_factory):
    """One root holding three of the four (mode, label_source) cells -- allen's exact shape.

    The missing cell is the point: `2d-tracked` is absent, so a scheme that fits both fractions
    as joint marginals would be over-constrained here. The two-level draw is not.
    """
    root = tmp_path_factory.mktemp('mixed') / 'ds'
    _session_2d(root / 'train' / 'a_2d_annot', label_source='annotated')
    _session_3d(root / 'train' / 'b_3d_annot', label_source='annotated')
    _session_3d(root / 'train' / 'c_3d_tracked', label_source='tracked')
    return root


@pytest.fixture(scope='session')
def tiny_root(tmp_path_factory):
    """A folder of two dataset roots -- the multi-dataset case."""
    root = tmp_path_factory.mktemp('tiny')
    _session_2d(root / 'ratlike' / 'train' / 'sess_a')
    _session_2d(root / 'ratlike' / 'val' / 'sess_b')
    _session_3d(root / 'mouselike' / 'train' / 'sess_c')
    _session_3d(root / 'mouselike' / 'train' / 'sess_moving', moving=True)
    return root


@pytest.fixture(scope='session')
def dataset_2d(tiny_root):
    return fmt.load_dataset(tiny_root / 'ratlike')


@pytest.fixture(scope='session')
def dataset_3d(tiny_root):
    return fmt.load_dataset(tiny_root / 'mouselike')


def _session_projected(path, T=4):
    """johnson-mouse's shape: per-camera 2D POSITIONS with no visibility assessment anywhere.

    Every 2D row is `projected` -- the annotator placed the point in this view but never judged
    whether it was actually seen there. The positions must survive, and no visibility target may
    be built from them, in either direction: `visible` would train "always visible" and the
    noisy-OR of "no camera says visible" would train "nothing is reconstructible".
    """
    W, H = 64, 48
    rig = _rig([(f'cam{i}', W, H, True, False, i + 1) for i in range(3)])
    K = len(KPTS_3D)
    lab = fmt.empty_labels(1, T, K, 3, mode3d=True, animal_ids=['m1'])
    rng = np.random.default_rng(7)
    lab.vis3d[:] = fmt.VISIBLE
    lab.points3d[:] = rng.uniform(-20, 20, size=(1, T, K, 3)).astype(np.float32)
    lab.vis2d[:] = fmt.PROJECTED
    lab.points2d[:] = rng.uniform(5, 40, size=(1, T, K, 3, 2)).astype(np.float32)

    groups = {'g000': fmt.Group('g000', T, fps=100.0)}
    fmt.write_session(path, mode='3d', units='mm', label_source='tracked', names=KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
    return lab


@pytest.fixture(scope='session')
def projected_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('projected')
    _session_projected(root / 'proj' / 'train' / 's')
    return root / 'proj'


def _session_centred_labels(path, T=24, labelled=(11, 12, 13)):
    """A group whose labels sit ONLY in the middle -- the shape the old loader could not use.

    posetail-pose's `get_start_ixs_train` admitted a window only if its FIRST frame had a finite
    coordinate, so a group like this yielded zero training windows and the natural annotation
    shape (a label with context on both sides) was silently unusable.
    """
    W = H = 48
    K = len(KPTS_3D)
    rig = _rig([('cam0', W, H, False, False, 0)])
    lab = fmt.empty_labels(1, T, K, 1, mode3d=False)
    lab.vis2d[0, list(labelled), :, 0] = fmt.VISIBLE
    lab.points2d[0, list(labelled), :, 0] = (
        10.0 + 10.0 * np.arange(len(labelled), dtype=np.float32))[:, None, None] % (W - 10)
    groups = {'g0': fmt.Group('g0', T)}
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=KPTS_3D,
                      rig=rig, groups=groups, labels={'g0': lab})
    _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return list(labelled)


@pytest.fixture(scope='session')
def centred_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('centred')
    _session_centred_labels(root / 'mid' / 'train' / 's')
    return root / 'mid'


@pytest.fixture(scope='session')
def dense_root(tmp_path_factory):
    """A fully-labelled 32-frame group -- long enough for a strided window to have room."""
    root = tmp_path_factory.mktemp('dense')
    _session_centred_labels(root / 'dense' / 'train' / 's', T=32, labelled=tuple(range(32)))
    return root / 'dense'
