"""A tiny synthetic dataset on disk, built through the public write path.

Small enough to be fast, but it exercises every branch the real datasets use: a 2D
single-camera multi-animal session, a 3D multi-camera session whose per-camera visibility is
coordinate-free (allen-mouse's shape), a moving camera, and an ignore region.
"""
import numpy as np
import pytest
from PIL import Image

from tailcyclenet import format as fmt

KPTS_2D = ['nose', 'left_ear', 'right_ear', 'tail_base']
KPTS_3D = ['nose', 'neck', 'tail_base']


def _cam(name, w, h, calibrated=True, moving=False, offset=(0.0, 0.0), idx=0):
    f = float(max(w, h))
    return fmt.Camera(
        name=name, type='pinhole', size=(w, h), offset=offset, image_size=(w, h), moving=moving,
        matrix=np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]) if calibrated else None,
        dist=np.zeros(5) if calibrated else None,
        rvec=np.array([0.0, 0.1 * idx, 0.0]) if calibrated else None,
        tvec=np.array([10.0 * idx, 0.0, 300.0]) if calibrated else None,
    )


def _write_frames(group_dir, cam, n_frames, size):
    d = group_dir / cam
    d.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8))
    for i in range(n_frames):
        img.save(d / f'{i:06d}.png')


def _session_2d(path, T=4, S=2):
    """rat-city's shape: one uncalibrated camera, several animals, pixel labels."""
    W, H = 64, 48
    cams = [_cam('cam0', W, H, calibrated=False)]
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
    fmt.write_session(path, mode='2d', units='px', names=KPTS_2D, cameras=cams, groups=groups,
                      labels={'g000': lab}, flip_pairs=[['left_ear', 'right_ear']],
                      provenance={'source': 'synthetic'})
    _write_frames(path / 'groups' / 'g000', 'cam0', T, (W, H))
    return lab


def _session_3d(path, T=4, moving=False):
    """allen-mouse's shape: native 3D plus coordinate-free per-camera visibility rows."""
    W, H = 64, 48
    cams = [_cam(f'cam{i}', W, H, idx=i + 1, moving=moving and i == 0) for i in range(3)]
    K = len(KPTS_3D)
    lab = fmt.empty_labels(1, T, K, 3, mode3d=True, animal_ids=['m1'])
    rng = np.random.default_rng(1)
    lab.vis3d[:] = fmt.VISIBLE
    lab.points3d[:] = rng.uniform(-50, 50, size=(1, T, K, 3)).astype(np.float32)
    lab.vis3d[0, 1, 2] = fmt.MISSING
    lab.points3d[0, 1, 2] = np.nan
    # per-camera visibility with NO x,y -- the rule 10 exemption
    lab.vis2d[:] = fmt.VISIBLE
    lab.vis2d[0, :, 0, 2] = fmt.MISSING
    if moving:
        lab.ext = np.tile(np.eye(4), (3, T, 1, 1))
        for t in range(T):
            lab.ext[0, t, 0, 3] = float(t)          # camera 0 slides along x
        for i, cam in enumerate(cams):
            if not cam.moving:
                lab.ext[i] = fmt._rt_to_ext(cam.rvec, cam.tvec)

    groups = {'g000': fmt.Group('g000', T, fps=200.0)}
    fmt.write_session(path, mode='3d', units='mm', names=KPTS_3D, cameras=cams, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for cam in cams:
        _write_frames(path / 'groups' / 'g000', cam.name, T, (W, H))
    return lab


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
