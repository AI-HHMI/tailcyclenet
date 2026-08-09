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
    d = group_dir / cam
    d.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8))
    for i in range(n_frames):
        img.save(d / f'{i:06d}.png')


def _session_2d(path, T=4, S=2):
    """rat-city's shape: one uncalibrated camera, several animals, pixel labels."""
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
    fmt.write_session(path, mode='2d', units='px', names=KPTS_2D, rig=rig, groups=groups,
                      labels={'g000': lab}, flip_pairs=[['left_ear', 'right_ear']],
                      provenance={'source': 'synthetic'})
    _write_frames(path / 'groups' / 'g000', 'cam0', T, (W, H))
    return lab


def _session_3d(path, T=4, moving=False):
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
    fmt.write_session(path, mode='3d', units='mm', names=KPTS_3D, rig=rig, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
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
    lab.points2d[0, list(labelled), :, 0] = np.array(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], np.float32)
    groups = {'g0': fmt.Group('g0', T)}
    fmt.write_session(path, mode='2d', units='px', names=KPTS_3D, rig=rig,
                      groups=groups, labels={'g0': lab})
    _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return list(labelled)


@pytest.fixture(scope='session')
def centred_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('centred')
    _session_centred_labels(root / 'mid' / 'train' / 's')
    return root / 'mid'
