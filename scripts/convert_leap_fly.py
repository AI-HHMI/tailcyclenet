#!/usr/bin/env python
"""Convert the LEAP fly cluster-sampled labels to tailcycle 2D format.

The LEAP training set stores 1,500 labelled 192x192 fly crops in HDF5 and the
corresponding 32-point coordinates in a MATLAB file.  The source has no visibility annotations; the requested training representation emits all finite coordinates as ``visible``.
Each sampled crop is an independent one-frame group; training clamp-pads these groups when it
needs a two-or-more-frame window.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/leap-fly')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
           'tailcycle-datasets/leap-fly-annotated')
H5_NAME = 'datasets/dsets_2018-05-03_cluster-sampled.k_10_n_150(5).h5'
MAT_NAME = 'datasets/dsets_2018-05-03_cluster-sampled.k_10_n_150.labels(6).mat'
SESSION = 'cluster_sampled'
NAMES = (
    'head', 'eyeL', 'eyeR', 'neck', 'thorax', 'abdomen',
    'forelegR1', 'forelegR2', 'forelegR3', 'forelegR4',
    'midlegR1', 'midlegR2', 'midlegR3', 'midlegR4',
    'hindlegR1', 'hindlegR2', 'hindlegR3', 'hindlegR4',
    'forelegL1', 'forelegL2', 'forelegL3', 'forelegL4',
    'midlegL1', 'midlegL2', 'midlegL3', 'midlegL4',
    'hindlegL1', 'hindlegL2', 'hindlegL3', 'hindlegL4',
    'wingL', 'wingR',
)
FLIP_PAIRS = (
    ('eyeL', 'eyeR'),
    ('forelegL1', 'forelegR1'), ('forelegL2', 'forelegR2'),
    ('forelegL3', 'forelegR3'), ('forelegL4', 'forelegR4'),
    ('midlegL1', 'midlegR1'), ('midlegL2', 'midlegR2'),
    ('midlegL3', 'midlegR3'), ('midlegL4', 'midlegR4'),
    ('hindlegL1', 'hindlegR1'), ('hindlegL2', 'hindlegR2'),
    ('hindlegL3', 'hindlegR3'), ('hindlegL4', 'hindlegR4'),
    ('wingL', 'wingR'),
)


def _mat_names(mat) -> list[str]:
    """Read and validate the MATLAB skeleton node names."""
    skeleton = mat['skeleton'].flat[0]
    nodes = [str(node.flat[0]) for node in skeleton.nodes.ravel()]
    if nodes != list(NAMES):
        raise ValueError(f'unexpected LEAP skeleton axis: {nodes!r}')
    return nodes


def _skeleton_edges(mat) -> list[tuple[str, str]]:
    """Convert MATLAB's one-based skeleton edge indices to name pairs."""
    edges = np.asarray(mat['skeleton'].flat[0].edges, dtype=np.int64)
    return [(NAMES[int(a) - 1], NAMES[int(b) - 1]) for a, b in edges]


def convert(src: Path, out: Path, clean: bool = False) -> None:
    """Write the LEAP crops and labels as one annotated tailcycle train session.

    MATLAB ``positions`` are stored as (row, column), or (y, x), and are reversed to tailcycle
    pixel order (x, y). The source has no visibility labels, so finite placements are marked
    visible; all independent sampled crops are assigned to the train split.
    """
    h5_path, mat_path = src / H5_NAME, src / MAT_NAME
    if not h5_path.is_file() or not mat_path.is_file():
        raise FileNotFoundError(f'expected {h5_path} and {mat_path}')
    if out.exists():
        if not clean:
            raise FileExistsError(f'{out} exists; pass --clean to replace it')
        shutil.rmtree(out)

    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    names = _mat_names(mat)
    edges = _skeleton_edges(mat)
    positions = np.asarray(mat['positions'], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[0:2] != (len(names), 2):
        raise ValueError(f'positions has unexpected shape {positions.shape}')

    with h5py.File(h5_path, 'r') as h5:
        boxes = np.asarray(h5['box'])
        expt_ids = np.asarray(h5['exptID']).reshape(-1).astype(np.int64)
        frame_ids = np.asarray(h5['framesIdx']).reshape(-1).astype(np.int64)
    if boxes.ndim != 4 or boxes.shape[1] != 1 or boxes.shape[2:] != (192, 192):
        raise ValueError(f'box has unexpected shape {boxes.shape}')
    n = boxes.shape[0]
    if positions.shape[2] != n or expt_ids.size != n or frame_ids.size != n:
        raise ValueError('HDF5 and MATLAB label counts disagree')
    if boxes.dtype != np.uint8:
        raise ValueError(f'box dtype is {boxes.dtype}, expected uint8')
    xy = np.transpose(positions, (2, 0, 1))[:, :, ::-1]
    if not np.isfinite(xy).all() or (xy < 0).any() or (xy[..., 0] >= 192).any() \
            or (xy[..., 1] >= 192).any():
        raise ValueError('labels contain non-finite or out-of-crop coordinates')

    session = out / 'train' / SESSION
    session.mkdir(parents=True, exist_ok=True)
    groups: dict[str, fmt.Group] = {}
    labels: dict[str, fmt.Labels] = {}
    for i in range(n):
        gid = f'e{int(expt_ids[i]):03d}_f{int(frame_ids[i]):09d}'
        if gid in groups:
            raise ValueError(f'duplicate group id {gid}')
        groups[gid] = fmt.Group(
            gid, 1, fps=float('nan'), source_video=f'LEAP experiment {int(expt_ids[i]):03d}',
            source_frame_start=int(frame_ids[i]) - 1, source_frame_step=1,
            notes='independent 192x192 LEAP cluster-sampled crop; source frame index was 1-based')
        lab = fmt.empty_labels(1, 1, len(names), 1, mode3d=False, animal_ids=['fly0'])
        lab.points2d[0, 0, :, 0] = xy[i]
        lab.vis2d[0, 0, :, 0] = fmt.VISIBLE
        labels[gid] = lab

        pixel_dir = session / 'groups' / gid / 'cam0'
        pixel_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(boxes[i, 0], mode='L').save(pixel_dir / '000000.png',
                                                   format='PNG', compress_level=1)

    rig = fmt.Rig(
        cgroup=__import__('aniposelib.cameras', fromlist=['CameraGroup']).CameraGroup([
            fmt.nominal_camera('cam0', (192, 192))]),
        offset={'cam0': (0.0, 0.0)}, moving={'cam0': False}, calibrated={'cam0': False})
    fmt.write_session(
        session, mode='2d', units='px', label_source='annotated', names=names, rig=rig,
        groups=groups, labels=labels, skeleton=edges, flip_pairs=FLIP_PAIRS,
        provenance={
            'source': str(src),
            'source_h5': str(h5_path),
            'source_labels': str(mat_path),
            'annotator': 'LEAP FlyAging annotation set',
            'annotator_tool': 'LEAP JointLabelGUI',
            'created': '2026-09-21',
            'converter': 'scripts/convert_leap_fly.py',
            'coordinate_note': 'LEAP positions are MATLAB row,column (y,x); converted to tailcycle x,y pixels',
            'visibility_note': 'LEAP supplies no visibility assessment; requested representation marks all finite placements visible',
            'split_note': 'all 1,500 cluster-sampled labelled crops are under train; source supplied no split',
        })
    print(f'converted {n} LEAP crops into {session}')


def main() -> None:
    """Parse paths and run the conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=SRC)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    convert(args.src, args.out, args.clean)


if __name__ == '__main__':
    main()
