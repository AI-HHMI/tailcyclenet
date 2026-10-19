"""Cameras and rigs: `Rig`, `nominal_camera`/`nominal_rig`, and `calibration.toml` read/write."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import FormatError


def nominal_camera(name: str, size, dist=None):
    """An aniposelib Camera for an uncalibrated single view (a 2D session).

    Focal length max(W, H) so normalised coords land in ~[-0.5, 0.5]; `set_size` must be the real
    image so `p2d / size` is pixel-normalised.
    """
    from aniposelib.cameras import Camera

    w, h = int(size[0]), int(size[1])
    f = float(max(w, h))
    cam = Camera(matrix=np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]),
                 dist=np.zeros(5) if dist is None else np.asarray(dist, float).ravel(),
                 rvec=np.zeros(3), tvec=np.zeros(3), name=name)
    cam.set_size((w, h))
    return cam


def nominal_rig(name: str, size, dist=None) -> Rig:
    """The one-camera, uncalibrated, static `Rig` of a 2D session: `nominal_camera` at no offset."""
    from aniposelib.cameras import CameraGroup

    return Rig(cgroup=CameraGroup([nominal_camera(name, size, dist)]), offset={name: (0.0, 0.0)},
               moving={name: False}, calibrated={name: False})


@dataclass
class Rig:
    """An aniposelib `CameraGroup` plus the three per-camera facts it does not model.

    `offset` is the stored image's origin in the sensor frame (project, then subtract offset);
    `moving` says per-frame extrinsics live in `extrinsics.pq`; `calibrated` says whether a
    matrix was really carried or `nominal_camera` invented one.
    """
    # aniposelib.cameras.CameraGroup
    cgroup: object
    offset: dict[str, tuple[float, float]]
    moving: dict[str, bool]
    calibrated: dict[str, bool]

    @property
    def cameras(self):
        """The aniposelib cameras of this rig."""
        return self.cgroup.cameras

    @property
    def names(self) -> list[str]:
        """Camera names in rig order."""
        return [c.get_name() for c in self.cgroup.cameras]

    def __len__(self) -> int:
        """Number of cameras in the rig."""
        return len(self.cgroup.cameras)

    def size(self, name: str) -> tuple[int, int]:
        """Pixel size of one camera.

        Inputs: name -- the camera's name.
        Outputs: (width, height) as ints.
        """
        return tuple(int(v) for v in self.by_name(name).get_size())

    def by_name(self, name: str):
        """The aniposelib camera for a name.

        Inputs: name -- the camera's name.
        Outputs: the Camera object.
        Side effects: raises KeyError if no camera has that name.
        """
        for c in self.cgroup.cameras:
            if c.get_name() == name:
                return c
        raise KeyError(f'no camera named {name!r}')

    def cam_type(self, name: str) -> str:
        """'fisheye' or 'pinhole' for one camera."""
        from aniposelib.cameras import FisheyeCamera
        return 'fisheye' if isinstance(self.by_name(name), FisheyeCamera) else 'pinhole'

    def tracktail(self, device='cpu', moving_ext: dict | None = None) -> list[dict]:
        """tracktail's camera dicts, DETACHED.

        On the torch branch aniposelib's intrinsics/extrinsics are `nn.Parameter`s; detaching stops
        gradients flowing into calibration we are not training.
        """
        import torch
        from tracktail.model.train_utils import format_camera

        out = []
        with torch.no_grad():
            for cam in self.cgroup.cameras:
                n = cam.get_name()
                d = format_camera(cam, {n: self.offset[n]}, self.cam_type(n), device,
                                  ext_override=None if moving_ext is None else moving_ext.get(n))
                camera = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in d.items()}
                camera['calibrated'] = bool(self.calibrated[n])
                out.append(camera)
        return out


def load_calibration(path: Path) -> Rig:
    """Parse a calibration.toml into a Rig. Camera order is the file's section order."""
    with open(path, 'rb') as f:
        doc = tomllib.load(f)
    return rig_from_doc(doc, str(path))


def rig_from_doc(doc: dict, where: str) -> Rig:
    """`load_calibration`'s body, over an already-parsed document.

    Split out for `adopt.plan`, which patches missing camera sizes before building the Rig.
    Distortion vectors are padded to five OpenCV coefficients at this load boundary because
    tracktail's projection helper consumes all five; the on-disk calibration is left unchanged.
    """
    from aniposelib.cameras import CameraGroup

    path = where
    blocks = [b for k, b in doc.items() if k != 'metadata' and isinstance(b, dict)]
    if not blocks:
        raise FormatError(f'{path}: no camera sections')

    cams, offset, moving, calibrated = [], {}, {}, {}
    for block in blocks:
        name = str(block.get('name', ''))
        if 'size' not in block:
            raise FormatError(f'{path}: camera {name!r} has no size')
        if 'matrix' in block:
            camera_doc = dict(block)
            distortions = list(camera_doc.get('distortions') or [])
            if len(distortions) < 5:
                camera_doc['distortions'] = distortions + [0.0] * (5 - len(distortions))
            cams.append(CameraGroup.from_dicts([camera_doc]).cameras[0])
        else:
            cams.append(nominal_camera(name, block['size'], block.get('distortions')))
        offset[name] = tuple(float(v) for v in block.get('offset', (0.0, 0.0)))
        moving[name] = bool(block.get('moving', False))
        calibrated[name] = 'matrix' in block
    return Rig(cgroup=CameraGroup(cams), offset=offset, moving=moving, calibrated=calibrated)


def dump_calibration(path: Path, rig: Rig) -> None:
    """Write calibration.toml: aniposelib's own dicts plus `offset` / `moving`.

    An uncalibrated camera writes only what it knows (name, size, offset), so reading it back
    rebuilds the nominal camera rather than resurrecting an invented matrix as a calibration.
    """
    import toml

    doc = {}
    for i, cam in enumerate(rig.cgroup.cameras):
        name = cam.get_name()
        if rig.calibrated[name]:
            block = cam.get_dict()
        else:
            block = {'name': name, 'size': [int(v) for v in cam.get_size()]}
        block['offset'] = [float(v) for v in rig.offset[name]]
        block['moving'] = bool(rig.moving[name])
        doc[f'cam_{i}'] = block
    doc['metadata'] = getattr(rig.cgroup, 'metadata', {}) or {}
    path.write_text(toml.dumps(doc))
