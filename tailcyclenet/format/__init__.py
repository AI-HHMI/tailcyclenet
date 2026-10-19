"""Read, write and validate the `tailcycle-dataset` format; `docs/annotation_format.md` is the spec.

`from tailcyclenet import format as fmt` is the API: the status codes and `FormatError` live here,
and every public name of the submodules is re-exported (calibration, tables, session, write,
discovery, registry, validate). The codes are defined BEFORE the submodule imports, which read them.
"""
from __future__ import annotations

# Visibility codes, shared by both label tables. -1 doubles as "no row".
# PROJECTED: a position with no visibility claim -- never train visibility on it.
UNLABELED, MISSING, VISIBLE, PROJECTED = -1, 0, 1, 2
# The statuses that carry coordinates.
POSITIONED = (VISIBLE, PROJECTED)
# Instance codes. -1 doubles as "no row" == no determination.
INST_NONE, INST_ABSENT, INST_PRESENT, INST_LABELED = -1, 0, 1, 2

KPT_STATUS = {'unlabeled': UNLABELED, 'missing': MISSING, 'visible': VISIBLE,
              'projected': PROJECTED}
INST_STATUS = {'absent': INST_ABSENT, 'present': INST_PRESENT, 'labeled': INST_LABELED}
IMAGE_EXTS = ('.png', '.jpg')
VIDEO_EXTS = ('.mp4', '.avi')
SPLITS = ('train', 'val', 'test')
# session.toml `labels`: who produced the labels. Closed vocabulary, so a typo is a FormatError;
# exposed on Session as `label_source` so as not to shadow the `Session.labels()` method.
LABEL_SOURCES = ('annotated', 'tracked')


class FormatError(Exception):
    """A dataset on disk does not satisfy docs/annotation_format.md."""


from .calibration import nominal_camera, nominal_rig, Rig, load_calibration, rig_from_doc, dump_calibration  # noqa: E402,F401
from .tables import TableWriter, write_table, DICT_COLS  # noqa: E402,F401
from .session import Labels, triangulate_group, Group, Session, video_group, VideoSession, empty_labels  # noqa: E402,F401
from .write import write_session, link  # noqa: E402,F401
from .discovery import Dataset, load_dataset, sessions_for, load_datasets  # noqa: E402,F401
from .registry import Registry  # noqa: E402,F401
from .validate import validate_session, validate_dataset  # noqa: E402,F401
from .prediction import load_predictions  # noqa: E402,F401
