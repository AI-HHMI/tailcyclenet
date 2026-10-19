"""Reading prediction sessions (and legacy `.npz` predictions); the writer is `session_writer`.

A session holds one calibration, mode and keypoint axis, so a run covering multiple source
sessions is refused rather than merged (`refuse_multi_session`).
"""
from __future__ import annotations

from pathlib import Path


from ..format import sessions_for


def refuse_multi_session(data, split):
    """One session per run, checked before the checkpoint loads -> the single session.

    A session directory holds one calibration, one mode and one keypoint axis, so a prediction
    session cannot represent several at once.
    """
    ds_name, sessions = sessions_for(Path(data), split)
    if len(sessions) != 1:
        raise SystemExit(
            f'--data covers {len(sessions)} session(s) and --out is ONE session directory. A '
            'session holds one calibration, one mode and one keypoint axis, and these do not '
            'agree. Point --data at a single session directory (tailcyclenet reads one '
            f'directly), or run once per session. Sessions: '
            f'{[s.session_id for s in sessions][:5]}')
    return ds_name, sessions[0]
