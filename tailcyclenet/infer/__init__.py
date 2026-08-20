"""THE inference program: the window loop, the driver over a dataset, and the command line.

`scripts/infer.py` is a shim onto `main` here. Splitting it that way is what lets the driver be
imported and tested rather than read as text -- see `dev/plans/infer_one_entry_point_and_ram_budget.md`.

THE PRIVATE NAMES ARE RE-EXPORTED ON PURPOSE. Five of them are imported across this boundary
today, and one is production rather than test code: `detector/evaluate.py` takes `_window_starts`
so `eval_detector.py --deploy` scores over the same windows the pose loop uses. Dropping them from
this list breaks callers silently at import time, which is the one thing a pure move must not do.
"""
from .cli import build_parser, main
from .driver import run_dataset
from .store import FrameStore
from .window import (ANCHORS, CARRY_SOURCES, ORACLE_CORRUPTIONS, OUTCOMES, InferConfig,
                     boxes_from_points, merge_blocks, run_blocks, run_group, self_prompt,
                     _build_prior, _corrupt_prior, _crop_views, _deploy_box_prompt,
                     _window_starts)

__all__ = ['ANCHORS', 'CARRY_SOURCES', 'ORACLE_CORRUPTIONS', 'OUTCOMES', 'FrameStore',
           'InferConfig', 'boxes_from_points', 'merge_blocks', 'run_blocks', 'run_group',
           'self_prompt', 'run_dataset', 'build_parser', 'main',
           '_build_prior', '_corrupt_prior', '_crop_views', '_deploy_box_prompt', '_window_starts']
