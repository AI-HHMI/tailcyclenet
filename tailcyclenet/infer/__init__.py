"""The inference program: the window loop, the driver over a dataset, and the command line.

Private names are re-exported on purpose -- `detector/evaluate.py` imports `_window_starts` --
so dropping one breaks callers silently at import time.
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
