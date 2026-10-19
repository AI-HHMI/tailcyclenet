"""The driver: one run over a dataset. Everything `run_group` is not.

Flag resolution and the refusals that must happen before a checkpoint loads, the group loop with
detection and association interleaved, and the prediction session written a block at a time. One
source session per run: `--out` names one session directory, and a session holds one calibration,
one mode and one keypoint axis.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .detect import detector_boxes
from .loading import load_deploy_detector, load_pose, load_scorer, load_source
from .provenance import run_provenance
from .resolve import (infer_config, refuse_before_load, refuse_out_dir, refuse_scorer_rig,
                      select_groups, trained_window)
from .session_writer import SessionWriter
from .traces import write_det_trace, write_overlap_trace
from .window import run_blocks


def _fmt_hms(seconds: float) -> str:
    """`H:MM:SS`, or `?` for a not-yet-observed rate (`seconds` is NaN or negative)."""
    if not seconds == seconds or seconds < 0:
        return '?'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


class _Progress:
    """Frames processed/remaining across the WHOLE run (every group), throttled to print no more
    than once per `min_interval` seconds so it can be called from inside the per-block loop
    without spamming stdout on a fast run. `total` is frames, not windows or blocks -- the same
    unit the per-group summary line already reports, so the two are directly comparable.

    The rate is a running average from the run's own start, not an instantaneous one: block sizes
    vary (memory-derived), and per-block timing would make the ETA jitter with the block boundary
    rather than the actual pace of the run.
    """

    def __init__(self, total: int, min_interval: float = 15.0):
        """Start the clock. `total` is the frame count the run expects to process overall;
        `min_interval` is the minimum wall-clock gap between printed lines."""
        self.total = max(int(total), 0)
        self.done = 0
        self.t0 = time.time()
        self._last_print = self.t0
        self.min_interval = min_interval

    def update(self, n: int, force: bool = False) -> None:
        """Add `n` finished frames and print a progress line if `min_interval` has elapsed
        since the last one, the run just completed, or `force` says to print unconditionally."""
        self.done += int(n)
        now = time.time()
        if not force and (now - self._last_print) < self.min_interval and self.done < self.total:
            return
        self._last_print = now
        elapsed = now - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else float('nan')
        frac = self.done / self.total if self.total else 1.0
        print(f'progress: {self.done}/{self.total} frames ({frac:.1%})  '
              f'elapsed {_fmt_hms(elapsed)}  remaining {_fmt_hms(remaining)}  '
              f'({rate:.1f} frames/s)', flush=True)


def _report_group(key, wall, stats, det_stats, detected):
    """Print one group's wall/decode split, box fill, association counts and window outcomes.

    Inputs: key ('session/group'), wall (s), stats (`run_blocks`' stats dict), det_stats
        (`detector_boxes`' stats), detected (a detector ran). Outputs: None. Side effects: prints.
    """
    _dec, _h, _m = (stats.get('decode_s', 0.0), stats.get('decode_hits', 0),
                    stats.get('decode_misses', 0))
    print(f'{key}: {wall:.1f}s wall, {_dec:.1f}s of decode work across threads '
          f'({_dec / max(wall, 1e-9):.2f}x wall), store {_h}/{_h + _m} hits '
          f'({_h / max(_h + _m, 1):.2f})')
    if detected:
        _sl = det_stats.get('slots', 0)
        if _sl:
            filled = det_stats.get('filled', 0) / _sl
            print(f'{key}: boxes in {filled:.3f} of (animal, frame, camera) slots'
                  f'{"   <-- LOW: try a smaller --det-score" if filled < 0.25 else ""}')
        _raw = det_stats.get('association_raw_offered', 0)
        _pre = det_stats.get('association_pre_link', 0)
        _kept = det_stats.get('association_kept', 0)
        if _raw:
            print(f'{key}: association offered {_raw}, seated before link {_pre}, '
                  f'kept {_kept}')
    _window_counts = {k: v for k, v in stats.items() if k.startswith('window_')}
    if _window_counts:
        print(f'{key}: window outcomes ' + ' '.join(
            f'{k.removeprefix("window_")}={v}' for k, v in sorted(_window_counts.items())))


def run_dataset(args):
    """One inference run: refuse, load, then predict every group into one prediction session.

    Inputs: args -- the parsed `infer` namespace (`cli.build_parser`).
    Outputs: None. Side effects: writes the prediction session under `--out` a block at a time
        (and the optional traces); prints progress and per-group summaries.

    Every pure-argument check runs before the checkpoint loads. The RAM budget is resolved once
    before anything allocates and re-resolved after the weights are resident. The group loop is
    STREAMED: each block is written as it finishes, so nothing is proportional to clip length;
    `f0` is a SOURCE frame index, asserted rather than trusted.
    """
    refuse_before_load(args)

    from tailcyclenet import memory as _memory
    _budget = _memory.current(override_gb=args.max_ram)
    print(f'ram: {_budget}')

    refuse_out_dir(args)
    ds_name, sess, src_prov = load_source(args)
    refuse_scorer_rig(args.scorer, sess)
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config, registry, ckpt = load_pose(args, device)
    trained_frames, trained_px = trained_window(args, config)
    scorer_run = load_scorer(args, model, registry, trained_frames, trained_px, device)
    scorer, _, scorer_shared_encoder, _ = scorer_run
    cfg = infer_config(args, config, sess, trained_frames, trained_px, device)
    boxes = dict(np.load(args.boxes, allow_pickle=True)) if args.boxes else {}
    det, det_wh, det_red, det_tile, det_boxsrc = load_deploy_detector(args, device, ds_name, cfg)
    if args.dataset_name and not args.videos:
        print(f'registry: reading session keypoints as {args.dataset_name!r}, not {ds_name!r}')
        ds_name = args.dataset_name
    _budget = _memory.rebudget(override_gb=args.max_ram)
    print(f'ram: {_budget}')

    gids = select_groups(args, sess, cfg)
    writer = SessionWriter(args.out, sess, registry,
                           run_provenance(args, sess, ds_name, ckpt, cfg, config, src_prov,
                                          scorer_run, det_tile, det_red, det_boxsrc),
                           gids, quality_enabled=scorer is not None)
    det_trace_groups = {}
    overlap_rows = {}
    _completed = False
    try:
        sess.preload()
        _total_frames = sum(max(0, min(sess.groups[g].n_frames,
                                      cfg.frame_stop or cfg.max_frames or sess.groups[g].n_frames)
                                - cfg.frame_start) for g in gids)
        progress = _Progress(_total_frames)
        for gid in gids:
            key = f'{sess.session_id}/{gid}'
            boxes_for, det_stats, n_want = None, {}, 0
            if det is not None:
                n_want = args.max_animals or max(1, len(sess.labels(gid).animal_ids))
                n_det = args.det_top_k or n_want
                print(f'{key}: detecting up to {n_det} per camera, {n_want} animal row(s)'
                      f'{"" if args.max_animals else " (from the labels; set --max-animals)"}',
                      flush=True)
                boxes_for = detector_boxes(
                    det, det_wh, sess, gid, args, device, det_red, det_tile, n_det, n_want,
                    stats=det_stats, box_source=det_boxsrc)
            if args.boxes and key not in boxes:
                raise SystemExit(
                    f'{args.boxes}: no entry for {key!r}. Falling back to the labels here would '
                    'quietly turn this into the GT-crop upper bound. Keys present: '
                    f'{sorted(k for k in boxes if not k.startswith("__"))[:5]} ...')
            f0, w0 = cfg.frame_start, 0
            n_frames = n_fin = n_pt = 0
            _t_group, _stats = time.time(), {}
            if args.overlap_trace:
                _stats['capture_overlap_agreement'] = True
            for blk in run_blocks(model, sess, gid, registry, ds_name, cfg,
                                  box_points=boxes.get(key), boxes_for=boxes_for, n_rows=n_want,
                                  stats=_stats, scorer=scorer,
                                  shared_scene_encoder=scorer_shared_encoder):
                assert f0 == int(blk['window_start'][0]), (f0, int(blk['window_start'][0]))
                writer.write_block(gid, blk, f0, w0)
                writer.write_window_records(blk.get('window_records', ()))
                f0 += blk['pred'].shape[1]
                w0 += blk['outcome'].shape[1]
                n_frames += blk['pred'].shape[1]
                n_fin += int(np.isfinite(blk['pred']).all(-1).sum())
                n_pt += int(np.isfinite(blk['pred']).all(-1).size)
                progress.update(blk['pred'].shape[1])
            span = ('' if not (cfg.frame_start or cfg.frame_stop)
                    else f' [{cfg.frame_start}, {f0})')
            print(f'{key}: {n_frames} frames{span}, {n_fin / max(n_pt, 1):.3f} finite')
            if args.det_trace and det is not None:
                det_trace_groups[key] = det_stats.get('decode_trace', [])
            if args.overlap_trace:
                overlap_rows[key] = (_stats.get('overlap_agreement', []),
                                     _stats.get('overlap_census', {}),
                                     _stats.get('prior_oob', []),
                                     _stats.get('model_conf', []))
            writer.write_identity_events(gid, det_stats.get('identity_events', []))
            _report_group(key, time.time() - _t_group, _stats, det_stats, det is not None)
        _completed = True

    finally:
        writer.close(complete=_completed)
    progress.update(0, force=True)
    if args.det_trace:
        write_det_trace(args.det_trace, det_trace_groups)
        print(f'wrote detector trace {args.det_trace}')
    if args.overlap_trace:
        write_overlap_trace(args.overlap_trace, overlap_rows)
    if args.identity_bridge:
        from . import bridge as _bridge
        _cfg = _bridge.BridgeConfig()
        _decisions = _bridge.bridge_session(args.out, _cfg, int(cfg.n_frames))
        print(_bridge.summarise(_decisions))
    print(f'wrote {args.out} ({len(gids)} group(s))')

