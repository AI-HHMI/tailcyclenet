#!/usr/bin/env python
"""Run a trained model. The only entry point that touches a checkpoint.

    # every group of a split, cropping from the labels (the GT-crop upper bound)
    pixi run python scripts/infer.py --run runs/w9 --data <dataset> --split test --out pred.npz

    # one session, query-free
    pixi run python scripts/infer.py --run runs/w9 --data <dataset>/test/<session> \\
        --anchor none --out pred.npz

    # crops from a detections file (the deployment number)
    pixi run python scripts/infer.py --run runs/w9 --data <dataset> --boxes dets.npz --out p.npz

A run folder carries its own config and keypoint registry, so `--run` is the whole model
specification and a config/checkpoint mismatch cannot happen.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.checkpoints import load_run
from tailcyclenet.format import Session, load_dataset
from tailcyclenet.infer import ANCHORS, CARRY_SOURCES, InferConfig, run_group


def sessions_for(path: Path, split: str):
    """(dataset_name, [Session]) from either a session directory or a dataset root."""
    path = Path(path)
    if (path / 'session.toml').exists():
        return path.parent.parent.name, [Session.load(path)]
    ds = load_dataset(path)
    return ds.name, ds.sessions.get(split, [])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path, help='dataset root or one session dir')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--anchor', default='carry', choices=ANCHORS,
                    help="'labels' is an ORACLE, not a deployment number")
    ap.add_argument('--overlap', type=int, default=8,
                    help='frames each window shares with its predecessor. 8, and it is the one '
                         'number that suits both dimensions: 3dpop reads -0.626 mm against 4 over '
                         '16 sessions (and -0.355 [-0.751, -0.063] SIG on the 120-frame protocol, '
                         'where overlap 2 costs +0.460 and 12 is no better), while 2D still '
                         'improves at 12 -- but only by +0.787 px [-0.111, +1.951] n.s. over 8 on '
                         'rat-city, at a small monotone identity cost. It is the SEAM COUNT against '
                         'the SEAM SIZE and both terms depend on the clip; sweep it per root.')
    ap.add_argument('--n-frames', type=int, default=None, help='default: the run\'s own')
    ap.add_argument('--boxes', type=Path, default=None,
                    help='npz of crop points per group; default is to crop from the labels')
    ap.add_argument('--detector', type=Path, default=None,
                    help='a detector run folder. THE deployment path: boxes come from pixels, '
                         'not from labels.')
    ap.add_argument('--det-input-wh', type=int, nargs=2, default=None,
                    help='letterbox size the detector was TRAINED at. Only needed for a '
                         'posetail-pose checkpoint, which keeps it in a config file rather than '
                         'in the weights.')
    ap.add_argument('--det-score', type=float, default=0.99,
                    help='objectness floor for a detection. 0.99, not the 0.05 that reads like a '
                         'floor: the objectness is SATURATED -- 98.5%% of rat-city\'s boxes and '
                         '99.98%% of 3dpop\'s sit at exactly 1.0 -- so anything between 0.05 and 0.5 '
                         'moves 1-3%% of boxes and does nothing. At 0.99 it is worth MOTA +0.074 '
                         '[+0.009, +0.154] on 3dpop with MPJPE -2.11 mm and coverage +0.010 also '
                         'SIG, and +0.073 on rat-city. The gain is `fp_none`, which is what a score '
                         'threshold should remove. Its SIZE tracks how many boxes are actually '
                         'below it (6-8%% on those two, 0.9%% on calms21, where it reads +0.012 '
                         'n.s.), so a detector whose scores are not saturated wants this lowered -- '
                         'the per-group box coverage printed below is how that shows up.')
    ap.add_argument('--det-cache', type=Path, default=None,
                    help='npz of per-group detector boxes; read if it exists, written if not. '
                         'Detection is the expensive half of a run and depends on no model, so '
                         'several arms over one clip set should share ONE set of boxes -- which '
                         'makes them a matched comparison by construction instead of by trusting '
                         'the detector to be deterministic. The box source is recorded in it and '
                         'checked on load, so a cache from a different detector cannot be reused '
                         'silently.')
    ap.add_argument('--link-boxes', action=argparse.BooleanOptionalAction, default=True,
                    help='follow one animal per instance row across frames -- per-frame Hungarian '
                         'on centre distance in units of the box side, gated at one side, with '
                         'births into empty rows and expiry after a window. ON BY DEFAULT, and it '
                         'only ever runs where `--track` cannot: `detect_group` builds the tracker '
                         'when `track and C > 1` and otherwise falls through to here, so this is '
                         'the whole of 2D single-view identity. It was off while `--track` was on, '
                         'which left the shipped 2D default with NO cross-frame identity at all -- '
                         'score-ordered rows, and the union crop spanning several animals. Every 2D '
                         'number in reports 11 and 13 came from an explicit opt-in. '
                         '`--no-link-boxes` restores that memoryless pass.')
    ap.add_argument('--box-prompt', default='auto', choices=('auto', 'none', 'labels', 'detector'),
                    help='DEPLOYMENT BOX PROMPT (report 27), 2D single-camera. DEFAULT `auto`: a '
                         'box-model checkpoint deploys with the box (from the detector if one is '
                         'given, else the GT `labels` ORACLE) at --crop-inflate 1.5 + --refine; a '
                         'plain model is unaffected. `labels` is an ORACLE, not a deployment '
                         'number. `none` forces the box off.')
    ap.add_argument('--crop-inflate', type=float, default=None,
                    help='inflate every crop about its centre. DEFAULT: 1.5 when the box is on '
                         '(the WIDE regime where the box is load-bearing, report 27 section 9m), '
                         'else 1.0.')
    ap.add_argument('--crop-source', default='boxes', choices=('boxes', 'keypoints'),
                    help="where the window's crop comes from. 'boxes' (default) unions the "
                         "detector's per-frame boxes -- what every recorded number uses. "
                         "'keypoints' runs THE CROP RULE on the detector's own keypoints over the "
                         'window, which is the one thing a union of boxes cannot reproduce: the '
                         'per-frame extents are unioned BEFORE squaring rather than after. Needs '
                         'a keypoint-trained detector and is ignored without one. Bounded by the '
                         'same overlap test `--refine` carries.')
    ap.add_argument('--max-move', type=float, default=1.0,
                    help='THE GATE, in box sides, shared by `--track` and `--link-boxes`: a '
                         'target-detection pair further apart than this is not the same animal and '
                         'is unavailable to the Hungarian. 1.0 was calibrated against CENTROID '
                         'displacement -- p90 0.06-0.11 body lengths on every multi-animal root, so '
                         '10-16x headroom -- and that calibration does not transfer to any cost '
                         'that measures something other than a centroid. Sweepable so the headroom '
                         'can be checked rather than assumed.')
    ap.add_argument('--track', action=argparse.BooleanOptionalAction, default=True,
                    help='3D multiview only, and ON BY DEFAULT. ONE cross-view target set with one '
                         'affinity and one Hungarian, replacing per-frame `associate` plus '
                         '`link_rows` -- see tailcyclenet/detector/track.py and dev/reports/12. '
                         'Those two never talked to each other, which is where the teleporting rows '
                         'and the starved animals come from. `--no-track` restores the memoryless '
                         'pass. What it buys, measured (dev/reports/13): over 480 frames the error '
                         'of the memoryless pass grows +0.6 mm/window to 39.4 mm while this holds '
                         '12-13 mm flat (-0.02/window), because the union crop widens 193 -> 230 px '
                         'and this keeps it at 187; the worst crop halves (p99 750 -> 376 px, max '
                         '1312 -> 570); box slots filled rise 0.866 -> 0.888 (p10 0.653 -> 0.732); '
                         'and it is 5-8x faster, widening with camera count, which is what makes a '
                         'multi-animal 16-camera rig runnable at all. What it does NOT buy: on '
                         'report 11 §2\'s 120-frame protocol none of report 12 §5\'s pre-registered '
                         'endpoints moves (coverage -0.001, MOTA +0.035, miss -0.016, all n.s.) -- '
                         'that benchmark is six windows long, too short to show a per-window effect. '
                         'It subsumes --link-boxes on a multi-camera rig.')
    ap.add_argument('--min-views', type=int, default=2, choices=(1, 2),
                    help='3D only. 2 is cross-view association as it stands: every instance is '
                         'built from a camera PAIR, so an animal only one view saw is dropped from '
                         'the frame entirely. 1 also emits each leftover box as a single-view '
                         'instance -- coverage against precision, since a leftover box is exactly '
                         'one the geometry never corroborated. Whether the pose model can USE a '
                         'one-camera 3D window is the run\'s own `[data].prob_2d_only`: the shipped '
                         '3dpop and rat-city runs set 0.25, configs/w9.toml sets 0, and under 0 it '
                         'is an untrained input shape. Measured on 3dpop: no metric moves.')
    ap.add_argument('--max-animals', type=int, default=0)
    # DETECTION BUDGET, SEPARATE FROM THE ROW COUNT. `--max-animals` used to set both, so sweeping
    # the row count also moved how many boxes the detector was allowed to emit and neither lever
    # could be read alone (`link_rows`, spare rows). Default 0 = follow `--max-animals`, which is
    # what the two did when they were one number.
    ap.add_argument('--det-top-k', type=int, default=0,
                    help='detections kept per frame-camera; 0 follows --max-animals')
    # THE KEYPOINT IDENTITY CUES (dev/reports/16 §9, 19 §4). All default-off, all VETOES over an
    # unchanged centroid cost -- each may only remove an edge the centre gate already accepted, so
    # `--max-move`'s calibration in box sides is untouched and a wrong cue costs a missed match
    # rather than a wrong one. A detection with too few keypoints ABSTAINS; it never vetoes.
    ap.add_argument('--pose-nms', type=float, default=None, metavar='FRAC',
                    help='INSTANCE-LEVEL NMS on the seated rows (report 20 lead 1, maDLC\'s '
                         '`Assembly.intersection_with`). Drops the lower-scored of two rows whose '
                         'keypoint-containment overlap `min(#kpts of A in B\'s box / |A|, ...)` '
                         'exceeds FRAC; 0.8 is maDLC\'s value. NOT IoU -- two touching animals '
                         'overlap almost equally by IoU and it is zero under fast motion. Aimed at '
                         '`fp_dup`, which is 90%% of calms21\'s detector-minus-GT FP rise and ~10%% '
                         'of 3dpop\'s, so expect it to matter on crowded-overlap roots and be a '
                         'near-no-op elsewhere. Default off; its fire rate is printed. '
                         'QUANTISED: the overlap is a fraction of K keypoints, so at K = 4 it takes '
                         'only {0, .25, .5, .75, 1} and this flag has FIVE meaningful settings, not '
                         'a continuum -- 0.6 and 0.7 are byte-identical (both mean "3 of 4"). Quote '
                         'it as a keypoint COUNT, not as a float, and note that maDLC\'s 0.8 means '
                         'something different at K = 17 than at K = 4. Same trap as '
                         '--min-match-kpts (eval rule 9).')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='predict only the first N frames of each group. A PREFIX, not a sample: '
                         '`carry` needs the frames contiguous.')
    ap.add_argument('--render', type=Path, default=None,
                    help='also write <dir>/<session>__<group>__<cam>.mp4 of the prediction. A 3D '
                         'prediction is REPROJECTED into each rendered camera.')
    ap.add_argument('--render-cams', default='0',
                    help='comma-separated camera indices to render. A reprojection hides a depth '
                         'error along its own ray, so a 3D render wants more than one.')
    ap.add_argument('--render-zoom', type=int, default=0,
                    help='side, in SOURCE pixels, of a window that follows the prediction. 0 '
                         'draws the whole frame, which on a 3208x2200 rig makes a 100 px mouse '
                         'unreadable. A view only -- it changes nothing that was predicted.')
    ap.add_argument('--refine', action=argparse.BooleanOptionalAction, default=None,
                    help='DEFAULT: ON IN 3D, OFF IN 2D -- derived from the session\'s own mode, '
                         'because the two dimensions disagree and the code knows which it is in. '
                         '3dpop over 16 sessions / 47 units, paired, one lever: MPJPE -0.962 mm '
                         '[-2.104, -0.216] SIG, p75 -0.749 SIG, p95 -8.17 SIG, with coverage, MOTA '
                         'and idsw all null -- a clean win, which retires report 11\'s "loses to '
                         '--crop-source keypoints at 2x compute". In 2D it is a TRADE: bulk accuracy '
                         'improves on both roots (rat-city p75 -0.744 SIG, coverage +0.008 SIG; '
                         'calms21 p75 -4.14 SIG) while calms21 identity gets SIG worse (MOTA '
                         '-0.0435, idsw +0.0085, coverage -0.0053) -- 2x the seed floor, so 2D does '
                         'not get it by default. `--refine` / `--no-refine` overrides either way. '
                         'What it does: '
                         're-crop each window to the first pass\'s OWN prediction and predict '
                         'again, label-free. The prediction re-enters the crop rule as if it were '
                         'the labels, so the second pass sees the box a GT crop would have given -- '
                         'the only arm that beat every detector crop on 3dpop. Costs one extra '
                         'forward AND one extra decode per animal per window: the crop moves, so '
                         'neither the pixels nor the scene encode can be reused.')
    ap.add_argument('--refine-px', type=int, default=None,
                    help='run --refine\'s FIRST pass at this input resolution instead of the '
                         'run\'s own image_size. Refine\'s gain is MAGNIFICATION, not coordinate '
                         'frame, so pass 1 only has to LOCALISE. calms21 2D: 96 px beats full-res '
                         'refine outright (6.651 vs 6.765) at a third of the overhead. 3dpop 3D: '
                         '192 is a NULL against 256 (+0.062 mm paired) at a quarter of the pixels, '
                         'and 96-128 trade ~1.7 mm for +0.021 coverage and +0.03 MOTA. 64 IS THE '
                         'CLIFF on both roots. No default: the floor scales with `patch_size` and '
                         'with how big the animal is in the crop, so it is per-root.')
    ap.add_argument('--vis-thresh', type=float, default=None,
                    help='withhold an (animal, frame) row whose MEDIAN `vis_pred` logit across '
                         'keypoints is below this. Measured against a rate-matched random rejection '
                         '-- the only honest control, since any rejection flatters a mean over '
                         'matched points -- it is worth MOTA +0.049 SIG on 3dpop at 7.3%% of rows '
                         'and 0.601 -> 0.628 on rat-city at 14%%, where the control gains nothing. '
                         'A LOGIT, and NOT PORTABLE: rat-city sits at a median of +2.7 and 3dpop at '
                         '+15.4, so pick it per dataset from the run\'s own `conf` field. Applies '
                         'to what is reported, never to the carried prompt.')
    ap.add_argument('--kpt-chunk', type=int, default=0,
                    help='decode keypoints in slices of this size, reusing one scene encode. '
                         'Lowers peak memory on large keypoint sets; the prediction is '
                         'unchanged. 0 = one pass.')
    ap.add_argument('--prefetch-windows', type=int, default=1,
                    help='decode this many windows AHEAD of the one currently forwarding on the '
                         'GPU (dev/reports/31). BIT-EXACT: the prediction is unchanged at any '
                         'value -- the pose loop was decode-bound with the GPU idle during every '
                         'decode, and this overlaps the next window\'s decode with the current '
                         'one\'s forward. `carried` (the anchor/carry prompt) is still read and '
                         'written strictly in window order on the main thread, so the SEQUENCE '
                         'of forwards is untouched. 0 restores the exact old serial loop -- no '
                         'prefetch pool is even created. Costs one extra small `crops` buffer '
                         '(already cropped to `image_size`), not a second full-frame decode.')
    ap.add_argument('--oracle-corrupt', default=None,
                    help='ONLY with --anchor labels, and never a deployment arm: break the oracle '
                         'prior on purpose and see how far the output follows it. `off:<x>` offsets '
                         'the WHOLE pose by x crop widths in one direction (the shape of a lag, '
                         'which i.i.d. training jitter never is); `stale:<n>` supplies the pose from '
                         'n frames earlier; `other` supplies the NEIGHBOURING animal\'s. Those are '
                         'the three failures deployment produces and none of them is in training. '
                         'The ratio of output displacement to prior displacement is the echo '
                         'coefficient alpha, and alpha is what decides whether the prompt needs '
                         'retraining.')
    ap.add_argument('--min-box-frames', type=int, default=1,
                    help='how many finite (frame, camera) boxes a row needs before it gets a window '
                         'crop. 1 is what the loop always did, and it is how coverage gets '
                         'FABRICATED: one box out of T x C positions the crop for all 24 frames and '
                         'every one is marked `ok` -- 3dpop reports 0.000 of (row, frame) with no '
                         'pose against 2.1-2.2%% of (row, frame) with no camera at all. Raising this '
                         'LOWERS reported coverage on purpose.')
    ap.add_argument('--carry-source', default='triangulate', choices=CARRY_SOURCES,
                    help='3D only, and only under --anchor carry: what the next window is seeded '
                         'with. "triangulate" (default) hands back the ANCHOR-FREE estimate, '
                         're-derived from this window\'s pixels every frame. "pred" hands back the '
                         'reported prediction, which under gridresid_offset = "query" is '
                         '`prior + residual` -- a loop with gain, measured on johnson-mouse as a '
                         'sawtooth locked to the window boundary that costs 30%% of the animal\'s '
                         'motion. "pred" is what every arm before this flag existed did; it is kept '
                         'so that comparison can be made. 2D is identical either way.')
    ap.add_argument('--gridresid-offset', default=None, choices=('query', 'triangulated'),
                    help='STATE what this run\'s 3D residual was anchored on, for a run folder '
                         'written before `[model].gridresid_offset` existed. Those weights were '
                         'trained under unconditional per-frame re-anchoring, i.e. '
                         '"triangulated"; loading them as "query" infers `world = prior + '
                         'residual` from a head that learned `world = tri_t + residual_t`. Not a '
                         'sweep knob: on a run whose config names the key this must agree with it.')
    ap.add_argument('--dataset-name', default=None,
                    help='the registry entry to read this data\'s keypoints as, when the folder '
                         'name is not it. `rat-city` served by a `rat-city-annotated` run: same '
                         'names, same ids, different directory. Checked against the session\'s '
                         'own names, so a wrong value raises rather than relabelling.')
    ap.add_argument('--groups', default=None, help='comma-separated group ids to restrict to')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    # EVERY PURE-ARGUMENT CHECK BEFORE THE CHECKPOINT LOADS. These cost nothing and a typo
    # should not cost a 5.6 GB load; it also makes them testable without a GPU.

    if args.anchor == 'labels':
        # AN ORACLE PRIOR AND DETECTOR ROWS ARE INCOMPATIBLE, not merely a bad idea. `run_group`
        # seeds row `a` from LABEL row `a`, and once boxes come from a detector row `a` is a
        # score-ordered or association-ordered slot that is not label row `a` for any `a` -- so the
        # arm that exists to be an upper bound was being handed a different animal's ground truth,
        # and read as a poor oracle rather than as a broken one.
        if args.detector or args.boxes:
            raise SystemExit(
                '--anchor labels seeds row `a` from LABEL row `a`, but --detector/--boxes rows are '
                'score- or association-ordered and are not label rows. That is a different '
                "animal's ground truth, not an oracle. Use --anchor labels with the label crop "
                'path (no --detector, no --boxes), or --anchor carry / none with a box source.')
        print('WARNING: --anchor labels seeds the model with GROUND TRUTH. This is an oracle '
              'upper bound, not a deployment number. Label it as such wherever you quote it.')

    if args.oracle_corrupt:
        from tailcyclenet.infer import ORACLE_CORRUPTIONS
        kind = args.oracle_corrupt.split(':')[0]
        if kind not in ORACLE_CORRUPTIONS:
            raise SystemExit(f'--oracle-corrupt {args.oracle_corrupt!r}: kind must be one of '
                             f'{ORACLE_CORRUPTIONS} (off:<x> | stale:<n> | other)')
        if kind in ('off', 'stale') and ':' not in args.oracle_corrupt:
            raise SystemExit(f'--oracle-corrupt {kind} needs an amount, e.g. {kind}:0.5')
        if args.anchor != 'labels':
            raise SystemExit('--oracle-corrupt breaks the ORACLE prior, so it only means anything '
                             f'under --anchor labels; got {args.anchor!r}.')
        print(f'*** DIAGNOSTIC: the oracle prior is deliberately corrupted '
              f'({args.oracle_corrupt}). This measures the echo coefficient and is not a '
              'prediction of anything. ***')

    if args.refine_px is not None:
        # `--refine` is tri-state now (None = derive from the session mode), so only an EXPLICIT
        # `--no-refine` is a contradiction. Leaving it unset is fine and means "wherever refine runs".
        if args.refine is False:
            raise SystemExit('--refine-px sets the resolution of --refine\'s FIRST pass, and '
                             '--no-refine turns that pass off. Pick one.')
        # THE STRUCTURAL FLOOR, and it is the spatial analogue of gotcha 1. At `patch_size` 16 a
        # 16 px input gives a 1x1 token grid and the forward returns ALL-NaN with no exception,
        # while `run_group` still marks every window `ok` -- only `coverage 0.0000` reveals it.
        # 8 px raises. Two patches is the smallest input that can carry a spatial relation at all;
        # the MEASURED floor is far above it (64 px is 3.2x worse than not refining, on both roots).
        if args.refine_px < 32:
            raise SystemExit(f'--refine-px {args.refine_px} is below the structural floor of 32 '
                             '(about two patches). Below ~2 patches the forward returns all-NaN '
                             'with no exception. The measured floor is 96; 64 is already worse '
                             'than not refining at all.')

    device = args.device if torch.cuda.is_available() else 'cpu'
    over = ({'gridresid_offset': args.gridresid_offset} if args.gridresid_offset else None)
    model, config, registry, ckpt = load_run(args.run, args.checkpoint, device=device,
                                             model_overrides=over)
    print(f'model: {ckpt.name}  ({registry.n_keypoints} keypoints)  '
          f'query={config["model"].get("query", "prior")}  '
          f'gridresid_offset={config["model"]["gridresid_offset"]}')

    trained_frames = int(config['data'].get('n_frames', 24))
    # LONGER THAN THE TRAINED WINDOW IS NOT A KNOB. `n_frames` sizes the temporal pos_embed the
    # checkpoint carries; asking for more frames than it was built for interpolates at best and
    # shape-errors deep inside the encoder at worst. Shorter is safe -- val/test already enumerate
    # fixed windows -- so only the ceiling is guarded.
    if args.n_frames and args.n_frames > trained_frames:
        raise SystemExit(f'--n-frames {args.n_frames} exceeds the run\'s trained window '
                         f'({trained_frames}). Shorter windows are fine; longer is not the same '
                         'model.')
    trained_px = int(config['data'].get('image_size', 256))
    # LARGER THAN THE TRAINED INPUT IS NOT A KNOB EITHER, for a different reason: `PadToSize` only
    # ever pads UP, so a bigger input is not padded, not resized, and reaches the 2D head's fixed
    # `image_size`-wide canvas as an out-of-range position. Smaller is what the compensations in
    # `model._input_extent` exist for.
    if args.refine_px and args.refine_px > trained_px:
        raise SystemExit(f'--refine-px {args.refine_px} exceeds the run\'s image_size '
                         f'({trained_px}). A reduced first pass is the lever; a larger one is a '
                         'different model.')
    # THE BOX DEPLOYMENT RECIPE (report 27) is the DEFAULT FOR A BOX MODEL and inert otherwise, so
    # a plain run is unchanged and a box run needs no flags. `--box-prompt auto` resolves to
    # `detector` (if a detector/boxes file is given) or the `labels` ORACLE, and pulls
    # crop_inflate -> 1.5, refine -> on, refine_px -> 128 unless the user set them. Explicit flags
    # always win. A box on a plain model, or a box source with no boxes, resolves to `none`.
    model_is_box = config.get('model', {}).get('box_prompt', 'none') != 'none'
    box_prompt = args.box_prompt
    if box_prompt == 'auto':
        box_prompt = ('detector' if (model_is_box and (args.detector or args.boxes))
                      else 'labels' if model_is_box else 'none')
    if box_prompt != 'none' and not model_is_box:
        print(f'--box-prompt {box_prompt}: this run is not a box model '
              '([model].box_prompt = "none"), so the box is ignored.')
        box_prompt = 'none'
    box_on = box_prompt != 'none'
    crop_inflate = args.crop_inflate if args.crop_inflate is not None else (1.5 if box_on else 1.0)
    refine = args.refine if args.refine is not None else (True if box_on else None)
    refine_px = args.refine_px if args.refine_px is not None else (128 if box_on else None)
    if box_on:
        print(f'box deployment recipe (report 27): --box-prompt {box_prompt} '
              f'--crop-inflate {crop_inflate} --refine{"" if refine else " off"} '
              f'--refine-px {refine_px}')
    cfg = InferConfig(
        n_frames=args.n_frames or trained_frames,
        overlap=args.overlap, image_size=trained_px,
        min_crop_dim=int(config['data'].get('min_crop_dim', 64)),
        box_source=config['data'].get('box_source', 'keypoints'),
        anchor=args.anchor, max_animals=args.max_animals, max_frames=args.max_frames,
        kpt_chunk=args.kpt_chunk,
        vis_thresh=args.vis_thresh, refine=refine, refine_px=refine_px,
        carry_source=args.carry_source, min_box_frames=args.min_box_frames,
        oracle_corrupt=args.oracle_corrupt, device=device,
        crop_source=args.crop_source,
        box_prompt=box_prompt, crop_inflate=crop_inflate,
        prefetch_windows=args.prefetch_windows)
    if cfg.box_source != 'keypoints':
        print(f'crops: box_source={cfg.box_source} (from the run config); a session with no '
              'instances.pq falls back to its keypoints')

    boxes = dict(np.load(args.boxes, allow_pickle=True)) if args.boxes else {}
    # `det_tile` and `det_red` are initialised HERE and not only inside the branch: both go into
    # the cache stamp unconditionally, and a NameError there would only fire on the box-source
    # paths that have no detector at all -- which is exactly how `det_kpts` shipped broken.
    det = det_wh = det_tile = None
    det_red = False
    if args.detector:
        from tailcyclenet.detector import associate_group, detect_raw, load_detector
        det, det_wh, det_ds, det_mcd, det_red, det_boxsrc, det_tile, det_objq = load_detector(
            args.detector, device, input_wh=args.det_input_wh)
        # `--det-score` IS NOT PORTABLE ACROSS DETECTOR GENERATIONS, and this is the only place
        # that can tell. 0.99 was measured on detectors whose objectness is saturated; a
        # tiled/masked one reads q01 0.45-0.84 and loses two thirds of its detections to the same
        # number -- coverage 0.703 against 0.986 at 0.50 (dev/reports/21 0b). A warning and not a
        # refusal, and no automatic threshold: which value is right depends on whether coverage or
        # identity is the objective (0.50 maximises the first, 0.97 the second), and choosing on
        # the caller's behalf would hide the trade.
        if det_objq and args.det_score >= det_objq.get('q50', 0.0):
            print(f'WARNING: --det-score {args.det_score} is at or above this detector\'s MEDIAN '
                  f'objectness ({det_objq["q50"]:.4f}), so it discards at least half of the '
                  f'detections it offers. q01 {det_objq.get("q01", float("nan")):.4f} '
                  f'q10 {det_objq.get("q10", float("nan")):.4f} '
                  f'q90 {det_objq.get("q90", float("nan")):.4f}. This detector is NOT saturated; '
                  '0.99 was measured against ones that are.', flush=True)
        # A TILE-TRAINED DETECTOR IS DEPLOYED ON THE WHOLE FRAME AT ITS TRAINING SCALE, and
        # `det_wh` is its TILE size. `detect_group` derives the per-camera input from `det_tile`
        # (frame sizes vary WITHIN a root -- rat-city-annotated ships 4696x2048 beside 4500x2050),
        # so `det_wh` is only a fallback here and is printed as the tile it is.
        print(f'detector: {args.detector} ({det_wh[0]}x{det_wh[1]}'
              + (f' TILE at scale {det_tile:g}, whole-frame input derived per camera'
                 if det_tile else '')
              + f', trained on {det_ds!r}, boxes={det_boxsrc or "keypoints"})')
        # The detector regresses THE CROP RULE'S box, so its floor has to be the pose model's
        # floor. Same shapes and same losses if they differ, so nothing else would say so.
        if det_mcd != cfg.min_crop_dim:
            raise SystemExit(
                f'{args.detector}: trained at min_crop_dim={det_mcd}, but this run\'s '
                f'[data].min_crop_dim is {cfg.min_crop_dim}. The detector reproduces the crop '
                'rule; two floors are two rules.')
        # THE OTHER HALF OF THE SAME CONTRACT, and it is not a hard failure: a detector trained on
        # `instances` boxes is the best rat-city detector on record (recall 0.531 vs 0.429) while
        # every rat-city pose run was trained on keypoint-extent crops, so this arm is a legitimate
        # thing to run -- it just is not a detector-quality comparison against a keypoint-trained
        # one, because the crop source moved too (eval rule 4). The npz records which.
        if (det_boxsrc or 'keypoints') != cfg.box_source:
            print(f'WARNING: detector boxes are {det_boxsrc!r} but this run was trained on '
                  f'{cfg.box_source!r} crops. Two crop sources are two crop rules -- do not read '
                  'a delta against a run whose detector matched as a detector-quality result.')
    ds_name, sessions = sessions_for(args.data, args.split)
    # A registry is keyed by DATASET NAME, so deploying a run on a root it was not trained on --
    # the whole point of a shared keypoint vocabulary -- otherwise dies on the folder name alone.
    # `rat-city` and `rat-city-annotated` are the shipped instance: identical `names`, therefore
    # identical registry ids and the same embedding rows, and different directories. This is safe
    # to override because it is CHECKED rather than assumed: `Registry.ids_for` aligns to the
    # session's own `names` and raises on any name the registry does not hold, which is gotcha 4's
    # guard and the reason a per-dataset id vector is never applied positionally.
    if args.dataset_name:
        print(f'registry: reading session keypoints as {args.dataset_name!r}, not {ds_name!r}')
        ds_name = args.dataset_name
    want = set(args.groups.split(',')) if args.groups else None
    render_cams = [int(c) for c in args.render_cams.split(',') if c.strip() != '']

    # The box cache. `stamp` is every input the boxes depend on: reusing a cache written under a
    # different detector, threshold or animal count would quietly make one arm incomparable to
    # the next, which is the kind of mismatch that gets published (eval rule 4).
    #
    # ONLY THE OPTIONS THAT DIFFER FROM THEIR DEFAULTS, and sorted by name. A positional list of
    # every value invalidated every cache on disk each time a new flag was added to the end of it --
    # three times in one afternoon, twice in the middle of a sweep, each time refusing boxes that
    # were in fact exactly the right boxes. Under this form a new option carrying its default leaves
    # the stamp untouched, while any option a run actually set is still in it.
    #
    # `det_score` is UNCONDITIONAL, and that is the exception the rule needs: its default moved from
    # 0.05 to 0.99, so "equals the default" means different boxes before and after. Omitting it would
    # let a cache built under the old default be reused silently under the new one -- precisely the
    # mismatch this stamp exists to catch. Anything whose default may move belongs on this line.
    # `link_rev` for the same reason `det_score` is unconditional: the cache holds boxes that have
    # already been through `link_rows`, so changing that rule silently makes an old cache a
    # different box set. It USED to appear only when linking was on -- which was safe only while
    # linking defaulted off. Now that `--link-boxes` defaults ON, "equals the default" means a
    # different box set before and after, exactly as it did for `track`: a cache written under the
    # old default carries no `link_rev` and no `link_boxes` entry, and under this line is REFUSED
    # rather than reused as if it had been linked. This is the FOURTH instance of the same trap.
    #
    # `track` IS UNCONDITIONAL TOO, and for the third instance of the same reason: its default moved
    # from off to on, so "equals the default" means a different box set before and after. Every cache
    # written while it was off carries no `track` entry, and under this line those caches are now
    # REFUSED rather than silently reused as if they had been tracked -- which is the whole point of
    # the stamp. Deleting them is correct; reproducing an untracked arm needs `--no-track`.
    #
    # `tile_scale` is UNCONDITIONAL for the FIFTH instance of the same trap, and this one is not
    # about a moved default: it comes from the CHECKPOINT rather than the command line, so two runs
    # can differ in it with identical arguments. A tile-trained detector is deployed at a different
    # whole-frame input size than a letterbox-trained one, which is a different box set under an
    # otherwise identical stamp. Every cache written before the key existed carries no entry and is
    # now refused rather than reused as if the scales matched.
    #
    # `reduce` is unconditional for the SAME reason, and it is the second checkpoint-derived one:
    # `load_detector` reads it off the checkpoint, `detect_group` uses it to pick the decode
    # resolution the detector sees, and a different decode resolution is a different box set. It
    # was in neither list, so two detectors differing only in it shared a cache silently.
    #
    # **THE CACHE NOW HOLDS RAW DETECTIONS, SO THE ASSOCIATION OPTIONS LEAVE THIS STAMP.** `track`,
    # `link_boxes`, `link_rev`, `max_animals`, `min_views`, `dup_res_px` and `max_move` change only
    # what happens AFTER detection, and `associate_group` re-runs on every invocation -- microseconds
    # per frame against 44 ms of 4K decode. So one cache now serves every identity arm, which makes
    # those arms matched BY CONSTRUCTION rather than by trusting the detector to be deterministic
    # (eval rule 4), and is the whole reason the split exists.
    #
    # `raw_rev` is UNCONDITIONAL and is the SIXTH instance of the `det_score` trap, and the sharpest:
    # a raw cache and an associated one are the same shape, the same dtype and the same key names.
    # An old cache read as raw would be associated a SECOND time, silently. There is no default to
    # move here -- the meaning of the file changed -- so every cache written before the split is
    # refused. That is correct and nearly free: `scratch/phase*` caches hold BatchNorm-detector boxes
    # and are refused by `load_detector` already.
    #
    # `top_k` is what the raw depends on where `max_animals` used to be. It is one key rather than
    # two because the dependency is genuinely one thing: `--det-top-k` when set, and the animal count
    # `max_animals` implies when not. Conditional membership is what makes a stamp lie, so the key is
    # always present and its VALUE says which rule produced it.
    from tailcyclenet.detector import RAW_REV
    top_k_stamp = (str(args.det_top_k) if args.det_top_k
                   else f'from-max-animals:{args.max_animals}')
    stamp = repr(sorted([('det_score', str(args.det_score)), ('raw_rev', str(RAW_REV)),
                         ('top_k', top_k_stamp),
                         ('tile_scale', str(det_tile)), ('reduce', str(det_red))]
                        + [(k, str(getattr(args, k))) for k in
                           ('detector', 'det_input_wh', 'max_frames')
                           if getattr(args, k) != ap.get_default(k)]))
    det_cache, cache_dirty = {}, False
    if args.det_cache and args.det_cache.exists():
        loaded = dict(np.load(args.det_cache, allow_pickle=True))
        got = str(loaded.pop('__stamp__', ''))
        if got != stamp:
            raise SystemExit(f'{args.det_cache}: written for {got}\n'
                             f'{" " * len(str(args.det_cache))}  now running {stamp}\n'
                             'Delete it or point --det-cache somewhere else.')
        det_cache = loaded
        print(f'detector boxes: '
              f'{sum(1 for k in det_cache if not k.endswith(("|score", "|kpt")))} '
              f'cached group(s) from {args.det_cache}')

    results = {}
    # `renders` holds the futures still encoding; drained after the npz is written, so a slow encode
    # never delays the prediction reaching disk.
    render_pool = ThreadPoolExecutor(max_workers=1)
    renders = []
    for sess in sessions:
        sess.preload()
        for gid in sess.groups:
            if want and gid not in want:
                continue
            key = f'{sess.session_id}/{gid}'
            # ALL THREE INITIALISED TOGETHER, OUTSIDE THE BRANCH. `det_kpts` used to be bound
            # inside it, so every box source that is NOT a detector -- the GT-crop upper bound and
            # the whole `--boxes` path -- raised `UnboundLocalError` at the `run_group` call below,
            # after paying the checkpoint load. Same trap the comment above `det_tile` names.
            det_boxes = det_scores = det_kpts = None
            if det is not None:
                # Default to what the session actually holds, not to 1. `associate_group` caps
                # rows at this count, so a bare --detector run on a ten-animal dataset
                # used to return one animal per frame and read as a catastrophic miss rate
                # rather than as a missing flag.
                n_want = args.max_animals or max(1, len(sess.labels(gid).animal_ids))
                # DETECT AT `top_k`, ASSOCIATE AT `n_want`. They were one number until the split,
                # and welding them meant a sweep over the row count also moved the detection budget
                # -- which is why `link_rows`' spare-rows finding could not be run end to end.
                n_det = args.det_top_k or n_want
                if key in det_cache:
                    raw = (det_cache[key], det_cache.get(f'{key}|score'),
                           det_cache.get(f'{key}|kpt'))
                    # A CACHE WITHOUT KEYPOINTS CANNOT SERVE `--crop-source keypoints`, and the
                    # failure would be SILENT: `run_group` takes `det_kpts_stc is not None` as the
                    # switch, so a None here does not error, it quietly crops from the boxes and
                    # reports the arm under the other arm's name. That is the `--boxes`-key trap
                    # below, one flag over. Refused rather than warned, because the whole purpose of
                    # a shared cache is that two arms differ in exactly one lever.
                    if args.crop_source == 'keypoints' and raw[2] is None:
                        raise SystemExit(
                            f'{args.det_cache}: holds no keypoints for {key!r}, so --crop-source '
                            'keypoints would silently fall back to cropping from the boxes and '
                            'measure the arm it is being compared against. Delete it and re-detect '
                            '(a keypoint-trained detector fills this in), or drop --det-cache.')
                    # `flush` for the same reason the detecting branch has it: redirected to a log,
                    # stdout is block-buffered, so the CACHED path -- the fast one, which prints
                    # little else -- shows nothing for minutes and reads as a hung run.
                    print(f'{key}: up to {n_want} animal(s), raw boxes from --det-cache', flush=True)
                else:
                    print(f'{key}: detecting up to {n_det} per camera, {n_want} animal row(s)'
                          f'{"" if args.max_animals else " (from the labels; set --max-animals)"}',
                          flush=True)
                    _t_det = time.time()
                    raw = detect_raw(det, det_wh, sess, gid, n_det, device=device,
                                     score_thresh=args.det_score, reduce=det_red,
                                     max_frames=args.max_frames, tile_scale=det_tile)
                    _det_secs = time.time() - _t_det
                    # A keypoint-trained detector fills a third array, cached under its own key.
                    # This does NOT change what an old cache is allowed to satisfy: a box-only arm
                    # never looks at it, and a keypoint-crop arm is refused above rather than served
                    # boxes under the wrong name. Storing it is what lets the two crop sources share
                    # ONE box set and so differ in exactly one lever (eval rule 4) -- report 15 §6
                    # had to match its item-3 arms by configuration for want of this.
                    det_cache[key] = raw[0]
                    det_cache[f'{key}|score'] = raw[1]
                    if raw[2] is not None:
                        det_cache[f'{key}|kpt'] = raw[2]
                    cache_dirty = True
                    # CHECKPOINT AN EXPENSIVE GROUP IMMEDIATELY, rather than only at the end of the
                    # run. The end-of-run write below is still the one that matters for a short
                    # protocol, but it held rat-city's 57,594-frame group -- 3h18m of decode, 62 GB
                    # of JPEG -- in memory alone across a further ~3h pose pass, so any interruption
                    # lost the entire detection. That cache IS the artifact the long-clip benchmark
                    # exists to produce ("detect once, then every association arm is a CPU-minute"),
                    # and it did not survive the run that creates it.
                    #
                    # GATED ON THE TIME THE DETECTION ACTUALLY TOOK, not on a frame count: 60 s is
                    # "long enough that losing it would hurt", which is exactly the quantity in
                    # question. A 58-group protocol detects each group in seconds and so writes
                    # once, at the end, byte-identical to before; a single long group writes exactly
                    # once, immediately. Only a root with many SLOW groups pays repeated
                    # compression, and there it is buying back hours.
                    if args.det_cache and _det_secs > 60.0:
                        args.det_cache.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(args.det_cache, __stamp__=np.asarray(stamp),
                                            **det_cache)
                        print(f'{key}: detection took {_det_secs / 60:.1f} min -- checkpointed '
                              f'{args.det_cache}', flush=True)
                # THE ASSOCIATION HALF RUNS EVERY TIME, cached or not. It is microseconds per frame
                # against 44 ms of 4K decode, so recomputing it costs nothing measurable and buys
                # the property the cache exists for: two identity arms differ in exactly one lever
                # over byte-identical pixels.
                nms_stats = {}
                det_boxes, det_scores, det_kpts = associate_group(
                    raw, sess, gid, n_want, link=args.link_boxes, min_views=args.min_views,
                    track=args.track, max_move=args.max_move, stats=nms_stats,
                    pose_nms=args.pose_nms)
                # THE FIRE RATE IS THE NUMBER A RATE-MATCHED RANDOM CONTROL MUST BE MATCHED TO, and
                # it cannot be recovered afterwards from the npz -- a dropped row leaves no trace.
                # "Report the fire rate before the metric" means printed, not recoverable.
                if args.pose_nms is not None:
                    # `.get(..., 0)`, both keys: `identity.pose_nms` returns before writing EITHER
                    # key when the detector has no keypoint branch (`kpts is None`) -- a correct
                    # no-op, since the maDLC overlap it computes needs keypoints to exist at all.
                    # A keypoint-less detector is the NORMAL case for a 2D root (rat-city's own
                    # recipe omits --keypoints), so `nms_stats` being empty here is not a bug
                    # signal, and asserting `nms_pairs` unconditionally raised on every such run.
                    print(f'{key}: pose-nms dropped {nms_stats.get("nms_dropped", 0)} row(s) of '
                          f'{nms_stats.get("nms_pairs", 0)} overlapping pair(s)'
                          + (' (no keypoint branch -- pose-nms is a no-op)' if not nms_stats
                             else ''), flush=True)
                # HOW MUCH THE THRESHOLD LEFT. `--det-score` defaults to 0.99 because objectness is
                # saturated on every detector shipped here; a detector whose scores are NOT
                # saturated would lose most of its boxes to that, and this line is where that shows
                # up rather than downstream as an unexplained miss rate.
                filled = float(np.isfinite(det_boxes).all(-1).mean())
                print(f'{key}: boxes in {filled:.3f} of (animal, frame, camera) slots'
                      f'{"   <-- LOW: try a smaller --det-score" if filled < 0.25 else ""}')
            # A MISSING `--boxes` KEY IS NOT AN ABSENT ARGUMENT. `boxes.get(key)` returning None
            # silently falls back to cropping from the LABELS, so a run whose keys did not match --
            # a different session naming, a stale npz, a typo in one group -- reported the GT-crop
            # oracle under a heading that said otherwise. Nothing in the output said which.
            if args.boxes and key not in boxes:
                raise SystemExit(
                    f'{args.boxes}: no entry for {key!r}. Falling back to the labels here would '
                    'quietly turn this into the GT-crop upper bound. Keys present: '
                    f'{sorted(k for k in boxes if not k.startswith("__"))[:5]} ...')
            out = run_group(model, sess, gid, registry, ds_name, cfg,
                            box_points=boxes.get(key), boxes_stc=det_boxes,
                            det_kpts_stc=det_kpts)
            if det_scores is not None:
                # The objectness each crop was accepted on, beside the prediction it produced.
                # `--det-score` is then an offline sweep instead of a re-detection per threshold.
                out['det_score'] = det_scores[:, :out['pred'].shape[1]]
            results[key] = out
            print(f'{key}: {out["pred"].shape} '
                  f'{np.isfinite(out["pred"]).all(-1).mean():.3f} finite')
            if args.render is not None:
                from tailcyclenet.render import render_group
                # RENDER ON A BACKGROUND THREAD so the loop can predict the next group while this
                # one encodes. A render of a 480-frame 4696x2048 clip is comparable in cost to the
                # inference that produced it, and it depends on nothing the loop mutates afterwards
                # -- `out['pred']` and `det_boxes` are finished arrays by here.
                #
                # ONE worker, not several. `_read_video`'s lock is now PER CONTAINER, so two
                # renders of two cameras would genuinely overlap their decodes -- but a render
                # holds a clip's worth of full frames and the encode, not the decode, is where a
                # render's time goes. The reason is memory now, not the lock.
                for ci in render_cams:
                    cam_name = sess.cam_names[ci]
                    # The per-frame boxes the crop rule was fed, in each row's own colour: a box
                    # with no skeleton in it is the disagreement worth seeing, and row `a` is not
                    # label row `a` once boxes come from a detector. `crop` is per WINDOW and the
                    # windows overlap, so it is not the array to draw here.
                    bx = (det_boxes[:, :out['pred'].shape[1], ci]
                          if det_boxes is not None else None)
                    renders.append((key, render_pool.submit(
                        render_group, sess, gid, out['pred'],
                        args.render / f'{key.replace("/", "__")}__{cam_name}.mp4',
                        cam=ci, zoom=args.render_zoom, boxes=bx)))
                # Report whatever has landed, without waiting for anything.
                for k, fut in [r for r in renders if r[1].done()]:
                    print(f'{k}: wrote {fut.result()}')
                    renders.remove((k, fut))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, out in results.items():
        for field, value in out.items():
            flat[f'{key}|{field}'] = value
    flat['__keys__'] = np.asarray(list(results), object)
    flat['__run__'] = np.asarray(str(args.run))
    flat['__anchor__'] = np.asarray(cfg.anchor)
    flat['__boxes__'] = np.asarray(
        str(args.detector) if args.detector else
        (str(args.boxes) if args.boxes else 'labels'))
    # WHICH CROP RULE PRODUCED THESE PIXELS, in the file rather than in a shell history. The run's
    # own `[data].box_source` on the label path; the detector's own on the deployment path, which
    # is the one that can disagree with it.
    flat['__box_source__'] = np.asarray((det_boxsrc or 'keypoints') if args.detector
                                        else cfg.box_source)
    # WHAT THE CROP WAS BUILT FROM, which `__box_source__` above does NOT say -- that one is the
    # detector's TRAINING target. Two arms differing only in `--crop-source` were previously
    # identical in their own provenance, so report 15 §6's pair could be told apart only by
    # filename. `--refine` rides here too, being the other re-crop lever, so a three-way comparison
    # is legible from the files alone.
    # `refine` is resolved per session from its mode, so read the RESOLVED flag off the results
    # rather than off `cfg`, which may still hold the tri-state None.
    did_refine = any(bool(r.get('refine')) for r in results.values())
    flat['__crop_source__'] = np.asarray(
        f'{cfg.crop_source}{"+refine" if did_refine else ""}'
        f'{f"@{cfg.refine_px}px" if did_refine and cfg.refine_px else ""}')
    np.savez_compressed(args.out, **flat)
    print(f'wrote {args.out} ({len(results)} group(s))')

    # THE PREDICTION IS ON DISK FIRST, then the renders are waited on. A render is a view and must
    # never be able to lose a run that has already paid for its inference.
    if renders:
        print(f'waiting on {len(renders)} render(s) still encoding', flush=True)
    for k, fut in renders:
        print(f'{k}: wrote {fut.result()}')
    render_pool.shutdown()

    # AFTER the prediction is on disk, never before: the cache is an optimisation for the next
    # run and a failure writing it must not lose the run that just paid for the detection.
    if args.det_cache and cache_dirty:
        args.det_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.det_cache, __stamp__=np.asarray(stamp), **det_cache)
        # COUNT GROUPS, NOT KEYS. Each group holds boxes plus `|score` plus `|kpt` where a
        # keypoint detector supplied them, so `len()` read 174 for 58 groups -- it was already
        # double-counting before the keypoint key existed, which is why 2x looked plausible.
        print(f'wrote {args.det_cache} '
              f'({sum(1 for k in det_cache if not k.endswith(("|score", "|kpt")))} '
              'group(s) of boxes)')


if __name__ == '__main__':
    main()
