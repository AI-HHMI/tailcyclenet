#!/usr/bin/env python
"""Run a trained model. The only entry point that touches a checkpoint.

    # one session directory, cropping from the labels (the GT-crop upper bound)
    pixi run python scripts/infer.py --run runs/<name> --data <dataset>/test/<session> --out pred/

    # one session, query-free
    pixi run python scripts/infer.py --run runs/<name> --data <dataset>/test/<session> \\
        --anchor none --out pred/

    # crops from a detections file (the deployment number)
    pixi run python scripts/infer.py --run runs/<name> --data <dataset>/test/<session> \\
        --boxes dets.npz --out pred/

    # raw footage plus an anipose calibration, straight off the camera files
    pixi run python scripts/infer.py --run runs/<name> --out pred/ --videos rec/ \\
        --calibration anipose/calibration.toml --cam-regex 'cam([0-9]+)_' \\
        --detector runs/det-<name> --max-animals 4

`--out` is a prediction SESSION DIRECTORY (session.toml, calibration.toml, groups.pq and the
label tables), written a block at a time -- not an npz. A run folder carries its own config and
keypoint registry, so `--run` is the whole model specification and a config/checkpoint mismatch
cannot happen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .window import ANCHORS, CARRY_SOURCES




def build_parser() -> argparse.ArgumentParser:
    """The command line. A function, not a module-level block, so a test can assert against the
    parser object rather than regex this file."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path)
    # EXACTLY ONE INPUT, and the mutual exclusion is checked by hand below `parse_args` so the
    # error names BOTH flags rather than argparse's one-sided message.
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--data', type=Path,
                     help='ONE session directory (a dataset root works only if it holds a '
                          'single session in --split). --out is one session, and a session '
                          'carries one calibration, one mode and one keypoint axis.')
    src.add_argument('--videos', type=Path, nargs='+', default=None,
                     help='RAW FOOTAGE INSTEAD OF A SESSION DIRECTORY: files and/or directories '
                          '(a directory expands to its .mp4/.avi children, sorted, NOT '
                          'recursively). THE SESSION IS BUILT IN MEMORY -- nothing is staged, and '
                          'nothing is written but --out. Needs --calibration, and --cam-regex '
                          'unless the calibration names exactly one camera. NO LABELS, therefore '
                          'NO SCORING: scripts/eval.py has nothing to compare a video-sourced '
                          'prediction against, and a number needs annotations, i.e. a converter. '
                          'It also makes --max-animals and a box source (--detector/--boxes) '
                          'mandatory, since neither can be recovered from footage.')
    ap.add_argument('--calibration', type=Path, default=None,
                    help='with --videos: an ANIPOSELIB-LAYOUT calibration.toml -- what '
                         '`format.load_calibration` reads, which anipose itself writes and this '
                         'repo dumps. "It is a toml" is NOT the same claim as "it is THIS toml": '
                         # `%%`: argparse expands every help string as `help % params`, so a bare
                         # `%Y` is a format spec and `--help` dies for every flag at once.
                         'a multiview_calib / OpenCV-YAML rig (`%%YAML:1.0` Cam*.yaml, `rc_ext` '
                         'as '
                         'a rotation MATRIX) needs a conversion script ending in '
                         '`format.dump_calibration`. That is a converter\'s job and stays out of '
                         'this flag.')
    ap.add_argument('--cam-regex', default=None,
                    help="with --videos: anipose's own `[triangulation] cam_regex`, applied to "
                         "each video's STEM -- the camera is `search(rx, stem).group(1)` and the "
                         'group id is `sub(rx, "", stem)`. So `cam0_trial3.mp4` under '
                         "'cam([0-9]+)_' is camera '0', group 'trial3', and `cam1_trial3.mp4` "
                         'joins it as the same group\'s second camera. THE CAMERA NAME IS THE '
                         "CAPTURE GROUP, so it is '0' and not 'cam0' -- the calibration must name "
                         'it that way, and this is the number-one thing to get wrong. Two '
                         'documented supersets of anipose: a pattern with NO capture group '
                         "matches the whole string ('Cam[0-9]+' -> 'Cam2005325', which is what a "
                         'raw rig dump needs), and a one-camera calibration needs no regex at '
                         'all. Several groups in one invocation is expected and free.')
    ap.add_argument('--session-id', default=None,
                    help='with --videos: the `session/group` key every downstream table is '
                         "written under. Default: the videos' common parent directory name.")
    ap.add_argument('--group-id', default=None,
                    help='with --videos: used ONLY when the regex leaves every remainder empty, '
                         'i.e. one raw recording per invocation (`Cam2005325.mp4` under '
                         "'Cam[0-9]+' leaves ''). INERT otherwise. Some empty and some not is a "
                         'genuine ambiguity and is refused. Default: the session id.')
    ap.add_argument('--units', default='mm',
                    help='with --videos: the 3D units. A DECLARATION about the calibration, not a '
                         'measurement, and it CANNOT BE CHECKED here -- a calibration in metres '
                         'declared as mm produces a prediction 1000x off with no symptom, because '
                         "nothing downstream knows the animal's size.")
    ap.add_argument('--fps', type=float, default=None,
                    help="with --videos: override the container's own fps. Reaches groups.pq "
                         'only; nothing in inference reads it.')
    ap.add_argument('--assoc-res-max-px', type=float, default=30.0,
                    help='with --videos, 3D only: `Session.assoc_res_max_px`, which '
                         '`CrossViewTracker` and `associate` both read as their reprojection '
                         "gate. The format's default, and it was measured on NO ad-hoc rig -- it "
                         'is a PIXEL gate on a reprojection residual, so a 4K rig and a 640x480 '
                         'rig should not share it. Sweep per rig, exactly as --det-score is swept '
                         'per checkpoint.')
    ap.add_argument('--trim-to-shortest', action='store_true',
                    help="with --videos: opt in to n_frames = min over a group's cameras. "
                         'Without it a length disagreement is REFUSED, because a one-frame offset '
                         '(a dropped trigger, usually harmless) and a 40,000-frame one (two '
                         'different recordings sharing a group id) look identical to a min().')
    ap.add_argument('--dump-session', type=Path, default=None,
                    help='DEBUG ONLY: write the constructed --videos session out through '
                         '`format.write_session`, pixels symlinked, and carry on -- so '
                         '`validate_session` and your own eyes can be pointed at what the flags '
                         'produced. Not the mechanism, and on no default path.')
    # DEFAULT None, NOT 'test': the videos path has to tell whether it was PASSED (it is inert
    # there and would otherwise be silently ignored).
    ap.add_argument('--split', default=None,
                    help='default: test. Inert with --videos, and refused rather than ignored.')
    ap.add_argument('--out', required=True, type=Path,
                    help='the prediction SESSION directory to write: session.toml, '
                         'calibration.toml, groups.pq and the label tables, in '
                         'docs/annotation_format.md. No pixels -- [provenance] '
                         'source_session says where they are.')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--anchor', default='carry', choices=ANCHORS,
                    help="'labels' is an ORACLE, not a deployment number")
    ap.add_argument('--overlap', type=int, default=4,
                    help='frames each window shares with its predecessor. Default 4; 8 suits 3D, '
                         '2D improves through 12 (seam count vs seam size -- sweep per root).')
    ap.add_argument('--n-frames', type=int, default=None, help='default: the run\'s own')
    ap.add_argument('--boxes', type=Path, default=None,
                    help='npz of crop points per group; default is to crop from the labels')
    ap.add_argument('--detector', type=Path, default=None,
                    help='a detector run folder. THE deployment path: boxes come from pixels, '
                         'not from labels. A run folder defaults to its latest complete '
                         'detector_it*.pth; pass --detector-checkpoint best for historical '
                         'detector.pth or pass an explicit checkpoint filename.')
    ap.add_argument('--detector-checkpoint', default='latest',
                    help="detector checkpoint selector when --detector is a run folder: 'latest' "
                         '(default), \'best\' (historical detector.pth), or an explicit filename.')
    ap.add_argument('--allow-detector-transfer', action='store_true', default=False,
                    help='permit a detector checkpoint whose recorded [dataset] does not match '
                         'this session\'s own dataset root. A detector is trained per dataset '
                         '(scale/appearance are dataset-specific), so this is refused by default; '
                         'pass this flag only for an explicit, labelled transfer-evaluation arm.')
    ap.add_argument('--det-input-wh', type=int, nargs=2, default=None,
                    help='letterbox size the detector was TRAINED at. Only needed for a '
                         'posetail-pose checkpoint, which keeps it in a config file rather than '
                         'in the weights.')
    ap.add_argument('--det-score', type=float, default=0.01,
                    help='objectness floor for a detection. 0.01 (default, CHANGED from 0.5 then '
                         '0.05) is measured against the current iou_aware_obj+COCO-pretrained '
                         'recipe: on rat-city-combined, MOTA peaks at 0.01 (0.795, dev/scratch/'
                         'detscore/ratcity_v2best_lowend_sweep.log) -- clearly above 0.05 (0.640) '
                         'and above 0.0 (0.747, more false positives with no MOTA gain); on '
                         'allen-mouse-combined and 3dpop, 0.01 is BYTE-IDENTICAL to 0.05 (no '
                         'detections score in that band at all), so lowering further is free '
                         'there. False negatives cost more than the extra false positives this '
                         'invites -- accepted deliberately: 3D multiview already rejects a '
                         'candidate whose reprojection residual exceeds `assoc_res_max_px` '
                         '(default 30px, Session.assoc_res_max_px) after association, and NMS '
                         '(centre-distance + IoU) already suppresses duplicates before it: a '
                         'looser floor feeds MORE candidates into filters that already exist, not '
                         'an unfiltered flood. 2D single-camera has no reprojection check to fall '
                         'back on -- `--pose-nms` is the nearest lever (off by default, helps '
                         'rat-city 2D, hurts calms21; sweep per root). Still sweep per checkpoint.')
    ap.add_argument('--det-nms-iou', type=float, default=0.5,
                    help="decode's box-NMS IoU threshold. 0.5 was HARDCODED and unreachable "
                         'before detector_v2 plan A1; every external detector (RTMDet 0.65, '
                         'DLC/SLEAP instance-level 0.8) ships higher -- sweep upward, not around '
                         '0.5.')
    ap.add_argument('--det-nms-center-dist', type=float, default=0.5,
                    help='SECOND, independent NMS survival test in units of box side (scale-free): '
                         'a candidate is also suppressed if its centre sits within this many box '
                         'sides of an already-kept box, regardless of IoU -- catches '
                         'near-concentric duplicates IoU-only NMS misses. 0.5 (default) is '
                         'CONFIRMED (detector_v2 plan A5, 2 seeds/2 roots, cuts fp_dup a lot); '
                         'pass a negative value to disable and restore the pre-A5 behaviour.')
    ap.add_argument('--link-boxes', action=argparse.BooleanOptionalAction, default=True,
                    help='follow one animal per instance row across frames -- per-frame Hungarian '
                         'on centre distance in units of the box side, gated at one side. The '
                         'whole of 2D single-view identity; `--no-link-boxes` restores the '
                         'memoryless pass.')
    ap.add_argument('--box-prompt', default='auto', choices=('auto', 'none', 'labels', 'detector'),
                    help='DEPLOYMENT box prompt, 2D single-camera. `auto` deploys a box model with '
                         'the detector box (--crop-inflate 1.5 + --refine) and REFUSES to run a '
                         'box model without a detector/boxes file; `labels` is an explicit ORACLE; '
                         '`none` forces the box off.')
    ap.add_argument('--crop-inflate', type=float, default=None,
                    help='inflate every crop about its centre. DEFAULT 1.5 when the box is on, '
                         'else 1.0.')
    ap.add_argument('--crop-source', default='boxes', choices=('boxes', 'keypoints'),
                    help="where the window's crop comes from. 'boxes' (default) unions the "
                         "detector's per-frame boxes; 'keypoints' runs THE CROP RULE on the "
                         'detector\'s own keypoints (needs a keypoint-trained detector, ignored '
                         'without one).')
    ap.add_argument('--max-move', type=float, default=1.0,
                    help='THE GATE, in box sides, shared by `--track` and `--link-boxes`: a pair '
                         'further apart than this is not the same animal. 1.0 is ~10x headroom on '
                         'every multi-animal root.')
    ap.add_argument('--track', action=argparse.BooleanOptionalAction, default=True,
                    help='3D multiview only, ON BY DEFAULT: ONE cross-view target set with one '
                         'affinity and one Hungarian, replacing per-frame `associate` plus '
                         '`link_rows`. Buys scale and flat error over long clips; `--no-track` '
                         'restores the memoryless pass.')
    ap.add_argument('--min-views', type=int, default=2, choices=(1, 2),
                    help='3D only. 2 builds every instance from a camera PAIR (single-view animals '
                         'dropped); 1 also emits leftover boxes as single-view instances. Whether '
                         'the pose model can use one is [data].prob_2d_only.')
    ap.add_argument('--max-animals', type=int, default=0)
    # Detection budget, separate from the row count: `--max-animals` used to set both, so neither
    # lever could be read alone. 0 = follow `--max-animals`.
    ap.add_argument('--det-top-k', type=int, default=0,
                    help='detections kept per frame-camera; 0 follows --max-animals')
    # The keypoint identity cues: all default-off VETOES over an unchanged centroid cost -- each
    # may only remove an edge the centre gate already accepted, and a detection with too few
    # keypoints abstains.
    ap.add_argument('--pose-nms', type=float, default=None, metavar='FRAC',
                    help='INSTANCE-LEVEL NMS on the seated rows: drop the lower-scored of two rows '
                         'whose keypoint-containment overlap exceeds FRAC. NOT IoU. QUANTISED by K '
                         '(at K=4 there are five settings). Default off.')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='predict only the first N frames of each group (a PREFIX, so `carry` '
                         'stays contiguous). = --start-frame 0 --end-frame N; refused together '
                         'with either.')
    # The frame range serves both input paths: it is a window-loop lever, not an input-format
    # one, which is why these sit outside any input-specific group.
    ap.add_argument('--start-frame', type=int, default=0,
                    help='first SOURCE frame to predict, per group (half-open [start, end)). The '
                         'output `frame` column is always the source index. A ranged run is not a '
                         'slice of the whole-clip answer: identity columns are comparable only '
                         'between runs that start at the same frame.')
    ap.add_argument('--end-frame', type=int, default=0,
                    help='one past the last SOURCE frame to predict, per group; 0 = to the end, '
                         'past the end clamps. A group shorter than --start-frame is skipped by '
                         'name; every group skipped is refused. NOT A RESUME.')
    ap.add_argument('--refine', action=argparse.BooleanOptionalAction, default=None,
                    help='DEFAULT: on in 3D, off in 2D (derived from the session\'s own mode). '
                         'Re-crops each window to the first pass\'s OWN prediction and predicts '
                         'again, label-free. Costs one extra forward + decode per animal per '
                         'window.')
    ap.add_argument('--refine-px', type=int, default=None,
                    help='run --refine\'s FIRST pass at this resolution instead of the run\'s '
                         'image_size (the gain is magnification; pass 1 only localises). 96-192 is '
                         'the plateau, 64 the cliff. No default.')
    ap.add_argument('--vis-thresh', type=float, default=None,
                    help='withhold an (animal, frame) row whose median `vis_pred` logit is below '
                         'this. A logit, not portable across datasets; applies to what is '
                         'reported, never the carried prompt.')
    ap.add_argument('--kpt-chunk', type=int, default=0,
                    help='decode keypoints in slices of this size, reusing one scene encode. '
                         'Lowers peak memory on large keypoint sets; the prediction is '
                         'unchanged. 0 = one pass.')
    ap.add_argument('--prefetch-windows', type=int, default=1,
                    help='decode this many windows AHEAD of the one forwarding. BIT-EXACT: the '
                         'prediction is unchanged at any value. 0 restores the serial loop.')
    ap.add_argument('--oracle-corrupt', default=None,
                    help='ONLY with --anchor labels, and never a deployment arm: break the oracle '
                         'prior on purpose and see how far the output follows it. `off:<x>` offsets '
                         'the WHOLE pose by x crop widths in one direction (the shape of a lag, '
                         'which i.i.d. training jitter never is); `stale:<n>` supplies the pose from '
                         'n frames earlier; `other` supplies the NEIGHBOURING animal\'s (row a+1); '
                         '`near` supplies the NEAREST ELIGIBLE animal\'s pose instead of the fixed '
                         'a+1 row; `swap:<n>` transposes n pairs of this row\'s own keypoints. '
                         'None of these is in training except through dataset.py\'s '
                         'prompt_swap_kpt_pairs / prompt_swap_animal, which swap/near are the '
                         'direct probes for. The ratio of output displacement to prior displacement '
                         'is the echo coefficient alpha, and alpha is what decides whether the '
                         'prompt needs retraining.')
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
    ap.add_argument('--max-ram', type=float, default=None, metavar='GB',
                    help='HOST RAM this run may spend on its own pixel buffers -- the decord '
                         'reader cache, how many CAMERAS the detector decodes at once, and the '
                         "window loop's camera concurrency and frame cache. DEFAULT: derived, as "
                         'the smallest of the cgroup limit (walked up EVERY ancestor), LSF\'s own '
                         'LSB_CG_MEMLIMIT/LSB_MEMLIMIT, MemAvailable and MemTotal -- never '
                         'SC_PHYS_PAGES, which is the MACHINE\'s memory and under a 16 GB cap '
                         'still reported this host\'s 503 GB, a 20x over-estimate and exactly the '
                         'case an LSF job dies in. EVERY KNOB THIS SIZES IS OUTPUT-NEUTRAL, so '
                         'lowering it costs wall clock and never a predicted number -- which is '
                         'why it sizes the CAMERA axis and never the detector batch: batch is NOT '
                         'inert (16 vs 3 moves boxes 0.204 px, scores 1.7e-03), see '
                         'tests/test_memory_budget.py. '
                         'TAILCYCLENET_MAX_RAM_GB does the same for paths with no CLI.')
    return ap


def main(argv=None):
    """Parse and run; `scripts/infer.py` is a shim onto this."""
    from .driver import run_dataset

    ap = build_parser()
    args = ap.parse_args(argv)
    # Exactly one of, checked by hand so the error names both flags.
    if bool(args.data) == bool(args.videos):
        ap.error('exactly one of --data (a session directory in docs/annotation_format.md) or '
                 '--videos (raw footage plus --calibration) is required.')
    if args.videos and not args.calibration:
        ap.error('--videos needs --calibration: an aniposelib-layout calibration.toml. There is '
                 'no geometry in a filename.')
    if args.calibration and not args.videos:
        ap.error('--calibration only means anything with --videos; a session directory carries '
                 'its own calibration.toml.')
    return run_dataset(args)
