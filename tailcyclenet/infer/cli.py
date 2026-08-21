#!/usr/bin/env python
"""Run a trained model. The only entry point that touches a checkpoint.

    # every group of a split, cropping from the labels (the GT-crop upper bound)
    pixi run python scripts/infer.py --run runs/<name> --data <dataset> --split test --out pred.npz

    # one session, query-free
    pixi run python scripts/infer.py --run runs/<name> --data <dataset>/test/<session> \\
        --anchor none --out pred.npz

    # crops from a detections file (the deployment number)
    pixi run python scripts/infer.py --run runs/<name> --data <dataset> --boxes dets.npz --out p.npz

A run folder carries its own config and keypoint registry, so `--run` is the whole model
specification and a config/checkpoint mismatch cannot happen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .window import ANCHORS, CARRY_SOURCES




def build_parser() -> argparse.ArgumentParser:
    """THE command line. A function, not a module-level block, so a test can assert against the
    parser object rather than regex this file -- which is what several of them used to do."""
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
    # DEFAULT None, NOT 'test'. The videos path has to be able to tell whether it was PASSED,
    # since it is inert there and silently ignoring it would let a user believe they selected
    # something. The directory path resolves `args.split or 'test'`.
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
                    help='frames each window shares with its predecessor. The default is 4 by '
                         'direct instruction (2026-08-21); the measured evidence says 8 suits 3D '
                         '(3dpop -0.626 mm against 4 over 16 sessions, and -0.355 [-0.751, -0.063] '
                         'SIG on the 120-frame protocol, where overlap 2 costs +0.460 and 12 is no '
                         'better), while 2D still improves at 12 -- but only by +0.787 px '
                         '[-0.111, +1.951] n.s. over 8 on rat-city, at a small monotone identity '
                         'cost. It is the SEAM COUNT against the SEAM SIZE and both terms depend '
                         'on the clip; sweep it per root.')
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
    ap.add_argument('--det-score', type=float, default=0.5,
                    help='objectness floor for a detection. 0.5, NOT 0.99: saturation is a '
                         'property of the RECIPE, not the dataset, and the current detector '
                         'generation is NOT saturated (q01 0.45-0.84) -- 0.99 keeps only 26-33%% '
                         'of detections (coverage 0.703, MOTA 0.622) where 0.5 keeps ~99%% '
                         '(coverage 0.986, MOTA 0.524) and 0.97 maximises identity (MOTA 0.723). '
                         '0.5 is the coverage-favouring default; 0.97 is the identity-favouring '
                         'choice. Sweep per checkpoint: read the objectness quantiles recorded in '
                         'it, and watch the per-group box coverage printed below.')
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
                         'box-model checkpoint deploys with the box from the detector at '
                         '--crop-inflate 1.5 + --refine; a plain model is unaffected. `auto` '
                         'REFUSES to run a box model without a detector/boxes file rather than '
                         'silently falling back to ground truth -- `labels` is an EXPLICIT opt-in '
                         'ORACLE (it warns), `none` forces the box off.')
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
                         'runs set 0.25, configs/3d.toml sets 0, and under 0 it '
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
                         '`carry` needs the frames contiguous. Exactly --start-frame 0 '
                         '--end-frame N, and it stays because every recorded protocol number is a '
                         '--max-frames 120 invocation. REFUSED together with either of them '
                         'rather than ordered: "120 frames from 300" and "up to frame 120, from '
                         '300" read with equal force, and a precedence rule would make one of '
                         'them silently wrong.')
    # THE FRAME RANGE SERVES BOTH INPUT PATHS, because it is a WINDOW-LOOP lever and not an
    # input-format one: a group carries its TRUE length either way and the range bounds only what
    # is PREDICTED. That is what lets one implementation serve a session directory and raw videos
    # alike, and it is why these sit outside any input-specific group.
    ap.add_argument('--start-frame', type=int, default=0,
                    help='first SOURCE frame to predict, per group. Half-open [start, end) -- '
                         "range's convention, --chunk's convention, and every slice in this repo. "
                         'THE `frame` COLUMN IN THE OUTPUT IS ALWAYS THE SOURCE INDEX, and '
                         'groups.pq keeps the group\'s FULL n_frames, so `load_predictions` hands '
                         'back a full-length array that is NaN outside the range and eval.py / '
                         '--chunk / --vs need no change. Re-basing to 0 would score frames '
                         '[start, end) against labels [0, end-start) -- the `chunk_frames` '
                         'failure exactly (coverage 0.4656 against 0.9891), which survived '
                         'because it looks like a pipeline degrading over a clip. '
                         'A RANGED RUN IS NOT A SLICE OF THE WHOLE-CLIP ANSWER: the detector '
                         'boxes are byte-identical and the per-frame accuracy columns are '
                         'comparable, but --track and --link-boxes carry state across frames and '
                         '--anchor carry has no prior at `start`, so the IDENTITY columns are '
                         'comparable only between two runs that START at the same frame.')
    ap.add_argument('--end-frame', type=int, default=0,
                    help='one past the last SOURCE frame to predict, per group. 0 = to the end. '
                         'Past the end CLAMPS, exactly as --max-frames does. A group shorter than '
                         '--start-frame is SKIPPED BY NAME rather than refused, because a ragged '
                         'root with one short group must still be runnable -- but if EVERY group '
                         'is skipped the run IS refused: writing an empty session and exiting 0 '
                         'is the worst of both. '
                         'NOT A RESUME. It bounds which frames are predicted; it does not '
                         'reconstruct the tracker or the carried prior the whole-clip run would '
                         'have held at that frame.')
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
    """Parse and run. `scripts/infer.py` is a six-line shim onto this."""
    from .driver import run_dataset

    ap = build_parser()
    args = ap.parse_args(argv)
    # EXACTLY ONE OF, checked by hand so the message names both. argparse's mutually-exclusive
    # group cannot express "required" here without making its own error name only one of them.
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
