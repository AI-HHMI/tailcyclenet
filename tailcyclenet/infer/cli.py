#!/usr/bin/env python
"""Run a trained model. The only entry point that touches a checkpoint.

    # one session directory, cropping from the labels (the GT-crop upper bound)
    pixi run python scripts/infer.py --pose runs/<name> --data <dataset>/test/<session> --out pred/

    # one session, query-free
    pixi run python scripts/infer.py --pose runs/<name> --data <dataset>/test/<session> \\
        --anchor none --out pred/

    # crops from a detections file (the deployment number)
    pixi run python scripts/infer.py --pose runs/<name> --data <dataset>/test/<session> \\
        --boxes dets.npz --out pred/

    # raw footage plus an anipose calibration, straight off the camera files
    pixi run python scripts/infer.py --pose runs/<name> --out pred/ --videos rec/ \\
        --calibration anipose/calibration.toml --cam-regex 'cam([0-9]+)_' \\
        --detector runs/det-<name> --max-animals 4

`--out` is a prediction SESSION DIRECTORY (session.toml, calibration.toml, groups.pq and the
label tables), written a block at a time -- not an npz. A run folder carries its own config and
keypoint registry (and so does each of its checkpoint files), so `--pose` -- older spelling
`--run`, kept -- is the whole model specification and a config/checkpoint mismatch
cannot happen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..detector.boxes import NMS_CENTER_DIST
from .window import ANCHORS, CARRY_SOURCES

REMOVED_FLAGS = {
    '--pose-nms': 'it pruned rows by detector-keypoint containment, and the detector keypoint '
                  'branch is deleted (every shipped detector had n_keypoints = 0); v0.3.4 is the '
                  'last release that has it',
    '--crop-source': "its only other value, 'keypoints', needed the deleted detector keypoint "
                     'branch; crops always come from boxes; v0.3.4 is the last release that has '
                     'it',
    '--independent-windows': 'per-window association was unmeasured and opt-in; v0.3.4 is the '
                             'last release that has it',
    '--fill-from': 'its own help called it actively harmful with no gate; --identity-bridge '
                   'drops the quarantined rows; v0.3.4 is the last release that has it',
    '--det-input-wh': 'posetail-pose detector compat; a detector checkpoint records its own '
                      'input size; v0.3.4 is the last release that has it',
}


class _RemovedFlag(argparse.Action):
    """A deleted flag: passing it exits naming the flag and why, never silently ignored."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Refuse the flag by name with its `REMOVED_FLAGS` reason."""
        raise SystemExit(f'{option_string} is deleted: {REMOVED_FLAGS[option_string]}.')


def _add_removed_flags(ap):
    """Register every `REMOVED_FLAGS` entry as a hidden flag that refuses itself by name."""
    for flag in REMOVED_FLAGS:
        ap.add_argument(flag, action=_RemovedFlag, nargs='*', help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    """The `infer` command line, as a parser object (tests assert against it).

    `--data`/`--videos` exclusion is re-checked by hand in `main` so the error names both flags.
    `--split` defaults to None so the videos path can tell it was passed. Help strings use `%%`
    for a literal percent (argparse expands `help % params`). Deleted flags stay registered
    (`REMOVED_FLAGS`) and exit naming themselves. Advanced and diagnostic flags sit in their own
    groups; measurements behind each default live in CLAUDE.md and dev/reports.
    """
    ap = argparse.ArgumentParser(prog='tailcyclenet infer', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pose', '--run', dest='run', required=True, type=Path,
                    help='pose run folder, or ONE pose checkpoint .pth (each embeds its config '
                         'and keypoint registry). --run is the older spelling.')
    ap.add_argument('--pose-checkpoint', '--checkpoint', dest='checkpoint', default=None,
                    help='checkpoint filename when --pose is a run folder (refused beside a '
                         'file); default: the latest training checkpoint.')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--data', type=Path,
                     help='ONE session directory (a dataset root only if it holds a single '
                          'session in --split).')
    src.add_argument('--videos', type=Path, nargs='+', default=None,
                     help='raw footage instead of a session: files and/or directories (a '
                          'directory expands to its .mp4/.avi children, not recursively). The '
                          'session is built in memory; with no labels there is no scoring, and '
                          '--max-animals plus a box source (--detector/--boxes) are mandatory.')
    ap.add_argument('--out', required=True, type=Path,
                    help='the prediction SESSION directory (session.toml, calibration.toml, '
                         'groups.pq, points3d.pq, points2d.pq); no pixels.')
    ap.add_argument('--split', default=None,
                    help='default: test. Inert with --videos, and refused rather than ignored.')
    ap.add_argument('--dataset-name', default=None,
                    help="the registry entry to read this data's keypoints as, when the folder "
                         "name is not it; checked against the session's own names.")
    ap.add_argument('--groups', default=None, help='comma-separated group ids to restrict to')
    ap.add_argument('--device', default='cuda:0')

    vid = ap.add_argument_group('--videos input')
    vid.add_argument('--calibration', type=Path, default=None,
                     help='an ANIPOSELIB-layout calibration.toml (what `format.load_calibration` '
                          'reads). Another rig format needs a converter ending in '
                          '`format.dump_calibration`.')
    vid.add_argument('--cam-regex', default=None,
                     help="anipose's `[triangulation] cam_regex`, applied to each video's stem: "
                          'the camera is the CAPTURE GROUP (`cam([0-9]+)_` -> camera "0", not '
                          '"cam0") and the group id is the remainder. A pattern with no capture '
                          'group matches the whole string; a one-camera calibration needs none.')
    vid.add_argument('--session-id', default=None,
                     help="the session key every table is written under. Default: the videos' "
                          'common parent directory name.')
    vid.add_argument('--group-id', default=None,
                     help='used ONLY when the regex leaves every remainder empty; some empty and '
                          'some not is refused. Default: the session id.')
    vid.add_argument('--units', default=None,
                     help='units declaration. Default px for a 2D single-camera session; a 3D '
                          'run MUST state it (calibration.toml has no units field).')
    vid.add_argument('--fps', type=float, default=None,
                     help="override the container's fps (reaches groups.pq only).")
    vid.add_argument('--trim-to-shortest', action='store_true',
                     help="opt in to n_frames = min over a group's cameras; a length "
                          'disagreement is refused without it.')

    win = ap.add_argument_group('window loop')
    win.add_argument('--anchor', default='carry', choices=ANCHORS,
                     help="'labels' is an ORACLE, not a deployment number")
    win.add_argument('--overlap', type=int, default=4,
                     help='frames each window shares with its predecessor (sweep per root).')
    win.add_argument('--n-frames', type=int, default=None, help='default: the run\'s own')
    win.add_argument('--max-frames', type=int, default=0,
                     help='predict only the first N frames of each group; = --start-frame 0 '
                          '--end-frame N, refused together with either.')
    win.add_argument('--start-frame', type=int, default=0,
                     help='first SOURCE frame to predict, per group (half-open [start, end)). '
                          'Identity columns compare only between runs that start together.')
    win.add_argument('--end-frame', type=int, default=0,
                     help='one past the last SOURCE frame per group; 0 = to the end. Not a '
                          'resume.')
    win.add_argument('--refine', action=argparse.BooleanOptionalAction, default=None,
                     help="re-crop each window to the first pass's own prediction and predict "
                          'again. Default: on in 3D, off in 2D (on for a box model).')
    win.add_argument('--refine-all-cams', action=argparse.BooleanOptionalAction, default=False,
                     help='TR1, 3D --refine: pass 2 re-crops every calibrated camera from pass '
                          "1's pose with a per-camera overlap test (drop a camera whose refined"
                          ' box misses its pass-1 box; add one with no pass-1 box when the pose'
                          ' reprojects into its frame). Default: off (all-or-nothing over the '
                          'pass-1 cameras). measured: 3dpop Seq59 coverage/MPJPE unchanged, '
                          'idsw 13->20; byte-identical on Seq29 and a 2-bird clip (2026-10-19). '
                          'With --refine-consensus (R2): swap:0.25 MOTA P02 Seq41 0.633->0.737, '
                          'Seq59 0.852->0.849; drop:0.25 Seq59 0.870->0.890, Seq29 0.852->0.870 '
                          '(2026-10-20).')
    win.add_argument('--refine-consensus', action=argparse.BooleanOptionalAction, default=False,
                     help='R1, 3D --refine: after pass 1, a camera whose own 2D head disagrees with '
                          'the fused pose (median reprojection residual > --refine-consensus-res '
                          'box sides), or whose median vis_pred_2d logit is < '
                          '--refine-consensus-vis when that is set, is an outlier; with >= 2 cameras left, pass 1 '
                          're-runs without it and pass 2 crops every camera (the outlier too, if '
                          'in frame) from that consensus pose. Never acts with < 3 pass-1 '
                          'cameras. With --refine-all-cams (R2) a camera with no pass-1 box is '
                          'also cropped from the consensus pose. Default: off. measured (3dpop, '
                          'one lever vs base, --box-corrupt swap:0.25): MOTA Seq29 0.783->0.836, '
                          'P02 Seq41 0.633->0.704, Seq59 0.852->0.856, MPJPE -3..-16 mm; '
                          'full 3dpop test (residual-only, L4): clean counts identical to base, '
                          'swap:0.25 MOTA 0.739->0.788, drop:0.25 inert; with R4 on, R1 adds '
                          'nothing (swap 0.847 R4 vs 0.837 R1+R4); '
                          'clean boxes: metrics unchanged; allen mouse1: the vis test flags nearly '
                          'every camera at pass-1 resolution (2026-10-20).')
    win.add_argument('--refine-consensus-res', type=float, default=0.5, metavar='SIDES',
                     help='R1 residual threshold in crop-box sides (default 0.5).')
    win.add_argument('--refine-consensus-vis', type=float, default=float('-inf'),
                     metavar='LOGIT',
                     help='R1 visibility threshold, median vis_pred_2d logit (default -inf = off, '
                          'residual test only). measured: not portable -- at 0.0 pass-1 logits '
                          'are negative on ~all allen mouse1 cameras and R1 stands down in 401/434 '
                          'camera-windows; residual-only matches the 0.0 test on 3dpop and is '
                          'byte-identical to base on mouse1 (2026-10-20).')
    win.add_argument('--refine-carry-fallback', action=argparse.BooleanOptionalAction,
                     default=False,
                     help='R4, 3D: a camera with no box this window, or whose crop centre is more '
                          'than one crop side from the crop rule on the carried pose reprojected '
                          'into it, is cropped for pass 1 from that carried pose. Only with a '
                          'fresh carry and at least one camera agreeing with it; under '
                          '--refine-consensus such a camera that fails the test is dropped; '
                          'stands down on a duplicate carry (--refine-carry-dup-radius) and keeps '
                          'the box where the carried crop is off in scale (--refine-carry-scale-max). '
                          'Default: off. measured with both gates (3dpop test, 58 sessions / 145 '
                          'units, L4): clean boxes MOTA 0.9058->0.9058 (miss/fp/idsw identical, '
                          'MPJPE -0.004 mm); drop:0.25 MOTA 0.847->0.902 (+0.065 [+0.049, '
                          '+0.089]), MPJPE -6.8 mm; swap:0.25 0.739->0.847 (+0.170 [+0.137, '
                          '+0.207]), MPJPE -9.9 mm (2026-10-21).')
    win.add_argument('--refine-carry-dup-radius', type=float, default=0.75, metavar='SIDES',
                     help='R4 gate: stand down for an animal whose carried pose is within this many '
                          'crop sides of another slot\'s carry in >= 2 cameras (the tracker\'s '
                          'duplicate rule; a duplicate carry is a drifted or ghost row). 0 = the '
                          'ungated R4. Default 0.75.')
    win.add_argument('--refine-carry-scale-max', type=float, default=2.0, metavar='FACTOR',
                     help='R4 gate: a camera whose carried crop is more than this factor larger or '
                          'smaller than the median of the animal\'s own boxes this window keeps its '
                          'box (an exploded or collapsed carry). 0 = off. Default 2.0.')
    win.add_argument('--refine-px', type=int, default=None,
                     help="run --refine's FIRST pass at this resolution (96-192 plateau, 64 "
                          'cliff). No default.')
    win.add_argument('--box-prompt', default='firstonly',
                     choices=('auto', 'firstonly', 'none', 'labels', 'detector'),
                     help='box prompt for a box model: `auto` uses the detector/boxes file and '
                          "refuses without one; `firstonly` (default) gives it only at each "
                          "animal's first window; `labels` is an ORACLE; `none` forces it off.")
    win.add_argument('--crop-inflate', type=float, default=None,
                     help='inflate every crop about its centre. Default 1.5 with the box on, '
                          'else 1.0.')
    win.add_argument('--crop-inflate-adaptive', action=argparse.BooleanOptionalAction,
                     default=False,
                     help='R3: per (animal, window, camera), inflate the pass-1 crop by the largest '
                          'factor in [1, --crop-inflate-max] that touches no OTHER animal\'s box '
                          'in that camera this window; --crop-inflate where none has one. Needs '
                          '--detector boxes to see neighbours. Default: off. measured (3dpop, '
                          'clean boxes): Seq29 MOTA 0.920->0.930, fp_dup 44->20; Seq59 MOTA '
                          '0.9226->0.9222, MPJPE -0.37 mm; P02 Seq41 MPJPE +0.36 mm; no help '
                          'under swap (2026-10-20).')
    win.add_argument('--crop-inflate-max', type=float, default=2.0,
                     help='R3 ceiling on the adaptive inflation (default 2.0).')
    win.add_argument('--vis-thresh', type=float, default=None,
                     help='PER-CAMERA gate: withhold an (animal, frame, camera) 2D row whose median '
                          'per-camera visibility logit (`vis_pred_2d`) is below this; the 3D row is '
                          'withheld only when every camera is gated (reported rows only, never the '
                          'carried prompt).')
    win.add_argument('--min-box-frames', type=int, default=1,
                     help='finite (frame, camera) boxes a row needs before it gets a window '
                          'crop; raising it lowers reported coverage on purpose.')

    det = ap.add_argument_group('boxes and detection')
    det.add_argument('--boxes', type=Path, default=None,
                     help='npz of crop points per group; default is to crop from the labels')
    det.add_argument('--detector', type=Path, default=None,
                     help='a detector run folder or packaged .pth: THE deployment path (boxes '
                          'from pixels).')
    det.add_argument('--detector-checkpoint', default='latest',
                     help="selector when --detector is a run folder: 'latest' (default), 'last', "
                          "'best' (historical detector.pth), or a filename.")
    det.add_argument('--det-score', type=float, default=0.01,
                     help='objectness floor for a detection; favours recall (the 3D '
                          'reprojection gate and NMS filter downstream). Sweep per checkpoint.')
    det.add_argument('--det-nms-iou', type=float, default=0.5,
                     help="decode's box-NMS IoU threshold.")
    det.add_argument('--det-nms-center-dist', type=float, default=NMS_CENTER_DIST,
                     help='second NMS test in box sides: suppress a candidate whose centre is '
                          'this close to a kept box; negative disables.')
    det.add_argument('--max-animals', type=int, default=0)
    det.add_argument('--det-top-k', type=int, default=0,
                     help='detections kept per frame-camera; 0 follows --max-animals')
    det.add_argument('--allow-detector-transfer', action='store_true', default=False,
                     help="permit a detector trained on another dataset (a labelled transfer "
                          'arm only).')

    ident = ap.add_argument_group('identity and association')
    ident.add_argument('--link-boxes', action=argparse.BooleanOptionalAction, default=True,
                       help='2D: follow one animal per row across frames (per-frame Hungarian on '
                            'centre distance, gated at --max-move).')
    ident.add_argument('--track', action=argparse.BooleanOptionalAction, default=True,
                       help='3D multiview: one cross-view target set across frames; --no-track '
                            'restores per-frame association.')
    ident.add_argument('--max-move', type=float, default=1.25,
                       help='the identity gate in box sides, shared by --track and --link-boxes.')
    ident.add_argument('--min-views', type=int, default=2, choices=(1, 2),
                       help='3D: 2 builds every instance from a camera pair; 1 also emits '
                            'single-view instances.')
    ident.add_argument('--max-age', type=int, default=8,
                       help='frames without evidence before a lost identity is retired (--track '
                            'and --link-boxes).')
    ident.add_argument('--duplicate-radius', type=float, default=0.75,
                       help='duplicate proximity in box sides for the 3D --track backstop.')
    ident.add_argument('--duplicate-persist', type=int, default=5,
                       help='in-band frames before the 3D --track backstop resolves a duplicate.')
    ident.add_argument('--assoc-res-max-px', type=float, default=None,
                       help="override the session's reprojection gate (a PIXEL gate: sweep per "
                            "rig). Default: the session's own value.")
    ident.add_argument('--assoc-res-max-sides', type=float, default=None,
                       help='TR5: gate cross-view grouping on the reprojection residual in BOX '
                            "SIDES (median over views of residual / that view's box side) "
                            'instead of pixels; when set it REPLACES the pixel gate for '
                            'association only (the GT triangulation keeps assoc_res_max_px). '
                            'Default: off (pixel gate). measured: 3dpop Seq59/Seq29 at 0.19 (= '
                            '30 px at the 157 px median side): MOTA -0.0003, idsw 13->10 / '
                            '0->3, IDF1 -0.012 [-0.045, +0.009] (2026-10-19).')
    ident.add_argument('--assoc-score-order', action=argparse.BooleanOptionalAction,
                       default=False,
                       help='TR6a: rank cross-view candidate pairs by (support, lower box '
                            'score, residual) instead of (support, residual). Default: off. '
                            'measured: 3dpop Seq59/Seq29 MOTA +0.0000, fp_dup 97->84, idsw '
                            '13->17 / 0->2, IDF1 -0.007 (2026-10-19).')
    ident.add_argument('--track-camera-claims', action=argparse.BooleanOptionalAction,
                       default=False,
                       help='TR2, 3D --track: after the targets x groups match, let each live '
                            'target claim one detection per camera that no seated group used '
                            '(per-camera Hungarian, same box-side gate). Births still need 2 '
                            'views. Default: off. measured: 3dpop Seq59 on TR1: 253 claims, '
                            'coverage unchanged, fp_dup 97->80, idsw 13->18, IDF1 0.848->0.779 '
                            '(2026-10-19).')
    ident.add_argument('--track-age-decay', type=float, default=None, metavar='LAMBDA',
                       help='T2, 3D --track: replace the additive sticky-age term with the '
                            "paper's multiplicative decay aff * exp(-LAMBDA * age). Default: "
                            'off (additive). measured: 3dpop test at 0.05, 57 of 58 sessions '
                            'byte-identical, every metric delta 0.0000; train Seq30 idsw '
                            '27->25, IDF1 0.844->0.897 (2026-10-19).')
    ident.add_argument('--track-affinity-2d', type=float, default=0.0, metavar='W2',
                       help='TR3, 3D --track: blend a per-camera 2D continuity term into the '
                            'target x group affinity, (1 - W2) * A_3D + W2 * A_2D. Default: 0 '
                            '(A_3D only). measured: 3dpop test at 0.6 on T2 (0.05): MOTA '
                            '+0.0001 [-0.0001, +0.0004], idsw 22->23, IDF1 -0.0001; train Seq30'
                            ' idsw 25->20 (2026-10-19).')
    ident.add_argument('--identity-bridge', action='store_true',
                       help='OPT-IN post-pass repairing the slot permutation a hard interaction '
                            'leaves; TRADES MISSES FOR IDENTITY (quote miss and coverage with '
                            'its idsw). Edits the written session in place.')

    sco = ap.add_argument_group('scorer')
    sco.add_argument('--scorer', type=Path, default=None,
                     help='optional framewise scorer run folder or .pth; its scores are written '
                          'as metadata, predictions are unchanged.')
    sco.add_argument('--scorer-checkpoint', default=None,
                     help='checkpoint filename when --scorer is a run folder; default: latest.')
    sco.add_argument('--allow-pose-scorer-mismatch', action='store_true', default=False,
                     help='allow different pose/scorer video-encoder weights (two encodes).')

    spd = ap.add_argument_group('speed and memory (output-neutral)')
    spd.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16',
                     help='autocast dtype for the scene encode only.')
    spd.add_argument('--camera-batch', action=argparse.BooleanOptionalAction, default=True,
                     help='encode all cameras in one scene-encoder call per window.')
    spd.add_argument('--assoc-fast', action=argparse.BooleanOptionalAction, default=True,
                     help='TR7: cross-view association with cached rays/projections and pairs '
                          'that fail their own residual gate dropped before support counting; '
                          'identical groups. Default: on (owner, 2026-10-19); --no-assoc-fast is the reference loop. '
                          'measured: byte-identical on 3dpop Seq59 and Seq29 (all six tables), '
                          'association 1.9x, wall -36%% on Seq29; byte-identical on allen-mouse-tracked (golden) (2026-10-19).')
    spd.add_argument('--kpt-chunk', type=int, default=0,
                     help='decode keypoints in slices of this size (lower peak memory); 0 = one '
                          'pass.')
    spd.add_argument('--prefetch-windows', type=int, default=1,
                     help='decode this many windows ahead (bit-exact at any value).')
    spd.add_argument('--max-ram', type=float, default=None, metavar='GB',
                     help='host RAM this run may spend on pixel buffers. Default: the smallest '
                          'of the cgroup limit, LSF limit, MemAvailable and MemTotal. Every '
                          'knob it sizes is output-neutral. TAILCYCLENET_MAX_RAM_GB does the same '
                          'without a CLI.')

    adv = ap.add_argument_group('advanced and diagnostic')
    adv.add_argument('--carry-source', default='triangulate', choices=CARRY_SOURCES,
                     help='3D --anchor carry: seed the next window with the anchor-free '
                          'estimate (default) or the reported prediction (`pred`, the old arm).')
    adv.add_argument('--gridresid-offset', default=None, choices=('query', 'triangulated'),
                     help='state the 3D residual anchor of a run folder written before '
                          '`[model].gridresid_offset` existed. Not a sweep knob.')
    adv.add_argument('--assoc-random-drop', type=float, default=0.0, metavar='P',
                     help='DIAGNOSTIC: reject each cross-view group that passes the gates with '
                          'probability P (seed 0): the rate-matched random control for a gating '
                          'lever (eval rule 14). Never a deployment arm.')
    adv.add_argument('--window-predictions', action='store_true',
                     help='write every window\'s pre-gate predictions to '
                          '<out>/window_predictions.pq')
    adv.add_argument('--det-trace', type=Path, default=None,
                     help='write per-frame/camera detector decode-stage counts as JSON.')
    adv.add_argument('--overlap-trace', type=Path, default=None,
                     help='write the output-neutral overlap test-retest capture as an npz.')
    adv.add_argument('--oracle-corrupt', default=None,
                     help='ONLY with --anchor labels: break the oracle prior on purpose to '
                          'measure the echo coefficient: off:<x> (whole-body offset, crop '
                          'widths), stale:<n>, other, near, swap:<n>.')
    adv.add_argument('--box-corrupt', default=None, metavar='SPEC',
                     help='DIAGNOSTIC, needs --detector: break the detector boxes before cropping, '
                          'per (animal, window, camera), seeded (deterministic given SPEC): '
                          'drop:<p> | shift:<p>:<sides> | shrink:<p>:<f> | swap:<p> (the nearest '
                          "other row's box), comma-joined. Never a deployment arm. measured: "
                          '3dpop Seq59 MOTA 0.923 clean, 0.870 drop:0.25, 0.922 shift:0.25:0.5, '
                          '0.923 shrink:0.25:0.6, 0.852 swap:0.25 (2026-10-20).')
    adv.add_argument('--dump-session', type=Path, default=None,
                     help='DEBUG: write the constructed --videos session (pixels symlinked) and '
                          'carry on.')
    _add_removed_flags(ap)
    return ap


def main(argv=None):
    """Parse and run; `scripts/infer.py` is a shim onto this.

    The `--data`/`--videos` mutual exclusion is checked by hand (not by argparse) so the error
    names both flags.
    """
    from .driver import run_dataset

    ap = build_parser()
    args = ap.parse_args(argv)
    if bool(args.data) == bool(args.videos):
        ap.error('exactly one of --data (a session directory in docs/annotation_format.md) or '
                 '--videos (raw footage) is required.')
    if args.calibration and not args.videos:
        ap.error('--calibration only means anything with --videos; a session directory carries '
                 'its own calibration.toml.')
    return run_dataset(args)
