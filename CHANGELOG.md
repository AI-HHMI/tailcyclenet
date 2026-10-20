# Changelog

## 0.4.1 (unreleased)

### Changes that move numbers
- `--assoc-fast` is on by default: byte-identical association, about 1.9x faster (`--no-assoc-fast` is the reference loop).
- `--refine-carry-fallback` (R4) is on by default in 3D: a camera with no box, or a box that disagrees with the fresh carried
  pose, is cropped from that pose; it stands down on a duplicate carry and when the carried crop is off in scale. 3dpop test:
  clean boxes MOTA unchanged, 25% dropped boxes +0.065, 25% swapped boxes +0.170. `--no-refine-carry-fallback` restores the
  old crops.
- `eval --min-match-kpts` defaults to 0.5; pass 0 to reproduce numbers published before 0.4.1.

### New opt-in switches
- `--box-corrupt` (diagnostic), `--refine-consensus` (residual-only camera consensus), `--crop-inflate-adaptive`,
  `--refine-all-cams`, `--track-camera-claims`, `--track-age-decay`, `--track-affinity-2d`, `--assoc-res-max-sides`,
  `--assoc-score-order`, `--assoc-random-drop` (diagnostic). All measured null or mixed on 3dpop; see each `--help`.

## 0.4.0 (2026-10-19)

Architecture cleanup (`dev/plans/architecture_cleanup.md`, untracked). **Import paths break; the CLI does not.**
`tailcyclenet <command>` and `scripts/*.py` keep their names and flags, except the deleted ones below.

### Layout
- One level of packages: `format/ data/ loader/ pose/ detector/ scorer/ infer/ evaluation/ render/`, plus
  `config.py runtime.py checkpoints.py distributed.py memory.py packaging.py tracktail_compat.py`.
- `infer/` is the whole deployment program (detection, association, tracking, bridge, scorer QC, `--videos`).
- `scripts/` is `convert/ export/ dataset_tools/ diagnostics/ tools/ cluster/` plus the command shims; the converter
  rename table is `scripts/convert/README.md`. New `tailcyclenet package` command.
- The wheel ships only `base.toml`, `detector.toml`, `scorer.toml` from `configs/`.

### Changes that move numbers
- Detector recipe: one NMS centre distance (0.3) for selection and deployment; FPs on GT-less groups are counted;
  mosaic paste avoids the view's own boxes; `[training].seed` seeds augmentation.
- Muon: `kpt_embed` and the missing-query tokens train at `kpt_lr` (older Muon checkpoints warm-start, not resume).
- 3D rotation augmentation no longer writes visibility 0 on unassessed camera slots.
- Val/test windows that fail to build are skipped, not swapped; scorer val pools correctly across ranks.
- `--vis-thresh` is a per-camera gate on `vis_pred_2d`; a 3D row is withheld only when every camera is gated.
- `eval` scores a ranged run only inside its `[provenance]` frame range.
- A resumed run continues its sampler stream instead of replaying it; a resume on a different `--devices` count is
  refused.

### Deleted (raise by name where they were flags/keys)
- Detector keypoint branch (`--pose-nms`, `--crop-source keypoints`, `[data].keypoints`), legacy detector
  architectures (`trimmed`, `nano`..`x`, COCO pretraining), `--det-input-wh`, the report-73 transfer flags.
- `--independent-windows`, `--fill-from`, pre-marker prediction-session readers.
- Scorer `--window-offset`, `[data].val_offset`, unreachable corruption gates and aliases (accepted at inert values),
  the per-root scorer configs.
- `TAILCYCLENET_READER_CACHE`, deepfly3d v1/v2 converters, pre-schema resume branches.
- v0.3.4 is the last release that loads the deleted detector architectures and prediction formats.

### Fixes
About 40 crash and silent-misbehaviour fixes across inference, detector, scorer, data and training resume
(`[training].checkpoint_path` accepts a run folder or packaged `.pth`; warm start uses the checkpoint's registry;
resume no longer re-downloads the base; `validate_session` checks spec rules 4/7/8 and refuses `projected` in
`points3d.pq`).
