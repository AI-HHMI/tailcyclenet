# tailcyclenet

Finetune a [posetail](https://pypi.org/project/posetail/) point tracker into an animal pose
estimator. Three settings, one model: **3D multiview**, **3D single-view**, **2D single-view**.

This is a clean rebuild of `../posetail-pose` (~30,000 lines, 86 generated arm configs, ~20
architecture switches, ~20 partly-redundant eval scripts). Everything that was an *experiment*
there is deleted here. What survives is the one architecture that won, the one crop rule, the one
window loop, and a data format that serves both hand annotation and bulk training.

---

## Layout

```
pyproject.toml        package + [tool.pixi.*] env. There is no separate pixi.toml.
CLAUDE.md             this file
docs/                 HUMAN-OWNED. Do not edit without explicit instruction.
  annotation_format.md  THE data format spec. Code is written against it, not the reverse.
dev/
  plans/              design docs, one per non-obvious decision
  reports/            progress + measurement reports as things land
scratch/              LLM scratch space. May be deleted between sessions. Artifacts untracked.
tailcyclenet/
  format.py           read/validate a session -> dense arrays; the keypoint registry
  dataset.py          torch Dataset over the format; 2D/3D/fake-2D; collate
  crop.py             crop_box_for_points -- THE crop rule, int32-exact
  model.py            PoseTrackerEncoder + build_model + the query-free scene centre
  query_encoder.py    PoseQueryEncoder with missing-query tokens
  checkpoints.py      run folders, save/load, warm start
  metrics.py          MPJPE / PCK / multi-instance matching / MOTA
  infer.py            THE inference path (windowed, crops, tracks, fusion)
  detector/           YOLOX-Nano box predictor + cross-view association
scripts/              convert_v4.py  train.py  train_detector.py  infer.py  eval.py
configs/              example configs, hand-written. NOT generated, NOT one per experiment.
tests/
```

**Who may write where.** `dev/` and `scratch/` are yours. `docs/` is the human's — ask first.
`../posetail-pose` and `../../posetail/posetail-next` are read-only reference.

---

## Environment

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"
pixi run test
```

`posetail==0.3.0` comes from **PyPI**, pinned. A checkout of the same version lives at
`../../posetail/posetail-next` for reading source; do not depend on it by path and do not modify
it. (posetail-pose's `path = "../posetail-next"` dep is how it ended up importing nothing after a
directory move.)

The `LD_LIBRARY_PATH` prepend in `[tool.pixi.activation.env]` is load-bearing: the env ships
`libstdc++.so.6.0.35`, the host may ship 6.0.29 with no `CXXABI_1.3.15`, and without it `import
scipy.optimize` dies inside `_highspy` with an error that names only CXXABI.

---

## The data format

Spec: `docs/annotation_format.md`. Summary of the shape:

```
<dataset_root>/<split>/<session_id>/
    session.toml       keypoint identity (names, mode, units) + provenance
    calibration.toml   ALL camera geometry: aniposelib + type + crop offset + image_size + moving
    groups.pq          one row per group (a group is a CONTIGUOUS CLIP)
    keypoints.pq       per-camera 2D + per-camera visibility
    points3d.pq        3D -- first-class, not derived from 2D
    instances.pq       boxes / present / absent          (optional)
    extrinsics.pq      per-frame extrinsics, moving cams (optional)
    groups/<gid>/<cam>/000000.jpg …   or   groups/<gid>/<cam>.mp4   (symlinks fine)
```

Five things that are easy to get wrong:

- **Split is a directory, not a column.** None of the four shipped datasets split cleanly by
  session — rat-city's train/val/test are one recording — so a session-level field could not
  express them.
- **`mode` is per-session.** One `train/` may hold 2D and 3D sessions; the model trains on both
  and both head-bank slots (`mode_idx` 0 and 1) get gradient. Cross-session agreement covers
  `names` only.
- **A group is a contiguous clip**, `n_frames` unbounded. rat-city's 57,594 frames are ONE group
  whose `cam0` is a single symlink. ~900 symlinks across all four datasets, not 550k.
- **`status` is the visibility channel**, in both label tables. Dictionary-encoded in parquet, so
  `vis = codes == VISIBLE` is a vectorized int8 compare.
- **`keypoints.pq` `x,y` may be null on a `visible` row** iff a `points3d` row exists for the same
  key. That row is a *visibility observation* — allen-mouse ships a real per-camera
  `vis (S,T,K,7)` with no per-camera 2D, and this is how it is kept without inventing positions.

---

## Training

```bash
pixi run python scripts/train.py --config configs/w9_allen.toml
```

`[data].path` is either a dataset root (has `train/`, optionally `val/` and `test/`) or a folder
whose children are dataset roots. In the second case keypoint names are prefixed with the dataset
folder name (`rat-city-nose`) and one estimator trains across all of them; the keypoint embedding
table is what makes this work.

`n_keypoints` is **derived** from the registry, never configured. The registry is written to the
run folder as `keypoint_registry.toml` and read back at inference. Given an existing registry, a
later run **appends** new names so old ids — and the embedding rows behind them — survive warm
start.

### The architecture: one switch

```toml
[model]
query = "prior"   # per-keypoint prior + missing-query tokens  (was posetail-pose's w9_honest)
# query = "none"  # query-free: no prior at all
```

`query = "prior"` carries a per-keypoint prior `kpt_prior (B,K,R)` plus `prompt_time (B,K)`.
Every position-derived fusion term — `pos`, `patch`, `vis`, and `depth` in 3D — carries a
**learned no-query token** where a keypoint has no prior, instead of a value computed from the
crop centre and presented as real. The instance-anchor machinery from posetail-pose
(`instance_anchor`, `anchor_mode`, `anchor_fallback`, `anchor_dropout`, `anchor_noise`,
`anchor_attn_bias`) is **deleted**, not defaulted off.

Also deleted, deliberately: `WideQueryEncoder` / `query_encoder='wide'`, `query_pos_embedding`,
`kpt_table_mlp`, the crowd head, distractor crops, prompt corruption, `crop_side_mode`,
`curriculum`. If 3D accuracy disappoints, the first thing to try is re-adding `wide` — it beat
`pose` on allen-mouse 3D (3.395 vs 4.021 mm) in posetail-pose.

---

## Inference: one entry point

```bash
pixi run python scripts/infer.py --data <dataset|session|video> --run runs/<x>/ --out pred.npz
```

There is **one** window loop. posetail-pose had three, and all three got it wrong differently.
Box sources are the annotation set, a detections npz, or a per-dataset detector. Prompt regimes:
`none` (query-free), `carry` (previous window's own prediction — what deployment does, requires
`overlap >= 1`), `self` (two passes per window). Rendering is a flag, not a separate script.

`scripts/eval.py` is offline and model-free: prediction npz + annotation set → MPJPE (paired
bootstrap), PCK, coverage, MOTA/miss/FP/idsw. Multi-animal rows report **matched** MPJPE: row
index is not identity once boxes come from a detector, and scoring row-to-row measured 385 px on
flies that are 30 px across.

## The detector

```bash
pixi run python scripts/train_detector.py --data <ONE dataset root> --out runs/det-<name>
pixi run python scripts/infer.py --run runs/w9 --data <dataset> --detector runs/det-<name> ...
```

**One detector per dataset**, and `--input-wh` defaults to an aspect-matched size rather than a
square. This is not fussiness: rat-city's frames are 2.29:1, so a square 416 letterbox wastes 56%
of the canvas and delivers the median rat at 15.8 x 12.5 px — about 2 x 1.6 cells at stride 8 and
absent from strides 16 and 32, so two thirds of the FPN cannot represent it. Same detector on
square 1024x1024 fly frames reaches AP50 0.985 where rat-city sits near 0.50.

The regression target is `crop.crop_box_for_points`, i.e. the detector reproduces *the crop the
pose model was trained on*, not "a box around the animal". `tests/test_detector.py` asserts that
against the crop rule directly.

---

## Gotchas — every one of these has already cost someone a day

1. **T = 1 is not usable.** `posetail/posetail/encoder_decoder.py:748` computes
   `gT = T // tubelet_size` → 0, so the pos_embed is zero-length; `tracker_encoder.py:518` has the
   same shape. The fix existed on the abandoned `memory` branch
   (`gT = feat.shape[1] // (gH*gW)`) and was **lost in the moving-cams merge**. Never sample
   fewer than 2 frames; single-frame groups are padded at ingest.
2. **`scene_features=` and `cube_scale=` were dropped from `TrackerEncoder.forward` in 0.3.0.**
   Encoder sharing for inference goes through `SceneRepresentation` directly, or the private
   `_forward_window` / `_decode_from_scene`.
3. **`batch_size` is structurally 1.** `custom_collate` keeps only item 0's `cgroup`
   (`posetail_dataset.py:398`) and the model takes one camera group per batch. This is why there
   is no DDP. Known ceiling, not a bug to fix casually.
4. **Keypoint identity ≠ array position.** The library drops keypoints with <2 valid frames, so
   `N` shrinks and positions stop matching ids. The loader must never filter. This failure is
   invisible in the loss curve.
5. **Keypoint ids ride in the occlusion channel**, and the stock `QueryEncoder` clamps
   `occlusion+1` into `[0,2]`. Never share that tensor between the two consumers.
6. **`vis` and `vis_2d` are both-or-neither** — supplying one dies inside einops. And
   `get_eval_metrics` wants the trailing dim `(B,T,N,1)`.
7. **The crop rule is exact, not approximate.** `crop.py` is lifted verbatim from posetail-pose's
   verified copy (`crop_box_for_points` does not exist in 0.3.0 — it was a `memory`-branch
   method). A test asserts it is int32-exact against `crop_cgroup_to_points`. If that fails,
   every detector number is invalid.
8. **Moving-camera inference is not supported upstream.**
   `inference_utils.load_camera_group_from_metadata` ignores `moving_cams` entirely; we build the
   camera group ourselves via `format_camera_group(..., moving_ext={cam: (T,4,4)})`. Only
   `TrackerEncoder` is moving-cam-safe — `ScorerEncoder` and `TrackerTapNext` shape-error on
   `(T,3)` centres.
9. **allen-mouse's npz is column-sorted.** `pose3d.npz['pose']` is ordered by
   `sorted(f'{name}_{axis}')` while `keypoints` is name-sorted, which transposes all 8
   `X` / `X-base` pairs — 16 of 47 keypoints. The converter applies the permutation once. Zipping
   `pose` against `keypoints` silently mislabels them and nothing downstream notices.

---

## Evaluation rules carried over from posetail-pose

These are not style preferences; each one was learned by publishing a wrong number.

1. **Label the axis.** Within-session and cross-animal results invert. Say which one a number is.
2. **Carry the reference through the identical pipeline.** A baseline computed a different way is
   not a baseline.
3. **Pair the bootstrap.** Unpaired intervals on the same windows overstate uncertainty enough to
   hide real effects.
4. **Match the controls.** An arm that differs in two keys measures neither.
5. **A log statistic cannot tell you a run converged.** Use two pinned checkpoints on identical
   windows.
6. **`err` is a mean over matched frames.** Decompose coverage before quoting any delta — a
   method that predicts fewer, easier frames looks better and is not.
7. **Anchor and prior inputs are GT-derived.** They must be gated off at eval by default. In
   posetail-pose their absence inflated *every* anchored number ever published there.
8. Only MOTA replicates across seeds, and only above a ±0.023 seed floor.

Reproduction note: posetail-pose's `reports/golden_allen_j3.json` is an exact-reproduction
contract for *that* pipeline. This repo will not match it bit-for-bit and should not claim to.
The check here is a band: allen-mouse cross-animal MPJPE near 3.394 mm, human-vs-human baseline
2.208 mm. A large gap is a port bug, and the first suspects are the allen column-sort permutation
and the crop rule.
