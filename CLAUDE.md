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
dev/                  UNTRACKED. working notes, not the repo's product
  plans/              design docs, one per non-obvious decision
  reports/            progress + measurement reports as things land
scratch/              UNTRACKED. LLM scratch space. May be deleted between sessions.
tailcyclenet/
  format.py           read/validate a session -> dense arrays; the keypoint registry
  dataset.py          torch Dataset over the format; 3D multiview / 3D single-view / 2D
  crop.py             crop_box_for_points -- THE crop rule, int32-exact
  model.py            PoseTrackerEncoder + build_model + the query-free scene centre
  query_encoder.py    PoseQueryEncoder with missing-query tokens
  checkpoints.py      run folders, save/load, warm start
  metrics.py          MPJPE / PCK / multi-instance matching / MOTA
  infer.py            THE inference path: one window loop
  detector/           YOLOX-Nano box predictor + cross-view association
scripts/              convert_v4.py  train.py  train_detector.py  infer.py  eval.py
configs/              example configs, hand-written. NOT generated, NOT one per experiment.
tests/
```

**Who may write where.** `dev/` and `scratch/` are yours, and both are gitignored — a conclusion
that should outlive the session belongs in this file or `docs/`. `docs/` is the human's — ask first.
`../posetail-pose` and `../../posetail/posetail-next` are read-only reference.

---

## Environment

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"
pixi run test
```

`posetail==0.3.2` comes from **PyPI**, pinned. A checkout of the same version lives at
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
    calibration.toml   ALL camera geometry: aniposelib CameraGroup + crop offset + moving
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
  and both head-bank slots (`mode_idx` 0 and 1) get gradient. Sessions need not agree on `names`
  either: `Dataset.names` is the **union** over the root's sessions and `Registry.ids_for` remaps
  each session's own axis onto it **by name**, so a session may reorder the keypoints or carry a
  subset of them. `allen-mouse-combined` is exactly this — 80 hand-annotated sessions in
  anatomical order beside a tracked one in name-sorted order.
- **A group is a contiguous clip**, `n_frames` unbounded. rat-city's 57,594 frames are ONE group
  whose `cam0` is a single symlink. ~900 symlinks across all four datasets, not 550k.
  **A group's length is not its label count.** allen's annotated groups are 65 frames carrying
  exactly ONE labelled frame — context around a single hand-annotated still — while a tracked
  group is labelled throughout. This is why a train window's T is derived from the labels rather
  than configured (gotcha 1), and why an index entry is a poor sampling weight (`_pool_weights`).
- **`status` is the visibility channel**, in both label tables. Dictionary-encoded in parquet, so
  `vis = codes == VISIBLE` is a vectorized int8 compare. But **coordinates live on `VISIBLE` *or*
  `PROJECTED`** (`fmt.POSITIONED`): `projected` is a position with no visibility claim, for a
  source that had its annotators place every keypoint in every view and never recorded occlusion
  (johnson-mouse: 1,235,334 "visible" vs 18 "not"). It must never reach a visibility target —
  `dataset.py` NaNs it out, and withholds `vis`/`vis_2d` entirely when a window has no assessment
  at all, because the noisy-OR would otherwise assert that nothing is reconstructible in 3D.
- **`keypoints.pq` `x,y` may be null on a `visible` row** iff a `points3d` row exists for the same
  key. That row is a *visibility observation* — allen-mouse ships a real per-camera
  `vis (S,T,K,7)` with no per-camera 2D, and this is how it is kept without inventing positions.

---

## Training

```bash
pixi run python scripts/train.py --config configs/w9.toml
```

`[data].path` is either a dataset root (has `train/`, optionally `val/` and `test/`) or a folder
whose children are dataset roots. In the second case keypoint names are prefixed with the dataset
folder name (`rat-city-nose`) and one estimator trains across all of them; the keypoint embedding
table is what makes this work.

`n_keypoints` is **derived** from the registry, never configured. The registry is written to the
run folder as `keypoint_registry.toml` and read back at inference. Given an existing registry, a
later run **appends** new names so old ids — and the embedding rows behind them — survive warm
start.

### The architecture: two switches

```toml
[model]
query = "prior"           # per-keypoint prior + missing-query tokens (posetail-pose w9_honest)
# query = "none"          # query-free: no prior at all
query_encoder = "wide"    # 512-dim, identity + time, + two query terms iff query = "prior"
# query_encoder = "pose"  # 256-dim, ten terms, 27 of 30 tensors inherited
```

They are **orthogonal**: `query` decides whether a prior is supplied, `query_encoder` decides
which module consumes it. There is no third key — `wide`'s two query terms (`qpos`, `patch`) are
**derived** from `query`, because under `query = "none"` the prior is never read, so `_query_ok`
is all-False for the whole run and both terms collapse to constant no-query tokens feeding dead
gate inputs. So `wide` + `none` is a 6-term encoder and `wide` + `prior` an 8-term one, and
`wide` + `none` **is** golden's `j3`.

`query = "prior"` carries a per-keypoint prior `kpt_prior (B,K,R)` plus `prompt_time (B,K)`.
Every position-derived fusion term — `pos`, `patch`, `vis`, and `depth` in 3D on `pose`; `qpos`
and `patch` on `wide` — carries a **learned no-query token** where a keypoint has no prior,
instead of a value computed from the crop centre and presented as real. `prompt_dropout` is the
fraction of **steps** that run fully query-free, drawn per item as the reference draws it, and
`prompt_noise_px` is the jitter on the priors it keeps — **in pixels**, converted for 3D by
`cube_scale` (world units per pixel), the same normalisation the metric losses use to enter the
Huber in pixels. One scalar in each session's own units could not work: `allen-mouse-combined`
alone holds 63 px sessions beside 14 mm ones. Ship it WITH the dropout — dropout alone is retired
at +0.146 SIG worse on fly. The
decode itself is per-keypoint independent — there is no attention across the query axis — but
`scene_center`, `scene_radius` and `cube_scale` are derived from the WHOLE `coords_q` set
(`tracker_encoder.py:318,356`) and scale the depth and 3D outputs. Measured on allen: a
per-keypoint draw at p = 0.5 puts `scene_center` 13.6 mm from its deployment value (0 of 200 draws
within 1 mm), where a per-item drop puts it at exactly 0. Per keypoint, the geometry the model
meets at deployment is never trained. The instance-anchor
machinery (`instance_anchor`, `anchor_mode`, `anchor_fallback`, `anchor_attn_bias`) is **deleted**,
not defaulted off — including `j7`, the best row on record (3.140 allen cross-animal).

`wide` is the pre-port encoder, and the record says it wins **whenever no prior is supplied**:
query-free, `pose`'s four distinguishing terms are computed from the same derived scene point for
every keypoint, so they are constants and `pose` is a lower-capacity `wide`. Tying its two query
terms to `query` also makes the old wiring trap unrepresentable — `wide` with both terms off
ignores `query_coords` entirely, which is how six posetail-pose configs declared an anchor,
trained, and were reported as anchored arms whose anchor was a literal no-op.

`query_patch_embedding` builds its `PatchProcessor` at the base checkpoint's `embed_dim` and
projects to the fusion width, so all ~5.4M of the pretrained patch CNN load by name. Building it
at 512 instead would inherit 93k of 5.5M and silently retrain the rest from noise.

### The 3D output: `gridresid_offset`

`output_mode = "gridresid"` reconstructs `world = query + R @ residual` with ONE anchor for the
whole window (`tracker_encoder.py:736`). What that anchor is is a switch:

- **`gridresid_offset = "query"`** — the native structure, kept **per keypoint** only where a real
  prior anchored it; every other keypoint falls back to that frame's own **triangulation**. Under
  `query = "none"` that is every point, so the prediction is the triangulation outright and the
  `grid` CE target is dropped (`losses.py:680` gates on `'grid' in outputs`).
- **`gridresid_offset = "triangulated"`** — recover the residual and re-add it to **each frame's**
  own triangulation, for every keypoint. This is posetail-pose's `_reanchor_per_frame`, which it
  applies unconditionally (`model.py:756`), measured 2.07 → 1.37 mm within-session.

Pair them with `query`: a residual measured from the query point only means something when the
query is a real prior, and per-frame re-anchoring exists specifically to rescue a scene-centre
anchor. So `prior` → `"query"` and `none` → `"triangulated"`, which also makes `wide` + `none` +
`"triangulated"` a faithful **golden j3** architecture.

The substitution uses the **detached** triangulation, and that is also the loss gate: the direct
term is then constant at unprompted points, contributing exactly zero gradient, so
`coords_loss_direct*` and `coords_softmax_3d` supervise query points only — without forking
`TotalLoss`. `prob_2d_only = 0`, because 3D single-view has no triangulation to fall back to; if
it is ever turned back on, non-query points leave the 3D target via `out['loss_kpt_mask']`.

**The shipped sweep is therefore a TWO-LEVER comparison** — `prior` and `none` differ in `query`
*and* `gridresid_offset`. That is deliberate (each arm gets the offset its query implies) but it
means a `prior` − `none` delta cannot be attributed to the prior alone. Say so when reporting it;
eval rule 4, and `improvement_leads.md` opens by retracting a claim of exactly this shape.
Isolating either lever needs a third arm.

Still deleted, deliberately: `kpt_table_mlp`, the crowd head, distractor crops, prompt corruption,
`crop_side_mode`, `curriculum`. CLAUDE.md used to claim wide beat pose "3.395 vs 4.021 mm" — that
is a **two-lever** comparison (`j2_jitter`: wide *and* crop jitter, 60k iters, vs `p3_package`:
pose, no jitter, **12k**). The one-key figure from that ledger is **3.535 vs 4.021**, unpaired,
one seed each, no CI, never re-run.

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

   Relatedly, **a training window's T is derived, not configured.** `[data].n_frames` is only a
   ceiling: `_frames` sizes each train window to the labelled span it covers, rounded up to an
   even number (tubelet 2), floor 2. The annotated sessions carry ONE labelled frame per 65-frame
   group, so a fixed T = 24 spent 24 encodes to supervise 1 — 40% of steps at `annot_frac = 0.4`.
   Val and test still enumerate fixed `n_frames` windows, or the metric would not be comparable
   across checkpoints. A train window may also be **strided**, by `[data].frame_strides` (default
   `[1]`, i.e. off) — posetail's `interval`. The derived-T rule then runs on a lattice of spacing
   s: only labels congruent to the anchor mod s are reachable, and T is capped by the room left on
   *that* lattice, not the room from frame 0. Note `SmoothnessLoss` has no notion of dt, so its
   effective weight rises with s; upstream ships both and never couples them, so neither does
   this. And **`SmoothnessLoss` raises below `smoothness_loss_order + 1` frames**
   (`losses.py:1146` narrows by `T - k`), so `run_batch` clamps the order per batch; at T = 2 it
   degrades to a first difference rather than being disabled.
2. **`scene_features=` and `cube_scale=` were dropped from `TrackerEncoder.forward` in 0.3.x.**
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
8. **The crop rule is exact, not approximate.** `crop.py` is lifted verbatim from posetail-pose's
   verified copy (`crop_box_for_points` does not exist in 0.3.x — it was a `memory`-branch
   method). A test asserts it is int32-exact against `crop_cgroup_to_points`. If that fails,
   every detector number is invalid.
9. **Moving-camera inference is not supported upstream.**
   `inference_utils.load_camera_group_from_metadata` ignores `moving_cams` entirely; we build the
   camera group ourselves via `format_camera_group(..., moving_ext={cam: (T,4,4)})`. Only
   `TrackerEncoder` is moving-cam-safe — `ScorerEncoder` and `TrackerTapNext` shape-error on
   `(T,3)` centres.
10. **allen-mouse's npz is column-sorted.** `pose3d.npz['pose']` is ordered by
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
