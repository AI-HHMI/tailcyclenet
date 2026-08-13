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
    track.py            ONE cross-view target set. `--track`; replaces associate+link_rows
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

A run folder also carries **`provenance.toml`** — the commit and a dirty flag at save time. It is
there because a config is not a provenance record and gotcha 12 is what that cost.

### The architecture: two switches

```toml
[model]
query = "prior"           # per-keypoint prior + missing-query tokens (posetail-pose w9_honest)
# query = "none"          # query-free: no prior at all
query_encoder = "wide"    # 512-dim, identity + time, + two query terms iff query = "prior"
# query_encoder = "pose"  # 256-dim, ten terms, 27 of 30 tensors inherited
```

They are **orthogonal**: `query` decides whether a prior is supplied, `query_encoder` decides
which module consumes it. `wide`'s two query terms (`qpos`, `patch`) **default to** `query`,
because under `query = "none"` the prior is never read, so `_query_ok` is all-False for the whole
run and both terms collapse to constant no-query tokens feeding dead gate inputs. So `wide` +
`none` is a 6-term encoder and `wide` + `prior` an 8-term one, and `wide` + `none` **is**
golden's `j3`.

Either term may be **overridden** — `query_pos_embedding = true, query_patch_embedding = false`
is the 7-term `j4_prior` recipe, 3.317 mm allen cross-animal with the anchor gated off and the
only non-anchor arm on record to beat golden j3's 3.394. Two combinations stay unbuildable, and
`build_model` says which key is wrong: both terms off under `query = "prior"` (the prior would
have no route into the encoder), and either term on under `query = "none"` (constant all run).
They only apply to `wide`; naming one beside `pose` asserts rather than silently doing nothing.

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
every keypoint, so they are constants and `pose` is a lower-capacity `wide`. Defaulting its two
query terms to `query` also keeps the old wiring trap unrepresentable — `wide` with both terms off
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

**`gridresid_offset` HAS NO DEFAULT and an absent key raises.** The two values load the same
tensors, so a checkpoint trained under one and built under the other produces numbers instead of an
exception — measured at **+23.1 mm MPJPE and −0.18 MOTA** on 3dpop (report 13 §0b). Gotcha 12.
`load_run(model_overrides=…)` / `scripts/infer.py --gridresid-offset` state the value for a run
folder written before the key existed; that is an assertion about weights nobody recorded, and it
is echoed as one.

The substitution uses the **detached** triangulation, and that is also the loss gate: the direct
term is then constant at unprompted points, contributing exactly zero gradient, so
`coords_loss_direct*` and `coords_softmax_3d` supervise query points only — without forking
`TotalLoss`. In 3D single-view there is no triangulation, so the substituted anchor is the
conf-weighted **mean** of the back-projected rays (`_rays_fallback`) — not the library's
`3d_pred_rays`, which is a weighted *sum*: `conf_pred_2d` is an unnormalised sigmoid and
`tracker_encoder.py:637` never divides by it, so at one camera it lands about half way from the
world origin to the animal.

**Turning off the 3D CE means NaN-ing `grid['anchor_local']`, never popping `grid`.**
`losses.py:680` gates `depth_softmax` — weight 1.5, the largest CE term in `w9.toml`, and
query-*independent* — on the same `'grid' in outputs`, and `losses.py:458` reads `f_eff` out of
the same dict to normalise the depth Huber. Measured on one unprompted window: popping took the
depth CE from 6.24 to off and the depth Huber from 0.617 to 39.49, a 64× renormalisation. A
non-finite anchor is the library's own off switch — `grid_softmax_loss` (`losses.py:45-52`) drops
non-finite targets and returns 0 when all are dropped.

**The shipped sweep is therefore a TWO-LEVER comparison** — `prior` and `none` differ in `query`
*and* `gridresid_offset`. That is deliberate (each arm gets the offset its query implies) but it
means a `prior` − `none` delta cannot be attributed to the prior alone. Say so when reporting it;
eval rule 4, and `improvement_leads.md` opens by retracting a claim of exactly this shape.
Isolating either lever needs a third arm.

**Prompt corruption is REOPENED, with two of its three parts built** (`prompt_offset_px`,
`prompt_stale_frames`, both default 0). The reason is that i.i.d. jitter is not the failure
deployment produces: Gaussian noise averages to zero over the keypoint set, so it teaches the model
to trust the prior's **centroid** exactly — the one quantity a whole-body lag gets wrong. The lag is
real (report 13 RC1: `--anchor carry` cost 23-39% of johnson-mouse's motion). Wrong-animal is *not* built, and that is
now a measurement rather than a shrug: `--oracle-corrupt other` displaces the prior by 1.65 crop
widths (a different bird) and moves the output by 0.008 — alpha 0.005, 61% of frames not moving at
all — because the bounds mask withdraws a prior that far outside the crop. A row swap's damage is the
wrong CROP, not a wrong prompt. The measured echo is alpha ~ 0.48-0.56 for offsets of a quarter to a
half of a crop width, collapsing to 0.10 at a full one as the mask starts firing, so the whole
dynamic range of the drift sits inside the gap the mask does not cover. Bar for keeping either: allen cross-animal inside the reproduction band (near
3.394 mm) *and* a smaller `motion_ratio` gap on johnson. Ships WITH `prompt_dropout`.

Still deleted, deliberately: `kpt_table_mlp`, the crowd head, distractor crops,
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

**`carry` feeds back the ANCHOR-FREE estimate, not the reported prediction** (`--carry-source`,
default `triangulate`). Under `gridresid_offset = "query"` the reported 3D output *is*
`query + R @ residual` with one anchor per keypoint for the whole window, so writing it back closes a
loop with gain: whatever the prior was wrong by is re-added and only the residual carries new
information. `3d_pred_triangulate` is re-derived from each window's own pixels
every frame and is supervised on every keypoint, so carrying it changes no reported output and breaks
only the feedback path. Worth, on 3dpop, MPJPE −0.50 mm / coverage +0.0025 / MOTA +0.0041, all SIG, at
+0.0011 of idsw; and it is what makes report 12's R3 safe to build.

**IT IS NOT THE FIX FOR THE MOTION LOSS, and the mechanism there is worth stating exactly.** `--anchor
carry` does lose **23-39% of johnson-mouse's motion** against the same run with no anchor — and every
consistency statistic in this repo *rewarded* that (best jerk 0.392 vs 0.507, best bone CV 0.148 vs
0.251), because a locked pose is a smooth one. But switching what is carried recovers only 0.5-9.1% of
the path (measured, report 13). The cause is query-anchoring itself: with ANY prior the output is
`prior + R @ residual` off one anchor for the whole window, so all its temporal variation comes from
the residual head, where `--anchor none` substitutes the per-frame triangulation. Only removing the
prior or re-anchoring the output per frame can touch that, which makes it a `prior` + `triangulated`
TRAINING question — the pre-`bcbfbc1` pairing — and an open one. **2D is bit-identical either way** — one camera has nothing to triangulate and the grid head
decodes absolute pixel bins — which is checked on calms21 and rat-city and pinned as a test, and is
what confines the risk of this to the 3D roots. `--carry-source pred` restores the old behaviour so
the comparison stays available.

Nothing in training or val exercises `carry`: the training prior is GT ± i.i.d. jitter or absent, and
val runs `none` then `self` (one window, same pixels, qt = 0). `--oracle-corrupt` (with
`--anchor labels`, diagnostic only) is how the echo coefficient is measured without a training run —
a whole-body offset in **crop widths**, a stale-frame prior, or the neighbouring animal's.

The npz also carries **`box_agree`** (S,T,C) — the predicted centroid's distance from the centre of
the crop box that produced its pixels, in units of one box side, reprojected in 3D. The pipeline held
two independent statements about where an animal is, the box and the pose, and nothing compared them.
In animal-size units, so unlike `--vis-thresh`'s logit one value means the same thing on every root.
And **`pred_tri`** / **`tri_degenerate`**, because every fix above rests on the triangulation and
nothing had ever scored it.

`scripts/eval.py` is offline and model-free: prediction npz + annotation set → MPJPE (paired
bootstrap), PCK, coverage, MOTA/miss/FP/idsw. Multi-animal rows report **matched** MPJPE: row
index is not identity once boxes come from a detector, and scoring row-to-row measured 385 px on
flies that are 30 px across. `--vs other.npz` pairs two prediction files over the groups both scored
and the points **both matched** — including coverage and the FP split, because a delta over points
only one arm attempted measures which arm declined more (eval rule 6).

**A window that predicted nothing says why.** `run_group` writes an `outcome` code per (animal,
window) — `ok / no box / no camera / no points / crop failed / decode failed` — and the crop box it
used. Five aborts wrote the same NaN before, so a coverage number could not be attributed to a cause.
The FP term is split too: `fp_dup` is a second prediction on an animal something else already claimed
and `fp_none` landed on nothing, which want opposite fixes. Measured on 3dpop, `fp_none` is ~10×
`fp_dup` on every arm, which is why cross-track arbitration is not worth building.

### The three inference levers that are measured (dev/reports/11_inference_verified.md)

- **`--det-score`, and it is now the DEFAULT at 0.99** (was 0.05, which was never a considered value
  — it is `decode`'s primitive floor inherited by the deployment path). The objectness is saturated —
  98.5% of rat-city's boxes and 99.98% of 3dpop's sit at exactly 1.0 — so a sweep over 0.05–0.5 moves
  1–3% of boxes and does nothing. At 0.99: MOTA **+0.074** [+0.009, +0.154] on 3dpop with MPJPE
  −2.11 mm and coverage +0.010 also SIG, and +0.073 on rat-city, whose MOTA at the new default
  (0.660) now beats its own GT-crop upper bound (0.635). **The size tracks how many boxes are
  actually below the threshold** — 6–8% on those two, 0.9% on calms21, where it reads +0.012 n.s.
  A detector whose scores are NOT saturated wants this lowered; `scripts/infer.py` prints per-group
  box coverage so that shows up where it happens. `decode` and `eval_detector.py` stay at 0.05 —
  they are the training and detector-scoring paths, and report 10's figures are all at 0.05.
  `det_score` is unconditional in the box-cache stamp for exactly this reason: the stamp otherwise
  records only non-defaults, so moving a default would let an old cache be reused silently.
- **`--vis-thresh`.** `vis_pred` was write-only. Read as a row gate it is worth MOTA +0.049
  [+0.011, +0.110] on 3dpop at 7.3% of rows and 0.601 → 0.628 on rat-city at 14% — and **−0.037 to
  −0.123 SIG on calms21**, whose converter writes every point `VISIBLE` so the head trained against
  an all-true target (`occlusion_acc = 1.0`), and whose baseline coverage of 0.995 leaves almost
  nothing junk to remove. So it is **dataset-conditional, not a default**, in two ways: the threshold
  is a LOGIT with no portable value (medians +2.7 / +4.0 / +15.4 across the three roots), and the
  lever itself can be negative. **Never quote it without the rate-matched random rejection of the
  same number of rows** — any rejection flatters a mean over matched points, and the control is the
  entire reason the number means anything. **Its measured justification is also not yet clean**: at
  `eval.py --min-match-kpts 0` a predicted row that shares ONE keypoint with a GT row can win the
  Hungarian on that keypoint alone, and this gate is precisely what makes rows sparse. Re-run it with
  the guard on before quoting it again. An all-NaN-confidence row is no longer silently kept either
  (`NaN < thresh` was False, so the row the model said least about was the one the gate could not
  touch).
- **`--n-frames` shorter is a trade, not a win.** 24 → 12 → 8 on 3dpop moves MOTA 0 → +0.106 → +0.130
  and pck@10 0.103 → 0.074 → 0.067, monotonically in both directions, with MPJPE inside its interval
  throughout. A shorter window shrinks the crop union AND cuts temporal context; the first buys
  instance recall, the second pays for it.
- **`--refine`** re-crops each window to the first pass's own prediction, label-free, at 2× the pose
  compute. It improves every PCK threshold on all three roots (calms21 +0.028/+0.031/+0.021) and
  leaves MPJPE, coverage and MOTA inside their intervals on both interval-bearing datasets. This is
  the arm that shrinks the crop union WITHOUT shortening the window. It is now **bounded**: a refined
  box that does not overlap the box it came from is rejected, because `crop_box_for_points` SQUARES
  the extent and a pose that wandered lands somewhere else entirely (the giant squares in
  `rat-city_best.npz`). And `crop` keeps the FIRST-PASS box — the refined one goes in `crop_refined`,
  since `crop` is the only record of what the box source offered and every coverage and
  crop-inflation number in reports 08 and 11 is computed from it.
- **`--min-box-frames`** (default 1 = unchanged). One finite box out of T × C used to position a crop
  for all 24 frames and mark every one `ok`: 3dpop reports 0.000 of (row, frame) with no pose against
  2.1–2.2% with no camera at all. Raising it LOWERS reported coverage, which is the point; it moves
  the same number the row matcher moves, so the two need separate arms.

On 3dpop the first two together beat a 7.7×-compute detector (MPJPE 56.17 vs 56.91, MOTA 0.613 vs
0.572) with no retraining.

**Read the FP split per dataset before choosing a fix.** The detector's FP rise over the GT crop is
91% `fp_none` on 3dpop (4 cameras, low overlap) and 90% `fp_dup` on calms21 (two mice, heavy
overlap). Cross-track arbitration is therefore worthless on the first and addresses nearly the whole
term on the second — opposite conclusions from one undifferentiated `fp_rate` of +0.029 either way.

**Do NOT put the window union box through `crop_box_for_points`.** 08 §1.3 asks for it on gotcha-8
grounds and it is measured worse: 3dpop +3.06 mm MPJPE and −0.032 MOTA, both SIG, and rat-city −0.040
MOTA. The union of per-frame crop-rule boxes is already near-square (aspect median 1.047) and
squaring it again grows the p90 box AREA by 82%. A detector box is already a crop-rule box, so the
union already satisfies the `min_crop_dim` floor; and the rule cannot be reproduced from boxes anyway,
because the per-frame extents that would be unioned before squaring are not recoverable.

**A GT crop is only an upper bound where the labels are dense.** The crop rule follows the *labelled*
keypoints, and rat-city labels 2.02 of its 4 per animal-frame, so its GT crop is built from one or two
points and floors at 64 px — the detector arms beat it on MPJPE there. Everywhere the labels are
complete the GT crop wins by a wide margin: +8.57 mm [+4.08, +14.02] on 3dpop (17 of 17) and
+14.93 px [+9.06, +21.21] on calms21 (7 of 7). calms21 is the control that settles it — 2D,
multi-animal and *heavier* overlap than rat-city, so **the inversion is label sparsity and nothing
else**. The "GT crop" row means different things on different roots; say which.

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

`--boxes` is part of that rule, and **nothing used to check it across the two models**: a detector
trained on `instances.pq` boxes serves a keypoint-trained pose run a different crop rule, silently.
`scripts/infer.py` now warns and records `__box_source__` in the npz — a warning and not a failure,
because `scratch/phase1/rat-city-inst` is the best rat-city detector on record (recall 0.531 vs 0.429)
while every rat-city pose run is keypoint-trained, so the mismatched pair is worth running. Its delta
against a matched pair moves two keys and is not a detector-quality result.

**`--link-boxes` is a per-frame Hungarian on CENTRE DISTANCE over the box side, gated at one side.**
It used to be ungated IoU with a force-assign, and all three parts were wrong. IoU ranks by shape
agreement, which is not identity — replaying calms21 frame 301→302 from the box cache, IoU scored the
WRONG mouse at 0.512 against the right one's 0.233 (two touching 220 px mice overlap almost equally)
and the pose error jumped from 4–10 px to 60–82 px; IoU is also exactly zero under fast motion, where
it cannot rank at all. The gate at one box side has 10–16× headroom over real motion (p90 centre
displacement is 0.06–0.11 body lengths on every multi-animal root) and rejects the 3.4–3.9% of 3dpop
row transitions that jump beyond a body length. And an unmatched row now stays EMPTY instead of taking
`free.pop(0)`, an arbitrary leftover: that was rat-city row 9, whose normal-sized boxes teleported
across the arena and whose window union came out **1924×1924 against a 244 px rat**. Unclaimed
detections may be born into an empty row; `last` expires after one window. **`LINK_REV` is in the
`--det-cache` stamp** because the cache stores boxes that have already been linked, so changing this
rule silently makes an old cache a different box set.

**`--track` is the target state, and it is what deletes the above.** `detector/track.py`: ONE
cross-view target set with one affinity and one Hungarian, replacing per-frame `associate` plus
`link_rows`, which never exchanged anything with each other or with `carried`. Report 12 §2.1 measures
the memoryless pass leaving **17.2% of offered boxes unclaimed** at 15 px jitter where target matching
claims all of them, and §2.2 measures it at **4.1 s/frame at C = 16** against ~0.6 ms — johnson-mouse
is a 16-camera rig, so any multi-animal 16-camera rig is currently unrunnable. `associate` stays for
BIRTHS, which is the one place a memoryless pairwise search is right. The affinity is the reprojection
distance of the target's held 3D point over the detection's box side — the same test as report 12's
eq 4 point-to-ray, in the same units and with the same gate as `--link-boxes`, and with no `alpha_3d`
constant to calibrate. Off by default until measured on 3dpop.

**Every detector box is bounded in `unletterbox_boxes`.** `yolox.py:167` decodes a side as
`exp(clamp(-6,6)) * stride` — up to ~12,910 px, ~137,000 source px after a 1/7 letterbox — and IoU-only
NMS cannot suppress it (its IoU with the real box it swallows is ~0). Clamped into the frame; a box
with no positive area comes back NaN, which every consumer already reads as "no box here".

**`min_views = 2` in `associate` was never a threshold — it is the algorithm.** Every instance is
built from a cross-camera PAIR, so `len(members) >= 2` holds by construction and the check could not
fire. `min_views = 1` is a different rule that emits each leftover box as a single-view instance;
`--dup-res-px` gates one that reprojects onto an instance already accepted in that camera, which is
`fp_dup` rather than coverage. Both ship default-off: on 3dpop `min_views = 1` halves the `no box`
outcome rate (0.016 → 0.008) and moves no metric, because `associate` stops at the session's animal
count and with four cameras it already fills every row. And whether the pose model can *use* a
one-camera 3D window is the run's own `[data].prob_2d_only` — 0.25 in the shipped 3dpop and rat-city
runs, **0** in `configs/w9.toml`, where it would be an untrained input shape.

Detection is the expensive half of a run, so `--det-cache` exists to share one box set across arms —
which makes them matched by construction instead of by trusting the detector to be deterministic. Its
stamp is the set of box-affecting options that **differ from their defaults**, deliberately: a
positional list of every value meant that adding one flag refused every cache on disk, which happened
three times in one afternoon and twice mid-sweep.

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
   *that* lattice, not the room from frame 0. `SmoothnessLoss` has no notion of dt — `torch.diff`
   is an UNDIVIDED difference — so its k-th difference grows like `s^k` (256× at k = 4, s = 4) and
   its magnitude would ride on a per-item draw. `_tune_smoothness` divides that out per batch,
   reading the stride off `sample_info['stride']`. What it cannot fix, and what any
   `frame_strides` arm must declare: the HINGE is scale-invariant, and striding genuinely loosens
   it — the threshold tracks the trajectory's k-th derivative, which grows like `s^k`, while the
   per-frame jitter it exists to catch is white and s-independent. **This used to be documented
   backwards** ("effective weight rises with s"): the magnitude rose, the hinge got looser.
   And **`SmoothnessLoss` raises below `smoothness_loss_order + 1` frames**
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
11. **Nothing in the parent process may decode video before the loader forks.** On a video-backed
   root that initialises decord in the parent, and the forked workers then deadlock in a futex
   while holding an open container: 0% GPU, ~0 worker CPU, no traceback, no timeout, forever.
   `scripts/train.py` materialises its fixed val windows before the train loader's first `next()`
   (`persistent_workers` forks there, not at construction), so it used to be the parent that
   decoded; it now pulls them through a one-worker `DataLoader`, which is byte-identical because
   `PoseDataset` seeds val items by index. Measured on calms21: hangs at every reader-cache size,
   never hangs when the parent decodes nothing.
   `scratch/calms21_loader_repro.py` isolates it in seconds, without the model.
   **3dpop is video-backed too and survives it** — which is the only reason the five-dataset sweep
   never hit this, and why "the sweep works" was not evidence that the pose loader was fork-safe.
   Of the shipped roots only calms21 and 3dpop hold `.mp4`; the rest are image directories and
   cannot trigger it — but **any** video-backed root can, including ones a converter makes later
   (`scratch/johnson-mouse/.../johnson-mouse-combined` is 16 cameras of `.mp4`), so this is a
   property of the pixels, not a list of four names.

   This is also why **the reader cache may never be sized by probing.** `_reader_cache_size`
   (`dataset.py`) derives it from a camera count, a frame size and `get_worker_info()` — all
   parsed-toml or in-process facts — because opening one `VideoReader` in the parent to measure
   anything is the deadlock. A single process streaming windows gets `len(rig)`, a loader worker
   gets 4, both clamped by half of physical RAM split across `num_workers`, and
   `TAILCYCLENET_READER_CACHE` is an override rather than a per-dataset requirement. A cache below
   the camera count misses on *every* call: a 16-camera rig at 4 ran detection 2.5× slower.

12. **A RUN FOLDER USED TO RECORD NO COMMIT, and one config key had a default it could not
   justify.** `runs/3dpop-prior` trained under unconditional per-frame re-anchoring, finished nine
   hours before `bcbfbc1` replaced that with the query-anchored residual, and carries no
   `gridresid_offset` — so `cfg.pop('gridresid_offset', 'query')` loaded it as the architecture it was
   not trained as, for weeks, silently. It cost **+23.1 mm MPJPE [+22.3, +24.0] and −0.18 MOTA** on
   3dpop against reading the same weights correctly, and the pose visibly lagged the animal until the
   bounds mask dropped the prior and the fallback snapped it back to the triangulation it *was*
   trained on. Both halves are now closed: `save_run_meta` writes `provenance.toml`, and the key
   raises rather than defaulting. **The `-none` siblings are mismatched harder**: query-free,
   `_query_anchored` substitutes the triangulation at every keypoint, so the residual head's output is
   discarded outright. Keyless generations: `runs/*`, `runs/20260810/*`, `runs/20260810_1711/*`. 2D is
   unaffected (`forward` returns at `model.py:300` before both offset paths), so rat-city and
   branson-fly are keyless and harmless. Any 3D number published off those three needs re-checking.

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
9. **The matcher itself can be fooled.** `match_instances`' cost is a mean over SHARED keypoints, so
   at `--min-match-kpts 0` a row sharing one keypoint is scored on that keypoint and can hijack a GT
   row. It is a FRACTION of K, not a count — K is 4 on rat-city and 47 on allen. Default 0 keeps every
   published number reproducible; any claim about a lever that changes row sparsity needs it on.
10. **Pairing is complete-case, which flatters the arm that failed more.** A group where either side
   is non-finite leaves the comparison, so `paired_bootstrap` returns `n_dropped` and `eval.py --vs`
   prints it. A delta over 9 of 17 groups is not a delta over 17.
11. **There is now a temporal statistic, and there was not before.** `motion_ratio` / `path_length`,
   paired over the steps both arms attempted. Every consistency number this repo had — jerk, bone CV —
   *rewards* a prediction that stopped moving, which is how RC1 stayed invisible. And `box_agree` is
   the pose-against-its-own-box check, in animal-size units.

Reproduction note: posetail-pose's `reports/golden_allen_j3.json` is an exact-reproduction
contract for *that* pipeline. This repo will not match it bit-for-bit and should not claim to.
The check here is a band: allen-mouse cross-animal MPJPE near 3.394 mm, human-vs-human baseline
2.208 mm. A large gap is a port bug, and the first suspects are the allen column-sort permutation
and the crop rule.
