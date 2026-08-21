# tailcyclenet

Finetune a [posetail](https://pypi.org/project/posetail/) point tracker into an animal pose
estimator. Three settings, one model: **3D multiview**, **3D single-view**, **2D single-view**.

This is a clean rebuild of `../posetail-pose`. Everything that was an *experiment* there is
deleted here; what survives is the one architecture that won, the one crop rule, the one window
loop, and a data format that serves both hand annotation and bulk training.

**THIS FILE IS THE CURRENT STATE, NOT THE TRAIL.** What is here is what a reader must know to
avoid breaking something: the defaults, the invariants, the gotchas, the eval rules. Measurements
and refutations live in `dev/reports/`, which is untracked — so state findings here as standing
rules rather than citing a report number. **Do not grow this file with narrative.**

---

## Layout

```
pyproject.toml        package + [tool.pixi.*] env. There is no separate pixi.toml.
CLAUDE.md             this file
docs/                 HUMAN-OWNED. Do not edit without explicit instruction.
  annotation_format.md  THE data format spec. Code is written against it, not the reverse.
dev/                  UNTRACKED. working notes, not the repo's product
scratch/              UNTRACKED. LLM scratch space. May be deleted between sessions.
tailcyclenet/
  format.py           read/validate a session -> dense arrays; the keypoint registry
  dataset.py          torch Dataset over the format; 3D multiview / 3D single-view / 2D
  crop.py             crop_box_for_points -- THE crop rule, int32-exact
  model.py            PoseTrackerEncoder + build_model + the query-free scene centre
  query_encoder.py    WideQueryEncoder + the box-prompt subclass, missing-query tokens
  checkpoints.py      run folders, save/load, warm start, config `extends`
  distributed.py      the world-size axis: what a config key means on N gpus + the rank guards
  metrics.py          MPJPE / PCK / multi-instance matching / MOTA
  adopt.py            --videos: filenames + an anipose calibration -> a Session, IN MEMORY
  video.py            THE one place a video container is opened and decoded (PyAV)
  infer/              THE inference path. `scripts/infer.py` is a 17-line shim onto `cli.main`.
    window.py           one window loop (`run_blocks`, a BLOCK of windows at a time)
    store.py            THE one decode site: (camera, source frame) -> full frame, evicted by block
    predictions.py      the prediction SESSION writer + `load_predictions` (npz still readable)
    driver.py           one run over a dataset: group loop, detection interleaved, session write
    cli.py              `build_parser()` + `main()`
  memory.py           ONE host RAM budget: cgroup ancestry / LSF / MemAvailable / MemTotal
  detector/           YOLOX-Nano box predictor + cross-view association
    track.py            ONE cross-view target set. `--track`
    identity.py         pose_nms, the one keypoint identity lever that works
scripts/              convert_*.py  train.py  train_detector.py  infer.py  eval.py
configs/              base.toml + 2d.toml + 3d.toml + detector.toml. Hand-written.
tests/
```

**Who may write where.** `dev/` and `scratch/` are yours, and both are gitignored — a conclusion
that should outlive the session belongs in this file or `docs/`. `docs/` is the human's — ask
first. `../posetail-pose` and `../../posetail/posetail-next` are read-only reference.

**Commit messages are ONE LINE.** No body, no bullet list, no `Co-Authored-By`, and no mention of
Claude or any other assistant.

---

## Environment

**Filesystem scope: never `find /` or search outside this repo and `~/ghome/projects/tailcycle`.**
Run folders currently live at `/groups/karashchuk/home/karashchukl/results/tailcyclenet/runs`, not
under the repo.

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"
pixi run test
pixi run lint
```

`posetail==0.3.5` comes from **PyPI**, pinned. A checkout of the same version lives at
`../posetail-next` for reading source; do not depend on it by path and do not modify it.

**0.3.5 landed every workaround this repo carried — the monkeypatch layer is deleted, not
maintained.** `synth_motion_*` and the moving-crop geometry in `crop.py` it was the last consumer
of are also deleted rather than defaulted off, so an old config naming them fails loudly.

**`format.py`'s moving-rig support (`moving_ext` / `extrinsics.pq` / `moving = true`) is KEPT** —
that is the format spec's contract and 0.3.5 supports it end to end. No shipped root exercises it.
Do not delete it as dead code.

The `LD_LIBRARY_PATH` prepend in `[tool.pixi.activation.env]` is load-bearing: the env ships
`libstdc++.so.6.0.35`, the host may ship 6.0.29 with no `CXXABI_1.3.15`, and without it
`import scipy.optimize` dies inside `_highspy` with an error that names only CXXABI.

**`[tool.ruff.lint].select` is pinned to `E4/E7/E9/F`.** ruff 0.16 widened its implied default set
to include rules whose autofixes rewrite import blocks and `dict()` calls repo-wide. Widening it is
a deliberate act with its own diff, not something a version bump does to you.

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
    extrinsics.pq      per-frame extrinsics, moving cams (optional; NO shipped root has one)
    groups/<gid>/<cam>/000000.jpg …   or   groups/<gid>/<cam>.mp4   (symlinks fine)
```

Five things that are easy to get wrong:

- **Split is a directory, not a column.** None of the shipped datasets split cleanly by session, so
  a session-level field could not express them.
- **`mode` is per-session.** One `train/` may hold 2D and 3D sessions; both head-bank slots get
  gradient. Sessions need not agree on `names` either: `Dataset.names` is the **union** and
  `Registry.ids_for` remaps each session's own axis onto it **by name**, so a session may reorder
  or subset the keypoints.
- **A group is a contiguous clip**, `n_frames` unbounded — a whole 57k-frame recording may be ONE
  group whose `cam0` is a single symlink. **A group's length is not its label count**: an annotated
  group may be 65 frames carrying exactly ONE labelled frame. This is why a train window's T is
  derived rather than configured (gotcha 1), and why an index entry is a poor sampling weight.
- **`status` is the visibility channel**, in both label tables, dictionary-encoded so
  `vis = codes == VISIBLE` is a vectorized int8 compare. But **coordinates live on `VISIBLE` *or*
  `PROJECTED`** (`fmt.POSITIONED`): `projected` is a position with no visibility claim, for a
  source whose annotators placed every keypoint in every view and never recorded occlusion. It must
  never reach a visibility target — `dataset.py` NaNs it out, and withholds `vis`/`vis_2d` entirely
  when a window has no assessment at all, because the noisy-OR would otherwise assert that nothing
  is reconstructible in 3D.
- **`keypoints.pq` `x,y` may be null on a `visible` row** iff a `points3d` row exists for the same
  key. That row is a *visibility observation* — a root may ship real per-camera visibility with no
  per-camera 2D, and this is how it is kept without inventing positions.

### Per-root traps

- **`rat-city-annotated`** (APT `.lbl`, `scripts/convert_apt_lbl.py`): 1,087 groups of 65 frames
  with ONE labelled frame at index 32, K = 4, names identical to the tracked root so both share
  registry ids and embedding rows. **Its per-camera visibility is the calms21 failure mode by
  decision** — APT records an occluded-but-placed point (22% of slots) and the format has no such
  status, so those are written `visible` with their coordinates. Never use this root's vis head as
  a row gate without the rate-matched random control. **And its `p` column is x-block-then-y-block**:
  `labels{i}.p` is `(2K,n)` and decodes as `reshape(2,K,n)`. Both readings look plausible; the
  interleaved one puts 28% of points outside the frame against 0.01%. Asserted on VALUES in
  `tests/test_convert_apt_lbl.py`, not on shapes. A labelled frame also labels only *some* of the
  animals in it.
- **`regions.pq`** is APT's "Label Box": an area the annotator certified as completely labelled.
  Polarity is measured, not inferred from the name. **The file's ABSENCE is the claim of exhaustive
  labelling**, so a session with no regions writes an **empty** one rather than none — `None` and
  `(0,6)` are different answers. Membership is by **CENTROID, not overlap**. A region is kept only
  on the group whose own labelled frame it sits on.
- **`3dpop` ships an `instances.pq`** written from v3, which restores animals that v4's
  score-cleaning deleted outright — birds physically in frame with no row anywhere, so every
  prediction on one scored as a false positive. It flips `box_source = 'instances'` from inert to
  live, so no new 3dpop number is comparable to an older one without `box_source = "keypoints"`.
  And in 3D a `present` row has no box to localise against, so `_in_ignore` excuses every unmatched
  prediction on a frame carrying one — about a quarter of 3dpop's frames. **Quote `fp_ignored`
  beside MOTA or the number is unreadable.**
- **`rat-city` IS A SYMLINK to `rat-city-tracked`**, and the `*-combined` roots are symlink farms,
  so a fix in one place lands in up to three roots and `find` without `-L` undercounts them.
- **A root's file date cannot settle whether its tables are current — regenerate and diff.** The
  one root that was actually stale would have looked *fine* on dates.
- **THE STATUS POLICY: `missing` claims a real assessment; absence means none was made.** Checked
  by `scripts/check_status.py`, not by prose — run it after any re-conversion. On the roots as
  they ship: every tracked root (3dpop, rat-city-tracked, branson-fly, johnson-mouse-tracked,
  calms21) already writes NaN/absent as no row, which the spec defines as identical to
  `unlabeled`. **`allen-mouse-tracked`'s 2.1M `missing` keypoint rows are not NaNs** — they come
  from the npz's own per-camera `vis` array, written only where the 3D point is finite, and are
  the only dense per-camera occlusion target in the repo; do not read "no coordinate" as "no
  assessment" and demote them. **calms21 has zero NaN and zero holes** (its converter asserts
  finiteness) and stays `visible` on purpose — see `Session.has_visibility_assessment` in the
  losses section below for what that means for training. The three small, deliberate exceptions
  — rat-city-annotated's 220 APT skips, allen-mouse-annotated's 211 points3d holes
  (`convert_annotated.py::triangulate_group`'s 1-view-or-over-reprojection-gate case, left as no
  row — its SEPARATE 178 `missing` rows are the 0-view-and-every-camera-assessed case, already
  written `missing` today, and are counted in `MISSING_OK` rather than `HOLE_EXEMPTIONS`),
  johnson-mouse-annotated's 19 outlier-filtered holes — are named with their exact counts in
  `scripts/check_status.py`'s `HOLE_EXEMPTIONS`; a count that moves there is a re-conversion
  silently disagreeing with the documented reason. See
  `dev/plans/status_consistency_and_occlusion.md` for the full survey.

### Which roots can carry a benchmark

| root | test groups | frames/group (min/med/max) | usable? |
|---|---|---|---|
| **3dpop** | 58 | 755 / 908 / 2,680 | **yes — the best split in the repo** |
| **calms21** | 19 | 4,881 / 17,483 / 23,810 | **yes, and long** |
| rat-city (= `-tracked`) | 1 | 500 | **no — this is the DEGENERATE one** |
| rat-city-combined | 21 | 20x65 + 1x500 | marginal |
| allen / johnson `-tracked` | 1 | 500 / 950 | no |
| allen / johnson `-annotated`, `-combined` | — | — | **NO `test/` SPLIT AT ALL** — every number on those roots is a `val` number |
| branson-fly | 1 | 500 | no |

**The long-clip protocol this repo keeps asking for already exists as data.** 3dpop's 120-frame
truncation discards 88% of its split, and calms21's 262,107 test frames are entirely unused. Both
at `--chunk 500` are chunks of MANY independent clips, so the within-clip caveat is far weaker
there than on rat-city's single group.

---

## Training

```bash
pixi run python scripts/train.py --config configs/3d.toml --data <root>   # or configs/2d.toml
```

**THE TWO CONFIGS DIFFER IN THREE KEYS AND THAT IS THE POINT.** `configs/base.toml` holds
everything shared; `2d.toml` and `3d.toml` `extends` it and add only what the setting genuinely
requires — `cams_to_sample`, `val_cams_to_sample`, `prob_2d_only`, all three camera-count questions
a one-camera root cannot ask. Resolution is **one level deep by design**
(`checkpoints.load_config`): a chain of overlays would put the difference back out of sight.

`[data].path` is either a dataset root (has `train/`, optionally `val/` and `test/`) or a folder
whose children are dataset roots. In the second case keypoint names are prefixed with the dataset
folder name and one estimator trains across all of them; the keypoint embedding table is what makes
this work.

`n_keypoints` is **derived** from the registry, never configured. The registry is written to the
run folder as `keypoint_registry.toml` and read back at inference. A later run **appends** new
names so old ids — and the embedding rows behind them — survive warm start. `warm_start` copies the
base rows into the grown table keyed on the base registry's own length; if that does not match the
checkpoint's row count the copy is **refused rather than guessed**, because a mis-applied row copy
points each embedding row at a different body part.

A run folder also carries **`provenance.toml`** — the commit and a dirty flag at save time. A
config is not a provenance record, and gotcha 12 is what that cost.

**THE ENCODER UNFREEZES MID-RUN, AND `[model].video_encoder_requires_grad` IS THE ONLY THING THAT
SAYS SO.** A bool means never / from step 0; an **int is the iteration to unfreeze at**, with
`video_encoder_finetune_last_n_layers` for how many trailing blocks. Shipped: **8 blocks at
iteration 10,000**, and **that is a default, not a result** — every number in this file was measured
with the encoder frozen for its whole life, so a run started after this is not comparable to one
before it; `video_encoder_requires_grad = false` restores the old arm exactly.
`[training].freeze_encoder` is DELETED and naming it raises: it re-froze the encoder after
`build_model`, so two sources of truth for one fact would have let the loop's silently win.

The model side is posetail 0.3.5's. Three things this repo adds in `tailcyclenet/unfreeze.py`,
each a silent failure otherwise:

- **A frozen param is in no param group**, so newly-trainable tensors must be ADDED, routed by the
  same `route_param` build time uses — a bare flag flip hands the encoder gradients nothing steps.
- **They must be NEW groups, not pre-registered ones.** Schedule-free advances `k` on every `step()`
  regardless of gradient and the averaging weight is `~1/k`, so a group grad-less for 10,000 steps
  folds its encoder into the averaged iterate at 1/10000 — leaving `model_state_eval` (what is
  deployed, and what `best` is selected on) holding a barely-moved encoder.
- **The clip target is recomputed after a fire**, or the new tensors are unclipped and unchecked for
  non-finiteness. `encoder_lr_scale` stops being inert at the unfreeze; the new group warms from its
  own `k = 0`, which is why there is no warmup key.

**REPO-LOCAL DIVERGENCE:** upstream leaves `norms_block` — applied at the hierarchical taps and
feeding the decoder — frozen behind a trainable block. This repo also unfreezes the norms whose
layer is inside the range. Unmeasured. **Resume replays the unfreeze onto a frozen-layout
optimizer**, because `load_state_dict` matches groups by position; a count mismatch raises.

**MUON IS THE OPTIMIZER, AND AN ABSENT `[training.optimizer].optimizer` KEY MEANS MUON.** Ported and
**not** re-measured on this repo's roots. Three traps: **`nn.Embedding` weights are 2D but
excluded**, because each row is a keypoint identity and orthogonalising mixes body parts; **resuming
an AdamW-SF run folder requires `optimizer = "schedulefree"`**, or train.py refuses the mismatched
checkpoint by name rather than `KeyError`-ing mid-restart; and the **fresh 2D matrices go to Muon at
`kpt_lr`**, one lever off the reference and the first suspect if a Muon arm disappoints.

**ONLY THE AdamW HALF IS GRADIENT-CLIPPED, AND A MUON RUN'S `grad_norm` IS A DIFFERENT QUANTITY.**
Muon orthogonalizes its own gradients inside `step()`, so their raw norm is not a step size — and
those matrices carry ~95% of the global norm, so a shared clip throttled the fresh heads and
embeddings by an order of magnitude. `train/grad_norm` and `train/grad_norm_muon` are not comparable
to each other, and only the first is comparable to a schedulefree run. **A run started before this
fix is not comparable to one after it.** Raising `max_grad_norm` past 10 does NOT help.

### Multiple GPUs

```bash
pixi run python scripts/train.py --config configs/3d.toml --data <root> --devices 4
```

One rank per GPU on ONE node, Lightning Fabric + DDP. `--devices 1` takes no wrapper at all and is
the loop it always was. **The mechanism, every gotcha and every measurement are in
`dev/reports/37_multi_gpu_ddp.md`; these are the standing rules.**

- **`batch_size` IS STRUCTURALLY 1 PER RANK, SO THE WORLD IS THE BATCH DIMENSION.** The collate
  keeps one camera group per batch; DDP averaging N items is the only batch this repo has.
- **EVERY ITERATION COUNT IN A CONFIG IS A TOTAL ACROSS RANKS**, and `learning_rate` / `kpt_lr`
  are scaled by **sqrt(world)** (the multipliers are not, so every ratio holds). **A multi-GPU
  number is TWO levers off a single-GPU one**; `provenance.toml` records which it was.
- **THE STAGED UNFREEZE RE-WRAPS THE MODULE, AND THAT IS NOT OPTIONAL.** DDP registers parameters
  when the module is WRAPPED and never re-checks, so a tensor unfrozen later is never all-reduced
  — each rank would train its own encoder, silently. The reference has this bug latently.
- **A DDP STEP COSTS THE SLOWEST RANK'S ITEM**, so the cost-determining draws (camera count, the
  `prob_2d_only` coin) are synchronised across ranks by `StepSampler`'s ordinal — worth 1.57x →
  1.18x at 4 ranks. **Anything that varies per item and multiplies work belongs in `_shape`.**
  Measure it with the encoder UNFROZEN or the cost spread collapses and it reads as a null.
- **`check_ranks_agree` raises at every checkpoint boundary if the ranks have parted company.** A
  correct run reads **exactly 0**; the one real bug it has caught reads **~3e-07**. `tol = 1e-9`
  sits between those two measured populations — **do not loosen it toward 1e-3**, which is 2,000x
  above the failure it exists to catch. A drift of 1e-7 is not a rounding wobble, it is two ranks
  training two different models, and it grows.
- **Anything rank-0-only that touches WEIGHTS must still run everywhere.** `save_checkpoint`'s
  schedule-free eval/train toggle is not bit-exact, so it takes `write: bool` and every call site
  runs on every rank as `save_checkpoint(..., write=is0)`. Gating the whole call on `is0`
  reintroduces the bug.
- Smaller ones, each a hang or a silent correlation otherwise: a skipped step is a **collective**
  decision; val is **sharded** and gathered before reducing; the sampler takes a **per-rank
  generator** (a `DistributedSampler` is wrong here); `[data].num_workers` is **per rank**;
  `16-mixed` is refused by name.

### The architecture: two switches

```toml
[model]
query = "prior"           # per-keypoint prior + missing-query tokens
# query = "none"          # query-free: no prior at all
box_prompt = "film"       # FiLM on the identity term
# box_prompt = "none"     # byte-identical to a config without the key
```

`query` decides whether a prior is supplied. `query_encoder` has one value, `"wide"`; the
256-dim `pose` encoder and the `term` box encoder were config-unreachable and are deleted, and
naming either raises by name.

`wide`'s two query terms (`qpos`, `patch`) **default to** `query`, because under `query = "none"`
the prior is never read and both terms would collapse to constant no-query tokens feeding dead gate
inputs. Either term may be **overridden** — `query_pos_embedding = true, query_patch_embedding =
false` is the 7-term variant, the only non-anchor arm on record to beat the 6-term one. Two
combinations stay unbuildable and `build_model` says which key is wrong.

`query = "prior"` carries a per-keypoint prior plus `prompt_time`. Every position-derived fusion
term carries a **learned no-query token** where a keypoint has no prior, instead of a value computed
from the crop centre and presented as real. `prompt_dropout` is the fraction of **steps** that run
fully query-free; `prompt_noise_px` is the jitter on the priors it keeps — **in pixels**, converted
for 3D by `cube_scale`. One scalar in each session's own units could not work: one root holds 63 px
sessions beside 14 mm ones. Ship it WITH the dropout — dropout alone is significantly worse.

**`prompt_dropout` IS DRAWN PER ITEM, NOT PER KEYPOINT.** The decode is per-keypoint independent,
but `scene_center`, `scene_radius` and `cube_scale` are derived from the WHOLE `coords_q` set. A
per-keypoint draw at p = 0.5 puts `scene_center` 13.6 mm from its deployment value (0 of 200 draws
within 1 mm); per item it is exactly 0. **Per keypoint, the geometry the model meets at deployment
is never trained.**

**`prompt_offset_px` is a WHOLE-BODY offset and it is the corruption with the shape of a deployment
failure.** i.i.d. jitter averages to zero over the keypoint set, so it teaches the model to trust
the prior's *centroid* exactly — the one quantity a whole-body lag gets wrong. The two differ ONLY
through the scalars that read the whole prior set, where a whole-body offset moves the centroid by
the full lag while i.i.d. draws cancel to `sigma/sqrt(K)`. **`base.toml` ships 5.0 on BOTH 2D and 3D
as a consistency choice, not a measured optimum** — the mechanism is 3D-derived and arguably
*unmeasurable* in 2D rather than harmful there. Say so when reporting a 3D arm: it is no longer
paired with the only measured value, 8.0.

The instance-anchor machinery is **deleted**, not defaulted off — including the best row on record.
`query_patch_embedding` builds its `PatchProcessor` at the base checkpoint's `embed_dim` and
projects to the fusion width, so all ~5.4M of the pretrained patch CNN load by name; building it at
512 would inherit 93k of 5.5M and silently retrain the rest from noise.

### The 3D output: `gridresid_offset`

`output_mode = "gridresid"` reconstructs `world = query + R @ residual` with ONE anchor for the
whole window. What that anchor is is a switch:

- **`"query"`** — the native structure, kept **per keypoint** only where a real prior anchored it;
  every other keypoint falls back to that frame's own **triangulation**. Under `query = "none"` that
  is every point, so the prediction is the triangulation outright and the `grid` CE target is
  dropped. **A COROLLARY THAT COSTS AN EXPERIMENT: query-free, `out_3d` never reaches the output**,
  so every `[model]` key that only affects it — `grid_decode_space`, `log_3d_output`,
  `head_3d_grid_radius`, the subpixel refinement — is INERT under `--anchor none`. A sweep over one
  of them there returns bit-identical numbers, and that is not evidence the key does not matter.
  Probe with `--anchor labels` or a prompted arm.
- **`"triangulated"`** — recover the residual and re-add it to **each frame's** own triangulation,
  for every keypoint.

Pair them with `query`: `prior` → `"query"`, `none` → `"triangulated"`. **`query = "prior"` +
`"triangulated"` is REFUTED** at **+0.289 mm [+0.115, +0.473] worse**, with the prompt doing the
least accuracy work of any arm — per-frame re-anchoring frees the motion by leaving no static anchor
to distort, and costs the accuracy by leaving the prior nothing to contribute. **You cannot get
per-frame freedom and a load-bearing prior from this one key.**

**`gridresid_offset` HAS NO DEFAULT and an absent key raises.** The two values load the same
tensors, so a checkpoint trained under one and built under the other produces numbers instead of an
exception. Gotcha 12. `load_run(model_overrides=)` / `--gridresid-offset` state the value for a run
folder written before the key existed; that is an assertion about weights nobody recorded, and it is
echoed as one.

The substitution uses the **detached** triangulation, which is also the loss gate: the direct term
is then constant at unprompted points, so `coords_loss_direct*` and `coords_softmax_3d` supervise
query points only without forking `TotalLoss`. In 3D single-view there is no triangulation, so the
substituted anchor is the conf-weighted **mean** of the back-projected rays — not the library's
`3d_pred_rays`, which is a weighted *sum* (`conf_pred_2d` is an unnormalised sigmoid and nothing
divides by it, so at one camera it lands about half way from the world origin to the animal).

**Turning off the 3D CE means NaN-ing `grid['anchor_local']`, never popping `grid`.**
`depth_softmax` — weight 1.5, the largest CE term, and query-*independent* — gates on the same
`'grid' in outputs`, and the depth Huber reads `f_eff` out of the same dict. Popping it took the
depth CE from 6.24 to off and the depth Huber from 0.617 to 39.49, a 64x renormalisation. A
non-finite anchor is the library's own off switch.

### The losses: five keys fire in 2D from `TotalLoss` itself, plus one repo-local key

`TotalLoss.forward` branches on `coords_true.shape[-1]`. The `R == 2` block computes
`coords_loss_2d`, `coords_softmax_2d`, `smoothness_loss_2d` and the two smoothness shape keys;
everything else is `_nan()`'d and dropped.

**`TotalLoss` ITSELF SUPERVISES NO VISIBILITY ON A 2D-ONLY RUN.** Both of its vis terms live in
the 3D `else`, and `vis_loss_weight = 5.0` — the largest weight in the block — gets no gradient
there. **`tailcyclenet/losses.py::PoseLoss` is the fix**, not a library key: `[training.losses]
vis_loss_2d_weight` (default **0.0**, so an absent key is bit-identical to every run on record)
adds a per-camera BCE on `outputs['vis_pred_2d']` against the loader's own `vis_2d` — the same
tensor `noisy_or_logit` returns UNCHANGED at one camera, so this supervises exactly what
`--vis-thresh` gates on. `scripts/train.py` builds `PoseLoss` in place of `TotalLoss`
unconditionally; every other `[training.losses]` key is unaffected.

**THE LOADER GATES THE TARGET AT THE SESSION LEVEL, NOT JUST PER WINDOW.** A `missing` row
supervises occlusion in 2D exactly as it already did in 3D — a `MISSING`/`VISIBLE` window builds a
real target, `PROJECTED`/`UNLABELED` masks out per-point as NaN (posetail's own per-entry mask).
But a `tracked` session whose table is 100% `visible` (calms21, rat-city-tracked, branson-fly) has
no NaN to mask on — every row is finite — so `Session.has_visibility_assessment` withholds the
whole session's target instead: no `missing` row anywhere in a `tracked` table means no assessment
ever happened, and training "always visible" from a target that asserts nothing is the calms21
failure mode CLAUDE.md already warns about for `rat-city-annotated`, one level up. An `annotated`
session is never gated this way, even at zero `missing` rows.

**`rat-city-annotated` and calms21 must still not be used to judge or gate this head.**
rat-city-annotated's 2,500 occluded-but-placed points are written `visible` on purpose (the
bullet above), so its target calls real occlusions positive; calms21's target is withheld
entirely (previous paragraph) and MUST NOT be turned on by naming it in
`Session.has_visibility_assessment`'s gate. `allen-mouse-annotated` is the one root with a dense,
honest 2D occlusion channel (41.9% `missing`) and is where a nonzero weight has to be measured
first — see `dev/plans/status_consistency_and_occlusion.md`.

That, and not "the converter wrote every point VISIBLE", is why `--vis-thresh` on a 2D root used
to gate on **an untrained head** — see the table below for what changes once the weight is on.

`conf_loss_weight`, `conf_2d_loss_weight` and `coords_loss_rays_weight` ship at 0.0 as documented
off-switches for terms that do exist. `gamma`, `feature_loss_weight` and `pixel_thresh` were
DELETED: the first two gate on outputs `TrackerEncoder` never emits, and the third feeds only the
two conf losses. `coords_loss_direct_weight` is applied **twice**, so its effective weight is double
what the config says.

---

## Inference: one entry point

```bash
pixi run python scripts/infer.py --data <ONE session dir> --run runs/<x>/ --out pred/
# or, straight off raw footage:
pixi run python scripts/infer.py --run runs/<x>/ --out pred/ \
    --videos rec/ --calibration anipose/calibration.toml --cam-regex 'cam([0-9]+)_' \
    --detector runs/det-<x> --max-animals 4
```

**`--out` IS A SESSION DIRECTORY IN `docs/annotation_format.md`, NOT AN NPZ**, written a block at a
time so nothing is proportional to clip length: `session.toml` (with a `[provenance]` naming the
run, the checkpoint file, the detector and the SOURCE SESSION, all absolute), `calibration.toml`,
`groups.pq`, `points3d.pq`, `keypoints.pq` (the per-camera 2D pose, which a 3D run used to
discard), `instances.pq` (box + `score` + `box_agree`) and a non-spec `windows.pq`. **No pixels and
no `groups/`** -- so `validate_session` reports one rule-7 error per (group, camera) and nothing
else.

**NOTHING SHIPPED READS `[provenance] source_session`, and this file used to claim `render.py`
does.** It does not: `render.py` takes its own `--data` and reads a prediction **npz**, and has
never been updated for the prediction-session format. `predictions.py` reads only
`source_session_id`. The `source_*` keys are what a future render path would read.

### `--videos`: raw footage plus an anipose calibration

**THE SESSION IS BUILT IN MEMORY (`tailcyclenet/adopt.py`). NOTHING IS STAGED.** `Group.source` is
the ONLY filesystem entry point in the decode path and it is a cache over `_src`, so
`format.video_group` pre-fills it and `pixels()`, `Group.dir` and `session.path` are never
reached; `format.VideoSession` overrides `_table` to `None` so a colliding `path` cannot adopt
someone else's parquet. `tests/test_format.py` pins both. **The cost: `validate_session` can no
longer be the acceptance test**, so `tests/test_adopt.py` compares the in-memory session against
the on-disk one `write_session` produces -- once, in CI. `--dump-session` is the debug artefact,
not the mechanism.

- **The naming rule is anipose's `cam_regex`, verbatim**: camera = `search(rx, stem).group(1)`,
  group = `sub(rx, '', stem)`. **THE CAMERA NAME IS THE CAPTURE GROUP**, so `cam([0-9]+)_` yields
  `0` and not `cam0` -- the most common error by a wide margin, and refusal 2 says so. Two
  supersets: a **group-less** pattern matches the whole string (`Cam[0-9]+` -> `Cam2005325`, which
  is what a raw rig dump needs, and which does NOT consume the separator), and a one-camera
  calibration needs no regex. **An empty group id is about DISAGREEMENT, not emptiness**: all
  empty is one group (`--group-id`), some empty and some not is refused.
- **`--calibration` is the aniposelib layout and nothing else.** A `multiview_calib`/OpenCV-YAML
  rig needs a converter script ending in `dump_calibration`; that stays out of `adopt.py`.
- **No labels means no eval**, and it makes `--max-animals` (with `--detector`) and a box source
  mandatory. Refusals 1-11 are PURE and fire above the checkpoint load and above any decode; 12-14
  need the probe. `--anchor labels` / `--box-prompt labels` are refused, not warned.
- **The keypoint axis comes from the run's own `keypoint_registry.toml`** -- `--dataset-name`, or
  the sole entry, printed. This also retires the `path.parent.parent.name` trick: `ds_name` is
  stated outright.
- **`--videos` IS AN INPUT PATH, NOT A SECOND PIPELINE**, pinned as byte-identity against the same
  videos reached through a hand-authored session directory.
- **Provenance is the CLI INPUTS, not the derived map**: `source_videos` (the RESOLVED, expanded
  file list -- **the one non-scalar provenance value in this repo**), `source_calibration`,
  `source_cam_regex`, `source_group_id`, and `source_session = ""` deliberately. `plan()` is pure
  over the filenames, so it re-derives the map exactly, and
  `adopt.session_from_prediction` **CHECKS ITSELF** against the prediction's own `groups.pq` and
  `calibration.toml`. A per-(group, camera) key would collide (`a_b`/`c` against `a`/`b_c`).
- **`build()` OPENS CONTAINERS IN THE PARENT PROCESS (gotcha 10).** Safe from `scripts/infer.py`,
  which never forks. **Never call `adopt` from `scripts/train.py`.** The probe opens one reader at
  a time (pool of 4) and drops it -- it must NOT use `dataset._reader`, whose size would be fixed
  before the run's own budget is resolved, and whose cost is quadratic in the reader count. On a
  16-camera raw rig the probe is minutes of cold opens; it prints per camera, and it warms the
  page cache the run then reads through.
- **Frame counts come from decord, not from container metadata**, because `n_frames` is a promise
  that every index in `[0, T)` decodes through the same reader the loader uses.
- **A ragged group is refused, not min()'d** (`--trim-to-shortest` opts in): a dropped trigger and
  two different recordings sharing a group id look identical to a `min`.

### `--start-frame` / `--end-frame`: a frame range on BOTH inputs

**A WINDOW-LOOP LEVER, NOT AN INPUT-FORMAT ONE**, which is why one implementation serves both.
Half-open `[start, end)`, per group, `0` = to the end.

- **`frame` IN THE OUTPUT IS ALWAYS THE SOURCE INDEX**, and `groups.pq` keeps the group's FULL
  `n_frames`. `load_predictions` therefore densifies back to a full-length array that is NaN
  outside the range, and `eval.py` / `--chunk` / `--vs` need no change. Re-basing to 0 would score
  frames `[start, end)` against labels `[0, end-start)` -- the `chunk_frames` failure exactly.
- **`--max-frames N` IS `--start-frame 0 --end-frame N`** and resolves into one quantity; the two
  spellings are **REFUSED together**, not ordered, because "120 frames from 300" and "up to frame
  120, from 300" read with equal force.
- **A RANGED RUN IS NOT A SLICE OF THE WHOLE-CLIP ANSWER.** The detector boxes ARE byte-identical
  (next bullet) and the accuracy columns are comparable, but `--track` and `--link-boxes` carry
  state and `carry` has no prior at `start`, so the IDENTITY columns are comparable only between
  runs that START together. **And byte-identity on the accuracy columns holds only where the
  window partition agrees**: `_window_starts` pulls the LAST window back to end exactly at
  `--end-frame`, which is not a whole-clip window boundary unless the range runs to the clip end,
  so the few frames past the range's final start came from a different window (eval rule 11's
  family). Pinned both ways in `tests/test_infer.py`.
- **THE DETECTION CURSOR OPENS ON A GLOBAL `_DET_BATCH` BOUNDARY AT OR BELOW `--start-frame`, and
  the lead-in is discarded.** `detect_raw` ASSERTS the alignment because a short leading batch is
  an input SHAPE the whole-clip pass never produces. Do NOT relax that assert and do NOT quantise
  the user's `--start-frame` by 16. Leaving the cursor at 0 is not merely wasteful -- it pulls the
  whole prefix through the frame store, which is sized from the WORK.
- **A group shorter than `--start-frame` is SKIPPED BY NAME; every group skipped is REFUSED.**
- **NOT A RESUME.** It bounds which frames are predicted; it does not reconstruct the tracker or
  the carried prior the whole-clip run would have held at that frame.
- It also retires `make_clips.py`'s cut: a whole-video group plus a range indexes into the shared
  mp4 (`VideoReader.get_batch` seeks), with no keyframe-lattice constraint and no duplicated
  pixels.

**ONE SOURCE SESSION PER RUN, refused before the checkpoint loads.** A session holds one
calibration, one mode and one keypoint axis. A 58-group 3dpop protocol is now 16 invocations and 16
directories; `eval.py` still keys on `session/group`, so scoring them together is a reader change
that is NOT built. **One session may hold MANY groups** -- twelve trials of a four-camera rig is
one `--videos` invocation.

**`--data` AND `--videos` ARE EXACTLY-ONE-OF**, and both contribute the same provenance KEYS with
different values, which is what keeps the branches honest: `SessionWriter` raises on a key given
twice with two values. `--split` defaults to `None` rather than `"test"` so the videos path can
tell whether it was PASSED (it is inert there, and refused rather than ignored).

**`scripts/eval.py` STILL READS EVERY EXISTING `.npz`** -- `load_predictions` dispatches on the
path -- because every published number lives in one.

There is **one** window loop. Box sources are the annotation set, a detections npz, or a
per-dataset detector. Prompt regimes: `none` (query-free), `carry` (previous window's own
prediction — what deployment does, requires `overlap >= 1`), `self` (two passes), `labels`
(ORACLE, gated off by default — see eval rule 7). **Rendering is `scripts/render.py`, no longer a flag**: the loop never holds a whole clip's `pred`.

### The measured defaults

The 3D column is 3dpop, **16 sessions / 47 `--chunk 500` units**, paired, one lever off
`--anchor carry --overlap 8`; the 2D column names its root because **the two 2D roots disagree**
(rat-city 500 f / 5 units; calms21 6 sessions x 2000 f / 24 units).

| flag | default | 2D | 3D |
|---|---|---|---|
| `--anchor` | `carry` | **ROOT-CONDITIONAL**: rat-city +5.04 px SIG worse, calms21 MOTA +0.082 SIG better | mean NULL; carry wins the bulk |
| `--overlap` | **8** | 12 best, 8 within noise (+0.79 n.s.) | −0.626 mm vs 4 |
| `--refine` | **on in 3D, off in 2D** | accuracy yes, identity no | **−0.962 mm [−2.104, −0.216] SIG** |
| `--crop-source` | `boxes` | **`boxes`** | **NULL** (+0.181 [−0.182, +0.546]) |
| `--det-score` | 0.5 | **per-checkpoint** | per-checkpoint |
| `--track` | on | **INERT** (`C > 1` gate) | on |
| `--carry-source` | `triangulate` | **INERT** (bit-identical) | on |
| `--pose-nms` | off | +0.0223 MOTA on rat-city, **harmful on calms21** | untested |
| `--vis-thresh` | off | **untrained at `vis_loss_2d_weight = 0.0` (the shipped default) -- UNMEASURED at nonzero, and rat-city-annotated/calms21 must not be the arm that measures it** | +0.049 on 3dpop |
| `--refine-px` | none | **SWEEP IT**: 96 best on calms21, **+15.8 px / MOTA −0.238 SIG on rat-city** | 192 ≈ 256; 96–128 trade mm for MOTA. **INTERACTS WITH `--crop-inflate`** |
| `--box-prompt` | `auto` | **LIVE, 2D single-camera** | **LIVE, PER CAMERA** |
| `--crop-inflate` | 1.5 iff box model | wide pass-1 + tight pass-2 | measured on 3dpop as a secondary axis alongside the box |
| `--min-views`, `--max-move`, `--min-box-frames` | 2 / 1.0 / 1 | clean nulls or no gain available | |
| `--prefetch-windows` | **1** | bit-exact at any value | bit-exact at any value |
| `--max-ram` | derived | **byte-exact at any value** (it sizes the BLOCK) | byte-exact at any value |

**THE `+0.049 on 3dpop` `--vis-thresh` FIGURE IS NOT A 3dpop RESULT.** 3dpop ships only
`points3d.pq`, so `lab.vis2d is None` and the loader has always emitted `vis = vis_2d = None` for
it -- `TotalLoss` then hard-zeros both visibility terms, so 3dpop's own training never touches the
head `--vis-thresh` gates on. Whatever produced that gate figure came from the base checkpoint or
another root in the training mix, not from 3dpop's own supervision. Unchecked; do not re-quote it
as evidence for or against this root without finding out which.

**`--anchor` IS ROOT-CONDITIONAL IN 2D, AND "HARMFUL IN 2D" WAS ONE ROOT GENERALISED.** The
discriminator is **ANIMAL COUNT, not clip length** — a 12-animal root loses at 500 frames too,
because a carried prior there is often the wrong animal's pose. The loop **SATURATES rather than
diverging**, braked by the bounds mask at about one crop width. In 3D the per-frame triangulation
de-loops the feedback and the sign flips. `carry` also LOCKS the pose, so read `motion_ratio` beside
it. **Use `--anchor none` on a crowded 2D root or a long 2D clip.**

- **`carry` feeds back the ANCHOR-FREE estimate.** Under `gridresid_offset = "query"` the reported
  output *is* `query + R @ residual`, so writing it back closes a loop with gain; the triangulation
  is re-derived from each window's own pixels and breaks only the feedback path. Worth −0.50 mm over
  480 frames and a NULL over 120 — too few hand-offs to accumulate. **2D is bit-identical either
  way**, pinned as a test.
- **It is NOT the fix for the motion loss.** With ANY prior the output hangs off one anchor for the
  whole window, so only removing the prior or re-anchoring per frame can touch it. A TRAINING
  question, and open.
- **`--overlap`'s SIGN transfers across dimensionality and its OPTIMUM does not.** The optimum is
  SEAM COUNT against SEAM SIZE, both of which depend on motion and animal count. **Sweep per root.**
- **`--pose-nms` is the one identity lever that works.** Intersect two rows' boxes, take
  `min(kpts of A in B / |A|, kpts of B in A / |B|)`, drop the lower-scored row. **Deliberately not
  IoU.** **QUANTISED BY K**: at K = 4 there are five settings and `0.6` and `0.7` are byte-identical
  — **quote it as a COUNT ("3 of 4")**. Harmful on a root at ceiling; the discriminator is whether
  `fp_dup` is a live term.
- **It composes with `--overlap` by plain additivity** — it fires upstream of the window loop, so
  its rate is identical at every overlap. **"Arm A makes arm B viable" is an artefact whenever B's
  fire rate is unchanged.** **Check additivity before sweeping a grid**; it turns N×M into N+M.
- **`box_agree` is a 3D DIAGNOSTIC.** In 2D the pose is decoded inside its own crop, so the centroid
  is at most half a box side from the centre by construction. As a row gate it is **portable where
  `--vis-thresh` is not**. Not a default; the rate-matched control is mandatory.
- **A window that predicted nothing says why.** `run_group` writes an `outcome` per (animal, window)
  — `ok / no box / no camera / no points / crop failed / decode failed`. Five aborts wrote the same
  NaN before, so coverage could not be attributed to a cause; on a long 2D clip **100% of coverage
  loss is `no box`**. The FP term splits too: `fp_dup` is a second prediction on a claimed animal,
  `fp_none` landed on nothing, and they want opposite fixes.
- **`--det-score 0.99` is wrong for any detector whose objectness is not saturated**, and
  **saturation is a property of the RECIPE, not the dataset**. `infer.py` warns when the threshold is
  at or above the checkpoint's recorded median. Sweep per checkpoint: 0.50 maximises coverage, 0.97
  identity. `decode` and `eval_detector.py` stay at 0.05 — training and scoring paths.

**VIDEO IS READ THROUGH PyAV, NOT decord, AND THAT WAS A MEMORY FIX RATHER THAN A PREFERENCE.**
decord loads whole containers into memory (dmlc/decord#80; #197 is the same symptom, and it is
unmaintained since 0.6.0 in 2021). On sixteen 21 GB recordings that is 336 GB: a `--videos` run
peaked at **456 GB under `--max-ram 24`** and had to be killed at 9 GB free on a shared node, and
at a reader cache of 4 decord is **OOM-killed inside one round of 16 cameras under a 150 GB cap**
where PyAV sits **flat at 1.59 GB**. **THE SWAP IS OUTPUT-NEUTRAL AND THAT IS MEASURED** --
bit-identical to decord on every video root (johnson h264 3208x2200, 3dpop mpeg4 3840x2160,
calms21 mpeg4), sampled at BOTH ENDS of each container because the seek path is what differs, and
bit-identical through `read_frames` itself including clamp-padded REPEATS and OUT-OF-ORDER indices
(`tests/test_video.py`). So no recorded number moves. **`TAILCYCLENET_VIDEO_BACKEND=decord`
restores the old reader** and is kept for bisecting, because every recorded number was produced
with it. **OpenCV IS REFUSED DESPITE PASSING HERE**: `CAP_PROP_POS_FRAMES` is documented at 8 to
-3 frames off on MP4/AVC1 (opencv#9053), and an off-by-one frame is a different picture of a
moving animal, silently. PyAV seeks to the preceding KEYFRAME and decodes forward COUNTING
FRAMES, so the index arithmetic is ours. **REPORTS 38/39's DECODE NUMBERS HAVE BEEN RE-MEASURED ON PyAV; THE OLD ONES ARE VOID.** They were
void for two non-overlapping reasons -- most were taken on JPEG DIRECTORIES (four of six shipped
roots), and the rest were video but on decord, whose `num_threads=1` is a regime PyAV never runs
in. What replaces them, all on a 16-camera 3208x2200 h264 rig, decode-only, 3 replicates:

- **THE READER-CACHE CLIFF IS 5.1x AND IS A THRESHOLD, NOT A CURVE** (cache 4 -> 7.4 s, 8 -> 6.3,
  **16 -> 1.5**). The access pattern is a CYCLE, so the whole win arrives only when the last camera
  fits. Report 38's 2.4x was right in kind and understated.
- **THE PRICE OF A READER IS LINEAR, ~0.053 GB/megapixel** (0.418/0.383/0.376/0.373 GB per reader
  at 1/4/8/16), not decord's `0.035 * MP * n^2`. The whole rig costs **6 GB where the old law
  demanded 63**, and because that law grew the cache as `sqrt(budget)` it gave only **10 readers at
  `--max-ram 128`** -- the cliff was unreachable at ANY budget. Refit; `--max-ram 32` now holds
  this rig, and the warning says so, because **a linear price inverts and a quadratic one does
  not**.
- **CAMERA CONCURRENCY IS 2.8x FROM 1 -> 4 AND THEN SATURATES** (4 -> 16 buys 9%, inside the
  spread). `_CAM_DECODE = 4` is unchanged and now measured rather than inherited.
- **`thread_type = AUTO` IS WORTH 3.2x AND IS ALSO A MEMORY LEVER.** Per-container threading and
  cross-container concurrency SUBSTITUTE: AUTO reaches at `_CAM_DECODE` 4 what NONE needs 16 to
  approach, and each concurrent camera holds a whole window of FULL frames. **decord was the NONE
  row**, which is why report 38 saw 3.5x from concurrency and report 39 saw none from raising it.
- **DECODE'S SHARE IS NO LONGER A CONSTANT.** `FrameStore.decode_s` accumulates and the driver
  prints it per group beside the store hit rate. Reported as a MULTIPLE OF WALL, not a percentage:
  decodes overlap up to `cam_decode`, so the thread-sum exceeds elapsed time and the JPEG-era
  "84.8%" framing is unbounded above 100%. **A store hit rate near 0 on a multi-camera rig IS the
  cliff**, diagnosable without a stopwatch.

**STILL OPEN: `FRACTION_READERS` 0.25 vs `FRACTION_STORE` 0.65 IS NOW BACKWARDS ON EVIDENCE** --
readers buy 5.1x at the threshold, the store's extra windows buy 7% -- but raising the reader share
pushes a small budget below one window of store, turning a working run into a refusal. It needs the
store's floor re-measured first and is not an arithmetic change.

**HOST RAM IS A BUDGET, AND `--max-ram GB` IS A CEILING ON THE PROCESS, NOT AN ALLOWANCE FOR THE
BUFFERS.** **AND IT IS NOW CHECKED RATHER THAN MERELY DERIVED FROM**: every consumer sized itself
off the budget and nothing verified the TOTAL, which is how a run reached 456 GB with nothing in
its output saying so. `memory.check_peak` warns ONCE, naming the phase, at the block boundary
where `trim()` already runs -- diagnosis, not enforcement, and only against a STATED budget, since
an inferred one is what was lying around rather than a promise. The budget is also resolved ABOVE
both input branches: it sat below them, so `--max-ram` was not in effect during the `--videos`
probe at all. `memory.py` resolves one budget as the smallest of the cgroup limit (walked up EVERY
ancestor), LSF's `LSB_CG_MEMLIMIT`/`LSB_MEMLIMIT`, `MemAvailable` and `MemTotal` — never
`SC_PHYS_PAGES`, which is the MACHINE's memory and under a 16 GB cap still reported this host's
503 GB, a 20x error and exactly the case an LSF job dies in. It sizes the decord cache
(`FRACTION_READERS` 0.25), how many CAMERAS the detector decodes at once (`FRACTION_DETECT` 0.10),
and **how many windows a BLOCK holds** (`FRACTION_STORE` 0.65).

**`FRACTION_DETECT` NO LONGER OVERLAPS NOTHING IN TIME.** It was the loosest of the three because
detection finished before the pose loop began; under one pass the detector's letterbox buffers are
live WHILE the store holds the block, so its share came down. **`FRACTION_READERS` did NOT move,
and the obvious argument for moving it is wrong**: "a stored frame needs no reader" is true of the
number of READS and false of the number of open CONTAINERS — every block still touches every
camera. Dropping it to 0.05 takes johnson from 16 readers to 10, and 16 ran at 61 s against 11 at
149 s. **The 0.10/0.65 split is reasoned, not measured, under one pass.**

**EVERY KNOB IT SIZES IS OUTPUT-NEUTRAL, and that is what licenses sizing anything from free
memory** — a 288 GB budget and a 24 GB cgroup produce byte-identical output on both video roots,
and `tests/test_memory_budget.py` pins BLOCK SIZE as the newest instance of that rule. It
is why the budget sizes the CAMERA axis and never `detect_raw`'s `batch`: **`batch` is NOT inert**
(16 vs 3 moves boxes 0.204 px and scores 1.7e-03, because cuDNN picks algorithms per input shape),
yet `tests/test_detector.py` classifies it as `plumbing`, which holds ONLY because the value is
pinned at 16 and no flag reaches it. `eval_detector.py --batch-size` is therefore a live hazard:
change it and two runs are not the same boxes, with nothing in either output saying so.

**AN UNCONSTRAINED RSS PEAK IS RETAINED ALLOCATOR ARENA, NOT THE WORKING SET, so never quote one
as "how much this needs".** johnson at 120 frames peaks at 177 GB on an idle host and runs the same
work in 10.3 GB under a 24 GB cap. Score memory under a cap (`systemd-run --user --scope -p
MemoryMax=NG`), because the failure being prevented is an OOM kill: on that cap the pre-budget code
sat pinned at 24.0 GB, swapped 67 GB and hit the limit 92,787 times. `MALLOC_MMAP_THRESHOLD_` is
REFUTED as a fix — 10% of peak for 43% of wall clock.

**THE LONG CLIP IS FIXED, AND THE FIX IS THE BLOCK.** `run_blocks` runs a group a BLOCK of windows
at a time — sized so the block's frames fit `FRACTION_STORE` — and yields each block for the writer
to append, so no array is proportional to clip length. The whole-clip `np.full` allocations are
gone: 82 GB of result arrays at 720,000 frames became a bounded block plus a parquet file. `carried`
crosses a boundary for free (it is per-animal state), and the seam frames a block's last window
touches belong to the NEXT block, which is what makes block output byte-identical to whole-clip
output (eval rule 11: a frame belongs to the last window containing it).

**WHAT REMAINS PROPORTIONAL, AND IS NOT OURS:** the `--boxes` npz a caller supplies, and
`sess.preload()`'s dense LABEL arrays for the source session.

**ONE DECODE PER (CAMERA, SOURCE FRAME).** `infer/store.py` is the only decode site; the detector,
refine pass 1, refine pass 2 and the next window's overlap all read it. It was **seven** decodes:
three overlapping windows x two refine passes, plus `detect_raw`'s own pass over the group.

**AND IT DOES NOT DEGRADE.** A budget too small for ONE WINDOW of full frames is a `SystemExit`
naming the arithmetic and the `--max-ram` that would fit, not a silent 3x re-decode. There is
nowhere to degrade to: pass 2 crops from the same frames pass 1 did, so dropping them means either
decoding twice or re-cropping from the stored crop, and the second double-resamples. This is why
`[data].n_frames` is **12**: one window on a 16-camera 3208x2200 rig is 4.07 GB at 12 against
8.13 at 24. `[data].n_frames` and `[model].stride_length` are ONE quantity and `check_window_length`
refuses a config where they disagree.

**`soft_argmax_threshold = 60` IS LOAD-BEARING — do not widen it.** Swept against the labels,
10/20/30/45/**60**/120/∞ read 8.114/8.032/7.991/7.981/**7.976**/8.374/**11.032** mm: removing the
truncation costs +3.06 mm (38%). It suppresses spurious far modes rather than introducing a bias. A
plain attribute, not in any `state_dict`, so it is sweepable with no retrain — and it is settled.

**`grid_decode_space = "warped"` is the RIGHT value and the library's default is not.** `"head"`
averages convex-spaced bin centres directly and overshoots through the warp at large motion.
Reverting costs +0.19 mm at every quantile. Only measurable on a PROMPTED arm.

**`--refine`'s GAIN IS MAGNIFICATION, NOT COORDINATE FRAME — so PASS 1 ONLY HAS TO LOCALISE.**
Re-centring the crop at a fixed SIDE recovered 42% of the mean gain and 0.4% of pck@10 (a null).
**`--refine-px` is that corollary shipped**: a plateau from 96–192 and a **cliff at 64** in both
dimensions, with 96 px beating full-resolution refine outright in 2D, because a low-res pass 1
contracts its pose toward the crop centre and so yields a *tighter* pass-2 box. **No shipped
default** — the floor scales with `patch_size` and with how big the animal sits in the crop.

**`image_size` MEANS THREE UNRELATED THINGS AND ONLY ONE IS WRONG FOR A SMALLER INPUT**: a padding
target (`PadToSize`, disabled — or the crop sits in the corner of a zero canvas at 4x the error),
the 2D head's fixed output canvas (a WEIGHT SHAPE, so callers rescale), and **the pixel extent of
the input**. 0.3.5 splits the third out as `input_size=`, and `model.forward` passes the cameras'
own extent through. All three were silent library bugs; the two that matter were worth 45.2 mm on
the triangulation and exactly `image_size/px` on the gridresid residual.

**AND THE UNCORRECTED VERSION READS BETTER ON MEAN MPJPE, which is a box-size confound.** Skipping
the corrections splays the pass-1 pose into a crop **14–26% wider**, partially DISABLING refinement
— worth −6 mm on the mean while costing **+257 mm at p99**. **Score a resolution bug on the FORWARD
against its own full-resolution reference**, not on a downstream metric. Same shape as eval rule 4.

**Do NOT put the window union box through `crop_box_for_points`.** Measured worse (+3.06 mm,
−0.032 MOTA): the union of per-frame crop-rule boxes is already near-square, and squaring it again
grows the p90 box AREA by 82%. A detector box is already a crop-rule box, so the union already
satisfies the `min_crop_dim` floor.

**A GT crop is only an upper bound where the labels are dense.** A root labelling 2 of 4 points
builds its GT crop from one or two points and floors at 64 px — detector arms beat it there. Where
labels are complete the GT crop wins wide (+8.57 mm, +14.93 px). **The inversion is label sparsity
and nothing else.** The "GT crop" row means different things on different roots; say which.

`scripts/eval.py` is offline and model-free: a prediction (session dir or npz) + annotation set → MPJPE (paired
bootstrap), PCK, coverage, MOTA/miss/FP/idsw.

**A METRIC THAT FAILS AT A CHUNK BOUNDARY IS A METRIC BUG UNTIL PROVEN OTHERWISE.** `chunk_frames`
sliced the LABEL arrays only where `label.shape[1] == pred.shape[1]`, so a prediction that is a
`--max-frames` PREFIX of its group failed that test and each chunk got the whole group's labels —
and `score` truncates to the shorter, so chunk 0 was right and **every later chunk scored its own
frames against frames 0..n−1 again**. Coverage read 0.4656 against 0.9891 fixed; MPJPE 98.6 against
26.5. It survived because it looks exactly like a pipeline degrading over a clip, which this repo
has a documented lever for. Fixed. **Real degradation does not respect the scoring unit's edges.**

**THE BOOTSTRAP RESAMPLES GROUPS, AND A LONG CLIP IS ONE GROUP** — so rat-city's whole test split
returns `DEGENERATE` on every delta ever measured there, and the roots this repo most wants
long-clip numbers from have the fewest groups. `--chunk N` splits each group into N-frame scoring
units. Two things make it a resampling change and not a metric change, and the second is not
obvious: the chunks PARTITION the frames, and **the match radius is the whole group's** — sized per
chunk it swung 27.6 to 101.9 px across ten chunks of one clip, and chunks scored under different
radii are not exchangeable, which is the one thing a bootstrap needs. `--vs` chunks the second file
too, or the two sides key on different names. Chunks of one clip are more alike than independent
clips, so this is **within-clip** uncertainty and optimistic against the between-clip kind. Say
which one a number is. Multi-animal rows report **matched** MPJPE: row index is not identity once
boxes come from a detector, and scoring row-to-row measured 385 px on 30 px flies. `--vs` pairs two
files over the groups both scored and the points **both matched** — including coverage and the FP
split, because a delta over points only one arm attempted measures which arm declined more.

---

## The detector

```bash
pixi run python scripts/train_detector.py --config configs/detector.toml
pixi run python scripts/infer.py --run runs/<name> --data <dataset> --detector runs/det-<name>
```

**THE DETECTOR RECIPE LIVES IN A CONFIG FILE, NOT ON THE CLI.** `configs/detector.toml` ships every
default with its evidence; the only CLI knobs left are `--out`/`--iters`/`--device`. Per-root
recipes are key overrides in YOUR config, not a second shipped file. An unknown key in any block
RAISES rather than silently training at defaults, and the run folder records the effective
`config.toml` + `provenance.toml` like a pose run does. Checkpoints are byte-compatible.

**One detector per dataset**, and `input_wh` defaults to an aspect-matched size rather than a
square. This is not fussiness: rat-city's frames are 2.29:1, so a square 416 letterbox wastes 56% of
the canvas and delivers the median rat at 15.8 x 12.5 px — about 2 x 1.6 cells at stride 8 and
absent from strides 16 and 32, so two thirds of the FPN cannot represent it. The same detector on
square 1024x1024 fly frames reaches AP50 0.985 where rat-city sits near 0.50.

The regression target is `crop.crop_box_for_points`, i.e. the detector reproduces *the crop the pose
model was trained on*, not "a box around the animal". `tests/test_detector.py` asserts that against
the crop rule directly. `--boxes` is part of that rule and a mismatch across the two models is a
warning, not a failure — the best rat-city detector on record is `instances`-trained while every
rat-city pose run is keypoint-trained, so the mismatched pair is worth running.

**A DETECTOR ARM NEEDS A SAME-RECIPE REPLICATE, NOT JUST A PAIRED INTERVAL.** Four same-recipe runs
spread val r@.5 by **0.0122** and best-iteration by 8,000, and a replicate with NO lever moves
coverage +0.0575 and miss −0.0619, both SIG. **ONE arm against ONE baseline establishes nothing
here, on any column.** **AND A FIXED `--det-score` MEASURES CALIBRATION** whenever a lever moves the
objectness distribution — re-score every arm at a matched threshold; several "significant" results
evaporated to this. **`--det-cache` IS DELETED and this is what it cost**: detection and pose are
ONE pass over the video now, so there is no separable detection phase to store and an association
sweep costs a full decode pass instead of a CPU-minute. Two arms are no longer matched by
construction -- match them by `[provenance]`, which records every input the boxes depend on.

- **`--use-regions` is the one detector lever with a positive accuracy result** (MOTA +0.0489, via
  `fp_none`, at an idsw cost). Default-off. `--tile-wh` is what makes the mask viable — a hard mask
  on FULL-FRAME input is measured dead. Both default off and ORTHOGONAL, so the untiled unmasked
  path is byte-identical, asserted rather than assumed.
- **A TILE IS JUST A TRANSFORM**: a tile at source origin `(ox, oy)` at scale `s` is exactly the
  letterbox `(s, (-ox*s, -oy*s))`, so boxes, regions and the single `warpAffine` all tile by
  substituting it, and **inference is one whole-frame forward as before**.
- **`--tile-scale` IS A RESOLUTION KNOB** — certified *area* is scale-invariant because
  `CENTER_RADIUS` is 2.5 **cells**. **Ship tiles at scale 1.0, and DO NOT ALSO RESIZE THE TILE**:
  the invariant is the animal's size in INPUT pixels, and tiling-then-downscaling is the number-one
  reported failure.
- **A TILED CHECKPOINT'S `input_wh` IS ITS TILE SIZE, NOT ITS DEPLOYMENT INPUT SIZE** (gotcha 11's
  shape). `tile_scale` rides in the checkpoint, `load_detector` **raises** if a tiled one lacks it,
  and the input is derived **per camera** — one root ships 4696x2048 beside 4500x2050.
- **`--rotate-deg` is default-off and its 180-degree setting is REFUTED** on two roots and two label
  sources in one direction. The knob survives because `_rotated_rect_max_inscribed` is 90-degree
  PERIODIC, so a smaller amplitude retains no more area and is a different trade, not a safer one.
  **In-training val recall disagreed with the end-to-end result in both directions three times** —
  score a detector end to end.
- **`--link-boxes` is a per-frame Hungarian on CENTRE DISTANCE over the box side, gated at one
  side.** It was ungated IoU with a force-assign and all three parts were wrong: IoU ranks by shape
  agreement, which is not identity, and is exactly zero under fast motion where it cannot rank at
  all. The gate has 10-16x headroom (p90 centre displacement is 0.06-0.11 body lengths everywhere).
  An unmatched row stays EMPTY rather than taking a leftover, which is what produced a
  **1924x1924 union against a 244 px rat**. *Default-on, and its paired arm has never been run.* `max_age = 24` and `birth_age` are pinned
  constants unreachable from any CLI.

**EVERY SENTENCE IN THIS `--track` BLOCK IS A 3D MULTIVIEW STATEMENT.** `CrossViewTracker` is built
only when `track and C > 1`, and every 2D root is `C == 1`; `associate` never runs there either. **In
2D `associate_group` reduces to a truncation of the score-ordered `decode` survivors followed by
`link_rows`.** The code path does not execute — say "2D" or "3D" before any sentence about `--track`.

- **`--track` IS THE DEFAULT, and it is for LONG CLIPS, not for crowding.** Over 480 frames the
  memoryless pass grows +0.6 mm/window to 39.4 mm while the tracker holds 12-13 mm flat and the
  worst crop halves — **so the error growth over a clip is the CROP degrading, not the prompt.** On
  a 120-frame protocol nothing moves (6 windows); on crowded clips it reads −7.4 mm, but those were
  selected FOR crowding and stratifying by animal count shows **no dose-response**. What it buys is
  scale: **4.1 s/frame at C = 16** for the memoryless pass against ~0.6 ms. `associate` stays for
  BIRTHS. **`track` is UNCONDITIONAL in the cache stamp**, so caches written while it was off are
  REFUSED.
- **`--track` made every birth-time lever irrelevant.** Births are 0.041% of associations, so a rule
  firing at birth has a ceiling of hundredths of a percent against a ±0.023 MOTA seed floor. **That
  rate is one root's and is not portable. Measure the POPULATION a lever governs before its
  selectivity.**
- **`min_views = 2` was never a threshold — it is the algorithm.** Every instance is built from a
  cross-camera PAIR, so the check could not fire. `min_views = 1` is a different rule that emits
  leftover boxes as single-view instances; it halves the `no box` rate and moves no metric. Whether
  the pose model can *use* one is `[data].prob_2d_only`, **0** in the shipped configs.
- **`--max-animals` / `--det-top-k`: sweep per root, never reason.** "S = animal count + 2" was
  +0.112 MOTA on a 500-frame clip and **does not reproduce at 115x the length** — the sign REVERSES
  and the magnitude is 37x smaller, because the spare rows only FILL on the short clip and those
  fills were false positives. **Measure the drop rate before porting a row count.**

**DETECTION AND ASSOCIATION ARE STILL SPLIT, but they now run INSIDE the window loop.**
`detect_raw(frames=, read=)` detects a slice of the clip over frames the store already holds, and
`associate_group(state=)` / `link_rows(state=)` carry the tracker and the matcher across every call
boundary. Two rules make that safe and both are pinned by tests:

- **A DETECTION SLICE MUST START ON A GLOBAL `batch` BOUNDARY.** `_units` partitions on
  `range(0, T, batch)`, so a slice starting anywhere else forwards a short leading batch — a shape
  the whole-clip pass never produces, and cuDNN picks algorithms per shape. `detect_raw` asserts it.
  This is why the detection cursor is NOT the block cursor: blocks are sized by free memory, and a
  budget-derived batch would move the boxes.
- **ASSOCIATION STATE CROSSES THE BOUNDARY, or block size changes identity.** `state=None`
  reproduces the old single-call behaviour byte for byte, so every other caller is unaffected.

**Every detector box is bounded in `unletterbox_boxes`.** A side decodes as `exp(clamp(-6,6)) *
stride` — up to ~12,910 px — and IoU-only NMS cannot suppress it (its IoU with the box it swallows
is ~0). Clamped into the frame; a zero-area box comes back NaN, which every consumer reads as "no
box here".

**THE COST OF DETECTION IS THE VIDEO DECODE, NOT THE GPU** — 0.86 ms of forward per frame-camera
against a 44 ms 4K MPEG-4 decode, at 0-3% GPU. **Do not chase NVDEC**: at 3840x2160 MPEG-4 Part 2 is
32,400 macroblocks against NVDEC's 8,192 cap for that codec, so an H100 cannot decode these files at
all. The remaining lever is process count.

**THE BIGGEST LEVER IN THE REPO IS NOT A FLAG.** Pose on a GT crop is **19.3 px at coverage 1.000**
against **44.7 px at 0.566** through the detector, and on the long clip 100% of coverage loss is
`no box`. Every identity lever fought over ±0.02 MOTA while the box path costs 25 px and 43% of
coverage.

---

## Gotchas — every one of these has already cost someone a day

1. **T = 1 is not usable.** `encoder_decoder.py` computes `gT = T // tubelet_size` → 0, so the
   pos_embed is zero-length. **FIXED UPSTREAM in 0.3.5**: a clip shorter than one tubelet now
   raises. Never sample fewer than 2 frames; single-frame groups are padded at ingest.

   Relatedly, **a training window's T is derived, not configured.** `[data].n_frames` is only a
   ceiling: `_frames` sizes each train window to the labelled span it covers, rounded up to an even
   number (tubelet 2), floor 2. Val and test still enumerate fixed `n_frames` windows, or the metric
   would not be comparable across checkpoints. A train window may also be **strided** by
   `[data].frame_strides` (default `[1]`); the derived-T rule then runs on a lattice of spacing s,
   and T is capped by the room left on *that* lattice. `SmoothnessLoss` has no notion of dt —
   `torch.diff` is an UNDIVIDED difference — so its k-th difference grows like `s^k` (256x at
   k = 4, s = 4) and `_tune_smoothness` divides that out per batch. What it cannot fix, and what any
   `frame_strides` arm must declare: the HINGE is scale-invariant, so **striding genuinely loosens
   it** — the threshold tracks the trajectory's k-th derivative while the per-frame jitter it exists
   to catch is white and s-independent. And `SmoothnessLoss` raises below `order + 1` frames, so the
   order is clamped per batch; at T = 2 it degrades to a first difference rather than being disabled.
2. **`scene_features=` and `cube_scale=` were dropped from `TrackerEncoder.forward` in 0.3.x —
   RESTORED in 0.3.5.** `forward` takes `scene_features=` again (plus `input_size=`), and
   `share_scene` / `_forward_window` pass the stashed encode through that argument instead of
   routing around the public API. `cube_scale` remains a derived quantity, not an argument.
3. **`batch_size` is structurally 1 PER RANK.** `custom_collate` keeps only item 0's `cgroup` and
   the model takes one camera group per batch. The batch dimension this repo has is therefore the
   WORLD (`--devices N`, above): DDP averages one item per rank. Raising the per-rank batch is
   still a known ceiling, not a bug to fix casually.
4. **Keypoint identity ≠ array position.** The library drops keypoints with <2 valid frames, so `N`
   shrinks and positions stop matching ids. The loader must never filter. **This failure is
   invisible in the loss curve.**
5. **Keypoint ids ride in the occlusion channel**, and the stock `QueryEncoder` clamps
   `occlusion+1` into `[0,2]`. **0.3.5 makes the misuse loud** — values outside `{-1,0,1}` raise —
   but the rule stands: never share that tensor between the two consumers.
6. **`vis` and `vis_2d` are both-or-neither** — supplying one dies inside einops (0.3.5 adds the
   same guard on the loss side, so the silent discard is also loud now). And `get_eval_metrics`
   wants the trailing dim `(B,T,N,1)`.
7. **The crop rule is exact, not approximate.** posetail 0.3.5 exposes the same rule as a public
   `crop_box_for_points`, and the local one stays as a documented superset: it adds `pad` (0 for a
   caller holding an already-padded stored extent) and None on an all-NaN input, both load-bearing
   for the detector. A test asserts it is int32-exact against `crop_cgroup_to_points`. If that
   fails, **every detector number is invalid.**
8. **Moving-camera inference was not supported upstream.** 0.3.5's
   `load_camera_group_from_metadata` honours `moving_cams: true`; we build the camera group via
   `format_camera_group(..., moving_ext=)`. Only `TrackerEncoder` is moving-cam-safe —
   `ScorerEncoder` and `TrackerTapNext` shape-error on `(T,3)` centres. **No shipped root has a
   moving rig**, so this path is exercised only by the synthetic fixtures in the tests.
9. **allen-mouse's npz is column-sorted.** `pose3d.npz['pose']` is ordered by
   `sorted(f'{name}_{axis}')` while `keypoints` is name-sorted, which transposes all 8 `X` /
   `X-base` pairs — 16 of 47 keypoints. The converter applies the permutation once. Zipping `pose`
   against `keypoints` silently mislabels them and nothing downstream notices.
10. **Nothing in the parent process may decode video before the loader forks.** The forked workers
    deadlock in a futex while holding an open container: 0% GPU, ~0 worker CPU, no traceback, no
    timeout, forever. `scripts/train.py` materialises its fixed val windows through a one-worker
    `DataLoader`, byte-identical because `PoseDataset` seeds val items by index. **Any video-backed
    root can trigger it, including ones a converter makes later: a property of the pixels, not a
    list of names.**

    This is also why **the reader cache may never be sized by probing**: opening one `VideoReader`
    in the parent to measure anything IS the deadlock. `_reader_cache_size` derives it from a camera
    count, a frame size and `get_worker_info()` — all parsed-toml or in-process facts, now joined by
    `/proc` and `/sys` through `memory.py`, which opens no data path either. **DO NOT SET
    `TAILCYCLENET_READER_CACHE` BY HAND**: at best a no-op repeating the derived value, at worst it
    overrides the per-worker clamp that stands between a 16-camera rig and swapping. `--max-ram` is
    the knob a human reaches for. If a run needs a different cache the fix belongs in
    `_reader_cache_size`, where it is testable.

    **THE COST OF `n` OPEN READERS IS QUADRATIC IN `n`, AND THE CACHE IS NOT AN LRU.** Measured on
    johnson, trimming the arena after each so this is retained memory: 0.28 / 3.58 / 15.34 / 59.38
    GB at 1 / 4 / 8 / 16 readers — `~0.035/megapixel * n^2`. The control that makes it a fact about
    readers: the same 16 cameras with each reader RELEASED immediately costs **0.01 GB**. Reopening
    is only 41.5 ms, so a small cache is a good trade and the budget is better spent on the frame
    cache, which is linear and removes reader pressure outright. And the access pattern is a CYCLE
    (every window touches every camera in order), which is the one pattern LRU cannot serve: below
    the cycle length it takes **zero** hits, so eleven cached readers ran a clip in 149 s against
    two readers' 222 s and sixteen's 61 s. Eviction is RANDOM for that reason.
11. **A RUN FOLDER USED TO RECORD NO COMMIT, and one config key had a default it could not
    justify.** A 3D run trained under unconditional per-frame re-anchoring, finished nine hours
    before the commit that replaced it, and carried no `gridresid_offset` — so it loaded as the
    architecture it was not trained as, for weeks, silently. **+23.1 mm MPJPE [+22.3, +24.0] and
    −0.18 MOTA**, with the pose visibly lagging the animal until the bounds mask dropped the prior.
    Both halves are now closed: `save_run_meta` writes `provenance.toml`, and the key raises rather
    than defaulting. **The query-free siblings are mismatched harder**: the triangulation is
    substituted at every keypoint so the residual head's output is discarded outright. 2D is
    unaffected (`forward` returns before both offset paths). **Any run folder with no
    `gridresid_offset` key predates the check, and any 3D number published off one needs
    re-checking.**

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
6. **`err` is a mean over matched frames.** Decompose coverage before quoting any delta — a method
   that predicts fewer, easier frames looks better and is not. *(Under a Hungarian row matcher this
   INVERTS: deleting a good row leaves its GT to re-match against a worse one, so a rate-matched
   random control costs +0.42 px MPJPE rather than gaining. Rejection is not free either way.)*
7. **Anchor and prior inputs are GT-derived.** They must be gated off at eval by default. In
   posetail-pose their absence inflated *every* anchored number ever published there.
8. Only MOTA replicates across seeds, and only above a ±0.023 seed floor.
9. **The matcher itself can be fooled.** `match_instances`' cost is a mean over SHARED keypoints, so
   at `--min-match-kpts 0` a row sharing one keypoint is scored on that keypoint and can hijack a GT
   row. It is a FRACTION of K, not a count — K ranges 4 to 47 across roots. Default 0 keeps every
   published number reproducible. **But it is punitive at small K with sparse labels, so quote it
   with both**: rat-city's absolute MOTA falls 0.587 → 0.092 at 0.5 while the *delta* between two
   arms barely moves. **Use it for deltas, not absolutes, and never carry one value across roots.**
10. **Pairing is complete-case, which flatters the arm that failed more.** A group where either side
    is non-finite leaves the comparison, so `paired_bootstrap` returns `n_dropped` and `--vs` prints
    it. A delta over 9 of 17 groups is not a delta over 17.
11. **A PER-WINDOW STATISTIC MUST USE THE SEAM RULE'S OWN FRAME→WINDOW ASSIGNMENT.** A frame in an
    overlap belongs to the LAST window containing it. Slicing `[start : start + n_frames]` instead
    hands every window its neighbours' frames too, which SMOOTHS the per-window error: it reported
    ZERO burst windows where the correct binning finds up to seven, at identical indices. It reads
    as "the effect does not reproduce" rather than as a binning bug.
12. **A LABEL-FREE IDENTITY STATISTIC IS A SMOKE ALARM, NOT A METRIC, and a SELF-NORMALISED one
    ranks arms BACKWARDS** — correlation **−0.621** against true `idsw`, because a row swapping on
    every step has an enormous p99 so almost nothing exceeds it. Any per-unit-normalised
    discontinuity measure inherits this. The ABSOLUTE form passes the gate at **+0.982** — but
    excluding one broken arm the correlation is **−0.558** over five working arms. So it detects a
    box path that is BROKEN, with no labels and an enormous margin, and cannot rank two that both
    WORK. **And never score a cue on the statistic it optimises.**
13. **There is a temporal statistic**: `motion_ratio`, paired over the steps both arms attempted.
    Every other consistency number — jerk, bone CV — *rewards* a prediction that stopped moving,
    which is how the motion lock stayed invisible. **But `motion_ratio` near 1 is necessary and NOT
    sufficient.** Read it beside the error and identity columns, exactly as `err` must be read
    beside coverage. **It is also not single-signed** — it reads LOCK where the animal moves and
    JITTER where it does not — so set a bar as |ratio − 1| on the root the arm was trained on, and
    never pool the two.
14. **A rejection or gating lever is scored against a RATE-MATCHED RANDOM CONTROL, always.** Any
    rejection improves a mean over matched points; the control is what says whether the cue did.

**NAME THE SPLIT.** `allen-mouse-combined/val` is a seen animal on an unseen session, so it yields a
seen-animal number. A genuinely cross-animal number needs a split that holds an animal out. Scoring
a seen-animal arm against a cross-animal reference compares two axes and reads the ~0.9 mm
difference in axis as a result. The reproduction check against posetail-pose is a BAND, not
bit-equality: allen cross-animal MPJPE near 3.394 mm, human-vs-human baseline 2.208 mm. A large gap
is a port bug, and the first suspects are the allen column-sort permutation and the crop rule.

**`checkpoint_best` IS SELECTED ON `val`, THE PRIOR-FREE PASS**, not on `val_self`. Their argmins
differ by up to 13,600 iterations in practice, so on any run whose deployable regime is prompted,
`best` is not the checkpoint you want — and it penalises exactly the arm that trades query-free
accuracy for prompted accuracy. Score `best` AND `last`.

---

## The deleted levers, in one line each

Every one below was measured, refuted, and REMOVED. Listed so nobody re-proposes one.

**Inference:** `--axis-veto`, `--kpt-affinity`, `--kpt-centre`, `--axis-cost`, `--swap-repair`,
`--random-veto` (their rate-matched control), `--stitch`, `--dup-res-px`, `--prior-vis-thresh`,
`--seam blend`, `--moving-crop`.
**Detector training:** `--ignore-band`, `--ema-decay`, `--warmup-frac`, `--augment-photometric`,
`--identity` / `--id-weight` (and the YOLOX identity head behind them).
**Config:** `moving_crop`; `synth_motion_*` (a measured null); the dead loss weights `gamma`,
`feature_loss_weight`, `pixel_thresh`; the inert model keys `corr_radius`, `use_volume_embedding`,
`occlusion_embedding`, `mode_3d`, `cross_attn_dim`; `query_encoder = "pose"` and
`box_prompt = "term"` (config-unreachable; both raise by name).
**Model:** `PoseQueryEncoder`, `BoxTermEncoder`, and the moving-crop geometry in `crop.py`
(`moving_boxes`, `crop_to_points_{2d,3d}_moving`, `apply_crop_moving`, `static_offset`,
`with_static_offset`, `jitter_params`).
**Never built, measured and refuted:** an `--anchor self` fixed-point iteration, a coarse-to-fine
grid decode, "anchor refine", and `--anchor detector`.

**THE BODY-AXIS CUE IS REAL AND EVERY MECHANISM THAT SPENDS IT IS REFUTED — all three forms are now
tried.** A veto, a permutation and a Hungarian cost term were each built, measured on BOTH roots and
lost; the ranking a K = 4 body axis supplies is too noisy to spend. **Do not propose a fourth
without first raising the cue's quality.** Two more are dead on POPULATION rather than on the cue: a
learned identity head reaches 25.1% on contested decisions against 8.3% chance, but the centroid
already settles 98.8%, so a perfect head could flip **≤0.31%** of associations; and SLEAP's
`connect_single_track_breaks` fires **0 times in 57,593 consecutive steps**. **When refuting a
mechanism, refute it on the quantity a better model cannot change.**

**A PRIOR HELPS ONLY WHEN IT IS INDEPENDENT EVIDENCE, and that retires four proposals at once.**
Iterating `--anchor self` to a fixed point, a coarse-to-fine grid decode, and seeding pass 2 from
pass 1's own pose all feed the model a prior derived from the SAME pixels the forward is about to
see, so it carries nothing the forward does not have and its errors correlate with the ones it is
meant to fix. `carry` is worth 10.2 px precisely because its prior comes from DIFFERENT pixels. The
fourth, `--anchor detector`, satisfies independence and **fails on the POPULATION instead**: the
detector's keypoints are dense (96–100% of slots) and nowhere near accurate enough — 3x worse than
the answer they would seed.

**THE PRE-SCREEN THAT FALLS OUT IS WORTH MORE THAN THE LEVER: a prior must be more accurate than the
prediction it seeds, and `kpt_agree` reported exactly that distance.** (It is **no longer written**
— it was `(S,T,C,K)`, the largest pose-side array by a factor of K, for a diagnostic nothing gates
on. Recompute it from `instances.pq` and the detector if the pre-screen is wanted again.) It
predicts both refutations for ~1 min of CPU, before any inference — the same shape as the
`fp_dup`-must-be-live rule for `--pose-nms`. And `kpt_agree` is CIRCULAR under any detector-seeded
regime, so it must never be quoted as a win for one.

---

## What is owed

Open work is sized and prioritised in `dev/plans/owed.md`. The headline items:

- **`calms21/test` at `--chunk 500`**, the same treatment `3dpop/test` already got (58 groups
  untruncated give 145 units and an interval 2.8x tighter than the 120-frame protocol's).
  `eval.py` still resamples chunks rather than sessions, which is the honest interval on a
  58-clip split.
- **calms21's "identity collapse" is an ARM, not the root** — two of its clips reach MOTA 1.000 at
  coverage 1.0000 on current code. The collapsing arm differs in FIVE levers, so "a delta measured
  on top of that is uninformative" is true of that arm and must not be generalised. What is owed is
  the one-lever test.
- **MEASURED, LARGE, AND DELIBERATELY NOT SHIPPED: a WIDER pass-1 crop under `--refine`.** On 3dpop
  9 paired cells all agree in sign (MOTA +0.011 to +0.099, p99 better by 83–313 mm while p75 moves
  ≤ ±1.6) — it deletes catastrophic rows rather than improving good ones. **On calms21 it is a
  catastrophe** (+71.18 px, MOTA −1.0229). **The discriminator is one label-free ratio off the
  detection cache: pass-1 crop side ÷ distance to the nearest other animal.** Fitted on two roots;
  the bar for shipping is a second 3D root plus rat-city as the discriminator's test.
- **`--link-boxes` is default-on and never measured**, and `aug_rotation_deg = 45` is the one
  surviving unmeasured augmentation.
- **THE `--videos` PATH IS BUILT AND UNMEASURED ON A REAL RIG.** The smoke test and the wall-clock
  work are `dev/plans/infer_from_videos_and_calibration.md` §12-13: johnson `mouse_2_validate`,
  16 cameras of 3208x2200 raw mp4 against the cut-clip reference. **§13.8 step 1 -- the
  raw-symlink session at matched `--max-ram` -- is the number everything else hangs off and needs
  NO new code; do it before proposing any lever.** Two traps carried from there: **report 39's
  decode conclusions are JPEG conclusions** (four of six shipped roots are image directories, so
  the reader cache cannot fire on them) and **the 40 s cold open is first-touch-per-node, not a
  per-eviction price** (steady state 1.04 s). The reader-cache penalty measured on SHORT
  containers is 2.4x; the fix is a `--max-ram` large enough to hold the rig, **but never below
  ~12, which is an OOM kill with an empty log rather than a refusal**.
- **`render.py` on a video-sourced prediction.** Two pre-existing gaps: it still reads an npz
  rather than a prediction session, and it is not wired to `adopt.session_from_prediction`.
