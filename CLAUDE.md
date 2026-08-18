# tailcyclenet

Finetune a [posetail](https://pypi.org/project/posetail/) point tracker into an animal pose
estimator. Three settings, one model: **3D multiview**, **3D single-view**, **2D single-view**.

This is a clean rebuild of `../posetail-pose`. Everything that was an *experiment* there is
deleted here; what survives is the one architecture that won, the one crop rule, the one window
loop, and a data format that serves both hand annotation and bulk training.

**THIS FILE IS THE CURRENT STATE, NOT THE TRAIL.** Every measurement, refutation and retraction
lives in `dev/reports/`, indexed and dated. What is here is what a reader must know to avoid
breaking something: the defaults, the invariants, the gotchas, the eval rules — with the report
number beside anything that needs a *why*. **Do not grow this file with narrative**; that is what
put it at 108 KB, and `dev/reports/24_lever_audit_and_cleanup.md` is the cut that brought it back.

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
  checkpoints.py      run folders, save/load, warm start, config `extends`
  metrics.py          MPJPE / PCK / multi-instance matching / MOTA
  infer.py            THE inference path: one window loop
  detector/           YOLOX-Nano box predictor + cross-view association
    track.py            ONE cross-view target set. `--track`; replaces associate+link_rows
    identity.py         pose_nms, the one keypoint identity lever that works
scripts/              convert_v4.py  train.py  train_detector.py  infer.py  eval.py
configs/              base.toml + 2d.toml + 3d.toml. Hand-written, NOT one per experiment.
tests/
```

**Who may write where.** `dev/` and `scratch/` are yours, and both are gitignored — a conclusion
that should outlive the session belongs in this file or `docs/`. `docs/` is the human's — ask
first. `../posetail-pose` and `../../posetail/posetail-next` are read-only reference.

**Commit messages are ONE LINE.** No body, no bullet list, no `Co-Authored-By`, and no mention of
Claude or any other assistant. What a change means belongs in this file or in `dev/reports/`.

---

## Environment

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"
pixi run test
```

`posetail==0.3.5` comes from **PyPI**, pinned. A checkout of the same version lives at
`../posetail-next` for reading source; do not depend on it by path and do not modify it.
(posetail-pose's `path = "../posetail-next"` dep is how it ended up importing nothing after a
directory move.)

**0.3.5 LANDED EVERY WORKAROUND THIS REPO CARRIED — the monkeypatch layer is deleted, not
maintained** (report 33). Diffing installed 0.3.2 against the 0.3.5 checkout before the upgrade
showed all four `patches.py` entries verbatim upstream: `cube.project_cam` right-aligns a
per-frame `(T,2)` camera `offset` (the library assumed `offset` was static; `ext` already
carried the time axis), `undistort_points` accepts one via the same `_align_offset` helper,
`get_camera_scale` collapses it to frame 0 (a Jacobian, so exact), and `points_to_rays`
resolves it from the pinned `ext`. Also landed: `crop_box_for_points` as a public function,
`TrackerEncoder.forward` re-accepting `scene_features=` and gaining `input_size=`, the T=1
zero-token guard, the smoothness-order clamp, the occlusion-channel id guard, the `vis_true` /
`vis_true_cams` both-or-neither guard, moving-cams inference, and a checkpoint config-recording
refusal that is the upstream version of this repo's `gridresid_offset` no-default rule.
`tailcyclenet/__init__.py` no longer imports a patches module.

**`synth_motion_*` is deleted, not defaulted off** (measured NULL, report 30; the keys now fail
loudly as unknown). So is the moving-crop geometry it was the last consumer of: `crop.moving_boxes`,
`crop_to_points_{2d,3d}_moving`, `apply_crop_moving`, `static_offset`, `with_static_offset` and
`jitter_params` are gone from `crop.py`, and the two `static_offset` call sites use
`cam['offset']` / the raw cgroup directly — identical on every shipped root, since nothing
carries a `(T,2)` offset any more and 0.3.5 collapses one inside `get_camera_scale` anyway.
`format.py`'s moving-rig support (`moving_ext` / `extrinsics.pq` / `moving = true`) is KEPT —
that is the format spec's contract, and 0.3.5 now supports it end to end; no shipped root
exercises it.

The `LD_LIBRARY_PATH` prepend in `[tool.pixi.activation.env]` is load-bearing: the env ships
`libstdc++.so.6.0.35`, the host may ship 6.0.29 with no `CXXABI_1.3.15`, and without it
`import scipy.optimize` dies inside `_highspy` with an error that names only CXXABI.

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

- **Split is a directory, not a column.** None of the four shipped datasets split cleanly by
  session — rat-city's train/val/test are one recording — so a session-level field could not
  express them.
- **`mode` is per-session.** One `train/` may hold 2D and 3D sessions; both head-bank slots get
  gradient. Sessions need not agree on `names` either: `Dataset.names` is the **union** and
  `Registry.ids_for` remaps each session's own axis onto it **by name**, so a session may reorder
  or subset the keypoints. `allen-mouse-combined` is exactly this — 80 hand-annotated sessions in
  anatomical order beside a tracked one in name-sorted order.
- **A group is a contiguous clip**, `n_frames` unbounded. rat-city's 57,594 frames are ONE group
  whose `cam0` is a single symlink (~900 symlinks across all four datasets, not 550k).
  **A group's length is not its label count**: allen's annotated groups are 65 frames carrying
  exactly ONE labelled frame. This is why a train window's T is derived rather than configured
  (gotcha 1), and why an index entry is a poor sampling weight.
- **`status` is the visibility channel**, in both label tables, dictionary-encoded so
  `vis = codes == VISIBLE` is a vectorized int8 compare. But **coordinates live on `VISIBLE` *or*
  `PROJECTED`** (`fmt.POSITIONED`): `projected` is a position with no visibility claim, for a
  source whose annotators placed every keypoint in every view and never recorded occlusion
  (johnson-mouse: 1,235,334 "visible" vs 18 "not"). It must never reach a visibility target —
  `dataset.py` NaNs it out, and withholds `vis`/`vis_2d` entirely when a window has no assessment
  at all, because the noisy-OR would otherwise assert that nothing is reconstructible in 3D.
- **`keypoints.pq` `x,y` may be null on a `visible` row** iff a `points3d` row exists for the same
  key. That row is a *visibility observation* — allen-mouse ships a real per-camera `vis (S,T,K,7)`
  with no per-camera 2D, and this is how it is kept without inventing positions.

### Per-root traps

- **`rat-city-annotated`** (APT `.lbl`, `scripts/convert_apt_lbl.py`): 1,087 groups of 65 frames
  with ONE labelled frame at index 32, K = 4, names identical to the tracked root so both share
  registry ids and embedding rows. **Its per-camera visibility is the calms21 failure mode by
  decision** — APT records an occluded-but-placed point (`occ == 1`, 22.0% of slots) and the format
  has no such status, so those are written `visible` with their coordinates. Never use this root's
  vis head as a row gate without the rate-matched random control. **And its `p` column is
  x-block-then-y-block**: `labels{i}.p` is `(2K,n)` and decodes as `reshape(2,K,n)`. Both readings
  look plausible on a 4696x2048 frame; the interleaved one puts 28.25% of points outside the frame
  against 0.01%. Asserted on VALUES in `tests/test_convert_apt_lbl.py`, not on shapes. A labelled
  frame also labels only *some* of the rats in it — a median of 2 where the tracked root finds 11.
- **`regions.pq`** is APT's "Label Box": an area the annotator certified as completely labelled.
  Polarity is measured, not inferred from the name. **The file's ABSENCE is the claim of exhaustive
  labelling**, so the two `test/` sessions write an **empty** one rather than none — `None` and
  `(0,6)` are different answers. Membership is by **CENTROID, not overlap**. A region is kept only
  on the group whose own labelled frame it sits on (drops 140 of 636, printed by the converter).
- **`3dpop` ships an `instances.pq`** written from v3 (1,469,421 rows), which restores five pigeons
  v4's score-cleaning deleted outright — birds physically in frame with no row anywhere, so every
  prediction on one scored as a false positive. It flips `box_source = 'instances'` from inert to
  live, so **no new 3dpop number is comparable to reports 10-14 without `box_source = "keypoints"`**.
  And in 3D a `present` row has no box to localise against, so `_in_ignore` excuses every unmatched
  prediction on a frame carrying one — 24.6% of 3dpop's frames. **Quote `fp_ignored` beside MOTA
  or the number is unreadable.**
- **`rat-city` IS A SYMLINK to `rat-city-tracked`**, and the `*-combined` roots are symlink farms,
  so a fix in one place lands in up to three roots and `find` without `-L` undercounts them.
- **A root's file date cannot settle whether its tables are current — regenerate and diff.** Four
  of five roots carry a converter commit *after* the root was written, and one converter was
  created two hours after the data it produced. All were current; the one that was actually stale
  (`rat-city/val`, 169 of 745 rows with the wrong status) would have looked *fine* on dates.

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

**The long-clip protocol this repo keeps asking for already exists as data** (report 24 §1.3):
3dpop's 120-frame truncation discards 88% of its split, and calms21's 262,107 test frames are
entirely unused. Both at `--chunk 500` are chunks of MANY independent clips, so the within-clip
caveat is far weaker there than on rat-city's single group.

---

## Training

```bash
pixi run python scripts/train.py --config configs/3d.toml --data <root>   # or configs/2d.toml
```

**THE TWO CONFIGS DIFFER IN THREE KEYS AND THAT IS THE POINT.** `configs/base.toml` holds
everything shared; `2d.toml` and `3d.toml` `extends` it and add only what the setting genuinely
requires — `cams_to_sample`, `val_cams_to_sample`, `prob_2d_only`, all three camera-count questions
a one-camera root cannot ask. `2d.toml` adds NO `[data]` key at all. Resolution is **one level deep
by design** (`checkpoints.load_config`): a chain of overlays would put the difference back out of
sight, which is the thing the split exists to show.

`[data].path` is either a dataset root (has `train/`, optionally `val/` and `test/`) or a folder
whose children are dataset roots. In the second case keypoint names are prefixed with the dataset
folder name and one estimator trains across all of them; the keypoint embedding table is what makes
this work.

`n_keypoints` is **derived** from the registry, never configured. The registry is written to the
run folder as `keypoint_registry.toml` and read back at inference. A later run **appends** new
names so old ids — and the embedding rows behind them — survive warm start. `warm_start` copies the
base rows into the grown table keyed on the base registry's own length; if that does not match the
checkpoint's row count the copy is **refused rather than guessed**, because a mis-applied row copy
points each embedding row at a different body part. (`_filter_shape_mismatch` drops a tensor whose
shape changed *whole*, so before this, growing the registry reset every trained row to noise.)

A run folder also carries **`provenance.toml`** — the commit and a dirty flag at save time. A
config is not a provenance record, and gotcha 12 is what that cost.

**THE ENCODER UNFREEZES MID-RUN NOW, AND `[model].video_encoder_requires_grad` IS THE ONLY THING
THAT SAYS SO** (report 35). A bool means never / from step 0; an **int is the iteration to unfreeze
at**, with `video_encoder_finetune_last_n_layers` for how many trailing blocks. Shipped: **8 blocks
at iteration 10,000**, and **that is a default, not a result** — every number in this file was
measured with the encoder frozen for its whole life, so a run started after this is not comparable
to one before it; `video_encoder_requires_grad = false` restores the old arm exactly.
`[training].freeze_encoder` is DELETED and naming it raises: it re-froze the encoder after
`build_model`, so two sources of truth for one fact would have let the loop's silently win.

The model side is **entirely posetail 0.3.5's** — `TrackerEncoder.unfreeze_video_encoder` owns the
gate, the idempotence and the last-N selection. Three things this repo had to add, each a silent
failure otherwise. **A FROZEN PARAM IS IN NO PARAM GROUP, EVER**, so `build_muon`'s
`requires_grad` filter means a bare flag flip would hand the encoder gradients nothing steps —
`tailcyclenet/unfreeze.py` adds the newly-trainable tensors, routed by the same `route_param`
build time uses. **THEY MUST BE NEW GROUPS, NOT PRE-REGISTERED ONES**: both schedule-free impls
advance `k`/`weight_sum` on every `step()` regardless of whether a param has a grad, and the
averaging weight is `ckp1 ~ 1/k` — a group grad-less for 10,000 steps would fold its encoder into
the averaged iterate at 1/10000, so `model_state_eval` (what is deployed, and what `best` is
selected on) would hold a barely-moved encoder while `model_state` held a finetuned one. And the
**clip target is recomputed after a fire**, or the new tensors are unclipped and unchecked for
non-finiteness (report 34b). `encoder_lr_scale = 0.1` stops being inert at the unfreeze; the new
AdamW group warms from its own `k = 0`, which is why there is no separate warmup key.

**REPO-LOCAL DIVERGENCE:** upstream unfreezes `blocks[-N:]` plus an `encoder.norm` VJEPA 2.1 does
not have, leaving `norms_block` — applied at the hierarchical taps and feeding the decoder —
frozen behind a trainable block. This repo also unfreezes the norms whose layer is inside the
range. Unmeasured, and separately measurable by dropping it. **Resume replays the unfreeze onto a
frozen-layout optimizer** rather than building an unfrozen one, because `load_state_dict` matches
groups by position; a count mismatch raises naming both keys.

**MUON IS THE OPTIMIZER, AND AN ABSENT `[training.optimizer].optimizer` KEY MEANS MUON** (report
34, ported from posetail-next; **not** re-measured on this repo's roots, so it is the reference's
~0.05-kubric-per-iter result and not one of ours). SF-Muon runs `torch.optim.Muon` on the 2D
transformer matrices — `decoder.{cross_attns,mlps,camera_attns,temporal_attns}` + `scene_encoder.
kv_proj` + the encoder blocks when unfrozen — with AdamW-schedule-free on everything else
(`tailcyclenet/optim.py`). Three traps a reader must not rediscover: **`nn.Embedding` weights are
2D but excluded**, because each row is a keypoint identity and orthogonalising mixes body parts (the
`warm_start` row-copy hazard); **resuming an AdamW-SF run folder requires `optimizer =
"schedulefree"`**, or train.py refuses the mismatched checkpoint by name rather than `KeyError`-ing
mid-restart (the `gridresid_offset` rule, for optimizer state); and the **fresh 2D matrices go to
Muon at `kpt_lr`** — one lever off the reference, so if a Muon arm disappoints that group is the
first suspect to revert. `optimizer = "schedulefree"` restores the exact three-group AdamW-SF
optimizer. `muon_schedulefree = false` drops the averaged iterate, and `save_checkpoint` then
writes no `model_state_eval` (`has_averaged_iterate` is False).

**ONLY THE AdamW HALF IS GRADIENT-CLIPPED, AND A MUON RUN'S `grad_norm` IS A DIFFERENT QUANTITY**
(report 34b). Muon orthogonalizes its own gradients inside `step()`, so their raw norm is not a step
size — and the Muon-routed matrices carry **~95% of the global grad norm**, so the old shared
`clip_grad_norm_(model.parameters())` computed a ~0.05 coefficient and applied it to the fresh
heads/embeddings/norms too, throttling them **8-20x**. The clip now targets `opt.adamw_params`;
non-finiteness is still checked on BOTH halves. `train/grad_norm` is the clipped AdamW half and
`train/grad_norm_muon` the unclipped Muon half — the two are not comparable to each other, and only
the first is comparable to a schedulefree run. **A run started before this fix is not comparable to
one after it.** Raising `max_grad_norm` past 10 does NOT help (measured); the residual is that Muon
warms up ~2-3x slower on this Adam-pretrained checkpoint, which is the documented
Adam-pretrained/Muon-finetune mismatch and the reason the default is still inherited, not measured. No DDP: `batch_size` is 1 and there
is no world-size LR scaling, so the reference's `sqrt(world_size)` machinery is not ported.

### The architecture: two switches

```toml
[model]
query = "prior"           # per-keypoint prior + missing-query tokens
# query = "none"          # query-free: no prior at all
query_encoder = "wide"    # 512-dim, identity + time, + two query terms iff query = "prior"
# query_encoder = "pose"  # 256-dim, ten terms
```

They are **orthogonal**: `query` decides whether a prior is supplied, `query_encoder` decides which
module consumes it. `wide`'s two query terms (`qpos`, `patch`) **default to** `query`, because under
`query = "none"` the prior is never read and both terms would collapse to constant no-query tokens
feeding dead gate inputs. So `wide` + `none` is a 6-term encoder and **is** golden's `j3`.

Either term may be **overridden** — `query_pos_embedding = true, query_patch_embedding = false` is
the 7-term `j4_prior` recipe (3.317 mm allen cross-animal, the only non-anchor arm on record to beat
golden j3's 3.394). Two combinations stay unbuildable and `build_model` says which key is wrong.
They only apply to `wide`; naming one beside `pose` asserts rather than silently doing nothing —
which is how six posetail-pose configs declared an anchor, trained, and were reported as anchored
arms whose anchor was a literal no-op.

`query = "prior"` carries a per-keypoint prior plus `prompt_time`. Every position-derived fusion
term carries a **learned no-query token** where a keypoint has no prior, instead of a value computed
from the crop centre and presented as real. `prompt_dropout` is the fraction of **steps** that run
fully query-free; `prompt_noise_px` is the jitter on the priors it keeps — **in pixels**, converted
for 3D by `cube_scale`, the same normalisation the metric losses use. One scalar in each session's
own units could not work: `allen-mouse-combined` holds 63 px sessions beside 14 mm ones. Ship it
WITH the dropout — dropout alone is retired at +0.146 SIG worse on fly.

The decode is per-keypoint independent, but `scene_center`, `scene_radius` and `cube_scale` are
derived from the WHOLE `coords_q` set — which is why `prompt_dropout` is drawn per ITEM and not per
keypoint. Measured on allen: a per-keypoint draw at p = 0.5 puts `scene_center` 13.6 mm from its
deployment value (0 of 200 draws within 1 mm); per item it is exactly 0. **Per keypoint, the
geometry the model meets at deployment is never trained.**

**`prompt_offset_px` is a WHOLE-BODY offset and it is the corruption with the shape of a deployment
failure.** i.i.d. jitter averages to zero over the keypoint set, so it teaches the model to trust
the prior's *centroid* exactly — the one quantity a whole-body lag gets wrong. Per keypoint the two
are indistinguishable (0.7086 vs 0.7084 mm at sigma 8 px); they differ ONLY through the scalars
that read the whole prior set, where a whole-body offset moves the centroid by the full lag while
i.i.d. draws cancel to sigma/sqrt(K) — **6.6x apart at allen's K = 44**. Measured at 8.0 on allen
3D: `prior_self` −0.277 mm [−0.395, −0.171] paired, and the carried prior's distortion of the
motion cut 29x. **`base.toml` ships 5.0 on BOTH 2D and 3D as a consistency choice, not a measured
optimum** — the mechanism is 3D-derived, so report 17 §7 argues it is *unmeasurable* in 2D rather
than harmful there. Say so when reporting a 3D arm: it is no longer paired with the `offset8` result.

The instance-anchor machinery is **deleted**, not defaulted off — including `j7`, the best row on
record (3.140 allen cross-animal).

`query_patch_embedding` builds its `PatchProcessor` at the base checkpoint's `embed_dim` and
projects to the fusion width, so all ~5.4M of the pretrained patch CNN load by name. Building it at
512 instead would inherit 93k of 5.5M and silently retrain the rest from noise.

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
  for every keypoint (posetail-pose's `_reanchor_per_frame`, 2.07 → 1.37 mm within-session).

Pair them with `query`: `prior` → `"query"`, `none` → `"triangulated"`. **`query = "prior"` +
`"triangulated"` is REFUTED** at **+0.289 mm [+0.115, +0.473] worse**, with the prompt doing the
least accuracy work of any arm — per-frame re-anchoring frees the motion by leaving no static anchor
to distort, and costs the accuracy by leaving the prior nothing to contribute. The same property
both times. **You cannot get per-frame freedom and a load-bearing prior from this one key.**

**`gridresid_offset` HAS NO DEFAULT and an absent key raises.** The two values load the same
tensors, so a checkpoint trained under one and built under the other produces numbers instead of an
exception — measured at **+23.1 mm MPJPE and −0.18 MOTA**. Gotcha 12. `load_run(model_overrides=)`
/ `--gridresid-offset` state the value for a run folder written before the key existed; that is an
assertion about weights nobody recorded, and it is echoed as one.

The substitution uses the **detached** triangulation, which is also the loss gate: the direct term
is then constant at unprompted points, so `coords_loss_direct*` and `coords_softmax_3d` supervise
query points only without forking `TotalLoss`. In 3D single-view there is no triangulation, so the
substituted anchor is the conf-weighted **mean** of the back-projected rays — not the library's
`3d_pred_rays`, which is a weighted *sum* (`conf_pred_2d` is an unnormalised sigmoid and nothing
divides by it, so at one camera it lands about half way from the world origin to the animal).

**Turning off the 3D CE means NaN-ing `grid['anchor_local']`, never popping `grid`.**
`depth_softmax` — weight 1.5, the largest CE term, and query-*independent* — gates on the same
`'grid' in outputs`, and the depth Huber reads `f_eff` out of the same dict. Measured on one window:
popping took the depth CE from 6.24 to off and the depth Huber from 0.617 to 39.49, a 64x
renormalisation. A non-finite anchor is the library's own off switch.

### The losses: five keys fire in 2D and the rest are 3D-only

`TotalLoss.forward` branches on `coords_true.shape[-1]`. The `R == 2` block computes
`coords_loss_2d`, `coords_softmax_2d`, `smoothness_loss_2d` and the two smoothness shape keys;
everything else is `_nan()`'d and dropped.

**SO NOTHING SUPERVISES VISIBILITY ON A 2D-ONLY RUN.** Both vis terms live in the 3D `else`, and
`vis_loss_weight = 5.0` — the largest weight in the block — gets no gradient there. That, and not
"the converter wrote every point VISIBLE", is why `--vis-thresh` cannot work on a 2D root: **the
head it gates on was never trained** (report 24 §1.2).

`conf_loss_weight`, `conf_2d_loss_weight` and `coords_loss_rays_weight` ship at 0.0 as documented
off-switches for terms that do exist. `gamma`, `feature_loss_weight` and `pixel_thresh` were
DELETED: the first two gate on outputs `TrackerEncoder` never emits (`coords_pred_iters` survives
only as a comment in that file; `feature_planes_levels` does not appear at all), and the third feeds
only the two conf losses. `coords_loss_direct_weight` is applied **twice**, so its effective weight
is double what the config says.

### Synthetic motion: DELETED, measured NULL

`synth_motion_prob` / `_amp` / `_frames` (reports 22, 29, 30) are **gone** — removed with the
0.3.5 upgrade (report 33), not defaulted off, so an old config naming them fails loudly. What
they did, for the record: an annotated group carries ONE labelled frame in 65, so the derived-T
rule gives it T = 2 and the annotated half of the corpus trains NO temporal machinery. The keys
emitted a T-frame window instead and **slid the animal through a crop that stands still** — in 2D
the displacement was added to the coordinates, in 3D it moved the world point, sized through
`get_camera_scale` so one `amp` meant the same crop-width displacement on a px session and a mm
one. The static crop rule then bounded the SLID points, making the box the UNION over the window.

**MEASURED AND IT IS A NULL (report 30).** rat-city at matched 60k reads a carry cost of **+1.12
against its control's +1.56 with both MPJPE deltas n.s.** — the moving-camera first version's
+4.47 is gone and geometry genuinely was that failure's mechanism, but nothing replaces it with a
gain. allen **cannot measure the endpoint at all**: the carry cost is ±0.01 mm on *both* arms,
because in 3D the per-frame triangulation de-loops the feedback — and johnson's carry cost is
negative on both arms for the same reason, so **the endpoint was measurable on exactly one root in
this repo**. johnson was the one root with a positive signal (+1.41 to +3.44 mm, 4 of 4 cells SIG)
and **94% of that is two chunks whose mean is 112 mm against a median of 4** — both arms break on
the same chunks, so the defensible size was the **~3% in the working regime**, on 6 of 6 chunks.
The live suspect was the background-slides-with-the-animal ceiling, which is a compositing or
pseudo-label fix and not a config key. The `best`/`last` iteration trap (report 22 §8d — an
under-trained `best` reading 10x the matched cell, sign flipped) is a general eval rule and
survives in this file under "checkpoint_best IS SELECTED ON val".

Still deleted, deliberately: `kpt_table_mlp`, the crowd head, distractor crops, `crop_side_mode`,
`curriculum`, `moving_crop`, and — with the 0.3.5 upgrade — `synth_motion_*` and the
moving-crop geometry it was the last consumer of (report 33). CLAUDE.md used to claim wide beat
pose "3.395 vs 4.021 mm" — that
is a **two-lever** comparison; the one-key figure is **3.535 vs 4.021**, unpaired, one seed, no CI,
never re-run.

---

## Inference: one entry point

```bash
pixi run python scripts/infer.py --data <dataset|session|video> --run runs/<x>/ --out pred.npz
```

There is **one** window loop. posetail-pose had three and all three got it wrong differently. Box
sources are the annotation set, a detections npz, or a per-dataset detector. Prompt regimes: `none`
(query-free), `carry` (previous window's own prediction — what deployment does, requires
`overlap >= 1`), `self` (two passes), `labels` (ORACLE, gated off by default, because in the project
this descends from, ungated GT-derived priors inflated *every* anchored number ever published).
Rendering is a flag, not a separate script.

### The measured defaults

The 3D column is 3dpop, **16 sessions / 47 `--chunk 500` units**, paired, one lever off
`--anchor carry --overlap 8`; the 2D column names its root because **the two 2D roots disagree**
(rat-city 500 f / 5 units; calms21 6 sessions x 2000 f / 24 units). Report 24 §10.

| flag | default | 2D | 3D | report |
|---|---|---|---|---|
| `--anchor` | `carry` | **ROOT-CONDITIONAL**: rat-city +5.04 px SIG worse, calms21 MOTA +0.082 SIG better | mean NULL; carry wins the bulk | 24 §10 |
| `--overlap` | **8** | 12 best, 8 within noise (+0.79 n.s.) | −0.626 mm vs 4 | 24 §10, 13 |
| `--refine` | **on in 3D, off in 2D** | accuracy yes, identity no | **−0.962 mm [−2.104, −0.216] SIG** | 24 §10 |
| `--crop-source` | `boxes` | **`boxes`** | **NULL** (+0.181 [−0.182, +0.546]) | 24 §10 |
| `--det-score` | 0.99 | **per-checkpoint** | per-checkpoint | 21 §0b |
| `--track` | on | **INERT** (`C > 1` gate) | on | 20 §0a |
| `--carry-source` | `triangulate` | **INERT** (bit-identical) | on | 13 RC1 |
| `--pose-nms` | off | +0.0223 MOTA on rat-city, **harmful on calms21** | untested | 21 §9k |
| `--vis-thresh` | off | **cannot work** (untrained head) | +0.049 on 3dpop | 24 §1.2 |
| `--refine-px` | none | **SWEEP IT**: 96 is best on calms21 and **+15.8 px / MOTA −0.238 SIG on rat-city** | 192 ≈ 256; 96–128 trade mm for MOTA. **INTERACTS WITH `--crop-inflate`**: report 24 §9m's wide pass-1 crop is harmful at 96, so a box model's default 128 is the nearest measured neighbour, not itself swept at 1.5× inflation | 24 §9h, §9j, §10, §9m |
| `--box-prompt` | `auto` | **LIVE, 2D single-camera** (report 27) | **LIVE, PER CAMERA** (report 33 — this used to be skipped on every 3D/multi-camera window and ran box-withheld silently) | 27, 33 |
| `--crop-inflate` | 1.5 iff box model | wide pass-1 + tight pass-2, report 27's mechanism | measured on 3dpop as a secondary axis alongside the box (report 33) | 27, 33 |
| `--min-views`, `--max-move`, `--min-box-frames` | 2 / 1.0 / 1 | clean nulls or no gain available | | |
| `--prefetch-windows` | **1** | bit-exact at any value (report 31) | bit-exact at any value | 31 |

**`--prefetch-windows` IS A PURE PERFORMANCE LEVER, NOT AN ACCURACY ONE** (report 31). The pose
loop was decode-bound with the GPU idle for the whole decode of every window; this overlaps
window `wi+1`'s decode with window `wi`'s forward, on a background thread that never reads or
writes `carried` (`--anchor carry`'s state), so the prediction is bit-identical at any value --
measured on real 3dpop and johnson-mouse sessions, both anchors, `--refine` on and off. **On a
video root it cannot make the pipeline GPU-bound** — 3dpop's decode is ~44 ms/frame-camera against
a ~0.9 ms forward and no CUDA video decoder in this env can touch these files (report 14 §4,
reconfirmed) — it only removes the GPU-idle stall, worth 1.5-1.7x wall clock on 3dpop (4 cam) and
johnson-mouse (16 cam, jpeg). `--prefetch-windows 0` reproduces today's exact serial loop and
memory profile.

**`--anchor` IN 2D IS ROOT-CONDITIONAL, AND "HARMFUL IN 2D" WAS ONE ROOT GENERALISED** (report 24
§10.2). On **calms21**, 6 sessions x 2000 frames, `carry` is significantly BETTER: **MOTA +0.0821
[+0.0320, +0.1240]**, idsw −0.0024 SIG, err p99 −40.7 SIG, MPJPE −3.60 n.s. On **rat-city** it is
significantly worse: +5.04 px [+2.10, +7.77] at 500 frames and +52.42 px at 57,594. **The
discriminator is ANIMAL COUNT, not clip length** — rat-city loses at 500 frames too, and it has
12 rats against calms21's 2, so a carried prior there is often the wrong animal's pose and a wrong
prior is worse than none. `carry` also LOCKS the pose (motion_ratio −0.207 SIG on calms21), so read
`motion_ratio` beside it. The default stays `carry`; use `--anchor none` on a crowded 2D root.

The rat-city mechanism, unchanged and still the reason to distrust `carry` on a long 2D clip:
**+52.42 px MPJPE [+50.77, +54.05] and −0.4563 MOTA** on rat-city's
57,594-frame clip, and −9.144 px even on a 500-frame one. **The loop SATURATES rather than
diverging** — it starts at the prior-free value, climbs ~5x over roughly the first 300 windows, then
plateaus for the remaining 2,500 (slope −0.0032 px/window, zero burst windows). An unbounded loop is
a stability bug that worsens indefinitely; a saturating one has a fixed cost you measure once. The
brake is the bounds mask, which fires at about one crop width. In 3D the per-frame triangulation
de-loops the feedback and the sign flips. **Use `--anchor none` for a long 2D deployment run.**
`idsw` is the one column `carry` wins, which is a locked pose being a consistent one.

**`--det-score 0.99` IS WRONG FOR ANY DETECTOR WHOSE OBJECTNESS IS NOT SATURATED.** 0.99 was chosen
against detectors with 98.5% of boxes at exactly 1.0; the tiled/masked generation reads q01
0.45–0.84 and loses two thirds of its detections to the same number — coverage **0.703 against 0.986
at 0.50**, MOTA 0.622 against **0.723 at 0.97**. **Saturation is a property of the RECIPE, not the
dataset** (`--ignore-band` re-saturated it), so no constant is right for both generations.
`train_detector.py` records the val objectness quantiles in the checkpoint and `infer.py` **warns
when the chosen threshold is at or above the median**. Sweep per checkpoint: 0.50 maximises
coverage, 0.97 maximises identity. `decode` and `eval_detector.py` stay at 0.05 — they are the
training and detector-scoring paths.

**`carry` feeds back the ANCHOR-FREE estimate** (`--carry-source`, default `triangulate`). Under
`gridresid_offset = "query"` the reported output *is* `query + R @ residual`, so writing it back
closes a loop with gain: whatever the prior was wrong by is re-added and only the residual carries
new information. `3d_pred_triangulate` is re-derived from each window's own pixels and supervised on
every keypoint, so carrying it changes no reported output and breaks only the feedback path. Worth
−0.50 mm on 3dpop's two 480-frame hard clips — **and a NULL on the 58-group 120-frame protocol**,
because 120 frames is five hand-offs and too few for a loop effect to accumulate. Quote −0.50 only
with "2 groups, 480 frames" attached. **2D is bit-identical either way** and pinned as a test.

**IT IS NOT THE FIX FOR THE MOTION LOSS.** `--anchor carry` loses 23-39% of johnson-mouse's motion,
and every consistency statistic *rewarded* that (a locked pose is a smooth one — best jerk 0.392 vs
0.507, best bone CV 0.148 vs 0.251). Switching what is carried recovers only 0.5-9.1% of the path.
The cause is query-anchoring itself: with ANY prior the output is `prior + R @ residual` off one
anchor for the whole window, so all its temporal variation comes from the residual head. Only
removing the prior or re-anchoring per frame can touch it, which makes it a TRAINING question and an
open one.

**`--overlap`'s SIGN transfers across dimensionality and its OPTIMUM does not.** 3dpop has an
interior optimum at 8 (−0.355 mm [−0.751, −0.063] SIG; overlap 12 is no better than 4, and overlap 2
costs +0.460 mm and −0.011 MOTA). 2D is still improving at 12 (−0.408 px against overlap 4, where 8
buys −0.235), at a small monotone identity cost. The optimum is the SEAM COUNT against the SEAM
SIZE, and both terms depend on the clip's motion and animal count. **Sweep it per root.** Overlap 2
and 4 give the SAME window count, so overlap 2's cost cannot be a context effect — only the carried
frame index and the left-context differ, at identical compute.

**`--pose-nms` IS THE ONE IDENTITY LEVER THAT WORKS.** maDLC's `Assembly.intersection_with`:
intersect two rows' boxes, take `min(kpts of A in B / |A|, kpts of B in A / |B|)`, drop the
lower-scored row above the threshold. **Deliberately not IoU**, for the reason `--link-boxes`
abandoned it. On rat-city MOTA **+0.0223 [+0.0179, +0.0267]** at 3-of-4, beating its rate-matched
random control by **+0.058** — at an identical rejection budget it costs 0.24 points of coverage
against random's 4.08 (17x) and removes 3.7x more duplicates. **QUANTISED BY K**: the overlap can
only take `{0,.25,.5,.75,1}` at K = 4, so the flag has five settings and `0.6` and `0.7` are
byte-identical. **Quote it as a COUNT ("3 of 4").** Harmful on calms21, which is at ceiling —
root-conditional and default-off. The discriminator is whether `fp_dup` is a live term to attack.

**`--pose-nms` COMPOSES WITH `--overlap` BY PLAIN ADDITIVITY.** It runs per frame inside
`associate_group`, upstream of the window loop, so its fire rate is **byte-identical at overlap 4, 8
and 12**. Summing the one-lever deltas predicts the measured pair on every column, several exact to
four decimals. Two consequences: the pair needs **no joint sweep**, and **"arm A makes arm B viable"
is an artefact whenever B's fire rate is unchanged** — 2-of-4's MPJPE cost appeared to go from SIG
to null under overlap 8 while its own cost never moved; a constant was added and the SUM crossed
zero. **Check additivity before sweeping a grid**; it turns an N×M grid into N+M.

**`box_agree`** (S,T,C) is the predicted centroid's distance from its crop-box centre, in box sides,
reprojected in 3D. The pipeline held two independent statements about where an animal is — the box
and the pose — and nothing compared them. **It is a 3D DIAGNOSTIC**: in 2D the pose is decoded
inside its own crop, so the centroid is at most about half a box side from the centre by
construction (every 2D arm reads p99 0.31-0.56, max 0.80). In 3D the pose is triangulated and its
reprojection into any one camera can land anywhere. As a row gate it works and is **portable where
`--vis-thresh` is not** — no learned head, no per-dataset threshold — worth −0.29 mm and +0.007
MOTA at 1.24% of rows on 3dpop, beating its rate-matched control at every threshold tried. Neither
is a default and the rate-matched control is mandatory for both.

**A window that predicted nothing says why.** `run_group` writes an `outcome` per (animal, window) —
`ok / no box / no camera / no points / crop failed / decode failed` — and the crop box it used. Five
aborts wrote the same NaN before, so a coverage number could not be attributed to a cause. On the
long 2D clip **100% of coverage loss is `no box`** (24.89%, 0.0000 for every other outcome), which
is what makes the crop path the biggest lever in the repo. The FP term is split too: `fp_dup` is a
second prediction on a claimed animal and `fp_none` landed on nothing; they want opposite fixes. On
3dpop `fp_none` is ~10x `fp_dup` (4 cameras, low overlap) and on calms21 90% is `fp_dup` (two mice,
heavy overlap) — **opposite conclusions from one undifferentiated `fp_rate`.**

**`soft_argmax_threshold = 60` IS LOAD-BEARING — do not widen it.** It masks the softmax to
`|argmax - index| <= 60` of the shipped 1024 bins. It reads like a bias that pulls extremities
inward, and it does move them — but swept against the labels it is the OPTIMUM:
10/20/30/45/**60**/120/∞ read 8.114/8.032/7.991/7.981/**7.976**/8.374/**11.032** mm. Removing the
truncation costs +3.06 mm (38%): it suppresses spurious far modes rather than introducing a bias,
and 256 and ∞ are identical so the posterior never spreads further. A plain attribute, not in any
`state_dict`, so it is sweepable with no retrain — which is how this was settled, and it is settled.

**`grid_decode_space = "warped"` is the RIGHT value and the library's default is not.** `"head"`
averages the convex-spaced bin centres directly and overshoots through the warp at large motion;
`"warped"` averages in the uniform space first and is overshoot-free. Reverting costs +0.19 mm
(6.4%) at every quantile. Only measurable on a PROMPTED arm.

**`--refine`'s GAIN IS MAGNIFICATION, NOT COORDINATE FRAME — so PASS 1 ONLY HAS TO LOCALISE.** An
ablation that re-centred the crop while holding its SIDE fixed recovered 42% of the mean gain, 3% of
the median and 0.4% of pck@10 (a null). Pass 2's advantage is ~1.5x more pixels per animal.
**`--refine-px` is that corollary shipped**: calms21 2D reads a plateau from 96–192, a knee at 80 and
a **cliff at 64** (3.2x worse than not refining), with **96 px beating full-resolution refine
outright** — 6.651 against 6.765 — because a low-res pass 1 contracts its pose slightly toward the
crop centre and so yields a *tighter* pass-2 box. 3dpop 3D: 192 against 256 is a **NULL** (+0.062 mm
paired) at a quarter of the pass-1 pixels; 96–128 give back ~1.7 mm but recover +0.021 coverage and
**+0.03 MOTA**; 64 is the cliff there too. **No shipped default** — the floor scales with
`patch_size` and with how big the animal sits in the crop.

**`image_size` MEANS THREE UNRELATED THINGS AND ONLY ONE IS WRONG FOR A SMALLER INPUT.** A padding
target (`PadToSize` — disabled, or the crop sits in the corner of a zero canvas at 4x the error for
no speedup), the 2D head's fixed output canvas (a WEIGHT SHAPE; the head reports at normalised
position × `image_size` whatever the input size, so callers rescale), and **the pixel extent of the
input**. posetail 0.3.5 splits the third out as `input_size=` on `TrackerEncoder.forward`, and
`model.forward` passes the cameras' own extent (`px = max(c['size'])`) through, so the pad target,
the 2D-head rescale and the gauge corrections all land inside the forward at every site that read
`image_size` that way (report 26 §5b/5c/5d). The 2D post-hoc rescale in `model.forward` is gone
because the library's `2d_pred` is already in input pixels; `run_group`'s external `scales` division
is unchanged. All three were library bugs, all silent, and the two that matter were worth 45.2 mm
on the triangulation and exactly `image_size/px` on the gridresid residual.

**AND THE UNCORRECTED VERSION READS BETTER ON MEAN MPJPE, which is a box-size confound.** Skipping
the corrections splays the pass-1 pose, which squares up into a crop **14–26% wider** and so
partially DISABLES refinement — worth −6 mm on 3dpop's mean while costing **+257 mm at p99** and
nothing at p75/p90. **Score a resolution bug on the FORWARD against its own full-resolution
reference**, not on a downstream metric. Same shape as eval rule 4.

**Do NOT put the window union box through `crop_box_for_points`.** 08 §1.3 asks for it on gotcha-8
grounds and it is measured worse: 3dpop +3.06 mm and −0.032 MOTA, rat-city −0.040 MOTA. The union of
per-frame crop-rule boxes is already near-square (aspect p50 1.03-1.15) and squaring it again grows
the p90 box AREA by 82%. A detector box is already a crop-rule box, so the union already satisfies
the `min_crop_dim` floor.

**A GT crop is only an upper bound where the labels are dense.** rat-city labels 2.02 of its 4
points per animal-frame, so its GT crop is built from one or two points and floors at 64 px — the
detector arms beat it on MPJPE there. Everywhere labels are complete the GT crop wins wide:
+8.57 mm [+4.08, +14.02] on 3dpop and +14.93 px on calms21 — and calms21 is the control that settles
it, being 2D, multi-animal and *heavier* overlap than rat-city. **The inversion is label sparsity
and nothing else.** The "GT crop" row means different things on different roots; say which.

`scripts/eval.py` is offline and model-free: prediction npz + annotation set → MPJPE (paired
bootstrap), PCK, coverage, MOTA/miss/FP/idsw.

**A METRIC THAT FAILS AT A CHUNK BOUNDARY IS A METRIC BUG UNTIL PROVEN OTHERWISE.** `chunk_frames`
sliced the LABEL arrays only where `label.shape[1] == pred.shape[1]`, so a prediction that is a
`--max-frames` PREFIX of its group failed that test on every field and each chunk got the whole
group's labels — and `score` truncates to the shorter, so chunk 0 was right and **every later chunk
scored its own frames against frames 0..n−1 again**. Measured on 6 calms21 sessions whose
predictions are good to a median 8-11 px in EVERY chunk: coverage **0.4656 against 0.9891** fixed,
MPJPE **98.6 against 26.5**, and per-chunk MOTA 0.76-0.95 on chunk 0 beside **−0.36 to −1.00 on
chunks 1-3 of all six**. It survived because it looks exactly like a pipeline degrading over a clip,
which this repo has a documented lever for. Fixed; report 24 §10.3. Real degradation does not respect
the scoring unit's edges.

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
# template -- set [data].path and [training].out. `extends` works one level deep.
pixi run python scripts/infer.py --run runs/w9 --data <dataset> --detector runs/det-<name> ...
```

**THE DETECTOR RECIPE LIVES IN A CONFIG FILE, NOT ON THE CLI.** `configs/detector.toml` ships
every default (including `augment`/`augment_strong`/`rotate-deg`/`yolox` with the same evidence
labels the old flag help carried); the only CLI knobs left are `--out`/`--iters`/`--device`
overrides. Per-root recipes (report 32 §2.2) are key overrides in YOUR config, not a second
shipped file. An unknown key in any block RAISES rather than silently training at defaults, and
the run folder records the effective `config.toml` + `provenance.toml` like a pose run does.
Checkpoints are byte-compatible: `load_detector`, `eval_detector.py` and `infer.py` are untouched.

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
coverage +0.0575, miss −0.0619 and `kpt_agree` +0.1124, all SIG. **ONE arm against ONE baseline
establishes nothing here, on any column.** Only effects larger than a whole seed survive. **AND A
FIXED `--det-score` MEASURES CALIBRATION** whenever a lever moves the objectness distribution, so
every arm must be re-scored at a matched threshold; several "significant" results have evaporated to
this. Inference arms off a shared `--det-cache` are subject to neither.

**`--use-regions` is the one detector lever with a positive accuracy result**: MOTA **+0.0489
[+0.0079, +0.1010]**, entirely via `fp_none` −0.0576, at an idsw +0.0074 cost. Default-off.
`--tile-wh` is what makes the mask viable — a hard mask on FULL-FRAME input is measured dead (a 69%
positive rate among certified anchors against 0.68% unmasked, with 17% of frames carrying no
certified negative at all). Both default off and they are ORTHOGONAL, so the untiled unmasked path
is byte-identical — asserted, not assumed. **A TILE IS JUST A TRANSFORM**: a tile at source origin
`(ox, oy)` at scale `s` is exactly the letterbox `(s, (-ox*s, -oy*s))`, so boxes, regions and the
single `warpAffine` all tile by substituting it, and **inference is one whole-frame forward as
before**. **`--tile-scale` IS A RESOLUTION KNOB**: positive rate among supervised anchors is 48.3%
at 0.25 and 5.2% at 1.0 while certified *area* is scale-invariant, because `CENTER_RADIUS` is 2.5
**cells**. **Ship tiles at scale 1.0, and DO NOT ALSO RESIZE THE TILE** — the invariant is the
animal's size in INPUT pixels, and tiling-then-downscaling is the number-one reported failure.

**A TILED CHECKPOINT'S `input_wh` IS ITS TILE SIZE, NOT ITS DEPLOYMENT INPUT SIZE.** Gotcha 12's
shape. Deployment letterboxes the whole frame at the same scale, so `tile_scale` rides in the
checkpoint, `load_detector` **raises** if a tiled checkpoint lacks it, and the input is derived **per
camera** (rat-city-annotated ships 4696x2048 beside 4500x2050).

**`--rotate-deg` ships default-off and its 180-degree setting is REFUTED** — +3.33 px on hand labels
and +1.28 px on the tracker clip, idsw ×1.7, err p99 +29.9, two roots and two label sources in one
direction. The knob survives because that refutes one SETTING on one root, and because
`_rotated_rect_max_inscribed` is 90-degree PERIODIC, so a smaller amplitude retains no more area and
is a different trade rather than obviously a safer one. **In-training val recall disagreed with the
end-to-end result in both directions three times**, so score a detector end to end.

**`--link-boxes` is a per-frame Hungarian on CENTRE DISTANCE over the box side, gated at one side.**
It used to be ungated IoU with a force-assign and all three parts were wrong: IoU ranks by shape
agreement, which is not identity — replaying calms21 frame 301→302, IoU scored the WRONG mouse at
0.512 against the right one's 0.233, and the pose error jumped from 4-10 px to 60-82 px — and IoU is
exactly zero under fast motion, where it cannot rank at all. The gate has 10-16x headroom over real
motion (p90 centre displacement is 0.06-0.11 body lengths on every multi-animal root). An unmatched
row now stays EMPTY instead of taking an arbitrary leftover: that was rat-city row 9, whose window
union came out **1924x1924 against a 244 px rat**. **`LINK_REV` is in the `--det-cache` stamp.**
*It is default-on and its paired arm has never been run.*

**EVERY SENTENCE IN THIS `--track` BLOCK IS A 3D MULTIVIEW STATEMENT.** `--track` is structurally
inert on every 2D root — `detector/__init__.py` builds `CrossViewTracker` only when `track and
C > 1`, and rat-city, calms21 and branson-fly are all `C == 1`. `associate` never runs there either;
it is a cross-view pairwise search. **In 2D `associate_group` reduces to a truncation of the
score-ordered `decode` survivors followed by `link_rows`.** This is not a caveat about effect size —
the code path does not execute. Say "2D" or "3D" before any sentence about `--track`, and never
quote a 2D identity result without naming the root (branson-fly and calms21 are at ceiling, so
rat-city is the only 2D root with an identity problem at all, and its GT is another tracker's
output).

**`--track` IS THE DEFAULT** (`--no-track` restores the memoryless pass), and **it is for LONG
CLIPS, not for crowding.** Over 480 frames the memoryless pass grows +0.6 mm/window to 39.4 mm while
the tracker holds 12-13 mm flat and is 5-8x faster; the union crop widens 193 → 230 px against 187,
the worst crop halves (p99 750 → 376), and box slots filled rise 0.866 → 0.888. **So the error
growth over a clip is the CROP degrading, not the prompt** — shown directly, not inferred. On the
58-group 120-frame protocol none of the pre-registered endpoints move, because that benchmark is
6 windows and too short to show the effect it fixes; on two crowded clips it reads −7.4 mm and
+0.089 MOTA, but those clips were selected FOR crowding and a stratification by animal count shows
**no dose-response**. What it unambiguously buys is scale: the memoryless pass leaves 17.2% of
offered boxes unclaimed and runs at **4.1 s/frame at C = 16** against ~0.6 ms, so any multi-animal
16-camera rig is otherwise unrunnable. `associate` stays for BIRTHS, the one place a memoryless
pairwise search is right. **`track` is UNCONDITIONAL in the `--det-cache` stamp**, so every cache
written while it was off is REFUSED rather than reused as if it had been tracked.

**`--track` MADE EVERY BIRTH-TIME LEVER IRRELEVANT.** Births are 104 events in 10,054 frames —
0.041% of associations, one per 97 frames — so any rule firing at birth has a ceiling of a few
hundredths of a percent against a ±0.023 MOTA seed floor. **That rate is 3dpop's and is not
portable** (rat-city sees 7 births in 500 frames, for an unrelated reason). **Measure the POPULATION
a lever governs before its selectivity.**

**`min_views = 2` was never a threshold — it is the algorithm.** Every instance is built from a
cross-camera PAIR, so `len(members) >= 2` holds by construction and the check could not fire.
`min_views = 1` is a different rule that emits each leftover box as a single-view instance; on 3dpop
it halves the `no box` outcome rate and moves no metric, because `associate` stops at the session's
animal count and with four cameras it already fills every row. Whether the pose model can *use* a
one-camera 3D window is the run's own `[data].prob_2d_only`, which is **0** in the shipped configs.

**`--max-animals` / `--det-top-k`: sweep per root, never reason.** "S = animal count + 2" was
measured **+0.112 MOTA** on a 500-frame clip and **does not reproduce at 115x the length** — the sign
REVERSES and the magnitude is **37x smaller** (+0.0030), with coverage 48x smaller. The mechanism is
in `fill`: on the short clip the spare rows FILLED and those fills were false positives; on the long
clip the same ~7.8 rows are occupied at every S. It is also a SINGLE-CAMERA result: under `--track`
at four cameras, S = 14 and S = 16 come out identical to four decimals, because the row count
saturates once the tracker has claimed what it needs. **Measure the drop rate before porting a row
count** — spare rows help exactly where the matcher cannot seat detections it was offered.

**DETECTION AND ASSOCIATION ARE SPLIT, AND `--det-cache` HOLDS RAW DETECTIONS.** `detect_raw` is
pixels → per-camera detections ranked by score; `associate_group` is the tracker/`associate`/
`link_rows` half, microseconds per frame. So every identity arm shares ONE detection pass and is
matched BY CONSTRUCTION rather than by trusting the detector to be deterministic, and `track`,
`link_boxes`, `max_animals`, `min_views` and `max_move` all LEAVE the stamp — they change nothing in
the file. `raw_rev` is unconditional in their place and is the sharpest instance of the trap: a raw
cache and a pre-split associated one are the same shape, dtype and key names, so an old one read as
raw would be associated a SECOND time, silently. **Every pre-split cache is refused.**

The stamp is the set of box-affecting options that **differ from their defaults**, deliberately: a
positional list of every value meant that adding one flag refused every cache on disk, which
happened three times in one afternoon and twice mid-sweep. `det_score`, `raw_rev`, `top_k`,
`tile_scale` and `reduce` are unconditional — because **moving a default would otherwise let an old
cache be reused silently**, which has now happened five times.

**Every detector box is bounded in `unletterbox_boxes`.** `yolox.py` decodes a side as
`exp(clamp(-6,6)) * stride` — up to ~12,910 px, ~137,000 source px after a 1/7 letterbox — and
IoU-only NMS cannot suppress it (its IoU with the real box it swallows is ~0). Clamped into the
frame; a box with no positive area comes back NaN, which every consumer reads as "no box here".

**THE COST OF DETECTION IS THE VIDEO DECODE, NOT THE GPU.** The YOLOX-Nano forward is 0.86 ms per
frame-camera against a 44 ms decode of one 3840x2160 MPEG-4 frame, 50x, and detection runs at 0-3%
GPU no matter what. Four things were wrong and all four are fixed bit-identically: a per-frame
uint8→float conversion through torch, whose intraop pool is `nproc` wide, at **67 ms/frame against
numpy's 1.0** and 62% of the whole pass; ONE GLOBAL LOCK on `_read_video` where the state is per
container, so four cameras could not overlap; no frame pool on the image-directory roots; and no
prefetch. 3dpop 136 s → 21 s quiet, **471 s → 38 s on a busy host** (the old path's cost RISES with
contention), CPU 1451% → 411%, RSS 85 → 18 GB. **Do not chase NVDEC**: 3dpop and calms21 are both
MPEG-4 Part 2, and at 3840x2160 that is 32,400 macroblocks against NVDEC's 8,192 cap for that codec
— an H100 cannot decode these files at all. The remaining lever is process count.

---

## Gotchas — every one of these has already cost someone a day

1. **T = 1 is not usable.** `encoder_decoder.py` computes `gT = T // tubelet_size` → 0, so the
   pos_embed is zero-length. **FIXED UPSTREAM in 0.3.5**: a clip shorter than one tubelet now
   raises (`view.shape[1] < tubelet_size`), and `gT` is derived from the tokens the encoder
   actually produced. Never sample fewer than 2 frames; single-frame groups are padded at ingest.

   Relatedly, **a training window's T is derived, not configured.** `[data].n_frames` is only a
   ceiling: `_frames` sizes each train window to the labelled span it covers, rounded up to an even
   number (tubelet 2), floor 2. The annotated sessions carry ONE labelled frame per 65-frame group,
   so a fixed T = 24 spent 24 encodes to supervise 1. Val and test still enumerate fixed `n_frames`
   windows, or the metric would not be comparable across checkpoints. A train window may also be
   **strided** by `[data].frame_strides` (default `[1]`); the derived-T rule then runs on a lattice
   of spacing s, and T is capped by the room left on *that* lattice. `SmoothnessLoss` has no notion
   of dt — `torch.diff` is an UNDIVIDED difference — so its k-th difference grows like `s^k` (256x
   at k = 4, s = 4) and `_tune_smoothness` divides that out per batch. What it cannot fix, and what
   any `frame_strides` arm must declare: the HINGE is scale-invariant, so **striding genuinely
   loosens it** — the threshold tracks the trajectory's k-th derivative while the per-frame jitter
   it exists to catch is white and s-independent. **This used to be documented backwards.** And
   `SmoothnessLoss` raises below `order + 1` frames, so the order is clamped per batch; at T = 2 it
   degrades to a first difference rather than being disabled.
2. **`scene_features=` and `cube_scale=` were dropped from `TrackerEncoder.forward` in 0.3.x —
   RESTORED in 0.3.5.** `forward` takes `scene_features=` again (plus `input_size=`), and
   `share_scene` / `_forward_window` pass the stashed encode through that argument instead of
   routing around the public API. `cube_scale` remains a derived quantity, not an argument.
3. **`batch_size` is structurally 1.** `custom_collate` keeps only item 0's `cgroup` and the model
   takes one camera group per batch. This is why there is no DDP. Known ceiling, not a bug to fix
   casually.
4. **Keypoint identity ≠ array position.** The library drops keypoints with <2 valid frames, so `N`
   shrinks and positions stop matching ids. The loader must never filter. **This failure is
   invisible in the loss curve.**
5. **Keypoint ids ride in the occlusion channel**, and the stock `QueryEncoder` clamps
   `occlusion+1` into `[0,2]`. **0.3.5 makes the misuse loud** — values outside `{-1,0,1}` raise —
   but the rule stands: never share that tensor between the two consumers.
6. **`vis` and `vis_2d` are both-or-neither** — supplying one dies inside einops (0.3.5 adds the
   same guard on the loss side, so the silent discard is also loud now). And
   `get_eval_metrics` wants the trailing dim `(B,T,N,1)`.
8. **The crop rule is exact, not approximate.** `crop.py` is lifted verbatim from posetail-pose's
   verified copy. posetail 0.3.5 now exposes the same rule as a public `crop_box_for_points`
   (report 26 #23), and the local one stays as a documented superset: it adds `pad` (0 for a
   caller holding an already-padded stored extent) and None on an all-NaN input, both load-bearing
   for the detector. A test asserts it is int32-exact against `crop_cgroup_to_points`. If that
   fails, **every detector number is invalid.**
9. **Moving-camera inference was not supported upstream.** 0.3.5's
   `load_camera_group_from_metadata` honours `moving_cams: true`; we build the camera group via
   `format_camera_group(..., moving_ext=)`. Only `TrackerEncoder` is moving-cam-safe —
   `ScorerEncoder` and `TrackerTapNext` shape-error on `(T,3)` centres. **No shipped root has a
   moving rig** — every `calibration.toml` reads `moving = false` and no `extrinsics.pq` exists
   anywhere — so this path is exercised only by the synthetic fixtures in the tests.
10. **allen-mouse's npz is column-sorted.** `pose3d.npz['pose']` is ordered by
   `sorted(f'{name}_{axis}')` while `keypoints` is name-sorted, which transposes all 8 `X` /
   `X-base` pairs — 16 of 47 keypoints. The converter applies the permutation once. Zipping `pose`
   against `keypoints` silently mislabels them and nothing downstream notices.
11. **Nothing in the parent process may decode video before the loader forks.** The forked workers
   deadlock in a futex while holding an open container: 0% GPU, ~0 worker CPU, no traceback, no
   timeout, forever. `scripts/train.py` materialises its fixed val windows before the train loader's
   first `next()`, so it used to be the parent that decoded; it now pulls them through a one-worker
   `DataLoader`, byte-identical because `PoseDataset` seeds val items by index.
   `scratch/calms21_loader_repro.py` isolates it in seconds, without the model. **3dpop is
   video-backed too and survives it** — which is the only reason the five-dataset sweep never hit
   this, and why "the sweep works" was not evidence the pose loader was fork-safe. Any video-backed
   root can trigger it, including ones a converter makes later: a property of the pixels, not a list
   of names.

   This is also why **the reader cache may never be sized by probing**: opening one `VideoReader` in
   the parent to measure anything IS the deadlock. `_reader_cache_size` derives it from a camera
   count, a frame size and `get_worker_info()` — all parsed-toml or in-process facts. **DO NOT SET
   `TAILCYCLENET_READER_CACHE` BY HAND**: at best a no-op repeating the derived value, at worst it
   overrides the per-worker clamp that stands between a 16-camera rig and swapping. A cache below
   the camera count misses on *every* call (a 16-camera rig at 4 ran detection 2.5x slower). If a
   run needs a different cache the fix belongs in `_reader_cache_size`, where it is testable.
12. **A RUN FOLDER USED TO RECORD NO COMMIT, and one config key had a default it could not
   justify.** `runs/3dpop-prior` trained under unconditional per-frame re-anchoring, finished nine
   hours before the commit that replaced it, and carries no `gridresid_offset` — so it loaded as the
   architecture it was not trained as, for weeks, silently. **+23.1 mm MPJPE [+22.3, +24.0] and
   −0.18 MOTA**, with the pose visibly lagging the animal until the bounds mask dropped the prior.
   Both halves are now closed: `save_run_meta` writes `provenance.toml`, and the key raises rather
   than defaulting. **The `-none` siblings are mismatched harder**: query-free, the triangulation is
   substituted at every keypoint so the residual head's output is discarded outright. Keyless
   generations: `runs/*`, `runs/20260810*`. 2D is unaffected (`forward` returns before both offset
   paths). **Any 3D number published off those needs re-checking.**

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
   row. It is a FRACTION of K, not a count — K is 4 on rat-city and 47 on allen. Default 0 keeps
   every published number reproducible; any claim about a lever that changes row sparsity needs it
   on. **But it is punitive at small K with sparse labels, so quote it with both**: rat-city's
   absolute MOTA falls 0.587 → 0.092 at 0.5 while the *delta* between two arms barely moves
   (+0.051 → +0.059). Use it for deltas, not absolutes, and never carry one value across roots
   (0.25 rounds to 1 at K = 4, i.e. identical to 0 there).
10. **Pairing is complete-case, which flatters the arm that failed more.** A group where either side
   is non-finite leaves the comparison, so `paired_bootstrap` returns `n_dropped` and `--vs` prints
   it. A delta over 9 of 17 groups is not a delta over 17.
11. **A PER-WINDOW STATISTIC MUST USE THE SEAM RULE'S OWN FRAME→WINDOW ASSIGNMENT.** A frame in an
   overlap belongs to the LAST window containing it. Slicing `[start : start + n_frames]` instead
   hands every window its neighbours' frames too, which SMOOTHS the per-window error: it reported
   ZERO burst windows on every 3dpop clip where the correct binning finds up to seven, at identical
   indices. It reads as "the effect does not reproduce" rather than as a binning bug.
12. **A LABEL-FREE IDENTITY STATISTIC IS A SMOKE ALARM, NOT A METRIC, and a SELF-NORMALISED one
   ranks arms BACKWARDS** — correlation **−0.621** against true `idsw`, because a row swapping on
   every step has an enormous p99 so almost nothing exceeds it, and the memoryless box path scores
   BEST of six arms. Any per-unit-normalised discontinuity measure inherits this. The ABSOLUTE form
   passes the gate at **+0.982** — but **excluding that one broken arm the correlation is −0.558
   over five working arms**. So it detects a box path that is BROKEN, with no labels and an enormous
   margin, and cannot rank two that both WORK. **And never score a cue on the statistic it
   optimises.**
13. **There is a temporal statistic and there was not before**: `motion_ratio`, paired over the steps
   both arms attempted. Every consistency number this repo had — jerk, bone CV — *rewards* a
   prediction that stopped moving, which is how the motion lock stayed invisible. **But
   `motion_ratio` near 1 is necessary and NOT sufficient**: `--seam blend` moved it −0.016 toward 1
   while making identity worse, and `moving_crop` moved it on all six roots while buying no accuracy
   anywhere. Read it beside the error and identity columns, exactly as `err` must be read beside
   coverage. **It is also not single-signed** — it reads LOCK where the animal moves (johnson 0.757)
   and JITTER where it does not (allen 1.983, an 11x slower animal) — so set a bar as |ratio − 1| on
   the root the arm was trained on, and never pool the two.

Reproduction note: posetail-pose's `reports/golden_allen_j3.json` is an exact-reproduction contract
for *that* pipeline. This repo will not match it bit-for-bit and should not claim to. The check is a
band: allen-mouse cross-animal MPJPE near 3.394 mm, human-vs-human baseline 2.208 mm. A large gap is
a port bug, and the first suspects are the allen column-sort permutation and the crop rule.

**BUT NAME THE SPLIT, because `allen-mouse-combined/val` CANNOT PRODUCE IT.** All five of its val
sessions are a seen animal on an unseen session, so that root yields a seen-animal number
(reference: `prior_self` 2.484, `prior_none` 2.642, `none` 2.786; golden on that axis is 2.274).
3.394 is genuinely cross-animal and needs a split that holds an animal out. Scoring a seen-animal
arm against 3.394 compares two axes and reads the ~0.9 mm difference in axis as a result.

**`checkpoint_best` IS SELECTED ON `val`, THE PRIOR-FREE PASS**, not on `val_self`. Their argmins
differ by up to 13,600 iterations in practice, so on any run whose deployable regime is prompted,
`best` is not the checkpoint you want — and it penalises exactly the arm that trades query-free
accuracy for prompted accuracy. Score `best` AND `last`.

---

## The deleted levers, in one line each

Every one below was measured, refuted, and REMOVED (`dev/reports/24_lever_audit_and_cleanup.md`).
Listed so nobody re-proposes one; the measurement is in the report and the reason is at the deletion
site.

**Inference:** `--axis-veto`, `--kpt-affinity`, `--kpt-centre`, `--axis-cost`, `--swap-repair`,
`--random-veto` (their rate-matched control), `--stitch`, `--dup-res-px`, `--prior-vis-thresh`,
`--seam blend`, `--moving-crop`.
**Detector training:** `--ignore-band`, `--ema-decay`, `--warmup-frac`, `--augment-photometric`,
`--identity` / `--id-weight` (and the YOLOX identity head behind them).
**Config:** `moving_crop`; `synth_motion_*` (report 33 — a measured null, deleted with the 0.3.5
upgrade); the dead loss weights `gamma`, `feature_loss_weight`, `pixel_thresh`; the
inert model keys `corr_radius`, `use_volume_embedding`, `occlusion_embedding`, `mode_3d`,
`cross_attn_dim`; and six `configs/datasets/*.toml` keys nothing ever read.
**Never built, measured in `scratch/` and refuted there:** an `--anchor self` fixed-point iteration,
a coarse-to-fine grid decode (the knob does not exist — `g3d_lo`/`g3d_hi` are never read at
inference), "anchor refine", and `--anchor detector`. See the independence paragraph below.

**THE CUE IS REAL AND EVERY MECHANISM THAT SPENDS IT IS REFUTED — all three forms are now tried.** A
veto, a permutation and a Hungarian cost term were each built, measured on BOTH roots and lost; the
ranking a K = 4 body axis supplies is too noisy to spend in any of them. **Do not propose a fourth
without first raising the cue's quality.** Two more are dead on POPULATION rather than on the cue: a
learned identity head reaches 25.1% on genuinely contested decisions against 8.3% chance (39.6% at
3x the training), but the centroid alone already settles 98.8%, so a perfect head could flip
**≤0.31%** of associations; and SLEAP's `connect_single_track_breaks` fires **0 times in 57,593
consecutive steps**. **When refuting a mechanism, refute it on the quantity a better model cannot
change.**

**A PRIOR HELPS ONLY WHEN IT IS INDEPENDENT EVIDENCE, and that retires four proposals at once.**
Iterating `--anchor self` to a fixed point, a coarse-to-fine grid decode, and seeding pass 2 from
pass 1's own pose ("anchor refine", −2.8 px where it is the *only* prior, which is the cleanest test
and the worst result) all feed the model a prior derived from the SAME pixels the forward is about
to see, so it carries nothing the forward does not have and its errors correlate with the ones it is
meant to fix. `carry` is worth 10.2 px precisely because its prior comes from DIFFERENT pixels.
**The fourth, `--anchor detector`, is the one that satisfies the independence test and it fails on
the POPULATION instead** (report 24 §9k): the detector's keypoints are dense enough (96–100% of
slots) and nowhere near accurate enough — p50 **31.8 mm** on 3dpop and **95.6 px** on rat-city
against the model's own prior-free median of 9.96 mm, i.e. **3x worse than the answer it would
seed**, and 10–35x the `prompt_noise_px` / `prompt_offset_px` the prompt trained against. Costs
+8.34 mm and +13.03 px.

**THE PRE-SCREEN THAT FALLS OUT IS WORTH MORE THAN THE LEVER: a prior must be more accurate than the
prediction it seeds, and `kpt_agree` on any prior-free arm already reports exactly that distance**
(3dpop 0.140 box sides, rat-city 0.721). Either number predicts both refutations for ~1 min of CPU,
before any inference. Same shape as the `fp_dup`-must-be-live rule for `--pose-nms`. And `kpt_agree`
is CIRCULAR under any detector-seeded regime — the model is handed the keypoints it is then scored
against — so it must never be quoted as a win for one.

**`--pose-nms` is the one identity lever that works.** `--rotate-deg` survives with its 180-degree
setting refuted. `synth_motion_*` was measured as a null on the one root that can measure it
(report 30) and is **deleted** with the 0.3.5 upgrade (report 33), along with the moving-crop
geometry and `patches.py` it was the last consumer of.

---

## What is owed

**Blocking.** `3dpop/test` is DONE — all 58 groups untruncated at `--chunk 500` give **145 units,
2.86M matched points, MPJPE 9.586 mm [8.548, 11.104]**, an interval **2.8x tighter** than the
120-frame protocol's, and its units come from 58 DISTINCT sessions rather than one clip
(`scratch/longclip/`, report 24 §7). `calms21/test` still wants the same treatment, and `eval.py`
still resamples chunks rather than sessions — which is the honest interval on a 58-clip split. **calms21's "identity collapse" is an ARM, not the root** — report 24 §6 reaches **MOTA 1.000,
coverage 1.0000, MPJPE 6.379 px** on two of its clips with `--anchor carry --link-boxes --track
--refine`, reproduced end to end on current code. The collapsing arm (115 px, MOTA −1.0 on six of
ten chunks) differs from it in FIVE levers — session, pose run, detector, `--refine`, clip length —
so report 23 §2's "a delta measured on top of that is uninformative" is true of that arm and must
not be generalised. What is owed is the one-lever test: `--refine` on the failing session, which
shrinks exactly the crop §6c measures as wider than the two mice are apart.

**Cheap, off artifacts already on disk.** `--pose-nms` on 3dpop K = 17 (~1 min CPU, and the only
outstanding test of the `fp_dup`-must-be-live discriminator — **a rule that says in advance which
roots a lever will help is worth more than the lever**); `--overlap 12 + --pose-nms 0.6`, predicted
to displace the current recommendation; `--overlap 16`.

**MEASURED, LARGE, AND DELIBERATELY NOT SHIPPED: a WIDER pass-1 crop under `--refine`** (report 24
§9m, `scratch/refine_wide/`). Inflating the pass-1 box about its centre is free — the same forward
on a differently-positioned box, and pass 2's box is a function of the pass-1 PREDICTION, so it does
not move. On 3dpop, 9 paired cells all agree in sign: MPJPE −6.9 to −15.2 mm, coverage +0.004 to
+0.088, **MOTA +0.011 to +0.099** (3–4x the seed floor), with **p99 better by 83–313 mm in every arm
while p75 moves ≤ ±1.6** — it deletes catastrophic rows, it does not improve good ones. **INDEPENDENT
of `--refine-px`** (the same size at full-resolution pass 1), except at 96 px where it is harmful.
**On calms21 it is a catastrophe**: 1.5x costs +71.18 px and MOTA −1.0229, `fp_dup` +0.4813 — both
rows on the same mouse on 48% of frames. **The discriminator is one label-free ratio off the
detection cache: pass-1 crop side ÷ distance to the nearest other animal** (3dpop 0.38, helps;
calms21 3.81, collapses) — where the crop already spans a neighbour, the only cue for WHICH animal
is that it is CENTRED, and widening destroys it. The rule is fitted on two roots. **The bar for
shipping is a second 3D root (or 3dpop's other 57 test groups at `--chunk 500`) plus rat-city as the
discriminator's test** — one group per root is not enough to license a flag here.

**Larger.** `synth_motion_*` is DELETED (report 33) — measured: null on rat-city, unmeasurable on
allen, **+3% in the working regime on johnson** (report 30). What the null leaves behind is
**johnson's other two clips** and **why two of its six chunks break on both arms** — a clip or
box-path failure that is 94% of that root's headline delta and worth more than the lever ever was
— plus **allen re-run iteration-matched at 50,000** (both sides have that checkpoint) and the one
change that could plausibly have rescued the arm: stopping the background from sliding with the
animal, via compositing or real neighbouring frames. The pose loader's `aug_rotation_deg = 45` has
no accuracy number in either dimension and is the one surviving unmeasured augmentation.
`--link-boxes` is default-on and never measured. `link_rows`' `max_age = 24` and `birth_age` are
pinned constants unreachable from any CLI that govern birth and expiry on a long clip.

**THE BIGGEST LEVER IN THE REPO IS NOT A FLAG.** Pose on a GT crop is **19.3 px at coverage 1.000**
against **44.7 px at 0.566** through the detector, and on the long clip 100% of coverage loss is
`no box`. Every identity lever in reports 19-21 fought over ±0.02 MOTA while the box path costs
25 px and 43% of coverage.
