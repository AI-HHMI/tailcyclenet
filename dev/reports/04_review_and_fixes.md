# Review of the whole repo, and the fixes

A read of all 5,147 lines against `../../posetail/posetail-next` (the library) and
`../posetail-pose` (the predecessor). 48 tests passed before this work and none of them touched
`model.py`, `query_encoder.py`, `infer.py`, `metrics.py` or `checkpoints.py`. 77 pass now.

The headline: **no 3D forward on a moving-camera rig could ever have run.** Everything else is
smaller.

## The moving-camera path

`posetail-next/posetail/posetail/cube.py:95-99` states the contract — a `(T,4,4)` extrinsic aligns
against **axis -3** of the points. Four violations, in four places.

1. **`PoseQueryEncoder` dropped two branches the stock encoder has.** `query_coords` is the
   flattened `(t n)` query axis, so axis -3 is the BATCH axis: projection came back with the batch
   axis silently replaced by `T`, and the visibility term raised inside einops. Stock
   `QueryEncoder.forward` un-flattens to `b t n r` first, for exactly this reason.
   `posetail-pose/posetail_pose/kpt_query_encoder.py:361,392` has the identical omission — the gap
   was inherited from the sibling, not from upstream, and it survived because posetail-pose never
   ran a moving rig through its own encoder.

   The `uniform` fast path also had to be disabled on a moving rig: one shared query point still
   projects to a different pixel every frame, so a width-1 axis cannot represent the answer.

2. **The camera group was built without `moving_ext` at four of five sites** (`infer` ×2,
   `detect_group`, `BoxDataset`). Five copies of one construction collapsed into
   `Session.cgroup(gid, frames)`. The `BoxDataset` one was a *target* bug, not a speed one: it
   projects `(S,K,3)` whose axis -3 is the ANIMAL, so a `(T,4,4)` extrinsic drew the detector's
   regression boxes around animal `i` seen through frame `i`'s pose.

   `cgroup` also gives **every** camera the per-frame form when any camera moves —
   `_decode_from_scene` stacks `cam['ext']` across cameras (`tracker_encoder.py:623`), so a mixed
   rig with one `(T,4,4)` and two `(4,4)` cameras is a stack error.

3. **An incomplete `extrinsics.pq` became the identity.** The array is pre-filled with `eye(4)`,
   which is a valid-looking pose at the world origin. Now raised in `Session.labels` (so it fires
   whether or not anyone validated) and reported by `validate_session` rule 13.

4. **`scene_center` used frame 0 only.** Now one ray per (camera, frame). Static rigs are
   bit-identical; a moving rig's centre moved 1.4 mm on the fixture.

## `kpt_chunk`

Plumbed through `PoseTrackerEncoder.forward` and structurally unable to work: the keypoint ids and
the query-validity mask sat on a full-K stash the library knows nothing about, so every chunk got
the whole thing. `_tile_to_query_axis` asserts on a non-divisible chunk and returns the WRONG
identities whenever it happens to divide.

posetail-pose avoided half of this by routing ids through the `occlusion` channel, which the
library slices for free (`posetail_pose/model.py:718`); gotcha #5 deliberately freed that channel
here. Its `_query_ok` had the same bug unfixed, so there was no precedent to copy.

Fixed by overriding `_decode_from_scene` with a running cursor, asserted to land exactly on K.
Verified by instrumentation: `kpt_chunk=2` on K=3 produces two calls given ids `(0,1)` then `(2,)`,
and the prediction is bit-identical to the unchunked pass.

## Configs that were accepted and wrong

| key | what happened |
|---|---|
| `output_mode` | The library **defaults to `direct`**, so omitting the key was as bad as setting it wrong. `_reanchor_per_frame` recovers the residual as `3d_pred_cams_direct - query`, an identity the library only makes true for `residual`/`gridresid` (`tracker_encoder.py:670`). Under `direct`/`grid`/`gridnorm` it re-anchors a prediction that was never query-anchored and corrupts `coords_pred` with no error anywhere. Now asserted **and** `setdefault`-ed. |
| `use_volume_embedding` | No volume term is built here; failed as an assertion deep inside a warm start. |
| `mode_3d` | Upstream it picks the *class*; here it is stored and never read, so `tapnext` was accepted and ignored. |
| `image_size` | Two keys, `[model]` and `[data]`, that must agree and nothing checked. `PadToSize` only pads UP, so a smaller data size leaves the crop in the corner of a zero-padded canvas; either way 2D shifts by half the difference and 3D scales by their ratio. |
| `n_frames = 1` | Gotcha #1, unguarded. `test_window_is_at_least_two_frames` asserted `shape[1] >= 1`, which is **true of the exact 1-frame window it exists to forbid** — it passed while the guard did not exist. Repaired to `== 2`. |

## The window loop

- **A detector row is not a label row.** `S` comes from the box source; `src[a]` was evaluated for
  every one of them on every anchor mode. Plain IndexError the moment a detector offered more
  animals than the session labelled.
- **`p[-overlap]`** ran off the front of a window shorter than the step.
- **A camera without a box dropped the whole animal.** Found during end-to-end verification:
  association leaves an unmatched camera NaN, and requiring a box in *every* camera took coverage
  to **0.000** on a three-camera rig where two views had the animal. The model is trained on
  camera subsets already (`cams_to_sample`, and `prob_2d_only` trains the one-camera case), so a
  subset is a supported input. Coverage 0.000 → 1.000.

## Reporting

- `eval.py` built its PCK arrays by re-deriving the slicing and got it wrong whenever `pred` and
  `true` disagreed on S or T — comparing one animal's points against another's. It now reuses the
  arrays the table was computed from.
- Multi-animal rows quoted a **matched** error beside a **rowwise** coverage.
- One MPJPE and one PCK were printed over mixed 2D and 3D rows, averaging pixels with millimetres.
  Now one block per mode.
- `mota`'s `ignore` dropped **every** unmatched prediction on any frame containing a
  present-but-unannotated animal, and `match_instances` took an `ignore` argument it never used.
  Now per-prediction against the ignore *box* where the format carries one, with the frame-wide
  fallback retained and **counted** as `fp_ignored`.
- `match_instances` was an O(T·Sp·St) Python loop — 5.7M calls per eval on rat-city's single
  57,594-frame group. Vectorised, and checked bit-identical against the loop over 60 random cases.

## The keypoint registry

`CLAUDE.md` promises append-only ids so the embedding rows survive a warm start. It was not
implemented: every run called `Registry.build` with no base. Discovery order is a directory
listing, and reversing it moved `mouselike` from ids `(0,1,2)` to `(4,5,6)` — each row of
`kpt_embed` then means a different body part than the checkpoint trained it to mean, invisibly.
`train.py` now takes the run folder's registry, then the warm-start checkpoint's, as a base.

## Training

The validation loop existed only as two config keys. It now reports **two regimes**, following the
lesson recorded in `posetail-pose/scripts/train_pose.py:231-245`:

- `val/*` — **prior-free**. The loader's `kpt_prior` is ground truth (evaluation rule 7).
- `val_self/*` — **self-prompted**: predict prior-free, re-query at the model's own frame-0
  prediction. Label-free, and the regime a `query = "prior"` arm actually deploys in. Prior-free
  alone is a structurally different forward — `uniform` flips True and the patch term collapses to
  one broadcast patch — so selecting checkpoints on it judges the arm by a path it is not trained
  on. Shares `infer.self_prompt` with the window loop rather than being a second copy.

Metrics come from `get_eval_metrics`, not the loss. wandb is wired from the existing `[wandb]`
block, plus `log.jsonl` in the run folder. **wandb does not own the run folder** — both reference
implementations set `exp_dir = wandb.run.dir`, which is how eval ends up needing a wandb path to
find a checkpoint; `--out` stays authoritative here.

## Checked, and correct — no change

- The **allen column-sort permutation** (gotcha #10, named as first suspect for any accuracy gap):
  `convert_v4.py:182-189` applies it in the same direction as posetail-pose's `COORD_PERM`,
  permuting coordinates into name order and leaving the name-indexed `vis` alone.
- `rotate_camera_image_plane_3d` and `rotate_camera_group` both handle `(T,4,4)` explicitly.
- `crop.apply_crop` / `_resize_camera` touch only `mat`/`size`/`offset`.
- `time_embed_mode`, `principal_point_embedding`, `intrinsic_embedding`, `occlusion_embedding`,
  `grid_decode_space`, `enable_subpixel_refinement` are all genuinely optional here.
  `stride_length != n_frames` is designed for, and left tolerated.

## Reverted

Declaring `toml` / `opencv` / `pillow` / `wandb` / `schedulefree` / `decord` in `pyproject.toml`.
They are imported directly, so it looked right — but posetail pins them **exactly**
(`wandb==0.19.5`, `opencv-python==4.9.0.80`, `toml==0.10.2`) behind its own `==0.3.2` pin, all
captured in `pixi.lock`. Re-declaring either duplicates a pin that conflicts the moment posetail
moves, or asks for `*` and conflicts now — conda's wandb 0.28.1 against posetail's 0.19.5 makes the
solve unsatisfiable. Tried, reverted, and the reason is a comment in `pyproject.toml` so nobody
retries it.

## What is still not covered

`scripts/eval.py` and `scripts/train.py` are exercised by hand, not by test. The end-to-end runs
below were done once each; a regression in them will not be caught automatically.

Everything below ran on synthetic fixtures with a deliberately small random model
(`tests/test_model.py:SMALL`, ViT-base), so these are plumbing results, not accuracy results:

- train → val (both regimes) → checkpoint → wandb offline → `log.jsonl`, 2D and moving-3D
- re-run into the same `--out`: `keypoint_registry.toml` byte-identical
- `train_detector.py` on a moving rig
- `infer.py --detector --kpt-chunk 2 --max-animals 2` on a moving rig → `eval.py`

**A caveat on the smoke runs.** With a random tiny model the 2D grid head saturates — `2d_pred`
has std 0 and every query decodes to the grid centre — so `val` and `val_self` come out equal to
four significant figures in 2D. They differ in 3D (48.05 vs 48.16). That the prior is genuinely
reaching the model is asserted at the query encoder instead
(`test_the_prior_reaches_the_query_encoder`), where it is observable regardless of the head.

## Stale after this work

Three `CLAUDE.md` gotchas now describe fixed behaviour — #9 (moving-camera inference unsupported),
#5's implication that `kpt_chunk` is unreachable, and #1's claim that the loader guards T=1 by
padding. Not edited: `CLAUDE.md` is the project's instruction file.
