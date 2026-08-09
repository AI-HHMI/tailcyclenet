# `tailcycle-dataset` — the training-data format (spec)

One file, the contract. This document is the specification `tests/test_format.py` and
`tailcyclenet/format.py` are written against; when the two disagree, this document wins.

## 1. Purpose

One format for **both** hand annotation and bulk training data. Those were previously two shapes
and neither could do the other's job:

- **`annotated-frames/`** — DLC wide `CollectedData.csv`, a `config.toml` for crop offsets, an
  aniposelib `calibration.toml`, PNGs per camera. Single isolated frames, single animal, no
  temporal context, no way to say "an animal is here but nobody labeled it".
- **`posetail-finetuning-v4`** — `<dataset>/<split>/<session>/<trial>/{pose3d,pose2d}.npz +
  metadata.yaml + img/<cam>/%06d.jpg`, `pose` shaped `(S,T,K,R)`. Multi-animal and temporal, but
  a derived binary blob with no provenance, no per-camera 2D observation, no boxes, and no
  distinction between "assessed and occluded" and "never looked at".

The difference between a hand-labeled session and a machine-tracked one is **how many rows carry
labels**, not what shape the data has. This format encodes that and nothing else.

## 2. Decisions (locked)

1. **Split is a directory level, not a field.** `<dataset>/<split>/<session>/`. Train/val/test
   remains a *consumer* concept — there is no `split` column anywhere — but it is expressed as a
   path because none of the shipped datasets split cleanly by session (rat-city's train, val and
   test are three frame ranges of one recording).
2. **A group is a contiguous clip.** `n_frames` is unbounded; `n_frames = 1` is legal and means
   "no context". A 57,594-frame recording is one group. This is what v4 called a trial.
3. **Multi-animal from the start.** `animal_id` is a cross-view, cross-frame identity.
4. **The 3D layer is first-class.** A session may carry native 3D (`points3d.pq`), native
   per-camera 2D (`keypoints.pq`), or both. 3D is *not* required to be derived from 2D.
5. **`mode` is a per-session property.** One `train/` may hold 2D and 3D sessions together;
   cross-session agreement covers `names` only.
6. **`status` is the visibility channel**, in every label table. There is no separate `vis`
   column and no NaN-means-something convention.
7. **Sparsity is a row count, not a mode.** A row exists only where a determination was made; a
   missing row means "no determination". A dense session simply has a row everywhere.
8. **Partial assessment is legal.** Completeness is a read-time policy; the format never
   enforces it.
9. **A box per animal per view, always optional** — for `labeled`, `present` and `absent` alike.
   When given it is a human judgment enclosing the animal; it is *not* required to match the
   keypoint extent or the crop rule.
10. **No `annotator` column.** Multiple annotators = separate dataset roots, the annotator named
    in `session.toml` `[provenance]`. Two annotators on one point would collide with the row key.
11. **All camera geometry lives in `calibration.toml`**, including the crop. `session.toml` is
    keypoint identity and provenance only.
12. **Tables are Parquet.** `.pq`, tidy long form, one row per assessed thing.

## 3. Directory layout

```
<dataset_root>/
  <split>/                            # train | val | test. `train` required for training.
    <session_id>/                     # the folder name IS the session id
      session.toml                    # keypoint identity + provenance
      calibration.toml                # ALL camera geometry incl. crop; always present
      groups.pq                       # one row per group
      keypoints.pq                    # per-camera 2D + per-camera visibility   (optional)
      points3d.pq                     # 3D                                      (optional)
      instances.pq                    # boxes / present / absent                (optional)
      extrinsics.pq                   # per-frame extrinsics, moving cameras    (optional)
      groups/
        <group_id>/
          <cam>/000000.jpg 000001.jpg …      # image dir, one per camera, OR
          <cam>.mp4                          # video, one per camera
```

At least one of `keypoints.pq` / `points3d.pq` must exist and be non-empty.

Either the image dir or the video may be a **symlink**; the converter for
`posetail-finetuning-v4` symlinks whole directories, which is why converting 475 GB of pixels
costs ~900 symlinks rather than 550,000.

Session-scoped, because cameras, crop and calibration are session properties and repeating them
per row or per group would be pure duplication. A single-session dataset degenerates to one flat
table set.

## 4. `session.toml`

```toml
mode  = "3d"        # "3d" -> >= 2 cameras + full calibration; "2d" -> exactly 1 camera
units = "mm"        # units of 3D. Pixel labels are ALWAYS pixels, in every mode.

names      = [ ... ]   # ordered keypoint names -- THE authority for the keypoint axis
skeleton   = []        # OPTIONAL, usually empty: [["a","b"], ...] name pairs
flip_pairs = []        # OPTIONAL, usually empty: [["left_x","right_x"], ...]

[provenance]
source         = "..."          # e.g. "posetail-finetuning-v4/rat-city"
annotator      = ""             # empty when a single annotator authored the root
annotator_tool = "..."
created        = "YYYY-MM-DD"
```

There is no `schema` field, no dataset `name`, no `session_id` and no `fps`: the folder name is
the session id, this document is the version, and fps is declared per group (§6).

- **Keypoint names live here, per session.** Every session under one dataset root must carry
  identical `names` (validation rule 3). `mode` and `units` may differ between sessions — that is
  what lets a `train/` mix 2D and 3D.
- The tables are **name-indexed**, so a column-order artifact like allen-mouse's
  (`pose` column-sorted while `keypoints` is name-sorted, transposing 16 of 47 keypoints) cannot
  occur in this format. Resolve it in the converter, once.
- `skeleton` and `flip_pairs` are optional and expected to be empty for most data — they are
  needed only for bone-length diagnostics and bilateral flip augmentation. When present, every
  name must be in `names` and `flip_pairs` must be an involution.

## 5. `calibration.toml`

An aniposelib `CameraGroup` file with **two** added per-camera keys, `offset` and `moving`. It
is **always present**, including for `mode = "2d"`, so there is exactly one place camera geometry
lives. Everything else is aniposelib's own schema, read and written by `CameraGroup`; lens model
is aniposelib's existing `fisheye` flag, not a new field.

```toml
[cam_0]
name        = "cam088"
size        = [1200, 800]      # size of the image ON DISK; asserted against the files
matrix      = [[1852.4, 0.0, 719.5], [0.0, 1852.4, 539.5], [0.0, 0.0, 1.0]]
distortions = [-0.0361, 0.0, 0.0, 0.0, 0.0]
rotation    = [0.0041, -0.0017, 0.0045]     # Rodrigues, world -> cam
translation = [2.5517, -1.2033, 310.44]
fisheye     = false            # aniposelib's own flag; picks the FisheyeCamera class
offset      = [120, 140]       # origin of the stored image inside the sensor frame (ADDED)
moving      = false            # per-frame extrinsics live in extrinsics.pq         (ADDED)

[metadata]
```

**`size` is the stored image; `matrix` is in sensor coordinates.** Projecting is "apply `matrix`,
then subtract `offset`" -- posetail's convention (`cube.project_cam` subtracts `cam['offset']`
after the intrinsic matmul; `undistort_points` adds it back) and aniposelib's (`set_size` is the
image being worked on). Note the example above: a principal point of 719.5 belongs to a
1440-wide sensor, while the file on disk is 1200 wide at `offset = [120, 140]`. There is
deliberately **no sensor-size field**: nothing consumes it, none of the source datasets record
it, and a field nobody reads is a field that silently goes wrong.

**Why the crop lives here.** `offset` is the only thing relating the pixels on disk to the sensor
the calibration describes, so it belongs with the calibration and not in a second file that can
drift from it. This is the direct answer to a measured failure: skipping the offset turned 2.38 mm
inter-annotator 3D disagreement into 16.9 mm, because a *common* offset does **not** cancel
between annotators — it makes rays non-intersecting, and triangulation amplifies a 2 px labelling
difference into ~12 mm.

**Coordinate rule, stated once and enforced:** every `x`/`y` in the tables is in the coordinate
system of **the image on disk**. The mapping to the sensor lives in exactly one place,
`offset`.

For `mode = "2d"`: one camera, `offset = [0, 0]`, and
`matrix` / `rotation` / `translation` / `distortions` may be omitted — the consumer substitutes a
nominal pinhole (posetail's `_make_nominal_2d_camera`). Writing real values for a single
calibrated camera is legal and lets the same session be read in 3D single-view mode.

`offset` and `moving` are ignored by aniposelib's loader — verified: `CameraGroup.load` on a file
written by `dump_calibration` returns the right cameras and sizes — so an anipose user can read
this file directly. `tailcyclenet` uses `CameraGroup.from_dicts` for the camera itself and reads
only those two keys separately; there is no reimplementation of a camera in this repo.

`tailcyclenet` pins the **`pytorch` branch** of aniposelib, whose `Camera` is an `nn.Module` with
torch intrinsics and extrinsics, so projection and triangulation run on GPU. One consequence to
know about: those tensors are `nn.Parameter`s and arrive with `requires_grad=True`, so the
adapter that hands cameras to posetail detaches them — otherwise every projection in the loss
would build an autograd graph through the calibration.

## 6. `groups.pq`

| column | type | req | meaning |
|---|---|---|---|
| `group_id` | str | ✓ | unique in session; equals the folder name under `groups/` |
| `n_frames` | int32 ≥ 1 | ✓ | images (or video frames) per camera |
| `fps` | float32 | | fps of the source recording — **the only place fps is declared** |
| `source_video` | str | | provenance: which recording this came from |
| `source_frame_start` | int32 | | absolute source index of group frame `0` |
| `source_frame_step` | int32 | | default `1` |
| `notes` | str | | free text |

No `split` (§2.1), no `include` (delete or rename the group folder, or filter consumer-side on
`group_id` / `notes`), and **no `label_frames`** — which frames carry labels is a fact about the
label tables, and the consumer must scan them to build a validity mask anyway.

## 7. `keypoints.pq` — per-camera 2D and per-camera visibility

One row per **assessed** (group, frame, animal, camera, bodypart).

| column | type | req | meaning |
|---|---|---|---|
| `group_id`, `animal_id`, `camera`, `bodypart` | dictionary\<int32,str\> | ✓ | the key |
| `frame` | int32 | ✓ | 0-based index into the group |
| `status` | dictionary\<int8,str\> | ✓ | `visible` \| `missing` \| `unlabeled` |
| `x`, `y` | float32 | | stored-image px. See the rule below. |
| `score` | float32 | | confidence in [0,1]; null for human labels |

- **`visible`** — the point is there. `x,y` are required **unless** a `points3d` row exists for
  the same `(group_id, frame, animal_id, bodypart)`, in which case the row is a *visibility
  observation*: "visible in this camera; the position is in the 3D layer". This exists because
  allen-mouse ships a real per-camera `vis (S,T,K,7)` array and no per-camera 2D, and reprojecting
  the 3D to manufacture an `x,y` would present a derived number as an observation.
- **`missing`** — looked at, judged occluded or outside the frame. `x,y` are null: an occluded
  point is not annotated. The occluded-vs-outside distinction is deliberately collapsed; it is
  resolved downstream by reprojecting derived 3D whenever 3D exists.
- **`unlabeled`** — explicitly recorded as not assessed. Consumers treat it **exactly like an
  absent row**. It exists so annotation software can persist progress state in the file the
  annotator is editing. A finished file contains none.
- **No row** — not labeled. Semantically identical to `unlabeled` for consumers.

## 8. `points3d.pq` — the 3D layer

One row per **assessed** (group, frame, animal, bodypart). No `camera` column: a 3D point is not
a per-camera observation.

| column | type | req | meaning |
|---|---|---|---|
| `group_id`, `animal_id`, `bodypart` | dictionary\<int32,str\> | ✓ | the key |
| `frame` | int32 | ✓ | |
| `status` | dictionary\<int8,str\> | ✓ | `visible` \| `missing` \| `unlabeled` |
| `x`, `y`, `z` | float32 | ✓ iff `visible` | in `session.toml` `units` |
| `score` | float32 | | |

Same enum, same semantics as §7. `missing` here means "this point exists on this animal at this
frame and is not observable" — the honest 3D counterpart of an occlusion, distinct from a row
that was never written.

A session may carry `points3d.pq` alone (allen-mouse, 3dpop), `keypoints.pq` alone (rat-city,
branson-fly), or both (a calibrated multi-view session with real 2D labels *and* a triangulated
3D solution). The consumer derives what it needs: 2D from 3D by projection, 3D from 2D by
triangulation. Neither derivation is stored.

## 9. `instances.pq` — boxes and the ignore region (optional)

One row per animal per view per frame **where a determination was made**.

| column | type | req | meaning |
|---|---|---|---|
| `group_id`, `animal_id`, `camera` | dictionary\<int32,str\> | ✓ | the key |
| `frame` | int32 | ✓ | |
| `x0,y0,x1,y1` | float32 | | box in stored-image px; always optional |
| `status` | dictionary\<int8,str\> | ✓ | `labeled` \| `present` \| `absent` |
| `notes` | str | | |

- `labeled` — keypoint or 3D rows exist for this instance (partially or fully).
- `present` — the animal is in this view but was not annotated. **An ignore region**: a
  prediction landing here is neither a true nor a false positive. This exists because 73% of a
  tracker's false positives on rat-city were measured to be real animals the annotator skipped
  (`--ignore-unobserved`, MOTA 0.692 → 0.709). Without a box it is a weaker statement — a presence
  assertion with no region — but legal.
- `absent` — explicitly asserted not visible in this view. A genuine negative.

**Box encoding.** `[x0, x1) × [y0, y1)`, top-left inclusive, bottom-right exclusive, so width is
`x1 - x0`. This matches the array-slicing convention of the crop rule (`tailcyclenet/crop.py`,
int32 `[x0, y0, x1, y1]`). Floats are allowed — annotators are not pixel-snapped. A box with
`x1 <= x0` or `y1 <= y0` is empty and equivalent to no box. Boxes are not clipped to `size`;
a validator may warn.

`animal_id` is a **cross-view identity**: the same string in `cam088` and `cam091` is the same
physical animal. That is what makes triangulation possible without a separate association step,
and it is checkable — the validator reprojects a triangulated instance and flags ids whose median
residual exceeds `assoc_res_max_px` (rule 12).

## 10. `extrinsics.pq` — moving cameras (optional)

Absent ⇒ every camera is static and its extrinsic comes from `calibration.toml`.

| column | type | req | meaning |
|---|---|---|---|
| `group_id`, `camera` | dictionary\<int32,str\> | ✓ | the key |
| `frame` | int32 | ✓ | |
| `ext` | list\<float64\>[16] | ✓ | 4×4 world→cam, **row-major** |

A camera appearing here must have `moving = true` in `calibration.toml`, and must have a row for
every frame of every group it appears in — a partially-specified moving camera is an error, not
an interpolation request. Intrinsics and distortion stay time-invariant (posetail asserts this).

This maps onto posetail 0.3.0's moving-camera support directly:
`format_camera_group(cgroup, offset_dict, cam_type, device, moving_ext={name: (T,4,4)})`.

## 11. Validation contract

Checkable rules, so `tests/test_format.py` can be written without re-deriving them.

1. Every session has `session.toml` with `mode`, `units` and non-empty `names`. The session id is
   the folder name; the split is its parent folder name.
2. `names` has no duplicates; every `skeleton` / `flip_pairs` name is in `names`; the flip map is
   an involution.
3. **Cross-session agreement:** every session under one dataset root carries identical `names`.
   `mode` and `units` may differ.
4. `calibration.toml` exists, and its camera `name`s are exactly the camera dirs/videos present
   under every group. Every camera has a non-empty `name`, a `size` and an `offset`.
5. `mode = "3d"` ⇒ ≥ 2 cameras, each with `matrix`, `rotation`, `translation`.
   `mode = "2d"` ⇒ exactly 1 camera.
6. Every `bodypart` ∈ `names`; every `camera` ∈ `calibration.toml`; every `group_id` ∈
   `groups.pq`; every `frame` ∈ `[0, n_frames)`.
7. Each group folder has exactly one `<cam>/` dir **or** one `<cam>.mp4` per declared camera —
   not both. An image dir holds exactly `n_frames` files named `%06d.png` or `%06d.jpg`,
   contiguous from `000000`, **one extension per directory**. (Positional and contiguous because
   the loader indexes by `sorted(listdir)[start:end:step]`; mixing extensions would produce two
   files claiming the same index.)
8. Every image's dimensions equal that camera's `size`. Video frame size likewise.
9. No duplicate key in any table: `(group_id, frame, animal_id, camera, bodypart)` for
   `keypoints.pq`, `(group_id, frame, animal_id, bodypart)` for `points3d.pq`,
   `(group_id, frame, animal_id, camera)` for `instances.pq`, `(group_id, frame, camera)` for
   `extrinsics.pq`.
10. A `visible` row has its coordinates: `points3d.pq` requires `x,y,z`; `keypoints.pq` requires
    `x,y` **unless** a `points3d` row exists for the same key. `missing` and `unlabeled` rows have
    null coordinates in both tables.
11. If `instances.pq` exists, every `visible`/`missing` keypoint row has a matching instance row
    with `status = labeled`, and a `labeled` instance has at least one `visible`/`missing` row.
    `unlabeled` rows are exempt — they are progress markers. Fewer than `K` assessed rows is
    legal.
12. `mode = "3d"` with per-camera 2D: every `animal_id` with ≥ 2 `labeled` views triangulates
    with median reprojection residual below `assoc_res_max_px` (default 30.0). Catches cross-view
    id mismatches.
13. Every camera in `extrinsics.pq` has `moving = true`, and covers every frame of every group in
    which it appears.
14. **Consumer-side leak rule:** a session must not be used both for training and for held-out
    evaluation. Enforced at ingest by session *folder name*, not by the files. Roots that differ
    only in `[provenance] annotator` are the carve-out — they share sessions by design.
    rat-city violates the spirit of this by construction (one recording, three frame ranges); the
    validator warns and does not fail, because the alternative is discarding the dataset.

## 12. What the consumer reads

`tailcyclenet/format.py` scatters each session's long tables **once** into dense arrays, in the
parent process, so forked dataloader workers share the pages copy-on-write:

```
animal_ids  (S,)                      str, sorted
points3d    (S, T, K, 3)   float32    NaN where not visible
vis3d       (S, T, K)      int8       -1 unlabeled/absent, 0 missing, 1 visible
points2d    (S, T, K, C, 2) float32   NaN where not visible or not observed
vis2d       (S, T, K, C)   int8       same codes
boxes       (S, T, C, 4)   float32    NaN where no box
instance    (S, T, C)      int8       -1 none, 0 absent, 1 present(ignore), 2 labeled
ext         (C, T, 4, 4)   float64    only when a camera is moving
```

`K` is the length of `names`; `C` the number of cameras; `S` the number of distinct
`animal_id`s in the group; `T` the group's `n_frames`.

`status` → visibility is a vectorized int8 compare on the dictionary codes, not a string
comparison — this is the whole reason `status` is dictionary-encoded. Scatter cost is ~0.5 s for
rat-city's 2.8M rows and ~3 s for branson-fly's 16M.

## 13. Migration

### From `posetail-finetuning-v4`

| v4 | here |
|---|---|
| `<dataset>/<split>/<session>/<trial>/` | `<dataset>/<split>/<session>/groups/<trial>/` — one trial is one group |
| `pose3d.npz['pose'] (S,T,K,3)` | `points3d.pq`; NaN → no row, finite → `visible` |
| `pose2d.npz['pose'] (S,T,K,2)` | `keypoints.pq` with the single camera |
| `pose3d.npz['vis'] (S,T,K,C)` | `keypoints.pq` `status`, `x,y` null (rule 10's exemption) |
| `pose3d.npz['keypoints']` | `session.toml` `names` — but see the column-sort note below |
| `pose3d.npz['ids'] (S,)` | `animal_id`; row index when absent |
| `metadata.yaml` `intrinsic/extrinsic/distortion_matrices` | `calibration.toml` |
| `metadata.yaml` `offset_dict`, `camera_widths/heights` | `calibration.toml` `offset`, `size` |
| `metadata.yaml` `fps`, `num_frames` | `groups.pq` `fps`, `n_frames` |
| `img/<cam>/` or `vid/<cam>.mp4` | `groups/<gid>/<cam>/` or `groups/<gid>/<cam>.mp4` — symlinked |

**allen-mouse's column sort.** Its `pose` array is ordered by `sorted(f'{name}_{axis}')` while
`keypoints` is name-sorted, which transposes all 8 `X` / `X-base` pairs — 16 of 47 keypoints. The
converter applies the repair permutation once and asserts the result. In this format the tables
are name-indexed, so the artifact cannot recur; do not carry a `coord_perm` field forward.

### From `annotated-frames/`

`n_frames = 1` is a fully valid group and is exactly that case: one annotated moment becomes one
single-frame group, `CollectedData.csv` melts to `keypoints.pq` (NaN cells become `missing` rows,
unassessed content becomes no rows), `config.toml`'s `offset = [x0,y0,w,h]` splits into
`calibration.toml`'s `offset` + `size`, `anipose_metadata.csv`'s `framenum` becomes
`source_frame_start`, and `CollectedData_lili.csv` / `_lara.csv` become two dataset roots
distinguished by `[provenance] annotator`.

## 14. Authoring guidance

- **`n_frames = 1` groups are materialised as ≥ 2 frames by the loader.** `T = 1` routes posetail
  down its image path into the `gT = T // tubelet_size = 0` bug
  (`encoder_decoder.py:748`, `tracker_encoder.py:518` in 0.3.0). Authoring `n_frames = 1` is fine;
  the loader duplicates or pads. Real context frames are better than a duplicated one — the
  duplication handicap was measured at **+0.405 mm (+41.8%)**.
- **Prefer ≥ 2 labeled frames per group.** Frame 0's prediction is algebraically independent of
  the triangulation, so it is the one frame where per-frame anchoring contributes nothing. One
  label at frame 0 is admissible but forfeits the anchor on its only label.
- **A label anywhere in the group is usable.** Unlike the old v4 loader — whose
  `get_start_ixs_train` admitted a window only if its *first* frame had a finite coordinate, so a
  group with a centered label yielded zero windows — this format's loader places the window
  itself. Centered labels are the natural shape and are supported.

## 15. Alternatives considered

- **CSV instead of Parquet.** Human-editable and diffable, which genuinely matters for hand
  annotation. Rejected because the same format must hold ~26M rows of dense tracking data, where
  CSV costs ~10× the bytes and a string-compare per `status` lookup. Parquet's dictionary encoding
  gives int8 codes for free. A `pq`↔`csv` converter is three lines of pyarrow if a human needs to
  edit a session by hand.
- **A `split` column.** Rejected: it makes a *consumer* decision look like a property of the data,
  and rule 14 (leak protection) becomes unenforceable when the same session carries both labels.
- **Dense arrays on disk (npz) instead of tables.** Faster to load, but cannot express "no
  determination" distinctly from "assessed missing" without a parallel status array per layer,
  cannot grow a column without a rewrite, and re-introduces exactly the v4 opacity this format
  exists to remove. The dense arrays are a *read* representation (§12), built on load.
- **Root-level `dataset.toml`** carrying keypoint identity and `max_instances`. Removed: with
  identity per session and cross-session agreement as a validation rule, the root level carried
  only repetition.
- **Keeping cameras in `session.toml`** with calibration separate. Removed: two files that must
  agree about the same cameras is a drift surface, and the crop offset is the single most
  consequential number in the whole format (16.9 mm vs 2.38 mm).
- **Per-group self-contained directory** (`group.toml` + tables + calib + frames). Maximally
  movable, but duplicates calibration hundreds of times and makes "enumerate all groups" an O(n)
  directory walk instead of one table read.
- **DLC wide CSV** (row per image, column per bodypart × coord). Familiar, but cannot express
  multiple animals without exploding the header, cannot express a box, stores sparse labels as a
  mostly-NaN dense grid, and `NaN` conflates "assessed missing" with "never looked at" — which is
  precisely what `status` exists to separate.
- **An `annotator` column.** Rejected: two annotators on one point collide with the row key
  (rule 9), and separate roots keep the human-baseline comparison clean.

## 16. Worked examples

### 16.1 Dense 2D multi-animal — rat-city

One recording, 12 rats, one overhead camera, 4 keypoints. Three frame ranges become three
groups in three splits of one session name.

```
rat-city/train/cohort7_20251209_1659/
  session.toml     mode="2d" units="px" names=["nose","left_ear","right_ear","tail_base"]
  calibration.toml  [cam_0] name="cam0" size=[4696,2048] offset=[0,0]
  groups.pq        group_id="ix0" n_frames=57594 fps=40.0 source_frame_start=0
  keypoints.pq     2,764,512 rows
  groups/ix0/cam0 -> …/posetail-finetuning-v3/rat-city/train/cohort7_20251209_1659/ix0/img/cam0
```

```
group_id,frame,animal_id,camera,bodypart,status,x,y,score
ix0,0,a01,cam0,nose,visible,2092.7,169.7,
ix0,0,a01,cam0,left_ear,missing,,,
ix0,0,a01,cam0,tail_base,visible,1979.9,302.4,
```

### 16.2 Native 3D with per-camera visibility — allen-mouse

7 cameras, 1 animal, 47 keypoints, and a real `vis (1,T,47,7)` array with no per-camera 2D. The
3D goes in `points3d.pq`; the visibility goes in `keypoints.pq` as coordinate-free rows.

```
group_id,frame,animal_id,bodypart,status,x,y,z,score          # points3d.pq
g_ix11343,0,a1,nose,visible,12.4,-88.1,203.7,
g_ix11343,0,a1,R-ear,missing,,,,

group_id,frame,animal_id,camera,bodypart,status,x,y,score      # keypoints.pq
g_ix11343,0,a1,cam088,nose,visible,,,                          # visibility observation
g_ix11343,0,a1,cam091,nose,missing,,,
```

### 16.3 Multi-animal with an ignore region

Two animals, two cameras; `m1` is labeled in both views, `m2` is present-but-unannotated in both.
A detector prediction on `m2` is neither a true nor a false positive.

```
group_id,frame,animal_id,camera,x0,y0,x1,y1,status              # instances.pq
g010,0,m1,cam088,120.0,80.0,260.0,240.0,labeled
g010,0,m2,cam088,340.0,300.0,480.0,430.0,present
g010,0,m1,cam091,90.0,60.0,230.0,220.0,labeled
g010,0,m2,cam091,310.0,280.0,450.0,410.0,present

group_id,frame,animal_id,camera,bodypart,status,x,y            # keypoints.pq
g010,0,m1,cam088,nose,visible,190.0,100.0
g010,0,m1,cam088,tail_base,missing,,
g010,0,m1,cam088,R-ear,unlabeled,,                              # annotation-software state
g010,0,m1,cam091,nose,visible,160.0,80.0
g010,0,m1,cam091,tail_base,visible,200.0,200.0
```

`m2` has no keypoint rows at all — consistent with rule 11, since its instance status is
`present`, not `labeled`.

### 16.4 Single frame, no context — a hand-annotated moment

```
group_id,n_frames,fps,source_video,source_frame_start
g000,1,200.0,cam746_2024-03-26T10_34_30.mp4,3439
```

The backwards-compatible path: exactly today's `annotated-frames/` case. The loader materialises
it as 2 frames (§14).

### 16.5 A moving camera

```
[cam_0]                                    # calibration.toml
name = "handheld"
size = [1920, 1080]
matrix = [[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]]
offset = [0, 0]
moving = true                              # rotation/translation ignored; extrinsics.pq wins
```

```
group_id,frame,camera,ext                  # extrinsics.pq
g000,0,handheld,[1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0]
g000,1,handheld,[0.999,-0.017,0.0,1.2, 0.017,0.999,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0]
```
