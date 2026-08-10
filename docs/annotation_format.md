# `tailcycle-dataset` — a format for animal pose training data

A directory layout and a set of tables for storing labelled pose data: 2D or 3D, single- or
multi-camera, single- or multi-animal, hand-annotated or machine-derived. This document is the
specification; there is no version field anywhere in the format, because this document is the
version.

## 1. Purpose

Pose training data usually arrives in one of two shapes, and neither can do the other's job:

- **Isolated labelled frames** — a wide CSV of body-part coordinates, one row per image, plus a
  calibration file and a folder of stills. Easy to author and to read, but it has no temporal
  context, usually assumes one animal, and cannot say "an animal is here but nobody labelled it".
- **Derived dense arrays** — a `(instances, frames, keypoints, coords)` binary blob per clip.
  Multi-animal and temporal, but opaque: no provenance, no per-camera observations, no boxes, and
  a single `NaN` that conflates "assessed and occluded" with "never looked at".

The difference between a hand-labelled session and a machine-tracked one is **how many rows carry
labels**, not what shape the data has. This format encodes that and nothing else.

## 2. Decisions (locked)

1. **Split is a directory level, not a field.** `<dataset>/<split>/<session>/`. Train/val/test
   stays a *consumer* concept — there is no `split` column anywhere — but it is expressed as a
   path, because datasets frequently do not split cleanly by session. A single recording split
   into train, val and test frame ranges is common, and no session-level field can express it.
2. **A group is a contiguous clip.** `n_frames` is unbounded; `n_frames = 1` is legal and means
   "no context". A 57,000-frame recording is one group.
3. **Multi-animal from the start.** `animal_id` is a cross-view, cross-frame identity.
4. **The 3D layer is first-class.** A session may carry native 3D, native per-camera 2D, or both.
   3D is *not* required to be derived from 2D.
5. **`mode` is a per-session property.** One `train/` may hold 2D and 3D sessions together.
   Sessions need not agree on `names` either: the keypoint axis is per session, and a consumer
   resolves it **by name** against the root's union (§4, rule 3).
6. **`status` is the visibility channel**, in every label table. There is no separate `vis`
   column and no NaN-means-something convention.
7. **Sparsity is a row count, not a mode.** A row exists only where a determination was made; a
   missing row means "no determination". A dense session simply has a row everywhere.
8. **Partial assessment is legal.** Completeness is a read-time policy; the format never
   enforces it.
9. **A box per animal per view, always optional** — for `labeled`, `present` and `absent` alike.
   When given it is a human judgment enclosing the animal; it is *not* required to match the
   keypoint extent or any particular crop rule.
10. **No `annotator` column.** Multiple annotators = separate dataset roots, the annotator named
    in `session.toml` `[provenance]`. Two annotators on one point would collide with the row key.
11. **All camera geometry lives in `calibration.toml`**, including the crop. `session.toml` is
    keypoint identity and provenance only.
12. **Tables are Parquet.** `.pq`, tidy long form, one row per assessed thing.

## 3. Directory layout

```
<dataset_root>/
  <split>/                            # train | val | test
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

Either the image directory or the video may be a **symlink**, and a converter is encouraged to
symlink whole directories rather than copy or link individual frames. A dense dataset then costs
one link per camera per group instead of one per frame — hundreds of links rather than hundreds
of thousands, for the same hundreds of gigabytes of pixels.

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
source         = "..."          # where this session came from
annotator      = ""             # empty when a single annotator authored the root
annotator_tool = "..."
created        = "YYYY-MM-DD"
```

There is no dataset `name`, no `session_id` and no `fps`: the folder name is the session id, and
fps is declared per group (§6).

- **Keypoint names live here, per session, and `names` is that session's keypoint axis.** Sessions
  under one dataset root need not carry identical `names`: a session may list the same keypoints
  in a different order, or only a **subset** of them. The root's axis is the **union** of its
  sessions' names, and a consumer maps each session onto it by name. `mode` and `units` may differ
  too — that is what lets a `train/` mix 2D and 3D.

  A session must, however, hold up its own end: a `bodypart` in a table that is not in that
  session's `names` is an error (rule 6). Dropping a keypoint from `names` means dropping its
  rows. Validation rule 3 reports the disagreements it sees, because a "missing" keypoint is more
  often a typo than a decision.
- The tables are **name-indexed**, which is what makes a keypoint-axis ordering mistake
  impossible to carry forward, and what makes the mixed-`names` root above safe rather than
  merely tolerated. See §13 for why that matters.
- `skeleton` and `flip_pairs` are optional and expected to be empty for most data — they are
  needed only for bone-length diagnostics and bilateral flip augmentation. When present, every
  name must be in `names`, and `flip_pairs` lists each pair **once** (the involution is its
  symmetric closure).
- `assoc_res_max_px` (optional, default `30.0`) is the reprojection-residual gate used by
  validation rule 12. Rigs with high-resolution sensors need a looser value.

## 5. `calibration.toml`

An [aniposelib](https://github.com/lambdaloop/aniposelib) `CameraGroup` file with **two** added
per-camera keys, `offset` and `moving`. It is **always present**, including for `mode = "2d"`, so
there is exactly one place camera geometry lives. Everything else is aniposelib's own schema, so
an aniposelib-based tool can read this file directly; lens model is aniposelib's existing
`fisheye` flag, not a new field.

```toml
[cam_0]
name        = "cam088"
size        = [1200, 800]      # size of the image ON DISK; asserted against the files
matrix      = [[1852.4, 0.0, 719.5], [0.0, 1852.4, 539.5], [0.0, 0.0, 1.0]]
distortions = [-0.0361, 0.0, 0.0, 0.0, 0.0]
rotation    = [0.0041, -0.0017, 0.0045]     # Rodrigues, world -> cam
translation = [2.5517, -1.2033, 310.44]
fisheye     = false            # aniposelib's own flag; picks its FisheyeCamera class
offset      = [120, 140]       # origin of the stored image inside the sensor frame (ADDED)
moving      = false            # per-frame extrinsics live in extrinsics.pq         (ADDED)

[metadata]
```

**`size` is the stored image; `matrix` is in sensor coordinates.** Projecting a world point is
therefore "apply `matrix`, then subtract `offset`", and un-projecting adds `offset` back. Note
the example: a principal point of 719.5 belongs to a 1440-wide sensor, while the file on disk is
1200 wide at `offset = [120, 140]`. There is deliberately **no sensor-size field**: nothing
consumes it, and a field nobody reads is a field that silently goes wrong.

**Why the crop lives with the calibration.** `offset` is the only thing relating the pixels on
disk to the sensor the calibration describes, so it belongs with the calibration rather than in a
second file that can drift from it. This is not a stylistic preference: on a seven-camera rodent
rig, dropping the offset turned 2.38 mm of inter-annotator 3D disagreement into **16.9 mm**. A
*common* offset does not cancel between annotators — it makes the rays non-intersecting, and
triangulation amplifies a 2 px labelling difference into roughly 12 mm.

For `mode = "2d"`: one camera, `offset = [0, 0]`, and `matrix` / `rotation` / `translation` /
`distortions` may all be omitted — a consumer then substitutes a nominal pinhole (focal length
`max(W, H)`, identity extrinsic). Writing real values for a single calibrated camera is legal and
lets the same session also be read as 3D single-view.

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
label tables, and a consumer must scan them to build a validity mask anyway.

## 7. `keypoints.pq` — per-camera 2D and per-camera visibility

One row per **assessed** (group, frame, animal, camera, bodypart).

| column | type | req | meaning |
|---|---|---|---|
| `group_id`, `animal_id`, `camera`, `bodypart` | dictionary\<int32,str\> | ✓ | the key |
| `frame` | int32 | ✓ | 0-based index into the group |
| `status` | dictionary\<int8,str\> | ✓ | `visible` \| `missing` \| `unlabeled` |
| `x`, `y` | float32 | | stored-image px; **null** unless `visible`. See the rule below. |
| `score` | float32 | | confidence in [0,1]; null for human labels |

- **`visible`** — the point is there. `x,y` are required **unless** a `points3d` row exists for
  the same `(group_id, frame, animal_id, bodypart)`, in which case the row is a *visibility
  observation*: "visible in this camera; the position is in the 3D layer". This exists because
  some pipelines produce a genuine per-camera visibility array with no per-camera 2D at all, and
  reprojecting the 3D to manufacture an `x,y` would present a derived number as an observation.
- **`missing`** — looked at, judged occluded or outside the frame. `x,y` are null: an occluded
  point is not annotated. The occluded-vs-outside distinction is deliberately collapsed; it is
  resolved downstream by reprojecting derived 3D whenever 3D exists.
- **`unlabeled`** — explicitly recorded as not assessed. Consumers treat it **exactly like an
  absent row**. It exists so annotation software can persist progress state in the file the
  annotator is editing. A finished file contains none.
- **No row** — not labelled. Semantically identical to `unlabeled` for consumers.

Empty coordinates are written as parquet **nulls**, not as `NaN`: a null costs a validity bit
rather than four bytes, and any reader sees the absence rather than a sentinel it has to know
about.

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

A session may carry `points3d.pq` alone, `keypoints.pq` alone, or both (a calibrated multi-view
session with real 2D labels *and* a triangulated 3D solution). The consumer derives what it
needs: 2D from 3D by projection, 3D from 2D by triangulation. Neither derivation is stored.

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
  prediction landing here is neither a true nor a false positive. This exists because it is
  measurable: on a multi-animal rodent recording, **73% of a tracker's false positives were real
  animals the annotator had skipped**, and excluding them moved MOTA from 0.692 to 0.709.
- `absent` — explicitly asserted not visible in this view. A genuine negative.

**Box encoding.** `[x0, x1) × [y0, y1)`, top-left inclusive, bottom-right exclusive, so width is
`x1 - x0`. This matches the usual array-slicing convention, so an integer box can be used as a
crop directly. Floats are allowed — annotators are not pixel-snapped. A box with `x1 <= x0` or
`y1 <= y0` is empty and equivalent to no box. Boxes are not clipped to `size`; a validator may
warn.

`animal_id` is a **cross-view identity**: the same string in two cameras is the same physical
animal. That is what makes triangulation possible without a separate association step, and it is
checkable — a validator reprojects a triangulated instance and flags ids whose median residual
exceeds `assoc_res_max_px` (rule 12).

## 10. `extrinsics.pq` — moving cameras (optional)

Absent ⇒ every camera is static and its extrinsic comes from `calibration.toml`.

| column | type | req | meaning |
|---|---|---|---|
| `group_id`, `camera` | dictionary\<int32,str\> | ✓ | the key |
| `frame` | int32 | ✓ | |
| `ext` | list\<float64\>[16] | ✓ | 4×4 world→cam, **row-major** |

A camera appearing here must have `moving = true` in `calibration.toml`, and must have a row for
every frame of every group it appears in — a partially-specified moving camera is an error, not
an interpolation request. Intrinsics and distortion stay time-invariant; a rig with a zoom lens
is outside this format as written.

## 11. Validation contract

Checkable rules, so a validator can be written without re-deriving them.

1. Every session has `session.toml` with `mode`, `units` and non-empty `names`. The session id is
   the folder name; the split is its parent folder name.
2. `names` has no duplicates; every `skeleton` / `flip_pairs` name is in `names`; no keypoint
   appears in two flip pairs with different partners, and no pair maps a name to itself.
3. **Cross-session agreement (WARNING, not an error):** a session whose `names` are a reordering
   of, or a subset of, the root's union is reported and then resolved by name. `mode` and `units`
   may differ. There is no failing case here — a name no other session has simply widens the
   union, which is why the warning names it.
4. `calibration.toml` exists, and its camera `name`s are exactly the camera dirs/videos present
   under every group. Every camera has a non-empty `name`, a `size` and an `offset`.
5. `mode = "3d"` ⇒ ≥ 2 cameras, each with `matrix`, `rotation`, `translation`.
   `mode = "2d"` ⇒ exactly 1 camera.
6. Every `bodypart` ∈ `names`; every `camera` ∈ `calibration.toml`; every `group_id` ∈
   `groups.pq`; every `frame` ∈ `[0, n_frames)`.
7. Each group folder has exactly one `<cam>/` dir **or** one `<cam>.mp4` per declared camera —
   not both. An image dir holds exactly `n_frames` files named `%06d.png` or `%06d.jpg`,
   contiguous from `000000`, **one extension per directory**. (Positional and contiguous because
   a consumer indexes by `sorted(listdir)[start:end:step]`; mixing extensions would produce two
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
    with median reprojection residual below `assoc_res_max_px`. Catches cross-view id mismatches.
13. Every camera in `extrinsics.pq` has `moving = true`, and covers every frame of every group in
    which it appears.
14. **Consumer-side leak rule:** a session must not be used both for training and for held-out
    evaluation. Enforced at ingest by session *folder name*, not by the files. Roots that differ
    only in `[provenance] annotator` are the carve-out — they share sessions by design. A dataset
    whose splits are frame ranges of one recording violates the spirit of this by construction; a
    validator should **warn rather than fail**, because the alternative is discarding the dataset.

## 12. What a consumer reads

A reading path is expected to scatter each session's long tables **once** into dense arrays, and
to do it before forking any worker processes so the arrays are shared rather than duplicated:

```
animal_ids  (S,)                       str, sorted
points3d    (S, T, K, 3)    float32    NaN where not visible
vis3d       (S, T, K)       int8       -1 unlabeled/absent, 0 missing, 1 visible
points2d    (S, T, K, C, 2) float32    NaN where not visible or not observed
vis2d       (S, T, K, C)    int8       same codes
boxes       (S, T, C, 4)    float32    NaN where no box
instance    (S, T, C)       int8       -1 none, 0 absent, 1 present(ignore), 2 labeled
ext         (C, T, 4, 4)    float64    only when a camera is moving
```

`K` is the length of `names`; `C` the number of cameras; `S` the number of distinct `animal_id`s
in the group; `T` the group's `n_frames`.

`status` → visibility is a vectorized int8 compare on the parquet dictionary codes, not a string
comparison — this is the whole reason `status` is dictionary-encoded, and it is why the format is
parquet rather than CSV. Scattering costs well under a second for a few million rows.

**A note on the third state.** Many training losses accept only a two-state visibility target. If
yours does, be careful how you collapse `unlabeled`: mapping it to "not visible" trains the model
on an assertion nobody made. Mapping it to a masked-out value is correct, and a loss that
silently accepts `NaN` may not do what you expect — check that the *gradient* survives, not just
that the loss is finite.

## 13. Converting from an array format

Most existing pose data is a `(instances, frames, keypoints, coords)` array plus a metadata file.
The mechanical parts are easy; these are the parts that bite.

- **Verify the keypoint axis order against the name list, do not assume it.** A writer that
  builds the coordinate array from `sorted(column_names)` but the name list from
  `unique(bare_names)` produces two *different* orders wherever one name is a prefix of another —
  `-` sorts before `_`, so `X-base_x` precedes `X_x` as a column while `X` precedes `X-base` as a
  name. That silently transposes every such pair. It is invisible in any aggregate statistic;
  what exposes it is anatomy. On a rodent skeleton, the wrist-to-knuckle bone measured 2.54 mm
  under the repaired ordering and 4.34 mm under the raw one — the raw ordering had wired the
  wrist to the fingertip. Note that swapping *both* endpoints of an edge leaves its length
  unchanged, so the fingertip-to-knuckle bones looked identical either way; only edges with one
  endpoint in a transposed pair move.
- **A NaN is not a `missing` row.** If the source encodes only "no value", write **no row**.
  Writing `missing` claims someone looked and judged the point occluded, which is an annotation
  that was never made. Reserve `missing` for a source that genuinely records assessment.
- **Symlink the pixels.** One link per camera per group; do not copy, and do not link per frame.
- **Check the frame counts.** Array length and image count disagreeing by one is common; truncate
  to the pixels and report it per group rather than padding silently.
- **A fully-unlabelled clip should be dropped, loudly.** It will otherwise be a group every
  consumer rejects, for reasons no one can see from the files.
- **Calibration that differs between clips means they are different sessions.** Calibration is a
  session property in this format; clips that disagree about it cannot share a session.

### From isolated labelled frames

`n_frames = 1`, one group per annotated moment, is a fully valid session and is exactly this
case: a wide CSV melts to `keypoints.pq` (NaN cells become `missing` rows, unassessed content
becomes no rows), a crop rectangle `[x0, y0, w, h]` splits into `calibration.toml`'s `offset` +
`size`, a source frame number becomes `source_frame_start`, and two annotators' files become two
dataset roots distinguished by `[provenance] annotator`.

## 14. Authoring guidance

- **`n_frames = 1` is legal, but give context frames when you can.** Some video encoders cannot
  process a single frame at all and require a consumer to duplicate or pad it; and a duplicated
  frame is a measurable handicap, not just an inelegance — on one 3D rodent dataset, duplicating
  a single labelled frame instead of supplying real context cost **+0.405 mm (+41.8%)**.
- **A label anywhere in the group is usable, and a centered label is the preferred shape.** A
  consumer is expected to place its window *around* the label rather than requiring the label at
  the window's edge. Beware of loaders that admit a window only if its *first* frame carries a
  label — under that rule a group with a centered label yields **zero** windows, silently.
- **Frame 0 is the weakest place to put a lone label** for any model that refines a prediction
  against its own temporal context, since frame 0 has none preceding it.
- **Context frames need no labels of their own.**

## 15. Alternatives considered

- **CSV instead of Parquet.** Human-editable and diffable, which genuinely matters for hand
  annotation. Rejected because the same format must hold tens of millions of rows of dense
  tracking data, where CSV costs ~10× the bytes and a string compare per `status` lookup.
  Parquet's dictionary encoding gives int8 codes for free. A `pq`↔`csv` converter is a few lines
  if a human needs to edit a session by hand.
- **A `split` column.** Rejected: it makes a *consumer* decision look like a property of the
  data, and rule 14 (leak protection) becomes unenforceable when the same session carries both.
- **Dense arrays on disk instead of tables.** Faster to load, but cannot express "no
  determination" distinctly from "assessed missing" without a parallel status array per layer,
  cannot grow a column without a rewrite, and re-introduces exactly the opacity this format
  exists to remove. The dense arrays are a *read* representation (§12), built on load.
- **Root-level `dataset.toml`** carrying keypoint identity and instance counts. Removed: with
  identity per session and cross-session agreement as a validation rule, the root level carried
  only repetition.
- **Keeping cameras in `session.toml`** with calibration separate. Removed: two files that must
  agree about the same cameras is a drift surface, and the crop offset is the single most
  consequential number in the format (16.9 mm vs 2.38 mm, §5).
- **Per-group self-contained directory** (`group.toml` + tables + calibration + frames).
  Maximally movable, but duplicates calibration hundreds of times and makes "enumerate all
  groups" an O(n) directory walk instead of one table read.
- **A wide CSV** (row per image, column per bodypart × coord). Familiar, but cannot express
  multiple animals without exploding the header, cannot express a box, stores sparse labels as a
  mostly-NaN dense grid, and `NaN` conflates "assessed missing" with "never looked at" — which is
  precisely what `status` exists to separate.
- **An `annotator` column.** Rejected: two annotators on one point collide with the row key
  (rule 9), and separate roots keep a human-vs-human comparison clean.

## 16. Worked examples

### 16.1 Dense 2D, multi-animal, one camera

One long overhead recording, 12 animals, 4 keypoints. Three frame ranges become three groups in
three splits that share a session name — which rule 14 warns about and does not reject.

```
arena-2d/train/rec_20251209/
  session.toml      mode="2d" units="px" names=["nose","left_ear","right_ear","tail_base"]
  calibration.toml  [cam_0] name="cam0" size=[4696,2048] offset=[0,0]
  groups.pq         group_id="clip0" n_frames=57594 fps=40.0 source_frame_start=0
  keypoints.pq      2,764,512 rows
  groups/clip0/cam0 -> (symlink to the frame directory)
```

```
group_id,frame,animal_id,camera,bodypart,status,x,y,score
clip0,0,a01,cam0,nose,visible,2092.7,169.7,
clip0,0,a01,cam0,left_ear,missing,,,
clip0,0,a01,cam0,tail_base,visible,1979.9,302.4,
```

### 16.2 Native 3D with per-camera visibility

7 cameras, 1 animal, 47 keypoints, and a real per-camera visibility array with no per-camera 2D.
The 3D goes in `points3d.pq`; the visibility goes in `keypoints.pq` as coordinate-free rows —
the §7 exemption to rule 10.

```
group_id,frame,animal_id,bodypart,status,x,y,z,score          # points3d.pq
g000,0,a1,nose,visible,12.4,-88.1,203.7,
g000,0,a1,R-ear,missing,,,,

group_id,frame,animal_id,camera,bodypart,status,x,y,score      # keypoints.pq
g000,0,a1,cam088,nose,visible,,,                               # visibility observation
g000,0,a1,cam091,nose,missing,,,
```

If that visibility came from a detector confidence threshold rather than a human, say so in
`[provenance]` and carry the confidence in `score`.

### 16.3 Multi-animal with an ignore region

Two animals, two cameras; `m1` is labelled in both views, `m2` is present-but-unannotated in
both. A detector prediction on `m2` is neither a true nor a false positive.

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

The backwards-compatible path: one annotated moment, one group. A consumer materialises it as
≥ 2 frames if its model needs them (§14).

### 16.5 A moving camera

```toml
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
