# johnson-mouse-tracked → tailcycle format

`scripts/convert_v4.py`, run 2026-08-10.

    src  /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4/johnson-mouse
    out  /groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/johnson-mouse-tracked

    pixi run python scripts/convert_v4.py --dataset johnson-mouse --clean --validate

The dense machine-tracked counterpart of `johnson-mouse-annotated`. Same 16-camera 3208×2200 rig,
same 24 keypoint names, one animal (`a00`), `mode="3d"`, `units="mm"`, fps 180.

## What landed

| split | session | group | T | source frames | 3D coverage |
|---|---|---|---|---|---|
| train | `mouse1` | `2024_11_25_13_52_32_ix0` | 8,000 | 0–7999 | 0.826 |
| val | `mouse1` | `2024_11_25_13_52_32_ix8050` | 32 | 8050–8081 | 0.897 |
| test | `mouse1` | `2024_11_25_13_52_32_ix9000` | 950 | 9000–9949 | 0.863 |

8,982 frames × 16 cameras = 143,712 images, **48 symlinks** (one per camera per group, into
`posetail-finetuning-v3`), 2.9 MB of tables and toml, zero pixels copied. 178,851 `visible` rows in
`points3d.pq`.

No `keypoints.pq` and no `instances.pq`: the source has no per-camera 2D and — unlike allen-mouse —
no per-camera `vis` array, so there is nothing honest to put in either.

`validate_dataset(check_images=True)`: **0 errors, 2 `[rule 14 WARNING]`s** (`mouse1` in
train/val, val/test).

## Verification

`scratch/check_johnson_tracked.py`, all four green:

1. **Calibration round-trip.** `points3d.pq` re-read from disk, projected through the written
   `calibration.toml`: **100.00% of points in-bounds in all 16 cameras** on every group, median
   landing at ≈(1590, 990) on a 3208×2200 sensor. A mangled toml round-trip or a wrong extrinsic
   convention is what this catches, and it is the only place it would show — there is no 2D to
   reproject against.
2. **Row count.** 158,496 / 689 / 19,666 `visible` rows written vs the same counts of finite poses
   in the source npz. Nothing invented, nothing lost.
3. **Anatomy vs the annotated sibling root** (below).
4. **Loader smoke.** `PoseDataset(root, split, LoaderConfig(n_frames=8))`: 1 train index entry
   (train indexes at animal granularity and draws the start per epoch — 1 group × 1 animal), 4 val
   windows, 16 views of `(8,256,256,3)`, `coords (8,24,3)`, 16-camera cgroup, all pixels finite.

`pixi run test`: 77 passed, 1 skipped. All four pre-existing v4 datasets still `--dry-run` clean.

## Anatomy — and what it says about the labels

Bone medians, this root vs `johnson-mouse-annotated` (hand-labelled, 21 trials of the same rig):

```
edge                 tracked  annotated   delta
EarL-Neck              13.61      14.59   -0.98
Neck-SpineL            26.67      33.03   -6.36
SpineL-TailBase        24.74      25.11   -0.38
ShoulderL-ElbowL       16.33      16.37   -0.04
ElbowL-WristL          11.88      13.71   -1.82
WristL-HandL            1.27       3.11   -1.84
ElbowR-WristR          10.89      14.02   -3.13
KneeL-AnkleL           18.95      19.75   -0.80
AnkleL-FootL            7.73      10.19   -2.46
TailBase-Tail1Q        18.57      19.13   -0.56
```

Two readings, and the distinction matters:

- **The keypoint axis is right.** A transposition moves the edges with one transposed endpoint by a
  lot and leaves the rest untouched (spec §13, and how the allen bug was caught). What is here
  instead is a *uniform* shrinkage across all 20 comparable edges, −0.04 to −6.4 mm, with the
  absolute values all at mouse scale. That is not what a transposition looks like. It is also
  independently confirmed: `npz['keypoints']` is now asserted equal to the spec `names` at
  conversion time, and `column_sort_perm` is the identity for these names.
- **The tracked labels are smoothed toward the body.** Every bone is shorter than the human
  measurement, worst on the two that need the most 3D precision: `Neck-SpineL` (−6.4 mm) and the
  distal limb, where `WristL-HandL` collapses from 3.11 mm to 1.27 mm — the tracker is putting the
  hand nearly on top of the wrist. Different recordings, so this is not a controlled comparison,
  but it is the expected regression-to-mean of a network's 3D output and it is a reason to treat
  this root as a consistency target rather than as ground truth.

## Two judgement calls

**1. `Snout` and `TailTip` are all-NaN in all three trials.** 22 of 24 keypoints carry data. Both
names stay in `names` because the pose axis is 24 wide and positions must line up; they simply get
no rows, which is exactly what the format says an unassessed point is. Anything trained here gets
no gradient on those two.

Sparsest keypoints that do carry data (train): `HandL` 0.56, `WristR` 0.61, `HandR` 0.63,
`ElbowL` 0.73, `WristL` 0.78. Everything else is ≥0.87.

**2. `source_frame_start` now comes from the trial name.** v4 cuts one recording into
`<recording>_ix<N>` trials, where N is the source frame of the trial's frame 0. `convert_v4` used
to hardcode `0`, which for `..._ix9000` is a false statement about the provenance — and here the
offsets are precisely what shows the three splits do not overlap. A `_ix(\d+)$` regex on the group
id, defaulting to 0, so no other dataset changes.

## Three converter fixes this dataset forced

1. **`allow_pickle=True` on the npz loads.** johnson writes `keypoints` as `dtype('O')` where
   allen writes `<U14` and 3dpop `<U16`; touching it under the numpy default raises
   `ValueError: Object arrays cannot be loaded when allow_pickle=False`. `npz.files` membership
   (the `'ids' in npz` probe) is unaffected, but the flag is set there too.
2. **The `npz keypoints == spec names` assert is now unconditional.** It used to live inside the
   `if spec.get('npz_column_sorted')` branch, so a dataset that ships a name list and does *not*
   set that unrelated repair flag — johnson and 3dpop, both — was never checked against its spec at
   all. This is the assert that makes gotcha 4 impossible to carry forward; it has nothing to do
   with the allen column-sort repair. Verified green on all five datasets.
3. **`out_name` in the dataset spec.** `allen-mouse-tracked` on disk had been renamed by hand —
   nothing in the repo produced that path, so re-running `--dataset allen-mouse` would have written
   a *second*, divergent root at `tailcycle-datasets/allen-mouse`. `out_dir()` now resolves
   `spec.get('out_name', name)`, and `configs/datasets/allen-mouse.toml` declares it. The `--clean`
   and `--validate` paths go through the same helper.

## Before quoting a number from this root

1. **It is within-session, within-animal.** Train, val and test are frame ranges 0–7999 /
   8050–8081 / 9000–9949 of one 10,000-frame recording of one mouse. Leak-free at the frame level;
   not at the session level, and not at the animal level at all. Rule 14 warns rather than fails
   for exactly this shape, and evaluation rule 1 means the axis has to be named on any number.
2. **The labels are tracked, not annotated.** `points3d.pq` here is a tracker's output — see the
   bone shrinkage above. The hand-annotated axis for this rig is `johnson-mouse-annotated`; a
   cross-animal, cross-session number needs that root, not this one.
3. **The val split is 32 frames.** Four non-overlapping 8-frame windows. That is a smoke test, not
   a measurement.

## Known gaps

- **No `configs/datasets/johnson-mouse.toml` detector entry.** 3208×2200 is 1.46:1, so the
  square-letterbox trap in CLAUDE.md applies to `--input-wh` when a detector is trained here. Same
  note as the annotated root.
- **The two johnson roots use different keypoint *orderings*** — this one is the npz's name-sorted
  order (which `convert_v4` indexes `pose` against positionally), the annotated root is anatomical.
  Same 24 names, same skeleton, same flip pairs. Cosmetic: the registry prefixes ids by dataset
  folder name, so the two roots get separate embedding rows either way.
