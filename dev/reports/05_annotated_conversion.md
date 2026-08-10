# allen-mouse-annotated: converting the anivia export

`scripts/convert_annotated.py` converts

```
animal-datasets/allen-mouse-training/motor-observatory-frames-posetail-2026-08-06
  -> animal-datasets-processed/tailcycle-datasets/allen-mouse-annotated
```

```bash
pixi run python scripts/convert_annotated.py --clean --validate
```

The export was already spec-*shaped* — session.toml, calibration.toml, per-group frame dirs — but
not readable by `tailcyclenet.format`. Nine things differed; all nine are closed below.

## What came out

| | train | val |
|---|---|---|
| 3D sessions (7 cameras) | 13 | 4 |
| 2D sessions (1 camera) | 63 | 0 |
| **total** | **76** | **4** |

1,884 symlinks, 1.5 MB of tables and toml, zero pixels copied. `val/` is the four sessions of
`motor-observatory-frames-2025-02-13-eval`; everything else is `train/`.

3D layer: **5,533 visible**, 35 gated out, 178 `missing`. Validation: **0 errors, 0 warnings**.
Round trip over all 80 written sessions: **0 rows lost, 0 invented**.

## The nine gaps

1. **CSV → parquet.** `groups/keypoints/instances.csv` → `write_session`.
2. **No split level** → `train/` and `val/` directories (`--val-root`).
3. **`calibration.toml` `size` was the sensor.** The file said `[1440, 1080]`; the JPEGs are the
   crop (1000×1000, 1080×880, 1160×880 or 1200×800 by session), and the truth lived in
   `session.toml` `[cameras.<c>] image_wh` / `offset`. Now `size` is the image on disk and
   `offset` is the crop origin, which is what rule 8 checks and what spec §5's 16.9-vs-2.38 mm
   lesson is about.
4. **`[cameras.*]` left `session.toml`** — geometry lives in `calibration.toml` only.
5. **`groups.csv label_frames` dropped** (spec §6). `fps` / `source_video` /
   `source_frame_start` / `_step` / `notes` carry over verbatim.
6. **Two keypoint orders in one export.** 24 sessions use one order; three
   (`behavior_715873_2024-02-16_14-39-00`, both `behavior_715875_*`) swap the `RF-*` and `LH-*`
   blocks. Same 47 names, same skeleton, same flip_pairs. Rule 3 wants one order per root, so the
   majority order is written everywhere — the tables are name-indexed, so nothing moves but the
   list.
7. **No 3D layer at all** → triangulated, below.
8. **`motor-observatory-frames-2024-05-12` → 2D**, below.
9. **Pixels symlinked**, one link per camera per group.

## The 3D layer

`mode = "3d"` sessions must carry `points3d.pq`: `dataset.py:278` reads `lab.points3d` and
`format.labels()` does not triangulate, so a keypoints-only 3D session crashes the loader.

Per `(group, frame, animal, bodypart)`, with `offset` added back before triangulating (the 2D is
in stored-image px, the calibration in sensor px) and the stored 2D never touched:

| condition | `points3d.pq` | count |
|---|---|---|
| ≥2 `visible` views, median reprojection residual ≤ 20 px | `visible` + x,y,z | 5,533 |
| ≥2 views but over the gate, **or** exactly one view | `unlabeled` → no row | 35 + 176 |
| 0 views visible **and** every camera assessed the point | `missing` | 178 |
| otherwise | no row | — |

Residuals over the 3D roots: median 3.3 px, p95 9.1 px, p99 17.4 px. The 20 px gate rejects 0.63%
of triangulable points; 10 px would reject 3.9%, 30 px 0.27%. `--max-reproj-px` moves it.

`unlabeled` is written as an absent row, which spec §7 says consumers treat identically ("a
finished file contains none"). If the distinction between *tried and rejected* and *never
assessed* ever needs to survive on disk, `points3d.pq` has to be written through
`fmt.write_table` directly rather than through `write_session`'s dense path.

Anatomy of the result — the check that catches a keypoint-axis mistake, which no aggregate does:

```
nose      - L-ear           median  24.84 mm   n=124
L-ear     - R-ear           median  16.80 mm   n=123
LF-wrist  - LF-index-base   median   3.06 mm   n=106
```

posetail-pose measured that wrist-to-knuckle bone at 2.54 mm under the repaired keypoint ordering
and 4.34 mm under the transposed one. 3.06 mm sits with the repaired one.

**Gotcha for anyone reusing this:** aniposelib on the pytorch branch holds intrinsics as
`nn.Parameter`, so `triangulate` and `reprojection_error` die on *"Can't call numpy() on Tensor
that requires grad"* unless wrapped in `torch.no_grad()`.

## The 2024-05-12 root → 2D

Its cameras were not properly synced, so frame *i* of two views is not one instant and
triangulating them would invent a 3D label. `mode = "2d"` is exactly one camera (rule 5), so each
of the 9 sessions becomes 7 single-camera sessions named `<session_id>__<cam>` — 63 in all, with
`units = "px"`, `keypoints.pq` and `instances.pq` filtered to that camera, and no `points3d.pq`.

Real intrinsics, distortion, extrinsic and `offset` are kept. They are legal in a 2D session, cost
nothing, and let a session be read later as 3D single-view; the 2D training path fixes
`cam_ix = [0]` and never looks at them. What is *not* kept is the cross-camera geometry.

Corroboration, gathered before the split was made: triangulating this root anyway gives median
residuals of 1.2–4.1 px in seven of nine sessions, but `behavior_729499_2024-03-26_10-03-00` and
`behavior_729500_2024-03-26_10-32-00` put `cam712` at **57 px** median.

## Dropped, loudly

A group with no `visible` row carries no supervision, and `dataset._labelled_frames` counts a
`missing` row as labelled — such a group yields index entries `_item` rejects forever, and eight
consecutive rejections raise. So:

- `behavior_750096_2024-08-16_12-55-33` — 3,243 rows, **zero** visible: every point assessed as
  occluded across all 10 groups. Whole session dropped.
- `behavior_729499_2024-03-26_10-03-00` — 10 of 28 groups have no keypoint rows at all (2D root,
  so dropped per camera).
- `behavior_750095_2024-08-05_11-13-16` (val) — 1 of 4 groups.
- Scattered single groups in the 2D root where one camera saw nothing, e.g.
  `behavior_729500_2024-04-05_10-15-35__cam746` drops 3 of 14.

## Two things to know before quoting a number from this

1. **The val axis is within-animal, cross-session.** Animals 750095, 750096 and 769890 appear in
   both splits on different days. No session id is in two splits, so rule 14 passes and there is
   no leak in the format's sense — but this is *not* a cross-animal generalisation measurement,
   and per evaluation rule 1 the axis has to be named on any number taken from it.
2. **Sampling is 87% 2D.** 63 of 76 train sessions are single-camera 2D, and `balance_datasets`
   balances across dataset *roots*, not sessions — a 60-item draw came back 56 × 2D, 4 × 3D. If
   the 3D multiview path is what is being trained, this root wants weighting, or pairing with
   `allen-mouse-tracked`.

## Verification run

- `--validate` → `fmt.validate_dataset(..., check_images=True)`: 0 errors, 0 warnings across 80
  sessions (this opens one image per group per camera and compares against `size`).
- Round trip (on by default, `--no-check` to skip): every source keypoint row re-read through
  `Session.labels()` and compared by `(group, frame, animal, camera, bodypart, status, x, y)` —
  80 sessions, 0 mismatches.
- Loader smoke: `PoseDataset` gives 1,065 train windows / 63 val windows, K = 47, and returns both
  2D items (R = 2, one camera) and 3D items (R = 3, seven cameras) with finite pixels.
- `scripts/train.py --iters 60` against the new root: warm start clean, loss 1.61 → 1.32, 0 items
  skipped.
