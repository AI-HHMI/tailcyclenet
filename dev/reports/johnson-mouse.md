# johnson-mouse → tailcycle format

`scripts/convert_johnson.py`, run 2026-08-10.

    src  /groups/karashchuk/karashchuklab/animal-datasets/johnson-mouse/merge
    out  /groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/johnson-mouse-annotated

    pixi run python scripts/convert_johnson.py --clean

## What landed

| | train | val |
|---|---|---|
| sessions (= trials) | 21 | 21 |
| groups | 623 | 311 |
| frames (= framesets) | 3,243 | 371 |
| labelled views | 51,473 | 5,883 |

16 cameras, 3208×2200, 24 keypoints, one animal (`a00`), `mode="3d"`, `units="mm"`.
57,824 symlinks, 0 broken, 7.1 MB of metadata on top of the untouched source pixels.
`validate_dataset`: **0 errors**, 21 warnings (all `[rule 14 WARNING]`, below).

Group size: 934 groups over 3,614 frames, median T=1, max T=55, 605 groups at T=1, and **63% of
frames sit in groups of T≥8**.

## Verification

- **Reprojection, read back from disk** (`--check`, on by default): median **0.58–0.63 px** on
  every one of the 42 sessions, p95 ≤ 1.22, p99 ≤ 1.50. Both the 3D and the calibration are
  re-read from the written `points3d.pq` / `calibration.toml`, so a mangled toml round-trip fails
  here too — the wrong transpose convention reads as 1684 px, not 0.6.
- **Boxes and 2D compared exhaustively against the source JSON**: all 57,356 boxes and all
  1,376,526 visible 2D points match bit-for-bit as float32.
- **Rule 11 checked by hand** (the validator does not implement it): the 57,356 `labeled`
  instance rows are exactly the 57,356 `(group, frame, animal, camera)` keys carrying keypoint
  rows, in every session.
- **Loader smoke test**: `PoseDataset(root, 'train', LoaderConfig(n_frames=8))` builds 623 train /
  311 val items; an item yields 16 views, `coords (8,24,3)`, `vis_2d (8,24,16)`, 16-camera cgroup.
- `pixi run test`: 77 passed, 1 skipped (nothing in `tailcyclenet/` changed).

## Four judgement calls

**1. The 3D layer is derived.** The source has no 3D at all, but a `mode="3d"` session is
unusable without `points3d.pq` — `dataset.py:278` and `detector/data.py:96` both index
`lab.points3d` unconditionally, and the failure is an opaque `RuntimeError: 8 consecutive items
failed to build`. `mode="2d"` demands exactly one camera. So `points3d.pq` is DLT-triangulated
from the 16-view 2D and `provenance.points3d_source` says so. It is not an independent
measurement and should never be quoted as one; at 0.6 px over 16 views it is nonetheless
better-determined than most datasets' shipped 3D.

**2. Groups are runs, not single frames.** Annotated frames arrive in strided runs — modal source
step 1, 4 or 6 depending on trial — separated by gaps of hundreds to tens of thousands of frames.
Cutting a group wherever the gap exceeds 8 source frames (`--max-gap`) gives the numbers above.
One group per frameset would have been 3,614 groups of T=1, and the loader would then read the
same JPEG 24× per item (`dataset.py:242`); the format doc measures that at +0.405 mm / +41.8%
cost. `--max-gap 30` would push T≥8 coverage to 76% but stitches across ~0.5 s jumps, which a
temporal model sees as a cut.

**3. 240 triangulated points are gated out.** The first 10 framesets of
`train/2025_02_12_10_12_30` (source frames 11490–11526) have all 16 views disagreeing by
200–2400 px, while frame 11 of the same group sits at 0.62 px. That is a source-side problem —
mis-synchronised or misassigned views — not a converter one. Because the 3D here is *derived*, it
carries a quality gate the 2D does not: a point whose median reprojection exceeds
`--gate-reproj-px` (default 10) gets no 3D row. **The 2D behind those points is exported
verbatim** — it is the annotators' work, not ours to censor. Without the gate that trial's p99 was
230 px; with it, 1.22 px. Worth a look by whoever owns the source.

**4. Unannotated views get no instance row.** 415 train / 53 val images belong to a frameset but
carry no annotation. They are symlinked (the format requires every camera to have `n_frames`
images) and left `unlabeled` — not `present`. `present` would claim the mouse is in view, and the
source records no such determination. It is 0.8% of images, so the lost ignore-region benefit is
negligible.

## Three source traps this converter works around

1. **`cv2.FileStorage` silently returns garbage for 13 of the 21 calibration trials.** Every trial
   from `2026_04_20_15_30_41` onward writes bare integers (`0`, not `0.`) under `dt: d`, and
   OpenCV parses them into ~2³² and −5.9e18. `parse_calib` is a regex reader instead. The existing
   `posetail_preprocessing/datasets/johnson_mouse.py:camera_from_path` uses `cv2.FileStorage` and
   would corrupt those 13 trials with no symptom.
2. **All five distortion coefficients matter.** The same reference converter does `dist[2:] = 0`,
   dropping p1/p2/k3. That is harmless on the legacy 2024_11_26 calibration (only k1,k2 nonzero)
   and costs 7.7 px on 2025_10_20. Note that aniposelib's `get_dict` writes `n_dist = 2` into
   `calibration.toml` alongside all five coefficients; the round-trip through `load_calibration`
   preserves all five — verified.
3. **1,871 ghost jpgs** sit on disk unreferenced by the current annotations (leftovers from a
   cleanup; the `.bak_ghost_jpgs` files are the pre-cleanup versions). The converter is driven
   entirely from the JSON `images` list; a filesystem walk would enrol them as unlabelled frames.

Calibration convention, verified against all four transpose combinations on all 21 trials:
`matrix = intrinsicMatrix.T`, `rvec = Rodrigues(R.T)`, `tvec = T`, `offset = (0,0)` (full sensor,
no crop). The three wrong combinations land at 146–2107 px.

## Known gaps

- **`fps` is NaN.** The `merge` tree carries no frame rate anywhere; the mp4s and `_meta.csv` live
  in sibling legacy directories that no longer correspond to these trials. Inventing one would
  make `source_frame_step` look like a duration.
- **Rule 14 warnings are expected and honest.** All 21 trial names appear under both `train/` and
  `val/`, because the source split is per-frameset, not per-session — the user asked for the
  splits to match the source. The partition is leakage-free at the frame level (0 overlapping
  framesets, 0 overlapping filenames) but *not* at the session level, so any number from this
  dataset is **within-session**, and must be labelled as such (eval rule 1).
- **No `configs/datasets/johnson-mouse.toml`.** Add one when a detector is trained here; 3208×2200
  is 1.46:1, so the square-letterbox trap in CLAUDE.md applies to `--input-wh`.
- 0.025% of source keypoints fall outside the image; they are exported with their real
  (occasionally negative) coordinates and left to the crop rule.
