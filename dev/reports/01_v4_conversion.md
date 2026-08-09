# Converting posetail-finetuning-v4 → tailcycle-dataset

`scripts/convert_v4.py`, run 2026-08-09. Output:
`/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/`

## Result

| dataset | mode | cams | sessions | groups | frames | K | S |
|---|---|---|---|---|---|---|---|
| rat-city | 2d | 1 | 3 | 3 | 58,158 | 4 | 12 |
| allen-mouse | 3d | 7 | 3 | 45 | 22,032 | 47 | 1 |
| 3dpop | 3d | 4 | 118 | 118 | 121,985 | 17 | 1–10 |
| branson-fly | 2d | 1 | 10 | 194 | 96,532 | 21 | 5–10 |

**226 MB, 984 symlinks, 271 parquet files.** All four validate with 0 errors. The 475 GB of
pixels stays in `posetail-finetuning-v3`; one symlink per camera per group.

Warnings are all rule 14 (a session name in more than one split) and all genuine: rat-city and
allen-mouse are single recordings split by frame range, and 3dpop reuses a pigeon cohort across
sequences. The rule warns rather than fails for exactly this reason.

## The allen column-sort repair, verified

`pose3d.npz['pose']` is stored column-sorted while `keypoints` is name-sorted, transposing 16 of
47 keypoints. Bone lengths settle it, because swapping *both* endpoints of an edge leaves its
length unchanged — which is why the `base → tip` edges below look identical and why this bug is
invisible to a naive check. The `wrist → base` edges are the ones that move:

| edge | repaired | raw |
|---|---|---|
| RF-pinkie-base → RF-pinkie | 1.84 | 1.84 |
| LF-pinkie-base → LF-pinkie | 1.96 | 1.96 |
| **RF-wrist → RF-pinkie-base** | **2.54** | **4.34** |
| **LF-wrist → LF-pinkie-base** | **2.55** | **4.39** |

Raw connects the wrist to the *fingertip* (71% longer). Repaired connects it to the knuckle.
`scratch/check_allen_perm.py` also asserts the converted `points3d` equals the repaired source
exactly over 19,338 finite points, with NaNs preserved.

## Decisions the data forced

- **A v4 session becomes one session only when its trials agree on calibration.** allen-mouse's
  `metadata.yaml` is byte-identical across all 45 trials → one session, 45 groups. 3dpop
  calibrates per *sequence* → 118 sessions of one group each, named
  `<Pigeon>__<Sequence>`. Data-driven, not a per-dataset branch: calibration is a session
  property in this format, so trials that disagree about it are not one session.
- **NaN becomes no row, never `missing`.** v4 has exactly one "no label" state. Writing `missing`
  would claim someone looked and judged the point occluded. allen-mouse's per-camera `vis` array
  is a real assessment and is the only place `missing` is written — as coordinate-free
  `keypoints.pq` rows, the §7 rule-10 exemption.
- **3dpop's pose is exactly one frame longer than its pixels**, in every trial (900 vs 899, 954
  vs 953, 1206 vs 1205 …). Truncated to the pixel count, reported per group.
- **`3dpop/val/Pigeon01/Sequence49` is 100% NaN in the source** — v4's score-cleaning removed
  every point in those 16 frames. The group is dropped and the session is not written, with both
  facts printed. It is the only such group.

## Two bugs the validator caught

1. `_remap` validated the whole parquet dictionary rather than the rows of the group being read.
   A parquet dictionary is per *file*, so a branson-fly session holding 5–10 flies across trials
   has `animal_id` values no individual group has seen. Every branson-fly session failed.
   Regression test: `test_animal_count_may_vary_between_groups`.
2. `flip_pairs` lists each pair once, so the involution is the symmetric closure; checking
   `flip[b] == a` on the raw dict rejected every valid file.

## Not done

- **branson-fly keypoint names are placeholders** (`kp00..kp20`, `names_provisional = true` in
  its spec). No preprocessing module for it exists anywhere, and its npz carries no `keypoints`
  array. Training is unaffected — the embedding is positional — but flip augmentation and any
  per-keypoint report would be wrong, so `skeleton` and `flip_pairs` stay empty.
- **No boxes.** v4 has none; `instances.pq` is absent everywhere. The detector derives boxes from
  the crop rule, so nothing needs them yet. The ignore-region semantics (`present`) that were
  worth +0.017 MOTA on rat-city are unreachable until someone annotates them.
- **No moving cameras in any of the four.** `extrinsics.pq` is implemented and tested against a
  synthetic session; no real data exercises it yet.
