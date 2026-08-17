# tailcyclenet

Finetune a [posetail](https://pypi.org/project/posetail/) point tracker into an animal pose
estimator. Three settings, one model: **3D multiview**, **3D single-view**, **2D single-view**.

The pipeline detects animals, crops them, and decodes per-keypoint poses through a window loop.
It reads a single annotation format (`docs/annotation_format.md`) that serves both hand annotation
and bulk training across datasets of differing keypoint sets, camera counts, and dimensionality.

> Working notes, design docs, and every measurement live in `dev/` (untracked). `CLAUDE.md` is the
> authoritative engineering reference — the invariants, gotchas, and eval rules. This README is the
> short version.

---

## Setup

Dependencies are managed with [pixi](https://pixi.sh). `posetail==0.3.2` is pinned from PyPI.

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"   # sanity check
pixi run test                                        # test suite
```

Notes:
- The `LD_LIBRARY_PATH` prepend in `pyproject.toml` is load-bearing — the env ships a newer
  `libstdc++` than some hosts, and without it `import scipy.optimize` dies naming only `CXXABI`.
- `tailcyclenet/patches.py` monkeypatches the pinned library (per-frame camera offset); it is
  applied automatically on import. Every entry is an upstream bug to be deleted when the pin moves.

---

## Layout

```
tailcyclenet/     library: format, dataset, crop rule, model, inference, metrics, detector
scripts/          train.py  train_detector.py  infer.py  eval.py  convert_*.py
configs/          base.toml + 2d.toml + 3d.toml (hand-written; extends one level deep)
docs/             annotation_format.md — the data format spec (human-owned)
tests/            invariants (crop rule, patches, converters, geometry)
```

---

## Training

Pose model (one estimator, trains across all dataset roots under `[data].path`):

```bash
# 3D (multiview / single-view)
pixi run python scripts/train.py --config configs/3d.toml --data <root>

# 2D (single-view)
pixi run python scripts/train.py --config configs/2d.toml --data <root>
```

The two configs differ in exactly three keys (`cams_to_sample`, `val_cams_to_sample`,
`prob_2d_only`) — all camera-count questions a one-camera root cannot ask. `n_keypoints` is derived
from the data, never configured.

**Recommended settings** (all shipped in the configs; see `dev/reports/25_recommended_parameters.md`
for the evidence class behind each):

- `query = "prior"`, `query_encoder = "wide"`, `gridresid_offset = "query"` — the winning
  architecture. `gridresid_offset` has **no default and must be stated** per run.
- `prompt_dropout = 0.5`, `prompt_noise_px = 2.5`, `prompt_offset_px = 8.0` (3D) / `0.0` (2D). Ship
  the prior corruption *with* dropout — dropout alone is measurably worse.
- `box_source` — **state it explicitly per root** (`instances` for rat-city; `keypoints` for
  calms21/johnson; either for 3dpop depending on comparability). It is a live default that moved
  underneath cached artifacts.
- `n_iterations = 60000`, `learning_rate = 1e-4`, `freeze_encoder = true`, `seed = 0` (golden chain).

Detector (one per dataset, reproduces the pose model's crop rule):

```bash
# 3D roots — train with keypoints (enables --crop-source keypoints at inference)
pixi run python scripts/train_detector.py --data <root> --out runs/det-<name> \
    --keypoints --augment --min-box-px 32 --boxes <match the pose run> --iters 20000

# 2D roots
pixi run python scripts/train_detector.py --data <root> --out runs/det-<name> \
    --augment --min-box-px 32 --boxes <match the pose run> --iters 20000
```

- `--boxes` **must equal the pose run's `box_source`**.
- `--rotate-deg` and `--augment-photometric` are **refuted** — leave off.
- A same-recipe replicate is mandatory before believing any detector arm: one seed moves coverage
  by more than most levers.

---

## Inference

```bash
pixi run python scripts/infer.py --data <dataset|session|video> --run runs/<x>/ \
    --detector runs/det-<name> --det-cache <cache>.npz --out pred.npz
pixi run python scripts/eval.py pred.npz --data <root> --split test --chunk 500
```

There is **one** window loop. Detection is cached separately (`--det-cache`), so every arm shares
one detection pass and comparisons are matched by construction. **The CLI defaults are not the
recommendation** — set the flags below explicitly.

### Recommended flags per root

| flag | 3dpop (3D/4cam) | calms21 (2D) | rat-city (2D/12) | johnson (3D/16) |
|---|---|---|---|---|
| `--anchor` | `carry` | `carry` | **`none`** | `carry` |
| `--overlap` | **8** | **4** | **12** | 8 |
| `--crop-source` | **`keypoints`** | `boxes` | `boxes` | `keypoints` |
| `--refine` | no | **yes** | untested | — |
| `--pose-nms` | untested | **no (harmful)** | **0.6 (= 3 of 4)** | — |
| `--track` | on | inert (C=1) | inert (C=1) | **on (mandatory)** |
| `--det-score` | calibrate | 0.99 | **0.97** | calibrate |
| `--max-animals` | count | count | count (12) | count |
| `--vis-thresh` | optional | **never** | **never** | optional |

### The rules behind the table

- **`--anchor carry` is harmful in 2D crowded/long clips.** On rat-city it costs +52 px MPJPE and
  −0.46 MOTA (the carried prior is often the wrong animal). Use `--anchor none` there. In 3D the
  per-frame triangulation de-loops the feedback and `carry` wins.
- **`--overlap`: sign transfers across dimensionality, optimum does not.** 3D has an interior
  optimum at 8; 2D keeps improving to 12. Sweep per root.
- **`--crop-source`: root-conditional.** `keypoints` wins in 3D (dense labels), `boxes` in 2D
  (sparse per-frame keypoints floor the crop too small).
- **`--refine` is the biggest lever on calms21** (~2× better) and costs only **+31% wall clock**
  (not 2×) — the video decode dominates the second pass.
- **`--pose-nms` is the one identity lever that works, and only where `fp_dup` is a live term**
  (rat-city yes; calms21 harmful). Quote it as a **count** (K=4 → 4 settings; 0.6 = 3 of 4).
- **`--det-score 0.99` is wrong for any detector whose objectness isn't saturated.** Sweep per
  checkpoint: 0.97 maximises identity, 0.50 maximises coverage.
- **`--vis-thresh` cannot work in 2D** — nothing supervises the visibility head there.
- Always run `eval.py` with `--chunk 500` on long clips (the bootstrap resamples groups, so a
  single long clip returns `DEGENERATE`). Use `--min-match-kpts 0.5` for deltas, `0` for absolutes.

### The one thing bigger than any flag

Pose on a GT crop is ~19 px at coverage 1.0 vs ~45 px at 0.57 through the detector; on long 2D
clips **100% of coverage loss is `no box`**. The detector/box path dominates every identity lever
combined. That, not a flag, is the frontier.

---

## Reference

- `CLAUDE.md` — full engineering reference: invariants, gotchas, eval rules.
- `docs/annotation_format.md` — the data format spec.
- `dev/reports/25_recommended_parameters.md` — every recommended parameter with its evidence.
- `dev/reports/24_lever_audit_and_cleanup.md` — what was measured, refuted, and removed.
