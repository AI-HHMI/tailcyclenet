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

**Recommended settings** (see `dev/reports/32_recommended_parameters_v2.md` for the evidence class
behind each; report 25 is superseded):

- `query = "prior"`, `query_encoder = "wide"`, `gridresid_offset = "query"` — the winning
  architecture. `gridresid_offset` has **no default and must be stated** per run.
- `prompt_dropout = 0.5`, `prompt_noise_px = 2.5`. `prompt_offset_px` — **8.0 in `3d.toml`,
  0.0 in `2d.toml`** (the shipped 5.0-on-both is a consistency choice, not a measurement; only 8.0
  on allen 3D has ever been measured, −0.277 mm paired). Ship prior corruption *with* dropout.
- `box_prompt = "film"` — the box prompt (report 27), now defaulted on. **Use it on crowded dense-K
  2D roots (calms21); set `"none"` explicitly on 3D and single-animal roots** — 3D is unmeasured and
  structurally skipped at inference.
- `box_source` — **state it explicitly per root** (`instances` for rat-city; `keypoints` for
  calms21/johnson; either for 3dpop depending on comparability). It is a live default that moved
  underneath cached artifacts and is absent from every shipped config.
- `n_iterations = 60000`, `learning_rate = 1e-4`, `freeze_encoder = true`, `seed = 0` (golden chain).

Detector (one per dataset, reproduces the pose model's crop rule):

```bash
# 3D roots — train with keypoints (enables --crop-source keypoints at inference)
pixi run python scripts/train_detector.py --data <root> --out runs/det-<name> \
    --keypoints --min-box-px 32 --boxes keypoints --iters 20000 [--reduce for JPEG roots]

# 2D roots
pixi run python scripts/train_detector.py --data <root> --out runs/det-<name> \
    --min-box-px 32 --boxes <match the pose run> --iters 20000
```

- `--boxes` **must equal the pose run's `box_source`** (`keypoints` on calms21/johnson — their
  stored MARS/COCO boxes agree with the crop rule on 0.000 of instances).
- **`--augment`, `--augment-strong`, `--rotate-deg 45`, `--yolox tiny` are now the defaults, on a
  prior rather than a measurement** ("more augmentation → more generalization"). The repo's evals
  are in-domain and cannot see that benefit: `--rotate-deg 45` is genuinely untested (the ±180
  refutation does not transfer — `_rotated_rect_max_inscribed` is 90°-periodic), `--yolox tiny` is
  indistinguishable-not-worse (its lead is smaller than its own seed swing; costs 4.4× params over
  `trimmed`), and `--augment-strong` contains a component refuted downstream (additive noise).
  **Reproducing any pre-2026-08-17 detector needs `--no-augment-strong --rotate-deg 0 --yolox
  trimmed --no-augment`** — a fresh no-flag detector is a new recipe, not a continuation of the old.
- `--augment-strong` is **undefined under `--use-regions`** (raises at construction) — pick one on
  the rat-city tiled recipe.
- `--min-box-px 32` everywhere; **64 for 3dpop if the budget allows** (and raise `--max-input-px`,
  which caps it). `--keypoints` on for 3D, off for 2D.
- A same-recipe replicate is mandatory before believing any detector arm: one seed moves coverage
  by more than most levers, and `--lr 1e-3` is unmeasured (10× DLC/SLEAP/APT).

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

Current defaults (report 25 predates all of them): `--overlap 8`, `--refine` derived (on 3D / off
2D), `--box-prompt auto`, `--prefetch-windows 1` (bit-exact, performance only).

### Recommended flags per root

| flag | 3dpop (3D/4cam) | calms21 (2D) | rat-city (2D/12) | johnson (3D/16) |
|---|---|---|---|---|
| `--anchor` | `carry` | `carry` | **`none`** | `carry` |
| `--overlap` | **8** | **8** (4 is a null) | **12** | 8 |
| `--crop-source` | `boxes` | `boxes` | `boxes` | `boxes` |
| `--refine` | **on (derived)** | **on for accuracy** | off (derived) | on (derived) |
| `--refine-px` | 192 ≈ 256 | **96** (1 session) | **≥192** | sweep |
| `--pose-nms` | untested | **no (harmful)** | **0.6 (= 3 of 4)** | — |
| `--track` | on | inert (C=1) | inert (C=1) | **on (mandatory)** |
| `--det-score` | 0.5 (default) | 0.5 (default) | **0.97 (identity)** | 0.5 (default) |
| `--max-animals` | count | count | count (12) | count |
| `--vis-thresh` | optional | **never** | **never** | optional |

### The rules behind the table

- **`--anchor carry` is root-conditional by ANIMAL COUNT, not clip length.** calms21 (2 mice):
  +0.082 MOTA SIG; rat-city (12 rats): **+52 px MPJPE / −0.46 MOTA** — use `--anchor none` there.
  In 3D the per-frame triangulation de-loops the feedback and `carry` wins.
- **`--overlap`: sign transfers across dimensionality, optimum does not.** 3D has an interior
  optimum at 8; 2D improves to 12 (though 12-vs-8 is n.s.). Sweep per root.
- **`--crop-source` is `boxes` everywhere** — `keypoints` is a measured NULL in 3D (+0.181 mm) and
  refuted in 2D (sparse per-frame keypoints floor the crop too small).
- **`--refine` is now derived: on in 3D (−0.962 mm SIG), off in 2D** (its calms21 accuracy gain
  costs identity, MOTA −0.0435 SIG). It costs **+31% wall clock**, not 2×.
- **`--pose-nms` is the one identity lever that works, and only where `fp_dup` is a live term**
  (rat-city yes; calms21 harmful). Quote it as a **count** (K=4 → 4 settings; 0.6 = 3 of 4).
- **`--det-score` defaults to 0.5 (coverage), not 0.99.** Saturation is a property of the recipe,
  not the dataset: 0.99 keeps 26–33% of detections on the current generation, 0.5 keeps ~99%. 0.97
  maximises identity. Sweep per checkpoint — read the objectness quantiles recorded in it.
- **`--vis-thresh` cannot work in 2D** — nothing supervises the visibility head there.
- **`--box-prompt auto` needs a detector/boxes file**: a box model run without one now **crashes**
  rather than silently falling back to the GT `labels` oracle. Pass `--detector`/`--boxes`
  (deployment), `--box-prompt labels` for the explicit oracle (it warns), or `--box-prompt none` to
  withhold the box. A box model also pulls `--crop-inflate 1.5` and `--refine-px 128` automatically.
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
- `dev/reports/32_recommended_parameters_v2.md` — every recommended parameter with its evidence
  (supersedes report 25).
- `dev/reports/24_lever_audit_and_cleanup.md` — what was measured, refuted, and removed.
