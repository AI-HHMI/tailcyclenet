# tailcyclenet

Finetune a [posetail](https://pypi.org/project/posetail/) point tracker into an animal pose
estimator. Three settings, one model: **3D multiview**, **3D single-view**, **2D single-view**.

The pipeline detects animals, crops them, and decodes per-keypoint poses through a window loop.
It reads a single annotation format (`docs/annotation_format.md`) that serves both hand annotation
and bulk training across datasets of differing keypoint sets, camera counts, and dimensionality.

> `CLAUDE.md` is the authoritative engineering reference — the defaults, invariants, gotchas and
> eval rules. This README is the short version: what this is, how to install it, how to run it.

---

## Setup

Dependencies are managed with [pixi](https://pixi.sh). `posetail==0.3.5` is pinned from PyPI.

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"   # sanity check
pixi run test                                        # test suite
pixi run lint                                        # ruff
```

Notes:
- The `LD_LIBRARY_PATH` prepend in `pyproject.toml` is load-bearing — the env ships a newer
  `libstdc++` than some hosts, and without it `import scipy.optimize` dies naming only `CXXABI`.
- posetail >= 0.3.5 ships every behaviour this repo once monkeypatched (per-frame camera
  offsets, `crop_box_for_points`, `scene_features=`/`input_size=` on the tracker forward);
  there is no patch layer anymore.

---

## Layout

```
tailcyclenet/     library: format, dataset, crop rule, model, inference, metrics, detector
scripts/          train.py  train_detector.py  infer.py  eval.py  convert_*.py
configs/          base.toml + 2d.toml + 3d.toml + detector.toml (extends one level deep)
configs/datasets/ per-dataset keypoint and skeleton definitions
docs/             annotation_format.md — the data format spec (human-owned)
tests/            invariants (crop rule, converters, geometry)
```

---

## Training

One estimator trains across every dataset root under `[data].path`; a keypoint embedding table is
what lets roots with different keypoint sets share a model.

```bash
# 3D (multiview / single-view)
pixi run python scripts/train.py --config configs/3d.toml --data <root>

# 2D (single-view)
pixi run python scripts/train.py --config configs/2d.toml --data <root>
```

The two configs differ in exactly three keys (`cams_to_sample`, `val_cams_to_sample`,
`prob_2d_only`) — all camera-count questions a one-camera root cannot ask. `n_keypoints` is derived
from the data, never configured.

The configs are hand-written and every default carries its reasoning as a comment. Per-root
overrides belong in your own config that `extends` one of these, not in a CLI flag.
`[model].gridresid_offset` has **no default and must be stated** — the two values load the same
tensors, so a mismatch produces numbers rather than an exception. See `CLAUDE.md`.

Unfreezing the video encoder costs activations *per camera*, because the scene encoder runs once
per camera: roughly +0.6 GB per camera frozen, plus ~0.022 GB per unfrozen block per camera. Check
the budget before submitting a job on a wide rig.

---

## Detector

```bash
pixi run python scripts/train_detector.py --config configs/detector.toml
```

The recipe lives in the config, not on the CLI — every default is there with its evidence, and an
unknown key raises rather than silently training at a default. Only `--out`, `--iters` and
`--device` override.

One detector per dataset, and `input_wh` defaults to an aspect-matched size rather than a square: a
square letterbox on a wide frame wastes most of the canvas and can put the animal below the stride
the FPN can represent. The regression target is `crop.crop_box_for_points` — the detector
reproduces *the crop the pose model was trained on*, so `[data].boxes` must equal the pose run's
`[data].box_source`.

---

## Inference and eval

```bash
pixi run python scripts/infer.py --data <dataset|session|video> --run runs/<name> \
    --detector runs/det-<name> --det-cache <cache>.npz --out pred.npz
pixi run python scripts/eval.py pred.npz --data <root> --split test --chunk 500
```

There is **one** window loop. Detection is cached separately (`--det-cache`), so every arm shares
one detection pass and comparisons are matched by construction. **The CLI defaults are not the
recommendation** — the good settings are root-conditional, so sweep them per root, and see
`CLAUDE.md` for which direction each one moves and what it trades against.

Current defaults: `--overlap 8`, `--refine` derived (on 3D / off 2D), `--box-prompt auto`,
`--prefetch-windows 1` (bit-exact, performance only).

Four rules that are *not* root-conditional:

- **`--vis-thresh` cannot work in 2D** — nothing supervises the visibility head there.
- **`--box-prompt auto` needs a detector or boxes file**: a box model run without one **crashes**
  rather than silently falling back to the GT `labels` oracle. Pass `--detector`/`--boxes` for
  deployment, `--box-prompt labels` for the explicit oracle (it warns), or `--box-prompt none` to
  withhold the box. A box model also pulls `--crop-inflate 1.5` and `--refine-px 128` automatically.
- Always run `eval.py` with `--chunk 500` on long clips — the bootstrap resamples groups, so a
  single long clip returns `DEGENERATE`.
- Use `--min-match-kpts 0.5` for deltas and `0` for absolutes.

The largest lever is not a flag: pose accuracy on a ground-truth crop is far better, at full
coverage, than through the detector, and on a long clip essentially all coverage loss is `no box`.
Fix the crop path before tuning identity flags.

---

## Reference

- `CLAUDE.md` — full engineering reference: defaults, invariants, gotchas, eval rules.
- `docs/annotation_format.md` — the data format spec.
