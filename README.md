# tailcyclenet

Finetune a [posetail](https://pypi.org/project/posetail/) point tracker into an animal pose
estimator. Three settings, one model: **3D multiview**, **3D single-view**, **2D single-view**.

The pipeline detects animals, crops them, and decodes per-keypoint poses through a single window
loop. It reads one annotation format (`docs/annotation_format.md`) that serves both hand annotation
and bulk training across datasets of differing keypoint sets, camera counts, and dimensionality.
This README is the committed reference: what this is, how to run it, and the invariants a
contributor must not break.

---

## Setup

Dependencies are managed with [pixi](https://pixi.sh). `posetail==0.3.5` is pinned from PyPI.

```bash
pixi install
pixi run python -c "import posetail, tailcyclenet"   # sanity check
pixi run test                                        # test suite
pixi run lint                                        # ruff
```

- The `LD_LIBRARY_PATH` prepend in `pyproject.toml` is load-bearing — the env ships a newer
  `libstdc++` than some hosts, and without it `import scipy.optimize` dies naming only `CXXABI`.
- posetail >= 0.3.5 ships every behaviour this repo once monkeypatched (per-frame camera offsets,
  `crop_box_for_points`, `scene_features=`/`input_size=` on the tracker forward); there is no patch
  layer anymore.

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

# one node, N gpus: one item per rank, gradients averaged by DDP
pixi run python scripts/train.py --config configs/3d.toml --data <root> --devices 4
```

Facts that are easy to get wrong:

- The two configs differ in exactly three keys — `cams_to_sample`, `val_cams_to_sample`,
  `prob_2d_only` — all camera-count questions a one-camera root cannot ask. `n_keypoints` is
  **derived from the data, never configured**.
- The per-rank batch is structurally **1**; `--devices N` is the only batch dimension this repo has.
  Every iteration count in a config is a **total across ranks** (60,000 is 60,000 samples on any
  gpu count) and the learning rate is scaled by `sqrt(N)`, so a multi-gpu run is two levers off a
  single-gpu one; `provenance.toml` records which it was.
- `[model].gridresid_offset` has **no default and must be stated** — the two values load the same
  tensors, so a mismatch produces numbers rather than an exception.
- A run folder writes `keypoint_registry.toml` (the derived keypoint axis) and `provenance.toml`
  (commit + dirty flag). A config is not a provenance record.
- The video encoder unfreezes mid-run per `[model].video_encoder_requires_grad` (a bool, or an int
  iteration to unfreeze at). A run started before the shipped default (8 blocks at 10,000) is not
  comparable to one after; `false` restores the old arm.

---

## Detector

```bash
pixi run python scripts/train_detector.py --config configs/detector.toml
```

The recipe lives in the config, not on the CLI — every default is there with its evidence, and an
unknown key raises rather than silently training at a default. Only `--out`, `--iters` and
`--device` override.

One detector per dataset, and `input_wh` defaults to an aspect-matched size rather than a square:
a square letterbox on a wide frame wastes most of the canvas and can put the animal below the stride
the FPN can represent. The regression target is `crop.crop_box_for_points` — the detector
reproduces *the crop the pose model was trained on*, so `[data].boxes` must equal the pose run's
`[data].box_source`.

---

## Inference and eval

```bash
# one source session (a dataset root works only if it holds a single session in --split)
pixi run python scripts/infer.py --run runs/<name> --data <session-dir> --split test \
    --detector runs/det-<name> --out pred/

# or, straight off raw footage + an anipose calibration
pixi run python scripts/infer.py --run runs/<name> --out pred/ \
    --videos rec/ --calibration anipose/calibration.toml --cam-regex 'cam([0-9]+)_' \
    --detector runs/det-<name> --max-animals 4

pixi run python scripts/eval.py pred/ --data <root> --split test --chunk 500
```

- **`--out` is a prediction session directory** (`session.toml`, `calibration.toml`, `groups.pq`,
  `points3d.pq`, `keypoints.pq`, `instances.pq`, `windows.pq`), written a block at a time so nothing
  is proportional to clip length. `eval.py` and `render.py` both read it; `render.py` finds its own
  pixels via the session's `[provenance]`.
- **`--data` and `--videos` are exactly-one-of**, and a run is **one source session** (which may
  hold many groups). For `--videos`, the camera name is the regex **capture group** and the session
  is built in memory — nothing is staged.
- There is **one** window loop. Box sources: annotations, a detections npz (`--boxes`), or a
  per-dataset detector (`--detector`). Prompt regimes: `none` (query-free), `carry` (previous
  window's own prediction — what deployment does), `self` (two passes), `labels` (an oracle, gated
  off by default).

**Defaults are not the recommendation.** The good settings are root-conditional, so sweep them per
root. Current values: `--anchor carry`, `--overlap 4`, `--refine` derived (on 3D / off 2D),
`--track on`, `--box-prompt auto`, `--prefetch-windows 1` (bit-exact, performance only),
`--max-ram` derived from the host. In particular `--anchor` is root-conditional in 2D (a carried
prior on a crowded root is often the wrong animal's pose) and `--overlap`'s optimum is seam-count
against seam-size — sweep per root.

Four rules that are *not* root-conditional:

- **`--vis-thresh` has no meaning in 2D at the shipped default** — the visibility head is only
  trained when `[training.losses].vis_loss_2d_weight` is nonzero (default `0.0`).
- **`--box-prompt auto` needs a detector or boxes file.** A box-model run without one refuses rather
  than silently falling back to the GT oracle. Pass `--detector`/`--boxes`, or `--box-prompt none`
  to withhold the box.
- Always run `eval.py` with `--chunk 500` on long clips — the bootstrap resamples groups, so a
  single long clip returns `DEGENERATE`.
- Use `--min-match-kpts 0.5` for deltas and `0` for absolutes.

The largest lever is not a flag: pose accuracy on a ground-truth crop is far better, at full
coverage, than through the detector, and on a long clip essentially all coverage loss is `no box`.
Fix the crop path before tuning identity flags.

---

## Reference

- `docs/annotation_format.md` — the data format spec (human-owned).
