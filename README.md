# tailcyclenet

Animal pose estimation with temporal priors, for **3D multiview**, **3D single-view**, and
**2D single-view** video — one model, one data format, one inference path.

`tailcyclenet` is the pose and quality-scoring half of the Tailcycle framework. It finetunes a
[tracktail](https://pypi.org/project/tracktail/) point tracker (formerly `posetail`) — a
V-JEPA 2.1 video encoder plus a query decoder, extended to multiview with calibrated cross-camera
attention — into:

- a **pose estimator**: each named keypoint becomes a learned query, so one network can train
  across datasets with different keypoint sets, camera counts, and dimensionality;
- a **box detector** that finds the animals and gives the pose model its crops;
- a **track-quality scorer**: trained contrastively on synthetically corrupted trajectories, it
  scores a pose track per frame and keypoint **without ground truth**, so you can find the
  frames worth annotating next.

The intended loop: annotate a few clips → train pose → run it on new video → score the output →
annotate the worst-scoring frames → retrain.

Rough cost: inference runs end to end (including video decode) at about 20 fps per camera on an
L4 GPU with about 500 MB of GPU memory per camera; finetuning on a new dataset takes about a day
on one L4.

---

## Install

### pip

```bash
# 1. torch built for your CUDA version (the default PyPI wheel is often CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. tailcyclenet
pip install tailcyclenet

# 3. aniposelib's `pytorch` branch (required: camera geometry runs on the GPU in every setting,
#    2D included; the PyPI release of aniposelib does not have it)
pip install "aniposelib @ git+https://github.com/lambdaloop/anipose-lib.git@pytorch"
```

This installs the `tailcyclenet` command: `train`, `train-detector`, `train-scorer`, `infer`,
`score`, `eval`, `render`. Run `tailcyclenet <command> --help` for every option. The shipped
configs are bundled in the package. Only the dataset converters need a source checkout (below).

### From source (pixi)

```bash
git clone https://github.com/AI-HHMI/tailcyclenet && cd tailcyclenet
pixi install
pixi run python -c "import tailcyclenet"
```

The pixi environment sets `LD_LIBRARY_PATH` so its own `libstdc++` wins over an older system
copy; without it, `import scipy.optimize` can fail with a `CXXABI` error. From a checkout, every
command is also available as `pixi run tailcyclenet <command> ...`, or as the script under
`scripts/` (`train.py`, `train_detector.py`, `train_scorer.py`, `infer.py`, `score_session.py`,
`eval.py`, `render.py`) with the same flags.

### Weights that are downloaded

- Pose training warm-starts from the pretrained tracker on Hugging Face
  (`[training].checkpoint_path`, default `ai-hhmi/posetail-static-animal`), which is downloaded
  and cached on first use. A local checkpoint folder also works.
- The detector backbone uses ImageNet weights, which are fetched on first use. If you set
  `[model].pretrained = "coco"`, the COCO YOLOX weights go to `~/.cache/tailcyclenet/weights/`
  (override with `--weights-dir` or `$TAILCYCLENET_CACHE_DIR`).

On a machine without internet access, download these on a connected one first.

---

## 1. Prepare your data

Everything reads one format, specified in [`docs/annotation_format.md`](docs/annotation_format.md).
In short:

```
my-dataset/
  train/  val/  test/              # the split is the directory
    <session_id>/
      session.toml                 # mode ("2d"/"3d"), units, labels ("annotated"/"tracked"), keypoint names
      calibration.toml             # aniposelib camera file (+ per-camera `offset`); always present
      groups.pq                    # one row per clip ("group"), with n_frames and fps
      keypoints.pq                 # per-camera 2D labels + visibility status   (optional)
      points3d.pq                  # 3D labels                                  (optional)
      instances.pq                 # boxes / "an animal is here"                (optional)
      groups/<group_id>/<cam>.mp4  # or <cam>/000000.jpg ...; symlinks are fine
```

- A **session** is one recording setup (one calibration, one mode). A **group** is one
  contiguous clip, of any length. A clip with only one labelled frame is fine; the unlabelled
  frames still provide temporal context.
- Sessions do not need the same keypoints: names are matched by name across sessions and
  datasets.
- 2D sessions have one camera and can omit the intrinsics and extrinsics.
- `status` separates "visible", "assessed and occluded" (`missing`), and "never looked at" (no
  row). Don't write `missing` for a plain NaN.

`tailcyclenet.format.write_session` writes the tables from arrays, and
`tailcyclenet.format.validate_dataset(load_dataset(root))` checks a root against the spec:

```python
from tailcyclenet.format import load_dataset, validate_dataset
problems = validate_dataset(load_dataset("my-dataset"))
print("\n".join(problems) or "ok")
```

`scripts/convert_*.py` are converters for the datasets this project was developed on. They are
useful examples of how to write one.

---

## 2. Train the pose model

```bash
tailcyclenet train --data my-dataset --out runs/my-pose
# from a checkout:
pixi run python scripts/train.py --data my-dataset --out runs/my-pose
```

**The shipped recipe (`configs/base.toml`, used when `--config` is omitted) should work for most
datasets, 2D and 3D alike. Start with it.** To change something, write a small TOML containing
only the keys you want to change; it is layered over `base.toml` automatically:

```toml
# my-pose.toml
[data]
path = "my-dataset"

[training]
out = "runs/my-pose"
```

The overrides that are actually useful in practice:

| key | default | when to change it |
|---|---|---|
| `[training].n_iterations` | `60000` | Raise it (e.g. 100k–120k) for a large dataset, and to several hundred thousand when training one model across many datasets on several GPUs. |
| `[training].checkpoint_path` | the pretrained tracker | Point it at an earlier pose run (or its `checkpoints/checkpoint_last.pth`) to finetune from your own model, e.g. a model already trained on the same species. If it holds the full training state and matches this config, training *continues* from its iteration; pass `--no-resume` (or use a packaged `.pth`) to start a fresh schedule from those weights. |
| `[data].flip_2d_prob`, `[data].flip_2d_modes` | `0.0`, `["horizontal"]` | Set `0.5` to enable left/right flip augmentation; add `"vertical"` for top-down or bottom-up views. Requires `flip_pairs` in every `session.toml`. |
| `[model].intrinsic_embedding` | `true` | Set `false` to stop conditioning on camera intrinsics when your calibration's intrinsics are only approximate. |
| `[data].num_workers` | `8` | Lower it if the host runs out of RAM (it is per GPU). |

Everything else (augmentation, prompt corruption, losses, architecture) is tuned and best left
alone.

- `[data].path` can also be a **folder of dataset roots**. One model then trains across all of
  them, with keypoint names prefixed by the dataset folder name.
- **Multiple GPUs** (one node): `--devices N`; the default `-1` uses every visible GPU. Each GPU
  processes one training item per step, so the number of GPUs is the batch size. Iteration
  counts are totals across GPUs.
- **Resuming**: re-running the same command with the same `--out` continues from
  `checkpoints/checkpoint_last.pth`. `--no-resume` starts over.
- Logging goes to Weights & Biases if it is installed (configured in `[wandb]`, written inside the
  run folder by default); `--no-wandb` turns it off.

A run folder contains `checkpoints/`, the effective `config.toml`, `keypoint_registry.toml` (the
keypoint axis the model was trained on), and `provenance.toml` (the git commit).

## 3. Train the detector

The detector predicts a box per animal per camera and reproduces the crop the pose model was
trained on. Its recipe lives entirely in a config. Point it at your data:

```toml
# my-detector.toml   (layered over the shipped configs/detector.toml)
[data]
path = "my-dataset"          # a dataset root, or a folder of roots

[training]
out = "runs/my-detector"
```

```bash
tailcyclenet train-detector --config my-detector.toml
# from a checkout:
pixi run python scripts/train_detector.py --config my-detector.toml
```

Only `--out`, `--iters`, and `--device` override the config on the command line. An unknown key
is an error, not a silent default.

**As with pose, the shipped recipe should be fine for most people** — it is built to generalise
across rigs (full-rotation and flip augmentation, photometric augmentation, synthetic
backgrounds). Useful overrides:

| key | default | when to change it |
|---|---|---|
| `[training].iters` | `8000` | Enough for one dataset. Raise it (tens of thousands) for a detector trained on a folder of many datasets. |
| `[data].boxes` | `"instances"` | Boxes come from `instances.pq` where a session has one, and from the keypoints otherwise. Set `"keypoints"` if your stored boxes are drawn differently from the animal's keypoint extent. |
| `[data].boxes_by_dataset` | `{}` | The same choice per dataset when training on a folder of roots, e.g. `{ my-other-dataset = "keypoints" }`. |
| `[training].batch_size`, `[training].num_workers` | `8`, `8` | Lower them if you run out of GPU or host memory. |

## 4. Train the scorer (optional)

The scorer is warm-started from a trained pose run and learns, from synthetic corruptions of your
own labels, to rank a correct track above a corrupted one. It needs no extra annotation.

```bash
tailcyclenet train-scorer --data my-dataset --checkpoint runs/my-pose --out runs/my-scorer
# from a checkout:
pixi run python scripts/train_scorer.py --data my-dataset --checkpoint runs/my-pose --out runs/my-scorer
```

`--checkpoint` should be the pose run you will run inference with. The scorer then shares that
model's video encoder at inference time, so scoring adds little cost.

The shipped `configs/scorer.toml` (used when `--config` is omitted) is meant to be used as is. In
your own overlay (passed as `--config`, layered over `scorer.toml`), the keys worth setting are `[data].path`, `[training].n_iterations` (default `60000`; larger for
multi-dataset scorers), and any `[model]` key you changed for the pose run. The scorer's
architecture must match the pose run it starts from, so if the pose run set
`intrinsic_embedding = false`, set it here too.

---

## 5. Run inference

On a session in the dataset format:

```bash
tailcyclenet infer --pose runs/my-pose --detector runs/my-detector \
    --data my-dataset/test/<session_id> --out pred/
```

Straight from raw videos plus an [anipose](https://anipose.readthedocs.io) calibration (nothing
is converted or copied to disk):

```bash
tailcyclenet infer --pose runs/my-pose --detector runs/my-detector --out pred/ \
    --videos recordings/ --calibration calibration.toml --cam-regex 'cam([0-9]+)_' \
    --max-animals 2
```

`--cam-regex` is anipose's `cam_regex`: the capture group is the camera name, and the rest of
the filename is the clip name. A single-camera 2D video needs no calibration.

Add `--scorer runs/my-scorer` to either command to write a per-frame, per-keypoint
`quality_score` (and `quality_precision`) beside every prediction. Predictions themselves are
not changed or filtered.

Things to know:

- **Always pass `--detector` (or `--boxes`) for real use.** Without a box source, `--data`
  inference crops from the ground-truth labels. That is an upper bound on accuracy, not what you
  would get on new video.
- `--pose` and `--detector` accept a run folder (the latest checkpoint is used) or a single
  packaged `.pth` (see below).
- One inference run reads **one session** (one calibration, one keypoint set), which may contain
  many clips. `--data` and `--videos` are mutually exclusive.
- `--start-frame` / `--end-frame` (or `--max-frames`) restrict the range; frame numbers in the
  output always refer to the source video.
- `--max-ram GB` caps host memory (by default, it is derived from the job or container limit).
  It changes speed, never the predictions.
- The defaults are a reasonable starting point for both 2D and 3D. The flags most worth trying
  on a new dataset are `--overlap` (frames shared between consecutive windows), `--anchor`
  (whether each window is seeded with the previous window's prediction: `carry`, the default, or
  `none`), and `--refine` / `--refine-px` (a second, tighter crop; on by default in 3D).

### Output

`--out` is a **prediction session** in the same format as the input:

| file | contents |
|---|---|
| `session.toml` | keypoint names, mode, and `[provenance]` (which run, checkpoint, data, and flags produced it) |
| `calibration.toml`, `groups.pq` | copied from the source |
| `points3d.pq` | 3D predictions (3D sessions), in the session's units |
| `points2d.pq` | per-camera 2D predictions |
| `instances.pq` | the boxes each animal was cropped from |
| `windows.pq` | per-window bookkeeping, including why a window produced nothing |

The tables are written as inference proceeds, so memory does not grow with clip length.
`session.toml` gets `complete = true` only when the run finishes.

---

## 6. Evaluate, score, and visualise

```bash
# accuracy and tracking metrics against the labels (needs a labelled source session)
tailcyclenet eval pred/ --data my-dataset --split test

# draw predictions over the original video
tailcyclenet render --pred pred/ --out clips/ --boxes
```

- `eval` reports MPJPE (or pixel error in 2D), PCK, coverage, and MOTA/IDF1 for multi-animal
  data, with bootstrap confidence intervals over clips. For a few very long clips, add
  `--chunk 500` to split each into 500-frame scoring units. `--vs other_pred/` compares two
  prediction sessions on the same frames.
- `render` finds the source video through the prediction's provenance; colour is per animal, so
  an identity swap shows up as a colour change.

To review an existing **tracked** dataset (for example, machine-tracked labels you are considering
training on), score it with the scorer and get a worst-first list of windows and keypoints:

```bash
tailcyclenet score --run runs/my-scorer --data my-tracked-dataset --split train \
    --out qc/ --device cuda:0
```

This writes `scores.pq` and `report.txt`. Scores are relative: lower means more likely wrong,
and the report ranks rather than thresholds.

---

## 7. Share a model

```bash
pixi run python scripts/package_checkpoint.py --kind pose     --run runs/my-pose     --out my-pose.pth
pixi run python scripts/package_checkpoint.py --kind detector --run runs/my-detector --out my-detector.pth
pixi run python scripts/package_checkpoint.py --kind scorer   --run runs/my-scorer   --out my-scorer.pth
```

A packaged `.pth` contains its config and keypoint registry, without the optimizer state, and
can be passed directly to `--pose` / `--detector` / `--scorer`.

---

## Repository layout

```
tailcyclenet/         the library
  format.py           read/validate/write the data format
  dataset.py          training data loader (2D, 3D single-view, 3D multiview)
  model.py            the pose model on top of the tracktail tracker
  train.py            pose training (tailcyclenet train)
  detector/           box detector and cross-view association
  scorer/             track-quality scorer: model, training, QC scoring
  infer/              the inference window loop and prediction writer
  adopt.py            raw videos + anipose calibration -> an in-memory session
  eval.py, metrics.py evaluation
  render.py           visualisation
configs/              base.toml (pose), detector.toml, scorer.toml, configs/datasets/ (skeletons)
scripts/              command-line entry points, dataset converters, utilities
docs/                 annotation_format.md -- the data format specification
tests/
```

## Development

```bash
pixi run test    # pytest
pixi run lint    # ruff
```

## License

BSD-3-Clause.
