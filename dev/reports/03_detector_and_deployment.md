# The detector and the end-to-end path

## What runs

```
raw pixels -> YOLOX-Nano -> boxes -> NMS (2D) / cross-view association (3D) -> pose -> npz -> eval
```

No labels are touched anywhere in that chain. Verified on branson-fly:

```
detector: 400 iters, loss 4.92 -> 1.38, recall@IoU0.5 = 0.806, 0.66M params, 0.046 s/it
infer --detector: (5, 500, 21, 2), 1.000 finite
```

Both models are smoke-trained (60 and 400 iterations), so no accuracy claim is made.

## One detector per dataset, and why the input size is not cosmetic

`--input-wh` defaults to an **aspect-matched** size at roughly a 416² pixel budget, not a square.
rat-city's frames are 4696×2048 (2.29:1). A square 416 letterbox scales by
`min(416/2048, 416/4696) = 0.089`, so the frame becomes 416×181 inside a 416×416 canvas — 56%
black padding — and the median rat arrives at 15.8 × 12.5 px. YOLOX pools at strides 8/16/32, so
that animal is ~2 × 1.6 cells at the finest level and **absent from the other two**: two thirds
of the FPN cannot represent it. The same detector on branson-fly's square 1024² frames delivers
the median fly at 26.5 × 28.1 px and reaches AP50 0.985 where rat-city sits near 0.50.

That is also why training one detector across datasets is not offered. One letterbox cannot
serve both, and `BoxDataset` raises rather than silently accepting a collection.

## The property that matters

The regression target is `crop.crop_box_for_points` — the detector reproduces **the crop the
pose model was trained on**, not "a box around the animal". Those are different objectives and
only the first preserves downstream accuracy (measured in posetail-pose at +0.022 mm for a
detector crop vs a GT crop within-session). `tests/test_detector.py::test_targets_are_the_crop_rule`
asserts the stored targets equal the crop rule's box after undoing the letterbox.

An animal with no finite point in a view gets a **NaN box, not a dropped frame**: objectness
still has to learn "nothing here", and dropping those views would train a detector that has
never seen an empty image.

## A metric that was measuring the wrong thing

The first end-to-end score on branson-fly read **385 px MPJPE** on flies that are ~30 px across.
The cause was not the model: `eval.py` compared prediction row `a` to label row `a`, and once
boxes come from a detector **row index is not identity** — the detector's rows are NMS survivors
ordered by score, and nothing ties row 0 at frame *t* to row 0 at frame *t+1*, let alone to
labelled animal 0.

Multi-animal rows now report **Hungarian-matched** MPJPE, with `unmatched` printed beside it so
a method that predicts one animal well and ignores nine cannot hide. Same predictions, same
labels: **385 px → 64 px**.

A related default was wrong for the same reason: the MOTA match radius was derived from a
*per-axis* keypoint span and came out at 1.6 px on rat-city — tighter than the labelling noise —
so every instance counted as both a miss and a false positive and MOTA went negative. It is now
half the diagonal of the animal's own keypoint box (20.5 px on rat-city).

## Deliberately simple, and what that costs

- **Association uses box centres only.** Triangulate every cross-camera pair, keep groups whose
  point reprojects into each contributing view within `assoc_res_max_px`, greedy by residual,
  each box used once. Two animals overlapping in one view are resolved by another view but not
  by that one, and a two-view group is accepted on a residual two rays can always satisfy
  exactly — hence the `min_views` floor. Appearance features or a joint epipolar solve are real
  work and are not justified until a measurement says association is the bottleneck.
- **There is no tracker.** Detector rows are untracked across frames. Feeding them straight to
  the pose model is an honest per-window deployment baseline and nothing more; identity across
  windows is what `--anchor carry` provides for a *known* instance, not what associates a new
  detection to an old one.
- **Centre-prior assignment, not SimOTA.** With one class and a handful of instances, dynamic-k
  buys nothing and adds a second thing that can be wrong.

## Repo state

40 tests, all passing. ~3,900 lines against posetail-pose's ~30,000.

Not done: rendering (`--render` is documented in the plan but not implemented), and no model has
been trained to convergence, so nothing here has been compared against the reference numbers
(allen-mouse cross-animal 3.394 mm, human baseline 2.208 mm).
