# The model, the loader, and the first training runs

## What is here

| file | lines | what |
|---|---|---|
| `tailcyclenet/format.py` | 930 | the format: read, write, validate, keypoint registry |
| `tailcyclenet/dataset.py` | 407 | the loader: 3D multiview / 3D single-view / 2D single-view |
| `tailcyclenet/query_encoder.py` | 316 | keypoint identity + honest no-query tokens |
| `tailcyclenet/model.py` | 243 | query construction, per-frame re-anchoring |
| `tailcyclenet/checkpoints.py` | 154 | run folders, warm start |
| `tailcyclenet/crop.py` | 126 | THE crop rule |
| `scripts/train.py` | 202 | the trainer |
| `scripts/convert_v4.py` | 357 | the converter |

**2,735 lines**, against posetail-pose's ~30,000. One architecture switch (`query`), one config
per intent rather than 86 generated arms, 31 tests.

## Training works

`allen-mouse`, 300 iterations, warm-started from `ukt14i7c`:

```
     20/300  loss 4.1882  1.08s/it  skipped 0
    100/300  loss 2.3274  0.87s/it  skipped 0
    200/300  loss 1.3918  0.75s/it  skipped 0
    300/300  loss 1.3524  0.72s/it  skipped 0
```

All four datasets at once — 1,735 windows, **89 keypoints**, 2D and 3D and 3D-single-view
interleaved in one run:

```
     20/120  loss 2.2362  [allen-mouse/3d/1cam]
     40/120  loss 2.2771  [3dpop/3d]
     60/120  loss 2.2136  [rat-city/2d]
     80/120  loss 1.7472  [branson-fly/2d]
    120/120  loss 1.1759  [allen-mouse/3d]
```

These are smoke runs, not converged models. A falling loss curve is not evidence that anything
works; it is evidence that nothing is structurally broken.

## The warm start is nearly total

Six parameters start fresh — the keypoint identity table, its LayerNorm, and the three no-query
tokens. Everything else is inherited, **including the 5.53M-parameter patch processor**, because
this encoder fuses at `embed_dim = 256` and so every one of its tensors loads verbatim.

The fusion gate is *inflated* rather than dropped: the base's 10-term gate is placed into this
model's 11-term one term by term, matched **by name** (the orders differ — stock leads with
`patch`, this leads with `kpt`), with the identity row and block left at zero. So at init the
identity term is gated at `sigmoid(0) = 0.5`, every inherited term is weighted exactly as the
base weighted it, and the model starts at the pretrained fusion behaviour rather than at noise.

## Two corrections found by building it

**1. `prob_2d_only` is 3D single-view, not 2D.** I first implemented it as "project the 3D onto
one camera and train in pixels". posetail's own path (`posetail_dataset.py:838-846, 971-975`)
subsets the cameras to one and adds `p2d`, and **never reassigns `coords`** — the targets stay
world-metric. So the mode teaches metric 3D from a single view, which is one of the three
settings this repo is for. Caught by reading the library rather than by any test.

**2. A NaN visibility target silently destroys every gradient.** posetail's visibility term is a
plain `binary_cross_entropy_with_logits(..., reduction='mean')` with no missing-value mask
(`losses.py:901`). The format has three visibility states, so `unlabeled` mapped naturally to
NaN — and the result was that the **forward loss stayed finite and healthy-looking while 36 of
40 steps were skipped on a non-finite gradient**. The forward drops the NaN term; its gradient
still flows.

`unlabeled` therefore collapses to "not visible" in the loader. That is safe rather than merely
convenient: `vis2d` is only unlabeled where the point has no 3D label either, so it carries no
coordinate supervision in any case. The three-state distinction stays in the format, where it
belongs; it is this consumer that cannot express it.

Both the skip counter and the anomaly trace mattered. A trainer that did not count skips would
have shown a perfectly plausible loss curve produced by 4 of 40 steps.

## Deliberate simplifications, and what they cost

- **Keypoint ids no longer travel through the occlusion channel.** posetail-pose smuggled them
  there because `TrackerEncoder.forward`'s only per-keypoint channel is `occlusion`, and the
  stock encoder clamps `occlusion + 1` into `[0, 2]` — so the two consumers could never share
  the tensor and mis-packing it was invisible. Here the model stashes `_kpt_ids` on the query
  encoder directly. The occlusion channel is now free, and the format carries real per-camera
  visibility, so feeding it is a one-line change rather than an architecture. Not done: that
  would be a new arm, not a port.
- **The instance anchor is deleted, not defaulted off** — `instance_anchor`, `anchor_mode`,
  `anchor_fallback`, `anchor_dropout`, `anchor_noise`, `anchor_attn_bias`. What remains is the
  per-keypoint prior, which is what `w9_honest` actually used at runtime.
- **`WideQueryEncoder` is gone.** It beat the pose encoder on allen-mouse 3D in posetail-pose
  (3.395 vs 4.021 mm). If 3D accuracy disappoints, this is the first thing to try re-adding.
- **`batch_size` is structurally 1** and there is no DDP, because posetail's collate keeps only
  item 0's camera group. Unchanged from posetail-pose; recorded as a ceiling, not fixed.
- **No moving-camera canonicalisation augmentation.** The loader threads per-frame extrinsics
  through to the model, but the fix-camera/move-world augmentation is not implemented — no
  dataset has moving cameras yet.

## Not yet done

Inference, metrics, eval, and the detector. Nothing has been evaluated against the reference
numbers, so no accuracy claim is made here.
