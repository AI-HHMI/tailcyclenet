"""The scorer's Dataset: a `PoseDataset` whose items are ready (good, bad, anchor) triplets.

Built is CPU-bound (two decodes, cv2 warpAffine, imgaug), so it runs inside the DataLoader
workers -- pipelined with the GPU step -- rather than synchronously in the training loop. Base
items arrive as CPU tensors, so `make_triplet`'s work stays on CPU and the loop moves the triplet
to the device.

The VAL split is a FIXED-SEED held-out set: its selection AND its corruption draws are keyed on
`(seed, idx)`, so `triplet_acc` measures the same windows under the same corruptions at every
checkpoint. Without that, `checkpoint_best` would be selected on a moving target.
"""
import numpy as np
import torch

from .triplet import build_corruptors_for, make_triplet, triplet_collate

GETITEM_MAX_RETRIES = 8


class ScorerDataset(torch.utils.data.Dataset):
    """Wraps a `PoseDataset` so each item is a triplet dict.

    A triplet build can fail for reasons the base loader calls legitimate -- a rotation that
    swings the animal off-frame, a crop that degenerates, a window whose surviving keypoints drop
    below 2. Any worker exception would crash the rank and hang a DDP job on the next collective,
    so a failure retries a DIFFERENT sample, exactly as `PoseDataset.__getitem__` does for its own
    rejects. Exhausting the retries returns None, which the training loop skips.
    """

    def __init__(self, base, corruption_cfg):
        """Inputs: base -- a built `PoseDataset`; corruption_cfg -- the `[scorer.corruption]`
        block, including `min_valid_frames` and the `mag_3d`/`mag_2d` magnitude dicts.
        Outputs: none. Side effects: builds the dense/sparse corruptor pairs once, so they are
        forked into every worker rather than rebuilt per item.
        """
        self.base = base
        self.cfg = dict(corruption_cfg)
        self.corruptors = build_corruptors_for(self.cfg)

    def __len__(self):
        """Inputs: none. Outputs: the number of base windows. Side effects: none."""
        return len(self.base)

    def _streams(self, idx):
        """The item's rng, its shape stream, and whether to freeze the corruption RNG.

        Train entropy-seeds the item stream so workers do not replay one another's augmentation;
        val/test key it on `(seed, idx)` so a metric is reproducible. The CORRUPTION draws come
        from the ambient torch RNG (the library's `PointCorruptor` has no generator argument), so
        on val/test torch is seeded per item as well -- otherwise the same window would be
        corrupted differently at every checkpoint and `triplet_acc` would compare nothing.

        Inputs: idx -- the window index.
        Outputs: (item_rng, shape_rng, frozen).
        Side effects: seeds the ambient torch RNG when `frozen` is True.
        """
        frozen = not self.base.train
        rng = np.random.default_rng(None if not frozen else (self.base.seed, idx))
        shape_rng = np.random.default_rng((self.base.seed, 0x5AFE, idx)) if frozen else rng
        if frozen:
            torch.manual_seed((int(self.base.seed) * 1000003 + int(idx)) % (2 ** 31))
        return rng, shape_rng, frozen

    def __getitem__(self, idx):
        """Build one triplet, retrying other windows on a failed build.

        Inputs: idx -- the requested window index (or `(ordinal, idx)` from `StepSampler`, which
                the base loader consumes and this passes through untouched).
        Outputs: a triplet dict, or None when every retry failed.
        Side effects: decodes video frames; draws from the item, ambient torch and imgaug RNGs.
        """
        base_idx = idx[1] if isinstance(idx, tuple) else idx
        for _ in range(GETITEM_MAX_RETRIES):
            rng, shape_rng, _frozen = self._streams(base_idx)
            shape = self.base._shape(shape_rng)
            sel = self.base._select(base_idx, rng, shape)
            if sel is not None:
                trip = make_triplet(self.base, sel, rng, self.cfg, self.corruptors)
                if trip is not None:
                    return trip
            base_idx = int(np.random.default_rng().integers(len(self.base)))
        return None


def scorer_collate(batch):
    """Collate for the scorer: one full batch-1 triplet per step.

    Inputs: batch -- a list holding exactly one triplet dict or None.
    Outputs: that dict, or None.
    Side effects: none.
    """
    return triplet_collate(batch)


def triplet_to_device(trip, device):
    """Move a worker-built triplet's tensors to the device, leaving its scalars as they are.

    The camera dicts move key by key because a `dict` is not a tensor; `make_triplet` can also
    hand back a cgroup whose entries are already device-local in an in-process build.

    Inputs: trip -- a triplet dict or None; device -- the target device.
    Outputs: the same dict, in place, with its tensors moved.
    Side effects: mutates `trip`.
    """
    if trip is None:
        return None
    for key in ('good', 'bad', 'anchor'):
        views, coords, cgroup = trip[key]
        trip[key] = ([v.to(device) for v in views], coords.to(device),
                     [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}
                      for cam in cgroup])
    for key in ('kpt_ids', 'counts', 'occlusion'):
        if trip.get(key) is not None:
            trip[key] = trip[key].to(device)
    return trip
