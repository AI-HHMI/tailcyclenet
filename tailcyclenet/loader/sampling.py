"""Sampling plumbing for the pose loader: the per-rank session shard, the step-indexed sampler, the
spawn-safe dataset handoff, worker initialisation, and the batch collate.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from tracktail.datasets.tracktail_dataset import custom_collate


def shard_sessions(sessions, rank: int = 0, world_size: int = 1):
    """Return a deterministic disjoint session shard, preserving source order.

    ``world_size=1`` is exactly the historical list.  Session metadata remains visible to
    registry construction; only the returned sessions are preloaded by ``PoseDataset``.
    """
    world_size, rank = int(world_size), int(rank)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f'invalid session shard rank={rank}, world_size={world_size}')
    return list(sessions)[rank::world_size]


class StepSampler(torch.utils.data.Sampler):
    """Draws indices with replacement and yields `(ordinal, index)`.

    The ordinal is the global training step -- the SAME on every rank while the index is not --
    and `PoseDataset.__getitem__` keys its cost-determining draws on it. Replacement, and no
    `DistributedSampler`: an index entry is one (session, group, animal) whatever the group's
    length, so it is a poor sampling weight and `_pick` re-draws inside `__getitem__`;
    independent per-rank streams is what the single-gpu loop already did.
    """

    def __init__(self, n: int, num_samples: int, generator=None, start: int = 0):
        """A sampler yielding `(ordinal, index)` pairs, drawing indices with replacement.

        Inputs: n -- the number of index entries.
                num_samples -- how many steps to yield.
                generator -- torch RNG for the index draws.
                start -- the first ordinal (the global step).
        """
        self.n, self.num_samples, self.generator, self.start = int(n), int(num_samples), generator, int(start)

    def __len__(self):
        """Number of steps this sampler will yield."""
        return self.num_samples

    def __iter__(self):
        """Yield `(start + k, index)` for k over `num_samples` random draws."""
        idx = torch.randint(0, self.n, (self.num_samples,), generator=self.generator)
        for k, i in enumerate(idx.tolist()):
            yield (self.start + k, i)


def _from_shared_pickle(buf: torch.Tensor):
    """Unpickle the object `SpawnHandoff` wrote into a shared-memory byte tensor."""
    return pickle.loads(memoryview(buf.numpy()))


class SpawnHandoff(Dataset):
    """Hand a big dataset to `spawn` loader workers through shared memory.

    Starting a `spawn` worker writes the pickled dataset into a pipe. The child reads that pipe only
    after it has imported torch/tracktail (~4 s), and a pipe holds 64 KB. So for a 361 MB
    PoseDataset each `Process.start()` blocks for that worker's import plus unpickle (~5 s), and
    the workers start ONE AT A TIME: 8 workers took ~40 s, on every rank. Here the dataset is
    pickled once into a shared-memory tensor, and torch's own reducer sends only a handle down
    the pipe. Every start then returns at once, the imports run in parallel, and each worker
    unpickles the same bytes it would have read from the pipe. A worker receives the wrapped
    dataset itself, not this wrapper.
    """

    def __init__(self, dataset):
        """Wrap `dataset`; nothing is pickled until the first worker starts."""
        self.dataset = dataset
        self._buf = None

    def __len__(self):
        """Length of the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, i):
        """Item `i` of the wrapped dataset (the in-process / `num_workers=0` path)."""
        return self.dataset[i]

    def __reduce__(self):
        """Pickle as (unpickle, shared bytes); the bytes are written once for all workers."""
        if self._buf is None:
            data = pickle.dumps(self.dataset, protocol=pickle.HIGHEST_PROTOCOL)
            self._buf = torch.empty(len(data), dtype=torch.uint8).share_memory_()
            self._buf.numpy()[:] = np.frombuffer(data, dtype=np.uint8)
        return _from_shared_pickle, (self._buf,)


def worker_init(worker_id):
    """A DataLoader `worker_init_fn`: pin cv2's and torch's thread pools to one thread, reseed
    imgaug (its global RNG survives fork otherwise, so every worker draws the same augmentations),
    and force imgaug's hue LUT cache to build here (a `spawn`ed dataset never re-runs `__init__`).
    numpy is deliberately NOT reseeded -- torch already decorrelates it per worker.
    """
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        import imgaug
        import imgaug.augmenters as iaa
    except ImportError:
        return
    imgaug.random.seed(int(np.random.randint(2 ** 31)))
    iaa.AddToHue()


def pose_collate(batch):
    """tracktail's collate for the first ten fields, plus this repo's three.

    `custom_collate` keeps only item 0's `cgroup` and forbids mixing 2D and 3D -- which is why
    batch_size is structurally 1 PER RANK. The same reason covers K: sessions may carry different
    keypoint subsets, so a mixed batch would fail the `kpt_ids` stack. Loud, and unreachable at
    batch_size 1.

    The box prompt field exists only when a box model is training (`_item` appends a 14th field
    then); it is absent for a plain run, so nothing downstream sees it.
    """
    batch = [b for b in batch if b is not None]
    out = custom_collate([b[:10] for b in batch])
    out['kpt_ids'] = torch.stack([b[10] for b in batch])
    out['kpt_prior'] = torch.stack([b[11] for b in batch])
    out['prompt_t'] = torch.stack([b[12] for b in batch])
    if len(batch[0]) > 13:
        out['box_prompt'] = torch.stack([b[13] for b in batch])
    return out
