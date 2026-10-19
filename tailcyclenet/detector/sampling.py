"""Sampling for `BoxDataset`: index weights (`IndexWeights`), samplers, collation."""
from __future__ import annotations

import numpy as np
import torch


class IndexWeights:
    """Sampling weights and realised draw mixes over a `BoxDataset`'s `index` (a mixin).

    Reads `index`, `datasets`, `root_ids` and `root_names`; writes nothing.
    """

    def _base_weights(self, indices, annot_frac=None):
        """Existing view/cohort rule applied to the selected index positions."""
        indices = np.asarray(indices, dtype=np.int64)
        n = len(indices)
        src = np.array([self.index[int(i)][0].label_source for i in indices])
        keys = np.array([f'{self.index[int(i)][0].path}/{self.index[int(i)][1]}/'
                         f'{self.index[int(i)][3]}' for i in indices])
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        per_entry = 1.0 / counts[inv].astype(np.float64)
        present = [c for c in sorted(set(src.tolist()))]
        if annot_frac is not None and not 0.0 <= float(annot_frac) <= 1.0:
            raise ValueError(f'annot_frac must be in [0, 1], got {annot_frac}')
        w = np.zeros(n, dtype=np.float64)
        for cohort in present:
            m = src == cohort
            if annot_frac is not None and len(present) > 1 and cohort in ('annotated', 'tracked'):
                share = float(annot_frac) if cohort == 'annotated' else 1.0 - float(annot_frac)
            else:
                share = float(m.sum()) / n
            tot = per_entry[m].sum()
            if tot > 0:
                w[m] = per_entry[m] * (share / tot)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('default_train_weights: weights are all zero or non-finite')
        return w / w.sum()

    def default_train_weights(self, annot_frac=None, balance_datasets=True):
        """View-uniform sampling weights, optionally balanced to equal root shares.

        One-root output is exactly the historical rule. With multiple roots, `annot_frac` applies
        within each root if balanced, and globally if roots are pooled.
        """
        all_indices = np.arange(len(self.index), dtype=np.int64)
        if not balance_datasets or len(self.datasets) == 1:
            return self._base_weights(all_indices, annot_frac)
        w = np.zeros(len(self.index), dtype=np.float64)
        for root_id in range(len(self.datasets)):
            ix = np.flatnonzero(np.asarray(self.root_ids) == root_id)
            if ix.size:
                w[ix] = self._base_weights(ix, annot_frac) / len(self.datasets)
        return w

    def cohort_mix(self, weights=None):
        """Realised share of train draws per cohort. Reporting only -- the mix is invisible in
        the loss curve, so `detector.train` prints it, exactly as `PoseDataset.mix` is.
        """
        src = np.array([s.label_source for s, _, _, _ in self.index])
        w = (np.full(len(self.index), 1.0 / max(len(self.index), 1)) if weights is None
             else np.asarray(weights, dtype=np.float64) / np.sum(weights))
        return {c: float(w[src == c].sum()) for c in sorted(set(src.tolist()))}

    def root_mix(self, weights=None):
        """Realised draw probability for each dataset root."""
        ids = np.asarray(self.root_ids, dtype=np.int64)
        w = (np.full(len(self.index), 1.0 / max(len(self.index), 1)) if weights is None
             else np.asarray(weights, dtype=np.float64) / np.sum(weights))
        return {name: float(w[ids == i].sum()) for i, name in enumerate(self.root_names)}

    def root_cohort_mix(self, weights=None):
        """Realised label-source shares within each dataset root."""
        ids = np.asarray(self.root_ids, dtype=np.int64)
        w = (np.full(len(self.index), 1.0 / max(len(self.index), 1)) if weights is None
             else np.asarray(weights, dtype=np.float64) / np.sum(weights))
        out = {}
        for i, name in enumerate(self.root_names):
            m = ids == i
            total = w[m].sum()
            src = np.array([s.label_source for s, _, _, _ in self.index])
            out[name] = {cohort: float(w[m & (src == cohort)].sum() / total)
                         for cohort in sorted(set(src[m].tolist()))}
        return out

    def train_weights(self, annot_frac=None, alpha=None, balance_datasets=True):
        """Compose cohort/view and alpha weighting, then enforce root balance if requested."""
        if not balance_datasets or len(self.datasets) == 1:
            base = self.default_train_weights(annot_frac, balance_datasets=False)
            alpha_w = self.alpha_weights(alpha)
            return base if alpha_w is None else base * alpha_w
        weights = np.zeros(len(self.index), dtype=np.float64)
        ids = np.asarray(self.root_ids, dtype=np.int64)
        alpha_w = self.alpha_weights(alpha)
        for root_id in range(len(self.datasets)):
            ix = np.flatnonzero(ids == root_id)
            local = self._base_weights(ix, annot_frac)
            if alpha_w is not None:
                local *= alpha_w[ix]
            total = local.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f'train_weights: root {self.root_names[root_id]!r} has no '
                                 'positive finite sampling weight')
            weights[ix] = local / total / len(self.datasets)
        return weights

    def alpha_weights(self, alpha):
        """Per-entry weights giving each GROUP (session, gid) total draw probability
        proportional to `n_views ** alpha` (per-entry weight `n_views ** (alpha - 1)`).

        `alpha = 1.0` is frame-uniform (all ones), `0.0` group-uniform, `0.5` sqrt-damps between
        the two; `None` (default) returns None. A no-op only where every group has the same
        `n_views` (3dpop does not, and responds: detector_v2 plan B3). Composes with the default
        train weights by elementwise product (renormalised) in `detector.train`.
        """
        if alpha is None:
            return None
        alpha = float(alpha)
        keys = np.array([f'{s.path}/{g}' for s, g, _, _ in self.index])
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_views = counts[inv].astype(np.float64)
        w = n_views ** (alpha - 1.0)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('alpha_weights: sampling weights are all zero or non-finite -- '
                             'check alpha')
        return w


class ChunkShuffle(torch.utils.data.Sampler):
    """A shuffle that keeps a worker inside a few videos at a time.

    `BoxDataset.index` is built session-by-session, so a run of contiguous index positions stays
    inside one video, and a plain `shuffle=True` would send every item to a different container
    and thrash `dataset._reader`'s cache (486 ms per batch of 16 against a 16 ms GPU step, on a
    multi-session root). Shuffling BLOCKS instead, and pooling `mix` of them so a batch still
    spans several sessions, costs 40 ms -- and a batch drawn from one video is still a correlated
    gradient step. PASS `chunk=dataset.chunk`; the default 512 was set against a one-group root.

    Image-directory datasets do not need this and are not harmed by it: their frames are separate
    files, so locality buys nothing and costs nothing.
    """

    def __init__(self, n, chunk=512, mix=4, seed=23):
        """Build the block shuffle over `n` items with `chunk`-sized locality blocks."""
        self.n, self.chunk, self.mix, self.seed = n, chunk, mix, seed
        self.epoch = 0

    def __len__(self):
        """Number of items shuffled."""
        return self.n

    def __iter__(self):
        """Yield the epoch's shuffled item order, block-local and `mix`-pooled."""
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        starts = np.arange(0, self.n, self.chunk)
        rng.shuffle(starts)
        for i in range(0, len(starts), self.mix):
            pool = np.concatenate([np.arange(s, min(s + self.chunk, self.n))
                                   for s in starts[i:i + self.mix]])
            rng.shuffle(pool)
            yield from (int(j) for j in pool)


class CohortSampler(torch.utils.data.Sampler):
    """Draw weighted index positions with replacement.

    `default_train_weights` always returns a view-uniform base and `alpha_weights` may modify it.
    Replacement makes a configured cohort share a property of each draw rather than of a finite
    epoch. `__len__` remains the index length so `iters` keeps its existing meaning.

    Locality is deliberately traded for the weighted draw. Train inputs for the multi-cohort roots
    are frame directories; calms21 is the video-backed root and has one cohort.
    """

    def __init__(self, weights, num_samples=None, seed=23):
        """Build a replacement sampler over the index from `weights` (see class docstring).

        One cumulative array and one `searchsorted` per draw, as `PoseDataset._pick` does.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or not w.size:
            raise ValueError(f'weights must be a non-empty 1-D array, got shape {w.shape}')
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError('weights must be finite, non-negative and not all zero')
        self.cum = np.cumsum(w / w.sum())
        self.num_samples = int(len(w) if num_samples is None else num_samples)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        """Number of draws per epoch."""
        return self.num_samples

    def __iter__(self):
        """Yield `num_samples` weighted draws with replacement.

        `searchsorted` can return `len(cum)` on a draw of exactly 1.0, so draws are clamped
        rather than letting a one-in-2^53 sample raise IndexError 14,000 iterations in.
        """
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        draws = np.searchsorted(self.cum, rng.random(self.num_samples))
        yield from (int(j) for j in np.minimum(draws, len(self.cum) - 1))


def box_collate(batch):
    """Collate `BoxDataset` items (named dicts) into one named-dict batch.

    Pad the box axis with NaN so a batch can hold different animal counts. NaN, not zero, and
    that carries all the way to the loss: a padded row is "no animal", which is the same signal
    a real animal with no finite point in this view sends.

    Items are explicit named dicts (`x`, `boxes`, `negative_supervision`), dispatched BY NAME,
    not by rank or tuple length.
    """
    xs = torch.stack([b['x'] for b in batch])
    n = max(b['boxes'].shape[0] for b in batch)
    boxes = torch.full((len(batch), n, 4), float('nan'))
    for i, b in enumerate(batch):
        boxes[i, :b['boxes'].shape[0]] = b['boxes']
    out = {'x': xs, 'boxes': boxes}

    if any('negative_supervision' in b for b in batch):
        out['negative_supervision'] = torch.tensor(
            [bool(b.get('negative_supervision', True)) for b in batch], dtype=torch.bool)
    return out


def split_batch(batch):
    """A collated batch dict -> `(x, boxes, negative_supervision_or_None)`."""
    return batch['x'], batch['boxes'], batch.get('negative_supervision')
