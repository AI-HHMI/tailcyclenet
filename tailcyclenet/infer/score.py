"""`tailcyclenet score`: score a tracked root with a trained scorer and rank its worst windows.

Body of `tailcyclenet score` and `scripts/score_session.py` (`main()`). The deliverable is a score
table plus a WORST-FIRST ranking a human can act on -- offline QC of data that has no hand
labels, NOT inference gating and NOT identity arbitration. Writes `scores.pq` (per source frame
and keypoint), `report.txt` and `provenance.toml` under `--out`; lower means more likely wrong.

Scores are RELATIVE. Nothing here is a calibrated probability, no cross-root or cross-keypoint
threshold is implied, and the report deliberately prints no threshold: the ranking is the product.

The dataset root is READ-ONLY. No column is added to `keypoints.pq` / `points3d.pq` -- the format
spec is the human's and is untouched. Everything this writes goes to `--out`.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from ..checkpoints import provenance
from ..scorer.checkpoint import load_scorer_run
from ..scorer.contract import refuse_deleted_data_keys, scorer_contract, scorer_output_granularity
# Importing dataset eagerly pulls tracktail's legacy training dataset. Keep QC's
# table-only import free of pandas; scoring imports the loader only when it actually runs.
PoseDataset = None
if TYPE_CHECKING:
    from ..loader.config import LoaderConfig


def _loader_config(config: dict) -> LoaderConfig:
    """The run's `[data]` block as a `LoaderConfig`, with the box forced off.

    Inputs: config -- the scorer run's merged config.
    Outputs: a `LoaderConfig`.
    Side effects: none, or raises SystemExit naming unknown keys.
    """
    from ..loader.config import SCORER_PLUMBING, loader_config_from

    data_cfg = dict(config.get('data', {}))
    refuse_deleted_data_keys(data_cfg)
    return loader_config_from(data_cfg, extra=SCORER_PLUMBING)


def _check_names(registry, target_registry, data) -> None:
    """Refuse a target whose keypoints the scorer was not trained on, LISTING the unknown ones.

    The scorer is conditioned on the keypoint registry through `kpt_embed`: a calms21-trained
    scorer has no row for an allen or 3dpop keypoint, and the failure would otherwise be an index
    error at scoring time or -- worse -- a silently wrong identity vector. That is a real limit on
    the QC path's reach: one trained scorer per registry.

    Inputs: registry -- the scorer's registry; target_registry -- the target root's; data -- the
            path, for the message.
    Outputs: None, or raises SystemExit listing the unknown names.
    Side effects: none.
    """
    known = set(registry.names)
    extra = [n for n in target_registry.names if n not in known]
    if extra:
        raise SystemExit(
            f'{data}: keypoint(s) {extra} are not in this scorer\'s registry. A scorer is '
            'conditioned on the keypoints it was trained on, so it cannot score them -- train a '
            'scorer for this root. Refusing rather than scoring them with a meaningless row.')


def _span_indices(dataset, spans: dict) -> list[int]:
    """The `dataset.index` positions whose window STARTS inside a requested span.

    Filtering the INDEX, not each decoded item, is the whole point of `spans`: it exists so a
    targeted look at a clip's bad stretch is affordable, and decoding a window only to discover it
    is out of range spends exactly the cost the flag avoids (a long tracked group is thousands of
    windows, and a root is many groups).

    Inputs: dataset -- a `PoseDataset` built with `train=False`; spans -- `{(session, group,
            animal): (lo, hi)}`, a half-open start range `[lo, hi)`, the keys spelled as the loader's own `row` carries them.
    Outputs: the kept index positions, in index order.
    Side effects: none -- no window is realised, so no frame is decoded.
    """
    out = []
    animal_ids: dict[tuple[str, str], list[str]] = {}
    for i, item in enumerate(dataset.index):
        key = (item.session.session_id, item.gid)
        if key not in animal_ids:
            animal_ids[key] = [str(a) for a in item.session.labels(item.gid).animal_ids]
        ids = animal_ids[key]
        if item.animal >= len(ids):
            continue
        rng = spans.get((key[0], key[1], ids[item.animal]))
        if rng is not None and rng[0] <= int(item.start) < rng[1]:
            out.append(i)
    return out


def _to_device(views, coords, cgroup, kpt_ids, device):
    """Batch, and move to the model's device, everything one window's forward needs.

    A window arrives on CPU from the loader; the model may be on a GPU. Every tensor the forward
    touches has to travel -- the views, the coordinates, the camera dict AND the keypoint ids --
    because `conv3d` raises on a CPU input under a CUDA weight, and the camera tensors feed the
    decoder's geometry.

    Inputs: views -- a list of per-camera frame tensors; coords -- [T,K,R]; cgroup -- the camera
            dict; kpt_ids -- [K] int64; device -- the target device.
    Outputs: the same four, with views and coords batched to one window and everything moved.
        `kpt_ids` keeps its [K] shape -- the caller batches it.
    Side effects: none.
    """
    dev = torch.device(device)
    views = [v[None].to(dev, non_blocking=True) for v in views]
    coords = coords[None].to(dev, non_blocking=True)
    kpt_ids = kpt_ids.to(dev)
    cgroup = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
              for cam in cgroup]
    return views, coords, cgroup, kpt_ids


def _alias_registry(registry, target: str, alias: str):
    """`registry` with `alias`'s keypoint ids also filed under the dataset name `target`.

    A multi-dataset registry spells names `<dataset>-<keypoint>`, so a scorer trained on
    `qdmouse4m-ref-cleaned-v2` knows no `qdmouse4m-ref-full-*` name even when the two roots share a
    rig and a skeleton. The alias reuses the SAME ids (the same embedding rows), renaming only.

    Inputs: registry -- the scorer's registry; target -- the scored root's dataset name; alias --
            a dataset name the registry already holds.
    Outputs: a new Registry; `registry` itself is untouched.
    Side effects: raises SystemExit when `alias` is not in the registry.
    """
    from ..format.registry import Registry

    known = dict(registry.datasets)
    if alias not in known:
        raise SystemExit(f'--dataset-name {alias!r} is not in this scorer\'s registry; it holds '
                         f'{sorted(known)}')
    if target == alias:
        return registry
    ids = known[alias]
    names = list(registry.names)
    pre = f'{alias}-'
    for i in ids:
        if names[i].startswith(pre):
            names[i] = f'{target}-{names[i][len(pre):]}'
    out = {k: v for k, v in known.items() if k != alias}
    out[target] = tuple(ids)
    return Registry(names=tuple(names), datasets=tuple(sorted(out.items())))


def _cams_config(lc, cams):
    """The loader config with the per-window camera count `cams` applies.

    Inputs: lc -- the run's LoaderConfig; cams -- 'all' (every camera of the session; the
            default since 2026-10-28), 'run' or None (the run's own TRAINING `cams_to_sample`, a
            per-window random subset seeded by the window index -- how every score before that
            date was made), 'val' (`val_cams_to_sample`), or an int.
    Outputs: a LoaderConfig.
    Side effects: raises ValueError on an unknown spelling.
    """
    if cams is None or cams == 'run':
        return lc
    if cams == 'val':
        return replace(lc, cams_to_sample=lc.val_cams_to_sample)
    if cams == 'all':
        return replace(lc, cams_to_sample=1 << 16)
    try:
        n = int(cams)
    except (TypeError, ValueError):
        raise ValueError(f'cams must be run, val, all or an int, got {cams!r}') from None
    if n < 1:
        raise ValueError(f'cams must be >= 1, got {n}')
    return replace(lc, cams_to_sample=n)


def default_num_workers() -> int:
    """Loader workers for a scoring pass: every core this process may use but one, at most 8."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(0, min(8, cores - 1))


class _WindowStream(torch.utils.data.IterableDataset):
    """`ds.get_once` over `where`, each loader worker walking ONE contiguous chunk.

    Contiguous, not strided: consecutive windows of a group are consecutive frames, so a worker
    that owns a run of them keeps every camera's decoder moving forward (`PyAVReader.get_batch`
    continues instead of seeking) and decodes each source frame once. Yields `(index, item)`;
    `item` is None for a window that failed to build.
    """

    def __init__(self, ds, where, n_chunks: int):
        """Split `where` into `n_chunks` contiguous, near-equal chunks."""
        super().__init__()
        self.ds = ds
        self.chunks = [list(map(int, c)) for c in np.array_split(np.asarray(where, dtype=np.int64),
                                                                max(1, int(n_chunks)))]

    def __iter__(self):
        """Yield this worker's chunk, in index order."""
        info = torch.utils.data.get_worker_info()
        chunk = self.chunks[0 if info is None else info.id]
        for i in chunk:
            yield i, self.ds.get_once(i)


def _stream_worker_init(_worker_id: int) -> None:
    """A scoring worker holds a reader for every camera and leaves the cores to the decoders."""
    from ..data import reader_cache
    reader_cache.want_full_rig()
    torch.set_num_threads(1)


def _items(ds, where, num_workers: int, prefetch: int = 2):
    """An iterator of `(index, item)` over `where`; arrival order is NOT `where` order.

    `num_workers <= 0` decodes in this process, in order. Otherwise the decode runs in
    `_WindowStream` workers so it overlaps the model's forward (gotcha 10: nothing here has
    decoded in the parent, and the parent only reads tables). The workers fork HERE, eagerly,
    so a caller can move its model to the GPU only afterwards.
    """
    if num_workers <= 0:
        return ((i, ds.get_once(i)) for i in where)
    loader = torch.utils.data.DataLoader(
        _WindowStream(ds, where, num_workers), batch_size=None, num_workers=num_workers,
        prefetch_factor=max(1, int(prefetch)), worker_init_fn=_stream_worker_init,
        persistent_workers=False)
    return iter(loader)


def score_root(run: Path, data: str, split: str, device='cpu', limit: int | None = None,
               val_stride: int | None = None, spans: dict | None = None, coverage: list[dict] | None = None, *,
               checkpoint_info: dict | None = None, checkpoint: str | None = None,
               session: str | None = None, start: int | None = None,
               stop: int | None = None, num_workers: int = 0, cams='all',
               dataset_name: str | None = None, precision: str = 'bf16',
               camera_batch: bool = True) -> tuple:
    """Score every window of `data`'s `split` with the scorer in `run`.

    Inputs: run -- a scorer run folder; data -- a dataset root; split -- which split to score;
            device -- where to run the model; limit -- stop after this many windows (a long clip
            is thousands of them, and a first look must not need the whole split);
            val_stride -- window
            spacing, defaulting to the run's `n_frames` (non-overlapping); session -- score only
            the exact session id; start/stop -- zero-based half-open positions in the resulting
            window index (after session and span filtering); spans -- restrict
            scoring to windows starting inside a half-open frame range, as {(session, group,
            animal): (lo, hi)}, which is what makes a targeted look at a clip's bad stretch affordable;
            coverage -- optional list populated with one record per requested window; checkpoint --
            an explicit checkpoint filename passed to the scorer loader (for example
            ``checkpoint_best.pth``); checkpoint_info -- optional mutable mapping populated with
            the resolved checkpoint file and training iteration for output provenance;
            num_workers -- decode in this many loader workers, each walking a contiguous chunk
            of windows, overlapping the forward (0 = in this process); cams -- per-window camera
            count, see `_cams_config`; dataset_name -- read the root's keypoints as this
            registry dataset (`_alias_registry`); precision / camera_batch -- the scene-encode
            speed settings, `set_scene_speed`, with inference's defaults (bf16 autocast for the
            video encoder only; every camera in one encoder call).
    With workers, `limit` caps the windows REQUESTED (a failed window still counts); in this
    process it caps the windows scored. Rows come out in index order either way.
    The session for each row comes from the index entry: coordinates and keypoint ids retain the
    session's own name order, which may reorder or subset the dataset registry.
    Outputs: (Polars DataFrame of per-keypoint scores, the scorer's registry, the run's config).
    Side effects: decodes video frames; puts the model in eval mode. With workers the model is
    loaded on the CPU and moved to `device` only after they fork (they never touch CUDA).
    """
    model, config, registry, ckpt = load_scorer_run(
        Path(run), checkpoint=checkpoint, device='cpu' if num_workers > 0 else device)
    if checkpoint_info is not None:
        checkpoint_info['checkpoint_file'] = str(ckpt)
        checkpoint_data = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
        checkpoint_info['checkpoint_iteration'] = checkpoint_data.get('iteration')
    lc = _cams_config(_loader_config(config), cams)
    if dataset_name is not None:
        registry = _alias_registry(registry, Path(str(data).rstrip('/')).name, dataset_name)
    if val_stride is not None:
        lc = replace(lc, val_stride=int(val_stride))
    dataset_cls = PoseDataset
    if dataset_cls is None:
        from ..loader.dataset import PoseDataset as dataset_cls
    ds = dataset_cls(data, split, lc, registry_base=registry, train=False)
    _check_names(registry, ds.registry, data)

    where = list(range(len(ds)))
    if session is not None:
        session_where = [i for i in where
                         if str(ds.index[i].session.session_id) == str(session)]
        if not session_where:
            raise SystemExit(f'no session {session!r} found in {data}/{split}')
        where = session_where
        print(f'{len(where)} windows belong to session {session!r}')
    if spans is not None:
        span_where = set(_span_indices(ds, spans))
        if not span_where:
            raise SystemExit(
                'no window starts inside the given spans, so there is nothing to score: check '
                'the session/group/animal keys, and that span_start is a WINDOW start (a multiple '
                'of the run n_frames)')
        where = [i for i in where if i in span_where]
        print(f'{len(where)} windows remain inside the requested spans')
    if start is not None or stop is not None:
        lo = 0 if start is None else int(start)
        hi = len(where) if stop is None else int(stop)
        total = len(where)
        if lo < 0 or hi < lo or hi > total:
            raise ValueError(f'invalid window range [{lo}, {hi}) for {total} windows')
        where = where[lo:hi]
        print(f'window range [{lo}, {hi}) of {total}: {len(where)} windows')
    print(f'scoring up to {len(where)} windows from {data}/{split} with {ckpt.name}')

    output_granularity = scorer_output_granularity(config)
    rows_by: dict[int, list] = {}
    cov_by: dict[int, list] = {}
    model.eval()
    n_seen = 0
    if num_workers > 0 and limit is not None:
        where = where[:int(limit)]
    stream = _items(ds, where, num_workers)
    if num_workers > 0:
        model = model.to(device)
    model.set_scene_speed(precision=precision, camera_batch=camera_batch)
    t_start = time.perf_counter()
    for i, item in stream:
        if limit is not None and n_seen >= limit:
            break
        rows = rows_by.setdefault(int(i), [])
        cov = cov_by.setdefault(int(i), [])
        requested = ds.index[i]
        expected_session = requested.session.session_id
        expected_group = requested.gid
        expected_animal = str(requested.session.labels(requested.gid).animal_ids[requested.animal])
        expected_start = int(requested.start)
        if item is None:
            if coverage is not None:
                cov.append({'index': int(i), 'session': expected_session,
                            'group': expected_group, 'animal': expected_animal,
                            'start': expected_start, 'status': 'unscorable',
                            'reason': 'item_build_failed'})
            continue
        row_identity = (str(item[5]['session']), str(item[5]['group']),
                        str(item[5]['animal']), int(item[5]['start']))
        expected_identity = (str(expected_session), str(expected_group), expected_animal,
                             expected_start)
        if row_identity != expected_identity:
            if coverage is not None:
                cov.append({'index': int(i), 'session': expected_session,
                            'group': expected_group, 'animal': expected_animal,
                            'start': expected_start, 'status': 'identity_mismatch',
                            'reason': f'returned {row_identity}'})
                coverage.extend(c for j in where for c in cov_by.get(int(j), ()))
            raise RuntimeError(f'QC index {i} returned {row_identity}, expected '
                               f'{expected_identity}; refusing cross-group scoring')
        if coverage is not None:
            cov.append({'index': int(i), 'session': expected_session,
                        'group': expected_group, 'animal': expected_animal,
                        'start': expected_start, 'status': 'scored', 'reason': ''})
        n_seen += 1
        sess = ds.index[i].session
        views, coords, _vis, frames, cgroup, row, _qt, _v2, _p2d, _occ, kpt_ids, _pr, _pt = \
            item[:13]
        views, coords, cgroup, kpt_ids = _to_device(
            views, coords, cgroup, kpt_ids, device)
        with torch.no_grad():
            scores, precision = model(views, coords, cgroup, kpt_ids[None])
        scores = scores[0].detach().cpu().numpy()
        precision = precision[0].detach().cpu().numpy()
        observed_slots = torch.isfinite(coords[0]).all(-1).cpu().numpy()
        names = list(sess.names)
        if output_granularity == 'frame':
            if scores.ndim != 2:
                raise RuntimeError(
                    f'framewise scorer returned {scores.shape!r} after batch removal; expected '
                    '[T,K]. Check the run output_granularity and checkpoint contract.')
            source_frames = (frames.detach().cpu().numpy() if torch.is_tensor(frames)
                             else np.asarray(frames))
            source_frames = np.asarray(source_frames).reshape(-1)
            if len(source_frames) != scores.shape[0]:
                raise RuntimeError(
                    f'framewise scorer returned T={scores.shape[0]} but loader supplied '
                    f'{len(source_frames)} source frames; refusing ambiguous QC ownership')
            for local_t in range(scores.shape[0]):
                source_frame = int(source_frames[local_t])
                for ki in range(scores.shape[1]):
                    is_observed = bool(observed_slots[local_t, ki])
                    rows.append({
                        'dataset': row['dataset'], 'session': row['session'],
                        'group': row['group'], 'animal': row['animal'], 'mode': row['mode'],
                        'window_start': int(row['start']), 'start': int(row['start']),
                        'local_t': int(local_t), 'frame': source_frame,
                        'keypoint': names[ki] if ki < len(names) else str(ki),
                        'score': float(scores[local_t, ki]) if is_observed else float('nan'),
                        'precision': float(precision[local_t, ki]) if is_observed else float('nan'),
                        'observed': is_observed, 'n_cams': len(cgroup),
                    })
        else:
            if scores.ndim != 1:
                raise RuntimeError(
                    f'sequence scorer returned {scores.shape!r} after batch removal; expected [K]')
            observed = observed_slots.sum(0)
            for ki in range(scores.shape[0]):
                rows.append({
                    'dataset': row['dataset'], 'session': row['session'], 'group': row['group'],
                    'animal': row['animal'], 'mode': row['mode'], 'start': int(row['start']),
                    'keypoint': names[ki] if ki < len(names) else str(ki),
                    'score': float(scores[ki]), 'precision': float(precision[ki]),
                    'n_observed_frames': int(observed[ki]), 'n_cams': len(cgroup),
                })

    elapsed = time.perf_counter() - t_start
    if n_seen:
        print(f'scored {n_seen} windows in {elapsed:.1f} s ({elapsed / n_seen:.3f} s/window, '
              f'{num_workers} decode worker(s))')
    rows = [r for i in where for r in rows_by.get(int(i), ())]
    if coverage is not None:
        coverage.extend(c for i in where for c in cov_by.get(int(i), ()))

    if rows:
        table = pl.DataFrame(rows)
    elif output_granularity == 'frame':
        table = pl.DataFrame(schema={
            'dataset': pl.String, 'session': pl.String, 'group': pl.String,
            'animal': pl.String, 'mode': pl.String, 'window_start': pl.Int64,
            'start': pl.Int64, 'local_t': pl.Int64, 'frame': pl.Int64,
            'keypoint': pl.String, 'score': pl.Float64, 'precision': pl.Float64,
            'observed': pl.Boolean, 'n_cams': pl.Int64,
        })
    else:
        table = pl.DataFrame(schema={
            'dataset': pl.String, 'session': pl.String, 'group': pl.String,
            'animal': pl.String, 'mode': pl.String, 'start': pl.Int64,
            'keypoint': pl.String, 'score': pl.Float64, 'precision': pl.Float64,
            'n_observed_frames': pl.Int64, 'n_cams': pl.Int64,
        })
    return table, registry, config


def non_missing(table: pl.DataFrame, name: str) -> pl.Expr:
    """Match the table backend's default missing-value policy for a grouping/aggregate column."""
    expr = pl.col(name).is_not_null()
    if table.schema[name] in (pl.Float32, pl.Float64):
        expr = expr & pl.col(name).is_not_nan()
    return expr


def _aggregate_input(table: pl.DataFrame) -> pl.DataFrame:
    """Make IEEE NaNs participate in Polars aggregates like nulls do."""
    columns = []
    for name in ("score", "n_observed_frames"):
        if name in table.columns and table.schema[name] in (pl.Float32, pl.Float64):
            columns.append(pl.col(name).fill_nan(None))
    return table.with_columns(columns) if columns else table


def _as_report_float(value) -> float:
    """Render null aggregate values with the historical ``nan`` spelling."""
    return float("nan") if value is None else float(value)


def _canonical_frame_table(table: pl.DataFrame) -> pl.DataFrame:
    """Apply the declared last-window/last-occurrence ownership rule to frame rows.

    ``score_root`` returns the raw contextual window table so overlapping windows remain auditable.
    The user-facing table has one observed/missing row per source frame/keypoint.  Sorting by
    ``window_start`` then ``local_t`` makes the reducer deterministic and agrees with the inference
    ownership convention; duplicate clamp copies therefore cannot multiply a QC aggregate.
    """
    required = {'frame', 'keypoint', 'local_t'}
    if table.is_empty() or not required.issubset(table.columns):
        return table
    order = [c for c in ('dataset', 'session', 'group', 'animal', 'frame', 'keypoint',
                         'window_start', 'start', 'local_t') if c in table.columns]
    table = table.sort(order, nulls_last=True, maintain_order=True)
    keys = [c for c in ('dataset', 'session', 'group', 'animal', 'frame', 'keypoint')
            if c in table.columns]
    return table.group_by(keys, maintain_order=True).last()


def _frame_mode(table: pl.DataFrame) -> bool:
    """Return whether a QC table carries framewise ownership columns."""
    return {'frame', 'local_t'}.issubset(table.columns)


def _rank_frame(table: pl.DataFrame, top: int = 10) -> str:
    """Worst-first report for source-frame/keypoint rows."""
    aggregate = _aggregate_input(table)
    if 'observed' in aggregate.columns:
        valid = pl.col('observed').fill_null(False)
    else:
        valid = pl.lit(True)
    valid = valid & non_missing(aggregate, 'score')
    usable = aggregate.filter(valid)
    if usable.is_empty():
        return 'no observed frame/keypoint rows were scored'
    group_keys = ['dataset', 'session', 'group', 'animal']
    kpt_keys = ['dataset', 'session', 'group', 'keypoint']
    per_group = (usable.group_by(group_keys, maintain_order=True)
                 .agg(pl.col('score').min().alias('worst'),
                      pl.col('score').median().alias('median'), pl.len().alias('n'))
                 .sort(['worst', *group_keys], nulls_last=True, maintain_order=True))
    per_kpt = (usable.group_by(kpt_keys, maintain_order=True)
               .agg(pl.col('score').median().alias('median'), pl.len().alias('n'))
               .sort(['median', *kpt_keys], nulls_last=True, maintain_order=True))
    n_groups = usable.get_column('group').n_unique()
    lines = [f'scored {usable.height} observed (window, frame, keypoint) rows over '
             f'{n_groups} group(s)', '',
             'per-group minimum frame/keypoint score (worst first):']
    for row in per_group.head(top).iter_rows(named=True):
        lines.append(f'  {float(row["worst"]):>9.4f}  '
                     f'{row["dataset"]}/{row["session"]}/{row["group"]}/animal{row["animal"]}'
                     f'  (median {float(row["median"]):.4f}, {int(row["n"])} points)')
    lines += ['', 'worst (group, keypoint) frame pairs by median score:']
    for row in per_kpt.head(top).iter_rows(named=True):
        lines.append(f'  {float(row["median"]):>9.4f}  '
                     f'{row["dataset"]}/{row["session"]}/{row["group"]}/{row["keypoint"]}'
                     f'  (n {int(row["n"])})')
    root_median = per_kpt.get_column('median').median()
    weak = 0 if root_median is None else sum(
        value is not None and value < root_median
        for value in per_kpt.get_column('median').to_list())
    lines += ['', f'{weak} of {per_kpt.height} (group, keypoint) pairs fall below this '
              "root's own median. That is a RANK, not a threshold."]
    return '\n'.join(lines)


def rank(table: pl.DataFrame, top: int = 10) -> str:
    """The worst-first report: worst windows, then worst (group, keypoint) pairs.

    Scores are relative, so the report gives RANKS and counts and never a threshold -- a reader
    acting on a cut-off would be inventing one.

    Inputs: table -- ``score_root``'s Polars DataFrame; top -- how many rows to show per section.
    Outputs: the report as a string.
    Side effects: none. Missing grouping keys are excluded like pandas `groupby(dropna=True)`,
    and NaN scores are normalized to null before Polars aggregation.
    """
    if table.is_empty():
        return 'no windows were scored'
    if _frame_mode(table):
        return _rank_frame(_canonical_frame_table(table), top)

    group_keys = ['dataset', 'session', 'group', 'animal']
    keypoint_keys = ['dataset', 'session', 'group', 'keypoint']
    aggregate = _aggregate_input(table)
    group_input = aggregate.filter(pl.all_horizontal([non_missing(aggregate, k)
                                                       for k in group_keys]))
    kpt_input = aggregate.filter(pl.all_horizontal([non_missing(aggregate, k)
                                                     for k in keypoint_keys]))
    per_group = (group_input.group_by(group_keys, maintain_order=True)
                 .agg(pl.col('score').min().alias('worst'),
                      pl.col('score').median().alias('median'),
                      pl.len().alias('n'))
                 .sort(['worst', *group_keys], nulls_last=True, maintain_order=True))
    per_kpt = (kpt_input.group_by(keypoint_keys, maintain_order=True)
               .agg(pl.col('score').median().alias('median'),
                    pl.len().alias('n'),
                    pl.col('n_observed_frames').median().alias('obs'))
               .sort(['median', *keypoint_keys], nulls_last=True, maintain_order=True))

    n_groups = table.filter(non_missing(table, 'group')).get_column('group').n_unique()
    lines = [f'scored {table.height} (window, keypoint) rows over '
             f'{n_groups} group(s)',
             '',
             'per-group minimum keypoint score (worst first):']
    for row in per_group.head(top).iter_rows(named=True):
        worst = _as_report_float(row['worst'])
        median = _as_report_float(row['median'])
        lines.append(f'  {worst:>9.4f}  {row["dataset"]}/{row["session"]}/{row["group"]}'
                     f'/animal{row["animal"]}  (median {median:.4f}, {int(row["n"])} points)')

    lines += ['', 'worst (group, keypoint) pairs by median score:']
    for row in per_kpt.head(top).iter_rows(named=True):
        median = _as_report_float(row['median'])
        obs = _as_report_float(row['obs'])
        lines.append(f'  {median:>9.4f}  {row["dataset"]}/{row["session"]}/{row["group"]}'
                     f'/{row["keypoint"]}  (n {int(row["n"])}, obs {obs:.0f})')

    root_median = per_kpt.get_column('median').median()
    weak = 0 if root_median is None else sum(
        value is not None and value < root_median
        for value in per_kpt.get_column('median').to_list())
    lines += ['', f'{weak} of {per_kpt.height} (group, keypoint) pairs fall below this '
              'root\'s own median. That is a RANK, not a threshold.']
    return '\n'.join(lines)


def write_outputs(out: Path, table: pl.DataFrame, run: Path, data: str, split: str,
                  report: str, coverage: list[dict] | None = None, *,
                  checkpoint_file: str | Path | None = None,
                  checkpoint_iteration: int | None = None,
                  output_granularity: str | None = None,
                  session: str | None = None, start: int | None = None,
                  stop: int | None = None, val_stride: int | None = None,
                  cams=None, dataset_name: str | None = None, precision: str | None = None,
                  camera_batch: bool | None = None) -> None:
    """Write mode-aware QC tables, report, provenance, and optional coverage.

    Framewise ``table`` is the raw contextual window table returned by :func:`score_root`.  It is
    retained as ``window_scores.pq`` and reduced to one source-frame/keypoint row in
    ``scores.pq`` using the deterministic last-window/last-occurrence rule.  Sequence tables keep
    the historical single-file schema.
    """
    import toml

    out.mkdir(parents=True, exist_ok=True)
    frame_mode = _frame_mode(table) if output_granularity is None else output_granularity == 'frame'
    canonical = _canonical_frame_table(table) if frame_mode else table
    if frame_mode:
        table.write_parquet(out / 'window_scores.pq', compression='snappy')
    canonical.write_parquet(out / 'scores.pq', compression='snappy')
    (out / 'report.txt').write_text(report + '\n')
    output_provenance = {
        **provenance(), 'scorer_run': str(run), 'source_root': str(data), 'split': split,
        'n_rows': int(len(canonical)), 'output_granularity': 'frame' if frame_mode else 'sequence',
    }
    config_path = Path(run) / 'config.toml'
    if config_path.exists():
        import tomllib
        with config_path.open('rb') as handle:
            run_config = tomllib.load(handle)
        scorer_cfg = run_config.get('scorer', {})
        corr_cfg = scorer_cfg.get('corruption', {})
        for key, value in scorer_contract(run_config).items():
            if value is not None:
                output_provenance[f'scorer_{key}'] = value
        for key in ('loss_schema', 'corruption_mask_semantics', 'source_frame_duplicate_policy'):
            if key in scorer_cfg:
                output_provenance[key] = scorer_cfg[key]
        for key in ('segment_prob', 'segment_count', 'segment_len_frames',
                    'full_window_share', 'segment_types', 'min_corrupt_px', 'max_clean_px',
                    'reference_gate', 'out_of_view_policy'):
            if key in corr_cfg:
                output_provenance[f'corruption_{key}'] = corr_cfg[key]
        if 'val_stride' in run_config.get('data', {}):
            output_provenance['val_stride'] = run_config['data']['val_stride']
    if frame_mode:
        output_provenance.update({
            'n_raw_rows': int(len(table)),
            'frame_reducer': 'last_window_last_occurrence',
            'score_table': 'scores.pq',
            'raw_score_table': 'window_scores.pq',
        })
    if checkpoint_file is not None:
        output_provenance['checkpoint_file'] = str(checkpoint_file)
    if checkpoint_iteration is not None:
        output_provenance['checkpoint_iteration'] = int(checkpoint_iteration)
    if session is not None:
        output_provenance['session'] = str(session)
    if start is not None:
        output_provenance['window_start_index'] = int(start)
    if stop is not None:
        output_provenance['window_stop_index'] = int(stop)
    if val_stride is not None:
        output_provenance['val_stride'] = int(val_stride)
    if cams is not None:
        output_provenance['cams'] = str(cams)
    if dataset_name is not None:
        output_provenance['dataset_name'] = str(dataset_name)
    if precision is not None:
        output_provenance['precision'] = str(precision)
    if camera_batch is not None:
        output_provenance['camera_batch'] = bool(camera_batch)
    (out / 'provenance.toml').write_text(toml.dumps(output_provenance))
    if coverage is not None:
        coverage_schema = {
            'index': pl.Int64, 'session': pl.String, 'group': pl.String,
            'animal': pl.String, 'start': pl.Int64, 'status': pl.String, 'reason': pl.String,
        }
        pl.DataFrame(coverage, schema=coverage_schema).write_csv(out / 'coverage.csv')
    print(f'wrote {out}/scores.pq, {out}/report.txt, {out}/provenance.toml')


class _Deleted(argparse.Action):
    """A retired flag: refuse it by name and say why, rather than "unrecognized arguments"."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Raise SystemExit naming the flag.

        Inputs: the argparse Action protocol. Outputs: never returns.
        Side effects: raises SystemExit.
        """
        raise SystemExit(
            f'{option_string} is deleted: report 60 measured that offsetting the scorer\'s window '
            f'lattice does not make it catch more errors ("do not offset the scorer\'s windows")')


def main(argv=None):
    """Score a tracked root and write a worst-first QC report.

    Inputs: argv -- argument list, or None for `sys.argv`.
    Outputs: a process exit code.
    Side effects: writes `scores.pq`, `report.txt` and `provenance.toml` under `--out`; framewise
        runs also write raw `window_scores.pq` and canonicalize rows by source-frame ownership.
    """
    parser = argparse.ArgumentParser(prog='tailcyclenet score', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='a scorer run folder')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint filename; checkpoint_best.pth is validation-selected and '
                             'must be explicitly named')
    parser.add_argument('--data', required=True, help='the tracked root to score')
    parser.add_argument('--split', default='test', help='which split of --data to score')
    parser.add_argument('--session', default=None,
                        help='score one exact session id instead of the whole split')
    parser.add_argument('--start', type=int, default=None,
                        help='zero-based inclusive window-index start')
    parser.add_argument('--stop', type=int, default=None,
                        help='zero-based exclusive window-index stop')
    parser.add_argument('--out', required=True,
                        help='where QC artefacts go (frame mode also writes window_scores.pq)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--window-offset', action=_Deleted, help=argparse.SUPPRESS)
    parser.add_argument('--spans-csv', default=None,
                        help='CSV of session,group,animal,span_start,span_len: score only windows '
                             'starting inside each span (a targeted look at a clip bad stretch)')
    parser.add_argument('--val-stride', type=int, default=None,
                        help='window spacing; default is the run n_frames (non-overlapping)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='decode workers, each walking a contiguous run of windows so every '
                             'camera decodes forward once; default: cores - 1, at most 8; 0 = '
                             'decode in this process')
    parser.add_argument('--cams', default='all',
                        help="cameras per window: 'all' (default), 'run' (the run's training "
                             "cams_to_sample, a per-window random subset; scores before "
                             "2026-10-28), 'val' (val_cams_to_sample), or an int")
    parser.add_argument('--dataset-name', default=None,
                        help="read --data's keypoints as this dataset of the scorer's registry "
                             '(e.g. a full-length root of the root the scorer was trained on)')
    parser.add_argument('--no-tf32', action='store_true',
                        help='disable TF32 matmuls/convs (on by default)')
    parser.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16',
                        help='autocast dtype for the scene (video encoder) encode only, as in '
                             'infer; fp32 reproduces scores before 2026-10-28')
    parser.add_argument('--camera-batch', action=argparse.BooleanOptionalAction, default=True,
                        help='encode all cameras in one scene-encoder call per window')
    args = parser.parse_args(argv)
    from ..runtime import set_tf32
    set_tf32(not args.no_tf32)
    num_workers = default_num_workers() if args.num_workers is None else int(args.num_workers)
    if str(args.device).startswith('cpu') and args.num_workers is None:
        num_workers = 0
    spans = None
    if args.spans_csv:
        import csv as _csv
        spans = {}
        with open(args.spans_csv) as f:
            for r in _csv.DictReader(f):
                lo = int(r['span_start'])
                spans[(r['session'], r['group'], str(r['animal']))] = (lo, lo + int(r['span_len']))
    coverage = []
    checkpoint_info = {}
    table, _registry, _config = score_root(
        Path(args.run), args.data, args.split, args.device, args.limit,
        val_stride=args.val_stride, spans=spans, coverage=coverage,
        checkpoint=args.checkpoint, checkpoint_info=checkpoint_info,
        session=args.session, start=args.start, stop=args.stop, num_workers=num_workers,
        cams=args.cams, dataset_name=args.dataset_name, precision=args.precision,
        camera_batch=args.camera_batch)
    report = rank(table, args.top)
    print(report)
    write_outputs(Path(args.out), table, Path(args.run), args.data, args.split, report, coverage,
                  checkpoint_file=checkpoint_info.get('checkpoint_file'),
                  checkpoint_iteration=checkpoint_info.get('checkpoint_iteration'),
                  output_granularity=scorer_output_granularity(_config), session=args.session,
                  start=args.start, stop=args.stop, val_stride=args.val_stride,
                  cams=args.cams, dataset_name=args.dataset_name, precision=args.precision,
                  camera_batch=args.camera_batch)
    return 0
