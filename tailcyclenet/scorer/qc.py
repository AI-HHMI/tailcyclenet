"""Score a tracked root with a trained scorer and rank its worst windows.

Section 8 of the plan. The deliverable is a score table plus a WORST-FIRST ranking a human can act
on -- offline QC of data that has no hand labels, NOT inference gating and NOT identity
arbitration.

Scores are RELATIVE. Nothing here is a calibrated probability, no cross-root or cross-keypoint
threshold is implied, and the report deliberately prints no threshold: the ranking is the product.

The dataset root is READ-ONLY. No column is added to `keypoints.pq` / `points3d.pq` -- the format
spec is the human's and is untouched. Everything this writes goes to `--out`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..checkpoints import load_scorer_run, provenance
from ..dataset import LoaderConfig, PoseDataset


def _loader_config(config: dict) -> LoaderConfig:
    """The run's `[data]` block as a `LoaderConfig`, with the box forced off.

    Inputs: config -- the scorer run's merged config.
    Outputs: a `LoaderConfig`.
    Side effects: none, or raises SystemExit naming unknown keys.
    """
    data_cfg = dict(config.get('data', {}))
    known = set(LoaderConfig.__dataclass_fields__) | {'path', 'num_workers'}
    unknown = set(data_cfg) - known
    if unknown:
        raise SystemExit(f'[data]: unknown key(s) {sorted(unknown)}')
    return LoaderConfig(**{k: v for k, v in data_cfg.items()
                           if k in LoaderConfig.__dataclass_fields__})


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


def _windows(dataset):
    """Enumerate the dataset's fixed windows as `(session, item)`, skipping build failures.

    The SESSION comes back with the item because the keypoint AXIS belongs to it: `coords` and
    `kpt_ids` are both laid out in the session's own `names` order, which may reorder or subset the
    dataset's. Labelling a score with the dataset's name order would mislabel every reordered
    session -- and `Registry.ids_for` exists precisely because that ordering differs per session.

    Inputs: dataset -- a `PoseDataset` built with `train=False`.
    Outputs: an iterator of `(session, item)`.
    Side effects: decodes video frames.
    """
    for i in range(len(dataset)):
        item = dataset[i]
        if item is not None:
            yield dataset.index[i].session, item


def score_root(run: Path, data: str, split: str, device='cpu', limit: int | None = None) -> tuple:
    """Score every window of `data`'s `split` with the scorer in `run`.

    Inputs: run -- a scorer run folder; data -- a dataset root; split -- which split to score;
            device -- where to run the model; limit -- stop after this many windows (a long clip
            is thousands of them, and a first look must not need the whole split).
    Outputs: (DataFrame of per-keypoint scores, the scorer's registry, the run's config).
    Side effects: decodes video frames; puts the model in eval mode.
    """
    model, config, registry, ckpt = load_scorer_run(Path(run), device=device)
    lc = _loader_config(config)
    ds = PoseDataset(data, split, lc, registry_base=registry, train=False)
    _check_names(registry, ds.registry, data)
    print(f'scoring {len(ds)} windows from {data}/{split} with {ckpt.name}')

    rows = []
    model.eval()
    for n_seen, (sess, item) in enumerate(_windows(ds)):
        if limit is not None and n_seen >= limit:
            break
        views, coords, _vis, _frames, cgroup, row, _qt, _v2, _p2d, _occ, kpt_ids, _pr, _pt = \
            item[:13]
        views = [v[None] for v in views]
        coords = coords[None]
        with torch.no_grad():
            scores, precision = model(views, coords, cgroup, kpt_ids[None])
        scores = scores[0].cpu().numpy()
        precision = precision[0].cpu().numpy()
        observed = torch.isfinite(coords[0]).all(-1).sum(0).cpu().numpy()
        names = list(sess.names)
        for ki in range(scores.shape[0]):
            rows.append({
                'dataset': row['dataset'], 'session': row['session'], 'group': row['group'],
                'animal': row['animal'], 'mode': row['mode'], 'start': int(row['start']),
                'keypoint': names[ki] if ki < len(names) else str(ki),
                'score': float(scores[ki]), 'precision': float(precision[ki]),
                'n_observed_frames': int(observed[ki]), 'n_cams': len(cgroup),
            })
    return pd.DataFrame(rows), registry, config


def rank(table: pd.DataFrame, top: int = 10) -> str:
    """The worst-first report: worst windows, then worst (group, keypoint) pairs.

    Scores are relative, so the report gives RANKS and counts and never a threshold -- a reader
    acting on a cut-off would be inventing one.

    Inputs: table -- `score_root`'s DataFrame; top -- how many rows to show per section.
    Outputs: the report as a string.
    Side effects: none.
    """
    if table.empty:
        return 'no windows were scored'
    lines = [f'scored {len(table)} (window, keypoint) rows over '
             f'{table.group.nunique()} group(s)',
             '',
             'per-group minimum keypoint score (worst first):']
    per_group = (table.groupby(['dataset', 'session', 'group', 'animal'])
                 .agg(worst=('score', 'min'), median=('score', 'median'), n=('score', 'size'))
                 .reset_index().sort_values('worst'))
    for _, r in per_group.head(top).iterrows():
        lines.append(f'  {r["worst"]:>9.4f}  {r["dataset"]}/{r["session"]}/{r["group"]}'
                     f'/animal{r["animal"]}  (median {r["median"]:.4f}, {int(r["n"])} points)')

    lines += ['', 'worst (group, keypoint) pairs by median score:']
    per_kpt = (table.groupby(['dataset', 'session', 'group', 'keypoint'])
               .agg(median=('score', 'median'), n=('score', 'size'),
                    obs=('n_observed_frames', 'median'))
               .reset_index().sort_values('median'))
    for _, r in per_kpt.head(top).iterrows():
        lines.append(f'  {r["median"]:>9.4f}  {r["dataset"]}/{r["session"]}/{r["group"]}'
                     f'/{r["keypoint"]}  (n {int(r["n"])}, obs {r["obs"]:.0f})')

    weak = per_kpt['median'] < per_kpt['median'].median()
    lines += ['', f'{int(weak.sum())} of {len(per_kpt)} (group, keypoint) pairs fall below this '
                  'root\'s own median. That is a RANK, not a threshold.']
    return '\n'.join(lines)


def write_outputs(out: Path, table: pd.DataFrame, run: Path, data: str, split: str,
                  report: str) -> None:
    """Write `scores.pq`, `report.txt` and `provenance.toml` under `out`.

    Inputs: out -- the output directory; table -- the score DataFrame; run -- the scorer run;
            data -- the scored root; split -- the split scored; report -- the ranking text.
    Outputs: none.
    Side effects: creates `out` and writes three files. The scored root is not touched.
    """
    import toml

    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / 'scores.pq', index=False)
    (out / 'report.txt').write_text(report + '\n')
    (out / 'provenance.toml').write_text(toml.dumps({
        **provenance(), 'scorer_run': str(run), 'source_root': str(data), 'split': split,
        'n_rows': int(len(table)),
    }))
    print(f'wrote {out}/scores.pq, {out}/report.txt, {out}/provenance.toml')
