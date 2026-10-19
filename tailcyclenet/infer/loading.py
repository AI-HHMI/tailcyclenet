"""Loading the networks a run deploys, and the compatibility rules between them and the source."""
from __future__ import annotations


def share_pose_scorer_encoder(pose_model, scorer_model, *, allow_mismatch,
                               pose_run, scorer_run) -> bool:
    """Share the raw video backbone, refusing unequal weights without explicit opt-in."""
    from ..model import share_video_encoder_if_identical

    shared = share_video_encoder_if_identical(pose_model, scorer_model)
    if not shared and not allow_mismatch:
        raise SystemExit(f'{scorer_run}: raw video-encoder weights do not exactly match '
                         f'{pose_run}; refusing two different scoring/pose encoders. Pass '
                         '--allow-pose-scorer-mismatch to opt in to separate encoder passes.')
    return shared


def _dataset_family(name: str) -> str:
    """The leading `-`-token of a dataset name: 'rat-city-combined' and 'rat-city-tracked' are
    both 'rat'; 'johnson-mouse-combined-aug' and 'johnson-mouse-combined' are both 'johnson'.

    A detector-vs-session dataset check needs SOME notion of "the same dataset", but this repo
    routinely trains a detector and a pose run, and evaluates them, on differently-suffixed roots
    that are the same underlying species/rig (`-combined` / `-aug` / `-tracked` / `-annotated`
    variants -- see `scratch/phase15/run.sh`'s own `rat-city-tracked` session against a
    `rat-city-combined` detector). Exact string equality would refuse that ALREADY-WORKING,
    ALREADY-CHECKED pairing. The first token is a coarse but effective family key for every
    dataset name in this repo; it is not exact by construction, so it only guards the unambiguous
    cross-species accident (a calms21 detector on a rat-city session), not a same-family variant.
    """
    return str(name).split('-', 1)[0].strip().lower()


def detector_matches_source(trained_datasets, trained_on, ds_name):
    """Multi-root checkpoints require exact membership; legacy/single-root keeps family matching."""
    if len(trained_datasets) > 1:
        return ds_name in trained_datasets
    return bool(trained_on and ds_name
                and _dataset_family(trained_on) == _dataset_family(ds_name))


def detector_box_source(det, ds_name, default):
    """Effective detector target for this source root, with a unique-family fallback."""
    by_root = getattr(det, 'box_sources', {}) or {}
    if ds_name in by_root:
        return by_root[ds_name]
    family = _dataset_family(ds_name) if ds_name else ''
    matches = [value for name, value in by_root.items()
               if family and _dataset_family(name) == family]
    return matches[0] if len(matches) == 1 else default
