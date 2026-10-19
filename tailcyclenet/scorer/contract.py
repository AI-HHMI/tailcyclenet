"""The scorer's checkpoint contract: its output granularity and the semantics a run is trained under.

A checkpoint records both so a resume, a warm start or a consumer can refuse a mismatch by name
before building a model; `scorer.checkpoint` applies these checks on load.
"""
from __future__ import annotations


# Scorer output mode is part of the model/checkpoint contract.  A scorer run written before
# framewise output existed has no key at all; treating an absent key as ``sequence`` is what keeps
# those weights byte-compatible.  New configs get an explicit value from configs/scorer.toml.
SCORER_OUTPUT_GRANULARITIES = ('sequence', 'frame')


def scorer_output_granularity(config: dict | None, *, legacy_default: str = 'sequence') -> str:
    """Resolve a scorer's output mode without guessing from tensor names.

    ``load_config`` overlays the scorer family base, so a new config normally contains
    ``[scorer].output_granularity = "frame"``.  Run folders are read back as already-merged
    TOML and old folders do not contain the key; those deliberately resolve to the legacy
    sequence contract.  ``legacy_default`` is exposed for callers that need to make the policy
    explicit while testing a partially-built config.
    """
    if legacy_default not in SCORER_OUTPUT_GRANULARITIES:
        raise ValueError(f'legacy_default must be one of {SCORER_OUTPUT_GRANULARITIES}, '
                         f'got {legacy_default!r}')
    value = (config or {}).get('scorer', {}).get('output_granularity')
    if value is None:
        value = legacy_default
    value = str(value).lower()
    if value not in SCORER_OUTPUT_GRANULARITIES:
        raise ValueError(
            f'[scorer].output_granularity = {value!r} is not one of '
            f'{SCORER_OUTPUT_GRANULARITIES}. Choose "frame" for the new framewise scorer or '
            '"sequence" for the legacy scorer.')
    return value


def scorer_checkpoint_granularity(ckpt: dict, *, legacy_default: str = 'sequence') -> str:
    """Resolve a checkpoint's scorer mode from metadata, with an old-file fallback.

    The top-level field is intentionally redundant with the embedded config.  It makes refusal
    possible before constructing a large scorer, and protects against a caller changing the run
    config after a checkpoint was written.  Old checkpoints have neither field and are legacy
    sequence checkpoints.
    """
    if not isinstance(ckpt, dict):
        raise ValueError(f'scorer checkpoint must be a dictionary, got {type(ckpt).__name__}')
    metadata = ckpt.get('scorer_metadata')
    candidates = []
    for key in ('output_granularity', 'scorer_output_granularity'):
        if key in ckpt:
            candidates.append((key, ckpt[key]))
    if isinstance(metadata, dict) and 'output_granularity' in metadata:
        candidates.append(('scorer_metadata.output_granularity',
                           metadata['output_granularity']))
    embedded = ckpt.get('config')
    if isinstance(embedded, dict) and isinstance(embedded.get('scorer'), dict):
        if 'output_granularity' in embedded['scorer']:
            candidates.append(('config.scorer.output_granularity',
                               embedded['scorer']['output_granularity']))
    values = {str(v).lower() for _, v in candidates}
    if len(values) > 1:
        details = ', '.join(f'{name}={value!r}' for name, value in candidates)
        raise ValueError(f'scorer checkpoint has conflicting output_granularity metadata: {details}')
    value = next(iter(values), None)
    if value is None:
        value = scorer_output_granularity(embedded, legacy_default=legacy_default)
    value = str(value).lower()
    if value not in SCORER_OUTPUT_GRANULARITIES:
        raise ValueError(
            f'scorer checkpoint has invalid output_granularity {value!r}; expected '
            f'{SCORER_OUTPUT_GRANULARITIES}')
    return value


def require_scorer_granularity(requested: str, checkpoint: str, *, where='checkpoint') -> None:
    """Raise a named refusal when a full scorer state crosses output modes."""
    requested = str(requested).lower()
    checkpoint = str(checkpoint).lower()
    if requested not in SCORER_OUTPUT_GRANULARITIES:
        raise ValueError(f'requested scorer output_granularity {requested!r} is invalid')
    if checkpoint not in SCORER_OUTPUT_GRANULARITIES:
        raise ValueError(f'{where}: checkpoint output_granularity {checkpoint!r} is invalid')
    if requested != checkpoint:
        raise ValueError(
            f'{where}: scorer output_granularity mismatch: checkpoint is {checkpoint!r}, but '
            f'the requested run is {requested!r}. Full-state scorer resume across sequence/frame '
            'modes is refused; use a pose checkpoint as a warm start instead.')


def scorer_contract(config: dict | None) -> dict:
    """Extract the scorer loss, mask, and corruption settings that define a run's semantics."""
    config = config or {}
    scorer = config.get('scorer', {}) if isinstance(config, dict) else {}
    corr = scorer.get('corruption', {}) if isinstance(scorer, dict) else {}
    data = config.get('data', {}) if isinstance(config, dict) else {}
    mode = scorer_output_granularity(config)
    segment_count = corr.get('n_segments', corr.get('segment_count', 1))
    if isinstance(segment_count, (list, tuple)):
        segment_count = list(segment_count)
    else:
        segment_count = int(segment_count)
    loss_schema = scorer.get('loss_schema')
    mask_semantics = scorer.get('corruption_mask_semantics')
    duplicate_policy = scorer.get('source_frame_duplicate_policy')
    if mode == 'sequence':
        if loss_schema in (None, 'framewise-v1'):
            loss_schema = 'sequence-v0'
        if mask_semantics in (None, 'far-observed-in-view-anchor-observed'):
            mask_semantics = 'legacy'
        if duplicate_policy in (None, 'inverse_multiplicity'):
            duplicate_policy = 'legacy'
    else:
        loss_schema = loss_schema or 'framewise-v1'
        mask_semantics = mask_semantics or 'far-observed-in-view-anchor-observed'
        duplicate_policy = duplicate_policy or 'inverse_multiplicity'
    sequence_gate = bool(corr.get('sequence_far_gate', False))
    if mode == 'sequence' and not sequence_gate:
        corruption_values = {
            'min_corrupt_px': 0.0, 'max_clean_px': 0.0, 'reference_gate': 'legacy',
            'reference_margin_px': 0.0, 'segment_prob': 0.0, 'segment_count': 1,
            'segment_len_frames': [1, 1], 'full_window_share': 0.0,
            'segment_types': None, 'no_active_slot_policy': 'legacy',
            'out_of_view_policy': 'legacy',
        }
    else:
        corruption_values = {
            'min_corrupt_px': float(corr.get('min_corrupt_px', 0.0)),
            'max_clean_px': float(corr.get('max_clean_px', 0.0)),
            'reference_gate': corr.get('reference_gate', 'source_far'),
            'reference_margin_px': float(corr.get('reference_margin_px', 0.0)),
            'segment_prob': float(corr.get('segment_prob', 0.0)),
            'segment_count': segment_count,
            'segment_len_frames': corr.get('segment_len_frames',
                                            corr.get('segment_length_frames', [2, 6])),
            'full_window_share': float(corr.get(
                'segment_full_window_share', corr.get(
                    'full_window_share', corr.get('segment_full_window_prob', 0.0)))),
            'segment_types': corr.get('segment_types'),
            'no_active_slot_policy': corr.get('no_active_slot_policy', 'retry'),
            'out_of_view_policy': corr.get('out_of_view_policy', 'exclude'),
        }
    return {
        'output_granularity': mode,
        'loss_schema': loss_schema,
        'corruption_mask_semantics': mask_semantics,
        'source_frame_duplicate_policy': duplicate_policy,
        'two_d_sampling': scorer.get('two_d_sampling', 'legacy-3d-single-view'),
        'prob_2d_only': float(data.get('prob_2d_only', 0.25)),
        'n_frames': int(data['n_frames']) if 'n_frames' in data else None,
        'frame_strides': list(data['frame_strides']) if 'frame_strides' in data else None,
        'min_valid_frames': int(scorer.get('min_valid_frames', corr.get('min_valid_frames', 6))),
        'const_offset_prob': float(corr.get('const_offset_prob', 0.0)),
        'frame_noise_prob': float(corr.get('frame_noise_prob', 0.0)),
        'gradual_drift_prob': float(corr.get('gradual_drift_prob', 0.0)),
        'sinusoid_prob': float(corr.get('sinusoid_prob', 0.0)),
        'point_drop_prob': float(corr.get('point_drop_prob', 0.0)),
        'point_drop_max_frac': float(corr.get('point_drop_max_frac', 0.0)),
        'point_drop_bernoulli_rate': float(corr.get('point_drop_bernoulli_rate', 0.0)),
        'mag_2d': dict(corr.get('mag_2d', {})),
        'mag_3d': dict(corr.get('mag_3d', {})),
        'pointwise_weight': float(scorer.get('pointwise_weight', 0.0)),
        'pointwise_balance': bool(scorer.get('pointwise_balance', True)),
        'pointwise_label_smoothing': float(scorer.get('pointwise_label_smoothing', 0.0)),
        'inactive_consistency_weight': float(scorer.get('inactive_consistency_weight', 0.0)),
        'anchor_consistency_weight': float(scorer.get('anchor_consistency_weight', 0.0)),
        **corruption_values,
        'sequence_far_gate': sequence_gate,
    }


def scorer_contract_mismatches(config: dict, ckpt: dict) -> dict:
    """Return contract disagreements against the request and within checkpoint sources."""
    expected = scorer_contract(config)
    metadata = ckpt.get('scorer_metadata') if isinstance(ckpt, dict) else None
    embedded = ckpt.get('config') if isinstance(ckpt, dict) else None
    sources = []
    if isinstance(metadata, dict) and isinstance(metadata.get('contract'), dict):
        sources.append(('metadata', metadata['contract']))
    if isinstance(embedded, dict):
        sources.append(('config', scorer_contract(embedded)))
    if isinstance(metadata, dict) and not sources:
        sources.append(('metadata', {key: metadata[key] for key in (
            'output_granularity', 'loss_schema', 'corruption_mask_semantics',
            'source_frame_duplicate_policy') if key in metadata}))
    mismatches = {}
    for source_name, source in sources:
        for key, value in expected.items():
            if key in source and source[key] != value:
                mismatches[f'{source_name}.{key}'] = (source[key], value)
    for left_index, (left_name, left) in enumerate(sources):
        for right_name, right in sources[left_index + 1:]:
            for key in set(left) & set(right):
                if left[key] != right[key]:
                    mismatches[f'{left_name}.{key} vs {right_name}.{key}'] = (
                        left[key], right[key])
    return mismatches


def require_scorer_contract(config: dict, ckpt: dict, *, where='checkpoint') -> None:
    """Refuse a scorer checkpoint whose recorded loss/data contract changed."""
    if scorer_output_granularity(config) == 'frame':
        metadata = ckpt.get('scorer_metadata') if isinstance(ckpt, dict) else None
        metadata_contract = metadata.get('contract') if isinstance(metadata, dict) else None
        embedded = ckpt.get('config') if isinstance(ckpt, dict) else None
        required = set(scorer_contract({'scorer': {'output_granularity': 'frame'}}))
        if isinstance(metadata_contract, dict):
            missing = required - set(metadata_contract)
            if missing:
                embedded_contract = scorer_contract(embedded) if isinstance(embedded, dict) else {}
                unresolved = missing - set(embedded_contract)
                if unresolved:
                    raise ValueError(
                        f'{where}: incomplete frame scorer contract metadata; missing '
                        f'{sorted(unresolved)}')
        elif not isinstance(embedded, dict):
            raise ValueError(f'{where}: frame scorer checkpoint has no complete semantic contract')
    mismatches = scorer_contract_mismatches(config, ckpt)
    if mismatches:
        details = ', '.join(f'{key}: checkpoint={old!r}, requested={new!r}'
                            for key, (old, new) in sorted(mismatches.items()))
        raise ValueError(f'{where}: scorer semantic contract mismatch ({details})')


def scorer_checkpoint_fields(config: dict, selection_metric: str | None = None,
                             selection_value: float | None = None) -> dict:
    """The scorer-only keys `checkpoints.save_checkpoint(kind='scorer', extra=...)` writes beside
    the weights: the granularity (twice, for old readers), the contract, and the selection."""
    mode = scorer_output_granularity(config)
    contract = scorer_contract(config)
    out = {'output_granularity': mode, 'scorer_output_granularity': mode,
           'scorer_metadata': {
               'output_granularity': mode,
               'loss_schema': contract['loss_schema'],
               'corruption_mask_semantics': contract['corruption_mask_semantics'],
               'source_frame_duplicate_policy': contract['source_frame_duplicate_policy'],
               'contract': contract}}
    if selection_metric is not None:
        out['scorer_selection_metric'] = str(selection_metric)
    if selection_value is not None:
        out['scorer_selection_value'] = float(selection_value)
    return out
