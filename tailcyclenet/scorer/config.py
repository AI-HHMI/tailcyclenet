"""The scorer family's config: known keys, the selection metric, and value validation."""
from __future__ import annotations

from .contract import SCORER_OUTPUT_GRANULARITIES, scorer_output_granularity
from .corruption import SEGMENT_TYPES

SCORER_HEAD_KEYS = ('pool_num_heads', 'score_hidden', 'use_precision', 'output_granularity')

# Settings consumed by the scorer head/loss.  Keeping this list here gives configs the same typo
# refusal as the pose trainer: a misspelled loss arm must not silently run the historical default.
SCORER_CONFIG_KEYS = {
    *SCORER_HEAD_KEYS,
    'triplet_margin', 'precision_reg_weight', 'score_reg_weight', 'min_valid_frames',
    'pointwise_weight', 'pointwise_balance', 'pointwise_label_smoothing',
    'inactive_consistency_weight', 'anchor_consistency_weight', 'selection_metric',
    'loss_schema', 'corruption_mask_semantics', 'source_frame_duplicate_policy',
    'two_d_sampling', 'collapse_patience', 'same_label_pull_weight',
}
FRAME_SELECTION_METRICS = {
    'triplet_acc', 'active_triplet_acc', 'score_gap', 'active_score_gap',
    'pointwise_auroc', 'pointwise_ap', 'localization', 'active_fraction',
}
SEQUENCE_SELECTION_METRICS = {'triplet_acc', 'score_gap'}


def resolve_selection_metric(config: dict, output_granularity: str) -> str:
    """Resolve a validation metric that the selected mode actually emits."""
    configured = str(config.get('scorer', {}).get(
        'selection_metric',
        'active_triplet_acc' if output_granularity == 'frame' else 'triplet_acc')).removeprefix('val/')
    if output_granularity == 'sequence':
        aliases = {'active_triplet_acc': 'triplet_acc', 'active_score_gap': 'score_gap'}
        configured = aliases.get(configured, configured)
        if configured not in SEQUENCE_SELECTION_METRICS:
            raise SystemExit(
                f'[scorer].selection_metric = {configured!r} is not emitted in sequence mode; '
                f'choose one of {sorted(SEQUENCE_SELECTION_METRICS)}')
    return configured


_ALIAS = 'a duplicate spelling no config or run folder ever used; set '


DELETED_CORRUPTION_KEYS = {
    'sequence_far_gate': (False, 'no run folder ever enabled it; frame mode is the gated scorer, '
                                 'and sequence mode keeps its ungated legacy loss'),
    'reference_margin_px': (0.0, 'it only parameterised the deleted reference_gate = '
                                 '"independent_far"'),
    'n_segments': (1, _ALIAS + 'segment_count'),
    'segment_length_frames': ([1, 1], _ALIAS + 'segment_len_frames'),
    'segment_full_window_share': (0.0, _ALIAS + 'full_window_share'),
    'segment_full_window_prob': (0.0, _ALIAS + 'full_window_share'),
}


def validate_scorer_config(config: dict, *, legacy_run: bool = False) -> str:
    """Validate scorer-only settings and return the resolved output mode.

    The validation is intentionally performed before a dataset/model is built.  Frame-only terms
    in sequence mode are a semantic mismatch, not harmless unused knobs; likewise a sequence
    keypoint gate cannot be mixed into the framewise active mask.  ``legacy_run`` is useful when
    validating a pre-frame run folder whose absent mode has already been resolved to sequence.
    """
    scorer = dict(config.get('scorer', {}))
    mode = scorer_output_granularity(config)
    if legacy_run and 'output_granularity' not in scorer:
        mode = 'sequence'
    unknown = set(scorer) - SCORER_CONFIG_KEYS - {'corruption'}
    if unknown:
        raise SystemExit(
            f'[scorer]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at defaults and report as the arm it is not. Known keys: '
            f'{sorted(SCORER_CONFIG_KEYS | {"corruption"})}')
    if mode not in SCORER_OUTPUT_GRANULARITIES:
        raise SystemExit(f'[scorer].output_granularity must be one of '
                         f'{SCORER_OUTPUT_GRANULARITIES}, got {mode!r}')
    corr = dict(scorer.get('corruption', {}))
    for key, (inert, why) in DELETED_CORRUPTION_KEYS.items():
        if key in corr and corr.pop(key) != inert:
            raise SystemExit(f'[scorer.corruption].{key} is deleted: {why}; only its inert '
                             f'value {inert!r} is still accepted')
    if 'output_granularity' in corr and str(corr['output_granularity']).lower() != mode:
        raise SystemExit(
            '[scorer.corruption].output_granularity must match '
            f'[scorer].output_granularity={mode!r}')
    if ('source_frame_duplicate_policy' in corr
            and corr['source_frame_duplicate_policy'] != scorer.get(
                'source_frame_duplicate_policy', 'inverse_multiplicity')):
        raise SystemExit(
            '[scorer.corruption].source_frame_duplicate_policy must match the top-level '
            '[scorer] value; set only one policy')
    pointwise = float(scorer.get('pointwise_weight', 0.0))
    frame_only = ('pointwise_weight', 'pointwise_balance', 'pointwise_label_smoothing',
                  'inactive_consistency_weight', 'anchor_consistency_weight')
    if mode == 'sequence':
        active = {k: scorer.get(k) for k in frame_only
                  if k in scorer and k not in ('pointwise_balance', 'pointwise_label_smoothing')}
        bad = {k: v for k, v in active.items() if float(v or 0.0) != 0.0}
        if bad:
            raise SystemExit(
                f'[scorer]: frame-only setting(s) {bad} require '
                'output_granularity = "frame"; sequence mode keeps the legacy loss.')
        if float(scorer.get('same_label_pull_weight', 1.0)) != 1.0:
            raise SystemExit(
                '[scorer].same_label_pull_weight applies to the frame loss only; the installed '
                'sequence loss always pulls at 1.0, so a sequence config must set it to 1.0')
    else:
        if pointwise > 0 and float(scorer.get('score_reg_weight', 0.0)) != 0.0:
            raise SystemExit(
                '[scorer]: score_reg_weight must be 0 when pointwise_weight > 0; the signed '
                'pointwise level anchor and score regularizer would fight each other')
    selection = str(scorer.get('selection_metric',
                                 'active_triplet_acc' if mode == 'frame' else 'triplet_acc'))
    if selection.startswith('val/'):
        selection = selection[4:]
    if selection not in FRAME_SELECTION_METRICS:
        raise SystemExit(
            f'[scorer].selection_metric = {selection!r} is unknown. Choose one of '
            f'{sorted(FRAME_SELECTION_METRICS)}')
    known_corr = {
        'const_offset_prob', 'frame_noise_prob', 'gradual_drift_prob', 'sinusoid_prob',
        'point_drop_prob', 'point_drop_max_frac', 'point_drop_bernoulli_rate',
        'mag_3d', 'mag_2d', 'min_corrupt_px', 'max_clean_px', 'reference_gate',
        'segment_prob', 'segment_count', 'segment_len_frames', 'full_window_share',
        'segment_types', 'no_active_slot_policy',
        'out_of_view_policy', 'output_granularity', 'source_frame_duplicate_policy',
    }
    if 'min_valid_frames' in corr:
        raise SystemExit(
            '[scorer.corruption].min_valid_frames is not read; set [scorer].min_valid_frames, '
            'which the corruption path copies down')
    unknown_corr = set(corr) - known_corr
    if unknown_corr:
        raise SystemExit(
            f'[scorer.corruption]: unknown key(s) {sorted(unknown_corr)}. Known keys: '
            f'{sorted(known_corr)}')
    segment_prob = float(corr.get('segment_prob', 0.0))
    if not 0.0 <= segment_prob <= 1.0:
        raise SystemExit('[scorer.corruption].segment_prob must be in [0, 1]')
    full_share = float(corr.get('full_window_share', 0.0))
    if not 0.0 <= full_share <= 1.0:
        raise SystemExit('[scorer.corruption] segment full-window share must be in [0, 1]')
    count_value = corr.get('segment_count', 1)
    count_values = list(count_value) if isinstance(count_value, (list, tuple)) else [count_value]
    if len(count_values) not in (1, 2):
        raise SystemExit('[scorer.corruption] segment count must be an integer or [lo, hi]')
    count_values = [int(value) for value in count_values]
    if count_values[0] < 1 or (len(count_values) == 2 and count_values[1] < count_values[0]):
        raise SystemExit('[scorer.corruption] segment count must satisfy 1 <= lo <= hi')
    length_value = corr.get('segment_len_frames', [1, 1])
    length_values = list(length_value) if isinstance(length_value, (list, tuple)) else [length_value]
    if len(length_values) not in (1, 2):
        raise SystemExit('[scorer.corruption] segment length must be an integer or [lo, hi]')
    length_values = [int(value) for value in length_values]
    if length_values[0] < 1 or (len(length_values) == 2 and length_values[1] < length_values[0]):
        raise SystemExit('[scorer.corruption] segment length must satisfy 1 <= lo <= hi')
    requested_types = corr.get('segment_types', SEGMENT_TYPES)
    if requested_types is None:
        raise SystemExit('[scorer.corruption].segment_types must be a sequence of names')
    if isinstance(requested_types, str):
        requested_types = (requested_types,)
    unknown_types = set(requested_types) - set(SEGMENT_TYPES)
    if unknown_types:
        raise SystemExit(
            f'[scorer.corruption].segment_types cannot be gated: {sorted(unknown_types)}')
    min_corrupt = float(corr.get('min_corrupt_px', 0.0))
    max_clean = float(corr.get('max_clean_px', 0.0))
    if (min_corrupt < 0 or max_clean < 0
            or (min_corrupt > 0 and min_corrupt <= max_clean)
            or (min_corrupt == 0 and max_clean > 0)):
        raise SystemExit(
            '[scorer.corruption] requires 0 <= max_clean_px < min_corrupt_px when a far '
            f'threshold is configured, or both zero for moved-mask fallback; '
            f'got min={min_corrupt}, max={max_clean}')
    duplicate_policy = str(scorer.get('source_frame_duplicate_policy', 'inverse_multiplicity'))
    if duplicate_policy != 'inverse_multiplicity':
        raise SystemExit(
            '[scorer].source_frame_duplicate_policy must be "inverse_multiplicity"; no other '
            'frame weighting policy is implemented')
    reference_gate = str(corr.get('reference_gate', 'source_far'))
    out_of_view = str(corr.get('out_of_view_policy', 'exclude'))
    if out_of_view != 'exclude':
        raise SystemExit(
            '[scorer.corruption].out_of_view_policy must be "exclude"; visibility-preserving '
            'resampling is a separate unimplemented arm')
    no_active = str(corr.get('no_active_slot_policy', 'retry'))
    if no_active != 'retry':
        raise SystemExit(
            '[scorer.corruption].no_active_slot_policy must be "retry"; the dataset retry '
            'chain is the only implemented policy')
    if reference_gate == 'independent_far':
        raise SystemExit(
            '[scorer.corruption].reference_gate = "independent_far" is deleted: no scorer '
            'dataset ever had the independent reference coordinates it needs, so every config '
            'refused it. Use "source_far" (or "none").')
    if reference_gate not in {'none', 'source_far'}:
        raise SystemExit(
            f'[scorer.corruption].reference_gate = {reference_gate!r} is not one of '
            '"none" | "source_far"')
    return mode


def scorer_kwargs(config: dict) -> dict:
    """The scorer's own head settings out of `[scorer]`.

    Inputs: config -- the merged run config.
    Outputs: a dict with exactly `SCORER_HEAD_KEYS`.
    Side effects: none.
    """
    cfg = dict(config.get('scorer', {}))
    cfg.pop('corruption', None)
    cfg.setdefault('output_granularity', scorer_output_granularity(config))
    return {k: cfg[k] for k in SCORER_HEAD_KEYS if k in cfg}


def corruption_config(config: dict) -> dict:
    """The `[scorer.corruption]` block, with `min_valid_frames` folded in from `[scorer]`.

    `min_valid_frames` lives on `[scorer]` and is consumed by the corruption path, so it is copied
    down rather than looked up twice.

    Inputs: config -- the merged run config.
    Outputs: the corruption dict.
    Side effects: none.
    """
    cfg = dict(config['scorer'].get('corruption', {}))
    cfg['min_valid_frames'] = int(config['scorer'].get('min_valid_frames', 6))
    cfg.setdefault('output_granularity', scorer_output_granularity(config))
    cfg.setdefault('source_frame_duplicate_policy', config['scorer'].get(
        'source_frame_duplicate_policy', 'inverse_multiplicity'))
    return cfg
