"""Configuration/checkpoint compatibility contracts for the scorer granularity switch."""

import pytest
import torch

from tailcyclenet.config import SCORER_CONFIG, load_config
from tailcyclenet.scorer.checkpoint import save_scorer_checkpoint
from tailcyclenet.scorer.contract import (
    require_scorer_contract,
    require_scorer_granularity,
    scorer_checkpoint_granularity,
    scorer_contract,
    scorer_contract_mismatches,
    scorer_output_granularity,
)
from tailcyclenet.scorer.losses import FrameTripletScorerLoss
from tailcyclenet.scorer.config import validate_scorer_config


def test_new_family_config_defaults_to_frame_but_absent_run_key_is_sequence():
    config = load_config('configs/scorer.toml', base=SCORER_CONFIG)
    assert scorer_output_granularity(config) == 'frame'
    assert scorer_output_granularity({'scorer': {}}) == 'sequence'
    assert validate_scorer_config({'scorer': {}, 'model': {}, 'data': {}}) == 'sequence'


def test_checkpoint_granularity_is_read_and_refused_crosswise():
    assert scorer_checkpoint_granularity({'kind': 'scorer', 'config': {'scorer': {}}}) == 'sequence'
    assert scorer_checkpoint_granularity({'kind': 'scorer', 'output_granularity': 'frame'}) == 'frame'
    require_scorer_granularity('frame', 'frame')
    for want, got in (('frame', 'sequence'), ('sequence', 'frame')):
        with pytest.raises(ValueError, match='output_granularity mismatch'):
            require_scorer_granularity(want, got)
    with pytest.raises(ValueError, match='conflicting output_granularity'):
        scorer_checkpoint_granularity({'output_granularity': 'frame',
                                       'scorer_output_granularity': 'sequence'})


def test_contract_normalises_sequence_defaults_and_keeps_frame_ranges():
    config = {
        'scorer': {
            'output_granularity': 'sequence',
            'loss_schema': 'framewise-v1',
            'corruption_mask_semantics': 'far-observed-in-view-anchor-observed',
            'source_frame_duplicate_policy': 'inverse_multiplicity',
            'corruption': {'segment_count': [1, 2]},
        }
    }
    contract = scorer_contract(config)
    assert contract['loss_schema'] == 'sequence-v0'
    assert contract['corruption_mask_semantics'] == 'legacy'
    assert contract['source_frame_duplicate_policy'] == 'legacy'
    assert contract['segment_count'] == 1
    assert not scorer_contract_mismatches(config, {'scorer_metadata': {'contract': contract}})
    frame = {'scorer': {'output_granularity': 'frame', 'corruption': {'segment_count': [1, 2]}}}
    assert scorer_contract(frame)['segment_count'] == [1, 2]


def test_true_2d_sampling_is_in_the_contract_and_legacy_checkpoints_fall_back_to_config():
    config = {'scorer': {'output_granularity': 'frame', 'two_d_sampling': 'true-2d-hybrid-v1'},
              'data': {'prob_2d_only': 0.2}}
    old = {'config': {'scorer': {'output_granularity': 'frame'}, 'data': {'prob_2d_only': 0.2}}}
    assert 'config.two_d_sampling' in scorer_contract_mismatches(config, old)

    config['scorer']['two_d_sampling'] = 'legacy-3d-single-view'
    metadata_contract = scorer_contract(old['config'])
    metadata_contract.pop('two_d_sampling')
    metadata_contract.pop('prob_2d_only')
    require_scorer_contract(config, {**old, 'scorer_metadata': {'contract': metadata_contract}})


_DELETED = [('sequence_far_gate', False, True), ('reference_margin_px', 0.0, 2.0),
            ('n_segments', 1, [1, 2]), ('segment_length_frames', [1, 1], [2, 6]),
            ('segment_full_window_share', 0.0, 0.1), ('segment_full_window_prob', 0.0, 0.1)]


def _frame(**corruption):
    return {'scorer': {'output_granularity': 'frame', 'corruption': corruption}}


def test_validate_scorer_config_refusals():
    """Deleted keys at live values, the deleted gate, a shadowed `min_valid_frames`, and a
    positive clean band with a zero corrupt threshold all raise by name."""
    cases = [(_frame(**{k: live}), rf'\[scorer\.corruption\]\.{k} is deleted')
             for k, _inert, live in _DELETED]
    cases += [
        (_frame(reference_gate='independent_far'), 'independent_far" is deleted'),
        ({'scorer': {'output_granularity': 'frame', 'min_valid_frames': 6,
                     'corruption': {'min_valid_frames': 2}}}, r'\[scorer\]\.min_valid_frames'),
        (_frame(min_corrupt_px=0.0, max_clean_px=0.5), 'both zero'),
    ]
    for config, match in cases:
        with pytest.raises(SystemExit, match=match):
            validate_scorer_config(config)


def test_validate_scorer_config_accepts_inert_and_live_values():
    """Old run-folder configs carry deleted keys at inert values; those must still load."""
    for key, inert, _live in _DELETED:
        assert validate_scorer_config(_frame(**{key: inert})) == 'frame', key
    for gate in ('source_far', 'none'):
        assert validate_scorer_config(_frame(reference_gate=gate)) == 'frame'
    validate_scorer_config({'scorer': {'output_granularity': 'frame', 'min_valid_frames': 6,
                                       'corruption': {}}})
    # `n_frames` and `stride_length` may differ (owner decision 2026-10-19).
    assert validate_scorer_config({'scorer': {}, 'data': {'n_frames': 12},
                                   'model': {'stride_length': 24}}) == 'sequence'


def test_frame_checkpoint_stamps_and_saves_the_loss_scale(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.add_module('frame_loss', FrameTripletScorerLoss(pointwise_weight=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {'scorer': {'output_granularity': 'frame', 'loss_schema': 'framewise-v1'}}
    save_scorer_checkpoint(tmp_path, 3, model, optimizer, config)
    checkpoint = torch.load(tmp_path / 'checkpoints' / 'checkpoint_last.pth',
                            map_location='cpu', weights_only=False)
    assert checkpoint['output_granularity'] == 'frame'
    assert checkpoint['scorer_metadata']['loss_schema'] == 'framewise-v1'
    assert 'frame_loss.pointwise_log_scale' in checkpoint['model_state']
