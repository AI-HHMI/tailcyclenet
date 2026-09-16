"""Configuration/checkpoint compatibility contracts for the scorer granularity switch."""

import pytest
import torch

from tailcyclenet.checkpoints import (
    _SCORER_CONFIG,
    load_config,
    require_scorer_granularity,
    save_checkpoint,
    scorer_checkpoint_granularity,
    scorer_contract,
    scorer_contract_mismatches,
    scorer_output_granularity,
)
from tailcyclenet.scorer.losses import FrameTripletScorerLoss
from tailcyclenet.scorer.train import validate_scorer_config


def test_new_family_config_defaults_to_frame_but_absent_run_key_is_sequence():
    config = load_config('configs/scorer.toml', base=_SCORER_CONFIG)
    assert scorer_output_granularity(config) == 'frame'
    assert scorer_output_granularity({'scorer': {}}) == 'sequence'
    assert validate_scorer_config({'scorer': {}, 'model': {}, 'data': {}}) == 'sequence'


def test_old_and_new_checkpoint_modes_are_refused_crosswise():
    old = {'kind': 'scorer', 'config': {'scorer': {}}}
    new = {'kind': 'scorer', 'output_granularity': 'frame'}
    assert scorer_checkpoint_granularity(old) == 'sequence'
    assert scorer_checkpoint_granularity(new) == 'frame'
    require_scorer_granularity('sequence', 'sequence')
    require_scorer_granularity('frame', 'frame')
    with pytest.raises(ValueError, match='output_granularity mismatch'):
        require_scorer_granularity('frame', 'sequence')
    with pytest.raises(ValueError, match='output_granularity mismatch'):
        require_scorer_granularity('sequence', 'frame')


def test_conflicting_checkpoint_mode_metadata_is_refused():
    with pytest.raises(ValueError, match='conflicting output_granularity'):
        scorer_checkpoint_granularity({
            'output_granularity': 'frame',
            'scorer_output_granularity': 'sequence',
        })


def test_sequence_contract_normalizes_inherited_frame_defaults_and_ranges():
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


def test_scorer_window_length_must_match_model_stride():
    config = {'scorer': {}, 'data': {'n_frames': 12}, 'model': {'stride_length': 24}}
    with pytest.raises(SystemExit, match='n_frames.*stride_length'):
        validate_scorer_config(config)


def test_zero_min_corrupt_threshold_cannot_have_positive_clean_band():
    config = {'scorer': {'output_granularity': 'frame',
                         'corruption': {'min_corrupt_px': 0.0, 'max_clean_px': 0.5}}}
    with pytest.raises(SystemExit, match='both zero'):
        validate_scorer_config(config)


def test_frame_contract_preserves_ranged_segment_count():
    config = {'scorer': {'output_granularity': 'frame',
                         'corruption': {'n_segments': [1, 2]}}}
    assert scorer_contract(config)['segment_count'] == [1, 2]


def test_frame_checkpoint_stamps_and_saves_the_loss_scale(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.add_module('frame_loss', FrameTripletScorerLoss(pointwise_weight=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {'scorer': {'output_granularity': 'frame', 'loss_schema': 'framewise-v1'}}
    save_checkpoint(tmp_path, 3, model, optimizer, config, kind='scorer')
    checkpoint = torch.load(tmp_path / 'checkpoints' / 'checkpoint_last.pth',
                            map_location='cpu', weights_only=False)
    assert checkpoint['output_granularity'] == 'frame'
    assert checkpoint['scorer_metadata']['loss_schema'] == 'framewise-v1'
    assert 'frame_loss.pointwise_log_scale' in checkpoint['model_state']
