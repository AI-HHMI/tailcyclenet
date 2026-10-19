"""Model loading at inference: pose/scorer encoder sharing and the detector guard."""
from __future__ import annotations

import numpy as np
import pytest

from tailcyclenet.infer.loading import detector_box_source, detector_matches_source
from tailcyclenet.infer.window import run_group
from tailcyclenet.pose.model import build_model
from tests.helpers import SMALL, _infer_cfg, _scorer


def test_scorer_encoder_mismatch_requires_explicit_opt_in():
    from tailcyclenet.infer.loading import share_pose_scorer_encoder

    pose = build_model(SMALL, n_keypoints=5).eval()
    scorer = _scorer(5, stride_length=4).eval()
    with pytest.raises(SystemExit, match='--allow-pose-scorer-mismatch'):
        share_pose_scorer_encoder(pose, scorer, allow_mismatch=False,
                                   pose_run='pose-run', scorer_run='scorer-run')
    assert not share_pose_scorer_encoder(pose, scorer, allow_mismatch=True,
                                          pose_run='pose-run', scorer_run='scorer-run')
    assert scorer.scene_encoder.encoder is not pose.scene_encoder.encoder


def test_integrated_scorer_shares_scene_and_preserves_pose(rat):
    model, sess, registry, name = rat
    from tailcyclenet.pose.model import share_video_encoder_if_identical

    scorer = _scorer(registry.n_keypoints, stride_length=4,
                     output_granularity='frame').eval()
    scorer.scene_encoder.encoder.load_state_dict(model.scene_encoder.encoder.state_dict())
    assert share_video_encoder_if_identical(model, scorer)
    cfg = _infer_cfg(anchor='none', refine=False)
    pose_only = run_group(model, sess, 'g000', registry, name, cfg)
    integrated = run_group(model, sess, 'g000', registry, name, cfg, scorer=scorer,
                           shared_scene_encoder=True)
    np.testing.assert_array_equal(pose_only['pred'], integrated['pred'])
    assert integrated['quality'].shape == pose_only['pred'].shape[:-1]
    assert integrated['quality_precision'].shape == integrated['quality'].shape
    assert np.isfinite(integrated['quality']).any()


def test_multi_root_detector_inference_guard_and_box_source_resolution():
    """Inference accepts a multi-root detector only on a root it trained on (by name), and takes
    that root's own box source, falling back to the session's."""
    assert detector_matches_source(['allen-mouse-combined', 'johnson-mouse-combined-aug'], '',
                                    'allen-mouse-combined')
    assert not detector_matches_source(['allen-mouse-combined', 'johnson-mouse-combined-aug'],
                                        '', 'allen-mouse-annotated')
    assert detector_matches_source(['allen-mouse-combined'], 'allen-mouse-combined',
                                    'allen-mouse-annotated')

    class Detector:
        box_sources = {'allen-mouse-combined': 'instances',
                       'johnson-mouse-combined-aug': 'keypoints'}

    detector = Detector()
    assert detector_box_source(detector, 'allen-mouse-combined', 'keypoints') == 'instances'
    assert detector_box_source(detector, 'johnson-mouse-annotated', 'instances') == 'keypoints'
    assert detector_box_source(detector, 'rat-city', 'keypoints') == 'keypoints'
