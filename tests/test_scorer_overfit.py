"""Phase 3's exit criterion: the scorer can OVERFIT a handful of windows.

`tests/test_scorer_triplet.py` establishes that the triplet is well-formed. This establishes that
the whole path -- dataset, triplet, model, loss, backward, step -- actually learns, which no unit
test of the parts can show. If a tiny model on a couple of windows cannot separate good from bad,
nothing downstream (Gate B, the calibration sweep, the real runs) can mean anything.

The metric is the loss's own `triplet_acc` = P(score(good) > score(bad)) over the overfit windows,
which is the same quantity Gate B headlines -- so a failure here is a failure of the objective, not
of a proxy for it.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from posetail.posetail.losses_scorer import TripletScorerLoss
from tailcyclenet.dataset import LoaderConfig, PoseDataset
from tailcyclenet.scorer.dataset import ScorerDataset, scorer_collate, triplet_to_device

from .test_scorer_model import SMALL

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)

CORRUPTION = {
    'const_offset_prob': 0.5, 'frame_noise_prob': 0.5,
    'gradual_drift_prob': 0.5, 'sinusoid_prob': 0.5,
    'point_drop_prob': 0.0, 'min_valid_frames': 2,
    'mag_3d': {'const_offset': 15.0, 'frame_noise': 10.0,
               'gradual_drift': 24.0, 'sinusoid': 16.0},
    'mag_2d': {'const_offset': 10.0, 'frame_noise': 8.0,
               'gradual_drift': 12.0, 'sinusoid': 10.0},
}


@pytest.fixture(scope='module')
def overfit_root(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path_factory.mktemp('overfit')
    C._session_3d(root / 'mouselike' / 'train' / 'sess_c')
    C._session_3d_multi(root / 'mouselike' / 'train' / 'sess_m', T=4)
    return root


def _windows(loader, n):
    """Collect `n` built triplets, skipping any that failed to build."""
    out = []
    for trip in loader:
        if trip is not None:
            out.append(trip)
        if len(out) >= n:
            break
    return out


def _overfit(root, iterations=120, n_windows=2, lr=1e-3, seed=0):
    """Train on a handful of frozen windows and report the final triplet accuracy.

    The windows are collected ONCE and reused, so this measures whether the objective is learnable
    rather than whether the loader keeps producing consistent samples.

    Inputs: root -- the dataset root; iterations -- optimisation steps; n_windows -- how many
            distinct windows to memorise; lr -- AdamW learning rate; seed -- torch seed.
    Outputs: (final `triplet_acc`, the loss history's last `score_gap`).
    Side effects: builds a model and Dataset, decodes video.
    """
    from tailcyclenet.scorer.model import build_scorer

    torch.manual_seed(seed)
    ds = ScorerDataset(PoseDataset(root / 'mouselike', 'train', CFG, train=True), CORRUPTION)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=scorer_collate,
                        num_workers=0)
    windows = _windows(loader, n_windows)
    assert len(windows) == n_windows, (
        f'only {len(windows)} of {n_windows} windows built -- the fixture must hold enough '
        'windows for the overfit to be meaningful')

    n_kpt = max(int(w['kpt_ids'].max()) + 1 for w in windows)
    cfg = {k: v for k, v in SMALL.items() if k != 'gridresid_offset'}
    model = build_scorer({**cfg, 'box_prompt': 'film', 'video_encoder_pretrained': False,
                          'gridresid_offset': 'query'}, n_kpt,
                         pool_num_heads=4, score_hidden=32, use_precision=True)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    loss_fn = TripletScorerLoss(margin=0.25, precision_reg_weight=0.15, score_reg_weight=0.0)

    acc = gap = float('nan')
    for _ in range(iterations):
        for trip in windows:
            trip = triplet_to_device(trip, torch.device('cpu'))
            opt.zero_grad(set_to_none=True)
            scores, precision, labels = model.score_triplet(trip)
            total = loss_fn(scores, precision, labels)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            opt.step()
            hist = loss_fn.collapse_history()
            acc = hist['triplet_acc']
            gap = hist['score_gap']
            loss_fn.reset_history()
    return acc, gap


def test_the_scorer_overfits_a_handful_of_windows(overfit_root):
    """A tiny model must drive `triplet_acc` to 1.0 on a couple of windows.

    This is the phase-3 exit gate. 120 AdamW steps at lr 1e-3 on two frozen windows is far more
    capacity than the task needs; failing to reach 1.0 means the path is broken, not under-trained.
    """
    acc, gap = _overfit(overfit_root)
    assert acc == pytest.approx(1.0), (
        f'the scorer did not overfit: triplet_acc={acc:.3f}, score_gap={gap:.4g}. '
        'A perfect score on two windows is the floor, so this is a broken path rather than an '
        'under-trained model.')


def test_the_overfit_curves_are_reproducible(overfit_root):
    """Same seed, same windows, same result -- so a failure means the CODE changed.

    Without this, the overfit test above could fail for a reason that is not the code, which is
    exactly the trap the Gate A.1 file fell into twice.
    """
    a, _ = _overfit(overfit_root, iterations=30)
    b, _ = _overfit(overfit_root, iterations=30)
    assert a == b, f'two identical overfit runs disagreed: {a} vs {b}'


def test_an_untrained_scorer_is_near_chance(overfit_root):
    """The control the overfit test needs: at init, `triplet_acc` must NOT already be 1.0.

    A freshly built model's scores are essentially arbitrary, so a good/bad separation of 1.0
    before any training would mean the comparison is being made somewhere that does not depend on
    the weights -- and the overfit test above would then prove nothing.
    """
    from tailcyclenet.scorer.model import build_scorer

    torch.manual_seed(0)
    ds = ScorerDataset(PoseDataset(overfit_root / 'mouselike', 'train', CFG, train=True),
                       CORRUPTION)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=scorer_collate, num_workers=0)
    windows = _windows(loader, 4)
    n_kpt = max(int(w['kpt_ids'].max()) + 1 for w in windows)
    cfg = {k: v for k, v in SMALL.items()}
    model = build_scorer({**cfg, 'box_prompt': 'film', 'video_encoder_pretrained': False}, n_kpt,
                         pool_num_heads=4, score_hidden=32, use_precision=True)
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for trip in windows:
            scores, _p, _l = model.score_triplet(triplet_to_device(trip, torch.device('cpu')))
            hits += int((scores[:, 0] > scores[:, 1]).sum())
            total += scores.shape[0]
    acc = hits / max(total, 1)
    assert acc < 1.0, (
        f'an untrained scorer already separates good from bad perfectly ({acc:.3f}); the overfit '
        'test would then measure nothing')
