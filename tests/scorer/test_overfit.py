"""Phase 3's exit criterion: the scorer can OVERFIT a handful of windows, on the shipped frame path.

`tests/scorer/test_triplet.py` establishes that the triplet is well-formed. This establishes that
the whole path -- dataset, triplet, model, loss, backward, step -- actually learns, which no unit
test of the parts can show. If a tiny model on a couple of windows cannot separate good from bad,
nothing downstream (Gate B, the calibration sweep, the real runs) can mean anything.

The recipe is `configs/scorer.toml`'s (`output_granularity = "frame"`, its corruption block and
loss weights), shrunk only where the 4-frame fixture or the tiny model require it. The metric is
the frame loss's own `active_triplet_acc` = P(score(good) > score(bad)) over the active rows, the
quantity checkpoint selection uses -- so a failure here is a failure of the objective, not of a
proxy for it.

The windows are built on the FROZEN (val) stream: a train-mode `PoseDataset`/`ScorerDataset`
entropy-seeds every item by design, which made two "identical" runs train on different windows.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from tailcyclenet import config as cfgmod
from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.scorer.config import corruption_config
from tailcyclenet.scorer.dataset import ScorerDataset, scorer_collate, triplet_to_device
from tailcyclenet.scorer.evaluate import scorer_loss
from tailcyclenet.scorer.losses import build_scorer_loss
from tests.helpers import REPO, SMALL, _session_2d, _session_3d, _session_3d_multi

pytestmark = pytest.mark.slow

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)


def _shipped():
    """`configs/scorer.toml` with the fixture-sized overrides: 4-frame windows hold at most 4
    distinct observed frames, so `min_valid_frames` (6) would put every window on the sparse menu."""
    config = cfgmod.load_config(str(REPO / 'configs/scorer.toml'), base=cfgmod.SCORER_CONFIG)
    config['scorer'] = {**config['scorer'], 'min_valid_frames': 2}
    return config


@pytest.fixture(scope='module')
def overfit_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('overfit')
    _session_3d(root / 'mouselike' / 'train' / 'sess_c')
    _session_3d_multi(root / 'mouselike' / 'train' / 'sess_m', T=4)
    _session_2d(root / 'ratlike' / 'train' / 'sess_a')
    _session_2d(root / 'ratlike' / 'train' / 'sess_b')
    return root


def _windows(root, session, n):
    """`n` built triplets from the frozen stream, skipping any that failed to build."""
    ds = ScorerDataset(PoseDataset(root / session, 'train', CFG, train=False),
                       corruption_config(_shipped()))
    out = []
    for trip in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=scorer_collate,
                           num_workers=0):
        if trip is not None:
            out.append(trip)
        if len(out) >= n:
            break
    assert len(out) == n, (
        f'only {len(out)} of {n} windows built -- the fixture must hold enough windows for the '
        'overfit to be meaningful')
    return out


def _model(windows, seed):
    from tailcyclenet.scorer.model import build_scorer

    torch.manual_seed(seed)
    n_kpt = max(int(w['kpt_ids'].max()) + 1 for w in windows)
    model = build_scorer({**SMALL, 'box_prompt': 'film', 'video_encoder_pretrained': False}, n_kpt,
                         pool_num_heads=4, score_hidden=32, use_precision=True,
                         output_granularity='frame')
    loss_fn = build_scorer_loss(_shipped(), 'frame')
    model.add_module('frame_loss', loss_fn)
    return model, loss_fn


def _overfit(root, iterations=120, n_windows=2, lr=1e-3, seed=0, session='mouselike'):
    """Train on a handful of frozen windows and report the final active triplet accuracy.

    Inputs: root -- the dataset root; iterations -- passes over the windows; n_windows -- how many
            distinct windows to memorise; lr -- AdamW learning rate; seed -- the model-init seed;
            session -- `mouselike` (3D multiview) or `ratlike` (2D single-view).
    Outputs: (final `active_triplet_acc`, its `active_score_gap`).
    Side effects: builds a model and Dataset, decodes video.
    """
    windows = _windows(root, session, n_windows)
    model, loss_fn = _model(windows, seed)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    acc = gap = float('nan')
    for _ in range(iterations):
        for trip in windows:
            trip = triplet_to_device(trip, torch.device('cpu'))
            opt.zero_grad(set_to_none=True)
            scores, precision, labels = model.score_triplet(trip)
            total = scorer_loss(loss_fn, scores, precision, labels, trip, 'frame')
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            opt.step()
        hist = loss_fn.collapse_history()
        acc, gap = hist['active_triplet_acc'], hist['active_score_gap']
        loss_fn.reset_history()
    return acc, gap


@pytest.mark.parametrize('session,mode', [('mouselike', '3d'), ('ratlike', '2d')])
def test_the_scorer_overfits_a_handful_of_windows(overfit_root, session, mode):
    """A tiny model must drive `active_triplet_acc` to 1.0 on a couple of windows.

    This is the phase-3 (3D) and phase-4 (2D) exit gate. 120 AdamW passes at lr 1e-3 over two
    frozen windows is far more capacity than the task needs; failing to reach 1.0 means the path
    is broken, not under-trained. The 2D arm is the same objective over `transfer_points_2d`, the
    anchor's coordinate carry, which is the part with no 3D analogue.
    """
    acc, gap = _overfit(overfit_root, session=session, n_windows=2)
    assert acc == pytest.approx(1.0), (
        f'{mode}: the scorer did not overfit {session}: active_triplet_acc={acc:.3f}, '
        f'score_gap={gap:.4g}. A perfect score on two windows is the floor, so this is a broken '
        'path rather than an under-trained model.')


def test_the_overfit_curves_are_reproducible(overfit_root):
    """Same seed, same windows, same result -- so a failure means the CODE changed.

    Compared on the score gap as well as the accuracy: two runs that agree on a 0/1-ish accuracy
    can still have trained differently.
    """
    a = _overfit(overfit_root, iterations=30)
    b = _overfit(overfit_root, iterations=30)
    assert a == b, f'two identical overfit runs disagreed: {a} vs {b}'


def test_an_untrained_scorer_is_near_chance(overfit_root):
    """The control the overfit test needs: at init, the accuracy must NOT already be 1.0.

    A freshly built model's scores are essentially arbitrary, so a good/bad separation of 1.0
    before any training would mean the comparison is being made somewhere that does not depend on
    the weights -- and the overfit test above would then prove nothing.
    """
    windows = _windows(overfit_root, 'mouselike', 3)
    model, loss_fn = _model(windows, seed=0)
    model.eval()
    with torch.no_grad():
        for trip in windows:
            trip = triplet_to_device(trip, torch.device('cpu'))
            scores, precision, labels = model.score_triplet(trip)
            scorer_loss(loss_fn, scores, precision, labels, trip, 'frame')
    acc = loss_fn.collapse_history()['active_triplet_acc']
    assert acc < 1.0, (
        f'an untrained scorer already separates good from bad perfectly ({acc:.3f}); the overfit '
        'test would then measure nothing')
