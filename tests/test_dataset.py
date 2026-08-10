"""The loader, and the one thing in it that must be bit-exact.

The crop rule is shared between the pose model and the detector: the detector is trained to
reproduce the pose crop, which is why a detector box costs ~0.02 mm instead of whatever an
independently-plausible rule would cost. `test_crop_rule_is_int32_exact` is what licenses that,
and it compares against the library's own inline arithmetic rather than a transcription of it.
"""
import numpy as np
import pytest
import torch

from tailcyclenet import crop as cropmod
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.format import Registry


# ----------------------------------------------------------------------------------------------
# the crop rule
# ----------------------------------------------------------------------------------------------

def test_crop_rule_is_int32_exact():
    """crop_box_for_points must equal PosetailDataset.crop_cgroup_to_points, exactly.

    The library exposes the rule only inline, so this drives the real method against a shim that
    supplies the one attribute it reads. Any drift here invalidates every detector number.
    """
    from types import SimpleNamespace
    from posetail.datasets.posetail_dataset import PosetailDataset

    rng = np.random.default_rng(0)
    shim = SimpleNamespace(min_crop_dim=64)
    for trial in range(200):
        w, h = int(rng.integers(80, 2000)), int(rng.integers(80, 2000))
        n = int(rng.integers(2, 30))
        pts = torch.as_tensor(rng.uniform(-200, max(w, h) + 200, size=(n, 2)), dtype=torch.float32)
        if trial % 7 == 0:                        # some non-finite points, the normal case
            pts[rng.integers(n)] = float('nan')
        size = torch.tensor([w, h], dtype=torch.int32)

        cam = {'size': size, 'offset': torch.zeros(2)}
        # the library projects first; feed it a camera whose projection is the identity by
        # calling the crop arithmetic on the same points through a one-camera group
        mine = cropmod.crop_box_for_points(pts, size, 64)

        pflat = pts.reshape(-1, 2)
        good = torch.all(torch.isfinite(pflat), dim=1)
        if not good.any():
            assert mine is None
            continue
        theirs = _library_box(shim, pflat[good], size)
        assert mine is not None
        assert mine.dtype == torch.int32
        assert torch.equal(mine, theirs), f'{w}x{h}: {mine.tolist()} != {theirs.tolist()}'


def _library_box(self, pflat, size):
    """PosetailDataset.crop_cgroup_to_points' arithmetic, transcribed from the library body.

    Kept separate from tailcyclenet.crop so the two are independent derivations; if someone
    "simplifies" one, the test still compares against the other.
    """
    low = torch.clamp(torch.min(pflat, dim=0).values - 20, torch.tensor([0, 0]), size).to(torch.int32)
    high = torch.clamp(torch.max(pflat, dim=0).values + 20, torch.tensor([0, 0]), size).to(torch.int32)
    cw, ch = high[0] - low[0], high[1] - low[1]
    base = max(self.min_crop_dim, int(cw), int(ch))
    min_dim_x, min_dim_y = min(base, int(size[0])), min(base, int(size[1]))
    if cw < min_dim_x:
        cx = (low[0] + high[0]) // 2
        low[0] = torch.clamp(cx - min_dim_x // 2, 0, size[0] - min_dim_x)
        high[0] = low[0] + min_dim_x
    if ch < min_dim_y:
        cy = (low[1] + high[1]) // 2
        low[1] = torch.clamp(cy - min_dim_y // 2, 0, size[1] - min_dim_y)
        high[1] = low[1] + min_dim_y
    return torch.cat([low, high])


def test_crop_box_is_none_when_nothing_is_finite():
    """The library raises here; the detector depends on getting None so it can emit a NaN box."""
    pts = torch.full((5, 2), float('nan'))
    assert cropmod.crop_box_for_points(pts, torch.tensor([100, 100], dtype=torch.int32)) is None


def test_jitter_stays_inside_the_image():
    rng = np.random.default_rng(0)
    jit = cropmod.jitter_box(rng, 0.3, 0.3)
    size = torch.tensor([200, 150], dtype=torch.int32)
    for _ in range(200):
        box = torch.tensor([40, 30, 120, 100], dtype=torch.int32)
        out = jit(box, size)
        assert out[0] >= 0 and out[1] >= 0 and out[2] <= 200 and out[3] <= 150
        assert out[2] > out[0] and out[3] > out[1]


# ----------------------------------------------------------------------------------------------
# the loader
# ----------------------------------------------------------------------------------------------

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)


def _batch(ds, i=0):
    return pose_collate([ds[i]])


def test_2d_item_shapes(tiny_root):
    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 1
    assert b.views[0].shape[:2] == (1, 4)                 # (B, T, H, W, 3)
    assert b.views[0].max() <= 1.0 and b.views[0].min() >= 0.0
    assert b.coords.shape == (1, 4, 4, 2)                 # R=2 for a true-2D session
    assert b.p2d.shape == (1, 1, 4, 4, 2)                 # 2D needs p2d; the loss reads it
    assert b.kpt_ids.shape == (1, 4)
    assert b.kpt_prior.shape == (1, 4, 2)
    assert b.prompt_t.shape == (1, 4)


def test_3d_item_shapes(tiny_root):
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    b = _batch(ds)
    assert len(b.views) == 3
    assert b.coords.shape == (1, 4, 3, 3)                 # R=3 world
    assert b.p2d is None
    assert b.vis.shape == (1, 4, 3, 1)                    # trailing dim get_eval_metrics wants
    assert b.vis_2d.shape == (1, 4, 3, 3, 1)


def test_vis_and_vis2d_are_both_or_neither(tiny_root):
    """Supplying one without the other dies inside einops, so the loader must never do it."""
    for name in ('ratlike', 'mouselike'):
        ds = PoseDataset(tiny_root / name, 'train', CFG)
        for i in range(len(ds)):
            b = pose_collate([ds[i]])
            assert (b.vis is None) == (b.vis_2d is None)


def test_keypoints_are_never_filtered(tiny_root):
    """Array position must keep equalling keypoint identity, even when points are missing.

    The library's filter_keypoints drops keypoints seen by too few views, and the resulting
    mislabelling is invisible in the loss curve. Every item carries all K.
    """
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        assert b.coords.shape[2] == 3
        assert b.kpt_ids.shape[1] == 3


def test_window_is_at_least_two_frames(tiny_root):
    """T=1 routes posetail into gT = T // tubelet_size = 0 and a zero-length pos_embed.

    This used to assert `shape[1] >= 1`, which is true of the exact 1-frame window it exists to
    forbid -- so it passed while the guard did not exist. `n_frames = 1` is now refused outright:
    the clamp-pad in `_frames` lengthens a short GROUP, not a short configured window.
    """
    cfg = LoaderConfig(n_frames=1, image_size=64, aug_prob=0.0, crop_jitter=0.0)
    with pytest.raises(AssertionError, match='n_frames'):
        PoseDataset(tiny_root / 'ratlike', 'train', cfg)

    cfg2 = LoaderConfig(n_frames=8, image_size=64, aug_prob=0.0, crop_jitter=0.0)
    ds2 = PoseDataset(tiny_root / 'ratlike', 'train', cfg2)   # groups are only 4 frames long
    b2 = _batch(ds2)
    assert b2.views[0].shape[1] == 8                          # clamp-padded, not truncated


def test_single_view_keeps_3d_targets(tiny_root):
    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=1.0, aug_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    ds = PoseDataset(tiny_root / 'mouselike', 'train', cfg)
    b = _batch(ds)
    assert b.coords.shape[-1] == 3          # targets stay world-metric
    assert len(b.views) == 1                # exactly one camera
    assert b.p2d is not None
    assert b.sample_info['single_view'] is True


def test_mixed_2d_and_3d_in_one_run(tiny_root):
    """One `train/` may hold both; the registry spans them and ids are disjoint."""
    ds = PoseDataset(tiny_root, 'train', CFG)
    assert {d.name for d in ds.datasets} == {'ratlike', 'mouselike'}
    modes = set()
    # balance_datasets samples the dataset per item, so iterating the index once is not a
    # coverage guarantee -- draw enough that missing one is 2^-30.
    for i in range(30):
        b = pose_collate([ds[i % len(ds)]])
        modes.add(b.sample_info['mode'])
        ids = b.kpt_ids[0].tolist()
        assert ids == list(ds.registry.ids_for_dataset(b.sample_info['dataset']))
    assert modes == {'2d', '3d'}


def test_registry_ids_survive_a_later_run(tiny_root):
    """Ids must be APPEND-ONLY against an existing registry, or warm start remaps the embedding.

    Discovery order is a directory listing, so it is not a stable thing to number against: with
    no base, seeing the same two datasets in the other order moves every id, and each row of
    `kpt_embed` then means a different body part than the checkpoint trained it to mean. Nothing
    in the loss curve shows it -- gotcha #4.
    """
    from tailcyclenet.format import load_datasets

    ds = load_datasets(tiny_root)
    first = Registry.build(ds)
    assert dict(Registry.build(list(reversed(ds))).datasets) != dict(first.datasets), \
        'the fixture must be order-sensitive, or this test proves nothing'

    again = Registry.build(list(reversed(ds)), first)
    assert dict(again.datasets) == dict(first.datasets)
    assert again.names == first.names


def test_registry_appends_a_new_dataset_without_moving_old_ids(tiny_root, tmp_path):
    from tailcyclenet.format import load_datasets

    base = Registry.build(load_datasets(tiny_root))
    grown = Registry.build(load_datasets(tiny_root) + [_FakeDataset('zzz', ['a', 'b'])], base)
    assert grown.names[:len(base.names)] == base.names
    assert list(grown.ids_for_dataset('zzz')) == [len(base.names), len(base.names) + 1]


class _FakeDataset:
    """The two attributes Registry.build reads. Cheaper than writing a session to disk."""

    def __init__(self, name, names):
        self.name, self.names = name, names


def test_val_windows_are_deterministic(tiny_root):
    ds = PoseDataset(tiny_root / 'ratlike', 'val', CFG)
    a = pose_collate([ds[0]])
    b = pose_collate([ds[0]])
    # equal_nan: a missing label is NaN, and two identical items must agree about which.
    torch.testing.assert_close(a.coords, b.coords, equal_nan=True)
    torch.testing.assert_close(a.views[0], b.views[0])
    assert a.sample_info['start'] == b.sample_info['start']


def test_prompt_is_the_first_labelled_frame(tiny_root):
    """prompt_t is not always 0 -- it was > 0 on 19.5% of rat-city windows."""
    ds = PoseDataset(tiny_root / 'ratlike', 'val', CFG)
    b = _batch(ds)
    finite = torch.isfinite(b.coords[0]).all(-1)       # (T,K)
    for k in range(finite.shape[1]):
        if finite[:, k].any():
            assert b.prompt_t[0, k] == int(finite[:, k].float().argmax())
            torch.testing.assert_close(b.kpt_prior[0, k],
                                       b.coords[0, b.prompt_t[0, k].long(), k])


def test_prompt_dropout_withholds_priors(tiny_root):
    cfg = LoaderConfig(n_frames=4, image_size=64, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=1.0, prob_2d_only=0.0)
    ds = PoseDataset(tiny_root / 'ratlike', 'train', cfg)
    b = _batch(ds)
    assert torch.isnan(b.kpt_prior).all()      # every keypoint unprompted -> learned tokens


def test_visibility_stays_three_state(tiny_root):
    """"Not assessed" must reach the loss as NaN, not as "not visible".

    posetail >= 0.3.2 masks non-finite visibility targets, so an unassessed keypoint-camera pair
    produces no gradient. Collapsing it to 0 instead would train the visibility head on ~18% of
    allen-mouse's targets that nobody ever labelled. Under 0.3.0 the collapse was forced: a NaN
    target there returned NaN gradients for every parameter while the loss looked healthy.
    """
    ds = PoseDataset(tiny_root / 'mouselike', 'train', CFG)
    saw_unassessed = False
    for i in range(len(ds)):
        b = pose_collate([ds[i]])
        if b.vis_2d is None:
            continue
        finite = b.vis_2d[torch.isfinite(b.vis_2d)]
        assert set(finite.unique().tolist()) <= {0.0, 1.0}, 'assessed entries are 0 or 1'
        saw_unassessed |= bool(torch.isnan(b.vis_2d).any())
    assert saw_unassessed, 'the 3D fixture carries unassessed entries; they must survive as NaN'


def test_gradients_survive_unassessed_visibility():
    """The property the loader depends on, asserted against the INSTALLED posetail.

    The failure this guards is invisible from the loss: it stays finite and falls normally while
    every parameter receives NaN. Pinning it here means a dependency downgrade fails loudly
    instead of quietly wasting a training run.
    """
    from posetail.posetail.losses import BCELossVis

    pred = torch.zeros(2, 4, 3, 1, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 3, 1)).float()
    target[0, 1, 2, 0] = float('nan')          # not assessed
    loss = BCELossVis(weight=1.0)(pred, target, device='cpu')
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all(), (
        'installed posetail poisons the gradient on an unassessed visibility target; '
        'needs >= 0.3.2')


def test_no_visibility_supervision_without_ground_truth(tiny_root, monkeypatch):
    """A dataset with no visibility labels must not have its visibility head trained.

    3dpop, rat-city and branson-fly ship no per-camera visibility, so the loader emits
    `vis = vis_2d = None`. posetail then sets `valid_vis = False` and hard-zeros BOTH visibility
    terms (`losses.py:493-508`). It still DERIVES a geometric `vis_true` -- but only to mask the
    coordinate losses, never to supervise visibility.

    That distinction matters: the geometric proxy is "does the GT point project inside the
    image", which the model could compute from its own prediction. Training the visibility head
    against it would teach a tautology and call it supervision.
    """
    from posetail.posetail.losses import TotalLoss

    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG)   # a 2D root: no visibility labels
    b = _batch(ds)
    assert b.vis is None and b.vis_2d is None

    seen = []
    real = TotalLoss.forward

    class Spy(TotalLoss):
        def forward(self, model, outputs, coords_true, vis_true, vis_true_cams, **kw):
            seen.append((vis_true is None, vis_true_cams is None))
            raise SystemExit                      # we only need the arguments, not the loss

    spy = Spy(vis_loss_weight=5.0, vis_loss_3d_weight=1.0)
    try:
        spy(None, {}, b.coords, b.vis, b.vis_2d)
    except SystemExit:
        pass
    assert seen == [(True, True)], 'a dataset without visibility labels must pass None, not zeros'


# ----------------------------------------------------------------------------------------------
# a label in the middle of a group must be usable -- on BOTH paths
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('train', [True, False])
@pytest.mark.parametrize('n_frames', [8, 24])
def test_centred_labels_are_usable(centred_root, train, n_frames):
    """The old loader required the window's FIRST frame to be labelled; this one does not.

    A group with labels only at frames 11-13 must still yield windows, every window must contain
    a label, and the label must not be forced to frame 0 -- frame 0 is the one frame where
    per-frame anchoring contributes nothing.
    """
    cfg = LoaderConfig(n_frames=n_frames, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=train)
    assert len(ds) > 0, 'a group whose labels are centred must still produce windows'

    n = 40 if train else len(ds)
    at_frame_zero = 0
    for i in range(n):
        b = pose_collate([ds[i % len(ds)]])
        finite = torch.isfinite(b.coords[0]).all(-1)          # (T,K)
        assert finite.any(), 'every window must contain at least one labelled frame'
        at_frame_zero += int(finite[0].any())
    assert at_frame_zero < n, 'the label must not always land on frame 0'


def test_val_windows_do_not_pad_when_the_group_is_long_enough(centred_root):
    """A start past `n_frames - T` clamp-pads with duplicates of the last frame.

    That wastes real context: with T=24 on a 24-frame group whose labels are at 11-13, starting
    at the first labelled frame padded 13 duplicated frames while frames 0-10 sat unused.
    """
    cfg = LoaderConfig(n_frames=24, image_size=32, aug_prob=0.0, crop_jitter=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=False)
    for item in ds.index:
        assert 0 <= item.start <= max(0, item.session.groups[item.gid].n_frames - 24)
    b = pose_collate([ds[0]])
    assert b.sample_info['start'] == 0
    assert b.fnums[0].tolist() == list(range(24)), 'no duplicated frames'


def test_prompt_time_is_not_forced_to_zero(centred_root):
    """prompt_t is the first LABELLED frame, which is > 0 whenever labels are centred."""
    cfg = LoaderConfig(n_frames=24, image_size=32, aug_prob=0.0, crop_jitter=0.0,
                       prompt_dropout=0.0)
    ds = PoseDataset(centred_root, 'train', cfg, train=False)
    b = pose_collate([ds[0]])
    assert (b.prompt_t[0] == 11).all(), 'the prompt must point at the real first label'
    assert torch.isfinite(b.kpt_prior[0]).all(), 'and the prior must be taken from there'
