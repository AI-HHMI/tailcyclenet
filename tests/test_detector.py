"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
import re
import shutil
import sys

import numpy as np
import pytest
import torch
from pathlib import Path

from tailcyclenet.crop import crop_box_for_points
from tailcyclenet.detector.dataset import BoxDataset
from tailcyclenet.detector.sampling import ChunkShuffle, CohortSampler, box_collate, split_batch
from tailcyclenet.detector.yolox import YOLOXNano
from tailcyclenet.detector.assign import assign, assign_tal
from tailcyclenet.detector.boxes import box_iou, paired_iou
from tailcyclenet.detector.loss import ciou_loss, detector_loss, giou_loss
from tailcyclenet.detector.decode import decode
from tailcyclenet.detector.letterbox import letterbox, unletterbox_boxes
from tailcyclenet.detector.config import TRAINING_KEYS, load_detector_config
from tailcyclenet.detector.augment import photometric, random_affine

sys.path.insert(0, str(Path(__file__).parent))
import conftest as cf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _box_only_session(path, group_ids=('video_ix0', 'video_ix10')):
    from tailcyclenet import format as fmt

    T = 2
    rig = cf._rig([('cam0', 64, 48, False, False, 0)])
    groups = {gid: fmt.Group(gid, T) for gid in group_ids}
    labels = {}
    for gid in group_ids:
        lab = fmt.empty_labels(2, T, len(cf.KPTS_2D), 1, mode3d=False,
                               animal_ids=['labeled', 'present'])
        lab.points2d = lab.vis2d = None
        lab.instance = np.full((2, T, 1), fmt.INST_NONE, np.int8)
        lab.boxes = np.full((2, T, 1, 4), np.nan, np.float32)
        lab.instance[0, :, 0] = fmt.INST_LABELED
        lab.boxes[0, :, 0] = [8, 7, 28, 29]
        lab.instance[1, :, 0] = fmt.INST_PRESENT
        lab.boxes[1, :, 0] = [35, 8, 55, 30]
        labels[gid] = lab
    fmt.write_session(path, mode='2d', units='px', label_source='annotated',
                      names=cf.KPTS_2D, rig=rig, groups=groups, labels=labels)
    for gid in group_ids:
        cf._write_frames(path / 'groups' / gid, 'cam0', T, (64, 48))


def test_forward_shapes_and_anchor_order():
    m = YOLOXNano(version='hybrid')
    x = torch.zeros(2, 3, 128, 160)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0], \
        'anchor_points must match forward()s flattening order exactly'
    assert boxes.shape == (2, anchors.shape[0], 4)
    assert (boxes[..., 2] >= boxes[..., 0]).all() and (boxes[..., 3] >= boxes[..., 1]).all()


def test_chunk_shuffle_is_a_permutation_that_keeps_locality():
    """Every index exactly once, few videos at a time: `_reader` caches per worker, so a wider
    draw re-opens containers (486 ms/batch vs 40)."""
    chunk, mix, n_pools = 512, 4, 3
    n = chunk * mix * n_pools                     # exact multiple, so pools are position-aligned
    s = ChunkShuffle(n, chunk=chunk, mix=mix, seed=0)
    order = list(iter(s))
    assert sorted(order) == list(range(n)), 'must visit every index exactly once'
    assert order != list(range(n)), 'must actually shuffle'
    for i in range(0, n, chunk * mix):            # one pool = at most `mix` distinct videos
        assert len({j // chunk for j in order[i:i + chunk * mix]}) == mix
    assert list(iter(s)) != order, 'a second epoch must reshuffle'


def _two_cohort_root(tmp_path, n_annot_frames=2, n_tracked_frames=30):
    """A root with ONE `annotated` session and ONE `tracked` session, deliberately lopsided.

    The shape rat-city-combined actually has: a handful of annotated stills against one long
    tracked clip, so an unweighted draw and a weighted one cannot agree by accident.
    """
    from tailcyclenet import format as fmt
    from tests.conftest import KPTS_3D, _rig, _write_frames

    W = H = 48
    root = tmp_path / 'two_cohort'
    for name, source, n_lab, T in (('a', 'annotated', n_annot_frames, n_annot_frames),
                                   ('t', 'tracked', n_tracked_frames, n_tracked_frames)):
        path = root / 'train' / name
        lab = fmt.empty_labels(1, T, len(KPTS_3D), 1, mode3d=False)
        frames = list(range(n_lab))
        lab.vis2d[0, frames, :, 0] = fmt.VISIBLE
        lab.points2d[0, frames, :, 0] = (
            10.0 + 3.0 * np.arange(len(frames), dtype=np.float32))[:, None, None] % (W - 10)
        fmt.write_session(path, mode='2d', units='px', label_source=source, names=KPTS_3D,
                          rig=_rig([('cam0', W, H, False, False, 0)]),
                          groups={'g0': fmt.Group('g0', T)}, labels={'g0': lab})
        _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return root


def _multi_detector_roots(tmp_path):
    """Two colliding roots; the larger one has four tracked sessions instead of one."""
    import shutil

    root_dir = tmp_path / 'roots'
    root_dir.mkdir()
    small = _two_cohort_root(tmp_path / 'small_src', n_annot_frames=2, n_tracked_frames=4)
    large = _two_cohort_root(tmp_path / 'large_src', n_annot_frames=2, n_tracked_frames=20)
    shutil.copytree(small, root_dir / 'alpha')
    shutil.copytree(large, root_dir / 'zeta')
    for i in range(3):
        shutil.copytree(root_dir / 'zeta' / 'train' / 't',
                        root_dir / 'zeta' / 'train' / f't_extra_{i}')
    return root_dir


def test_multi_root_weights_balance_roots_pool_and_validate_names(tmp_path):
    """Balanced: equal root shares with each root's cohort split intact; pooled: shares follow
    the combined population; `alpha` composes; an unknown `boxes_by_dataset` root is refused."""
    root = _multi_detector_roots(tmp_path)
    ds = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=0)
    weights = ds.train_weights(annot_frac=0.25, balance_datasets=True)
    assert ds.root_names == ['alpha', 'zeta']
    assert ds.root_mix(weights) == pytest.approx({'alpha': 0.5, 'zeta': 0.5})
    for cohorts in ds.root_cohort_mix(weights).values():
        assert cohorts == pytest.approx({'annotated': 0.25, 'tracked': 0.75})

    pooled = ds.train_weights(annot_frac=0.25, balance_datasets=False)
    assert pooled == pytest.approx(ds.default_train_weights(0.25, balance_datasets=False))
    assert ds.root_mix(pooled)['zeta'] > 0.65
    assert ds.cohort_mix(pooled) == pytest.approx({'annotated': 0.25, 'tracked': 0.75})

    alpha = ds.train_weights(annot_frac=None, alpha=0.5, balance_datasets=True)
    assert ds.root_mix(alpha) == pytest.approx({'alpha': 0.5, 'zeta': 0.5})
    single = BoxDataset(root / 'alpha', 'train', input_wh=(64, 64), max_frames_per_group=0)
    np.testing.assert_array_equal(single.train_weights(0.5, 0.5, True),
                                  single.default_train_weights(0.5) * single.alpha_weights(0.5))

    with pytest.raises(ValueError, match='unknown dataset roots'):
        BoxDataset(root, 'train', boxes_by_dataset={'typo': 'instances'})


def test_annot_frac_is_inert_on_a_single_cohort_split(dense_root):
    """The key must not affect an all-`tracked` split (3dpop/calms21/branson-fly) -- calms21 is
    the one train-time video root, where decode locality still costs."""
    ds = BoxDataset(dense_root, 'train', input_wh=(64, 64))
    assert len(ds.cohort_mix()) == 1, 'fixture must be single-cohort for this test to mean anything'
    base = ds.default_train_weights(None)
    for frac in (0.0, 0.25, 0.5, 1.0):
        w = ds.default_train_weights(frac)
        np.testing.assert_allclose(w, base, err_msg=f'annot_frac={frac} must be inert on a '
                                   'single-cohort split')


def test_annot_frac_sets_the_cohort_share_uniformly_within_a_cohort(tmp_path):
    """The mix becomes the configured number, and within a cohort a group's weight does not scale
    with its surviving frame count (the same accident one level down)."""
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    assert ds.cohort_mix()['annotated'] < 0.2, 'fixture must be lopsided or this proves nothing'
    for frac in (0.25, 0.75):
        mix = ds.cohort_mix(ds.default_train_weights(frac))
        assert mix['annotated'] == pytest.approx(frac)
        assert mix['tracked'] == pytest.approx(1.0 - frac)
    w = ds.default_train_weights(0.5)
    src = np.array([s.label_source for s, _, _, _ in ds.index])
    for cohort in ('annotated', 'tracked'):
        assert len(np.unique(w[src == cohort])) == 1
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match='annot_frac'):
            ds.default_train_weights(bad)

    # The weights are only a claim until the sampler is drawn from.
    s = CohortSampler(w, num_samples=20000, seed=0)
    first = list(iter(s))
    assert len(first) == len(s) and min(first) >= 0 and max(first) < len(ds)
    assert (src[np.array(first)] == 'annotated').mean() == pytest.approx(0.5, abs=0.02)
    assert list(iter(s)) != first, 'a second epoch must redraw'


def _two_group_root(tmp_path, n1=4, n2=32, label_source='tracked'):
    """Two SINGLE-COHORT sessions with different frame counts, isolated from `annot_frac`."""
    from tailcyclenet import format as fmt
    from tests.conftest import KPTS_3D, _rig, _write_frames

    W = H = 48
    root = tmp_path / 'two_group'
    for name, T in (('a', n1), ('b', n2)):
        path = root / 'train' / name
        lab = fmt.empty_labels(1, T, len(KPTS_3D), 1, mode3d=False)
        lab.vis2d[0, :, :, 0] = fmt.VISIBLE
        lab.points2d[0, :, :, 0] = (
            10.0 + 3.0 * np.arange(T, dtype=np.float32))[:, None, None] % (W - 10)
        fmt.write_session(path, mode='2d', units='px', label_source=label_source, names=KPTS_3D,
                          rig=_rig([('cam0', W, H, False, False, 0)]),
                          groups={'g0': fmt.Group('g0', T)}, labels={'g0': lab})
        _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return root


def test_max_frames_per_group_zero_drops_the_cap(tmp_path):
    """B1a: 0 means every labelled frame is indexed; the default cap (40) is unchanged."""
    root = _two_group_root(tmp_path, n1=4, n2=32)
    capped = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=8, seed=0)
    uncapped = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=0, seed=0)
    assert len(capped) == 4 + 8       # group 'a' (4 frames) under the cap, 'b' (32) truncated to 8
    assert len(uncapped) == 4 + 32    # nothing truncated
    # 40 (the shipped default) must be UNCHANGED from the pre-B1a behaviour.
    default = BoxDataset(root, 'train', input_wh=(64, 64), seed=0)
    assert len(default) == 4 + 32     # both groups are under the default cap of 40


def test_alpha_weights_interpolate_frame_and_group_uniform(tmp_path):
    """alpha=None is no weighting, 1.0 frame-uniform, 0.0 equal group totals; on uniform group
    sizes (3dpop's shape) every alpha is a free no-op."""
    ds = BoxDataset(_two_group_root(tmp_path / 'a', n1=4, n2=32), 'train', input_wh=(64, 64),
                    max_frames_per_group=0)
    assert ds.alpha_weights(None) is None
    w = ds.alpha_weights(1.0)
    assert np.allclose(w, w[0])
    w = ds.alpha_weights(0.0)
    keys = np.array([f'{s.session_id}/{g}' for s, g, _, _ in ds.index])
    totals = [float(w[keys == k].sum()) for k in np.unique(keys)]
    assert totals[0] == pytest.approx(totals[1])

    even = BoxDataset(_two_group_root(tmp_path / 'b', n1=16, n2=16), 'train', input_wh=(64, 64),
                      max_frames_per_group=0)
    for alpha in (0.0, 0.5):
        assert len(np.unique(np.round(even.alpha_weights(alpha), 10))) == 1


# --- detector training config (configs/detector.toml + tailcyclenet/detector/config.py) ------

BARE = """
[data]
path = "/tmp/ds"
[model]
yolox = "hybrid"
[training]
out = "/tmp/run"
"""


def _write_config(tmp_path, text, name='config.toml'):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_detector_config_bare_overlay_inherits_the_shipped_recipe(tmp_path):
    """The implicit base is `configs/detector.toml`: an absent key inherits its shipped value (not
    an old code default), and `--out`/`--iters`/`--device` land in [training]."""
    cfg = load_detector_config(_write_config(tmp_path, BARE))
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    assert d['balance_datasets'] is True and d['boxes_by_dataset'] == {}
    assert d['annot_frac'] == 0.6
    assert 'wandb' not in cfg
    assert m['p2'] is True and m['pretrained'] == ''          # hybrid: no pretrained source
    assert (t['assignment'], t['box_loss'], t['tal_topk']) == ('tal', 'ciou', 20)
    assert 'assignment' in TRAINING_KEYS
    assert t['weight_decay'] == 0.01 and t['max_pos_per_gt'] == 10 and t['box_weight'] == 15.0
    assert t['iou_aware_obj'] is True and t['iou_aware_warmup'] == 2000
    assert (t['optimizer'], t['muon_momentum'], t['muon_lr_scale']) == ('muon', 0.95, 1.0)
    assert t['warmup_steps'] == 500 and (t['beta1'], t['beta2']) == (0.9, 0.95)
    assert t['shared_head'] is False and t['fpn_upsample'] == 'bilinear'

    overlay = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[training]
out = "/tmp/run-det"
""", 'overlay.toml')
    cfg = load_detector_config(overlay, out='/tmp/run-det', iters=7, device='cpu')
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    assert d['boxes'] == 'instances' and d['min_crop_dim'] == 64 and d['min_box_px'] == 32
    assert d['max_input_px'] == 4 * 416 * 416 and d['val_frames_per_group'] == 8
    assert 'frames_per_group' not in d and 'use_regions' not in d and 'keypoints' not in d
    assert d['augment'] is True and d['augment_strong'] is True and d['rotate_deg'] == 180.0
    assert d['reduce'] is False and d['hflip'] is True
    assert d['input_wh'] is None and d['tile_wh'] is None
    assert d['tile_scale'] == 1.0 and d['tile_bg_per_frame'] == 1
    assert m['yolox'] == 'convnext-t' and m['pretrained'] == 'imagenet'
    assert (t['out'], t['iters'], t['device']) == ('/tmp/run-det', 7, 'cpu')
    assert t['batch_size'] == 8 and t['lr'] == 1e-3 and t['num_workers'] == 8 and t['seed'] == 23
    assert t['eval_every'] == 2000 and t['eval_batches'] == 25


_REFUSED = [  # (block, line, match): each is a decision the loader must refuse by name
    ('data', 'bogus = 1', 'unknown key'),
    ('model', 'bogus = 1', 'unknown key'),
    ('training', 'bogus = 1', 'unknown key'),
    ('data', 'boxes = "nope"', 'boxes'),
    ('model', 'yolox = "nope"', 'yolox'),
    ('model', 'pretrained = "/some/path.pth"', 'pretrained'),
    ('data', 'balance_datasets = "false"', 'balance_datasets'),
    ('data', 'boxes_by_dataset = []', 'boxes_by_dataset must be a table'),
    ('data', 'annot_frac = 1.5', 'annot_frac'),
    ('training', 'weight_decay = -0.1', 'weight_decay'),
    ('training', 'assignment = "bad"', 'assignment'),
    ('training', 'box_loss = "bad"', 'box_loss'),
    ('training', 'optimizer = "sgd"', 'optimizer'),
    ('training', 'fpn_upsample = "bicubic"', 'fpn_upsample'),
    ('wandb', 'mode = "invalid"', r'\[wandb\]\.mode'),
    (None, 'extends = "../detector.toml"', 'extends'),
]


def test_detector_config_refuses_bad_keys_and_values_by_name(tmp_path):
    for i, (block, line, match) in enumerate(_REFUSED):
        text = BARE + '[wandb]\n' if block == 'wandb' else BARE
        if block is None:
            text = line + '\n' + text
        elif line.split(' =')[0] + ' =' in text:
            text = re.sub(rf'^{line.split(" =")[0]} = .*$', line, text, flags=re.M)
        else:
            text = text.replace(f'[{block}]\n', f'[{block}]\n{line}\n', 1)
        with pytest.raises(SystemExit, match=match):
            load_detector_config(_write_config(tmp_path, text, f'bad{i}.toml'))


def test_detector_config_wandb_block_and_logging(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace

    from tailcyclenet.detector.train import (_init_detector_wandb, _log_detector_wandb,
                                             _wandb_eval_metrics)

    cfg = load_detector_config(_write_config(tmp_path, BARE + """[wandb]
project_name = "tailcycle-detector"
path = "/tmp/wandb"
mode = "online"
"""))
    assert cfg['wandb'] == {'project_name': 'tailcycle-detector', 'path': '/tmp/wandb',
                            'mode': 'online'}

    scores = {'val_macro': {'r50': 0.75, 'n_gt': 12, 'nan': float('nan')},
              'val_root_facemap': {'r50': 0.5, 'label': 'ignored'}}
    assert _wandb_eval_metrics(scores, {'q50': 0.87}) == {
        'eval/val_macro/r50': 0.75, 'eval/val_macro/n_gt': 12.0,
        'eval/val_root_facemap/r50': 0.5, 'train/objectness_q50': 0.87}

    calls = {}
    fake = ModuleType('wandb')
    fake.run = SimpleNamespace(name='test-run')
    fake.init = lambda **kwargs: calls.update(init=kwargs)
    fake.log = lambda values, step: calls.update(log=(values, step))
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    config = {'wandb': {'project_name': 'tailcycle-detector', 'path': str(tmp_path),
                        'mode': 'online'}}
    wb = _init_detector_wandb(config, tmp_path / 'run')
    _log_detector_wandb(wb, {'train/loss': 1.25}, 50)
    assert calls['init']['project'] == 'tailcycle-detector'
    assert calls['init']['mode'] == 'online' and calls['init']['config'] is config
    assert calls['log'] == ({'train/loss': 1.25}, 50)


def test_load_detector_config_overrides_apply_before_the_required_check(tmp_path):
    """`--out` must rescue `out = ""` (the override used to land AFTER the required-check), and
    `--iters`/`--device` share that path."""
    cfg = _write_config(tmp_path, BARE.replace('"/tmp/run"', '""') + 'iters = 100\n')
    loaded = load_detector_config(cfg, out=tmp_path / 'run', iters=5, device='cuda:1')
    assert loaded['training']['out'] == str(tmp_path / 'run')
    assert loaded['training']['iters'] == 5 and loaded['training']['device'] == 'cuda:1'
    with pytest.raises(SystemExit, match='out.*required'):
        load_detector_config(cfg)


def test_detector_config_round_trips_through_the_run_folder(tmp_path):
    """The recorded config.toml (None values dropped) loads back to the same recipe."""
    from tailcyclenet.detector.config import load_detector_config

    src = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "keypoints"
min_crop_dim = 64
input_wh = [96, 64]
tile_wh = []
[model]
yolox = "hybrid"
[training]
out = "/tmp/run"
iters = 2
""", 'src.toml')
    cfg = load_detector_config(src, out=str(tmp_path / 'run'), iters=2, device='cpu')
    import toml
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'config.toml').write_text(toml.dumps(cfg))
    again = load_detector_config(tmp_path / 'run' / 'config.toml')
    assert again['data']['input_wh'] == [96, 64] and again['data']['tile_wh'] is None
    assert again['data']['boxes'] == 'keypoints'
    assert again['training']['iters'] == 2
    assert again['model']['yolox'] == 'hybrid'


def test_per_root_detector_overlays_load_and_raise_val_frames_per_group(tmp_path):
    """Each shipped per-root overlay carries `[data].path` and raises `val_frames_per_group`
    past 8: each of those roots has ONE val group."""
    root = Path(__file__).resolve().parent.parent / 'configs' / 'detector'
    want = {'rat-city.toml': 64, 'branson-fly.toml': 32, '3dpop.toml': 16}
    for name, expect in want.items():
        cfg = load_detector_config(root / name, out=tmp_path / name)
        assert cfg['data']['path'], f'{name}: [data].path must be set (no CLI override exists)'
        assert cfg['data']['val_frames_per_group'] == expect, name
        assert cfg['data']['val_frames_per_group'] > 8, \
            f'{name}: must exceed the shipped default of 8 or the overlay does nothing'


def test_giou_is_zero_for_a_perfect_box():
    b = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    assert float(giou_loss(b, b)) < 1e-5
    assert float(giou_loss(b, b + 200)) > 1.0        # disjoint -> worse than any overlap


def test_tal_finds_edge_candidates_without_center_radius():
    anchors = torch.tensor([[4., 4., 8.], [12., 12., 8.], [20., 20., 8.], [80., 80., 8.]])
    gt = torch.tensor([[0., 0., 100., 100.]])
    obj = torch.zeros(4)
    boxes = torch.tensor([[0., 0., 8., 8.], [4., 4., 16., 16.],
                          [12., 12., 24., 24.], [70., 70., 90., 90.]])
    tal_pos, tal_gt = assign_tal(anchors, gt, obj, boxes, topk=3)
    center_pos, _ = assign(anchors, gt)
    assert center_pos.numel() <= 1
    assert tal_pos.numel() >= 3 and bool((tal_gt == 0).all())


def test_ciou_has_finite_nonzero_gradient_for_disjoint_boxes():
    pred = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
    target = torch.tensor([[5., 5., 6., 6.]])
    loss = ciou_loss(pred, target).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0


def test_recall_v2_loss_switches_are_live_together():
    torch.manual_seed(11)
    anchors = YOLOXNano(version='hybrid').anchor_points(128, 128, 'cpu')
    obj = torch.randn(1, anchors.shape[0], requires_grad=True)
    boxes = torch.rand(1, anchors.shape[0], 4) * 128
    # Make every decoded box valid for the synthetic assignment quality calculation.
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:].abs() + 1.0
    gt = torch.tensor([[[8., 8., 96., 96.]]])
    loss, parts = detector_loss(obj, boxes, anchors, gt, assignment='tal', box_loss_fn='ciou')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(obj.grad).all() and parts['n_pos'] > 0


def test_detector_loss_giou_path_is_byte_identical():
    torch.manual_seed(12)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10., 10., 40., 40.]]])
    a, ap = detector_loss(obj, boxes, anchors, gt)
    b, bp = detector_loss(obj, boxes, anchors, gt, box_loss_fn='giou')
    assert torch.equal(a, b) and ap == bp


def test_assign_ignores_nan_boxes():
    """An animal absent from a view gets a NaN box, and objectness must learn "nothing here"."""
    m = YOLOXNano(version='hybrid')
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[float('nan')] * 4])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == 0

    gt = torch.tensor([[float('nan')] * 4, [30.0, 30.0, 90.0, 90.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    assert (gix == 1).all(), 'positives must point at the finite box, not the NaN one'


def test_every_positive_anchor_is_inside_its_own_box():
    """A positive whose centre is outside its GT box has an unreachable regression target.
    `inside | near` shipped and left 71% of rat-city's positives unreachable."""
    m = YOLOXNano(version='hybrid')
    anchors = m.anchor_points(256, 256, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 60.0, 70.0],        # small: radius reaches well past it
                       [100.0, 30.0, 240.0, 200.0]])    # large: inside reaches past the radius
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    cx, cy = anchors[pos, 0], anchors[pos, 1]
    box = gt[gix]
    assert ((cx > box[:, 0]) & (cx < box[:, 2]) & (cy > box[:, 1]) & (cy < box[:, 3])).all(), \
        'a positive anchor outside its assigned box cannot reach the target it is given'
    assert len(set(gix.tolist())) == 2, 'both boxes must keep positives'


def test_assign_gives_each_anchor_one_box():
    m = YOLOXNano(version='hybrid')
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [30.0, 30.0, 110.0, 110.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == len(set(pos.tolist())), 'an anchor claimed by two boxes cancels'


def test_max_pos_per_gt_caps_a_large_box_and_leaves_a_small_one_alone():
    """T2.2: None is uncapped and unchanged; K caps a big box's candidacy without evicting a
    small box's own anchors, in `assign` and in the loss's `n_pos`."""
    anchors = YOLOXNano(version='hybrid').anchor_points(256, 256, torch.device('cpu'))
    for a, b in zip(assign(anchors, torch.tensor([[20.0, 20.0, 100.0, 100.0]])),
                    assign(anchors, torch.tensor([[20.0, 20.0, 100.0, 100.0]]),
                           max_pos_per_gt=None)):
        torch.testing.assert_close(a, b)

    whole = torch.tensor([[10.0, 10.0, 246.0, 246.0]])
    assert assign(anchors, whole)[0].numel() > 10
    pos, gix = assign(anchors, whole, max_pos_per_gt=10)
    assert pos.numel() <= 10 and bool((gix == 0).all())

    gt = torch.tensor([[10.0, 10.0, 130.0, 130.0], [200.0, 200.0, 216.0, 216.0]])
    small_u = int((assign(anchors, gt)[1] == 1).sum())
    assert 0 < small_u < 10, 'the small box needs an uncapped count under K'
    _, gix_c = assign(anchors, gt, max_pos_per_gt=10)
    assert int((gix_c == 1).sum()) == small_u and int((gix_c == 0).sum()) <= 10

    torch.manual_seed(8)
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 256
    _, up = detector_loss(obj, boxes, anchors, whole[None])
    _, cp = detector_loss(obj, boxes, anchors, whole[None], max_pos_per_gt=10)
    assert up['n_pos'] > 10 >= cp['n_pos']


def test_loss_is_finite_with_no_animal_anywhere():
    m = YOLOXNano(version='hybrid')
    x = torch.zeros(2, 3, 128, 128)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 128, x.device)
    gt = torch.full((2, 1, 4), float('nan'))
    loss, parts = detector_loss(obj, boxes, anchors, gt)
    assert torch.isfinite(loss) and parts['n_pos'] == 0


def test_detector_loss_drops_negatives_on_incomplete_frames_but_keeps_positives():
    anchors = torch.tensor([[5.0, 5.0, 8.0], [40.0, 40.0, 8.0]])
    obj = torch.zeros(2, 2, requires_grad=True)
    pred_boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [35.0, 35.0, 45.0, 45.0]]] * 2)
    gt = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]] * 2)
    neg = torch.tensor([True, False])
    loss, parts = detector_loss(obj, pred_boxes, anchors, gt, negative_supervision=neg)
    loss.backward()
    assert parts['negative_frames'] == 0.5
    assert obj.grad[0, 1] != 0, 'a complete frame trains its background negative'
    assert obj.grad[1, 0] != 0, 'an incomplete frame still trains its positive'
    assert obj.grad[1, 1] == 0, 'an incomplete frame trains no background negative'
    same, _ = detector_loss(obj.detach(), pred_boxes, anchors, gt,
                            negative_supervision=torch.tensor([True, True]))
    base, _ = detector_loss(obj.detach(), pred_boxes, anchors, gt)
    assert float(same) == float(base)


def test_detector_loss_iou_aware_holds_hard_target_during_warmup():
    """`it < iou_aware_warmup` keeps the hard 1.0 target: an early near-zero IoU must not teach
    objectness to stay off."""
    torch.manual_seed(3)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5             # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    warm, wp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000, it=0)
    hard, hp = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(warm) == float(hard), 'during warmup the loss must equal the hard-target one'
    assert wp['iou_target'] == pytest.approx(1.0)


def test_detector_loss_iou_aware_switches_to_iou_target_after_warmup():
    """Past warmup (or `it=None`) the positive's target is the DETACHED IoU of its predicted and
    GT box, matching a direct `paired_iou`."""
    torch.manual_seed(4)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5              # bad boxes -> IoU well under 1
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    past, pp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000,
                             it=2000)
    hard, _ = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(past) != float(hard), 'past warmup the loss must differ from the hard-target one'
    pos, gix = assign(anchors, gt[0])
    assert pos.numel()
    want = float(paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1).mean())
    assert pp['iou_target'] == pytest.approx(want, abs=1e-5) and want < 1.0
    none_it, p1 = detector_loss(obj, boxes, anchors, gt, iou_aware=True, it=None)
    torch.testing.assert_close(none_it, past)
    assert p1['iou_target'] == pytest.approx(pp['iou_target'])


def test_iou_aware_weight_forcing_is_keyed_on_pos_mask_not_target_value():
    """The guard forcing a true positive's objectness weight back to 1 under
    `negative_supervision` must key on `pos_mask`, not the target value -- under `iou_aware`
    with bad boxes the target is near 0."""
    torch.manual_seed(6)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 5              # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]]])
    no_negatives = torch.tensor([False])                   # supervises NO negatives

    _, sp = detector_loss(obj, boxes, anchors, gt, negative_supervision=no_negatives,
                          iou_aware=True, iou_aware_warmup=0, it=1)
    assert sp['iou_target'] < 0.5, 'the test needs a genuinely low IoU to be a real check'
    # The OBJ term (not the box term, which iou_aware never touches) must be identical: the
    # positive's weight was forced to 1 in both cases, only the BCE TARGET value differs, and a
    # BCE(logit, 1.0) vs BCE(logit, iou) at weight=1 are different numbers -- so this checks obj
    # is computed off the same forced weight, not that the two total losses match.
    pos, gix = assign(anchors, gt[0])
    assert pos.numel()
    logit = obj[0, pos]
    target_hard = torch.ones_like(logit)
    target_soft = paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1)
    import torch.nn.functional as F
    norm = max(pos.numel(), 1)          # `detector_loss`'s own `obj_all / max(n_pos, B)`, B=1 here
    expected_hard_obj = F.binary_cross_entropy_with_logits(
        logit, target_hard, reduction='sum') / norm
    expected_soft_obj = F.binary_cross_entropy_with_logits(
        logit, target_soft, reduction='sum') / norm
    # Re-derived directly rather than relying on `hard`/`soft` above (which also include
    # box_weight * box, identical in both arms and therefore not informative here) -- read `obj`
    # off `parts` instead.
    _, hp = detector_loss(obj, boxes, anchors, gt, negative_supervision=no_negatives, iou_aware=False)
    assert hp['obj'] == pytest.approx(float(expected_hard_obj), rel=1e-4)
    assert sp['obj'] == pytest.approx(float(expected_soft_obj), rel=1e-4)


def test_box_arithmetic_has_one_copy():
    """`detector.boxes` is the one copy of centre/side/IoU arithmetic: numpy (`link_rows`) and
    torch (association/NMS) agree, and `paired_iou` is `box_iou`'s diagonal."""
    from tailcyclenet.detector.boxes import box_center_dist
    from tailcyclenet.detector.boxes import centres, sides
    b = np.array([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 7.0, 9.0]], np.float32)
    np.testing.assert_array_equal(centres(b), centres(torch.from_numpy(b)).numpy())
    np.testing.assert_array_equal(sides(b), sides(torch.from_numpy(b)).numpy())
    np.testing.assert_array_equal(centres(b), [[5.0, 10.0], [6.0, 7.0]])
    np.testing.assert_array_equal(sides(b), [15.0, 3.0])

    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    c = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [5.0, 5.0, 15.0, 15.0]])
    torch.testing.assert_close(paired_iou(a, c), box_iou(a, c).diagonal())
    assert 0.0 < float(paired_iou(a, c)[1]) < 1.0
    # centre distance is in units of box side, so it is scale-free
    assert box_center_dist(a * 2, c * 2)[1, 1].item() == pytest.approx(
        box_center_dist(a, c)[1, 1].item())


def test_decode_suppresses_duplicates():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.], [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2, 'the two overlapping boxes must collapse to one'
    assert s[0] > s[1]


def test_decode_trace_reports_score_nms_and_top_k_without_changing_output():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.],
                          [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    plain = decode(logits, boxes, top_k=1, iou_thresh=0.5)
    traced = decode(logits, boxes, top_k=1, iou_thresh=0.5, return_trace=True)
    got = traced[:2]
    trace = traced[2]
    for a, b in zip(plain, got):
        assert torch.equal(a, b), 'diagnostic tracing must not alter deployed boxes/scores'
    assert (trace['n_total'], trace['n_score'], trace['n_nms'], trace['n_top_k']) == (3, 3, 2, 1)
    assert trace['score_boxes'].shape[0] == 3
    assert trace['nms_boxes'].shape[0] == 2


def test_decode_center_dist_is_on_by_default():
    """A5: a near-concentric duplicate (IoU 0.16) survives IoU-only NMS; the default
    centre-distance NMS collapses it, and `center_dist_thresh=None` restores IoU-only."""
    boxes = torch.tensor([[10., 10., 50., 50.], [-20., -20., 80., 80.], [400., 400., 440., 440.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2 and s[0] > s[1]
    assert decode(logits, boxes, top_k=5, iou_thresh=0.5, center_dist_thresh=None)[0].shape[0] == 3


def test_reduce_factor_never_decodes_below_the_target():
    """A reduced decode must still downscale into the target; under tiling the comparison is
    against the deployed whole-frame size, where tile_scale 1.0 decodes natively."""
    from tailcyclenet.detector.letterbox import reduce_factor

    assert reduce_factor((4696, 2048), (640, 288)) == 4       # rat-city: 1174x512, still above
    assert reduce_factor((4696, 2048), (896, 384)) == 4       # 1174x512 still clears 896x384
    assert reduce_factor((1024, 570), (544, 320)) == 1        # calms21: 1/2 is already below
    assert reduce_factor((640, 480), (640, 480)) == 1
    for src, out in (((4696, 2048), (640, 288)), ((3840, 2160), (544, 320))):
        n = reduce_factor(src, out)
        assert src[0] / n >= out[0] and src[1] / n >= out[1], 'the remaining resize must be a '\
            'downscale, or the letterbox is upsampling a decimated frame'
    size = (4696, 2048)                                   # rat-city, tiled at tile_scale 1.0
    for tile in ((640, 640), (640, 288)):
        assert reduce_factor(size, tile) > 1, f'{tile}: the old comparison must decimate'
    assert reduce_factor(size, size) == 1, 'deployment decodes natively, so training must too'
    assert reduce_factor(size, (size[0] * 0.125, size[1] * 0.125)) > 1


def test_letterbox_scale_is_in_source_units_under_reduction():
    """`unletterbox_boxes` never sees the decode: a reduced decode that changed the returned
    scale would move every box by the reduction factor (4x on rat-city)."""
    full = np.zeros((2048, 4696, 3), np.uint8)
    quarter = np.zeros((512, 1174, 3), np.uint8)
    a, s_a, p_a = letterbox(full, (640, 288))
    b, s_b, p_b = letterbox(quarter, (640, 288), src_wh=(4696, 2048))
    assert a.shape == b.shape
    assert s_a == s_b and p_a == p_b
    box = torch.tensor([[100.0, 50.0, 700.0, 400.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * s_b + p_b[0]
    moved[:, 1::2] = moved[:, 1::2] * s_b + p_b[1]
    torch.testing.assert_close(unletterbox_boxes(moved, s_b, p_b), box, atol=1e-3, rtol=0)


def test_unletterbox_clamps_a_runaway_box_into_the_frame():
    """A decoded side can reach ~12,910 px; IoU-only NMS cannot suppress it, so the clamp bounds
    it into the frame and a zero-area box comes back NaN.
    """
    from tailcyclenet.detector.letterbox import unletterbox_boxes

    b = torch.tensor([[-5000.0, -5000.0, 20000.0, 20000.0], [10.0, 10.0, 9.0, 40.0]])
    out = unletterbox_boxes(b, 1.0, (0, 0), src_wh=(640, 480))
    assert out[0].tolist() == [0.0, 0.0, 640.0, 480.0]
    assert torch.isnan(out[1]).all(), 'a box with no positive area is not a detection'
    # Without `src_wh` -- the training path, which has no frame to clamp against -- nothing changes.
    assert unletterbox_boxes(b, 1.0, (0, 0))[0, 2].item() == 20000.0


def test_a_video_read_locks_per_container_not_globally():
    """`dataset._read_video` locks PER PATH: one global lock would kill `detect_group`'s multi-
    camera decode overlap (3.5x on 3dpop)."""
    from tailcyclenet import dataset as ds

    a, b = ds._read_lock_for('/x/cam0.mp4'), ds._read_lock_for('/x/cam1.mp4')
    assert a is not b
    assert ds._read_lock_for('/x/cam0.mp4') is a
    with a:                       # holding one container's lock must not block another's
        assert b.acquire(blocking=False)
        b.release()


def test_targets_are_the_crop_rule(tiny_root):
    """The regression target must BE crop_box_for_points, not something similar to it."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2)
    item = ds[0]
    x, boxes = item['x'], item['boxes']
    assert x.shape == (3, 128, 128)
    sess, gid, f, ci = ds.index[0]
    lab = sess.labels(gid)
    cam = sess.rig.tracktail()[ci]
    for s in range(boxes.shape[0]):
        want = crop_box_for_points(
            torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32),
            cam['size'], ds.min_crop_dim)
        got = boxes[s]
        if want is None:
            assert torch.isnan(got).all()
            continue
        # the stored box is the crop rule's box, letterboxed; undo that and it must come back
        img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
        _, scale, pad = letterbox(img, ds.input_wh)
        back = unletterbox_boxes(got[None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_augmented_targets_are_still_the_crop_rule(tiny_root):
    """Augmentation warps the GEOMETRY and re-derives the box; scaling the box would scale the
    pad but not the `min_crop_dim` floor -- a box the crop rule can never emit."""
    from tailcyclenet.detector.augment import random_affine

    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                       max_frames_per_group=2)
    sess, gid, f, ci = ds.index[0]
    cam = sess.rig.tracktail()[ci]
    warp = random_affine(cam['size'], np.random.default_rng([ds.seed, 0]))
    got = ds.boxes_for(0, warp)

    _, scale, pad = letterbox(np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8),
                              ds.input_wh)
    lab = sess.labels(gid)
    for s in range(got.shape[0]):
        p = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
        p = p @ torch.as_tensor(warp[:, :2]).T + torch.as_tensor(warp[:, 2])
        w, h = float(cam['size'][0]), float(cam['size'][1])
        outside = (p[:, 0] < 0) | (p[:, 0] > w) | (p[:, 1] < 0) | (p[:, 1] > h)
        want = crop_box_for_points(torch.where(outside[:, None], torch.nan, p),
                                   cam['size'], ds.min_crop_dim)
        if want is None:
            assert torch.isnan(got[s]).all()
            continue
        back = unletterbox_boxes(got[s][None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)

    # And with no warp it is byte-identical to the unaugmented loader -- `augment` must be a key
    # you turn on, not a thing that leaks into a run that did not ask for it.
    torch.testing.assert_close(ds.boxes_for(0), plain.boxes_for(0))


def test_augment_decodes_real_pixels_through_getitem(tiny_root):
    """`augment=True` must survive a real `__getitem__` (the only caller of `photometric`): a
    cleanup once left a `NameError` on every augmented item."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    for i in range(min(2, len(ds))):
        item = ds[i]
        x = item['x']
        _boxes = item['boxes']
        assert x.shape == (3, 48, 64)
        assert torch.isfinite(x).all()
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, \
            'the photometric gain must still land in the normalised [0, 1] range'


def test_rotation_off_is_byte_identical_and_on_turns_about_its_centre():
    """`rotate_deg=0` reproduces the pre-rotation matrix draw for draw (drawing an angle and
    zeroing it would reseat every later draw); rotation turns about `centre`."""
    def before(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5):
        w, h = float(size[0]), float(size[1])
        s = rng.uniform(*scale)
        sx = -s if rng.random() < hflip else s
        cx, cy = w / 2, h / 2
        return np.array([[sx, 0.0, cx - sx * cx + rng.uniform(-translate, translate) * w],
                         [0.0, s, cy - s * cy + rng.uniform(-translate, translate) * h]],
                        np.float32)

    for seed in range(5):
        a = before((4696, 2048), np.random.default_rng([0, seed]))
        b = random_affine((4696, 2048), np.random.default_rng([0, seed]))
        assert np.array_equal(a, b), f'seed {seed} moved'

    # And the rotation is about `centre`, which is the whole reason tiling and rotation compose.
    M = random_affine((1000, 1000), np.random.default_rng(1), scale=(1.0, 1.0), translate=0.0,
                      hflip=0.0, rotate_deg=180.0, centre=(300.0, 700.0))
    fixed = M @ np.array([300.0, 700.0, 1.0])
    np.testing.assert_allclose(fixed, [300.0, 700.0], atol=1e-3)


def test_photometric_is_one_gain_draw():
    """`photometric(img, rng)` consumes exactly ONE `uniform(0.7, 1.3)` and applies it as a
    clipped gain -- the augmentation stream's draw order rests on this."""
    img = (np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3) % 200).astype(np.uint8)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    out1 = photometric(img, rng1)
    gain = rng2.uniform(0.7, 1.3)
    np.testing.assert_array_equal(out1, np.clip(img * gain, 0, 255).astype(np.uint8))
    assert rng1.uniform() == rng2.uniform()


def test_a_tiled_item_turns_about_its_own_tile(tiny_root):
    """A tile is cut AFTER the warp, so the warp turns about the tile's centre (about the frame
    centre only 0.075 of rat-city tiles kept their animal)."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(32, 32), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, rotate_deg=180.0,
                    tile_wh=(32, 32), tile_scale=1.0)
    i = next(j for j in range(len(ds)) if ds.origins[j] is not None)
    ox, oy = ds.origins[i]
    tw, th = ds._tile_extent()
    assert ds._warp_centre(i) == (ox + tw / 2, oy + th / 2)

    sess, _, _, ci = ds.index[i]
    size = tuple(sess.rig.size(sess.cam_names[ci]))
    M = random_affine(size, np.random.default_rng([ds.seed, i]), hflip=0.0, rotate_deg=180.0,
                      translate=0.0, scale=(1.0, 1.0), centre=ds._warp_centre(i))
    c = np.array([ox + tw / 2, oy + th / 2, 1.0])
    np.testing.assert_allclose(M @ c, c[:2], atol=1e-3)


def test_chunk_is_one_containers_worth_of_index(tiny_root):
    """`ChunkShuffle`'s block must be one video: a hardcoded 512 spanned 13 calms21 videos and
    ran the reader cache at a 16% hit rate."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), max_frames_per_group=2)
    n_src = len({(s.session_id, g, c) for s, g, _, c in ds.index})
    assert ds.chunk == len(ds.index) // n_src
    assert ds.chunk < len(ds.index) or n_src == 1


def test_collate_pads_uneven_animal_counts():
    a = {'x': torch.zeros(3, 8, 8), 'boxes': torch.zeros(2, 4)}
    b = {'x': torch.zeros(3, 8, 8), 'boxes': torch.zeros(5, 4)}
    got = box_collate([a, b])
    assert got['x'].shape == (2, 3, 8, 8) and got['boxes'].shape == (2, 5, 4)
    assert torch.isnan(got['boxes'][0, 2:]).all()


def test_an_untiled_checkpoints_tile_scale_is_dropped(tmp_path):
    """An untiled checkpoint records `tile_scale = 1.0`; reaching `detect_group` it would
    letterbox the whole frame at `frame_wh * scale`."""
    import conftest as cf

    from tailcyclenet.detector.checkpoint import load_detector
    p = tmp_path / 'detector.pth'
    base = cf._detector_checkpoint()
    torch.save({**base, 'tile_wh': None, 'tile_scale': 1.0}, p)
    assert load_detector(p).tile_scale is None
    # ...and a genuinely tiled one still keeps it, or the tiled path loses its whole point.
    torch.save({**base, 'tile_wh': [640, 640], 'tile_scale': 0.5}, p)
    assert load_detector(p).tile_scale == 0.5


def test_tiled_targets_are_still_the_crop_rule(tiny_root):
    """The tiled box is RE-DERIVED in source px, never scaled by tile_scale."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2,
                    tile_wh=(32, 32), tile_scale=1.0)
    from tailcyclenet.detector.letterbox import tile_transform
    for i in range(min(3, len(ds))):
        sess, gid, f, ci = ds.index[i]
        boxes = ds.boxes_for(i)
        lab = sess.labels(gid)
        cam = sess.rig.tracktail()[ci]
        ox, oy = ds.origins[i]
        tw, th = ds._tile_extent()
        for s in range(boxes.shape[0]):
            pts = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
            keep = ((pts[:, 0] >= ox) & (pts[:, 0] <= ox + tw) &
                    (pts[:, 1] >= oy) & (pts[:, 1] <= oy + th))
            want = crop_box_for_points(torch.where(keep[:, None], pts, torch.nan),
                                       cam['size'], ds.min_crop_dim)
            if want is None:
                assert torch.isnan(boxes[s]).all()
                continue
            scale, pad = tile_transform((ox, oy), ds.tile_scale)
            back = unletterbox_boxes(boxes[s][None], scale, pad)[0]
            torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_an_off_frame_tile_drops_its_points_and_is_grey(tiny_root):
    """Out-of-tile behaves like out-of-frame (no box), and off-frame pixels are grey, not wrapped."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    tile_wh=(16, 16), tile_scale=1.0)
    ds.origins[0] = (1000.0, 1000.0)        # the fixture's points are in [5, 43]
    assert torch.isnan(ds.boxes_for(0)).all()
    ds.origins[0] = (-16.0, -16.0)          # wholly outside, up and left
    x = ds[0]['x']
    assert x.shape == (3, 16, 16)
    torch.testing.assert_close(x, torch.full_like(x, 114 / 255.0))


def test_strong_augment_never_drops_a_box(tiny_root):
    """Appearance ops and cutout never move a box -- only mosaic-lite may ADD one, fully interior."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    for i in range(min(2, len(ds))):
        base = ds.boxes_for(i)
        for _ in range(4):
            assert ds[i]['boxes'].shape[0] >= base.shape[0]


def test_mosaic_paste_is_fully_interior_and_reencodes_the_crop_rule(tiny_root):
    """The appended box is `src_box + translation`, entirely inside `input_wh`."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    rng = np.random.default_rng(1)
    for i in range(min(3, len(ds))):
        base = ds.boxes_for(i)
        img, _ = ds._load_letterbox(i)
        boxes, img2 = ds._mosaic_paste(i, base.clone(), img.copy(), rng)
        if boxes.shape[0] == base.shape[0]:
            continue    # no finite source box was found in a few tries; not this fixture's job
        new = boxes[base.shape[0]:]
        assert (new[:, 0] >= 0).all() and (new[:, 1] >= 0).all()
        assert (new[:, 2] <= ds.input_wh[0]).all() and (new[:, 3] <= ds.input_wh[1]).all()
        return
    pytest.skip('no fixture item produced a finite mosaic source box in the tries allotted')


# --- box sources --------------------------------------------------------------------------


def test_box_source_instances_uses_only_labeled_stored_boxes(tmp_path):
    """Stored labeled corners are targets, not the crop rule over keypoints."""
    from tailcyclenet import format as fmt

    root = tmp_path / 'ds'
    path = root / 'train' / 's'
    cf._session_2d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    lab.instance[1, 1, 0] = fmt.INST_LABELED  # one stored positive alongside keypoints-only a01
    fmt.write_session(path, mode='2d', units='px', label_source='annotated', names=sess.names,
                      rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=sess.flip_pairs)

    ds = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=4, box_source='instances')
    base = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8,
                      max_frames_per_group=4)
    assert len(ds) == 1
    sess, gid, frame, cam_ix = ds.index[0]
    boxes = ds[0]['boxes']
    plain_index = next(i for i, (_, _, ff, _) in enumerate(base.index) if ff == frame)
    plain = base[plain_index]['boxes']
    lab = sess.labels(gid)
    cam = sess.rig.tracktail()[cam_ix]
    img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
    _, scale, pad = letterbox(img, ds.input_wh)
    want = crop_box_for_points(torch.as_tensor(lab.boxes[1, frame, cam_ix]).view(2, 2),
                               cam['size'], ds.min_crop_dim, pad=0)
    back = unletterbox_boxes(boxes[1][None], scale, pad)[0]
    torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)
    assert not torch.isfinite(boxes[0]).any(), 'no stored box means no target, despite keypoints'
    assert torch.isfinite(plain[0]).all(), 'the keypoint source would produce a crop box'


def test_box_source_instances_trains_present_boxes_without_negatives(tiny_root):
    """A present box is a positive, and it makes the frame's background unreliable."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=0, box_source='instances')
    # every other frame has only labeled rows without stored boxes: no target, no negatives
    assert [f for _, _, f, _ in ds.index] == [1]
    item = ds[0]
    assert not torch.isfinite(item['boxes'][0]).any()
    assert torch.isfinite(item['boxes'][1]).all(), 'present boxes are positive targets'
    assert item['negative_supervision'] is False


def test_negative_supervision_follows_instance_statuses(tmp_path):
    """Negatives need labeled/absent evidence and no present row, with or without a box."""
    from tailcyclenet import format as fmt

    T = 5
    path = tmp_path / 'ds' / 'train' / 'boxes'
    rig = cf._rig([('cam0', 64, 48, False, False, 0)])
    lab = fmt.empty_labels(2, T, len(cf.KPTS_2D), 1, mode3d=False, animal_ids=['a', 'b'])
    lab.points2d = lab.vis2d = None
    lab.instance = np.full((2, T, 1), fmt.INST_NONE, np.int8)
    lab.boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    lab.instance[0, 0, 0] = fmt.INST_LABELED           # labeled box: positives + negatives
    lab.boxes[0, 0, 0] = [8, 7, 28, 29]
    lab.instance[0, 1, 0] = fmt.INST_ABSENT            # absent-only: all-background example
    lab.instance[0, 2, 0] = fmt.INST_LABELED           # labeled box beside a boxless present
    lab.boxes[0, 2, 0] = [8, 7, 28, 29]
    lab.instance[1, 2, 0] = fmt.INST_PRESENT
    lab.instance[0, 3, 0] = fmt.INST_PRESENT           # boxless present only: nothing to train
    # frame 4 has no instance row at all: no claim, not a negative
    fmt.write_session(path, mode='2d', units='px', label_source='annotated',
                      names=cf.KPTS_2D, rig=rig, groups={'g': fmt.Group('g', T)},
                      labels={'g': lab})
    cf._write_frames(path / 'groups' / 'g', 'cam0', T, (64, 48))

    ds = BoxDataset(path.parent.parent, 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=0)
    got = {f: ds[i]['negative_supervision'] for i, (_, _, f, _) in enumerate(ds.index)}
    assert got == {0: True, 1: True, 2: False}
    assert not torch.isfinite(ds[1]['boxes']).any(), 'absent rows are not positives'
    batch = box_collate([ds[0], ds[1], ds[2]])
    assert batch['negative_supervision'].tolist() == [True, True, False]


def test_box_only_sessions_fall_back_to_labeled_instance_boxes(tmp_path):
    """A session with no keypoint tables trains its stored boxes, and indexes beside a pose one."""
    root = tmp_path / 'ds'
    path = root / 'train' / 'boxes_only'
    _box_only_session(path)
    assert not (path / 'keypoints.pq').exists()
    ds = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8)
    assert len(ds) == 4
    assert {item[1] for item in ds.index} == {'video_ix0', 'video_ix10'}
    assert len(ds.box_only_fallback_sessions) == 1
    for i in range(len(ds)):
        boxes = ds.boxes_for(i)
        assert torch.isfinite(boxes[0]).all()
        assert torch.isfinite(boxes[1]).all(), 'present boxes are positive targets'

    batch = box_collate([ds[0], ds[1]])
    assert batch['negative_supervision'].tolist() == [False, False]
    assert split_batch(batch)[2] is batch['negative_supervision']

    cf._session_2d(root / 'train' / 'pose')
    mixed = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8)
    assert {sess.session_id for sess, _, _, _ in mixed.index} == {'boxes_only', 'pose'}
    assert mixed.box_only_fallback_sessions == {path}


# --- cross-view tracker ------------------------------------------------------------------


def _lever_rig(specs):
    """A NON-DEGENERATE rig: every camera at its own tilt and height, so rays through different
    animals are SKEW (a coplanar rig triangulates any pair to residual ~0). specs entries are
    (rvec_x, rvec_y, tvec_y)."""
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.format import Rig

    cams = []
    for i, (rx, ry, ty) in enumerate(specs):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([rx, ry, 0.0]),
                     tvec=np.array([0.0, ty, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    return Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
               moving=dict.fromkeys(names, False),
               calibrated=dict.fromkeys(names, True)).tracktail()


def _lever_boxes(cg, worlds, side=200.0):
    """Project `worlds` into every camera of `cg` as square boxes, in `CrossViewTracker.step`'s
    argument shapes. Large by default: every gate in track.py is in box sides."""
    from tailcyclenet.infer.track import _project

    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, np.asarray(worlds, np.float32))
        per_cam.append(torch.stack([uv[:, 0] - side / 2, uv[:, 1] - side / 2,
                                    uv[:, 0] + side / 2, uv[:, 1] + side / 2], -1))
        scores.append(torch.ones(len(worlds)))
    return per_cam, scores


def _lost_in_one_camera():
    """Animal A seen in cameras 0 and 1, MISSED in camera 2 where the only detection is animal
    B. Outputs: (cg, per_cam, scores, A)."""
    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a = np.array([0.0, 0.0, 0.0], np.float32)
    b = np.array([120.0, 0.0, 0.0], np.float32)
    per_cam, scores = _lever_boxes(cg, [a, b])
    per_cam[2], scores[2] = per_cam[2][1:], torch.ones(1)
    return cg, per_cam, scores, a


def _flat_rig():
    """Three coplanar cameras with optical axes through the origin 
    (fine where no residual gate is read)."""
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    return Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
               moving=dict.fromkeys(names, False),
               calibrated=dict.fromkeys(names, True)).tracktail()


def _dup_scene():
    """One animal detected twice in every camera (two identical box sets)."""
    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    return cg, point, per_cam, scores, duplicate, [torch.ones(2) for _ in scores]


def test_duplicate_backstop_retires_shields_and_logs_without_changing_output():
    """After `duplicate_persist` in-band frames the backstop retires the weaker duplicate, its
    freed slot does not re-birth it (the immortal duplicate of 3dpop Seq59, report 53), and
    `tracker.events` records born/retired/shielded/birth_refused -- recording changes nothing."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, _, _, _, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    outs = [tr.step(cg, duplicate, dup_scores) for _ in range(6)]
    assert len(tr.targets) == 1, 'the retired duplicate must stay retired'
    rows = [int(np.isfinite(b).all(-1).any(-1).sum()) for b, _, _ in outs[1:]]
    assert rows == [1] * 5, 'one row on the animal from the retirement on'
    first = int(np.isfinite(outs[1][0]).all(-1).any(-1).argmax())
    assert int(np.isfinite(outs[-1][0]).all(-1).any(-1).argmax()) == first, \
        'the surviving row must keep the animal'

    by_event = {}
    for e in tr.events:
        assert set(e) == {'frame', 'slot', 'event', 'detail'}, e
        assert 0 <= e['frame'] < 6, e
        by_event.setdefault(e['event'], []).append(e)
    assert len(by_event.get('born', [])) == 2
    retired = by_event.get('retired_duplicate', [])
    assert len(retired) == 1
    shielded = by_event.get('shielded', [])
    assert len(shielded) == 1 and shielded[0]['slot'] == retired[0]['detail']['winner']
    assert retired[0]['detail']['persisted'] >= 2
    assert retired[0]['detail']['loser_age'] >= 0 and retired[0]['detail']['winner_age'] >= 0
    assert by_event.get('birth_refused')

    tr2 = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for t, (boxes, sc, claimed) in enumerate(outs):
        b2, s2, c2 = tr2.step(cg, duplicate, dup_scores)
        np.testing.assert_array_equal(np.nan_to_num(b2, nan=-9e9),
                                      np.nan_to_num(boxes, nan=-9e9), err_msg=f'boxes at t={t}')
        np.testing.assert_array_equal(np.nan_to_num(s2, nan=-9e9),
                                      np.nan_to_num(sc, nan=-9e9), err_msg=f'scores at t={t}')
        np.testing.assert_array_equal(c2, claimed, err_msg=f'claimed at t={t}')


def test_identity_events_record_a_slot_dying_of_old_age():
    """`died` carries the age that crossed `max_age`: without it a starved slot reads like one
    still alive but unclaimed."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=3)
    tr.step(cg, per_cam, scores)
    assert [e['event'] for e in tr.events] == ['born']

    empty = [torch.zeros((0, 4)) for _ in cg]
    empty_scores = [s.new_zeros(0) for s in empty]
    for _ in range(6):
        tr.step(cg, empty, empty_scores)
    died = [e for e in tr.events if e['event'] == 'died']
    assert len(died) == 1 and died[0]['slot'] == 0, tr.events
    assert died[0]['detail']['age'] > tr.max_age, died[0]
    assert not tr.targets


def test_shield_survives_the_winner_vacating_for_one_frame():
    """The shield is anchored to the ANIMAL, not the winner's this-frame claims: a one-frame
    vacate let the duplicate re-seat (~3 refires per 160 frames, 3dpop Seq59)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, point, per_cam, scores, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for _ in range(2):
        tr.step(cg, duplicate, dup_scores)
    assert len(tr.targets) == 1 and tr._dup_anchor, 'a retirement must anchor its winner'
    anchored = dict(tr._dup_anchor)
    far = torch.tensor([600.0, 0.0, 0.0])
    far_cam, far_scores = _lever_boxes(cg, [far.numpy()], side=100.0)
    tr.step(cg, far_cam, far_scores)          # the winner's only work this frame is elsewhere
    assert tr._dup_anchor.keys() == anchored.keys(), 'the anchor must outlive a one-frame vacate'
    for _ in range(4):
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    seated = [t['point'] for t in tr.targets.values()
              if float(torch.linalg.norm(t['point'] - point)) < 300.0]
    assert len(seated) == 1, 'the duplicate must not re-seat on the anchored animal'
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 1


def test_shield_dies_when_the_winners_own_target_dies():
    """A winner that ages out must take its shield with it, or the next frame's unrelated birth
    at that spot is wrongly refused."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, point, per_cam, scores, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2, max_age=2)
    for _ in range(2):
        tr.step(cg, duplicate, dup_scores)
    assert tr._dup_shield, 'a retirement must shield its winner'
    empty = [b.new_zeros((0, 4)) for b in per_cam]
    empty_scores = [s.new_zeros(0) for s in scores]
    for _ in range(4):                        # starve the winner past max_age: it must die
        tr.step(cg, empty, empty_scores)
    assert not tr.targets, 'the winner must have aged out'
    assert not tr._dup_shield, 'the shield must die with the target it protects'
    boxes, _, _ = tr.step(cg, per_cam, scores)  # an unrelated single-set birth at the same spot
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 1, \
        'a dead shield must not refuse an unrelated birth'


def test_duplicate_persistence_does_not_survive_a_members_death():
    """A `_dup_contact` counter must not resume across a slot's death, or a rebirth on a new
    animal retires early."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, point, per_cam, scores, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=5, max_age=2)
    for _ in range(3):                        # 3 of 5 in-band frames: short of retirement
        tr.step(cg, duplicate, dup_scores)
    key = frozenset(tr.targets)
    assert tr._dup_contact.get(key, 0) == 3
    empty = [b.new_zeros((0, 4)) for b in per_cam]
    empty_scores = [s.new_zeros(0) for s in scores]
    for _ in range(4):                        # both slots starve past max_age and die together
        tr.step(cg, empty, empty_scores)
    assert not tr.targets
    assert not tr._dup_contact, 'a dead pair must not leave a stale counter behind'
    for _ in range(3):                        # rebirth: must count from 0, not resume at 3
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 2, \
        'a rebirth must not inherit a stale persistence count and retire early'


def test_refused_duplicate_does_not_consume_the_free_slot():
    """A6b: a refused duplicate birth must retry the SAME free slot, or a genuine group behind
    it is crowded out of the frame."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    pointA = np.array([0.0, 0.0, 0.0])
    pointB = np.array([250.0, 0.0, 0.0])    # far from A: a distinct animal, not a duplicate
    per_cam, scores = _lever_boxes(cg, [pointA, pointB], side=100.0)
    dupA = [torch.cat([b[:1], b[:1].clone()]) for b in per_cam]      # A twice per camera
    dup_scores = [torch.ones(2) for _ in scores]
    genuine = [b[1:2] for b in per_cam]                              # B, one box per camera

    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for _ in range(2):                       # both slots seat at A; the backstop retires one
        boxes, _, _ = tr.step(cg, dupA, dup_scores)
    assert len(tr.targets) == 1, 'the backstop must retire exactly one slot here'
    survivor = int(np.isfinite(boxes).all(-1).any(-1).argmax())

    frames23 = [torch.cat([d, g]) for d, g in zip(dupA, genuine)]    # [A, A, B] per camera
    boxes3, _, _ = tr.step(cg, frames23, [torch.ones(3) for _ in scores])
    other = 1 - survivor
    assert np.isfinite(boxes3[other]).all(-1).any(), \
        'the genuine birth must seat in the freed slot -- the refusal must not consume it'
    assert any(e['event'] == 'birth_refused' for e in tr.events), \
        'the duplicate refusals must still be recorded'


def test_age_stickiness_resolves_two_slots_on_one_group():
    """Two slots stranded on one animal must not flip-flop on near-ties: the fresher keeps the
    group, the stale one starves (3dpop Seq59)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    tr = CrossViewTracker(2, max_res_px=30.0, max_age=4)
    # Seat both targets on the same animal, one stale (age 2) and one fresh (age 0).
    tr.targets[0] = {'point': point.clone(), 'age': 0}
    tr.targets[1] = {'point': point.clone(), 'age': 2}
    rows = []
    for _ in range(8):
        boxes, _, _ = tr.step(cg, per_cam, scores)
        rows.append(boxes)
    assert all(np.isfinite(boxes).all(-1).any(-1).sum() == 1 for boxes in rows), \
        'two slots must never both claim one group'
    assert len(tr.targets) == 1, 'the stale duplicate must starve out'
    assert np.isfinite(rows[-1]).all(-1).any(-1).sum() == 1


def test_age_sticky_term_never_inverts_a_real_match():
    """Past `AGE_STICKY_CAP` the packed cost `affinity * 16 + CAP - age` went negative and a
    real high-affinity match read as unavailable (only when `max_age` > the cap)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    # An 80mm offset against a 200px box side is a real, well-supported match (well inside the
    # `max_move` gate) but not a perfect one -- affinity comfortably short of 1.0, exactly the
    # regime the inverted sign discards.
    per_cam, scores = _lever_boxes(cg, [(point + torch.tensor([80.0, 0.0, 0.0])).numpy()],
                                   side=200.0)
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=30)
    tr.targets[0] = {'point': point.clone(), 'age': 24}
    out, sc, claimed = tr.step(cg, per_cam, scores)
    assert np.isfinite(out).all(-1).any(), \
        'a stale (age 24) slot lost its own high-affinity candidate to nothing'
    assert tr.targets[0]['age'] == 0, 'a real match must reset age, not leave the slot starving'


def test_a_pointless_target_expires_instead_of_burning_a_slot_forever():
    """A `--min-views 1` target has an all-NaN point the matcher never sees; it must still age
    out."""
    import torch

    from tailcyclenet.infer.track import CrossViewTracker

    tr = CrossViewTracker(2, max_age=3)
    tr.targets[0] = {'point': torch.full((3,), float('nan')), 'age': 0}
    assert 0 in tr.targets

    # No detections at all: the pointless target must still age out, freeing its slot.
    for _ in range(tr.max_age + 1):
        tr.step([], [], [])
    assert 0 not in tr.targets, 'a target the matcher can never see must still be able to expire'

    # A target WITH a point is unaffected by this path -- it ages through the normal loop.
    tr2 = CrossViewTracker(2, max_age=3)
    tr2.targets[0] = {'point': torch.zeros(3), 'age': 0}
    tr2.step([], [], [])
    assert tr2.targets[0]['age'] == 1, 'the finite-point target must age exactly once per frame'


def test_min_views_1_admits_the_box_no_pair_claimed(tmp_path):
    """`min_views = 2` is the algorithm; `min_views = 1` emits each leftover box as a single-
    view instance with no invented depth."""

    from tailcyclenet.infer.associate import associate
    from tailcyclenet.format import Session

    cf._session_3d(tmp_path / 'ds' / 'test' / 's')
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    cams = sess.cgroup('g000', 0)

    # One animal both of the first two cameras see at the image centre, plus a box in camera 2 that
    # no pair can agree with (it sits in a corner).
    centre = torch.tensor([[24., 16., 40., 32.]])
    per_cam = [centre, centre, torch.cat([centre, torch.tensor([[0., 0., 8., 8.]])])]

    two = associate(cams, per_cam, max_res_px=30.0, min_views=2)
    one = associate(cams, per_cam, max_res_px=30.0, min_views=1)
    assert len(one) == len(two) + 1, f'{len(two)} -> {len(one)}: the leftover box must appear'
    extra = one[len(two)]
    assert list(extra['boxes']) == [2] and torch.isnan(extra['point']).all(), \
        'a single ray has no triangulated point, and inventing a depth would be a lie'


def test_min_views_1_single_view_birth_holds_no_slot():
    """A single-view group (NaN point by design) is written but never seated: seating it
    rotated one animal through every slot and then lost it (T1)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    box = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
    boxes_per_cam = [box] + [torch.zeros((0, 4)) for _ in cg[1:]]
    scores_per_cam = [torch.ones(1)] + [torch.zeros(0) for _ in cg[1:]]
    tr = CrossViewTracker(4, max_res_px=30.0, max_age=5, min_views=1)
    for _ in range(12):
        out, _, claimed = tr.step(cg, boxes_per_cam, scores_per_cam)
        rows = [s for s in range(4) if np.isfinite(out[s]).all(-1).any()]
        assert rows == [0]
        torch.testing.assert_close(torch.as_tensor(out[0, 0]), box[0])
        assert claimed[0, 0] == 0
        assert not tr.targets
    assert not [e for e in tr.events if e['event'] == 'born']


def test_joint_association_refuses_a_binding_the_per_camera_hungarian_accepts():
    """Joint association matches whole cross-view groups, so a slot holding A cannot bind B's
    box in the camera that missed A."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, per_cam, scores, a = _lost_in_one_camera()

    tr = CrossViewTracker(1, max_res_px=30.0)
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    out, _, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0].tolist() == [0, 0, -1], 'the joint decision must not reach into camera 2'
    np.testing.assert_allclose(tr.targets[0]['point'].numpy(), a, atol=1e-3)
    assert tr.targets[0]['age'] == 0, 'a slot that matched a group has evidence and must not age'


def test_joint_association_still_births_ages_and_resumes_like_the_shipped_path():
    """Two crossing animals with the score order swapped every frame: births on frame 0, no row
    jumps, a one-frame miss ages without dropping, the SAME slots resume."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a, b = np.array([-80.0, 0.0, 0.0]), np.array([80.0, 0.0, 0.0])
    tr = CrossViewTracker(2, max_res_px=30.0)
    rows = []
    for t in range(12):
        w = [a + [12.0 * t, 0, 0], b - [12.0 * t, 0, 0]]
        per_cam, scores = _lever_boxes(cg, w if t % 2 == 0 else w[::-1], side=40.0)
        rows.append(tr.step(cg, per_cam, scores)[0])

    assert all(np.isfinite(r).all(-1).any(-1).sum() == 2 for r in rows), 'an animal was lost'
    for s in (0, 1):
        cx = np.array([r[s, 0, [0, 2]].mean() for r in rows])
        assert np.abs(np.diff(cx)).max() < 30.0, f'row {s} jumped: {np.diff(cx)}'

    empty = [torch.zeros((0, 4)) for _ in cg], [torch.zeros((0,)) for _ in cg]
    out, _, _ = tr.step(cg, *empty)
    assert not np.isfinite(out).any() and len(tr.targets) == 2, \
        'a one-frame detector miss must not end a track under joint either'
    w = [a + [12.0 * 11, 0, 0], b - [12.0 * 11, 0, 0]]
    resumed, _, _ = tr.step(cg, *_lever_boxes(cg, w, side=40.0))
    for s in (0, 1):
        assert abs(resumed[s, 0, [0, 2]].mean() - rows[-1][s, 0, [0, 2]].mean()) < 30.0, \
            f'row {s} resumed on the other animal'


def test_the_tracker_and_associate_agree_on_a_single_uncrowded_animal():
    """On an unambiguous single animal the tracker must reproduce `associate`, so a crowded-
    case delta is a crowding result."""
    import numpy as np

    from tailcyclenet.infer.associate import associate
    from tailcyclenet.infer.track import CrossViewTracker, _project


    cg = _flat_rig()

    tr = CrossViewTracker(1, max_res_px=30.0)
    for t in range(5):
        w = np.array([[10.0 * t, 0.0, 0.0]], np.float32)
        per_cam = []
        for cam in cg:
            uv = _project(cam, w)
            per_cam.append(torch.stack([uv[:, 0] - 20, uv[:, 1] - 20,
                                        uv[:, 0] + 20, uv[:, 1] + 20], -1))
        scores = [torch.ones(1) for _ in cg]
        got, _, _ = tr.step(cg, per_cam, scores)
        ref = associate(cg, per_cam, max_res_px=30.0, max_instances=1)
        assert len(ref) == 1
        for c, box in ref[0]['boxes'].items():
            np.testing.assert_allclose(got[0, c], box.numpy(), atol=1e-4)


def test_support_count_counts_independently_corroborating_cameras():
    """E1a/E3: `_support_count` counts OTHER cameras near the reprojection, excluding the
    seeding pair and already-`used` detections."""

    from tailcyclenet.infer.associate import _support_count
    from tailcyclenet.infer.track import _project

    cg = _flat_rig()

    p3d = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    proj = [_project(cam, p3d.reshape(1, 3).numpy()) for cam in cg]     # (1,2) per camera

    # Every camera has a detection exactly at the true reprojection.
    centres_all = [p.clone() for p in proj]
    assert _support_count(cg, (0, 1), centres_all, set(), p3d, max_res_px=5.0) == 1, \
        'excluding the seeding pair (0, 1), only camera 2 remains and it corroborates'

    # Camera 2's detection moved far away: no corroboration left.
    centres_far = [proj[0].clone(), proj[1].clone(), proj[2] + 1000.0]
    assert _support_count(cg, (0, 1), centres_far, set(), p3d, max_res_px=5.0) == 0

    # Camera 2's detection is still at the true point, but already `used` by an earlier group --
    # an already-claimed detection must not corroborate a second candidate.
    assert _support_count(cg, (0, 1), centres_all, {(2, 0)}, p3d, max_res_px=5.0) == 0


def test_a_nan_box_is_skipped_by_both_cross_view_paths():
    """A NaN box (an anchor in the letterbox padding band) must be skipped by `associate`, the
    tracker, and `min_views = 1`."""

    from tailcyclenet.infer.associate import associate
    from tailcyclenet.infer.track import CrossViewTracker, _project

    cg = _flat_rig()

    world = np.array([[0.0, 0.0, 0.0]], np.float32)
    nan_row = torch.full((1, 4), float('nan'))
    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, world)
        good = torch.stack([uv[:, 0] - 20, uv[:, 1] - 20, uv[:, 0] + 20, uv[:, 1] + 20], -1)
        per_cam.append(torch.cat([good, nan_row]))       # one real animal, one dead box
        scores.append(torch.ones(2))

    got = associate(cg, per_cam, max_res_px=20.0)
    assert len(got) == 1, 'the real animal must still be found beside a NaN box'
    assert bool(torch.isfinite(got[0]['point']).all())
    # The NaN box is not silently adopted as a member of it.
    assert all(j == 0 for j in got[0]['members'].values())

    tr = CrossViewTracker(2, max_res_px=20.0)
    boxes, _, _ = tr.step(cg, per_cam, scores)
    assert np.isfinite(boxes[0]).all(), 'the tracked animal must come back with real boxes'

    # And `min_views = 1` does not emit the dead box as a single-view instance.
    solo = associate(cg, per_cam, max_res_px=20.0, min_views=1)
    for g in solo:
        for c, b in g['boxes'].items():
            assert torch.isfinite(b).all(), f'camera {c} emitted a NaN box as an instance'


def test_measured_tracker_configuration_is_the_default():
    """The measured zero-switch configuration is what the constructor ships."""
    import inspect

    from tailcyclenet.infer.track import CrossViewTracker

    sig = inspect.signature(CrossViewTracker.__init__).parameters
    assert sig['max_age'].default == 8 and sig['max_move'].default == 1.25


# --- link_rows / associate_group ----------------------------------------------------------


def test_link_rows_follows_one_animal():
    """Unlinked rows are score-ordered, so the window-union crop (what `run_group` crops to)
    spans several animals; linking must undo a per-frame swap."""
    from tailcyclenet.infer.associate import link_rows

    T = 20
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        a = np.array([10 + t, 10, 40 + t, 40], np.float32)      # drifts right
        b = np.array([110 - t, 10, 140 - t, 40], np.float32)    # drifts left
        boxes[0, t, 0], boxes[1, t, 0] = (a, b) if t % 2 == 0 else (b, a)

    union = lambda x: np.concatenate([x[..., :2].min(0), x[..., 2:].max(0)], -1)  # noqa: E731
    before = union(boxes[0, :, 0])
    linked = link_rows(boxes.copy())
    after = union(linked[0, :, 0])

    assert (before[2] - before[0]) > 120, 'the swapped rows should span the whole frame'
    assert (after[2] - after[0]) < (before[2] - before[0]) / 2, \
        f'linking must shrink the union crop, got {after} from {before}'
    # Every frame still holds both animals -- linking reorders, it never drops.
    assert np.isfinite(linked).all()


def test_link_rows_gates_on_size_and_never_force_assigns():
    """A row that matches nothing stays EMPTY (`free.pop(0)` once handed it a teleporting box),
    a jump of more than one box side is not motion, and a one-frame miss does not break the
    chain (matching is against the LAST KNOWN box)."""
    from tailcyclenet.infer.associate import link_rows

    boxes = np.full((2, 6, 1, 4), np.nan, np.float32)
    boxes[0, :, 0] = [10, 10, 40, 40]                          # a stationary animal
    boxes[1, 3, 0] = [900, 900, 930, 930]                      # one far-away detection, once
    linked = link_rows(boxes.copy())
    assert np.allclose(linked[0, :, 0, 0], 10), 'the tracked row must not move'

    jump = np.full((1, 2, 1, 4), np.nan, np.float32)
    jump[0, 0, 0] = [0, 0, 30, 30]
    jump[0, 1, 0] = [300, 0, 330, 30]                          # 10 box sides away
    assert not np.isfinite(link_rows(jump.copy())[0, 1, 0]).all()
    jump[0, 1, 0] = [20, 0, 50, 30]                            # 0.67 of a side: ordinary motion
    assert np.isfinite(link_rows(jump)[0, 1, 0]).all()

    gap = np.full((2, 4, 1, 4), np.nan, np.float32)
    for t in range(4):
        gap[0, t, 0] = [10 + t, 10, 40 + t, 40]
        gap[1, t, 0] = [110, 10, 140, 40]
    gap[:, 2, 0] = np.nan                                      # the detector sees nothing at t=2
    linked = link_rows(gap.copy())
    assert linked[0, 3, 0][0] == 13 and linked[1, 3, 0][0] == 110


def test_link_rows_prefers_the_nearer_box_where_iou_prefers_the_wrong_one():
    """IoU ranks by shape agreement, which is not identity: a true continuation that shrank loses
    to a size-matched wrong animal. Centre distance is not fooled.
    """
    from tailcyclenet.infer.associate import link_rows

    last = np.array([100.0, 100.0, 300.0, 300.0], np.float32)     # remembered, 200 px
    near = np.array([170.0, 170.0, 230.0, 230.0], np.float32)      # same centre, 60 px
    wide = np.array([150.0, 150.0, 350.0, 350.0], np.float32)      # 0.35 sides away, 200 px
    iou = box_iou(torch.as_tensor(last[None]), torch.as_tensor(np.stack([near, wide])))[0]
    assert iou[1] > iou[0], f'the fixture must be one IoU gets wrong, got {iou.tolist()}'

    boxes = np.stack([np.stack([last, near]),
                      np.stack([np.full(4, np.nan, np.float32), wide])])[:, :, None]
    linked = link_rows(boxes.astype(np.float32).copy())
    assert np.allclose(linked[0, 1, 0], near), \
        f'centre distance must keep the nearer box, got {linked[0, 1, 0]}'
    # ...and the other detection is a BIRTH into the empty row, not a dropped animal.
    assert np.allclose(linked[1, 1, 0], wide)


def test_associate_group_state_carries_across_a_split(tmp_path):
    """Blocks are sized by the RAM budget, so N calls with `state=` must equal ONE call byte
    for byte, in both branches (tracker C > 1, `link_rows` C == 1); the scene makes a stateless
    split FAIL."""

    from tailcyclenet.infer.associate import associate_group
    from tailcyclenet.format import Session

    T = 12
    for tag, cams, track in (('mv', 3, True), ('sv', 1, False)):
        d = tmp_path / tag / 'test' / 's'
        (cf._session_3d if cams > 1 else cf._session_2d)(d, T=T)
        sess = Session.load(d)
        sess.preload()

        D, C, S = 4, cams, 2
        box = np.full((D, T, C, 4), np.nan, np.float32)
        sc = np.full((D, T, C), np.nan, np.float32)
        for t in range(T):
            for c in range(C):
                for a in range(S):
                    # Two animals drifting apart inside the fixture's 64x48 frame. What makes each
                    # branch discriminating differs: the tracker seats a vanished row's survivor
                    # differently; link_rows needs the DETECTION ORDER to alternate.
                    if C > 1 and a == 1 and 4 <= t < 7:
                        continue
                    d = a if C > 1 else (a + t) % 2
                    x = 4.0 + 24.0 * a + 0.5 * t
                    box[d, t, c] = [x, 12, x + 12, 32]
                    sc[d, t, c] = 0.9 - 0.1 * d
        raw = (box, sc)

        def run(state):
            parts = []
            for lo, hi in ((0, 5), (5, T)):
                sub = (box[:, lo:hi].copy(), sc[:, lo:hi].copy())
                parts.append(associate_group(sub, sess, 'g000', S, link=not track, track=track,
                                             **({} if state is None else {'state': state})))
            return [np.concatenate([p[i] for p in parts], axis=1) for i in range(2)]

        whole = associate_group(raw, sess, 'g000', S, link=not track, track=track)
        joined = run({})
        for i, name in enumerate(('boxes', 'scores')):
            np.testing.assert_array_equal(
                np.nan_to_num(joined[i], nan=-9e9), np.nan_to_num(whole[i], nan=-9e9),
                err_msg=f'{name} differ at C={cams}, track={track}: a block boundary changed the '
                        'association, so the RAM budget can change the answer')

        # THE TEST MUST BE ABLE TO FAIL. Without the carry, this scene disagrees -- if it stops
        # disagreeing, the scene has gone back to being one a fresh matcher reproduces by luck
        # and the assertion above is vacuous.
        assert not np.array_equal(np.nan_to_num(run(None)[0], nan=-9e9),
                                  np.nan_to_num(whole[0], nan=-9e9)), \
            f'C={cams}, track={track}: a STATELESS split reproduces the whole clip here, so this ' \
            'scene does not exercise the carry at all'


def test_associate_group_threads_max_age_and_frame_base(tmp_path, monkeypatch):
    """`--max-age` reaches `CrossViewTracker` (C > 1) and `link_rows` (C == 1), and a ranged run's
    events are SOURCE frames (`frame_base` added back, A6a)."""
    import tailcyclenet.infer.associate as detmod
    from tailcyclenet.format import Session

    T = 4
    d = tmp_path / 'test' / 's'
    cf._session_3d(d, T=T)
    sess = Session.load(d)
    sess.preload()
    cg = sess.cgroup('g000')
    per_cam, scores = _lever_boxes(cg, [np.array([0.0, 0.0, 0.0])], side=100.0)
    box = np.tile(np.stack([b[0].numpy() for b in per_cam])[None, None], (1, T, 1, 1))
    sc = np.tile(np.stack([s[0].numpy() for s in scores])[None, None], (1, T, 1))
    raw = (box.astype(np.float32), sc.astype(np.float32))

    state = {}
    detmod.associate_group(raw, sess, 'g000', 1, track=True, max_age=5, state=state)
    assert state['tracker'].max_age == 5
    assert state['tracker'].events[0]['frame'] == 0
    ranged = {}
    detmod.associate_group(raw, sess, 'g000', 1, track=True, state=ranged, frame_base=32)
    assert ranged['tracker'].events[0]['frame'] == 32

    seen = {}

    def spy_link_rows(boxes, scores=None, **kw):
        seen.update(kw)
        return boxes, scores

    monkeypatch.setattr(detmod, 'link_rows', spy_link_rows)

    class _FakeSession:
        assoc_res_max_px = 30.0
        rig = type('Rig', (), {'moving': {}})()

        def cgroup(self, gid, t=None):
            return []

    nan = (np.full((1, 2, 1, 4), np.nan, np.float32), np.full((1, 2, 1), np.nan, np.float32))
    detmod.associate_group(nan, _FakeSession(), 'g000', 1, link=True, track=False, max_age=7)
    assert seen.get('max_age') == 7


def test_moving_rig_associate_group_fetches_extrinsics_at_source_frames(tmp_path):
    """D1.1: a moving rig's extrinsics are fetched at SOURCE frames (`frame_base` + local), not
    the batch-local index."""

    from tailcyclenet.infer.associate import associate_group
    from tailcyclenet.format import Session

    class _RecordingSession:
        """Delegate to a real Session but record every `cgroup` frame argument."""

        def __init__(self, real):
            self._real = real
            self.frames = []

        def __getattr__(self, name):
            return getattr(self._real, name)

        def cgroup(self, gid, frames=None):
            self.frames.append(frames)
            return self._real.cgroup(gid, frames)

    d = tmp_path / 'test' / 's'
    cf._session_3d(d, T=12, moving=True)
    sess = Session.load(d)

    T, C = 4, 3
    raw = (np.full((1, T, C, 4), np.nan, np.float32),
           np.full((1, T, C), np.nan, np.float32))

    whole = _RecordingSession(sess)
    associate_group(raw, whole, 'g000', 1, track=True)
    assert whole.frames == [0, 1, 2, 3], \
        'frame_base=0 (the default) must keep the whole-clip behaviour byte-identical'

    ranged = _RecordingSession(sess)
    associate_group(raw, ranged, 'g000', 1, track=True, frame_base=6)
    assert ranged.frames == [6, 7, 8, 9], \
        "a moving rig's extrinsics must be fetched at SOURCE frames, not batch-local ones"


# --- CLI defaults and provenance ----------------------------------------------------------


def test_the_tracker_is_the_default_and_help_renders(capsys):
    """`--track` is ON by default with `--no-track` to restore the memoryless pass; `--help`
    renders for both CLIs (a bare `%` in a help string kills argparse with a format TypeError)."""
    import inspect

    from tailcyclenet.infer.associate import associate_group
    from tailcyclenet.infer.cli import build_parser
    from tailcyclenet.detector.train import main as train_main

    sig = inspect.signature(associate_group)
    assert sig.parameters['track'].default is True
    assert sig.parameters['link'].default is False, 'link is opt-in at the function level'
    ap = build_parser()
    assert ap.get_default('track') is True
    opts = {s for a in ap._actions for s in a.option_strings}
    assert '--no-track' in opts and '--det-cache' not in opts
    assert '--det-score' in ap.format_help()

    with pytest.raises(SystemExit) as e:
        train_main(['--help'])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert all(f in out for f in ('--config', '--out', '--iters', '--device'))


def test_provenance_records_every_box_affecting_option():
    """Every `detect_raw` parameter that changes a detection is recorded, at every value
    (conditional membership is what makes a record lie)."""
    import argparse
    import inspect

    from tailcyclenet.infer.detect import detect_raw
    from tailcyclenet.infer.provenance import box_provenance

    args = argparse.Namespace(detector=None, det_score=0.5, det_top_k=0,
                              max_animals=0, max_frames=0, frame_start=0, frame_stop=0,
                              det_nms_iou=0.5, det_nms_center_dist=None)
    prov = box_provenance(args, None, False, None)

    # Everything `detect_raw` takes that can change the detections. The rest are plumbing --
    # `batch` is in there under protest: it is NOT inert but it is pinned rather
    # than exposed, so no run can differ in it. `input_wh` is the checkpoint's own, so
    # `detector_checkpoint` records it. `frames`/`read` are the block loop's plumbing and
    # change no pixel. See tests/test_memory_budget.py.
    plumbing = {'det', 'session', 'gid', 'device', 'batch', 'frames', 'read', 'trace',
                'trace_detail', 'embed_out', 'input_wh'}
    params = set(inspect.signature(detect_raw).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'score_thresh': 'det_score', 'top_k': 'det_top_k',
             'iou_thresh': 'det_nms_iou', 'center_dist_thresh': 'det_nms_center_dist',
             'box_source': 'det_box_source'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change the detections and are not recorded in the prediction: {missing}. Two runs '
        'differing in one of them would be indistinguishable from their output alone.')

    # UNCONDITIONAL, every key, always. The stamp recorded only the options that DIFFERED from
    # their defaults, because a positional list invalidated every cache on disk each time a flag
    # was added. Nothing here has that pressure, and conditional membership is what made the stamp
    # need five exceptions to its own rule.
    args2 = argparse.Namespace(detector='d.pt', det_score=0.97,
                               det_top_k=24, max_animals=2, max_frames=120,
                               frame_start=300, frame_stop=500, det_nms_iou=0.65,
                               det_nms_center_dist=0.25)
    assert set(box_provenance(args2, 1.0, True, 'instances')) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'


def test_every_identity_lever_is_recorded_in_the_prediction():
    """Every `associate_group` parameter that changes which row a detection lands in is
    recorded, unconditionally: two stored 2D arms once differed only by directory name."""
    import inspect

    from tailcyclenet.infer.associate import associate_group
    from tailcyclenet.infer.provenance import identity_provenance

    from tailcyclenet.infer.cli import build_parser

    args = build_parser().parse_args(['--run', 'r', '--out', 'o'])
    prov = identity_provenance(args)

    # Everything `associate_group` takes that can change which row a detection lands in. The rest
    # is plumbing: `raw`/`session`/`gid` are the inputs themselves, `max_instances` is already
    # recorded as `max_animals` by `box_provenance`, `stats` is a diagnostic sink, `state` is
    # the block-boundary carry (recorded nowhere because it is derived, not chosen), and
    # `frame_base` is derived from `--start-frame`, already recorded as `frame_start` by
    # `box_provenance` -- it does not choose which row a detection lands in, only which
    # SOURCE frame the tracker's own step 0 is (A6a).
    plumbing = {'raw', 'session', 'gid', 'max_instances', 'stats', 'state', 'frame_base'}
    params = set(inspect.signature(associate_group).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'link': 'link_boxes'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change which row a detection lands in and are not recorded in the prediction: '
        f'{missing}. Two runs differing in one of them would be indistinguishable from their '
        'output alone, which is exactly how the stored 2D suppression arms became unmatchable.')

    # UNCONDITIONAL, every key, always -- same rule as `box_provenance`: conditional membership
    # is what makes a record lie, because an absent key reads as "not used" and not as "unknown".
    args2 = build_parser().parse_args([
        '--run', 'r', '--out', 'o', '--no-track', '--no-link-boxes', '--min-views', '1',
        '--max-move', '2.0', '--max-age', '24', '--duplicate-radius', '0.9',
        '--duplicate-persist', '8', '--identity-bridge'])
    assert set(identity_provenance(args2)) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'
    assert identity_provenance(args2)['max_age'] == 24


def test_detector_run_directory_resolves_latest_last_and_best(tmp_path):
    """A run folder resolves to its highest-numbered checkpoint by default (an incomplete run
    too); `last` still demands `detector_last.pth`; `best` is `detector.pth`."""
    from tailcyclenet.detector.checkpoint import load_detector, resolve_detector_checkpoint
    from tailcyclenet.infer.cli import build_parser

    run = tmp_path / 'run'
    run.mkdir()
    torch.save({'iteration': 2}, run / 'detector_it000002.pth')
    (run / 'config.toml').write_text('[training]\niters = 4\n')
    with pytest.raises(ValueError, match='detector_last.pth'):
        resolve_detector_checkpoint(run, checkpoint='last')
    (run / 'metrics.json').write_text('[{"iteration": 2}]')
    torch.save({'iteration': 2}, run / 'detector_last.pth')
    assert resolve_detector_checkpoint(run, checkpoint='latest') == run / 'detector_it000002.pth'
    assert resolve_detector_checkpoint(run, checkpoint='detector_it000002.pth') == \
        run / 'detector_it000002.pth'

    torch.save({'iteration': 4}, run / 'detector_it000004.pth')
    torch.save({'iteration': 2}, run / 'detector.pth')
    torch.save({'iteration': 4}, run / 'detector_last.pth')
    (run / 'metrics.json').write_text('[{"iteration": 2}, {"iteration": 4}]')
    assert resolve_detector_checkpoint(run) == run / 'detector_it000004.pth'
    assert resolve_detector_checkpoint(run, checkpoint='last') == run / 'detector_last.pth'
    assert resolve_detector_checkpoint(run, checkpoint='best') == run / 'detector.pth'

    args = build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o'])
    assert args.detector_checkpoint == 'latest'
    assert load_detector.__defaults__[-1] == 'latest'


# --- scoring --------------------------------------------------------------------------------


def test_summarise_and_overall_report_fp_dup_and_fp_none_as_rates():
    """D1: `box_mota`'s fp_dup/fp_none survive into the per-group summary as per-gt rates (NaN,
    not 0.0, where nothing was scored) and `overall` weights them by n_gt, skipping NaN groups."""
    from tailcyclenet.detector.evaluate import _summarise, overall

    s = {'n_gt': 4, 'hit50': 3, 'hit75': 2, 'iou': 3.0, 'fp': 1,
         'mota': [{'mota': 0.5, 'gt': 4, 'fp_ignored': 1, 'fp_dup': 1, 'fp_none': 2, 'misses': 3}]}
    r = _summarise(s)
    assert r['fp_ignored'] == 1
    assert (r['fp_dup'], r['fp_none'], r['miss']) == pytest.approx((1 / 4, 2 / 4, 3 / 4))
    r = _summarise({'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0, 'fp': 0, 'mota': []})
    assert np.isnan(r['fp_dup']) and np.isnan(r['fp_none']) and np.isnan(r['miss'])

    row = {'r50': 1, 'r75': 1, 'iou': 1, 'fp': 0, 'mota': 1.0}
    o = overall({'a': {**row, 'n_gt': 2, 'fp_ignored': 0, 'fp_dup': 0.5, 'fp_none': 0.0,
                       'miss': 0.25},
                 'b': {**row, 'n_gt': 6, 'fp_ignored': 2, 'fp_dup': 0.0, 'fp_none': 1.0,
                       'miss': 0.0},
                 'c': {**row, 'n_gt': 2, 'fp_ignored': 0, 'fp_dup': float('nan'),
                       'fp_none': float('nan'), 'miss': float('nan')}})
    assert o['fp_ignored'] == 2
    assert o['fp_dup'] == pytest.approx((2 * 0.5) / 8)
    assert o['fp_none'] == pytest.approx((6 * 1.0) / 8)
    assert o['miss'] == pytest.approx((2 * 0.25) / 8)


def test_box_mota_splits_fp_into_dup_and_none_and_counts_misses():
    from tailcyclenet.detector.evaluate import box_mota

    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0],             # exact match -> TP
                         [1.0, 1.0, 11.0, 11.0],             # near the claimed GT -> dup
                         [1000.0, 1000.0, 1010.0, 1010.0]])  # far away -> none
    r = box_mota({0: (pred, gt, None, None)})
    assert r['fp'] == 2 and r['fp_dup'] == 1 and r['fp_none'] == 1 and r['misses'] == 0

    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1000.0, 1000.0, 1010.0, 1010.0]])
    r = box_mota({0: (pred[:1], gt, None, None)})
    assert r['misses'] == 1 and r['fp'] == 0 and r['fp_dup'] == 0 and r['fp_none'] == 0


def test_deployment_score_reads_zero_untrained_and_one_when_forced(tmp_path):
    """Seeded fresh model: the rare-positive objectness prior sits below `det_score`, so fills
    read exactly 0 (union sides NaN); forcing every objectness logit positive fills every slot --
    the zero is the prior firing, not a broken wire."""
    torch.manual_seed(0)
    from tailcyclenet.detector.diagnostics import gt_crop_sides
    from tailcyclenet.evaluation.deployment import deployment_score
    from tailcyclenet.format import Session

    cf._session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    model = YOLOXNano(version='hybrid')
    kw = dict(input_wh=(64, 64), top_k=8, max_animals=2, n_frames=2, overlap=0, min_box_frames=1)
    r = deployment_score(model, sess, gid, det_score=0.5, **kw)
    assert r['det_fill'] == 0.0 and r['slot_fill'] == 0.0 and r['window_miss'] == 1.0
    assert r['n_windows'] == 4 and r['n_gt'] == 8
    assert all(v != v for v in r['union_side_px'].values())
    assert r['gt_side_px'][0.5] > 0

    for m in model.head.obj_pred:
        torch.nn.init.constant_(m.bias, 10.0)
    r = deployment_score(model, sess, gid, det_score=0.05, **kw)
    assert r['det_fill'] == 1.0
    assert not any(v != v for v in r['union_side_px'].values())

    # 3D: `gt_crop_sides` projects points3d per camera (a 3D session has no points2d).
    cf._session_3d(tmp_path / 'r3' / 'test' / 's0', T=4)
    s3 = Session.load(tmp_path / 'r3' / 'test' / 's0')
    s3.preload()
    sides = gt_crop_sides(s3, list(s3.groups)[0], min_crop_dim=8)
    assert sides.size > 0 and np.all(sides >= 8)


def test_eval_detector_deploy_prints_the_scored_t_under_det_max_frames(tmp_path, monkeypatch,
                                                                       capsys):
    """`eval_detector.py --deploy` runs end to end on a saved checkpoint, and its T column is the
    `--det-max-frames` prefix actually scored, not the group's raw length (a 3dpop run printed
    T=899..2680 while scoring 120)."""
    from scripts import eval_detector

    root = tmp_path / 'root'
    cf._session_2d(root / 'test' / 's1', T=4, S=2)
    ckpt = tmp_path / 'detector.pth'
    torch.save(cf._detector_checkpoint(input_wh=[64, 64], min_crop_dim=8,
                                       box_source='keypoints'), ckpt)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(ckpt), '--data', str(root), '--split', 'test',
        '--deploy', '--no-track', '--link-boxes', '--score-thresh', '0.001', '--device', 'cpu',
        '--n-frames', '2', '--overlap', '0', '--det-max-frames', '2'])
    eval_detector.main()
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if '/g' in ln and ln.strip()[0] != '=')
    assert line.split()[1] == '2', f'expected the truncated T=2 in {line!r}'


# --- model and checkpoint ----------------------------------------------------------------


def test_norm_groups_always_divides_the_channel_count():
    """A GroupNorm whose count does not divide its channels is a constructor error."""
    from tailcyclenet.detector.yolox import norm_groups
    for c in list(range(1, 200)) + [24, 48, 96, 192]:
        g = norm_groups(c)
        assert 1 <= g <= c and c % g == 0


def test_groupnorm_has_no_running_statistics_and_no_batch_dependence():
    """Train and inference are the SAME computation (BatchNorm's statistics are what would make
    train-on-tiles / infer-on-frames unsafe), and one item's output ignores the rest of the batch
    (so a resolution sweep that changes the batch measures resolution, not normalisation)."""
    torch.manual_seed(0)
    m = YOLOXNano(version='hybrid')
    assert list(m.named_buffers()) == []
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())
    x = torch.rand(3, 3, 96, 128)
    m.train()
    a = m(x)[0]
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(a, m(x)[0])
        torch.testing.assert_close(m(x[1:2])[0], m(x)[0][1:2], rtol=1e-4, atol=1e-5)


def test_load_detector_round_trips_and_refuses_by_name(tmp_path):
    """The checkpoint's own `yolox_version`/`p2`/`obj_quantiles` decide what loads (an unrecorded
    quantile set is `{}`, not a guess); a BatchNorm, a version-less (`trimmed`) or a pose
    checkpoint is refused by name."""
    from tailcyclenet.detector.checkpoint import load_detector

    with pytest.raises(ValueError, match='hybrid'):
        YOLOXNano(version='medium')
    p = tmp_path / 'detector.pth'
    m = YOLOXNano(version='hybrid', p2=True)
    base = cf._detector_checkpoint(m, input_wh=(96, 96))
    torch.save(base, p)
    ld = load_detector(p)
    assert ld.model.version == 'hybrid' and ld.model.p2 is True
    assert ld.model.STRIDES == (4, 8, 16, 32)
    torch.testing.assert_close(ld.model.state_dict()['head.obj_pred.0.bias'],
                               m.state_dict()['head.obj_pred.0.bias'])
    assert ld.obj_quantiles == {}
    q = {'q01': 0.452, 'q10': 0.601, 'q50': 0.883, 'q90': 0.981}
    torch.save({**base, 'obj_quantiles': q}, p)
    assert load_detector(p).obj_quantiles == q

    refused = {'bn normalisation': {'model_state': {}, 'input_wh': [128, 128],
                                    'yolox_version': 'hybrid'},
               "'trimmed' is a deleted architecture": {k: v for k, v in base.items()
                                                        if k != 'yolox_version'},
               "'pose' checkpoint, not a detector": {'kind': 'pose', 'model_state': {}}}
    for match, ckpt in refused.items():
        torch.save(ckpt, p)
        with pytest.raises(ValueError, match=match):
            load_detector(p)


def test_p2_true_adds_a_stride_4_level():
    """`p2=True` adds a stride-4 level whose anchors match the forward's flattening exactly."""
    base = YOLOXNano(version='hybrid')
    m = YOLOXNano(version='hybrid', p2=True)
    assert m.STRIDES == (4, 8, 16, 32)
    assert base.STRIDES == (8, 16, 32), 'the OTHER instance must be unaffected'
    x = torch.zeros(1, 3, 128, 160)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0]
    assert set(anchors[:, 2].tolist()) == {4.0, 8.0, 16.0, 32.0}
    n_base = sum(p.numel() for p in base.parameters())
    n_p2 = sum(p.numel() for p in m.parameters())
    assert n_p2 > n_base


def test_hybrid_backbone_shape_contract_and_finite_grads():
    from tailcyclenet.detector.hybrid import HybridBackbone

    bb = HybridBackbone(p2=True)
    H, W = 128, 192
    feats = bb(torch.rand(1, 3, H, W))
    assert len(feats) == len(bb.out_channels) == 4
    for f, s, c in zip(feats, (4, 8, 16, 32), bb.out_channels):
        assert f.shape == (1, c, H // s, W // s)
    sum(f.float().sum() for f in feats).backward()
    for name, p in bb.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), f'{name}: non-finite grad'


def test_gap_levers_build_the_head_and_neck_they_name():
    """`shared_head=False` adds a separate objectness tower; `fpn_upsample` defaults to nearest at
    the constructor (the shipped config sets bilinear)."""
    shared = YOLOXNano(version='hybrid', shared_head=True)
    unshared = YOLOXNano(version='hybrid', shared_head=False)
    assert not hasattr(shared.head, 'obj_convs') and hasattr(unshared.head, 'obj_convs')
    assert sum(p.numel() for p in unshared.parameters()) > \
        sum(p.numel() for p in shared.parameters())
    assert shared.neck.fpn_upsample == 'nearest'
    model = YOLOXNano(version='hybrid', fpn_upsample='bilinear')
    obj, boxes = model(torch.rand(1, 3, 64, 64))
    assert torch.isfinite(obj).all() and torch.isfinite(boxes).all()


def test_detector_packaging_preserves_config(tmp_path):
    from scripts.package_checkpoint import package_detector
    from tailcyclenet.detector.checkpoint import load_detector

    run = tmp_path / 'run'
    run.mkdir()
    config = {'data': {'input_wh': [96, 96]}, 'model': {'yolox': 'hybrid'},
              'training': {'iters': 2, 'seed': 7}}
    (run / 'config.toml').write_text('[training]\niters = 2\n')
    model = YOLOXNano(version='hybrid')
    source = {'iteration': 2, 'model_state': model.state_dict(), 'input_wh': [96, 96],
              'norm': 'gn', 'yolox_version': 'hybrid', 'p2': False, 'config': config}
    torch.save(source, run / 'detector_last.pth')
    torch.save(source, run / 'detector_it000002.pth')
    (run / 'metrics.json').write_text('[{"iteration": 2}]')
    out = tmp_path / 'detector.pth'
    package_detector(run, out)
    _ld = load_detector(out)
    loaded, wh = _ld.model, _ld.input_wh
    packaged = torch.load(out, map_location='cpu', weights_only=False)
    assert wh == (96, 96)
    assert packaged['config'] == config
    assert packaged['kind'] == 'detector'
    assert loaded.version == 'hybrid'


def test_build_detector_optimizer_adamw_matches_param_count():
    from tailcyclenet.detector.train import build_detector_optimizer

    model = YOLOXNano(version='hybrid')
    opt, sched = build_detector_optimizer(
        model, {'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 5e-4, 'iters': 10},
        {'pretrained': ''})
    assert sum(p.numel() for g in opt.param_groups for p in g['params']) == \
        sum(p.numel() for p in model.parameters())
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


# --- train_detector end to end -----------------------------------------------------------


def test_train_detector_end_to_end_wires_every_lever(tmp_path, dense_root, monkeypatch):
    """A 2-iteration in-process run with the non-default levers on: every config value reaches
    `detector_loss` (box_weight was once a dead key; `max_pos_per_gt = 0` means uncapped), the run
    folder holds the artefacts, and the checkpoint is self-contained and reloads its architecture."""
    import tomllib

    import tailcyclenet.detector.train as td
    from tailcyclenet.detector.checkpoint import load_detector

    seen = []
    real_loss = td.detector_loss

    def spy(*a, **kw):
        seen.append(kw)
        return real_loss(*a, **kw)

    monkeypatch.setattr(td, 'detector_loss', spy)
    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
tile_wh = []
[model]
yolox = "hybrid"
p2 = true
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
box_weight = 3.0
max_pos_per_gt = 0
iou_aware_obj = true
iou_aware_warmup = 0
shared_head = false
fpn_upsample = "bilinear"
""")
    td.main(['--config', str(cfg)])
    assert len(seen) == 2
    kw = seen[0]
    assert kw['box_weight'] == 3.0 and kw['max_pos_per_gt'] is None
    assert kw['iou_aware'] is True and kw['iou_aware_warmup'] == 0
    assert (kw['assignment'], kw['box_loss_fn']) == ('tal', 'ciou')

    for name in ('detector_it000002.pth', 'detector_last.pth', 'detector.pth', 'metrics.json',
                 'config.toml', 'provenance.toml'):
        assert (out / name).exists(), name
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['data']['input_wh'] == [48, 48] and recorded['training']['iters'] == 2
    ld = load_detector(out / 'detector.pth')
    assert tuple(ld.input_wh) == (48, 48) and ld.min_crop_dim == 16
    assert ld.box_source == 'keypoints' and ld.tile_scale is None
    assert ld.model.p2 is True and hasattr(ld.model.head, 'obj_convs')
    assert ld.model.neck.fpn_upsample == 'bilinear'
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['kind'] == 'detector' and ckpt['optimizer_kind'] == 'muon'
    assert ckpt['yolox_version'] == 'hybrid' and ckpt['config'] == recorded
    with open(out / 'provenance.toml', 'rb') as f:
        assert ckpt['provenance'] == tomllib.load(f)

    lone = tmp_path / 'elsewhere' / 'detector.pth'
    lone.parent.mkdir()
    shutil.copy(out / 'detector_it000002.pth', lone)
    model = load_detector(out / 'detector_it000002.pth').model
    shutil.rmtree(out)
    lone_model = load_detector(lone).model
    for k, v in model.state_dict().items():
        assert torch.equal(v, lone_model.state_dict()[k]), k
