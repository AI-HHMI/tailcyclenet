"""The training loop's guards. Not the model, not the loader -- the two branches around them.

A non-finite loss is the one that matters: it is rare, it only fires on data nobody can construct
on demand, and getting it wrong costs a 72-hour cluster job that dies in the first two minutes.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _train_module():
    """Import scripts/train.py without running main()."""
    spec = importlib.util.spec_from_file_location('tcn_train', REPO / 'scripts' / 'train.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch():
    """The attributes `run_batch` touches, and nothing else."""
    return SimpleNamespace(
        views=[torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)],
        cgroup=[{'size': torch.tensor([4, 4])}],
        sample_info={'mode': '3d'},
        kpt_ids=torch.zeros(1, 2, dtype=torch.long),
        kpt_prior=torch.zeros(1, 2, 3),
        prompt_t=torch.zeros(1, 2, dtype=torch.int32),
        coords=torch.zeros(1, 2, 2, 3),
        vis=None, vis_2d=None, p2d=None)


def test_non_finite_loss_becomes_a_skipped_step():
    """posetail RAISES on a poisoned sub-loss; the loop needs a NaN it can skip. The raise is
    right for its own purpose but killed two sweep jobs in their first 100 seconds over an
    intermittent bad step; `run_batch` converts it into the signal the loop's own
    `torch.isfinite(loss)` guard already reads.
    """
    tr = _train_module()
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}   # noqa: E731

    def poisoned(*a, **k):
        raise ValueError("sub-loss(es) went non-finite while attached to the autograd graph: "
                         "['coords_loss', 'smoothness_loss_3d']")

    loss, out = tr.run_batch(model, poisoned, _batch(), 'cpu')
    assert not torch.isfinite(loss), 'a non-finite sub-loss must come back as NaN, not raise'
    assert loss.grad_fn is None, 'the skipped step must carry no graph into the next iteration'
    assert out is not None


def test_other_value_errors_still_propagate():
    """Only the non-finite case is converted. A real bug must not be swallowed as a skipped step:
    a run that silently skips every iteration looks exactly like a run that trained."""
    tr = _train_module()
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}   # noqa: E731

    def broken(*a, **k):
        raise ValueError('shape mismatch: this is a real bug, not a bad step')

    with pytest.raises(ValueError, match='real bug'):
        tr.run_batch(model, broken, _batch(), 'cpu')


def test_short_window_survives_the_smoothness_loss():
    """T = 2 must not raise inside `SmoothnessLoss`, and must not be silently disabled either:
    the loss narrows by `T - k`, so a window shorter than `order + 1` raises on a negative length.
    0.3.5 clamps `order` to `T - 1` inside the loss; this pins that the repo's per-batch rule
    (`_tune_smoothness`) degrades to a first difference at T = 2 and restores the order after.
    """
    from posetail.posetail.losses import TotalLoss

    mod = _train_module()
    loss_fn = TotalLoss(smoothness_loss_3d_weight=0.5, smoothness_loss_2d_weight=0.5,
                        smoothness_loss_order=4)
    assert loss_fn.smoothness_loss_3d.order == 4

    # The library's own clamp makes the raw call safe at T = 2 (0.3.5); assert the forward runs
    # rather than the old pre-0.3.5 raise.
    pred = torch.zeros(1, 2, 3, 3)
    assert torch.isfinite(loss_fn.smoothness_loss_3d(pred, pred, torch.ones(1, 2, 3, 1),
                                                     time_dim=1))

    # ...and `_tune_smoothness` applies the same rule per batch: degraded to a first difference,
    # not disabled.
    mod._tune_smoothness(loss_fn, 2)
    assert loss_fn.smoothness_loss_3d.order == 1
    assert torch.isfinite(loss_fn.smoothness_loss_3d(pred, pred, torch.ones(1, 2, 3, 1),
                                                     time_dim=1))

    # A full-length window gets the configured order back -- the clamp is per batch, not sticky.
    mod._tune_smoothness(loss_fn, 24)
    assert loss_fn.smoothness_loss_3d.order == 4


def test_stride_is_divided_out_of_the_smoothness_weight():
    """`torch.diff` is UNDIVIDED, so a stride-s window's k-th difference is s^k times a stride-1
    one and the term's effective weight would ride on a per-item draw. Checked against the loss's
    own output (a pure ramp reads s times larger at stride s), and idempotent -- re-derived from
    the configured weight each call.
    """
    from posetail.posetail.losses import TotalLoss

    mod = _train_module()
    loss_fn = TotalLoss(smoothness_loss_3d_weight=0.5, smoothness_loss_2d_weight=0.5,
                        smoothness_loss_order=1)
    vis = torch.ones(1, 4, 3, 1)

    def excess(stride):
        mod._tune_smoothness(loss_fn, 4, stride)
        t = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1) * stride
        pred = t.expand(1, 4, 3, 3).contiguous()           # a ramp at this stride's rate
        return float(loss_fn.smoothness_loss_3d(pred, torch.zeros_like(pred), vis, time_dim=1))

    base = excess(1)
    assert base > 0, 'the probe must actually reach the hinge, or the test proves nothing'
    for s in (2, 4):
        assert abs(excess(s) - base) < 1e-6, f'stride {s} changed the term by {excess(s) / base:.1f}x'

    # And it is idempotent -- re-derived from the configured weight each call, never compounded.
    mod._tune_smoothness(loss_fn, 4, 1)
    assert loss_fn.smoothness_loss_3d.weight == 0.5


def test_run_batch_routes_2d_visibility_on_its_own_wire():
    """`batch.vis_2d` means a DIFFERENT thing depending on mode: in 3D it is posetail's own
    per-camera term (`vis_true_cams`, both-or-neither with `vis_true`); in 2D it must reach
    `PoseLoss`'s own `vis_2d_true` keyword, which `TotalLoss.forward` does not have and never sees.
    """
    tr = _train_module()
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 2),   # noqa: E731
                             'vis_pred_2d': torch.zeros(1, 1, 2, 2)}

    seen = []

    def spy(model_, out, coords_true, vis_true, vis_true_cams, vis_2d_true=None, **kw):
        seen.append(dict(vis_true=vis_true, vis_true_cams=vis_true_cams,
                         vis_2d_true=vis_2d_true))
        return torch.tensor(0.0)

    batch_2d = SimpleNamespace(
        views=[torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)],
        cgroup=[{'size': torch.tensor([4, 4])}],
        sample_info={'mode': '2d'},
        kpt_ids=torch.zeros(1, 2, dtype=torch.long),
        kpt_prior=torch.zeros(1, 2, 2),
        prompt_t=torch.zeros(1, 2, dtype=torch.int32),
        coords=torch.zeros(1, 2, 2, 2),
        vis=None, vis_2d=torch.ones(1, 2, 2, 1, 1), p2d=torch.zeros(1, 1, 2, 2, 2))
    tr.run_batch(model, spy, batch_2d, 'cpu')
    assert seen[-1]['vis_true'] is None
    assert seen[-1]['vis_true_cams'] is None, '2D must never reach vis_true_cams'
    assert seen[-1]['vis_2d_true'] is not None

    batch_3d = _batch()
    batch_3d.vis_2d = torch.ones(1, 2, 2, 1, 1)          # a real 3D per-camera target this time
    tr.run_batch(model, spy, batch_3d, 'cpu')
    assert seen[-1]['vis_true_cams'] is not None, '3D keeps the library wire unchanged'
    assert seen[-1]['vis_2d_true'] is None, '3D must never reach the 2D-only term'


def test_a_checkpoint_round_trips_enough_to_resume_from(tmp_path):
    """`optimizer_state` was written from day one and read by NOTHING, so a relaunch into the same
    --out restarted at iteration 0 and overwrote both ~5.6 GB files. Pins the part that can be
    silently wrong: the schedule-free `z` iterate survives, and resuming takes the RAW weights
    rather than the averaged eval ones.
    """
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.checkpoints import save_checkpoint
    from tailcyclenet.format import Registry

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
    opt.train()
    # SEVERAL steps: schedule-free's averaged iterate coincides with the raw one at step 1, so a
    # single step cannot tell "resume from the raw weights" from "resume from the eval weights".
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
    config = {'model': {}, 'data': {'image_size': 64}, 'training': {'seed': 23}}
    registry = Registry(names=('nose',), datasets=(('ds', (0,)),))
    save_checkpoint(tmp_path / 'run', 1234, model, opt, config, registry=registry)

    ck = torch.load(tmp_path / 'run' / 'checkpoints' / 'checkpoint_last.pth',
                    map_location='cpu', weights_only=False)
    assert int(ck['iteration']) == 1234, 'the iteration is what a resume restarts at'
    assert ck['config'] == config
    assert ck['keypoint_registry'] == registry.to_dict()
    # The two iterates are genuinely different, or "resume from the raw one" is a distinction
    # without a difference and this test would pass on the wrong weights.
    assert not torch.equal(ck['model_state']['weight'], ck['model_state_eval']['weight'])

    fresh = torch.nn.Linear(4, 4)
    fresh_opt = AdamWScheduleFree(fresh.parameters(), lr=1e-3)
    fresh.load_state_dict(ck['model_state'])
    fresh_opt.load_state_dict(ck['optimizer_state'])
    assert torch.equal(fresh.weight, ck['model_state']['weight'])
    assert fresh_opt.state, 'no optimizer state came back, so the schedule-free z is lost'
    assert any('z' in s for s in fresh_opt.state.values()), \
        'the z iterate is the state that makes this a resume rather than a restart'


# -- config guards -----------------------------------------------------------------------------

KNOWN_TRAINING = {'n_iterations', 'seed', 'checkpoint_path', 'checkpoint_revision', 'max_grad_norm',
                  'checkpoint_freq', 'val_freq', 'val_batches', 'print_freq', 'out', 'optimizer',
                  'losses'}


def test_the_video_encoder_download_is_skipped_only_when_a_checkpoint_will_overwrite_it():
    """`main()`'s first `build_model()` call must skip the VJEPA2 download exactly when a resume
    or a warm-start checkpoint is about to overwrite every tensor it produces; a fresh run (with
    neither) still needs the real pretrained weights, since nothing else supplies them.
    """
    src = (Path(__file__).parent.parent / 'scripts' / 'train.py').read_text()
    assert 'skip_video_encoder_download' in src
    assert 'will_load_full_checkpoint' in src
    assert "resumed.exists() and not args.no_resume" in src
    assert "not args.no_warm_start and train_cfg.get('checkpoint_path')" in src
    assert 'or checkpoint is not None' in src


def test_checkpoint_path_is_a_resume_only_for_full_training_state():
    """`main()` resumes from `[training].checkpoint_path` only when the checkpoint carries the
    full training state -- raw weights + optimizer state + iteration (what `save_checkpoint`
    and the reference repo's train loop write). A packaged pose checkpoint (weights only) can
    never resume; it is a warm start. The predicates `main()` gates on are wired into it."""
    from tailcyclenet.checkpoints import full_training_state

    assert full_training_state({'model_state': {}, 'optimizer_state': {}, 'iteration': 5})
    assert not full_training_state({'model_state': {}, 'iteration': 5}), \
        'weights without optimizer state is a warm start, not a resume'
    assert not full_training_state({'model_state': {}}), 'no iteration means no resume'
    assert not full_training_state({})
    assert not full_training_state(None)

    src = (Path(__file__).parent.parent / 'scripts' / 'train.py').read_text()
    for name in ('full_training_state', 'state_matches_optimizer_kind',
                 'optimizer_layout_matches'):
        assert name in src, f'main() must gate the checkpoint_path resume on {name}'


def test_load_config_layers_over_the_repo_base_by_default(tmp_path):
    """A pose config that names no base still gets `configs/base.toml` as its defaults -- the
    `extends` line is gone, not needed: the overlay IS the whole difference."""
    from tailcyclenet.checkpoints import load_config

    overlay = tmp_path / 'overlay.toml'
    overlay.write_text('[data]\npath = "/some/root"\n')
    cfg = load_config(overlay)
    assert cfg['data']['path'] == '/some/root'
    assert cfg['model']['query'] == 'prior', 'the model defaults must come from base.toml'
    assert cfg['training']['n_iterations'] == 60000
    assert cfg['training']['optimizer']['optimizer'] == 'muon'
    assert cfg['training']['losses']['delta'] == 6


def test_load_config_extends_is_deleted_and_raises(tmp_path):
    """`extends` is deleted from the config language: EVERY config layers over its family's
    base automatically (pose -> `configs/base.toml`, detector -> `configs/detector.toml`), so
    the key is not needed and naming it is refused by name."""
    from tailcyclenet.checkpoints import load_config

    p = tmp_path / 'old.toml'
    p.write_text('extends = "base.toml"\n[data]\npath = "/x"\n')
    with pytest.raises(SystemExit, match='extends'):
        load_config(p)


def test_load_config_layers_over_an_explicit_base(tmp_path):
    """The detector loader passes `configs/detector.toml` as `base`; a config layered over a
    non-pose base must not inherit the pose base's blocks."""
    from tailcyclenet.checkpoints import load_config

    base = tmp_path / 'det_base.toml'
    base.write_text('[data]\nboxes = "instances"\n')
    p = tmp_path / 'overlay.toml'
    p.write_text('[training]\niters = 4\n')
    cfg = load_config(p, base=base)
    assert cfg == {'data': {'boxes': 'instances'}, 'training': {'iters': 4}}


def test_known_training_keys_match_the_guard_in_main():
    """The `[training]` allow-list must stay in step with what `main()` actually reads: a typo'd
    key yields a silent default (no validation, no `checkpoint_best.pth`) -- an arm reporting its
    own control.
    """
    src = (Path(__file__).parent.parent / 'scripts' / 'train.py').read_text()
    assert 'known_training = {' in src, 'the [training] unknown-key guard is gone'
    # every key the guard allows is either read from train_cfg or is a sub-block
    for key in KNOWN_TRAINING - {'optimizer', 'losses', 'out'}:
        assert f"train_cfg.get('{key}'" in src or f"train_cfg['{key}']" in src, \
            f'{key} is allowed by the guard but nothing reads it'


def test_val_is_skipped_only_for_a_missing_split_not_a_config_error():
    """`no val/` may be swallowed; a config error inside `PoseDataset` may NOT. The old bare
    except caught both, so a broken session printed one line and then trained with no validation.
    """
    src = (Path(__file__).parent.parent / 'scripts' / 'train.py').read_text()
    body = src[src.index('    val_ds = None'):src.index('    nw = args.num_workers')]
    assert 'except' not in body, 'val construction must not be wrapped in a bare except again'
    assert "'val').is_dir()" in body, 'the missing-split case must be TESTED for, not caught'


def test_infer_does_not_restate_loader_defaults():
    """`LoaderConfig` owns its defaults; the CLI restating them is how `box_source` diverged.
    Reads `tailcyclenet/infer/driver.py`, where the config resolution moved when inference did.
    """
    src = (Path(__file__).parent.parent / 'tailcyclenet' / 'infer' / 'driver.py').read_text()
    for literal in ("'n_frames', 24", "'image_size', 256", "'min_crop_dim', 64",
                    "'box_source', 'keypoints'"):
        assert literal not in src, f'{literal} restates a LoaderConfig default at the CLI boundary'


def test_det_score_has_one_default():
    """Two defaults meant whichever caller omitted it got a different detector. Asserts against
    the PARSER rather than the file's text, which broke on reformats that changed nothing.
    """
    import inspect

    from tailcyclenet.detector import detect_raw
    from tailcyclenet.infer import build_parser

    sig = inspect.signature(detect_raw).parameters['score_thresh'].default
    cli = build_parser().get_default('det_score')
    assert sig == cli, f'detect_raw defaults to {sig} but the CLI to {cli}'
