"""The `select`/`realise`/`targets` seam of `PoseDataset`, the one the scorer builds on.

Byte-identity of the loader's output is the golden harness's job; this file checks the seam's
structure: each case reaches its branch, selection is deterministic, the stored labels are not
written through, and two realisations of one selection are independent.
"""
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet.dataset import LoaderConfig, PoseDataset


def _same(a, b, path='item'):
    """Recursive element-wise equality, NaN-aware (`torch.equal` is not, and windows hold NaNs)."""
    if a is None or b is None:
        assert a is None and b is None, f'{path}: {a!r} vs {b!r}'
        return
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f'{path}: tensor vs {type(b)}'
        assert a.shape == b.shape, f'{path}: shape {tuple(a.shape)} vs {tuple(b.shape)}'
        assert a.dtype == b.dtype, f'{path}: dtype {a.dtype} vs {b.dtype}'
        if a.dtype.is_floating_point:
            ok = bool(((a == b) | (a.isnan() & b.isnan())).all())
        else:
            ok = torch.equal(a, b)
        assert ok, f'{path}: values differ'
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f'{path}: ndarray vs {type(b)}'
        assert a.shape == b.shape, f'{path}: ndarray shape {a.shape} vs {b.shape}'
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'), \
            f'{path}: ndarray values differ'
    elif isinstance(a, dict):
        assert set(a) == set(b), f'{path}: keys {set(a) ^ set(b)}'
        for k in a:
            _same(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f'{path}: len {len(a)} vs {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{path}[{i}]')
    elif isinstance(a, float):
        assert a == b or (np.isnan(a) and np.isnan(b)), f'{path}: {a} vs {b}'
    else:
        assert a == b, f'{path}: {a!r} vs {b!r}'


def _seed_all(item_seed, shape_seed, aug_seed, rot_seed):
    """Seed every stream the loader draws from: the item Generator, plus the GLOBAL `np.random`
    (`rotate_camera_group`'s world-gauge angle), imgaug's global RNG, and `random`/`torch`."""
    import imgaug.random

    np.random.seed(rot_seed)
    random.seed(rot_seed)
    torch.manual_seed(rot_seed)
    imgaug.random.seed(aug_seed)
    return np.random.default_rng(item_seed), np.random.default_rng(shape_seed)


def _states(rng):
    """A snapshot of every RNG's state, for comparing draw CONSUMPTION."""
    import imgaug.random

    return {
        'item_rng': rng.bit_generator.state,
        'np': np.random.get_state(),
        'python': random.getstate(),
        'torch': torch.random.get_rng_state(),
        # imgaug's own `RNG` wrapper has no `.bit_generator`; `.state` is its numpy array.
        'imgaug': imgaug.random.get_global_rng().state,
    }


def _run(ds, idx, seed=0, shape_seed=0x5AFE, aug_seed=0xA06, rot_seed=0xB07):
    """One `_item` plus the RNG state it left behind."""
    rng, shape_rng = _seed_all(seed, shape_seed, aug_seed, rot_seed)
    shape = ds.shape(shape_rng)
    out = ds._item(idx, rng, shape)
    return out, _states(rng)


# --- the fixture: 2D and 3D, train AND val, plus a moving rig ------------------------------

KPTS_2D = ['nose', 'left_ear', 'right_ear', 'tail_base']
KPTS_3D = ['nose', 'neck', 'tail_base']


@pytest.fixture(scope='session')
def seam_root(tmp_path_factory):
    """2D and 3D, train and val, plus a moving rig (`tiny_root` has no 3D val split)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path_factory.mktemp('seam')
    C._session_2d(root / 'ratlike' / 'train' / 'sess_a')
    C._session_2d(root / 'ratlike' / 'val' / 'sess_b')
    C._session_3d(root / 'mouselike' / 'train' / 'sess_c')
    C._session_3d(root / 'mouselike' / 'val' / 'sess_v')
    C._session_3d(root / 'mouselike' / 'train' / 'sess_moving', moving=True)
    return root


def _L(**over):
    """A LoaderConfig with the fixture's small geometry, unless the case overrides it."""
    return LoaderConfig(**{'n_frames': 4, 'image_size': 64, **over})


# Each case names the DRAW SITES it exists to cover.
_CASES = {
    'plain': _L(prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0, prompt_dropout=0.0),
    # Rotation + jitter + appearance aug + cutout + grayscale.
    'augmented': _L(prob_2d_only=0.0, aug_prob=1.0, crop_jitter=0.3, crop_jitter_scale=0.3,
                    grayscale_prob=0.5, prompt_dropout=0.0),
    # The 3D prompt-noise branch: `px` from `get_camera_scale` (the oracle's repaired import).
    'prompt-noise': _L(prob_2d_only=0.0, aug_prob=0.5, crop_jitter=0.2,
                       prompt_noise_px=3.0, prompt_offset_px=5.0, prompt_dropout=0.5),
    # The stale-prior and keypoint-pair-swap branches.
    'prompt-swaps': _L(prob_2d_only=0.0, aug_prob=0.5, prompt_stale_frames=2,
                       prompt_swap_kpt_pairs=0.5, prompt_swap_animal=0.5, prompt_dropout=0.25),
    # The BOX PROMPT branch -- `_targets`'s box block and `box_prompt`'s module import.
    'box-prompt': _L(prob_2d_only=0.0, aug_prob=0.5, box_prompt='film', box_prompt_dropout=0.5,
                     box_prompt_jitter=0.2, box_prompt_scale_jitter=0.2),
    # The RANGED crop-inflate draw (a per-item uniform draw, not a scalar).
    'ranged-inflate': _L(prob_2d_only=0.0, aug_prob=0.5, crop_inflate=[0.9, 1.5]),
    # Val geometry: jitter and crop-inflate draws OFF, a fixed window.
    'val-geometry': _L(prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0, crop_inflate=[0.9, 1.5]),
}


def test_the_covered_cases_actually_reach_their_branches(seam_root):
    """Each case OBSERVABLY reaches the branch it covers -- once, configs that never entered the
    prompt-noise and box-prompt paths passed on an unusable oracle."""
    # The box-prompt case must actually append the box field (13 fields -> 14).
    plain = _run(PoseDataset(seam_root / 'ratlike', 'train', _CASES['plain'], train=True), 0)[0]
    boxed = _run(PoseDataset(seam_root / 'ratlike', 'train', _CASES['box-prompt'], train=True), 0)[0]
    assert plain is not None and boxed is not None
    assert len(plain) == 13, f'the plain case unexpectedly has {len(plain)} fields'
    assert len(boxed) == 14, ('the box-prompt case did not append a box -- the box branch is '
                             'not being exercised')
    assert boxed[13] is not None and boxed[13].shape[-1] == 4

    # The prompt-noise case must actually move the prior (it reaches `get_camera_scale`).
    noisy = _run(PoseDataset(seam_root / 'mouselike', 'train', _CASES['prompt-noise'], train=True),
                 0)[0]
    assert noisy is not None
    assert torch.isfinite(noisy[11]).any(), 'the noise case produced no finite prior at all'
    assert not _same_or_false(plain[11], noisy[11]), \
        'the prompt-noise case produced the SAME prior as the plain case -- its branch did not run'

    # The ranged-inflate case must DRAW per item on train, and be the range's MIDPOINT on val.
    tr = PoseDataset(seam_root / 'mouselike', 'train', _CASES['ranged-inflate'], train=True)
    seen = set()
    for seed in (0, 1, 2, 3, 4, 5):
        sel = tr.select(0, np.random.default_rng(seed), tr.shape(np.random.default_rng(1)))
        assert sel is not None
        assert 0.9 <= sel.inflate <= 1.5, f'inflate {sel.inflate} outside the configured range'
        seen.add(round(sel.inflate, 6))
    assert len(seen) > 1, 'train crop-inflate did not vary -- the ranged draw is not running'

    va = PoseDataset(seam_root / 'mouselike', 'val', _CASES['val-geometry'], train=False)
    vsel = va.select(0, np.random.default_rng(0), va.shape(np.random.default_rng(1)))
    assert vsel is not None
    assert vsel.inflate == pytest.approx(1.2), \
        f'val crop-inflate must be the range MIDPOINT (1.2), got {vsel.inflate}'


def _same_or_false(a, b):
    try:
        _same(a, b, 'x')
        return True
    except AssertionError:
        return False


# --- structural properties the scorer depends on -----------------------------------------


def test_the_ground_truth_arrays_are_not_written_through(seam_root):
    """`select` shares storage with the stored labels; realising must not write through them (the
    scorer realises one selection twice)."""
    cfg = _CASES['augmented']
    ds = PoseDataset(seam_root / 'mouselike', 'train', cfg, train=True)
    sel = ds.select(0, np.random.default_rng(7), ds.shape(np.random.default_rng(1)))
    assert sel is not None
    before3d = sel.lab.points3d.copy()
    before2d = None if sel.lab.points2d is None else sel.lab.points2d.copy()

    ds.realise(sel, np.random.default_rng(11))
    np.testing.assert_array_equal(sel.lab.points3d, before3d)
    if before2d is not None:
        np.testing.assert_array_equal(sel.lab.points2d, before2d)


@pytest.mark.parametrize('case', ['augmented', 'box-prompt'])
def test_two_realisations_of_one_selection_are_independent(seam_root, case):
    """Realising twice leaves no trace: a replay after a different view equals the first view."""
    cfg = _CASES[case]
    for root_name in (('ratlike', 'mouselike') if case == 'box-prompt' else ('mouselike',)):
        ds = PoseDataset(seam_root / root_name, 'train', cfg, train=True)
        sel_a = ds.select(0, np.random.default_rng(3), ds.shape(np.random.default_rng(1)))
        assert sel_a is not None

        rng, _ = _seed_all(100, 1, 0xA06, 0xB07)
        first = ds.realise(sel_a, rng)
        rng, _ = _seed_all(200, 1, 0xA06, 0xB07)          # a DIFFERENT view of the same selection
        ds.realise(sel_a, rng)
        rng, _ = _seed_all(100, 1, 0xA06, 0xB07)
        replay = ds.realise(sel_a, rng)                   # the same view again

        assert first is not None and replay is not None, f'{case}/{root_name}: realise returned None'
        _same(first.coords, replay.coords, 'coords')
        for i, (a, b) in enumerate(zip(first.views, replay.views)):
            _same(a, b, f'views[{i}]')
        assert first.r == replay.r
        assert first.boxes is not None and len(first.boxes) == len(replay.boxes)
        for i, (a, b) in enumerate(zip(first.boxes, replay.boxes)):
            _same(a, b, f'boxes[{i}]')


def test_select_is_deterministic_given_idx_rng_and_shape(seam_root):
    """One selection per `(idx, rng, shape)` -- the scorer realises one, so it cannot re-draw."""
    cfg = _CASES['prompt-swaps']
    ds = PoseDataset(seam_root / 'mouselike', 'train', cfg, train=True)
    shape = ds.shape(np.random.default_rng(5))
    a = ds.select(0, np.random.default_rng(9), shape)
    b = ds.select(0, np.random.default_rng(9), shape)
    assert a is not None and b is not None
    _same(a.frames, b.frames, 'frames')
    _same(a.coords, b.coords, 'coords')
    assert a.cam_names == b.cam_names and a.cam_ix == b.cam_ix
    assert a.animal == b.animal and a.inflate == b.inflate
