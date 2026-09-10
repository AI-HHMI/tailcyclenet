"""Gate A.1: the `_select`/`_realise`/`_targets` split is BYTE-IDENTICAL to the old `_item`.

The split exists so a track-quality triplet can realise ONE selection into two independent views
(`tailcyclenet/scorer/`). It is approved only on the condition that the pose loader does not move,
and that condition is not self-evident: the three parts share one RNG stream, so a draw that
changed order -- or a helper that started consuming entropy it did not before -- would silently
reshape the training distribution while every shape stayed correct.

The comparison is against the REAL pre-refactor function, frozen in `tests/legacy_item.py`.
Asserting that `_item` agrees with `_select`+`_realise`+`_targets` would prove nothing: `_item` only
delegates to them.

Three things this file is deliberate about, each learned from an earlier version of it failing:

  * **Every RNG that the loader touches is seeded and its POST-CALL STATE compared**, not just the
    output. Matching outputs alone cannot establish that the draw CONSUMPTION is unchanged -- a
    reordering that happens to commute would pass.
  * **Old-vs-old reproducibility is established first.** If the oracle cannot reproduce itself,
    every old-vs-new comparison below it is meaningless.
  * **The oracle's imports are repaired, not its algorithm.** The frozen copy still needs
    `get_camera_scale` and the real `box_prompt` module to RUN; a `NameError` there would show up
    as a passing test if the config never reached that branch.
"""
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet.dataset import LoaderConfig, PoseDataset

from .legacy_item import LegacyItemMixin


class _Legacy(LegacyItemMixin, PoseDataset):
    """`PoseDataset` whose `_item` is the pre-refactor implementation (the mixin wins the MRO).

    `PoseDataset.__getitem__`'s retry loop calls `self._item`, so this drives the OLD code through
    the SAME entry point the new one uses.
    """


def _same(a, b, path='item'):
    """Recursive element-wise equality, NaN-aware.

    `torch.equal` is deliberately NOT the primary check: it is not NaN-aware, and a
    partially-labelled window is full of NaNs by design, so it reports a mismatch for two
    IDENTICAL tensors. That trap made an earlier version of this file fail against itself.
    """
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
    """Seed every stream the loader can draw from, so two runs are comparable.

    FOUR sources, because the loader uses four and only one of them is the item's own rng:

    * the item `Generator` (rotation draw, crop jitter, grayscale, cutout, the prompt corruptions);
    * `np.random` -- `PosetailDataset.rotate_camera_group` draws the 3D world-gauge angle from the
      GLOBAL numpy RNG (`rvec = np.random.uniform(...)`), not from the item stream;
    * `imgaug` -- the appearance pipelines draw from imgaug's own global RNG, seeded with entropy
      when the pipelines are built, so two datasets otherwise differ in the PIXELS while every
      coordinate agrees;
    * `random` / `torch` -- seeded for completeness; a helper that reached for either would
      otherwise make this test flaky rather than failing.

    `np.random` and `imgaug` are pre-existing warts in the loader, not the refactor's doing.
    """
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


def _same_state(a, b, path):
    """Recursive equality over whatever a RNG state snapshot happens to contain.

    The containers are not uniform -- `np.random.get_state()` is a tuple holding an ndarray, the
    item generator's state is a nested dict, imgaug's is a bare ndarray -- so a hand-written
    two-level comparison gets it wrong. Written recursively so a new snapshot key cannot silently
    be skipped.
    """
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f'{path}: state keys differ'
        for k in a:
            _same_state(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), f'{path}: state arity differs'
        for i, (x, y) in enumerate(zip(a, b)):
            _same_state(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f'{path}: ndarray vs {type(b)}'
        assert np.array_equal(a, b), f'{path}: RNG state array differs'
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f'{path}: RNG state tensor differs'
    else:
        assert a == b, f'{path}: RNG state differs ({a!r} vs {b!r})'


def _run(ds, idx, seed=0, shape_seed=0x5AFE, aug_seed=0xA06, rot_seed=0xB07):
    """One `_item` plus the RNG state it left behind."""
    rng, shape_rng = _seed_all(seed, shape_seed, aug_seed, rot_seed)
    shape = ds._shape(shape_rng)
    out = ds._item(idx, rng, shape)
    return out, _states(rng)


# --- the fixture: 2D and 3D, train AND val, plus a moving rig ------------------------------

KPTS_2D = ['nose', 'left_ear', 'right_ear', 'tail_base']
KPTS_3D = ['nose', 'neck', 'tail_base']


@pytest.fixture(scope='session')
def seam_root(tmp_path_factory):
    """A root holding every case the seam must not disturb: 2D and 3D, train and val, moving.

    Built here rather than reusing `tiny_root` because `tiny_root` has NO 3D `val` split -- a
    previous version of this file silently dropped every 3D val case for that reason and reported
    success anyway.
    """
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


# Each case names the DRAW SITES it exists to cover. Anything the loader can branch on should
# appear in at least one, or the branch is not covered by this test at all.
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
    'box-prompt-dropped': _L(prob_2d_only=0.0, aug_prob=0.5, box_prompt='film',
                             box_prompt_dropout=0.0, box_prompt_frames='first'),
    # The RANGED crop-inflate draw (a per-item uniform draw, not a scalar).
    'ranged-inflate': _L(prob_2d_only=0.0, aug_prob=0.5, crop_inflate=[0.9, 1.5]),
    # 3D single-view via `prob_2d_only`, and a ranged camera count.
    'single-view': _L(prob_2d_only=1.0, aug_prob=1.0, crop_jitter=0.2, cams_to_sample=[2, 3]),
    # Val geometry: jitter and crop-inflate draws OFF, a fixed window.
    'val-geometry': _L(prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0, crop_inflate=[0.9, 1.5]),
}


def _cases_for(name):
    """Which (root, split) pairs a case is meaningful on, and whether that split is train."""
    if name in ('box-prompt', 'box-prompt-dropped'):
        return [('ratlike', 'train', True), ('mouselike', 'train', True)]   # 2D and 3D boxes
    if name in ('ranged-inflate', 'single-view', 'prompt-noise', 'prompt-swaps'):
        return [('mouselike', 'train', True)]
    if name == 'val-geometry':
        return [('ratlike', 'val', False), ('mouselike', 'val', False)]
    return [('ratlike', 'train', True), ('ratlike', 'val', False),
            ('mouselike', 'train', True), ('mouselike', 'val', False)]


@pytest.mark.parametrize('case', sorted(_CASES))
def test_split_is_byte_identical_to_the_old_item(seam_root, case):
    """Same seeds, same index, same shape -> identical item AND identical RNG consumption."""
    cfg = _CASES[case]
    for root_name, split, train in _cases_for(case):
        root = seam_root / root_name
        new = PoseDataset(root, split, cfg, train=train)
        old = _Legacy(root, split, cfg, train=train)
        assert len(new) == len(old) > 0, f'{case}/{root_name}/{split}: fixture holds no windows'

        compared = 0
        for idx in range(min(len(new), 4)):
            for seed in (0, 1, 12345):
                a, sa = _run(old, idx, seed)
                b, sb = _run(new, idx, seed)
                assert (a is None) == (b is None), \
                    f'{case}/{root_name}/{split} idx={idx} seed={seed}: None mismatch'
                if a is None:
                    continue
                assert len(a) == len(b), 'tuple arity changed'
                _same(a, b, f'[{case} {root_name}/{split} idx={idx} seed={seed}]')
                # Outputs matching is NOT sufficient: a reordering that commutes would pass.
                _same_state(sa, sb, f'[{case} {root_name}/{split} idx={idx} seed={seed}]')
                compared += 1
        assert compared > 0, f'{case}/{root_name}/{split}: nothing built -- vacuous'


def test_the_oracle_reproduces_itself(seam_root):
    """Old-vs-old must match before any old-vs-new comparison means anything.

    If the frozen oracle could not reproduce its own output, every byte-identity assertion above
    would be measuring the harness's RNG control rather than the refactor.
    """
    cfg = _CASES['prompt-swaps']
    old = _Legacy(seam_root / 'mouselike', 'train', cfg, train=True)
    a, sa = _run(old, 0, 7)
    b, sb = _run(old, 0, 7)
    assert a is not None and b is not None
    _same(a, b, '[oracle self-check]')
    _same_state(sa, sb, '[oracle self-check]')


def test_the_unseeded_globals_are_the_reason_a_naive_version_fails(seam_root):
    """A NEGATIVE control for the seeding: without controlling `np.random`/imgaug the oracle does
    NOT reproduce itself.

    This is what makes `_seed_all` load-bearing rather than cargo-culted -- and it is the exact
    trap an earlier version of this file fell into, reporting a refactor failure that was its own.
    """
    import imgaug.random

    cfg = _CASES['augmented']
    old = _Legacy(seam_root / 'ratlike', 'train', cfg, train=True)

    # Deliberately NOT seeded: two runs in a row.
    a = old._item(0, np.random.default_rng(0), old._shape(np.random.default_rng(1)))
    imgaug.random.seed(999)          # perturb imgaug between the two
    b = old._item(0, np.random.default_rng(0), old._shape(np.random.default_rng(1)))
    assert a is not None and b is not None
    try:
        _same(a, b, '[unseeded control]')
        differs = False
    except AssertionError:
        differs = True
    assert differs, ('the unseeded control did NOT differ -- the appearance/rotation draws are '
                     'apparently deterministic here, so this test cannot demonstrate anything and '
                     'the seeding in _seed_all is not justified by it')


def test_rejection_paths_agree(tmp_path):
    """Windows that FAIL to build must fail identically, and consume the same RNG doing it.

    A rejection is a draw too: `__getitem__` retries with `idx = int(rng.integers(len(index)))`, so
    a rejection that moved changes which sample the NEXT attempt sees. The rejection is produced
    deterministically by DELETING one frame file -- `read_frames` returns None and `_realise` bails
    -- which is the `_realise`-level failure the seam introduces.

    (A too-long `n_frames` does NOT reject: `_frames` clamp-pads, so a 4-frame group asked for 8
    frames repeats frames instead of failing. An earlier version of this test asserted otherwise.)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path / 'ds'
    C._session_2d(root / 'ratlike' / 'train' / 'sess_a')
    (root / 'ratlike' / 'train' / 'sess_a' / 'groups' / 'g000' / 'cam0' / '000001.png').unlink()

    cfg = _L(prob_2d_only=0.0, aug_prob=0.5, crop_jitter=0.0)
    new = PoseDataset(root / 'ratlike', 'train', cfg, train=True)
    old = _Legacy(root / 'ratlike', 'train', cfg, train=True)

    rejected = built = 0
    for idx in range(len(new)):
        for seed in (0, 3):
            a, sa = _run(old, idx, seed)
            b, sb = _run(new, idx, seed)
            assert (a is None) == (b is None), f'idx={idx} seed={seed}: rejection mismatch'
            if a is None:
                rejected += 1
            else:
                built += 1
                _same(a, b, f'[reject-case idx={idx} seed={seed}]')
            _same_state(sa, sb, f'[reject-case idx={idx} seed={seed}]')
    assert rejected > 0, 'nothing was rejected -- this case does not exercise the path'


def test_the_covered_cases_actually_reach_their_branches(seam_root):
    """Each case must OBSERVABLY reach the branch it exists to cover.

    This is the guard against the failure mode this file already had once: configs that looked
    like they covered the prompt-noise and box-prompt paths, but never entered either, so the
    oracle's missing `get_camera_scale` import and its `from . import box_prompt` (which would
    have resolved against `tests/`) were never executed and the test passed on an unusable oracle.
    A green suite that does not run the code is worse than a red one.
    """
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
        sel = tr._select(0, np.random.default_rng(seed), tr._shape(np.random.default_rng(1)))
        assert sel is not None
        assert 0.9 <= sel.inflate <= 1.5, f'inflate {sel.inflate} outside the configured range'
        seen.add(round(sel.inflate, 6))
    assert len(seen) > 1, 'train crop-inflate did not vary -- the ranged draw is not running'

    va = PoseDataset(seam_root / 'mouselike', 'val', _CASES['val-geometry'], train=False)
    vsel = va._select(0, np.random.default_rng(0), va._shape(np.random.default_rng(1)))
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
    """`_select` hands out the stored coordinates; nothing may write through them.

    `torch.as_tensor` on the label array SHARES storage, so an in-place op after `_select` would
    corrupt the session for every later item. The old fused `_item` had the same exposure but
    could not be asked this question; the split makes it load-bearing, because the scorer realises
    one selection TWICE.
    """
    cfg = _CASES['augmented']
    ds = PoseDataset(seam_root / 'mouselike', 'train', cfg, train=True)
    sel = ds._select(0, np.random.default_rng(7), ds._shape(np.random.default_rng(1)))
    assert sel is not None
    before3d = sel.lab.points3d.copy()
    before2d = None if sel.lab.points2d is None else sel.lab.points2d.copy()

    ds._realise(sel, np.random.default_rng(11))
    np.testing.assert_array_equal(sel.lab.points3d, before3d)
    if before2d is not None:
        np.testing.assert_array_equal(sel.lab.points2d, before2d)


@pytest.mark.parametrize('case', ['augmented', 'prompt-swaps', 'box-prompt'])
def test_two_realisations_of_one_selection_are_independent(seam_root, case):
    """The two-view property the scorer depends on: realising twice must leave no trace.

    A replay of the same view after an intervening different view must be identical to the first
    -- which fails silently if any helper ever starts mutating `sel` in place.
    """
    cfg = _CASES[case]
    for root_name in (('ratlike', 'mouselike') if case == 'box-prompt' else ('mouselike',)):
        ds = PoseDataset(seam_root / root_name, 'train', cfg, train=True)
        sel_a = ds._select(0, np.random.default_rng(3), ds._shape(np.random.default_rng(1)))
        assert sel_a is not None

        _, first = _seed_all(100, 1, 0xA06, 0xB07), None
        rng, _ = _seed_all(100, 1, 0xA06, 0xB07)
        first = ds._realise(sel_a, rng)
        rng, _ = _seed_all(200, 1, 0xA06, 0xB07)          # a DIFFERENT view of the same selection
        ds._realise(sel_a, rng)
        rng, _ = _seed_all(100, 1, 0xA06, 0xB07)
        replay = ds._realise(sel_a, rng)                   # the same view again

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
    shape = ds._shape(np.random.default_rng(5))
    a = ds._select(0, np.random.default_rng(9), shape)
    b = ds._select(0, np.random.default_rng(9), shape)
    assert a is not None and b is not None
    _same(a.frames, b.frames, 'frames')
    _same(a.coords, b.coords, 'coords')
    assert a.cam_names == b.cam_names and a.cam_ix == b.cam_ix
    assert a.animal == b.animal and a.inflate == b.inflate
