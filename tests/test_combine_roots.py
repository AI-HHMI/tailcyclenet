"""scripts/combine_roots.py -- the collision rule and the refusal to clobber.

Two things here would be silently wrong rather than loud. The first is the SPLIT-INVARIANCE of
the disambiguating suffix: `rat-city-tracked` carries one session name in train, val AND test
(its splits are frame ranges of one 57,594-frame recording), so a suffix applied only in the
split where the collision happens would make one session look like two to anything enforcing the
rule-14 leak check by folder name. The second is idempotency: the sweep preflight exists because
a half-built root looks fine to `-d train` and fails hours into a job, so a re-run must be a
no-op and an unexpected entry must stop the build rather than be silently overwritten.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def cr():
    spec = importlib.util.spec_from_file_location(
        'tcn_combine_roots', REPO / 'scripts' / 'combine_roots.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _root(base, name, sessions):
    """sessions: {split: [session name, ...]}. Only session.toml's EXISTENCE is read here."""
    root = base / name
    for split, names in sessions.items():
        for s in names:
            (root / split / s).mkdir(parents=True)
            (root / split / s / 'session.toml').write_text('mode = "2d"\n')
    return root


def test_collision_suffix_is_split_invariant(cr, tmp_path):
    """A colliding name is suffixed on BOTH sides in EVERY split; the rest stay plain."""
    a = _root(tmp_path, 'src-tracked', {'train': ['shared'], 'val': ['shared'], 'test': ['shared']})
    b = _root(tmp_path, 'src-annotated', {'train': ['shared', 'only_b'], 'val': ['other']})
    plan = cr.plan([a, b])

    assert set(plan) == {'train', 'val', 'test'}
    assert set(plan['train']) == {'shared__src-tracked', 'shared__src-annotated', 'only_b'}
    # the tracked session keeps ONE name across all three splits -- the property this test exists for
    assert set(plan['val']) == {'shared__src-tracked', 'other'}
    assert set(plan['test']) == {'shared__src-tracked'}
    assert plan['train']['shared__src-tracked'] == a / 'train' / 'shared'
    assert plan['train']['only_b'] == b / 'train' / 'only_b'


def test_build_is_idempotent_and_refuses_to_clobber(cr, tmp_path):
    a = _root(tmp_path, 'src-a', {'train': ['s1'], 'val': ['s2']})
    b = _root(tmp_path, 'src-b', {'train': ['s3']})
    out = tmp_path / 'src-combined'
    plan = cr.plan([a, b])

    assert cr.build(out, plan, force=False) == (3, 0)
    link = out / 'train' / 's1'
    assert link.is_symlink() and (link / 'session.toml').exists()
    assert link.readlink() == Path('../../src-a/train/s1')     # RELATIVE: the tree stays movable

    assert cr.build(out, plan, force=False) == (0, 0)          # a re-run writes nothing

    (out / 'train' / 'stale').symlink_to('../../src-a/train/s1')
    with pytest.raises(SystemExit):                            # stale link: refuse, then prune
        cr.build(out, plan, force=False)
    assert cr.build(out, plan, force=True) == (0, 1)

    (out / 'val' / 'real_dir').mkdir()
    with pytest.raises(SystemExit):                            # never delete a real directory
        cr.build(out, plan, force=True)
