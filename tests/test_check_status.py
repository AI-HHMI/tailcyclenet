"""`scripts/check_status.py`'s structural symlink-farm detection."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _mod():
    """Load the script as a module."""
    spec = importlib.util.spec_from_file_location('tcn_check_status',
                                                  REPO / 'scripts' / 'check_status.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_root_is_a_symlink_farm_only_if_every_split_is(tmp_path):
    """The decision used to be taken on the FIRST split with sessions: a symlinked `train/` hid
    a real `test/` from the audit."""
    m = _mod()
    real = tmp_path / 'real'
    (real / 's0').mkdir(parents=True)
    (real / 's1').mkdir()

    mixed = tmp_path / 'mixed'
    (mixed / 'train').mkdir(parents=True)
    (mixed / 'train' / 's0').symlink_to(real / 's0')
    (mixed / 'test' / 's1').mkdir(parents=True)
    assert not m._is_symlink_farm(mixed)

    farm = tmp_path / 'farm'
    for split, s in (('train', 's0'), ('test', 's1')):
        (farm / split).mkdir(parents=True)
        (farm / split / s).symlink_to(real / s)
    assert m._is_symlink_farm(farm)
    assert not m._is_symlink_farm(tmp_path / 'empty')
