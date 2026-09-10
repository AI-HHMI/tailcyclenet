"""The entry-point contract that a DDP rank respawn depends on.

Above one GPU, Lightning re-runs the command once per rank from `sys.argv`
(`lightning_fabric/strategies/launchers/subprocess_script.py`): a `-m` entry becomes
`[python, '-m', __main__.__spec__.name] + sys.argv[1:]`, and a script entry becomes
`[python, os.path.abspath(sys.argv[0])] + sys.argv[1:]`. So `sys.argv[1:]` must still BEGIN with
the subcommand, or the respawned rank dispatches to whichever option happens to be first.

That is the failure 0.1.9 shipped: the dispatcher folded the subcommand into `argv[0]`
(`'<script> train'`), which removed it from `sys.argv[1:]` and turned the script path into one
string containing a space. Every multi-GPU run died at rank 1 before completing an iteration,
while every single-GPU run was unaffected because it never spawns a rank. Nothing covered this.

The other half of the contract is that the command's parser must NOT see the token: it is the same
`argparse.ArgumentParser` a direct `python scripts/<command>.py` run uses, and those read
`sys.argv[1:]` via a bare `parse_args()`.
"""
from __future__ import annotations

import sys

import pytest

from tailcyclenet import __main__ as dispatcher
from tailcyclenet import train as train_module


def test_dispatch_keeps_the_subcommand_for_a_respawn_and_withholds_it_from_the_parser(monkeypatch):
    """`sys.argv[1:]` starts with the token after dispatch, while main() got the token-free args."""
    seen = []
    monkeypatch.setattr(train_module, 'main', lambda argv=None: seen.append(argv))
    monkeypatch.setattr(sys, 'argv', ['tailcyclenet', 'train', '--iters', '4', '--no-wandb'])

    dispatcher.main()

    assert sys.argv[1:] == ['train', '--iters', '4', '--no-wandb']
    assert seen == [['--iters', '4', '--no-wandb']]


def test_a_respawned_argv_redispatches_to_the_same_command(monkeypatch):
    """The argv a rank inherits must name the command again, not be read as an option."""
    seen = []
    monkeypatch.setattr(train_module, 'main', lambda argv=None: seen.append(argv))
    monkeypatch.setattr(sys, 'argv', ['tailcyclenet', 'train', '--iters', '4'])

    dispatcher.main()
    inherited = list(sys.argv[1:])

    seen.clear()
    monkeypatch.setattr(sys, 'argv', ['/somewhere/tailcyclenet/__main__.py', *inherited])
    dispatcher.main()

    assert inherited[0] == 'train'
    assert seen == [['--iters', '4']]


def test_a_first_argument_that_is_not_a_command_is_refused(monkeypatch):
    """An option in command position is an error rather than a silently wrong dispatch."""
    monkeypatch.setattr(sys, 'argv', ['tailcyclenet', '--iters', '4'])

    with pytest.raises(SystemExit, match='is not a tailcyclenet command'):
        dispatcher.main()
