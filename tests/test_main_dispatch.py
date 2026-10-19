"""The entry-point contract a DDP rank respawn depends on: Lightning re-runs `sys.argv`, so
`sys.argv[1:]` must still BEGIN with the subcommand (0.1.9 folded it into argv[0] and every
multi-GPU run died at rank 1), while the command's parser must not see the token."""
from __future__ import annotations

import sys

import pytest

from tailcyclenet import __main__ as dispatcher
from tailcyclenet.pose import train as train_module


def test_dispatch_keeps_the_subcommand_and_a_respawn_redispatches(monkeypatch):
    """After dispatch `sys.argv[1:]` starts with the token, main() got the token-free args, and
    the argv a rank inherits dispatches to the same command again."""
    seen = []
    monkeypatch.setattr(train_module, 'main', lambda argv=None: seen.append(argv))
    monkeypatch.setattr(sys, 'argv', ['tailcyclenet', 'train', '--iters', '4', '--no-wandb'])
    dispatcher.main()
    inherited = list(sys.argv[1:])
    assert inherited == ['train', '--iters', '4', '--no-wandb']
    monkeypatch.setattr(sys, 'argv', ['/somewhere/tailcyclenet/__main__.py', *inherited])
    dispatcher.main()
    assert seen == [['--iters', '4', '--no-wandb']] * 2


def test_a_first_argument_that_is_not_a_command_is_refused(monkeypatch):
    """An option in command position is an error rather than a silently wrong dispatch."""
    monkeypatch.setattr(sys, 'argv', ['tailcyclenet', '--iters', '4'])

    with pytest.raises(SystemExit, match='is not a tailcyclenet command'):
        dispatcher.main()


def test_scorer_commands_dispatch_to_their_modules(monkeypatch):
    """The scorer's train and score entry points are reachable from a pip install."""
    import importlib

    for command, module_name in [('train-scorer', 'tailcyclenet.scorer.train'),
                                 ('score', 'tailcyclenet.scorer.score')]:
        module = importlib.import_module(module_name)
        seen = []
        monkeypatch.setattr(module, 'main', lambda argv=None, seen=seen: seen.append(argv))
        monkeypatch.setattr(sys, 'argv', ['tailcyclenet', command, '--out', 'x'])
        dispatcher.main()
        assert sys.argv[1:] == [command, '--out', 'x']
        assert seen == [['--out', 'x']]
