"""`tailcyclenet <subcommand> ...` -- the pip-installed entry point (`[project.scripts]`).

Each subcommand is the same `main()` `scripts/*.py` has always called; this only adds the
dispatch layer so a pip install has something to run without cloning the repo for `scripts/`.
Each subcommand owns its own `argparse.ArgumentParser`, which sets `prog` to
`tailcyclenet <subcommand>`; argv is re-parsed without the subcommand token rather than nesting
parsers, so `--help` on a subcommand is the same text running the underlying script directly
produces, minus the entry-point path.
"""
from __future__ import annotations

import sys

COMMANDS = {
    'train': ('tailcyclenet.train', 'finetune a pose model'),
    'train-detector': ('tailcyclenet.train_detector', 'train a YOLOX-Nano box detector'),
    'infer': ('tailcyclenet.infer', 'run a trained model'),
    'eval': ('tailcyclenet.eval', 'score predictions against labels'),
    'render': ('tailcyclenet.render', 'draw a prediction over its own pixels'),
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch `tailcyclenet <command> ...` to that command's own `main()`.

    Inputs: argv -- defaults to `sys.argv[1:]`.
    Side effects: prints usage and raises SystemExit(2) with no/unknown command, else calls the
                  target module's `main(rest)`, which owns everything past the command name --
                  the same `argparse.ArgumentParser` a direct `python scripts/<command>.py`
                  invocation reads. `sys.argv` is left as the process received it, i.e.
                  `[argv0, command, *rest]`; see the note on the rank respawn below.

    Why `sys.argv` must keep the command token while `main()` is handed `rest` separately.
    Lightning rebuilds the command for each DDP rank from `sys.argv`
    (`lightning_fabric/strategies/launchers/subprocess_script.py`): for a `-m` entry it runs
    `[sys.executable, '-m', __main__.__spec__.name] + sys.argv[1:]`, so the child only re-dispatches
    to the same subcommand if `sys.argv[1:]` still BEGINS with it. But each subcommand's parser is
    the one a direct `python scripts/<command>.py` run uses, and those call `parse_args()` against
    `sys.argv[1:]`, which requires the token to be ABSENT. The two needs conflict on a single
    `sys.argv`, so they are separated: `sys.argv` keeps the token (respawn-safe) and `rest` is
    passed in explicitly (parse-correct). This is why the subcommand modules take an `argv`
    parameter at all -- `infer/cli.py` already did.

    The previous version folded the command into `argv[0]` (`'<script> <command>'`) to name it in
    argparse's `prog`. That satisfied neither: the `-m` route then dropped the subcommand (the
    dispatcher saw `--data`), and the script route was handed `os.path.abspath()` of a path with an
    embedded space. Both failed only above one GPU, because a single-GPU run never spawns a rank.
    The subcommand name now reaches argparse through each parser's `prog` instead.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ('-h', '--help'):
        prog = 'tailcyclenet'
        lines = [f'usage: {prog} <command> ...', '', 'commands:']
        lines += [f'  {name:<15s} {help_}' for name, (_, help_) in COMMANDS.items()]
        lines.append(f"\nrun `{prog} <command> --help` for a command's own options.")
        print('\n'.join(lines))
        raise SystemExit(0 if argv else 2)
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        raise SystemExit(f'{command!r} is not a tailcyclenet command. '
                         f'Choose from: {", ".join(COMMANDS)}.')
    module_name, _ = COMMANDS[command]
    import importlib
    mod = importlib.import_module(module_name)
    sys.argv = [sys.argv[0], command, *rest]
    mod.main(rest)


if __name__ == '__main__':
    main()
