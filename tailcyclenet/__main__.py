"""`tailcyclenet <subcommand> ...` -- the pip-installed entry point (`[project.scripts]`).

Each subcommand is the same `main()` `scripts/*.py` has always called; this only adds the
dispatch layer so a pip install has something to run without cloning the repo for `scripts/`.
Each subcommand owns its own `argparse.ArgumentParser`; argv is re-parsed without the subcommand
token rather than nesting parsers, so `--help` on a subcommand is byte-identical to running the
underlying script directly.
"""
from __future__ import annotations

import sys

COMMANDS = {
    'train': ('tailcyclenet.train', 'finetune a pose model'),
    'train-detector': ('tailcyclenet.train_detector', 'train a YOLOX-Nano box detector'),
    'infer': ('tailcyclenet.infer', 'run a trained model'),
    'eval': ('tailcyclenet.eval', 'score predictions against labels'),
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch `tailcyclenet <command> ...` to that command's own `main()`.

    Inputs: argv -- defaults to `sys.argv[1:]`.
    Side effects: prints usage and raises SystemExit(2) with no/unknown command, or rewrites
                  `sys.argv` and calls the target module's `main()`, which owns everything past
                  the command name -- the same `argparse.ArgumentParser` a direct
                  `python scripts/<command>.py` invocation reads, so `argv[0]` is rewritten to
                  name the subcommand actually run rather than `__main__.py`.
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
    sys.argv = [f'{sys.argv[0]} {command}', *rest]
    mod.main()


if __name__ == '__main__':
    main()
