#!/usr/bin/env python
"""Run a trained model. The only entry point that touches a checkpoint.

The program itself is `tailcyclenet.infer` -- `cli.py` for the command line, `driver.py` for the
run over a dataset, `window.py` for the window loop. This file is the documented invocation
(`pixi run python scripts/infer.py ...`) and nothing else, so that the half of the program that
used to live here can be imported and tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.infer import main

if __name__ == '__main__':
    main()
