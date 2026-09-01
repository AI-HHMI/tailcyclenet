#!/usr/bin/env python
"""Finetune a posetail tracker into a pose estimator.

    pixi run python scripts/train.py --config configs/base.toml [--devices N]

The program itself is `tailcyclenet.train`; this file is the documented invocation, so the
half of the program that used to live here can be imported and tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.train import main

if __name__ == '__main__':
    main()
