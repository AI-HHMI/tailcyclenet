#!/usr/bin/env python
"""Score a prediction file against the labels. Offline and model-free.

    pixi run python scripts/eval.py pred.npz --data <dataset> --split test

The program itself is `tailcyclenet.eval`; this file is the documented invocation, so the half
of the program that used to live here can be imported and tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.eval import main

if __name__ == '__main__':
    main()
