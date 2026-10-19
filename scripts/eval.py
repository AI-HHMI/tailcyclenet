#!/usr/bin/env python
"""Score prediction sessions or legacy archives; independent-window sessions are scored by slot without identity metrics.

    pixi run python scripts/eval.py pred.npz --data <dataset> --split test

The program itself is `tailcyclenet.evaluation.eval`; this file is the documented invocation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.evaluation.eval import main

if __name__ == '__main__':
    main()
