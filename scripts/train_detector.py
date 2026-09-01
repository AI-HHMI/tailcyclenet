#!/usr/bin/env python
"""Train the box predictor, one detector per dataset.

    pixi run python scripts/train_detector.py --config configs/detector.toml

The program itself is `tailcyclenet.train_detector`; this file is the documented invocation, so
the half of the program that used to live here can be imported and tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.train_detector import main

if __name__ == '__main__':
    main()
