#!/usr/bin/env python
"""Train one box predictor on one dataset root or a folder of roots.

    pixi run python scripts/train_detector.py --config configs/detector.toml

The program itself is `tailcyclenet.detector.train`; this file is the documented invocation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.detector.train import main

if __name__ == '__main__':
    main()
