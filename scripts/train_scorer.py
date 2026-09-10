#!/usr/bin/env python
"""CLI shim for `tailcyclenet.scorer.train`, mirroring `scripts/train_detector.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.scorer.train import main

if __name__ == '__main__':
    raise SystemExit(main())
