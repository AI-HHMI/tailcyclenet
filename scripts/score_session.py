#!/usr/bin/env python
"""CLI shim for `tailcyclenet.scorer.score` (`tailcyclenet score`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.scorer.score import main

if __name__ == '__main__':
    raise SystemExit(main())
