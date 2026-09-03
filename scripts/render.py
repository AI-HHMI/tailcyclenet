#!/usr/bin/env python
"""Draw a prediction over the pixels it was made from. Offline, and model-free.

    pixi run python scripts/render.py --pred pred/ --out clips/

The program itself is `tailcyclenet.render`; this file is the documented invocation, so the half
of the program that used to live here can be imported and tested. `tailcyclenet render` runs the
same `main()`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.render import main

if __name__ == '__main__':
    main()
