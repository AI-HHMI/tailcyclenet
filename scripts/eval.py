#!/usr/bin/env python
"""`tailcyclenet eval` from a checkout; `--help` lists its options."""
from tailcyclenet.__main__ import run

if __name__ == '__main__':
    raise SystemExit(run('eval'))
