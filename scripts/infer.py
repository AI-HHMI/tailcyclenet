#!/usr/bin/env python
"""`tailcyclenet infer` from a checkout; `--help` lists its options."""
from tailcyclenet.__main__ import run

if __name__ == '__main__':
    raise SystemExit(run('infer'))
