#!/usr/bin/env bash
# Run any repo Python entry point on an LSF L4 compute host, in the project's own environment.
# This is the generic launcher; scripts/job_l4_infer.sh is the inference-shaped wrapper.
#
# The env is invoked by ABSOLUTE PATH with the project's activation environment rather than through
# `pixi run`. A compute node has no network, and pixi re-solves the lock whenever it looks stale --
# which fails there with a DNS error instead of falling back. The three variables below are the
# whole of `[tool.pixi.activation.env]` in pyproject.toml; keep them in step with it.
#
# Usage: job_l4_run.sh <script.py> [args...]
set -euo pipefail

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet

cd "$REPO"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPYCACHEPREFIX="${HOME}/.cache/tailcyclenet/pycache"
export MALLOC_ARENA_MAX=2

echo "host $(hostname)  argv: $*"
nvidia-smi || true
exec "$REPO/.pixi/envs/default/bin/python" -u "$@"
