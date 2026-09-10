#!/usr/bin/env bash
# One Gate-D inference pass, executed on an LSF L4 compute host. Submit via
# scripts/submit_l4_infer.sh; do not run this directly on the workstation.
#
# The env is invoked by ABSOLUTE PATH with the project's own activation environment rather than
# through `pixi run`. A compute node has no network, and pixi re-solves the lock whenever it looks
# stale -- which fails there with a DNS error instead of falling back, so `pixi run` is not a
# launcher that can be relied on. The three variables below are the whole of
# `[tool.pixi.activation.env]` in pyproject.toml; keep them in step with it.
#
# Usage: job_l4_infer.sh <session-name> <out-dir> [extra infer.py args...]
set -euo pipefail

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
DATA="${DATA:-/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/3dpop}"
RUN="${RUN:?set RUN to a pose run folder}"
SPLIT="${SPLIT:-test}"

SESSION=$1
OUT=$2
shift 2

cd "$REPO"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPYCACHEPREFIX="${HOME}/.cache/tailcyclenet/pycache"
export MALLOC_ARENA_MAX=2

echo "host $(hostname)  session $SESSION  out $OUT"
nvidia-smi || true
exec "$REPO/.pixi/envs/default/bin/python" -u scripts/infer.py \
    --run "$RUN" \
    --data "$DATA/$SPLIT/$SESSION" \
    --out "$OUT" \
    "$@"
