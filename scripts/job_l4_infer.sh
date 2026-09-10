#!/usr/bin/env bash
# One Gate-D inference pass, executed on an LSF L4 compute host. Submit via
# scripts/submit_l4_infer.sh; do not run this directly on the workstation. The environment comes
# from scripts/job_l4_run.sh, which is the generic launcher for every L4 job in this repo.
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

exec /bin/bash "$REPO/scripts/job_l4_run.sh" scripts/infer.py \
    --run "$RUN" \
    --data "$DATA/$SPLIT/$SESSION" \
    --out "$OUT" \
    "$@"
