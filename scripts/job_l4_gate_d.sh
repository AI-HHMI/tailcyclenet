#!/usr/bin/env bash
# Gate D in one L4 job: score the PREDICTION tracks, then correlate the scores with the pose
# model's real disagreement against the reference (plan section 7.4).
#
# The input is scratch/gated/hybrid: a directory per source session holding the PREDICTION's
# session.toml / points3d.pq (symlinks into scratch/gated/pred-*) plus a `groups` symlink to the
# SOURCE session's pixel tree. A prediction session on its own carries no pixels -- the scorer
# scores a track together with its video -- so the hybrid is what makes the predicted track
# scorable. scripts/make_gate_d_hybrids.sh builds it.
#
# Usage: job_l4_gate_d.sh [scorer-run] [scorer-config]
set -euo pipefail

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
SCORER_RUN="${1:-/groups/karashchuk/home/karashchukl/results/tailcyclenet/scorers/scorer-3dpop}"
SCORER_CONFIG="${2:-configs/scorer-3dpop.toml}"
REF_ROOT="${REF_ROOT:-/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/3dpop}"
GATED="$REPO/scratch/gated"

cd "$REPO"

echo "### scoring the prediction tracks $(date +%H:%M:%S)"
/bin/bash scripts/job_l4_run.sh scripts/score_session.py \
    --run "$SCORER_RUN" \
    --data "$GATED/hybrid" \
    --split test \
    --out "$GATED/qc" \
    --device cuda \
    --top 8

echo "### correlation vs reference disagreement $(date +%H:%M:%S)"
/bin/bash scripts/job_l4_run.sh scripts/scorer_error_correlation.py \
    --scores "$GATED/qc/scores.pq" \
    --pred-root "$GATED/hybrid" \
    --ref-root "$REF_ROOT" \
    --split test \
    --config "$SCORER_CONFIG"

echo "### done $(date +%H:%M:%S)"
