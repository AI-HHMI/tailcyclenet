#!/usr/bin/env bash
# Submit one Gate-D inference pass per 3dpop test session on the LSF gpu_l4 queue.
#
# One session per job, because the sessions are independent and a single job that walked all of
# them would serialise ~1.5 h of GPU behind one failure. Each job writes its own prediction
# session directory, so a re-run skips what already exists.
#
#   bash scripts/submit_l4_infer.sh --dry-run
#   RUN=<pose run> bash scripts/submit_l4_infer.sh
#   RUN=<pose run> WALL=04:00 bash scripts/submit_l4_infer.sh Pigeon10__Sequence59_n10_28062022
#
# Run from the workstation. `ssh login2 bjobs` shows the queue; `bpeek <id>` tails one job.
set -euo pipefail

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
DATA="${DATA:-/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/3dpop}"
SPLIT="${SPLIT:-test}"
RUN="${RUN:-/groups/karashchuk/home/karashchukl/results/tailcyclenet/runs/boxwidenf12distractor-20260820_1840/3dpop-box-wide-unfreeze-nframes12-distractor-a010-k007}"
OUTDIR="${OUTDIR:-$REPO/scratch/gated}"
QUEUE="${QUEUE:-gpu_l4}"
WALL="${WALL:-04:00}"
GPUS="${GPUS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---box-prompt labels}"
LOGS=/groups/karashchuk/home/karashchukl/logs/tailcyclenet
JOB="$REPO/scripts/job_l4_infer.sh"

# The small clean clips plus the two clips whose duplicate episodes are the named known-bad
# cases (Sequence29 / Sequence59). The small ones are cheap and give the correlation its
# low-disagreement end; the two long ones supply the high end and Gate D's part 2.
DEFAULT_SESSIONS=(
    Pigeon01__Sequence49_n01_28062022
    Pigeon01__Sequence56_n01_28062022
    Pigeon01__Sequence3_n01_01072022
    Pigeon01__Sequence14_n01_13072022
    Pigeon01__Sequence8_n01_01072022
    Pigeon01__Sequence12_n01_13072022
    Pigeon05__Sequence29_n05_04072022
    Pigeon10__Sequence59_n10_28062022
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
    DRY_RUN=1
    shift
fi

# `bsub` exists on the LSF login node, not on this workstation. Re-dispatch there so the script
# has one entry point from either side; the remote copy finds bsub and proceeds normally.
LOGIN_HOST="${LOGIN_HOST:-login2}"
if ! command -v bsub >/dev/null 2>&1; then
    printf -v remote 'cd %q && ' "$REPO"
    printf -v envs \
        'RUN=%q DATA=%q SPLIT=%q OUTDIR=%q QUEUE=%q WALL=%q GPUS=%q EXTRA_ARGS=%q LOGIN_HOST=%q ' \
        "$RUN" "$DATA" "$SPLIT" "$OUTDIR" "$QUEUE" "$WALL" "$GPUS" "$EXTRA_ARGS" "$LOGIN_HOST"
    printf -v rest '%s' ''
    fwd=()
    (( DRY_RUN )) && fwd+=(--dry-run)
    (( $# )) && fwd+=("$@")
    (( ${#fwd[@]} )) && printf -v rest '%q ' "${fwd[@]}"
    echo "no local bsub -- re-dispatching to $LOGIN_HOST"
    exec ssh "$LOGIN_HOST" "${remote}${envs}bash scripts/submit_l4_infer.sh ${rest}"
fi
if (( $# )); then
    SESSIONS=("$@")
else
    SESSIONS=("${DEFAULT_SESSIONS[@]}")
fi

[[ -x "$JOB" ]] || { echo "FATAL: missing executable job script: $JOB" >&2; exit 1; }
[[ -d "$DATA/$SPLIT" ]] || { echo "FATAL: no $DATA/$SPLIT" >&2; exit 1; }
[[ -f "${RUN%/}/config.toml" || -f "$RUN" ]] || {
    echo "FATAL: RUN is not a run folder or a .pth: $RUN" >&2; exit 1; }
mkdir -p "$LOGS" "$OUTDIR"

SLOTS=$(( 8 * GPUS ))
for S in "${SESSIONS[@]}"; do
    SHORT=$(printf '%s' "$S" | sed -E 's/_n[0-9]+_[0-9]+$//')
    OUT="$OUTDIR/pred-$SHORT"
    TAG="tcn-infer-${SHORT}-l4-$(date +%Y%m%d_%H%M%S)"
    cmd=(bsub
         -J "$TAG"
         -e "$LOGS/$TAG.err"
         -o "$LOGS/$TAG.out"
         -n "$SLOTS"
         -q "$QUEUE"
         -R "span[hosts=1]"
         -gpu "num=$GPUS"
         -W "$WALL"
         -env "RUN=$RUN,DATA=$DATA,SPLIT=$SPLIT"
         /bin/bash "$JOB" "$S" "$OUT" ${EXTRA_ARGS})
    if (( DRY_RUN )); then
        printf '%q ' "${cmd[@]}"
        echo
        continue
    fi
    if [[ -f "$OUT/session.toml" ]]; then
        echo "skip $SHORT (already inferred)"
        continue
    fi
    echo "submit $SHORT -> $OUT"
    "${cmd[@]}"
done
