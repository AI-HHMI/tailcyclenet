#!/usr/bin/env bash
# Submit one native deeperfly run per extracted Ramdya-fly session to gpu_l4.
#
# From an LSF login node:
#   bash scripts/submit_deeperfly_l4.sh --dry-run
#   bash scripts/submit_deeperfly_l4.sh
#
# Source validation is deliberately strict: four conditions, seven cameras, exactly 900
# numbered JPEGs per camera.  The worker links (never edits) those images and writes a
# converter-compatible native results.h5 plus status.json under BASE/deeperfly_outputs/lsf.
set -Eeuo pipefail
shopt -s nullglob

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
BASE="${BASE:-/groups/karashchuk/karashchuklab/animal-datasets/ramdya-fly}"
OUT_ROOT="${OUT_ROOT:-$BASE/deeperfly_outputs/lsf}"
STAGE_ROOT="${STAGE_ROOT:-$BASE/deeperfly_staging/lsf}"
LOG_ROOT="${LOG_ROOT:-$BASE/deeperfly_outputs/lsf_logs}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-$REPO/configs/deeperfly_ramdya_lsf_900.toml}"
DEEPERFLY_ROOT="${DEEPERFLY_ROOT:-$BASE/deeperfly}"
DEEPERFLY_ENV="${DEEPERFLY_ENV:-$DEEPERFLY_ROOT/.venv-lsf}"
DEEPERFLY_BIN="${DEEPERFLY_BIN:-$DEEPERFLY_ENV/bin/deeperfly}"
WORKER="${WORKER:-$REPO/scripts/run_deeperfly_l4_session.sh}"
QUEUE="${QUEUE:-gpu_l4}"
WALL="${WALL:-08:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-197}"
SCALE_MM_PER_UNIT="${TAILCYCLE_SCALE_MM_PER_UNIT:-0.456}"
LOGIN_HOST="${LOGIN_HOST:-login2}"
NO_REMOTE="${NO_REMOTE:-0}"
# Keep submission fast on the shared filesystem.  The worker performs the exhaustive
# per-frame check immediately before inference; set STRICT_SOURCE=1 for a slower audit.
STRICT_SOURCE="${STRICT_SOURCE:-0}"
DRY_RUN=0
FORCE=0

CONDITIONS=(
    aDN-GAL4_Control
    MDN-GAL4_Control
    aDN-GAL4_UAS-CsChrimson
    MDN-GAL4_UAS-CsChrimson
)

usage() {
    cat <<'EOF'
usage: submit_deeperfly_l4.sh [--dry-run] [--force] [CONDITION SESSION ...]

With no pairs, discover all 197 extracted Ramdya sessions.  A pair may also be
written CONDITION/SESSION.  --dry-run performs all source and duplicate checks
and prints the exact bsub command without submitting it.

Environment overrides: BASE OUT_ROOT STAGE_ROOT LOG_ROOT CONFIG_TEMPLATE
DEEPERFLY_ROOT DEEPERFLY_ENV DEEPERFLY_BIN WORKER QUEUE WALL GPUS CPUS EXPECTED_SESSIONS LOGIN_HOST NO_REMOTE STRICT_SOURCE.
EOF
}

while (( $# )); do
    case "$1" in
        --dry-run|-n) DRY_RUN=1; shift ;;
        --force) FORCE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

[[ "$CPUS" == 8 ]] || { echo "FATAL: gpu_l4 workflow requires CPUS=8 (got $CPUS)" >&2; exit 2; }
[[ "$GPUS" == 1 ]] || { echo "FATAL: this workflow submits one L4 GPU per job" >&2; exit 2; }
[[ "$SCALE_MM_PER_UNIT" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "FATAL: TAILCYCLE_SCALE_MM_PER_UNIT must be numeric" >&2; exit 2; }
# LSF exists on login2, not generally on a workstation.  Preserve one documented
# entry point, while NO_REMOTE=1 makes dry-run/tests print locally without SSH.  Dispatch
# before touching the source tree: the dataset is mounted on the LSF login/compute hosts,
# and a workstation need not have the 197-session tree mounted.
if ! command -v bsub >/dev/null 2>&1 && [[ "$NO_REMOTE" != 1 ]]; then
    printf -v envs 'BASE=%q OUT_ROOT=%q STAGE_ROOT=%q LOG_ROOT=%q CONFIG_TEMPLATE=%q DEEPERFLY_ROOT=%q DEEPERFLY_ENV=%q DEEPERFLY_BIN=%q WORKER=%q QUEUE=%q WALL=%q GPUS=%q CPUS=%q EXPECTED_SESSIONS=%q TAILCYCLE_SCALE_MM_PER_UNIT=%q LOGIN_HOST=%q STRICT_SOURCE=%q NO_REMOTE=0 ' \
        "$BASE" "$OUT_ROOT" "$STAGE_ROOT" "$LOG_ROOT" "$CONFIG_TEMPLATE" "$DEEPERFLY_ROOT" "$DEEPERFLY_ENV" "$DEEPERFLY_BIN" "$WORKER" "$QUEUE" "$WALL" "$GPUS" "$CPUS" "$EXPECTED_SESSIONS" "$SCALE_MM_PER_UNIT" "$LOGIN_HOST" "$STRICT_SOURCE"
    rest=()
    (( DRY_RUN )) && rest+=(--dry-run)
    (( FORCE )) && rest+=(--force)
    rest+=("$@")
    args=''
    if (( ${#rest[@]} )); then
        printf -v args '%q ' "${rest[@]}"
    fi
    echo "no local bsub -- re-dispatching to $LOGIN_HOST" >&2
    exec ssh "$LOGIN_HOST" "cd $(printf '%q' "$REPO") && $envs bash $(printf '%q' "$REPO/scripts/submit_deeperfly_l4.sh") $args"
fi
if ! command -v bsub >/dev/null 2>&1; then
    (( DRY_RUN && NO_REMOTE == 1 )) || { echo "FATAL: bsub not found (set NO_REMOTE=1 only for local dry-run)" >&2; exit 1; }
fi

[[ -x "$WORKER" ]] || { echo "FATAL: missing executable worker: $WORKER" >&2; exit 1; }
[[ -f "$CONFIG_TEMPLATE" ]] || { echo "FATAL: missing config template: $CONFIG_TEMPLATE" >&2; exit 1; }
[[ -d "$BASE" ]] || { echo "FATAL: missing BASE: $BASE" >&2; exit 1; }

# The submit-time source gate uses one glob per camera rather than 900 NFS stat calls.
# The worker performs the exhaustive per-frame check after dispatch, before inference.
validate_source() {
    local condition=$1 session=$2 raw="$BASE/$condition/extracted/$session/images"
    [[ -d "$raw" ]] || { echo "FATAL: missing images: $raw" >&2; return 1; }
    local c i f count
    for (( c=0; c<7; c++ )); do
        files=("$raw/camera_${c}_img_"*.jpg)
        count=${#files[@]}
        (( count == 900 )) || {
            echo "FATAL: $condition/$session camera $c has $count JPEGs, expected 900" >&2
            return 1
        }
        [[ -f "$raw/camera_${c}_img_000000.jpg" &&
           -f "$raw/camera_${c}_img_000899.jpg" ]] || {
            echo "FATAL: $condition/$session camera $c lacks frame endpoints" >&2
            return 1
        }
        if [[ "$STRICT_SOURCE" == 1 ]]; then
            for (( i=0; i<900; i++ )); do
                f="$raw/camera_${c}_img_$(printf '%06d' "$i").jpg"
                [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; return 1; }
            done
        fi
    done
}

# Discovery is deterministic and intentionally does not follow arbitrary directories.  The
# expected count is an integrity check for the advertised all-197 operation; set
# EXPECTED_SESSIONS=0 only for a deliberately partial inventory/dry run.
DISCOVER_ALL=$(( $# == 0 ))
declare -a SESSIONS=()
if (( DISCOVER_ALL )); then
    for condition in "${CONDITIONS[@]}"; do
        extracted="$BASE/$condition/extracted"
        [[ -d "$extracted" ]] || { echo "FATAL: missing extracted directory: $extracted" >&2; exit 1; }
        shopt -s nullglob
        for d in "$extracted"/*_behData_images; do
            [[ -d "$d" ]] || continue
            SESSIONS+=("$condition/${d##*/}")
        done
        shopt -u nullglob
    done
else
    # Positional input is either CONDITION SESSION pairs or CONDITION/SESSION tokens.
    if (( $# == 1 )); then
        [[ "$1" == */* ]] || { echo "FATAL: use CONDITION SESSION or CONDITION/SESSION" >&2; exit 2; }
        SESSIONS=("$1")
    elif (( $# % 2 == 0 )); then
        while (( $# )); do
            SESSIONS+=("$1/$2")
            shift 2
        done
    else
        echo "FATAL: positional inputs must be CONDITION SESSION pairs" >&2; exit 2
    fi
fi

if (( EXPECTED_SESSIONS > 0 && ${#SESSIONS[@]} != EXPECTED_SESSIONS && DISCOVER_ALL )); then
    echo "FATAL: discovered ${#SESSIONS[@]} sessions, expected $EXPECTED_SESSIONS" >&2
    echo "      override EXPECTED_SESSIONS=0 only for a deliberate partial inventory" >&2
    exit 1
fi
(( ${#SESSIONS[@]} > 0 )) || { echo "FATAL: no sessions" >&2; exit 1; }

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

active_job() {
    local job=$1
    command -v bjobs >/dev/null 2>&1 || return 1
    # -J filters before parsing, and the status field is present in every LSF bjobs format.
    bjobs -noheader -J "$job" 2>/dev/null \
        | grep -Eq '[[:space:]](PEND|RUN|WAIT|UNKWN)([[:space:]]|$)'
}

for item in "${SESSIONS[@]}"; do
    condition=${item%%/*}
    session=${item#*/}
    case "$condition" in
        aDN-GAL4_Control|MDN-GAL4_Control|aDN-GAL4_UAS-CsChrimson|MDN-GAL4_UAS-CsChrimson) ;;
        *) echo "FATAL: unknown condition: $condition" >&2; exit 2 ;;
    esac
    [[ "$session" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*_behData_images$ ]] || {
        echo "FATAL: invalid session basename: $session" >&2; exit 2; }
    validate_source "$condition" "$session"

    out="$OUT_ROOT/$condition/$session"
    # Reuse the old lock namespace to prevent old df3d/native deeperfly overlap.
    lock="$BASE/.deepfly3d3d_lsf_locks/${condition}__${session}.lock"
    job="dfl4_${condition}_${session}"
    legacy_job="df3d_${condition}_${session}"
    logbase="$LOG_ROOT/${condition}__${session}"
    if [[ -f "$out/status.json" ]] && grep -q '"state": "complete"' "$out/status.json" && (( ! FORCE )); then
        echo "skip $condition/$session (complete)"
        continue
    fi
    if [[ -e "$lock" ]] || active_job "$job" || active_job "$legacy_job"; then
        echo "skip $condition/$session (already queued/running or legacy lock: $job/$legacy_job)"
        continue
    fi
    if [[ -e "$out" ]] && (( ! FORCE )); then
        echo "skip $condition/$session (partial/failed output; use --force to replace): $out" >&2
        continue
    fi

    cmd=(bsub
        -q "$QUEUE"
        -n "$CPUS"
        -gpu "num=$GPUS"
        -R "span[hosts=1]"
        -W "$WALL"
        -J "$job"
        -oo "$logbase.out"
        -eo "$logbase.err"
        -env "BASE=$BASE,OUT_ROOT=$OUT_ROOT,LOG_ROOT=$LOG_ROOT,STAGE_ROOT=$STAGE_ROOT,CONFIG_TEMPLATE=$CONFIG_TEMPLATE,DEEPERFLY_ROOT=$DEEPERFLY_ROOT,DEEPERFLY_ENV=$DEEPERFLY_ENV,DEEPERFLY_BIN=$DEEPERFLY_BIN,TAILCYCLE_SCALE_MM_PER_UNIT=$SCALE_MM_PER_UNIT,FORCE=$FORCE"
        /bin/bash "$WORKER" "$condition" "$session")
    if (( DRY_RUN )); then
        printf '%q ' "${cmd[@]}"
        echo
    else
        echo "submit $condition/$session -> $out"
        "${cmd[@]}"
    fi
done
