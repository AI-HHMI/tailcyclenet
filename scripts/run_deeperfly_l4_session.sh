#!/usr/bin/env bash
# Run one native deeperfly Ramdya recording on an LSF L4 host.
#
# The source tree is read-only.  The worker links the first 900 JPEGs from all seven
# source cameras into a private temporary recording, runs the native deeperfly pipeline,
# validates results.h5, and removes the staging tree.  Native units are deliberately
# retained; the 0.456 mm reference scale is conversion metadata only here.
set -Eeuo pipefail

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
BASE="${BASE:-/groups/karashchuk/karashchuklab/animal-datasets/ramdya-fly}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-$REPO/configs/deeperfly_ramdya_lsf_900.toml}"
DEEPERFLY_ROOT="${DEEPERFLY_ROOT:-$BASE/deeperfly}"
# Use an environment whose Python interpreter is visible on both login2 and compute hosts.
DEEPERFLY_ENV="${DEEPERFLY_ENV:-$DEEPERFLY_ROOT/.venv-lsf}"
DEEPERFLY_BIN="${DEEPERFLY_BIN:-$DEEPERFLY_ENV/bin/deeperfly}"
OUT_ROOT="${OUT_ROOT:-$BASE/deeperfly_outputs/lsf}"
LOG_ROOT="${LOG_ROOT:-$BASE/deeperfly_outputs/lsf_logs}"
STAGE_ROOT="${STAGE_ROOT:-$BASE/deeperfly_staging/lsf}"
FRAMES="${FRAMES:-900}"
CAMERAS="${CAMERAS:-7}"
SCALE_MM_PER_UNIT="${TAILCYCLE_SCALE_MM_PER_UNIT:-0.456}"
FORCE="${FORCE:-0}"
KEEP_STAGE="${KEEP_STAGE:-0}"

# The native checkout is separate from tailcyclenet's Pixi environment.  LSF compute
# hosts expose CUDA through modules; the shared .venv-lsf avoids per-job uv setup.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

usage() {
    cat >&2 <<'EOF'
usage: run_deeperfly_l4_session.sh CONDITION SESSION

CONDITION is one of the four Ramdya condition directories. SESSION is the exact
*_behData_images directory basename below CONDITION/extracted.
EOF
}

(( $# == 2 )) || { usage; exit 2; }
CONDITION=$1
SESSION=$2
case "$CONDITION" in
    aDN-GAL4_Control|MDN-GAL4_Control|aDN-GAL4_UAS-CsChrimson|MDN-GAL4_UAS-CsChrimson) ;;
    *) echo "FATAL: unknown condition: $CONDITION" >&2; exit 2 ;;
esac
[[ "$SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*_behData_images$ ]] || {
    echo "FATAL: SESSION must be a safe basename ending _behData_images: $SESSION" >&2; exit 2; }
[[ "$FRAMES" == 900 && "$CAMERAS" == 7 ]] || {
    echo "FATAL: this workflow is fixed at 900 frames and seven cameras (got $FRAMES/$CAMERAS)" >&2
    exit 2
}
[[ "$SCALE_MM_PER_UNIT" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "FATAL: TAILCYCLE_SCALE_MM_PER_UNIT must be numeric" >&2
    exit 2
}

RAW="$BASE/$CONDITION/extracted/$SESSION/images"
OUT="$OUT_ROOT/$CONDITION/$SESSION"
# Reuse the old lock namespace so old df3d and native deeperfly submissions cannot overlap.
LOCK="$BASE/.deepfly3d3d_lsf_locks/${CONDITION}__${SESSION}.lock"
CONFIG_SHA=unknown
STAGE=

fail() { echo "FATAL: $*" >&2; exit 1; }
[[ -d "$RAW" ]] || fail "missing source image directory: $RAW"
[[ -f "$CONFIG_TEMPLATE" ]] || fail "missing native config template: $CONFIG_TEMPLATE"
[[ -d "$DEEPERFLY_ROOT" ]] || fail "missing deeperfly checkout: $DEEPERFLY_ROOT"
[[ -x "$DEEPERFLY_BIN" ]] || fail "missing native deeperfly executable: $DEEPERFLY_BIN"

# Refuse a short, ragged, or renamed source.  The staged names intentionally match the
# pilot config's filename = camera_0 ... camera_6 prefix contract.
for (( c=0; c<CAMERAS; c++ )); do
    for (( i=0; i<FRAMES; i++ )); do
        f="$RAW/camera_${c}_img_$(printf '%06d' "$i").jpg"
        [[ -f "$f" ]] || fail "camera $c is missing frame $i: $f"
    done
    mapfile -t files < <(printf '%s\n' "$RAW/camera_${c}_img_"*.jpg)
    (( ${#files[@]} == FRAMES )) || fail "camera $c has ${#files[@]} JPEGs, expected $FRAMES"
done

# A completed result is immutable by default.  FORCE=1 is an explicit reprocessing request;
# it can never remove a lock held by another worker.
if [[ -f "$OUT/status.json" ]] && grep -q '"state": "complete"' "$OUT/status.json" && [[ "$FORCE" != 1 ]]; then
    echo "skip $CONDITION/$SESSION (complete: $OUT/results.h5)"
    exit 0
fi
if [[ -e "$LOCK" ]]; then
    echo "skip $CONDITION/$SESSION (worker lock exists: $LOCK)"
    exit 0
fi
mkdir -p "$(dirname "$LOCK")"
mkdir "$LOCK" || { echo "skip $CONDITION/$SESSION (worker lock won by another job)"; exit 0; }

write_status() {
    # All values written below are fixed strings, paths, or numeric values from validated
    # basenames.  This keeps status.json available even if the native environment lacks Python.
    local state=$1 code=$2 detail=$3
    mkdir -p "$OUT"
    local tmp="$OUT/.status.json.tmp.$$"
    cat >"$tmp" <<EOF
{
  "state": "$state",
  "detail": "$detail",
  "exit_code": $code,
  "condition": "$CONDITION",
  "session": "$SESSION",
  "source_images": "$RAW",
  "staged_frames": $FRAMES,
  "staged_cameras": $CAMERAS,
  "result_h5": "$OUT/results.h5",
  "native_log": "$OUT/deeperfly.run.log",
  "native_log_external": "$RUN_LOG",
  "inspect_log": "$OUT/deeperfly.inspect.log",
  "config": "$OUT/deeperfly_config.toml",
  "config_sha256": "$CONFIG_SHA",
  "deeperfly_env": "$DEEPERFLY_ENV",
  "deeperfly_bin": "$DEEPERFLY_BIN",
  "native_units": "deeperfly_configured_rig_units",
  "tailcycle_reference_scale_mm": $SCALE_MM_PER_UNIT,
  "tailcycle_scale_applied": false,
  "h5_schema": ["pose2d/points", "pose2d/conf", "pose2d/cameras", "bundle_adjustment/cameras", "triangulation/points3d", "triangulation/reproj_error", "skeleton"]
}
EOF
    mv -f "$tmp" "$OUT/status.json"
}

cleanup() {
    local rc=$?
    if [[ "$KEEP_STAGE" == 1 ]]; then
        echo "keeping temporary recording: ${STAGE:-<not-created>}" >&2
    elif [[ -n "${STAGE:-}" ]]; then
        rm -rf -- "$STAGE"
    fi
    rmdir "$LOCK" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT

if [[ -e "$OUT" ]]; then
    [[ "$FORCE" == 1 ]] || fail "output exists but is not marked complete (use FORCE=1): $OUT"
    rm -rf -- "$OUT"
fi
mkdir -p "$OUT" "$STAGE_ROOT" "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/${CONDITION}__${SESSION}.native.log"
STAGE=$(mktemp -d "$STAGE_ROOT/.${CONDITION}__${SESSION}.XXXXXX")

for (( c=0; c<CAMERAS; c++ )); do
    for (( i=0; i<FRAMES; i++ )); do
        name="camera_${c}_img_$(printf '%06d' "$i").jpg"
        ln -s -- "$RAW/$name" "$STAGE/$name"
    done
done
cp -- "$CONFIG_TEMPLATE" "$OUT/deeperfly_config.toml"
CONFIG_SHA=$(sha256sum "$OUT/deeperfly_config.toml" | awk '{print $1}')
cat >"$OUT/run_metadata.toml" <<EOF
# Immutable provenance for the native run. Native coordinates are not rescaled here.
condition = "$CONDITION"
session = "$SESSION"
source_images = "$RAW"
stage_frames = $FRAMES
stage_cameras = $CAMERAS
config = "$OUT/deeperfly_config.toml"
config_sha256 = "$CONFIG_SHA"
native_log_external = "$RUN_LOG"
deeperfly_env = "$DEEPERFLY_ENV"
deeperfly_bin = "$DEEPERFLY_BIN"
native_units = "deeperfly_configured_rig_units"
tailcycle_reference_scale_mm = $SCALE_MM_PER_UNIT
tailcycle_scale_applied = false
EOF

# Run from the native checkout so the shared deeperfly environment is used directly.
# The H5 is intentionally the native results.h5 schema.
# tailcyclenet's environment.  The H5 is intentionally the native results.h5 schema.
set +e
(
    cd "$DEEPERFLY_ROOT" &&
    "$DEEPERFLY_BIN" run "$STAGE" -c "$OUT/deeperfly_config.toml" -o "$OUT" --overwrite
) 2>&1 | tee "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
# Native output management may rewrite config.toml; keep our exact request separately so
# status/provenance remains self-contained even on a failed run.
mkdir -p "$OUT"
cp -- "$CONFIG_TEMPLATE" "$OUT/deeperfly_config.toml"
CONFIG_SHA=$(sha256sum "$OUT/deeperfly_config.toml" | awk '{print $1}')
cp -- "$RUN_LOG" "$OUT/deeperfly.run.log"
cat >"$OUT/run_metadata.toml" <<EOF
# Immutable provenance for the native run. Native coordinates are not rescaled here.
condition = "$CONDITION"
session = "$SESSION"
source_images = "$RAW"
stage_frames = $FRAMES
stage_cameras = $CAMERAS
config = "$OUT/deeperfly_config.toml"
config_sha256 = "$CONFIG_SHA"
native_log_external = "$RUN_LOG"
deeperfly_env = "$DEEPERFLY_ENV"
deeperfly_bin = "$DEEPERFLY_BIN"
native_units = "deeperfly_configured_rig_units"
tailcycle_reference_scale_mm = $SCALE_MM_PER_UNIT
tailcycle_scale_applied = false
EOF
if (( run_rc != 0 )); then
    write_status failed "$run_rc" native_run
    exit "$run_rc"
fi
[[ -s "$OUT/results.h5" ]] || { write_status failed 1 missing_results_h5; exit 1; }

# Inspect is the native schema/900-frame acceptance check.  It catches a truncated
# recording before a converter can mistake a partial H5 for a complete session.
set +e
(
    cd "$DEEPERFLY_ROOT" && "$DEEPERFLY_BIN" inspect "$OUT/results.h5"
) >"$OUT/deeperfly.inspect.log" 2>&1
inspect_rc=$?
set -e
if (( inspect_rc != 0 )); then
    write_status failed "$inspect_rc" inspect
    exit "$inspect_rc"
fi
if ! grep -Eq 'frames:[[:space:]]*900([[:space:]]|$)' "$OUT/deeperfly.inspect.log"; then
    write_status failed 1 wrong_frame_count
    exit 1
fi
if ! grep -Eq 'views:[[:space:]]*7' "$OUT/deeperfly.inspect.log" \
    || ! grep -Eq 'has 3D:[[:space:]]*True' "$OUT/deeperfly.inspect.log"; then
    write_status failed 1 incompatible_h5_schema
    exit 1
fi
write_status complete 0 ok
echo "complete $CONDITION/$SESSION -> $OUT/results.h5"
