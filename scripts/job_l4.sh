#!/usr/bin/env bash
# One tailcyclenet process executed on an LSF L4 compute host. Pattern copied from
# ../comparison/models/deeplabcut/job_l4.sh (dev/plans/identity_bridge_and_reid.md §8.4).
#
#   bsub -q gpu_l4 -gpu "num=1" -n 8 -R "span[hosts=1] rusage[mem=122880]" \
#        -J tcn-infer-<clip> -o ~/logs/tailcyclenet/<clip>.out \
#        bash scripts/job_l4.sh infer --data <session> --run <run> --out <pred-dir>
#
# `KIND` selects the script; everything after it is forwarded verbatim. Pass `--max-ram`
# explicitly on every LSF invocation (CLAUDE.md: an under-detected cgroup/LSF memory limit
# silently re-sizes the reader cache and the block store -- verify the `ram:` line in the first
# job's log before submitting the rest). Never request `-n 16`: the L4 limit is `-n 8` per GPU.
set -euo pipefail

REPO=/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
KIND=$1
shift

case "$KIND" in
    infer)             SCRIPT=scripts/infer.py ;;
    train)             SCRIPT=scripts/train.py ;;
    train_detector)    SCRIPT=scripts/train_detector.py ;;
    train_crop_reid)   SCRIPT=scripts/train_crop_reid.py ;;
    eval)              SCRIPT=scripts/eval.py ;;
    *) echo "FATAL: expected infer|train|train_detector|train_crop_reid|eval, got $KIND" >&2
       exit 2 ;;
esac

cd "$REPO"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true
export PATH="/groups/karashchuk/home/karashchukl/bin:/home/karashchukl@hhmi.org/.pixi/bin:$PATH"
# Unbuffered stdout: a redirected `bsub -o` pipe is fully (not line-) buffered, so a script that
# logs a short line every N iterations can sit invisible in the log for minutes even though it is
# actively running -- a real hang and a buffered-but-alive process are then indistinguishable
# from the log alone. This makes every job's progress line land as soon as it is printed.
export PYTHONUNBUFFERED=1

nvidia-smi
exec pixi run python "$SCRIPT" "$@"
