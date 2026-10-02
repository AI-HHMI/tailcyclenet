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
    train_detector_eval) SCRIPT='' ;;
    eval)              SCRIPT=scripts/eval.py ;;
    *) echo "FATAL: expected infer|train|train_detector|train_detector_eval|eval, got $KIND" >&2
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
if [[ "$KIND" == train_detector_eval ]]; then
    CONFIG=$1; OUT=$2; EVALSPEC=$3
    shift 3
    pixi run python scripts/train_detector.py --config "$CONFIG" --out "$OUT" "$@"
    pixi run python - "$OUT" "$EVALSPEC" <<'PY'
import sys, tomllib
from pathlib import Path
out, spec = Path(sys.argv[1]), tomllib.loads(Path(sys.argv[2]).read_text())
for entry in spec.get('eval', []):
    name, data, split, evalset = (entry[k] for k in ('name', 'data', 'split', 'evalset'))
    extra = entry.get('extra', [])
    for ckpt in (f"detector_it{int(tomllib.loads((out / 'config.toml').read_text())['training']['iters']):06d}.pth", 'detector_it004000.pth'):
        path = out / ckpt
        if not path.exists():
            continue
        dest = out / 'eval' / f'{name}__{path.stem}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        cmd = ['pixi', 'run', 'python', 'scripts/eval_detector.py', '--run', str(out), '--checkpoint', ckpt, '--data', data, '--split', split, '--evalset', evalset, '--out', str(dest), '--boxes', 'keypoints', *extra]
        with dest.with_suffix('.log').open('w') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f'eval {name} {ckpt}: exit {rc} -> {dest}', flush=True)
PY
else
    exec pixi run python "$SCRIPT" "$@"
fi
