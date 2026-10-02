#!/usr/bin/env python
"""Submit detector transfer arms from a JSON manifest, respecting project L4 capacity."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/groups/karashchuk/home/karashchukl/results/tailcyclenet/detectors/transfer')


def main():
    """Validate capacity, submit the requested arms, and append each job to the manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument('arms', type=Path, help='JSON list of {arm, config, evalspec, seed}')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    arms = json.loads(args.arms.read_text())
    listing = subprocess.run(['ssh', 'login2', 'bjobs', '-w'], check=True,
                             capture_output=True, text=True).stdout
    active = sum('gpu_l4' in line for line in listing.splitlines()[1:])
    if active + len(arms) > 8:
        raise SystemExit(f'refusing {len(arms)} jobs: {active} project L4 jobs already active')
    manifest = ROOT / 'scratch/detector_transfer/manifest.tsv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        name = arm['arm']
        out = str(RESULTS / f'{name}-s{arm["seed"]}')
        cmd = ['ssh', 'login2', 'bsub', '-q', 'gpu_l4', '-gpu', 'num=1', '-n', '8', '-R',
               'span[hosts=1] rusage[mem=122880]', '-W', '3:00', '-J', f'tcn-dtx-{name}',
               '-o', f'~/logs/tailcyclenet/dtx-{name}.out', 'bash', 'scripts/job_l4.sh',
               'train_detector_eval', arm['config'], out, arm['evalspec']]
        print(shlex.join(cmd))
        jobid = 'DRY-RUN'
        if not args.dry_run:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            jobid = result.stdout.strip().split('<')[-1].split('>')[0]
        with manifest.open('a') as stream:
            stream.write(f'{name}\t{arm["config"]}\t{out}\t{arm["evalspec"]}\t{jobid}\t{time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')


if __name__ == '__main__':
    main()
