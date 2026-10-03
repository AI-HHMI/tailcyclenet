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
LOGS = '/groups/karashchuk/home/karashchukl/logs/tailcyclenet'


def main():
    """Validate capacity, submit the requested arms, and append each job to the manifest.

    `ssh` joins its arguments with spaces and the remote shell re-splits them, so the whole
    remote command is ONE pre-quoted string, run from the repo (job_l4.sh is a relative path).
    The run folder is named by the arm alone (arm names already carry their seed). A dry run
    prints and writes nothing.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('arms', type=Path, help='JSON list of {arm, config, evalspec, seed}')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cap', type=int, default=32, help='project L4 job cap (owner: 32 from 2026-10-03)')
    args = ap.parse_args()
    arms = json.loads(args.arms.read_text())
    listing = subprocess.run(['ssh', 'login2', 'bjobs', '-w'], check=True,
                             capture_output=True, text=True).stdout
    active = sum('gpu_l4' in line for line in listing.splitlines()[1:])
    if active + len(arms) > args.cap:
        raise SystemExit(f'refusing {len(arms)} jobs: {active} project L4 jobs already active')
    manifest = ROOT / 'scratch/detector_transfer/manifest.tsv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        name = arm['arm']
        out = str(RESULTS / name)
        bsub = ['bsub', '-q', 'gpu_l4', '-gpu', 'num=1', '-n', '8', '-R',
                'span[hosts=1] rusage[mem=122880]', '-W', '3:00', '-J', f'tcn-dtx-{name}',
                '-o', f'{LOGS}/dtx-{name}.out', 'bash', 'scripts/job_l4.sh',
                'train_detector_eval', arm['config'], out, arm['evalspec']]
        remote = f'mkdir -p {LOGS} && cd {shlex.quote(str(ROOT))} && {shlex.join(bsub)}'
        cmd = ['ssh', 'login2', remote]
        print(shlex.join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        jobid = result.stdout.split('<', 1)[1].split('>', 1)[0]
        print(f'  -> job {jobid}')
        with manifest.open('a') as stream:
            stream.write(f'{name}\t{arm["config"]}\t{out}\t{arm["evalspec"]}\t{jobid}\t{time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')


if __name__ == '__main__':
    main()
