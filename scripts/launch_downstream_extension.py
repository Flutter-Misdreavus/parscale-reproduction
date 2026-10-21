#!/usr/bin/env python3
"""Start one bounded scoring service, independent of the executor's lifetime."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    task = json.loads((run / 'task.json').read_text())
    if run.name != 'parscale-20261021-downstream-rest' or task['extension_protocol'] != 'remaining-three-v1':
        raise RuntimeError('Unexpected fixed task')
    if (ROOT / 'experiments' / run.name / 'task.json').read_bytes() != (run / 'task.json').read_bytes():
        raise RuntimeError('Frozen task mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit before launching')
    unit = run.name + '-score.service'
    env_keys = ['PARSCALE_RUNTIME_DIR', 'PARSCALE_TORCH_SITE', 'PARSCALE_GPU_UUID']
    for key in env_keys:
        if not os.environ.get(key):
            raise RuntimeError('Missing runtime environment: ' + key)
    end = datetime.fromisoformat(task['scoring_deadline_utc']).timestamp() + 60
    runtime = min(31800, int(end - datetime.now(timezone.utc).timestamp()))
    if runtime < task['minimum_scoring_start_seconds'] + 900:
        raise RuntimeError('Insufficient time for verification and complete matrix')
    (run / 'score-launch.lock').mkdir()
    (run / 'raw').mkdir(exist_ok=True)
    command = ['systemd-run', '--user', '--unit', unit,
               '--property', 'RuntimeMaxSec=' + str(runtime),
               '--property', 'TimeoutStopSec=15', '--property', 'KillMode=control-group',
               '--property', 'WorkingDirectory=' + str(ROOT),
               '--property', 'StandardOutput=append:' + str(run / 'raw/score-service.log'),
               '--property', 'StandardError=append:' + str(run / 'raw/score-service.log')]
    for key in env_keys:
        command += ['--setenv', key + '=' + os.environ[key]]
    command += [sys.executable, str(ROOT / 'scripts/downstream_extension_service.py'), str(run)]
    # The launch lock is retained even on failure: no implicit retries.
    subprocess.run(command, check=True, timeout=20)
    receipt = {'run_id': run.name, 'unit': unit, 'runtime_max_seconds': runtime,
               'launched_at': datetime.now(timezone.utc).isoformat(),
               'head_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'independent_of_executor': True}
    (run / 'score-service.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
