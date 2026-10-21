#!/usr/bin/env python3
"""Run the scoring command in its own bounded systemd cgroup and record exit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]).resolve()
started = datetime.now(timezone.utc)
code = subprocess.call(['bash', str(root/'scripts/run_downstream_extension.sh'), str(run)], cwd=root)
record = {'run_id': run.name, 'command_exit_code': code,
          'started_at': started.isoformat(), 'finished_at': datetime.now(timezone.utc).isoformat()}
path = run / 'score-service-exit.json'
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps(record, indent=2)+'\n')
tmp.replace(path)
raise SystemExit(code)
