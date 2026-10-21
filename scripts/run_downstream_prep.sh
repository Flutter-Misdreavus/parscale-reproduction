#!/usr/bin/env bash
# Fixed preparation driver for parscale-20261021-downstream-prep.
# Usage: scripts/run_downstream_prep.sh <run_dir> <prepare|smoke>
set -euo pipefail
RUN_DIR="$(realpath "${1:?run directory}")"
MODE="${2:?prepare|smoke}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/runtime_env.sh"
export PYTHONPATH="${PARSCALE_RUNTIME_DIR:?}/downstream-pkgs:${PARSCALE_TORCH_SITE:-}:$PYTHONPATH"
cd "$REPO"
test -f "$RUN_DIR/task.json"
mkdir -p "$RUN_DIR/raw"
if [ "$MODE" = "smoke" ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
else
  export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
fi
export TOKENIZERS_PARALLELISM=false
git rev-parse HEAD | tee "$RUN_DIR/raw/${MODE}.run.sha"
git status --porcelain | tee "$RUN_DIR/raw/${MODE}.git-status"
python3 scripts/downstream_evaluation.py --run-dir "$RUN_DIR" --mode "$MODE" 2>&1 | tee "$RUN_DIR/raw/${MODE}.log"
