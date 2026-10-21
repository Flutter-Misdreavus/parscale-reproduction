#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?run directory}")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/runtime_env.sh"
cd "$REPO"
test -f "$RUN_DIR/task.json"
mkdir "$RUN_DIR/full-once.lock"
mkdir -p "$RUN_DIR/raw"
git rev-parse HEAD > "$RUN_DIR/full-once.lock/run.sha"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EVAL_RUN_DIR="$RUN_DIR"
exec > "$RUN_DIR/raw/full.log" 2>&1
python3 scripts/verify_models.py --run-dir "$RUN_DIR"
python3 scripts/full_evaluation.py --run-dir "$RUN_DIR" --validation-only \
  --reference-validation "$REPO/runs/parscale-20261021-cache-diagnostic-v4/diagnostics.json"
python3 scripts/full_evaluation.py --run-dir "$RUN_DIR"
