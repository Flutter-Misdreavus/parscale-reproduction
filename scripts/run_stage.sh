#!/usr/bin/env bash
# Reproducible driver for the ParScale smoke stage.
#
# Usage: scripts/run_stage.sh <run_id> <run_dir> [smoke|full]
# Requires the pinned checkpoints to have been pre-fetched into the HF cache
# (see the download command recorded in the run report/log).
set -euo pipefail

RUN_ID="${1:?run_id}"
RUN_DIR="${2:?run_dir}"
STAGE="${3:-smoke}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/runtime_env.sh"

export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export TOKENIZERS_PARALLELISM=false
export EVAL_RUN_DIR="$RUN_DIR"

mkdir -p "$RUN_DIR/raw/logs"
cd "$REPO"

echo "[stage] git HEAD $(git rev-parse HEAD) status=$(git status --porcelain | wc -l) dirty-files" | tee "$RUN_DIR/raw/logs/stage.log"

/usr/bin/time -v python3 scripts/fit_scaling.py \
  --out "$RUN_DIR/fit-results.json" \
  --run-id "$RUN_ID" 2>&1 | tee "$RUN_DIR/raw/logs/fit.log"

/usr/bin/time -v python3 scripts/evaluate.py \
  --stage "$STAGE" \
  --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/raw/logs/evaluate.log"
