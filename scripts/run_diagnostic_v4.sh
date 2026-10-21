#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?run directory}")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RT=/path/to/parscale-runtime
cd "$REPO"
test -f "$RUN_DIR/task.json"
# mkdir is atomic: no second execution may silently replace the first attempt.
mkdir "$RUN_DIR/diagnostic-once.lock"
mkdir -p "$RUN_DIR/raw"
git rev-parse HEAD > "$RUN_DIR/diagnostic-once.lock/run.sha"
export CUDA_VISIBLE_DEVICES=GPU-REDACTED
export PYTHONPATH="$RT/pkgs:/path/to/python/site-packages:$REPO"
export HF_HOME="$RT/hf" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EVAL_RUN_DIR="$RUN_DIR" PARSCALE_SYSTEMD_UNIT=parscale-cache-diagnostic-v4.service
exec timeout --signal=TERM --kill-after=5 895 python3 scripts/cache_diagnostic_v2.py \
  --run-dir "$RUN_DIR" --phase all --record-utilization-only > "$RUN_DIR/raw/diagnostic.log" 2>&1
