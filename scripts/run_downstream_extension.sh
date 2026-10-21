#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?run directory}")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/runtime_env.sh"
export PYTHONPATH="$PARSCALE_RUNTIME_DIR/downstream-pkgs:${PARSCALE_TORCH_SITE:?}:$PYTHONPATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"
mkdir -p "$RUN_DIR/raw"
git rev-parse HEAD > "$RUN_DIR/raw/full.run.sha"
python3 scripts/supervised_downstream_extension.py --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/raw/full.log"
