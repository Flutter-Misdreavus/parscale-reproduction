#!/usr/bin/env bash
# Source from a driver after setting REPO. Values belong to the local machine.
RT="${PARSCALE_RUNTIME_DIR:?Set PARSCALE_RUNTIME_DIR to your prepared runtime directory}"
export CUDA_VISIBLE_DEVICES="${PARSCALE_GPU_UUID:?Set PARSCALE_GPU_UUID to your authorized GPU UUID}"
export PYTHONPATH="$RT/pkgs:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RT/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
