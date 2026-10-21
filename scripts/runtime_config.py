"""Resolve machine-specific values without publishing infrastructure identity."""
import os


def configured_gpu_uuid(recorded_uuid):
    if recorded_uuid != "GPU-REDACTED":
        return recorded_uuid
    value = os.environ.get("PARSCALE_GPU_UUID", "")
    if not value.startswith("GPU-") or value == "GPU-REDACTED":
        raise RuntimeError("Set PARSCALE_GPU_UUID to the GPU authorized for this new run")
    return value
