"""Accelerator → runtime-version mapping.  One source of truth."""

from __future__ import annotations

RUNTIME_FOR_FAMILY: dict[str, str] = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v6e": "v2-alpha-tpuv6e",
}


def family_of(accel: str) -> str:
    """Return the TPU generation family for an accelerator type string."""
    for prefix in sorted(RUNTIME_FOR_FAMILY, key=len, reverse=True):
        if accel.startswith(prefix):
            return prefix
    raise ValueError(f"unknown accelerator family for {accel!r}")


def runtime_for(accel: str) -> str:
    """Return the gcloud --version string for a given accelerator type."""
    family = family_of(accel)
    return RUNTIME_FOR_FAMILY[family]
