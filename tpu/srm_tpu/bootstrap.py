"""On-VM bootstrap: detect accel, install wheels, smoke-test.

Runs on the TPU VM after `scripts/bootstrap.sh` has set up uv + python3.11.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from srm_tpu.inventory import Inventory
from srm_tpu.pools import runtime_for


def detect_accel() -> str:
    """Detect the TPU accelerator type from the metadata server."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/attributes/accelerator-type",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            accel = resp.read().decode().strip()
            if accel:
                return accel
    except Exception:
        pass
    return "unknown"


def run(cmd: list[str], step: str) -> None:
    print(f"[bootstrap] {step}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


def main(inventory_path: str | None = None) -> None:
    inv = Inventory.load(Path(inventory_path) if inventory_path else None)

    # 1. Detect accelerator.
    accel = detect_accel()
    runtime = runtime_for(accel)
    print(f"[bootstrap] accel={accel} runtime={runtime}")

    # 2. apt-get install.
    apt_packages = inv.bootstrap.apt
    if apt_packages:
        run(["sudo", "apt-get", "install", "-y", *apt_packages], step="apt")

    # 3. Force python3.11 venv (if not already).
    venv_path = Path(".venv")
    if not venv_path.exists():
        py = inv.bootstrap.python
        run(["uv", "venv", "--python", f"python{py}", str(venv_path)], step="venv")

    # 4. Project install.
    if inv.bootstrap.project_install:
        install_parts = inv.bootstrap.project_install.split()
        run(install_parts, step="project_install")

    # 5. Install torchax stack.
    t = inv.bootstrap.torch
    run([
        "uv", "pip", "install",
        f"torch=={t.torch}",
        f"torchaudio=={t.torchaudio}",
        f"jax[{t.jax}]",
        f"torchax{t.torchax}" if t.torchax != "*" else "torchax",
    ], step="torchax_stack")

    # 6. Extra pip deps.
    for dep in inv.bootstrap.extra_pip:
        run(["uv", "pip", "install", dep], step=f"extra_pip:{dep}")

    # 7. Smoke test.
    print(f"[bootstrap] smoke_test: {inv.bootstrap.smoke_test}")
    result = subprocess.run(
        ["uv", "run", "python", "-c", inv.bootstrap.smoke_test],
        capture_output=True, text=True, errors="replace",
    )
    if result.returncode != 0:
        print(f"[bootstrap] SMOKE TEST FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[bootstrap] smoke test passed — {result.stdout.strip()}")
    print("[bootstrap] done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
