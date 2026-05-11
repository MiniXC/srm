"""Background worker daemonisation — replaces `nohup bash -c`."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def spawn_worker(
    vm_name: str,
    *,
    inventory_path: str,
    log_dir: Path,
    pid_dir: Path,
    extra_argv: list[str] | None = None,
) -> int:
    """Start a detached worker process and record its pidfile.

    The worker re-invokes `srm-tpu <extra_argv>`.

    Returns the pid of the spawned child.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{vm_name}.log"
    pid_path = pid_dir / f"{vm_name}.pid"

    argv = [
        sys.executable, "-m", "srm_tpu.daemon", "worker", vm_name,
        "--inventory", inventory_path,
    ]
    if extra_argv:
        argv.extend(extra_argv)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            argv,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    pid_path.write_text(str(proc.pid))
    return proc.pid


def stop_worker(vm_name: str, *, pid_dir: Path) -> bool:
    """Kill the process group for a daemon worker by pidfile.

    Returns True if killed.
    """
    pid_path = pid_dir / f"{vm_name}.pid"
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    pid_path.unlink(missing_ok=True)
    return True


def worker_status(vm_name: str, *, pid_dir: Path) -> dict:
    """Return {'pid': int|None, 'alive': bool} for a daemon worker."""
    pid_path = pid_dir / f"{vm_name}.pid"
    if not pid_path.exists():
        return {"pid": None, "alive": False}
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return {"pid": None, "alive": False}
    try:
        os.kill(pid, 0)
        return {"pid": pid, "alive": True}
    except OSError:
        return {"pid": pid, "alive": False}


def _daemon_worker() -> None:
    """Entry point for `python -m srm_tpu.daemon worker <vm-name> --inventory ...`.

    Re-dispatches to `srm-tpu <extra_argv>` via the CLI.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("vm_name")
    parser.add_argument("--inventory", default="srm-tpu.yaml")
    args, extra = parser.parse_known_args()

    from srm_tpu.cli import app

    sys.argv = ["srm-tpu"] + extra
    app()


if __name__ == "__main__":
    _daemon_worker()
