"""Typed wrappers around `gcloud compute tpus tpu-vm ...`.

Every public function takes `dry_run: bool = False`.  When True, the command
is logged and a stub is returned without execution.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable

from srm_tpu.inventory import Pool
from srm_tpu.log import Logger
from srm_tpu.retry import AttemptResult, classify

_GCLOUD = "gcloud"


def _run(
    argv: list[str],
    *,
    log: Logger,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    log.command(argv)
    if dry_run:
        # Return a stub success.
        return subprocess.CompletedProcess(argv, 0, stdout="[dry-run]\n", stderr="")
    return subprocess.run(argv, capture_output=True, text=True, errors="replace")


def create_vm(
    pool: Pool,
    name: str,
    *,
    project: str,
    log: Logger,
    dry_run: bool = False,
) -> AttemptResult:
    argv = [
        _GCLOUD, "compute", "tpus", "tpu-vm", "create", name,
        f"--project={project}",
        f"--zone={pool.zone}",
        f"--accelerator-type={pool.accel}",
        f"--version={pool.runtime}",
    ]
    if pool.spot:
        argv.append("--spot")

    t0 = time.monotonic()
    proc = _run(argv, log=log, dry_run=dry_run)
    elapsed = time.monotonic() - t0

    result = AttemptResult(
        ok=proc.returncode == 0,
        attempt=0,  # caller fills
        returncode=proc.returncode,
        elapsed_s=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
        classification=classify(proc.returncode, proc.stderr),
    )
    log.attempt(result)
    return result


def describe_vm(
    name: str,
    zone: str,
    *,
    project: str,
) -> dict | None:
    """Return the JSON description of a VM, or None if it doesn't exist."""
    proc = subprocess.run(
        [
            _GCLOUD, "compute", "tpus", "tpu-vm", "describe", name,
            f"--project={project}", f"--zone={zone}",
            "--format=json",
        ],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


def list_vms(
    zone: str,
    *,
    project: str,
) -> list[dict]:
    """List all TPU VMs in a zone."""
    proc = subprocess.run(
        [
            _GCLOUD, "compute", "tpus", "tpu-vm", "list",
            f"--project={project}", f"--zone={zone}",
            "--format=json",
        ],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return []


def list_vms_all_zones(
    zones: Iterable[str],
    *,
    project: str,
) -> list[dict]:
    """List TPU VMs across all provided zones, sorted by zone then name."""
    results: list[dict] = []
    for zone in zones:
        for vm in list_vms(zone, project=project):
            results.append(vm)
    results.sort(key=lambda v: (v.get("name", "").split("/")[-1]))
    return results


def delete_vm(
    name: str,
    zone: str,
    *,
    project: str,
    log: Logger,
    dry_run: bool = False,
) -> AttemptResult:
    argv = [
        _GCLOUD, "compute", "tpus", "tpu-vm", "delete", name,
        f"--project={project}", f"--zone={zone}", "--quiet",
    ]
    t0 = time.monotonic()
    proc = _run(argv, log=log, dry_run=dry_run)
    elapsed = time.monotonic() - t0
    return AttemptResult(
        ok=proc.returncode == 0,
        attempt=0,
        returncode=proc.returncode,
        elapsed_s=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
        classification=classify(proc.returncode, proc.stderr),
    )


def delete_by_filter(
    zones: Iterable[str],
    *,
    project: str,
    name_regex: str | None = None,
    state: str | None = None,
    log: Logger,
    dry_run: bool = False,
) -> list[str]:
    """Delete VMs matching a name regex and/or state across zones.

    Returns the list of VM names that were deleted (or would be, in dry-run).
    """
    deleted: list[str] = []
    name_pat = re.compile(name_regex) if name_regex else None

    for zone in zones:
        for vm in list_vms(zone, project=project):
            vm_name = vm.get("name", "").split("/")[-1]
            vm_state = vm.get("state", "")
            if name_pat and not name_pat.search(vm_name):
                continue
            if state and vm_state.lower() != state.lower():
                continue

            log.info(f"deleting {vm_name} (zone={zone}, state={vm_state})")
            delete_vm(vm_name, zone, project=project, log=log, dry_run=dry_run)
            deleted.append(vm_name)
    return deleted


def ssh(
    name: str,
    zone: str,
    *,
    project: str,
    command: str | None = None,
    worker: str = "all",
    dry_run: bool = False,
) -> int:
    argv = [
        _GCLOUD, "compute", "tpus", "tpu-vm", "ssh", name,
        f"--project={project}", f"--zone={zone}", f"--worker={worker}",
    ]
    if command:
        argv += ["--command", command]

    if dry_run:
        print(f"[dry-run] {' '.join(argv)}")
        return 0
    return subprocess.call(argv)


def scp(
    local: Path,
    remote: str,
    name: str,
    zone: str,
    *,
    project: str,
    worker: str = "all",
    log: Logger,
    dry_run: bool = False,
) -> int:
    argv = [
        _GCLOUD, "compute", "tpus", "tpu-vm", "scp",
        str(local), f"{name}:{remote}",
        f"--project={project}", f"--zone={zone}", f"--worker={worker}",
    ]
    log.command(argv)
    if dry_run:
        return 0
    return subprocess.call(argv)
