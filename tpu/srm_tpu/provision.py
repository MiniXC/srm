"""Provisioning pipeline: poll_create → wait_ready → push_env → launch_run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from srm_tpu.gcloud import (
    create_vm,
    delete_vm,
    describe_vm,
    scp,
    ssh,
)
from srm_tpu.inventory import Inventory, Pool
from srm_tpu.log import Logger
from srm_tpu.retry import AttemptResult, RetryPolicy
from srm_tpu.secrets import write_env_file

LAUNCH_SCRIPT = r"""set -euo pipefail
if [[ ! -d {workdir}/.git ]]; then
    git clone --branch {branch} {repo} {workdir}
fi
cd {workdir}
git fetch --quiet origin
git checkout {branch}
git pull --ff-only
install -m 600 ~/.env.srm {workdir}/.env
sudo apt-get install -y tmux >/dev/null 2>&1 || true
if tmux has-session -t {tmux} 2>/dev/null; then
    echo "tmux session {tmux} already running; not relaunching"
else
    tmux new-session -d -s {tmux} "{command} 2>&1 | tee -a {log}"
fi
tmux ls
"""


@dataclass(frozen=True)
class LaunchSpec:
    """Everything that varies between `srm-tpu launch` invocations."""

    vm_name: str
    pool: str  # resolves in inventory
    config: str = ""
    command: str = ""
    retry_on_exit_codes: tuple[int, ...] = (130,)
    max_retries: int = 50
    env: Mapping[str, str] | None = None


@dataclass
class ProvisionResult:
    vm_name: str
    pool: str
    state: Literal["LAUNCHED", "FAILED", "DRY_RUN"]
    attempts: int
    elapsed_s: float
    log_path: Path | None = None


def poll_create(
    pool: Pool,
    name: str,
    *,
    project: str,
    policy: RetryPolicy,
    log: Logger,
    dry_run: bool = False,
) -> AttemptResult:
    """Repeatedly try to create a TPU VM until success or exhaustion."""
    attempt = 0
    t0 = time.monotonic()

    while True:
        # Check if the VM already exists (e.g. from a prior timed-out attempt).
        existing = describe_vm(name, pool.zone, project=project)
        if existing and existing.get("state") not in ("TERMINATED",):
            log.info(f"{name} already exists (state={existing.get('state')}) — skipping create")
            return AttemptResult(
                ok=True, attempt=attempt, returncode=0, elapsed_s=0,
                stdout="", stderr="", classification="success",
            )

        attempt += 1
        result = create_vm(pool, name, project=project, log=log, dry_run=dry_run)
        result.attempt = attempt

        if result.ok or result.classification == "success":
            return result

        if policy.max_attempts and attempt >= policy.max_attempts:
            log.error(f"exhausted after {attempt} attempts in pool {pool.name}")
            return result

        # Sleep before retry.
        delay = min(policy.base_sleep_s * (policy.backoff ** (attempt - 1)), policy.max_sleep_s)
        log.debug(f"sleeping {delay:.0f}s before attempt {attempt + 1}")
        if not dry_run:
            time.sleep(delay)


def wait_ready(
    name: str,
    zone: str,
    *,
    project: str,
    poll_interval_s: float = 15.0,
    log: Logger,
) -> Literal["READY", "PREEMPTED", "TERMINATED", "FAILED", "UNKNOWN"]:
    """Poll until the VM is READY or hits a terminal state."""
    while True:
        vm = describe_vm(name, zone, project=project)
        if vm is None:
            log.warn(f"{name}: VM not found during wait_ready")
            return "UNKNOWN"

        state = vm.get("state", "UNKNOWN")
        log.info(f"{name}: state={state}")
        if state == "READY":
            return "READY"
        if state in ("PREEMPTED", "TERMINATED", "FAILED"):
            return state

        time.sleep(poll_interval_s)


def push_env(
    name: str,
    zone: str,
    env: Mapping[str, str],
    *,
    project: str,
    remote_path: str = "~/.env.srm",
    log: Logger,
    dry_run: bool = False,
) -> None:
    """SCP a secrets file onto the VM."""
    tmp = write_env_file(env)
    try:
        log.info(f"pushing env ({', '.join(env)}) to {name}")
        scp(tmp, remote_path, name, zone, project=project, log=log, dry_run=dry_run)
    finally:
        tmp.unlink(missing_ok=True)


def launch_run(
    name: str,
    zone: str,
    *,
    project: str,
    inventory: Inventory,
    spec: LaunchSpec,
    log: Logger,
    dry_run: bool = False,
) -> None:
    """SSH in, clone/pull repo, copy env, start tmux session. Idempotent."""
    cfg = inventory.project
    rendered_command = spec.command.replace("{config}", spec.config)
    script = LAUNCH_SCRIPT.format(
        workdir=cfg.remote_workdir,
        branch=cfg.remote_branch,
        repo=cfg.remote_repo,
        tmux=cfg.tmux_session,
        command=rendered_command,
        log=f"{cfg.remote_workdir}-run.log",
    )
    log.info(f"launching tmux session '{cfg.tmux_session}' on {name}")
    ssh(name, zone, project=project, command=script, dry_run=dry_run)


def provision(
    spec: LaunchSpec,
    *,
    inventory: Inventory,
    secrets: Mapping[str, str],
    policy: RetryPolicy | None = None,
    log: Logger | None = None,
    dry_run: bool = False,
) -> ProvisionResult:
    """End-to-end: poll_create → wait_ready → push_env → launch_run."""
    if log is None:
        log = Logger()
    if policy is None:
        policy = RetryPolicy()

    pool = inventory.pools.get(spec.pool)
    if pool is None:
        raise ValueError(f"unknown pool: {spec.pool!r}")

    project = inventory.project.gcp_project
    t0 = time.monotonic()

    if dry_run:
        log.info(f"[dry-run] would provision {spec.vm_name} in pool {pool.name}")
        return ProvisionResult(
            vm_name=spec.vm_name, pool=pool.name,
            state="DRY_RUN", attempts=0, elapsed_s=0,
        )

    # 1. Create the VM (poll until capacity).
    create_result = poll_create(
        pool, spec.vm_name, project=project, policy=policy, log=log, dry_run=dry_run,
    )
    if not create_result.ok:
        log.error(f"failed to create {spec.vm_name}: {create_result.stderr_tail()}")
        return ProvisionResult(
            vm_name=spec.vm_name, pool=pool.name,
            state="FAILED", attempts=create_result.attempt,
            elapsed_s=time.monotonic() - t0,
        )

    # 2. Wait for READY.  If preempted mid-provision, delete + retry.
    while True:
        state = wait_ready(spec.vm_name, pool.zone, project=project, log=log)
        if state == "READY":
            break
        if state in ("PREEMPTED", "TERMINATED", "FAILED"):
            log.event("preempted", vm_name=spec.vm_name, state=state)
            delete_vm(spec.vm_name, pool.zone, project=project, log=log)
            time.sleep(10)
            create_result = poll_create(
                pool, spec.vm_name, project=project, policy=policy, log=log,
            )
            if not create_result.ok:
                return ProvisionResult(
                    vm_name=spec.vm_name, pool=pool.name,
                    state="FAILED", attempts=create_result.attempt,
                    elapsed_s=time.monotonic() - t0,
                )

    # 3. Push secrets.
    push_env(spec.vm_name, pool.zone, secrets, project=project, log=log, dry_run=dry_run)

    # 4. Launch the training run inside tmux.
    launch_run(
        spec.vm_name, pool.zone, project=project,
        inventory=inventory, spec=spec, log=log, dry_run=dry_run,
    )

    log.info(f"DONE — {spec.vm_name} is running in pool {pool.name}")
    return ProvisionResult(
        vm_name=spec.vm_name, pool=pool.name,
        state="LAUNCHED", attempts=create_result.attempt,
        elapsed_s=time.monotonic() - t0,
    )
