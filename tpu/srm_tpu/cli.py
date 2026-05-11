"""`srm-tpu` — CLI entry point."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from srm_tpu import daemon
from srm_tpu.provision import poll_create, LaunchSpec, ProvisionResult, provision
from srm_tpu.gcloud import (
    delete_by_filter,
    describe_vm,
    list_vms_all_zones,
    ssh,
)
from srm_tpu.inventory import Inventory, InventoryError, Pool
from srm_tpu.log import Logger
from srm_tpu.retry import RetryPolicy
from srm_tpu.secrets import load_dotenv, select

app = typer.Typer(no_args_is_help=True)


# -- helpers ----------------------------------------------------------------

def _log(json: bool, verbose: bool) -> Logger:
    return Logger(json_mode=json, level=1 if verbose else 0)


def _inv(inventory_path: Optional[str]) -> Inventory:
    try:
        return Inventory.load(Path(inventory_path) if inventory_path else None)
    except InventoryError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(2)


def _resolve_pool(
    pool_name: Optional[str],
    inv: Inventory,
) -> list[Pool]:
    if pool_name:
        pool = inv.pools.get(pool_name)
        if pool is None:
            typer.echo(f"Error: unknown pool '{pool_name}'", err=True)
            raise typer.Exit(2)
        return [pool]
    return list(inv.pools.values())


# -- commands ---------------------------------------------------------------

@app.command()
def pools(
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Print the inventory's pools as a table."""
    inv = _inv(inventory)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="TPU Pools")
    table.add_column("Name", style="cyan")
    table.add_column("Accel")
    table.add_column("Zone")
    table.add_column("Spot")
    table.add_column("Instances")
    table.add_column("Runtime")

    for pool in inv.pools.values():
        table.add_row(
            pool.name,
            pool.accel,
            pool.zone,
            "✓" if pool.spot else "—",
            str(pool.instances),
            pool.runtime,
        )
    console.print(table)


@app.command()
def request(
    pool: Annotated[
        Optional[str], typer.Option("--pool", help="Pool name (default: all pools)")
    ] = None,
    parallel: Annotated[
        int, typer.Option("--parallel", "-n", help="Number of VMs to request")
    ] = 1,
    prefix: Annotated[str, typer.Option("--prefix", help="VM name prefix")] = "srm",
    detached: Annotated[
        bool, typer.Option("--detached", help="Run in background")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print commands without executing")
    ] = False,
    json_logs: Annotated[
        bool, typer.Option("--json-logs", help="Emit JSON records")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Bring up TPU VMs, polling until capacity."""
    inv = _inv(inventory)
    log = _log(json_logs, verbose)
    policy = RetryPolicy()

    pools_list = _resolve_pool(pool, inv)
    project = inv.project.gcp_project

    vm_names: list[str] = []
    for i in range(parallel):
        pool_obj = pools_list[i % len(pools_list)]
        vm_name = f"{prefix}-{pool_obj.name}-{i + 1:02d}"
        vm_names.append(vm_name)

        if detached:
            pid = daemon.spawn_worker(
                vm_name,
                inventory_path=str(inventory or "srm-tpu.yaml"),
                log_dir=Path(inv.project.log_dir),
                pid_dir=Path(inv.project.pid_dir),
                extra_argv=["request", "--pool", pool_obj.name],
            )
            log.info(f"[detached] {vm_name}: pid={pid}")
        else:
            result = poll_create(
                pool_obj, vm_name, project=project, policy=policy, log=log,
                dry_run=dry_run,
            )
            if not result.ok:
                log.error(f"failed to create {vm_name}")
                raise typer.Exit(1)

    if detached:
        log.info(f"spawned {len(vm_names)} workers — check {inv.project.log_dir} for logs")
    else:
        log.info(f"created {len(vm_names)} VM(s) — {', '.join(vm_names)}")


@app.command()
def launch(
    pool: Annotated[str, typer.Option("--pool", help="Pool name")],
    config: Annotated[str, typer.Option("--config", help="Path to experiment config YAML")],
    command: Annotated[str, typer.Option("--command", help="Command to run in tmux")],
    prefix: Annotated[str, typer.Option("--prefix", help="VM name prefix")] = "srm",
    parallel: Annotated[
        int, typer.Option("--parallel", "-n", help="Number of VMs")
    ] = 1,
    detached: Annotated[
        bool, typer.Option("--detached", help="Run in background")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print commands without executing")
    ] = False,
    json_logs: Annotated[
        bool, typer.Option("--json-logs", help="Emit JSON records")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
    inventory_path: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Provision + push env + launch training on TPU VMs."""
    inv = _inv(inventory_path)
    log = _log(json_logs, verbose)
    policy = RetryPolicy()

    secrets = select(load_dotenv(), inv.secrets)

    for i in range(parallel):
        vm_name = f"{prefix}-{i + 1:02d}" if parallel > 1 else prefix
        spec = LaunchSpec(
            vm_name=vm_name,
            pool=pool,
            config=config,
            command=command,
        )

        if detached:
            pid = daemon.spawn_worker(
                vm_name,
                inventory_path=str(inventory_path or "srm-tpu.yaml"),
                log_dir=Path(inv.project.log_dir),
                pid_dir=Path(inv.project.pid_dir),
                extra_argv=["launch", "--pool", pool, "--config", config,
                            "--command", command, "--prefix", vm_name],
            )
            log.info(f"[detached] {vm_name}: pid={pid}")
        else:
            result = provision(
                spec, inventory=inv, secrets=secrets, policy=policy, log=log,
                dry_run=dry_run,
            )
            if result.state == "FAILED":
                log.error(f"provision failed for {vm_name}")
                raise typer.Exit(1)

    if detached:
        log.info(f"spawned {parallel} worker(s) — check {inv.project.log_dir}")


@app.command()
def status(
    pool: Annotated[
        Optional[str], typer.Option("--pool", help="Filter by pool name")
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Show local poll loops + all TPU VMs in known zones."""
    inv = _inv(inventory)
    pools_list = _resolve_pool(pool, inv)
    zones = sorted({p.zone for p in pools_list})
    project = inv.project.gcp_project

    vms = list_vms_all_zones(zones, project=project)

    if not vms:
        typer.echo("(no TPU VMs found)")
    else:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="TPU VMs")
        table.add_column("Name", style="cyan")
        table.add_column("Zone")
        table.add_column("Accel")
        table.add_column("State", style="bold")
        table.add_column("Created")

        for vm in vms:
            name = vm.get("name", "?").split("/")[-1]
            zone = vm.get("zone", "?").split("/")[-1] if "zone" in vm else "?"
            accel = vm.get("acceleratorType", "?").split("/")[-1] if "acceleratorType" in vm else "?"
            state = vm.get("state", "?")
            created = vm.get("createTime", "?")
            state_style = "red" if state in ("PREEMPTED", "TERMINATED") else "green"
            table.add_row(name, zone, accel, f"[{state_style}]{state}[/{state_style}]", created)
        console.print(table)

    # Also show local daemon workers.
    pid_dir = Path(inv.project.pid_dir)
    if pid_dir.exists():
        workers = list(pid_dir.glob("*.pid"))
        if workers:
            typer.echo("\nLocal daemon workers:")
            for w in workers:
                name = w.stem
                ws = daemon.worker_status(name, pid_dir=pid_dir)
                typer.echo(f"  {name}: pid={ws['pid']} alive={ws['alive']}")


@app.command(name="list")
def list_vms(
    pool: Annotated[
        Optional[str], typer.Option("--pool", help="Filter by pool zone")
    ] = None,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """List every TPU VM across known zones."""
    inv = _inv(inventory)
    pools_list = _resolve_pool(pool, inv)
    zones = sorted({p.zone for p in pools_list})
    project = inv.project.gcp_project

    vms = list_vms_all_zones(zones, project=project)
    if not vms:
        typer.echo("(no TPU VMs found)")
        return

    for vm in vms:
        name = vm.get("name", "?").split("/")[-1]
        state = vm.get("state", "?")
        print(f"{name:<30} {state}")


@app.command(name="ssh")
def ssh_cmd(
    name: str,
    command: Annotated[Optional[str], typer.Option("--command")] = None,
    zone: Annotated[Optional[str], typer.Option("--zone")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """SSH into a TPU VM (auto-detects zone)."""
    inv = _inv(inventory)
    project = inv.project.gcp_project

    if zone is None:
        # Auto-detect: find the VM across all known zones.
        all_zones = sorted({p.zone for p in inv.pools.values()})
        zone = None
        for z in all_zones:
            if describe_vm(name, z, project=project):
                zone = z
                break
        if zone is None:
            typer.echo(f"VM '{name}' not found in any known zone", err=True)
            raise typer.Exit(1)

    ssh(name, zone, project=project, command=command, dry_run=dry_run)


@app.command()
def tail(
    name: str,
    follow: Annotated[bool, typer.Option("-f", "--follow")] = False,
    n: Annotated[int, typer.Option("-n")] = 20,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Tail the local log for a daemon worker."""
    inv = _inv(inventory)
    log_path = Path(inv.project.log_dir) / f"{name}.log"
    if not log_path.exists():
        typer.echo(f"no log file at {log_path}", err=True)
        raise typer.Exit(1)

    import subprocess

    cmd = ["tail", "-n", str(n)]
    if follow:
        cmd.append("-F")
    cmd.append(str(log_path))
    subprocess.call(cmd)


@app.command()
def logs(
    name: str,
    remote_path: Annotated[
        str, typer.Option("--remote-path", help="Path to log file on VM")
    ] = "~/srm-run.log",
    zone: Annotated[Optional[str], typer.Option("--zone")] = None,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Tail the in-tmux log on a remote VM."""
    inv = _inv(inventory)
    project = inv.project.gcp_project

    if zone is None:
        all_zones = sorted({p.zone for p in inv.pools.values()})
        for z in all_zones:
            if describe_vm(name, z, project=project):
                zone = z
                break
        if zone is None:
            typer.echo(f"VM '{name}' not found", err=True)
            raise typer.Exit(1)

    ssh(name, zone, project=project, command=f"tail -n 50 -F {remote_path}")


@app.command()
def stop(
    name: str,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Kill a local poll loop (no VM impact)."""
    inv = _inv(inventory)
    pid_dir = Path(inv.project.pid_dir)
    killed = daemon.stop_worker(name, pid_dir=pid_dir)
    if killed:
        typer.echo(f"stopped worker for {name}")
    else:
        typer.echo(f"no running worker found for {name}")


@app.command()
def delete(
    name: Annotated[
        Optional[str], typer.Argument(help="VM name (skip if using --filter)")
    ] = None,
    filter: Annotated[
        Optional[str],
        typer.Option(
            "--filter",
            help="Delete VMs matching criteria: 'state=PREEMPTED' or 'name~regex'",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Delete TPU VMs by name or filter."""
    inv = _inv(inventory)
    log = Logger()
    project = inv.project.gcp_project
    zones = sorted({p.zone for p in inv.pools.values()})

    if filter:
        name_regex = None
        state = None
        for part in filter.split(","):
            part = part.strip()
            if part.startswith("name~"):
                name_regex = part[5:]
            elif part.startswith("state="):
                state = part[6:]

        deleted = delete_by_filter(
            zones,
            project=project,
            name_regex=name_regex,
            state=state,
            log=log,
            dry_run=dry_run,
        )
        typer.echo(f"deleted {len(deleted)} VM(s)")
        return

    if name is None:
        typer.echo("Provide a VM name or --filter", err=True)
        raise typer.Exit(2)

    # Find the zone for this VM.
    zone = None
    for z in zones:
        if describe_vm(name, z, project=project):
            zone = z
            break
    if zone is None:
        typer.echo(f"VM '{name}' not found", err=True)
        raise typer.Exit(1)

    from srm_tpu.gcloud import delete_vm

    delete_vm(name, zone, project=project, log=log, dry_run=dry_run)
    typer.echo(f"deleted {name}")


@app.command()
def teardown(
    pool: Annotated[
        Optional[str], typer.Option("--pool", help="Pool to teardown (default: all)")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Delete all VMs + kill all local poll loops."""
    inv = _inv(inventory)
    log = Logger()
    project = inv.project.gcp_project
    pools_list = _resolve_pool(pool, inv)
    zones = sorted({p.zone for p in pools_list})

    # Delete every VM in these zones.
    deleted = delete_by_filter(zones, project=project, log=log, dry_run=dry_run)
    typer.echo(f"deleted {len(deleted)} VM(s)")

    # Kill local daemon workers.
    pid_dir = Path(inv.project.pid_dir)
    if pid_dir.exists():
        for pidfile in pid_dir.glob("*.pid"):
            daemon.stop_worker(pidfile.stem, pid_dir=pid_dir)
        typer.echo("stopped all local workers")


@app.command()
def bake(
    vm_name: Annotated[Optional[str], typer.Option("--vm-name")] = None,
    inventory: Annotated[
        Optional[str], typer.Option("--inventory", help="Path to srm-tpu.yaml")
    ] = None,
) -> None:
    """Run the bootstrap recipe on a VM."""
    if vm_name:
        inv = _inv(inventory)
        project = inv.project.gcp_project
        # Find zone.
        all_zones = sorted({p.zone for p in inv.pools.values()})
        zone = None
        for z in all_zones:
            if describe_vm(vm_name, z, project=project):
                zone = z
                break
        if zone is None:
            typer.echo(f"VM '{vm_name}' not found", err=True)
            raise typer.Exit(1)

        ssh(
            vm_name, zone, project=project,
            command="cd ~/srm && bash scripts/bootstrap.sh",
        )
    else:
        # Run locally (e.g. already on the VM).
        from srm_tpu.bootstrap import main as bootstrap_main

        bootstrap_main(inventory)


@app.command()
def run(
    command: Annotated[str, typer.Option("--command", help="Command to run")],
    config: Annotated[Optional[str], typer.Option("--config")] = None,
    retry_on_preempt: Annotated[
        bool, typer.Option("--retry-on-preempt")
    ] = False,
) -> None:
    """On-VM entry point: set PJRT_DEVICE, enable torchax, run command."""
    import os

    os.environ.setdefault("PJRT_DEVICE", "TPU")

    if config:
        command = command.replace("{config}", config)

    attempt = 0
    max_retries = 50 if retry_on_preempt else 1

    while attempt < max_retries:
        attempt += 1
        print(f"[srm-tpu run] attempt {attempt}: {command}")
        rc = subprocess_call(command)
        if rc == 0:
            break
        if rc == 130 and retry_on_preempt:
            print(f"[srm-tpu run] preempted (rc=130); retrying in 30s")
            time.sleep(30)
        else:
            print(f"[srm-tpu run] exit {rc}; retrying in 30s")
            time.sleep(30)

    if attempt >= max_retries:
        print(f"[srm-tpu run] gave up after {max_retries} attempts")
        sys.exit(1)


def subprocess_call(cmd: str) -> int:
    import subprocess as sp

    return sp.call(cmd, shell=True)


if __name__ == "__main__":
    app()
