"""YAML inventory loader — the single source of truth for pools and config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from srm_tpu.pools import runtime_for


class InventoryError(Exception):
    """Raised when the inventory YAML is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    gcp_project: str
    default_zone: str = "europe-west4-a"
    log_dir: str = ".srm-tpu/logs"
    pid_dir: str = ".srm-tpu/pids"
    remote_repo: str = ""
    remote_branch: str = "main"
    remote_workdir: str = "~/srm"
    tmux_session: str = "srm"


@dataclass(frozen=True)
class Pool:
    name: str
    accel: str
    zone: str
    spot: bool
    instances: int
    runtime: str = ""


@dataclass(frozen=True)
class TorchPin:
    torch: str = "2.7.1"
    torchaudio: str = "2.7.1"
    jax: str = "tpu"
    torchax: str = "*"


@dataclass(frozen=True)
class BootstrapRecipe:
    python: str = "3.11"
    apt: tuple[str, ...] = ()
    torch: TorchPin = field(default_factory=TorchPin)
    project_install: str = "uv sync --extra dev --extra tpu"
    extra_pip: tuple[str, ...] = ()
    smoke_test: str = (
        "import torch, torchax; "
        "t = torch.randn(2, device='jax'); "
        "print(t.device, torchax.__version__)"
    )


@dataclass(frozen=True)
class Inventory:
    project: ProjectConfig
    pools: dict[str, Pool]
    secrets: tuple[str, ...]
    bootstrap: BootstrapRecipe

    @classmethod
    def load(cls, path: Path | None = None) -> Inventory:
        if path is None:
            path = Path("srm-tpu.yaml")

        if not path.exists():
            raise InventoryError(f"inventory file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "project" not in raw:
            raise InventoryError("missing required top-level key: project")

        proj_raw = raw["project"]
        project = ProjectConfig(
            gcp_project=_require(proj_raw, "gcp_project"),
            default_zone=proj_raw.get("default_zone", "europe-west4-a"),
            log_dir=proj_raw.get("log_dir", ".srm-tpu/logs"),
            pid_dir=proj_raw.get("pid_dir", ".srm-tpu/pids"),
            remote_repo=proj_raw.get("remote_repo", ""),
            remote_branch=proj_raw.get("remote_branch", "main"),
            remote_workdir=proj_raw.get("remote_workdir", "~/srm"),
            tmux_session=proj_raw.get("tmux_session", "srm"),
        )

        pools: dict[str, Pool] = {}
        for name, p in raw.get("pools", {}).items():
            if not isinstance(p, dict):
                raise InventoryError(f"pools.{name}: expected a mapping")
            accel = p.get("accel", "")
            instances = p.get("instances", 0)
            if not accel:
                raise InventoryError(f"pools.{name}.accel is required")
            if not isinstance(instances, int) or instances < 1:
                raise InventoryError(
                    f"pools.{name}.instances must be a positive integer, got {instances!r}"
                )
            pools[name] = Pool(
                name=name,
                accel=accel,
                zone=p.get("zone", project.default_zone),
                spot=p.get("spot", True),
                instances=instances,
                runtime=p.get("runtime", "") or runtime_for(accel),
            )

        secrets: tuple[str, ...] = tuple(raw.get("secrets", []))

        boot_raw = raw.get("bootstrap", {})
        torch_raw = boot_raw.get("torch", {})
        bootstrap = BootstrapRecipe(
            python=str(boot_raw.get("python", "3.11")),
            apt=tuple(boot_raw.get("apt", [])),
            torch=TorchPin(
                torch=str(torch_raw.get("torch", "2.7.1")),
                torchaudio=str(torch_raw.get("torchaudio", "2.7.1")),
                jax=str(torch_raw.get("jax", "tpu")),
                torchax=str(torch_raw.get("torchax", "*")),
            ),
            project_install=str(
                boot_raw.get("project_install", "uv sync --extra dev --extra tpu")
            ),
            extra_pip=tuple(boot_raw.get("extra_pip", [])),
            smoke_test=str(
                boot_raw.get(
                    "smoke_test",
                    "import torch, torchax; t = torch.randn(2, device='jax'); "
                    "print(t.device, torchax.__version__)",
                )
            ),
        )

        return cls(
            project=project,
            pools=pools,
            secrets=secrets,
            bootstrap=bootstrap,
        )


def _require(raw: dict, path: str) -> str:
    *parts, key = path.split(".")
    for part in parts:
        raw = raw.get(part, {})  # type: ignore[assignment]
    value = raw.get(key)  # type: ignore[union-attr]
    if not value:
        raise InventoryError(f"{path} is required")
    return str(value)
