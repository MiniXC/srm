"""Checkpoint I/O for local paths and `gs://` URIs.

Uses `gsutil` for GCS (preinstalled on every TPU VM), falls back to local
filesystem otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol


class Backend(Protocol):
    def is_remote(self, path: str) -> bool: ...
    def save(self, state: Any, path: str) -> None: ...
    def load(self, path: str, map_location: str = "cpu") -> Any: ...
    def exists(self, path: str) -> bool: ...
    def copy(self, src: str, dst: str) -> None: ...


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


class GsutilBackend:
    @staticmethod
    def is_remote(path: str) -> bool:
        return _is_gcs(path)

    @staticmethod
    def save(state: Any, path: str) -> None:
        if _is_gcs(path):
            fd, tmp = tempfile.mkstemp(suffix=".pt")
            path_obj = Path(tmp)
            try:
                import torch

                torch.save(state, tmp)
                _run(["gsutil", "-q", "cp", tmp, path])
            finally:
                path_obj.unlink(missing_ok=True)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            import torch

            torch.save(state, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Any:
        import torch

        if _is_gcs(path):
            fd, tmp = tempfile.mkstemp(suffix=".pt")
            path_obj = Path(tmp)
            try:
                _run(["gsutil", "-q", "cp", path, tmp])
                return torch.load(tmp, map_location=map_location)
            finally:
                path_obj.unlink(missing_ok=True)
        return torch.load(path, map_location=map_location)

    @staticmethod
    def exists(path: str) -> bool:
        if _is_gcs(path):
            try:
                _run(["gsutil", "-q", "stat", path])
                return True
            except subprocess.CalledProcessError:
                return False
        return Path(path).exists()

    @staticmethod
    def copy(src: str, dst: str) -> None:
        if _is_gcs(src) or _is_gcs(dst):
            _run(["gsutil", "-q", "cp", src, dst])
        else:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


_DEFAULT: Backend = GsutilBackend()


def set_backend(b: Backend) -> None:
    global _DEFAULT
    _DEFAULT = b


def save_state(state: Any, path: str) -> None:
    _DEFAULT.save(state, path)


def load_state(path: str, map_location: str = "cpu") -> Any:
    return _DEFAULT.load(path, map_location=map_location)


def exists(path: str) -> bool:
    return _DEFAULT.exists(path)


def copy(src: str, dst: str) -> None:
    _DEFAULT.copy(src, dst)
