""".env loader with whitelist enforcement."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


class SecretError(Exception):
    """Raised when a required secret is missing from .env."""


def load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """Parse a KEY=VALUE .env file.  Ignores blank lines and comments."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env[key] = value
    return env


def select(
    env: Mapping[str, str],
    whitelist: Iterable[str],
) -> dict[str, str]:
    """Return only the keys in `whitelist`. Raise SecretError if any are missing."""
    result: dict[str, str] = {}
    missing: list[str] = []
    for key in whitelist:
        value = env.get(key, os.environ.get(key))
        if value:
            result[key] = value
        else:
            missing.append(key)
    if missing:
        raise SecretError(
            f"secrets {missing!r} are listed in the inventory but were not found "
            f"in .env or the environment"
        )
    return result


def write_env_file(env: Mapping[str, str]) -> Path:
    """Write env vars to a temporary file with mode 0600."""
    fd, path = tempfile.mkstemp(suffix=".env", prefix="srm_env_")
    with os.fdopen(fd, "w") as f:
        for key, value in env.items():
            f.write(f"{key}={value}\n")
    os.chmod(path, 0o600)
    return Path(path)
