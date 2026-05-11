"""Logging: one logger, two formats (human + JSON lines)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any, TextIO


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _human(fields: dict[str, Any]) -> str:
    ts = fields.get("ts", "")
    kind = fields.get("kind", "?")
    parts = [f"[{ts}]"]
    if "vm_name" in fields:
        parts.append(f"[{fields['vm_name']}]")
    if kind == "attempt":
        parts.append(f"attempt={fields.get('attempt', '?')}")
        parts.append(f"cls={fields.get('classification', '?')}")
        if fields.get("returncode") is not None:
            parts.append(f"rc={fields['returncode']}")
        parts.append(f"dt={fields.get('elapsed_s', '?')}s")
        tail = fields.get("stderr_tail", "")
        if tail:
            parts.append(f'last="{tail[:80]}"')
    elif kind == "command":
        parts.append(" ".join(fields.get("argv", [])))
    elif kind == "error":
        parts.append(f"ERROR: {fields.get('message', '')}")
    else:
        parts.append(str(fields))
    return " ".join(parts)


@dataclass
class Logger:
    json_mode: bool = False
    level: int = 0  # 0=INFO, 1=DEBUG
    out: TextIO | None = None

    def __post_init__(self) -> None:
        if self.out is None:
            self.out = sys.stderr

    # -- structured ---------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        self._emit({"kind": kind, **fields})

    def command(self, argv: list[str]) -> None:
        self._emit({"kind": "command", "argv": argv})

    def attempt(self, result: "AttemptResult", **fields: Any) -> None:  # noqa: F821
        self._emit({
            "kind": "attempt",
            "attempt": result.attempt,
            "ok": result.ok,
            "returncode": result.returncode,
            "elapsed_s": round(result.elapsed_s, 2),
            "classification": result.classification,
            "stderr_tail": result.stderr_tail(),
            **fields,
        })

    def error(self, msg: str, exc: BaseException | None = None) -> None:
        fields: dict[str, Any] = {"kind": "error", "message": msg}
        if exc is not None:
            fields["exception"] = repr(exc)
        self._emit(fields)

    # -- plain --------------------------------------------------------------

    def info(self, msg: str) -> None:
        self._print(f"[INFO] {msg}")

    def debug(self, msg: str) -> None:
        if self.level >= 1:
            self._print(f"[DEBUG] {msg}")

    def warn(self, msg: str) -> None:
        self._print(f"[WARN] {msg}")

    # -- internals ----------------------------------------------------------

    def _emit(self, fields: dict[str, Any]) -> None:
        fields.setdefault("ts", _now())
        if self.json_mode:
            self._print(json.dumps(fields, default=str))
        else:
            self._print(_human(fields))

    def _print(self, line: str) -> None:
        if self.out is not None:
            print(line, file=self.out, flush=True)
