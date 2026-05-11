"""One retry loop, one regex, one log format — centralised."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

RETRYABLE_PATTERNS = (
    "no more capacity",
    "Insufficient capacity",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "resourceExhausted",
    "Stockout",
    "currently unavailable",
    "tenant project creation",
    '"code": 8',
    '"code": 10',
    "HttpError",
    "503",
    "504",
    "deadline exceeded",
    "Internal error",
    "already exists",  # treat as success-after-the-fact
)
RETRYABLE_RE = re.compile(
    "|".join(map(re.escape, RETRYABLE_PATTERNS)), re.IGNORECASE
)

Classification = Literal["success", "retryable", "unknown", "fatal", "timeout"]


def classify(returncode: int | None, stderr: str) -> Classification:
    if returncode == 0:
        return "success"
    if returncode is None:
        return "timeout"
    if RETRYABLE_RE.search(stderr):
        return "retryable"
    return "unknown"


@dataclass
class RetryPolicy:
    max_attempts: int = 0  # 0 = infinite
    base_sleep_s: float = 30.0
    max_sleep_s: float = 300.0
    backoff: float = 1.0  # 1.0 = constant, >1 = exponential
    attempt_timeout_s: float | None = None


@dataclass
class AttemptResult:
    ok: bool
    attempt: int
    returncode: int | None
    elapsed_s: float
    stdout: str
    stderr: str
    classification: Classification
    _start_s: float = field(default_factory=time.time, compare=False)

    def stderr_tail(self, n: int = 1) -> str:
        lines = [l for l in self.stderr.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""
