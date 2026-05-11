"""Correlation: Pearson/Spearman with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def correlate(
    distances: pd.DataFrame,
    mos: pd.DataFrame,
    *,
    method: Literal["pearson", "spearman"] = "spearman",
    bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-(year, subtask) correlation between Wasserstein distance and MOS.

    distances: cols [year, subtask, system, distance]
    mos:       cols [year, subtask, system, mean_mos]
    """
    merged = distances.merge(mos, on=["year", "subtask", "system"])

    rows = []
    for (year, subtask), grp in merged.groupby(["year", "subtask"]):
        if len(grp) < 3:
            continue  # need at least 3 systems

        x = grp["distance"].values
        y = grp["mean_mos"].values

        rho = _corr(x, y, method)
        ci_lo, ci_hi = _bootstrap_ci(x, y, method, bootstrap, seed)

        rows.append({
            "year": year,
            "subtask": subtask,
            "n_systems": len(grp),
            "rho": rho,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    return pd.DataFrame(rows)


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    # Spearman: correlate ranks.
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_vals = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_vals[i] = _corr(x[idx], y[idx], method)
    lo = np.percentile(boot_vals, 2.5)
    hi = np.percentile(boot_vals, 97.5)
    return float(lo), float(hi)
