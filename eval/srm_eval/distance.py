"""Distance metrics for distribution comparison.

Primary: wasserstein_2_perdim — 1D 2-Wasserstein per feature dimension, averaged.
This is what TTSDS uses internally for 1-D benchmarks.
Also: frechet (Gaussian W2) and sliced wasserstein for diagnostics.
"""

from __future__ import annotations

import numpy as np


def wasserstein_2_perdim(x: np.ndarray, y: np.ndarray) -> float:
    """1-D 2-Wasserstein distance per feature dim, averaged (TTSDS's default).

    When x and y have different sizes, subsamples the larger to match
    the smaller, runs 10 times with np.random.seed(0), and averages.
    """
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return float("inf")

    rng = np.random.RandomState(0)
    distances = np.zeros(x.shape[1])

    for d in range(x.shape[1]):
        vals = np.zeros(10)
        for run in range(10):
            x_d = x[:, d]
            y_d = y[:, d]
            if x.shape[0] > y.shape[0]:
                idx = rng.choice(x.shape[0], n, replace=False)
                x_d = x_d[idx]
            elif y.shape[0] > x.shape[0]:
                idx = rng.choice(y.shape[0], n, replace=False)
                y_d = y_d[idx]
            vals[run] = np.mean((np.sort(x_d) - np.sort(y_d)) ** 2) ** 0.5
        distances[d] = np.mean(vals)

    return float(np.mean(distances))


def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet / Gaussian Wasserstein-2 distance between two (N, D) sets."""
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    diff = mu_x - mu_y
    diff_sq = np.dot(diff, diff)

    # sqrt(sigma_x @ sigma_y)
    prod = sigma_x @ sigma_y
    # Use svd for stable sqrt.
    from scipy import linalg
    s, _ = linalg.eigh(prod)
    s = np.maximum(s, 0)
    sqrt_trace = np.sum(np.sqrt(s))

    fd = diff_sq + np.trace(sigma_x) + np.trace(sigma_y) - 2 * sqrt_trace
    return float(max(fd, 0) ** 0.5)


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 128,
    seed: int = 0,
) -> float:
    """Sliced Wasserstein-1 distance via random 1-D projections."""
    rng = np.random.RandomState(seed)
    d = x.shape[1]
    projections = rng.randn(n_projections, d)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    distances = np.zeros(n_projections)
    for i in range(n_projections):
        px = x @ projections[i]
        py = y @ projections[i]
        n = min(len(px), len(py))
        idx_x = rng.choice(len(px), n, replace=False) if len(px) > n else np.arange(n)
        idx_y = rng.choice(len(py), n, replace=False) if len(py) > n else np.arange(n)
        distances[i] = np.mean(np.abs(np.sort(px[idx_x]) - np.sort(py[idx_y])))

    return float(np.mean(distances))
