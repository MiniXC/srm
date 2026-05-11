"""Test distance metrics on simple known distributions."""

import numpy as np
import pytest
from srm_eval.distance import wasserstein_2_perdim, frechet_distance


def test_wasserstein_identical():
    x = np.random.RandomState(0).randn(100, 5)
    d = wasserstein_2_perdim(x, x)
    assert d == pytest.approx(0.0, abs=0.001)


def test_wasserstein_separated():
    x = np.zeros((100, 3))
    y = np.ones((100, 3)) * 10
    d = wasserstein_2_perdim(x, y)
    assert d == pytest.approx(10.0, abs=0.1)


def test_wasserstein_different_sizes():
    x = np.random.RandomState(0).randn(50, 3)
    y = np.random.RandomState(1).randn(200, 3)
    d = wasserstein_2_perdim(x, y)
    assert d > 0
    assert np.isfinite(d)


def test_frechet_identical():
    x = np.random.RandomState(0).randn(100, 5)
    d = frechet_distance(x, x)
    assert d == pytest.approx(0.0, abs=0.001)


def test_frechet_separated():
    x = np.zeros((100, 5))
    y = np.ones((100, 5)) * 5
    # Frechet: mean diff vector has norm sqrt(5 * 5²) = sqrt(125) ≈ 11.18
    d = frechet_distance(x, y)
    assert d == pytest.approx(11.18, rel=0.01)
