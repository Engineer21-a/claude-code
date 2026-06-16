"""The Numba kernels must compute exactly what the NumPy reference computes."""

import numpy as np
import pytest

from pdeforge import backend


@pytest.mark.skipif(not backend.HAS_NUMBA, reason="Numba backend not installed")
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, np.inf])
def test_weighted_lp_norm_matches_reference(p):
    rng = np.random.default_rng(42)
    error = rng.standard_normal((128, 96))
    weight = rng.random((128, 96))
    fast = backend.weighted_lp_norm(error, weight, p, cell_volume=1e-3)
    ref = backend.reference_weighted_lp_norm(error, weight, p, cell_volume=1e-3)
    assert np.isclose(fast, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not backend.HAS_NUMBA, reason="Numba backend not installed")
def test_cellwise_order_matches_reference():
    rng = np.random.default_rng(7)
    # Include exactly-equal entries to exercise the near-zero/NaN guard.
    f = rng.random(500)
    m = f.copy()
    m[::3] += rng.random(m[::3].shape) * 0.1
    c = m + rng.standard_normal(500) * 0.05
    fast = backend.cellwise_order(f, m, c, r=2.0)
    ref = backend.reference_cellwise_order(f, m, c, r=2.0)
    assert np.allclose(fast, ref, equal_nan=True)
