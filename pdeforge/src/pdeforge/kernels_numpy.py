"""Pure-NumPy reference implementations of the performance-critical kernels.

These are always available (no Numba required) and define the *ground truth*
that the JIT-compiled kernels in :mod:`pdeforge.kernels` are checked against in
``tests/test_backend_parity.py``.  Keeping a readable reference next to the fast
path is deliberate: the performance claim is only worth anything if it is
demonstrably the same computation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["weighted_lp_norm", "cellwise_order"]


def weighted_lp_norm(
    error: np.ndarray, weight: np.ndarray, p: float, cell_volume: float
) -> float:
    """Discrete weighted ``L^p`` grid-function norm.

    Approximates ``(integral w |e|^p dV)^(1/p)`` by midpoint quadrature on a
    uniform grid::

        ( cell_volume * sum_i w_i |e_i|^p )^(1/p)

    ``p = inf`` returns the weighted max-norm ``max_i w_i |e_i|``.
    """
    e = np.abs(error.ravel())
    w = weight.ravel()
    if np.isinf(p):
        return float(np.max(w * e))
    total = cell_volume * np.sum(w * e**p)
    return float(total ** (1.0 / p))


def cellwise_order(
    phi_fine: np.ndarray,
    phi_med: np.ndarray,
    phi_coarse: np.ndarray,
    r: float,
    atol: float = 1e-14,
) -> np.ndarray:
    """Per-cell observed order of accuracy from three co-located solutions.

    For each cell the local order is ``p = ln(|phi_c - phi_m| / |phi_m - phi_f|)
    / ln(r)``.  Cells whose denominator underflows (already converged, so the
    order is undefined) or whose ratio is non-positive are returned as ``nan``.
    """
    f = phi_fine.ravel()
    m = phi_med.ravel()
    c = phi_coarse.ravel()
    out = np.full(f.shape, np.nan, dtype=np.float64)
    num = np.abs(c - m)
    den = np.abs(m - f)
    log_r = np.log(r)
    valid = (den > atol) & (num > atol)
    out[valid] = np.log(num[valid] / den[valid]) / log_r
    return out.reshape(phi_fine.shape)
