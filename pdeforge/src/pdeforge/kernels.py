"""Numba-JIT implementations of the performance-critical kernels.

These mirror :mod:`pdeforge.kernels_numpy` exactly but are compiled to parallel
machine code with ``@njit(parallel=True)``.  They are worth JIT-compiling
because both are *reductions with branchy per-element logic* (near-zero guards,
sign handling) that vectorise awkwardly and run slowly as Python loops -- the
sweet spot for Numba.  ``benchmarks/bench_kernels.py`` reports the measured
speedup and asserts bit-for-bit agreement with the reference path.

Importing this module fails cleanly if Numba is unavailable; callers should go
through :mod:`pdeforge.backend`, which falls back to the NumPy reference.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["weighted_lp_norm", "cellwise_order"]


@njit(cache=True, parallel=True, fastmath=False)
def _lp_sum(error: np.ndarray, weight: np.ndarray, p: float) -> float:
    total = 0.0
    for i in prange(error.size):
        total += weight[i] * abs(error[i]) ** p
    return total


@njit(cache=True, parallel=True, fastmath=False)
def _linf(error: np.ndarray, weight: np.ndarray) -> float:
    best = 0.0
    for i in prange(error.size):
        # ``best = max(best, v)`` is the form Numba recognises as a parallel
        # max-reduction; a plain ``if v > best`` is not and silently misbehaves.
        best = max(best, weight[i] * abs(error[i]))
    return best


def weighted_lp_norm(
    error: np.ndarray, weight: np.ndarray, p: float, cell_volume: float
) -> float:
    """Numba-accelerated weighted ``L^p`` grid-function norm."""
    e = np.ascontiguousarray(error.ravel(), dtype=np.float64)
    w = np.ascontiguousarray(weight.ravel(), dtype=np.float64)
    if np.isinf(p):
        return float(_linf(e, w))
    total = cell_volume * _lp_sum(e, w, float(p))
    return float(total ** (1.0 / p))


@njit(cache=True, parallel=True, fastmath=False)
def _cellwise_order(
    f: np.ndarray, m: np.ndarray, c: np.ndarray, log_r: float, atol: float
) -> np.ndarray:
    out = np.full(f.size, np.nan)
    for i in prange(f.size):
        num = abs(c[i] - m[i])
        den = abs(m[i] - f[i])
        if den > atol and num > atol:
            out[i] = np.log(num / den) / log_r
    return out


def cellwise_order(
    phi_fine: np.ndarray,
    phi_med: np.ndarray,
    phi_coarse: np.ndarray,
    r: float,
    atol: float = 1e-14,
) -> np.ndarray:
    """Numba-accelerated per-cell observed order of accuracy."""
    f = np.ascontiguousarray(phi_fine.ravel(), dtype=np.float64)
    m = np.ascontiguousarray(phi_med.ravel(), dtype=np.float64)
    c = np.ascontiguousarray(phi_coarse.ravel(), dtype=np.float64)
    out = _cellwise_order(f, m, c, float(np.log(r)), float(atol))
    return out.reshape(phi_fine.shape)
