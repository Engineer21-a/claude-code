"""Verification metrics: error norms, observed order of accuracy, and GCI.

Two complementary stories live here:

* **Direct order** (used when the exact solution is known, i.e. MMS): fit the
  discretisation error against the grid spacing and read off the slope.  This is
  what proves a *code* is correct.
* **Roache / Celik GCI** (used when the exact solution is *not* known -- the
  everyday simulation case): estimate the order and a numerical-uncertainty band
  from three successive grids, and check that they sit in the asymptotic range.

References
----------
* P. J. Roache, "Perspective: A Method for Uniform Reporting of Grid Refinement
  Studies", J. Fluids Eng. 116 (1994).
* I. Celik et al., "Procedure for Estimation and Reporting of Uncertainty Due to
  Discretization in CFD Applications", J. Fluids Eng. 130 (2008).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import backend
from .solver import SolveResult

__all__ = [
    "error_norms",
    "least_squares_order",
    "pairwise_orders",
    "richardson_extrapolation",
    "GCIResult",
    "grid_convergence_index",
]


def error_norms(result: SolveResult, p_values=(2.0, np.inf)) -> dict[str, float]:
    """Discrete error norms over the interior of a :class:`SolveResult`."""
    err = result.interior_error
    weight = np.ones_like(err)
    cell_volume = result.h ** len(result.shape)
    out: dict[str, float] = {}
    for p in p_values:
        key = "linf" if np.isinf(p) else f"l{int(p)}"
        out[key] = backend.weighted_lp_norm(err, weight, p, cell_volume)
    return out


def least_squares_order(hs, errors) -> tuple[float, float, float]:
    """Fit ``log(E) = log(C) + p*log(h)``; return ``(p, logC, R^2)``."""
    log_h = np.log(np.asarray(hs, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    a = np.vstack([log_h, np.ones_like(log_h)]).T
    (p, log_c), *_ = np.linalg.lstsq(a, log_e, rcond=None)
    pred = a @ np.array([p, log_c])
    ss_res = float(np.sum((log_e - pred) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(p), float(log_c), r2


def pairwise_orders(hs, errors) -> list[float]:
    """Observed order between each consecutive pair of grids."""
    hs = np.asarray(hs, dtype=float)
    errors = np.asarray(errors, dtype=float)
    orders = []
    for i in range(len(hs) - 1):
        orders.append(float(np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])))
    return orders


def richardson_extrapolation(phi1: float, phi2: float, phi3: float, r: float):
    """Constant-ratio Richardson extrapolation from fine(1)->coarse(3) values.

    Returns ``(p, phi_extrapolated)`` where ``p`` is the observed order and
    ``phi_extrapolated`` is the ``h -> 0`` estimate.
    """
    p = float(np.log(abs((phi3 - phi2) / (phi2 - phi1))) / np.log(r))
    phi_ext = float(phi1 + (phi1 - phi2) / (r**p - 1.0))
    return p, phi_ext


@dataclass
class GCIResult:
    """Grid Convergence Index outcome following Celik et al. (2008)."""

    p: float  # apparent order of accuracy
    phi_extrapolated: float  # Richardson-extrapolated value
    gci_fine: float  # GCI on the fine-grid pair (fraction, not %)
    gci_medium: float  # GCI on the medium-grid pair
    asymptotic_ratio: float  # ~1.0 => grids are in the asymptotic range

    @property
    def in_asymptotic_range(self) -> bool:
        return abs(self.asymptotic_ratio - 1.0) < 0.1


def grid_convergence_index(hs, phis, fs: float = 1.25) -> GCIResult:
    """Three-grid GCI per Celik (2008), allowing a non-constant refinement ratio.

    Parameters
    ----------
    hs:
        Representative cell sizes, fine -> coarse (``h1 < h2 < h3``).
    phis:
        The corresponding solution functional values ``phi1, phi2, phi3``.
    fs:
        Factor of safety; 1.25 for studies using three or more grids.
    """
    h1, h2, h3 = (float(v) for v in hs)
    phi1, phi2, phi3 = (float(v) for v in phis)
    r21 = h2 / h1
    r32 = h3 / h2
    eps21 = phi2 - phi1
    eps32 = phi3 - phi2
    s = float(np.sign(eps32 / eps21)) if eps21 != 0 else 1.0

    # Fixed-point iteration for the apparent order p (Celik eq. with q(p)).
    p = 2.0
    for _ in range(100):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(eps32 / eps21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = p_new

    phi_ext = (r21**p * phi1 - phi2) / (r21**p - 1.0)

    def _rel(a: float, b: float) -> float:
        return abs((a - b) / a) if a != 0 else abs(a - b)

    e_a21 = _rel(phi1, phi2)
    e_a32 = _rel(phi2, phi3)
    gci_fine = fs * e_a21 / (r21**p - 1.0)
    gci_medium = fs * e_a32 / (r32**p - 1.0)
    asymptotic_ratio = gci_medium / (r21**p * gci_fine) if gci_fine != 0 else float("nan")
    return GCIResult(float(p), float(phi_ext), float(gci_fine), float(gci_medium), float(asymptotic_ratio))
