"""Drive a grid-refinement study end to end and render a PASS/FAIL verdict.

This is the function a user actually calls: hand it a :class:`PDEProblem` and it
solves on a sequence of refined (nested) grids, measures the discretisation
error, fits the observed order of accuracy, runs the GCI uncertainty estimate,
computes a per-cell order field, and decides whether the code passes
verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import backend
from .problem import PDEProblem
from .solver import SolveResult, solve
from .verify import (
    GCIResult,
    error_norms,
    grid_convergence_index,
    least_squares_order,
    pairwise_orders,
)

__all__ = ["ConvergenceStudy", "run_convergence_study"]

# Nested grids (n = 2^k * 16 + 1) so coarse points are a strict subset of fine
# points -- lets us co-locate solutions for the per-cell order field for free.
DEFAULT_RESOLUTIONS_1D = (17, 33, 65, 129, 257)
DEFAULT_RESOLUTIONS_2D = (17, 33, 65, 129)


@dataclass
class ConvergenceStudy:
    """Full results of a verification study."""

    problem: PDEProblem
    resolutions: tuple[int, ...]
    hs: list[float]
    errors_l2: list[float]
    errors_linf: list[float]
    observed_order: float
    r_squared: float
    pairwise: list[float]
    gci: GCIResult
    results: list[SolveResult] = field(repr=False)
    cell_order_field: np.ndarray = field(repr=False)
    tolerance: float = 0.25
    backend_used: str = backend.BACKEND

    @property
    def theoretical_order(self) -> int:
        return self.problem.theoretical_order

    @property
    def order_deficit(self) -> float:
        return self.theoretical_order - self.observed_order

    @property
    def passed(self) -> bool:
        """PASS iff the observed order is within tolerance of the theoretical one.

        A one-sided slack above the theoretical order is allowed (super-
        convergence is not a defect); falling short by more than ``tolerance``
        is a failure -- exactly what catches a degraded stencil.
        """
        return self.observed_order >= self.theoretical_order - self.tolerance

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _solution_functional(result: SolveResult) -> float:
    """A scalar, grid-independent functional for the GCI: the solution L2 norm."""
    m = result.interior_mask.ravel()
    u_int = result.u_numeric.ravel()[m]
    weight = np.ones_like(u_int)
    cell_volume = result.h ** len(result.shape)
    return backend.weighted_lp_norm(u_int, weight, 2.0, cell_volume)


def _restrict(result: SolveResult, target_n: int) -> np.ndarray:
    """Subsample a (nested-grid) field down to ``target_n`` points per axis."""
    stride = (result.n - 1) // (target_n - 1)
    field_nd = result.u_numeric.reshape(result.shape)
    if len(result.shape) == 1:
        return field_nd[::stride]
    return field_nd[::stride, ::stride]


def _cell_order_field(results: list[SolveResult]) -> np.ndarray:
    """Per-cell observed order from the three finest (nested) solutions."""
    fine, med, coarse = results[-1], results[-2], results[-3]
    target_n = coarse.n
    pf = _restrict(fine, target_n)
    pm = _restrict(med, target_n)
    pc = coarse.u_numeric.reshape(coarse.shape)
    return backend.cellwise_order(pf, pm, pc, r=2.0)


def run_convergence_study(
    problem: PDEProblem,
    resolutions: tuple[int, ...] | None = None,
    *,
    tolerance: float = 0.25,
    **solve_kwargs,
) -> ConvergenceStudy:
    """Solve on a refinement ladder and assemble a :class:`ConvergenceStudy`."""
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTIONS_1D if problem.dim == 1 else DEFAULT_RESOLUTIONS_2D
    resolutions = tuple(sorted(resolutions))  # ascending n => descending h

    results = [solve(problem, n, **solve_kwargs) for n in resolutions]
    hs = [r.h for r in results]
    norms = [error_norms(r) for r in results]
    errors_l2 = [d["l2"] for d in norms]
    errors_linf = [d["linf"] for d in norms]

    p, _log_c, r2 = least_squares_order(hs, errors_l2)
    pw = pairwise_orders(hs, errors_l2)

    # GCI on the three finest grids (fine -> coarse) using the solution norm.
    functionals = [_solution_functional(r) for r in results]
    fine_to_coarse_h = hs[-1:-4:-1]
    fine_to_coarse_phi = functionals[-1:-4:-1]
    gci = grid_convergence_index(fine_to_coarse_h, fine_to_coarse_phi)

    cell_order = _cell_order_field(results)

    return ConvergenceStudy(
        problem=problem,
        resolutions=resolutions,
        hs=hs,
        errors_l2=errors_l2,
        errors_linf=errors_linf,
        observed_order=p,
        r_squared=r2,
        pairwise=pw,
        gci=gci,
        results=results,
        cell_order_field=cell_order,
        tolerance=tolerance,
        backend_used=backend.BACKEND,
    )
