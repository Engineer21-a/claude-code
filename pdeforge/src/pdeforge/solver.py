"""Method-of-lines / direct solvers for the supported PDE families.

Boundary conditions are imposed by *partitioning* the grid unknowns into an
interior set ``I`` and a boundary set ``B``.  The boundary values are taken from
the manufactured exact solution (this is verification, so the exact boundary is
known), and their contribution is moved to the right-hand side::

    du_I/dt = A_II u_I + A_IB u_B(t) + f_I(t)

This keeps the discrete operators identical to the pure stencils in
:mod:`pdeforge.operators` and makes the interior convergence rate clean and
unpolluted by boundary bookkeeping.

Time integration:

* ``heat`` / ``advdiff`` -- Crank-Nicolson (2nd order, unconditionally stable for
  diffusion), factorised once with a sparse LU.
* ``wave`` -- the classic explicit 2nd-order central-in-time update.
* ``poisson`` -- a single sparse solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import sympy as sp
from scipy.sparse.linalg import splu, spsolve

from .mms import t as t_sym
from .operators import gradient, laplacian
from .problem import PDEProblem

__all__ = ["SolveResult", "solve"]


@dataclass
class SolveResult:
    """Outcome of a single-grid solve, with everything reports/plots need."""

    n: int
    h: float
    coords: list[np.ndarray]
    shape: tuple[int, ...]
    interior_mask: np.ndarray
    u_numeric: np.ndarray  # full field (boundary = exact, interior = computed)
    u_exact: np.ndarray  # full field
    t_final: float

    @property
    def interior_error(self) -> np.ndarray:
        m = self.interior_mask.ravel()
        return (self.u_numeric.ravel() - self.u_exact.ravel())[m]


def _make_field_evaluator(expr: sp.Expr, problem: PDEProblem, mesh: list[np.ndarray]):
    """Return ``f(tau) -> flattened full-grid field`` for a symbolic expression."""
    syms = problem.spatial_symbols
    time_dep = not problem.is_steady
    args = (*syms, t_sym) if time_dep else syms
    fn = sp.lambdify(args, expr, "numpy")
    shape = mesh[0].shape

    def evaluate(tau: float) -> np.ndarray:
        call_args = (*mesh, tau) if time_dep else tuple(mesh)
        val = np.asarray(fn(*call_args), dtype=float)
        return np.broadcast_to(val, shape).ravel().copy()

    return evaluate


def _build_mesh(problem: PDEProblem, n: int):
    coords = [np.linspace(lo, hi, n) for (lo, hi) in problem.bounds]
    spacings = [(hi - lo) / (n - 1) for (lo, hi) in problem.bounds]
    if problem.dim == 1:
        mesh = [coords[0]]
        shape: tuple[int, ...] = (n,)
    else:
        gx, gy = np.meshgrid(coords[0], coords[1], indexing="ij")
        mesh = [gx, gy]
        shape = (n, n)
    return coords, spacings, mesh, shape


def _interior_indices(shape: tuple[int, ...]):
    mask = np.ones(shape, dtype=bool)
    if len(shape) == 1:
        mask[0] = mask[-1] = False
    else:
        mask[0, :] = mask[-1, :] = False
        mask[:, 0] = mask[:, -1] = False
    flat = mask.ravel()
    return mask, np.where(flat)[0], np.where(~flat)[0]


def _spatial_operator(problem: PDEProblem, shape, spacings, advection_scheme, spatial_order):
    """Build the physical spatial operator ``A`` so that ``u_t = A u + f`` (or steady)."""
    lap = laplacian(shape, tuple(spacings), order=spatial_order)
    kind = problem.kind
    if kind == "heat":
        return (problem.params["alpha"] * lap).tocsr()
    if kind == "advdiff":
        alpha = problem.params["alpha"]
        velocity = problem.params["velocity"]
        grads = gradient(shape, tuple(spacings), advection_scheme, velocity)
        op = alpha * lap
        for vk, gk in zip(velocity, grads, strict=True):
            op = op - vk * gk
        return op.tocsr()
    if kind == "wave":
        return (problem.params["c"] ** 2 * lap).tocsr()
    if kind == "poisson":
        return (-problem.params["alpha"] * lap).tocsr()
    raise ValueError(f"unknown kind {problem.kind!r}")


def solve(
    problem: PDEProblem,
    n: int,
    *,
    advection_scheme: str = "central",
    spatial_order: int = 2,
    cfl: float = 0.5,
) -> SolveResult:
    """Solve ``problem`` on an ``n``-point-per-axis uniform grid.

    Parameters
    ----------
    n:
        Points per axis, including both endpoints.
    advection_scheme:
        Passed through to the first-derivative operator for ``advdiff``; the
        knob the bug-injection demo flips to ``"upwind"`` (1st order).
    spatial_order:
        Formal order of the spatial discretisation (2 or 4).
    cfl:
        Sets the time step as ``cfl * h`` (heat/advdiff) or ``cfl * h / c``
        (wave); proportional to ``h`` so temporal and spatial error refine
        together.
    """
    coords, spacings, mesh, shape = _build_mesh(problem, n)
    mask, idx_i, idx_b = _interior_indices(shape)
    h = float(min(spacings))

    eval_exact = _make_field_evaluator(problem.exact, problem, mesh)
    eval_source = _make_field_evaluator(problem.source, problem, mesh)

    op = _spatial_operator(problem, shape, spacings, advection_scheme, spatial_order)
    a_ii = op[idx_i][:, idx_i].tocsc()
    a_ib = op[idx_i][:, idx_b].tocsr()

    if problem.kind == "poisson":
        u_full = eval_exact(0.0)
        f_full = eval_source(0.0)
        u_b = u_full[idx_b]
        rhs = f_full[idx_i] - a_ib.dot(u_b)
        u_i = spsolve(a_ii, rhs)
        u_num = u_full.copy()
        u_num[idx_i] = u_i
        u_exact = u_full
        return SolveResult(n, h, coords, shape, mask, u_num, u_exact, 0.0)

    t_final = problem.t_final
    if problem.kind in ("heat", "advdiff"):
        dt = cfl * h
        nt = max(1, int(np.ceil(t_final / dt)))
        dt = t_final / nt
        n_i = len(idx_i)
        identity = sps.identity(n_i, format="csc")
        m_left = (identity - 0.5 * dt * a_ii).tocsc()
        m_right = (identity + 0.5 * dt * a_ii).tocsr()
        lu = splu(m_left)

        def forcing(tau: float) -> np.ndarray:
            return eval_source(tau)[idx_i] + a_ib.dot(eval_exact(tau)[idx_b])

        u_i = eval_exact(0.0)[idx_i]
        g_n = forcing(0.0)
        for step in range(nt):
            tau_next = (step + 1) * dt
            g_np = forcing(tau_next)
            rhs = m_right.dot(u_i) + 0.5 * dt * (g_n + g_np)
            u_i = lu.solve(rhs)
            g_n = g_np

    elif problem.kind == "wave":
        c = problem.params["c"]
        dt = cfl * h / c
        nt = max(1, int(np.ceil(t_final / dt)))
        dt = t_final / nt
        s_ii = a_ii.tocsr()
        eval_dudt = _make_field_evaluator(sp.diff(problem.exact, t_sym), problem, mesh)

        u0 = eval_exact(0.0)
        v0 = eval_dudt(0.0)
        f0 = eval_source(0.0)[idx_i]
        u_prev = u0[idx_i]
        acc0 = s_ii.dot(u_prev) + a_ib.dot(u0[idx_b]) + f0
        u_curr = u_prev + dt * v0[idx_i] + 0.5 * dt**2 * acc0
        tau = dt
        for step in range(1, nt):
            acc = s_ii.dot(u_curr) + a_ib.dot(eval_exact(tau)[idx_b]) + eval_source(tau)[idx_i]
            u_next = 2.0 * u_curr - u_prev + dt**2 * acc
            u_prev, u_curr = u_curr, u_next
            tau = (step + 1) * dt
        u_i = u_curr
    else:  # pragma: no cover - guarded above
        raise ValueError(f"unknown kind {problem.kind!r}")

    u_exact = eval_exact(t_final)
    u_num = u_exact.copy()
    u_num[idx_i] = u_i
    return SolveResult(n, h, coords, shape, mask, u_num, u_exact, t_final)
