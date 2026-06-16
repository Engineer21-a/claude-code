"""Method of Manufactured Solutions (MMS).

MMS is the gold standard for *code verification*: you choose a smooth, analytic
field ``u_exact`` that is **not** a natural solution of the PDE, substitute it
into the governing equation, and let the residual *define* a source term ``f``.
By construction ``u_exact`` then solves the PDE ``L[u] = f`` exactly, so the
discretization error -- the only thing left -- is known to machine precision on
every grid.  Comparing the observed convergence rate against the scheme's formal
order is then a rigorous, automatable test of correctness.

References
----------
* P. J. Roache, *Verification and Validation in Computational Science and
  Engineering*, Hermosa, 1998.
* K. Salari and P. Knupp, *Code Verification by the Method of Manufactured
  Solutions*, Sandia report SAND2000-1444, 2000.
"""

from __future__ import annotations

import sympy as sp

# Canonical symbols shared across the package.
x, y, t = sp.symbols("x y t", real=True)

__all__ = ["x", "y", "t", "laplacian_expr", "advection_expr", "derive_source"]


def laplacian_expr(u: sp.Expr, dim: int) -> sp.Expr:
    """Symbolic Laplacian of ``u`` in ``dim`` spatial dimensions."""
    lap = sp.diff(u, x, 2)
    if dim == 2:
        lap += sp.diff(u, y, 2)
    return lap


def advection_expr(u: sp.Expr, velocity: tuple[float, ...], dim: int) -> sp.Expr:
    """Symbolic advection term ``v . grad(u)``."""
    adv = velocity[0] * sp.diff(u, x)
    if dim == 2:
        adv += velocity[1] * sp.diff(u, y)
    return adv


def derive_source(kind: str, u: sp.Expr, params: dict, dim: int) -> sp.Expr:
    """Derive the MMS source term that makes ``u`` an exact solution.

    The PDE conventions (matching :mod:`pdeforge.solver`) are:

    ===========  =================================================
    ``kind``     governing equation
    ===========  =================================================
    ``heat``     ``u_t = alpha * lap(u) + f``
    ``advdiff``  ``u_t + v . grad(u) = alpha * lap(u) + f``
    ``poisson``  ``-alpha * lap(u) = f``
    ``wave``     ``u_tt = c**2 * lap(u) + f``
    ===========  =================================================
    """
    lap = laplacian_expr(u, dim)
    if kind == "heat":
        alpha = params["alpha"]
        return sp.simplify(sp.diff(u, t) - alpha * lap)
    if kind == "advdiff":
        alpha = params["alpha"]
        adv = advection_expr(u, params["velocity"], dim)
        return sp.simplify(sp.diff(u, t) + adv - alpha * lap)
    if kind == "poisson":
        alpha = params["alpha"]
        return sp.simplify(-alpha * lap)
    if kind == "wave":
        c = params["c"]
        return sp.simplify(sp.diff(u, t, 2) - c**2 * lap)
    raise ValueError(f"unknown PDE kind {kind!r}")
