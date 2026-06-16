"""The manufactured source must make the exact solution satisfy the PDE exactly."""

import sympy as sp

import pdeforge as pf
from pdeforge.mms import advection_expr, laplacian_expr, t


def _residual(problem):
    u = problem.exact
    f = problem.source
    lap = laplacian_expr(u, problem.dim)
    kind = problem.kind
    if kind == "heat":
        return sp.diff(u, t) - problem.params["alpha"] * lap - f
    if kind == "advdiff":
        adv = advection_expr(u, problem.params["velocity"], problem.dim)
        return sp.diff(u, t) + adv - problem.params["alpha"] * lap - f
    if kind == "poisson":
        return -problem.params["alpha"] * lap - f
    if kind == "wave":
        return sp.diff(u, t, 2) - problem.params["c"] ** 2 * lap - f
    raise AssertionError(kind)


def test_manufactured_source_is_consistent():
    for factory in (
        pf.heat_1d,
        pf.heat_2d,
        pf.advection_diffusion_1d,
        pf.poisson_2d,
        pf.wave_1d,
    ):
        problem = factory()
        residual = sp.simplify(_residual(problem))
        assert residual == 0, f"{problem.name}: residual {residual} != 0"
