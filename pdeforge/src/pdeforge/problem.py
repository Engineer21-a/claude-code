"""Symbolic specification of a PDE verification problem.

A :class:`PDEProblem` bundles everything needed to *forge* a verified solver:
the governing equation (by ``kind``), the spatial domain, physical parameters,
and a manufactured exact solution from which the source term is derived
automatically.  The factory functions at the bottom build a representative
problem for each supported equation so users (and the test-suite) get a
one-liner starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .mms import derive_source, t, x, y

__all__ = [
    "PDEProblem",
    "heat_1d",
    "heat_2d",
    "advection_diffusion_1d",
    "poisson_2d",
    "wave_1d",
]


@dataclass(frozen=True)
class PDEProblem:
    """A fully specified, self-verifying PDE problem.

    Attributes
    ----------
    name:
        Human-readable label used in reports.
    kind:
        One of ``"heat"``, ``"advdiff"``, ``"poisson"``, ``"wave"``.
    dim:
        Number of spatial dimensions (1 or 2).
    bounds:
        Domain extent per axis, e.g. ``((0.0, 1.0),)`` or ``((0, 1), (0, 1))``.
    params:
        Physical coefficients (``alpha``, ``velocity``, ``c``) as needed.
    exact:
        The manufactured exact solution as a sympy expression in ``x[, y][, t]``.
    t_final:
        Final time for time-dependent problems (ignored for ``poisson``).
    theoretical_order:
        Formal order of accuracy expected from the default scheme.
    """

    name: str
    kind: str
    dim: int
    bounds: tuple[tuple[float, float], ...]
    params: dict
    exact: sp.Expr
    t_final: float = 0.0
    theoretical_order: int = 2
    source: sp.Expr = field(init=False)

    def __post_init__(self) -> None:
        src = derive_source(self.kind, self.exact, self.params, self.dim)
        # ``frozen=True`` blocks normal assignment; set the derived field directly.
        object.__setattr__(self, "source", src)

    @property
    def is_steady(self) -> bool:
        return self.kind == "poisson"

    @property
    def spatial_symbols(self) -> tuple[sp.Symbol, ...]:
        return (x,) if self.dim == 1 else (x, y)

    def describe(self) -> str:
        """Return a short multi-line LaTeX-ish description for reports."""
        eqs = {
            "heat": r"u_t = \alpha\,\nabla^2 u + f",
            "advdiff": r"u_t + \mathbf{v}\cdot\nabla u = \alpha\,\nabla^2 u + f",
            "poisson": r"-\alpha\,\nabla^2 u = f",
            "wave": r"u_{tt} = c^2\,\nabla^2 u + f",
        }
        lines = [
            f"Governing equation: $ {eqs[self.kind]} $",
            f"Manufactured solution: $ u = {sp.latex(self.exact)} $",
            f"Derived source: $ f = {sp.latex(self.source)} $",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Factory functions: one representative problem per supported equation.
# --------------------------------------------------------------------------- #


def heat_1d(alpha: float = 0.1, t_final: float = 0.2) -> PDEProblem:
    """1D heat equation with a decaying sinusoid as the manufactured solution."""
    exact = sp.sin(sp.pi * x) * sp.exp(-0.5 * t)
    return PDEProblem(
        name="1D heat equation",
        kind="heat",
        dim=1,
        bounds=((0.0, 1.0),),
        params={"alpha": alpha},
        exact=exact,
        t_final=t_final,
    )


def heat_2d(alpha: float = 0.1, t_final: float = 0.1) -> PDEProblem:
    """2D heat equation with a separable decaying mode."""
    exact = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.exp(-0.5 * t)
    return PDEProblem(
        name="2D heat equation",
        kind="heat",
        dim=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        params={"alpha": alpha},
        exact=exact,
        t_final=t_final,
    )


def advection_diffusion_1d(
    alpha: float = 0.05, velocity: float = 1.0, t_final: float = 0.2
) -> PDEProblem:
    """1D advection-diffusion -- the workhorse for the bug-injection demo."""
    exact = sp.sin(sp.pi * x) * sp.exp(-0.5 * t)
    return PDEProblem(
        name="1D advection-diffusion",
        kind="advdiff",
        dim=1,
        bounds=((0.0, 1.0),),
        params={"alpha": alpha, "velocity": (velocity,)},
        exact=exact,
        t_final=t_final,
    )


def poisson_2d(alpha: float = 1.0) -> PDEProblem:
    """2D Poisson problem (steady), solved with a single sparse solve."""
    exact = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    return PDEProblem(
        name="2D Poisson equation",
        kind="poisson",
        dim=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        params={"alpha": alpha},
        exact=exact,
    )


def wave_1d(c: float = 1.0, t_final: float = 0.2) -> PDEProblem:
    """1D wave equation with a forced standing wave."""
    exact = sp.sin(sp.pi * x) * sp.cos(2.0 * t)
    return PDEProblem(
        name="1D wave equation",
        kind="wave",
        dim=1,
        bounds=((0.0, 1.0),),
        params={"c": c},
        exact=exact,
        t_final=t_final,
    )
