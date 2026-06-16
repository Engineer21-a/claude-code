"""pdeforge -- forge numerically *verified* PDE solvers from symbolic equations.

The pipeline, in one import:

    >>> import pdeforge as pf
    >>> problem = pf.heat_2d()                 # symbolic spec + auto MMS source
    >>> study = pf.run_convergence_study(problem)
    >>> study.verdict, round(study.observed_order, 2)
    ('PASS', 2.0)
    >>> pf.generate_report(study, "out/")      # doctest: +SKIP

See the README for the full story.  Everything here is grounded in the standard
verification & validation literature (Roache 1998; Salari & Knupp 2000; Celik
et al. 2008; ASME V&V 20-2009).
"""

from __future__ import annotations

from . import backend
from .problem import (
    PDEProblem,
    advection_diffusion_1d,
    heat_1d,
    heat_2d,
    poisson_2d,
    wave_1d,
)
from .report import generate_report
from .solver import SolveResult, solve
from .study import ConvergenceStudy, run_convergence_study
from .verify import (
    GCIResult,
    error_norms,
    grid_convergence_index,
    least_squares_order,
    pairwise_orders,
    richardson_extrapolation,
)

__version__ = "0.1.0"

HAS_NUMBA = backend.HAS_NUMBA
BACKEND = backend.BACKEND

__all__ = [
    "__version__",
    "HAS_NUMBA",
    "BACKEND",
    # problems
    "PDEProblem",
    "heat_1d",
    "heat_2d",
    "advection_diffusion_1d",
    "poisson_2d",
    "wave_1d",
    # solving
    "SolveResult",
    "solve",
    # verification
    "ConvergenceStudy",
    "run_convergence_study",
    "error_norms",
    "least_squares_order",
    "pairwise_orders",
    "richardson_extrapolation",
    "grid_convergence_index",
    "GCIResult",
    # reporting
    "generate_report",
]
