"""Verify a 1D advection-diffusion solver with the 2nd-order central scheme.

    python examples/advection_diffusion.py
"""

from __future__ import annotations

import pdeforge as pf


def main() -> None:
    problem = pf.advection_diffusion_1d(alpha=0.01, velocity=1.0)
    study = pf.run_convergence_study(problem, advection_scheme="central")
    report = pf.generate_report(study, "reports/advection_diffusion")
    print(f"{problem.name} (central): observed order {study.observed_order:.3f} -> {study.verdict}")
    print(f"report written : {report}")


if __name__ == "__main__":
    main()
