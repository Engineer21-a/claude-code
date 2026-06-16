"""Verify a 2D Poisson solver (steady, single sparse solve) and write a report.

    python examples/poisson_2d.py
"""

from __future__ import annotations

import pdeforge as pf


def main() -> None:
    problem = pf.poisson_2d()
    print(f"Forging a verified solver for: {problem.name}")
    print(problem.describe(), "\n")

    study = pf.run_convergence_study(problem)
    report = pf.generate_report(study, "reports/poisson_2d")
    print(f"observed order : {study.observed_order:.3f}  ->  {study.verdict}")
    print(f"report written : {report}")


if __name__ == "__main__":
    main()
