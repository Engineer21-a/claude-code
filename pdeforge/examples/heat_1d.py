"""Verify a 1D heat-equation solver (Crank-Nicolson) and write a report.

    python examples/heat_1d.py
"""

from __future__ import annotations

import pdeforge as pf


def main() -> None:
    problem = pf.heat_1d()
    study = pf.run_convergence_study(problem)
    report = pf.generate_report(study, "reports/heat_1d")
    print(f"{problem.name}: observed order {study.observed_order:.3f} -> {study.verdict}")
    print(f"report written : {report}")


if __name__ == "__main__":
    main()
