"""Verify a 2D heat-equation solver and write a full V&V report.

    python examples/heat_2d.py
"""

from __future__ import annotations

import pdeforge as pf


def main() -> None:
    problem = pf.heat_2d()
    print(f"Forging a verified solver for: {problem.name}")
    print(problem.describe(), "\n")

    study = pf.run_convergence_study(problem)
    report = pf.generate_report(study, "reports/heat_2d")

    print(f"observed order : {study.observed_order:.3f} (theoretical {study.theoretical_order})")
    print(f"R^2            : {study.r_squared:.5f}")
    print(f"GCI (fine)     : {study.gci.gci_fine * 100:.3f} %")
    print(f"asymptotic     : ratio {study.gci.asymptotic_ratio:.3f}")
    print(f"verdict        : {study.verdict}")
    print(f"report written : {report}")


if __name__ == "__main__":
    main()
