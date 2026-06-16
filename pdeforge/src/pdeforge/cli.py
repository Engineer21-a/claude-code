"""Command-line interface: ``pdeforge run`` and ``pdeforge demo``."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from . import (
    BACKEND,
    PDEProblem,
    advection_diffusion_1d,
    generate_report,
    heat_1d,
    heat_2d,
    poisson_2d,
    run_convergence_study,
    wave_1d,
)

PROBLEMS: dict[str, Callable[..., PDEProblem]] = {
    "heat_1d": heat_1d,
    "heat_2d": heat_2d,
    "advdiff_1d": advection_diffusion_1d,
    "poisson_2d": poisson_2d,
    "wave_1d": wave_1d,
}


def _run_one(name: str, output_dir: Path) -> bool:
    problem = PROBLEMS[name]()
    study = run_convergence_study(problem)
    report = generate_report(study, output_dir / name)
    flag = "✅" if study.passed else "❌"
    print(
        f"{flag} {problem.name:<26} observed p = {study.observed_order:5.3f} "
        f"(theory {study.theoretical_order})  ->  {study.verdict}   [{report}]"
    )
    return study.passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pdeforge",
        description="Forge numerically verified PDE solvers from symbolic equations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="verify a single named problem")
    p_run.add_argument("problem", choices=sorted(PROBLEMS))
    p_run.add_argument("-o", "--output", default="reports", type=Path)

    p_demo = sub.add_parser("demo", help="verify every built-in problem")
    p_demo.add_argument("-o", "--output", default="reports", type=Path)

    args = parser.parse_args(argv)
    print(f"pdeforge — kernel backend: {BACKEND}\n")

    if args.command == "run":
        ok = _run_one(args.problem, args.output)
        return 0 if ok else 1

    if args.command == "demo":
        results = [_run_one(name, args.output) for name in sorted(PROBLEMS)]
        return 0 if all(results) else 1

    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
