"""THE headline demo: the tool autonomously catches a numerical bug.

We take a correct advection-diffusion solver (2nd-order central differencing) and
inject a single, *plausible* defect: switch the advection term to 1st-order
upwind differencing -- the kind of "let's add some upwinding for stability"
change that ships in real codebases and silently destroys the formal order of
accuracy.

No human inspects the stencil.  pdeforge runs the manufactured-solution
convergence study on both builds and the verdict speaks for itself: the correct
build PASSES at order 2, the buggy build is flagged FAIL as its order collapses
toward 1.

    python examples/bug_injection_demo.py
"""

from __future__ import annotations

import pdeforge as pf

RESOLUTIONS = (17, 33, 65, 129, 257)


def main() -> None:
    problem = pf.advection_diffusion_1d(alpha=0.01, velocity=1.0)

    print("=" * 70)
    print("  pdeforge bug-injection demo —", problem.name)
    print("=" * 70)
    print(problem.describe(), "\n")

    correct = pf.run_convergence_study(problem, resolutions=RESOLUTIONS, advection_scheme="central")
    buggy = pf.run_convergence_study(problem, resolutions=RESOLUTIONS, advection_scheme="upwind")

    pf.generate_report(correct, "reports/bug_demo_correct", title="Advection-diffusion (correct, central)")
    pf.generate_report(buggy, "reports/bug_demo_buggy", title="Advection-diffusion (BUG: 1st-order upwind)")

    print(f"{'build':<34}{'observed order':>16}{'verdict':>10}")
    print("-" * 60)
    print(f"{'correct  (2nd-order central)':<34}{correct.observed_order:>16.3f}{correct.verdict:>10}")
    print(f"{'buggy    (1st-order upwind) ':<34}{buggy.observed_order:>16.3f}{buggy.verdict:>10}")
    print()

    assert correct.passed, "the correct build should pass"
    assert not buggy.passed, "the bug should have been detected"
    print("pdeforge detected the injected defect automatically — no human in the loop. ✅")
    print("Reports: reports/bug_demo_correct/report.md  and  reports/bug_demo_buggy/report.md")


if __name__ == "__main__":
    main()
