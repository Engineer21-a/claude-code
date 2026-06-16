"""The "wow" test: an injected stencil bug is caught as an order collapse + FAIL.

Switching the advection term from the 2nd-order central difference to the
1st-order upwind difference is exactly the kind of well-intentioned change an
engineer makes for stability and then forgets about.  The verification machinery
must notice the resulting order degradation with no human in the loop.
"""

import pdeforge as pf

RESOLUTIONS = (17, 33, 65, 129)


def test_correct_scheme_passes():
    problem = pf.advection_diffusion_1d(alpha=0.01, velocity=1.0)
    study = pf.run_convergence_study(problem, resolutions=RESOLUTIONS, advection_scheme="central")
    assert study.passed
    assert abs(study.observed_order - 2.0) < 0.2


def test_injected_bug_is_detected():
    problem = pf.advection_diffusion_1d(alpha=0.01, velocity=1.0)
    study = pf.run_convergence_study(problem, resolutions=RESOLUTIONS, advection_scheme="upwind")
    assert not study.passed
    assert study.verdict == "FAIL"
    # The order should collapse from 2 towards 1.
    assert study.observed_order < 1.3
