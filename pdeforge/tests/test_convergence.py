"""The headline self-validating tests: every scheme hits its theoretical order."""

import pytest

import pdeforge as pf

RESOLUTIONS_1D = (17, 33, 65, 129)
RESOLUTIONS_2D = (17, 33, 65)


@pytest.mark.parametrize(
    "factory, resolutions",
    [
        (pf.heat_1d, RESOLUTIONS_1D),
        (pf.heat_2d, RESOLUTIONS_2D),
        (pf.advection_diffusion_1d, RESOLUTIONS_1D),
        (pf.poisson_2d, RESOLUTIONS_2D),
        (pf.wave_1d, RESOLUTIONS_1D),
    ],
)
def test_second_order_schemes_converge_at_order_two(factory, resolutions):
    study = pf.run_convergence_study(factory(), resolutions=resolutions)
    assert study.passed
    assert abs(study.observed_order - 2.0) < 0.15
    assert study.r_squared > 0.999
    # GCI must agree and the grids must be in the asymptotic range.
    assert abs(study.gci.p - 2.0) < 0.2
    assert study.gci.in_asymptotic_range


def test_fourth_order_laplacian_beats_second_order():
    # A 4th-order interior stencil should converge visibly faster than 2nd order
    # even with the low-order boundary closure.
    study = pf.run_convergence_study(
        pf.poisson_2d(), resolutions=(17, 33, 65), spatial_order=4
    )
    assert study.observed_order > 3.0
