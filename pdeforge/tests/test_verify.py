"""Verification math recovers known orders and GCI quantities on synthetic data."""

import numpy as np

from pdeforge.verify import (
    grid_convergence_index,
    least_squares_order,
    pairwise_orders,
    richardson_extrapolation,
)


def _synthetic_errors(p_true, c=0.5, hs=(0.1, 0.05, 0.025, 0.0125)):
    return list(hs), [c * h**p_true for h in hs]


def test_least_squares_recovers_order():
    for p_true in (1.0, 2.0, 4.0):
        hs, errs = _synthetic_errors(p_true)
        p, _logc, r2 = least_squares_order(hs, errs)
        assert abs(p - p_true) < 1e-10
        assert r2 > 1 - 1e-12


def test_pairwise_orders_recovers_order():
    hs, errs = _synthetic_errors(2.0)
    assert np.allclose(pairwise_orders(hs, errs), 2.0)


def test_richardson_extrapolation_recovers_order_and_limit():
    p_true, exact, c, r = 2.0, 3.0, 0.7, 2.0
    h1 = 0.01
    phi1 = exact + c * h1**p_true
    phi2 = exact + c * (h1 * r) ** p_true
    phi3 = exact + c * (h1 * r * r) ** p_true
    p, phi_ext = richardson_extrapolation(phi1, phi2, phi3, r)
    assert abs(p - p_true) < 1e-9
    assert abs(phi_ext - exact) < 1e-9


def test_gci_apparent_order_and_asymptotic_range():
    # Monotone, second-order-convergent functional with constant ratio r=2.
    p_true, exact, c, r = 2.0, 5.0, 0.3, 2.0
    h1 = 0.02
    hs = [h1, h1 * r, h1 * r * r]
    phis = [exact + c * h**p_true for h in hs]
    gci = grid_convergence_index(hs, phis)
    assert abs(gci.p - p_true) < 1e-6
    assert abs(gci.phi_extrapolated - exact) < 1e-6
    assert gci.in_asymptotic_range
    assert gci.gci_fine < gci.gci_medium  # finer grid -> smaller uncertainty
