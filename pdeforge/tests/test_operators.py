"""Finite-difference operators reproduce known derivatives at their formal order."""

import numpy as np

from pdeforge.operators import first_derivative_1d, laplacian, second_derivative_1d


def _interior(n):
    return slice(1, n - 1)


def test_second_derivative_on_sine_is_second_order():
    errors = []
    hs = []
    for n in (33, 65, 129):
        h = 1.0 / (n - 1)
        x = np.linspace(0.0, 1.0, n)
        u = np.sin(np.pi * x)
        d2 = second_derivative_1d(n, h) @ u
        exact = -(np.pi**2) * np.sin(np.pi * x)
        sl = _interior(n)
        errors.append(np.max(np.abs(d2[sl] - exact[sl])))
        hs.append(h)
    order = np.log(errors[0] / errors[-1]) / np.log(hs[0] / hs[-1])
    assert 1.9 < order < 2.1


def test_central_first_derivative_is_second_order():
    errors, hs = [], []
    for n in (33, 65, 129):
        h = 1.0 / (n - 1)
        x = np.linspace(0.0, 1.0, n)
        d1 = first_derivative_1d(n, h, "central") @ np.sin(np.pi * x)
        exact = np.pi * np.cos(np.pi * x)
        sl = _interior(n)
        errors.append(np.max(np.abs(d1[sl] - exact[sl])))
        hs.append(h)
    order = np.log(errors[0] / errors[-1]) / np.log(hs[0] / hs[-1])
    assert 1.9 < order < 2.1


def test_upwind_first_derivative_is_first_order():
    errors, hs = [], []
    for n in (33, 65, 129):
        h = 1.0 / (n - 1)
        x = np.linspace(0.0, 1.0, n)
        d1 = first_derivative_1d(n, h, "upwind") @ np.sin(np.pi * x)
        exact = np.pi * np.cos(np.pi * x)
        sl = _interior(n)
        errors.append(np.max(np.abs(d1[sl] - exact[sl])))
        hs.append(h)
    order = np.log(errors[0] / errors[-1]) / np.log(hs[0] / hs[-1])
    assert 0.9 < order < 1.1


def test_laplacian_2d_interior_block_is_symmetric():
    # The interior-interior block (what the solver actually inverts) is the
    # symmetric Dirichlet Laplacian; the full operator is not symmetric because
    # boundary rows are zeroed.
    nx = ny = 9
    h = 1.0 / (nx - 1)
    lap = laplacian((nx, ny), (h, h)).toarray()
    mask = np.ones((nx, ny), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    interior = np.where(mask.ravel())[0]
    core = lap[np.ix_(interior, interior)]
    assert core.shape == ((nx - 2) * (ny - 2), (nx - 2) * (ny - 2))
    assert np.allclose(core, core.T)
