"""Finite-difference operators on uniform grids, assembled as ``scipy.sparse`` matrices.

The operators here are *pure stencils*: every interior row uses the standard
finite-difference coefficients and boundary rows are left as zeros.  Boundary
conditions are applied later (see :mod:`pdeforge.solver`) by partitioning the
unknowns into interior and boundary sets.  Keeping the stencils free of
boundary bookkeeping makes them trivial to reason about and to unit-test.

All matrices act on grid functions flattened in C-order.  For 2D problems the
field ``U`` has shape ``(nx, ny)`` with ``U[i, j] = u(x_i, y_j)`` and is
flattened so that the linear index is ``i * ny + j``.  With this convention a
2D operator is the Kronecker sum of its 1D parts::

    L2 = kron(D2x, I_y) + kron(I_x, D2y)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "second_derivative_1d",
    "first_derivative_1d",
    "laplacian",
    "gradient",
]


def _zero_boundary_rows(mat: sp.lil_matrix, n: int) -> None:
    """Zero the first and last rows of an ``n x n`` matrix (in place)."""
    mat[0, :] = 0.0
    mat[n - 1, :] = 0.0


def second_derivative_1d(n: int, h: float, order: int = 2) -> sp.csr_matrix:
    """Second-derivative operator ``d^2/dx^2`` on ``n`` equispaced points.

    Parameters
    ----------
    n:
        Number of grid points including both endpoints.
    h:
        Grid spacing.
    order:
        Formal order of accuracy of the interior stencil. ``2`` uses the
        classic ``(1, -2, 1) / h**2`` stencil; ``4`` uses the five-point
        ``(-1, 16, -30, 16, -1) / (12 h**2)`` stencil.
    """
    mat = sp.lil_matrix((n, n))
    if order == 2:
        inv = 1.0 / h**2
        for i in range(1, n - 1):
            mat[i, i - 1] = inv
            mat[i, i] = -2.0 * inv
            mat[i, i + 1] = inv
    elif order == 4:
        inv = 1.0 / (12.0 * h**2)
        # Use the 4th-order stencil on rows that have two neighbours on each
        # side; fall back to 2nd order on the first/last interior row so the
        # operator stays well defined without ghost points.
        for i in range(1, n - 1):
            if 2 <= i <= n - 3:
                mat[i, i - 2] = -1.0 * inv
                mat[i, i - 1] = 16.0 * inv
                mat[i, i] = -30.0 * inv
                mat[i, i + 1] = 16.0 * inv
                mat[i, i + 2] = -1.0 * inv
            else:
                inv2 = 1.0 / h**2
                mat[i, i - 1] = inv2
                mat[i, i] = -2.0 * inv2
                mat[i, i + 1] = inv2
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported order {order!r}; use 2 or 4")
    _zero_boundary_rows(mat, n)
    return mat.tocsr()


def first_derivative_1d(
    n: int, h: float, scheme: str = "central", velocity_sign: float = 1.0
) -> sp.csr_matrix:
    """First-derivative operator ``d/dx`` on ``n`` equispaced points.

    Parameters
    ----------
    scheme:
        ``"central"`` gives the 2nd-order ``(-1, 0, 1) / (2 h)`` stencil.
        ``"upwind"`` gives the 1st-order one-sided stencil whose direction is
        chosen from ``velocity_sign`` (backward difference for positive flow).
        The first-order option exists so the verification machinery can *catch*
        the order degradation it causes -- see the bug-injection demo.
    velocity_sign:
        Sign of the advecting velocity; only used by the upwind scheme.
    """
    mat = sp.lil_matrix((n, n))
    if scheme == "central":
        inv = 1.0 / (2.0 * h)
        for i in range(1, n - 1):
            mat[i, i - 1] = -inv
            mat[i, i + 1] = inv
    elif scheme == "upwind":
        inv = 1.0 / h
        for i in range(1, n - 1):
            if velocity_sign >= 0.0:  # backward difference
                mat[i, i - 1] = -inv
                mat[i, i] = inv
            else:  # forward difference
                mat[i, i] = -inv
                mat[i, i + 1] = inv
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown scheme {scheme!r}; use 'central' or 'upwind'")
    _zero_boundary_rows(mat, n)
    return mat.tocsr()


def laplacian(shape: tuple[int, ...], spacings: tuple[float, ...], order: int = 2) -> sp.csr_matrix:
    """Assemble a 1D or 2D Laplacian as a Kronecker sum of 1D operators."""
    if len(shape) == 1:
        return second_derivative_1d(shape[0], spacings[0], order=order)
    if len(shape) == 2:
        nx, ny = shape
        hx, hy = spacings
        d2x = second_derivative_1d(nx, hx, order=order)
        d2y = second_derivative_1d(ny, hy, order=order)
        ix = sp.identity(nx, format="csr")
        iy = sp.identity(ny, format="csr")
        return (sp.kron(d2x, iy) + sp.kron(ix, d2y)).tocsr()
    raise ValueError("only 1D and 2D grids are supported")


def gradient(
    shape: tuple[int, ...],
    spacings: tuple[float, ...],
    scheme: str = "central",
    velocity: tuple[float, ...] = (1.0,),
) -> list[sp.csr_matrix]:
    """Return the list of first-derivative operators, one per spatial axis."""
    if len(shape) == 1:
        return [first_derivative_1d(shape[0], spacings[0], scheme, np.sign(velocity[0]) or 1.0)]
    if len(shape) == 2:
        nx, ny = shape
        hx, hy = spacings
        ix = sp.identity(nx, format="csr")
        iy = sp.identity(ny, format="csr")
        gx = first_derivative_1d(nx, hx, scheme, np.sign(velocity[0]) or 1.0)
        gy = first_derivative_1d(ny, hy, scheme, np.sign(velocity[1]) or 1.0)
        return [sp.kron(gx, iy).tocsr(), sp.kron(ix, gy).tocsr()]
    raise ValueError("only 1D and 2D grids are supported")
