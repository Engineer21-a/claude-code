"""Backend selector for the performance-critical kernels.

At import time this picks the Numba-compiled kernels when they are available and
transparently falls back to the pure-NumPy reference otherwise.  The rest of the
package always calls through here, so the library runs identically -- only
slower -- on a machine without Numba.  ``HAS_NUMBA`` and ``BACKEND`` are exposed
so reports and benchmarks can state which path executed.
"""

from __future__ import annotations

from . import kernels_numpy

try:  # pragma: no cover - exercised implicitly by whichever path is installed
    from . import kernels as _numba_kernels

    HAS_NUMBA = True
    BACKEND = "numba"
    weighted_lp_norm = _numba_kernels.weighted_lp_norm
    cellwise_order = _numba_kernels.cellwise_order
except ImportError:  # pragma: no cover - only when numba is unavailable
    # Restrict to ImportError so a genuine bug inside kernels.py (e.g. a typo or
    # a Numba compilation error) fails loudly instead of silently degrading to
    # the NumPy backend and skipping the parity tests.
    HAS_NUMBA = False
    BACKEND = "numpy"
    weighted_lp_norm = kernels_numpy.weighted_lp_norm
    cellwise_order = kernels_numpy.cellwise_order

# Always expose the reference path explicitly for parity testing / benchmarks.
reference_weighted_lp_norm = kernels_numpy.weighted_lp_norm
reference_cellwise_order = kernels_numpy.cellwise_order

__all__ = [
    "HAS_NUMBA",
    "BACKEND",
    "weighted_lp_norm",
    "cellwise_order",
    "reference_weighted_lp_norm",
    "reference_cellwise_order",
]
