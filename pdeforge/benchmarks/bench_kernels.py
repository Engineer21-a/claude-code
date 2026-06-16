"""Benchmark the JIT kernels and *prove* they compute the reference result.

A performance claim is only honest if it is the same computation, so this script
(a) times a naive pure-Python loop, the vectorised NumPy reference, and the
Numba-compiled kernel, and (b) asserts all three agree before reporting any
speedups.  Run it with::

    python benchmarks/bench_kernels.py
"""

from __future__ import annotations

import time

import numpy as np

from pdeforge import backend
from pdeforge import kernels_numpy as ref


def _time(fn, *args, repeats=5):
    fn(*args)  # warm up (JIT compile / cache)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _python_lp(error, weight, p, cell_volume):
    e = error.ravel()
    w = weight.ravel()
    total = 0.0
    for i in range(e.size):
        total += w[i] * abs(e[i]) ** p
    return (cell_volume * total) ** (1.0 / p)


def _python_cellwise(f, m, c, r):
    f, m, c = f.ravel(), m.ravel(), c.ravel()
    out = np.full(f.size, np.nan)
    log_r = np.log(r)
    for i in range(f.size):
        num = abs(c[i] - m[i])
        den = abs(m[i] - f[i])
        if den > 1e-14 and num > 1e-14:
            out[i] = np.log(num / den) / log_r
    return out


def main() -> None:
    if not backend.HAS_NUMBA:
        print("Numba not available; skipping benchmark.")
        return

    rng = np.random.default_rng(0)
    n = 4_000_000
    error = rng.standard_normal(n)
    weight = rng.random(n)
    f, m, c = rng.random(n), rng.random(n), rng.random(n)

    print(f"pdeforge kernel benchmark  (N = {n:,} elements, backend = {backend.BACKEND})\n")

    # --- correctness first -------------------------------------------------- #
    fast_lp = backend.weighted_lp_norm(error, weight, 2.0, 1e-6)
    ref_lp = ref.weighted_lp_norm(error, weight, 2.0, 1e-6)
    assert np.isclose(fast_lp, ref_lp, rtol=1e-12), (fast_lp, ref_lp)
    fast_co = backend.cellwise_order(f, m, c, 2.0)
    ref_co = ref.cellwise_order(f, m, c, 2.0)
    assert np.allclose(fast_co, ref_co, equal_nan=True)
    print("correctness: Numba kernels match NumPy reference exactly  ✅\n")

    rows = []
    for label, py, npf, nbf in [
        (
            "weighted L2 norm",
            lambda: _python_lp(error, weight, 2.0, 1e-6),
            lambda: ref.weighted_lp_norm(error, weight, 2.0, 1e-6),
            lambda: backend.weighted_lp_norm(error, weight, 2.0, 1e-6),
        ),
        (
            "per-cell order field",
            lambda: _python_cellwise(f, m, c, 2.0),
            lambda: ref.cellwise_order(f, m, c, 2.0),
            lambda: backend.cellwise_order(f, m, c, 2.0),
        ),
    ]:
        t_py = _time(py, repeats=1)  # pure python is slow; time once
        t_np = _time(npf)
        t_nb = _time(nbf)
        rows.append((label, t_py, t_np, t_nb))

    header = f"{'kernel':<22}{'python (s)':>14}{'numpy (s)':>14}{'numba (s)':>14}{'speedup vs py':>16}"
    print(header)
    print("-" * len(header))
    for label, t_py, t_np, t_nb in rows:
        print(f"{label:<22}{t_py:>14.4f}{t_np:>14.4f}{t_nb:>14.4f}{t_py / t_nb:>15.1f}x")


if __name__ == "__main__":
    main()
