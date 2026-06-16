# pdeforge

**Forge numerically *verified* PDE solvers from symbolic equations — and let the
tool write the V&V report and catch the bugs.**

`pdeforge` automates the single most respected, most tedious, and most
error-prone task in scientific computing: **solution verification**. You hand it
a PDE written in plain mathematics; it manufactures an exact solution, derives
the source term symbolically, generates a fast finite-difference solver, runs a
grid-refinement study, measures the **observed order of accuracy**, computes the
**Grid Convergence Index** (ASME V&V 20 / Celik 2008), and emits a
publication-quality report with an automated **PASS / FAIL** verdict.

It is grounded end-to-end in the standard verification & validation literature
(Roache 1998; Salari & Knupp 2000; Celik et al. 2008; Oberkampf & Roy 2010) —
nothing here is a black box.

---

## The 30-second pitch: it finds bugs a human reviewer misses

A correct advection-diffusion solver uses 2nd-order central differencing. Someone
"adds a little upwinding for stability" — a plausible one-line change that ships
in real codebases — and silently destroys the formal order of accuracy. No human
inspects the stencil. `pdeforge` runs the convergence study on both builds:

```
build                               observed order   verdict
------------------------------------------------------------
correct  (2nd-order central)                 2.007      PASS
buggy    (1st-order upwind)                  0.988      FAIL

pdeforge detected the injected defect automatically — no human in the loop. ✅
```

That is `examples/bug_injection_demo.py`, verbatim. The buggy build's report is
committed at [`docs/sample_report_failing/report.md`](docs/sample_report_failing/report.md);
a passing one at [`docs/sample_report/report.md`](docs/sample_report/report.md).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # numpy, scipy, sympy, numba, matplotlib

pdeforge demo                  # verify every built-in problem, write reports
python examples/bug_injection_demo.py
python benchmarks/bench_kernels.py
pytest -q                      # the test-suite *is* the proof (see below)
```

In Python:

```python
import pdeforge as pf

problem = pf.heat_2d()                      # symbolic spec; MMS source derived automatically
study   = pf.run_convergence_study(problem) # solve on refined grids, measure order, run GCI
print(study.verdict, round(study.observed_order, 3))   # -> PASS 2.0
pf.generate_report(study, "reports/heat_2d")            # figures + Markdown V&V report
```

---

## How it works — the pipeline

```
   symbolic PDE              MMS                 FD solver            verification           report
  (sympy, plain math) ─▶ manufacture exact ─▶ method of lines /  ─▶ observed order,    ─▶ figures + Markdown
   u_t = α∇²u + f         solution & derive    Crank–Nicolson /     GCI, asymptotic-       + PASS/FAIL verdict
                          source term f        sparse direct        range check
```

1. **Specify** (`problem.py`, `mms.py`). Write the PDE and a smooth manufactured
   solution `u_exact` with `sympy`. The source term `f = L[u_exact]` is derived
   symbolically, so `u_exact` solves the PDE **exactly** and the discretisation
   error is known to machine precision on every grid. This is the
   Method of Manufactured Solutions — the gold standard for code verification.
2. **Discretise** (`operators.py`, `solver.py`). 2nd/4th-order finite-difference
   operators as `scipy.sparse` matrices. Time integration via Crank–Nicolson
   (heat/advection-diffusion), an explicit 2nd-order scheme (wave), or a single
   sparse solve (Poisson). Dirichlet/Neumann boundaries handled by interior/
   boundary partitioning so the operators stay clean and the convergence is
   unpolluted.
3. **Verify** (`verify.py`, `study.py`). Two complementary stories: a *direct*
   least-squares fit of error vs `h` (when the exact solution is known), and the
   *three-grid Roache/Celik GCI* with an asymptotic-range check (the everyday
   case where it is not).
4. **Report** (`report.py`). A convergence plot with reference slopes, a
   solution/error figure, a per-cell order-of-accuracy map, a grid table, the GCI
   uncertainty band, and the verdict.

---

## Why it convinces skeptics

- **Every claim is checkable.** The test-suite is *self-validating*: it asserts
  that each scheme hits its **theoretical** order (2nd-order schemes → observed
  `p ≈ 2.00`, `R² = 1.0000`) and that the GCI lands in the asymptotic range. If
  the numerics were wrong, the tests would fail.
- **It catches real defects.** `test_bug_detection.py` injects a degraded stencil
  and asserts the verdict flips to **FAIL** — the tool earns trust by finding
  bugs, not by asserting correctness.
- **The fast path is the same computation.** The Numba kernels are checked
  bit-for-bit against a readable NumPy reference (`test_backend_parity.py`), and
  there is a pure-NumPy fallback if Numba is unavailable.
- **Standards, not vibes.** Roache (1998), Salari & Knupp (SAND2000-1444),
  Celik et al. (J. Fluids Eng. 2008), ASME V&V 20-2009.

---

## Performance (Numba)

The two performance-critical kernels — the weighted `Lᵖ` error norm and the
per-cell observed-order field — are branchy reductions that vectorise awkwardly,
so they are JIT-compiled with `@njit(parallel=True)`. Measured on 4,000,000
elements (`benchmarks/bench_kernels.py`), having first **asserted** the Numba
result equals the NumPy reference:

| kernel               | pure Python | NumPy   | Numba   | speedup vs Python |
|----------------------|------------:|--------:|--------:|------------------:|
| weighted L2 norm     |   1.150 s   | 0.028 s | 0.015 s |            ~78×   |
| per-cell order field |   2.968 s   | 0.128 s | 0.010 s |           ~293×   |

Numba also beats the vectorised NumPy reference on both kernels (single-pass,
parallel, no temporaries). If Numba is not installed, `pdeforge` runs identically
— just slower — through the NumPy path; `pdeforge.BACKEND` reports which ran.

---

## Supported problems

| factory                       | equation                                   | scheme                         | order |
|-------------------------------|--------------------------------------------|--------------------------------|:-----:|
| `heat_1d`, `heat_2d`          | `u_t = α∇²u + f`                            | Crank–Nicolson                 |   2   |
| `advection_diffusion_1d`      | `u_t + v·∇u = α∇²u + f`                     | central (or upwind, 1st order) |   2   |
| `poisson_2d`                  | `−α∇²u = f`                                 | sparse direct                  |   2   |
| `wave_1d`                     | `u_tt = c²∇²u + f`                          | explicit central-in-time       |   2   |

A 4th-order Laplacian is available (`spatial_order=4`) and the framework detects
the higher rate automatically.

---

## Project layout

```
src/pdeforge/
  problem.py      symbolic PDE spec + factory problems
  mms.py          Method of Manufactured Solutions (symbolic source derivation)
  operators.py    finite-difference stencils -> scipy.sparse matrices
  solver.py       method-of-lines / Crank–Nicolson / direct solves
  verify.py       error norms, observed order, Richardson, GCI (Celik 2008)
  study.py        orchestrates the refinement study + PASS/FAIL verdict
  report.py       matplotlib figures + Markdown report
  kernels.py      Numba JIT kernels      kernels_numpy.py  NumPy reference
  backend.py      transparent numba/numpy selector
  cli.py          `pdeforge run` / `pdeforge demo`
tests/            self-validating convergence, bug-detection, parity, MMS, math
examples/         one runnable script per problem + the bug-injection demo
benchmarks/       kernel speedup table (with correctness assertion)
docs/             committed sample reports (a PASS and the headline FAIL)
```

## License

MIT — see [LICENSE](LICENSE).
