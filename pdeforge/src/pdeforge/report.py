"""Publication-quality V&V report generation (figures + Markdown).

Turns a :class:`~pdeforge.study.ConvergenceStudy` into a self-contained report:
a log-log convergence plot with theoretical/observed reference slopes, a
solution/error figure, a per-cell order-of-accuracy map, a grid-refinement
table, the GCI uncertainty summary, and a bold PASS/FAIL verdict -- the
deliverable a simulation engineer would otherwise spend a day assembling by
hand.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .study import ConvergenceStudy  # noqa: E402

__all__ = ["generate_report"]


def _plot_convergence(study: ConvergenceStudy, path: Path) -> None:
    hs = np.asarray(study.hs)
    err = np.asarray(study.errors_l2)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(hs, err, "o-", color="#1f77b4", lw=2, ms=7, label="measured $L_2$ error")

    # Reference slope at the theoretical order, anchored at the coarsest grid.
    p_th = study.theoretical_order
    ref = err[0] * (hs / hs[0]) ** p_th
    ax.loglog(hs, ref, "k--", lw=1.5, label=f"theoretical slope $p={p_th}$")

    ax.set_xlabel("grid spacing $h$")
    ax.set_ylabel(r"discretisation error $\|u_h - u_{exact}\|_2$")
    ax.set_title(
        f"Convergence: observed $p = {study.observed_order:.3f}$ "
        f"($R^2 = {study.r_squared:.4f}$)"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_solution(study: ConvergenceStudy, path: Path) -> None:
    res = study.results[-1]
    if len(res.shape) == 1:
        x = res.coords[0]
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(x, res.u_exact, "k-", lw=2, label="exact")
        ax.plot(x, res.u_numeric, "o", color="#d62728", ms=3, label="numeric")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"Solution at finest grid (n={res.n})")
        ax.legend()
    else:
        x, y = res.coords
        err = (res.u_numeric - res.u_exact).reshape(res.shape)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        c0 = axes[0].pcolormesh(x, y, res.u_numeric.reshape(res.shape).T, shading="auto")
        axes[0].set_title(f"Numeric solution (n={res.n})")
        fig.colorbar(c0, ax=axes[0])
        c1 = axes[1].pcolormesh(x, y, err.T, shading="auto", cmap="coolwarm")
        axes[1].set_title("Pointwise error")
        fig.colorbar(c1, ax=axes[1])
        for ax in axes:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_cell_order(study: ConvergenceStudy, path: Path) -> None:
    field = study.cell_order_field
    coarse = study.results[-3]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    if field.ndim == 1:
        ax.plot(coarse.coords[0], field, "o-", color="#2ca02c", ms=4)
        ax.axhline(study.theoretical_order, color="k", ls="--", label="theoretical")
        ax.set_xlabel("$x$")
        ax.set_ylabel("local observed order $p$")
        ax.legend()
    else:
        extent = (
            float(coarse.coords[0][0]),
            float(coarse.coords[0][-1]),
            float(coarse.coords[1][0]),
            float(coarse.coords[1][-1]),
        )
        im = ax.imshow(field.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax, label="local observed order $p$")
    ax.set_title("Per-cell observed order of accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _convergence_table(study: ConvergenceStudy) -> str:
    lines = [
        "| grid $n$ | spacing $h$ | $L_2$ error | $L_\\infty$ error | pairwise order |",
        "|---------:|------------:|------------:|------------------:|---------------:|",
    ]
    pw = [float("nan")] + study.pairwise  # first grid has no predecessor
    for i, n in enumerate(study.resolutions):
        order = "—" if i == 0 else f"{pw[i]:.3f}"
        lines.append(
            f"| {n} | {study.hs[i]:.3e} | {study.errors_l2[i]:.3e} "
            f"| {study.errors_linf[i]:.3e} | {order} |"
        )
    return "\n".join(lines)


def generate_report(
    study: ConvergenceStudy, output_dir: str | Path, *, title: str | None = None
) -> Path:
    """Render figures and a Markdown report into ``output_dir``; return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    _plot_convergence(study, fig_dir / "convergence.png")
    _plot_solution(study, fig_dir / "solution.png")
    _plot_cell_order(study, fig_dir / "cell_order.png")

    gci = study.gci
    title = title or f"Verification report: {study.problem.name}"
    verdict_banner = (
        "> ## ✅ VERDICT: **PASS**"
        if study.passed
        else "> ## ❌ VERDICT: **FAIL**"
    )

    md = f"""# {title}

{verdict_banner}
>
> Observed order **{study.observed_order:.3f}** vs theoretical **{study.theoretical_order}** \
(deficit {study.order_deficit:+.3f}, tolerance ±{study.tolerance}).

*Generated by pdeforge on {_dt.date.today().isoformat()} — kernel backend: `{study.backend_used}`.*

## Problem

{study.problem.describe()}

The source term above was **derived automatically** by the Method of Manufactured
Solutions, so the exact solution is known to machine precision and the
discretisation error below is exact.

## Grid-refinement study

{_convergence_table(study)}

Least-squares fit of $\\log E$ vs $\\log h$ over all grids gives observed order
**$p = {study.observed_order:.3f}$** with $R^2 = {study.r_squared:.4f}$.

![Convergence](figures/convergence.png)

## Discretisation uncertainty (GCI, Celik 2008)

| quantity | value |
|----------|-------|
| apparent order $p$ | {gci.p:.3f} |
| Richardson-extrapolated value | {gci.phi_extrapolated:.6g} |
| fine-grid GCI | {gci.gci_fine * 100:.3f} % |
| medium-grid GCI | {gci.gci_medium * 100:.3f} % |
| asymptotic-range ratio | {gci.asymptotic_ratio:.4f} |
| in asymptotic range? | {"yes" if gci.in_asymptotic_range else "no"} |

## Solution and error

![Solution](figures/solution.png)

## Per-cell observed order

![Per-cell order](figures/cell_order.png)

## References

1. P. J. Roache, *Verification and Validation in Computational Science and
   Engineering*, Hermosa, 1998.
2. K. Salari, P. Knupp, *Code Verification by the Method of Manufactured
   Solutions*, SAND2000-1444, 2000.
3. I. Celik et al., *Procedure for Estimation and Reporting of Uncertainty Due to
   Discretization in CFD Applications*, J. Fluids Eng. 130(7), 2008.
4. ASME V&V 20-2009, *Standard for Verification and Validation in Computational
   Fluid Dynamics and Heat Transfer*.
"""
    report_path = out / "report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path
