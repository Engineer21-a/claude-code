"""A curated showcase: define several problems, verify them, plot and tabulate.

Run with::

    python examples/showcase.py

Produces a console results table plus three figures in ``reports/showcase/``:
a combined convergence plot, an observed-vs-theoretical order bar chart, and a
panel of per-cell order maps. The narrative interpretation lives in the README
of the run and in the chat that accompanies this script.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pdeforge as pf  # noqa: E402

RES_1D = (17, 33, 65, 129, 257)
RES_2D = (17, 33, 65, 129)


@dataclass
class Case:
    key: str
    title: str
    blurb: str
    study: pf.ConvergenceStudy
    color: str


def build_cases() -> list[Case]:
    """Define the problem set and run a verification study for each."""
    specs = [
        (
            "heat1d",
            "1D heat (diffusion)",
            "u_t = α u_xx. Parabolic smoothing; solved with Crank–Nicolson (2nd order).",
            pf.heat_1d(),
            dict(resolutions=RES_1D),
            "#1f77b4",
        ),
        (
            "poisson2d",
            "2D Poisson (steady)",
            "-α∇²u = f. Elliptic equilibrium; one sparse direct solve (2nd order).",
            pf.poisson_2d(),
            dict(resolutions=RES_2D),
            "#2ca02c",
        ),
        (
            "advdiff_central",
            "1D advection–diffusion (central)",
            "u_t + v u_x = α u_xx. Transport + diffusion; 2nd-order central differencing.",
            pf.advection_diffusion_1d(alpha=0.01, velocity=1.0),
            dict(resolutions=RES_1D, advection_scheme="central"),
            "#9467bd",
        ),
        (
            "wave1d",
            "1D wave (hyperbolic)",
            "u_tt = c² u_xx. Non-dissipative propagation; explicit central-in-time (2nd order).",
            pf.wave_1d(),
            dict(resolutions=RES_1D),
            "#ff7f0e",
        ),
        (
            "poisson2d_o4",
            "2D Poisson, 4th-order stencil",
            "Same elliptic problem solved with a 4th-order Laplacian; expected steeper slope.",
            replace(pf.poisson_2d(), theoretical_order=4),
            dict(resolutions=RES_2D, spatial_order=4),
            "#17becf",
        ),
        (
            "advdiff_bug",
            "1D advection–diffusion (BUG: upwind)",
            "Identical problem but advection switched to 1st-order upwind — the injected defect.",
            pf.advection_diffusion_1d(alpha=0.01, velocity=1.0),
            dict(resolutions=RES_1D, advection_scheme="upwind"),
            "#d62728",
        ),
    ]
    cases = []
    for key, title, blurb, problem, kwargs, color in specs:
        # The bug case sets its own expected order to 1 so the verdict is fair:
        # we are asking "is this scheme delivering *its* advertised order?"
        study = pf.run_convergence_study(problem, **kwargs)
        cases.append(Case(key, title, blurb, study, color))
    return cases


def print_table(cases: list[Case]) -> None:
    head = f"{'problem':<38}{'obs. order':>11}{'R^2':>9}{'GCI fine %':>12}{'asympt.':>9}{'verdict':>9}"
    print(head)
    print("-" * len(head))
    for c in cases:
        s = c.study
        print(
            f"{c.title:<38}{s.observed_order:>11.3f}{s.r_squared:>9.4f}"
            f"{s.gci.gci_fine * 100:>12.3f}{s.gci.asymptotic_ratio:>9.3f}{s.verdict:>9}"
        )


def plot_convergence(cases: list[Case], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in cases:
        s = c.study
        style = "o--" if not s.passed else "o-"
        ax.loglog(s.hs, s.errors_l2, style, color=c.color, lw=2, ms=6,
                  label=f"{c.title}  (p≈{s.observed_order:.2f})")
    # Guide slopes for orders 1, 2, 4 anchored to a common point.
    h = np.array([min(min(c.study.hs) for c in cases), max(max(c.study.hs) for c in cases)])
    anchor = max(max(c.study.errors_l2) for c in cases)
    for order, ls in [(1, ":"), (2, "-."), (4, "--")]:
        ax.loglog(h, anchor * (h / h.max()) ** order, ls, color="gray", lw=1, alpha=0.7)
        ax.text(h.min(), anchor * (h.min() / h.max()) ** order, f" slope {order}",
                color="gray", fontsize=8, va="center")
    ax.set_xlabel("grid spacing $h$")
    ax.set_ylabel(r"$L_2$ discretisation error")
    ax.set_title("pdeforge showcase — convergence of every scheme")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_order_bars(cases: list[Case], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = np.arange(len(cases))
    obs = [c.study.observed_order for c in cases]
    th = [c.study.theoretical_order for c in cases]
    ax.bar(xs - 0.2, obs, width=0.4, label="observed", color=[c.color for c in cases])
    ax.bar(xs + 0.2, th, width=0.4, label="theoretical", color="lightgray", edgecolor="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.title.replace(" (", "\n(") for c in cases], fontsize=7, rotation=0)
    ax.set_ylabel("order of accuracy $p$")
    ax.set_title("Observed vs theoretical order")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    out = Path("reports/showcase")
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases()

    print("\nProblem set\n" + "=" * 70)
    for c in cases:
        print(f"• {c.title}\n    {c.blurb}")
    print()
    print_table(cases)

    plot_convergence(cases, out / "convergence_all.png")
    plot_order_bars(cases, out / "order_bars.png")
    # Per-problem reports (figures + Markdown) for the full set.
    for c in cases:
        pf.generate_report(c.study, out / c.key, title=f"Showcase: {c.title}")
    print(f"\nFigures and per-problem reports written under {out}/")


if __name__ == "__main__":
    main()
