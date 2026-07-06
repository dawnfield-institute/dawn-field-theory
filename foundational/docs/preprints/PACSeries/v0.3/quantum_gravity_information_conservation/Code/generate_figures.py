#!/usr/bin/env python3
"""
generate_figures.py — Regenerate PACSeries Paper 8 figures from Data/results/.

Usage:
    python generate_figures.py        # writes PNGs into ../Figures/

Reads the frozen result snapshots in ../Data/results/ (not the source repo),
so figures are reproducible from the shipped package alone.
"""
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data" / "results"
FIGS = HERE.parent / "Figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, GREY, RED, GREEN = "#2b6cb0", "#a0aec0", "#c53030", "#2f855a"


def load(pat):
    files = sorted(glob.glob(str(DATA / pat)))
    return json.load(open(files[-1])) if files else None


def fig_planck_routes():
    """Four routes to the Planck scale — a bracket, not a convergence (exp_02)."""
    d = load("exp_02_planck_from_negotiation_*")
    routes = d["T1"]["routes"]
    order = ["heisenberg", "landauer", "negotiation", "schwarzschild"]
    labels = ["Heisenberg\n(0.5)", "Landauer\n1/ln2=1.44",
              "Negotiation\nL_MVAE=1.63", "Schwarzschild\n(2.0)"]
    vals = [routes[k]["l_planck_units"] for k in order]
    # inner routes (Landauer, Negotiation) vs outer bracket (Heisenberg, Schwarzschild)
    colors = [GREY, BLUE, BLUE, GREY]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(labels, vals, color=colors)
    ax.axhspan(routes["landauer"]["l_planck_units"],
               routes["negotiation"]["l_planck_units"],
               color=BLUE, alpha=0.10)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(r"$\ell$  (Planck units)")
    ax.set_title("Four routes to the Planck scale: inner pair converges "
                 f"{d['T1']['inner_spread']:.2f}$\\times$, outer pair brackets "
                 f"({d['T1']['full_spread']:.0f}$\\times$ span)")
    ax.set_ylim(0, 2.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_planck_routes.png")
    plt.close(fig)


def fig_page_curve_unitarity():
    """epsilon-PAC violation prevents the Page curve returning to zero (exp_06)."""
    d = load("exp_06_page_curve_unitarity_*")
    er = d["T4"]["epsilon_results"]
    eps = [e["epsilon"] for e in er]
    final = [e["final_entropy"] for e in er]
    peak = [e["peak_entropy"] for e in er]
    x = np.arange(len(eps))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w/2, peak, w, color=GREY, label="peak entropy (Page time)")
    ax.bar(x + w/2, final, w, color=RED, label="final entropy ($k=N$)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\varepsilon$={e}" for e in eps])
    ax.set_ylabel("entanglement entropy (nats)")
    ax.set_title("Page curve unitarity: only exact PAC ($\\varepsilon=0$) "
                 "returns entropy to zero")
    ax.annotate("returns to 0\n(full unitarity)", xy=(0, 0.3),
                xytext=(0.4, max(peak) * 0.45), fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_page_curve_unitarity.png")
    plt.close(fig)


def fig_hawking_coefficient():
    """Hawking T*M = 1/(8pi) from cascade geometry (4pi solid angle x 2 round-trip) (exp_05)."""
    d = load("exp_05_hawking_from_pac_*")
    tm = d["T1"]["mean_TM_standard"]
    cv = d["T1"]["cv_standard"]
    t2 = d["T2"]
    computed = t2["coefficient"]
    expected = t2["expected"]
    denom = t2["geometric_denominator"]      # 8*pi = 25.133
    solid = t2["solid_angle"]                # 4*pi
    rt = t2["round_trip"]                    # 2
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(["computed\n(cascade geometry)", "$1/(8\\pi)$\n(Hawking)"],
                  [computed, expected], color=[BLUE, GREY], width=0.55)
    for b, v in zip(bars, [computed, expected]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.0008, f"{v:.7f}",
                ha="center", fontsize=10)
    ax.set_ylabel(r"$T\cdot M$  (Planck units)")
    ax.set_ylim(0, expected * 1.35)
    ax.set_title("Hawking radiation from PAC conservation: "
                 r"$T\cdot M = 1/(8\pi)$ from geometry alone")
    ax.text(0.5, 0.72,
            f"$8\\pi = 4\\pi \\times 2$  (solid angle {solid:.3f} $\\times$ "
            f"round-trip {rt})\ndenominator $= {denom:.3f}$\n"
            f"constant across 12 orders of mass  (CV $= {cv:.1e}$)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#edf2f7", ec=GREY))
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_hawking_coefficient.png")
    plt.close(fig)


def fig_arrow_of_time():
    """Super-exponential irreversibility: Loschmidt echo error ~ phi^(2n) (exp_09)."""
    d = load("exp_09_stochastic_irreversibility_*")
    ed = d["T2"]["echo_data"]
    n = [e["n"] for e in ed]
    err = [e["echo_error"] for e in ed]
    phi = (1 + 5 ** 0.5) / 2
    ref = [err[0] * phi ** (ni - n[0]) for ni in n]   # phi^n: base-phi per cascade level
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.semilogy(n, err, "o-", color=BLUE, lw=1.6, label="Loschmidt echo error")
    ax.semilogy(n, ref, "--", color=RED, lw=1.2, label=r"$\varphi^{n}$ (per-level Landauer factor)")
    ax.set_xlabel("cascade depth $n$")
    ax.set_ylabel("forward/reverse divergence")
    ax.set_title("Arrow of time: forward/reverse echo diverges as $\\varphi^{n}$ "
                 "(Landauer irreversibility)")
    ax.annotate(f"$n=100$: {err[4]:.1e}", xy=(n[4], err[4]),
                xytext=(n[1], err[4] * 3), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_arrow_of_time.png")
    plt.close(fig)


def main():
    for fn in (fig_planck_routes, fig_page_curve_unitarity,
               fig_hawking_coefficient, fig_arrow_of_time):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
