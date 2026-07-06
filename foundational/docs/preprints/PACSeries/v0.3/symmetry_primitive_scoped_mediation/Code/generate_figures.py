#!/usr/bin/env python3
"""
generate_figures.py — Regenerate PACSeries Paper 7 figures from Data/results/.

Usage:
    python generate_figures.py        # writes PNGs into ../Figures/

Reads the frozen result snapshots in ../Data/results/ (not the source repo),
so figures are reproducible from the shipped package alone.

Figures:
    fig1_force_hierarchy.png   — force coupling vs Fibonacci cascade depth (M6 exp_04)
    fig2_alpha_ranking.png     — alpha_EM: #1 of 10,440 Fibonacci combinations (M6 exp_09)
    fig3_scope_attenuation.png — emergent 1/phi attenuation ratio (M7 exp_04)
    fig4_phi_convergence.png   — phi (and b-nacci) from relational self-reference (M7 exp_01)
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
BLUE, GREY, RED, GOLD = "#2b6cb0", "#a0aec0", "#c53030", "#b7791f"
PHI = (1 + 5 ** 0.5) / 2
INV_PHI = 1 / PHI


def load(pat):
    files = sorted(glob.glob(str(DATA / pat)))
    return json.load(open(files[-1])) if files else None


def fig_force_hierarchy():
    """Coupling strength vs Fibonacci cascade depth; alpha ~ phi^-d."""
    d = load("exp_04_coupling*")
    alpha_em = d["alpha_em"]["dft_formula"]                 # depth 13
    alpha_g = 10 ** d["alpha_g"]["log10_predicted"]         # depth 183
    alpha_s = d["strong_coupling"]["measured"]              # depth ~3
    # weak coupling ~ phi^-7 (paper Section 10.1)
    forces = [("Strong", 3, alpha_s),
              ("Weak", 7, INV_PHI ** 7),
              ("EM", 13, alpha_em),
              ("Gravity", 183, alpha_g)]
    depths = np.array([f[1] for f in forces])
    vals = np.array([f[2] for f in forces])
    dd = np.linspace(1, 190, 400)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(dd, INV_PHI ** dd, color=GREY, lw=1.3, ls="--",
            label=r"$\varphi^{-d}$ (Fibonacci-depth law)")
    ax.scatter(depths, vals, color=BLUE, zorder=5, s=60)
    # stagger labels so the three shallow-depth forces do not overlap
    label_offsets = {"Strong": (10, 6e0), "Weak": (12, 3e-2),
                     "EM": (18, 2e-4), "Gravity": (150, 3e-36)}
    for name, dp, v in forces:
        xt, yt = label_offsets[name]
        ax.annotate(f"{name} (d={dp})", xy=(dp, v), xytext=(xt, yt),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7),
                    fontsize=9, color=BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("Fibonacci cascade depth $d$")
    ax.set_ylabel(r"coupling strength $\alpha$")
    ax.set_title(r"Force hierarchy from Fibonacci depth: $\alpha \sim \varphi^{-d}$")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_force_hierarchy.png")
    plt.close(fig)


def fig_alpha_ranking():
    """alpha_EM formula ranks #1 of 10,440 Fibonacci combinations."""
    d = load("exp_09_alpha_em*")
    cs = d["combinatorial_search"]
    top5 = cs["top5"]
    ppm = [e["ppm"] for e in top5]
    labels = ["#1\n(DFT)"] + [f"#{i}" for i in range(2, len(ppm) + 1)]
    colors = [RED] + [GREY] * (len(ppm) - 1)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(labels, ppm, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("error (ppm) — lower is better")
    ax.set_title(f"$\\alpha_{{EM}}$ formula ranks #1 of {cs['total_combinations']:,} "
                 "Fibonacci combinations")
    ax.text(0, ppm[0] * 1.4, f"{ppm[0]:.1f} ppm", ha="center", color=RED, fontsize=10)
    factor = ppm[1] / ppm[0]
    ax.annotate(f"{factor:.0f}$\\times$ better\nthan #2", xy=(1, ppm[1]),
                xytext=(1.6, ppm[1] * 0.5),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_alpha_ranking.png")
    plt.close(fig)


def fig_scope_attenuation():
    """Emergent per-level attenuation ratio clusters near 1/phi (M7 exp_04)."""
    d = load("exp_04_inv_phi_attenuation*")
    t1 = d["test1_emergent"]
    ratios = np.array(t1["all_ratios"])
    mean_r = t1["overall_ratio"]
    r2 = t1["overall_r2"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(ratios, bins=12, color=BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(INV_PHI, color=GOLD, lw=2, ls=":",
               label=r"$1/\varphi = 0.618$")
    ax.axvline(mean_r, color=RED, lw=2,
               label=f"emergent mean = {mean_r:.3f}")
    ax.axvline(0.5, color=GREY, lw=1.5, ls="--",
               label="single-scale control = 0.5")
    ax.set_xlabel("measured inter-level decay ratio")
    ax.set_ylabel("count")
    ax.set_title(f"Emergent $1/\\varphi$ scope attenuation from multi-scale drive "
                 f"($R^2={r2:.3f}$)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_scope_attenuation.png")
    plt.close(fig)


def fig_phi_convergence():
    """phi (and b-nacci constants) from cross-scale relational self-reference."""
    d = load("exp_01_self_reference*")
    fc = d["fibonacci_convergence"]
    bs = sorted(int(b) for b in fc)
    measured = [fc[str(b)]["gen_phi"] for b in bs]
    predicted = [fc[str(b)]["R_predicted"] for b in bs]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(bs, predicted, "o-", color=GREY, lw=1.4, ms=9,
            label="predicted $b$-nacci root")
    ax.plot(bs, measured, "x", color=RED, ms=11, mew=2,
            label="measured (relational self-reference)")
    ax.scatter([2], [measured[0]], s=220, facecolors="none",
               edgecolors=GOLD, lw=2, zorder=6)
    ax.annotate(r"$b=2:\ \varphi = 1.6180$", xy=(2, measured[0]),
                xytext=(2.3, 1.68), fontsize=10, color=GOLD)
    ax.set_xticks(bs)
    ax.set_xlabel("branching factor $b$")
    ax.set_ylabel("cross-scale ratio $R = D/S$")
    ax.set_title(r"$\varphi$ from relational self-reference (subordinate$_n$ = dominant$_{n+1}$)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_phi_convergence.png")
    plt.close(fig)


def main():
    for fn in (fig_force_hierarchy, fig_alpha_ranking,
               fig_scope_attenuation, fig_phi_convergence):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
