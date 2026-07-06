#!/usr/bin/env python3
"""
generate_figures.py — Regenerate PACSeries Paper 10 figures from Data/results/.

Usage:
    python generate_figures.py        # writes PNGs into ../Figures/

Reads the frozen result snapshots in ../Data/results/ (not the source repo),
so figures are reproducible from the shipped package alone.

Figures:
  fig1_gauge_closure_f7   — F_7 = 13 = 1 + 3 + 8 + 1 gauge dimension closure (exp_03, M12)
  fig2_lorentz_closure    — 15 so(3,1) commutators close to machine precision (exp_11, M12)
  fig3_killing_signature  — Killing-form signature selectivity: only A_1 -> sl(2,C) is
                            indefinite (Minkowski) (exp_11 M12 + exp_09 M13)
  fig4_scorecard          — per-experiment scores across M12 and M13/13.5
"""
import json
import glob
import re
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
BLUE, GREY, RED, GREEN, PURPLE = "#2b6cb0", "#a0aec0", "#c53030", "#2f855a", "#6b46c1"


def load(pat):
    files = sorted(glob.glob(str(DATA / pat)))
    return json.load(open(files[-1])) if files else None


def parse_sig(s):
    """'(3, 3)' or '(3,3)' -> (3,3)."""
    nums = [int(x) for x in re.findall(r"-?\d+", str(s))]
    return tuple(nums[:2])


def fig_gauge_closure():
    d = load("milestone12/exp_03_*")
    t3 = d["tests"]["T3"]
    parts = [("U(1)", t3["U1_generators"], BLUE),
             ("SU(2)", t3["SU2_generators"], GREEN),
             ("SU(3)", t3["SU3_generators"], PURPLE),
             ("Higgs", t3["Higgs_scalar"], GREY)]
    total = t3["total"]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    left = 0
    for label, val, c in parts:
        ax.barh(0, val, left=left, color=c, edgecolor="white")
        ax.text(left + val / 2, 0, f"{label}\n{val}", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        left += val
    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks(range(0, total + 1))
    ax.set_xlabel("gauge dimension")
    ax.set_title(r"Gauge closure: $F_7 = 13 = 1 + 3 + 8 + 1$ "
                 r"(U(1) + SU(2) + SU(3) + Higgs)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_gauge_closure_f7.png")
    plt.close(fig)


def fig_lorentz_closure():
    d = load("milestone12/exp_11_*")
    t1 = d["tests"]["T1"]
    maxerr = t1["max_closure_error"]
    # The three independent commutator families of so(3,1): [J,J], [K,K], [J,K].
    # exp_11 reports each family's residual plus the overall maximum-closure error.
    fams = [("$[J_i,J_j]=i\\epsilon_{ijk}J_k$", abs(t1.get("JJ_error", 0.0))),
            ("$[K_i,K_j]=-i\\epsilon_{ijk}J_k$", abs(t1.get("KK_error", 0.0))),
            ("$[J_i,K_j]=i\\epsilon_{ijk}K_k$", abs(t1.get("JK_error", 0.0))),
            ("overall max\n(15 relations)", abs(maxerr))]
    labels = [f for f, _ in fams]
    vals = [max(v, 1e-18) for _, v in fams]  # floor exact zeros for log scale
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(range(len(vals)), vals,
                  color=[GREEN, GREEN, GREEN, BLUE])
    ax.set_yscale("log")
    ax.axhline(2.2e-16, color=GREY, ls=":", lw=1.2, label="double-precision $\\epsilon$ (~2.2e-16)")
    for i, (_, v) in enumerate(fams):
        txt = "0 (exact)" if v == 0.0 else f"{v:.1e}"
        ax.text(i, vals[i] * 1.4, txt, ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("commutator closure residual")
    ax.set_ylim(1e-18, 1e-14)
    ax.set_title(r"Lorentz algebra closes: $\mathfrak{sl}(2,\mathbb{C}) \cong \mathfrak{so}(3,1)$"
                 "\nall 15 independent commutators verified below double-precision $\\epsilon$")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_lorentz_closure.png")
    plt.close(fig)


def fig_killing_signature():
    d = load("milestone13/exp_09_*")
    t1 = d["tests"]["T1"]
    algs = [("$\\mathfrak{su}(2)$\n(compact, $A_1$)", parse_sig(t1["su2_signature"])),
            ("$\\mathfrak{su}(3)$\n(compact, $A_2$)", parse_sig(t1["su3_signature"])),
            ("$\\mathfrak{sl}(2,\\mathbb{C})$\n($A_1$ + SEC)", parse_sig(t1["sl2c_signature"]))]
    labels = [a for a, _ in algs]
    npos = [s[0] for _, s in algs]
    nneg = [s[1] for _, s in algs]
    x = np.arange(len(algs))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x, npos, color=GREEN, label="positive eigenvalues (+)")
    ax.bar(x, [-n for n in nneg], color=RED, label="negative eigenvalues (−)")
    for i, (p, n) in enumerate(zip(npos, nneg)):
        ax.text(i, p + 0.15, f"({p},{n})", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Killing-form eigenvalue count")
    ax.set_ylim(-9, 5)
    ax.set_title("Only SEC-complexified $A_1$ gives an indefinite (Lorentzian) Killing form\n"
                 "compact algebras are definite; $\\mathfrak{sl}(2,\\mathbb{C})$ is signature (3,3) → Minkowski (1,3)")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_killing_signature.png")
    plt.close(fig)


def fig_scorecard():
    rows = []
    for ms, tag in (("milestone12", "M12"), ("milestone13", "M13")):
        for f in sorted(glob.glob(str(DATA / ms / "*.json"))):
            d = json.load(open(f))
            name = d["experiment"]
            m = re.search(r"exp_(\d+)", name)
            num = int(m.group(1)) if m else 0
            inv = tag == "M13" and num >= 14
            rows.append((f"{tag}·{num:02d}", d.get("score", 0), d.get("total", 4), tag, inv))
    labels = [r[0] for r in rows]
    frac = [r[1] / r[2] if r[2] else 0 for r in rows]
    colors = []
    for _, s, t, tag, inv in rows:
        f = s / t if t else 0
        if f >= 0.999:
            colors.append(GREEN)
        elif f >= 0.5:
            colors.append("#d69e2e")
        else:
            colors.append(RED)
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(rows)), frac, color=colors)
    for i, (_, s, t, tag, inv) in enumerate(rows):
        ax.text(i, frac[i] + 0.02, f"{s}/{t}", ha="center", fontsize=7)
    # divider between M12 and M13, and M13 core vs investigation
    n12 = sum(1 for r in rows if r[3] == "M12")
    n13core = sum(1 for r in rows if r[3] == "M13" and not r[4])
    ax.axvline(n12 - 0.5, color="k", lw=1)
    ax.axvline(n12 + n13core - 0.5, color=GREY, ls="--", lw=1)
    ax.text(n12 / 2 - 0.5, 1.12, "M12 (49/52)", ha="center", fontsize=10, fontweight="bold")
    ax.text(n12 + n13core / 2 - 0.5, 1.12, "M13 core (48/52)", ha="center", fontsize=10, fontweight="bold")
    ax.text(n12 + n13core + 1.5, 1.12, "M13.5 inv. (5/16)", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("test pass fraction")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Per-experiment scorecard — Paper 10 (green = 4/4, amber = partial, red = majority-fail)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_scorecard.png")
    plt.close(fig)


def main():
    for fn in (fig_gauge_closure, fig_lorentz_closure, fig_killing_signature, fig_scorecard):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
