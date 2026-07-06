#!/usr/bin/env python3
"""
generate_figures.py — Regenerate PACSeries Paper 11 figures from Data/results/.

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
BLUE, GREY, RED = "#2b6cb0", "#a0aec0", "#c53030"


def load(pat):
    files = sorted(glob.glob(str(DATA / pat)))
    return json.load(open(files[-1])) if files else None


def fig_noncommutativity():
    d = load("exp_p13_bell*")
    sweep = d["T3"]["sweep"]
    types = [e["type"] for e in sweep]
    nc = [e["NC"] for e in sweep]
    colors = [RED if v > 1e-9 else GREY for v in nc]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(types, nc, color=colors)
    ax.set_ylabel(r"$\mathcal{NC}$  (non-commutativity)")
    ax.set_title("Non-commutativity across ADE types — only $D_4$ (triality) is non-zero")
    ax.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(nc):
        if v > 1e-9:
            ax.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=9, color=RED)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_noncommutativity.png")
    plt.close(fig)


def fig_bell_chsh():
    d = load("exp_p13_bell*")
    sweep = d["T3"]["sweep"]
    types = [e["type"] for e in sweep]
    smax = [e["S_max"] for e in sweep]
    tsirelson = 2 * np.sqrt(2)
    colors = [BLUE if s > 2 else GREY for s in smax]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(types, smax, color=colors)
    ax.axhline(2.0, color="k", ls="--", lw=1, label="classical bound $S=2$")
    ax.axhline(tsirelson, color=RED, ls=":", lw=1.2, label=r"Tsirelson $2\sqrt{2}=2.828$")
    ax.set_ylabel("CHSH $S_{\\max}$ (orbit-Laplacian rotation)")
    ax.set_ylim(0, 3.0)
    ax.set_title("Bell violation by topology: trivial Aut ($E_7,E_8$) cannot violate ($S<2$)")
    ax.legend(fontsize=9, loc="lower right")
    # honest note: orbit-Laplacian rotation on D4xD4 reaches 2.514 (88.8% of Tsirelson);
    # full Tsirelson saturation requires the ideal SU(2) generator (T5).
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_bell_chsh.png")
    plt.close(fig)


def fig_zeno():
    d = load("exp_p15_path*")
    zd = d["T3"]["zeno_data"]
    N = [e["N"] for e in zd]
    P = [e["P_survive"] for e in zd]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(N, P, "o-", color=BLUE, lw=1.6)
    ax.set_xscale("log")
    ax.axhline(P[0], color=GREY, ls="--", lw=1, label=f"free evolution ($N=1$): {P[0]:.3f}")
    # mark anti-Zeno dip (N=2) and Zeno freezing (large N)
    ax.annotate("anti-Zeno dip", xy=(N[1], P[1]), xytext=(N[1] * 1.3, P[1] - 0.12),
                arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=9)
    ax.annotate(f"Zeno freezing\n$P={P[-1]:.3f}$", xy=(N[-1], P[-1]),
                xytext=(N[-1] * 0.25, P[-1] - 0.18),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
    ax.set_xlabel("number of measurements $N$")
    ax.set_ylabel("survival probability $P_\\mathrm{survive}$")
    ax.set_title("Quantum Zeno / anti-Zeno on $D_4$ orbit space")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_zeno.png")
    plt.close(fig)


def fig_decoherence():
    d = load("exp_p16_deco*")
    t4 = d["T4"]
    labels = ["orbit eigenstate\n(pointer basis)", "superposition\n$|+\\rangle$"]
    vals = [t4["min_purity_orbit"], t4["min_purity_super"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=[BLUE, RED])
    ax.set_ylabel("minimum purity under dephasing")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_title("Einselection: the orbit basis is the pointer basis")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_decoherence.png")
    plt.close(fig)


def main():
    for fn in (fig_noncommutativity, fig_bell_chsh, fig_zeno, fig_decoherence):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
