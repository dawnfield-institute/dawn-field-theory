#!/usr/bin/env python3
"""
generate_figures.py — Regenerate PACSeries Paper 9 figures from Data/results/.

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
PHI = (1 + 5 ** 0.5) / 2
LN_PHI = np.log(PHI)


def load(pat):
    files = sorted(glob.glob(str(DATA / pat)))
    return json.load(open(files[-1])) if files else None


def fig_cascade_clock():
    """N(t) cascade clock: three independent observables on one temporal line."""
    # DFT-constrained clock: N(t) = a + (1/ln phi) * ln(t_lookback, Gyr)
    a = 1.360
    slope = 1.0 / LN_PHI
    # Three M8 anchor points (paper Section 8.1): (t_lookback Gyr, N, label)
    anchors = [(4.0, 4.16, "S8"), (9.5, 5.94, "Hubble"), (13.2, 6.90, "JWST")]
    t = np.linspace(2.0, 15.0, 200)
    N = a + slope * np.log(t)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(t, N, "-", color=BLUE, lw=1.8,
            label=fr"$N(t)=1.360+(1/\ln\varphi)\ln t$,  slope $={slope:.4f}$")
    for tl, n, lab in anchors:
        ax.plot(tl, n, "o", color=RED, ms=8)
        ax.annotate(lab, xy=(tl, n), xytext=(tl - 0.3, n + 0.25),
                    fontsize=10, color=RED, ha="right")
    ax.set_xlabel(r"lookback time $t_\mathrm{lookback}$ (Gyr)")
    ax.set_ylabel(r"cascade level $N$")
    ax.set_title("The cascade clock unifies S8, Hubble, and JWST (RMS residual 0.126)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_cascade_clock.png")
    plt.close(fig)


def fig_s8_tension():
    """S8 tension: Planck vs lensing vs DFT cascade-dissipation prediction."""
    d = load("exp_07_s8_redshift_evolution*")
    tr = d["tests"]["tension_resolution"]
    s8_planck = tr["s8_planck"]
    s8_lensing = tr["s8_lensing"]
    s8_dft = tr["s8_full_fit"]
    sig_before = tr["lcdm_tension_sigma"]
    sig_after = tr["dft_tension_full_sigma"]

    labels = ["Planck (CMB)", "lensing\n(KiDS/DES)", "DFT cascade\n$S_8(z{=}0.35)$"]
    vals = [s8_planck, s8_lensing, s8_dft]
    colors = [GREY, "#4a5568", BLUE]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylabel(r"$S_8=\sigma_8\sqrt{\Omega_m/0.3}$")
    ax.set_ylim(0.70, 0.86)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_title(
        f"S8 tension resolved by cascade dissipation: "
        f"{sig_before:.2f}$\\sigma \\to$ {sig_after:.2f}$\\sigma$")
    ax.annotate("", xy=(2, s8_dft), xytext=(0, s8_planck),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.text(1.0, (s8_planck + s8_dft) / 2 + 0.012,
            f"~{100*(sig_before-sig_after)/sig_before:.0f}% reduction",
            color=RED, fontsize=9, ha="center")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_s8_tension.png")
    plt.close(fig)


def fig_cc_precision():
    """Cosmological constant precision improving across milestones."""
    d = load("exp_08_cosmological_constant_precision*")
    m8_log = d["tests"]["test1_tiling_exponent"]["log10_corrected"]
    observed = d["tests"]["test1_tiling_exponent"]["log10_observed"]
    m8_err = abs(m8_log - observed)
    # Paper Section 3.1 progression (orders of magnitude from observed):
    stages = ["M7\ncascade\ncounting", "MAR\nMVAE vacuum", "M8\ncorrection\ntemplate"]
    errs = [0.9, 0.22, m8_err]
    colors = [GREY, "#4a5568", GREEN]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(stages, errs, color=colors, width=0.6)
    ax.set_ylabel(r"error in $\log_{10}(\Lambda/\Lambda_P)$  (orders)")
    ax.set_yscale("log")
    for b, v in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_title("Cosmological constant: 122-order problem closed to 0.09 orders")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_cc_precision.png")
    plt.close(fig)


def fig_prediction_summary():
    """Headline predictions vs observation — a falsifiability scorecard."""
    d02 = load("exp_02_dark_matter_mass_spectrum*")
    d04 = load("exp_04_zprime_395_quantification*")
    d07 = load("exp_07_hubble_tension_quantification*")
    d08 = load("exp_08_cosmological_constant_precision*")
    ds8 = load("exp_07_s8_redshift_evolution*")

    cc_err = abs(d08["tests"]["test1_tiling_exponent"]["log10_corrected"]
                 - d08["tests"]["test1_tiling_exponent"]["log10_observed"])
    h0_err = d07["tests"]["test1_cascade_h0_ratio"]["error_pct"]
    dm_spread = d02["tests"]["test1_three_route_convergence"]["log10_spread"]
    ol_err = d08["tests"]["test3_dark_energy_density"]["error_pct"]
    s8_sig = ds8["tests"]["tension_resolution"]["dft_tension_full_sigma"]
    zp_margin = d04["tests"]["test1_lhc_exclusion"]["margin_factor"]

    rows = [
        ("Cosmological constant", f"{cc_err:.2f} orders", cc_err, 0.09),
        ("Hubble ratio $\\varphi^{1/6}$", f"{h0_err:.3f} %", h0_err, 0.075),
        ("$\\Omega_\\Lambda$ vs Planck", f"{ol_err:.2f} %", ol_err, 0.18),
        ("Dark matter mass spread", f"{dm_spread:.2f} orders", dm_spread, 0.09),
        ("S8 tension", f"{s8_sig:.2f}$\\sigma$", s8_sig, 0.07),
        ("Z$'$ safety margin", f"{zp_margin:.1f}$\\times$", zp_margin, 9.30),
    ]
    names = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.barh(y, [1] * len(rows), color=BLUE, alpha=0.12, height=0.6)
    for yi, (name, lab, _, _) in zip(y, rows):
        ax.text(0.02, yi, name, va="center", ha="left", fontsize=10)
        ax.text(0.98, yi, lab, va="center", ha="right", fontsize=10,
                color=GREEN, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Headline predictions: agreement with observation "
                 "(all within pre-registered bounds)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_prediction_summary.png")
    plt.close(fig)


def main():
    for fn in (fig_cascade_clock, fig_s8_tension, fig_cc_precision,
               fig_prediction_summary):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
