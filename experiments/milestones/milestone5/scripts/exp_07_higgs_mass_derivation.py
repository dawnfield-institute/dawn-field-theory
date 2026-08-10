#!/usr/bin/env python3
"""
Milestone 5 - Exp 07: Higgs Boson Mass from DFT First Principles
=================================================================

GOAL: Derive M_H = 125.25 GeV from Fibonacci / phi / Xi structure,
      consistent with the DFT correction template used for alpha, sin^2(theta_W),
      the W mass, and the muon/electron mass ratio.

EXISTING DFT RESULTS (used as building blocks):
  alpha   = F3/(F4*phi*F10) * (1 - F10/(4*pi*F7^2))   = 1/137.036  (5.7 ppm)
  sin^2(theta_W) = F4/F7 = 3/13 = 0.2308              (0.19% error)
  M_W     = M_Z * sqrt(1 - 3/13) = M_Z * sqrt(10/13)
  Koide Q = F3/F4 = 2/3                                (exact)
  mu/e    = F4 * F6^2 * (1 + 1/F7) = 206.769           (5 ppm)
  Correction template: 1 +/- F_a / (n * pi * F_b^2)

STRATEGY:
  1. Ratio analysis — express M_H/M_Z, M_H/v, lambda in simple terms
  2. Fibonacci combination search — brute-force (F_a, F_b) pairs
  3. Correction template refinement — apply DFT perturbative corrections
  4. Lambda (Higgs quartic) decomposition — lambda = M_H^2/(2*v^2)
  5. Summary table — rank all candidates by precision

This is a PURE THEORY experiment — no simulator, no external deps.
"""

import math
import json
import os
from datetime import datetime
from itertools import product

# ============================================================
# Constants
# ============================================================

PHI      = (1 + math.sqrt(5)) / 2
LN_PHI   = math.log(PHI)
GAMMA_EM = 0.5772156649015329
XI       = GAMMA_EM + LN_PHI          # 1.05843...
PI       = math.pi

# Fibonacci sequence F1..F12
FIB = {i: v for i, v in enumerate([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144], start=1)}
FIB_RANGE = range(1, 13)  # F1..F12

# PDG 2024 measured values
M_H      = 125.25           # GeV  (Higgs mass)
M_H_ERR  = 0.17             # GeV
M_Z      = 91.1876           # GeV  (Z boson mass)
M_W      = 80.377            # GeV  (W boson mass)
VEV      = 246.22            # GeV  (Higgs vacuum expectation value)
G_F      = 1.1663788e-5      # GeV^-2  (Fermi constant)
ALPHA_EM = 1.0 / 137.036     # fine structure constant
ALPHA_S  = 0.1179            # strong coupling at M_Z

# Derived SM quantities
LAMBDA_H = M_H**2 / (2 * VEV**2)     # Higgs quartic coupling ~ 0.1293
SIN2_TW  = 1 - (M_W / M_Z)**2        # Weinberg angle from masses
COS_TW   = M_W / M_Z

# DFT Weinberg angle
SIN2_TW_DFT = 3.0 / 13.0             # F4/F7 = 0.23077
COS_TW_DFT  = math.sqrt(1 - SIN2_TW_DFT)  # sqrt(10/13)

# ============================================================
# Helpers
# ============================================================

def ppm_error(predicted, measured):
    """Return error in parts per million."""
    return abs(predicted - measured) / abs(measured) * 1e6

def pct_error(predicted, measured):
    """Return percent error."""
    return abs(predicted - measured) / abs(measured) * 100.0

def fib_label(idx):
    """Pretty label: F3 = 2, etc."""
    return f"F{idx}={FIB[idx]}"


def separator(title):
    """Section separator for output."""
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


# ============================================================
# Section 1: Ratio Analysis
# ============================================================

def ratio_analysis():
    """Compute key mass ratios and identify near-Fibonacci structure."""
    separator("1. RATIO ANALYSIS")

    ratios = {
        "M_H / M_Z":           M_H / M_Z,
        "M_H / M_W":           M_H / M_W,
        "M_H / v":             M_H / VEV,
        "v / M_H":             VEV / M_H,
        "M_H^2 / v^2":        M_H**2 / VEV**2,
        "lambda = M_H^2/(2v^2)": LAMBDA_H,
        "M_H / (M_Z * phi)":  M_H / (M_Z * PHI),
        "M_H / (M_Z * sqrt(phi))": M_H / (M_Z * math.sqrt(PHI)),
        "v / (M_Z * phi)":    VEV / (M_Z * PHI),
        "M_H * phi / v":      M_H * PHI / VEV,
        "M_H / (v / sqrt(2*phi))": M_H / (VEV / math.sqrt(2 * PHI)),
        "2*lambda":            2 * LAMBDA_H,
        "sqrt(2*lambda)":      math.sqrt(2 * LAMBDA_H),
        "1/lambda":            1.0 / LAMBDA_H,
        "lambda * pi":         LAMBDA_H * PI,
        "lambda * phi":        LAMBDA_H * PHI,
        "lambda * F7":         LAMBDA_H * FIB[7],
        "lambda * F7 * phi":   LAMBDA_H * FIB[7] * PHI,
        "lambda * 4*pi":       LAMBDA_H * 4 * PI,
        "Xi / lambda":         XI / LAMBDA_H,
    }

    print(f"\n{'Ratio':<35s} {'Value':>14s}   Nearest Fibonacci context")
    print("-" * 72)

    results = []
    for name, val in ratios.items():
        # Find nearest F_a/F_b
        best_frac = None
        best_err  = 1e30
        for a in FIB_RANGE:
            for b in FIB_RANGE:
                frac = FIB[a] / FIB[b]
                err = abs(val - frac) / max(abs(val), 1e-30)
                if err < best_err:
                    best_err = err
                    best_frac = (a, b, frac)
        ctx = ""
        if best_frac and best_err < 0.10:
            a, b, frac = best_frac
            ctx = f"~ F{a}/F{b} = {frac:.6f}  ({best_err*100:.2f}%)"
        print(f"  {name:<33s} {val:>14.8f}   {ctx}")
        results.append({"name": name, "value": val, "nearest_fib": ctx})

    return results


# ============================================================
# Section 2: Fibonacci Combination Search
# ============================================================

def fibonacci_search():
    """Brute-force search over Fibonacci pair formulas for M_H."""
    separator("2. FIBONACCI COMBINATION SEARCH")

    candidates = []

    for a in FIB_RANGE:
        fa = FIB[a]
        for b in FIB_RANGE:
            fb = FIB[b]
            if fa == fb and a == b:
                continue  # skip trivial F_a/F_a = 1

            # --- M_H = M_Z * F_a / F_b ---
            pred = M_Z * fa / fb
            if 50 < pred < 300:
                err = ppm_error(pred, M_H)
                candidates.append({
                    "formula": f"M_Z * F{a}/F{b} = M_Z * {fa}/{fb}",
                    "predicted": pred,
                    "ppm": err,
                    "category": "linear_ratio"
                })

            # --- M_H = M_Z * sqrt(F_a / F_b) ---
            if fa / fb > 0:
                pred = M_Z * math.sqrt(fa / fb)
                if 50 < pred < 300:
                    err = ppm_error(pred, M_H)
                    candidates.append({
                        "formula": f"M_Z * sqrt(F{a}/F{b}) = M_Z * sqrt({fa}/{fb})",
                        "predicted": pred,
                        "ppm": err,
                        "category": "sqrt_ratio"
                    })

            # --- M_H = M_Z * F_a * phi / F_b ---
            pred = M_Z * fa * PHI / fb
            if 50 < pred < 300:
                err = ppm_error(pred, M_H)
                candidates.append({
                    "formula": f"M_Z * F{a}*phi/F{b} = M_Z * {fa}*phi/{fb}",
                    "predicted": pred,
                    "ppm": err,
                    "category": "phi_ratio"
                })

            # --- M_H = v * F_a / (F_b * phi^n) for n=0,1,2 ---
            for n in range(3):
                pred = VEV * fa / (fb * PHI**n)
                if 50 < pred < 300:
                    err = ppm_error(pred, M_H)
                    candidates.append({
                        "formula": f"v * F{a}/(F{b}*phi^{n}) = {VEV}*{fa}/({fb}*phi^{n})",
                        "predicted": pred,
                        "ppm": err,
                        "category": "vev_ratio"
                    })

            # --- M_H = v / sqrt(F_a * phi^n) for n=0,1 ---
            for n in range(2):
                denom = fa * PHI**n
                if denom > 0:
                    pred = VEV / math.sqrt(denom)
                    if 50 < pred < 300:
                        err = ppm_error(pred, M_H)
                        candidates.append({
                            "formula": f"v / sqrt(F{a}*phi^{n}) = {VEV}/sqrt({fa}*phi^{n})",
                            "predicted": pred,
                            "ppm": err,
                            "category": "vev_sqrt"
                        })

            # --- lambda = F_a / (F_b * phi^n * pi^m) ---
            for n in range(3):
                for m in range(3):
                    lam = fa / (fb * PHI**n * PI**m)
                    if 0.01 < lam < 1.0:
                        pred_mh = VEV * math.sqrt(2 * lam)
                        if 50 < pred_mh < 300:
                            err = ppm_error(pred_mh, M_H)
                            candidates.append({
                                "formula": (f"v*sqrt(2*F{a}/(F{b}*phi^{n}*pi^{m})) "
                                            f"[lambda={lam:.6f}]"),
                                "predicted": pred_mh,
                                "ppm": err,
                                "category": "lambda_decomp"
                            })

    # --- Special combinations involving Xi ---
    for a in FIB_RANGE:
        fa = FIB[a]
        for b in FIB_RANGE:
            fb = FIB[b]
            # M_H = M_Z * Xi * F_a / F_b
            pred = M_Z * XI * fa / fb
            if 50 < pred < 300:
                err = ppm_error(pred, M_H)
                candidates.append({
                    "formula": f"M_Z * Xi * F{a}/F{b} = M_Z * Xi * {fa}/{fb}",
                    "predicted": pred,
                    "ppm": err,
                    "category": "xi_ratio"
                })
            # M_H = v * Xi / (F_a * phi)
            pred = VEV * XI / (fa * PHI)
            if 50 < pred < 300:
                err = ppm_error(pred, M_H)
                candidates.append({
                    "formula": f"v * Xi / (F{a}*phi) = {VEV}*Xi/({fa}*phi)",
                    "predicted": pred,
                    "ppm": err,
                    "category": "xi_vev"
                })

    # --- phi-power forms: M_H = M_Z * phi^n for real n ---
    # Check: M_H/M_Z = phi^n => n = ln(M_H/M_Z)/ln(phi)
    n_phi = math.log(M_H / M_Z) / math.log(PHI)
    print(f"\n  M_H/M_Z = phi^n  =>  n = {n_phi:.6f}")
    print(f"  (nearest integer/half-integer: {round(2*n_phi)/2:.1f})")

    # --- sqrt(2) * phi forms ---
    pred = M_Z * math.sqrt(2) * PHI**0.5
    err = ppm_error(pred, M_H)
    candidates.append({
        "formula": f"M_Z * sqrt(2) * sqrt(phi) = {pred:.4f}",
        "predicted": pred,
        "ppm": err,
        "category": "special"
    })

    pred = M_Z * PHI * math.sqrt(PHI) / math.sqrt(2)
    err = ppm_error(pred, M_H)
    candidates.append({
        "formula": f"M_Z * phi * sqrt(phi) / sqrt(2) = {pred:.4f}",
        "predicted": pred,
        "ppm": err,
        "category": "special"
    })

    # Sort by precision
    candidates.sort(key=lambda c: c["ppm"])

    print(f"\n  Total candidates generated: {len(candidates)}")
    print(f"\n  TOP 30 CANDIDATES (by ppm error):")
    print(f"  {'Rank':<5s} {'ppm':>10s} {'Predicted':>12s} {'Formula'}")
    print("  " + "-" * 70)
    for i, c in enumerate(candidates[:30]):
        print(f"  {i+1:<5d} {c['ppm']:>10.1f} {c['predicted']:>12.4f}  {c['formula']}")

    return candidates


# ============================================================
# Section 3: Correction Template Search
# ============================================================

def correction_search(base_candidates):
    """
    Take the best base formulas and apply the DFT correction template:
       M_H = base * (1 +/- F_a / (n * pi * F_b^2))
    to see if perturbative corrections improve precision.
    """
    separator("3. CORRECTION TEMPLATE REFINEMENT")

    # Take top 20 base candidates
    top_bases = base_candidates[:20]
    corrected = []

    for base_info in top_bases:
        base_val = base_info["predicted"]
        base_formula = base_info["formula"]

        for a in FIB_RANGE:
            fa = FIB[a]
            for b in FIB_RANGE:
                fb = FIB[b]
                for n in [1, 2, 3, 4]:
                    denom = n * PI * fb**2
                    if denom == 0:
                        continue
                    corr = fa / denom

                    # Plus correction
                    pred_plus = base_val * (1 + corr)
                    err_plus = ppm_error(pred_plus, M_H)
                    corrected.append({
                        "formula": f"[{base_formula}] * (1 + F{a}/({n}*pi*F{b}^2))",
                        "predicted": pred_plus,
                        "ppm": err_plus,
                        "base_ppm": base_info["ppm"],
                        "correction": corr,
                        "sign": "+",
                        "category": "corrected"
                    })

                    # Minus correction
                    pred_minus = base_val * (1 - corr)
                    err_minus = ppm_error(pred_minus, M_H)
                    corrected.append({
                        "formula": f"[{base_formula}] * (1 - F{a}/({n}*pi*F{b}^2))",
                        "predicted": pred_minus,
                        "ppm": err_minus,
                        "base_ppm": base_info["ppm"],
                        "correction": corr,
                        "sign": "-",
                        "category": "corrected"
                    })

    corrected.sort(key=lambda c: c["ppm"])

    print(f"\n  Total corrected candidates: {len(corrected)}")
    print(f"\n  TOP 30 CORRECTED (by ppm error):")
    print(f"  {'Rank':<5s} {'ppm':>10s} {'base ppm':>10s} {'Predicted':>12s} {'Formula'}")
    print("  " + "-" * 90)
    for i, c in enumerate(corrected[:30]):
        print(f"  {i+1:<5d} {c['ppm']:>10.1f} {c['base_ppm']:>10.1f} "
              f"{c['predicted']:>12.6f}  {c['formula']}")

    return corrected


# ============================================================
# Section 4: Lambda (Higgs quartic) Decomposition
# ============================================================

def lambda_analysis():
    """
    Investigate the Higgs quartic coupling lambda = M_H^2/(2*v^2).
    In SM: M_H = v * sqrt(2*lambda).  So lambda is the key parameter.
    """
    separator("4. LAMBDA (HIGGS QUARTIC) DECOMPOSITION")

    print(f"\n  lambda = M_H^2 / (2*v^2) = {LAMBDA_H:.8f}")
    print(f"  1/lambda = {1/LAMBDA_H:.6f}")
    print(f"  lambda * pi = {LAMBDA_H * PI:.8f}")
    print(f"  lambda * 4*pi = {LAMBDA_H * 4 * PI:.8f}")
    print(f"  lambda * phi = {LAMBDA_H * PHI:.8f}")
    print(f"  lambda * F7 = {LAMBDA_H * FIB[7]:.8f}")
    print(f"  lambda / alpha_s = {LAMBDA_H / ALPHA_S:.6f}  (ratio to strong coupling)")
    print(f"  lambda / alpha_EM = {LAMBDA_H / ALPHA_EM:.6f}")
    print(f"  Xi * lambda = {XI * LAMBDA_H:.8f}")
    print(f"  sqrt(lambda) = {math.sqrt(LAMBDA_H):.8f}")
    print(f"  sqrt(lambda) * phi = {math.sqrt(LAMBDA_H) * PHI:.8f}")

    # Search for lambda = F_a / (F_b * phi^n * pi^m * 2^k)
    print(f"\n  Searching for lambda = F_a / (F_b * phi^n * pi^m * 2^k) ...")
    lambda_candidates = []

    for a in FIB_RANGE:
        fa = FIB[a]
        for b in FIB_RANGE:
            fb = FIB[b]
            for n in range(4):
                for m in range(4):
                    for k in range(4):
                        denom = fb * PHI**n * PI**m * 2**k
                        if denom == 0:
                            continue
                        lam = fa / denom
                        if 0.05 < lam < 0.5:
                            err = pct_error(lam, LAMBDA_H)
                            pred_mh = VEV * math.sqrt(2 * lam)
                            mh_ppm = ppm_error(pred_mh, M_H)
                            lambda_candidates.append({
                                "formula": f"F{a}/(F{b}*phi^{n}*pi^{m}*2^{k}) = {fa}/({fb}*phi^{n}*pi^{m}*2^{k})",
                                "lambda": lam,
                                "lambda_err_pct": err,
                                "predicted_MH": pred_mh,
                                "MH_ppm": mh_ppm,
                            })

    # Also test lambda = F_a * phi^n / (F_b * pi^m * 2^k)
    for a in FIB_RANGE:
        fa = FIB[a]
        for b in FIB_RANGE:
            fb = FIB[b]
            for n in range(1, 4):
                for m in range(4):
                    for k in range(4):
                        denom = fb * PI**m * 2**k
                        if denom == 0:
                            continue
                        lam = fa * PHI**n / denom
                        if 0.05 < lam < 0.5:
                            err = pct_error(lam, LAMBDA_H)
                            pred_mh = VEV * math.sqrt(2 * lam)
                            mh_ppm = ppm_error(pred_mh, M_H)
                            lambda_candidates.append({
                                "formula": f"F{a}*phi^{n}/(F{b}*pi^{m}*2^{k}) = {fa}*phi^{n}/({fb}*pi^{m}*2^{k})",
                                "lambda": lam,
                                "lambda_err_pct": err,
                                "predicted_MH": pred_mh,
                                "MH_ppm": mh_ppm,
                            })

    # Xi-based lambda forms
    for a in FIB_RANGE:
        fa = FIB[a]
        for n in range(3):
            for m in range(3):
                denom = fa * PHI**n * PI**m
                if denom == 0:
                    continue
                lam = XI / denom
                if 0.05 < lam < 0.5:
                    err = pct_error(lam, LAMBDA_H)
                    pred_mh = VEV * math.sqrt(2 * lam)
                    mh_ppm = ppm_error(pred_mh, M_H)
                    lambda_candidates.append({
                        "formula": f"Xi/(F{a}*phi^{n}*pi^{m}) = Xi/({fa}*phi^{n}*pi^{m})",
                        "lambda": lam,
                        "lambda_err_pct": err,
                        "predicted_MH": pred_mh,
                        "MH_ppm": mh_ppm,
                    })

    lambda_candidates.sort(key=lambda c: c["MH_ppm"])

    print(f"\n  Total lambda candidates: {len(lambda_candidates)}")
    print(f"\n  TOP 30 LAMBDA DECOMPOSITIONS:")
    print(f"  {'Rank':<5s} {'MH ppm':>10s} {'lambda':>12s} {'lam err%':>10s} {'M_H pred':>12s} {'Formula'}")
    print("  " + "-" * 90)
    for i, c in enumerate(lambda_candidates[:30]):
        print(f"  {i+1:<5d} {c['MH_ppm']:>10.1f} {c['lambda']:>12.8f} "
              f"{c['lambda_err_pct']:>10.4f} {c['predicted_MH']:>12.4f}  {c['formula']}")

    return lambda_candidates


# ============================================================
# Section 5: Physical Consistency Checks
# ============================================================

def physical_checks():
    """
    Verify internal consistency with other DFT results.
    """
    separator("5. PHYSICAL CONSISTENCY CHECKS")

    # Check: M_W from DFT Weinberg angle
    mw_dft = M_Z * COS_TW_DFT
    print(f"\n  DFT Weinberg angle: sin^2(theta_W) = 3/13 = {SIN2_TW_DFT:.6f}")
    print(f"  Measured sin^2(theta_W) = {SIN2_TW:.6f}  (from M_W/M_Z)")
    print(f"  M_W (DFT) = M_Z * sqrt(10/13) = {mw_dft:.4f} GeV  "
          f"(vs measured {M_W} GeV, {ppm_error(mw_dft, M_W):.0f} ppm)")

    # Check: v from G_F
    v_from_gf = 1.0 / math.sqrt(math.sqrt(2) * G_F)
    print(f"\n  v from G_F: v = 1/sqrt(sqrt(2)*G_F) = {v_from_gf:.4f} GeV  "
          f"(vs {VEV} GeV)")

    # Key relationship: M_H = v * M_H/v
    ratio_mh_v = M_H / VEV
    print(f"\n  M_H / v = {ratio_mh_v:.8f}")
    print(f"  sqrt(2*lambda) = {math.sqrt(2*LAMBDA_H):.8f}")
    print(f"  (these should be equal: diff = {abs(ratio_mh_v - math.sqrt(2*LAMBDA_H)):.2e})")

    # Check if M_H/v ~ 1/phi^n or F_a/F_b
    print(f"\n  1/phi   = {1/PHI:.8f}")
    print(f"  1/phi^2 = {1/PHI**2:.8f}")
    print(f"  M_H/v   = {ratio_mh_v:.8f}  (between 1/phi^2 and 1/phi)")
    print(f"  M_H/v * phi = {ratio_mh_v * PHI:.8f}")
    print(f"  F5/F7 = 5/13 = {5/13:.8f}  (vs M_H/v = {ratio_mh_v:.8f}, "
          f"err = {pct_error(5/13, ratio_mh_v):.3f}%)")
    print(f"  F4/F6 = 3/8 = {3/8:.8f}   (vs M_H/(v*sqrt(2)) = "
          f"{ratio_mh_v/math.sqrt(2):.8f}, err = {pct_error(3/8, ratio_mh_v/math.sqrt(2)):.3f}%)")

    # Electroweak relation: M_H vs M_W, M_Z
    print(f"\n  M_H / M_Z = {M_H/M_Z:.8f}")
    print(f"  M_H / M_W = {M_H/M_W:.8f}")
    print(f"  M_H / (M_W + M_Z) = {M_H/(M_W+M_Z):.8f}")
    print(f"  (M_W + M_Z) / M_H = {(M_W+M_Z)/M_H:.8f}")
    # Is M_H close to (M_W + M_Z)/phi ?
    pred = (M_W + M_Z) / PHI
    print(f"  (M_W + M_Z) / phi = {pred:.4f} GeV  "
          f"(err = {pct_error(pred, M_H):.3f}%)")

    # Is M_H ~ sqrt(M_W * M_Z * phi) ?
    pred = math.sqrt(M_W * M_Z * PHI)
    print(f"  sqrt(M_W * M_Z * phi) = {pred:.4f} GeV  "
          f"(err = {pct_error(pred, M_H):.3f}%)")


# ============================================================
# Section 6: Grand Summary
# ============================================================

def grand_summary(base_candidates, corrected_candidates, lambda_candidates):
    """Merge all results and produce a ranked master list."""
    separator("6. GRAND SUMMARY — TOP 20 ACROSS ALL CATEGORIES")

    all_results = []

    for c in base_candidates[:50]:
        all_results.append({
            "formula": c["formula"],
            "predicted": c["predicted"],
            "ppm": c["ppm"],
            "category": c["category"],
        })

    for c in corrected_candidates[:50]:
        all_results.append({
            "formula": c["formula"],
            "predicted": c["predicted"],
            "ppm": c["ppm"],
            "category": c["category"],
        })

    for c in lambda_candidates[:50]:
        all_results.append({
            "formula": c["formula"],
            "predicted": c["predicted_MH"],
            "ppm": c["MH_ppm"],
            "category": "lambda_decomp",
        })

    # Deduplicate by rounding predicted to 6 decimal places
    seen = set()
    unique = []
    for r in all_results:
        key = round(r["predicted"], 6)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: r["ppm"])

    print(f"\n  {'Rank':<5s} {'ppm':>10s} {'Predicted':>12s} {'Category':<16s} {'Formula'}")
    print("  " + "-" * 100)
    for i, r in enumerate(unique[:20]):
        cat = r["category"][:15]
        print(f"  {i+1:<5d} {r['ppm']:>10.1f} {r['predicted']:>12.6f} {cat:<16s} {r['formula']}")

    # Highlight best
    if unique:
        best = unique[0]
        print(f"\n  BEST CANDIDATE:")
        print(f"    Formula:   {best['formula']}")
        print(f"    Predicted: {best['predicted']:.8f} GeV")
        print(f"    Measured:  {M_H} +/- {M_H_ERR} GeV")
        print(f"    Error:     {best['ppm']:.1f} ppm  ({best['ppm']/10000:.4f}%)")
        within_exp = abs(best["predicted"] - M_H) < M_H_ERR
        print(f"    Within experimental error: {'YES' if within_exp else 'NO'}")

    return unique[:20]


# ============================================================
# Save Results
# ============================================================

def save_results(ratio_data, top_results):
    """Save to JSON in results directory."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(results_dir, f"exp_07_higgs_{timestamp}.json")

    data = {
        "experiment": "exp_07_higgs_mass_derivation",
        "timestamp": datetime.now().isoformat(),
        "measured": {
            "M_H_GeV": M_H,
            "M_H_err_GeV": M_H_ERR,
            "M_Z_GeV": M_Z,
            "M_W_GeV": M_W,
            "v_GeV": VEV,
            "lambda_quartic": LAMBDA_H,
        },
        "constants": {
            "phi": PHI,
            "Xi": XI,
            "gamma_EM": GAMMA_EM,
        },
        "top_20_results": top_results,
        "ratio_analysis": ratio_data,
    }

    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n  Results saved to: {outpath}")
    return outpath


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("  EXP 07: HIGGS BOSON MASS FROM DFT FIRST PRINCIPLES")
    print("  Dawn Field Theory — Milestone 5")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 72)

    print(f"\n  Target: M_H = {M_H} +/- {M_H_ERR} GeV")
    print(f"  Reference: M_Z = {M_Z} GeV, v = {VEV} GeV")
    print(f"  lambda = {LAMBDA_H:.8f}, phi = {PHI:.8f}, Xi = {XI:.8f}")

    # Run all sections
    ratio_data = ratio_analysis()
    base_candidates = fibonacci_search()
    corrected = correction_search(base_candidates)
    lambda_candidates = lambda_analysis()
    physical_checks()
    top_results = grand_summary(base_candidates, corrected, lambda_candidates)

    # Save
    save_results(ratio_data, top_results)

    separator("DONE")
    print(f"\n  Experiment complete. Review top candidates for DFT Higgs mass formula.")
    print(f"  Next steps:")
    print(f"    1. Check if best formula connects to electroweak symmetry breaking")
    print(f"    2. Verify correction term has same structure as alpha, sin^2(theta_W)")
    print(f"    3. Look for deeper connection: is lambda = f(alpha, sin^2(theta_W))?")
    print()


if __name__ == "__main__":
    main()
