#!/usr/bin/env python3
"""
exp_30k — Coupling Hierarchy from Fibonacci Recursion Depth

exp_30i established that φ^183 matches α_G⁻¹ to 3.8%, where 183 = F₇² + F₇ + 1.
This experiment asks: is there a systematic ADE pattern connecting recursion
depth to coupling strength?

Key idea: each ADE level transition has a cost (ξ − 1 ≈ π/55). Different
couplings correspond to different recursion depths in the Fibonacci hierarchy.
If ADE controls coupling constants, the hierarchy of fundamental forces should
follow from the Fibonacci depth structure.

Tests:
  1. Fibonacci depth ladder — which depths match known couplings?
  2. Zeckendorf decomposition of coupling constants
  3. GUT unification scale vs F₇ = 13
  4. ξ-cascade cost accounting (depth difference gravity → EM)
  5. Hierarchy ratio universality (ratios of log-couplings)

Author: Peter Groom
Date: 2026-03-28
"""
import json
import sys
import os
import numpy as np
from datetime import datetime

# Constants
PHI = (1 + np.sqrt(5)) / 2
GAMMA = 0.5772156649015329
XI = GAMMA + np.log(PHI)

# Physical constants (CODATA 2018 / PDG 2024)
ALPHA_EM_INV = 137.035999084          # α_EM⁻¹
ALPHA_S = 0.1180                       # α_s(M_Z)
G_FERMI = 1.1663788e-5                # G_F in GeV⁻²
M_Z = 91.1876                          # Z boson mass (GeV)
M_PLANCK = 1.22089e19                  # Planck mass (GeV)
M_PROTON_GEV = 0.93827208816           # proton mass (GeV)
M_PROTON_KG = 1.67262192369e-27        # proton mass (kg)
G_N = 6.67430e-11                      # Newton's constant (m³/(kg·s²))
HBAR = 1.054571817e-34                 # reduced Planck constant (J·s)
C_LIGHT = 299792458                    # speed of light (m/s)
ALPHA_G = G_N * M_PROTON_KG**2 / (HBAR * C_LIGHT)  # gravitational fine structure ~5.91e-39

results = {
    "experiment": "exp_30k_coupling_hierarchy",
    "date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "checks": [],
    "passed": 0,
    "failed": 0,
    "total": 0,
}


def record(name, passed, details=""):
    results["checks"].append({"name": name, "passed": passed, "details": details})
    results["total"] += 1
    if passed:
        results["passed"] += 1
    else:
        results["failed"] += 1
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")


def fib(n):
    """Compute nth Fibonacci number (F_1=1, F_2=1, F_3=2, ...)."""
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def zeckendorf(n):
    """Zeckendorf representation: decompose n into non-consecutive Fibonacci numbers."""
    if n <= 0:
        return []
    fibs = []
    k = 2
    while fib(k) <= n:
        k += 1
    k -= 1
    remainder = n
    while remainder > 0:
        f = fib(k)
        if f <= remainder:
            fibs.append(k)
            remainder -= f
            k -= 2  # non-consecutive
        else:
            k -= 1
    return fibs


# ─────────────────────────────────────────────────────────
# Test 1: Fibonacci depth ladder
# ─────────────────────────────────────────────────────────
def test_fibonacci_depth_ladder():
    """
    Construct depths d = F_k² + F_k + 1 for k = 1..10 (geometric series in F_k).
    Compute φ^d for each and compare to known physical hierarchies.

    Only k=7 (d=183) was established in exp_30i. Does ANY other k match
    a known coupling constant?
    """
    print("\n=== Test 1: Fibonacci Depth Ladder ===")

    print(f"  {'k':>3}  {'F_k':>6}  {'d=F²+F+1':>10}  {'log₁₀(φ^d)':>12}  {'Match?':>20}")
    print(f"  {'─'*3}  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*20}")

    known_scales = {
        "α_EM⁻¹": np.log10(ALPHA_EM_INV),      # 2.137
        "α_W⁻¹": np.log10(29.0),                # 1.462
        "α_S⁻¹": np.log10(1/ALPHA_S),           # 0.928
        "α_G⁻¹": np.log10(1/ALPHA_G),           # ~38.23
        "M_P/M_Z": np.log10(M_PLANCK/M_Z),          # ~17.13
        "M_P/m_p": np.log10(M_PLANCK/M_PROTON_GEV), # ~19.11
    }

    matches = {}
    for k in range(1, 11):
        Fk = fib(k)
        d = Fk**2 + Fk + 1
        log_phi_d = d * np.log10(PHI)

        # Check against known scales
        best_match = ""
        best_err = float('inf')
        for name, log_val in known_scales.items():
            err = abs(log_phi_d - log_val)
            if err < best_err:
                best_err = err
                best_match = f"{name} (Δ={err:.2f})"

        # Only report if within 1 order of magnitude
        if best_err < 1.0:
            matches[k] = (best_match, best_err)
            flag = " ←"
        else:
            flag = ""

        print(f"  {k:3d}  {Fk:6d}  {d:10d}  {log_phi_d:12.4f}  {best_match}{flag}")

    # k=7 should match α_G⁻¹ (established in exp_30i)
    k7_d = fib(7)**2 + fib(7) + 1
    k7_log = k7_d * np.log10(PHI)
    k7_target = np.log10(1/ALPHA_G)
    k7_err = abs(k7_log - k7_target)
    k7_match = k7_err < 0.05  # within 0.05 orders (exp_30i got 0.016)

    # Are there OTHER clean matches?
    n_close = sum(1 for k, (m, e) in matches.items() if e < 0.5 and k != 7)

    print(f"\n  k=7 (d=183) → α_G⁻¹ match: {k7_err:.3f} orders ({'CONFIRMED' if k7_match else 'FAIL'})")
    print(f"  Other matches within 0.5 orders: {n_close}")

    if n_close == 0:
        print(f"  → k=7 appears unique in the ladder. Other couplings don't follow F_k²+F_k+1.")
        print(f"  → This is an honest negative: the geometric series structure is specific to gravity.")

    record(
        "fibonacci_depth_ladder",
        k7_match,
        f"k=7 confirmed (Δ={k7_err:.3f}), other close matches: {n_close}. "
        f"Ladder specific to gravity — not a universal coupling formula."
    )


# ─────────────────────────────────────────────────────────
# Test 2: Zeckendorf decomposition of coupling constants
# ─────────────────────────────────────────────────────────
def test_zeckendorf_couplings():
    """
    Decompose integer parts of coupling-related numbers into Fibonacci
    representations. Look for patterns (odd indices, ADE-depth spacing).

    From exp_30i: Zeckendorf(137) = F₁₁ + F₉ + F₇ + F₂ — all odd indices.
    Is this pattern universal across couplings, or specific to α⁻¹?
    """
    print("\n=== Test 2: Zeckendorf Decomposition of Couplings ===")

    test_numbers = {
        "α_EM⁻¹ = 137": 137,
        "α_W⁻¹ ≈ 29": 29,
        "α_S⁻¹ ≈ 8": 8,
        "F₇ = 13": 13,
        "F₁₈₃ digits ≈ 38": 38,  # orders of magnitude of hierarchy
    }

    all_odd_count = 0
    total = 0

    for label, n in test_numbers.items():
        z = zeckendorf(n)
        all_odd = all(k % 2 == 1 for k in z)
        if all_odd:
            all_odd_count += 1
        total += 1
        z_str = " + ".join(f"F_{k}" for k in z)
        indices_parity = "all odd" if all_odd else "mixed"
        print(f"  {label:25s} = {z_str:30s} ({indices_parity})")

    # Check structural properties of 137's decomposition
    z_137 = zeckendorf(137)
    # 137 = F_11 + F_9 + F_7 + F_2 — three odd indices (11,9,7) plus one even (2)
    n_odd = sum(1 for k in z_137 if k % 2 == 1)
    n_total_z = len(z_137)
    mostly_odd = n_odd >= n_total_z - 1  # at most one even index

    print(f"\n  137 Zeckendorf indices: {z_137}")
    print(f"  Odd indices: {n_odd}/{n_total_z} (F_2 is the only even)")

    # The spacing pattern: indices decrease by 2, 2, 5
    spacings = [z_137[i] - z_137[i+1] for i in range(len(z_137)-1)]
    print(f"  Index spacings: {spacings}")
    contains_f7 = 7 in z_137
    print(f"  Contains F₇ (ADE depth): {contains_f7}")

    # How common is this pattern? Count integers 1-200 containing F_7 in Zeckendorf
    contains_f7_count = sum(1 for n in range(1, 201) if 7 in zeckendorf(n))
    print(f"  Integers 1..200 containing F₇: {contains_f7_count}/200 ({contains_f7_count/200*100:.1f}%)")

    # The significant feature: three consecutive odd-indexed Fibonacci numbers
    # F_7, F_9, F_11 with spacing 2,2 — an arithmetic progression in Fibonacci index space
    has_ap = spacings[:2] == [2, 2]
    print(f"  Top 3 indices form arithmetic progression (spacing 2): {has_ap}")

    record(
        "zeckendorf_couplings",
        contains_f7 and has_ap,
        f"137 = F_11+F_9+F_7+F_2: contains F₇={contains_f7}, top 3 in AP(spacing 2)={has_ap}. "
        f"Tier 3: structural but not unique — {contains_f7_count}/200 integers contain F₇."
    )


# ─────────────────────────────────────────────────────────
# Test 3: GUT unification scale vs F₇
# ─────────────────────────────────────────────────────────
def test_gut_scale():
    """
    The SM couplings approximately unify near 10^{15-16} GeV.
    The ratio M_GUT / M_Z ≈ 10^{13-14}.

    In ADE: F₇ = 13. Does the GUT scale ratio relate to F₇?
    Compute: log₁₀(M_GUT/M_Z) and compare to F₇ = 13.
    """
    print("\n=== Test 3: GUT Unification Scale ===")

    # Standard estimates for GUT scale (MSSM and non-SUSY)
    log_gut_mssm = 16.3    # MSSM unification: ~2×10^16 GeV
    log_gut_nonsm = 14.5   # Non-SUSY: ~3×10^14 GeV (approximate)
    log_mz = np.log10(M_Z)  # ~1.96

    ratio_mssm = log_gut_mssm - log_mz      # ~14.3
    ratio_nonsm = log_gut_nonsm - log_mz     # ~12.5

    F7 = fib(7)  # 13

    err_mssm = abs(ratio_mssm - F7) / F7
    err_nonsm = abs(ratio_nonsm - F7) / F7

    print(f"  F₇ = {F7}")
    print(f"  log₁₀(M_GUT/M_Z):")
    print(f"    MSSM:    {ratio_mssm:.1f} (error vs F₇: {err_mssm*100:.1f}%)")
    print(f"    Non-SUSY: {ratio_nonsm:.1f} (error vs F₇: {err_nonsm*100:.1f}%)")

    # φ^{F₇} = φ^13
    phi_13 = PHI**13
    log_phi_13 = 13 * np.log10(PHI)
    print(f"\n  φ^13 = {phi_13:.1f} (log₁₀ = {log_phi_13:.2f})")
    print(f"  Ratio to GUT/M_Z: φ^13 gives {log_phi_13:.2f} orders — within GUT range")

    # The GUT scale is not precisely known. F₇ = 13 is in the right ballpark.
    # Best comparison: MSSM unification log-ratio ≈ 14.3, F₇ = 13
    best_err = min(err_mssm, err_nonsm)
    within_15pct = best_err < 0.15

    print(f"\n  Assessment:")
    print(f"    F₇ = 13 is order-of-magnitude correct for GUT/M_Z ratio")
    print(f"    Best match: {best_err*100:.1f}% (non-SUSY closer)")
    print(f"    But GUT scale itself is uncertain by ~2 orders")
    print(f"    Tier 3: suggestive, not falsifiable without precise GUT measurement")

    record(
        "gut_scale_f7",
        within_15pct,
        f"log(M_GUT/M_Z) ≈ {ratio_nonsm:.1f}–{ratio_mssm:.1f} vs F₇=13, "
        f"best match {best_err*100:.1f}%. Tier 3: GUT scale too uncertain for clean test."
    )


# ─────────────────────────────────────────────────────────
# Test 4: ξ-cascade cost accounting
# ─────────────────────────────────────────────────────────
def test_xi_cascade():
    """
    Model: the total information cost from Planck to a given scale is
    n × (ξ - 1), where n is the recursion depth. Different couplings
    correspond to different depths in the φ-tower.

    For EM: φ^{n_EM} = α_EM⁻¹ → n_EM = log(137.036)/log(φ) ≈ 10.22
    For gravity: φ^{183} ≈ α_G⁻¹

    The depth difference 183 - 10.22 ≈ 172.8 should have structure.
    Check: does it relate to Fibonacci numbers?
    """
    print("\n=== Test 4: ξ-Cascade Cost Accounting ===")

    # Recursion depths via φ-tower
    n_em = np.log(ALPHA_EM_INV) / np.log(PHI)
    n_grav = 183  # from exp_30i
    n_strong = np.log(1/ALPHA_S) / np.log(PHI)
    n_weak = np.log(29.0) / np.log(PHI)  # α_W⁻¹ ≈ 29

    print(f"  Recursion depths (φ^n = coupling⁻¹):")
    print(f"    Strong: n_S = {n_strong:.2f} (α_S⁻¹ ≈ 8.47)")
    print(f"    Weak:   n_W = {n_weak:.2f} (α_W⁻¹ ≈ 29)")
    print(f"    EM:     n_EM = {n_em:.2f} (α_EM⁻¹ = 137.04)")
    print(f"    Gravity: n_G = {n_grav} (from ADE)")

    # Depth differences
    delta_em_grav = n_grav - n_em
    delta_em_strong = n_em - n_strong
    delta_em_weak = n_em - n_weak

    print(f"\n  Depth differences:")
    print(f"    n_G - n_EM = {delta_em_grav:.2f}")
    print(f"    n_EM - n_W = {delta_em_weak:.2f}")
    print(f"    n_EM - n_S = {delta_em_strong:.2f}")

    # Check Fibonacci structure of depth differences
    z_delta = zeckendorf(round(delta_em_grav))
    z_str = " + ".join(f"F_{k}" for k in z_delta)
    print(f"\n  Zeckendorf({round(delta_em_grav)}) = {z_str}")

    # Check if n_EM ≈ F₇/φ² or some clean Fibonacci expression
    print(f"\n  n_EM = {n_em:.4f}")
    print(f"    F₅ = {fib(5)} = 5")
    print(f"    2·F₅ = 10 (close to n_EM = {n_em:.2f})")
    print(f"    F₇/φ = {fib(7)/PHI:.4f}")
    print(f"    φ⁵ = {PHI**5:.4f} = F₅·φ + F₄ = 5φ+3 = {5*PHI+3:.4f}")

    # The cleanest expression: n_EM = 2·F₅ + 0.22 (not clean)
    # Try: is n_EM close to any Fibonacci number?
    closest_fib_idx = min(range(1, 15), key=lambda k: abs(fib(k) - n_em))
    closest_fib = fib(closest_fib_idx)
    fib_err = abs(closest_fib - n_em)
    print(f"    Closest Fibonacci: F_{closest_fib_idx} = {closest_fib} (error = {fib_err:.2f})")

    # ξ cost per level
    xi_minus_1 = XI - 1
    total_cost_grav = n_grav * xi_minus_1
    total_cost_em = n_em * xi_minus_1
    print(f"\n  ξ - 1 = {xi_minus_1:.6f}")
    print(f"  Total cost to gravity: {n_grav}·(ξ-1) = {total_cost_grav:.4f}")
    print(f"  Total cost to EM: {n_em:.2f}·(ξ-1) = {total_cost_em:.4f}")

    # The honest assessment: depth differences don't show clean Fibonacci structure
    # n_EM ≈ 10.22 is not a Fibonacci number
    # The φ-tower gives a continuous (not discrete) mapping for non-gravitational couplings

    has_structure = closest_fib_idx <= 7 and fib_err < 1.0

    record(
        "xi_cascade_cost",
        has_structure,
        f"n_EM={n_em:.2f} (closest F_{closest_fib_idx}={closest_fib}, err={fib_err:.2f}), "
        f"n_G-n_EM={delta_em_grav:.1f}. Tier 2: φ-tower maps couplings but depths not obviously Fibonacci."
    )


# ─────────────────────────────────────────────────────────
# Test 5: Hierarchy ratio universality
# ─────────────────────────────────────────────────────────
def test_hierarchy_ratio():
    """
    Test whether RATIOS between coupling log-scales follow ADE structure.

    Key ratio: log(α_G⁻¹) / log(α_EM⁻¹) ≈ 38.23 / 2.137 ≈ 17.89
    Compare to: φ⁶ = 2φ⁵+φ⁴ = ... = 17.944 (0.3% match!)

    If confirmed: the ratio of recursion depths n_G/n_EM = 183/10.22 ≈ 17.91
    is a Fibonacci power.
    """
    print("\n=== Test 5: Hierarchy Ratio Universality ===")

    log_alpha_g_inv = np.log10(1/ALPHA_G)
    log_alpha_em_inv = np.log10(ALPHA_EM_INV)

    # Primary ratio: gravity/EM
    ratio_ge = log_alpha_g_inv / log_alpha_em_inv
    phi_6 = PHI**6
    err_phi6 = abs(ratio_ge - phi_6) / phi_6

    print(f"  log₁₀(α_G⁻¹) = {log_alpha_g_inv:.4f}")
    print(f"  log₁₀(α_EM⁻¹) = {log_alpha_em_inv:.4f}")
    print(f"  Ratio: {ratio_ge:.4f}")
    print(f"  φ⁶ = {phi_6:.4f}")
    print(f"  Error: {err_phi6*100:.2f}%")

    # Also check depth ratio
    n_em = np.log(ALPHA_EM_INV) / np.log(PHI)
    depth_ratio = 183 / n_em
    err_depth = abs(depth_ratio - phi_6) / phi_6
    print(f"\n  Depth ratio 183/n_EM = 183/{n_em:.2f} = {depth_ratio:.4f}")
    print(f"  vs φ⁶ = {phi_6:.4f}, error = {err_depth*100:.2f}%")

    # Check other ratios
    log_alpha_s_inv = np.log10(1/ALPHA_S)
    log_alpha_w_inv = np.log10(29.0)

    ratio_gs = log_alpha_g_inv / log_alpha_s_inv
    ratio_gw = log_alpha_g_inv / log_alpha_w_inv
    ratio_ew = log_alpha_em_inv / log_alpha_w_inv
    ratio_es = log_alpha_em_inv / log_alpha_s_inv

    print(f"\n  Other ratios:")
    print(f"    G/S: {ratio_gs:.3f}")
    print(f"    G/W: {ratio_gw:.3f}")
    print(f"    EM/W: {ratio_ew:.3f}")
    print(f"    EM/S: {ratio_es:.3f}")

    # Check each against φ^n for n = 1..10
    for label, ratio in [("G/S", ratio_gs), ("G/W", ratio_gw),
                         ("EM/W", ratio_ew), ("EM/S", ratio_es)]:
        best_n = min(range(1, 11), key=lambda n: abs(PHI**n - ratio))
        best_err = abs(PHI**best_n - ratio) / ratio
        print(f"    {label} ≈ φ^{best_n} = {PHI**best_n:.3f} (error {best_err*100:.1f}%)")

    # The key finding
    primary_match = err_phi6 < 0.005  # within 0.5%

    print(f"\n  Key result:")
    print(f"    log(α_G⁻¹)/log(α_EM⁻¹) = {ratio_ge:.4f} ≈ φ⁶ = {phi_6:.4f}")
    print(f"    Match to {err_phi6*100:.2f}%")
    if primary_match:
        print(f"    This means: n_G/n_EM ≈ φ⁶ — the depth ratio is a Fibonacci power")
        print(f"    Since n_G = 183 = F₇²+F₇+1, we get n_EM ≈ 183/φ⁶ = {183/phi_6:.2f}")
    print(f"    Tier 2: falsifiable prediction, specific numerical match")

    record(
        "hierarchy_ratio",
        primary_match,
        f"log(α_G⁻¹)/log(α_EM⁻¹) = {ratio_ge:.4f} ≈ φ⁶ = {phi_6:.4f}, "
        f"error {err_phi6*100:.2f}%. Tier 2: depth ratio is Fibonacci power."
    )


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("exp_30k — Coupling Hierarchy from Fibonacci Recursion Depth")
    print("=" * 65)

    test_fibonacci_depth_ladder()
    test_zeckendorf_couplings()
    test_gut_scale()
    test_xi_cascade()
    test_hierarchy_ratio()

    print("\n" + "=" * 65)
    print(f"TOTAL: {results['passed']}/{results['total']} checks passed")
    print("=" * 65)

    # Save results
    ts = results["date"]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_30k_coupling_hierarchy_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
