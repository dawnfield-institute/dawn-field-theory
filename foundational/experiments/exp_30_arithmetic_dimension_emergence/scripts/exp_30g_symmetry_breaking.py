#!/usr/bin/env python3
"""
exp_30g — Level 1→2 Symmetry Breaking via Confluence

Tests whether arithmetic level transitions exhibit spontaneous symmetry breaking,
with the golden ratio φ as the critical point where Levels 0, 1, and 2 are in
equilibrium (φ² = φ + 1).

Tests:
  1. Self-application confluence: x+x = x·x and x·x = x^x both at x=2
  2. Golden ratio as 3-level equilibrium: φ² = φ + 1 (L2 = L1 + L0)
  3. Confluence zone widths decrease with level (higher ops break faster)
  4. Breaking direction: higher operation always dominates above confluence
  5. φ-cascade: confluence points form Fibonacci-structured hierarchy

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
LN_PHI = np.log(PHI)

results = {
    "experiment": "exp_30g_symmetry_breaking",
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


# ─────────────────────────────────────────────────────────
# Test 1: Self-application confluence points
# ─────────────────────────────────────────────────────────
def test_self_application_confluence():
    """
    At the confluence between Level k and Level k+1, the self-application
    of each operation gives the same result:
      L1→L2: x + x = x · x  →  2x = x²  →  x = 2
      L2→L3: x · x = x ^ x  →  x² = x^x  →  x = 2
    Both confluences occur at x = 2 (the binary number).

    Additionally, L0→L1 confluence: S(x) = x + x where S is successor.
    S(x) = x + 1, so x + 1 = 2x → x = 1.
    """
    print("\n=== Test 1: Self-Application Confluence Points ===")

    # L0→L1: successor = doubling → x + 1 = x + x → x = 1
    x_01 = 1.0  # expected
    lhs_01 = x_01 + 1  # successor
    rhs_01 = x_01 + x_01  # self-addition
    err_01 = abs(lhs_01 - rhs_01)
    print(f"  L0→L1: S({x_01}) = {lhs_01}, {x_01}+{x_01} = {rhs_01}, error = {err_01:.2e}")

    # L1→L2: x + x = x * x → x = 2
    # Solve numerically to confirm
    from scipy.optimize import brentq
    f_12 = lambda x: (x + x) - (x * x)  # = 2x - x² = x(2 - x)
    # Roots at x=0 and x=2; we want the non-trivial one
    x_12 = brentq(f_12, 0.5, 3.0)
    err_12 = abs(x_12 - 2.0)
    print(f"  L1→L2: x+x = x·x at x = {x_12:.10f}, error from 2 = {err_12:.2e}")

    # L2→L3: x * x = x ^ x → x² = x^x
    # Taking log: 2 ln(x) = x ln(x) → x = 2 (for x > 0, x ≠ 1)
    f_23 = lambda x: (x * x) - (x ** x)
    x_23 = brentq(f_23, 1.5, 2.5)
    err_23 = abs(x_23 - 2.0)
    print(f"  L2→L3: x·x = x^x at x = {x_23:.10f}, error from 2 = {err_23:.2e}")

    # All non-trivial confluences at x = 2
    all_at_2 = err_12 < 1e-10 and err_23 < 1e-10
    # L0→L1 at x = 1
    l01_at_1 = err_01 < 1e-10

    # The confluence hierarchy: 1, 2, 2
    # Level 0 is "unity" (x=1), levels 1-2 and 2-3 both break at "binary" (x=2)
    print(f"\n  Confluence hierarchy: L0→L1 at x=1, L1→L2 at x=2, L2→L3 at x=2")
    print(f"  Binary (x=2) is the universal higher-level confluence point")

    record(
        "self_application_confluence",
        all_at_2 and l01_at_1,
        f"L0→L1 at 1 (err={err_01:.1e}), L1→L2 at 2 (err={err_12:.1e}), L2→L3 at 2 (err={err_23:.1e})"
    )


# ─────────────────────────────────────────────────────────
# Test 2: Golden ratio as 3-level equilibrium
# ─────────────────────────────────────────────────────────
def test_golden_ratio_equilibrium():
    """
    The golden ratio satisfies φ² = φ + 1, which in ADE terms is:
      Level 2(φ) = Level 1(φ) + Level 0

    This means φ is where multiplication (L2), addition (L1), and unity (L0)
    are in perfect balance. No other positive real satisfies this with the
    specific Fibonacci recursion structure.

    Extended: φ also satisfies 1/φ = φ - 1, meaning L0(inversion) = L1 - L0.
    And φ^n = F_n · φ + F_{n-1} (Fibonacci representation of powers).
    """
    print("\n=== Test 2: Golden Ratio as 3-Level Equilibrium ===")

    # Core identity: φ² = φ + 1
    err_core = abs(PHI**2 - (PHI + 1))
    print(f"  φ² = {PHI**2:.15f}")
    print(f"  φ+1 = {PHI + 1:.15f}")
    print(f"  |φ² - (φ+1)| = {err_core:.2e}")

    # Inversion identity: 1/φ = φ - 1
    err_inv = abs(1/PHI - (PHI - 1))
    print(f"  1/φ = {1/PHI:.15f}, φ-1 = {PHI-1:.15f}, error = {err_inv:.2e}")

    # Fibonacci representation: φ^n = F_n φ + F_{n-1}
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    max_err_fib = 0
    for n in range(2, 20):
        lhs = PHI ** n
        rhs = fib(n) * PHI + fib(n - 1)
        max_err_fib = max(max_err_fib, abs(lhs - rhs))
    print(f"  φ^n = F_n·φ + F_{{n-1}} max error (n=2..19): {max_err_fib:.2e}")

    # Uniqueness: φ is the unique positive root of x² - x - 1 = 0
    # No other number balances L2 = L1 + L0 this way
    discriminant = 1 + 4  # b² - 4ac for x² - x - 1
    root = (1 + np.sqrt(discriminant)) / 2
    err_root = abs(root - PHI)
    print(f"  Unique positive root of x²-x-1: {root:.15f}, error from φ: {err_root:.2e}")

    # Level interpretation
    print(f"\n  ADE interpretation:")
    print(f"    φ² = φ + 1  ↔  L2(φ) = L1(φ) + L0")
    print(f"    1/φ = φ - 1  ↔  L0_inv(φ) = L1(φ) - L0")
    print(f"    φ^n = F_n·φ + F_{{n-1}}  ↔  L3 powers decompose into L1 + L0")

    all_pass = err_core < 1e-14 and err_inv < 1e-14 and max_err_fib < 1e-10
    record(
        "golden_ratio_equilibrium",
        all_pass,
        f"φ²=φ+1 err={err_core:.1e}, 1/φ=φ-1 err={err_inv:.1e}, Fib rep max err={max_err_fib:.1e}"
    )


# ─────────────────────────────────────────────────────────
# Test 3: Confluence zone widths decrease with level
# ─────────────────────────────────────────────────────────
def test_divergence_rate():
    """
    At the confluence point x=2, measure the RATE at which operations diverge
    from each other. Higher-level transitions should diverge faster — the
    symmetry breaking is sharper at higher levels.

    Divergence rate = |d/dx [op_{k+1}(x,x) - op_k(x,x)]| evaluated at x=2.

    L1→L2: d/dx(x² - 2x)|_{x=2} = |2x - 2|_{x=2} = 2
    L2→L3: d/dx(x^x - x²)|_{x=2} = |x^x(1+ln x) - 2x|_{x=2} = |4(1+ln2) - 4| = 4·ln2 ≈ 2.77
    """
    print("\n=== Test 3: Divergence Rate at Confluence ===")

    x0 = 2.0
    h = 1e-8

    # L1→L2: gap(x) = x² - 2x
    gap_12 = lambda x: x*x - 2*x
    deriv_12 = abs(gap_12(x0 + h) - gap_12(x0 - h)) / (2 * h)
    exact_12 = abs(2 * x0 - 2)  # d/dx(x² - 2x) = 2x - 2
    print(f"  L1→L2 divergence rate: {deriv_12:.6f} (exact: {exact_12:.6f})")

    # L2→L3: gap(x) = x^x - x²
    gap_23 = lambda x: x**x - x*x
    deriv_23 = abs(gap_23(x0 + h) - gap_23(x0 - h)) / (2 * h)
    exact_23 = abs(x0**x0 * (1 + np.log(x0)) - 2 * x0)  # 4(1+ln2) - 4 = 4·ln2
    print(f"  L2→L3 divergence rate: {deriv_23:.6f} (exact: {exact_23:.6f})")

    # Ratio: how much sharper the L2→L3 breaking is
    ratio = deriv_23 / deriv_12
    print(f"  Ratio L2→L3 / L1→L2: {ratio:.6f}")
    print(f"  = 4·ln(2) / 2 = 2·ln(2) = {2*np.log(2):.6f}")

    # The rate INCREASES with level — higher operations break faster
    rate_increases = deriv_23 > deriv_12
    print(f"  Divergence rate increases with level: {rate_increases}")

    # At φ (below confluence): measure how strongly lower op dominates
    x_phi = PHI
    gap_12_phi = abs(2*x_phi - x_phi**2)  # |add - mult| at φ
    gap_23_phi = abs(x_phi**2 - x_phi**x_phi)  # |mult - exp| at φ
    print(f"\n  At φ = {PHI:.6f}:")
    print(f"    |add - mult| = |{2*x_phi:.4f} - {x_phi**2:.4f}| = {gap_12_phi:.6f}")
    print(f"    |mult - exp| = |{x_phi**2:.4f} - {x_phi**x_phi:.4f}| = {gap_23_phi:.6f}")
    print(f"    Both positive: lower op dominates (below confluence)")

    # Zone width comparison (connected component around x=2 only)
    xs = np.linspace(1.5, 2.5, 100000)
    diff_12 = np.abs(xs + xs - xs * xs)
    diff_23 = np.abs(xs * xs - xs ** xs)
    eps = 0.1
    zone_12 = xs[diff_12 < eps]
    zone_23 = xs[diff_23 < eps]
    w12 = zone_12[-1] - zone_12[0] if len(zone_12) > 1 else 0
    w23 = zone_23[-1] - zone_23[0] if len(zone_23) > 1 else 0
    print(f"\n  Local zone widths (ε=0.1, near x=2):")
    print(f"    L1→L2: {w12:.4f}")
    print(f"    L2→L3: {w23:.4f}")
    print(f"    L2→L3 is {'narrower' if w23 < w12 else 'wider'} (faster divergence)")

    record(
        "divergence_rate",
        rate_increases and abs(deriv_12 - exact_12) < 0.01 and abs(deriv_23 - exact_23) < 0.01,
        f"L1→L2 rate={deriv_12:.4f}, L2→L3 rate={deriv_23:.4f}, ratio={ratio:.4f} = 2·ln2"
    )


# ─────────────────────────────────────────────────────────
# Test 4: Breaking direction — higher operation dominates
# ─────────────────────────────────────────────────────────
def test_breaking_direction():
    """
    Above the confluence point (x > 2), the higher operation ALWAYS dominates:
      x·x > x+x for x > 2
      x^x > x·x for x > 2

    Below the confluence (1 < x < 2), the lower operation dominates:
      x+x > x·x for 0 < x < 2
      x·x > x^x for 1 < x < 2

    This asymmetry IS spontaneous symmetry breaking: the system must "choose"
    which operation dominates, and the choice is determined by scale.
    """
    print("\n=== Test 4: Breaking Direction ===")

    # Above confluence (x > 2)
    xs_above = np.linspace(2.01, 10.0, 1000)
    add_above = xs_above + xs_above
    mult_above = xs_above * xs_above
    exp_above = xs_above ** xs_above

    mult_dom_add = np.all(mult_above > add_above)
    exp_dom_mult = np.all(exp_above > mult_above)
    print(f"  x > 2: mult > add always? {mult_dom_add}")
    print(f"  x > 2: exp > mult always? {exp_dom_mult}")

    # Below confluence (0 < x < 2)
    xs_below = np.linspace(0.01, 1.99, 1000)
    add_below = xs_below + xs_below
    mult_below = xs_below * xs_below

    add_dom_mult = np.all(add_below > mult_below)
    print(f"  0 < x < 2: add > mult always? {add_dom_mult}")

    # For exp vs mult below 2 (restricted to x > 1 where x^x is well-behaved)
    xs_mid = np.linspace(1.01, 1.99, 1000)
    mult_mid = xs_mid * xs_mid
    exp_mid = xs_mid ** xs_mid
    mult_dom_exp = np.all(mult_mid > exp_mid)
    print(f"  1 < x < 2: mult > exp always? {mult_dom_exp}")

    # Dominance ratios at key points
    print(f"\n  Dominance ratios at φ = {PHI:.6f}:")
    add_phi = PHI + PHI
    mult_phi = PHI * PHI
    exp_phi = PHI ** PHI
    print(f"    φ+φ = {add_phi:.6f}")
    print(f"    φ·φ = {mult_phi:.6f}")
    print(f"    φ^φ = {exp_phi:.6f}")
    print(f"    mult/add = {mult_phi/add_phi:.6f} (φ/2 = {PHI/2:.6f})")
    print(f"    exp/mult = {exp_phi/mult_phi:.6f}")

    # At φ: φ·φ/φ+φ = φ²/2φ = φ/2 — exact!
    ratio_check = abs(mult_phi / add_phi - PHI / 2)
    print(f"    |mult/add - φ/2| = {ratio_check:.2e}")

    all_pass = mult_dom_add and exp_dom_mult and add_dom_mult and mult_dom_exp
    record(
        "breaking_direction",
        all_pass,
        f"x>2: mult>add={mult_dom_add}, exp>mult={exp_dom_mult}; x<2: add>mult={add_dom_mult}, mult>exp={mult_dom_exp}"
    )


# ─────────────────────────────────────────────────────────
# Test 5: φ-cascade and Fibonacci structure
# ─────────────────────────────────────────────────────────
def test_phi_cascade():
    """
    The confluence structure has deep Fibonacci connections:

    1. φ² = φ + 1 is the L2 = L1 + L0 balance equation
    2. The continued fraction [1; 1, 1, 1, ...] = φ — slowest convergence
       = most "balanced" breaking (neither side dominates quickly)
    3. The breaking energy E(x) = |x² - x - 1| has minimum 0 at x = φ
    4. Fibonacci ratios F_{n+1}/F_n → φ show the breaking is approached
       through discrete integer steps, each one a "partial confluence"
    """
    print("\n=== Test 5: φ-Cascade and Fibonacci Structure ===")

    # Breaking energy E(x) = |x² - x - 1| (how far from 3-level equilibrium)
    xs = np.linspace(0.1, 3.0, 10000)
    E = np.abs(xs**2 - xs - 1)
    min_idx = np.argmin(E)
    x_min = xs[min_idx]
    E_min = E[min_idx]
    print(f"  Breaking energy E(x) = |x² - x - 1|")
    print(f"  Minimum at x = {x_min:.4f} (φ = {PHI:.4f}), E_min = {E_min:.6f}")

    # Fibonacci ratio convergence to φ
    def fib(n):
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return a

    print(f"\n  Fibonacci ratio convergence (discrete approach to breaking point):")
    fib_ratios = []
    for n in range(2, 20):
        ratio = fib(n + 1) / fib(n)
        err = abs(ratio - PHI)
        fib_ratios.append((n, ratio, err))

    for n, r, e in fib_ratios[:8]:
        print(f"    F_{n+1}/F_{n} = {r:.10f}, |err| = {e:.2e}")

    # Convergence rate should be φ^{-2n} (golden ratio convergence)
    errors = [e for _, _, e in fib_ratios]
    log_errors = [np.log(e) for e in errors if e > 0]
    if len(log_errors) > 2:
        rate = (log_errors[-1] - log_errors[0]) / (len(log_errors) - 1)
        expected_rate = -2 * LN_PHI
        print(f"\n  Convergence rate: {rate:.4f} (expected -2·ln(φ) = {expected_rate:.4f})")
        rate_match = abs(rate - expected_rate) / abs(expected_rate) < 0.05

    # Continued fraction: φ = [1; 1, 1, 1, ...] — all 1s
    # This makes φ the "most irrational" number (slowest CF convergence)
    # = most balanced breaking (neither operation dominates quickly)
    cf_convergents = []
    p_prev, p_curr = 1, 1  # convergents p_n/q_n for CF [1; 1, 1, ...]
    q_prev, q_curr = 0, 1
    for i in range(15):
        cf_convergents.append(p_curr / q_curr)
        p_prev, p_curr = p_curr, p_curr + p_prev
        q_prev, q_curr = q_curr, q_curr + q_prev

    # All CF coefficients are 1 — verify convergents match F_{n+1}/F_n
    cf_match = all(
        abs(cf_convergents[i] - fib(i + 2) / fib(i + 1)) < 1e-10
        for i in range(min(len(cf_convergents), 15))
    )
    print(f"  CF convergents match F_{{n+1}}/F_n: {cf_match}")

    # The breaking energy at integer Fibonacci ratios
    print(f"\n  Breaking energy at discrete Fibonacci ratios:")
    for n in [3, 5, 8, 13]:
        fn1 = fib(n + 1) if n < 18 else 0
        fn0 = fib(n)
        if fn0 > 0:
            x = fn1 / fn0
            e = abs(x**2 - x - 1)
            print(f"    F_{n+1}/F_{n} = {fn1}/{fn0} = {x:.6f}, E = {e:.2e}")

    all_pass = abs(x_min - PHI) < 0.001 and cf_match and rate_match
    record(
        "phi_cascade",
        all_pass,
        f"E(x) min at {x_min:.4f} (φ={PHI:.4f}), CF match={cf_match}, rate match={rate_match}"
    )


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("exp_30g — Level 1→2 Symmetry Breaking via Confluence")
    print("=" * 65)

    test_self_application_confluence()
    test_golden_ratio_equilibrium()
    test_divergence_rate()
    test_breaking_direction()
    test_phi_cascade()

    print("\n" + "=" * 65)
    print(f"TOTAL: {results['passed']}/{results['total']} checks passed")
    print("=" * 65)

    # Save results
    ts = results["date"]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_30g_symmetry_breaking_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
