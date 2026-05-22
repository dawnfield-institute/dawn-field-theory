#!/usr/bin/env python3
"""
exp_03_mobius_fibonacci_mechanism.py
=====================================

THREE-PART INVESTIGATION: Why does F_10 = 55 appear in the Feigenbaum formula?

PART A: Mobius Perturbation Series Extension
    Push the correction series C = 4 - 4/F^2 + c_3/F^4 + ... to Level 6+.
    Test if coefficient ratios follow Fibonacci structure.
    If yes: the formula is a convergent Fibonacci-geometric series, not an empirical fit.

PART B: Orbit Modular Arithmetic
    Systematic analysis of 2^n mod F_k.
    Key observation: 2^10 mod 55 = 34 = F_9.
    Is this unique? When does 2^n mod F_n = F_{n-1}?

PART C: Self-Closing Formula Dissection
    Anatomy of delta = phi^{20/N}: what do 39, 160, 1371 encode?
    Can we derive them from Mobius geometry?
    Higher-order self-closing: can we beat 13 digits?
"""

import json
import time
from datetime import datetime
from pathlib import Path
from mpmath import mp, mpf, sqrt, pi, log, log10, fabs, phi as mpphi, power, nstr

# ============================================================
# CONSTANTS (200 dps)
# ============================================================
mp.dps = 200

DELTA_KNOWN = mpf(
    '4.66920160910299067185320382046620161725818557747576863274565134300'
    '41343302113147371386897440239480138173006257387285600977533512531'
    '02447093890875406413481915241755948313568379789691270958234299516'
)

R_INF_KNOWN = mpf(
    '3.56994567187094490184200515138649893676383691151483237813880114180'
    '76359246521972857194523735381046823974126482698024094429191909780'
    '31586727916449255185049578223115328302860028469636142720826649911'
)

ALPHA_KNOWN = mpf(
    '2.50290787509589282228390287321821578638127137672714997733619205677'
    '92354196397679065211552846227722096325396934454632514265681655994'
    '80509672680067318574011679217988247808050316114814100561203043728'
)

PHI = (1 + sqrt(5)) / 2

# Fibonacci cache
def fib(n, _cache={0: 0, 1: 1}):
    if n not in _cache:
        _cache[n] = fib(n-1) + fib(n-2)
    return _cache[n]

# Lucas cache
def luc(n, _cache={0: 2, 1: 1}):
    if n not in _cache:
        _cache[n] = luc(n-1) + luc(n-2)
    return _cache[n]


# ============================================================
# PART A: MOBIUS PERTURBATION SERIES EXTENSION
# ============================================================

def part_a_mobius_series():
    """
    The paper's Mobius perturbation series (S6):

        r_inf = pi * M_10(-1/phi + Delta_z)
        1/Delta_z = 1857 + C * (delta-4)/pi

    where C has a series in 1/F^2:
        C = c_0 + c_1/F^2 + c_2/F^4 + ...

    Paper found: c_0 = 4, c_1 = -4 (Level 2: C = 4 - 4/F^2 gives 9 digits).

    APPROACH:
    1. Extract C_exact from known r_inf and delta
    2. Expand C_exact - 4 + 4/F^2 to find higher-order terms
    3. Separately: extract r_inf correction series A_k from the formula residual
    """
    print("=" * 72)
    print("  PART A: MOBIUS PERTURBATION SERIES EXTENSION")
    print("=" * 72)

    F = mpf(55)  # F_10
    F9, F10, F11 = mpf(34), mpf(55), mpf(89)

    # === Step 1: Extract exact Delta_z and C_exact ===
    target = R_INF_KNOWN / pi
    z_exact = (F10 - target * F9) / (target * F10 - F11)
    Delta_z = z_exact - (-1/PHI)
    inv_Dz = 1 / Delta_z

    d4 = DELTA_KNOWN - 4
    d4_over_pi = d4 / pi

    # C_exact = (1/Delta_z - 1857) / ((delta-4)/pi)
    C_exact = (inv_Dz - 1857) / d4_over_pi

    print(f"\n  Exact Mobius seed:")
    print(f"    Delta_z     = {nstr(Delta_z, 30)}")
    print(f"    1/Delta_z   = {nstr(inv_Dz, 30)}")
    print(f"    C_exact     = {nstr(C_exact, 30)}")

    # === Step 2: Expand C in powers of 1/F^2 ===
    print(f"\n  C expansion in 1/F^2:")
    print(f"    C_exact = {nstr(C_exact, 25)}")

    # Level 0: c_0
    c0 = mpf(4)
    C_residual = C_exact - c0
    print(f"\n    c_0 = 4")
    print(f"    C - 4 = {nstr(C_residual, 25)}")

    # Level 1: c_1 / F^2
    c1 = C_residual * F**2
    C_residual1 = C_exact - c0 - c1/F**2
    print(f"\n    c_1 = (C-4) * F^2 = {nstr(c1, 25)}")
    print(f"    C - 4 - c_1/F^2 = {nstr(C_residual1, 25)}")

    # Level 2: c_2 / F^4
    c2 = C_residual1 * F**4
    C_residual2 = C_exact - c0 - c1/F**2 - c2/F**4
    print(f"\n    c_2 = residual * F^4 = {nstr(c2, 25)}")
    print(f"    residual = {nstr(C_residual2, 25)}")

    # Continue
    c_list = [c0, c1, c2]
    C_res = C_residual2
    for level in range(3, 12):
        c_k = C_res * F**(2*level)
        C_res = C_res - c_k / F**(2*level)
        c_list.append(c_k)

    # Test: what digits of r_inf does each level give?
    print(f"\n  Precision hierarchy:")
    print(f"  {'Level':>6}  {'C_approx':>25}  {'r_inf digits':>14}")
    print(f"  {'-'*6}  {'-'*25}  {'-'*14}")

    for level in range(len(c_list)):
        C_approx = sum(c_list[k] / F**(2*k) for k in range(level + 1))
        inv_Dz_approx = 1857 + C_approx * d4_over_pi
        Dz_approx = 1 / inv_Dz_approx
        z_approx = -1/PHI + Dz_approx
        r_approx = pi * (F11 * z_approx + F10) / (F10 * z_approx + F9)
        err = fabs(r_approx - R_INF_KNOWN) / R_INF_KNOWN
        digits = float(-log10(err)) if err > 0 else 200
        print(f"  {level:6d}  {nstr(C_approx, 20):>25}  {digits:14.1f}")

    # === Step 3: Analyze c_k coefficients ===
    print(f"\n  C-series coefficients:")
    print(f"  {'k':>3}  {'c_k':>28}  {'c_k/c_{k-1}':>20}  {'c_k as fraction?':>25}")
    print(f"  {'-'*3}  {'-'*28}  {'-'*20}  {'-'*25}")

    for k in range(len(c_list)):
        c = c_list[k]
        ratio = c_list[k] / c_list[k-1] if k > 0 and fabs(c_list[k-1]) > 0 else mpf(0)

        # Check if c_k is close to a simple integer or fraction
        near_int = ""
        rounded = int(float(c) + (0.5 if float(c) > 0 else -0.5))
        if fabs(c - rounded) < mpf('0.01'):
            near_int = f"~{rounded}"
        elif fabs(c) > 0:
            for denom in [1, 2, 3, 4, 5, 6, 7, 8, 13, 34, 55]:
                val = c * denom
                rval = int(float(val) + (0.5 if float(val) > 0 else -0.5))
                if fabs(val - rval) < mpf('0.01'):
                    near_int = f"~{rval}/{denom}"
                    break

        print(f"  {k:3d}  {nstr(c, 22):>28}  {nstr(ratio, 15):>20}  {near_int:>25}")

    # === Step 4: Check Fibonacci identities for coefficients ===
    print(f"\n  Fibonacci pattern checks:")
    print(f"    c_0 = {nstr(c_list[0], 15)} (= 4 = F_3 + 1?  or 2^2)")
    print(f"    c_1 = {nstr(c_list[1], 15)} (paper: -4)")
    print(f"    c_2 = {nstr(c_list[2], 15)}")
    if len(c_list) > 3:
        print(f"    c_3 = {nstr(c_list[3], 15)}")
    if len(c_list) > 4:
        print(f"    c_4 = {nstr(c_list[4], 15)}")
    if len(c_list) > 5:
        print(f"    c_5 = {nstr(c_list[5], 15)}")

    # Check ratios
    print(f"\n    Successive ratios c_k/c_{{k-1}}:")
    for k in range(1, min(8, len(c_list))):
        if fabs(c_list[k-1]) > mpf(10)**(-50):
            r = c_list[k] / c_list[k-1]
            print(f"      c_{k}/c_{k-1} = {nstr(r, 20)}")
            # Check against Fibonacci-related values
            for name, val in [("1", mpf(1)), ("-1", mpf(-1)), ("phi", PHI), ("-phi", -PHI),
                              ("1/phi", 1/PHI), ("-1/phi", -1/PHI), ("2", mpf(2)), ("-2", mpf(-2)),
                              ("phi^2", PHI**2), ("-phi^2", -PHI**2)]:
                if fabs(r - val) < mpf('0.1'):
                    print(f"        CLOSE TO {name} = {nstr(val, 15)}, diff = {nstr(r - val, 10)}")

    # === Step 5: r_inf CORRECTION SERIES from formula residual ===
    print("\n" + "-" * 72)
    print("  r_inf CORRECTION SERIES (beyond formula's 13 digits)")
    print("-" * 72)
    print("  r_inf = formula_value + A_2/F^8 - A_3/F^10 + A_4/F^12 - ...")

    # The formula from the paper
    d = sqrt(52 + 2*pi/F)
    inner = 17 - pi/(F*d)
    xi_m1 = pi / F
    k_corr = sqrt(mpf(3)/5 - xi_m1**2 / 7)
    base = pi * (F + sqrt(inner)) * (F + pi) / F**2
    correction = k_corr * pi**4 / F**6
    r_formula = base - correction

    print(f"\n  base    = {nstr(base, 30)}")
    print(f"  formula = {nstr(r_formula, 30)}")
    print(f"  r_inf   = {nstr(R_INF_KNOWN, 30)}")

    formula_error = r_formula - R_INF_KNOWN
    print(f"\n  formula - r_inf = {nstr(formula_error, 25)}")
    print(f"  |error| = {nstr(fabs(formula_error), 8)}")

    # Extract A_k: corrections of the form (-1)^k * A_k / F^(4+2k)
    # The formula already has A_1 = k_corr * pi^4 built in.
    # The residual starts at ~F^(-8) scale, so A_2 is first unknown.
    A_list = [None, k_corr * pi**4]  # A_1 is known from the formula

    r_current = r_formula
    sign = mpf(1)
    for k in range(2, 16):
        err = r_current - R_INF_KNOWN
        exponent = 4 + 2*k  # F^8, F^10, F^12, ...
        A_k = sign * err * F**exponent
        r_current = r_current - sign * A_k / F**exponent
        A_list.append(A_k)
        sign = -sign

        digits_now = float(-log10(fabs(r_current - R_INF_KNOWN) / R_INF_KNOWN)) if fabs(r_current - R_INF_KNOWN) > 0 else 200
        if k <= 8 or k % 3 == 0:
            print(f"    A_{k:2d} = {nstr(A_k, 22):>28}  (r_inf to {digits_now:.1f} digits)")

    # Ratio analysis - the key test
    print(f"\n  RATIO ANALYSIS (the paper claims A_3/A_2 = 6050 = 2*F_10^2):")
    print(f"  {'k':>3}  {'A_k/A_{k-1}':>25}  {'ratio':>14}  {'Fibonacci?':>30}")
    print(f"  {'-'*3}  {'-'*25}  {'-'*14}  {'-'*30}")

    for k in range(2, len(A_list)):
        if A_list[k] is not None and A_list[k-1] is not None and fabs(A_list[k-1]) > 0:
            ratio = A_list[k] / A_list[k-1]
            ratio_abs = fabs(ratio)

            fib_match = ""
            # Check against Fibonacci-related values
            for name, val in [
                ("2*F_10^2 = 6050", mpf(6050)),
                ("-2*F_10^2 = -6050", mpf(-6050)),
                ("F_10^2 = 3025", mpf(3025)),
                ("-F_10^2 = -3025", mpf(-3025)),
                ("F_10*F_9 = 1870", mpf(1870)),
                ("-F_10*F_9 = -1870", mpf(-1870)),
                ("2*F_10*F_9 = 3740", mpf(3740)),
                ("-2*F_10*F_9", mpf(-3740)),
            ]:
                if fabs(ratio - val) < fabs(val) * mpf('0.01'):
                    fib_match = f"NEAR {name} (d={nstr(ratio-val, 8)})"
                    break

            if not fib_match:
                # Check if ratio/F_10^2 is near a small integer
                r_over_f2 = ratio / F**2
                rounded = int(float(r_over_f2) + (0.5 if float(r_over_f2) > 0 else -0.5))
                if fabs(r_over_f2 - rounded) < mpf('0.1') and abs(rounded) < 100:
                    fib_match = f"~ {rounded} * F_10^2 = {rounded * 3025}"

            print(f"  {k:3d}  {nstr(ratio, 20):>25}  {float(ratio):>14.4f}  {fib_match:>30}")

    # === Step 6: Deep analysis of C_exact ===
    print("\n" + "-" * 72)
    print("  WHAT IS C_exact?")
    print("-" * 72)

    C = C_exact
    print(f"\n  C_exact = {nstr(C, 40)}")
    print(f"\n  Close to 4: C - 4 = {nstr(C - 4, 30)}")
    print(f"  -4/F^2 = {nstr(-4/F**2, 30)}")
    print(f"  C - (4 - 4/F^2) = {nstr(C - (4 - 4/F**2), 20)}")

    # What is the correction beyond 4 - 4/F^2?
    correction = C - (4 - 4/F**2)
    print(f"\n  Correction beyond paper's Level 2:")
    print(f"    correction = {nstr(correction, 25)}")
    print(f"    correction * F^2 = {nstr(correction * F**2, 20)}")
    print(f"    correction * F^4 = {nstr(correction * F**4, 20)}")
    print(f"    correction / pi = {nstr(correction / pi, 20)}")
    print(f"    correction * F^2 / pi = {nstr(correction * F**2 / pi, 20)}")

    # c_1 exact analysis
    c1_exact = c_list[1]
    print(f"\n  c_1 = {nstr(c1_exact, 30)}")
    print(f"  c_1 + 4 = {nstr(c1_exact + 4, 20)} (deviation from paper's -4)")
    print(f"  (c_1 + 4) / pi = {nstr((c1_exact + 4) / pi, 20)}")
    print(f"  (c_1 + 4) * F = {nstr((c1_exact + 4) * F, 20)}")
    print(f"  (c_1 + 4) * F^2 = {nstr((c1_exact + 4) * F**2, 20)}")
    print(f"  (c_1 + 4) * F / pi = {nstr((c1_exact + 4) * F / pi, 20)}")

    # Check: is C related to delta or alpha?
    print(f"\n  C vs universal constants:")
    print(f"    C * delta = {nstr(C * DELTA_KNOWN, 20)}")
    print(f"    C * alpha = {nstr(C * ALPHA_KNOWN, 20)}")
    print(f"    C / phi = {nstr(C / PHI, 20)}")
    print(f"    C * phi = {nstr(C * PHI, 20)}")
    print(f"    4 - C = {nstr(4 - C, 20)}")
    print(f"    (4 - C) * F^2 = {nstr((4 - C) * F**2, 20)}")
    print(f"    (4 - C) * F^2 / pi = {nstr((4 - C) * F**2 / pi, 20)}")

    # Check: is (4-C)*F^2 related to (delta-4)?
    print(f"\n  Key test: does (4-C)*F^2 relate to (delta-4)?")
    d4 = DELTA_KNOWN - 4
    ratio_cd = (4 - C) * F**2 / d4
    print(f"    (4-C)*F^2 / (delta-4) = {nstr(ratio_cd, 20)}")
    print(f"    ratio / pi = {nstr(ratio_cd / pi, 20)}")
    print(f"    ratio * pi = {nstr(ratio_cd * pi, 20)}")

    return c_list, A_list


# ============================================================
# PART B: ORBIT MODULAR ARITHMETIC
# ============================================================

def part_b_orbit_modular():
    """
    Key observation: 2^10 mod 55 = 34 = F_9.
    Question: Is 2^n mod F_n = F_{n-1} special?
    """
    print("\n\n" + "=" * 72)
    print("  PART B: ORBIT MODULAR ARITHMETIC")
    print("=" * 72)

    # === Test 1: 2^n mod F_n for various n ===
    print("\n  Test 1: 2^n mod F_n")
    print(f"  {'n':>3}  {'F_n':>8}  {'2^n mod F_n':>12}  {'= F_{n-1}?':>12}  {'F_{n-1}':>8}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")

    hits = []
    for n in range(3, 25):
        fn = fib(n)
        fn_1 = fib(n-1)
        mod_val = pow(2, n, fn) if fn > 0 else 0
        is_prev = (mod_val == fn_1)
        marker = "  YES <<<" if is_prev else ""
        print(f"  {n:3d}  {fn:8d}  {mod_val:12d}  {str(is_prev):>12s}  {fn_1:8d}{marker}")
        if is_prev:
            hits.append(n)

    print(f"\n  Hits (2^n mod F_n = F_{{n-1}}): n = {hits}")

    # === Test 2: 2^n mod F_k for all k, fixed n ===
    print("\n  Test 2: 2^10 mod F_k for various k")
    print(f"  {'k':>3}  {'F_k':>8}  {'2^10 mod F_k':>14}  {'Fib?':>8}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*14}  {'-'*8}")

    fib_set = set(fib(i) for i in range(30))
    for k in range(3, 20):
        fk = fib(k)
        mod_val = pow(2, 10, fk) if fk > 0 else 0
        is_fib = mod_val in fib_set
        fib_idx = ""
        if is_fib:
            for j in range(30):
                if fib(j) == mod_val:
                    fib_idx = f"F_{j}"
                    break
        print(f"  {k:3d}  {fk:8d}  {mod_val:14d}  {fib_idx:>8s}")

    # === Test 3: General 2^n mod F_k = F_{k-1} ===
    print("\n  Test 3: When does 2^n mod F_k = F_{k-1}? (n=1..100, k=3..20)")
    print(f"  {'k':>3}  {'F_k':>8}  {'F_{k-1}':>8}  {'smallest n':>12}  {'all n (<100)':>30}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*30}")

    for k in range(3, 21):
        fk = fib(k)
        fk_1 = fib(k-1)
        matching_n = [n for n in range(1, 101) if pow(2, n, fk) == fk_1]
        smallest = matching_n[0] if matching_n else None
        n_str = str(matching_n[:8]) + ("..." if len(matching_n) > 8 else "")
        print(f"  {k:3d}  {fk:8d}  {fk_1:8d}  {str(smallest):>12s}  {n_str:>30s}")

    # === Test 4: Pisano periods ===
    print("\n  Test 4: Pisano period pi(m) for m = 2^k")
    print("  (Period of Fibonacci mod m)")
    print(f"  {'m':>8}  {'pi(m)':>8}  {'notes':>20}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*20}")

    def pisano_period(m):
        """Compute the Pisano period: period of F_n mod m."""
        if m <= 1:
            return 1
        a, b = 0, 1
        for i in range(1, 6 * m + 10):
            a, b = b, (a + b) % m
            if a == 0 and b == 1:
                return i
        return None

    for k in range(1, 14):
        m = 2**k
        pp = pisano_period(m)
        notes = ""
        if pp and pp % 3 == 0:
            notes = f"= 3 * {pp//3}"
        print(f"  {m:8d}  {str(pp):>8s}  {notes:>20s}")

    # === Test 5: The order of 2 mod F_n ===
    print("\n  Test 5: Multiplicative order of 2 mod F_n")
    print(f"  {'n':>3}  {'F_n':>8}  {'ord_2(F_n)':>12}  {'F_n/ord':>10}  {'notes':>25}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*25}")

    from math import gcd

    def mult_order(a, m):
        """Multiplicative order of a mod m."""
        if gcd(a, m) != 1:
            return None
        val = a % m
        for k in range(1, m + 1):
            if val == 1:
                return k
            val = (val * a) % m
        return None

    for n in range(3, 21):
        fn = fib(n)
        if fn <= 1:
            continue
        order = mult_order(2, fn)
        ratio = fn / order if order else None
        notes = ""
        if order and n == order:
            notes = "*** n = ord ***"
        elif order and order == fn - 1:
            notes = "full order (F_n - 1)"
        print(f"  {n:3d}  {fn:8d}  {str(order):>12s}  {str(ratio)[:10] if ratio else '':>10s}  {notes:>25s}")

    # === Test 6: Period of 2 mod 55 and the exponent 20 ===
    print("\n  Test 6: Multiplicative order of 2 mod 55")
    order_2_55 = mult_order(2, 55)
    print(f"    ord(2 mod 55) = {order_2_55}")
    print(f"    Self-closing formula exponent = 20")
    print(f"    M_10 eigenvalue = phi^20")
    print(f"    MATCH: ord(2 mod F_10) = {order_2_55} = exponent in delta = phi^(20/N)")
    if order_2_55 == 20:
        print(f"    >>> The period of binary doubling mod F_10 IS the eigenvalue exponent! <<<")

    # Check: is this true for other Fibonacci?
    print(f"\n    Cross-check: ord(2 mod F_n) vs 2n for various n")
    print(f"    {'n':>3}  {'F_n':>8}  {'ord(2,F_n)':>12}  {'2n':>6}  {'match':>8}")
    print(f"    {'-'*3}  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*8}")
    for n in range(3, 21):
        fn = fib(n)
        if fn <= 1 or gcd(2, fn) > 1:
            continue
        order = mult_order(2, fn)
        match = order == 2*n if order else False
        marker = "YES <<<" if match else ""
        print(f"    {n:3d}  {fn:8d}  {str(order):>12s}  {2*n:6d}  {marker:>8s}")

    return hits


# ============================================================
# PART C: SELF-CLOSING FORMULA DISSECTION
# ============================================================

def part_c_self_closing():
    """
    delta = phi^{20/N} where N = sqrt(39 + 1/x), x = 160 + (delta-4)^2*(1 - 1/(1371 + delta-4))

    Questions:
    1. What are 39, 160, 1371 in terms of Fibonacci numbers?
    2. Can we derive them from the Mobius structure?
    3. Can we find a better formula?
    """
    print("\n\n" + "=" * 72)
    print("  PART C: SELF-CLOSING FORMULA DISSECTION")
    print("=" * 72)

    F10 = mpf(55)

    # === Step 1: Verify convergence ===
    print("\n  Step 1: Self-closing iteration")
    x = mpf(160)
    for i in range(8):
        N = sqrt(39 + 1/x)
        delta = PHI**(20/N)
        d4 = delta - 4
        x_new = 160 + d4**2 * (1 - 1/(1371 + d4))
        err = fabs(delta - DELTA_KNOWN)
        digits = float(-log10(err)) if err > 0 else 200
        print(f"    iter {i}: delta = {nstr(delta, 25)}, digits = {digits:.1f}")
        x = x_new

    # === Step 2: Analyze structural constants ===
    print("\n  Step 2: Structural constant decomposition")

    print("\n  39:")
    print(f"    = F_9 + F_5 = {fib(9)} + {fib(5)} = {fib(9) + fib(5)}")
    print(f"    = 3 * 13 = 3 * F_7")
    print(f"    = F_10 - F_8 + F_4 = {fib(10)} - {fib(8)} + {fib(4)} = {fib(10)-fib(8)+fib(4)} (= 37, NOT 39)")
    print(f"    Best: 39 = 34 + 5 = F_9 + F_5")
    # Check: is 39 special in N^2 = 39 + small?
    N_at_delta = 20 / float(log(DELTA_KNOWN) / log(PHI))
    print(f"    N = 20/log_phi(delta) = {N_at_delta:.15f}")
    print(f"    N^2 = {N_at_delta**2:.15f}")
    print(f"    N^2 - 39 = {N_at_delta**2 - 39:.15f}")
    print(f"    1/(N^2 - 39) = x = {1/(N_at_delta**2 - 39):.10f}")

    print("\n  160:")
    print(f"    = 2^5 * 5 = 32 * F_5")
    print(f"    = 8 * 20 = 2^3 * ord(2 mod 55)")
    print(f"    = F_10 * 3 - 5 = 55*3-5 = {55*3-5} (= 160, EXACT)")
    print(f"    Best: 160 = 8 * 20 (connects to ord(2 mod F_10) = 20)")

    print("\n  1371:")
    print(f"    = 3 * 457 = 3 * 457")
    print(f"    ~ F_10 * 5^2 - F_3 = 55 * 25 - 4 = {55*25-4}")
    print(f"    ~ F_10 * 25 - 4 = 1371, EXACT!")
    print(f"    = F_10 * F_5^2 - F_3")

    # === Step 3: Express everything through F_5 and F_10 ===
    print("\n  Step 3: F_5 = 5 and F_10 = 55 decomposition")
    print(f"    39 = F_9 + F_5 (uses both Fibonacci families)")
    print(f"    160 = 2^5 * F_5 = 32 * 5")
    print(f"    1371 = F_10 * F_5^2 - F_3 = 55 * 25 - 4")
    print(f"    1857 = F_10 * F_9 - F_7 = 55 * 34 - 13")
    print(f"    6050 = 2 * F_10^2 = 2 * 55^2")

    # === Step 4: Can we improve the self-closing formula? ===
    print("\n  Step 4: Improved self-closing formula search")

    # The current formula: x = 160 + (d4)^2 * (1 - 1/(1371 + d4))
    # Let's see what x_exact needs to be
    d4_exact = DELTA_KNOWN - 4
    N_exact = 20 * log(PHI) / log(DELTA_KNOWN)
    x_exact = 1 / (N_exact**2 - 39)

    print(f"    Exact x = {nstr(x_exact, 30)}")
    print(f"    Formula x = 160 + d4^2 * (1 - 1/(1371 + d4))")

    x_formula = 160 + d4_exact**2 * (1 - 1/(1371 + d4_exact))
    print(f"    Formula x = {nstr(x_formula, 30)}")
    print(f"    x_exact - x_formula = {nstr(x_exact - x_formula, 15)}")

    # The residual tells us about higher-order terms
    x_residual = x_exact - x_formula
    print(f"\n    Residual / d4^3 = {nstr(x_residual / d4_exact**3, 20)}")
    print(f"    Residual / d4^4 = {nstr(x_residual / d4_exact**4, 20)}")
    print(f"    Residual / (d4^3 / pi) = {nstr(x_residual * pi / d4_exact**3, 20)}")

    # Try: x = 160 + d4^2 * (1 - 1/(1371 + d4) + c3 * d4)
    # Solve for c3
    term2 = d4_exact**2 * (1 - 1/(1371 + d4_exact))
    c3_needed = (x_exact - 160 - term2) / (d4_exact**3)
    print(f"\n    c_3 correction: x = 160 + d4^2*(1 - 1/(1371+d4)) + c_3*d4^3")
    print(f"    c_3 = {nstr(c3_needed, 25)}")
    print(f"    c_3 * 55 = {nstr(c3_needed * 55, 20)}")
    print(f"    c_3 * 55^2 = {nstr(c3_needed * 55**2, 20)}")
    print(f"    c_3 * pi = {nstr(c3_needed * pi, 20)}")
    print(f"    1/c_3 = {nstr(1/c3_needed, 20)}")

    # Test the improved formula
    x_improved = 160 + d4_exact**2 * (1 - 1/(1371 + d4_exact)) + c3_needed * d4_exact**3
    N_improved = sqrt(39 + 1/x_improved)
    delta_improved = PHI**(20/N_improved)
    err_improved = fabs(delta_improved - DELTA_KNOWN)
    print(f"\n    With c_3 correction:")
    print(f"    delta digits: {float(-log10(err_improved)) if err_improved > 0 else 200:.1f}")

    # === Step 5: The 20 = 2*10 connection ===
    print("\n  Step 5: Why 20?")
    print(f"    M_10 eigenvalue at -1/phi: phi^20 = {nstr(PHI**20, 20)}")
    print(f"    20 = 2 * 10 (double the Fibonacci index)")
    print(f"    phi^20 = phi^(2*10) = (phi^10)^2")
    print(f"    phi^10 = {nstr(PHI**10, 20)}")
    print(f"    F_10 + F_9*phi = phi^10 (exact Fibonacci identity)")

    # Verify
    phi10_check = fib(10) + fib(9) * PHI
    print(f"    Check: 55 + 34*phi = {nstr(phi10_check, 20)}")
    print(f"    phi^10 = {nstr(PHI**10, 20)}")
    print(f"    Match: {nstr(fabs(phi10_check - PHI**10), 5)}")

    # === Step 6: What N encodes ===
    print(f"\n  Step 6: What is N?")
    print(f"    N_exact = {nstr(N_exact, 25)}")
    print(f"    N^2 = {nstr(N_exact**2, 25)}")
    print(f"    sqrt(40) = {nstr(sqrt(40), 25)}")
    print(f"    N - sqrt(40) = {nstr(N_exact - sqrt(40), 20)}")
    print(f"    N^2 - 39 = {nstr(N_exact**2 - 39, 20)}")
    print(f"    N^2 - 40 = {nstr(N_exact**2 - 40, 20)}")

    # Check: is N related to phi?
    print(f"\n    N / phi = {nstr(N_exact / PHI, 20)}")
    print(f"    N / phi^2 = {nstr(N_exact / PHI**2, 20)}")
    print(f"    N * phi = {nstr(N_exact * PHI, 20)}")
    print(f"    N - phi^3 = {nstr(N_exact - PHI**3, 20)}")
    print(f"    phi^3 = {nstr(PHI**3, 20)}")
    print(f"    N / (2*phi) = {nstr(N_exact / (2*PHI), 20)}")

    return N_exact, x_exact


# ============================================================
# PART D: SYNTHESIS
# ============================================================

def synthesis(c_list, A_list, orbit_hits, N_exact, x_exact):
    print("\n\n" + "=" * 72)
    print("  SYNTHESIS: FIBONACCI MECHANISM ASSESSMENT")
    print("=" * 72)

    # 1. C-series structure
    print("\n  1. MOBIUS C-SERIES (1/Delta_z = 1857 + C*(d-4)/pi):")
    print(f"     1857 = F_10 * F_9 - F_7")
    print(f"     C = c_0 + c_1/F^2 + c_2/F^4 + ...")
    for k in range(min(6, len(c_list))):
        print(f"     c_{k} = {nstr(c_list[k], 18)}")

    # 2. A coefficients (beyond formula)
    if A_list and len(A_list) > 3:
        print(f"\n  2. r_inf CORRECTION SERIES (beyond 13-digit formula):")
        for k in range(2, min(7, len(A_list))):
            if A_list[k] is not None:
                print(f"     A_{k} = {nstr(A_list[k], 18)}")
        # Check A_3/A_2
        if A_list[3] is not None and A_list[2] is not None and fabs(A_list[2]) > 0:
            r32 = A_list[3] / A_list[2]
            print(f"     A_3/A_2 = {nstr(r32, 15)}")
            print(f"     Paper claims 6050 = 2*F_10^2. Deviation: {nstr(r32 - 6050, 10)}")

    # 3. Orbit arithmetic
    print(f"\n  3. ORBIT MODULAR ARITHMETIC:")
    print(f"     2^10 mod 55 = {pow(2, 10, 55)} (= F_9 = 34)")
    print(f"     Values of n where 2^n mod F_n = F_{{n-1}}: {orbit_hits}")

    # 4. Self-closing
    print(f"\n  4. SELF-CLOSING CONSTANTS:")
    print(f"     39 = F_9 + F_5")
    print(f"     160 = 2^5 * F_5")
    print(f"     1371 = F_10 * F_5^2 - F_3")
    print(f"     All expressible through F_5=5 and F_10=55 (and small Fibonacci)")

    # 5. Overall assessment
    print(f"\n  5. KEY QUESTIONS ANSWERED:")
    print(f"     - Does the C-series converge? (check digits gained per level)")
    print(f"     - Do c_k ratios follow phi or Fibonacci pattern?")
    print(f"     - Is 2^n mod F_n = F_{{n-1}} unique to n=10?")
    print(f"     - Can the self-closing formula be improved beyond 13 digits?")


def run():
    t0 = time.time()

    c_list, A_list = part_a_mobius_series()
    orbit_hits = part_b_orbit_modular()
    N_exact, x_exact = part_c_self_closing()
    synthesis(c_list, A_list, orbit_hits, N_exact, x_exact)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = results_dir / f'exp_03_mobius_fibonacci_mechanism_{ts}.json'

    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'exp_03_mobius_fibonacci_mechanism.py',
        'elapsed_seconds': elapsed,
        'C_coefficients': [str(c) for c in c_list],
        'A_coefficients': [str(a) if a is not None else None for a in A_list],
        'orbit_hits': orbit_hits,
    }

    with open(fpath, 'w') as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"  Saved: {fpath}")


if __name__ == "__main__":
    run()
