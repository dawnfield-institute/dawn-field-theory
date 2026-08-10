"""
exp_26: Unified Correction Template — F_a/(mπF_b²) Across Couplings

MOTIVATION:
  Two independent physical constants share a correction structure:

    α_EM:    base × [1 − F₁₀/(4πF₇²)]    where F₁₀=55, F₇=13  →  5.7 ppm
    Gravity: F₁₈₃ × [1 + F₁₃/(πF₆²)]     where F₁₃=233, F₆=8  →  0.08% log₁₀

  Both have the form F_a/(mπF_b²) — a Fibonacci number divided by an
  integer multiple of π times a squared Fibonacci number. This could be:
    (a) coincidence (small Fibonacci numbers produce many near-matches)
    (b) a genuine correction template from the cascade topology

  This experiment tests the template systematically:
    1. Exact side-by-side comparison
    2. Phase decomposition (what do a, b, m mean physically?)
    3. Search for other constants that follow the same pattern
    4. Monte Carlo: how likely is this structural match by chance?

CHAIN:
  - exp_03 (α correction decomposition): F₁₀/(4πF₇²) = Phase I × III / Phase II²
  - exp_23 (F₁₈₃ correction): 1 + F₁₃/(πF₆²) = 2.159, residual 0.0008
  - PRE_STRUCTURAL_EMERGENCE.md: correction = "cross-phase product"
"""

import sys
import os
import numpy as np
from itertools import product as iterproduct
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, INV_PHI, LN_PHI, XI_BALANCE, FIB
from core.utils import experiment_header, save_results

# =====================================================================
# Physical Constants
# =====================================================================

ALPHA_EM_MEASURED = 7.2973525693e-3   # CODATA 2018
G = 6.67430e-11
HBAR = 1.054571817e-34
C = 2.99792458e8
M_PLANCK_GEV = 1.22089e19
M_PROTON_GEV = 0.938272088

# Other precision constants to test
ALPHA_S_MZ = 0.1180
SIN2_THETA_W = 0.23122
WEINBERG_ANGLE_RAD = np.arcsin(np.sqrt(SIN2_THETA_W))  # ~0.5017 rad

# Mass ratios
MU_E = 206.7682830
P_E = 1836.15267343
TAU_E = 3477.48

# Cosmological
OMEGA_C = 0.265
OMEGA_LAMBDA = 0.685

F = lambda n: FIB[n] if n < len(FIB) else _fib(n)

def _fib(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


# =====================================================================
# MAIN
# =====================================================================

def main():
    meta = experiment_header(
        'exp_26_unified_correction_template',
        'Testing F_a/(mπF_b²) as a universal correction template',
        paper='Paper 4/5',
        section='§corrections'
    )

    results = {**meta, 'tests': {}}

    # =================================================================
    # TEST 1: Exact Side-by-Side Comparison
    # =================================================================
    print("=" * 70)
    print("Test 1: Side-by-Side Correction Comparison")
    print("=" * 70 + "\n")

    # α_EM formula
    alpha_base = F(3) / (F(4) * PHI * F(10))  # 2/(3 × φ × 55)
    alpha_correction_val = F(10) / (4 * np.pi * F(7)**2)  # 55/(4π × 169)
    alpha_correction = 1 - alpha_correction_val
    alpha_predicted = alpha_base * alpha_correction
    alpha_error_ppm = abs(alpha_predicted - ALPHA_EM_MEASURED) / ALPHA_EM_MEASURED * 1e6

    # Gravity formula
    gravity_depth = F(7)**2 + F(7) + 1  # 183
    log10_f183 = gravity_depth * np.log10(PHI) - 0.5 * np.log10(5)
    mass_ratio_sq = (M_PLANCK_GEV / M_PROTON_GEV)**2
    log10_target = np.log10(mass_ratio_sq)
    gravity_correction_val = F(13) / (1 * np.pi * F(6)**2)  # 233/(π × 64)
    gravity_correction = 1 + gravity_correction_val
    log10_corrected = log10_f183 + np.log10(gravity_correction)
    gravity_residual = abs(log10_corrected - log10_target)

    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │             α_EM CORRECTION           │  GRAVITY CORRECTION  │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Form:   1 − F_a/(m·π·F_b²)          │  1 + F_a/(m·π·F_b²) │")
    print(f"  │  Sign:   MINUS (−)                    │  PLUS (+)            │")
    print(f"  │  F_a:    F₁₀ = {F(10):>5d}                  │  F₁₃ = {F(13):>5d}          │")
    print(f"  │  F_b:    F₇  = {F(7):>5d}                  │  F₆  = {F(6):>5d}            │")
    print(f"  │  m:      4                            │  1                   │")
    print(f"  │  Corr:   {alpha_correction_val:.6f}               │  {gravity_correction_val:.6f}           │")
    print(f"  │  Factor: {alpha_correction:.6f}               │  {gravity_correction:.6f}            │")
    print(f"  │  Result: {alpha_predicted:.10f}          │  log₁₀ = {log10_corrected:.6f}   │")
    print(f"  │  Target: {ALPHA_EM_MEASURED:.10f}          │  log₁₀ = {log10_target:.6f}   │")
    print(f"  │  Error:  {alpha_error_ppm:.1f} ppm                   │  {gravity_residual:.6f} log₁₀     │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Structural comparison
    print(f"\n  Structural comparison:")
    print(f"    Both use F₇ as the gauge depth anchor")
    print(f"    α_EM:    F_b = F₇ directly (EM gauge depth)")
    print(f"    Gravity: depth = F₇² + F₇ + 1 (cyclotomic of F₇)")
    print(f"    α_EM correction: a=10, b=7 → a = b + 3")
    print(f"    Gravity correction: a=13, b=6 → a = 2b + 1")
    print(f"    α_EM: m=4 (4π = solid angle)")
    print(f"    Gravity: m=1 (π = half-cycle)")

    # Index relationships
    print(f"\n  Index relationships:")
    print(f"    α_EM:    a − b = 10 − 7 = 3 = F₄")
    print(f"    Gravity: a − b = 13 − 6 = 7 = F₇  (!)") 
    print(f"    α_EM:    a + b = 10 + 7 = 17 = F₈ (?)")
    print(f"    Gravity: a + b = 13 + 6 = 19 (prime)")
    print(f"    α_EM:    b = 7 (F₇ = 13)")
    print(f"    Gravity: b = 6 (F₆ = 8)")
    print(f"    Note: F₆ and F₇ are consecutive Fibonacci indices")

    results['tests']['side_by_side'] = {
        'alpha': {
            'base': float(alpha_base),
            'correction_value': float(alpha_correction_val),
            'correction_factor': float(alpha_correction),
            'predicted': float(alpha_predicted),
            'measured': ALPHA_EM_MEASURED,
            'error_ppm': float(alpha_error_ppm),
            'F_a': 10, 'F_b': 7, 'm': 4, 'sign': 'minus',
        },
        'gravity': {
            'depth': gravity_depth,
            'correction_value': float(gravity_correction_val),
            'correction_factor': float(gravity_correction),
            'log10_corrected': float(log10_corrected),
            'log10_target': float(log10_target),
            'residual_log10': float(gravity_residual),
            'F_a': 13, 'F_b': 6, 'm': 1, 'sign': 'plus',
        },
        'shared_anchor': 'F₇ (EM gauge depth)',
        'a_minus_b': {'alpha': 3, 'gravity': 7, 'note': 'F₄ and F₇'},
        'status': 'INFO',
    }

    # =================================================================
    # TEST 2: Generalized Template Search
    #
    # For each known constant, search:
    #   base_expression × [1 ± F_a/(m·π·F_b²)]
    # where a=1..15, b=1..12, m=1..8, sign ∈ {+,−}
    # and base_expression is a simple Fibonacci ratio.
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 2: Template Search for Other Constants")
    print("=" * 70 + "\n")

    # Constants to test: for each, define plausible Fibonacci base expressions
    # and the measured value to match
    test_targets = {
        'sin²θ_W': {
            'measured': SIN2_THETA_W,
            'bases': [
                ('3/13', 3/13),   # F₄/F₇ — the known formula
                ('3/F₇', F(4)/F(7)),
            ],
        },
        'α_s(M_Z)': {
            'measured': ALPHA_S_MZ,
            'bases': [
                ('F₃/F₇', F(3)/F(7)),     # 2/13 ≈ 0.154
                ('1/(F₆+1)', 1/(F(6)+1)),  # 1/9 ≈ 0.111
                ('F₅/F₈', F(5)/F(8)),      # 5/21 ≈ 0.238
            ],
        },
        'μ/e': {
            'measured': MU_E,
            'bases': [
                ('F₈·F₇/φ', F(8)*F(7)/PHI),   # 21×13/φ ≈ 168.7
                ('F₁₁/φ²', F(11)/PHI**2),      # 89/2.618 ≈ 34
                ('F₁₂/φ', F(12)/PHI),           # 144/φ ≈ 89
            ],
        },
        'p/e': {
            'measured': P_E,
            'bases': [
                ('F₉·F₆·F₄/φ', F(9)*F(6)*F(4)/PHI),  # 34×8×3/φ ≈ 504
                ('F₁₂·F₇/F₆', F(12)*F(7)/F(6)),       # 144×13/8 = 234
                ('F₁₆/φ', F(16)/PHI),                   # 987/φ ≈ 610
            ],
        },
        'Ω_c': {
            'measured': OMEGA_C,
            'bases': [
                ('F₃·Ξ/F₆', F(3)*XI_BALANCE/F(6)),  # known match
                ('F₄/F₇', F(4)/F(7)),                 # 3/13 ≈ 0.231
                ('1/F₅', 1/F(5)),                      # 1/5 = 0.2
            ],
        },
    }

    found_corrections = {}

    for target_name, config in test_targets.items():
        measured = config['measured']
        best_match = None
        best_residual = float('inf')

        for base_label, base_val in config['bases']:
            if base_val <= 0:
                continue

            for sign in [+1, -1]:
                for a in range(2, 16):
                    fa = _fib(a)
                    for b in range(2, 13):
                        fb = _fib(b)
                        for m in [1, 2, 3, 4, 6, 8]:
                            denom = m * np.pi * fb**2
                            corr_val = fa / denom
                            if corr_val > 0.5:
                                continue  # Correction should be small
                            corr_factor = 1 + sign * corr_val
                            if corr_factor <= 0:
                                continue
                            predicted = base_val * corr_factor
                            if predicted <= 0:
                                continue
                            residual = abs(predicted - measured) / abs(measured)

                            if residual < best_residual:
                                best_residual = residual
                                best_match = {
                                    'base': base_label,
                                    'base_val': float(base_val),
                                    'sign': '+' if sign > 0 else '−',
                                    'a': a, 'b': b, 'm': m,
                                    'F_a': fa, 'F_b': fb,
                                    'correction_val': float(corr_val),
                                    'correction_factor': float(corr_factor),
                                    'predicted': float(predicted),
                                    'residual': float(residual),
                                    'residual_ppm': float(residual * 1e6),
                                    'a_minus_b': a - b,
                                }

        found_corrections[target_name] = best_match

        if best_match:
            sign_str = best_match['sign']
            print(f"  {target_name:12s}  base = {best_match['base']:15s}  "
                  f"[1 {sign_str} F_{best_match['a']}/({best_match['m']}πF_{best_match['b']}²)]  "
                  f"= {best_match['predicted']:.8f}  "
                  f"err = {best_match['residual_ppm']:.1f} ppm"
                  f"  (a−b = {best_match['a_minus_b']})")

    # How many found sub-100 ppm corrections?
    sub_100 = sum(1 for m in found_corrections.values()
                  if m and m['residual_ppm'] < 100)
    sub_1000 = sum(1 for m in found_corrections.values()
                   if m and m['residual_ppm'] < 1000)

    print(f"\n  Summary: {sub_100}/{len(found_corrections)} below 100 ppm, "
          f"{sub_1000}/{len(found_corrections)} below 1000 ppm")

    results['tests']['template_search'] = {
        'corrections_found': found_corrections,
        'n_sub_100ppm': sub_100,
        'n_sub_1000ppm': sub_1000,
        'status': 'PASS' if sub_100 >= 3 else 'FAIL',
    }

    # =================================================================
    # TEST 3: Monte Carlo — How Likely Is This Pattern?
    #
    # Generate random "corrections" of the form r/(mπs²) where r,s
    # are drawn from randomly ordered integers (not Fibonacci), m=1..8.
    # How often does a random correction match TWO independent constants
    # as well as Fibonacci does?
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 3: Monte Carlo — Probability of Structural Match")
    print("=" * 70 + "\n")

    rng = np.random.RandomState(42)
    n_mc = 5000

    # For each MC trial:
    #   Generate a random sequence of 15 positive integers (like Fibonacci)
    #   Try to match BOTH α_EM and gravity with the same template
    mc_both_match = 0
    mc_alpha_match = 0
    mc_gravity_match = 0

    alpha_threshold = 10  # ppm
    gravity_threshold = 0.001  # log₁₀

    for trial in range(n_mc):
        # Random "sequence" — sorted positive integers with growth rate ~φ
        seq = sorted(rng.choice(range(1, 2000), size=15, replace=False))

        trial_alpha_ok = False
        trial_gravity_ok = False

        # Try all template parameters
        for a in range(4, 15):  # Need decent-sized F_a
            for b in range(2, 10):
                for m in [1, 2, 3, 4, 6, 8]:
                    sa = seq[a]
                    sb = seq[b]
                    corr_val = sa / (m * np.pi * sb**2)
                    if corr_val > 0.5 or corr_val < 1e-6:
                        continue

                    # Test α_EM
                    for sign in [+1, -1]:
                        for base_a_idx in range(1, 6):
                            for base_b_idx in range(1, 8):
                                if base_b_idx == base_a_idx:
                                    continue
                                base_val = seq[base_a_idx] / (seq[base_b_idx] * PHI * seq[min(a, 14)])
                                if base_val <= 0 or base_val > 0.1:
                                    continue
                                pred = base_val * (1 + sign * corr_val)
                                err = abs(pred - ALPHA_EM_MEASURED) / ALPHA_EM_MEASURED * 1e6
                                if err < alpha_threshold:
                                    trial_alpha_ok = True

                    # Test gravity
                    for sign in [+1, -1]:
                        for depth_idx in range(3, 12):
                            depth = seq[depth_idx]**2 + seq[depth_idx] + 1
                            if depth > 5000:
                                continue
                            log_fd = depth * np.log10(PHI) - 0.5 * np.log10(5)
                            corrected = log_fd + np.log10(1 + sign * corr_val)
                            residual = abs(corrected - log10_target)
                            if residual < gravity_threshold:
                                trial_gravity_ok = True

                    if trial_alpha_ok and trial_gravity_ok:
                        break
                if trial_alpha_ok and trial_gravity_ok:
                    break
            if trial_alpha_ok and trial_gravity_ok:
                break

        if trial_alpha_ok:
            mc_alpha_match += 1
        if trial_gravity_ok:
            mc_gravity_match += 1
        if trial_alpha_ok and trial_gravity_ok:
            mc_both_match += 1

    frac_alpha = mc_alpha_match / n_mc
    frac_gravity = mc_gravity_match / n_mc
    frac_both = mc_both_match / n_mc
    frac_independent = frac_alpha * frac_gravity

    print(f"  Monte Carlo ({n_mc} trials, random integer sequences):")
    print(f"    α_EM match (<{alpha_threshold} ppm):     {mc_alpha_match}/{n_mc} = {frac_alpha:.4f}")
    print(f"    Gravity match (<{gravity_threshold} log₁₀): {mc_gravity_match}/{n_mc} = {frac_gravity:.4f}")
    print(f"    BOTH match:                    {mc_both_match}/{n_mc} = {frac_both:.4f}")
    print(f"    Independent expectation:       {frac_independent:.6f}")
    if frac_both > 0:
        enrichment = frac_both / max(frac_independent, 1/n_mc)
        print(f"    Enrichment (observed/expected): {enrichment:.1f}×")
    else:
        print(f"    Enrichment: 0/{frac_independent:.6f} = under-represented")

    t3_pass = frac_both < 0.05  # Less than 5% of random sequences match both

    results['tests']['monte_carlo'] = {
        'n_trials': n_mc,
        'alpha_threshold_ppm': alpha_threshold,
        'gravity_threshold_log10': gravity_threshold,
        'frac_alpha': float(frac_alpha),
        'frac_gravity': float(frac_gravity),
        'frac_both': float(frac_both),
        'frac_independent': float(frac_independent),
        'status': 'PASS' if t3_pass else 'FAIL',
    }

    # =================================================================
    # TEST 4: Index Pattern Analysis
    #
    # If the template is real, the indices (a, b, m) should have
    # structural meaning. Test whether a−b is always a Fibonacci number.
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 4: Index Pattern Analysis")
    print("=" * 70 + "\n")

    fib_set = set(FIB[:20])

    entries = [
        {'name': 'α_EM', 'a': 10, 'b': 7, 'm': 4, 'sign': '−'},
        {'name': 'Gravity', 'a': 13, 'b': 6, 'm': 1, 'sign': '+'},
    ]

    # Add any good matches from Test 2
    for tname, match in found_corrections.items():
        if match and match['residual_ppm'] < 100:
            entries.append({
                'name': tname,
                'a': match['a'], 'b': match['b'],
                'm': match['m'], 'sign': match['sign'],
            })

    print(f"  {'Name':12s}  {'a':>3s}  {'b':>3s}  {'m':>3s}  {'sign':>4s}  "
          f"{'a−b':>4s}  {'a−b∈Fib':>7s}  {'a+b':>4s}  {'a+b∈Fib':>7s}  "
          f"{'m∈Fib':>5s}")
    print(f"  {'-'*12}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*4}  "
          f"{'-'*4}  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*5}")

    n_fib_diff = 0
    n_entries = len(entries)

    for e in entries:
        a, b, m = e['a'], e['b'], e['m']
        diff = a - b
        summ = a + b
        diff_fib = diff in fib_set
        sum_fib = summ in fib_set
        m_fib = m in fib_set

        if diff_fib:
            n_fib_diff += 1

        print(f"  {e['name']:12s}  {a:3d}  {b:3d}  {m:3d}  {e['sign']:>4s}  "
              f"{diff:4d}  {'YES' if diff_fib else 'no':>7s}  "
              f"{summ:4d}  {'YES' if sum_fib else 'no':>7s}  "
              f"{'YES' if m_fib else 'no':>5s}")

    # Check if the (a, b) pairs follow a pattern
    print(f"\n  Index relationships:")
    print(f"    α_EM:    (a,b) = (10, 7) → a = F₇ index, b = F₇ index")
    print(f"    Gravity: (a,b) = (13, 6) → a = F₇ value, b = 7−1")
    print(f"    Note: α_EM's a−b = 3 = F₄ (small Fib)")
    print(f"    Note: Gravity's a−b = 7 = F₇ index (gauge depth index)")
    print(f"    Note: α_EM's m=4, Gravity's m=1 → ratio 4:1")

    # The sign pattern
    print(f"\n  Sign pattern:")
    print(f"    α_EM:    MINUS → correction reduces base (self-screening)")
    print(f"    Gravity: PLUS  → correction amplifies base (enhancement)")
    print(f"    EM screens itself (virtual pairs reduce coupling)")
    print(f"    Gravity is enhanced (mass-energy adds to gravitational effect)")

    t4_fib_frac = n_fib_diff / n_entries if n_entries > 0 else 0
    t4_pass = t4_fib_frac >= 0.5  # At least half have Fibonacci a−b

    results['tests']['index_patterns'] = {
        'entries': entries,
        'n_fib_diff': n_fib_diff,
        'fib_diff_fraction': float(t4_fib_frac),
        'status': 'PASS' if t4_pass else 'FAIL',
    }

    # =================================================================
    # SYNTHESIS
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    t1_s = 'INFO'
    t2_s = results['tests']['template_search']['status']
    t3_s = results['tests']['monte_carlo']['status']
    t4_s = results['tests']['index_patterns']['status']

    tests_pass = sum(1 for s in [t2_s, t3_s, t4_s] if s == 'PASS')

    print(f"\n  Test 1 (side-by-side):    {t1_s}")
    print(f"  Test 2 (template search): {t2_s}")
    print(f"  Test 3 (Monte Carlo):     {t3_s}")
    print(f"  Test 4 (index patterns):  {t4_s}")
    print(f"\n  Result: {tests_pass}/3 PASS")

    print(f"\n  The correction template F_a/(mπF_b²):")
    print(f"    α_EM:    1 − F₁₀/(4πF₇²) = 1 − {alpha_correction_val:.6f} → {alpha_error_ppm:.1f} ppm")
    print(f"    Gravity: 1 + F₁₃/(πF₆²)  = 1 + {gravity_correction_val:.6f} → {gravity_residual:.6f} log₁₀")
    print(f"    Both anchored to F₇ = 13 (EM gauge depth)")
    print(f"    Sign: − for EM (screening), + for gravity (enhancement)")

    results['falsification'] = {
        'test_id': 'F24',
        'hypothesis': (
            'F_a/(mπF_b²) is a universal Fibonacci correction template '
            'appearing across independent physical constants.'
        ),
        'chain': [
            f'Test 2 (template search): {t2_s} — {sub_100}/{len(found_corrections)} constants at <100 ppm',
            f'Test 3 (Monte Carlo): {t3_s} — {frac_both:.4f} random match rate',
            f'Test 4 (index patterns): {t4_s} — {t4_fib_frac:.0%} have Fibonacci a−b',
        ],
        'n_pass': f'{tests_pass}/3',
        'falsified': tests_pass < 1,
    }

    save_results(results, 'exp_26_unified_correction_template')


if __name__ == '__main__':
    main()
