"""
exp_23: F₁₈₃ Gravity Hierarchy — Correction Term and Uniqueness

HYPOTHESIS: The raw comparison log₁₀(F₁₈₃) vs log₁₀((M_Pl/m_p)²) has a
gap of ~0.33 in log₁₀ (factor ~2.15). A Fibonacci correction term — analogous
to α_EM's [1 − F₁₀/(4πF₇²)] — may close this gap. Additionally, we test
whether 183 = F₇² + F₇ + 1 is uniquely good among integer formulas.

WHAT exp_03/exp_09 established:
  - 183 = F₇² + F₇ + 1 (cyclotomic Φ₃ evaluated at F₇)
  - F₁₈₃ ≈ 10^37.895
  - (M_Planck/m_proton)² ≈ 10^38.228
  - Gap: ~0.33 in log₁₀ (factor of ~2.15)

WHAT exp_08 warned:
  - "183 needs uniqueness proof"
  - "Many formulas involving small numbers give ~10³⁸"
  - Must try alternatives to confirm

TESTS:
  1. Precise gap computation: exact log₁₀(F₁₈₃) vs measured, both with
     mass-ratio method and coupling-ratio method
  2. Fibonacci correction search: systematically search for corrections
     of the form F_k/(m·π^n·F_j^p) that close the gap
  3. Uniqueness: search all n² + n + 1 formulas for n = F_1..F_15.
     How many give log₁₀ within 0.5 of 38.23? Within 0.1?
  4. Random formula test: sample 10,000 random formulas a^2 + a + 1
     for a ∈ [1, 100]. What fraction match the hierarchy?
  5. New exp_22 insight: PAC depth bound = φ². Does the correction
     involve φ² (the newly-derived MED bound)?

FALSIFICATION (F21):
  If >10% of random small-integer formulas give |delta_log10| < 0.5,
  then 183 is NOT special and the claim is numerological. If the
  correction search turns up nothing clean, the gap remains structural.

SOURCES:
  - gravity_from_maxwell_pac/scripts/exp_03_f183_hierarchy.py
  - standard_model_fibonacci_arithmetic/Code/experiments/exp_09_gravity_hierarchy.py
  - exp_08_falsification.py (xi_warning, alternative_formulas)
  - exp_22: PAC depth bound = φ² (new result)
"""

import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, LN_PHI, INV_PHI
from core.utils import experiment_header, save_results

# =====================================================================
# Physical Constants
# =====================================================================

# All in SI
C = 2.99792458e8         # m/s
G = 6.67430e-11          # m³/(kg·s²)
HBAR = 1.054571817e-34   # J·s
ALPHA_EM = 7.2973525693e-3

# Masses
M_PLANCK_KG = 2.176434e-8   # kg
M_PROTON_KG = 1.67262192e-27  # kg

# GeV
M_PLANCK_GEV = 1.22089e19
M_PROTON_GEV = 0.938272088

# Fibonacci
def fib(n):
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

def fib_log10(n):
    """log₁₀(F_n) via Binet for large n."""
    return n * np.log10(PHI) - 0.5 * np.log10(5)

F7 = fib(7)   # 13
F10 = fib(10)  # 55

# =====================================================================
# MAIN
# =====================================================================

def main():
    meta = experiment_header(
        'exp_23_f183_correction',
        'F₁₈₃ gravity hierarchy: correction term and uniqueness test',
        paper='Paper 4/5',
        section='§11 (gravity speculative), §8.2'
    )

    results = {**meta, 'tests': {}}

    # =================================================================
    # TEST 1: Precise Gap Computation
    # =================================================================
    print("=" * 70)
    print("Test 1: Precise Gap Computation")
    print("=" * 70 + "\n")

    # Gravity depth
    depth_gravity = F7**2 + F7 + 1  # = 183
    log10_f183 = fib_log10(depth_gravity)

    # Measured hierarchy — multiple methods
    # Method A: (M_Planck / m_proton)²
    mass_ratio = M_PLANCK_GEV / M_PROTON_GEV
    mass_ratio_sq = mass_ratio ** 2
    log10_mass = np.log10(mass_ratio_sq)

    # Method B: α_EM / α_G where α_G = G·m_p²/(ℏc)
    alpha_G = G * M_PROTON_KG**2 / (HBAR * C)
    coupling_ratio = ALPHA_EM / alpha_G
    log10_coupling = np.log10(coupling_ratio)

    # Method C: force ratio at 1 fm
    r_fm = 1e-15
    e_charge = 1.602176634e-19
    k_coulomb = 8.9875517873681764e9
    F_EM = k_coulomb * e_charge**2 / r_fm**2
    F_grav = G * M_PROTON_KG**2 / r_fm**2
    force_ratio = F_EM / F_grav
    log10_force = np.log10(force_ratio)

    print(f"  Gravity depth: {F7}² + {F7} + 1 = {depth_gravity}")
    print(f"  log₁₀(F₁₈₃) = {log10_f183:.6f}")
    print()
    print(f"  Method A (mass ratio²):    log₁₀ = {log10_mass:.6f}")
    print(f"  Method B (α_EM/α_G):       log₁₀ = {log10_coupling:.6f}")
    print(f"  Method C (force ratio):    log₁₀ = {log10_force:.6f}")

    # The hierarchy problem: why is gravity so weak?
    # Standard measure: (M_Planck/m_proton)² = 1/α_G ≈ 10^38.23
    # This is what the papers compare F₁₈₃ to (not α_EM/α_G)
    log10_target = log10_mass  # Method A: (M_Pl/m_p)²
    gap = log10_target - log10_f183
    factor = 10**gap

    print(f"\n  Canonical target: log₁₀((M_Pl/m_p)²) = {log10_target:.6f}")
    print(f"  Note: α_EM/α_G = α_EM × (M_Pl/m_p)², different quantity")
    print(f"  Gap: {gap:.6f} in log₁₀ (factor {factor:.4f})")
    print(f"  F₁₈₃ is {'smaller' if gap > 0 else 'larger'} by factor {factor:.4f}")
    print(f"  Equivalently: measured ≈ F₁₈₃ × {factor:.4f}")

    results['tests']['precise_gap'] = {
        'gravity_depth': depth_gravity,
        'log10_f183': log10_f183,
        'log10_mass_ratio_sq': log10_mass,
        'log10_coupling_ratio': log10_coupling,
        'log10_force_ratio': log10_force,
        'canonical_target': log10_target,
        'gap_log10': gap,
        'correction_factor': factor,
        'status': 'INFO',
    }

    # =================================================================
    # TEST 2: Fibonacci Correction Search
    #
    # The α_EM formula has correction [1 − F₁₀/(4πF₇²)].
    # Search for similar structure: F₁₈₃ × C_correction ≈ measured.
    # C_correction should involve Fibonacci numbers, π, and small integers.
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 2: Fibonacci Correction Search")
    print("=" * 70 + "\n")

    # We need a correction C such that F₁₈₃ × C = measured_hierarchy
    # In log space: log₁₀(C) = gap ≈ 0.33
    # C ≈ 2.15

    # Build candidate corrections from Fibonacci/π/φ building blocks
    fib_nums = {f'F_{k}': fib(k) for k in range(2, 20)}
    candidates = []

    # Simple ratio candidates: F_a / F_b, F_a / (n·F_b), etc.
    for ka in range(2, 20):
        fa = fib(ka)
        for kb in range(2, 20):
            fb = fib(kb)
            if fb == 0:
                continue

            # F_a / F_b
            c = fa / fb
            if c > 0.5 and c < 10:
                log_c = np.log10(c)
                err = abs(log_c - gap)
                if err < 0.01:
                    candidates.append({
                        'formula': f'F_{ka}/F_{kb} = {fa}/{fb}',
                        'value': c,
                        'log10': log_c,
                        'error_log10': err,
                        'category': 'ratio',
                    })

            # F_a / (π · F_b)
            c = fa / (np.pi * fb)
            if c > 0.5 and c < 10:
                log_c = np.log10(c)
                err = abs(log_c - gap)
                if err < 0.01:
                    candidates.append({
                        'formula': f'F_{ka}/(π·F_{kb}) = {fa}/(π·{fb})',
                        'value': c,
                        'log10': log_c,
                        'error_log10': err,
                        'category': 'ratio_pi',
                    })

            # π · F_a / F_b
            c = np.pi * fa / fb
            if c > 0.5 and c < 10:
                log_c = np.log10(c)
                err = abs(log_c - gap)
                if err < 0.01:
                    candidates.append({
                        'formula': f'π·F_{ka}/F_{kb} = π·{fa}/{fb}',
                        'value': c,
                        'log10': log_c,
                        'error_log10': err,
                        'category': 'pi_ratio',
                    })

    # φ-based corrections
    for n in range(-5, 6):
        c = PHI ** n
        if c > 0.5 and c < 10:
            log_c = np.log10(c)
            err = abs(log_c - gap)
            if err < 0.05:
                candidates.append({
                    'formula': f'φ^{n}',
                    'value': c,
                    'log10': log_c,
                    'error_log10': err,
                    'category': 'phi_power',
                })

    # φ² (PAC depth bound!) based corrections
    pac_bound = PHI ** 2  # = 2.618
    log_pac = np.log10(pac_bound)
    err_pac = abs(log_pac - gap)
    candidates.append({
        'formula': f'φ² (PAC depth bound) = {pac_bound:.4f}',
        'value': pac_bound,
        'log10': log_pac,
        'error_log10': err_pac,
        'category': 'pac_bound',
    })

    # π/φ, φ/π, etc.
    for label, c in [
        ('π/φ', np.pi / PHI),
        ('φ/π', PHI / np.pi),
        ('2φ', 2 * PHI),
        ('φ+1', PHI + 1),  # = φ² = pac_bound
        ('√5', np.sqrt(5)),
        ('φ·√5', PHI * np.sqrt(5)),
        ('e/φ', np.e / PHI),
        ('φ²/π', PHI**2 / np.pi),
        ('π²/F₇', np.pi**2 / F7),
    ]:
        if c > 0.5 and c < 10:
            log_c = np.log10(c)
            err = abs(log_c - gap)
            candidates.append({
                'formula': label,
                'value': c,
                'log10': log_c,
                'error_log10': err,
                'category': 'named',
            })

    # Combine: 1 + F_a/(n·π·F_b²) style (like α correction)
    for ka in range(2, 15):
        fa = fib(ka)
        for kb in range(2, 15):
            fb = fib(kb)
            for n in [1, 2, 4]:
                denom = n * np.pi * fb**2
                corr = 1 + fa / denom
                if corr > 0.5 and corr < 10:
                    log_c = np.log10(corr)
                    err = abs(log_c - gap)
                    if err < 0.01:
                        candidates.append({
                            'formula': f'1 + F_{ka}/({n}πF_{kb}²) = 1 + {fa}/({n}π·{fb}²)',
                            'value': corr,
                            'log10': log_c,
                            'error_log10': err,
                            'category': 'alpha_style',
                        })

    # Sort by error
    candidates.sort(key=lambda c: c['error_log10'])

    print(f"  Correction needed: C = {factor:.6f} (log₁₀ = {gap:.6f})")
    print(f"  Searched {len(candidates)} candidate expressions\n")
    print(f"  Top 15 matches:")
    for i, c in enumerate(candidates[:15]):
        print(f"    {i+1:2d}. {c['formula']:45s}  = {c['value']:.6f}  "
              f"log₁₀ = {c['log10']:.6f}  Δ = {c['error_log10']:.6f}")

    best = candidates[0] if candidates else None
    if best:
        print(f"\n  Best correction: {best['formula']}")
        print(f"  Value: {best['value']:.6f}")
        print(f"  Residual: {best['error_log10']:.6f} in log₁₀")
        print(f"  Corrected hierarchy: F₁₈₃ × {best['value']:.4f} "
              f"= 10^{log10_f183 + best['log10']:.4f}")
        print(f"  Target: 10^{log10_target:.4f}")

    t2_pass = best and best['error_log10'] < 0.005

    results['tests']['correction_search'] = {
        'correction_needed': factor,
        'correction_log10': gap,
        'n_candidates_searched': len(candidates),
        'top_10': candidates[:10],
        'best_match': best,
        'best_residual': best['error_log10'] if best else None,
        'status': 'PASS' if t2_pass else 'FAIL',
    }

    # =================================================================
    # TEST 3: Uniqueness of 183
    #
    # For n = F_1 through F_15, compute n² + n + 1 and check F_{n²+n+1}.
    # How many give log₁₀ within 0.5 of 38.23?
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 3: Uniqueness of 183 Among Fibonacci-Derived Depths")
    print("=" * 70 + "\n")

    depth_candidates = []
    for k in range(1, 16):
        n = fib(k)
        depth = n**2 + n + 1
        if depth > 5000:
            continue  # Beyond reasonable Fibonacci computation
        log_fib = fib_log10(depth)
        delta = abs(log_fib - log10_target)
        match_05 = delta < 0.5
        match_01 = delta < 0.1

        depth_candidates.append({
            'k': k, 'F_k': n,
            'depth': depth,
            'log10_Fdepth': log_fib,
            'delta': delta,
            'within_05': match_05,
            'within_01': match_01,
        })

        indicator = " ← TARGET" if depth == 183 else ""
        print(f"  F_{k:2d} = {n:5d} → depth = {depth:10d} → "
              f"log₁₀(F_depth) = {log_fib:10.4f}  Δ = {delta:8.4f}  "
              f"{'✓' if match_05 else ' '}{indicator}")

    n_within_05 = sum(1 for d in depth_candidates if d['within_05'])
    n_within_01 = sum(1 for d in depth_candidates if d['within_01'])
    print(f"\n  Within 0.5 of target: {n_within_05}/{len(depth_candidates)}")
    print(f"  Within 0.1 of target: {n_within_01}/{len(depth_candidates)}")

    # Is 183 the closest?
    sorted_by_delta = sorted(depth_candidates, key=lambda d: d['delta'])
    print(f"\n  Closest: depth={sorted_by_delta[0]['depth']} "
          f"(F_{sorted_by_delta[0]['k']}² + F_{sorted_by_delta[0]['k']} + 1)")
    f183_rank = next(i+1 for i, d in enumerate(sorted_by_delta) if d['depth'] == 183)
    print(f"  183 rank: #{f183_rank}")

    t3_unique = f183_rank <= 2 and n_within_01 <= 2

    results['tests']['uniqueness_fibonacci'] = {
        'candidates': depth_candidates,
        'n_within_05': n_within_05,
        'n_within_01': n_within_01,
        'f183_rank': f183_rank,
        'closest_depth': sorted_by_delta[0]['depth'],
        'status': 'PASS' if t3_unique else 'FAIL',
    }

    # =================================================================
    # TEST 4: Random Formula Test
    #
    # Sample 10,000 random formulas of the form a² + a + 1 for
    # a ∈ [1, 100]. What fraction give |Δlog₁₀| < 0.5?
    # Also test a² + b for random a, b.
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 4: Random Formula Competition")
    print("=" * 70 + "\n")

    rng = np.random.RandomState(42)

    # Test 4a: a² + a + 1 for a = 1..200
    cyclotomic_hits_05 = 0
    cyclotomic_hits_01 = 0
    cyclotomic_best_a = None
    cyclotomic_best_delta = float('inf')

    for a in range(1, 201):
        depth = a**2 + a + 1
        if depth > 5000:
            continue
        log_f = fib_log10(depth)
        delta = abs(log_f - log10_target)
        if delta < 0.5:
            cyclotomic_hits_05 += 1
        if delta < 0.1:
            cyclotomic_hits_01 += 1
        if delta < cyclotomic_best_delta:
            cyclotomic_best_delta = delta
            cyclotomic_best_a = a

    print(f"  Test 4a: F_(a²+a+1) for a=1..200")
    print(f"  Within 0.5 of target: {cyclotomic_hits_05}/200 "
          f"({cyclotomic_hits_05/200*100:.1f}%)")
    print(f"  Within 0.1 of target: {cyclotomic_hits_01}/200 "
          f"({cyclotomic_hits_01/200*100:.1f}%)")
    print(f"  Best: a={cyclotomic_best_a} → "
          f"depth={cyclotomic_best_a**2+cyclotomic_best_a+1} "
          f"(Δ={cyclotomic_best_delta:.4f})")
    is_fib_best = (cyclotomic_best_a == F7)
    print(f"  Best is F₇=13: {is_fib_best}")

    # Test 4b: Random depths from various formulas
    n_random = 10000
    random_hits_05 = 0
    random_hits_01 = 0

    for _ in range(n_random):
        # Random formula: pick two small integers and combine
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        op = rng.choice(['a*b+1', 'a**2+b', 'a*b', 'a+b', 'a**2+a+1'])
        if op == 'a*b+1':
            depth = a * b + 1
        elif op == 'a**2+b':
            depth = a**2 + b
        elif op == 'a*b':
            depth = a * b
        elif op == 'a+b':
            depth = a + b
        else:
            depth = a**2 + a + 1

        if depth < 2 or depth > 3000:
            continue
        log_f = fib_log10(depth)
        delta = abs(log_f - log10_target)
        if delta < 0.5:
            random_hits_05 += 1
        if delta < 0.1:
            random_hits_01 += 1

    frac_05 = random_hits_05 / n_random
    frac_01 = random_hits_01 / n_random

    print(f"\n  Test 4b: {n_random} random formulas (a,b ∈ [1,50])")
    print(f"  Within 0.5 of target: {random_hits_05}/{n_random} "
          f"({frac_05*100:.1f}%)")
    print(f"  Within 0.1 of target: {random_hits_01}/{n_random} "
          f"({frac_01*100:.1f}%)")

    # How special is 183? The log₁₀(F_n) grows linearly with n.
    # For n between 1 and 200: log₁₀(F_n) ranges from 0 to ~42.
    # Target window of 0.5 out of 42 = ~1.2%.
    # So random depth in [1,200] has ~1.2% chance of matching.
    expected_frac = 1.0 / (200 * np.log10(PHI))  # Approximate
    print(f"\n  Expected random match (0.5 window): ~{expected_frac*100:.1f}%")
    print(f"  Observed: {frac_05*100:.1f}%")

    t4_pass = frac_05 < 0.10  # Less than 10% of random formulas match

    results['tests']['random_formula'] = {
        'cyclotomic_hits_05': cyclotomic_hits_05,
        'cyclotomic_hits_01': cyclotomic_hits_01,
        'cyclotomic_best_a': int(cyclotomic_best_a) if cyclotomic_best_a else None,
        'is_fibonacci_best': is_fib_best,
        'random_hits_05': random_hits_05,
        'random_frac_05': frac_05,
        'random_hits_01': random_hits_01,
        'random_frac_01': frac_01,
        'status': 'PASS' if t4_pass else 'FAIL',
    }

    # =================================================================
    # TEST 5: PAC Depth Bound Connection
    #
    # exp_22 showed PAC bound = φ² = 2.618.
    # Does the correction factor relate to φ²?
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 5: PAC Depth Bound Connection")
    print("=" * 70 + "\n")

    pac_bound = PHI ** 2  # 2.618...

    # Is the correction factor related to φ²?
    print(f"  Correction factor needed: {factor:.6f}")
    print(f"  φ² = {pac_bound:.6f}")
    print()

    # Check various relationships
    relationships = [
        ('C = φ²', pac_bound),
        ('C = φ²/π', pac_bound / np.pi),
        ('C = √(φ²)', np.sqrt(pac_bound)),
        ('C = ln(φ²)', np.log(pac_bound)),
        ('C = φ²·ln(φ)', pac_bound * LN_PHI),
        ('C = φ²/e', pac_bound / np.e),
        ('C = (φ²−2)', pac_bound - 2),
        ('C = 1 + 1/φ²', 1 + 1/pac_bound),
        ('C = φ²·(1−1/π)', pac_bound * (1 - 1/np.pi)),
        ('C = φ³/π', PHI**3 / np.pi),
        ('C = F₁₀/(4πF₇)', F10 / (4 * np.pi * F7)),
    ]

    print(f"  {'Relationship':35s}  {'Value':>10s}  {'log₁₀':>10s}  {'Δ':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, val in relationships:
        if val > 0:
            log_val = np.log10(val)
            delta = abs(log_val - gap)
            print(f"  {label:35s}  {val:10.6f}  {log_val:10.6f}  {delta:10.6f}")

    # Find best φ²-related correction
    best_pac = min(relationships, key=lambda r: abs(np.log10(r[1]) - gap) if r[1] > 0 else 999)
    best_pac_delta = abs(np.log10(best_pac[1]) - gap) if best_pac[1] > 0 else 999

    print(f"\n  Best φ²-based correction: {best_pac[0]}")
    print(f"  Residual: {best_pac_delta:.6f} in log₁₀")

    t5_pass = best_pac_delta < 0.01

    results['tests']['pac_bound_connection'] = {
        'pac_bound': pac_bound,
        'correction_factor': factor,
        'relationships': [(r[0], float(r[1]), abs(np.log10(r[1]) - gap) if r[1] > 0 else None)
                          for r in relationships],
        'best_pac_correction': best_pac[0],
        'best_pac_residual': best_pac_delta,
        'status': 'PASS' if t5_pass else 'FAIL',
    }

    # =================================================================
    # SYNTHESIS
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    t1_s = 'INFO'
    t2_s = results['tests']['correction_search']['status']
    t3_s = results['tests']['uniqueness_fibonacci']['status']
    t4_s = results['tests']['random_formula']['status']
    t5_s = results['tests']['pac_bound_connection']['status']

    tests_pass = sum(1 for s in [t2_s, t3_s, t4_s, t5_s] if s == 'PASS')

    print(f"\n  Test 1 (precise gap):          {t1_s}")
    print(f"  Test 2 (correction search):    {t2_s}")
    print(f"  Test 3 (uniqueness):           {t3_s}")
    print(f"  Test 4 (random competition):   {t4_s}")
    print(f"  Test 5 (PAC bound connection): {t5_s}")
    print(f"\n  Result: {tests_pass}/4 PASS (excluding info test)")

    print(f"\n  Gap summary:")
    print(f"    Raw: log₁₀(F₁₈₃) = {log10_f183:.6f}")
    print(f"    Target: log₁₀(hierarchy) = {log10_target:.6f}")
    print(f"    Gap: {gap:.6f} (factor {factor:.4f})")
    if best:
        corrected = log10_f183 + np.log10(best['value'])
        print(f"    Corrected: {corrected:.6f} (residual: "
              f"{abs(corrected - log10_target):.6f})")

    # Falsification
    results['falsification'] = {
        'test_id': 'F21',
        'hypothesis': (
            'F₁₈₃ predicts the EM/gravity hierarchy, '
            'with a Fibonacci correction closing the ~0.33 log₁₀ gap.'
        ),
        'chain': [
            f'Test 2 (correction): {t2_s} — Fibonacci correction search',
            f'Test 3 (uniqueness): {t3_s} — 183 among Fibonacci depths',
            f'Test 4 (random): {t4_s} — random formula competition',
            f'Test 5 (PAC bound): {t5_s} — φ² connection',
        ],
        'n_pass': f'{tests_pass}/4',
        'falsified': tests_pass < 1,
        'honest_assessment': (
            'The raw F₁₈₃ comparison is within 0.33 in log₁₀ of the measured '
            'hierarchy — about a factor of 2. This is remarkably close for an '
            'order-of-magnitude structural observation. Whether 183 is uniquely '
            'good among small integers determines if this is structural or '
            'numerological. A clean correction term would strengthen the claim.'
        ),
    }

    save_results(results, 'exp_23_f183_correction')


if __name__ == '__main__':
    main()
