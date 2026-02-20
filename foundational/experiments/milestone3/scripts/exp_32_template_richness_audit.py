#!/usr/bin/env python3
"""
Experiment 32 — Template Richness Audit
========================================

PACSeries Paper 4, §12 (strengthens or weakens joint significance claim)

KEY QUESTION: Given a 4-factor Fibonacci template, how many *arbitrary*
dimensionless physical constants can be matched below various precision
thresholds? If most PDG constants CAN be matched, individual formula
matches are cheap and only joint constraints matter. If most CANNOT,
individual matches carry more weight.

Method:
  1. Assemble ~50 dimensionless ratios from PDG / CODATA
  2. For each, search products/ratios of F₃–F₁₃ with φ, π, Ξ
     using the same template class as exp_09 (look-elsewhere for α)
  3. Report: hit rate at 100 ppm, 1%, 5% thresholds
  4. Compare hit rate for PAC-claimed constants vs unclaimed constants

This directly addresses the referee concern: "how many PDG constants
COULD you match with a 4-factor Fibonacci template?"
"""

import json
import os
import math
import itertools
from datetime import datetime

# ─── Fibonacci numbers ───────────────────────────────────────────
def fib(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

FIBS = {f'F{i}': fib(i) for i in range(2, 14)}  # F2=1 .. F13=233
FIB_VALUES = [fib(i) for i in range(2, 14)]

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi
GAMMA_E = 0.5772156649015329  # Euler-Mascheroni
XI = GAMMA_E + math.log(PHI)  # Ξ = γ + ln(φ) ≈ 1.0584

# Extended constant pool: Fibonacci numbers + transcendentals
CONSTANTS = {}
for i in range(2, 14):
    CONSTANTS[f'F{i}'] = fib(i)
CONSTANTS['phi'] = PHI
CONSTANTS['1/phi'] = 1/PHI
CONSTANTS['pi'] = PI
CONSTANTS['Xi'] = XI
CONSTANTS['gamma_E'] = GAMMA_E
CONSTANTS['ln_phi'] = math.log(PHI)

CONST_VALUES = list(CONSTANTS.values())
CONST_NAMES = list(CONSTANTS.keys())

# ─── PDG / CODATA dimensionless constants ────────────────────────
# These are dimensionless ratios or pure numbers from particle physics,
# atomic physics, and cosmology. Each has a well-measured value.
# We include both PAC-claimed and unclaimed constants.

PDG_CONSTANTS = {
    # === PAC-CLAIMED (these appear in Paper 4) ===
    'alpha_em': {'value': 7.2973525693e-3, 'claimed': True, 'source': 'CODATA 2022'},
    'sin2_theta_W': {'value': 0.23121, 'claimed': True, 'source': 'PDG 2022 on-shell'},
    'alpha_s_MZ': {'value': 0.1179, 'claimed': True, 'source': 'PDG 2022'},
    'koide_Q': {'value': 0.666661, 'claimed': True, 'source': 'PDG masses'},
    'mu_over_e': {'value': 206.7682830, 'claimed': True, 'source': 'CODATA 2022'},
    'tau_over_e': {'value': 3477.23, 'claimed': True, 'source': 'PDG 2022'},
    'proton_over_e': {'value': 1836.15267, 'claimed': True, 'source': 'CODATA 2022'},

    # === UNCLAIMED (these do NOT appear in Paper 4 formulas) ===
    # Lepton mass ratios (not claimed)
    'tau_over_mu': {'value': 16.8170, 'claimed': False, 'source': 'PDG 2022'},

    # Quark mass ratios (MS-bar at 2 GeV)
    'ms_over_md': {'value': 18.9, 'claimed': False, 'source': 'PDG 2022 (17-22 range)'},
    'mc_over_ms': {'value': 11.76, 'claimed': False, 'source': 'PDG 2022'},
    'mb_over_mc': {'value': 3.40, 'claimed': False, 'source': 'PDG 2022'},
    'mt_over_mb': {'value': 41.1, 'claimed': False, 'source': 'PDG 2022'},
    'mu_over_md': {'value': 0.474, 'claimed': False, 'source': 'PDG 2022 (0.38-0.58)'},

    # Hadron mass ratios
    'neutron_over_proton': {'value': 1.00137842, 'claimed': False, 'source': 'CODATA 2022'},
    'pion_charged_over_e': {'value': 273.132, 'claimed': False, 'source': 'PDG 2022'},
    'pion_neutral_over_e': {'value': 264.143, 'claimed': False, 'source': 'PDG 2022'},
    'kaon_over_pion': {'value': 3.540, 'claimed': False, 'source': 'PDG 2022'},
    'W_over_Z': {'value': 0.8768, 'claimed': False, 'source': 'PDG 2022'},
    'higgs_over_W': {'value': 1.557, 'claimed': False, 'source': 'PDG 2022'},
    'higgs_over_Z': {'value': 1.370, 'claimed': False, 'source': 'PDG 2022'},
    'Z_over_proton': {'value': 97.16, 'claimed': False, 'source': 'PDG 2022'},
    'W_over_proton': {'value': 85.67, 'claimed': False, 'source': 'PDG 2022'},

    # CKM matrix elements
    'Vus': {'value': 0.2243, 'claimed': False, 'source': 'PDG 2022'},
    'Vcb': {'value': 0.0422, 'claimed': False, 'source': 'PDG 2022'},
    'Vub': {'value': 0.00394, 'claimed': False, 'source': 'PDG 2022'},

    # PMNS mixing (sin² values)
    'sin2_theta12_PMNS': {'value': 0.307, 'claimed': False, 'source': 'PDG 2022'},
    'sin2_theta23_PMNS': {'value': 0.546, 'claimed': False, 'source': 'PDG 2022'},
    'sin2_theta13_PMNS': {'value': 0.0220, 'claimed': False, 'source': 'PDG 2022'},

    # Neutrino mass squared differences (eV²) — ratios
    'delta_m2_atm_over_sol': {'value': 32.5, 'claimed': False, 'source': 'PDG 2022'},

    # Cosmological parameters
    'omega_baryon': {'value': 0.0493, 'claimed': False, 'source': 'Planck 2018'},
    'omega_CDM': {'value': 0.265, 'claimed': False, 'source': 'Planck 2018'},
    'omega_lambda': {'value': 0.685, 'claimed': False, 'source': 'Planck 2018'},
    'baryon_to_photon': {'value': 6.12e-10, 'claimed': False, 'source': 'Planck 2018'},
    'sigma8': {'value': 0.811, 'claimed': False, 'source': 'Planck 2018'},
    'spectral_index_ns': {'value': 0.965, 'claimed': False, 'source': 'Planck 2018'},
    'hubble_h': {'value': 0.674, 'claimed': False, 'source': 'Planck 2018'},

    # QCD scale / proton mass
    'lambda_QCD_over_mp': {'value': 0.214, 'claimed': False, 'source': 'Approx. Λ_QCD ≈ 200 MeV'},

    # Electromagnetic / atomic
    'rydberg_times_a0_over_c': {'value': 7.297e-3, 'claimed': False, 'source': 'Equals α — cross-check'},
    'g_minus_2_electron': {'value': 1.15965218128e-3, 'claimed': False, 'source': 'CODATA 2022'},
    'g_minus_2_muon': {'value': 1.16592061e-3, 'claimed': False, 'source': 'FNAL 2023'},

    # Dimensionless gravity
    'alpha_gravity_proton': {'value': 5.91e-39, 'claimed': False, 'source': 'G·mp²/(ℏc)'},

    # Electroweak
    'GF_natural': {'value': 1.1664e-5, 'claimed': False, 'source': 'Fermi constant in GeV⁻²'},

    # Strong CP (upper bound)
    'theta_QCD_upper': {'value': 1e-10, 'claimed': False, 'source': 'Upper bound'},

    # Ratios involving electron mass
    'electron_over_proton': {'value': 5.446e-4, 'claimed': False, 'source': 'CODATA 2022'},

    # Nuclear binding
    'deuteron_BE_over_mp': {'value': 1.186e-3, 'claimed': False, 'source': '2.224 MeV / 938.3 MeV'},

    # Pion decay constant ratio
    'fpi_over_mpi': {'value': 0.939, 'claimed': False, 'source': '130.2 / 139.6 MeV'},

    # Additional particle ratios
    'eta_over_pion': {'value': 3.929, 'claimed': False, 'source': 'PDG 2022'},
    'rho_over_pion': {'value': 5.535, 'claimed': False, 'source': 'PDG 2022'},
    'J_psi_over_proton': {'value': 3.296, 'claimed': False, 'source': 'PDG 2022'},
    'upsilon_over_proton': {'value': 10.078, 'claimed': False, 'source': 'PDG 2022'},
}


# ─── Template search engine ─────────────────────────────────────

def generate_formulas_2factor(max_power=2):
    """
    Generate all 2-factor formulas: c₁^a × c₂^b
    where c₁, c₂ drawn from CONST_VALUES, a,b ∈ {-2,-1,1,2}
    """
    powers = [-2, -1, 1, 2]
    formulas = set()
    for i, c1 in enumerate(CONST_VALUES):
        for j, c2 in enumerate(CONST_VALUES):
            if j <= i:
                continue
            for a in powers:
                for b in powers:
                    try:
                        val = (c1 ** a) * (c2 ** b)
                        if val > 0 and math.isfinite(val):
                            formulas.add(val)
                    except (OverflowError, ZeroDivisionError):
                        pass
    return formulas


def generate_formulas_3factor():
    """
    Generate 3-factor formulas: c₁^a × c₂^b × c₃^c
    with powers in {-2, -1, 1, 2}.
    To manage combinatorial explosion, use a representative subset.
    """
    powers = [-1, 1, 2]  # Reduced power set for 3-factor
    formulas = set()
    n = len(CONST_VALUES)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for a in powers:
                    for b in powers:
                        for c in powers:
                            try:
                                val = (CONST_VALUES[i] ** a) * \
                                      (CONST_VALUES[j] ** b) * \
                                      (CONST_VALUES[k] ** c)
                                if val > 0 and math.isfinite(val):
                                    formulas.add(val)
                            except (OverflowError, ZeroDivisionError):
                                pass
    return formulas


def generate_formulas_4factor_additive():
    """
    Generate 4-factor formulas with additive corrections:
    c₁ × c₂^a × (1 + c₃/c₄) and c₁ × c₂^a × (1 - c₃/(c₄·c₅))
    This is the template class used for mass ratios in Paper 4.
    """
    formulas = set()
    # Use Fibonacci values only for outer products (matching Paper 4 style)
    fibs = FIB_VALUES
    for f1 in fibs:
        for f2 in fibs:
            for f3 in fibs:
                for f4 in fibs:
                    if f4 == 0:
                        continue
                    # Template: f1 × f2 × (1 + f3/f4)
                    try:
                        val = f1 * f2 * (1 + f3/f4)
                        if val > 0 and math.isfinite(val):
                            formulas.add(val)
                    except (OverflowError, ZeroDivisionError):
                        pass
                    # Template: f1 × f2^2 × (1 + 1/f3)
                    try:
                        val = f1 * f2**2 * (1 + 1/f3)
                        if val > 0 and math.isfinite(val):
                            formulas.add(val)
                    except (OverflowError, ZeroDivisionError):
                        pass

    # Also include φ and π in some slots
    for f1 in fibs:
        for f2 in fibs:
            for t in [PHI, PI, XI, 1/PHI]:
                try:
                    val = f1 * f2 * t
                    if val > 0 and math.isfinite(val):
                        formulas.add(val)
                    val = f1 * t / f2
                    if val > 0 and math.isfinite(val):
                        formulas.add(val)
                    val = f1 / (f2 * t)
                    if val > 0 and math.isfinite(val):
                        formulas.add(val)
                except (OverflowError, ZeroDivisionError):
                    pass

    return formulas


def best_match(target, formula_values):
    """Find the best-matching formula value for a target."""
    if target == 0 or not math.isfinite(target):
        return None, float('inf')
    best_val = None
    best_err = float('inf')
    for v in formula_values:
        if v == 0:
            continue
        err = abs(v - target) / abs(target)
        if err < best_err:
            best_err = err
            best_val = v
    return best_val, best_err


def main():
    print("=" * 70)
    print("TEMPLATE RICHNESS AUDIT")
    print("How many PDG constants can a Fibonacci template match?")
    print("=" * 70)
    print()

    # Generate formula pools
    print("Generating formula pools...")
    pool_2f = generate_formulas_2factor()
    print(f"  2-factor pool: {len(pool_2f):,} unique values")
    pool_3f = generate_formulas_3factor()
    print(f"  3-factor pool: {len(pool_3f):,} unique values")
    pool_4f = generate_formulas_4factor_additive()
    print(f"  4-factor additive pool: {len(pool_4f):,} unique values")

    # Combined pool
    all_formulas = pool_2f | pool_3f | pool_4f
    print(f"  Combined pool: {len(all_formulas):,} unique values")
    print()

    all_formula_list = sorted(all_formulas)

    # ─── Test each PDG constant ──────────────────────────────────
    thresholds = [100e-6, 0.01, 0.05]  # 100 ppm, 1%, 5%
    threshold_labels = ['100 ppm', '1%', '5%']

    results = {}
    claimed_results = {}
    unclaimed_results = {}

    print(f"{'Constant':<30s}  {'Value':>12s}  {'Best Match':>12s}  {'Error':>10s}  {'Claimed':>7s}")
    print("-" * 85)

    for name, info in sorted(PDG_CONSTANTS.items()):
        target = info['value']
        claimed = info['claimed']

        bval, berr = best_match(target, all_formula_list)

        results[name] = {
            'value': target,
            'claimed': claimed,
            'best_match': bval,
            'best_error': berr,
            'source': info['source'],
        }

        bucket = claimed_results if claimed else unclaimed_results
        bucket[name] = berr

        flag = '✓' if claimed else ' '
        err_str = f"{berr*100:.4f}%" if berr < 1 else f"{berr:.2e}"
        print(f"  {name:<28s}  {target:>12.6g}  {bval:>12.6g}  {err_str:>10s}  {flag:>7s}")

    print()

    # ─── Summary statistics ──────────────────────────────────────
    print("=" * 70)
    print("SUMMARY: Hit rates by threshold")
    print("=" * 70)
    print()

    for thresh, label in zip(thresholds, threshold_labels):
        n_claimed_hit = sum(1 for e in claimed_results.values() if e < thresh)
        n_claimed_total = len(claimed_results)
        n_unclaimed_hit = sum(1 for e in unclaimed_results.values() if e < thresh)
        n_unclaimed_total = len(unclaimed_results)
        n_all_hit = n_claimed_hit + n_unclaimed_hit
        n_all_total = n_claimed_total + n_unclaimed_total

        print(f"  Threshold: {label}")
        print(f"    Claimed constants:   {n_claimed_hit}/{n_claimed_total} "
              f"({100*n_claimed_hit/n_claimed_total:.0f}%)")
        print(f"    Unclaimed constants: {n_unclaimed_hit}/{n_unclaimed_total} "
              f"({100*n_unclaimed_hit/n_unclaimed_total:.0f}%)")
        print(f"    All constants:       {n_all_hit}/{n_all_total} "
              f"({100*n_all_hit/n_all_total:.0f}%)")
        print()

    # ─── Key finding ─────────────────────────────────────────────
    # Sort unclaimed by error to see which ones the template CAN match
    sorted_unclaimed = sorted(unclaimed_results.items(), key=lambda x: x[1])

    print("=" * 70)
    print("UNCLAIMED constants best-matched by Fibonacci template")
    print("(Top 10 by precision)")
    print("=" * 70)
    for name, err in sorted_unclaimed[:10]:
        print(f"  {name:<30s}  error = {err*100:.4f}%")
    print()

    print("=" * 70)
    print("UNCLAIMED constants WORST-matched by Fibonacci template")
    print("(Bottom 10)")
    print("=" * 70)
    for name, err in sorted_unclaimed[-10:]:
        err_str = f"{err*100:.4f}%" if err < 1 else f"{err:.2e}"
        print(f"  {name:<30s}  error = {err_str}")
    print()

    # ─── Discrimination metric ───────────────────────────────────
    import statistics
    claimed_errors = list(claimed_results.values())
    unclaimed_errors = list(unclaimed_results.values())

    claimed_median = statistics.median(claimed_errors)
    unclaimed_median = statistics.median(unclaimed_errors)

    print("=" * 70)
    print("DISCRIMINATION: Claimed vs Unclaimed")
    print("=" * 70)
    print(f"  Claimed median error:   {claimed_median*100:.4f}%")
    print(f"  Unclaimed median error: {unclaimed_median*100:.4f}%")
    print(f"  Ratio (unclaimed/claimed): {unclaimed_median/claimed_median:.1f}×")
    print()

    # ─── Save results ────────────────────────────────────────────
    output = {
        'experiment': 'exp_32_template_richness_audit',
        'description': 'Template richness audit — how many PDG constants match Fibonacci templates?',
        'paper': 'Paper 4',
        'section': '§12 (strengthens/weakens joint significance)',
        'timestamp': datetime.now().isoformat(),
        'milestone': 'milestone3',
        'pool_sizes': {
            '2_factor': len(pool_2f),
            '3_factor': len(pool_3f),
            '4_factor_additive': len(pool_4f),
            'combined': len(all_formulas),
        },
        'n_constants_tested': {
            'total': len(PDG_CONSTANTS),
            'claimed': len(claimed_results),
            'unclaimed': len(unclaimed_results),
        },
        'hit_rates': {},
        'discrimination': {
            'claimed_median_error': claimed_median,
            'unclaimed_median_error': unclaimed_median,
            'ratio': unclaimed_median / claimed_median,
        },
        'per_constant': results,
    }

    for thresh, label in zip(thresholds, threshold_labels):
        n_c = sum(1 for e in claimed_results.values() if e < thresh)
        n_u = sum(1 for e in unclaimed_results.values() if e < thresh)
        output['hit_rates'][label] = {
            'claimed': f"{n_c}/{len(claimed_results)}",
            'unclaimed': f"{n_u}/{len(unclaimed_results)}",
            'all': f"{n_c+n_u}/{len(PDG_CONSTANTS)}",
            'claimed_pct': round(100*n_c/len(claimed_results), 1),
            'unclaimed_pct': round(100*n_u/len(unclaimed_results), 1),
        }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_32_template_richness_audit_{ts}.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {path}")


if __name__ == '__main__':
    main()
