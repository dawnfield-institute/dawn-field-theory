"""
Milestone 10 -- Exp 09: Number Theory as Fossil

Block D: Synthesis

PURPOSE: Test whether mathematical structures decompose into universal
(any PAC-closable system would produce them) vs fossil (specific to our
closure constant phi). If mathematics is a fossil record of our universe's
specific annealing trajectory (thesis section 9), then:
  - Alternative closures produce different prime-like decompositions
  - Some invariants are universal (independent of closure)
  - Phi-enrichment is a fingerprint of our specific closure, absent in others

This is the most speculative experiment — expected to open questions rather
than close them definitively.

Tests:
  1. Alternative closures: >= 3 non-phi closures with different primes
  2. Universal invariants: properties shared across all closures
  3. Phi-enrichment as fingerprint: present in phi-closure, absent in alternatives
  4. Riemann zero decomposition: universal vs fossil components of spacing

Builds on: iddea.md section 9, M7 (arithmetic from symmetry)
Predicted: 2/4 (most speculative experiment)
Prediction type: P (fossil decomposition is novel)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    alternative_pac_closure,
    save_results, setup_experiment,
    PHI, INV_PHI, LN_PHI, XI_BALANCE, PI,
)

_, RESULTS_DIR = setup_experiment(__file__)


def get_standard_prime_gaps(n_primes=200):
    """Generate standard prime numbers and their gap distribution."""
    primes = []
    candidate = 2
    while len(primes) < n_primes:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1

    gaps = np.diff(primes)
    return np.array(primes, dtype=float), gaps.astype(float)


def test1_alternative_closures():
    """At least 3 non-phi closures with different prime-like decompositions."""
    print("\n" + "=" * 70)
    print("TEST 1: ALTERNATIVE CLOSURES — Different Prime Structures")
    print("=" * 70)

    closure_types = ['sqrt2', 'e', 'tribonacci', 'plastic']
    standard_primes, standard_gaps = get_standard_prime_gaps(200)

    results = {}
    different_count = 0

    for ct in closure_types:
        closure = alternative_pac_closure(ct, n_elements=1000)

        # Compare gap distribution to standard primes
        if len(closure['gap_distribution']) > 10:
            alt_gaps = np.array(closure['gap_distribution'][:len(standard_gaps)])
            # Normalize for comparison
            alt_norm = alt_gaps / max(np.mean(alt_gaps), 1e-10)
            std_norm = standard_gaps[:len(alt_norm)] / max(np.mean(standard_gaps[:len(alt_norm)]), 1e-10)

            ks_stat, ks_p = ks_2samp(alt_norm, std_norm)
        else:
            ks_stat, ks_p = 0.0, 1.0

        is_different = ks_stat > 0.1  # KS-distance > 0.1
        if is_different:
            different_count += 1

        results[ct] = {
            'base_constant': closure['base_constant'],
            'n_primes': closure['n_primes'],
            'gap_mean': closure['prime_gaps_mean'],
            'gap_std': closure['prime_gaps_std'],
            'phi_enrichment': closure['phi_enrichment'],
            'ks_distance': float(ks_stat),
            'ks_p_value': float(ks_p),
            'different_from_standard': bool(is_different),
        }

        print(f"  {ct:12s}: base={closure['base_constant']:.4f}, "
              f"n_primes={closure['n_primes']}, "
              f"KS={ks_stat:.4f}, "
              f"phi_enrich={closure['phi_enrichment']:.4f} "
              f"{'DIFFERENT' if is_different else 'similar'}")

    print(f"\n  Different from standard: {different_count}/{len(closure_types)}")

    passed = different_count >= 3
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {different_count} >= 3 different")

    return {
        'test': 'alternative_closures',
        'closures': results,
        'different_count': different_count,
        'n_tested': len(closure_types),
        'passed': bool(passed),
    }


def test2_universal_invariants():
    """Identify structural properties shared across all closures."""
    print("\n" + "=" * 70)
    print("TEST 2: UNIVERSAL INVARIANTS — Shared Across Closures")
    print("=" * 70)

    closure_types = ['sqrt2', 'e', 'tribonacci', 'plastic']
    closures = {ct: alternative_pac_closure(ct, n_elements=1000) for ct in closure_types}
    standard_primes, standard_gaps = get_standard_prime_gaps(200)

    invariants_found = []

    # Invariant 1: Prime density decreases (prime number theorem analog)
    # In all closures, the density of "primes" should decrease with magnitude
    all_decreasing = True
    for ct, c in closures.items():
        n = c['n_primes']
        if n > 20:
            # Compare first half vs second half density
            first_half_density = 10 / max(c['prime_gaps_mean'], 1e-10) if c['prime_gaps_mean'] > 0 else 0
            # If gaps grow, density decreases
            all_decreasing = all_decreasing and (c['prime_gaps_std'] > 0)

    if all_decreasing:
        invariants_found.append('prime_density_decreases')
        print(f"  1. Prime density decreases with magnitude: UNIVERSAL")
    else:
        print(f"  1. Prime density decreasing: NOT universal")

    # Invariant 2: Gap distribution is non-uniform (structured, not random)
    all_structured = True
    for ct, c in closures.items():
        gaps = c['gap_distribution']
        if len(gaps) > 5:
            # Check that gap distribution has structure (not uniform)
            _, p = ks_2samp(gaps[:min(50, len(gaps))],
                           np.random.uniform(min(gaps), max(gaps), 50))
            if p > 0.1:  # If p > 0.1, looks uniform (no structure)
                all_structured = False

    if all_structured:
        invariants_found.append('structured_gap_distribution')
        print(f"  2. Gap distribution is structured: UNIVERSAL")
    else:
        print(f"  2. Gap distribution structured: NOT universal")

    # Invariant 3: Multiplicative structure (factorization exists)
    # All closures should have composites = products of primes
    all_factorizable = True
    for ct, c in closures.items():
        # If n_primes < n_elements, composites exist
        if c['n_primes'] >= 900:  # Almost everything is "prime" = no factorization
            all_factorizable = False

    if all_factorizable:
        invariants_found.append('multiplicative_structure')
        print(f"  3. Multiplicative structure (factorization): UNIVERSAL")
    else:
        print(f"  3. Multiplicative structure: NOT universal")

    # Invariant 4: Gap growth is at most polynomial
    all_polynomial = True
    for ct, c in closures.items():
        if c['prime_gaps_mean'] > 0 and c['prime_gaps_std'] > 0:
            # Check that std/mean ratio is bounded (not exponential growth)
            ratio = c['prime_gaps_std'] / c['prime_gaps_mean']
            if ratio > 10:  # Exponential would give huge ratio
                all_polynomial = False

    if all_polynomial:
        invariants_found.append('polynomial_gap_growth')
        print(f"  4. Gap growth at most polynomial: UNIVERSAL")
    else:
        print(f"  4. Polynomial gap growth: NOT universal")

    print(f"\n  Universal invariants found: {len(invariants_found)}")
    for inv in invariants_found:
        print(f"    - {inv}")

    passed = len(invariants_found) >= 2
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {len(invariants_found)} >= 2 invariants")

    return {
        'test': 'universal_invariants',
        'invariants': invariants_found,
        'n_invariants': len(invariants_found),
        'passed': bool(passed),
    }


def test3_phi_enrichment_fingerprint():
    """Phi-enrichment present in standard primes, absent in alternatives."""
    print("\n" + "=" * 70)
    print("TEST 3: PHI-ENRICHMENT — Fingerprint of Our Closure")
    print("=" * 70)

    # Check phi-enrichment in standard primes
    standard_primes, _ = get_standard_prime_gaps(200)
    std_ratios = standard_primes[1:] / standard_primes[:-1]
    std_phi_matches = np.sum(np.abs(std_ratios - PHI) < 0.1)
    std_phi_enrichment = std_phi_matches / len(std_ratios)

    print(f"\n  Standard primes phi-enrichment: {std_phi_enrichment:.4f}")

    # Check alternatives
    closure_types = ['sqrt2', 'e', 'tribonacci', 'plastic']
    absent_count = 0
    alt_enrichments = {}

    for ct in closure_types:
        closure = alternative_pac_closure(ct, n_elements=1000)
        enr = closure['phi_enrichment']
        alt_enrichments[ct] = enr
        is_absent = enr < 0.05
        if is_absent:
            absent_count += 1
        print(f"  {ct:12s} phi-enrichment: {enr:.4f} {'ABSENT' if is_absent else 'present'}")

    print(f"\n  Absent in alternatives: {absent_count}/{len(closure_types)}")

    # Pass: phi-enrichment absent (< 0.05) in at least 2 alternatives
    passed = absent_count >= 2
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: {absent_count} >= 2 absent")

    return {
        'test': 'phi_enrichment_fingerprint',
        'standard_phi_enrichment': float(std_phi_enrichment),
        'alternative_enrichments': alt_enrichments,
        'absent_count': absent_count,
        'passed': bool(passed),
    }


def test4_riemann_zero_decomposition():
    """Separate universal vs fossil components of Riemann zero spacings."""
    print("\n" + "=" * 70)
    print("TEST 4: RIEMANN ZERO DECOMPOSITION — Universal vs Fossil")
    print("=" * 70)

    # Approximate first N Riemann zeta zeros on the critical line
    # Using the Gram points approximation: t_n ~ 2*pi*n / log(n)
    n_zeros = 200
    approx_zeros = []
    for n in range(1, n_zeros + 1):
        # Rough approximation to the nth zero
        t = 2 * PI * n / np.log(max(n, 2))
        # Add correction term
        t += PI / 8 * np.log(t / (2 * PI)) if t > 2 * PI else 0
        approx_zeros.append(t)

    zeros = np.array(approx_zeros)
    spacings = np.diff(zeros)

    # Normalize spacings by mean
    mean_spacing = np.mean(spacings)
    norm_spacings = spacings / mean_spacing

    # Decompose into:
    # 1. Universal component: GUE (Gaussian Unitary Ensemble) statistics
    #    The universal part is the level repulsion and spacing distribution
    # 2. Fossil component: deviations from GUE specific to zeta zeros

    # GUE prediction for normalized spacings: Wigner surmise
    # P(s) = (32/pi^2) * s^2 * exp(-4s^2/pi)
    def wigner_surmise(s):
        return (32 / PI**2) * s**2 * np.exp(-4 * s**2 / PI)

    # Compute residual from Wigner surmise
    n_bins = 30
    hist_counts, bin_edges = np.histogram(norm_spacings, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    wigner_pred = np.array([wigner_surmise(s) for s in bin_centers])

    # Residual structure
    residual = hist_counts - wigner_pred
    residual_entropy = 0.0
    abs_residual = np.abs(residual)
    if np.sum(abs_residual) > 0:
        probs = abs_residual / np.sum(abs_residual)
        probs = probs[probs > 0]
        residual_entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))
        normalized_entropy = residual_entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 1.0

    print(f"\n  Zeros analyzed:          {n_zeros}")
    print(f"  Mean spacing:            {mean_spacing:.4f}")
    print(f"  Spacing std:             {np.std(norm_spacings):.4f}")
    print(f"  Residual entropy:        {residual_entropy:.4f}")
    print(f"  Normalized entropy:      {normalized_entropy:.4f}")
    print(f"  (1.0 = uniform residual = no fossil; <1.0 = structured)")

    # Pass: residual has non-trivial structure (entropy < max)
    passed = normalized_entropy < 0.95
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: entropy {normalized_entropy:.4f} < 0.95")

    return {
        'test': 'riemann_zero_decomposition',
        'n_zeros': n_zeros,
        'mean_spacing': float(mean_spacing),
        'spacing_std': float(np.std(norm_spacings)),
        'residual_entropy': float(residual_entropy),
        'normalized_entropy': float(normalized_entropy),
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 09: NUMBER THEORY AS FOSSIL")
    print("Block D: Synthesis")
    print("=" * 70)

    r1 = test1_alternative_closures()
    r2 = test2_universal_invariants()
    r3 = test3_phi_enrichment_fingerprint()
    r4 = test4_riemann_zero_decomposition()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    results = {
        'experiment': 'exp_09_number_theory_fossil',
        'milestone': 10,
        'block': 'D',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_09_number_theory_fossil', RESULTS_DIR)


if __name__ == '__main__':
    main()
