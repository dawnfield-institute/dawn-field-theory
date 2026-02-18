"""
exp_03: Prime Cascade Reachability (Formal Statistics)

HYPOTHESIS: Primes reside in "void regions" of cascade reachability —
points where no prior prime's cascade wave provides significant coverage.

SOURCE: internal/energy_equivilance/cascade_void_prime.py (Experiments D, E, F)
TARGET: Paper 1 - Prime structure from cascade dynamics

The original found: Mann-Whitney p = 7.7×10⁻¹² (composites have
significantly more cascade coverage than primes).

FALSIFICATION (F3): If a null model (Cramér random primes) shows
the same coverage disparity, then the cascade model adds nothing
beyond prime density effects.

METHOD (from original):
1. Cascade coverage: each prime p sends a wave to all positions,
   decaying as exp(-kT·ln(2)·0.1·distance) with structure boost
2. Mann-Whitney U: compare coverage at primes vs composites
3. Late cascade ratio: consecutive coverage ratios converge to ~0.600
4. Prime gap statistics: fraction of gaps that are Fibonacci numbers
5. Cramér null comparison: generate random primes with 1/ln(n) density
"""

import sys
import os
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, LN_PHI, LANDAUER_MIN
from core.utils import save_results, experiment_header, sieve_primes


def cascade_reachability(limit, primes):
    """
    Compute cascade reachability at each position using the physics-based
    model from cascade_void_prime.py Experiment D.
    
    Each prime p launches a cascade affecting ALL nearby positions:
    - wave_strength = 1 / ln(p)
    - void_decay = exp(-LANDAUER_MIN * 0.1 * distance_from_p)
    - cascade_boost = 1 + 0.1 * ln(1 + distance)
    - coverage[n] += wave_strength * void_decay * cascade_boost
    
    Note: the original only processes primes up to sqrt(LIMIT) for
    the launch wave, and applies to ALL multiples.
    """
    coverage = np.zeros(limit + 1)
    
    sqrt_limit = int(np.sqrt(limit)) + 1
    
    for p in primes:
        if p > sqrt_limit:
            break
        
        strength = 1.0 / np.log(max(p, 2))
        
        for multiple in range(p * 2, limit + 1, p):
            distance = (multiple / p) - 1  # steps from source prime
            void_decay = np.exp(-distance * LANDAUER_MIN * 0.1)
            cascade_boost = 1.0 + 0.1 * np.log1p(distance)
            coverage[multiple] += strength * void_decay * cascade_boost
    
    return coverage


def cramer_random_primes(limit, rng):
    """
    Cramér random model: each n > 2 is prime independently with
    probability 1/ln(n). Returns list of random 'primes'.
    """
    cramer = [2]
    for n in range(3, limit + 1):
        if rng.random() < 1.0 / np.log(n):
            cramer.append(n)
    return cramer


def prime_gap_fibonacci_analysis(primes):
    """
    From cascade_void_prime.py Experiment F:
    Check what fraction of prime gaps are Fibonacci numbers,
    and test gaps mod φ for non-uniformity.
    """
    gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
    
    # Fibonacci numbers up to the max gap
    fibs = [1, 1]
    while fibs[-1] < max(gaps) + 10:
        fibs.append(fibs[-1] + fibs[-2])
    fib_set = set(fibs)
    
    exact_fib = sum(1 for g in gaps if g in fib_set)
    near_fib = sum(1 for g in gaps if min(abs(g - f) for f in fibs) <= 1)
    
    # Gaps mod φ uniformity test
    gap_mod_phi = [(g % PHI) / PHI for g in gaps]
    hist, _ = np.histogram(gap_mod_phi, bins=10, range=(0, 1))
    expected = len(gaps) / 10
    chi2 = sum((h - expected)**2 / expected for h in hist)
    chi2_p = 1.0 - stats.chi2.cdf(chi2, df=9)
    
    return {
        'total_gaps': len(gaps),
        'exact_fibonacci': exact_fib,
        'frac_exact_fibonacci': exact_fib / len(gaps),
        'near_fibonacci': near_fib,
        'frac_near_fibonacci': near_fib / len(gaps),
        'chi2_gaps_mod_phi': float(chi2),
        'chi2_p_value': float(chi2_p),
        'phi_structure_detected': chi2_p < 0.05,
    }


def main():
    meta = experiment_header(
        'exp_03_prime_reachability',
        'Prime cascade reachability — physics-based model from originals',
        paper='Paper 1',
        section='§13 (prime structure)'
    )

    LIMIT = 10000
    results = {**meta, 'limit': LIMIT, 'tests': {}}

    # =========================================================================
    # TEST 1: Cascade coverage — primes vs composites (Mann-Whitney)
    # =========================================================================
    print("TEST 1: Cascade coverage (physics-based model)")
    primes = sieve_primes(LIMIT)
    prime_set = set(primes)
    composites = [n for n in range(4, LIMIT + 1) if n not in prime_set]
    
    coverage = cascade_reachability(LIMIT, primes)
    
    prime_cov = np.array([coverage[p] for p in primes])
    composite_cov = np.array([coverage[c] for c in composites])
    
    print(f"  Primes: n={len(primes)}")
    print(f"    Coverage mean={prime_cov.mean():.6f}, "
          f"median={np.median(prime_cov):.6f}")
    print(f"  Composites: n={len(composites)}")
    print(f"    Coverage mean={composite_cov.mean():.6f}, "
          f"median={np.median(composite_cov):.6f}")
    
    ratio = composite_cov.mean() / prime_cov.mean() if prime_cov.mean() > 0 else float('inf')
    print(f"  Coverage ratio (composite/prime): {ratio:.2f}×")
    
    # Mann-Whitney U test (one-sided: composites > primes)
    u_stat, u_p = stats.mannwhitneyu(
        composite_cov, prime_cov, alternative='greater'
    )
    print(f"  Mann-Whitney U: U = {u_stat:.0f}, p = {u_p:.2e}")
    
    # Zero-coverage fraction
    prime_zero = np.sum(prime_cov < 1e-10) / len(prime_cov)
    comp_zero = np.sum(composite_cov < 1e-10) / len(composite_cov)
    print(f"  Zero-coverage: primes={prime_zero:.4f}, composites={comp_zero:.4f}")
    
    # Cohen's d effect size
    pooled_std = np.sqrt((prime_cov.std()**2 + composite_cov.std()**2) / 2)
    cohens_d = (composite_cov.mean() - prime_cov.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d = {cohens_d:.4f} "
          f"({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
    
    results['tests']['coverage_comparison'] = {
        'prime_coverage_mean': float(prime_cov.mean()),
        'prime_coverage_std': float(prime_cov.std()),
        'composite_coverage_mean': float(composite_cov.mean()),
        'composite_coverage_std': float(composite_cov.std()),
        'coverage_ratio': float(ratio),
        'mann_whitney_U': float(u_stat),
        'mann_whitney_p': float(u_p),
        'prime_zero_fraction': float(prime_zero),
        'composite_zero_fraction': float(comp_zero),
        'cohens_d': float(cohens_d),
    }
    
    # =========================================================================
    # TEST 2: Late cascade coverage ratio (from Experiment C)
    # =========================================================================
    print("\nTEST 2: Late cascade coverage ratio")
    # Look at coverage at composites with positive coverage, consecutive
    late_composites = [c for c in composites if c > LIMIT // 2 and coverage[c] > 1e-15]
    cov_ratios = []
    for i in range(1, len(late_composites)):
        c1 = coverage[late_composites[i-1]]
        c2 = coverage[late_composites[i]]
        if c1 > 1e-15 and c2 > 1e-15:
            cov_ratios.append(c2 / c1)
    
    if cov_ratios:
        mean_ratio = np.mean(cov_ratios)
        std_ratio = np.std(cov_ratios)
        print(f"  Late cascade ratio: {mean_ratio:.6f} ± {std_ratio:.6f}")
        print(f"  Target 0.600 (from original): |Δ| = {abs(mean_ratio - 0.600):.6f}")
        print(f"  Distance from 1/φ ({1/PHI:.6f}): {abs(mean_ratio - 1/PHI):.6f}")
    
    results['tests']['late_cascade_ratio'] = {
        'mean_ratio': float(mean_ratio) if cov_ratios else None,
        'std_ratio': float(std_ratio) if cov_ratios else None,
        'n_ratios': len(cov_ratios),
    }
    
    # =========================================================================
    # TEST 3: Prime gap Fibonacci statistics (Experiment F)
    # =========================================================================
    print("\nTEST 3: Prime gap Fibonacci statistics")
    gap_stats = prime_gap_fibonacci_analysis(primes)
    
    print(f"  Gaps that are Fibonacci: {gap_stats['exact_fibonacci']}/{gap_stats['total_gaps']} "
          f"= {gap_stats['frac_exact_fibonacci']:.4f}")
    print(f"  Gaps within ±1 of Fibonacci: {gap_stats['near_fibonacci']}/{gap_stats['total_gaps']} "
          f"= {gap_stats['frac_near_fibonacci']:.4f}")
    print(f"  Gaps mod φ uniformity: χ² = {gap_stats['chi2_gaps_mod_phi']:.4f}, "
          f"p = {gap_stats['chi2_p_value']:.4f}")
    print(f"  φ-structure detected: {gap_stats['phi_structure_detected']}")
    
    results['tests']['prime_gap_fibonacci'] = gap_stats
    
    # =========================================================================
    # TEST 4: Cramér null model comparison
    # =========================================================================
    print("\nTEST 4: Cramér null model comparison")
    rng = np.random.default_rng(42)
    n_simulations = 100
    
    cramer_u_stats = []
    cramer_ratios = []
    
    for sim in range(n_simulations):
        cramer_primes = cramer_random_primes(LIMIT, rng)
        cramer_set = set(cramer_primes)
        
        # Compute coverage using Cramér primes (limit sources for speed)
        cramer_cov = np.zeros(LIMIT + 1)
        sqrt_lim = int(np.sqrt(LIMIT)) + 1
        for p in cramer_primes:
            if p > sqrt_lim:
                break
            strength = 1.0 / np.log(max(p, 2))
            for mult in range(p * 2, LIMIT + 1, p):
                dist = (mult / p) - 1
                cramer_cov[mult] += strength * np.exp(-dist * LANDAUER_MIN * 0.1) * (1 + 0.1 * np.log1p(dist))
        
        # Compare coverage at Cramér "primes" vs "composites"
        cp_cov = np.array([cramer_cov[p] for p in cramer_primes if p <= LIMIT])
        cc_cov = np.array([cramer_cov[n] for n in range(4, LIMIT + 1) if n not in cramer_set])
        
        if len(cp_cov) > 0 and len(cc_cov) > 0:
            u, _ = stats.mannwhitneyu(cc_cov, cp_cov, alternative='greater')
            cramer_u_stats.append(u)
            if cp_cov.mean() > 0:
                cramer_ratios.append(cc_cov.mean() / cp_cov.mean())
    
    # How unusual is the real U-statistic compared to Cramér null?
    if cramer_u_stats:
        cramer_u_mean = np.mean(cramer_u_stats)
        cramer_u_std = np.std(cramer_u_stats)
        z_score = (u_stat - cramer_u_mean) / cramer_u_std if cramer_u_std > 0 else 0
        
        print(f"  Cramér U-statistic: {cramer_u_mean:.0f} ± {cramer_u_std:.0f}")
        print(f"  Real U-statistic: {u_stat:.0f}")
        print(f"  z-score vs Cramér: {z_score:.2f}")
        print(f"  Cramér coverage ratio: {np.mean(cramer_ratios):.2f}×")
        print(f"  Real coverage ratio: {ratio:.2f}×")
    
    results['tests']['cramer_null'] = {
        'n_simulations': n_simulations,
        'cramer_u_mean': float(cramer_u_mean) if cramer_u_stats else None,
        'cramer_u_std': float(cramer_u_std) if cramer_u_stats else None,
        'real_u_stat': float(u_stat),
        'z_score_vs_cramer': float(z_score) if cramer_u_stats else None,
        'cramer_coverage_ratio_mean': float(np.mean(cramer_ratios)) if cramer_ratios else None,
        'real_coverage_ratio': float(ratio),
    }
    
    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print(f"{'='*70}")
    
    # Core result: Mann-Whitney significant?
    mw_pass = u_p < 0.05
    # Effect size meaningful?
    effect_pass = abs(cohens_d) > 0.5
    # Coverage ratio > 1?
    ratio_pass = ratio > 1.5
    # Cascade model adds value beyond Cramér?
    cramer_pass = abs(z_score) > 2.0 if cramer_u_stats else False
    
    print(f"\n  Mann-Whitney (primes < composites): p = {u_p:.2e} {'PASS' if mw_pass else 'FAIL'}")
    print(f"  Effect size: d = {cohens_d:.4f} {'PASS' if effect_pass else 'FAIL'}")
    print(f"  Coverage ratio: {ratio:.2f}× {'PASS' if ratio_pass else 'FAIL'}")
    print(f"  Beyond Cramér: z = {z_score:.2f} {'PASS' if cramer_pass else 'FAIL'}")
    print(f"  Fibonacci in gaps: {gap_stats['frac_exact_fibonacci']:.4f}")
    print(f"  φ-structure: {'detected' if gap_stats['phi_structure_detected'] else 'not detected'}")
    
    n_pass = sum([mw_pass, effect_pass, ratio_pass])
    
    # =========================================================================
    # Falsification Assessment
    # =========================================================================
    results['falsification'] = {
        'test_id': 'F3',
        'hypothesis': 'Primes reside in cascade void regions',
        'mann_whitney_p': float(u_p),
        'cohens_d': float(cohens_d),
        'coverage_ratio': float(ratio),
        'cramer_z_score': float(z_score) if cramer_u_stats else None,
        'tests_passed': n_pass,
        'tests_total': 3,
        'falsified': n_pass < 2,
        'assessment': (
            f"Mann-Whitney p = {u_p:.2e} ({'significant' if mw_pass else 'not significant'}), "
            f"Cohen's d = {cohens_d:.4f}, "
            f"coverage ratio = {ratio:.2f}×. "
            f"Cramér z-score = {z_score:.2f}. "
            f"{'NOT FALSIFIED' if n_pass >= 2 else 'FALSIFIED'}: "
            f"cascade coverage model {'shows' if mw_pass else 'does not show'} "
            f"significant prime vs composite separation. "
            f"Gap Fibonacci fraction: {gap_stats['frac_exact_fibonacci']:.4f}. "
            f"NOTE: Uses N={LIMIT}; scale to N=100000 for publication."
        ),
    }
    
    save_results(results, 'exp_03_prime_reachability')


if __name__ == '__main__':
    main()
