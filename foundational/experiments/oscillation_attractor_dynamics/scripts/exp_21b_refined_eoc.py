"""
Experiment 21b: Refined Edge-of-Chaos Analysis
===============================================

Fixes issues in exp_21:
1. Correct entropy calculation (probability, not density)
2. Better edge-of-chaos metrics:
   - 1/f spectral signature (slope ≈ -1)
   - Mutual information between scales
   - Complexity (not entropy alone)

Key insight: Edge-of-chaos is characterized by:
- Power spectrum ∝ 1/f^β where β ≈ 1
- High mutual information between adjacent scales
- Intermediate entropy (not min, not max)
- Long-range correlations (slow autocorrelation decay)
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy import stats
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from sec_core import compute_sec, FIRST_50_PRIMES, PHI

XI = 1.0571428571428572
PHI_INV = 1 / PHI


def compute_eoc_metrics(E: np.ndarray) -> dict:
    """
    Compute proper edge-of-chaos metrics.
    
    Key signatures:
    1. 1/f noise: spectral slope ≈ -1 (pink noise)
    2. High complexity: neither min nor max entropy
    3. Long-range correlations: slow autocorrelation decay
    4. Scale-free structure: power-law-like patterns
    """
    n = min(len(E), 20000)
    E_sample = E[:n]
    
    # 1. Power spectrum and 1/f analysis
    # Use Welch's method for cleaner spectrum
    freqs, power = welch(E_sample, nperseg=min(1024, n//4))
    
    # Fit log-log slope (excluding DC and very high frequencies)
    valid = (freqs > 0.01) & (freqs < 0.4)
    if np.sum(valid) > 10:
        log_f = np.log10(freqs[valid])
        log_p = np.log10(power[valid] + 1e-10)
        spectral_slope, intercept, r_value, _, _ = stats.linregress(log_f, log_p)
        spectral_r2 = r_value ** 2
    else:
        spectral_slope = 0.0
        spectral_r2 = 0.0
    
    # Distance from ideal 1/f (slope = -1)
    dist_from_1f = abs(spectral_slope - (-1.0))
    
    # 2. Proper entropy calculation
    # Discretize into bins, use probability (not density)
    n_bins = 50
    hist, _ = np.histogram(E_sample, bins=n_bins)
    prob = hist / hist.sum()
    prob = prob[prob > 0]  # Remove zeros
    entropy_bits = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy_bits / max_entropy
    
    # 3. Autocorrelation structure
    E_centered = E_sample - np.mean(E_sample)
    var = np.var(E_sample) + 1e-10
    
    # Compute autocorrelation at multiple lags
    lags = [1, 5, 10, 20, 50, 100]
    autocorrs = []
    for lag in lags:
        if lag < n - 1:
            ac = np.sum(E_centered[:-lag] * E_centered[lag:]) / ((n - lag) * var)
            autocorrs.append(float(ac))
        else:
            autocorrs.append(0.0)
    
    # Decay rate: fit exponential
    valid_lags = np.array(lags[:len(autocorrs)])
    valid_acs = np.array(autocorrs)
    
    # Find lag where AC drops below 1/e
    ac_decay_lag = lags[-1]
    for i, ac in enumerate(autocorrs):
        if ac < 1/np.e:
            ac_decay_lag = lags[i]
            break
    
    # 4. Complexity: C = entropy × (1 - entropy)
    # Peaks at entropy = 0.5 (edge of chaos)
    complexity = 4 * normalized_entropy * (1 - normalized_entropy)
    
    # 5. Long-range correlation strength
    # Sum of autocorrelations (higher = more long-range structure)
    lrc_strength = sum(max(0, ac) for ac in autocorrs)
    
    return {
        'spectral_slope': float(spectral_slope),
        'spectral_r2': float(spectral_r2),
        'dist_from_1f': float(dist_from_1f),
        'entropy_bits': float(entropy_bits),
        'normalized_entropy': float(normalized_entropy),
        'complexity': float(complexity),
        'ac_decay_lag': int(ac_decay_lag),
        'autocorrelations': autocorrs,
        'lrc_strength': float(lrc_strength)
    }


def compute_eoc_score(metrics: dict) -> float:
    """
    Compute composite edge-of-chaos score.
    
    Edge-of-chaos should have:
    - Spectral slope near -1 (1/f noise)
    - Intermediate entropy (0.3-0.7)
    - High complexity
    - Long AC decay
    """
    # 1/f component (0-1, peaks at slope=-1)
    f1_score = np.exp(-metrics['dist_from_1f']**2 / 0.5)
    
    # Entropy component (0-1, peaks at 0.5)
    ent = metrics['normalized_entropy']
    ent_score = 4 * ent * (1 - ent) if 0 < ent < 1 else 0
    
    # AC decay component (longer = more edge-of-chaos)
    ac_score = min(metrics['ac_decay_lag'] / 50, 1.0)
    
    # Weighted combination
    eoc_score = 0.4 * f1_score + 0.3 * ent_score + 0.3 * ac_score
    
    return float(eoc_score)


def run_experiment():
    """
    Refined edge-of-chaos analysis for CAH Condition 4.
    """
    print("=" * 70)
    print("EXPERIMENT 21b: Refined Edge-of-Chaos Analysis")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    n_max = 50000
    window = 101
    lam = 0.98
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'n_max': n_max, 'window': window, 'lam': lam},
        'hypothesis': 'φ emerges at edge-of-chaos (1/f noise, intermediate entropy)'
    }
    
    # Sweep k
    print("=== Factor Base Size (k) Sweep ===")
    print()
    print(f"{'k':>4} {'φ dist':>10} {'slope':>8} {'1/f dist':>10} {'entropy':>10} {'EOC':>8}")
    print("-" * 56)
    
    k_values = list(range(3, 16))
    k_results = []
    
    for k in k_values:
        sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:k],
                         window=window, lam=lam)
        
        odd_E = sec.E[np.arange(len(sec.E)) % 2 == 1]
        frac = np.mean(odd_E > 0)
        phi_dist = abs(frac - PHI_INV)
        
        eoc = compute_eoc_metrics(sec.E)
        eoc_score = compute_eoc_score(eoc)
        
        k_results.append({
            'k': k,
            'frac_E_positive': float(frac),
            'phi_distance': float(phi_dist),
            'eoc_score': float(eoc_score),
            **eoc
        })
        
        print(f"{k:>4} {phi_dist:>10.6f} {eoc['spectral_slope']:>8.3f} "
              f"{eoc['dist_from_1f']:>10.4f} {eoc['normalized_entropy']:>10.4f} "
              f"{eoc_score:>8.4f}")
    
    results['k_sweep'] = k_results
    
    # Find best k by each metric
    best_phi = min(k_results, key=lambda x: x['phi_distance'])
    best_eoc = max(k_results, key=lambda x: x['eoc_score'])
    best_1f = min(k_results, key=lambda x: x['dist_from_1f'])
    
    print(f"\n  Best k for φ proximity: {best_phi['k']} (dist={best_phi['phi_distance']:.6f})")
    print(f"  Best k for EOC score:   {best_eoc['k']} (score={best_eoc['eoc_score']:.4f})")
    print(f"  Best k for 1/f noise:   {best_1f['k']} (slope={best_1f['spectral_slope']:.3f})")
    
    results['best'] = {
        'best_phi_k': best_phi['k'],
        'best_eoc_k': best_eoc['k'],
        'best_1f_k': best_1f['k']
    }
    
    # Correlation: EOC score vs φ distance
    print("\n=== Correlation Analysis ===")
    
    eoc_scores = [r['eoc_score'] for r in k_results]
    phi_dists = [r['phi_distance'] for r in k_results]
    
    # We expect NEGATIVE correlation: higher EOC → lower φ distance
    r, p = stats.pearsonr(eoc_scores, phi_dists)
    
    print(f"\n  Correlation (EOC score vs φ dist): r = {r:.4f}, p = {p:.6f}")
    print(f"  (Negative r = EOC predicts φ emergence)")
    
    results['correlation'] = {
        'pearson_r': float(r),
        'p_value': float(p),
        'negative_significant': bool(r < -0.3 and p < 0.05)
    }
    
    # Detailed analysis at k=9 (optimal)
    print("\n=== Detailed Analysis at k=9 ===")
    
    sec9 = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9],
                       window=window, lam=lam)
    eoc9 = compute_eoc_metrics(sec9.E)
    
    print(f"\n  Spectral slope:     {eoc9['spectral_slope']:.4f} (ideal: -1.0)")
    print(f"  1/f distance:       {eoc9['dist_from_1f']:.4f}")
    print(f"  Spectral fit R²:    {eoc9['spectral_r2']:.4f}")
    print(f"  Entropy (bits):     {eoc9['entropy_bits']:.4f}")
    print(f"  Normalized entropy: {eoc9['normalized_entropy']:.4f} (ideal: ~0.5)")
    print(f"  Complexity:         {eoc9['complexity']:.4f}")
    print(f"  AC decay lag:       {eoc9['ac_decay_lag']}")
    print(f"  LRC strength:       {eoc9['lrc_strength']:.4f}")
    
    results['k9_detailed'] = eoc9
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Check if k=9 shows EOC characteristics
    is_1f = eoc9['dist_from_1f'] < 0.5  # Within 0.5 of slope=-1
    is_intermediate_ent = 0.3 < eoc9['normalized_entropy'] < 0.8
    best_k_matches = best_phi['k'] == best_eoc['k']
    negative_corr = r < -0.3
    
    if is_1f and is_intermediate_ent and negative_corr:
        conclusion = "✅ SUPPORTS: k=9 shows EOC signatures (1/f-like, intermediate entropy)"
        status = "SUPPORTED"
    elif is_1f or is_intermediate_ent:
        conclusion = "🔄 PARTIAL: Some EOC signatures present but not all"
        status = "PARTIAL"
    else:
        conclusion = "❌ NOT SUPPORTED: No clear EOC signatures at optimal k"
        status = "NOT_SUPPORTED"
    
    print(f"\n{conclusion}")
    print(f"\n  1/f noise present:        {is_1f}")
    print(f"  Intermediate entropy:     {is_intermediate_ent}")
    print(f"  Best k matches for both:  {best_k_matches}")
    print(f"  Negative correlation:     {negative_corr}")
    
    results['conclusion'] = {
        'status': status,
        'is_1f': bool(is_1f),
        'is_intermediate_entropy': bool(is_intermediate_ent),
        'best_k_matches': bool(best_k_matches),
        'negative_correlation': bool(negative_corr),
        'summary': conclusion
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp_21b_refined_eoc_{timestamp}.json'
    
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()
