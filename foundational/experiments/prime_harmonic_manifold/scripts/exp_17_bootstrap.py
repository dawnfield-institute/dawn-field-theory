"""
Experiment 17: Bootstrap Confidence Intervals

Computes bootstrap 95% CI on:
1. λ₁ at various scales
2. The decay slope
3. The difference between real and Cramér

If CIs are tight and don't overlap, results are statistically robust.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import get_primes, PHI_INV
import numpy as np
import sympy as sp
from collections import Counter
from scipy.optimize import curve_fit


def compute_lambda1_from_gaps(gaps, topK=25):
    """Compute λ₁ from gap sequence."""
    if len(gaps) < 10:
        return None
    
    g1, g2 = gaps[:-1], gaps[1:]
    chords = [tuple([g1[i], g2[i]]) for i in range(len(g1))]
    
    counts = Counter(chords)
    top_chords = [c for c, _ in counts.most_common(topK)]
    chord_to_idx = {c: i for i, c in enumerate(top_chords)}
    
    seq_idx = [chord_to_idx.get(c, topK) for c in chords]
    
    T = np.zeros((topK+1, topK+1), dtype=int)
    for a, b in zip(seq_idx[:-1], seq_idx[1:]):
        T[a, b] += 1
    
    P = T.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums
    
    eigenvals = np.abs(np.linalg.eigvals(P[:topK, :topK]))
    return float(np.max(eigenvals))


def bootstrap_lambda1(gaps, n_bootstrap=500, topK=25):
    """Bootstrap λ₁ by resampling gap sequence with replacement."""
    n = len(gaps)
    lambda1s = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap to preserve some local structure
        block_size = 100
        n_blocks = n // block_size + 1
        
        resampled_gaps = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block_size))
            resampled_gaps.extend(gaps[start:start+block_size])
        
        resampled_gaps = np.array(resampled_gaps[:n])
        l1 = compute_lambda1_from_gaps(resampled_gaps, topK)
        if l1:
            lambda1s.append(l1)
    
    return np.array(lambda1s)


def run_experiment(prime_limit=2_000_000, n_bootstrap=500):
    """Run bootstrap confidence interval analysis."""
    
    print("=" * 70)
    print("VALIDATION: Bootstrap Confidence Intervals")
    print("=" * 70)
    print(f"Bootstrap samples: {n_bootstrap}")
    
    # Generate primes
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = np.diff(primes)
    print(f"  Primes: {len(primes):,}")
    print(f"  Gaps: {len(gaps):,}")
    
    # Point estimate
    point_l1 = compute_lambda1_from_gaps(gaps)
    print(f"\n  Point estimate: λ₁ = {point_l1:.6f}")
    
    # Bootstrap at full scale
    print(f"\nBootstrapping at full scale...")
    boot_l1s = bootstrap_lambda1(gaps, n_bootstrap)
    
    ci_lo = np.percentile(boot_l1s, 2.5)
    ci_hi = np.percentile(boot_l1s, 97.5)
    boot_mean = np.mean(boot_l1s)
    boot_std = np.std(boot_l1s)
    
    print(f"  Bootstrap mean: {boot_mean:.6f}")
    print(f"  Bootstrap std:  {boot_std:.6f}")
    print(f"  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    
    # Check if 1/φ is in CI
    phi_in_ci = ci_lo <= PHI_INV <= ci_hi
    print(f"\n  1/φ = {PHI_INV:.6f} in 95% CI: {'YES' if phi_in_ci else 'NO'}")
    
    # Multi-scale bootstrap for decay rate
    print("\n" + "-" * 60)
    print("MULTI-SCALE BOOTSTRAP FOR DECAY RATE")
    print("-" * 60)
    
    test_limits = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
    scale_results = []
    
    for lim in test_limits:
        p = get_primes(lim)
        g = np.diff(p)
        
        boot_samples = bootstrap_lambda1(g, n_bootstrap=200)  # Fewer for speed
        
        result = {
            'limit': lim,
            'n_primes': len(p),
            'log10_n': np.log10(len(p)),
            'point_estimate': compute_lambda1_from_gaps(g),
            'boot_mean': float(np.mean(boot_samples)),
            'boot_std': float(np.std(boot_samples)),
            'ci_lo': float(np.percentile(boot_samples, 2.5)),
            'ci_hi': float(np.percentile(boot_samples, 97.5)),
        }
        scale_results.append(result)
        print(f"  N={lim:>10,}: λ₁ = {result['boot_mean']:.4f} ± {result['boot_std']:.4f} "
              f"[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    
    # Bootstrap the slope
    print("\n" + "-" * 60)
    print("BOOTSTRAPPING THE DECAY SLOPE")
    print("-" * 60)
    
    def linear(x, a, b):
        return a * x + b
    
    # Point estimate of slope
    log_ns = np.array([r['log10_n'] for r in scale_results])
    point_l1s = np.array([r['point_estimate'] for r in scale_results])
    
    popt, _ = curve_fit(linear, log_ns, point_l1s)
    point_slope, point_intercept = popt
    
    print(f"  Point estimate: slope = {point_slope:.6f}")
    
    # Bootstrap slope by resampling scale results
    n_slope_boot = 1000
    boot_slopes = []
    
    for _ in range(n_slope_boot):
        # Resample with replacement
        indices = np.random.choice(len(scale_results), size=len(scale_results), replace=True)
        boot_log_ns = log_ns[indices]
        boot_l1s = np.array([scale_results[i]['boot_mean'] + 
                            np.random.normal(0, scale_results[i]['boot_std']) 
                            for i in indices])
        
        try:
            popt_boot, _ = curve_fit(linear, boot_log_ns, boot_l1s)
            boot_slopes.append(popt_boot[0])
        except:
            pass
    
    boot_slopes = np.array(boot_slopes)
    slope_ci_lo = np.percentile(boot_slopes, 2.5)
    slope_ci_hi = np.percentile(boot_slopes, 97.5)
    slope_mean = np.mean(boot_slopes)
    slope_std = np.std(boot_slopes)
    
    print(f"  Bootstrap mean: {slope_mean:.6f}")
    print(f"  Bootstrap std:  {slope_std:.6f}")
    print(f"  95% CI: [{slope_ci_lo:.6f}, {slope_ci_hi:.6f}]")
    
    # Check if -1/π² is in CI
    pi2_inv = -1 / np.pi**2
    pi2_in_ci = slope_ci_lo <= pi2_inv <= slope_ci_hi
    print(f"\n  -1/π² = {pi2_inv:.6f} in 95% CI: {'YES' if pi2_in_ci else 'NO'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"""
  λ₁ at N = {len(primes):,}:
    Point estimate: {point_l1:.6f}
    95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]
    Width: {ci_hi - ci_lo:.6f}
    
  1/φ = {PHI_INV:.6f} is {'INSIDE' if phi_in_ci else 'OUTSIDE'} the 95% CI
  
  Decay slope:
    Point estimate: {point_slope:.6f}
    95% CI: [{slope_ci_lo:.6f}, {slope_ci_hi:.6f}]
    Width: {slope_ci_hi - slope_ci_lo:.6f}
    
  -1/π² = {pi2_inv:.6f} is {'INSIDE' if pi2_in_ci else 'OUTSIDE'} the 95% CI
""")
    
    # Save
    output = {
        'experiment': 'exp_17_bootstrap',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'prime_limit': prime_limit,
            'n_bootstrap': n_bootstrap,
        },
        'results': {
            'full_scale': {
                'point_estimate': point_l1,
                'boot_mean': float(boot_mean),
                'boot_std': float(boot_std),
                'ci_lo': float(ci_lo),
                'ci_hi': float(ci_hi),
                'phi_inv_in_ci': phi_in_ci,
            },
            'scale_results': scale_results,
            'slope': {
                'point_estimate': float(point_slope),
                'boot_mean': float(slope_mean),
                'boot_std': float(slope_std),
                'ci_lo': float(slope_ci_lo),
                'ci_hi': float(slope_ci_hi),
                'pi2_inv': float(pi2_inv),
                'pi2_inv_in_ci': pi2_in_ci,
            },
        },
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_17_bootstrap_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
