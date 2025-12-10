"""
exp_04_robustness.py
====================

Purpose: Test robustness of SEC prime detection under parameter variation.

Key Questions:
1. Is the effect scale-invariant?
2. How sensitive is it to λ (decay parameter)?
3. How sensitive is it to window size?
4. Does it work for different n ranges (small, medium, large)?

Traces:
- scale_tests: results at n=10K, 50K, 100K, 500K
- lambda_sweep: frac(E>0) vs λ ∈ [0.9, 0.999]
- window_sweep: frac(E>0) vs window ∈ [21, 501]
- range_tests: results for [1K,10K], [10K,100K], [100K,500K]
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sec_core import compute_sec, FIRST_50_PRIMES as PRIMES


def test_scale_invariance(scales: List[int]) -> List[Dict[str, Any]]:
    """Test SEC at different scales."""
    PHI_INV = 0.618033988749895
    results = []
    
    for n_max in scales:
        print(f"  Testing scale n_max={n_max:,}...")
        
        sec = compute_sec(
            n_max=n_max,
            factor_base=PRIMES[:10],
            window=101,
            lam=0.99
        )
        
        # Analyze odd numbers
        odds = np.arange(3, n_max + 1, 2)
        is_prime_mask = sec.prime_mask[odds]
        E_odds = sec.E[odds]
        
        # Get top 1% positive I
        I_odds = sec.I[odds]
        threshold = np.percentile(I_odds, 99)
        top_1pct = I_odds >= threshold
        
        prime_rate_top1pct = np.mean(is_prime_mask[top_1pct]) if np.any(top_1pct) else 0
        frac_E_positive = float(np.mean(E_odds > 0))
        baseline = np.mean(is_prime_mask)
        
        results.append({
            'n_max': n_max,
            'baseline_prime_rate': float(baseline),
            'top_1pct_prime_rate': float(prime_rate_top1pct),
            'enrichment': float(prime_rate_top1pct / baseline) if baseline > 0 else 0,
            'frac_E_positive': frac_E_positive,
            'error_vs_phi': frac_E_positive - PHI_INV
        })
    
    return results


def test_lambda_sensitivity(n_max: int, lambdas: List[float]) -> List[Dict[str, Any]]:
    """Test sensitivity to decay parameter λ."""
    PHI_INV = 0.618033988749895
    results = []
    
    for lmbda in lambdas:
        sec = compute_sec(
            n_max=n_max,
            factor_base=PRIMES[:10],
            window=101,
            lam=lmbda
        )
        
        odds = np.arange(3, n_max + 1, 2)
        E_odds = sec.E[odds]
        frac_positive = float(np.mean(E_odds > 0))
        
        results.append({
            'lambda': lmbda,
            'frac_E_positive': frac_positive,
            'error_vs_phi': frac_positive - PHI_INV
        })
    
    return results


def test_window_sensitivity(n_max: int, windows: List[int]) -> List[Dict[str, Any]]:
    """Test sensitivity to window size."""
    PHI_INV = 0.618033988749895
    results = []
    
    for window in windows:
        sec = compute_sec(
            n_max=n_max,
            factor_base=PRIMES[:10],
            window=window,
            lam=0.99
        )
        
        odds = np.arange(3, n_max + 1, 2)
        E_odds = sec.E[odds]
        frac_positive = float(np.mean(E_odds > 0))
        
        results.append({
            'window': window,
            'frac_E_positive': frac_positive,
            'error_vs_phi': frac_positive - PHI_INV
        })
    
    return results


def test_range_stability(n_max: int, ranges: List[tuple]) -> List[Dict[str, Any]]:
    """Test if SEC works in different n ranges."""
    PHI_INV = 0.618033988749895
    
    # Compute once for full range
    sec = compute_sec(
        n_max=n_max,
        factor_base=PRIMES[:10],
        window=101,
        lam=0.99
    )
    
    results = []
    
    for start, end in ranges:
        # Get odd indices for this range
        odds_in_range = np.arange(max(3, start | 1), min(n_max + 1, end + 1), 2)
        if len(odds_in_range) == 0:
            continue
        
        E_range = sec.E[odds_in_range]
        is_prime_range = sec.prime_mask[odds_in_range]
        
        frac_positive = float(np.mean(E_range > 0))
        baseline = float(np.mean(is_prime_range))
        
        results.append({
            'range': f'{start:,}-{end:,}',
            'n_count': len(odds_in_range),
            'baseline_prime_rate': baseline,
            'frac_E_positive': frac_positive,
            'error_vs_phi': frac_positive - PHI_INV
        })
    
    return results


def run_experiment(n_max: int, output_dir: str) -> Dict[str, Any]:
    """Run complete robustness analysis."""
    PHI_INV = 0.618033988749895
    
    print("\nRunning scale invariance tests...")
    scales = [10000, 50000]
    if n_max >= 100000:
        scales.append(100000)
    if n_max >= 500000:
        scales.append(500000)
    
    scale_results = test_scale_invariance(scales)
    
    print("\nRunning λ sensitivity test...")
    lambdas = [0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
    lambda_results = test_lambda_sensitivity(n_max, lambdas)
    
    print("\nRunning window sensitivity test...")
    windows = [21, 51, 101, 151, 201, 301, 501]
    window_results = test_window_sensitivity(n_max, windows)
    
    print("\nRunning range stability test...")
    ranges = [(1000, 10000), (10000, 50000)]
    if n_max >= 100000:
        ranges.append((50000, 100000))
    if n_max >= 500000:
        ranges.append((100000, 500000))
    
    range_results = test_range_stability(n_max, ranges)
    
    # Analyze stability
    scale_variance = np.var([r['frac_E_positive'] for r in scale_results])
    lambda_variance = np.var([r['frac_E_positive'] for r in lambda_results])
    window_variance = np.var([r['frac_E_positive'] for r in window_results])
    
    # All enrichments > 2.5x?
    enrichment_stable = all(r['enrichment'] > 2.5 for r in scale_results)
    
    # φ within 2% for all scales?
    phi_stable = all(abs(r['error_vs_phi']) < 0.02 for r in scale_results)
    
    results = {
        'parameters': {
            'max_n_tested': n_max,
            'scales_tested': scales,
            'lambdas_tested': lambdas,
            'windows_tested': windows,
            'ranges_tested': [list(r) for r in ranges]
        },
        'scale_tests': scale_results,
        'lambda_tests': lambda_results,
        'window_tests': window_results,
        'range_tests': range_results,
        'stability_analysis': {
            'scale_variance': float(scale_variance),
            'lambda_variance': float(lambda_variance),
            'window_variance': float(window_variance),
            'most_sensitive_to': 'lambda' if lambda_variance > max(scale_variance, window_variance) else
                                 'scale' if scale_variance > window_variance else 'window'
        },
        'validation': {
            'scale_invariant': scale_variance < 0.001,
            'enrichment_stable': enrichment_stable,
            'phi_stable_across_scales': phi_stable,
            'lambda_sensitive': lambda_variance > 0.01,
            'window_stable': window_variance < 0.01
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 04: Robustness Tests')
    parser.add_argument('--n_max', type=int, default=50000, help='Maximum n to test')
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 04: Robustness Tests")
    print("=" * 70)
    
    results = run_experiment(args.n_max, output_dir)
    
    # Display results
    print()
    print("-" * 70)
    print("SCALE INVARIANCE")
    print("-" * 70)
    print()
    print(f"{'Scale':<12} {'Enrichment':<12} {'frac(E>0)':<12} {'Error vs φ':<12}")
    print("-" * 48)
    for r in results['scale_tests']:
        print(f"{r['n_max']:>10,}   {r['enrichment']:.2f}x       {r['frac_E_positive']:.6f}   {r['error_vs_phi']:+.6f}")
    print()
    
    print("-" * 70)
    print("LAMBDA SENSITIVITY")
    print("-" * 70)
    print()
    print(f"{'λ':<10} {'frac(E>0)':<12} {'Error vs φ':<12}")
    print("-" * 34)
    for r in results['lambda_tests']:
        print(f"{r['lambda']:.3f}      {r['frac_E_positive']:.6f}   {r['error_vs_phi']:+.6f}")
    print()
    
    print("-" * 70)
    print("WINDOW SENSITIVITY")
    print("-" * 70)
    print()
    print(f"{'Window':<10} {'frac(E>0)':<12} {'Error vs φ':<12}")
    print("-" * 34)
    for r in results['window_tests']:
        print(f"{r['window']:<10} {r['frac_E_positive']:.6f}   {r['error_vs_phi']:+.6f}")
    print()
    
    print("-" * 70)
    print("RANGE STABILITY")
    print("-" * 70)
    print()
    print(f"{'Range':<20} {'Count':<12} {'frac(E>0)':<12} {'Error vs φ':<12}")
    print("-" * 56)
    for r in results['range_tests']:
        print(f"{r['range']:<20} {r['n_count']:<12,} {r['frac_E_positive']:.6f}   {r['error_vs_phi']:+.6f}")
    print()
    
    print("-" * 70)
    print("STABILITY ANALYSIS")
    print("-" * 70)
    print()
    sa = results['stability_analysis']
    print(f"  Scale variance:   {sa['scale_variance']:.6f}")
    print(f"  Lambda variance:  {sa['lambda_variance']:.6f}")
    print(f"  Window variance:  {sa['window_variance']:.6f}")
    print(f"  Most sensitive to: {sa['most_sensitive_to']}")
    print()
    
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)
    print()
    for key, passed in results['validation'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        # Some validations are "good" if they fail (like lambda_sensitive)
        if key == 'lambda_sensitive':
            status = "⚠️  SENSITIVE" if passed else "✅ STABLE"
        print(f"  {key}: {status}")
    print()
    
    # Save trace
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = os.path.join(output_dir, f'exp_04_robustness_{timestamp}.json')
    
    # Convert numpy bool_ to native bool for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_clean = convert_numpy(results)
    
    with open(trace_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"Trace saved: {os.path.basename(trace_file)}")


if __name__ == '__main__':
    main()