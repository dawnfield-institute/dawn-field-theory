#!/usr/bin/env python3
"""
Experiment 08: SEC Blind Analysis
=================================

Test if SEC's stress field partition really approaches 1/φ,
or if that's a post-hoc observation.

Methodology:
1. Compute SEC for various parameters (no target in mind)
2. Record the frac(E > 0) for each
3. Test what constant it's closest to
4. Check if it's stable or parameter-dependent

Author: Dawn Field Theory Research
Date: 2025-01-06
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add SEC core
SEC_CORE = r"c:\Users\peter\repos\core_workspace\dawn-field-theory\foundational\experiments\sec_prime_manifold\core"
sys.path.insert(0, SEC_CORE)

from sec_core import compute_sec, FIRST_50_PRIMES

# Constants to test against
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # 0.618033988749895

CANDIDATES = {
    '1/phi': PHI_INV,
    '3/5': 0.6,
    '2/3': 0.666667,
    '5/8': 0.625,
    '8/13': 0.615385,  # Fibonacci ratio
    '13/21': 0.619048,  # Fibonacci ratio
    '0.618': 0.618,
    '0.62': 0.62,
    'sqrt(2)-1': np.sqrt(2) - 1,  # 0.4142...
    '1-1/e': 1 - 1/np.e,  # 0.6321...
}


def run_sec_blind(n_max: int, factor_base_size: int, window: int, lam: float) -> dict:
    """Run SEC and return fraction of positive E."""
    factor_base = FIRST_50_PRIMES[:factor_base_size]
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=window, lam=lam)
    
    # Analyze odd numbers only (avoiding trivial 2)
    idx = np.arange(3, n_max + 1, 2)
    E = sec.E[idx]
    
    frac_positive = float((E > 0).mean())
    
    return {
        'n_max': n_max,
        'factor_base_size': factor_base_size,
        'window': window,
        'lambda': lam,
        'frac_positive': frac_positive,
    }


def find_closest_constant(value: float) -> tuple:
    """Find which candidate constant is closest to the value."""
    sorted_cands = sorted(CANDIDATES.items(), key=lambda x: abs(x[1] - value))
    closest = sorted_cands[0]
    return closest[0], closest[1], abs(closest[1] - value)


def main():
    print("=" * 70)
    print("EXPERIMENT 08: SEC Blind Analysis")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Question: Does SEC's stress partition actually approach 1/φ?")
    print("Methodology: Sweep parameters and record what fraction emerges.")
    print()
    
    results = []
    
    # Test 1: Vary n_max (scale invariance)
    print("Test 1: Varying scale (n_max)")
    print("-" * 50)
    for n_max in [10000, 50000, 100000, 500000]:
        r = run_sec_blind(n_max, factor_base_size=10, window=101, lam=0.99)
        closest_name, closest_val, error = find_closest_constant(r['frac_positive'])
        r['closest'] = closest_name
        r['error'] = error
        results.append(r)
        print(f"  n_max={n_max:>7,}: frac={r['frac_positive']:.6f} → closest: {closest_name} (error={error:.6f})")
    
    # Test 2: Vary factor base
    print()
    print("Test 2: Varying factor base size")
    print("-" * 50)
    for fb_size in [5, 10, 15, 20, 30]:
        r = run_sec_blind(50000, factor_base_size=fb_size, window=101, lam=0.99)
        closest_name, closest_val, error = find_closest_constant(r['frac_positive'])
        r['closest'] = closest_name
        r['error'] = error
        results.append(r)
        print(f"  factor_base={fb_size:>2}: frac={r['frac_positive']:.6f} → closest: {closest_name} (error={error:.6f})")
    
    # Test 3: Vary window size
    print()
    print("Test 3: Varying window size")
    print("-" * 50)
    for window in [51, 101, 201, 501]:
        r = run_sec_blind(50000, factor_base_size=10, window=window, lam=0.99)
        closest_name, closest_val, error = find_closest_constant(r['frac_positive'])
        r['closest'] = closest_name
        r['error'] = error
        results.append(r)
        print(f"  window={window:>3}: frac={r['frac_positive']:.6f} → closest: {closest_name} (error={error:.6f})")
    
    # Test 4: Vary lambda
    print()
    print("Test 4: Varying lambda (decay)")
    print("-" * 50)
    for lam in [0.9, 0.95, 0.99, 0.999]:
        r = run_sec_blind(50000, factor_base_size=10, window=101, lam=lam)
        closest_name, closest_val, error = find_closest_constant(r['frac_positive'])
        r['closest'] = closest_name
        r['error'] = error
        results.append(r)
        print(f"  lambda={lam:.3f}: frac={r['frac_positive']:.6f} → closest: {closest_name} (error={error:.6f})")
    
    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    fracs = [r['frac_positive'] for r in results]
    print(f"\nAll measured fractions:")
    print(f"  Mean:   {np.mean(fracs):.6f}")
    print(f"  Median: {np.median(fracs):.6f}")
    print(f"  Std:    {np.std(fracs):.6f}")
    print(f"  Range:  [{min(fracs):.6f}, {max(fracs):.6f}]")
    
    # How often was each constant closest?
    from collections import Counter
    closest_counts = Counter(r['closest'] for r in results)
    print(f"\nClosest constant frequency:")
    for name, count in closest_counts.most_common():
        print(f"  {name}: {count} times")
    
    # The key test: is 1/φ consistently the closest?
    phi_inv_closest = closest_counts.get('1/phi', 0)
    total = len(results)
    
    print()
    print("-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    
    if phi_inv_closest == total:
        print(f"✅ 1/φ was closest in ALL {total} tests")
        print(f"   Mean error: {np.mean([r['error'] for r in results if r['closest'] == '1/phi']):.6f}")
    elif phi_inv_closest > total / 2:
        print(f"⚠️ 1/φ was closest in {phi_inv_closest}/{total} tests")
        print("   The φ connection may be real but not universal.")
    else:
        other_winner = closest_counts.most_common(1)[0][0]
        print(f"❌ 1/φ was NOT the most common closest constant")
        print(f"   Most common: {other_winner} ({closest_counts[other_winner]}/{total})")
        print("   The φ connection may be parameter-dependent or coincidental.")
    
    # Check distance comparison
    print()
    mean_frac = np.mean(fracs)
    print(f"Overall mean fraction: {mean_frac:.6f}")
    print(f"Distance to 1/φ ({PHI_INV:.6f}): {abs(mean_frac - PHI_INV):.6f}")
    print(f"Distance to 3/5 (0.600000): {abs(mean_frac - 0.6):.6f}")
    print(f"Distance to 5/8 (0.625000): {abs(mean_frac - 0.625):.6f}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results,
        'summary': {
            'mean': float(np.mean(fracs)),
            'median': float(np.median(fracs)),
            'std': float(np.std(fracs)),
            'range': [float(min(fracs)), float(max(fracs))],
        },
        'closest_counts': dict(closest_counts),
        'phi_inv': PHI_INV,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = results_dir / f"exp_08_sec_blind_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {outfile}")


if __name__ == "__main__":
    main()
