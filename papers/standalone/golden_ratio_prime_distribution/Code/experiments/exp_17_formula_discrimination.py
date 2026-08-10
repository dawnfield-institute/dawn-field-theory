#!/usr/bin/env python3
"""
EXPERIMENT 17: Formula Discrimination and Coincidence Detection

GOAL: Determine if the D=3, size=9 connection is meaningful or coincidental.

APPROACH:
1. Test ALL simple formulas that predict 9 for 1D
2. Use 2D, 3D, 0D as discriminators
3. Statistical significance: How likely is 9 by chance?
4. Alternative explanations for why 9 works

If ANY formula matches MULTIPLE dimensions --> likely meaningful
If only 1D works --> probably coincidence
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Ensure reproducibility
np.random.seed(42)

def sieve_primes(n_max):
    """Generate primes up to n_max."""
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def gaussian_primes(n_max):
    """Generate Gaussian primes (complex primes) up to norm n_max."""
    gprimes = []
    # Split primes: p = 1 mod 4 splits as (a+bi)(a-bi)
    # Inert primes: p = 3 mod 4 stays prime
    # Ramified: 2 = -i(1+i)^2
    
    for a in range(int(n_max**0.5) + 1):
        for b in range(int(n_max**0.5) + 1):
            if a == 0 and b == 0:
                continue
            norm = a*a + b*b
            if norm <= n_max and is_prime_simple(norm):
                gprimes.append((a, b, norm))
    
    # Sort by norm
    gprimes.sort(key=lambda x: x[2])
    return gprimes

def is_prime_simple(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def sec_1d(primes, base_size, n_max, lam=0.99):
    """Standard 1D SEC on integers."""
    B = primes[:base_size]
    
    E = np.zeros(n_max)
    E_prev = 0
    
    for n in range(2, n_max):
        S_n = sum(1 for p in B if n % p == 0) / len(B)
        S_hat = np.mean([sum(1 for p in B if m % p == 0) / len(B) 
                        for m in range(2, n)]) if n > 2 else 0
        I_n = S_hat - S_n
        E[n] = lam * E_prev + I_n
        E_prev = E[n]
    
    return np.mean(E[1000:n_max] > 0)

def sec_2d_gaussian(gprimes, base_size, n_points, lam=0.99):
    """SEC on 2D lattice using Gaussian integers."""
    if len(gprimes) < base_size:
        return np.nan
    
    B = gprimes[:base_size]
    
    E = np.zeros(n_points)
    E_prev = 0
    
    # Sample points on 2D lattice
    for idx in range(n_points):
        a = (idx % 100) + 1
        b = (idx // 100) + 1
        
        # Count how many Gaussian primes divide a+bi
        S_n = 0
        for pa, pb, pnorm in B:
            # (a+bi) divisible by (pa+pbi) if (a+bi)/(pa+pbi) is Gaussian integer
            # Compute: (a+bi)(pa-pbi) / (pa^2 + pb^2)
            real_part = a * pa + b * pb
            imag_part = b * pa - a * pb
            if real_part % pnorm == 0 and imag_part % pnorm == 0:
                S_n += 1
        S_n /= len(B)
        
        # Running mean of S
        S_hat = np.mean([E[j] for j in range(max(0, idx-100), idx)]) if idx > 0 else 0
        I_n = S_hat - S_n
        E[idx] = lam * E_prev + I_n
        E_prev = E[idx]
    
    return np.mean(E[100:] > 0)

def test_formula_predictions():
    """Test many formulas that predict 9 for 1D, see which generalize."""
    
    print("=" * 70)
    print("FORMULA DISCRIMINATION TEST")
    print("=" * 70)
    print("\nFinding all simple formulas f(dim) where f(1) = 9...\n")
    
    # Generate formulas that predict 9 for dim=1
    # Using D as a free parameter
    formulas = []
    
    # D-based formulas
    for D in range(2, 10):
        # Power law: D^(k - dim) = 9 for dim=1 --> D^(k-1) = 9
        for k in range(1, 8):
            if D ** (k - 1) == 9:
                formulas.append({
                    'name': f'{D}^({k} - dim)',
                    'fn': lambda dim, d=D, kk=k: d ** (kk - dim) if kk > dim else 1,
                    'D': D, 'k': k, 'type': 'power'
                })
        
        # Linear: D * (k - dim) = 9 for dim=1 --> D * (k-1) = 9
        for k in range(2, 15):
            if D * (k - 1) == 9:
                formulas.append({
                    'name': f'{D} * ({k} - dim)',
                    'fn': lambda dim, d=D, kk=k: max(1, d * (kk - dim)),
                    'D': D, 'k': k, 'type': 'linear'
                })
        
        # Quadratic: D * (k - dim)^2 = 9 for dim=1
        for k in range(2, 10):
            if D * (k - 1)**2 == 9:
                formulas.append({
                    'name': f'{D} * ({k} - dim)^2',
                    'fn': lambda dim, d=D, kk=k: max(1, d * (kk - dim)**2),
                    'D': D, 'k': k, 'type': 'quadratic'
                })
    
    # Hardcoded constants that work for 1D
    formulas.append({
        'name': '9 (constant)',
        'fn': lambda dim: 9,
        'type': 'constant'
    })
    
    # Direct power of 3
    formulas.append({
        'name': '3^(3 - dim)',
        'fn': lambda dim: 3 ** (3 - dim) if dim <= 3 else 1,
        'type': 'power'
    })
    
    formulas.append({
        'name': '3 * (3 - dim + 1)',
        'fn': lambda dim: 3 * (3 - dim + 1),
        'type': 'linear'
    })
    
    formulas.append({
        'name': '3 * (4 - dim)',
        'fn': lambda dim: 3 * (4 - dim),
        'type': 'linear'
    })
    
    # Alternative: based on prime structure
    formulas.append({
        'name': '2^(dim+2) + 1',
        'fn': lambda dim: 2**(dim+2) + 1,
        'type': 'alt'
    })
    
    formulas.append({
        'name': '3^dim + 3^(2-dim)',
        'fn': lambda dim: int(3**dim + 3**(2-dim)) if dim <= 2 else 3**dim,
        'type': 'alt'
    })
    
    # Remove duplicates and verify 1D prediction
    unique_formulas = []
    seen = set()
    for f in formulas:
        pred_1d = f['fn'](1)
        pred_2d = f['fn'](2)
        key = (f['name'], pred_1d, pred_2d)
        if pred_1d == 9 and key not in seen:
            unique_formulas.append(f)
            seen.add(key)
    
    print(f"Found {len(unique_formulas)} unique formulas predicting 9 for 1D:\n")
    
    print(f"{'Formula':<25} {'1D':<6} {'2D':<6} {'3D':<6} {'0D':<6}")
    print("-" * 50)
    for f in unique_formulas:
        pred_1d = f['fn'](1)
        pred_2d = f['fn'](2)
        pred_3d = f['fn'](3)
        pred_0d = f['fn'](0)
        print(f"{f['name']:<25} {pred_1d:<6} {pred_2d:<6} {pred_3d:<6} {pred_0d:<6}")
    
    return unique_formulas

def test_empirical_2d(n_max=5000):
    """Thoroughly test 2D SEC to find true optimal."""
    
    print("\n" + "=" * 70)
    print("EMPIRICAL 2D GAUSSIAN SEC SCAN")
    print("=" * 70)
    
    gprimes = gaussian_primes(500)
    print(f"\nGenerated {len(gprimes)} Gaussian primes")
    print(f"First 10: {gprimes[:10]}")
    
    # Test many sizes
    sizes_to_test = list(range(2, min(30, len(gprimes))))
    results_2d = {}
    
    print(f"\nTesting {len(sizes_to_test)} base sizes...")
    for size in sizes_to_test:
        frac = sec_2d_gaussian(gprimes, size, n_max)
        error = abs(frac - 0.618034)
        results_2d[size] = {'frac': frac, 'error': error}
        print(f"  Size {size:2d}: frac(E>0) = {frac:.4f}, error = {error:.4f}")
    
    # Find best
    best_size = min(results_2d.keys(), key=lambda s: results_2d[s]['error'])
    print(f"\nBest 2D size: {best_size} (error = {results_2d[best_size]['error']:.4f})")
    
    return best_size, results_2d

def test_coincidence_probability():
    """Calculate probability that 9 works by chance."""
    
    print("\n" + "=" * 70)
    print("COINCIDENCE PROBABILITY ANALYSIS")
    print("=" * 70)
    
    # How special is size=9?
    # Test: if we pick random sizes, how often does one hit phi within 0.01?
    
    n_max = 10000
    primes = sieve_primes(n_max)
    
    # Test sizes 2-30
    results = {}
    for size in range(2, 31):
        frac = sec_1d(primes, size, n_max)
        error = abs(frac - 0.618034)
        results[size] = {'frac': frac, 'error': error}
    
    # How many sizes hit within different error thresholds?
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    for thresh in thresholds:
        hits = [s for s, r in results.items() if r['error'] < thresh]
        prob = len(hits) / len(results)
        print(f"\nError < {thresh}: {len(hits)}/29 sizes = {prob:.1%}")
        if hits:
            print(f"  Sizes: {hits}")
    
    # Is 9 uniquely best, or are there other near-optimal?
    sorted_by_error = sorted(results.items(), key=lambda x: x[1]['error'])
    print("\nTop 5 sizes by phi-match:")
    for size, r in sorted_by_error[:5]:
        print(f"  Size {size}: error = {r['error']:.6f}")
    
    # Statistical test: is size=9 significantly better than random choice?
    best_size, best_result = sorted_by_error[0]
    second_best_size, second_best = sorted_by_error[1]
    
    improvement = (second_best['error'] - best_result['error']) / second_best['error']
    print(f"\nSize {best_size} is {improvement:.1%} better than size {second_best_size}")
    
    return results, sorted_by_error

def investigate_why_9():
    """Deep dive: what's special about 9 primes specifically?"""
    
    print("\n" + "=" * 70)
    print("WHY SIZE 9? STRUCTURAL ANALYSIS")
    print("=" * 70)
    
    primes = sieve_primes(10000)
    first_9 = primes[:9]
    print(f"\nFirst 9 primes: {list(first_9)}")
    print(f"Product: {np.prod(first_9)}")
    print(f"Sum: {np.sum(first_9)}")
    print(f"Product/Sum: {np.prod(first_9) / np.sum(first_9):.2f}")
    
    # Log density at each prime
    print(f"\nLog density analysis:")
    for i, p in enumerate(first_9):
        log_density = np.log(i+1) / np.log(p) if p > 1 else 0
        print(f"  p_{i+1} = {p}: log({i+1})/log({p}) = {log_density:.4f}")
    
    # Prime counting at key thresholds
    print(f"\nPrime counting:")
    for x in [9, 23, 100]:
        pi_x = np.sum(primes <= x)
        log_ratio = pi_x / (x / np.log(x)) if x > 1 else 0
        print(f"  π({x}) = {pi_x}, x/ln(x) = {x/np.log(x):.1f}, ratio = {log_ratio:.2f}")
    
    # Is 9 related to prime gaps?
    gaps = np.diff(first_9)
    print(f"\nGaps between first 9 primes: {list(gaps)}")
    print(f"Mean gap: {np.mean(gaps):.2f}")
    print(f"Max gap: {np.max(gaps)}")
    
    # Fibonacci connection?
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    print(f"\nFibonacci up to 9th: {fib}")
    print(f"Fib(6) = 8, Fib(7) = 13, 9 is between them")
    print(f"9 is NOT Fibonacci")
    
    # Powers of 3
    print(f"\n9 = 3^2")
    print(f"Is 3 special? First odd prime, only prime p where p^2 < p_next")
    print(f"  2^2 = 4 > 3? Yes")
    print(f"  3^2 = 9 > 5? Yes, but 9 = 10 - 1 = 2*5 - 1")
    
    # Primorial analysis
    print(f"\nPrimorial analysis:")
    primorial = 1
    for i, p in enumerate(first_9):
        primorial *= p
        divisors_up_to_primorial = sum(1 for n in range(2, min(primorial, 1000)) 
                                       if any(n % q == 0 for q in first_9[:i+1]))
        coverage = divisors_up_to_primorial / min(primorial, 998) if primorial > 2 else 0
        print(f"  p_{i+1}# = {primorial:>15,d}, coverage = {coverage:.3f}")
    
    return first_9

def main():
    print("\n" + "=" * 70)
    print("EXPERIMENT 17: IS THE D=3 CONNECTION REAL OR COINCIDENCE?")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Testing whether D=3, size=9 connection is meaningful',
        'tests': {}
    }
    
    # 1. Formula discrimination
    formulas = test_formula_predictions()
    results['tests']['formulas'] = [
        {'name': f['name'], 
         'predictions': {str(d): f['fn'](d) for d in [0, 1, 2, 3]}}
        for f in formulas
    ]
    
    # 2. Empirical 2D test
    best_2d, results_2d = test_empirical_2d(n_max=3000)
    results['tests']['2d_empirical'] = {
        'best_size': best_2d,
        'all_results': {str(k): {'frac': float(v['frac']), 'error': float(v['error'])} 
                       for k, v in results_2d.items()}
    }
    
    # 3. Coincidence probability
    size_results, sorted_results = test_coincidence_probability()
    results['tests']['coincidence'] = {
        'all_sizes': {str(k): {'frac': float(v['frac']), 'error': float(v['error'])} 
                     for k, v in size_results.items()},
        'best_5': [(s, float(r['error'])) for s, r in sorted_results[:5]]
    }
    
    # 4. Why 9?
    first_9 = investigate_why_9()
    results['tests']['structural'] = {
        'first_9_primes': [int(p) for p in first_9],
        'product': int(np.prod(first_9)),
        'sum': int(np.sum(first_9))
    }
    
    # 5. Match formulas to empirical 2D
    print("\n" + "=" * 70)
    print("FORMULA MATCHING")
    print("=" * 70)
    
    print(f"\nEmpirical 2D optimal: {best_2d}")
    print("\nWhich formulas predict this?")
    matching_formulas = []
    for f in formulas:
        pred_2d = f['fn'](2)
        if pred_2d == best_2d:
            matching_formulas.append(f['name'])
            print(f"  ✓ {f['name']} predicts {pred_2d}")
    
    if not matching_formulas:
        print("  ✗ No formula predicts the empirical 2D optimal!")
        
        # Find closest predictions
        print("\n  Closest predictions:")
        for f in formulas:
            pred_2d = f['fn'](2)
            if abs(pred_2d - best_2d) <= 5:
                print(f"    {f['name']}: predicts {pred_2d}, off by {abs(pred_2d - best_2d)}")
    
    results['matching_formulas'] = matching_formulas
    
    # 6. Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    # Criteria for "real" vs "coincidence"
    criteria = {
        'formula_matches_2d': len(matching_formulas) > 0,
        'size_9_uniquely_best': sorted_results[0][0] == 9,
        'size_9_margin': sorted_results[0][1]['error'] < sorted_results[1][1]['error'] * 0.5,
        'multiple_formulas_agree': len(formulas) > 3
    }
    
    print("\nCriteria for meaningful connection:")
    for name, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    n_pass = sum(criteria.values())
    if n_pass >= 3:
        verdict = "LIKELY MEANINGFUL"
    elif n_pass >= 2:
        verdict = "INCONCLUSIVE - needs more evidence"
    else:
        verdict = "LIKELY COINCIDENCE"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  Passed {n_pass}/4 criteria")
    
    results['verdict'] = {
        'criteria': {k: bool(v) for k, v in criteria.items()},
        'n_pass': n_pass,
        'conclusion': verdict
    }
    
    # Save
    trace_dir = Path(__file__).parent.parent / 'traces'
    trace_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trace_file = trace_dir / f'exp_17_formula_discrimination_{timestamp}.json'
    
    with open(trace_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTrace saved: {trace_file.name}")
    
    return results

if __name__ == '__main__':
    main()
