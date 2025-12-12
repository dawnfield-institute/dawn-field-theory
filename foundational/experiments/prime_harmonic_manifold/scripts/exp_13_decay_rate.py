"""
Experiment 13: Decay Rate Analysis

Investigates whether the λ₁ decay rate (-0.0881/decade) connects
to known mathematical constants.

Key candidates:
- log(2)/log(10) ≈ 0.301
- 1/(2π) ≈ 0.159
- 1/e ≈ 0.368
- Euler-Mascheroni γ ≈ 0.577
- log₁₀(e)/π ≈ 0.138
- 1/(φ·log(10)) ≈ 0.268
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import PHI, PHI_INV
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
from collections import Counter


# Mathematical constants
EULER_GAMMA = 0.5772156649015329
LOG10_E = np.log10(np.e)  # ≈ 0.4343
LOG_2 = np.log(2)
LOG_10 = np.log(10)
PI = np.pi


def compute_lambda1(prime_limit, topK=25):
    """Compute λ₁ for primes up to limit."""
    primes_list = list(sp.primerange(2, prime_limit))
    if len(primes_list) < 100:
        return None, len(primes_list)
    
    primes = np.array(primes_list, dtype=float)
    gaps = np.diff(primes)
    
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
    return np.max(eigenvals), len(primes_list)


def run_experiment():
    """Test decay rate connections to mathematical constants."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Decay Rate Analysis")
    print("=" * 70)
    
    # Generate high-resolution scaling data
    print("\nGenerating high-resolution λ₁ data...")
    
    test_limits = [
        5_000, 7_500, 10_000, 15_000, 20_000, 30_000, 50_000,
        75_000, 100_000, 150_000, 200_000, 300_000, 500_000,
        750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000,
        5_000_000, 7_500_000, 10_000_000, 15_000_000, 20_000_000
    ]
    
    results = []
    for lim in test_limits:
        l1, n_primes = compute_lambda1(lim)
        if l1:
            log_n = np.log10(n_primes)
            log_lim = np.log10(lim)
            results.append({
                'limit': lim,
                'n_primes': n_primes,
                'lambda1': l1,
                'log10_n': log_n,
                'log10_lim': log_lim,
            })
            print(f"  N={lim:>12,}: λ₁ = {l1:.6f}")
    
    # Extract arrays
    log_n = np.array([r['log10_n'] for r in results])
    log_lim = np.array([r['log10_lim'] for r in results])
    lambda1 = np.array([r['lambda1'] for r in results])
    
    print("\n" + "=" * 70)
    print("FIT ANALYSIS")
    print("=" * 70)
    
    # Linear fit: λ₁ = a * log₁₀(N) + b
    def linear(x, a, b):
        return a * x + b
    
    popt_n, _ = curve_fit(linear, log_n, lambda1)
    a_n, b_n = popt_n
    
    popt_lim, _ = curve_fit(linear, log_lim, lambda1)
    a_lim, b_lim = popt_lim
    
    print(f"\n  Fit by # primes:  λ₁ = {a_n:.6f} × log₁₀(N) + {b_n:.6f}")
    print(f"  Fit by limit:     λ₁ = {a_lim:.6f} × log₁₀(L) + {b_lim:.6f}")
    
    # The key slope
    slope = a_n
    print(f"\n  KEY SLOPE = {slope:.6f}")
    
    # Test against known constants
    print("\n" + "-" * 60)
    print("CONSTANT MATCHING")
    print("-" * 60)
    
    candidates = [
        ("1/log(10)", -1/LOG_10),
        ("-log₁₀(2)", -np.log10(2)),
        ("-1/(2π)", -1/(2*PI)),
        ("-1/e", -1/np.e),
        ("-γ (Euler)", -EULER_GAMMA),
        ("-log₁₀(e)/π", -LOG10_E/PI),
        ("-1/(φ×log(10))", -1/(PHI * LOG_10)),
        ("-1/(φ²)", -1/PHI**2),
        ("-1/φ / log(10)", -PHI_INV/LOG_10),
        ("-log₁₀(φ)", -np.log10(PHI)),
        ("-1/(π×log(10))", -1/(PI * LOG_10)),
        ("-2/(π×log(100))", -2/(PI * np.log(100))),
        ("-γ/log(10)", -EULER_GAMMA/LOG_10),
        ("-1/π²", -1/PI**2),
        ("-log(2)/π", -LOG_2/PI),
        ("-1/(2×log(10))", -1/(2*LOG_10)),
        ("-1/(e×log(10))", -1/(np.e*LOG_10)),
    ]
    
    print(f"\n  {'Candidate':<25} {'Value':<12} {'Error':<12} {'Rel. Error':<10}")
    print("  " + "-" * 60)
    
    matches = []
    for name, val in candidates:
        error = abs(slope - val)
        rel_error = error / abs(slope)
        matches.append((name, val, error, rel_error))
        marker = "✓" if rel_error < 0.1 else ""
        print(f"  {name:<25} {val:<12.6f} {error:<12.6f} {rel_error*100:<8.2f}% {marker}")
    
    # Sort by error
    matches.sort(key=lambda x: x[2])
    
    print("\n" + "-" * 60)
    print("TOP 5 MATCHES")
    print("-" * 60)
    
    for i, (name, val, error, rel_error) in enumerate(matches[:5]):
        print(f"  {i+1}. {name}: {val:.6f} (error: {rel_error*100:.2f}%)")
    
    # Best match
    best_name, best_val, best_error, best_rel = matches[0]
    
    # Combined expressions
    print("\n" + "-" * 60)
    print("COMBINED EXPRESSION SEARCH")
    print("-" * 60)
    
    # Try a + b*c form
    combined = []
    for c1_name, c1 in [("1", 1), ("φ", PHI), ("1/φ", PHI_INV), ("π", PI), ("e", np.e), ("γ", EULER_GAMMA)]:
        for c2_name, c2 in [("1", 1), ("φ", PHI), ("1/φ", PHI_INV), ("π", PI), ("e", np.e), ("log(10)", LOG_10)]:
            for op, op_name in [(lambda a,b: -a/b, "-{}/{}"), (lambda a,b: -a*b, "-{}×{}"), 
                                (lambda a,b: -(a+b), "-({}+{})"), (lambda a,b: -(a-b), "-({}−{})")]:
                try:
                    val = op(c1, c2)
                    if -1 < val < 0:  # Reasonable range
                        error = abs(slope - val)
                        rel_error = error / abs(slope)
                        name = op_name.format(c1_name, c2_name)
                        combined.append((name, val, error, rel_error))
                except:
                    pass
    
    combined.sort(key=lambda x: x[2])
    
    print("\n  Best combined expressions:")
    for i, (name, val, error, rel_error) in enumerate(combined[:10]):
        if rel_error < 0.15:
            print(f"    {name}: {val:.6f} (error: {rel_error*100:.2f}%)")
    
    # Crossing point analysis
    print("\n" + "=" * 70)
    print("CROSSING POINT ANALYSIS")
    print("=" * 70)
    
    # Where does λ₁ = 1/φ?
    log_cross_phi = (PHI_INV - b_n) / a_n
    n_cross_phi = 10 ** log_cross_phi
    
    # Where does λ₁ = 1/2?
    log_cross_half = (0.5 - b_n) / a_n
    n_cross_half = 10 ** log_cross_half
    
    # Where does λ₁ = 1/e?
    log_cross_e = (1/np.e - b_n) / a_n
    n_cross_e = 10 ** log_cross_e
    
    print(f"\n  λ₁ = 1/φ ≈ 0.618 at N ≈ {n_cross_phi:,.0f} primes (10^{log_cross_phi:.2f})")
    print(f"  λ₁ = 1/2 = 0.500 at N ≈ {n_cross_half:,.0f} primes (10^{log_cross_half:.2f})")
    print(f"  λ₁ = 1/e ≈ 0.368 at N ≈ {n_cross_e:,.0f} primes (10^{log_cross_e:.2f})")
    
    # Check if crossing points relate to known sequences
    print("\n  Nearest notable numbers to crossing points:")
    
    # The n-th prime close to crossing
    from sympy import prime, primepi
    
    cross_prime = prime(int(n_cross_phi)) if n_cross_phi < 10**8 else "too large"
    print(f"    {int(n_cross_phi):,}th prime ≈ {cross_prime}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
  Measured decay rate: {slope:.6f} per decade
  
  Best single-constant match: {best_name} = {best_val:.6f} (error: {best_rel*100:.2f}%)
  
  Physical interpretation:
  - If slope = -1/π² ≈ -0.1013: Connection to quantum mechanics / wave equations
  - If slope = -log₁₀(φ) ≈ -0.209: Direct φ-logarithmic structure  
  - If slope = -1/(2log(10)) ≈ -0.217: Prime counting function connection
  
  Current best fit suggests: {best_name}
""")
    
    # Save results
    results_data = {
        'experiment': 'exp_13_decay_rate',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'n_points': len(results)},
        'results': {
            'slope_by_n': float(a_n),
            'intercept_by_n': float(b_n),
            'slope_by_limit': float(a_lim),
            'intercept_by_limit': float(b_lim),
            'data_points': results,
            'best_match': {
                'name': best_name,
                'value': float(best_val),
                'error': float(best_error),
                'rel_error': float(best_rel),
            },
            'top_matches': [(m[0], float(m[1]), float(m[3])) for m in matches[:5]],
            'crossing_points': {
                'phi_inv': float(n_cross_phi),
                'half': float(n_cross_half),
                'e_inv': float(n_cross_e),
            },
        },
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_13_decay_rate_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results_data


if __name__ == '__main__':
    run_experiment()
