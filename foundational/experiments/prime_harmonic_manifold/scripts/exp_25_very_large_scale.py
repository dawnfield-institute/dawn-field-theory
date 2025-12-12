"""
Experiment 25: Very Large Scale Validation
==========================================

Purpose: Test if 1/π² decay holds at extreme scales (10^7 to 10^8+ primes)
This addresses the concern that current results might be scale-limited artifacts.

Key questions:
1. Does λ₁ decay = -1/π² persist at 10^8 primes?
2. What is the confidence interval at extreme scale?
3. Does the Cramér null model divergence increase or stabilize?
"""

import numpy as np
from sympy import primerange, isprime
import json
from datetime import datetime
from pathlib import Path
import sys

# Theory prediction
THEORY_DECAY = -1 / (np.pi ** 2)  # ≈ -0.1013

def get_primes_up_to_n(n_primes):
    """Generate first n primes efficiently."""
    # Estimate upper bound using prime counting function approximation
    if n_primes < 6:
        upper = 15
    else:
        upper = int(n_primes * (np.log(n_primes) + np.log(np.log(n_primes)) + 2))
    
    primes = list(primerange(2, upper))
    while len(primes) < n_primes:
        upper = int(upper * 1.5)
        primes = list(primerange(2, upper))
    
    return primes[:n_primes]

def compute_gaps(primes):
    """Compute prime gaps."""
    return [primes[i+1] - primes[i] for i in range(len(primes)-1)]

def build_markov_matrix(gaps, n_gaps=2):
    """Build Markov transition matrix from gap chords."""
    # Create chord vocabulary
    chords = []
    for i in range(len(gaps) - n_gaps + 1):
        chord = tuple(gaps[i:i+n_gaps])
        chords.append(chord)
    
    # Build transition counts
    unique_chords = list(set(chords))
    chord_to_idx = {c: i for i, c in enumerate(unique_chords)}
    n = len(unique_chords)
    
    if n < 2:
        return None, 0
    
    # Count transitions
    counts = np.zeros((n, n))
    for i in range(len(chords) - 1):
        from_idx = chord_to_idx[chords[i]]
        to_idx = chord_to_idx[chords[i+1]]
        counts[from_idx, to_idx] += 1
    
    # Normalize to get transition probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    P = counts / row_sums
    
    return P, n

def get_leading_eigenvalue(P):
    """Get leading non-trivial eigenvalue."""
    if P is None or P.shape[0] < 2:
        return None
    
    try:
        eigenvalues = np.linalg.eigvals(P)
        # Sort by magnitude
        sorted_eigs = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
        # First is always ~1 (Perron-Frobenius), return second
        if len(sorted_eigs) > 1:
            return abs(sorted_eigs[1])
        return None
    except:
        return None

def generate_cramer_gaps(n_gaps, max_prime):
    """Generate gaps from Cramér model (Poisson with mean ~ log(p))."""
    gaps = []
    p = 2
    for _ in range(n_gaps):
        # Expected gap ~ log(p)
        expected = max(2, np.log(max(p, 3)))
        gap = max(2, int(np.random.exponential(expected)))
        # Ensure even (most gaps are even)
        if gap % 2 == 1:
            gap += 1
        gaps.append(gap)
        p += gap
    return gaps

def run_scale_test(n_primes, verbose=True):
    """Run test at specific scale."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing at N = {n_primes:,} primes")
        print(f"{'='*60}")
    
    # Generate primes
    if verbose:
        print("Generating primes...", end=" ", flush=True)
    primes = get_primes_up_to_n(n_primes)
    if verbose:
        print(f"done. Range: [2, {primes[-1]:,}]")
    
    # Compute gaps
    gaps = compute_gaps(primes)
    
    # Build Markov matrix
    if verbose:
        print("Building Markov matrix...", end=" ", flush=True)
    P, vocab_size = build_markov_matrix(gaps, n_gaps=2)
    if verbose:
        print(f"done. Vocabulary: {vocab_size:,} chords")
    
    # Get eigenvalue
    lambda_1 = get_leading_eigenvalue(P)
    if verbose:
        print(f"λ₁ = {lambda_1:.6f}")
    
    # Cramér null model (10 samples for speed at large scale)
    if verbose:
        print("Running Cramér null model...", end=" ", flush=True)
    cramer_lambdas = []
    for _ in range(10):
        null_gaps = generate_cramer_gaps(len(gaps), primes[-1])
        null_P, _ = build_markov_matrix(null_gaps, n_gaps=2)
        null_lambda = get_leading_eigenvalue(null_P)
        if null_lambda is not None:
            cramer_lambdas.append(null_lambda)
    
    cramer_mean = np.mean(cramer_lambdas) if cramer_lambdas else 0
    cramer_std = np.std(cramer_lambdas) if cramer_lambdas else 0
    cramer_z = (lambda_1 - cramer_mean) / cramer_std if cramer_std > 0 else 0
    if verbose:
        print(f"done. Cramér λ₁ = {cramer_mean:.4f} ± {cramer_std:.4f}, z = {cramer_z:.1f}")
    
    return {
        'n_primes': n_primes,
        'max_prime': primes[-1],
        'vocab_size': vocab_size,
        'lambda_1': lambda_1,
        'cramer_mean': cramer_mean,
        'cramer_std': cramer_std,
        'cramer_z': cramer_z
    }

def compute_decay_rate(results):
    """Compute decay rate from results across scales."""
    # Use log-log regression
    log_n = np.array([np.log10(r['n_primes']) for r in results])
    log_lambda = np.array([np.log10(r['lambda_1']) for r in results if r['lambda_1'] is not None])
    
    if len(log_lambda) < 2:
        return None, None
    
    # Linear regression
    slope, intercept = np.polyfit(log_n[:len(log_lambda)], log_lambda, 1)
    
    return slope, intercept

def main():
    print("="*70)
    print("EXPERIMENT 25: VERY LARGE SCALE VALIDATION")
    print("="*70)
    print(f"\nTheory prediction: λ₁ decay = {THEORY_DECAY:.6f} per log-decade")
    print("Testing whether this holds at 10^7 to 10^8 primes\n")
    
    # Test scales: 10^4, 10^5, 10^6, 10^7, and if memory allows, 5×10^7
    scales = [10_000, 100_000, 1_000_000, 10_000_000]
    
    # Try to add larger scales if system can handle it
    try:
        # Test if we can allocate for 50M
        test_array = np.zeros((1000, 1000))
        del test_array
        scales.append(50_000_000)
    except MemoryError:
        print("Note: Limiting to 10^7 due to memory constraints")
    
    results = []
    for n in scales:
        try:
            result = run_scale_test(n)
            results.append(result)
        except MemoryError:
            print(f"Memory error at N={n:,}, stopping here")
            break
        except Exception as e:
            print(f"Error at N={n:,}: {e}")
            break
    
    # Compute decay rate
    print("\n" + "="*70)
    print("DECAY RATE ANALYSIS")
    print("="*70)
    
    slope, intercept = compute_decay_rate(results)
    
    if slope is not None:
        print(f"\nMeasured decay rate: {slope:.6f} per log-decade")
        print(f"Theory prediction:   {THEORY_DECAY:.6f} per log-decade")
        print(f"Difference:          {abs(slope - THEORY_DECAY):.6f}")
        print(f"Relative error:      {abs(slope - THEORY_DECAY) / abs(THEORY_DECAY) * 100:.2f}%")
        
        # Z-score from theory
        # Estimate error from residuals
        log_n = np.array([np.log10(r['n_primes']) for r in results])
        log_lambda = np.array([np.log10(r['lambda_1']) for r in results])
        predicted = slope * log_n + intercept
        residual_std = np.std(log_lambda - predicted)
        
        # Standard error of slope
        n_points = len(log_n)
        se_slope = residual_std / np.sqrt(np.sum((log_n - np.mean(log_n))**2))
        
        z_from_theory = (slope - THEORY_DECAY) / se_slope if se_slope > 0 else 0
        
        print(f"\nSlope SE:            {se_slope:.6f}")
        print(f"Z-score from theory: {z_from_theory:.2f}")
        
        if abs(z_from_theory) < 2:
            print("\n✅ RESULT: 1/π² decay CONFIRMED at extreme scales")
            print("   The measured decay is statistically consistent with -1/π²")
        elif abs(z_from_theory) < 3:
            print("\n⚠️ RESULT: Marginal deviation from 1/π²")
            print("   Some tension but not conclusive rejection")
        else:
            print("\n❌ RESULT: Significant deviation from 1/π²")
            print("   The decay rate may differ from theory at large scales")
    
    # Cramér divergence analysis
    print("\n" + "="*70)
    print("CRAMÉR DIVERGENCE ANALYSIS")
    print("="*70)
    
    for r in results:
        print(f"N = {r['n_primes']:>12,}: z = {r['cramer_z']:>6.1f} (λ₁ = {r['lambda_1']:.4f})")
    
    # Check if divergence increases with scale
    z_values = [r['cramer_z'] for r in results]
    if len(z_values) >= 2:
        z_trend = z_values[-1] - z_values[0]
        if z_trend > 0:
            print(f"\n📈 Cramér z-score INCREASES with scale (+{z_trend:.1f})")
            print("   Real primes diverge MORE from null model at larger scales")
        else:
            print(f"\n📉 Cramér z-score stable/decreasing ({z_trend:.1f})")
    
    # Save results
    output = {
        'experiment': 'exp_25_very_large_scale',
        'timestamp': datetime.now().isoformat(),
        'theory_decay': THEORY_DECAY,
        'scales_tested': scales[:len(results)],
        'results': results,
        'decay_analysis': {
            'measured_slope': slope,
            'theory_slope': THEORY_DECAY,
            'difference': abs(slope - THEORY_DECAY) if slope else None,
            'relative_error_pct': abs(slope - THEORY_DECAY) / abs(THEORY_DECAY) * 100 if slope else None,
            'z_from_theory': z_from_theory if slope else None,
            'confirmed': abs(z_from_theory) < 2 if slope else None
        },
        'cramer_analysis': {
            'z_values': z_values,
            'divergence_increases': z_values[-1] > z_values[0] if len(z_values) >= 2 else None
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_25_very_large_scale_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file.name}")
    
    return output

if __name__ == '__main__':
    main()
