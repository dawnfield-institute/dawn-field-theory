#!/usr/bin/env python3
"""
EXPERIMENT 23: Why Gaussian Model Fails - Finding the Real Mechanism

The alternating AR(1) with Gaussian noise predicts P(E_odd > 0) ≈ 0.52
But we observe 0.618.

The difference (0.618 - 0.52 = 0.098) must come from:
1. Non-Gaussian I distribution
2. Serial correlation in I
3. Phase structure (prime vs composite patterns)
4. Something else

Let's find out.
"""

import numpy as np
from scipy import stats
import json
from datetime import datetime
from pathlib import Path

np.random.seed(42)

def sieve_primes(n_max):
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime, np.where(is_prime)[0]

def compute_sec(n_max, factor_base, window=101, lam=0.99):
    k = len(factor_base)
    
    S = np.zeros(n_max + 1)
    S_hat = np.zeros(n_max + 1)
    I = np.zeros(n_max + 1)
    E = np.zeros(n_max + 1)
    
    for n in range(2, n_max + 1):
        S[n] = sum(1 for p in factor_base if n % p == 0) / k
    
    half = window // 2
    for n in range(2, n_max + 1):
        lo = max(2, n - half)
        hi = min(n_max, n + half)
        S_hat[n] = S[lo:hi+1].mean()
    
    for n in range(2, n_max + 1):
        I[n] = S_hat[n] - S[n]
        E[n] = lam * E[n-1] + I[n]
    
    return S, S_hat, I, E

def analyze_I_distribution(I, odds):
    """Check if I is Gaussian."""
    I_odd = I[odds]
    
    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(I_odd[:5000])  # Limited sample
    
    # Skewness and kurtosis
    skew = stats.skew(I_odd)
    kurtosis = stats.kurtosis(I_odd)  # Excess kurtosis (0 for Gaussian)
    
    # Quantile comparison
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    empirical_quantiles = np.percentile(I_odd, np.linspace(1, 99, 100))
    
    # Rescale empirical
    I_mean, I_std = np.mean(I_odd), np.std(I_odd)
    empirical_standardized = (empirical_quantiles - I_mean) / I_std
    
    qq_deviation = np.mean(np.abs(empirical_standardized - theoretical_quantiles))
    
    return {
        'shapiro_p': float(shapiro_p),
        'skewness': float(skew),
        'kurtosis_excess': float(kurtosis),
        'qq_deviation': float(qq_deviation),
        'is_gaussian': shapiro_p > 0.05 and abs(skew) < 0.5 and abs(kurtosis) < 1
    }

def analyze_I_autocorrelation(I, n_max):
    """Check if I values are serially correlated."""
    odds = np.arange(3, n_max + 1, 2)
    I_odd = I[odds]
    
    # Autocorrelation at lag 1, 2, ... (in terms of odd positions)
    # Note: consecutive odds are 2 apart in actual indices
    
    autocorrs = []
    for lag in range(1, 20):
        if lag < len(I_odd) - 100:
            corr = np.corrcoef(I_odd[:-lag], I_odd[lag:])[0, 1]
            autocorrs.append(float(corr))
    
    # Also check autocorrelation of actual E transitions
    E_odd = []
    lam = 0.99
    E_prev = 0
    for n in range(3, n_max + 1, 2):
        E_curr = lam * lam * E_prev + lam * I[n-1] + I[n]
        E_odd.append(E_curr)
        E_prev = E_curr
    
    return {
        'I_autocorr_lags': autocorrs[:10],
        'significant_autocorr': any(abs(a) > 0.05 for a in autocorrs[:5])
    }

def analyze_prime_composite_structure(I, E, primes_mask, n_max):
    """Check if prime/composite patterns drive the deviation."""
    
    odds = np.arange(3, n_max + 1, 2)
    is_prime_odd = primes_mask[odds]
    
    I_odd = I[odds]
    E_odd = E[odds]
    
    # Separate by prime/composite
    I_prime = I_odd[is_prime_odd]
    I_composite = I_odd[~is_prime_odd]
    E_prime = E_odd[is_prime_odd]
    E_composite = E_odd[~is_prime_odd]
    
    # Key insight: primes have higher I than composites
    # This creates "bursts" of positive I at prime locations
    
    return {
        'I_prime_mean': float(np.mean(I_prime)),
        'I_composite_mean': float(np.mean(I_composite)),
        'I_difference': float(np.mean(I_prime) - np.mean(I_composite)),
        'E_prime_frac_positive': float(np.mean(E_prime > 0)),
        'E_composite_frac_positive': float(np.mean(E_composite > 0)),
        'prime_fraction_of_odds': float(np.mean(is_prime_odd))
    }

def simulate_with_actual_I(I, n_max, lam=0.99):
    """
    Simulate using ACTUAL I values but with different starting conditions
    to isolate the effect of the I distribution vs other factors.
    """
    
    odds = np.arange(3, n_max + 1, 2)
    I_odd = I[odds]
    
    # Permutation test: shuffle I values and see what happens
    n_trials = 100
    frac_results = []
    
    for _ in range(n_trials):
        I_shuffled = I.copy()
        # Shuffle I values among odds
        I_odd_shuffled = np.random.permutation(I_odd)
        
        # Recompute E with shuffled I
        E_shuffled = np.zeros(n_max + 1)
        for i, n in enumerate(odds):
            if n > 2:
                E_shuffled[n] = lam * E_shuffled[n-2] + I_odd_shuffled[i]
        
        frac = np.mean(E_shuffled[odds] > 0)
        frac_results.append(frac)
    
    return {
        'shuffled_frac_mean': float(np.mean(frac_results)),
        'shuffled_frac_std': float(np.std(frac_results)),
        'original_frac': float(np.mean(I[odds] > 0))  # This is I, not E
    }

def analyze_E_persistence(E, n_max):
    """
    Look at how long E stays positive/negative.
    
    If there are long "runs" of E > 0 or E < 0, this affects the fraction.
    """
    
    odds = np.arange(3, n_max + 1, 2)
    E_odd = E[odds]
    
    # Find run lengths
    signs = np.sign(E_odd)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    
    if len(sign_changes) > 1:
        run_lengths = np.diff(sign_changes)
        
        # Separate by sign
        positive_runs = []
        negative_runs = []
        
        for i in range(len(sign_changes) - 1):
            run_len = sign_changes[i+1] - sign_changes[i]
            if signs[sign_changes[i] + 1] > 0:
                positive_runs.append(run_len)
            else:
                negative_runs.append(run_len)
        
        return {
            'mean_run_length': float(np.mean(run_lengths)),
            'mean_positive_run': float(np.mean(positive_runs)) if positive_runs else 0,
            'mean_negative_run': float(np.mean(negative_runs)) if negative_runs else 0,
            'num_crossings': len(sign_changes),
            'crossing_rate': float(len(sign_changes) / len(E_odd))
        }
    else:
        return {'error': 'Not enough sign changes'}

def analyze_threshold_sensitivity(E, n_max):
    """
    Check: is frac(E>0) = 0.618 exact, or is it frac(E>ε) for some ε?
    """
    
    odds = np.arange(3, n_max + 1, 2)
    E_odd = E[odds]
    
    thresholds = np.linspace(-0.1, 0.1, 201)
    fracs = []
    
    for thresh in thresholds:
        frac = np.mean(E_odd > thresh)
        fracs.append(float(frac))
    
    # Find threshold that gives exactly 0.618
    phi_inv = 0.618034
    closest_idx = np.argmin(np.abs(np.array(fracs) - phi_inv))
    closest_thresh = thresholds[closest_idx]
    
    # Gradient at threshold=0
    idx_zero = np.argmin(np.abs(thresholds))
    gradient = (fracs[idx_zero + 1] - fracs[idx_zero - 1]) / (thresholds[idx_zero + 1] - thresholds[idx_zero - 1])
    
    return {
        'frac_at_zero': float(fracs[idx_zero]),
        'threshold_for_phi': float(closest_thresh),
        'gradient_at_zero': float(gradient),
        'frac_E_exactly_zero': float(np.mean(E_odd == 0))
    }

def decompose_by_digit_pattern(E, I, n_max):
    """
    Check if there's structure based on n mod small numbers.
    """
    
    odds = np.arange(3, n_max + 1, 2)
    E_odd = E[odds]
    
    results = {}
    for mod in [3, 5, 7, 9]:
        for residue in range(mod):
            if residue % 2 == 1:  # Only odd residues
                mask = (odds % mod) == residue
                if np.sum(mask) > 100:
                    frac = np.mean(E_odd[mask] > 0)
                    results[f'{residue}_mod_{mod}'] = float(frac)
    
    return results

def main():
    print("=" * 70)
    print("EXPERIMENT 23: FINDING THE REAL φ MECHANISM")
    print("=" * 70)
    
    n_max = 100000
    primes_mask, primes = sieve_primes(n_max)
    factor_base = primes[:9]
    
    S, S_hat, I, E = compute_sec(n_max, factor_base)
    odds = np.arange(3, n_max + 1, 2)
    
    actual_frac = np.mean(E[odds] > 0)
    print(f"\nActual frac(E>0) on odds: {actual_frac:.6f}")
    print(f"Target (1/φ): 0.618034")
    print(f"Error: {abs(actual_frac - 0.618034):.6f}")
    
    results = {'timestamp': datetime.now().isoformat()}
    
    # 1. Check Gaussianity
    print("\n1. Is I Gaussian?")
    gauss = analyze_I_distribution(I, odds)
    print(f"   Shapiro-Wilk p-value: {gauss['shapiro_p']:.4f}")
    print(f"   Skewness: {gauss['skewness']:.4f}")
    print(f"   Excess kurtosis: {gauss['kurtosis_excess']:.4f}")
    print(f"   Gaussian: {gauss['is_gaussian']}")
    results['gaussianity'] = gauss
    
    # 2. Check autocorrelation
    print("\n2. Is I serially correlated?")
    autocorr = analyze_I_autocorrelation(I, n_max)
    print(f"   First 5 autocorrs: {[f'{a:.4f}' for a in autocorr['I_autocorr_lags'][:5]]}")
    print(f"   Significant: {autocorr['significant_autocorr']}")
    results['autocorrelation'] = autocorr
    
    # 3. Prime/composite structure
    print("\n3. Prime vs composite structure:")
    pc = analyze_prime_composite_structure(I, E, primes_mask, n_max)
    print(f"   I_prime mean: {pc['I_prime_mean']:.6f}")
    print(f"   I_composite mean: {pc['I_composite_mean']:.6f}")
    print(f"   Difference: {pc['I_difference']:.6f}")
    print(f"   E_prime frac>0: {pc['E_prime_frac_positive']:.4f}")
    print(f"   E_composite frac>0: {pc['E_composite_frac_positive']:.4f}")
    results['prime_composite'] = pc
    
    # 4. E persistence
    print("\n4. E persistence (run lengths):")
    persist = analyze_E_persistence(E, n_max)
    print(f"   Mean run length: {persist.get('mean_run_length', 'N/A'):.2f}")
    print(f"   Mean positive run: {persist.get('mean_positive_run', 'N/A'):.2f}")
    print(f"   Mean negative run: {persist.get('mean_negative_run', 'N/A'):.2f}")
    print(f"   Crossing rate: {persist.get('crossing_rate', 'N/A'):.4f}")
    results['persistence'] = persist
    
    # 5. Threshold sensitivity
    print("\n5. Threshold sensitivity:")
    thresh = analyze_threshold_sensitivity(E, n_max)
    print(f"   frac(E>0): {thresh['frac_at_zero']:.6f}")
    print(f"   Threshold for exact φ: {thresh['threshold_for_phi']:.6f}")
    print(f"   Gradient at 0: {thresh['gradient_at_zero']:.4f}")
    results['threshold'] = thresh
    
    # 6. Residue class patterns
    print("\n6. Residue class patterns (frac E>0 by n mod k):")
    residues = decompose_by_digit_pattern(E, I, n_max)
    for key, val in sorted(residues.items()):
        print(f"   {key}: {val:.4f}")
    results['residues'] = residues
    
    # 7. THE KEY: Prime injection pattern
    print("\n" + "=" * 70)
    print("KEY INSIGHT: PRIME INJECTION PATTERN")
    print("=" * 70)
    
    # Primes have I_prime >> I_composite
    # When a prime occurs, it "kicks" E upward
    # Composites gradually decay E back down
    
    # The fraction of time E > 0 depends on:
    # - How much primes kick E up (I_prime - I_overall)
    # - How fast E decays (λ)
    # - How often primes occur (prime density)
    
    prime_kick = pc['I_prime_mean'] - np.mean(I[odds])
    print(f"\n   Prime 'kick' (I_prime - I_mean): {prime_kick:.6f}")
    print(f"   Prime density on odds: {pc['prime_fraction_of_odds']:.4f}")
    
    # Time between primes on average
    avg_prime_gap = 1 / pc['prime_fraction_of_odds']
    print(f"   Average gap between odd primes: {avg_prime_gap:.1f} odd numbers")
    
    # How much does E decay between primes?
    # E decays by λ² per odd step (since E alternates through evens)
    decay_per_step = 0.99 ** 2
    decay_over_gap = decay_per_step ** avg_prime_gap
    print(f"   E decay over avg gap: {decay_over_gap:.4f}")
    
    # Equilibrium analysis
    print("\n   Equilibrium analysis:")
    print(f"   At equilibrium, prime kicks balance decay")
    print(f"   Prime kick × prime_density ≈ decay rate × E_mean")
    
    E_odd_mean = np.mean(E[odds])
    print(f"   Actual E_odd_mean: {E_odd_mean:.6f}")
    
    print("\n" + "-" * 50)
    print("HYPOTHESIS: φ from asymmetric run lengths")
    print("-" * 50)
    
    print(f"""
    The Gaussian AR(1) model predicts frac ≈ 0.52 because it assumes
    symmetric positive/negative runs.
    
    But actual data shows:
    - Mean positive run: {persist.get('mean_positive_run', 'N/A'):.2f}
    - Mean negative run: {persist.get('mean_negative_run', 'N/A'):.2f}
    
    If positive runs are longer on average, frac(E>0) > 0.5
    
    The asymmetry comes from prime structure:
    - Primes inject LARGE positive kicks
    - Composites inject SMALL negative drift
    - This creates longer positive runs
    
    The specific value 1/φ may come from the prime density relationship
    to the kick/decay balance.
    """)
    
    # Final check: weighted time
    print("\n" + "-" * 50)
    print("WEIGHTED TIME ANALYSIS")
    print("-" * 50)
    
    # If positive runs are longer, weighted average differs from count
    E_odd = E[odds]
    
    # Time-weighted average of sign
    time_positive = np.sum(E_odd > 0)
    time_negative = np.sum(E_odd < 0)
    time_zero = np.sum(E_odd == 0)
    total = len(E_odd)
    
    print(f"   Time positive: {time_positive} ({time_positive/total:.4f})")
    print(f"   Time negative: {time_negative} ({time_negative/total:.4f})")
    print(f"   Time at zero: {time_zero}")
    
    print(f"\n   Ratio positive/negative: {time_positive/time_negative:.4f}")
    print(f"   Expected ratio for φ: {0.618034 / (1-0.618034):.4f}")
    
    results['weighted_time'] = {
        'time_positive': int(time_positive),
        'time_negative': int(time_negative),
        'ratio': float(time_positive/time_negative)
    }
    
    # Save
    trace_dir = Path(__file__).parent.parent / 'traces'
    trace_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trace_file = trace_dir / f'exp_23_real_mechanism_{timestamp}.json'
    
    # Convert any remaining numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    results_clean = json.loads(json.dumps(results, default=convert))
    
    with open(trace_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\nTrace saved: {trace_file.name}")
    
    return results

if __name__ == '__main__':
    main()
