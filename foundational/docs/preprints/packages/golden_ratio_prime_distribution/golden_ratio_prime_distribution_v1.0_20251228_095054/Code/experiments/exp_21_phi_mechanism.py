#!/usr/bin/env python3
"""
EXPERIMENT 21: Deriving the φ Mechanism

Goal: Understand WHY frac(E>0) → 1/φ on the odd manifold.

Approach:
1. Decompose E(n) into its components
2. Analyze the statistical properties of I(n) for odds
3. Model E as a stochastic process and derive the positive fraction
4. Identify where φ enters mathematically
"""

import numpy as np
from scipy import stats
from scipy.special import erfc
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

def compute_full_sec(n_max, factor_base, window=101, lam=0.99):
    """Compute SEC with all intermediate values."""
    k = len(factor_base)
    
    S = np.zeros(n_max + 1)
    S_hat = np.zeros(n_max + 1)
    I = np.zeros(n_max + 1)
    E = np.zeros(n_max + 1)
    
    # Compute S(n)
    for n in range(2, n_max + 1):
        S[n] = sum(1 for p in factor_base if n % p == 0) / k
    
    # Compute Ŝ(n) with sliding window
    half = window // 2
    for n in range(2, n_max + 1):
        lo = max(2, n - half)
        hi = min(n_max, n + half)
        S_hat[n] = S[lo:hi+1].mean()
    
    # Compute I(n) and E(n)
    for n in range(2, n_max + 1):
        I[n] = S_hat[n] - S[n]
        E[n] = lam * E[n-1] + I[n]
    
    return S, S_hat, I, E

def analyze_impulse_distribution(I, odds, primes_mask):
    """Analyze statistical properties of I(n) on odds."""
    I_odd = I[odds]
    
    # Separate by prime/composite
    is_prime_odd = primes_mask[odds]
    I_prime = I_odd[is_prime_odd]
    I_composite = I_odd[~is_prime_odd]
    
    return {
        'I_odd_mean': float(np.mean(I_odd)),
        'I_odd_std': float(np.std(I_odd)),
        'I_odd_skew': float(stats.skew(I_odd)),
        'I_odd_kurtosis': float(stats.kurtosis(I_odd)),
        'I_prime_mean': float(np.mean(I_prime)),
        'I_prime_std': float(np.std(I_prime)),
        'I_composite_mean': float(np.mean(I_composite)),
        'I_composite_std': float(np.std(I_composite)),
        'prime_fraction': float(np.mean(is_prime_odd)),
        'I_positive_fraction': float(np.mean(I_odd > 0))
    }

def theoretical_E_positive_fraction(I_mean, I_std, lam):
    """
    For AR(1) process E(n) = λE(n-1) + I(n) with I~N(μ,σ²):
    
    Stationary distribution: E ~ N(μ/(1-λ), σ²/(1-λ²))
    
    P(E > 0) = 1 - Φ(-μ_E/σ_E) = Φ(μ_E/σ_E)
    
    where μ_E = μ/(1-λ), σ_E = σ/√(1-λ²)
    """
    mu_E = I_mean / (1 - lam)
    sigma_E = I_std / np.sqrt(1 - lam**2)
    
    # P(E > 0) = P(Z > -μ_E/σ_E) = Φ(μ_E/σ_E)
    z = mu_E / sigma_E if sigma_E > 0 else 0
    p_positive = stats.norm.cdf(z)
    
    return {
        'mu_E': mu_E,
        'sigma_E': sigma_E,
        'z_score': z,
        'theoretical_positive_frac': p_positive
    }

def analyze_S_structure(S, odds, factor_base):
    """Understand how S(n) behaves on odds."""
    S_odd = S[odds]
    
    # S(n) for odd n can only take values k/|B| where k = 0,1,...,|B|-1
    # (since 2 never divides odd n, the 2-component is always 0)
    
    # Count how many factor base primes (excluding 2) divide each odd n
    fb_no_2 = [p for p in factor_base if p != 2]
    
    divisibility_counts = []
    for n in odds[:10000]:  # Sample
        count = sum(1 for p in factor_base if n % p == 0)
        divisibility_counts.append(count)
    
    counts_dist = np.bincount(divisibility_counts, minlength=len(factor_base)+1)
    counts_prob = counts_dist / len(divisibility_counts)
    
    # For odd n: 2 never divides, so S(odd) = (divisors among {3,5,7,...}) / k
    # But S_hat includes even neighbors where 2 DOES divide
    
    return {
        'S_odd_mean': float(np.mean(S_odd)),
        'S_odd_std': float(np.std(S_odd)),
        'divisibility_distribution': counts_prob.tolist(),
        'factor_base_size': len(factor_base),
        'expected_S_odd': float(np.mean(counts_prob * np.arange(len(counts_prob))) / len(factor_base))
    }

def analyze_S_hat_for_odds(S, odds, window=101):
    """
    Key insight: Ŝ(n) for odd n averages over a window that includes EVEN neighbors.
    
    For odd n, the window [n-50, n+50] contains ~50 evens and ~51 odds.
    The evens have S(even) that includes the 2-component.
    """
    half = window // 2
    n_max = len(S) - 1
    
    # For each odd n, compute contribution from even vs odd neighbors
    even_contrib = []
    odd_contrib = []
    
    for n in odds[:5000]:  # Sample
        lo = max(2, n - half)
        hi = min(n_max, n + half)
        
        neighbors = np.arange(lo, hi + 1)
        even_neighbors = neighbors[neighbors % 2 == 0]
        odd_neighbors = neighbors[neighbors % 2 == 1]
        
        S_even_mean = S[even_neighbors].mean() if len(even_neighbors) > 0 else 0
        S_odd_mean = S[odd_neighbors].mean() if len(odd_neighbors) > 0 else 0
        
        even_contrib.append(S_even_mean)
        odd_contrib.append(S_odd_mean)
    
    return {
        'S_hat_even_contrib_mean': float(np.mean(even_contrib)),
        'S_hat_odd_contrib_mean': float(np.mean(odd_contrib)),
        'even_minus_odd': float(np.mean(even_contrib) - np.mean(odd_contrib)),
        'note': 'Even neighbors have higher S due to 2-divisibility'
    }

def derive_phi_connection():
    """
    Attempt to derive why frac(E>0) = 1/φ.
    
    Key equations:
    - S(odd) = (divisors among {3,5,7,...,p_k}) / k
    - S(even) = (1 + divisors among {3,5,7,...,p_k}) / k  (2 always divides)
    - Ŝ(odd) ≈ 0.5 * S_even_mean + 0.5 * S_odd_mean
    - I(odd) = Ŝ(odd) - S(odd) ≈ 0.5 * (S_even_mean - S_odd_mean)
    
    Since S_even_mean - S_odd_mean = 1/k (the 2-component):
    I(odd) ≈ 1/(2k) for "typical" odd n
    
    But this is modified by prime/composite structure...
    """
    
    # The 2-component creates a systematic positive bias in I for odds
    # because Ŝ (which includes even neighbors) > S(odd) on average
    
    # For primes: S(prime) = 0 or 1/k (only the prime itself)
    # For composites: S(composite) varies
    
    # The question: why does the distribution of E land at 1/φ positive?
    
    return {
        'hypothesis': 'The 2-component creates systematic I > 0 bias for odds',
        'mechanism': 'Ŝ includes even neighbors with 2-divisibility, S(odd) excludes it',
        'expected_bias': '1/(2k) for typical odd n',
        'phi_question': 'Why does this bias produce exactly 1/φ positive fraction?'
    }

def test_2_component_hypothesis(n_max, primes):
    """Test: is the 2-component responsible for the signal?"""
    
    results = {}
    odds = np.arange(3, n_max + 1, 2)
    
    # Case 1: Factor base includes 2
    fb_with_2 = primes[:9]
    S1, S_hat1, I1, E1 = compute_full_sec(n_max, fb_with_2)
    frac1 = np.mean(E1[odds] > 0)
    
    # Case 2: Factor base excludes 2 (shift to {3,5,7,...,29})
    fb_without_2 = primes[1:10]  # {3,5,7,11,13,17,19,23,29}
    S2, S_hat2, I2, E2 = compute_full_sec(n_max, fb_without_2)
    frac2 = np.mean(E2[odds] > 0)
    
    # Analyze the I distributions
    I1_odd_mean = np.mean(I1[odds])
    I2_odd_mean = np.mean(I2[odds])
    
    results['with_2'] = {
        'factor_base': list(fb_with_2),
        'frac_E_positive': float(frac1),
        'I_odd_mean': float(I1_odd_mean),
        'error_vs_phi': float(abs(frac1 - 0.618034))
    }
    
    results['without_2'] = {
        'factor_base': list(fb_without_2),
        'frac_E_positive': float(frac2),
        'I_odd_mean': float(I2_odd_mean),
        'error_vs_phi': float(abs(frac2 - 0.618034))
    }
    
    # The key difference
    results['I_mean_difference'] = float(I1_odd_mean - I2_odd_mean)
    results['hypothesis_supported'] = I1_odd_mean > I2_odd_mean
    
    return results

def analyze_I_bias_source(n_max, factor_base, window=101):
    """
    Decompose I(n) = Ŝ(n) - S(n) for odd n.
    
    Ŝ(n) = (1/W) * Σ S(m) over window
         = (1/W) * [Σ S(even) + Σ S(odd)]
         
    For a window centered at odd n:
    - ~W/2 even neighbors, ~W/2 odd neighbors
    - S(even) = (1 + odd_factors) / k
    - S(odd) = odd_factors / k
    
    So: Ŝ(odd n) ≈ (1/2) * [(1 + μ_odd)/k + μ_odd/k]
                 = (1/2) * [1/k + 2*μ_odd/k]
                 = 1/(2k) + μ_odd/k
    
    And: I(odd n) = Ŝ - S ≈ 1/(2k) + μ_odd/k - S(n)
                  = 1/(2k) + (μ_odd - S(n)*k)/k
    
    The 1/(2k) term is the systematic bias from the 2-component!
    """
    
    k = len(factor_base)
    S, S_hat, I, E = compute_full_sec(n_max, factor_base, window)
    
    odds = np.arange(3, n_max + 1, 2)
    
    # Theoretical bias
    theoretical_bias = 1 / (2 * k)
    
    # Empirical I mean for odds
    empirical_I_mean = np.mean(I[odds])
    
    # What's the contribution beyond the 2-bias?
    residual = empirical_I_mean - theoretical_bias
    
    return {
        'factor_base_size': k,
        'theoretical_2_bias': float(theoretical_bias),
        'empirical_I_mean_odd': float(empirical_I_mean),
        'residual': float(residual),
        'ratio': float(empirical_I_mean / theoretical_bias) if theoretical_bias > 0 else None
    }

def model_E_as_random_walk(I, odds, lam=0.99):
    """
    Model E(n) for odd n as a correlated random walk.
    
    E(n) = λE(n-1) + I(n)
    
    For odd n, we skip even indices, so it's like:
    E(n) = λ²E(n-2) + λI(n-1) + I(n)  [but n-1 is even!]
    
    This is more complex than standard AR(1)...
    """
    
    I_odd = I[odds]
    
    # Compute E restricted to odds (but E evolves through evens too)
    # The actual E at odd positions depends on the full sequence
    
    # Autocorrelation of I at odd positions
    autocorr = []
    for lag in range(1, 20):
        if lag < len(I_odd):
            corr = np.corrcoef(I_odd[:-lag], I_odd[lag:])[0, 1]
            autocorr.append(float(corr))
    
    return {
        'I_odd_autocorrelation': autocorr,
        'note': 'E at odds depends on E at evens, which depends on I at evens'
    }

def compute_exact_positive_fraction(I_mean, I_std, lam, method='gaussian'):
    """
    For E(n) = λE(n-1) + I(n) with stationary I:
    
    E converges to stationary distribution with:
    μ_E = μ_I / (1-λ)
    σ_E = σ_I / √(1-λ²)
    
    If I is Gaussian, E is Gaussian, and P(E>0) = Φ(μ_E/σ_E)
    """
    
    mu_E = I_mean / (1 - lam)
    sigma_E = I_std / np.sqrt(1 - lam**2)
    
    if method == 'gaussian':
        z = mu_E / sigma_E if sigma_E > 0 else 0
        p_pos = stats.norm.cdf(z)
    else:
        p_pos = None
    
    # For P(E>0) = 1/φ ≈ 0.618, we need:
    # Φ(z) = 0.618
    # z = Φ⁻¹(0.618) ≈ 0.30
    
    z_for_phi = stats.norm.ppf(0.618034)
    
    # So we need μ_E/σ_E = 0.30
    # μ_I/(1-λ) / [σ_I/√(1-λ²)] = 0.30
    # μ_I * √(1-λ²) / [σ_I * (1-λ)] = 0.30
    # μ_I/σ_I * √(1+λ)/√(1-λ) = 0.30
    # μ_I/σ_I * √(1.99/0.01) = 0.30
    # μ_I/σ_I * 14.1 = 0.30
    # μ_I/σ_I = 0.0213
    
    required_I_ratio = z_for_phi * np.sqrt(1 - lam) / np.sqrt(1 + lam)
    
    return {
        'mu_E': float(mu_E),
        'sigma_E': float(sigma_E),
        'z_score': float(mu_E / sigma_E) if sigma_E > 0 else 0,
        'predicted_positive_frac': float(p_pos) if p_pos else None,
        'z_for_phi': float(z_for_phi),
        'required_I_mean_over_std_for_phi': float(required_I_ratio),
        'actual_I_mean_over_std': float(I_mean / I_std) if I_std > 0 else 0
    }

def main():
    print("=" * 70)
    print("EXPERIMENT 21: DERIVING THE φ MECHANISM")
    print("=" * 70)
    
    n_max = 100000
    primes_mask, primes = sieve_primes(n_max)
    factor_base = primes[:9]  # {2,3,5,7,11,13,17,19,23}
    
    print(f"\nFactor base: {list(factor_base)}")
    print(f"n_max: {n_max:,}")
    
    results = {'timestamp': datetime.now().isoformat()}
    
    # 1. Compute SEC
    print("\n1. Computing SEC...")
    S, S_hat, I, E = compute_full_sec(n_max, factor_base)
    odds = np.arange(3, n_max + 1, 2)
    
    frac_E_pos = np.mean(E[odds] > 0)
    print(f"   frac(E>0) on odds: {frac_E_pos:.6f}")
    print(f"   Error vs 1/φ: {abs(frac_E_pos - 0.618034):.6f}")
    
    # 2. Analyze I distribution
    print("\n2. Analyzing I(n) distribution on odds...")
    I_stats = analyze_impulse_distribution(I, odds, primes_mask)
    print(f"   I mean (odd): {I_stats['I_odd_mean']:.6f}")
    print(f"   I std (odd): {I_stats['I_odd_std']:.6f}")
    print(f"   I mean (primes): {I_stats['I_prime_mean']:.6f}")
    print(f"   I mean (composites): {I_stats['I_composite_mean']:.6f}")
    results['I_distribution'] = I_stats
    
    # 3. Theoretical prediction
    print("\n3. Theoretical AR(1) prediction...")
    theory = compute_exact_positive_fraction(
        I_stats['I_odd_mean'], 
        I_stats['I_odd_std'], 
        lam=0.99
    )
    print(f"   μ_E (stationary): {theory['mu_E']:.4f}")
    print(f"   σ_E (stationary): {theory['sigma_E']:.4f}")
    print(f"   z-score: {theory['z_score']:.4f}")
    print(f"   Predicted P(E>0): {theory['predicted_positive_frac']:.6f}")
    print(f"   Actual P(E>0): {frac_E_pos:.6f}")
    print(f"   z needed for φ: {theory['z_for_phi']:.4f}")
    print(f"   Required I_mean/I_std: {theory['required_I_mean_over_std_for_phi']:.6f}")
    print(f"   Actual I_mean/I_std: {theory['actual_I_mean_over_std']:.6f}")
    results['theory'] = theory
    
    # 4. Test 2-component hypothesis
    print("\n4. Testing 2-component hypothesis...")
    two_test = test_2_component_hypothesis(n_max, primes)
    print(f"   With 2: frac(E>0) = {two_test['with_2']['frac_E_positive']:.4f}")
    print(f"   Without 2: frac(E>0) = {two_test['without_2']['frac_E_positive']:.4f}")
    print(f"   I_mean difference: {two_test['I_mean_difference']:.6f}")
    print(f"   Hypothesis supported: {two_test['hypothesis_supported']}")
    results['two_component_test'] = two_test
    
    # 5. Analyze I bias source
    print("\n5. Analyzing I bias from 2-component...")
    bias = analyze_I_bias_source(n_max, factor_base)
    print(f"   Theoretical 2-bias (1/2k): {bias['theoretical_2_bias']:.6f}")
    print(f"   Empirical I mean: {bias['empirical_I_mean_odd']:.6f}")
    print(f"   Ratio: {bias['ratio']:.4f}")
    results['bias_analysis'] = bias
    
    # 6. The key question: why φ?
    print("\n" + "=" * 70)
    print("KEY ANALYSIS: WHY φ?")
    print("=" * 70)
    
    # The AR(1) model predicts P(E>0) based on I_mean/I_std
    # For P(E>0) = 0.618, we need z ≈ 0.30
    # This means I_mean/I_std ≈ 0.021 (given λ=0.99)
    
    actual_ratio = I_stats['I_odd_mean'] / I_stats['I_odd_std']
    predicted_ratio = theory['required_I_mean_over_std_for_phi']
    
    print(f"\nFor frac(E>0) = 1/φ, we need I_mean/I_std = {predicted_ratio:.6f}")
    print(f"Actual I_mean/I_std = {actual_ratio:.6f}")
    print(f"Match: {abs(actual_ratio - predicted_ratio) < 0.01}")
    
    # What determines I_mean/I_std?
    print("\nDecomposing I_mean and I_std:")
    print(f"  I_mean = {I_stats['I_odd_mean']:.6f}")
    print(f"  I_std = {I_stats['I_odd_std']:.6f}")
    
    # I_mean ≈ 1/(2k) from the 2-component
    # I_std depends on variance of S(n) for odds
    
    k = len(factor_base)
    predicted_I_mean = 1 / (2 * k)
    print(f"\n  Predicted I_mean from 2-component: 1/(2*{k}) = {predicted_I_mean:.6f}")
    print(f"  Actual I_mean: {I_stats['I_odd_mean']:.6f}")
    
    # For φ to emerge, we need a specific relationship between k and I_std
    # I_mean/I_std = 0.021
    # 1/(2k) / I_std = 0.021
    # I_std = 1/(2k * 0.021) = 1/(0.042k)
    
    required_I_std = 1 / (2 * k * predicted_ratio)
    print(f"\n  For φ, required I_std = {required_I_std:.6f}")
    print(f"  Actual I_std = {I_stats['I_odd_std']:.6f}")
    
    # 7. What controls I_std?
    print("\n" + "-" * 50)
    print("What determines I_std?")
    print("-" * 50)
    
    # I_std comes from variance in Ŝ - S
    # Var(I) ≈ Var(Ŝ) + Var(S) - 2Cov(Ŝ,S)
    # For large window, Var(Ŝ) → 0, so Var(I) ≈ Var(S)
    
    S_odd = S[odds]
    print(f"  Var(S) on odds: {np.var(S_odd):.6f}")
    print(f"  Std(S) on odds: {np.std(S_odd):.6f}")
    print(f"  Std(I) on odds: {I_stats['I_odd_std']:.6f}")
    
    # Var(S) for odds depends on how divisibility is distributed
    # S(odd) = (# of fb primes dividing n) / k
    # For independent divisibility with prob 1/p each:
    # Var(S) = (1/k²) * Σ p(1-p) where p = 1/p_i
    
    theoretical_var_S = sum((1/p) * (1 - 1/p) for p in factor_base if p > 2) / (k**2)
    print(f"  Theoretical Var(S) (independent): {theoretical_var_S:.6f}")
    print(f"  Actual Var(S): {np.var(S_odd):.6f}")
    
    results['mechanism'] = {
        'actual_I_ratio': float(actual_ratio),
        'required_I_ratio_for_phi': float(predicted_ratio),
        'match': bool(abs(actual_ratio - predicted_ratio) < 0.01),
        'I_mean_from_2_component': float(predicted_I_mean),
        'required_I_std_for_phi': float(required_I_std),
        'actual_I_std': float(I_stats['I_odd_std']),
        'S_var_theoretical': float(theoretical_var_S),
        'S_var_actual': float(np.var(S_odd))
    }
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("MECHANISM SUMMARY")
    print("=" * 70)
    
    print("""
    1. E(n) is an AR(1) process: E(n) = 0.99*E(n-1) + I(n)
    
    2. For AR(1) with Gaussian inputs, P(E>0) = Φ(μ_E/σ_E)
       where μ_E = μ_I/(1-λ), σ_E = σ_I/√(1-λ²)
    
    3. For P(E>0) = 1/φ ≈ 0.618:
       - Need z = Φ⁻¹(0.618) ≈ 0.30
       - Need μ_I/σ_I ≈ 0.021 (given λ=0.99)
    
    4. The 2-component provides μ_I ≈ 1/(2k):
       - Ŝ(odd) includes even neighbors where 2 divides
       - S(odd) never has 2 dividing
       - This creates systematic positive bias
    
    5. The divisibility variance provides σ_I:
       - σ_I ≈ √[Σ (1/p)(1-1/p)] / k
       - Depends on factor base composition
    
    6. φ emerges when 1/(2k) / σ_I ≈ 0.021
       - This is a specific relationship between k and the primes
       - Size k=9 appears to satisfy this relationship
    """)
    
    # Save results
    trace_dir = Path(__file__).parent.parent / 'traces'
    trace_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trace_file = trace_dir / f'exp_21_phi_mechanism_{timestamp}.json'
    
    with open(trace_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTrace saved: {trace_file.name}")
    
    return results

if __name__ == '__main__':
    main()
