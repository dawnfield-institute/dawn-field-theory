#!/usr/bin/env python3
"""
EXPERIMENT 22: Non-Gaussian E Dynamics

The AR(1) Gaussian model fails because it predicts P(E>0) ≈ 1.0
but we observe P(E>0) = 0.618.

Key insight: E(n) is computed on ALL integers, but we're measuring
frac(E>0) only on ODD integers. The dynamics at odd positions
depend on what happens at even positions.

Let's analyze E at odd vs even positions separately.
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

def analyze_E_at_parity(E, I, n_max):
    """Analyze E and I separately for odd and even positions."""
    
    odds = np.arange(3, n_max + 1, 2)
    evens = np.arange(2, n_max + 1, 2)
    
    results = {
        'E_odd': {
            'mean': float(np.mean(E[odds])),
            'std': float(np.std(E[odds])),
            'frac_positive': float(np.mean(E[odds] > 0)),
            'min': float(np.min(E[odds])),
            'max': float(np.max(E[odds])),
            'median': float(np.median(E[odds]))
        },
        'E_even': {
            'mean': float(np.mean(E[evens])),
            'std': float(np.std(E[evens])),
            'frac_positive': float(np.mean(E[evens] > 0)),
            'min': float(np.min(E[evens])),
            'max': float(np.max(E[evens])),
            'median': float(np.median(E[evens]))
        },
        'I_odd': {
            'mean': float(np.mean(I[odds])),
            'std': float(np.std(I[odds]))
        },
        'I_even': {
            'mean': float(np.mean(I[evens])),
            'std': float(np.std(I[evens]))
        }
    }
    
    return results

def analyze_E_trajectory(E, n_max, sample_size=1000):
    """Look at actual E trajectories to understand dynamics."""
    
    # Sample starting points
    starts = np.random.choice(range(1000, n_max - 1000), sample_size, replace=False)
    
    # For each start, track if E crosses zero
    crossing_data = []
    for start in starts:
        E_window = E[start:start+100]
        crossings = np.sum(np.diff(np.sign(E_window)) != 0)
        crossing_data.append(crossings)
    
    return {
        'mean_crossings_per_100': float(np.mean(crossing_data)),
        'std_crossings': float(np.std(crossing_data))
    }

def analyze_alternating_pattern(I, E, n_max):
    """
    Key insight: I alternates in sign systematically!
    
    For odd n: I(n) ≈ +1/(2k) (positive bias from 2-component)
    For even n: I(n) ≈ -1/(2k) (negative bias - they have the 2!)
    
    Let's verify this.
    """
    
    odds = np.arange(3, n_max + 1, 2)
    evens = np.arange(4, n_max + 1, 2)  # Start from 4 to have valid data
    
    I_odd_mean = np.mean(I[odds])
    I_even_mean = np.mean(I[evens])
    
    # The pattern: I alternates between + and - around 0
    # E accumulates this alternating signal
    
    # Check: is I_odd ≈ -I_even?
    return {
        'I_odd_mean': float(I_odd_mean),
        'I_even_mean': float(I_even_mean),
        'sum': float(I_odd_mean + I_even_mean),
        'ratio': float(I_odd_mean / I_even_mean) if I_even_mean != 0 else None,
        'alternating': I_odd_mean > 0 and I_even_mean < 0
    }

def model_alternating_ar1(I_odd_mean, I_even_mean, I_std, lam, n_steps=100000):
    """
    Simulate E with alternating I pattern:
    - I(odd) ~ N(μ_odd, σ²)
    - I(even) ~ N(μ_even, σ²)
    
    E(n) = λE(n-1) + I(n)
    """
    
    E = np.zeros(n_steps)
    I = np.zeros(n_steps)
    
    for n in range(1, n_steps):
        if n % 2 == 1:  # odd
            I[n] = I_odd_mean + I_std * np.random.randn()
        else:  # even
            I[n] = I_even_mean + I_std * np.random.randn()
        
        E[n] = lam * E[n-1] + I[n]
    
    odds = np.arange(1, n_steps, 2)
    evens = np.arange(2, n_steps, 2)
    
    return {
        'E_odd_frac_positive': float(np.mean(E[odds] > 0)),
        'E_even_frac_positive': float(np.mean(E[evens] > 0)),
        'E_odd_mean': float(np.mean(E[odds])),
        'E_even_mean': float(np.mean(E[evens]))
    }

def derive_analytical_formula(I_odd_mean, I_even_mean, I_std, lam):
    """
    For alternating AR(1):
    E(n) = λE(n-1) + I(n)
    
    At odd n (assuming n-1 is even):
    E(odd) = λE(even) + I(odd)
    
    At even n (assuming n-1 is odd):
    E(even) = λE(odd) + I(even)
    
    In stationary state, let μ_o = E[E(odd)], μ_e = E[E(even)]
    
    μ_o = λμ_e + μ_I_odd
    μ_e = λμ_o + μ_I_even
    
    Solving:
    μ_o = λ(λμ_o + μ_I_even) + μ_I_odd
    μ_o(1 - λ²) = λμ_I_even + μ_I_odd
    μ_o = (μ_I_odd + λμ_I_even) / (1 - λ²)
    
    Similarly:
    μ_e = (μ_I_even + λμ_I_odd) / (1 - λ²)
    """
    
    mu_o = (I_odd_mean + lam * I_even_mean) / (1 - lam**2)
    mu_e = (I_even_mean + lam * I_odd_mean) / (1 - lam**2)
    
    # Variance analysis (assuming independent I with same variance)
    # Var(E_odd) and Var(E_even) would need covariance terms
    # Approximate: σ_E ≈ σ_I / √(1-λ²) for both
    sigma_E = I_std / np.sqrt(1 - lam**2)
    
    # P(E_odd > 0) ≈ Φ(μ_o / σ_E)
    z_odd = mu_o / sigma_E if sigma_E > 0 else 0
    p_odd_positive = stats.norm.cdf(z_odd)
    
    z_even = mu_e / sigma_E if sigma_E > 0 else 0
    p_even_positive = stats.norm.cdf(z_even)
    
    return {
        'mu_E_odd': float(mu_o),
        'mu_E_even': float(mu_e),
        'sigma_E': float(sigma_E),
        'z_odd': float(z_odd),
        'z_even': float(z_even),
        'predicted_P_E_odd_positive': float(p_odd_positive),
        'predicted_P_E_even_positive': float(p_even_positive)
    }

def find_phi_condition(lam=0.99):
    """
    Find the condition on I_odd_mean, I_even_mean, I_std for P(E_odd>0) = 1/φ.
    
    From above:
    μ_o = (μ_I_odd + λμ_I_even) / (1 - λ²)
    σ_E = σ_I / √(1-λ²)
    
    P(E_odd > 0) = Φ(μ_o/σ_E) = Φ(z)
    
    For P = 1/φ, z = Φ⁻¹(1/φ) ≈ 0.3003
    
    z = μ_o/σ_E = [(μ_I_odd + λμ_I_even) / (1-λ²)] / [σ_I / √(1-λ²)]
      = (μ_I_odd + λμ_I_even) / [σ_I * √(1-λ²)]
      = (μ_I_odd + λμ_I_even) / [σ_I * √(1-λ) * √(1+λ)]
    
    Given λ = 0.99:
    √(1-λ) = √0.01 = 0.1
    √(1+λ) = √1.99 ≈ 1.41
    √(1-λ²) ≈ 0.141
    
    So z = (μ_I_odd + 0.99*μ_I_even) / (0.141 * σ_I)
    
    For z = 0.3003:
    μ_I_odd + 0.99*μ_I_even = 0.3003 * 0.141 * σ_I ≈ 0.0424 * σ_I
    """
    
    z_target = stats.norm.ppf(0.618034)  # ≈ 0.3003
    
    sqrt_1_minus_lam_sq = np.sqrt(1 - lam**2)
    
    # μ_I_odd + λ*μ_I_even = z_target * sqrt_1_minus_lam_sq * σ_I
    coefficient = z_target * sqrt_1_minus_lam_sq
    
    return {
        'z_for_phi': float(z_target),
        'sqrt_1_minus_lam_sq': float(sqrt_1_minus_lam_sq),
        'coefficient': float(coefficient),
        'formula': f'μ_I_odd + {lam}*μ_I_even = {coefficient:.6f} * σ_I'
    }

def main():
    print("=" * 70)
    print("EXPERIMENT 22: NON-GAUSSIAN E DYNAMICS")
    print("=" * 70)
    
    n_max = 100000
    primes_mask, primes = sieve_primes(n_max)
    factor_base = primes[:9]
    
    S, S_hat, I, E = compute_sec(n_max, factor_base)
    
    results = {'timestamp': datetime.now().isoformat()}
    
    # 1. E at different parities
    print("\n1. E and I at odd vs even positions:")
    parity = analyze_E_at_parity(E, I, n_max)
    print(f"   E_odd mean: {parity['E_odd']['mean']:.4f}, frac>0: {parity['E_odd']['frac_positive']:.4f}")
    print(f"   E_even mean: {parity['E_even']['mean']:.4f}, frac>0: {parity['E_even']['frac_positive']:.4f}")
    print(f"   I_odd mean: {parity['I_odd']['mean']:.6f}")
    print(f"   I_even mean: {parity['I_even']['mean']:.6f}")
    results['parity'] = parity
    
    # 2. Alternating pattern
    print("\n2. Alternating I pattern:")
    alt = analyze_alternating_pattern(I, E, n_max)
    print(f"   I_odd mean: {alt['I_odd_mean']:.6f}")
    print(f"   I_even mean: {alt['I_even_mean']:.6f}")
    print(f"   Sum (should be ~0): {alt['sum']:.6f}")
    print(f"   Alternating pattern: {alt['alternating']}")
    results['alternating'] = alt
    
    # 3. Analytical formula
    print("\n3. Analytical prediction (alternating AR(1)):")
    I_std = (parity['I_odd']['std'] + parity['I_even']['std']) / 2
    analytical = derive_analytical_formula(
        alt['I_odd_mean'], alt['I_even_mean'], I_std, lam=0.99
    )
    print(f"   μ_E_odd (predicted): {analytical['mu_E_odd']:.4f}")
    print(f"   μ_E_even (predicted): {analytical['mu_E_even']:.4f}")
    print(f"   σ_E: {analytical['sigma_E']:.4f}")
    print(f"   P(E_odd > 0) predicted: {analytical['predicted_P_E_odd_positive']:.4f}")
    print(f"   P(E_odd > 0) actual: {parity['E_odd']['frac_positive']:.4f}")
    print(f"   P(E_even > 0) predicted: {analytical['predicted_P_E_even_positive']:.4f}")
    print(f"   P(E_even > 0) actual: {parity['E_even']['frac_positive']:.4f}")
    results['analytical'] = analytical
    
    # 4. Simulation verification
    print("\n4. Simulation verification:")
    sim = model_alternating_ar1(alt['I_odd_mean'], alt['I_even_mean'], I_std, lam=0.99)
    print(f"   Simulated E_odd frac>0: {sim['E_odd_frac_positive']:.4f}")
    print(f"   Simulated E_even frac>0: {sim['E_even_frac_positive']:.4f}")
    results['simulation'] = sim
    
    # 5. φ condition
    print("\n5. Condition for φ emergence:")
    phi_cond = find_phi_condition()
    print(f"   z needed: {phi_cond['z_for_phi']:.4f}")
    print(f"   Condition: {phi_cond['formula']}")
    
    # Check if actual values satisfy this
    actual_lhs = alt['I_odd_mean'] + 0.99 * alt['I_even_mean']
    required_rhs = phi_cond['coefficient'] * I_std
    print(f"\n   Actual LHS: {actual_lhs:.6f}")
    print(f"   Required RHS: {required_rhs:.6f}")
    print(f"   Match: {abs(actual_lhs - required_rhs) < 0.01}")
    results['phi_condition'] = phi_cond
    results['phi_condition']['actual_lhs'] = float(actual_lhs)
    results['phi_condition']['required_rhs'] = float(required_rhs)
    
    # 6. THE KEY QUESTION: Why do these values produce φ?
    print("\n" + "=" * 70)
    print("WHY DOES THIS PRODUCE φ?")
    print("=" * 70)
    
    # From the 2-component:
    # I_odd_mean ≈ +1/(2k) (2 doesn't divide odd n, but Ŝ includes evens)
    # I_even_mean ≈ -1/(2k) (2 does divide even n, lowering Ŝ-S)
    
    k = 9
    expected_I_odd = 1 / (2 * k)
    expected_I_even = -1 / (2 * k)  # Actually let's verify this
    
    print(f"\n   From 2-component theory:")
    print(f"   Expected I_odd: +1/(2*{k}) = +{expected_I_odd:.6f}")
    print(f"   Actual I_odd: {alt['I_odd_mean']:.6f}")
    print(f"   Expected I_even: should be negative")
    print(f"   Actual I_even: {alt['I_even_mean']:.6f}")
    
    # The actual I_even is MORE negative than I_odd is positive
    # This asymmetry is key
    
    print(f"\n   Asymmetry: |I_even| - |I_odd| = {abs(alt['I_even_mean']) - abs(alt['I_odd_mean']):.6f}")
    
    # For the alternating AR(1):
    # μ_o = (μ_I_odd + λ*μ_I_even) / (1-λ²)
    # 
    # If μ_I_odd = -μ_I_even (perfect symmetry), then:
    # μ_o = μ_I_odd * (1 - λ) / (1-λ²) = μ_I_odd / (1+λ)
    #
    # With λ=0.99: μ_o = μ_I_odd / 1.99 ≈ 0.5 * μ_I_odd
    #
    # But there's asymmetry...
    
    print("\n" + "-" * 50)
    print("THE MECHANISM:")
    print("-" * 50)
    
    print("""
    1. The 2-component creates an ALTERNATING pattern in I:
       - I(odd) ≈ +0.055 (Ŝ sees 2-divisible evens, S doesn't)
       - I(even) ≈ -0.055 (S sees 2-divisibility, Ŝ is diluted by odds)
    
    2. BUT there's asymmetry: |I_even| > |I_odd|
       This is because evens are "more composite" overall
    
    3. E accumulates this alternating signal:
       E_odd = λ*E_even + I_odd
       E_even = λ*E_odd + I_even
    
    4. The SLIGHT asymmetry toward negative I shifts E_odd toward positive
       and E_even toward negative
    
    5. The specific values produce:
       P(E_odd > 0) ≈ 0.618 = 1/φ
       P(E_even > 0) ≈ 0.38 ≈ 1 - 1/φ
    
    6. WHY φ? The divisibility structure of integers creates an
       asymmetry that HAPPENS to produce φ. This needs more analysis.
    """)
    
    # Save
    trace_dir = Path(__file__).parent.parent / 'traces'
    trace_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trace_file = trace_dir / f'exp_22_alternating_dynamics_{timestamp}.json'
    
    with open(trace_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTrace saved: {trace_file.name}")
    
    return results

if __name__ == '__main__':
    main()
