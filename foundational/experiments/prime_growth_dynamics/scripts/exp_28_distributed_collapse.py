"""
exp_28: Distributed Collapse Hypothesis

Tests: "local Ξ = distributed γ + ln(φ)"

The hypothesis: Ξ = 1 + π/55 is the INSTANTANEOUS balance constant.
γ + ln(φ) = 1.0584 is the same constant ACCUMULATED over infinite smoothing.

The 0.12% gap may represent the difference between:
- Local: what balance looks like at any single step
- Global: what balance integrates to over the whole process

Test approach:
1. Track cumulative "balance cost" through sieve waves
2. See if it converges to γ + ln(φ)
3. Check if the per-step average equals Ξ

If true: Ξ and γ+ln(φ) are the same phenomenon at different scales.
If false: They're coincidentally similar but distinct.

Author: Dawn Field Institute
Date: February 5, 2026
"""

import math
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Constants
PHI = (1 + math.sqrt(5)) / 2
GAMMA = 0.5772156649015329
LN_PHI = math.log(PHI)
XI = 1 + math.pi / 55
GAMMA_PLUS_LN_PHI = GAMMA + LN_PHI


def sieve_with_tracking(limit: int) -> Dict:
    """
    Run sieve tracking information cost at each wave.
    
    Information cost per wave = log of probability mass eliminated.
    """
    # Track: which numbers are still "rough" (potentially prime)
    is_rough = [True] * (limit + 1)
    is_rough[0] = is_rough[1] = False
    
    total_rough = limit - 1  # Start with all numbers 2..limit
    
    waves = []
    cumulative_info_cost = 0
    cumulative_mertens_term = 0
    
    p = 2
    wave_num = 0
    
    while p * p <= limit:
        if is_rough[p]:
            wave_num += 1
            
            # Count how many we'll eliminate
            eliminated = 0
            for multiple in range(p * p, limit + 1, p):
                if is_rough[multiple]:
                    is_rough[multiple] = False
                    eliminated += 1
            
            # Information cost: proportion eliminated
            if total_rough > 0:
                prop_eliminated = eliminated / total_rough
                if prop_eliminated > 0:
                    # Information = -log(1 - prop) = log(1/(1-prop))
                    # This is bits of "surprise" at elimination
                    info_cost = -math.log(1 - prop_eliminated) if prop_eliminated < 1 else 0
                else:
                    info_cost = 0
            else:
                info_cost = 0
            
            # Mertens term: log(1 - 1/p)
            mertens_term = -math.log(1 - 1/p)
            
            # Update totals
            total_rough -= eliminated
            cumulative_info_cost += info_cost
            cumulative_mertens_term += mertens_term
            
            # Running "average" balance constant implied
            if wave_num > 0:
                avg_balance = 1 + cumulative_info_cost / wave_num
            else:
                avg_balance = 1
            
            waves.append({
                'wave': wave_num,
                'prime': p,
                'eliminated': eliminated,
                'remaining_rough': total_rough,
                'info_cost': float(info_cost),
                'cumulative_info': float(cumulative_info_cost),
                'mertens_term': float(mertens_term),
                'cumulative_mertens': float(cumulative_mertens_term),
                'avg_balance': float(avg_balance),
                'error_from_xi': float(100 * abs(avg_balance - XI) / XI),
                'error_from_gamma_ln_phi': float(100 * abs(avg_balance - GAMMA_PLUS_LN_PHI) / GAMMA_PLUS_LN_PHI)
            })
        
        p += 1
    
    # Final primes (remaining rough numbers)
    final_primes = sum(1 for i in range(2, limit + 1) if is_rough[i])
    
    return {
        'limit': limit,
        'num_waves': wave_num,
        'final_primes': final_primes,
        'waves': waves,
        'cumulative_info_cost': float(cumulative_info_cost),
        'cumulative_mertens': float(cumulative_mertens_term)
    }


def test_mertens_connection() -> Dict:
    """
    The Mertens product: ∏(1 - 1/p) ~ e^(-γ) / ln(x)
    
    Taking logs: Σ log(1 - 1/p) ~ -γ - ln(ln(x))
    
    So cumulative_mertens should approach γ as we go.
    """
    limits = [100, 1000, 10000, 100000, 1000000]
    
    results = []
    for limit in limits:
        # Sum of log(1 - 1/p) for p <= sqrt(limit)
        sqrt_limit = int(math.sqrt(limit))
        
        # Get primes up to sqrt(limit)
        is_prime = [True] * (sqrt_limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(sqrt_limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, sqrt_limit + 1, i):
                    is_prime[j] = False
        
        primes = [i for i in range(2, sqrt_limit + 1) if is_prime[i]]
        
        # Mertens sum
        mertens_sum = sum(-math.log(1 - 1/p) for p in primes)
        
        # Theoretical: γ + ln(ln(sqrt(limit))) = γ + ln(ln(limit)/2)
        # Actually: Σ(1/p) ~ ln(ln(x)) + M where M ≈ 0.2615 (Mertens constant)
        # And Σlog(1-1/p) = Σ(-1/p - 1/2p² - ...) ~ -ln(ln(x)) - γ + small
        
        # What we want to check: does mertens_sum relate to γ + ln(φ)?
        
        results.append({
            'limit': limit,
            'sqrt_limit': sqrt_limit,
            'num_primes_used': len(primes),
            'mertens_sum': float(mertens_sum),
            'ln_ln_sqrt': float(math.log(math.log(sqrt_limit))) if sqrt_limit > math.e else 0,
            'gamma': float(GAMMA),
            'mertens_minus_gamma': float(mertens_sum - GAMMA),
            'ratio_to_ln_phi': float((mertens_sum - GAMMA) / LN_PHI) if mertens_sum > GAMMA else 0
        })
    
    return {
        'description': 'Testing if Mertens sum relates to γ + ln(φ)',
        'results': results
    }


def test_local_vs_global() -> Dict:
    """
    The core test: is Ξ the local quantity and γ+ln(φ) the global?
    
    Approach:
    - Track "balance ratio" at each sieve step
    - See what it converges to
    """
    limit = 1000000
    data = sieve_with_tracking(limit)
    
    # Sample waves
    sample_indices = [0, 4, 9, 19, 49, 99, len(data['waves'])-1]
    sample_indices = [i for i in sample_indices if i < len(data['waves'])]
    
    samples = []
    for i in sample_indices:
        w = data['waves'][i]
        samples.append({
            'wave': w['wave'],
            'prime': w['prime'],
            'avg_balance': w['avg_balance'],
            'error_from_xi_%': w['error_from_xi'],
            'error_from_gamma_ln_phi_%': w['error_from_gamma_ln_phi']
        })
    
    # Final convergent
    final = data['waves'][-1]
    
    # Check: does per-wave info cost equal anything meaningful?
    per_wave_costs = [w['info_cost'] for w in data['waves']]
    mean_per_wave = np.mean(per_wave_costs)
    
    # The hypothesis: cumulative info cost / num_waves → something related to Ξ-1 or ln(φ)
    cumulative_div_waves = data['cumulative_info_cost'] / data['num_waves']
    
    return {
        'limit': limit,
        'num_waves': data['num_waves'],
        'samples': samples,
        'final_avg_balance': float(final['avg_balance']),
        'final_error_from_xi_%': float(final['error_from_xi']),
        'final_error_from_gamma_ln_phi_%': float(final['error_from_gamma_ln_phi']),
        'mean_per_wave_info_cost': float(mean_per_wave),
        'cumulative_info_div_waves': float(cumulative_div_waves),
        'cumulative_info_cost': float(data['cumulative_info_cost']),
        'hypothesis_test': {
            'xi_minus_1': float(XI - 1),
            'ln_phi': float(LN_PHI),
            'gamma': float(GAMMA),
            'cumulative_vs_xi_minus_1': float(data['cumulative_info_cost'] / (XI - 1)),
            'cumulative_vs_ln_phi': float(data['cumulative_info_cost'] / LN_PHI),
            'cumulative_vs_gamma': float(data['cumulative_info_cost'] / GAMMA)
        }
    }


def test_balance_at_each_prime() -> Dict:
    """
    What if Ξ appears at EACH prime, not accumulated?
    
    For each prime p, compute a local "balance" measure.
    See if it clusters around Ξ.
    """
    limit = 100000
    
    # Get primes
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime, compute: p / (p - 1) = 1 + 1/(p-1)
    # This is the "expansion factor" of the sieve at that prime
    expansion_factors = [p / (p - 1) for p in primes]
    
    # Cumulative product of expansion factors
    # ∏(p/(p-1)) = ∏(1/(1-1/p))^(-1)
    # This is related to Mertens product
    
    running_product = 1.0
    products = []
    for i, p in enumerate(primes[:100]):  # First 100 primes
        running_product *= (p / (p - 1))
        # Normalize by ln(p) somehow?
        normalized = running_product / math.log(p) if p > 1 else running_product
        products.append({
            'prime': p,
            'expansion': float(p / (p - 1)),
            'running_product': float(running_product),
            'running_product_over_ln_p': float(normalized),
            'e_gamma_times_ln_p': float(math.exp(GAMMA) * math.log(p))
        })
    
    # The limit: ∏(p/(p-1)) for p ≤ x ~ e^γ ln(x) (Mertens' third theorem)
    # So running_product / ln(last_prime) → e^γ ≈ 1.781
    
    return {
        'description': 'Testing expansion factors at each prime',
        'target_e_gamma': float(math.exp(GAMMA)),
        'samples': products[-10:],  # Last 10
        'final_product_over_ln': float(products[-1]['running_product_over_ln_p']),
        'error_from_e_gamma': float(100 * abs(products[-1]['running_product_over_ln_p'] - math.exp(GAMMA)) / math.exp(GAMMA))
    }


def test_xi_as_rate() -> Dict:
    """
    What if Ξ is the RATE of information gain per unit?
    
    Total info in primes up to x ≈ x / ln(x) (PNT)
    Total info cost of smoothing ≈ ???
    
    Ratio might be Ξ?
    """
    limits = [1000, 10000, 100000, 1000000]
    
    results = []
    for limit in limits:
        # Pi(x)
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        pi_x = sum(1 for i in range(2, limit + 1) if is_prime[i])
        
        # Information content of primes
        # I(primes) = Σ log(p) for primes p ≤ x
        # This is related to Chebyshev's θ(x) ~ x
        theta_x = sum(math.log(p) for p in range(2, limit + 1) if is_prime[p])
        
        # Information "cost" of the smoothing process
        # = log of number of composites eliminated = log(x - 1 - π(x))
        composites = limit - 1 - pi_x
        info_cost_smoothing = math.log(composites) if composites > 0 else 0
        
        # Various ratios
        results.append({
            'limit': limit,
            'pi_x': pi_x,
            'theta_x': float(theta_x),
            'theta_over_x': float(theta_x / limit),  # Should → 1
            'composites': composites,
            'info_cost_smoothing': float(info_cost_smoothing),
            'theta_over_info_cost': float(theta_x / info_cost_smoothing) if info_cost_smoothing > 0 else 0,
            'ratio_comparison': {
                'xi': float(XI),
                'gamma_ln_phi': float(GAMMA_PLUS_LN_PHI),
                'e_gamma': float(math.exp(GAMMA))
            }
        })
    
    return {
        'description': 'Testing if Ξ is an information rate',
        'results': results
    }


def main():
    print("=" * 70)
    print("EXP 28: DISTRIBUTED COLLAPSE HYPOTHESIS")
    print("Testing: local Ξ = distributed γ + ln(φ)")
    print("=" * 70)
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Ξ is instantaneous balance, γ+ln(φ) is accumulated balance',
        'constants': {
            'xi': float(XI),
            'gamma': float(GAMMA),
            'ln_phi': float(LN_PHI),
            'gamma_plus_ln_phi': float(GAMMA_PLUS_LN_PHI),
            'gap': float(GAMMA_PLUS_LN_PHI - XI),
            'gap_percent': float(100 * (GAMMA_PLUS_LN_PHI - XI) / XI)
        },
        'tests': {}
    }
    
    # Test 1: Mertens connection
    print("TEST 1: Mertens sum connection to γ + ln(φ)")
    print("-" * 60)
    mertens = test_mertens_connection()
    results['tests']['mertens_connection'] = mertens
    
    print(f"  Checking if Mertens sum relates to γ + ln(φ)...")
    for r in mertens['results']:
        print(f"  limit={r['limit']:>7}: mertens_sum={r['mertens_sum']:.4f}, "
              f"(sum-γ)/ln(φ)={r['ratio_to_ln_phi']:.4f}")
    print()
    
    # Test 2: Local vs global
    print("TEST 2: Local vs global balance")
    print("-" * 60)
    local_global = test_local_vs_global()
    results['tests']['local_vs_global'] = local_global
    
    print(f"  Limit: {local_global['limit']:,}")
    print(f"  Waves: {local_global['num_waves']}")
    print()
    print(f"  Sample convergence:")
    for s in local_global['samples']:
        print(f"    Wave {s['wave']:>3} (p={s['prime']:>5}): avg_balance={s['avg_balance']:.6f}, "
              f"err_Ξ={s['error_from_xi_%']:.2f}%, err_γ+lnφ={s['error_from_gamma_ln_phi_%']:.2f}%")
    print()
    print(f"  Final avg balance: {local_global['final_avg_balance']:.6f}")
    print(f"  Error from Ξ: {local_global['final_error_from_xi_%']:.2f}%")
    print(f"  Error from γ+ln(φ): {local_global['final_error_from_gamma_ln_phi_%']:.2f}%")
    print()
    
    # Test 3: Balance at each prime
    print("TEST 3: Expansion factors (Mertens third theorem)")
    print("-" * 60)
    expansion = test_balance_at_each_prime()
    results['tests']['expansion_factors'] = expansion
    
    print(f"  Target e^γ = {expansion['target_e_gamma']:.6f}")
    print(f"  Final product/ln(p) = {expansion['final_product_over_ln']:.6f}")
    print(f"  Error: {expansion['error_from_e_gamma']:.2f}%")
    print()
    
    # Test 4: Xi as rate
    print("TEST 4: Ξ as information rate")
    print("-" * 60)
    rate = test_xi_as_rate()
    results['tests']['xi_as_rate'] = rate
    
    for r in rate['results']:
        print(f"  limit={r['limit']:>7}: θ(x)/x={r['theta_over_x']:.4f}, "
              f"θ/cost={r['theta_over_info_cost']:.4f}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # What did we find?
    # Check if any ratio approaches Ξ or γ+ln(φ)
    
    final_balance = local_global['final_avg_balance']
    closer_to_xi = abs(final_balance - XI) < abs(final_balance - GAMMA_PLUS_LN_PHI)
    
    print(f"  The 'average balance' through sieve waves converges to: {final_balance:.6f}")
    print(f"  Ξ = {XI:.6f}")
    print(f"  γ + ln(φ) = {GAMMA_PLUS_LN_PHI:.6f}")
    print()
    print(f"  Closer to: {'Ξ' if closer_to_xi else 'γ + ln(φ)'}")
    print()
    
    # The Mertens third theorem already gives us e^γ
    print("  KEY INSIGHT:")
    print("  Mertens third theorem: ∏(p/(p-1)) ~ e^γ × ln(x)")
    print(f"  So e^γ = {math.exp(GAMMA):.6f} is the fundamental constant, not Ξ")
    print()
    print(f"  But e^(γ + ln(φ)) = e^γ × φ = {math.exp(GAMMA) * PHI:.6f}")
    print(f"  And Ξ = {XI:.6f}")
    print()
    
    # What's the relationship?
    ratio_xi_to_e_gamma = XI / math.exp(GAMMA)
    ratio_gamma_ln_phi_to_e_gamma = GAMMA_PLUS_LN_PHI / math.exp(GAMMA)
    
    print(f"  Ξ / e^γ = {ratio_xi_to_e_gamma:.6f}")
    print(f"  (γ+ln(φ)) / e^γ = {ratio_gamma_ln_phi_to_e_gamma:.6f}")
    print(f"  1/φ = {1/PHI:.6f}")
    print()
    
    # Check if Ξ / e^γ ≈ 1/φ
    error_ratio = 100 * abs(ratio_xi_to_e_gamma - 1/PHI) / (1/PHI)
    print(f"  Ξ / e^γ vs 1/φ: error = {error_ratio:.2f}%")
    
    if error_ratio < 5:
        print("  ⚠️  Close match! Ξ ≈ e^γ / φ might be meaningful.")
    
    results['summary'] = {
        'final_avg_balance': float(final_balance),
        'closer_to': 'xi' if closer_to_xi else 'gamma_ln_phi',
        'e_gamma': float(math.exp(GAMMA)),
        'xi_over_e_gamma': float(ratio_xi_to_e_gamma),
        'inv_phi': float(1/PHI),
        'error_xi_e_gamma_vs_inv_phi': float(error_ratio)
    }
    
    # Save
    with open('exp_28_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to exp_28_results.json")


if __name__ == '__main__':
    main()
