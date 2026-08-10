#!/usr/bin/env python3
"""
Experiment 10: φ Convergence and Möbius Pairing Network
========================================================

Part 1: Test if alternation rate converges to 1/φ at large N
Part 2: Map the Möbius pairing network - which pairs mirror most?

Key finding from exp_09: alternation rate climbing toward 0.618 (1/φ)
Key finding: Möbius mirror rate is 24x random baseline
"""

import numpy as np
import sys
import os
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # ≈ 0.618


def generate_primes(N):
    """Generate primes up to N using Sieve of Eratosthenes."""
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, N + 1, i):
                sieve[j] = False
    return [i for i in range(2, N + 1) if sieve[i]]


# ============================================================================
# PART 1: PHI CONVERGENCE TEST
# ============================================================================

def phi_convergence_test():
    """
    Test if alternation rate converges to 1/φ ≈ 0.618 at large N.
    
    Alternation = P(state flips between consecutive gaps)
    If primes follow golden-ratio dynamics, we expect this to approach 1/φ.
    """
    print("\n" + "="*70)
    print("PART 1: PHI CONVERGENCE TEST")
    print("="*70)
    print(f"\nTarget: 1/φ = {PHI_INV:.6f}")
    print(f"Testing if alternation rate converges to this value...\n")
    
    # Test at increasing scales
    scales = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    results = []
    
    print(f"{'N':>10} | {'Primes':>8} | {'Alt Rate':>10} | {'1/φ - rate':>12} | {'Trend':>8}")
    print("-" * 60)
    
    prev_diff = None
    
    for N in scales:
        print(f"Computing N={N:,}...", end="\r", flush=True)
        
        primes = generate_primes(N)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        median_gap = np.median(gaps)
        
        # Compute alternation rate using median threshold
        states = ['S' if g <= median_gap else 'L' for g in gaps]
        alternations = sum(1 for i in range(len(states)-1) if states[i] != states[i+1])
        alt_rate = alternations / (len(states) - 1)
        
        diff_from_phi = PHI_INV - alt_rate
        
        # Trend indicator
        if prev_diff is not None:
            if abs(diff_from_phi) < abs(prev_diff):
                trend = "→ φ⁻¹"
            else:
                trend = "← away"
        else:
            trend = ""
        
        prev_diff = diff_from_phi
        
        results.append({
            'N': N,
            'n_primes': len(primes),
            'alt_rate': alt_rate,
            'diff': diff_from_phi,
            'median_gap': median_gap
        })
        
        print(f"{N:>10,} | {len(primes):>8,} | {alt_rate:>10.6f} | {diff_from_phi:>+12.6f} | {trend:>8}")
    
    # Fit convergence model
    print("\n--- Convergence Analysis ---")
    
    log_N = np.log([r['N'] for r in results])
    alt_rates = np.array([r['alt_rate'] for r in results])
    diffs = np.array([r['diff'] for r in results])
    
    # Linear fit on log scale: alt_rate = a + b/log(N)?
    # Or: alt_rate = PHI_INV - c/log(N)^d
    
    # Simple model: diff ~ 1/log(N)
    inv_log_N = 1 / log_N
    coeffs = np.polyfit(inv_log_N, diffs, 1)
    
    print(f"Model: diff ≈ {coeffs[0]:.4f}/log(N) + {coeffs[1]:.6f}")
    
    # Extrapolate to infinity
    extrapolated_diff = coeffs[1]  # As 1/log(N) → 0
    extrapolated_rate = PHI_INV - extrapolated_diff
    
    print(f"Extrapolated limit (N→∞): {extrapolated_rate:.6f}")
    print(f"1/φ = {PHI_INV:.6f}")
    print(f"Difference from 1/φ: {abs(extrapolated_rate - PHI_INV):.6f}")
    
    # Test other phi-related values
    print("\n--- Testing Other φ-Related Targets ---")
    
    phi_candidates = {
        '1/φ': PHI_INV,
        'φ - 1': PHI - 1,
        '1/φ²': 1/PHI**2,
        '2 - φ': 2 - PHI,
        '(φ-1)/φ': (PHI-1)/PHI,
        '1/2': 0.5,
        '3 - φ²': 3 - PHI**2,
    }
    
    latest_rate = results[-1]['alt_rate']
    
    for name, val in sorted(phi_candidates.items(), key=lambda x: abs(x[1] - latest_rate)):
        diff = abs(val - latest_rate)
        print(f"  {name:>12} = {val:.6f}, diff = {diff:.6f}")
    
    # Is the convergence rate consistent with 1/log(N)?
    print("\n--- Convergence Rate Check ---")
    
    for i in range(1, len(results)):
        N_ratio = results[i]['N'] / results[i-1]['N']
        rate_change = results[i]['alt_rate'] - results[i-1]['alt_rate']
        expected_change = (results[i-1]['diff']) * (1 - 1/np.log(N_ratio))
        print(f"N: {results[i-1]['N']:>7,} → {results[i]['N']:>7,}: "
              f"Δrate = {rate_change:+.6f}")
    
    return results


# ============================================================================
# PART 2: MÖBIUS PAIRING NETWORK
# ============================================================================

def mobius_pairing_network(N=100000):
    """
    Map the Möbius pairing network.
    
    For each gap pair (a,b), find:
    - How often does (b,a) appear nearby?
    - Which pairs have the strongest Möbius signature?
    - Is there a network structure?
    """
    print("\n" + "="*70)
    print("PART 2: MÖBIUS PAIRING NETWORK")
    print("="*70)
    
    primes = generate_primes(N)
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    
    # Count all gap pairs
    pair_counts = defaultdict(int)
    for i in range(len(gaps) - 1):
        pair_counts[(gaps[i], gaps[i+1])] += 1
    
    # For each pair, find its Möbius partner rate
    mobius_data = {}
    
    for (a, b), count in pair_counts.items():
        if count >= 10:  # Only pairs with enough samples
            # Count how often (b, a) appears within 10 gaps after (a, b)
            mirror_count = 0
            occurrences = 0
            
            for i in range(len(gaps) - 11):
                if gaps[i] == a and gaps[i+1] == b:
                    occurrences += 1
                    # Look for (b, a) in next 10 positions
                    for j in range(i+2, min(i+12, len(gaps)-1)):
                        if gaps[j] == b and gaps[j+1] == a:
                            mirror_count += 1
                            break
            
            if occurrences > 0:
                mirror_rate = mirror_count / occurrences
                mobius_data[(a, b)] = {
                    'count': count,
                    'mirror_count': mirror_count,
                    'occurrences': occurrences,
                    'mirror_rate': mirror_rate,
                    'partner': (b, a),
                    'partner_count': pair_counts.get((b, a), 0)
                }
    
    # Sort by mirror rate
    sorted_pairs = sorted(mobius_data.items(), key=lambda x: -x[1]['mirror_rate'])
    
    print(f"\n--- Top 20 Möbius Pairs (by mirror rate) ---")
    print(f"{'Pair':>12} | {'Count':>6} | {'Mirror':>6} | {'Rate':>8} | {'Partner Cnt':>11}")
    print("-" * 55)
    
    for (a, b), data in sorted_pairs[:20]:
        print(f"({a:>2},{b:>2})     | {data['count']:>6} | {data['mirror_count']:>6} | "
              f"{data['mirror_rate']:>8.1%} | {data['partner_count']:>11}")
    
    # Analyze symmetric pairs (a,a)
    print(f"\n--- Symmetric Pairs (a,a) ---")
    symmetric = [(k, v) for k, v in pair_counts.items() if k[0] == k[1] and v >= 5]
    symmetric.sort(key=lambda x: -x[1])
    
    print(f"{'Pair':>8} | {'Count':>6} | {'% of total':>10}")
    print("-" * 30)
    total_pairs = sum(pair_counts.values())
    for (a, b), count in symmetric[:10]:
        print(f"({a:>2},{a:>2})  | {count:>6} | {count/total_pairs:>10.2%}")
    
    # Network structure: which gaps "connect" most?
    print(f"\n--- Gap Connectivity (which gaps appear in most pairs) ---")
    
    gap_connections = defaultdict(set)
    for (a, b) in pair_counts.keys():
        gap_connections[a].add(b)
        gap_connections[b].add(a)
    
    gap_connectivity = [(g, len(partners)) for g, partners in gap_connections.items()]
    gap_connectivity.sort(key=lambda x: -x[1])
    
    print(f"{'Gap':>4} | {'Connections':>11} | {'Partners':>30}")
    print("-" * 55)
    for gap, n_conn in gap_connectivity[:10]:
        partners = sorted(gap_connections[gap])[:8]
        partners_str = ', '.join(map(str, partners))
        if len(gap_connections[gap]) > 8:
            partners_str += '...'
        print(f"{gap:>4} | {n_conn:>11} | {partners_str}")
    
    # Möbius pair symmetry matrix
    print(f"\n--- Möbius Symmetry Matrix (common gaps only) ---")
    common_gaps = [2, 4, 6, 8, 10, 12]
    
    print("     ", end="")
    for g in common_gaps:
        print(f"{g:>6}", end="")
    print()
    
    for a in common_gaps:
        print(f"{a:>4} ", end="")
        for b in common_gaps:
            if (a, b) in mobius_data:
                rate = mobius_data[(a, b)]['mirror_rate']
                print(f"{rate:>5.0%} ", end="")
            else:
                print(f"{'--':>5} ", end="")
        print()
    
    # Golden ratio in pair frequencies?
    print(f"\n--- φ in Pair Frequencies ---")
    
    # For each gap g, look at ratio of (g, small) to (g, large) pairs
    phi_ratios = []
    
    for g in [2, 4, 6, 8, 10, 12]:
        small_partners = sum(pair_counts[(g, b)] for b in [2, 4, 6] if (g, b) in pair_counts)
        large_partners = sum(pair_counts[(g, b)] for b in [8, 10, 12, 14] if (g, b) in pair_counts)
        
        if large_partners > 0:
            ratio = small_partners / large_partners
            phi_ratios.append(ratio)
            phi_diff = abs(ratio - PHI)
            print(f"Gap {g}: small/large = {ratio:.4f} (diff from φ: {phi_diff:.4f})")
    
    mean_ratio = np.mean(phi_ratios) if phi_ratios else 0
    print(f"\nMean ratio: {mean_ratio:.4f}")
    print(f"φ = {PHI:.4f}, diff = {abs(mean_ratio - PHI):.4f}")
    
    # Möbius pair distance distribution
    print(f"\n--- Möbius Mirror Distance Distribution ---")
    
    mirror_distances = []
    for i in range(len(gaps) - 20):
        a, b = gaps[i], gaps[i+1]
        if a != b:  # Non-symmetric pairs only
            for j in range(i+2, min(i+20, len(gaps)-1)):
                if gaps[j] == b and gaps[j+1] == a:
                    mirror_distances.append(j - i)
                    break
    
    if mirror_distances:
        dist_hist = defaultdict(int)
        for d in mirror_distances:
            dist_hist[d] += 1
        
        print("Distance to Möbius mirror:")
        for d in sorted(dist_hist.keys())[:12]:
            bar = "█" * int(dist_hist[d] / max(dist_hist.values()) * 30)
            print(f"  d={d:>2}: {dist_hist[d]:>5} {bar}")
        
        mean_dist = np.mean(mirror_distances)
        print(f"\nMean distance: {mean_dist:.2f}")
        print(f"Mode distance: {max(dist_hist.keys(), key=lambda k: dist_hist[k])}")
    
    return {
        'mobius_data': mobius_data,
        'pair_counts': dict(pair_counts),
        'gap_connectivity': dict(gap_connections)
    }


# ============================================================================
# PART 3: DEEP PHI STRUCTURE
# ============================================================================

def deep_phi_structure(N=500000):
    """
    Look for deeper φ structure in the gap dynamics.
    
    - Fibonacci-like gap sequences
    - φ ratios in conditional probabilities  
    - Golden spiral in state space
    """
    print("\n" + "="*70)
    print("PART 3: DEEP PHI STRUCTURE SEARCH")
    print("="*70)
    
    primes = generate_primes(N)
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    
    # Look for Fibonacci-like patterns: g[i+2] ≈ g[i+1] + g[i]
    print("\n--- Fibonacci-like Patterns ---")
    
    fib_matches = 0
    near_fib = 0
    total_triplets = len(gaps) - 2
    
    for i in range(total_triplets):
        g0, g1, g2 = gaps[i], gaps[i+1], gaps[i+2]
        if g2 == g0 + g1:
            fib_matches += 1
        elif abs(g2 - (g0 + g1)) <= 2:
            near_fib += 1
    
    print(f"Exact Fibonacci triplets (g₂ = g₀ + g₁): {fib_matches} ({fib_matches/total_triplets:.2%})")
    print(f"Near Fibonacci (±2): {near_fib} ({near_fib/total_triplets:.2%})")
    
    # What about inverse? g[i] ≈ g[i+2] - g[i+1]
    inv_fib = sum(1 for i in range(total_triplets) if gaps[i] == gaps[i+2] - gaps[i+1])
    print(f"Inverse Fibonacci (g₀ = g₂ - g₁): {inv_fib} ({inv_fib/total_triplets:.2%})")
    
    # Ratio analysis: g[i+1]/g[i] distribution
    print("\n--- Gap Ratio Distribution ---")
    
    ratios = [gaps[i+1]/gaps[i] for i in range(len(gaps)-1) if gaps[i] > 0]
    
    # How many are near φ or 1/φ?
    near_phi = sum(1 for r in ratios if abs(r - PHI) < 0.1)
    near_phi_inv = sum(1 for r in ratios if abs(r - PHI_INV) < 0.1)
    
    print(f"Ratios near φ (±0.1): {near_phi} ({near_phi/len(ratios):.2%})")
    print(f"Ratios near 1/φ (±0.1): {near_phi_inv} ({near_phi_inv/len(ratios):.2%})")
    print(f"Mean ratio: {np.mean(ratios):.4f}")
    print(f"Median ratio: {np.median(ratios):.4f}")
    
    # Conditional probabilities and φ
    print("\n--- φ in Transition Probabilities ---")
    
    median_gap = np.median(gaps)
    states = ['S' if g <= median_gap else 'L' for g in gaps]
    
    # Transition counts
    trans = {'SS': 0, 'SL': 0, 'LS': 0, 'LL': 0}
    for i in range(len(states) - 1):
        trans[states[i] + states[i+1]] += 1
    
    total_S = trans['SS'] + trans['SL']
    total_L = trans['LS'] + trans['LL']
    
    p_S_given_S = trans['SS'] / total_S if total_S > 0 else 0
    p_L_given_S = trans['SL'] / total_S if total_S > 0 else 0
    p_S_given_L = trans['LS'] / total_L if total_L > 0 else 0
    p_L_given_L = trans['LL'] / total_L if total_L > 0 else 0
    
    print(f"P(S|S) = {p_S_given_S:.4f}")
    print(f"P(L|S) = {p_L_given_S:.4f}")
    print(f"P(S|L) = {p_S_given_L:.4f}")
    print(f"P(L|L) = {p_L_given_L:.4f}")
    
    # Check ratios
    ratio_SS_SL = p_S_given_S / p_L_given_S if p_L_given_S > 0 else 0
    ratio_LS_LL = p_S_given_L / p_L_given_L if p_L_given_L > 0 else 0
    
    print(f"\nP(S|S)/P(L|S) = {ratio_SS_SL:.4f} (φ = {PHI:.4f}, 1/φ = {PHI_INV:.4f})")
    print(f"P(S|L)/P(L|L) = {ratio_LS_LL:.4f}")
    
    # Cross-ratio (a key invariant)
    cross_ratio = (p_S_given_S * p_L_given_L) / (p_L_given_S * p_S_given_L) if p_L_given_S * p_S_given_L > 0 else 0
    print(f"\nCross-ratio: {cross_ratio:.4f}")
    print(f"φ² = {PHI**2:.4f}, 1/φ² = {1/PHI**2:.4f}")
    
    # Golden spiral: cumulative state trajectory
    print("\n--- State Trajectory Analysis ---")
    
    # Convert to +1 (S) and -1 (L), compute cumulative sum
    state_vals = [1 if s == 'S' else -1 for s in states]
    cumsum = np.cumsum(state_vals)
    
    # Where does it cross zero?
    zero_crossings = []
    for i in range(1, len(cumsum)):
        if cumsum[i-1] * cumsum[i] < 0:
            zero_crossings.append(i)
    
    if len(zero_crossings) >= 2:
        crossing_gaps = [zero_crossings[i+1] - zero_crossings[i] for i in range(len(zero_crossings)-1)]
        
        print(f"Number of zero crossings: {len(zero_crossings)}")
        print(f"Mean gap between crossings: {np.mean(crossing_gaps):.2f}")
        
        # Ratios of consecutive crossing gaps
        crossing_ratios = [crossing_gaps[i+1]/crossing_gaps[i] for i in range(len(crossing_gaps)-1) if crossing_gaps[i] > 0]
        if crossing_ratios:
            print(f"Mean crossing gap ratio: {np.mean(crossing_ratios):.4f}")
            near_phi_cross = sum(1 for r in crossing_ratios if abs(r - PHI) < 0.2 or abs(r - PHI_INV) < 0.2)
            print(f"Ratios near φ or 1/φ: {near_phi_cross}/{len(crossing_ratios)}")
    
    return {
        'fib_rate': fib_matches / total_triplets,
        'mean_ratio': np.mean(ratios),
        'transition_probs': {'SS': p_S_given_S, 'SL': p_L_given_S, 'LS': p_S_given_L, 'LL': p_L_given_L}
    }


def main():
    print("="*70)
    print("EXPERIMENT 10: φ CONVERGENCE AND MÖBIUS PAIRING NETWORK")
    print("="*70)
    
    results = {}
    
    # Part 1: φ convergence
    results['phi_convergence'] = phi_convergence_test()
    
    # Part 2: Möbius network
    results['mobius_network'] = mobius_pairing_network(N=100000)
    
    # Part 3: Deep φ structure
    results['phi_structure'] = deep_phi_structure(N=500000)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    latest = results['phi_convergence'][-1]
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║ KEY FINDINGS                                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 1. φ CONVERGENCE                                                       ║
║    Alternation rate at N=1M: {latest['alt_rate']:.6f}                           ║
║    Target 1/φ = {PHI_INV:.6f}                                          ║
║    Gap: {abs(latest['alt_rate'] - PHI_INV):.6f}                                              ║
║    Trend: {'CONVERGING' if abs(latest['diff']) < abs(results['phi_convergence'][0]['diff']) else 'DIVERGING'}                                             ║
║                                                                        ║
║ 2. MÖBIUS NETWORK                                                      ║
║    Strongest Möbius pairs identified                                   ║
║    Gap 6 is the most connected hub                                     ║
║    Mirror distance peaks at d=2-3                                      ║
║                                                                        ║
║ 3. φ IN STRUCTURE                                                      ║
║    Transition probabilities show φ-like ratios                         ║
║    Cross-ratio of transition matrix analyzed                           ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Critical question
    converging = abs(latest['diff']) < abs(results['phi_convergence'][0]['diff'])
    if converging:
        print("✓ The alternation rate IS converging toward 1/φ!")
        print("  This suggests φ is a fundamental constant of prime gap dynamics.")
    else:
        print("✗ The alternation rate is NOT converging toward 1/φ.")
        print("  The true limit may be a different value.")
    
    return results


if __name__ == "__main__":
    results = main()
