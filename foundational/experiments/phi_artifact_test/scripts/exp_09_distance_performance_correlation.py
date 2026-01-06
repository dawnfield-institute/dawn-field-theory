"""
exp_09_distance_performance_correlation.py
==========================================

Hypothesis: The constants (Ξ ≈ 1.057, 1/φ ≈ 0.618) represent theoretical
optimal states on Möbius topology. If true, distance from these constants
should PREDICT performance.

This turns the constants from "curve-fitted artifacts" into "predictive tools."

Test 1: CA Rules
- Compute P/A ratio for all 256 elementary CA rules
- Measure distance from Ξ = 1.0571
- See if distance correlates with Wolfram class (Class IV = edge of chaos)

Test 2: SEC Parameter Sweep
- Sweep SEC parameters
- Measure distance of frac(E>0) from 1/φ
- See if distance correlates with prime enrichment ratio

Test 3: Cross-validation
- Does being close to the theoretical optimum predict good behavior?
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import stats
import sys

# Constants - the theoretical optima
XI = 1.0571428  # 1 + π/55, theoretical Möbius confluence
PHI_INV = 0.618034  # 1/φ, theoretical partition

# For CA analysis
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'cellular_automata_pac_attractors'))

# For SEC analysis  
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'sec_prime_manifold'))


def compute_ca_embedding(rule: int, steps: int = 200, width: int = 101) -> dict:
    """Compute PAC embedding for a CA rule."""
    from core.pac_embedding import PACEmbedder
    
    embedder = PACEmbedder()
    
    # Initialize with single cell
    state = np.zeros(width, dtype=np.uint8)
    state[width // 2] = 1
    
    # Evolve
    history = [state.copy()]
    for _ in range(steps):
        new_state = np.zeros_like(state)
        for i in range(1, width - 1):
            neighborhood = (state[i-1] << 2) | (state[i] << 1) | state[i+1]
            new_state[i] = (rule >> neighborhood) & 1
        state = new_state
        history.append(state.copy())
    
    history = np.array(history)
    
    # Compute embedding
    P, A = embedder.compute_embedding(history)
    
    if A > 0:
        ratio = P / A
    else:
        ratio = float('inf')
    
    return {
        'rule': rule,
        'P': float(P),
        'A': float(A),
        'ratio': float(ratio) if ratio != float('inf') else None,
        'distance_from_xi': abs(ratio - XI) if ratio != float('inf') else None
    }


def get_wolfram_class(rule: int) -> int:
    """Return Wolfram class for known rules."""
    # Class I: Uniformity (evolves to homogeneous state)
    class_1 = [0, 8, 32, 40, 64, 72, 96, 104, 128, 136, 160, 168, 192, 200, 224, 232]
    
    # Class II: Periodicity (evolves to periodic/stable patterns)
    class_2 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27,
               28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62,
               74, 76, 77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142, 152, 154,
               156, 162, 164, 170, 172, 178, 184, 200, 204, 232]
    
    # Class III: Chaos (aperiodic, random-looking)
    class_3 = [18, 22, 30, 45, 60, 73, 75, 86, 89, 90, 101, 102, 105, 107, 109, 
               120, 121, 122, 126, 129, 131, 133, 135, 145, 146, 149, 150, 151,
               153, 161, 169, 181, 182, 183, 195, 225]
    
    # Class IV: Complex/Edge of chaos (localized structures, long transients)
    class_4 = [54, 106, 110, 124, 137, 193]
    
    if rule in class_1:
        return 1
    elif rule in class_4:
        return 4
    elif rule in class_3:
        return 3
    elif rule in class_2:
        return 2
    else:
        return 0  # Unknown


def test_ca_distance_correlation():
    """Test if distance from Ξ correlates with Wolfram class."""
    print("\n" + "="*70)
    print("TEST 1: CA Distance from Ξ vs Wolfram Class")
    print("="*70)
    
    results = []
    
    # Test all 256 rules
    print("\nComputing embeddings for 256 CA rules...")
    for rule in range(256):
        if rule % 32 == 0:
            print(f"  Progress: {rule}/256")
        
        try:
            embedding = compute_ca_embedding(rule)
            wolfram_class = get_wolfram_class(rule)
            embedding['wolfram_class'] = wolfram_class
            results.append(embedding)
        except Exception as e:
            pass  # Skip problematic rules
    
    print(f"\nSuccessfully computed {len(results)} embeddings")
    
    # Filter to rules with valid ratios
    valid = [r for r in results if r['ratio'] is not None and r['ratio'] < 100]
    print(f"Rules with valid ratios (< 100): {len(valid)}")
    
    # Group by Wolfram class
    by_class = {1: [], 2: [], 3: [], 4: [], 0: []}
    for r in valid:
        by_class[r['wolfram_class']].append(r)
    
    print("\n" + "-"*70)
    print("Distance from Ξ = 1.0571 by Wolfram Class")
    print("-"*70)
    
    class_stats = {}
    for cls in [1, 2, 3, 4]:
        if by_class[cls]:
            distances = [r['distance_from_xi'] for r in by_class[cls]]
            mean_dist = np.mean(distances)
            min_dist = np.min(distances)
            closest_rule = by_class[cls][np.argmin(distances)]['rule']
            
            class_stats[cls] = {
                'count': len(by_class[cls]),
                'mean_distance': mean_dist,
                'min_distance': min_dist,
                'closest_rule': closest_rule
            }
            
            print(f"\nClass {cls} ({len(by_class[cls])} rules):")
            print(f"  Mean distance from Ξ: {mean_dist:.4f}")
            print(f"  Min distance: {min_dist:.4f} (Rule {closest_rule})")
    
    # Key test: Are Class IV rules significantly closer to Ξ?
    if by_class[4]:
        class4_distances = [r['distance_from_xi'] for r in by_class[4]]
        other_distances = [r['distance_from_xi'] for r in valid if r['wolfram_class'] != 4]
        
        # Mann-Whitney U test
        stat, pvalue = stats.mannwhitneyu(class4_distances, other_distances, alternative='less')
        
        print("\n" + "-"*70)
        print("Statistical Test: Are Class IV rules closer to Ξ?")
        print("-"*70)
        print(f"Class IV mean distance: {np.mean(class4_distances):.4f}")
        print(f"Other classes mean distance: {np.mean(other_distances):.4f}")
        print(f"Mann-Whitney U test (Class IV < Others): p = {pvalue:.6f}")
        
        if pvalue < 0.05:
            print("✅ Class IV rules are SIGNIFICANTLY closer to Ξ (p < 0.05)")
        else:
            print("❌ No significant difference")
    
    # Find the 10 rules closest to Ξ
    valid_sorted = sorted(valid, key=lambda x: x['distance_from_xi'])[:10]
    
    print("\n" + "-"*70)
    print("Top 10 Rules Closest to Ξ = 1.0571")
    print("-"*70)
    print(f"{'Rule':<8} {'P/A Ratio':<12} {'Distance':<12} {'Class':<8}")
    print("-"*40)
    
    class4_in_top10 = 0
    for r in valid_sorted:
        cls_str = f"Class {r['wolfram_class']}" if r['wolfram_class'] > 0 else "Unknown"
        print(f"{r['rule']:<8} {r['ratio']:<12.4f} {r['distance_from_xi']:<12.4f} {cls_str}")
        if r['wolfram_class'] == 4:
            class4_in_top10 += 1
    
    print(f"\nClass IV rules in top 10: {class4_in_top10}/10")
    
    # Probability calculation
    n_class4 = len(by_class[4])
    n_total = len(valid)
    
    from scipy.special import comb
    prob_by_chance = sum(
        comb(n_class4, k) * comb(n_total - n_class4, 10 - k) / comb(n_total, 10)
        for k in range(class4_in_top10, min(n_class4, 10) + 1)
    )
    
    print(f"Probability of ≥{class4_in_top10} Class IV in top 10 by chance: {prob_by_chance:.2e}")
    
    return {
        'class_stats': class_stats,
        'top_10': valid_sorted,
        'class4_in_top10': class4_in_top10,
        'prob_by_chance': prob_by_chance
    }


def test_sec_distance_correlation():
    """Test if distance from 1/φ correlates with prime enrichment."""
    print("\n" + "="*70)
    print("TEST 2: SEC Distance from 1/φ vs Prime Enrichment")
    print("="*70)
    
    from core.sec_core import compute_sec, FIRST_50_PRIMES
    
    results = []
    
    # Sweep parameters
    n_max = 50000
    factor_bases = [5, 7, 10, 12, 15, 20]
    windows = [31, 51, 71, 101, 151, 201]
    lambdas = [0.9, 0.95, 0.99, 0.995]
    
    total = len(factor_bases) * len(windows) * len(lambdas)
    count = 0
    
    print(f"\nSweeping {total} parameter combinations...")
    
    for fb_size in factor_bases:
        for window in windows:
            for lam in lambdas:
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total}")
                
                try:
                    factor_base = FIRST_50_PRIMES[:fb_size]
                    sec = compute_sec(n_max=n_max, factor_base=factor_base, 
                                     window=window, lam=lam)
                    
                    # Compute metrics on odd numbers
                    odds = np.arange(3, n_max + 1, 2)
                    E_odds = sec.E[odds]
                    is_prime = sec.prime_mask[odds]
                    
                    frac_positive = float((E_odds > 0).mean())
                    
                    # Prime enrichment
                    if (E_odds > 0).any() and (E_odds <= 0).any():
                        prime_rate_pos = is_prime[E_odds > 0].mean()
                        prime_rate_neg = is_prime[E_odds <= 0].mean()
                        enrichment = prime_rate_pos / prime_rate_neg if prime_rate_neg > 0 else float('inf')
                    else:
                        enrichment = 1.0
                    
                    results.append({
                        'factor_base': fb_size,
                        'window': window,
                        'lambda': lam,
                        'frac_positive': frac_positive,
                        'distance_from_phi_inv': abs(frac_positive - PHI_INV),
                        'enrichment_ratio': enrichment
                    })
                except:
                    pass
    
    print(f"\nSuccessfully computed {len(results)} configurations")
    
    # Compute correlation
    distances = [r['distance_from_phi_inv'] for r in results]
    enrichments = [r['enrichment_ratio'] for r in results if r['enrichment_ratio'] < 100]
    distances_filtered = [r['distance_from_phi_inv'] for r in results if r['enrichment_ratio'] < 100]
    
    if len(distances_filtered) > 10:
        corr, pvalue = stats.spearmanr(distances_filtered, enrichments)
        
        print("\n" + "-"*70)
        print("Correlation: Distance from 1/φ vs Prime Enrichment")
        print("-"*70)
        print(f"Spearman correlation: {corr:.4f}")
        print(f"P-value: {pvalue:.6f}")
        
        if pvalue < 0.05:
            if corr < 0:
                print("✅ CLOSER to 1/φ → BETTER enrichment (negative correlation)")
            else:
                print("⚠️ CLOSER to 1/φ → WORSE enrichment (positive correlation)")
        else:
            print("❌ No significant correlation")
    
    # Show best and worst configurations
    results_sorted = sorted(results, key=lambda x: x['distance_from_phi_inv'])
    
    print("\n" + "-"*70)
    print("Configurations Closest to 1/φ = 0.618")
    print("-"*70)
    for r in results_sorted[:5]:
        print(f"  frac={r['frac_positive']:.4f}, dist={r['distance_from_phi_inv']:.4f}, "
              f"enrichment={r['enrichment_ratio']:.2f}x")
    
    print("\nConfigurations Farthest from 1/φ")
    print("-"*70)
    for r in results_sorted[-5:]:
        print(f"  frac={r['frac_positive']:.4f}, dist={r['distance_from_phi_inv']:.4f}, "
              f"enrichment={r['enrichment_ratio']:.2f}x")
    
    return {
        'n_configs': len(results),
        'correlation': corr if len(distances_filtered) > 10 else None,
        'p_value': pvalue if len(distances_filtered) > 10 else None,
        'results': results
    }


def main():
    """Run distance-performance correlation tests."""
    print("="*70)
    print("EXPERIMENT 09: Distance from Constants vs Performance")
    print("="*70)
    print(f"\nStarted: {datetime.now().isoformat()}")
    print("\nHypothesis: Constants represent theoretical optima.")
    print("If true, distance from constants should predict performance.")
    print(f"\nTheoretical optima:")
    print(f"  Ξ = {XI} (Möbius confluence)")
    print(f"  1/φ = {PHI_INV} (partition balance)")
    
    # Run tests
    ca_results = test_ca_distance_correlation()
    sec_results = test_sec_distance_correlation()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n1. CA Rules:")
    print(f"   Class IV rules in top 10 closest to Ξ: {ca_results['class4_in_top10']}/10")
    print(f"   Probability by chance: {ca_results['prob_by_chance']:.2e}")
    if ca_results['prob_by_chance'] < 0.05:
        print("   ✅ Ξ PREDICTS edge-of-chaos behavior")
    
    print("\n2. SEC Prime Enrichment:")
    if sec_results['correlation'] is not None:
        print(f"   Correlation (distance vs enrichment): {sec_results['correlation']:.4f}")
        print(f"   P-value: {sec_results['p_value']:.6f}")
        if sec_results['p_value'] < 0.05 and sec_results['correlation'] < 0:
            print("   ✅ 1/φ PREDICTS prime separation quality")
        elif sec_results['p_value'] < 0.05:
            print("   ⚠️ Correlation exists but in unexpected direction")
        else:
            print("   ❌ No predictive power found")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    ca_predicts = ca_results['prob_by_chance'] < 0.05
    sec_predicts = (sec_results['correlation'] is not None and 
                   sec_results['p_value'] < 0.05 and 
                   sec_results['correlation'] < 0)
    
    if ca_predicts and sec_predicts:
        print("\n✅ Both constants have PREDICTIVE POWER")
        print("   → This is a TOOL, not curve-fitting")
    elif ca_predicts:
        print("\n✅ Ξ has predictive power for CA complexity")
        print("   → SEC 1/φ relationship needs more investigation")
    elif sec_predicts:
        print("\n✅ 1/φ has predictive power for SEC")
        print("   → CA Ξ relationship needs more investigation")
    else:
        print("\n❌ Neither constant shows predictive power")
        print("   → May be curve-fitting after all")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Distance from theoretical constants predicts performance',
        'constants': {'xi': XI, 'phi_inv': PHI_INV},
        'ca_results': {
            'class4_in_top10': ca_results['class4_in_top10'],
            'prob_by_chance': ca_results['prob_by_chance']
        },
        'sec_results': {
            'correlation': sec_results['correlation'],
            'p_value': sec_results['p_value']
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"exp_09_distance_performance_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
