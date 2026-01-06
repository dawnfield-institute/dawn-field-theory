"""
exp_10_xi_predicts_class4.py
=============================

Test: Does distance from Ξ PREDICT edge-of-chaos behavior?

If Ξ is a theoretical optimum (Möbius confluence), then:
- Rules closer to Ξ should be more likely to be Class IV
- Distance from Ξ should correlate with computational complexity
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import comb

XI_TARGET = 1.0571428
CLASS_4_RULES = [54, 106, 110, 124, 137, 193]

def main():
    print("="*70)
    print("EXPERIMENT 10: Does Distance from Ξ Predict Class IV?")
    print("="*70)
    
    # Load existing CA results
    results_dir = Path(__file__).parent.parent.parent / 'cellular_automata_pac_attractors' / 'results'
    result_files = list(results_dir.glob('exp_07_definitive*.json'))
    
    if not result_files:
        print("No results found!")
        return
    
    with open(sorted(result_files)[-1]) as f:
        data = json.load(f)
    
    print(f"\nLoaded: {sorted(result_files)[-1].name}")
    
    # Get ALL rules with valid P/A
    all_data = []
    for key, r in data['rule_embeddings'].items():
        if r['A'] > 0:
            ratio = r['P'] / r['A']
            if ratio < 100:  # Filter outliers
                all_data.append({
                    'rule': int(key),
                    'ratio': ratio,
                    'distance': abs(ratio - XI_TARGET),
                    'is_class_4': int(key) in CLASS_4_RULES
                })
    
    print(f"Rules with valid P/A (< 100): {len(all_data)}")
    
    # Sort by distance from XI
    all_data.sort(key=lambda x: x['distance'])
    
    print(f"\n" + "-"*70)
    print(f"Top 15 closest to Ξ = {XI_TARGET}:")
    print("-"*70)
    print(f"{'Rank':<6}{'Rule':<8}{'P/A':<14}{'Distance':<14}{'Class IV?'}")
    print("-"*50)
    
    class_4_in_top_10 = 0
    class_4_in_top_15 = 0
    
    for i, r in enumerate(all_data[:15]):
        is_c4 = "★ YES" if r['is_class_4'] else ""
        print(f"{i+1:<6}{r['rule']:<8}{r['ratio']:<14.6f}{r['distance']:<14.6f}{is_c4}")
        if r['is_class_4']:
            if i < 10:
                class_4_in_top_10 += 1
            class_4_in_top_15 += 1
    
    print(f"\nClass IV rules in top 10: {class_4_in_top_10}/10")
    print(f"Class IV rules in top 15: {class_4_in_top_15}/15")
    
    # Probability calculations
    n_class4 = 6
    n_total = len(all_data)
    
    prob_10 = sum(
        comb(n_class4, k) * comb(n_total - n_class4, 10 - k) / comb(n_total, 10)
        for k in range(class_4_in_top_10, min(n_class4, 10) + 1)
    )
    
    prob_15 = sum(
        comb(n_class4, k) * comb(n_total - n_class4, 15 - k) / comb(n_total, 15)
        for k in range(class_4_in_top_15, min(n_class4, 15) + 1)
    )
    
    print(f"\nProbability of ≥{class_4_in_top_10} Class IV in top 10 by chance: {prob_10:.2e}")
    print(f"Probability of ≥{class_4_in_top_15} Class IV in top 15 by chance: {prob_15:.2e}")
    
    # Statistical test
    print(f"\n" + "="*70)
    print("STATISTICAL TEST: Are Class IV rules closer to Ξ?")
    print("="*70)
    
    class_4_distances = [d['distance'] for d in all_data if d['is_class_4']]
    other_distances = [d['distance'] for d in all_data if not d['is_class_4']]
    
    print(f"\nClass IV rules (n={len(class_4_distances)}):")
    print(f"  Mean distance from Ξ: {np.mean(class_4_distances):.6f}")
    print(f"  Median distance: {np.median(class_4_distances):.6f}")
    print(f"  Range: [{min(class_4_distances):.6f}, {max(class_4_distances):.6f}]")
    
    print(f"\nOther rules (n={len(other_distances)}):")
    print(f"  Mean distance from Ξ: {np.mean(other_distances):.6f}")
    print(f"  Median distance: {np.median(other_distances):.6f}")
    
    if len(class_4_distances) > 0 and len(other_distances) > 0:
        stat, pvalue = stats.mannwhitneyu(class_4_distances, other_distances, alternative='less')
        print(f"\nMann-Whitney U test (Class IV < Others):")
        print(f"  U statistic: {stat:.2f}")
        print(f"  p-value: {pvalue:.6f}")
        
        if pvalue < 0.01:
            print("\n✅ HIGHLY SIGNIFICANT (p < 0.01)")
            print("   Class IV rules are significantly closer to Ξ")
            print("   → Ξ PREDICTS edge-of-chaos behavior!")
        elif pvalue < 0.05:
            print("\n✅ SIGNIFICANT (p < 0.05)")
            print("   Class IV rules are closer to Ξ")
        else:
            print("\n❌ Not significant (p ≥ 0.05)")
    
    # Effect size
    all_class4_ranks = [i+1 for i, d in enumerate(all_data) if d['is_class_4']]
    expected_avg_rank = (len(all_data) + 1) / 2
    actual_avg_rank = np.mean(all_class4_ranks)
    
    print(f"\n" + "-"*70)
    print("Effect Size Analysis")
    print("-"*70)
    print(f"Class IV rules ranks (sorted by distance from Ξ): {all_class4_ranks}")
    print(f"Expected average rank (random): {expected_avg_rank:.1f}")
    print(f"Actual average rank: {actual_avg_rank:.1f}")
    print(f"Effect: Class IV rules rank {expected_avg_rank - actual_avg_rank:.1f} positions better than chance")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if pvalue < 0.05 and class_4_in_top_10 >= 3:
        print("\n✅ Ξ = 1.0571 has PREDICTIVE POWER for CA complexity")
        print("   - Class IV rules cluster near Ξ (not by fitting, but naturally)")
        print("   - This validates Ξ as a meaningful theoretical constant")
        print("   - Distance from Ξ could be used as a TOOL to identify")
        print("     edge-of-chaos behavior in other dynamical systems")
    else:
        print("\n⚠️ Results inconclusive - need more analysis")


if __name__ == "__main__":
    main()
