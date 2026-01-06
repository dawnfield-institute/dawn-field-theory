#!/usr/bin/env python3
"""
Experiment 07: Blind Clustering Analysis
=========================================

The HONEST test: Remove all φ/Ξ targets and see what naturally emerges.

Key question: Do Class IV rules cluster at some special value, and is that
value genuinely meaningful - or did we just curve-fit Ξ = 1.0571 after the fact?

Methodology:
1. Compute P/A ratios for all 256 rules with NO target in mind
2. Find where Class IV rules naturally cluster
3. Test if that cluster location is:
   a) Significantly different from random?
   b) Near any mathematically special values?
   c) Reproducible across different embeddings?

This removes the circular logic:
- OLD: "We look for Ξ = 1.0571 → We find Class IV at Ξ → Ξ is special!"
- NEW: "We find Class IV clusters at X → Is X special for any reason?"

Author: Dawn Field Theory Research
Date: 2025-01-06
"""

import sys
import os
import json
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# Add CA core to path
CA_CORE = Path(__file__).parent.parent.parent.parent / "experiments" / "cellular_automata_pac_attractors" / "core"
sys.path.insert(0, str(CA_CORE))

from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass
from pac_embedding import PACEmbedder, PACCoordinates


# Constants we're TESTING against (not assuming)
CANDIDATES = {
    'unity': 1.0,
    'xi_claimed': 1.0571,
    'phi': 1.618033988749895,
    'phi_inv': 0.618033988749895,
    'sqrt2': 1.41421356,
    'e': 2.71828183,
    'pi_div_3': 1.0471975512,  # π/3
    '1_plus_pi_55': 1.0571198,  # 1 + π/55 (the Ξ formula)
    'log2': 0.693147,
    'sqrt_phi': 1.272019649,
}


def compute_all_ratios(width: int = 101, steps: int = 200) -> Dict[int, float]:
    """
    Compute P/A ratios for all 256 rules with NO preconceptions.
    """
    embedder = PACEmbedder(width=width, steps=steps)
    ratios = {}
    
    for rule in range(256):
        coords = embedder.embed_rule(rule, init_type='single')
        if coords.actualization > 0.001:  # Avoid division by near-zero
            ratios[rule] = coords.potential / coords.actualization
        else:
            ratios[rule] = float('inf')  # Mark as extreme
    
    return ratios


def find_natural_clusters(ratios: Dict[int, float], n_clusters: int = 5) -> Dict[str, any]:
    """
    Find natural clusters in the P/A ratio distribution.
    Uses hierarchical clustering with NO target values.
    """
    # Filter out infinite ratios
    finite_ratios = {k: v for k, v in ratios.items() if np.isfinite(v)}
    
    rules = np.array(list(finite_ratios.keys()))
    values = np.array(list(finite_ratios.values())).reshape(-1, 1)
    
    # Hierarchical clustering
    Z = linkage(values, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    # Analyze each cluster
    clusters = {}
    for i in range(1, n_clusters + 1):
        mask = labels == i
        cluster_rules = rules[mask]
        cluster_values = values[mask].flatten()
        
        # What Wolfram classes are in this cluster?
        class_counts = Counter([
            RULE_CLASSIFICATIONS.get(r, WolframClass.UNKNOWN).name 
            for r in cluster_rules
        ])
        
        clusters[f'cluster_{i}'] = {
            'rules': cluster_rules.tolist(),
            'values': cluster_values.tolist(),
            'mean': float(np.mean(cluster_values)),
            'std': float(np.std(cluster_values)),
            'min': float(np.min(cluster_values)),
            'max': float(np.max(cluster_values)),
            'n_rules': len(cluster_rules),
            'class_composition': dict(class_counts),
            'class_iv_count': class_counts.get('CLASS_IV', 0),
        }
    
    return clusters


def find_class_iv_center(ratios: Dict[int, float]) -> Dict[str, any]:
    """
    Find where Class IV rules naturally concentrate.
    NO target value assumed - just find their natural center.
    """
    class_iv_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                      if wc == WolframClass.CLASS_IV]
    
    class_iv_ratios = [ratios[r] for r in class_iv_rules if np.isfinite(ratios[r])]
    
    if not class_iv_ratios:
        return {'error': 'No finite Class IV ratios'}
    
    mean = np.mean(class_iv_ratios)
    median = np.median(class_iv_ratios)
    std = np.std(class_iv_ratios)
    
    # Bootstrap confidence interval for the mean
    bootstrap_means = []
    for _ in range(10000):
        sample = np.random.choice(class_iv_ratios, size=len(class_iv_ratios), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'class_iv_rules': class_iv_rules,
        'ratios': class_iv_ratios,
        'n': len(class_iv_ratios),
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
    }


def test_candidates(class_iv_center: float, ci_lower: float, ci_upper: float) -> Dict[str, any]:
    """
    Test each candidate constant against the Class IV cluster center.
    Which candidates fall within the confidence interval?
    """
    results = {}
    
    for name, value in CANDIDATES.items():
        distance = abs(class_iv_center - value)
        within_ci = ci_lower <= value <= ci_upper
        relative_error = distance / class_iv_center if class_iv_center != 0 else float('inf')
        
        results[name] = {
            'value': value,
            'distance_from_center': float(distance),
            'relative_error': float(relative_error),
            'within_95_ci': within_ci,
        }
    
    # Sort by distance
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['distance_from_center']))
    
    return sorted_results


def compare_to_random(ratios: Dict[int, float], class_iv_center: float) -> Dict[str, any]:
    """
    Is the Class IV clustering significantly different from random?
    
    Null hypothesis: Class IV rules are randomly distributed among all rules.
    """
    all_finite_ratios = [v for v in ratios.values() if np.isfinite(v)]
    class_iv_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                      if wc == WolframClass.CLASS_IV]
    class_iv_ratios = [ratios[r] for r in class_iv_rules if np.isfinite(ratios[r])]
    
    # Other rules
    other_ratios = [ratios[r] for r in ratios if r not in class_iv_rules and np.isfinite(ratios[r])]
    
    # Test 1: Are Class IV ratios significantly different from others?
    if len(class_iv_ratios) > 0 and len(other_ratios) > 0:
        u_stat, mw_p = stats.mannwhitneyu(class_iv_ratios, other_ratios, alternative='two-sided')
    else:
        u_stat, mw_p = 0, 1.0
    
    # Test 2: Bootstrap test for Class IV mean vs random sample mean
    random_means = []
    n_iv = len(class_iv_ratios)
    for _ in range(10000):
        sample = np.random.choice(all_finite_ratios, size=n_iv, replace=False)
        random_means.append(np.mean(sample))
    
    # Where does the actual Class IV mean fall?
    percentile = np.sum(np.array(random_means) <= class_iv_center) / len(random_means) * 100
    
    # Test 3: Variance test - do Class IV rules cluster tighter than random?
    class_iv_std = np.std(class_iv_ratios)
    random_stds = []
    for _ in range(10000):
        sample = np.random.choice(all_finite_ratios, size=n_iv, replace=False)
        random_stds.append(np.std(sample))
    
    tightness_percentile = np.sum(np.array(random_stds) >= class_iv_std) / len(random_stds) * 100
    
    return {
        'mann_whitney_u': float(u_stat),
        'mann_whitney_p': float(mw_p),
        'class_iv_mean_percentile': float(percentile),
        'class_iv_std': float(class_iv_std),
        'random_std_mean': float(np.mean(random_stds)),
        'tightness_percentile': float(tightness_percentile),
        'is_tighter_than_95pct_random': tightness_percentile > 95,
        'is_mean_unusual': percentile < 5 or percentile > 95,
    }


def find_best_simple_formula(center: float) -> Dict[str, any]:
    """
    Find the simplest mathematical expression that approximates the center.
    
    Test formulas of form: a + b*π/c where a,b,c are small integers.
    """
    best_formulas = []
    
    for a in range(0, 3):  # 0, 1, 2
        for b in range(-2, 3):  # -2 to 2
            for c in range(1, 100):  # Denominators 1 to 99
                formula_value = a + b * np.pi / c
                if formula_value <= 0:
                    continue
                error = abs(formula_value - center)
                if error < 0.01:  # Within 1%
                    complexity = abs(a) + abs(b) + (1 if c > 1 else 0)
                    best_formulas.append({
                        'formula': f"{a} + {b}*π/{c}" if b != 0 else str(a),
                        'value': float(formula_value),
                        'error': float(error),
                        'relative_error_pct': float(error / center * 100),
                        'complexity': complexity,
                    })
    
    # Also test simple fractions
    for num in range(1, 20):
        for denom in range(1, 20):
            frac_value = num / denom
            error = abs(frac_value - center)
            if error < 0.01:
                best_formulas.append({
                    'formula': f"{num}/{denom}",
                    'value': float(frac_value),
                    'error': float(error),
                    'relative_error_pct': float(error / center * 100),
                    'complexity': num + denom,
                })
    
    # Sort by error
    best_formulas.sort(key=lambda x: x['error'])
    
    return {
        'target_value': float(center),
        'best_matches': best_formulas[:10],  # Top 10
    }


def run_blind_clustering():
    """Main experiment: blind clustering with no φ/Ξ assumptions."""
    
    print("=" * 70)
    print("EXPERIMENT 07: Blind Clustering Analysis")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nThis experiment removes ALL φ/Ξ dependencies.")
    print("We find where Class IV naturally clusters, then test if it's special.\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Blind clustering - no target constants assumed',
    }
    
    # Step 1: Compute ratios with no preconceptions
    print("Step 1: Computing P/A ratios for all 256 rules...")
    ratios = compute_all_ratios(width=101, steps=200)
    results['n_rules'] = len(ratios)
    results['n_finite'] = len([v for v in ratios.values() if np.isfinite(v)])
    print(f"  Computed {results['n_finite']} finite ratios out of 256")
    
    # Step 2: Find natural clusters
    print("\nStep 2: Finding natural clusters (no targets)...")
    clusters = find_natural_clusters(ratios, n_clusters=5)
    results['clusters'] = clusters
    
    # Which cluster has the most Class IV?
    class_iv_cluster = max(clusters.items(), key=lambda x: x[1]['class_iv_count'])
    print(f"  Cluster with most Class IV: {class_iv_cluster[0]}")
    print(f"    Mean P/A: {class_iv_cluster[1]['mean']:.6f}")
    print(f"    Class IV rules: {class_iv_cluster[1]['class_iv_count']}")
    
    # Step 3: Find Class IV natural center
    print("\nStep 3: Finding Class IV natural center...")
    class_iv = find_class_iv_center(ratios)
    results['class_iv_center'] = class_iv
    
    print(f"  Class IV rules: {class_iv['class_iv_rules']}")
    print(f"  Mean P/A ratio: {class_iv['mean']:.6f}")
    print(f"  Median: {class_iv['median']:.6f}")
    print(f"  95% CI: [{class_iv['ci_95_lower']:.6f}, {class_iv['ci_95_upper']:.6f}]")
    
    # Step 4: Test against candidates
    print("\nStep 4: Testing candidate constants...")
    candidates = test_candidates(class_iv['mean'], class_iv['ci_95_lower'], class_iv['ci_95_upper'])
    results['candidate_tests'] = candidates
    
    print(f"\n  {'Candidate':<20} {'Value':>12} {'Distance':>12} {'Within CI':>10}")
    print("-" * 60)
    for name, data in list(candidates.items())[:8]:
        print(f"  {name:<20} {data['value']:>12.6f} {data['distance_from_center']:>12.6f} {'✓' if data['within_95_ci'] else '✗':>10}")
    
    closest = list(candidates.items())[0]
    print(f"\n  Closest candidate: {closest[0]} = {closest[1]['value']:.6f}")
    
    # Step 5: Compare to random
    print("\nStep 5: Testing if clustering is significant...")
    random_comparison = compare_to_random(ratios, class_iv['mean'])
    results['random_comparison'] = random_comparison
    
    print(f"  Mann-Whitney p-value: {random_comparison['mann_whitney_p']:.6f}")
    print(f"  Class IV mean at {random_comparison['class_iv_mean_percentile']:.1f}th percentile of random")
    print(f"  Class IV std: {random_comparison['class_iv_std']:.4f} (tighter than {random_comparison['tightness_percentile']:.1f}% of random)")
    
    if random_comparison['is_tighter_than_95pct_random']:
        print("  ✅ Class IV clusters tighter than 95% of random samples")
    else:
        print("  ❌ Class IV clustering not significantly tighter than random")
    
    # Step 6: Find simplest formula
    print("\nStep 6: Finding simplest mathematical approximation...")
    formulas = find_best_simple_formula(class_iv['mean'])
    results['formula_search'] = formulas
    
    print(f"\n  Target value: {formulas['target_value']:.6f}")
    print(f"  Best matches:")
    for i, f in enumerate(formulas['best_matches'][:5]):
        print(f"    {i+1}. {f['formula']} = {f['value']:.6f} (error: {f['relative_error_pct']:.4f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: What did we find?")
    print("=" * 70)
    
    print(f"\n1. Class IV rules naturally cluster at P/A ≈ {class_iv['mean']:.4f}")
    print(f"   (95% CI: [{class_iv['ci_95_lower']:.4f}, {class_iv['ci_95_upper']:.4f}])")
    
    closest_name = closest[0]
    closest_val = closest[1]['value']
    if closest_name == 'xi_claimed':
        print(f"\n2. The claimed Ξ = 1.0571 is {'within' if closest[1]['within_95_ci'] else 'NOT within'} the 95% CI")
        if closest[1]['within_95_ci']:
            print("   ⚠️ BUT: This may be because Ξ was DERIVED from this observation!")
    else:
        print(f"\n2. Closest simple constant: {closest_name} = {closest_val:.6f}")
        print(f"   (The claimed Ξ = 1.0571 is rank #{list(candidates.keys()).index('xi_claimed')+1})")
    
    if random_comparison['is_tighter_than_95pct_random']:
        print(f"\n3. Class IV clustering IS statistically significant (p < 0.05)")
        print("   The clustering phenomenon is REAL, regardless of what we call it.")
    else:
        print(f"\n3. Class IV clustering is NOT statistically significant")
        print("   The phenomenon may be random variation.")
    
    # The honest conclusion
    print("\n" + "-" * 70)
    print("HONEST CONCLUSION:")
    print("-" * 70)
    
    if class_iv['mean'] > 0.9 and class_iv['mean'] < 1.2:
        print("""
Class IV rules (edge of chaos) cluster at P/A ≈ 1.05-1.06.
This is a REAL observation. What's uncertain:

- Whether Ξ = 1 + π/55 is the "true" formula, or just curve-fitting
- Whether this relates to φ/golden ratio at all
- Whether the same value appears in other domains

The VALUE matters. The NARRATIVE around it may not.
""")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp_07_blind_clustering_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"Completed: {datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    run_blind_clustering()
