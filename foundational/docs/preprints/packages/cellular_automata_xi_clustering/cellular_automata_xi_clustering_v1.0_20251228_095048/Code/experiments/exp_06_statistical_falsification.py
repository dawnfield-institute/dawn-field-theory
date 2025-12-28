#!/usr/bin/env python3
"""
Experiment 06: Statistical Falsification Battery
=================================================

Rigorous statistical tests to prove results aren't due to chance:
1. Bootstrap confidence intervals for Rule 110's P/A ratio
2. Permutation test: Class IV vs other classes
3. Kolmogorov-Smirnov test for distribution differences
4. Monte Carlo: probability of random systems hitting Ξ
5. Effect size (Cohen's d) calculations

Goal: Prove p < 0.0001 that Class IV rules cluster near Ξ by chance.
"""

import sys
import os
import json
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass
from pac_embedding import PACEmbedder

# Constants
XI = 1.0571
XI_TOLERANCE = 0.05  # Within 5% of Ξ


class StatisticalFalsifier:
    """Run rigorous statistical tests to rule out chance."""
    
    def __init__(self, width: int = 101, steps: int = 200):
        self.width = width
        self.steps = steps
        self.embedder = PACEmbedder(width=width, steps=steps)
        
    def get_pa_ratio(self, rule: int) -> float:
        """Get P/A ratio for a rule."""
        coords = self.embedder.embed_rule(rule)
        if coords.actualization > 0.001:
            return coords.potential / coords.actualization
        return float('inf')
    
    def get_pa_ratio_random_init(self, rule: int) -> float:
        """Get P/A ratio with random initial condition."""
        coords = self.embedder.embed_rule(rule, init_type='random')
        if coords.actualization > 0.001:
            return coords.potential / coords.actualization
        return float('inf')
    
    def test_bootstrap_confidence(self, n_bootstrap: int = 10000) -> Dict:
        """
        Test 1: Bootstrap confidence intervals for Rule 110's P/A ratio.
        
        If 95% CI contains Ξ, result is robust.
        """
        print("=== Test 1: Bootstrap Confidence Intervals ===")
        
        # Collect Rule 110 measurements with different conditions
        measurements = []
        
        # Single cell init (standard)
        for _ in range(100):
            ratio = self.get_pa_ratio(110)
            if ratio < 100:
                measurements.append(ratio)
        
        # Random init
        for _ in range(100):
            ratio = self.get_pa_ratio_random_init(110)
            if ratio < 100:
                measurements.append(ratio)
        
        measurements = np.array(measurements)
        
        # Bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(measurements, size=len(measurements), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        mean = np.mean(measurements)
        std = np.std(measurements)
        
        contains_xi = ci_lower <= XI <= ci_upper
        
        print(f"  Mean P/A ratio: {mean:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Contains Ξ={XI}: {contains_xi}")
        
        return {
            'n_measurements': len(measurements),
            'mean': float(mean),
            'std': float(std),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'ci_width': float(ci_upper - ci_lower),
            'contains_xi': contains_xi,
            'distance_from_xi': float(abs(mean - XI))
        }
    
    def test_permutation(self, n_permutations: int = 10000) -> Dict:
        """
        Test 2: Permutation test - Class IV vs other classes.
        
        Null hypothesis: Class IV is no different from other classes.
        """
        print("\n=== Test 2: Permutation Test ===")
        
        # Get ratios for Class IV
        class_iv_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                        if wc == WolframClass.CLASS_IV]
        class_iv_distances = []
        for rule in class_iv_rules:
            ratio = self.get_pa_ratio(rule)
            if ratio < 100:
                class_iv_distances.append(abs(ratio - XI))
        
        # Get ratios for other classes
        other_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                      if wc != WolframClass.CLASS_IV]
        other_distances = []
        for rule in other_rules:
            ratio = self.get_pa_ratio(rule)
            if ratio < 100:
                other_distances.append(abs(ratio - XI))
        
        # Observed difference (other classes are FARTHER from Ξ)
        observed_diff = np.mean(other_distances) - np.mean(class_iv_distances)
        
        print(f"  Class IV mean distance from Ξ: {np.mean(class_iv_distances):.4f}")
        print(f"  Other classes mean distance: {np.mean(other_distances):.4f}")
        print(f"  Observed difference: {observed_diff:.4f}")
        
        # Permutation distribution under null
        all_distances = class_iv_distances + other_distances
        n_iv = len(class_iv_distances)
        
        perm_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(all_distances)
            perm_iv = all_distances[:n_iv]
            perm_other = all_distances[n_iv:]
            perm_diffs.append(np.mean(perm_other) - np.mean(perm_iv))
        
        perm_diffs = np.array(perm_diffs)
        
        # One-tailed p-value (we expect Class IV to be CLOSER)
        p_value = np.mean(perm_diffs >= observed_diff)
        
        print(f"  Permutation p-value: {p_value:.6f}")
        
        return {
            'class_iv_n': len(class_iv_distances),
            'class_iv_mean_distance': float(np.mean(class_iv_distances)),
            'other_n': len(other_distances),
            'other_mean_distance': float(np.mean(other_distances)),
            'observed_difference': float(observed_diff),
            'n_permutations': n_permutations,
            'p_value': float(p_value),
            'significant_001': p_value < 0.001,
            'significant_0001': p_value < 0.0001
        }
    
    def test_kolmogorov_smirnov(self) -> Dict:
        """
        Test 3: Kolmogorov-Smirnov test for distribution differences.
        
        Tests if Class IV P/A distribution is different from other classes.
        """
        print("\n=== Test 3: Kolmogorov-Smirnov Test ===")
        
        # Collect many samples with different initial conditions
        class_iv_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                        if wc == WolframClass.CLASS_IV]
        
        class_iv_ratios = []
        for rule in class_iv_rules:
            for _ in range(50):
                ratio = self.get_pa_ratio(rule)
                if ratio < 100:
                    class_iv_ratios.append(ratio)
                ratio = self.get_pa_ratio_random_init(rule)
                if ratio < 100:
                    class_iv_ratios.append(ratio)
        
        other_ratios = []
        other_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                      if wc != WolframClass.CLASS_IV]
        for rule in other_rules:
            for _ in range(10):
                ratio = self.get_pa_ratio(rule)
                if ratio < 100:
                    other_ratios.append(ratio)
        
        class_iv_ratios = np.array(class_iv_ratios)
        other_ratios = np.array(other_ratios)
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(class_iv_ratios, other_ratios)
        
        print(f"  Class IV samples: {len(class_iv_ratios)}")
        print(f"  Other samples: {len(other_ratios)}")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  KS p-value: {ks_p:.6f}")
        
        return {
            'class_iv_n': len(class_iv_ratios),
            'class_iv_mean': float(np.mean(class_iv_ratios)),
            'class_iv_std': float(np.std(class_iv_ratios)),
            'other_n': len(other_ratios),
            'other_mean': float(np.mean(other_ratios)),
            'other_std': float(np.std(other_ratios)),
            'ks_statistic': float(ks_stat),
            'p_value': float(ks_p),
            'distributions_different': ks_p < 0.05
        }
    
    def test_monte_carlo(self, n_trials: int = 100000) -> Dict:
        """
        Test 4: Monte Carlo - probability of randomly hitting Ξ.
        
        What's the probability that a random system has P/A ≈ Ξ?
        """
        print("\n=== Test 4: Monte Carlo Random Baseline ===")
        
        # Random P/A ratios (uniform on [0,1] for each)
        random_near_xi = 0
        random_ratios = []
        
        for _ in range(n_trials):
            p = np.random.random()
            a = np.random.random()
            
            if a > 0.01:  # Avoid division by near-zero
                ratio = p / a
                random_ratios.append(ratio)
                
                if abs(ratio - XI) < XI_TOLERANCE:
                    random_near_xi += 1
        
        prob_random = random_near_xi / len(random_ratios)
        
        # How many Class IV rules are near Ξ?
        class_iv_rules = [r for r, wc in RULE_CLASSIFICATIONS.items() 
                        if wc == WolframClass.CLASS_IV]
        class_iv_near_xi = 0
        for rule in class_iv_rules:
            ratio = self.get_pa_ratio(rule)
            if ratio < 100 and abs(ratio - XI) < XI_TOLERANCE:
                class_iv_near_xi += 1
        
        prob_class_iv = class_iv_near_xi / len(class_iv_rules)
        
        enrichment = prob_class_iv / (prob_random + 1e-10)
        
        print(f"  Random probability of hitting Ξ: {prob_random:.6f}")
        print(f"  Class IV probability: {prob_class_iv:.4f}")
        print(f"  Enrichment factor: {enrichment:.1f}x")
        
        # Binomial test: is Class IV enrichment significant?
        # Under null, each Class IV rule has prob_random chance of hitting Ξ
        binom_p = stats.binom.sf(class_iv_near_xi - 1, len(class_iv_rules), prob_random)
        
        print(f"  Binomial p-value: {binom_p:.6f}")
        
        return {
            'n_random_trials': n_trials,
            'random_near_xi': random_near_xi,
            'prob_random': float(prob_random),
            'class_iv_total': len(class_iv_rules),
            'class_iv_near_xi': class_iv_near_xi,
            'prob_class_iv': float(prob_class_iv),
            'enrichment_factor': float(enrichment),
            'binomial_p_value': float(binom_p),
            'significant': binom_p < 0.05
        }
    
    def test_effect_size(self) -> Dict:
        """
        Test 5: Effect size (Cohen's d) calculation.
        
        How large is the difference between Class IV and others?
        """
        print("\n=== Test 5: Effect Size (Cohen's d) ===")
        
        # Get distances from Ξ for each class
        class_distances = {wc.name: [] for wc in WolframClass}
        
        for rule, wc in RULE_CLASSIFICATIONS.items():
            ratio = self.get_pa_ratio(rule)
            if ratio < 100:
                class_distances[wc.name].append(abs(ratio - XI))
        
        # Cohen's d: (mean1 - mean2) / pooled_std
        class_iv_d = np.array(class_distances['CLASS_IV'])
        
        results = {'class_comparisons': {}}
        
        for class_name, distances in class_distances.items():
            if class_name == 'CLASS_IV' or len(distances) == 0:
                continue
            
            other_d = np.array(distances)
            
            # Pooled standard deviation
            n1, n2 = len(class_iv_d), len(other_d)
            s1, s2 = np.std(class_iv_d, ddof=1), np.std(other_d, ddof=1)
            
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(other_d) - np.mean(class_iv_d)) / pooled_std
            else:
                cohens_d = 0
            
            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                interpretation = 'negligible'
            elif abs(cohens_d) < 0.5:
                interpretation = 'small'
            elif abs(cohens_d) < 0.8:
                interpretation = 'medium'
            else:
                interpretation = 'large'
            
            results['class_comparisons'][class_name] = {
                'cohens_d': float(cohens_d),
                'interpretation': interpretation,
                'class_iv_mean': float(np.mean(class_iv_d)),
                'other_mean': float(np.mean(other_d))
            }
            
            print(f"  Class IV vs {class_name}: d={cohens_d:.2f} ({interpretation})")
        
        return results
    
    def test_all_256_rules(self) -> Dict:
        """
        Test 6: Analyze all 256 rules, not just classified ones.
        
        Do unclassified rules near Ξ behave like Class IV?
        """
        print("\n=== Test 6: Full 256-Rule Analysis ===")
        
        all_ratios = []
        near_xi_rules = []
        far_from_xi_rules = []
        
        for rule in range(256):
            ratio = self.get_pa_ratio(rule)
            if ratio < 100:
                all_ratios.append((rule, ratio))
                
                if abs(ratio - XI) < XI_TOLERANCE:
                    near_xi_rules.append(rule)
                elif abs(ratio - XI) > 0.5:
                    far_from_xi_rules.append(rule)
        
        # Check Wolfram class of near-Ξ rules
        near_xi_classes = []
        for rule in near_xi_rules:
            wc = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
            near_xi_classes.append(wc.name)
        
        class_counts = Counter(near_xi_classes)
        
        print(f"  Rules near Ξ: {len(near_xi_rules)}")
        print(f"  Class distribution: {dict(class_counts)}")
        
        # What fraction of near-Ξ rules are Class IV?
        class_iv_fraction = class_counts.get('CLASS_IV', 0) / len(near_xi_rules) if near_xi_rules else 0
        
        print(f"  Fraction that are Class IV: {class_iv_fraction:.2%}")
        
        return {
            'total_rules': 256,
            'near_xi_count': len(near_xi_rules),
            'near_xi_rules': near_xi_rules,
            'class_distribution': dict(class_counts),
            'class_iv_fraction': float(class_iv_fraction)
        }


def main():
    print("=" * 70)
    print("EXPERIMENT 06: Statistical Falsification Battery")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Testing null hypothesis: Class IV rules are NOT special near Ξ")
    print()
    
    falsifier = StatisticalFalsifier(width=101, steps=200)
    
    # Run all tests
    test1 = falsifier.test_bootstrap_confidence()
    test2 = falsifier.test_permutation()
    test3 = falsifier.test_kolmogorov_smirnov()
    test4 = falsifier.test_monte_carlo()
    test5 = falsifier.test_effect_size()
    test6 = falsifier.test_all_256_rules()
    
    # Summary
    print("\n" + "=" * 70)
    print("FALSIFICATION SUMMARY")
    print("=" * 70)
    
    tests_passed = {
        'bootstrap_contains_xi': test1['contains_xi'],
        'permutation_significant': test2['significant_001'],
        'ks_distributions_different': test3['distributions_different'],
        'monte_carlo_enriched': test4['significant'],
        'large_effect_size': any(c['interpretation'] == 'large' 
                                 for c in test5['class_comparisons'].values())
    }
    
    all_pass = all(tests_passed.values())
    
    for test_name, passed in tests_passed.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}: {passed}")
    
    # Compute combined p-value (Fisher's method)
    p_values = [
        test2['p_value'],
        test3['p_value'],
        test4['binomial_p_value']
    ]
    p_values = [p for p in p_values if p > 0]
    
    if p_values:
        chi2_stat = -2 * sum(np.log(p) for p in p_values)
        combined_p = stats.chi2.sf(chi2_stat, 2 * len(p_values))
    else:
        combined_p = 1.0
    
    print(f"\n  Combined p-value (Fisher's method): {combined_p:.2e}")
    
    if combined_p < 0.0001:
        print("\n  🎯 RESULTS ARE STATISTICALLY UNDENIABLE")
        print(f"     Probability of chance: < 0.01%")
    elif combined_p < 0.01:
        print("\n  ✅ Results are highly significant (p < 0.01)")
    elif combined_p < 0.05:
        print("\n  ⚠️ Results are significant but need more data")
    else:
        print("\n  ❌ Results not significant - needs investigation")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp_06_falsification_{timestamp}.json"
    
    output = {
        "experiment": "exp_06_statistical_falsification",
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "bootstrap": test1,
            "permutation": test2,
            "kolmogorov_smirnov": test3,
            "monte_carlo": test4,
            "effect_size": test5,
            "full_256": test6
        },
        "summary": {
            "tests_passed": tests_passed,
            "all_pass": all_pass,
            "combined_p_value": float(combined_p),
            "conclusion": "Class IV rules cluster near Ξ with p < 0.0001" if combined_p < 0.0001 else "Needs more data"
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
