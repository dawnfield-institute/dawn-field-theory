#!/usr/bin/env python3
"""
Experiment 12: Ξ Random Baseline
================================

CORRECTED FALSIFICATION TEST

Tests whether ~1.057 is special or whether random systems also cluster there.

If random systems cluster at ~1.057 → Ξ is an artifact of our metrics
If only complex systems cluster there → Ξ marks genuine complexity boundary

Methodology:
- Generate random transition matrices (simulating "random rules")
- Compute P/A ratios using same embedding methodology as CA
- Compare distribution to Class IV distribution
"""

import sys
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass
from pac_embedding import PACEmbedder

XI = 1.0571428


def generate_random_transition_stats(n_samples: int = 1000, width: int = 101, steps: int = 200):
    """
    Generate random "rule-like" systems and compute P/A ratios.
    
    Uses random binary matrices as state evolution instead of CA rules.
    Applies same embedding metrics to measure P/A.
    """
    random_ratios = []
    
    for _ in range(n_samples):
        # Generate random binary evolution (like a random CA)
        history = np.random.randint(0, 2, size=(steps, width), dtype=np.uint8)
        
        # Compute same metrics as PACEmbedder
        # Entropy
        densities = history.mean(axis=1)
        hist, _ = np.histogram(densities, bins=20, range=(0, 1), density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            entropy = 0.0
        else:
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Mutual information
        joint_counts = {}
        for t in range(steps - 1):
            for i in range(width):
                key = (history[t, i], history[t + 1, i])
                joint_counts[key] = joint_counts.get(key, 0) + 1
        total = sum(joint_counts.values())
        if total == 0:
            mutual_info = 0.0
        else:
            p_x = np.zeros(2)
            p_y = np.zeros(2)
            for (x, y), count in joint_counts.items():
                p_x[x] += count
                p_y[y] += count
            p_x /= total
            p_y /= total
            mutual_info = 0.0
            for (x, y), count in joint_counts.items():
                p_xy = count / total
                if p_xy > 0 and p_x[x] > 0 and p_y[y] > 0:
                    mutual_info += p_xy * np.log2(p_xy / (p_x[x] * p_y[y]))
        
        # Structure factor
        power_spectra = []
        for row in history[steps // 2:]:
            fft = np.fft.fft(row.astype(float) - row.mean())
            power = np.abs(fft) ** 2
            power_spectra.append(power)
        avg_power = np.mean(power_spectra, axis=0)
        non_dc = avg_power[1:len(avg_power)//2]
        if len(non_dc) > 0 and np.mean(non_dc) > 0:
            structure = np.max(non_dc) / (np.mean(non_dc) + 1e-10)
        else:
            structure = 0.0
        
        # P/A calculation
        max_entropy = np.log2(20)
        norm_entropy = min(entropy / max_entropy, 1.0)
        potential = 1.0 - norm_entropy
        
        actualization = 0.5 * mutual_info + 0.3 * min(structure / 10, 1.0) + 0.1
        actualization = min(max(actualization, 0.01), 1.0)
        total = potential + actualization
        if total > 0:
            potential /= total
            actualization /= total
        
        if actualization > 0.001:
            ratio = potential / actualization
            if ratio < 50:  # Filter degenerate
                random_ratios.append(ratio)
    
    return np.array(random_ratios)


def get_class_iv_ratios():
    """Get P/A ratios for Class IV rules using standard embedding."""
    embedder = PACEmbedder(width=101, steps=200)
    class_iv_rules = [r for r, c in RULE_CLASSIFICATIONS.items() if c == WolframClass.CLASS_IV]
    
    ratios = []
    for rule in class_iv_rules:
        coords = embedder.embed_rule(rule)
        if coords.actualization > 0.001:
            ratio = coords.potential / coords.actualization
            if ratio < 50:
                ratios.append(ratio)
    
    return np.array(ratios), class_iv_rules


def run_random_baseline_test():
    """Run the random baseline falsification test."""
    
    print("=" * 70)
    print("EXPERIMENT 12: Ξ Random Baseline Test")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nFalsification Question: Does ~1.057 appear in random systems?")
    print("If yes → Ξ is metric artifact. If no → Ξ marks genuine complexity.\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'xi': XI,
        'falsification_target': 'Is Ξ ≈ 1.057 an artifact of embedding metrics?'
    }
    
    # Generate random baseline
    print("Generating 1000 random system P/A ratios...")
    random_ratios = generate_random_transition_stats(n_samples=1000)
    print(f"  Valid samples: {len(random_ratios)}")
    
    # Get Class IV ratios
    print("\nComputing Class IV P/A ratios...")
    class_iv_ratios, class_iv_rules = get_class_iv_ratios()
    print(f"  Class IV rules: {class_iv_rules}")
    print(f"  Valid ratios: {len(class_iv_ratios)}")
    
    # Compare distributions
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON")
    print("=" * 70)
    
    print(f"\nRandom systems:")
    print(f"  Mean P/A: {np.mean(random_ratios):.4f} ± {np.std(random_ratios):.4f}")
    print(f"  Median P/A: {np.median(random_ratios):.4f}")
    print(f"  Range: [{np.min(random_ratios):.4f}, {np.max(random_ratios):.4f}]")
    
    print(f"\nClass IV systems:")
    print(f"  Mean P/A: {np.mean(class_iv_ratios):.4f} ± {np.std(class_iv_ratios):.4f}")
    print(f"  Median P/A: {np.median(class_iv_ratios):.4f}")
    print(f"  Range: [{np.min(class_iv_ratios):.4f}, {np.max(class_iv_ratios):.4f}]")
    
    results['random_stats'] = {
        'n': len(random_ratios),
        'mean': float(np.mean(random_ratios)),
        'std': float(np.std(random_ratios)),
        'median': float(np.median(random_ratios)),
        'min': float(np.min(random_ratios)),
        'max': float(np.max(random_ratios))
    }
    
    results['class_iv_stats'] = {
        'n': len(class_iv_ratios),
        'mean': float(np.mean(class_iv_ratios)),
        'std': float(np.std(class_iv_ratios)),
        'median': float(np.median(class_iv_ratios)),
        'ratios': class_iv_ratios.tolist()
    }
    
    # Key test: How many random systems fall near Ξ?
    print("\n" + "=" * 70)
    print("Ξ PROXIMITY TEST")
    print("=" * 70)
    
    xi_tolerance = 0.05  # Within 5% of Ξ
    xi_low = XI * (1 - xi_tolerance)
    xi_high = XI * (1 + xi_tolerance)
    
    random_near_xi = np.sum((random_ratios >= xi_low) & (random_ratios <= xi_high))
    random_near_xi_pct = 100 * random_near_xi / len(random_ratios)
    
    class_iv_near_xi = np.sum((class_iv_ratios >= xi_low) & (class_iv_ratios <= xi_high))
    class_iv_near_xi_pct = 100 * class_iv_near_xi / len(class_iv_ratios) if len(class_iv_ratios) > 0 else 0
    
    print(f"\nWithin 5% of Ξ = {XI} (range [{xi_low:.4f}, {xi_high:.4f}]):")
    print(f"  Random systems: {random_near_xi}/{len(random_ratios)} ({random_near_xi_pct:.1f}%)")
    print(f"  Class IV systems: {class_iv_near_xi}/{len(class_iv_ratios)} ({class_iv_near_xi_pct:.1f}%)")
    
    results['xi_proximity'] = {
        'tolerance': xi_tolerance,
        'xi_low': float(xi_low),
        'xi_high': float(xi_high),
        'random_near_xi': int(random_near_xi),
        'random_near_xi_pct': float(random_near_xi_pct),
        'class_iv_near_xi': int(class_iv_near_xi),
        'class_iv_near_xi_pct': float(class_iv_near_xi_pct)
    }
    
    # Statistical test: Is Class IV enriched near Ξ compared to random?
    print("\n" + "=" * 70)
    print("STATISTICAL TEST")
    print("=" * 70)
    
    # Fisher's exact test for enrichment near Ξ
    contingency = [
        [class_iv_near_xi, len(class_iv_ratios) - class_iv_near_xi],
        [random_near_xi, len(random_ratios) - random_near_xi]
    ]
    
    odds_ratio, fisher_p = stats.fisher_exact(contingency)
    
    print(f"\nFisher's exact test (Class IV enrichment near Ξ):")
    print(f"  Odds ratio: {odds_ratio:.2f}")
    print(f"  p-value: {fisher_p:.6f}")
    
    results['fisher_test'] = {
        'odds_ratio': float(odds_ratio),
        'p_value': float(fisher_p),
        'significant': bool(fisher_p < 0.05)
    }
    
    # Mann-Whitney U test for distribution difference
    u_stat, mw_p = stats.mannwhitneyu(class_iv_ratios, random_ratios, alternative='two-sided')
    print(f"\nMann-Whitney U test (distribution difference):")
    print(f"  U statistic: {u_stat:.1f}")
    print(f"  p-value: {mw_p:.6f}")
    
    results['mann_whitney'] = {
        'u_statistic': float(u_stat),
        'p_value': float(mw_p),
        'significant': bool(mw_p < 0.05)
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("FALSIFICATION VERDICT")
    print("=" * 70)
    
    if random_near_xi_pct > 10:
        verdict = "POTENTIAL ARTIFACT"
        explanation = f"Random systems cluster near Ξ at {random_near_xi_pct:.1f}% rate - Ξ may be a metric artifact"
    elif fisher_p < 0.05 and odds_ratio > 1:
        verdict = "Ξ IS GENUINE"
        explanation = f"Class IV is {odds_ratio:.1f}x more likely to be near Ξ than random (p={fisher_p:.4f})"
    else:
        verdict = "INCONCLUSIVE"
        explanation = "Cannot distinguish Class IV from random at Ξ proximity"
    
    print(f"\nVerdict: {verdict}")
    print(f"Explanation: {explanation}")
    
    results['verdict'] = verdict
    results['explanation'] = explanation
    
    # Save results
    output_path = Path(__file__).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"exp_12_random_baseline_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved: {output_file}")
    
    return results


if __name__ == "__main__":
    run_random_baseline_test()
