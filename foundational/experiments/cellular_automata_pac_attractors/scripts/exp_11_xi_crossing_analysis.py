#!/usr/bin/env python3
"""
Experiment 11: Ξ-Crossing Analysis
===================================

Tests whether Class IV rules uniquely cross through Ξ ≈ 1.0571
during their P/A trajectory evolution.

Hypothesis: Class IV rules orbit around Ξ (dynamic equilibrium),
while other classes stay on one side (static or chaotic).
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass

WIDTH = 101
MAX_STEPS = 500
XI = 1.0571428


def compute_pa_fast(history, width):
    """Fast P/A computation matching PACEmbedder methodology."""
    steps = len(history)
    
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
        return 100.0
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
    if power_spectra:
        avg_power = np.mean(power_spectra, axis=0)
        non_dc = avg_power[1:len(avg_power)//2]
        if len(non_dc) > 0 and np.mean(non_dc) > 0:
            structure = np.max(non_dc) / (np.mean(non_dc) + 1e-10)
        else:
            structure = 0.0
    else:
        structure = 0.0
    
    # P/A calculation
    max_entropy = np.log2(20)
    norm_entropy = min(entropy / max_entropy, 1.0)
    potential = 1.0 - norm_entropy
    
    actualization = 0.5 * mutual_info + 0.3 * min(structure / 10, 1.0) + 0.1
    actualization = min(max(actualization, 0.01), 1.0)
    total = potential + actualization
    potential /= total
    actualization /= total
    return potential / actualization


def analyze_trajectory(rule):
    """Analyze P/A trajectory for a single rule."""
    ca = ElementaryCA(rule, WIDTH)
    state = ca.evolve_fast(MAX_STEPS, init_type='single')
    
    pas = []
    for steps in range(100, MAX_STEPS + 1, 25):
        history = state.history[:steps]
        pa = compute_pa_fast(history, WIDTH)
        if pa < 50:  # Filter degenerate cases
            pas.append(pa)
    
    if not pas:
        return {
            'crosses_xi': False,
            'min_dist': 100.0,
            'mean_dist': 100.0,
            'n_crossings': 0,
            'pa_range': (100.0, 100.0)
        }
    
    pas = np.array(pas)
    distances = np.abs(pas - XI)
    n_crossings = sum(1 for i in range(len(pas)-1) 
                      if (pas[i] - XI) * (pas[i+1] - XI) < 0)
    
    return {
        'crosses_xi': n_crossings > 0,
        'min_dist': float(np.min(distances)),
        'mean_dist': float(np.mean(distances)),
        'n_crossings': n_crossings,
        'pa_range': (float(np.min(pas)), float(np.max(pas)))
    }


def run_crossing_analysis():
    """Run full Ξ-crossing analysis."""
    
    print("=" * 70)
    print("EXPERIMENT 11: Ξ-Crossing Analysis")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nHypothesis: Class IV rules orbit around Ξ = {XI}")
    print(f"Other classes stay on one side of Ξ")
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'xi': XI,
        'parameters': {'width': WIDTH, 'max_steps': MAX_STEPS},
        'rule_results': [],
        'class_summaries': {}
    }
    
    # Analyze all classified rules
    class_data = {cls: {'crosses': 0, 'total': 0, 'min_dists': [], 'crossings': []} 
                  for cls in WolframClass}
    
    for rule, cls in sorted(RULE_CLASSIFICATIONS.items()):
        result = analyze_trajectory(rule)
        results['rule_results'].append({
            'rule': rule,
            'class': cls.name,
            **result
        })
        
        class_data[cls]['total'] += 1
        if result['crosses_xi']:
            class_data[cls]['crosses'] += 1
        class_data[cls]['min_dists'].append(result['min_dist'])
        class_data[cls]['crossings'].append(result['n_crossings'])
    
    # Summary by class
    print("=" * 70)
    print("SUMMARY BY WOLFRAM CLASS")
    print("=" * 70)
    print(f"{'Class':<12} {'Rules':<6} {'Cross Ξ':<12} {'Mean Min Dist':<15} {'Avg Crossings':<12}")
    print("-" * 70)
    
    for cls in [WolframClass.CLASS_I, WolframClass.CLASS_II, 
                WolframClass.CLASS_III, WolframClass.CLASS_IV]:
        data = class_data[cls]
        if data['total'] > 0:
            pct = 100 * data['crosses'] / data['total']
            valid_dists = [d for d in data['min_dists'] if d < 50]
            mean_min = np.mean(valid_dists) if valid_dists else 100.0
            avg_cross = np.mean(data['crossings'])
            
            results['class_summaries'][cls.name] = {
                'total_rules': data['total'],
                'rules_crossing_xi': data['crosses'],
                'crossing_percentage': pct,
                'mean_min_distance': mean_min,
                'avg_n_crossings': avg_cross
            }
            
            cross_str = f"{data['crosses']}/{data['total']} ({pct:.0f}%)"
            print(f"{cls.name:<12} {data['total']:<6} {cross_str:<12} {mean_min:<15.4f} {avg_cross:<12.2f}")
    
    # Statistical test: Is Class IV more likely to cross Ξ?
    print("\n" + "=" * 70)
    print("STATISTICAL TEST: Class IV vs Others")
    print("=" * 70)
    
    from scipy import stats
    
    class_iv_crosses = class_data[WolframClass.CLASS_IV]['crosses']
    class_iv_total = class_data[WolframClass.CLASS_IV]['total']
    
    other_crosses = sum(class_data[cls]['crosses'] 
                       for cls in [WolframClass.CLASS_I, WolframClass.CLASS_II, WolframClass.CLASS_III])
    other_total = sum(class_data[cls]['total'] 
                     for cls in [WolframClass.CLASS_I, WolframClass.CLASS_II, WolframClass.CLASS_III])
    
    # Fisher's exact test
    contingency = [[class_iv_crosses, class_iv_total - class_iv_crosses],
                   [other_crosses, other_total - other_crosses]]
    
    odds_ratio, p_value = stats.fisher_exact(contingency)
    
    print(f"\nClass IV crossing rate: {class_iv_crosses}/{class_iv_total} = {100*class_iv_crosses/class_iv_total:.1f}%")
    print(f"Other classes crossing rate: {other_crosses}/{other_total} = {100*other_crosses/other_total:.1f}%")
    print(f"Odds ratio: {odds_ratio:.2f}")
    print(f"Fisher's exact p-value: {p_value:.6f}")
    
    results['fisher_test'] = {
        'class_iv_crosses': int(class_iv_crosses),
        'class_iv_total': int(class_iv_total),
        'other_crosses': int(other_crosses),
        'other_total': int(other_total),
        'odds_ratio': float(odds_ratio),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }
    
    if p_value < 0.05:
        if odds_ratio > 1:
            print("\n✓ SIGNIFICANT: Class IV rules are MORE likely to cross Ξ")
        else:
            print("\n✗ SIGNIFICANT: Class IV rules are LESS likely to cross Ξ")
    else:
        print("\n○ NOT SIGNIFICANT")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"exp_11_xi_crossing_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved: {output_file}")
    
    return results


if __name__ == "__main__":
    run_crossing_analysis()
