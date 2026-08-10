#!/usr/bin/env python3
"""
Experiment 09: Ξ as Convergence Detector
========================================

Tests the hypothesis that Ξ ≈ 1.0571 represents a stable attractor point
and that P/A ratio evolution can detect convergence state.

Hypothesis:
- Class I/II: Start chaotic, P/A converges quickly toward some value
- Class III: P/A fluctuates indefinitely, never stabilizes
- Class IV: P/A stabilizes AT or NEAR Ξ = 1.0571

Methodology:
- Track P/A ratio as function of evolution time
- Compute P/A using windowed analysis (not cumulative)
- Measure distance from Ξ over time
- Compare trajectory patterns by Wolfram class
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass

# Constants
XI = 1.0571428  # Target P/A ratio
WINDOW_SIZE = 20  # Steps per window for P/A computation


def compute_window_entropy(window: np.ndarray) -> float:
    """Compute entropy over a spatial window of CA states."""
    # Density per row
    densities = window.mean(axis=1)
    
    # Shannon entropy of density distribution
    hist, _ = np.histogram(densities, bins=20, range=(0, 1), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-10))


def compute_window_mi(window: np.ndarray) -> float:
    """Compute mutual information between consecutive rows."""
    joint_counts = {}
    for t in range(len(window) - 1):
        for i in range(window.shape[1]):
            key = (window[t, i], window[t + 1, i])
            joint_counts[key] = joint_counts.get(key, 0) + 1
    
    total = sum(joint_counts.values())
    if total == 0:
        return 0.0
    
    p_x = np.zeros(2)
    p_y = np.zeros(2)
    for (x, y), count in joint_counts.items():
        p_x[x] += count
        p_y[y] += count
    p_x /= total
    p_y /= total
    
    mi = 0.0
    for (x, y), count in joint_counts.items():
        p_xy = count / total
        if p_xy > 0 and p_x[x] > 0 and p_y[y] > 0:
            mi += p_xy * np.log2(p_xy / (p_x[x] * p_y[y]))
    return mi


def compute_window_structure(window: np.ndarray) -> float:
    """Compute structure factor for window."""
    power_spectra = []
    for row in window:
        fft = np.fft.fft(row.astype(float) - row.mean())
        power = np.abs(fft) ** 2
        power_spectra.append(power)
    
    avg_power = np.mean(power_spectra, axis=0)
    non_dc = avg_power[1:len(avg_power)//2]
    if len(non_dc) == 0 or np.mean(non_dc) == 0:
        return 0.0
    return np.max(non_dc) / (np.mean(non_dc) + 1e-10)


def compute_window_pa_ratio(window: np.ndarray) -> float:
    """
    Compute P/A ratio for a window of CA evolution.
    
    Uses same methodology as PACEmbedder but on a window.
    """
    entropy = compute_window_entropy(window)
    mutual_info = compute_window_mi(window)
    structure = compute_window_structure(window)
    
    # Normalize entropy
    max_entropy = np.log2(20)
    norm_entropy = min(entropy / max_entropy, 1.0)
    
    # P: Low entropy = high potential
    potential = 1.0 - norm_entropy
    
    # A: MI + structure combination
    actualization = 0.5 * mutual_info + 0.3 * min(structure / 10, 1.0)
    actualization = min(max(actualization, 0.001), 1.0)  # Avoid zero
    
    # Normalize
    total = potential + actualization
    if total > 0:
        potential /= total
        actualization /= total
    
    # P/A ratio
    return potential / actualization if actualization > 0 else 100.0


def track_pa_trajectory(rule: int, width: int = 101, total_steps: int = 400) -> Dict:
    """
    Track P/A ratio evolution over time.
    
    Returns trajectory data for analysis.
    """
    ca = ElementaryCA(rule, width)
    state = ca.evolve_fast(total_steps, init_type='single')
    
    # Compute P/A ratio for sliding windows
    n_windows = (total_steps - WINDOW_SIZE) // (WINDOW_SIZE // 2)  # 50% overlap
    
    pa_trajectory = []
    time_points = []
    
    for i in range(n_windows):
        start = i * (WINDOW_SIZE // 2)
        end = start + WINDOW_SIZE
        if end > total_steps:
            break
        
        window = state.history[start:end]
        pa = compute_window_pa_ratio(window)
        
        pa_trajectory.append(pa)
        time_points.append((start + end) / 2)  # Midpoint of window
    
    # Compute trajectory statistics
    pa_array = np.array(pa_trajectory)
    distances_from_xi = np.abs(pa_array - XI)
    
    # Split into early (first half) and late (second half)
    mid = len(pa_array) // 2
    early_pa = pa_array[:mid]
    late_pa = pa_array[mid:]
    early_dist = distances_from_xi[:mid]
    late_dist = distances_from_xi[mid:]
    
    return {
        'rule': rule,
        'wolfram_class': RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN).name,
        'trajectory': pa_trajectory,
        'time_points': time_points,
        'distances_from_xi': distances_from_xi.tolist(),
        'statistics': {
            'mean_pa': float(np.mean(pa_array)),
            'std_pa': float(np.std(pa_array)),
            'mean_distance_from_xi': float(np.mean(distances_from_xi)),
            'final_distance_from_xi': float(distances_from_xi[-1]) if len(distances_from_xi) > 0 else None,
            'early_mean_distance': float(np.mean(early_dist)),
            'late_mean_distance': float(np.mean(late_dist)),
            'convergence_ratio': float(np.mean(late_dist) / (np.mean(early_dist) + 1e-10)),
            'is_converging_to_xi': float(np.mean(late_dist)) < float(np.mean(early_dist)),
            'final_pa': float(pa_array[-1]) if len(pa_array) > 0 else None
        }
    }


def run_convergence_experiment():
    """Run convergence detection experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 09: Ξ as Convergence Detector")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nHypothesis: P/A ratio converges to different values by class")
    print(f"Target: Ξ = {XI}")
    print(f"Window size: {WINDOW_SIZE} steps")
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'xi': XI, 'window_size': WINDOW_SIZE},
        'trajectories': [],
        'class_summaries': {}
    }
    
    # Test representative rules from each class
    test_rules = {
        WolframClass.CLASS_I: [0, 8, 32, 128],  # Die out
        WolframClass.CLASS_II: [4, 13, 108, 232],  # Periodic
        WolframClass.CLASS_III: [30, 45, 90, 150],  # Chaotic
        WolframClass.CLASS_IV: [54, 106, 110, 124, 137, 193]  # Edge of chaos
    }
    
    # Track all trajectories
    class_trajectories = {cls: [] for cls in test_rules.keys()}
    
    for wclass, rules in test_rules.items():
        print(f"\n=== {wclass.name} ===")
        for rule in rules:
            traj = track_pa_trajectory(rule, total_steps=400)
            results['trajectories'].append(traj)
            class_trajectories[wclass].append(traj)
            
            stats = traj['statistics']
            convergence = "→Ξ" if stats['is_converging_to_xi'] else "away"
            print(f"  Rule {rule:3d}: P/A = {stats['mean_pa']:.4f} ± {stats['std_pa']:.4f}, "
                  f"dist={stats['mean_distance_from_xi']:.4f}, {convergence}")
    
    # Summarize by class
    print("\n" + "=" * 70)
    print("CLASS SUMMARIES")
    print("=" * 70)
    print(f"{'Class':<12} {'Mean P/A':>10} {'Std P/A':>10} {'Dist from Ξ':>12} {'Converging?':>12}")
    print("-" * 60)
    
    for wclass in [WolframClass.CLASS_I, WolframClass.CLASS_II, 
                   WolframClass.CLASS_III, WolframClass.CLASS_IV]:
        if wclass not in class_trajectories:
            continue
            
        trajs = class_trajectories[wclass]
        if not trajs:
            continue
            
        all_mean_pa = [t['statistics']['mean_pa'] for t in trajs]
        all_std_pa = [t['statistics']['std_pa'] for t in trajs]
        all_dist = [t['statistics']['mean_distance_from_xi'] for t in trajs]
        converging = sum(1 for t in trajs if t['statistics']['is_converging_to_xi'])
        
        results['class_summaries'][wclass.name] = {
            'mean_pa': float(np.mean(all_mean_pa)),
            'std_pa_avg': float(np.mean(all_std_pa)),
            'mean_distance_from_xi': float(np.mean(all_dist)),
            'fraction_converging': converging / len(trajs)
        }
        
        conv_str = f"{converging}/{len(trajs)}"
        print(f"{wclass.name:<12} {np.mean(all_mean_pa):>10.4f} {np.mean(all_std_pa):>10.4f} "
              f"{np.mean(all_dist):>12.4f} {conv_str:>12}")
    
    # Key analysis: Do Class IV rules stabilize closer to Ξ?
    print("\n" + "=" * 70)
    print("KEY ANALYSIS: Class IV vs Others")
    print("=" * 70)
    
    class_iv_dists = [t['statistics']['mean_distance_from_xi'] 
                     for t in class_trajectories.get(WolframClass.CLASS_IV, [])]
    other_dists = []
    for wc in [WolframClass.CLASS_I, WolframClass.CLASS_II, WolframClass.CLASS_III]:
        other_dists.extend([t['statistics']['mean_distance_from_xi'] 
                           for t in class_trajectories.get(wc, [])])
    
    if class_iv_dists and other_dists:
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_ind(class_iv_dists, other_dists)
        
        print(f"\nClass IV mean distance from Ξ: {np.mean(class_iv_dists):.6f}")
        print(f"Other classes mean distance from Ξ: {np.mean(other_dists):.6f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        results['statistical_test'] = {
            'class_iv_mean_dist': float(np.mean(class_iv_dists)),
            'other_classes_mean_dist': float(np.mean(other_dists)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'class_iv_closer_to_xi': float(np.mean(class_iv_dists)) < float(np.mean(other_dists))
        }
        
        if p_value < 0.05:
            if np.mean(class_iv_dists) < np.mean(other_dists):
                print("\n✓ SIGNIFICANT: Class IV rules are closer to Ξ than other classes")
            else:
                print("\n✗ SIGNIFICANT: Class IV rules are FARTHER from Ξ (unexpected)")
        else:
            print("\n○ NOT SIGNIFICANT: No clear difference between Class IV and others")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"exp_09_convergence_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved: {output_file}")
    
    return results


if __name__ == "__main__":
    run_convergence_experiment()
