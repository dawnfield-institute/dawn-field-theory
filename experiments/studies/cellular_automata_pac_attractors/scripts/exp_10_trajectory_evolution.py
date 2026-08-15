#!/usr/bin/env python3
"""
Experiment 10: P/A Trajectory Evolution
========================================

Tracks P/A ratio evolution using SAME methodology as PACEmbedder
but computed at different evolution depths.

The question: Does P/A ratio converge to Ξ over time for Class IV rules?

Method:
- Run CA for increasing number of steps: 50, 100, 150, 200, 250, 300
- At each checkpoint, compute P/A using full history up to that point
- Track trajectory toward or away from Ξ = 1.0571
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass, CAState

# Constants
XI = 1.0571428


def compute_metrics_from_history(history: np.ndarray) -> Dict[str, float]:
    """Compute the same metrics as PACEmbedder from a history array."""
    steps, width = history.shape
    
    # 1. Entropy (spatial density distribution)
    densities = history.mean(axis=1)
    hist, _ = np.histogram(densities, bins=20, range=(0, 1), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        entropy = 0.0
    else:
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # 2. Block entropy (diversity of local patterns)
    block_size = 3
    pattern_counts = {}
    for t in range(steps):
        for i in range(width - block_size + 1):
            pattern = tuple(history[t, i:i+block_size])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    total = sum(pattern_counts.values())
    if total == 0:
        block_entropy = 0.0
    else:
        probs = np.array(list(pattern_counts.values())) / total
        block_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # 3. Mutual information (temporal predictability)
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
    
    # 4. Structure factor (spatial periodicity)
    power_spectra = []
    for row in history[steps // 2:]:  # Second half to skip transients
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
    
    return {
        'entropy': entropy,
        'block_entropy': block_entropy,
        'mutual_info': mutual_info,
        'structure': structure
    }


def compute_pa_ratio(history: np.ndarray) -> float:
    """Compute P/A ratio using PACEmbedder methodology."""
    metrics = compute_metrics_from_history(history)
    
    # Normalize entropy to [0, 1]
    max_entropy = np.log2(20)
    norm_entropy = min(metrics['entropy'] / max_entropy, 1.0)
    
    # P: Low entropy = high potential
    potential = 1.0 - norm_entropy
    
    # A: Combination of MI, structure, block entropy
    actualization = (0.5 * metrics['mutual_info'] + 
                    0.3 * min(metrics['structure'] / 10, 1.0) + 
                    0.2 * (metrics['block_entropy'] / 8))
    actualization = min(actualization, 1.0)
    
    # Normalize to sum = 1
    total = potential + actualization
    if total > 0:
        potential /= total
        actualization /= total
    
    # P/A ratio
    if actualization > 0.001:
        return potential / actualization
    else:
        return 100.0


def track_trajectory(rule: int, checkpoints: List[int], width: int = 101) -> Dict:
    """Track P/A ratio at multiple evolution depths."""
    max_steps = max(checkpoints)
    
    ca = ElementaryCA(rule, width)
    state = ca.evolve_fast(max_steps, init_type='single')
    
    trajectory = []
    for steps in checkpoints:
        history_slice = state.history[:steps]
        pa_ratio = compute_pa_ratio(history_slice)
        dist_from_xi = abs(pa_ratio - XI)
        trajectory.append({
            'steps': steps,
            'pa_ratio': pa_ratio,
            'distance_from_xi': dist_from_xi
        })
    
    return {
        'rule': rule,
        'class': RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN).name,
        'trajectory': trajectory
    }


def run_trajectory_experiment():
    """Run trajectory evolution experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 10: P/A Trajectory Evolution")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nQuestion: Does P/A ratio converge toward Ξ = {XI}?")
    print()
    
    checkpoints = [50, 100, 150, 200, 250, 300]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoints': checkpoints,
        'xi': XI,
        'trajectories': []
    }
    
    # Test representative rules from each class
    test_rules = {
        'CLASS_I': [0, 8],
        'CLASS_II': [4, 13, 108],
        'CLASS_III': [30, 45, 90, 150],
        'CLASS_IV': [54, 106, 110, 124, 137, 193]
    }
    
    for class_name, rules in test_rules.items():
        print(f"\n=== {class_name} ===")
        print(f"{'Rule':>6} |", end="")
        for cp in checkpoints:
            print(f" {cp:>7}", end="")
        print(" | Direction")
        print("-" * 70)
        
        for rule in rules:
            traj = track_trajectory(rule, checkpoints)
            results['trajectories'].append(traj)
            
            # Print trajectory
            print(f"{rule:>6} |", end="")
            for point in traj['trajectory']:
                print(f" {point['pa_ratio']:>7.3f}", end="")
            
            # Direction: compare first vs last
            first = traj['trajectory'][0]['distance_from_xi']
            last = traj['trajectory'][-1]['distance_from_xi']
            direction = "→ Ξ" if last < first else "← Ξ"
            print(f" | {direction}")
    
    # Summary: Final distances by class
    print("\n" + "=" * 70)
    print("FINAL P/A RATIOS AND DISTANCES FROM Ξ")
    print("=" * 70)
    
    class_finals = {}
    for traj in results['trajectories']:
        cls = traj['class']
        final = traj['trajectory'][-1]
        if cls not in class_finals:
            class_finals[cls] = []
        class_finals[cls].append({
            'rule': traj['rule'],
            'final_pa': final['pa_ratio'],
            'final_dist': final['distance_from_xi']
        })
    
    print(f"\n{'Class':<12} {'Rules':<20} {'Mean Final P/A':>15} {'Mean Dist from Ξ':>18}")
    print("-" * 70)
    
    for cls in ['CLASS_I', 'CLASS_II', 'CLASS_III', 'CLASS_IV']:
        if cls not in class_finals:
            continue
        finals = class_finals[cls]
        rules_str = ",".join(str(f['rule']) for f in finals)
        mean_pa = np.mean([f['final_pa'] for f in finals])
        mean_dist = np.mean([f['final_dist'] for f in finals])
        print(f"{cls:<12} {rules_str:<20} {mean_pa:>15.4f} {mean_dist:>18.4f}")
        
        results[f'{cls.lower()}_summary'] = {
            'mean_final_pa': float(mean_pa),
            'mean_final_distance': float(mean_dist)
        }
    
    # Key finding: Class IV at Ξ?
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    
    class_iv_finals = class_finals.get('CLASS_IV', [])
    if class_iv_finals:
        for f in sorted(class_iv_finals, key=lambda x: x['final_dist']):
            verdict = "≈ Ξ ✓" if f['final_dist'] < 0.01 else "≠ Ξ"
            print(f"  Rule {f['rule']:3d}: P/A = {f['final_pa']:.6f}, dist = {f['final_dist']:.6f} {verdict}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"exp_10_trajectory_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved: {output_file}")
    
    return results


if __name__ == "__main__":
    run_trajectory_experiment()
