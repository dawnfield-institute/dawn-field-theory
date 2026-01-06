#!/usr/bin/env python3
"""
Experiment 13: Alternative Embedding Test
==========================================

CORRECTED FALSIFICATION TEST

Tests whether CA Class IV clustering at ~1.057 is:
- A property of the RULES (genuine)
- Or an artifact of the EMBEDDING METRIC (spurious)

Methodology:
1. Compute P/A ratios using MULTIPLE different embedding approaches
2. If Class IV consistently clusters at ~1.057 regardless of metric → genuine
3. If clustering value changes with metric → artifact of metric choice

This directly addresses whether Ξ is real or constructed.
"""

import sys
import os
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict

# Wolfram classes for elementary CA rules
WOLFRAM_CLASSES = {
    1: [0, 8, 32, 40, 64, 96, 128, 136, 160, 168, 192, 224, 4, 12, 36, 44, 68, 
        76, 100, 108, 132, 140, 164, 172, 196, 204, 228, 236],
    2: [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 19, 23, 24, 25, 26, 27, 28, 29, 
        33, 34, 35, 37, 38, 42, 43, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 77, 
        78, 94, 104, 130, 138, 152, 154, 162, 170, 178, 184, 200, 232],
    3: [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150, 182],
    4: [41, 54, 106, 110, 124]
}

# All rules by class
ALL_RULES = {cls: rules for cls, rules in WOLFRAM_CLASSES.items()}


def evolve_ca(rule, width=100, steps=100, init='center'):
    """Evolve 1D cellular automaton."""
    # Initialize
    if init == 'center':
        state = np.zeros(width, dtype=np.uint8)
        state[width // 2] = 1
    elif init == 'random':
        state = np.random.randint(0, 2, width, dtype=np.uint8)
    else:
        state = np.array(init, dtype=np.uint8)
    
    # Build rule lookup
    rule_bin = format(rule, '08b')
    lookup = {i: int(rule_bin[7-i]) for i in range(8)}
    
    # Evolve
    history = [state.copy()]
    for _ in range(steps):
        new_state = np.zeros_like(state)
        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            new_state[i] = lookup[pattern]
        state = new_state
        history.append(state.copy())
    
    return np.array(history)


def embedding_original(history):
    """
    Original embedding: P = 1-entropy, A = MI + structure + block
    This is what found Ξ ≈ 1.057
    """
    # Entropy: overall disorder
    flat = history.flatten()
    p1 = np.mean(flat)
    if p1 == 0 or p1 == 1:
        entropy = 0
    else:
        entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
    
    # Mutual information (row-to-row)
    MI = 0
    for i in range(len(history) - 1):
        row1, row2 = history[i], history[i+1]
        # Joint distribution
        joint = np.zeros((2, 2))
        for a, b in zip(row1, row2):
            joint[a, b] += 1
        joint /= joint.sum()
        
        # Marginals
        p_a = joint.sum(axis=1)
        p_b = joint.sum(axis=0)
        
        # MI
        for a in range(2):
            for b in range(2):
                if joint[a, b] > 0 and p_a[a] > 0 and p_b[b] > 0:
                    MI += joint[a, b] * np.log2(joint[a, b] / (p_a[a] * p_b[b]))
    MI /= len(history) - 1
    
    # Structure factor (spatial autocorrelation)
    autocorr = []
    for row in history:
        if len(row) > 1:
            c = np.corrcoef(row[:-1], row[1:])[0, 1]
            if not np.isnan(c):
                autocorr.append(abs(c))
    structure = np.mean(autocorr) if autocorr else 0
    
    # Block entropy (2x2 patterns)
    blocks = defaultdict(int)
    for i in range(len(history) - 1):
        for j in range(len(history[0]) - 1):
            pattern = (history[i, j], history[i, j+1], history[i+1, j], history[i+1, j+1])
            blocks[pattern] += 1
    total = sum(blocks.values())
    block_entropy = 0
    for count in blocks.values():
        p = count / total
        if p > 0:
            block_entropy -= p * np.log2(p)
    block_entropy /= 4  # Normalize to [0, 1]
    
    # Original formula
    P = 1 - entropy
    A = 0.5 * MI + 0.3 * structure + 0.2 * block_entropy
    
    # Avoid division by zero
    if A < 0.001:
        A = 0.001
    
    return P / A


def embedding_pure_entropy(history):
    """
    Alternative 1: Pure entropy ratio
    P = 1 - entropy, A = block_entropy
    """
    flat = history.flatten()
    p1 = np.mean(flat)
    if p1 == 0 or p1 == 1:
        entropy = 0
    else:
        entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
    
    # Block entropy
    blocks = defaultdict(int)
    for i in range(len(history) - 1):
        for j in range(len(history[0]) - 1):
            pattern = (history[i, j], history[i, j+1], history[i+1, j], history[i+1, j+1])
            blocks[pattern] += 1
    total = sum(blocks.values())
    block_entropy = 0
    for count in blocks.values():
        p = count / total
        if p > 0:
            block_entropy -= p * np.log2(p)
    block_entropy /= 4
    
    P = 1 - entropy
    A = max(block_entropy, 0.001)
    
    return P / A


def embedding_compression(history):
    """
    Alternative 2: Compression ratio
    P = row uniqueness, A = pattern complexity
    """
    # Row uniqueness (how many unique rows)
    unique_rows = len(set(tuple(row) for row in history))
    row_ratio = unique_rows / len(history)
    
    # Column pattern complexity
    col_patterns = []
    for j in range(history.shape[1]):
        col = tuple(history[:, j])
        col_patterns.append(col)
    unique_cols = len(set(col_patterns))
    col_ratio = unique_cols / history.shape[1]
    
    P = row_ratio
    A = max(col_ratio, 0.001)
    
    return P / A


def embedding_temporal(history):
    """
    Alternative 3: Temporal dynamics
    P = stability, A = change rate
    """
    # Stability: how much stays the same
    same = 0
    for i in range(len(history) - 1):
        same += np.mean(history[i] == history[i+1])
    stability = same / (len(history) - 1)
    
    # Change rate: Hamming distance between successive rows
    changes = []
    for i in range(len(history) - 1):
        hamming = np.mean(history[i] != history[i+1])
        changes.append(hamming)
    change_rate = np.mean(changes)
    
    P = stability
    A = max(change_rate, 0.001)
    
    return P / A


def embedding_frequency(history):
    """
    Alternative 4: Frequency domain
    P = low frequency power, A = high frequency power
    """
    # 2D FFT of the history
    fft = np.fft.fft2(history.astype(float))
    power = np.abs(fft) ** 2
    
    # Low frequency: center quarter
    h, w = power.shape
    low = power[:h//4, :w//4].sum() + power[:h//4, -w//4:].sum()
    low += power[-h//4:, :w//4].sum() + power[-h//4:, -w//4:].sum()
    
    # High frequency: the rest
    high = power.sum() - low
    
    P = low / (power.sum() + 1e-10)
    A = max(high / (power.sum() + 1e-10), 0.001)
    
    return P / A


EMBEDDING_METHODS = {
    'original': embedding_original,
    'pure_entropy': embedding_pure_entropy,
    'compression': embedding_compression,
    'temporal': embedding_temporal,
    'frequency': embedding_frequency
}


def run_alternative_embedding_test():
    """Test if Class IV clustering persists across different embeddings."""
    
    print("=" * 70)
    print("EXPERIMENT 13: Alternative Embedding Test")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nFalsification Question: Does Class IV cluster at ~1.057 regardless of metric?")
    print("                        Or is it an artifact of the embedding choice?\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'falsification_target': 'Is Ξ ≈ 1.057 robust across embedding methods?'
    }
    
    # Compute P/A for all rules with all embeddings
    embedding_results = {name: {} for name in EMBEDDING_METHODS}
    
    print("Computing embeddings for all 256 rules...")
    
    for rule in range(256):
        history = evolve_ca(rule, width=100, steps=100, init='center')
        
        for name, embed_func in EMBEDDING_METHODS.items():
            try:
                pa_ratio = embed_func(history)
                embedding_results[name][rule] = pa_ratio
            except Exception as e:
                embedding_results[name][rule] = None
    
    print("Done.\n")
    
    # Analyze Class IV position in each embedding
    print("=" * 70)
    print("CLASS IV POSITION BY EMBEDDING METHOD")
    print("=" * 70)
    
    XI = 1.0571
    PHI = 1.618
    
    class4_rules = WOLFRAM_CLASSES[4]
    analysis = {}
    
    print(f"\n{'Method':<15} {'Class IV Mean':>14} {'Class IV Std':>12} {'Dist to Ξ':>12} {'Dist to φ':>12}")
    print("-" * 70)
    
    for name in EMBEDDING_METHODS:
        class4_values = [embedding_results[name][r] for r in class4_rules if embedding_results[name].get(r) is not None]
        
        if class4_values:
            mean_val = np.mean(class4_values)
            std_val = np.std(class4_values)
            dist_xi = abs(mean_val - XI)
            dist_phi = abs(mean_val - PHI)
            
            analysis[name] = {
                'class4_mean': float(mean_val),
                'class4_std': float(std_val),
                'class4_values': [float(v) for v in class4_values],
                'dist_to_xi': float(dist_xi),
                'dist_to_phi': float(dist_phi)
            }
            
            print(f"{name:<15} {mean_val:>14.4f} {std_val:>12.4f} {dist_xi:>12.4f} {dist_phi:>12.4f}")
    
    # Compare Class IV separation from other classes
    print("\n" + "=" * 70)
    print("CLASS IV SEPARATION FROM OTHER CLASSES")
    print("=" * 70)
    
    other_rules = [r for cls, rules in WOLFRAM_CLASSES.items() if cls != 4 for r in rules]
    
    print(f"\n{'Method':<15} {'C4 Mean':>10} {'Other Mean':>12} {'Separation':>12} {'T-stat':>10} {'p-value':>12}")
    print("-" * 70)
    
    for name in EMBEDDING_METHODS:
        class4_values = [embedding_results[name][r] for r in class4_rules if embedding_results[name].get(r) is not None]
        other_values = [embedding_results[name][r] for r in other_rules if embedding_results[name].get(r) is not None]
        
        if class4_values and other_values:
            c4_mean = np.mean(class4_values)
            other_mean = np.mean(other_values)
            separation = c4_mean - other_mean
            
            # t-test
            t_stat, p_val = stats.ttest_ind(class4_values, other_values)
            
            analysis[name]['other_mean'] = float(other_mean)
            analysis[name]['separation'] = float(separation)
            analysis[name]['t_stat'] = float(t_stat)
            analysis[name]['p_value'] = float(p_val)
            
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            print(f"{name:<15} {c4_mean:>10.4f} {other_mean:>12.4f} {separation:>12.4f} {t_stat:>10.2f} {p_str:>12}")
    
    results['analysis'] = analysis
    
    # Summary verdict
    print("\n" + "=" * 70)
    print("FALSIFICATION VERDICT")
    print("=" * 70)
    
    # Check if all methods show Class IV separation
    all_separate = all(
        analysis[name].get('p_value', 1) < 0.05 
        for name in EMBEDDING_METHODS 
        if name in analysis
    )
    
    # Check if all methods cluster near same value
    means = [analysis[name]['class4_mean'] for name in EMBEDDING_METHODS if name in analysis]
    mean_variance = np.var(means)
    
    xi_consistency = all(
        abs(analysis[name]['class4_mean'] - XI) < 0.3
        for name in EMBEDDING_METHODS
        if name in analysis
    )
    
    print(f"\nClass IV separates from other classes in all methods: {all_separate}")
    print(f"Class IV mean variance across methods: {mean_variance:.4f}")
    print(f"All methods cluster near Ξ (~1.057): {xi_consistency}")
    
    if all_separate and xi_consistency:
        verdict = "GENUINE"
        explanation = "Class IV consistently separates and clusters near ~1.057 regardless of embedding"
    elif all_separate and not xi_consistency:
        verdict = "PARTIAL - SEPARATION GENUINE, VALUE VARIES"
        explanation = "Class IV separates but the specific value (~1.057) depends on embedding choice"
    else:
        verdict = "LIKELY ARTIFACT"
        explanation = "Class IV separation is not robust across embedding methods"
    
    print(f"\nVERDICT: {verdict}")
    print(f"Explanation: {explanation}")
    
    results['verdict'] = verdict
    results['explanation'] = explanation
    results['all_separate'] = all_separate
    results['xi_consistency'] = xi_consistency
    results['mean_variance'] = float(mean_variance)
    
    # Save results
    output_path = Path(__file__).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"exp_13_alternative_embeddings_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved: {output_file}")
    
    return results


if __name__ == "__main__":
    run_alternative_embedding_test()
