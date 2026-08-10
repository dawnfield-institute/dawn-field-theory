"""
Experiment 05: Fibonacci Folding - Rigorous Controls
=====================================================

The previous experiment found strong Fibonacci/φ signals.
Now test if this is REAL or artifact:

1. Control: Random protein model (no evolution)
2. Control: Shuffled structure assignments
3. Analysis: Are certain Fibonacci numbers driving this?
4. Analysis: Does it scale with protein size?

Key question: Is Fibonacci SELECTED FOR by evolution/physics,
or just an artifact of small-number distributions?
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime

PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# From the real data
REAL_ELEMENT_LENGTHS = [
    # Distribution observed in exp_04
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 20
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 40
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 60
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 80
    3, 3, 3, 3, 3, 3, 3, 3,  # 88 threes total
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 20
    5, 5, 5, 5, 5, 5,  # 26 fives
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  # 20 fours
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,  # 15 sixes
    12, 12, 12, 12, 12, 12, 12, 12,  # 8 twelves
    7, 7, 7, 7, 7,  # some other lengths
    8, 8, 8, 8, 8, 8, 8,  # 8s
    9, 9, 9, 9,
    10, 10, 10,
    11, 11,
    13, 13, 13,
    14, 14,
    15, 15,
    16, 17, 18, 19, 20, 21, 22
]


def nearest_fibonacci(n):
    """Find nearest Fibonacci number and distance."""
    for i, f in enumerate(FIBONACCI):
        if f >= n:
            if i == 0:
                return f, abs(n - f)
            prev = FIBONACCI[i-1]
            if abs(n - prev) < abs(n - f):
                return prev, abs(n - prev)
            return f, abs(n - f)
    return FIBONACCI[-1], abs(n - FIBONACCI[-1])


def compute_fib_score(lengths):
    """Compute mean distance to nearest Fibonacci."""
    if len(lengths) == 0:
        return float('inf')
    distances = [nearest_fibonacci(l)[1] for l in lengths]
    return np.mean(distances)


def compute_fib_enrichment(lengths):
    """Compute Fibonacci exact match enrichment."""
    if len(lengths) == 0:
        return 0
    exact = sum(1 for l in lengths if l in FIBONACCI)
    max_l = max(lengths)
    # Expected under uniform distribution
    fib_in_range = len([f for f in FIBONACCI if f <= max_l])
    expected = len(lengths) * fib_in_range / max_l if max_l > 0 else 0
    return exact / expected if expected > 0 else 0


def run_control_analysis():
    """Run rigorous controls on the Fibonacci folding hypothesis."""
    print("=" * 60)
    print("Experiment 05: Fibonacci Folding - Rigorous Controls")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Real data baseline
    real_lengths = REAL_ELEMENT_LENGTHS
    real_fib_score = compute_fib_score(real_lengths)
    real_enrichment = compute_fib_enrichment(real_lengths)
    
    print(f"\n[1] REAL DATA BASELINE:")
    print(f"  N = {len(real_lengths)}")
    print(f"  Mean length: {np.mean(real_lengths):.2f}")
    print(f"  Fibonacci distance: {real_fib_score:.3f}")
    print(f"  Fibonacci enrichment: {real_enrichment:.2f}x")
    
    # Key insight: small numbers ARE biased toward Fibonacci
    # because Fibonacci is denser at small numbers
    print(f"\n[2] SMALL NUMBER BIAS CHECK:")
    print(f"  Fibonacci density by range:")
    for rng in [(1,5), (5,10), (10,20), (20,50)]:
        fib_in_range = len([f for f in FIBONACCI if rng[0] <= f < rng[1]])
        total_in_range = rng[1] - rng[0]
        density = fib_in_range / total_in_range
        print(f"    {rng[0]}-{rng[1]}: {fib_in_range}/{total_in_range} = {density:.1%}")
    
    # CONTROL 1: What if element lengths were uniformly distributed?
    print(f"\n[3] CONTROL 1: Uniform distribution (1 to max)")
    np.random.seed(42)
    max_len = max(real_lengths)
    uniform_scores = []
    for _ in range(1000):
        uniform_lengths = np.random.randint(1, max_len + 1, size=len(real_lengths))
        uniform_scores.append(compute_fib_score(uniform_lengths))
    
    uniform_mean = np.mean(uniform_scores)
    uniform_std = np.std(uniform_scores)
    z_vs_uniform = (real_fib_score - uniform_mean) / uniform_std
    
    print(f"  Uniform random Fib distance: {uniform_mean:.3f} ± {uniform_std:.3f}")
    print(f"  Real data Fib distance:      {real_fib_score:.3f}")
    print(f"  Z-score vs uniform: {z_vs_uniform:.2f}")
    print(f"  Real is {'CLOSER' if z_vs_uniform < 0 else 'NOT CLOSER'} to Fibonacci")
    
    results['tests']['vs_uniform'] = {
        'real_score': real_fib_score,
        'uniform_mean': uniform_mean,
        'uniform_std': uniform_std,
        'z_score': z_vs_uniform
    }
    
    # CONTROL 2: What if we match the LENGTH DISTRIBUTION but not Fibonacci?
    # This tests: is it just because we have many short elements?
    print(f"\n[4] CONTROL 2: Match length distribution, randomize Fibonacci alignment")
    
    # Shift all lengths by 0.5-1.5 to break Fibonacci alignment
    shifted_scores = []
    for _ in range(1000):
        # Randomly shift each length by +1 or -1 (avoiding breaking the distribution much)
        shift = np.random.choice([-1, 0, 1], size=len(real_lengths), p=[0.25, 0.5, 0.25])
        shifted = np.array(real_lengths) + shift
        shifted = np.maximum(1, shifted)  # Keep positive
        shifted_scores.append(compute_fib_score(shifted))
    
    shifted_mean = np.mean(shifted_scores)
    shifted_std = np.std(shifted_scores)
    z_vs_shifted = (real_fib_score - shifted_mean) / shifted_std
    
    print(f"  Shifted data Fib distance: {shifted_mean:.3f} ± {shifted_std:.3f}")
    print(f"  Real data Fib distance:    {real_fib_score:.3f}")
    print(f"  Z-score vs shifted: {z_vs_shifted:.2f}")
    
    results['tests']['vs_shifted'] = {
        'real_score': real_fib_score,
        'shifted_mean': shifted_mean,
        'shifted_std': shifted_std,
        'z_score': z_vs_shifted
    }
    
    # CONTROL 3: Exponential distribution (what physics might predict without selection)
    print(f"\n[5] CONTROL 3: Exponential distribution (random physical model)")
    exp_scores = []
    for _ in range(1000):
        # Exponential with same mean as real data
        exp_lengths = np.random.exponential(np.mean(real_lengths), size=len(real_lengths))
        exp_lengths = np.maximum(1, np.round(exp_lengths)).astype(int)
        exp_scores.append(compute_fib_score(exp_lengths))
    
    exp_mean = np.mean(exp_scores)
    exp_std = np.std(exp_scores)
    z_vs_exp = (real_fib_score - exp_mean) / exp_std
    
    print(f"  Exponential Fib distance: {exp_mean:.3f} ± {exp_std:.3f}")
    print(f"  Real data Fib distance:   {real_fib_score:.3f}")
    print(f"  Z-score vs exponential: {z_vs_exp:.2f}")
    
    results['tests']['vs_exponential'] = {
        'real_score': real_fib_score,
        'exp_mean': exp_mean,
        'exp_std': exp_std,
        'z_score': z_vs_exp
    }
    
    # ANALYSIS: Which Fibonacci numbers are enriched?
    print(f"\n[6] FIBONACCI BREAKDOWN: Which numbers are enriched?")
    length_counts = defaultdict(int)
    for l in real_lengths:
        length_counts[l] += 1
    
    print(f"  Length  Count  IsFib  LocalDensity")
    for length in sorted(set(real_lengths)):
        count = length_counts[length]
        is_fib = "FIB" if length in FIBONACCI else ""
        local_frac = count / len(real_lengths)
        print(f"    {length:2d}     {count:3d}   {is_fib:3s}    {local_frac:.1%}")
    
    # Core question: Is 3 driving everything?
    print(f"\n[7] SENSITIVITY: Remove length=3 and retest")
    no_threes = [l for l in real_lengths if l != 3]
    no_threes_score = compute_fib_score(no_threes)
    no_threes_enrichment = compute_fib_enrichment(no_threes)
    
    print(f"  Without 3s (N={len(no_threes)}):")
    print(f"    Fib distance: {no_threes_score:.3f} (was {real_fib_score:.3f})")
    print(f"    Enrichment:   {no_threes_enrichment:.2f}x (was {real_enrichment:.2f}x)")
    
    # Compare to uniform without 3
    uniform_no3_scores = []
    for _ in range(1000):
        uniform_no3 = np.random.choice([l for l in range(1, max_len+1) if l != 3], size=len(no_threes))
        uniform_no3_scores.append(compute_fib_score(uniform_no3))
    
    z_no3 = (no_threes_score - np.mean(uniform_no3_scores)) / np.std(uniform_no3_scores)
    print(f"    Z-score vs uniform (no 3s): {z_no3:.2f}")
    
    results['tests']['sensitivity_no_3'] = {
        'n_without_3': len(no_threes),
        'score_without_3': no_threes_score,
        'enrichment_without_3': no_threes_enrichment,
        'z_vs_uniform': z_no3
    }
    
    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT: Is Fibonacci folding REAL?")
    print("=" * 60)
    
    verdicts = []
    
    if z_vs_uniform < -2:
        verdicts.append(f"vs Uniform: z={z_vs_uniform:.1f} → SIGNIFICANT ✅")
    else:
        verdicts.append(f"vs Uniform: z={z_vs_uniform:.1f} → not significant ❌")
    
    if z_vs_shifted < -2:
        verdicts.append(f"vs Shifted: z={z_vs_shifted:.1f} → SIGNIFICANT ✅")
    else:
        verdicts.append(f"vs Shifted: z={z_vs_shifted:.1f} → not significant ❌")
    
    if z_vs_exp < -2:
        verdicts.append(f"vs Exponential: z={z_vs_exp:.1f} → SIGNIFICANT ✅")
    else:
        verdicts.append(f"vs Exponential: z={z_vs_exp:.1f} → not significant ❌")
    
    if z_no3 < -2:
        verdicts.append(f"Without 3s: z={z_no3:.1f} → STILL SIGNIFICANT ✅")
    else:
        verdicts.append(f"Without 3s: z={z_no3:.1f} → driven by 3s ⚠️")
    
    for v in verdicts:
        print(f"  {v}")
    
    significant = sum(1 for v in verdicts if '✅' in v)
    print(f"\n  Significant tests: {significant}/{len(verdicts)}")
    
    if significant >= 3:
        print("\n  CONCLUSION: Fibonacci enrichment appears REAL")
        print("  → Protein folding may follow PAC-like recursion")
    elif significant >= 2:
        print("\n  CONCLUSION: Fibonacci signal present but partially artifactual")
        print("  → Need more data / deeper analysis")
    else:
        print("\n  CONCLUSION: Fibonacci enrichment is likely ARTIFACT")
        print("  → Small number bias explains the pattern")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_05_folding_controls_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_control_analysis()
