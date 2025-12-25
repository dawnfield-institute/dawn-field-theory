"""
Experiment 07: Deep Structure in Möbius Pair Ratios
===================================================

Exp_06 found:
- Mean pair ratio = 1.466 (near φ)
- But global inc/dec = 1.0 (balanced)

This suggests φ is in the LOCAL structure of pairs, not global imbalance.

New tests:
1. Distribution shape of pair ratios (does it peak at φ?)
2. Gap-size dependence (do larger gaps show more φ?)
3. Conditional analysis: given gap g, what's the next-gap distribution?
4. Run-length of gap increases vs decreases
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import save_results
from sec_core import compute_sec, FIRST_50_PRIMES, PHI

XI = 1.0571428571428572


def ratio_distribution_analysis(gaps):
    """
    Analyze the full distribution of count(a,b)/count(b,a) ratios.
    """
    pairs = list(zip(gaps[:-1], gaps[1:]))
    pair_counts = Counter(pairs)
    
    ratios = []
    weights = []  # Weight by total count
    
    processed = set()
    for (a, b), count in pair_counts.items():
        if a == b or (b, a) in processed:
            continue
        
        if (b, a) in pair_counts:
            count_ab = count
            count_ba = pair_counts[(b, a)]
            ratio = max(count_ab, count_ba) / min(count_ab, count_ba)
            ratios.append(ratio)
            weights.append(count_ab + count_ba)
            processed.add((a, b))
            processed.add((b, a))
    
    ratios = np.array(ratios)
    weights = np.array(weights)
    
    # Histogram
    hist, bin_edges = np.histogram(ratios, bins=50, range=(1, 3))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Weighted mean
    weighted_mean = np.average(ratios, weights=weights)
    
    # Find mode
    mode_idx = np.argmax(hist)
    mode = bin_centers[mode_idx]
    
    return {
        "n_pairs": len(ratios),
        "mean": float(np.mean(ratios)),
        "weighted_mean": float(weighted_mean),
        "median": float(np.median(ratios)),
        "mode": float(mode),
        "std": float(np.std(ratios)),
        "phi": float(PHI),
        "histogram": hist.tolist(),
        "bin_centers": bin_centers.tolist()
    }


def gap_size_dependence(gaps):
    """
    Does the asymmetry vary with gap size?
    
    For each gap size g, look at pairs (g, x) and (x, g).
    """
    pairs = list(zip(gaps[:-1], gaps[1:]))
    pair_counts = Counter(pairs)
    
    gap_asymmetries = {}
    
    for g in range(2, 40, 2):  # Even gaps only
        # Count (g, x) pairs
        gx_total = sum(c for (a, b), c in pair_counts.items() if a == g)
        xg_total = sum(c for (a, b), c in pair_counts.items() if b == g)
        
        if gx_total > 0 and xg_total > 0:
            ratio = gx_total / xg_total
            gap_asymmetries[g] = {
                "gap": g,
                "as_first": gx_total,
                "as_second": xg_total,
                "ratio": ratio,
                "diff_from_1": abs(ratio - 1),
                "diff_from_phi": abs(ratio - PHI),
                "diff_from_phi_inv": abs(ratio - 1/PHI)
            }
    
    return gap_asymmetries


def conditional_next_gap(gaps, given_gap=6):
    """
    Given gap = g, what's the distribution of the next gap?
    """
    next_gaps = []
    for i in range(len(gaps) - 1):
        if gaps[i] == given_gap:
            next_gaps.append(gaps[i + 1])
    
    if not next_gaps:
        return {"error": f"no gaps of size {given_gap}"}
    
    counter = Counter(next_gaps)
    total = len(next_gaps)
    
    # Distribution
    dist = {g: c / total for g, c in counter.most_common(10)}
    
    # Mean and variance
    mean_next = np.mean(next_gaps)
    
    # How often is next gap > given_gap vs < given_gap?
    larger = sum(1 for g in next_gaps if g > given_gap)
    smaller = sum(1 for g in next_gaps if g < given_gap)
    equal = sum(1 for g in next_gaps if g == given_gap)
    
    return {
        "given_gap": given_gap,
        "n_occurrences": total,
        "mean_next": float(mean_next),
        "larger_count": larger,
        "smaller_count": smaller,
        "equal_count": equal,
        "larger_fraction": larger / total,
        "ratio_larger_smaller": larger / smaller if smaller > 0 else float('inf'),
        "top_next_gaps": dist
    }


def run_length_analysis(gaps):
    """
    Analyze runs of increasing vs decreasing gaps.
    
    This connects to SEC L+/L- = φ finding.
    """
    diffs = np.diff(gaps.astype(float))
    
    # Find runs
    pos_runs = []
    neg_runs = []
    zero_runs = []
    
    current_run = 1
    current_sign = np.sign(diffs[0]) if len(diffs) > 0 else 0
    
    for d in diffs[1:]:
        sign = np.sign(d)
        if sign == current_sign:
            current_run += 1
        else:
            if current_sign > 0:
                pos_runs.append(current_run)
            elif current_sign < 0:
                neg_runs.append(current_run)
            else:
                zero_runs.append(current_run)
            current_run = 1
            current_sign = sign
    
    # Add final run
    if current_sign > 0:
        pos_runs.append(current_run)
    elif current_sign < 0:
        neg_runs.append(current_run)
    
    pos_runs = np.array(pos_runs)
    neg_runs = np.array(neg_runs)
    
    if len(pos_runs) == 0 or len(neg_runs) == 0:
        return {"error": "insufficient runs"}
    
    mean_pos = np.mean(pos_runs)
    mean_neg = np.mean(neg_runs)
    
    # Run length distributions
    pos_hist, _ = np.histogram(pos_runs, bins=range(1, 15))
    neg_hist, _ = np.histogram(neg_runs, bins=range(1, 15))
    
    return {
        "n_positive_runs": len(pos_runs),
        "n_negative_runs": len(neg_runs),
        "mean_positive_run": float(mean_pos),
        "mean_negative_run": float(mean_neg),
        "L_plus_over_L_minus": float(mean_pos / mean_neg),
        "phi": float(PHI),
        "diff_from_phi": float(abs(mean_pos / mean_neg - PHI)),
        "diff_from_1": float(abs(mean_pos / mean_neg - 1)),
        "pos_run_histogram": pos_hist.tolist(),
        "neg_run_histogram": neg_hist.tolist()
    }


def alternation_pattern(gaps):
    """
    How often do gaps alternate (inc, dec, inc, dec...)?
    """
    diffs = np.diff(gaps.astype(float))
    signs = np.sign(diffs)
    
    # Count alternations vs same-direction
    alternations = np.sum(np.diff(signs) != 0)
    same_direction = np.sum(np.diff(signs) == 0)
    
    total = len(signs) - 1
    alt_fraction = alternations / total if total > 0 else 0
    
    return {
        "alternations": int(alternations),
        "same_direction": int(same_direction),
        "alternation_fraction": float(alt_fraction),
        "expected_random": 0.5,
        "excess_alternation": float(alt_fraction - 0.5)
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 07: Deep Structure in Möbius Pair Ratios")
    print("Looking for φ in local pair structure and run lengths")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_07_deep_mobius_structure",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Get prime gaps
    n_max = 1000000
    sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9], window=13, lam=0.99)
    primes = sec.primes[sec.primes > 10]
    gaps = np.diff(primes)
    
    print(f"\nAnalyzing {len(gaps)} prime gaps (n_max={n_max})")
    
    # Test 1: Ratio distribution
    print("\n[Test 1] Pair Ratio Distribution Shape")
    print("-" * 50)
    
    ratio_dist = ratio_distribution_analysis(gaps)
    print(f"  Mean ratio: {ratio_dist['mean']:.4f}")
    print(f"  Weighted mean: {ratio_dist['weighted_mean']:.4f}")
    print(f"  Median: {ratio_dist['median']:.4f}")
    print(f"  Mode: {ratio_dist['mode']:.4f}")
    print(f"  φ = {ratio_dist['phi']:.4f}")
    
    closest = min([ratio_dist['mean'], ratio_dist['weighted_mean'], ratio_dist['mode']], 
                  key=lambda x: abs(x - PHI))
    print(f"  Closest to φ: {closest:.4f}")
    
    results["tests"].append({"name": "ratio_distribution", "data": ratio_dist})
    
    # Test 2: Gap size dependence
    print("\n[Test 2] Gap Size Dependence of Asymmetry")
    print("-" * 50)
    
    gap_dep = gap_size_dependence(gaps)
    # Convert keys to strings for JSON
    gap_dep_str = {str(k): v for k, v in gap_dep.items()}
    print("  Gap | As First | As Second | Ratio")
    for g in sorted(gap_dep.keys())[:10]:
        d = gap_dep[g]
        marker = "←φ" if d['diff_from_phi'] < 0.2 else ""
        print(f"   {g:2d} | {d['as_first']:6d} | {d['as_second']:6d} | {d['ratio']:.3f} {marker}")
    
    results["tests"].append({"name": "gap_size_dependence", "data": gap_dep_str})
    
    # Test 3: Conditional next gap
    print("\n[Test 3] Conditional Next Gap Distribution")
    print("-" * 50)
    
    for g in [2, 4, 6, 8, 12]:
        cond = conditional_next_gap(gaps, given_gap=g)
        if "error" not in cond:
            print(f"  After gap {g}: mean next = {cond['mean_next']:.2f}, "
                  f"larger/smaller = {cond['ratio_larger_smaller']:.3f}")
    
    cond_results = {str(g): conditional_next_gap(gaps, g) for g in [2, 4, 6, 8, 10, 12]}
    results["tests"].append({"name": "conditional_next_gap", "data": cond_results})
    
    # Test 4: Run length analysis
    print("\n[Test 4] Run Length Analysis (L+/L- vs φ)")
    print("-" * 50)
    
    run_analysis = run_length_analysis(gaps)
    if "error" not in run_analysis:
        print(f"  Mean positive run (increasing): {run_analysis['mean_positive_run']:.4f}")
        print(f"  Mean negative run (decreasing): {run_analysis['mean_negative_run']:.4f}")
        print(f"  L+/L- ratio: {run_analysis['L_plus_over_L_minus']:.4f}")
        print(f"  φ = {run_analysis['phi']:.4f}")
        print(f"  Diff from φ: {run_analysis['diff_from_phi']:.4f}")
        
        if run_analysis['diff_from_phi'] < 0.1:
            print("  ✓ Run ratio matches φ!")
        elif run_analysis['diff_from_1'] < 0.1:
            print("  Run ratio ≈ 1 (symmetric)")
    
    results["tests"].append({"name": "run_length", "data": run_analysis})
    
    # Test 5: Alternation pattern
    print("\n[Test 5] Alternation Pattern")
    print("-" * 50)
    
    alt = alternation_pattern(gaps)
    print(f"  Alternation fraction: {alt['alternation_fraction']:.4f}")
    print(f"  Expected random: {alt['expected_random']:.4f}")
    print(f"  Excess alternation: {alt['excess_alternation']:.4f}")
    
    results["tests"].append({"name": "alternation", "data": alt})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    findings = []
    
    if abs(ratio_dist['weighted_mean'] - PHI) < 0.3:
        findings.append(f"Weighted mean ratio {ratio_dist['weighted_mean']:.3f} approaches φ")
    
    if "error" not in run_analysis and run_analysis['diff_from_phi'] < 0.2:
        findings.append(f"Run length ratio L+/L- = {run_analysis['L_plus_over_L_minus']:.3f} ≈ φ")
    
    if alt['excess_alternation'] > 0.05:
        findings.append(f"Excess alternation: {alt['alternation_fraction']:.3f} vs 0.5 expected")
    
    results["summary"] = {"findings": findings}
    
    for f in findings:
        print(f"✓ {f}")
    
    if not findings:
        print("No strong φ signatures detected in this analysis")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_07_deep_structure", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
