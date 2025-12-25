"""
Experiment 05: Möbius Structure Across Consecutive Gaps
=======================================================

Exp_04 showed Möbius isn't within single gaps. 
New hypothesis: The Möbius half-twist is between CONSECUTIVE gaps.

Gap_n → Gap_{n+1} where one is the "flipped reflection" of the other

This would mean:
- Prime gaps come in Möbius-twisted pairs
- The pattern (g1, g2, g3, ...) has antiperiodic structure
- Ξ emerges from the gap sequence spectral properties

Also testing: Does the prime gap SEQUENCE itself show antiperiodic FFT modes?
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import save_results
from sec_core import compute_sec, FIRST_50_PRIMES, PHI

XI = 1.0571428571428572


def gap_sequence_antiperiodicity(gaps):
    """
    Test if the gap sequence shows antiperiodic structure.
    
    f(n + N/2) = -f(n) for antiperiodic
    
    For gaps, test: gap[n] + gap[n + N/2] = constant?
    """
    n = len(gaps)
    half = n // 2
    
    # Compare first half to second half
    first_half = gaps[:half]
    second_half = gaps[half:2*half]
    
    # Antiperiodic: g1 + g2 ≈ constant for all pairs
    sums = first_half + second_half
    mean_sum = np.mean(sums)
    std_sum = np.std(sums)
    cv = std_sum / mean_sum if mean_sum > 0 else float('inf')
    
    # Alternative: correlation between first_half and -second_half
    corr, pval = pearsonr(first_half, -second_half)
    
    return {
        "n_gaps": n,
        "mean_pair_sum": float(mean_sum),
        "std_pair_sum": float(std_sum),
        "cv": float(cv),
        "antiperiodic_correlation": float(corr),
        "p_value": float(pval)
    }


def consecutive_gap_symmetry(gaps):
    """
    Test for Möbius relationship between consecutive gaps.
    
    If gap_n and gap_{n+1} are Möbius-twisted:
    There should be a consistent relationship g_{n+1} = f(g_n)
    """
    g1 = gaps[:-1]
    g2 = gaps[1:]
    
    # Basic correlation
    corr, pval = pearsonr(g1, g2)
    
    # Test various relationships
    # 1. Sum constant? g1 + g2 = C
    sums = g1 + g2
    sum_cv = np.std(sums) / np.mean(sums)
    
    # 2. Ratio constant? g2/g1 = C
    ratios = g2 / g1
    ratio_mean = np.mean(ratios)
    ratio_cv = np.std(ratios) / np.mean(ratios)
    
    # 3. Product constant? g1 * g2 = C
    products = g1 * g2
    prod_cv = np.std(products) / np.mean(products)
    
    return {
        "correlation": float(corr),
        "p_value": float(pval),
        "sum_mean": float(np.mean(sums)),
        "sum_cv": float(sum_cv),
        "ratio_mean": float(ratio_mean),
        "ratio_cv": float(ratio_cv),
        "product_mean": float(np.mean(products)),
        "product_cv": float(prod_cv)
    }


def gap_sequence_spectral(gaps, max_modes=100):
    """
    FFT of the gap sequence - look for Möbius (half-integer) modes.
    """
    # Center the gap sequence
    gaps_centered = gaps - np.mean(gaps)
    
    # FFT
    spectrum = np.abs(fft(gaps_centered))
    n = len(gaps)
    freqs = fftfreq(n)
    
    # Only positive frequencies
    positive = freqs > 0
    spectrum_pos = spectrum[positive][:max_modes]
    freqs_pos = freqs[positive][:max_modes]
    
    # Find dominant frequencies
    sorted_indices = np.argsort(spectrum_pos)[::-1]
    top_modes = []
    
    for idx in sorted_indices[:10]:
        freq = freqs_pos[idx]
        power = spectrum_pos[idx]
        # Classify as integer-like or half-integer-like
        mode_type = "integer" if abs(freq * n - round(freq * n)) < 0.1 else "half-integer"
        top_modes.append({
            "frequency": float(freq),
            "power": float(power),
            "mode_index": float(freq * n),
            "type": mode_type
        })
    
    # Count power in integer vs half-integer modes
    int_power = 0
    half_power = 0
    for i, (f, p) in enumerate(zip(freqs_pos, spectrum_pos)):
        mode_idx = f * n
        if abs(mode_idx - round(mode_idx)) < 0.25:
            int_power += p**2
        else:
            half_power += p**2
    
    ratio = np.sqrt(half_power / int_power) if int_power > 0 else 0
    
    return {
        "n_gaps": n,
        "top_modes": top_modes,
        "integer_power": float(int_power),
        "half_integer_power": float(half_power),
        "power_ratio": float(ratio),
        "xi_comparison": float(XI),
        "difference_from_xi": float(abs(ratio - XI))
    }


def gap_pair_classification(gaps):
    """
    Classify consecutive gap pairs and look for patterns.
    
    Möbius prediction: Pairs should fall into symmetric classes
    """
    pairs = list(zip(gaps[:-1], gaps[1:]))
    
    # Count pair frequencies
    from collections import Counter
    pair_counts = Counter(pairs)
    
    # Most common pairs
    common = pair_counts.most_common(20)
    
    # Check for (a,b) / (b,a) symmetry
    symmetric_count = 0
    total_pairs = len(pairs)
    
    for (a, b), count in pair_counts.items():
        if (b, a) in pair_counts:
            symmetric_count += min(count, pair_counts[(b, a)])
    
    symmetry_fraction = symmetric_count / (2 * total_pairs)
    
    return {
        "total_pairs": total_pairs,
        "unique_pairs": len(pair_counts),
        "most_common": [(list(p), c) for p, c in common],
        "symmetric_fraction": float(symmetry_fraction)
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 05: Möbius Structure Across Consecutive Gaps")
    print("Hypothesis: Half-twist between consecutive prime gaps")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_05_consecutive_gaps_mobius",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Get prime gaps
    n_max = 200000
    sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9], window=13, lam=0.99)
    primes = sec.primes[sec.primes > 10]
    gaps = np.diff(primes)
    
    print(f"\nAnalyzing {len(gaps)} prime gaps")
    print(f"Gap range: {gaps.min()} to {gaps.max()}")
    print(f"Mean gap: {np.mean(gaps):.2f}")
    
    # Test 1: Antiperiodicity in gap sequence
    print("\n[Test 1] Gap Sequence Antiperiodicity")
    print("-" * 50)
    
    anti = gap_sequence_antiperiodicity(gaps)
    print(f"  Mean pair sum (g_i + g_{{i+N/2}}): {anti['mean_pair_sum']:.2f}")
    print(f"  CV of pair sums: {anti['cv']:.4f}")
    print(f"  Antiperiodic correlation: {anti['antiperiodic_correlation']:.4f}")
    print(f"  P-value: {anti['p_value']:.2e}")
    
    if anti['antiperiodic_correlation'] > 0.3:
        print("  ✓ Gap sequence shows antiperiodic structure!")
    
    results["tests"].append({"name": "antiperiodicity", "data": anti})
    
    # Test 2: Consecutive gap symmetry
    print("\n[Test 2] Consecutive Gap Relationship")
    print("-" * 50)
    
    consec = consecutive_gap_symmetry(gaps)
    print(f"  g_i ↔ g_{{i+1}} correlation: {consec['correlation']:.4f}")
    print(f"  Mean ratio g_{{i+1}}/g_i: {consec['ratio_mean']:.4f}")
    print(f"  Ratio CV: {consec['ratio_cv']:.4f}")
    
    # Check if ratio relates to φ or ξ
    ratio = consec['ratio_mean']
    print(f"  Diff from φ: {abs(ratio - PHI):.4f}")
    print(f"  Diff from 1/φ: {abs(ratio - 1/PHI):.4f}")
    print(f"  Diff from ξ: {abs(ratio - XI):.4f}")
    
    results["tests"].append({"name": "consecutive_symmetry", "data": consec})
    
    # Test 3: Spectral analysis of gap sequence
    print("\n[Test 3] Gap Sequence Spectral Analysis")
    print("-" * 50)
    
    spectral = gap_sequence_spectral(gaps)
    print(f"  Integer mode power: {spectral['integer_power']:.2e}")
    print(f"  Half-integer mode power: {spectral['half_integer_power']:.2e}")
    print(f"  Power ratio: {spectral['power_ratio']:.4f}")
    print(f"  Expected Ξ: {spectral['xi_comparison']:.4f}")
    print(f"  Difference: {spectral['difference_from_xi']:.4f}")
    
    print("\n  Top frequency modes:")
    for mode in spectral['top_modes'][:5]:
        print(f"    f={mode['frequency']:.4f}, power={mode['power']:.1f}, type={mode['type']}")
    
    results["tests"].append({"name": "spectral", "data": spectral})
    
    # Test 4: Gap pair classification
    print("\n[Test 4] Gap Pair Classification")
    print("-" * 50)
    
    pairs = gap_pair_classification(gaps)
    print(f"  Total pairs: {pairs['total_pairs']}")
    print(f"  Unique pairs: {pairs['unique_pairs']}")
    print(f"  Symmetric fraction (a,b)↔(b,a): {pairs['symmetric_fraction']:.4f}")
    
    print("\n  Most common gap pairs:")
    for p, c in pairs['most_common'][:5]:
        print(f"    {p}: {c} times")
    
    results["tests"].append({"name": "pair_classification", "data": pairs})
    
    # Test 5: Run-length comparison (connects to SEC φ finding)
    print("\n[Test 5] Gap Run-Length Analysis")
    print("-" * 50)
    
    # Runs of increasing vs decreasing gaps
    diffs = np.diff(gaps)  # Change in gap size
    
    # Find runs of positive and negative changes
    pos_runs = []
    neg_runs = []
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
            current_run = 1
            current_sign = sign
    
    if pos_runs and neg_runs:
        mean_pos = np.mean(pos_runs)
        mean_neg = np.mean(neg_runs)
        run_ratio = mean_pos / mean_neg
        
        print(f"  Mean increasing run: {mean_pos:.3f}")
        print(f"  Mean decreasing run: {mean_neg:.3f}")
        print(f"  Ratio L+/L-: {run_ratio:.4f}")
        print(f"  φ = {PHI:.4f}")
        print(f"  Diff from φ: {abs(run_ratio - PHI):.4f}")
        
        if abs(run_ratio - PHI) < 0.1 or abs(run_ratio - 1/PHI) < 0.1:
            print("  ✓ Run ratio matches φ!")
        
        results["tests"].append({
            "name": "run_length",
            "data": {
                "mean_pos_run": float(mean_pos),
                "mean_neg_run": float(mean_neg),
                "run_ratio": float(run_ratio),
                "phi": float(PHI),
                "diff_from_phi": float(abs(run_ratio - PHI))
            }
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    findings = []
    
    if anti['antiperiodic_correlation'] > 0.3:
        findings.append("Gap sequence shows antiperiodic structure")
    
    if spectral['difference_from_xi'] < 0.3:
        findings.append(f"Spectral ratio {spectral['power_ratio']:.3f} approaches Ξ")
    
    if pairs['symmetric_fraction'] > 0.3:
        findings.append(f"High (a,b)↔(b,a) symmetry: {pairs['symmetric_fraction']:.2f}")
    
    results["summary"] = {"findings": findings}
    
    for f in findings:
        print(f"✓ {f}")
    
    if not findings:
        print("No strong Möbius signatures in gap sequence")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_05_consecutive_gaps", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
