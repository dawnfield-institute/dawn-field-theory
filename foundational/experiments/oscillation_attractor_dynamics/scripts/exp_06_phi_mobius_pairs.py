"""
Experiment 06: φ in Möbius Gap Pair Frequencies
================================================

Exp_05 found 47.5% (a,b)↔(b,a) symmetry in gap pairs.
Now test: Do the frequencies of these pairs follow φ ratios?

Predictions:
1. Ratio of symmetric to asymmetric pairs ≈ φ or 1/φ
2. Within Möbius pairs, count(a,b)/count(b,a) clusters around φ
3. Gap sizes that appear in Möbius pairs follow Fibonacci pattern
4. The "imbalance" within pairs relates to SEC run-length L+/L- = φ
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from collections import Counter, defaultdict
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import save_results
from sec_core import compute_sec, FIRST_50_PRIMES, PHI, FIBONACCI

XI = 1.0571428571428572


def analyze_mobius_pairs(gaps):
    """
    Deep analysis of (a,b)↔(b,a) Möbius pair structure.
    """
    pairs = list(zip(gaps[:-1], gaps[1:]))
    pair_counts = Counter(pairs)
    
    # Separate into symmetric (a,a) and asymmetric (a,b) where a≠b
    symmetric_pairs = {}  # (a,a) type
    mobius_pairs = {}     # (a,b) with a≠b that has a (b,a) partner
    orphan_pairs = {}     # (a,b) with no (b,a) partner
    
    processed = set()
    
    for (a, b), count in pair_counts.items():
        if (a, b) in processed:
            continue
            
        if a == b:
            symmetric_pairs[(a, b)] = count
            processed.add((a, b))
        else:
            reverse = (b, a)
            if reverse in pair_counts:
                # Möbius pair found
                mobius_pairs[(a, b)] = count
                mobius_pairs[(b, a)] = pair_counts[reverse]
                processed.add((a, b))
                processed.add((b, a))
            else:
                orphan_pairs[(a, b)] = count
                processed.add((a, b))
    
    return symmetric_pairs, mobius_pairs, orphan_pairs


def phi_ratio_analysis(mobius_pairs):
    """
    Test if count(a,b)/count(b,a) ratios cluster around φ.
    """
    ratios = []
    pair_data = []
    
    # Group by unordered pair
    processed = set()
    for (a, b), count in mobius_pairs.items():
        key = tuple(sorted([a, b]))
        if key in processed:
            continue
        processed.add(key)
        
        count_ab = mobius_pairs.get((a, b), 0)
        count_ba = mobius_pairs.get((b, a), 0)
        
        if count_ab > 0 and count_ba > 0:
            ratio = max(count_ab, count_ba) / min(count_ab, count_ba)
            ratios.append(ratio)
            pair_data.append({
                "pair": [a, b],
                "count_ab": count_ab,
                "count_ba": count_ba,
                "ratio": ratio,
                "diff_from_phi": abs(ratio - PHI),
                "diff_from_1": abs(ratio - 1.0)
            })
    
    ratios = np.array(ratios)
    
    # How many cluster near φ vs near 1?
    near_phi = np.sum(np.abs(ratios - PHI) < 0.3)
    near_one = np.sum(np.abs(ratios - 1.0) < 0.3)
    
    return {
        "n_mobius_pairs": len(ratios),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "std_ratio": float(np.std(ratios)),
        "near_phi_count": int(near_phi),
        "near_one_count": int(near_one),
        "phi": float(PHI),
        "diff_from_phi": float(abs(np.mean(ratios) - PHI)),
        "top_pairs": sorted(pair_data, key=lambda x: x["count_ab"] + x["count_ba"], reverse=True)[:10]
    }


def symmetric_asymmetric_ratio(symmetric, mobius, orphans):
    """
    Test: ratio of symmetric to asymmetric total counts.
    """
    sym_total = sum(symmetric.values())
    mobius_total = sum(mobius.values())
    orphan_total = sum(orphans.values())
    
    total = sym_total + mobius_total + orphan_total
    
    # Various ratio tests
    results = {
        "symmetric_count": sym_total,
        "mobius_count": mobius_total,
        "orphan_count": orphan_total,
        "total": total,
        "symmetric_fraction": sym_total / total,
        "mobius_fraction": mobius_total / total,
        "orphan_fraction": orphan_total / total
    }
    
    # Test ratios against φ
    if mobius_total > 0:
        ratio_sym_mob = sym_total / mobius_total
        results["sym_to_mobius_ratio"] = ratio_sym_mob
        results["sym_mob_diff_from_phi"] = abs(ratio_sym_mob - PHI)
        results["sym_mob_diff_from_phi_inv"] = abs(ratio_sym_mob - 1/PHI)
    
    if orphan_total > 0:
        ratio_mob_orph = mobius_total / orphan_total
        results["mobius_to_orphan_ratio"] = ratio_mob_orph
        results["mob_orph_diff_from_phi"] = abs(ratio_mob_orph - PHI)
    
    return results


def fibonacci_gap_enrichment(mobius_pairs, all_gaps):
    """
    Test if Fibonacci gap sizes are enriched in Möbius pairs.
    """
    fib_set = {2, 3, 5, 8, 13, 21, 34}
    
    # Get all gap sizes that appear in Möbius pairs
    mobius_gaps = set()
    for (a, b) in mobius_pairs.keys():
        mobius_gaps.add(a)
        mobius_gaps.add(b)
    
    # Count Fibonacci vs non-Fibonacci in Möbius
    fib_in_mobius = len(mobius_gaps & fib_set)
    nonfib_in_mobius = len(mobius_gaps - fib_set)
    
    # Expected by chance
    unique_gaps = set(all_gaps)
    fib_in_all = len(unique_gaps & fib_set)
    nonfib_in_all = len(unique_gaps - fib_set)
    
    if nonfib_in_all > 0 and nonfib_in_mobius > 0:
        enrichment = (fib_in_mobius / nonfib_in_mobius) / (fib_in_all / nonfib_in_all)
    else:
        enrichment = 0
    
    return {
        "fib_gaps_in_mobius": list(mobius_gaps & fib_set),
        "nonfib_gaps_in_mobius": len(mobius_gaps - fib_set),
        "fib_in_all_gaps": list(unique_gaps & fib_set),
        "enrichment": enrichment,
        "all_mobius_gaps": sorted(list(mobius_gaps))
    }


def pair_imbalance_analysis(mobius_pairs):
    """
    For each Möbius pair (a,b)/(b,a), measure the imbalance.
    
    Connect to SEC run-length: the "longer" direction should dominate by φ.
    """
    imbalances = []
    
    processed = set()
    for (a, b) in mobius_pairs.keys():
        key = tuple(sorted([a, b]))
        if key in processed or a == b:
            continue
        processed.add(key)
        
        count_ab = mobius_pairs.get((a, b), 0)
        count_ba = mobius_pairs.get((b, a), 0)
        
        # Imbalance: which direction is more common?
        # Positive if (small, large) more common, negative if (large, small)
        if a < b:
            imbalance = (count_ab - count_ba) / (count_ab + count_ba)
        else:
            imbalance = (count_ba - count_ab) / (count_ab + count_ba)
        
        imbalances.append({
            "pair": sorted([a, b]),
            "increasing_count": count_ab if a < b else count_ba,
            "decreasing_count": count_ba if a < b else count_ab,
            "imbalance": imbalance
        })
    
    imb_values = [d["imbalance"] for d in imbalances]
    
    # Aggregate: are increasing or decreasing gaps more common?
    inc_total = sum(d["increasing_count"] for d in imbalances)
    dec_total = sum(d["decreasing_count"] for d in imbalances)
    
    global_ratio = inc_total / dec_total if dec_total > 0 else 0
    
    return {
        "mean_imbalance": float(np.mean(imb_values)),
        "std_imbalance": float(np.std(imb_values)),
        "increasing_total": inc_total,
        "decreasing_total": dec_total,
        "inc_dec_ratio": global_ratio,
        "diff_from_phi": abs(global_ratio - PHI),
        "diff_from_phi_inv": abs(global_ratio - 1/PHI),
        "top_imbalanced": sorted(imbalances, key=lambda x: abs(x["imbalance"]), reverse=True)[:5]
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 06: φ in Möbius Gap Pair Frequencies")
    print("Testing if (a,b)↔(b,a) pair statistics follow golden ratio")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_06_phi_mobius_pairs",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Get prime gaps
    n_max = 500000
    sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9], window=13, lam=0.99)
    primes = sec.primes[sec.primes > 10]
    gaps = np.diff(primes)
    
    print(f"\nAnalyzing {len(gaps)} prime gaps (n_max={n_max})")
    
    # Classify pairs
    symmetric, mobius, orphans = analyze_mobius_pairs(gaps)
    
    print(f"  Symmetric (a,a): {len(symmetric)} types, {sum(symmetric.values())} total")
    print(f"  Möbius (a,b)↔(b,a): {len(mobius)//2} pair types, {sum(mobius.values())} total")
    print(f"  Orphan (no partner): {len(orphans)} types, {sum(orphans.values())} total")
    
    # Test 1: Ratio analysis within Möbius pairs
    print("\n[Test 1] count(a,b)/count(b,a) Ratio Distribution")
    print("-" * 50)
    
    phi_analysis = phi_ratio_analysis(mobius)
    print(f"  Mean ratio max/min: {phi_analysis['mean_ratio']:.4f}")
    print(f"  φ = {phi_analysis['phi']:.4f}")
    print(f"  Difference: {phi_analysis['diff_from_phi']:.4f}")
    print(f"  Pairs near φ (±0.3): {phi_analysis['near_phi_count']}")
    print(f"  Pairs near 1 (±0.3): {phi_analysis['near_one_count']}")
    
    if phi_analysis['diff_from_phi'] < 0.1:
        print("  ✓ Mean ratio matches φ!")
    
    results["tests"].append({"name": "phi_ratio_analysis", "data": phi_analysis})
    
    # Test 2: Symmetric/Asymmetric ratio
    print("\n[Test 2] Symmetric vs Möbius vs Orphan Ratios")
    print("-" * 50)
    
    sym_analysis = symmetric_asymmetric_ratio(symmetric, mobius, orphans)
    print(f"  Symmetric fraction: {sym_analysis['symmetric_fraction']:.4f}")
    print(f"  Möbius fraction: {sym_analysis['mobius_fraction']:.4f}")
    print(f"  Orphan fraction: {sym_analysis['orphan_fraction']:.4f}")
    
    if "sym_to_mobius_ratio" in sym_analysis:
        print(f"  Sym/Möbius ratio: {sym_analysis['sym_to_mobius_ratio']:.4f}")
        print(f"  Diff from 1/φ: {sym_analysis['sym_mob_diff_from_phi_inv']:.4f}")
    
    results["tests"].append({"name": "symmetric_asymmetric", "data": sym_analysis})
    
    # Test 3: Fibonacci enrichment
    print("\n[Test 3] Fibonacci Gap Enrichment in Möbius Pairs")
    print("-" * 50)
    
    fib_analysis = fibonacci_gap_enrichment(mobius, gaps)
    print(f"  Fibonacci gaps in Möbius: {fib_analysis['fib_gaps_in_mobius']}")
    print(f"  Enrichment factor: {fib_analysis['enrichment']:.2f}x")
    print(f"  All Möbius gap sizes: {fib_analysis['all_mobius_gaps'][:15]}...")
    
    results["tests"].append({"name": "fibonacci_enrichment", "data": fib_analysis})
    
    # Test 4: Imbalance analysis (connect to SEC L+/L-)
    print("\n[Test 4] Pair Imbalance (Increasing vs Decreasing)")
    print("-" * 50)
    
    imb_analysis = pair_imbalance_analysis(mobius)
    print(f"  Total increasing (a<b): {imb_analysis['increasing_total']}")
    print(f"  Total decreasing (a>b): {imb_analysis['decreasing_total']}")
    print(f"  Ratio inc/dec: {imb_analysis['inc_dec_ratio']:.4f}")
    print(f"  φ = {PHI:.4f}")
    print(f"  Diff from φ: {imb_analysis['diff_from_phi']:.4f}")
    print(f"  Diff from 1/φ: {imb_analysis['diff_from_phi_inv']:.4f}")
    
    if imb_analysis['diff_from_phi'] < 0.1 or imb_analysis['diff_from_phi_inv'] < 0.1:
        print("  ✓ Inc/Dec ratio matches φ!")
    
    results["tests"].append({"name": "imbalance_analysis", "data": imb_analysis})
    
    # Test 5: Scale dependence
    print("\n[Test 5] Scale Dependence of Ratios")
    print("-" * 50)
    
    scales = [10000, 50000, 100000, 200000, 500000]
    scale_results = []
    
    for n in scales:
        primes_n = sec.primes[(sec.primes > 10) & (sec.primes < n)]
        if len(primes_n) < 100:
            continue
        gaps_n = np.diff(primes_n)
        sym_n, mob_n, orph_n = analyze_mobius_pairs(gaps_n)
        
        if mob_n:
            imb_n = pair_imbalance_analysis(mob_n)
            scale_results.append({
                "n_max": n,
                "n_gaps": len(gaps_n),
                "inc_dec_ratio": imb_n["inc_dec_ratio"],
                "diff_from_phi": imb_n["diff_from_phi"]
            })
            print(f"  n={n}: inc/dec={imb_n['inc_dec_ratio']:.4f}, Δφ={imb_n['diff_from_phi']:.4f}")
    
    results["tests"].append({"name": "scale_dependence", "data": scale_results})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    findings = []
    
    if phi_analysis['diff_from_phi'] < 0.2:
        findings.append(f"Mean pair ratio {phi_analysis['mean_ratio']:.3f} approaches φ")
    
    if imb_analysis['diff_from_phi'] < 0.2 or imb_analysis['diff_from_phi_inv'] < 0.2:
        findings.append(f"Inc/Dec ratio {imb_analysis['inc_dec_ratio']:.3f} relates to φ")
    
    if fib_analysis['enrichment'] > 1.2:
        findings.append(f"Fibonacci gaps {fib_analysis['enrichment']:.1f}x enriched in Möbius pairs")
    
    results["summary"] = {"findings": findings}
    
    for f in findings:
        print(f"✓ {f}")
    
    if not findings:
        print("No strong φ signatures in Möbius pair frequencies")
        print(f"(Mean ratio = {phi_analysis['mean_ratio']:.3f}, φ = {PHI:.3f})")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_06_phi_mobius", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
