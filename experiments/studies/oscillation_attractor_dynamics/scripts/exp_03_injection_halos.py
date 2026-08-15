"""
Experiment 03: Prime Injection Halos - Crystallization Field Analysis
=====================================================================

Core reframe: Primes aren't attractors or zero-crossings.
Primes are INJECTION POINTS that seed structure into entropy soup.

Like SHA hash injection in cosmo.py:
- Prime p injects positive impulse I(p) > 0
- The inter-prime region is where crystallization occurs
- Composites are the crystallized structure
- The stress field E(n) should show characteristic "halo" shape between primes

This experiment measures:
1. The shape of E(n) between consecutive primes (the "crystallization halo")
2. Whether halo shape scales with prime gap size
3. Whether Fibonacci structure appears in the crystallization pattern
4. Whether φ governs the injection/crystallization balance
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy.stats import pearsonr
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import save_results
from sec_core import compute_sec, FIRST_50_PRIMES, PHI, FIBONACCI


def extract_inter_prime_regions(E, I, primes, min_gap=4, max_gap=30):
    """
    Extract stress field and impulse in regions between consecutive primes.
    
    Returns dict: gap_size -> list of (E_segment, I_segment) tuples
    """
    regions = defaultdict(list)
    
    for i in range(len(primes) - 1):
        p1, p2 = primes[i], primes[i + 1]
        gap = p2 - p1
        
        if min_gap <= gap <= max_gap:
            # Extract the inter-prime segment (excluding the primes themselves)
            E_seg = E[p1+1:p2].copy()
            I_seg = I[p1+1:p2].copy()
            regions[gap].append((E_seg, I_seg))
    
    return regions


def analyze_halo_shape(regions):
    """
    Analyze the average shape of E(n) in inter-prime regions.
    
    Hypothesis: Should show characteristic crystallization pattern
    - Initial positive (residual injection)
    - Decay toward negative (crystallization/collapse)
    - Pattern governed by φ
    """
    results = {}
    
    for gap_size, segments in regions.items():
        if len(segments) < 10:  # Need enough samples
            continue
            
        # Normalize each segment to [0, 1] position within gap
        n_points = gap_size - 1  # Points between primes
        
        # Average the stress field shape
        E_sum = np.zeros(n_points)
        I_sum = np.zeros(n_points)
        count = 0
        
        for E_seg, I_seg in segments:
            if len(E_seg) == n_points:
                E_sum += E_seg
                I_sum += I_seg
                count += 1
        
        if count > 0:
            E_avg = E_sum / count
            I_avg = I_sum / count
            
            # Characterize the shape
            results[gap_size] = {
                "count": count,
                "E_mean": float(np.mean(E_avg)),
                "E_std": float(np.std(E_avg)),
                "E_start": float(E_avg[0]),
                "E_end": float(E_avg[-1]),
                "E_min": float(np.min(E_avg)),
                "E_max": float(np.max(E_avg)),
                "E_trend": float(E_avg[-1] - E_avg[0]),  # Positive = rising, negative = falling
                "I_mean": float(np.mean(I_avg)),
                "I_positive_frac": float(np.mean(I_avg > 0)),
                "E_profile": E_avg.tolist(),
                "I_profile": I_avg.tolist()
            }
            
            # Find zero-crossing position if any
            signs = np.sign(E_avg)
            crossings = np.where(np.diff(signs) != 0)[0]
            if len(crossings) > 0:
                # Relative position of first crossing (0 = start, 1 = end)
                first_cross = crossings[0] / (n_points - 1)
                results[gap_size]["crossing_position"] = float(first_cross)
                results[gap_size]["crossing_vs_phi"] = float(abs(first_cross - 1/PHI))
    
    return results


def test_fibonacci_resonance(regions, fib_gaps=[2, 3, 5, 8, 13]):
    """
    Test if Fibonacci gap sizes show distinct crystallization patterns.
    """
    fib_results = {}
    non_fib_results = {}
    
    for gap_size, data in regions.items():
        if gap_size in fib_gaps:
            fib_results[gap_size] = data
        else:
            non_fib_results[gap_size] = data
    
    # Compare average properties
    fib_trends = [d["E_trend"] for d in fib_results.values()]
    non_fib_trends = [d["E_trend"] for d in non_fib_results.values()]
    
    comparison = {
        "fib_gaps_found": list(fib_results.keys()),
        "non_fib_gaps_found": list(non_fib_results.keys())[:10],  # Sample
        "fib_mean_trend": float(np.mean(fib_trends)) if fib_trends else None,
        "non_fib_mean_trend": float(np.mean(non_fib_trends)) if non_fib_trends else None,
    }
    
    if fib_trends and non_fib_trends:
        comparison["trend_difference"] = comparison["fib_mean_trend"] - comparison["non_fib_mean_trend"]
    
    return comparison


def test_phi_crossing(halo_data):
    """
    Test if zero-crossings occur at φ-related positions within gaps.
    """
    positions = []
    for gap_size, data in halo_data.items():
        if "crossing_position" in data:
            positions.append(data["crossing_position"])
    
    if not positions:
        return {"error": "no crossings found"}
    
    positions = np.array(positions)
    
    # Test against 1/φ ≈ 0.618
    phi_inv = 1 / PHI
    mean_pos = np.mean(positions)
    std_pos = np.std(positions)
    
    return {
        "n_gaps_with_crossing": len(positions),
        "mean_crossing_position": float(mean_pos),
        "std_crossing_position": float(std_pos),
        "phi_inverse": float(phi_inv),
        "difference_from_phi": float(abs(mean_pos - phi_inv)),
        "within_1_std_of_phi": float(abs(mean_pos - phi_inv)) < std_pos
    }


def test_injection_crystallization_balance(sec, primes):
    """
    Test the injection vs crystallization rate.
    
    Injection: sum of I(p) for primes
    Crystallization: sum of I(c) for composites
    """
    I = sec.I
    prime_set = set(primes)
    
    injection_total = 0
    crystallization_total = 0
    
    for n in range(3, len(I)):
        if n in prime_set:
            injection_total += I[n]
        else:
            crystallization_total += I[n]
    
    # Balance ratio
    total = abs(injection_total) + abs(crystallization_total)
    if total > 0:
        injection_frac = abs(injection_total) / total
        crystallization_frac = abs(crystallization_total) / total
    else:
        injection_frac = crystallization_frac = 0.5
    
    return {
        "injection_total": float(injection_total),
        "crystallization_total": float(crystallization_total),
        "injection_fraction": float(injection_frac),
        "crystallization_fraction": float(crystallization_frac),
        "ratio": float(abs(injection_total) / abs(crystallization_total)) if crystallization_total != 0 else 0,
        "phi_comparison": float(1/PHI),
        "difference_from_phi_inv": float(abs(injection_frac - 1/PHI))
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 03: Prime Injection Halos")
    print("Hypothesis: Primes inject structure; composites crystallize around them")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_03_injection_halos",
        "timestamp": datetime.now().isoformat(),
        "core_hypothesis": "Primes are entropy injection points, composites are crystallization",
        "tests": []
    }
    
    # Compute SEC
    n_max = 100000
    sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9], window=13, lam=0.99)
    primes = sec.primes[sec.primes > 100]  # Skip transient
    
    print(f"\nAnalyzing {len(primes)} primes up to {n_max}")
    
    # Test 1: Extract and analyze inter-prime regions
    print("\n[Test 1] Inter-Prime Crystallization Halos")
    print("-" * 50)
    
    regions = extract_inter_prime_regions(sec.E, sec.I, primes, min_gap=4, max_gap=30)
    halo_data = analyze_halo_shape(regions)
    
    print(f"  Gap sizes analyzed: {sorted(halo_data.keys())}")
    
    # Show a few examples
    for gap in [6, 8, 12]:
        if gap in halo_data:
            d = halo_data[gap]
            print(f"  Gap {gap}: E_trend={d['E_trend']:.4f}, count={d['count']}")
    
    results["tests"].append({
        "name": "halo_shapes",
        "gap_data": {str(k): v for k, v in halo_data.items()}
    })
    
    # Test 2: Fibonacci resonance
    print("\n[Test 2] Fibonacci Gap Resonance")
    print("-" * 50)
    
    fib_comparison = test_fibonacci_resonance(halo_data)
    print(f"  Fibonacci gaps found: {fib_comparison['fib_gaps_found']}")
    if fib_comparison.get('fib_mean_trend') is not None:
        print(f"  Fib gap mean trend: {fib_comparison['fib_mean_trend']:.4f}")
        print(f"  Non-fib gap mean trend: {fib_comparison['non_fib_mean_trend']:.4f}")
        if fib_comparison.get('trend_difference'):
            print(f"  Difference: {fib_comparison['trend_difference']:.4f}")
    
    results["tests"].append({
        "name": "fibonacci_resonance",
        "data": fib_comparison
    })
    
    # Test 3: φ at crossing positions
    print("\n[Test 3] Crossing Position vs φ")
    print("-" * 50)
    
    phi_test = test_phi_crossing(halo_data)
    if "error" not in phi_test:
        print(f"  Mean crossing position: {phi_test['mean_crossing_position']:.4f}")
        print(f"  1/φ = {phi_test['phi_inverse']:.4f}")
        print(f"  Difference: {phi_test['difference_from_phi']:.4f}")
        if phi_test['within_1_std_of_phi']:
            print("  ✓ Crossing position consistent with φ")
    else:
        print(f"  {phi_test['error']}")
    
    results["tests"].append({
        "name": "phi_crossing",
        "data": phi_test
    })
    
    # Test 4: Injection vs Crystallization balance
    print("\n[Test 4] Injection/Crystallization Balance")
    print("-" * 50)
    
    balance = test_injection_crystallization_balance(sec, primes)
    print(f"  Total injection (primes): {balance['injection_total']:.2f}")
    print(f"  Total crystallization (composites): {balance['crystallization_total']:.2f}")
    print(f"  Injection fraction: {balance['injection_fraction']:.4f}")
    print(f"  1/φ = {balance['phi_comparison']:.4f}")
    print(f"  Difference from 1/φ: {balance['difference_from_phi_inv']:.4f}")
    
    if balance['difference_from_phi_inv'] < 0.05:
        print("  ✓ Balance matches φ!")
    
    results["tests"].append({
        "name": "injection_crystallization_balance",
        "data": balance
    })
    
    # Test 5: Prime impulse signature
    print("\n[Test 5] Prime Injection Signature")
    print("-" * 50)
    
    prime_impulses = sec.I[primes]
    composite_mask = ~sec.prime_mask
    composite_mask[:3] = False
    composite_impulses = sec.I[composite_mask]
    
    print(f"  Mean prime impulse I(p): {np.mean(prime_impulses):.4f}")
    print(f"  Mean composite impulse I(c): {np.mean(composite_impulses):.4f}")
    print(f"  Prime/composite ratio: {np.mean(prime_impulses)/np.mean(composite_impulses):.2f}")
    
    # Primes should have POSITIVE impulse (injection)
    # Composites should have NEGATIVE impulse (crystallization/collapse)
    prime_positive_frac = np.mean(prime_impulses > 0)
    composite_negative_frac = np.mean(composite_impulses < 0)
    
    print(f"  Primes with I > 0: {100*prime_positive_frac:.1f}%")
    print(f"  Composites with I < 0: {100*composite_negative_frac:.1f}%")
    
    if prime_positive_frac > 0.5 and composite_negative_frac > 0.5:
        print("  ✓ Injection/crystallization signature confirmed!")
    
    results["tests"].append({
        "name": "impulse_signature",
        "data": {
            "prime_mean_impulse": float(np.mean(prime_impulses)),
            "composite_mean_impulse": float(np.mean(composite_impulses)),
            "prime_positive_fraction": float(prime_positive_frac),
            "composite_negative_fraction": float(composite_negative_frac),
            "signature_confirmed": prime_positive_frac > 0.5 and composite_negative_frac > 0.5
        }
    })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    sig = results["tests"][-1]["data"]
    bal = balance
    
    key_findings = []
    
    if sig["signature_confirmed"]:
        key_findings.append("✓ Primes inject (+), composites crystallize (-)")
    
    if bal["difference_from_phi_inv"] < 0.1:
        key_findings.append(f"✓ Injection fraction ≈ 1/φ (diff={bal['difference_from_phi_inv']:.3f})")
    
    results["summary"] = {
        "injection_signature_confirmed": sig["signature_confirmed"],
        "phi_balance_confirmed": bal["difference_from_phi_inv"] < 0.1,
        "key_findings": key_findings
    }
    
    for finding in key_findings:
        print(finding)
    
    if not key_findings:
        print("No strong confirmations yet - need parameter tuning")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_03_injection_halos", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
