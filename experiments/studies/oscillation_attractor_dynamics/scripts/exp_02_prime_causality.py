"""
Experiment 02: Prime Impulse → Zero-Crossing Causality
======================================================

The exp_01 results show primes are NOT at zero-crossings.
BUT proximity=0 showed 1.59x enrichment - exact matches are enriched!

New hypothesis: Primes CAUSE zero-crossings that occur AFTER them.
The prime injects an impulse I(p), and the system crosses zero
within a characteristic lag time.

This would mean:
- Primes → positive impulse → triggers crossing shortly after
- The crossing is the RESPONSE to the prime, not the prime itself
"""

import numpy as np
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import detect_zero_crossings, save_results
from sec_core import compute_sec, FIRST_50_PRIMES, PHI


def crossings_after_primes(E, prime_mask, max_lag=20):
    """
    For each prime p, find the first zero-crossing after p.
    Measure the lag distribution.
    """
    crossings, directions = detect_zero_crossings(E, start_idx=100)
    crossing_set = set(crossings)
    
    primes = np.where(prime_mask)[0]
    primes = primes[primes > 100]  # Skip transient
    
    lags = []
    for p in primes:
        # Find first crossing after p
        for lag in range(1, max_lag + 1):
            if p + lag in crossing_set:
                lags.append(lag)
                break
    
    return np.array(lags)


def crossings_before_primes(E, prime_mask, max_lag=20):
    """
    For each prime p, find the last zero-crossing before p.
    """
    crossings, _ = detect_zero_crossings(E, start_idx=100)
    crossing_set = set(crossings)
    
    primes = np.where(prime_mask)[0]
    primes = primes[primes > 100]
    
    lags = []
    for p in primes:
        for lag in range(1, max_lag + 1):
            if p - lag in crossing_set:
                lags.append(lag)
                break
    
    return np.array(lags)


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 02: Prime Impulse → Zero-Crossing Causality")
    print("Hypothesis: Primes CAUSE zero-crossings that occur shortly after")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_02_prime_causality",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Compute SEC
    n_max = 200000
    sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9], window=13, lam=0.99)
    
    # Test 1: Lag distribution for crossings AFTER primes
    print("\n[Test 1] Crossings AFTER primes (causal direction)")
    print("-" * 50)
    
    lags_after = crossings_after_primes(sec.E, sec.prime_mask, max_lag=30)
    
    if len(lags_after) > 0:
        print(f"  Primes analyzed: {len(np.where(sec.prime_mask)[0][100:])}")
        print(f"  Crossings found after: {len(lags_after)}")
        print(f"  Mean lag: {np.mean(lags_after):.2f}")
        print(f"  Median lag: {np.median(lags_after):.1f}")
        print(f"  Mode lag: {np.bincount(lags_after).argmax()}")
        
        # Lag histogram
        hist, bins = np.histogram(lags_after, bins=range(1, 32))
        peak_lag = np.argmax(hist) + 1
        print(f"  Peak lag: {peak_lag}")
        
        results["tests"].append({
            "name": "crossings_after_primes",
            "n_primes": len(lags_after),
            "mean_lag": float(np.mean(lags_after)),
            "median_lag": float(np.median(lags_after)),
            "std_lag": float(np.std(lags_after)),
            "peak_lag": int(peak_lag),
            "histogram": hist.tolist()
        })
    
    # Test 2: Lag distribution for crossings BEFORE primes (control)
    print("\n[Test 2] Crossings BEFORE primes (anti-causal, control)")
    print("-" * 50)
    
    lags_before = crossings_before_primes(sec.E, sec.prime_mask, max_lag=30)
    
    if len(lags_before) > 0:
        print(f"  Crossings found before: {len(lags_before)}")
        print(f"  Mean lag: {np.mean(lags_before):.2f}")
        print(f"  Median lag: {np.median(lags_before):.1f}")
        
        hist_before, _ = np.histogram(lags_before, bins=range(1, 32))
        peak_before = np.argmax(hist_before) + 1
        print(f"  Peak lag: {peak_before}")
        
        results["tests"].append({
            "name": "crossings_before_primes",
            "n_primes": len(lags_before),
            "mean_lag": float(np.mean(lags_before)),
            "median_lag": float(np.median(lags_before)),
            "peak_lag": int(peak_before),
            "histogram": hist_before.tolist()
        })
    
    # Test 3: Asymmetry analysis
    print("\n[Test 3] Causal Asymmetry")
    print("-" * 50)
    
    mean_after = np.mean(lags_after)
    mean_before = np.mean(lags_before)
    asymmetry = (mean_before - mean_after) / (mean_before + mean_after)
    
    print(f"  Mean lag AFTER primes: {mean_after:.2f}")
    print(f"  Mean lag BEFORE primes: {mean_before:.2f}")
    print(f"  Asymmetry: {asymmetry:.3f}")
    
    if asymmetry > 0.05:
        print("  ✓ Crossings occur CLOSER after primes → causal relationship")
    elif asymmetry < -0.05:
        print("  Crossings occur CLOSER before primes → anti-causal?")
    else:
        print("  No significant asymmetry")
    
    results["tests"].append({
        "name": "asymmetry",
        "mean_after": float(mean_after),
        "mean_before": float(mean_before),
        "asymmetry": float(asymmetry),
        "causal": asymmetry > 0.05
    })
    
    # Test 4: Crossing direction analysis
    print("\n[Test 4] Crossing Direction After Primes")
    print("-" * 50)
    
    crossings, directions = detect_zero_crossings(sec.E, start_idx=100)
    crossing_dir_map = dict(zip(crossings, directions))
    
    primes = np.where(sec.prime_mask)[0]
    primes = primes[primes > 100]
    
    pos_going = 0  # Crosses from negative to positive
    neg_going = 0  # Crosses from positive to negative
    
    for p in primes:
        for lag in range(1, 20):
            if p + lag in crossing_dir_map:
                if crossing_dir_map[p + lag] > 0:
                    pos_going += 1
                else:
                    neg_going += 1
                break
    
    total = pos_going + neg_going
    if total > 0:
        pos_frac = pos_going / total
        print(f"  Positive-going crossings: {pos_going} ({100*pos_frac:.1f}%)")
        print(f"  Negative-going crossings: {neg_going} ({100*(1-pos_frac):.1f}%)")
        print(f"  Ratio: {pos_frac/(1-pos_frac):.2f}" if pos_frac < 1 else "  All positive")
        
        # Compare to φ
        phi_inv = 1/PHI
        diff_from_phi = abs(pos_frac - phi_inv)
        print(f"  Difference from 1/φ (0.618): {diff_from_phi:.3f}")
        
        results["tests"].append({
            "name": "crossing_direction",
            "positive_going": pos_going,
            "negative_going": neg_going,
            "positive_fraction": float(pos_frac),
            "phi_comparison": float(phi_inv),
            "difference_from_phi": float(diff_from_phi)
        })
    
    # Test 5: Prime-crossing coincidence at EXACT position
    print("\n[Test 5] EXACT Prime-Crossing Coincidence")
    print("-" * 50)
    
    crossing_set = set(crossings)
    prime_set = set(primes)
    exact_matches = len(crossing_set & prime_set)
    
    expected = len(crossings) * len(primes) / n_max
    enrichment = exact_matches / expected if expected > 0 else 0
    
    print(f"  Crossings at EXACT prime positions: {exact_matches}")
    print(f"  Expected by chance: {expected:.1f}")
    print(f"  Enrichment: {enrichment:.2f}x")
    
    results["tests"].append({
        "name": "exact_coincidence",
        "exact_matches": exact_matches,
        "expected": float(expected),
        "enrichment": float(enrichment)
    })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results["summary"] = {
        "asymmetry": float(asymmetry),
        "causal_relationship": asymmetry > 0.05,
        "key_finding": "Primes influence crossing timing" if asymmetry > 0.05 else "No clear causal pattern"
    }
    
    print(f"Asymmetry: {asymmetry:.3f}")
    if asymmetry > 0.05:
        print("✓ Evidence for causal relationship: primes → crossings")
    else:
        print("No clear causal asymmetry detected")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_02_causality", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
