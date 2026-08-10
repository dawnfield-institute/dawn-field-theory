"""
Experiment 04: Möbius Structure in Prime Injection/Crystallization
===================================================================

Hypothesis: The prime→composite→prime cycle has Möbius topology.

The Möbius confluence operator shows:
- P_{t+1} = A_t(u+π, 1-v)  [half-twist + reflection]
- Antiperiodic modes satisfy f(u+π, 1-v) = -f(u, v)
- Ξ emerges from spectral ratio of (n+½)² / n² modes

For prime distribution:
- Primes inject at "P" points on the Möbius band
- Composites form in the "A" field (crystallization)
- The cycle P→A→P' has Möbius topology

Predictions:
1. Inter-prime intervals should show half-twist symmetry
2. The SEC stress field E(n) should be antiperiodic across prime gaps
3. Ξ ≈ 1.0571 should appear in the spectral structure of E(n)

This connects primes_again.md (oscillation attractors) with 
the Möbius confluence operator.
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

from oscillation_engine import save_results
from sec_core import compute_sec, FIRST_50_PRIMES, PHI

# Constants
XI = 1.0571428571428572  # From cellular automata validation


def test_half_twist_symmetry(E, primes, gap_size=6):
    """
    Test if E(n) in inter-prime regions shows half-twist symmetry.
    
    Möbius antiperiodic: f(x + L/2) ≈ -f(x) for half the period
    
    For a prime gap of size g, test if:
    E(p + k) ≈ -E(p + g - k) for k in [1, g/2]
    """
    correlations = []
    
    for i in range(len(primes) - 1):
        p1, p2 = primes[i], primes[i+1]
        gap = p2 - p1
        
        if gap == gap_size:
            # Extract the inter-prime segment
            segment = E[p1+1:p2]  # Exclude the primes themselves
            n = len(segment)
            
            if n >= 4:
                # Split into first and second half
                half = n // 2
                first_half = segment[:half]
                second_half = segment[n-half:][::-1]  # Reversed
                
                # Check antiperiodic: first_half ≈ -second_half
                correlation = np.corrcoef(first_half, -second_half)[0, 1]
                correlations.append(correlation)
    
    if correlations:
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        return {
            "gap_size": gap_size,
            "n_segments": len(correlations),
            "mean_antiperiodic_correlation": float(mean_corr),
            "std": float(std_corr),
            "is_antiperiodic": mean_corr > 0.5
        }
    return {"gap_size": gap_size, "error": "no segments found"}


def spectral_xi_emergence(E, start_idx=1000, segment_length=4096):
    """
    Test if Ξ emerges from the spectral structure of E(n).
    
    Möbius prediction: ratio of half-integer to integer modes = Ξ
    """
    # Extract a clean segment
    segment = E[start_idx:start_idx + segment_length]
    
    # FFT
    spectrum = np.abs(fft(segment))
    freqs = fftfreq(len(segment))
    
    # Only positive frequencies
    positive = freqs > 0
    spectrum_pos = spectrum[positive]
    freqs_pos = freqs[positive]
    
    # Bin into "integer-like" and "half-integer-like" modes
    # Based on normalized frequency: integer modes at k/N, half-integer at (k+0.5)/N
    
    n_modes = len(spectrum_pos)
    integer_power = 0
    half_integer_power = 0
    
    for i, (f, p) in enumerate(zip(freqs_pos, spectrum_pos)):
        # Determine if closer to integer or half-integer spacing
        mode_index = f * segment_length
        residual = mode_index - round(mode_index)
        
        if abs(residual) < 0.25:
            integer_power += p**2
        else:
            half_integer_power += p**2
    
    ratio = np.sqrt(half_integer_power / integer_power) if integer_power > 0 else 0
    
    return {
        "segment_length": segment_length,
        "integer_power": float(integer_power),
        "half_integer_power": float(half_integer_power),
        "power_ratio": float(ratio),
        "expected_xi": float(XI),
        "difference_from_xi": float(abs(ratio - XI))
    }


def confluence_cycle_detection(E, I, primes):
    """
    Test for confluence cycles: P → A → P'
    
    In the Möbius model:
    - Prime = P (potential injection)
    - Composites = A (actualization/crystallization)  
    - Next prime = P' (confluenced from A)
    
    Test: Does the "energy" at p' relate to the cumulative A in [p, p']?
    """
    results = []
    
    for i in range(len(primes) - 1):
        p1, p2 = primes[i], primes[i+1]
        
        # P (injection at prime)
        P_inject = I[p1]  # Impulse at first prime
        
        # A (crystallization in gap)
        A_cumulative = np.sum(E[p1+1:p2])  # Total stress in gap
        
        # P' (new potential at next prime)
        P_next = E[p2]  # Stress level at next prime
        
        # Confluence prediction: P' should relate to A through half-twist
        # Simple test: sign relationship
        results.append({
            "gap": p2 - p1,
            "P_inject": float(P_inject),
            "A_cumulative": float(A_cumulative),
            "P_next": float(P_next),
            "confluence_ratio": float(P_next / A_cumulative) if A_cumulative != 0 else 0
        })
    
    # Analyze confluence ratios
    ratios = [r["confluence_ratio"] for r in results if abs(r["confluence_ratio"]) < 10]
    
    if ratios:
        mean_ratio = np.mean(ratios)
        # Does this relate to XI or 1/XI?
        return {
            "n_cycles": len(ratios),
            "mean_confluence_ratio": float(mean_ratio),
            "abs_mean": float(np.mean(np.abs(ratios))),
            "xi_comparison": float(XI),
            "xi_inv_comparison": float(1/XI),
            "difference_from_xi": float(abs(np.mean(np.abs(ratios)) - XI)),
            "difference_from_xi_inv": float(abs(np.mean(np.abs(ratios)) - 1/XI))
        }
    return {"error": "no valid ratios"}


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 04: Möbius Structure in Prime Distribution")
    print("Hypothesis: Prime→composite→prime cycle has Möbius topology")
    print("=" * 70)
    
    results = {
        "experiment_id": "exp_04_mobius_structure",
        "timestamp": datetime.now().isoformat(),
        "core_hypothesis": "Ξ emerges from antiperiodic (Möbius) structure of prime gaps",
        "tests": []
    }
    
    # Compute SEC
    n_max = 100000
    sec = compute_sec(n_max=n_max, factor_base=FIRST_50_PRIMES[:9], window=13, lam=0.99)
    primes = sec.primes[sec.primes > 100]
    
    # Test 1: Half-twist symmetry in inter-prime regions
    print("\n[Test 1] Half-Twist (Antiperiodic) Symmetry")
    print("-" * 50)
    
    gap_results = {}
    for gap_size in [4, 6, 8, 10, 12, 14]:
        result = test_half_twist_symmetry(sec.E, primes, gap_size)
        if "error" not in result:
            gap_results[gap_size] = result
            print(f"  Gap {gap_size}: antiperiodic corr = {result['mean_antiperiodic_correlation']:.3f} (n={result['n_segments']})")
            if result['is_antiperiodic']:
                print(f"           ✓ Shows antiperiodic symmetry!")
    
    results["tests"].append({
        "name": "half_twist_symmetry",
        "data": gap_results
    })
    
    # Test 2: Spectral Ξ emergence
    print("\n[Test 2] Spectral Ξ Emergence")
    print("-" * 50)
    
    spectral = spectral_xi_emergence(sec.E, start_idx=1000, segment_length=4096)
    print(f"  Power ratio (half-int/int): {spectral['power_ratio']:.4f}")
    print(f"  Expected Ξ: {spectral['expected_xi']:.4f}")
    print(f"  Difference: {spectral['difference_from_xi']:.4f}")
    
    if spectral['difference_from_xi'] < 0.1:
        print("  ✓ Spectral ratio matches Ξ!")
    
    results["tests"].append({
        "name": "spectral_xi",
        "data": spectral
    })
    
    # Test 3: Confluence cycle detection
    print("\n[Test 3] Confluence Cycles (P → A → P')")
    print("-" * 50)
    
    confluence = confluence_cycle_detection(sec.E, sec.I, primes)
    if "error" not in confluence:
        print(f"  Cycles analyzed: {confluence['n_cycles']}")
        print(f"  Mean |confluence ratio|: {confluence['abs_mean']:.4f}")
        print(f"  Ξ = {confluence['xi_comparison']:.4f}")
        print(f"  1/Ξ = {confluence['xi_inv_comparison']:.4f}")
        print(f"  Diff from Ξ: {confluence['difference_from_xi']:.4f}")
        print(f"  Diff from 1/Ξ: {confluence['difference_from_xi_inv']:.4f}")
    
    results["tests"].append({
        "name": "confluence_cycles",
        "data": confluence
    })
    
    # Test 4: Combined E(p) at primes analysis
    print("\n[Test 4] Stress Field at Primes")
    print("-" * 50)
    
    E_at_primes = sec.E[primes]
    E_at_composites = sec.E[~sec.prime_mask][100:]  # Skip small composites
    
    print(f"  Mean E(prime): {np.mean(E_at_primes):.4f}")
    print(f"  Mean E(composite): {np.mean(E_at_composites):.4f}")
    print(f"  Ratio: {np.mean(E_at_primes)/np.mean(E_at_composites):.4f}")
    
    # Sign distribution
    prime_positive = np.mean(E_at_primes > 0)
    print(f"  E(prime) > 0: {100*prime_positive:.1f}%")
    
    results["tests"].append({
        "name": "stress_at_primes",
        "data": {
            "mean_E_prime": float(np.mean(E_at_primes)),
            "mean_E_composite": float(np.mean(E_at_composites)),
            "ratio": float(np.mean(E_at_primes)/np.mean(E_at_composites)),
            "prime_positive_fraction": float(prime_positive)
        }
    })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Möbius Structure in Primes")
    print("=" * 70)
    
    findings = []
    
    # Check for antiperiodic symmetry in any gap
    antiperiodic_gaps = [g for g, r in gap_results.items() if r.get('is_antiperiodic', False)]
    if antiperiodic_gaps:
        findings.append(f"✓ Antiperiodic symmetry in gaps: {antiperiodic_gaps}")
    
    if spectral['difference_from_xi'] < 0.2:
        findings.append(f"✓ Spectral ratio {spectral['power_ratio']:.3f} approaches Ξ={XI:.3f}")
    
    results["summary"] = {
        "antiperiodic_gaps": antiperiodic_gaps,
        "spectral_xi_match": spectral['difference_from_xi'] < 0.2,
        "findings": findings
    }
    
    for f in findings:
        print(f)
    
    if not findings:
        print("Möbius structure not clearly detected - may need different parameters")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, "exp_04_mobius_structure", results_dir)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
