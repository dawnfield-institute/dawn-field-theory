#!/usr/bin/env python3
"""
Experiment 15: Extended Harmonic Analysis
=========================================

Goal: Full spectral decomposition of the SEC stress field E(n).

Building on exp_12's discovery that 99.96% of power is in factor base primes,
this experiment:

1. Phase analysis: Are the prime harmonics phase-locked?
2. Harmonic interactions: Do primes interact multiplicatively?
3. Chirikov criterion: Is there resonance overlap?
4. Power-law scaling: How does amplitude scale with prime size?
5. Windowed spectrograms: How does spectrum evolve with n?
6. Cross-spectrum: Correlation between E and prime indicator

Trace output: results/exp_15_extended_harmonic_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, rfft, rfftfreq, fftfreq
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sec_core import (
    compute_sec, prime_sieve, FIRST_50_PRIMES, PHI
)

PHI_INV = 1 / PHI


def extract_harmonic_components(E: np.ndarray, factor_base: List[int]) -> Dict[str, Any]:
    """
    Extract amplitude and phase for each factor base prime harmonic.
    """
    n = len(E)
    E_centered = E - E.mean()
    
    # Full FFT for phase information
    fft_result = fft(E_centered)
    freqs = fftfreq(n)
    
    harmonics = {}
    total_power = np.sum(np.abs(fft_result)**2)
    
    for p in factor_base:
        # Find index closest to frequency 1/p
        target_freq = 1/p
        
        # Search both positive and negative frequencies
        idx_pos = np.argmin(np.abs(freqs - target_freq))
        idx_neg = np.argmin(np.abs(freqs + target_freq))
        
        # Use the one with larger amplitude
        if np.abs(fft_result[idx_pos]) > np.abs(fft_result[idx_neg]):
            idx = idx_pos
        else:
            idx = idx_neg
        
        amplitude = np.abs(fft_result[idx])
        phase = np.angle(fft_result[idx])
        power = amplitude**2
        
        harmonics[p] = {
            "frequency": float(freqs[idx]),
            "period": float(1/freqs[idx]) if freqs[idx] != 0 else float('inf'),
            "amplitude": float(amplitude),
            "phase": float(phase),
            "phase_degrees": float(np.degrees(phase)),
            "power": float(power),
            "power_fraction": float(power / total_power) if total_power > 0 else 0
        }
    
    return {
        "harmonics": harmonics,
        "total_power": float(total_power),
        "n_samples": n
    }


def analyze_phase_relationships(harmonics: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Analyze phase relationships between prime harmonics.
    
    Questions:
    - Are phases random or correlated?
    - Is there a consistent phase progression?
    - Do Fibonacci-adjacent primes have special phase relationships?
    """
    primes = sorted(harmonics.keys())
    phases = [harmonics[p]["phase"] for p in primes]
    
    # Phase differences between adjacent primes
    phase_diffs = []
    for i in range(len(primes) - 1):
        diff = phases[i+1] - phases[i]
        # Normalize to [-π, π]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        phase_diffs.append(diff)
    
    # Test for uniformity (Rayleigh test)
    # If phases are uniformly distributed, mean resultant length → 0
    complex_phases = [np.exp(1j * p) for p in phases]
    mean_resultant = np.abs(np.mean(complex_phases))
    
    # Circular statistics
    mean_phase = np.angle(np.mean(complex_phases))
    
    # Phase vs prime index correlation
    phase_unwrapped = np.unwrap(phases)
    corr_phase_idx = float(np.corrcoef(range(len(primes)), phase_unwrapped)[0, 1])
    
    # Phase vs log(prime) correlation
    log_primes = np.log(primes)
    corr_phase_logp = float(np.corrcoef(log_primes, phase_unwrapped)[0, 1])
    
    return {
        "primes": primes,
        "phases": phases,
        "phase_differences": phase_diffs,
        "mean_phase": float(mean_phase),
        "mean_phase_degrees": float(np.degrees(mean_phase)),
        "mean_resultant_length": float(mean_resultant),
        "phase_uniformity": mean_resultant < 0.3,  # Low = uniform
        "correlation_phase_vs_index": corr_phase_idx,
        "correlation_phase_vs_log_prime": corr_phase_logp
    }


def analyze_power_scaling(harmonics: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Analyze how harmonic amplitude scales with prime size.
    
    Questions:
    - Is there a power-law relationship: A(p) ~ p^α?
    - What is the scaling exponent?
    """
    primes = sorted(harmonics.keys())
    amplitudes = [harmonics[p]["amplitude"] for p in primes]
    
    # Log-log regression
    log_primes = np.log(primes)
    log_amps = np.log(amplitudes)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_primes, log_amps)
    
    return {
        "primes": primes,
        "amplitudes": amplitudes,
        "power_law_exponent": float(slope),
        "power_law_intercept": float(np.exp(intercept)),
        "power_law_r_squared": float(r_value**2),
        "power_law_p_value": float(p_value),
        "scaling_type": "power_law" if r_value**2 > 0.8 else "irregular"
    }


def analyze_harmonic_interactions(E: np.ndarray, factor_base: List[int]) -> Dict[str, Any]:
    """
    Look for multiplicative interactions between prime harmonics.
    
    If primes p and q interact, we might see power at frequencies:
    - 1/(p*q) (product)
    - |1/p - 1/q| (beat frequency)
    - 1/p + 1/q (sum frequency)
    """
    n = len(E)
    E_centered = E - E.mean()
    
    fft_result = rfft(E_centered)
    power = np.abs(fft_result)**2
    freqs = rfftfreq(n)
    
    interactions = []
    
    # Check all pairs
    for i, p in enumerate(factor_base):
        for q in factor_base[i+1:]:
            # Product frequency
            product_freq = 1 / (p * q)
            idx_prod = np.argmin(np.abs(freqs - product_freq))
            
            # Beat frequency
            beat_freq = abs(1/p - 1/q)
            idx_beat = np.argmin(np.abs(freqs - beat_freq))
            
            # Sum frequency
            sum_freq = 1/p + 1/q
            idx_sum = np.argmin(np.abs(freqs - sum_freq))
            
            interactions.append({
                "primes": [p, q],
                "product_period": p * q,
                "product_power": float(power[idx_prod]),
                "beat_period": float(1/beat_freq) if beat_freq > 0 else float('inf'),
                "beat_power": float(power[idx_beat]),
                "sum_power": float(power[idx_sum])
            })
    
    # Find strongest interactions
    interactions_sorted = sorted(interactions, key=lambda x: x["product_power"], reverse=True)
    
    return {
        "n_pairs": len(interactions),
        "top_interactions": interactions_sorted[:10],
        "average_product_power": float(np.mean([i["product_power"] for i in interactions])),
        "average_beat_power": float(np.mean([i["beat_power"] for i in interactions]))
    }


def compute_spectrogram(E: np.ndarray, window_size: int = 1000, 
                        overlap: int = 500) -> Dict[str, Any]:
    """
    Compute spectrogram to see how spectrum evolves with n.
    """
    f, t, Sxx = signal.spectrogram(E, fs=1.0, window='hann', 
                                    nperseg=window_size, noverlap=overlap)
    
    # Convert to periods
    periods = 1 / f[1:]  # Skip DC
    
    # Find power at prime periods for each time slice
    prime_power_evolution = {}
    factor_base = FIRST_50_PRIMES[:9]
    
    for p in factor_base:
        # Find closest period
        idx = np.argmin(np.abs(periods - p))
        power_vs_time = Sxx[idx + 1, :]  # +1 to skip DC
        prime_power_evolution[p] = power_vs_time.tolist()
    
    return {
        "n_time_bins": len(t),
        "n_freq_bins": len(f),
        "time_centers": t.tolist(),
        "prime_power_evolution": prime_power_evolution,
        "total_power_evolution": Sxx.sum(axis=0).tolist()
    }


def cross_spectrum_with_primes(E: np.ndarray, prime_mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute cross-spectrum between E and prime indicator.
    
    This reveals which frequencies are coherent between E and primality.
    """
    # Coherence between E and prime indicator
    f, coherence = signal.coherence(E, prime_mask.astype(float), fs=1.0, nperseg=1024)
    
    # Find peaks in coherence
    peak_indices = signal.find_peaks(coherence, height=0.1)[0]
    
    peaks = []
    for idx in peak_indices:
        if f[idx] > 0:
            peaks.append({
                "frequency": float(f[idx]),
                "period": float(1/f[idx]),
                "coherence": float(coherence[idx])
            })
    
    # Sort by coherence
    peaks_sorted = sorted(peaks, key=lambda x: x["coherence"], reverse=True)
    
    return {
        "mean_coherence": float(np.mean(coherence)),
        "max_coherence": float(np.max(coherence)),
        "n_significant_peaks": len(peak_indices),
        "top_coherent_periods": peaks_sorted[:10]
    }


def chirikov_overlap_analysis(harmonics: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Apply Chirikov resonance overlap criterion.
    
    In dynamical systems, chaos emerges when resonances overlap.
    K = Δω / δω where Δω is resonance width and δω is spacing.
    K > 1 indicates chaos.
    
    For primes: δω ∝ 1/p - 1/q, resonance width ∝ amplitude.
    """
    primes = sorted(harmonics.keys())
    
    overlaps = []
    for i in range(len(primes) - 1):
        p1, p2 = primes[i], primes[i+1]
        
        # Frequency spacing
        spacing = 1/p1 - 1/p2
        
        # Resonance widths (proportional to amplitude)
        width1 = harmonics[p1]["amplitude"] / 1000  # Normalized
        width2 = harmonics[p2]["amplitude"] / 1000
        
        # Overlap parameter
        if spacing > 0:
            K = (width1 + width2) / (2 * spacing)
        else:
            K = float('inf')
        
        overlaps.append({
            "primes": [p1, p2],
            "spacing": float(spacing),
            "combined_width": float(width1 + width2),
            "chirikov_K": float(K),
            "overlap": K > 1
        })
    
    n_overlapping = sum(1 for o in overlaps if o["overlap"])
    
    return {
        "overlap_pairs": overlaps,
        "n_overlapping": n_overlapping,
        "fraction_overlapping": n_overlapping / len(overlaps) if overlaps else 0,
        "mean_K": float(np.mean([o["chirikov_K"] for o in overlaps if o["chirikov_K"] < float('inf')])),
        "interpretation": "chaotic" if n_overlapping > len(overlaps) / 2 else "regular"
    }


def run_experiment(n_max: int = 50000, save_trace: bool = True) -> Dict[str, Any]:
    """Run extended harmonic analysis."""
    
    print("=" * 70)
    print("EXPERIMENT 15: Extended Harmonic Analysis")
    print("=" * 70)
    
    # Compute SEC
    factor_base = FIRST_50_PRIMES[:9]
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=101, lam=0.99)
    
    # Get E for odd integers
    idx = np.arange(3, n_max + 1, 2)
    E_odd = sec.E[idx]
    pm_odd = sec.prime_mask[idx]
    
    results = {}
    
    # 1. Extract harmonics
    print(f"\n" + "-" * 70)
    print("1. HARMONIC EXTRACTION")
    print("-" * 70)
    
    harm_result = extract_harmonic_components(E_odd, factor_base)
    results["harmonics"] = harm_result
    
    print(f"\n{'Prime':>6} {'Amplitude':>12} {'Phase (°)':>12} {'Power %':>12}")
    print("-" * 45)
    for p in factor_base:
        h = harm_result["harmonics"][p]
        print(f"{p:>6} {h['amplitude']:>12.2f} {h['phase_degrees']:>12.1f} {h['power_fraction']*100:>11.2f}%")
    
    # 2. Phase analysis
    print(f"\n" + "-" * 70)
    print("2. PHASE RELATIONSHIPS")
    print("-" * 70)
    
    phase_result = analyze_phase_relationships(harm_result["harmonics"])
    results["phase_analysis"] = phase_result
    
    print(f"\nMean phase: {phase_result['mean_phase_degrees']:.1f}°")
    print(f"Mean resultant length: {phase_result['mean_resultant_length']:.4f}")
    print(f"Phase uniformity: {'Yes' if phase_result['phase_uniformity'] else 'No'}")
    print(f"Correlation (phase vs index): {phase_result['correlation_phase_vs_index']:.4f}")
    print(f"Correlation (phase vs log p): {phase_result['correlation_phase_vs_log_prime']:.4f}")
    
    # 3. Power scaling
    print(f"\n" + "-" * 70)
    print("3. POWER-LAW SCALING")
    print("-" * 70)
    
    scaling_result = analyze_power_scaling(harm_result["harmonics"])
    results["power_scaling"] = scaling_result
    
    print(f"\nScaling type: {scaling_result['scaling_type']}")
    print(f"Power-law exponent: {scaling_result['power_law_exponent']:.4f}")
    print(f"R²: {scaling_result['power_law_r_squared']:.4f}")
    print(f"A(p) ~ {scaling_result['power_law_intercept']:.2f} × p^{scaling_result['power_law_exponent']:.2f}")
    
    # 4. Harmonic interactions
    print(f"\n" + "-" * 70)
    print("4. HARMONIC INTERACTIONS")
    print("-" * 70)
    
    interact_result = analyze_harmonic_interactions(E_odd, factor_base)
    results["interactions"] = interact_result
    
    print(f"\nNumber of prime pairs: {interact_result['n_pairs']}")
    print(f"Average product-frequency power: {interact_result['average_product_power']:.2f}")
    print(f"Average beat-frequency power: {interact_result['average_beat_power']:.2f}")
    print(f"\nTop interactions by product power:")
    for i, top in enumerate(interact_result["top_interactions"][:5]):
        print(f"  {top['primes']}: period {top['product_period']}, power {top['product_power']:.2f}")
    
    # 5. Spectrogram
    print(f"\n" + "-" * 70)
    print("5. SPECTRAL EVOLUTION")
    print("-" * 70)
    
    spec_result = compute_spectrogram(E_odd)
    results["spectrogram"] = spec_result
    
    print(f"\nTime bins: {spec_result['n_time_bins']}")
    print(f"Frequency bins: {spec_result['n_freq_bins']}")
    
    # Check if prime power is stable over time
    for p in [3, 5, 7]:
        power_vec = spec_result["prime_power_evolution"][p]
        cv = np.std(power_vec) / np.mean(power_vec) if np.mean(power_vec) > 0 else 0
        print(f"Prime {p} power stability (CV): {cv:.4f}")
    
    # 6. Cross-spectrum
    print(f"\n" + "-" * 70)
    print("6. COHERENCE WITH PRIMALITY")
    print("-" * 70)
    
    cross_result = cross_spectrum_with_primes(E_odd, pm_odd)
    results["cross_spectrum"] = cross_result
    
    print(f"\nMean coherence: {cross_result['mean_coherence']:.4f}")
    print(f"Max coherence: {cross_result['max_coherence']:.4f}")
    print(f"Significant coherent periods: {cross_result['n_significant_peaks']}")
    if cross_result["top_coherent_periods"]:
        print(f"\nTop coherent periods:")
        for i, peak in enumerate(cross_result["top_coherent_periods"][:5]):
            print(f"  Period {peak['period']:.1f}: coherence {peak['coherence']:.4f}")
    
    # 7. Chirikov analysis
    print(f"\n" + "-" * 70)
    print("7. CHIRIKOV RESONANCE OVERLAP")
    print("-" * 70)
    
    chirikov_result = chirikov_overlap_analysis(harm_result["harmonics"])
    results["chirikov"] = chirikov_result
    
    print(f"\nOverlapping pairs: {chirikov_result['n_overlapping']} / {len(chirikov_result['overlap_pairs'])}")
    print(f"Mean Chirikov K: {chirikov_result['mean_K']:.4f}")
    print(f"Interpretation: {chirikov_result['interpretation']}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
KEY FINDINGS:

1. PHASE STRUCTURE
   - Phases are {'uniform (random)' if phase_result['phase_uniformity'] else 'structured'}
   - Phase-index correlation: {phase_result['correlation_phase_vs_index']:.3f}

2. POWER SCALING
   - Amplitude ~ p^{scaling_result['power_law_exponent']:.2f}
   - {'Good' if scaling_result['power_law_r_squared'] > 0.8 else 'Weak'} power-law fit (R²={scaling_result['power_law_r_squared']:.3f})

3. COHERENCE
   - E and primality are {'coherent' if cross_result['max_coherence'] > 0.3 else 'weakly coherent'}
   - Max coherence: {cross_result['max_coherence']:.3f}

4. DYNAMICAL REGIME
   - System is {chirikov_result['interpretation']}
   - Mean overlap K: {chirikov_result['mean_K']:.3f}
""")
    
    # Validation
    validation = {
        "power_at_primes_dominates": sum(
            harm_result["harmonics"][p]["power_fraction"] 
            for p in factor_base
        ) > 0.5,
        "phases_have_structure": not phase_result["phase_uniformity"] or 
                                 abs(phase_result["correlation_phase_vs_log_prime"]) > 0.3,
        "power_law_holds": scaling_result["power_law_r_squared"] > 0.5,
        "coherent_with_primes": cross_result["max_coherence"] > 0.1
    }
    
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)
    for check, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    results["validation"] = validation
    
    # Save trace
    if save_trace:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"exp_15_extended_harmonic_{timestamp}.json"
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"\nTrace saved: {filepath.name}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=50000)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    
    run_experiment(n_max=args.n_max, save_trace=not args.no_trace)
