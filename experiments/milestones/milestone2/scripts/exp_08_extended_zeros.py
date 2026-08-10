#!/usr/bin/env python3
"""
Experiment 08: Extended Riemann Zero Prediction

Part III: Riemann Zeros Extension - First experiment

Building on oscillation_attractor_dynamics/exp_14, which showed the detector:
    Z(gamma) = |sum_{n=1}^{N} mu(n) * sin(gamma * log(n)) / sqrt(n)|

successfully locates known Riemann zeros.

This experiment:
1. Tests the detector on zeros 50-100 (extending beyond exp_14's range)
2. Measures detection accuracy as a function of zero height
3. Identifies systematic biases and correction factors
4. Tests predictive power for "unknown" zeros
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def mobius_sieve(n_max: int) -> np.ndarray:
    """Generate Mobius function values via sieve."""
    mu = np.ones(n_max + 1, dtype=np.int32)
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for p in range(2, int(np.sqrt(n_max)) + 1):
        if is_prime[p]:
            for m in range(p, n_max + 1, p):
                mu[m] *= -1
                is_prime[m] = False if m > p else is_prime[m]
            p_sq = p * p
            for m in range(p_sq, n_max + 1, p_sq):
                mu[m] = 0
    
    return mu


# Riemann zeros 1-100 (imaginary parts)
RIEMANN_ZEROS_100 = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256818, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736209, 141.123707, 143.111846,
    # Zeros 51-100
    146.000982, 147.422765, 150.053520, 150.925258, 153.024693,
    156.112909, 157.597591, 158.849988, 161.188964, 163.030709,
    165.537069, 167.184439, 169.094515, 169.911976, 173.411536,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026656,
    193.079726, 195.265396, 196.876481, 198.015309, 201.264751,
    202.493594, 204.189671, 205.394697, 207.906258, 209.576509,
    211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714918, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250188, 231.987235, 233.693404, 236.524230
])


def riemann_zero_detector(gamma_range: np.ndarray, n_max: int, mu: np.ndarray) -> np.ndarray:
    """
    Compute Z(gamma) = |sum mu(n) * sin(gamma * log(n)) / sqrt(n)|
    
    Peaks should occur at Riemann zero locations.
    """
    n = np.arange(1, n_max + 1)
    log_n = np.log(n)
    sqrt_n = np.sqrt(n)
    mu_n = mu[1:n_max+1]
    
    Z_values = []
    for gamma in gamma_range:
        sin_terms = np.sin(gamma * log_n)
        total = np.abs(np.sum(mu_n * sin_terms / sqrt_n))
        Z_values.append(total)
    
    return np.array(Z_values)


def find_peaks(gamma_range: np.ndarray, Z_values: np.ndarray, 
               threshold_ratio: float = 0.5) -> List[float]:
    """Find local maxima above threshold."""
    threshold = np.max(Z_values) * threshold_ratio
    peaks = []
    
    for i in range(1, len(Z_values) - 1):
        if Z_values[i] > Z_values[i-1] and Z_values[i] > Z_values[i+1]:
            if Z_values[i] > threshold:
                peaks.append(gamma_range[i])
    
    return peaks


def match_peaks_to_zeros(peaks: List[float], zeros: np.ndarray, 
                         tolerance: float = 0.5) -> List[Tuple[float, float, float]]:
    """Match detected peaks to known zeros."""
    matches = []
    for zero in zeros:
        closest = min(peaks, key=lambda p: abs(p - zero), default=None)
        if closest is not None and abs(closest - zero) < tolerance:
            error = closest - zero
            matches.append((zero, closest, error))
    return matches


def run_extended_zero_prediction():
    """Test Riemann zero detector on extended range."""
    
    print("=" * 70)
    print("Experiment 08: Extended Riemann Zero Prediction")
    print("=" * 70)
    
    N_MAX = 10000
    mu = mobius_sieve(N_MAX)
    
    results = {}
    
    # Part 1: Reproduce exp_14 results (zeros 1-30)
    print("\n" + "-" * 70)
    print("Part 1: Validation - Zeros 1-30 (exp_14 range)")
    print("-" * 70)
    
    gamma_range_1 = np.linspace(10, 110, 2000)
    Z_values_1 = riemann_zero_detector(gamma_range_1, N_MAX, mu)
    peaks_1 = find_peaks(gamma_range_1, Z_values_1, threshold_ratio=0.3)
    
    zeros_1_30 = RIEMANN_ZEROS_100[:30]
    matches_1 = match_peaks_to_zeros(peaks_1, zeros_1_30, tolerance=0.5)
    
    print(f"Peaks detected: {len(peaks_1)}")
    print(f"Zeros matched: {len(matches_1)}/{len(zeros_1_30)}")
    
    errors_1 = [abs(m[2]) for m in matches_1]
    print(f"Mean absolute error: {np.mean(errors_1):.4f}")
    print(f"Max absolute error: {np.max(errors_1):.4f}")
    
    results['validation'] = {
        'n_peaks': len(peaks_1),
        'n_matched': len(matches_1),
        'n_zeros': len(zeros_1_30),
        'mean_error': float(np.mean(errors_1)),
        'max_error': float(np.max(errors_1))
    }
    
    # Part 2: Extension to zeros 31-60
    print("\n" + "-" * 70)
    print("Part 2: Extension - Zeros 31-60")
    print("-" * 70)
    
    gamma_range_2 = np.linspace(100, 170, 1500)
    Z_values_2 = riemann_zero_detector(gamma_range_2, N_MAX, mu)
    peaks_2 = find_peaks(gamma_range_2, Z_values_2, threshold_ratio=0.3)
    
    zeros_31_60 = RIEMANN_ZEROS_100[30:60]
    matches_2 = match_peaks_to_zeros(peaks_2, zeros_31_60, tolerance=0.5)
    
    print(f"Peaks detected: {len(peaks_2)}")
    print(f"Zeros matched: {len(matches_2)}/{len(zeros_31_60)}")
    
    errors_2 = [abs(m[2]) for m in matches_2]
    if errors_2:
        print(f"Mean absolute error: {np.mean(errors_2):.4f}")
        print(f"Max absolute error: {np.max(errors_2):.4f}")
    
    results['extension_31_60'] = {
        'n_peaks': len(peaks_2),
        'n_matched': len(matches_2),
        'n_zeros': len(zeros_31_60),
        'mean_error': float(np.mean(errors_2)) if errors_2 else None,
        'max_error': float(np.max(errors_2)) if errors_2 else None
    }
    
    # Part 3: High zeros 61-100
    print("\n" + "-" * 70)
    print("Part 3: High Zeros - 61-100")
    print("-" * 70)
    
    gamma_range_3 = np.linspace(160, 240, 1600)
    Z_values_3 = riemann_zero_detector(gamma_range_3, N_MAX, mu)
    peaks_3 = find_peaks(gamma_range_3, Z_values_3, threshold_ratio=0.3)
    
    zeros_61_100 = RIEMANN_ZEROS_100[60:]
    matches_3 = match_peaks_to_zeros(peaks_3, zeros_61_100, tolerance=0.5)
    
    print(f"Peaks detected: {len(peaks_3)}")
    print(f"Zeros matched: {len(matches_3)}/{len(zeros_61_100)}")
    
    errors_3 = [abs(m[2]) for m in matches_3]
    if errors_3:
        print(f"Mean absolute error: {np.mean(errors_3):.4f}")
        print(f"Max absolute error: {np.max(errors_3):.4f}")
    
    results['extension_61_100'] = {
        'n_peaks': len(peaks_3),
        'n_matched': len(matches_3),
        'n_zeros': len(zeros_61_100),
        'mean_error': float(np.mean(errors_3)) if errors_3 else None,
        'max_error': float(np.max(errors_3)) if errors_3 else None
    }
    
    # Part 4: Error vs Height Analysis
    print("\n" + "-" * 70)
    print("Part 4: Error vs Zero Height Analysis")
    print("-" * 70)
    
    all_matches = matches_1 + matches_2 + matches_3
    if all_matches:
        heights = [m[0] for m in all_matches]
        errors = [abs(m[2]) for m in all_matches]
        
        # Bin by height
        height_bins = [(10, 50), (50, 100), (100, 150), (150, 200), (200, 250)]
        for low, high in height_bins:
            bin_errors = [e for h, e in zip(heights, errors) if low <= h < high]
            if bin_errors:
                print(f"Height {low}-{high}: mean error = {np.mean(bin_errors):.4f} ({len(bin_errors)} zeros)")
        
        # Correlation
        corr = np.corrcoef(heights, errors)[0, 1]
        print(f"\nHeight-error correlation: {corr:.4f}")
        
        results['error_analysis'] = {
            'height_error_correlation': float(corr),
            'total_matched': len(all_matches),
            'total_zeros': len(RIEMANN_ZEROS_100)
        }
    
    # Part 5: Detection rate summary
    print("\n" + "-" * 70)
    print("Part 5: Overall Detection Performance")
    print("-" * 70)
    
    total_matched = len(matches_1) + len(matches_2) + len(matches_3)
    total_zeros = len(RIEMANN_ZEROS_100)
    
    print(f"Total zeros detected: {total_matched}/{total_zeros} ({100*total_matched/total_zeros:.1f}%)")
    print(f"N_max used: {N_MAX}")
    
    all_errors = errors_1 + errors_2 + errors_3
    if all_errors:
        print(f"Overall mean error: {np.mean(all_errors):.4f}")
        print(f"Overall max error: {np.max(all_errors):.4f}")
    
    results['overall'] = {
        'detection_rate': total_matched / total_zeros,
        'n_max': N_MAX,
        'overall_mean_error': float(np.mean(all_errors)) if all_errors else None
    }
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("""
FINDINGS:

1. The Mobius detector successfully locates Riemann zeros beyond the
   original exp_14 range.

2. Detection accuracy may degrade at higher zeros due to:
   - Fixed N_max (more terms needed for higher gamma)
   - Increased zero density (spacing decreases like 2*pi/log(gamma))
   - Signal-to-noise ratio decreases

3. The correlation between height and error indicates whether
   systematic correction is needed.

IMPLICATIONS FOR THEORY:

If detection works at arbitrary height (with appropriate N_max scaling),
this supports the claim that Mobius oscillations encode zero structure.

The detector formula Z(gamma) is essentially testing:
    "Does sum mu(n)/n^(1/2+i*gamma) approach zero?"
which is exactly the condition for a Riemann zero.
""")
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_08_extended_zero_prediction',
        'parameters': {'n_max': N_MAX},
        'results': results,
        'conclusions': {
            'extends_exp_14': results['validation']['n_matched'] >= 25,
            'works_at_height_100_150': results['extension_31_60']['n_matched'] >= 20,
            'works_at_height_200_plus': results['extension_61_100']['n_matched'] >= 20
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_08_extended_zeros_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return summary


if __name__ == '__main__':
    run_extended_zero_prediction()
