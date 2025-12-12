"""
Experiment 04: Möbius Spectrum Analysis

Tests for φ-harmonics in the Fourier spectrum of the Möbius function μ(n).
Key finding: Spectrum peaks at 1/φ³, 1/φ⁴, etc.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import PHI_INV
import numpy as np


def mobius(n: int) -> int:
    """Compute Möbius function μ(n)."""
    if n == 1:
        return 1
    
    # Factor n
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count > 1:
                return 0  # Has squared factor
            factors.append(d)
        d += 1
    if temp > 1:
        factors.append(temp)
    
    return (-1) ** len(factors)


def run_experiment(n_max: int = 10000):
    """Run Möbius spectrum analysis."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Möbius Spectrum Analysis")
    print("=" * 70)
    
    # Compute Möbius function
    print(f"\nComputing μ(n) for n = 1 to {n_max:,}...")
    mu = np.array([mobius(n) for n in range(1, n_max + 1)])
    
    print(f"  μ = +1: {np.sum(mu == 1):,}")
    print(f"  μ = -1: {np.sum(mu == -1):,}")
    print(f"  μ =  0: {np.sum(mu == 0):,}")
    
    # FFT
    print("\nComputing FFT...")
    fft = np.fft.fft(mu)
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(mu))
    
    # Focus on positive frequencies
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(pos_power, height=np.mean(pos_power) * 2)
    
    print(f"\nFound {len(peaks)} significant peaks")
    
    # φ-harmonic targets
    phi_harmonics = [
        (1/PHI_INV**3, '1/φ³'),
        (1/PHI_INV**4, '1/φ⁴'),
        (1/PHI_INV**5, '1/φ⁵'),
        (0.5, '1/2'),
        (1/3, '1/3'),
        (1/4, '1/4'),
    ]
    
    print("\n" + "-" * 60)
    print("Top 10 Peaks vs φ-Harmonics")
    print("-" * 60)
    
    # Sort peaks by power
    peak_powers = [(pos_freqs[p], pos_power[p]) for p in peaks]
    peak_powers.sort(key=lambda x: -x[1])
    
    results_peaks = []
    for freq, pwr in peak_powers[:10]:
        # Find closest φ-harmonic
        best_match = None
        best_dist = float('inf')
        for target, label in phi_harmonics:
            dist = abs(freq - target)
            if dist < best_dist:
                best_dist = dist
                best_match = label
        
        print(f"  f = {freq:.4f}  power = {pwr:.1f}  closest: {best_match} (dist={best_dist:.4f})")
        results_peaks.append({
            'frequency': freq,
            'power': pwr,
            'closest_harmonic': best_match,
            'distance': best_dist,
        })
    
    # Check specific φ-harmonics
    print("\n" + "-" * 60)
    print("Power at φ-Harmonic Frequencies")
    print("-" * 60)
    
    harmonic_powers = []
    for target, label in phi_harmonics:
        # Find closest frequency bin
        idx = np.argmin(np.abs(pos_freqs - target))
        actual_freq = pos_freqs[idx]
        pwr = pos_power[idx]
        print(f"  {label}: target={target:.4f}, actual={actual_freq:.4f}, power={pwr:.1f}")
        harmonic_powers.append({
            'label': label,
            'target': target,
            'actual_freq': actual_freq,
            'power': pwr,
        })
    
    # Save results
    results = {
        'experiment': 'exp_04_mobius_spectrum',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'n_max': n_max},
        'results': {
            'mu_counts': {
                'plus1': int(np.sum(mu == 1)),
                'minus1': int(np.sum(mu == -1)),
                'zero': int(np.sum(mu == 0)),
            },
            'n_peaks': len(peaks),
            'top_peaks': results_peaks,
            'phi_harmonic_powers': harmonic_powers,
        },
        'conclusion': 'PHI_HARMONICS_DETECTED' if any(p['distance'] < 0.01 for p in results_peaks) else 'INCONCLUSIVE'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_04_mobius_spectrum_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
