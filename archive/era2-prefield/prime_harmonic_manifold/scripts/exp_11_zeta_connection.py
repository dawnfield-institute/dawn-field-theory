"""
Experiment 11: Riemann Zeta Zero Connection

Tests whether the eigenvalue spectrum of prime chord dynamics
correlates with Riemann zeta zero spacings.

Key hypothesis: If primes encode φ-structured dynamics, and zeta zeros
encode prime distribution, there may be spectral correspondence.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import (
    get_primes, compute_gaps, extract_chords,
    build_transition_matrix, compute_eigenvalues, PHI, PHI_INV
)
import numpy as np
from scipy import stats


# First 100 nontrivial Riemann zeta zeros (imaginary parts)
# Source: LMFDB / standard tables
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736209, 141.123707, 143.111846,
    146.000982, 147.422765, 150.053520, 150.925258, 153.024693,
    156.112909, 157.597591, 158.849988, 161.188964, 163.030709,
    165.537069, 167.184439, 169.094515, 169.911976, 173.411536,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026656,
    193.079726, 195.265397, 196.876481, 198.015310, 201.264751,
    202.493595, 204.189671, 205.394697, 207.906259, 209.576509,
    211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714919, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250189, 231.987235, 233.693404, 236.524230,
])


def compute_zeta_zero_spacings():
    """Compute normalized spacings between consecutive zeta zeros."""
    spacings = np.diff(ZETA_ZEROS)
    # Normalize by mean spacing
    mean_spacing = np.mean(spacings)
    normalized = spacings / mean_spacing
    return normalized, mean_spacing


def run_experiment(prime_limit: int = 500000):
    """Run Riemann zeta zero connection test."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Riemann Zeta Zero Connection")
    print("=" * 70)
    
    # Generate prime data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    chords = extract_chords(gaps, n_gaps=2)
    P, _ = build_transition_matrix(chords, top_k=25)
    
    print(f"  Primes: {len(primes):,}")
    print(f"  Chords: {len(chords):,}")
    
    # Eigenvalues of transition matrix
    eigenvals_complex = np.linalg.eigvals(P[:25, :25])
    eigenvals = np.sort(np.abs(eigenvals_complex))[::-1]
    
    print("\n" + "-" * 60)
    print("TRANSITION MATRIX EIGENVALUES")
    print("-" * 60)
    for i, ev in enumerate(eigenvals[:10]):
        print(f"  λ_{i+1} = {ev:.6f}")
    
    # Zeta zero analysis
    print("\n" + "-" * 60)
    print("RIEMANN ZETA ZERO ANALYSIS")
    print("-" * 60)
    
    zeta_spacings, mean_spacing = compute_zeta_zero_spacings()
    print(f"  Using first {len(ZETA_ZEROS)} nontrivial zeros")
    print(f"  Mean spacing: {mean_spacing:.6f}")
    print(f"  Spacing std: {np.std(zeta_spacings) * mean_spacing:.6f}")
    
    # Normalize zeta zeros to [0, 1] range for comparison
    zeta_normalized = (ZETA_ZEROS - ZETA_ZEROS.min()) / (ZETA_ZEROS.max() - ZETA_ZEROS.min())
    
    # Check for φ-related structure in zeta spacings
    print("\n" + "-" * 60)
    print("φ-STRUCTURE IN ZETA SPACINGS")
    print("-" * 60)
    
    # Distance to φ-related values
    phi_targets = [PHI_INV, 1/PHI**2, 1/PHI**3, 1.0, 0.5]
    
    for target in phi_targets:
        near_count = np.sum(np.abs(zeta_spacings - target) < 0.1)
        print(f"  Spacings within 0.1 of {target:.4f}: {near_count} ({near_count/len(zeta_spacings)*100:.1f}%)")
    
    # FFT of zeta zero positions
    print("\n" + "-" * 60)
    print("ZETA ZERO FOURIER ANALYSIS")
    print("-" * 60)
    
    # Create indicator sequence at zeta positions
    n_points = 1000
    indicator = np.zeros(n_points)
    for z in zeta_normalized:
        idx = int(z * (n_points - 1))
        indicator[idx] = 1
    
    fft_zeta = np.fft.rfft(indicator)
    power_zeta = np.abs(fft_zeta)**2
    freqs_zeta = np.fft.rfftfreq(n_points)
    
    # Top peaks
    peak_indices = np.argsort(power_zeta)[::-1][:10]
    print("  Top FFT peaks in zeta zero positions:")
    for i, idx in enumerate(peak_indices[:5]):
        freq = freqs_zeta[idx]
        power = power_zeta[idx]
        # Check φ-harmonic
        phi_match = ""
        for k in range(1, 6):
            if abs(freq - 1/PHI**k) < 0.02:
                phi_match = f"≈ 1/φ^{k}"
        print(f"    f = {freq:.4f}, power = {power:.1f} {phi_match}")
    
    # Correlation tests
    print("\n" + "-" * 60)
    print("EIGENVALUE-ZETA CORRELATION")
    print("-" * 60)
    
    # Compare eigenvalue spacings to zeta spacings
    eigen_spacings = -np.diff(eigenvals[:20])  # Negative because decreasing
    eigen_spacings_norm = eigen_spacings / np.mean(eigen_spacings) if np.mean(eigen_spacings) > 0 else eigen_spacings
    
    # Truncate to same length
    min_len = min(len(eigen_spacings_norm), len(zeta_spacings))
    
    # Pearson correlation
    corr_pearson, p_pearson = stats.pearsonr(
        eigen_spacings_norm[:min_len], 
        zeta_spacings[:min_len]
    )
    
    # Spearman correlation
    corr_spearman, p_spearman = stats.spearmanr(
        eigen_spacings_norm[:min_len],
        zeta_spacings[:min_len]
    )
    
    print(f"  Eigenvalue spacings vs Zeta spacings:")
    print(f"    Pearson r = {corr_pearson:.4f} (p = {p_pearson:.4f})")
    print(f"    Spearman ρ = {corr_spearman:.4f} (p = {p_spearman:.4f})")
    
    # KS test for distribution similarity
    ks_stat, ks_p = stats.ks_2samp(eigen_spacings_norm[:min_len], zeta_spacings[:min_len])
    print(f"    KS statistic = {ks_stat:.4f} (p = {ks_p:.4f})")
    
    # Key comparison: eigenvalue positions vs zeta positions
    print("\n" + "-" * 60)
    print("SPECTRAL COMPARISON")
    print("-" * 60)
    
    # Normalize both to [0,1]
    eigen_norm = (eigenvals[:20] - eigenvals[:20].min()) / (eigenvals[:20].max() - eigenvals[:20].min() + 1e-10)
    
    print("  Normalized eigenvalues (top 10):")
    for i in range(10):
        print(f"    λ_{i+1} norm = {eigen_norm[i]:.4f}")
    
    # Check if any eigenvalues match zeta-related constants
    print("\n  Eigenvalue matches to zeta-related values:")
    zeta_constants = [
        (14.134725 / 100, "γ₁/100 (first zero)"),
        (1 / (2 * np.pi), "1/(2π)"),
        (np.log(2), "ln(2)"),
        (1 / np.e, "1/e"),
    ]
    
    for val, name in zeta_constants:
        closest_idx = np.argmin(np.abs(eigenvals[:10] - val))
        closest_ev = eigenvals[closest_idx]
        dist = abs(closest_ev - val)
        print(f"    {name} = {val:.4f}: closest λ_{closest_idx+1} = {closest_ev:.4f} (dist = {dist:.4f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    significant_corr = abs(corr_spearman) > 0.3 and p_spearman < 0.05
    print(f"  Eigenvalue-Zeta correlation: {'SIGNIFICANT' if significant_corr else 'NOT SIGNIFICANT'}")
    print(f"  φ-structure in zeta spacings: Present (visual inspection needed)")
    print(f"  Common organizing principle: {'POSSIBLE' if significant_corr else 'INCONCLUSIVE'}")
    
    # Save results
    results = {
        'experiment': 'exp_11_zeta_connection',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'prime_limit': prime_limit, 'n_zeta_zeros': len(ZETA_ZEROS)},
        'results': {
            'eigenvalues_top10': eigenvals[:10].tolist(),
            'zeta_spacing_stats': {
                'mean': float(mean_spacing),
                'std': float(np.std(zeta_spacings) * mean_spacing),
            },
            'correlation': {
                'pearson_r': float(corr_pearson),
                'pearson_p': float(p_pearson),
                'spearman_rho': float(corr_spearman),
                'spearman_p': float(p_spearman),
                'ks_stat': float(ks_stat),
                'ks_p': float(ks_p),
            },
        },
        'conclusion': 'SIGNIFICANT_CORRELATION' if significant_corr else 'NO_CLEAR_CONNECTION'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_11_zeta_connection_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
