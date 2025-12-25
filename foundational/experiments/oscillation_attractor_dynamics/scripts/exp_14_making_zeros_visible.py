#!/usr/bin/env python3
"""
Experiment 14: Making Riemann Zeros Visible

Just as I(n) made primes visible (100% have I(p) > 0), can we find a
transformation that makes Riemann zeros visible?

The explicit formula tells us:
    ψ(x) - x = -Σ_ρ x^ρ/ρ - log(2π) - (1/2)log(1 - x^{-2})

So zeros appear as oscillations in ψ(x) - x when viewed in log(x) space.

Strategy:
1. Compute ψ(x) - x for x up to N
2. Transform to log(x) space (uniformly sample in log)
3. FFT to find the γ_k frequencies
4. See if we can "detect" zeros the way I(n) detects primes

This is the standard approach from analytic number theory, implemented
to see if it connects to our field dynamics.
"""

import torch
import torch.fft
import numpy as np
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# First 50 Riemann zeros for validation
RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029535, 111.874659,
    114.320220, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256818, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736208, 141.123707, 143.111845
]


def sieve_primes(n):
    """Sieve of Eratosthenes"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]


def compute_chebyshev_psi(N, primes):
    """
    Compute ψ(x) = Σ_{p^k ≤ x} log(p) for x = 1 to N
    This is the von Mangoldt summatory function.
    """
    psi = np.zeros(N + 1)
    for p in primes:
        log_p = np.log(p)
        pk = p
        while pk <= N:
            psi[pk:] += log_p
            pk *= p
    return psi


def detect_zeros_in_psi():
    """
    Main experiment: Find Riemann zeros in ψ(x) - x
    
    The explicit formula says:
    ψ(x) - x ≈ -Σ x^{1/2}[cos(γ log x)/|ρ| + ...]
    
    So in log(x) space, we should see frequencies at γ_1, γ_2, ...
    """
    print("\n" + "=" * 70)
    print("DETECTING RIEMANN ZEROS IN ψ(x) - x")
    print("=" * 70)
    
    N = 1000000  # Go big!
    print(f"Computing ψ(x) for x up to {N:,}...")
    
    primes = sieve_primes(N)
    print(f"Found {len(primes):,} primes")
    
    psi = compute_chebyshev_psi(N, primes)
    
    # Error term: ψ(x) - x
    x = np.arange(1, N + 1)
    error = psi[1:] - x
    
    print(f"Error term range: [{error.min():.2f}, {error.max():.2f}]")
    
    # Transform to log space
    # Sample uniformly in log(x) from log(100) to log(N)
    log_min = np.log(100)  # Start above small primes
    log_max = np.log(N)
    
    # Number of samples determines frequency resolution
    # We want to resolve zeros up to ~150, so need resolution ~ 1
    # Resolution = (log_max - log_min) / num_samples
    # For resolution 0.5: num_samples = 2 * (log_max - log_min) / 0.5
    num_samples = int(4 * (log_max - log_min))  # Resolution ~ 0.25
    print(f"Sampling {num_samples} points in log space")
    
    log_x = np.linspace(log_min, log_max, num_samples)
    x_samples = np.exp(log_x)
    
    # Interpolate error at these log-uniform points
    error_log = np.interp(x_samples, x, error)
    
    # Remove the x^{1/2} envelope to get pure oscillations
    # ψ(x) - x ≈ -Σ x^{1/2} * (oscillation terms)
    # So divide by sqrt(x) to normalize
    error_normalized = error_log / np.sqrt(x_samples)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(error_normalized))
    error_windowed = error_normalized * window
    
    # Move to GPU for FFT
    signal = torch.tensor(error_windowed, device=device, dtype=torch.float64)
    
    # FFT
    fft_result = torch.fft.rfft(signal)
    power = torch.abs(fft_result) ** 2
    
    # Frequency axis
    # The sampling rate in log space is num_samples / (log_max - log_min)
    # Frequency in "per log unit" converts to γ via f = γ/(2π)
    freq_axis = torch.fft.rfftfreq(len(signal), d=(log_x[1] - log_x[0]))
    # Convert to γ: actual frequency = 2π * freq_axis
    gamma_axis = 2 * np.pi * freq_axis.cpu().numpy()
    
    power_cpu = power.cpu().numpy()
    
    print("\n" + "-" * 60)
    print("SPECTRAL PEAKS (candidate zeros)")
    print("-" * 60)
    
    # Find peaks
    from scipy.signal import find_peaks
    
    # Look for peaks above the 95th percentile
    threshold = np.percentile(power_cpu, 95)
    peaks, properties = find_peaks(power_cpu, height=threshold, distance=3)
    
    # Sort by power
    sorted_peaks = sorted(peaks, key=lambda p: power_cpu[p], reverse=True)
    
    print(f"\nFound {len(peaks)} peaks above 95th percentile")
    print(f"\n{'Rank':>4} | {'γ detected':>12} | {'Power':>12} | {'Nearest true γ':>14} | {'Error':>8}")
    print("-" * 65)
    
    matches = []
    for rank, peak in enumerate(sorted_peaks[:30], 1):
        gamma_detected = gamma_axis[peak]
        peak_power = power_cpu[peak]
        
        # Find nearest true zero
        errors = [abs(gamma_detected - g) for g in RIEMANN_ZEROS]
        nearest_idx = np.argmin(errors)
        nearest_gamma = RIEMANN_ZEROS[nearest_idx]
        error = errors[nearest_idx]
        
        match = "✓" if error < 1.0 else ""
        if error < 1.0:
            matches.append((gamma_detected, nearest_gamma, error))
        
        print(f"{rank:>4} | {gamma_detected:>12.4f} | {peak_power:>12.2e} | {nearest_gamma:>14.4f} | {error:>8.3f} {match}")
    
    print(f"\n{len(matches)} peaks within 1.0 of a known zero!")
    
    return gamma_axis, power_cpu, matches


def validate_zero_detection():
    """
    Validate by checking if detected zeros match the first 20 known zeros
    """
    print("\n" + "=" * 70)
    print("VALIDATION: MATCHING DETECTED PEAKS TO KNOWN ZEROS")
    print("=" * 70)
    
    N = 1000000
    primes = sieve_primes(N)
    psi = compute_chebyshev_psi(N, primes)
    
    x = np.arange(1, N + 1)
    error = psi[1:] - x
    
    # High resolution in log space
    log_min = np.log(100)
    log_max = np.log(N)
    num_samples = 50000  # High resolution
    
    log_x = np.linspace(log_min, log_max, num_samples)
    x_samples = np.exp(log_x)
    error_log = np.interp(x_samples, x, error)
    error_normalized = error_log / np.sqrt(x_samples)
    
    window = np.hanning(len(error_normalized))
    error_windowed = error_normalized * window
    
    signal = torch.tensor(error_windowed, device=device, dtype=torch.float64)
    fft_result = torch.fft.rfft(signal)
    power = torch.abs(fft_result) ** 2
    
    freq_axis = torch.fft.rfftfreq(len(signal), d=(log_x[1] - log_x[0]))
    gamma_axis = 2 * np.pi * freq_axis.cpu().numpy()
    power_cpu = power.cpu().numpy()
    
    # For each known zero, find the nearest peak
    print(f"\n{'Zero #':>6} | {'True γ':>10} | {'Detected γ':>12} | {'Error':>8} | {'Found?':>8}")
    print("-" * 55)
    
    found_count = 0
    for i, true_gamma in enumerate(RIEMANN_ZEROS[:20], 1):
        # Find the gamma_axis value nearest to true_gamma
        idx = np.argmin(np.abs(gamma_axis - true_gamma))
        
        # Check if there's a peak nearby (within 5 bins)
        local_region = power_cpu[max(0, idx-5):min(len(power_cpu), idx+5)]
        local_max_idx = np.argmax(local_region) + max(0, idx-5)
        detected_gamma = gamma_axis[local_max_idx]
        
        # Is this a significant peak?
        is_peak = power_cpu[local_max_idx] > np.percentile(power_cpu, 90)
        error = abs(detected_gamma - true_gamma)
        
        found = "✓" if is_peak and error < 2.0 else "✗"
        if is_peak and error < 2.0:
            found_count += 1
        
        print(f"{i:>6} | {true_gamma:>10.4f} | {detected_gamma:>12.4f} | {error:>8.3f} | {found:>8}")
    
    print(f"\nFound {found_count}/20 zeros (within error < 2.0)")
    
    return found_count


def compare_with_i_n_detection():
    """
    Compare zero detection with how I(n) detects primes
    """
    print("\n" + "=" * 70)
    print("COMPARISON: ZERO DETECTION vs PRIME DETECTION")
    print("=" * 70)
    
    print("""
    PRIME DETECTION via I(n):
    - I(p) > 0 for 100% of primes
    - Detection lift: 5.52x asymptotic
    - Works at any scale
    - No prior knowledge of primes needed
    
    ZERO DETECTION via ψ(x)-x FFT:
    - Peaks at γ_k frequencies
    - Requires large N for resolution
    - Zeros appear as spectral peaks
    - Known zeros validate the method
    
    KEY DIFFERENCE:
    - Primes are "local" (detectable at specific n)
    - Zeros are "global" (only visible in aggregate spectrum)
    
    This mirrors the relationship:
    - Primes = points on the number line
    - Zeros = frequencies of oscillation across the entire line
    """)
    
    # Can we make zeros more "local"?
    print("\n" + "-" * 60)
    print("ATTEMPT: Localizing zero signatures")
    print("-" * 60)
    
    N = 100000
    primes = sieve_primes(N)
    psi = compute_chebyshev_psi(N, primes)
    
    x = np.arange(1, N + 1)
    error = psi[1:] - x
    
    # Instead of global FFT, use short-time Fourier transform
    # to see how zeros "activate" at different scales
    
    from scipy.signal import stft
    
    log_x = np.log(x)
    # Resample uniformly in log space
    log_uniform = np.linspace(log_x[0], log_x[-1], 10000)
    error_log = np.interp(log_uniform, log_x, error)
    error_norm = error_log / np.sqrt(np.exp(log_uniform))
    
    # STFT with window that resolves first few zeros
    nperseg = 256
    f, t, Zxx = stft(error_norm, fs=1/(log_uniform[1]-log_uniform[0]), 
                     nperseg=nperseg, noverlap=nperseg//2)
    
    # Convert frequency to gamma
    gamma_stft = 2 * np.pi * f
    
    # Find where each known zero is strongest
    print(f"{'Zero':>6} | {'γ':>10} | {'Peak log(x)':>12} | {'Peak x':>12}")
    print("-" * 50)
    
    for i, gamma in enumerate(RIEMANN_ZEROS[:10], 1):
        # Find frequency bin closest to gamma
        freq_idx = np.argmin(np.abs(gamma_stft - gamma))
        
        # Find time bin where this frequency is strongest
        time_power = np.abs(Zxx[freq_idx, :])
        peak_time_idx = np.argmax(time_power)
        
        # Convert back to log(x) and x
        peak_log_x = log_uniform[0] + t[peak_time_idx]
        peak_x = np.exp(peak_log_x)
        
        print(f"{i:>6} | {gamma:>10.3f} | {peak_log_x:>12.3f} | {peak_x:>12.0f}")


def build_zero_detector():
    """
    Build a "zero detector" analogous to prime detection via I(n)
    
    For primes: I(n) > threshold → likely prime
    For zeros: ??? → likely zero at this γ
    """
    print("\n" + "=" * 70)
    print("BUILDING A ZERO DETECTOR")
    print("=" * 70)
    
    print("""
    Goal: Create a function Z(γ) where peaks indicate Riemann zeros,
    analogous to I(n) where positive values indicate primes.
    
    Approach: Accumulate evidence across different scales
    """)
    
    # Multi-scale analysis
    scales = [10000, 50000, 100000, 500000, 1000000]
    
    # Gamma range to search
    gamma_search = np.linspace(10, 105, 1000)  # Cover first 30 zeros
    
    # Accumulator for zero evidence
    Z = np.zeros(len(gamma_search))
    
    for N in scales:
        print(f"  Processing N = {N:,}...")
        
        primes = sieve_primes(N)
        psi = compute_chebyshev_psi(N, primes)
        
        x = np.arange(1, N + 1)
        error = psi[1:] - x
        
        # Sample in log space
        log_x = np.linspace(np.log(100), np.log(N), 5000)
        x_samples = np.exp(log_x)
        error_log = np.interp(x_samples, x, error)
        error_norm = error_log / np.sqrt(x_samples)
        
        # For each candidate gamma, compute correlation with cos(gamma * log_x)
        for i, gamma in enumerate(gamma_search):
            oscillation = np.cos(gamma * log_x)
            # Correlation magnitude
            corr = np.abs(np.corrcoef(error_norm, oscillation)[0, 1])
            Z[i] += corr
    
    # Normalize
    Z = Z / len(scales)
    
    # Find peaks in Z
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(Z, height=np.percentile(Z, 90), distance=5)
    
    print(f"\n{'Rank':>4} | {'γ detected':>12} | {'Z score':>10} | {'Nearest true γ':>14} | {'Error':>8}")
    print("-" * 60)
    
    sorted_peaks = sorted(peaks, key=lambda p: Z[p], reverse=True)
    
    found = 0
    for rank, peak in enumerate(sorted_peaks[:20], 1):
        gamma_det = gamma_search[peak]
        z_score = Z[peak]
        
        # Nearest true zero
        errors = [abs(gamma_det - g) for g in RIEMANN_ZEROS]
        nearest_idx = np.argmin(errors)
        nearest = RIEMANN_ZEROS[nearest_idx]
        err = errors[nearest_idx]
        
        match = "✓" if err < 1.5 else ""
        if err < 1.5:
            found += 1
        
        print(f"{rank:>4} | {gamma_det:>12.3f} | {z_score:>10.4f} | {nearest:>14.3f} | {err:>8.3f} {match}")
    
    print(f"\nDetector found {found}/20 peaks matching known zeros")
    
    return gamma_search, Z, peaks


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 14: MAKING RIEMANN ZEROS VISIBLE")
    print("=" * 70)
    print("""
    Just as I(n) makes primes visible in the stress field,
    can we find a transformation that makes zeros visible?
    
    Using ψ(x) - x in log space, zeros should appear as spectral peaks.
    """)
    
    # Main detection
    gamma_axis, power, matches = detect_zeros_in_psi()
    
    # Validation
    found_count = validate_zero_detection()
    
    # Comparison with prime detection
    compare_with_i_n_detection()
    
    # Build zero detector
    gamma_search, Z, peaks = build_zero_detector()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ZERO DETECTION RESULTS")
    print("=" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║ METHOD                            │ ZEROS FOUND  │ SUCCESS RATE     ║
╠══════════════════════════════════════════════════════════════════════╣
║ FFT of ψ(x)-x/√x in log space     │    {found_count:2d}/20     │   {found_count/20*100:5.1f}%         ║
║ Multi-scale Z(γ) detector         │    see above │   varies         ║
╚══════════════════════════════════════════════════════════════════════╝

KEY INSIGHT:

The zeros ARE visible when we use the right transformation:
1. Compute psi(x) = sum of log(p) for p^k <= x  (Chebyshev function)
2. Take error: psi(x) - x
3. Normalize by sqrt(x) to remove envelope
4. Transform to log(x) space
5. FFT -> peaks at gamma_1, gamma_2, ...

This is the analytic number theory approach, validated against known zeros.

The connection to our I(n) work:
- I(n) detects primes LOCALLY (one n at a time)
- Z(gamma) detects zeros GLOBALLY (via spectrum of entire sequence)
- Primes are the "atoms", zeros are the "resonances"
    """)
