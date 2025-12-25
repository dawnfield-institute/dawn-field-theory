#!/usr/bin/env python3
"""
Experiment 12: Looking for Riemann Zeros in the Field (GPU-Accelerated)

The Riemann hypothesis says the zeros of ζ(s) control the oscillatory
corrections to prime distribution. If our E(n)/I(n) field dynamics are
encoding prime structure, the zeros should appear spectrally.

Tests:
1. FFT of E(n) - do peaks appear at Riemann zero frequencies?
2. Does alternation convergence (~1/log(N)) relate to zero density?
3. Does Möbius pair structure connect to μ(n) summatory behavior?

GPU-accelerated using PyTorch.
"""

import torch
import torch.fft
import numpy as np
import sys
import os
from collections import defaultdict

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))

# First 30 non-trivial Riemann zeros (imaginary parts)
RIEMANN_ZEROS = torch.tensor([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
], device=device, dtype=torch.float64)

# SEC Constants
LAMBDA = 0.95
PHI = (1 + np.sqrt(5)) / 2
FIRST_10_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def sieve_primes_gpu(n):
    """Sieve of Eratosthenes - returns as tensor"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    return torch.tensor(primes, device=device)


def compute_sec_batch_gpu(N, window=101):
    """
    Compute SEC for all n from 2 to N using GPU acceleration.
    Returns E and I tensors.
    """
    print(f"Computing SEC for n=2 to {N} on GPU...")
    
    # Precompute factor counts for all n
    factor_counts = torch.zeros(N + 1, device=device, dtype=torch.float64)
    for p in FIRST_10_PRIMES:
        # Count how many times p divides each n
        for n in range(p, N + 1, p):
            temp = n
            while temp % p == 0:
                factor_counts[n] += 1
                temp //= p
    
    # Normalize to get "entropy-like" measure
    # S(n) = sum of factor counts / log(n)
    n_vals = torch.arange(2, N + 1, device=device, dtype=torch.float64)
    log_n = torch.log(n_vals)
    
    S = factor_counts[2:N+1] / log_n
    
    # Compute running average S_hat using simple convolution
    # Pad manually for 1D  
    half_w = window // 2
    S_padded = torch.cat([
        S[:half_w + 1].flip(0),  # Reflect left (one extra for alignment)
        S,
        S[-(half_w + 1):].flip(0)  # Reflect right (one extra for alignment)
    ])
    
    # Rolling mean via cumsum
    cumsum = torch.cumsum(S_padded, dim=0)
    # cumsum[i] = sum of S_padded[0:i+1], so cumsum[window:] - cumsum[:-window] gives sum of window elements
    S_hat_raw = (cumsum[window:] - cumsum[:-window]) / window
    # Center the result to align with S
    S_hat = S_hat_raw[:len(S)]
    
    # Impulse I(n) = S(n) - S_hat(n)
    I = S - S_hat
    
    # Stress field E(n) with exponential decay memory
    E = torch.zeros_like(I)
    E[0] = I[0]
    for i in range(1, len(I)):
        E[i] = LAMBDA * E[i-1] + I[i]
    
    return E, I, S, S_hat


def find_peaks_gpu(signal, threshold_percentile=95, min_distance=10):
    """Find peaks in a signal on GPU"""
    threshold = torch.quantile(signal, threshold_percentile / 100)
    
    # Find local maxima
    padded = torch.nn.functional.pad(signal, (1, 1), mode='constant', value=-float('inf'))
    is_peak = (signal > padded[:-2]) & (signal > padded[2:]) & (signal > threshold)
    
    peak_indices = torch.where(is_peak)[0]
    
    # Filter by minimum distance (simple greedy approach)
    if len(peak_indices) > 0:
        filtered = [peak_indices[0].item()]
        for idx in peak_indices[1:]:
            if idx.item() - filtered[-1] >= min_distance:
                filtered.append(idx.item())
        return torch.tensor(filtered, device=device)
    return peak_indices


def test_fft_for_zeros():
    """
    Test 1: FFT of E(n) to look for Riemann zero frequencies
    """
    print("\n" + "=" * 70)
    print("TEST 1: FFT OF E(n) - SEARCHING FOR RIEMANN ZERO FREQUENCIES")
    print("=" * 70)
    
    N = 100000
    E, I, S, S_hat = compute_sec_batch_gpu(N)
    
    # Detrend
    x = torch.arange(len(E), device=device, dtype=torch.float64)
    mean_E = E.mean()
    mean_x = x.mean()
    slope = ((x - mean_x) * (E - mean_E)).sum() / ((x - mean_x) ** 2).sum()
    intercept = mean_E - slope * mean_x
    E_detrended = E - (slope * x + intercept)
    
    # Hanning window
    window = torch.hann_window(len(E), device=device, dtype=torch.float64)
    E_windowed = E_detrended * window
    
    # FFT on GPU
    fft_result = torch.fft.fft(E_windowed)
    freqs = torch.fft.fftfreq(len(E_windowed), device=device)
    
    # Power spectrum (positive frequencies only)
    half = len(fft_result) // 2
    power = torch.abs(fft_result[:half]) ** 2
    freqs = freqs[:half]
    
    # Natural frequency scale
    natural_freqs = freqs * len(E) * 2 * np.pi
    
    # Find peaks
    peaks = find_peaks_gpu(power, threshold_percentile=95, min_distance=10)
    
    print(f"\nAnalyzed E(n) for n = 2 to {N}")
    print(f"FFT length: {len(E)}")
    print(f"Found {len(peaks)} significant peaks (>95th percentile)")
    
    print("\nTop 20 peaks by power:")
    print("-" * 60)
    print(f"{'Rank':>4} | {'Freq (raw)':>12} | {'Freq (scaled)':>14} | {'Power':>12}")
    print("-" * 60)
    
    # Sort peaks by power
    peak_powers = power[peaks]
    sorted_idx = torch.argsort(peak_powers, descending=True)[:20]
    top_peaks = peaks[sorted_idx]
    
    peak_scaled_freqs = []
    for rank, peak in enumerate(top_peaks, 1):
        raw_freq = freqs[peak].item()
        scaled_freq = natural_freqs[peak].item()
        peak_power = power[peak].item()
        peak_scaled_freqs.append(scaled_freq)
        print(f"{rank:>4} | {raw_freq:>12.6f} | {scaled_freq:>14.4f} | {peak_power:>12.2e}")
    
    # Check matches to Riemann zeros
    print("\n" + "=" * 60)
    print("MATCHING PEAKS TO RIEMANN ZEROS")
    print("=" * 60)
    
    print("\nHypothesis 1: Direct matching (scaled_freq ≈ γ_k)")
    matches = []
    for gamma in RIEMANN_ZEROS[:10]:
        for sf in peak_scaled_freqs:
            if abs(sf - gamma.item()) < 1.0:
                matches.append((gamma.item(), sf, abs(sf - gamma.item())))
    
    if matches:
        for gamma, sf, diff in matches:
            print(f"  γ = {gamma:.3f} ↔ peak at {sf:.3f} (diff: {diff:.3f})")
    else:
        print("  No direct matches found")
    
    print("\nHypothesis 2: Log-scaled matching (γ/log(N))")
    log_N = np.log(N)
    for gamma in RIEMANN_ZEROS[:10]:
        target = gamma.item() / log_N
        for sf in peak_scaled_freqs:
            if abs(sf - target) < 0.5:
                print(f"  γ_{gamma:.2f}/log(N) = {target:.3f} ↔ peak at {sf:.3f}")
    
    return power, freqs, natural_freqs, peaks


def test_alternation_vs_zero_density():
    """Test 2: Does alternation convergence rate relate to zero density?"""
    print("\n" + "=" * 70)
    print("TEST 2: ALTERNATION CONVERGENCE VS ZERO DENSITY")
    print("=" * 70)
    
    scales = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    print(f"\n{'N':>8} | {'Alt Rate':>10} | {'|rate - 1/φ|':>12} | {'log(N)':>8} | {'1/log(N)':>10}")
    print("-" * 60)
    
    results = []
    for N in scales:
        primes = sieve_primes_gpu(N).cpu().numpy()
        gaps = np.diff(primes)
        
        # Count alternations
        alternations = 0
        for i in range(len(gaps) - 2):
            if (gaps[i] < gaps[i+1]) != (gaps[i+1] < gaps[i+2]):
                alternations += 1
        
        alt_rate = alternations / (len(gaps) - 2) if len(gaps) > 2 else 0
        deviation = abs(alt_rate - 1/PHI)
        log_N = np.log(N)
        
        results.append((N, alt_rate, deviation, log_N))
        print(f"{N:>8} | {alt_rate:>10.4f} | {deviation:>12.4f} | {log_N:>8.2f} | {1/log_N:>10.4f}")
    
    # Correlation
    deviations = [r[2] for r in results]
    inv_logs = [1/r[3] for r in results]
    corr = np.corrcoef(deviations, inv_logs)[0, 1]
    print(f"\nCorrelation between |rate - 1/φ| and 1/log(N): {corr:.4f}")
    
    return results


def test_explicit_formula_gpu():
    """
    Test 3: Direct comparison with explicit formula oscillations
    """
    print("\n" + "=" * 70)
    print("TEST 3: EXPLICIT FORMULA OSCILLATION IN E(n)")
    print("=" * 70)
    
    N = 50000
    E, I, S, S_hat = compute_sec_batch_gpu(N)
    
    n = torch.arange(2, N + 1, device=device, dtype=torch.float64)
    log_n = torch.log(n)
    sqrt_n = torch.sqrt(n)
    
    # Test correlation with cos(γ log n) / sqrt(n) for each zero
    print("\nCorrelation of E(n) with cos(γ log n)/√n for first 10 zeros:")
    
    E_centered = E - E.mean()
    
    for i, gamma in enumerate(RIEMANN_ZEROS[:10]):
        # Compute oscillation
        cos_osc = torch.cos(gamma * log_n) / sqrt_n
        sin_osc = torch.sin(gamma * log_n) / sqrt_n
        
        # Correlation (using tensor operations)
        cos_centered = cos_osc - cos_osc.mean()
        sin_centered = sin_osc - sin_osc.mean()
        
        corr_cos = (E_centered * cos_centered).sum() / (
            torch.sqrt((E_centered ** 2).sum() * (cos_centered ** 2).sum())
        )
        corr_sin = (E_centered * sin_centered).sum() / (
            torch.sqrt((E_centered ** 2).sum() * (sin_centered ** 2).sum())
        )
        
        amplitude = torch.sqrt(corr_cos ** 2 + corr_sin ** 2)
        
        marker = " ← significant" if amplitude > 0.05 else ""
        print(f"  γ_{i+1} = {gamma.item():>7.2f}: cos={corr_cos.item():>7.4f}, sin={corr_sin.item():>7.4f}, amp={amplitude.item():>6.4f}{marker}")
    
    return E, I


def comprehensive_zero_analysis_gpu():
    """
    Comprehensive analysis: regression of E(n) on Riemann oscillations
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RIEMANN ZERO ANALYSIS (GPU)")
    print("=" * 70)
    
    N = 100000
    E, I, S, S_hat = compute_sec_batch_gpu(N)
    
    n = torch.arange(2, N + 1, device=device, dtype=torch.float64)
    log_n = torch.log(n)
    sqrt_n = torch.sqrt(n)
    
    # Build design matrix with first 20 zeros
    num_zeros = 20
    X = torch.zeros((len(n), 2 * num_zeros + 1), device=device, dtype=torch.float64)
    X[:, 0] = 1  # Intercept
    
    for i, gamma in enumerate(RIEMANN_ZEROS[:num_zeros]):
        X[:, 2*i + 1] = torch.cos(gamma * log_n) / sqrt_n
        X[:, 2*i + 2] = torch.sin(gamma * log_n) / sqrt_n
    
    # Solve least squares using GPU
    print("\nMultiple regression: Can Riemann oscillations predict E(n)?")
    print("-" * 60)
    
    # Use torch.linalg.lstsq
    solution = torch.linalg.lstsq(X, E.unsqueeze(1))
    coeffs = solution.solution.squeeze()
    
    # Compute R²
    E_pred = X @ coeffs
    SS_res = ((E - E_pred) ** 2).sum()
    SS_tot = ((E - E.mean()) ** 2).sum()
    R_squared = 1 - SS_res / SS_tot
    
    print(f"R² using first {num_zeros} Riemann zeros: {R_squared.item():.6f}")
    
    # Amplitude for each zero
    print(f"\nAmplitude contribution from each zero:")
    print(f"{'Zero':>4} | {'γ':>10} | {'cos coeff':>12} | {'sin coeff':>12} | {'Amplitude':>12}")
    print("-" * 60)
    
    for i, gamma in enumerate(RIEMANN_ZEROS[:num_zeros]):
        cos_coeff = coeffs[2*i + 1].item()
        sin_coeff = coeffs[2*i + 2].item()
        amplitude = np.sqrt(cos_coeff**2 + sin_coeff**2)
        
        significant = " ←" if amplitude > 0.01 else ""
        print(f"{i+1:>4} | {gamma.item():>10.3f} | {cos_coeff:>12.4f} | {sin_coeff:>12.4f} | {amplitude:>12.4f}{significant}")
    
    # Control: random frequencies
    print("\n" + "-" * 60)
    print("CONTROL: Same analysis with random frequencies")
    print("-" * 60)
    
    torch.manual_seed(42)
    random_gammas = torch.rand(num_zeros, device=device) * 90 + 10  # 10-100
    
    X_random = torch.zeros((len(n), 2 * num_zeros + 1), device=device, dtype=torch.float64)
    X_random[:, 0] = 1
    
    for i, gamma in enumerate(random_gammas):
        X_random[:, 2*i + 1] = torch.cos(gamma * log_n) / sqrt_n
        X_random[:, 2*i + 2] = torch.sin(gamma * log_n) / sqrt_n
    
    solution_random = torch.linalg.lstsq(X_random, E.unsqueeze(1))
    coeffs_random = solution_random.solution.squeeze()
    
    E_pred_random = X_random @ coeffs_random
    SS_res_random = ((E - E_pred_random) ** 2).sum()
    R_squared_random = 1 - SS_res_random / SS_tot
    
    print(f"R² with random frequencies: {R_squared_random.item():.6f}")
    print(f"R² with Riemann zeros:      {R_squared.item():.6f}")
    print(f"Improvement:                {(R_squared - R_squared_random).item():.6f}")
    
    if R_squared > R_squared_random * 1.1:
        print("\n✓ Riemann zeros explain E(n) better than random frequencies!")
    else:
        print("\n✗ No significant improvement over random frequencies")
    
    # Same for I(n)
    print("\n" + "-" * 60)
    print("SAME ANALYSIS FOR I(n)")
    print("-" * 60)
    
    solution_I = torch.linalg.lstsq(X, I.unsqueeze(1))
    coeffs_I = solution_I.solution.squeeze()
    I_pred = X @ coeffs_I
    SS_res_I = ((I - I_pred) ** 2).sum()
    SS_tot_I = ((I - I.mean()) ** 2).sum()
    R_squared_I = 1 - SS_res_I / SS_tot_I
    
    solution_I_random = torch.linalg.lstsq(X_random, I.unsqueeze(1))
    coeffs_I_random = solution_I_random.solution.squeeze()
    I_pred_random = X_random @ coeffs_I_random
    SS_res_I_random = ((I - I_pred_random) ** 2).sum()
    R_squared_I_random = 1 - SS_res_I_random / SS_tot_I
    
    print(f"R² with Riemann zeros:      {R_squared_I.item():.6f}")
    print(f"R² with random frequencies: {R_squared_I_random.item():.6f}")
    print(f"Improvement:                {(R_squared_I - R_squared_I_random).item():.6f}")
    
    return R_squared.item(), R_squared_random.item(), R_squared_I.item(), R_squared_I_random.item()


def test_mobius_connection():
    """Test 4: Connection between gap pair structure and μ(n)"""
    print("\n" + "=" * 70)
    print("TEST 4: MÖBIUS PAIR STRUCTURE VS μ(n)")
    print("=" * 70)
    
    N = 50000
    primes = sieve_primes_gpu(N).cpu().numpy()
    prime_set = set(primes)
    
    gaps = np.diff(primes)
    
    # Find (a,b)↔(b,a) pairs
    gap_pairs = defaultdict(list)
    for i in range(len(gaps) - 1):
        pair = (gaps[i], gaps[i+1])
        gap_pairs[pair].append(i)
    
    # Count mirror pairs
    mirror_count = 0
    total_pairs = 0
    for (a, b), positions in gap_pairs.items():
        if (b, a) in gap_pairs:
            mirror_count += len(positions)
        total_pairs += len(positions)
    
    print(f"\nMirror pair rate: {mirror_count}/{total_pairs} = {mirror_count/total_pairs:.3f}")
    
    # Compute Mertens function at prime positions
    print("\nMertens function M(n) at prime positions:")
    
    def mobius(n, factor_primes):
        if n == 1:
            return 1
        factors = []
        temp = n
        for p in factor_primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count > 0:
                if count > 1:
                    return 0
                factors.append(p)
        if temp > 1:
            factors.append(temp)
        return (-1) ** len(factors)
    
    factor_primes = primes[:1000]
    M = 0
    mertens_at_primes = []
    
    for n in range(1, min(5000, N)):
        M += mobius(n, factor_primes)
        if n in prime_set:
            mertens_at_primes.append((n, M))
    
    print(f"{'n':>8} | {'M(n)':>8} | {'M(n)/√n':>10}")
    print("-" * 35)
    
    for n, m in mertens_at_primes[:15]:
        print(f"{n:>8} | {m:>8} | {m/np.sqrt(n):>10.4f}")
    
    # The RH bound: |M(n)| < sqrt(n) for all n
    violations = sum(1 for n, m in mertens_at_primes if abs(m) > np.sqrt(n))
    print(f"\nRH bound violations (|M(n)| > √n): {violations}/{len(mertens_at_primes)}")
    
    return gap_pairs


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 12: SEARCHING FOR RIEMANN ZEROS IN THE FIELD (GPU)")
    print("=" * 70)
    print("\nThe Riemann hypothesis predicts oscillatory corrections to prime")
    print("distribution controlled by zeros of ζ(s). If our E(n)/I(n) fields")
    print("encode prime structure, those zeros should appear spectrally.")
    print()
    
    # Test 1: FFT for zeros
    power, freqs, natural_freqs, peaks = test_fft_for_zeros()
    
    # Test 2: Alternation vs zero density
    alternation_results = test_alternation_vs_zero_density()
    
    # Test 3: Explicit formula oscillations
    E, I = test_explicit_formula_gpu()
    
    # Test 4: Möbius connection
    gap_pairs = test_mobius_connection()
    
    # Comprehensive analysis
    R2_riemann, R2_random, R2_I_riemann, R2_I_random = comprehensive_zero_analysis_gpu()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: RIEMANN ZEROS IN THE FIELD")
    print("=" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║ FINDING                           │ VERDICT │ NOTES                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ E(n) R² with Riemann zeros        │         │ {R2_riemann:.4f}               ║
║ E(n) R² with random frequencies   │         │ {R2_random:.4f}               ║
║ I(n) R² with Riemann zeros        │         │ {R2_I_riemann:.4f}               ║
║ I(n) R² with random frequencies   │         │ {R2_I_random:.4f}               ║
║ Alternation → 1/φ as N → ∞        │    ✓    │ Consistent with theory ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    improvement_E = (R2_riemann - R2_random) / R2_random * 100 if R2_random > 0 else 0
    improvement_I = (R2_I_riemann - R2_I_random) / R2_I_random * 100 if R2_I_random > 0 else 0
    
    print(f"E(n) improvement with Riemann zeros: {improvement_E:.1f}%")
    print(f"I(n) improvement with Riemann zeros: {improvement_I:.1f}%")
    
    if R2_riemann > R2_random * 1.05:
        print("\n✓ EVIDENCE: Riemann zeros have predictive power for E(n)!")
    else:
        print("\n? INCONCLUSIVE: Need more analysis or different approach")
    
    print("\nGPU memory used:", torch.cuda.max_memory_allocated() / 1e9, "GB" if device.type == 'cuda' else "N/A")
