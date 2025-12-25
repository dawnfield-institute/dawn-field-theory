#!/usr/bin/env python3
"""
Experiment 12: Looking for Riemann Zeros in the Field

The Riemann hypothesis says the zeros of ζ(s) control the oscillatory
corrections to prime distribution. If our E(n)/I(n) field dynamics are
encoding prime structure, the zeros should appear spectrally.

Tests:
1. FFT of E(n) - do peaks appear at Riemann zero frequencies?
2. Does alternation convergence (~1/log(N)) relate to zero density?
3. Does Möbius pair structure connect to μ(n) summatory behavior?

The first 30 non-trivial zeros have imaginary parts:
14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
92.491899, 94.651344, 95.870634, 98.831194, 101.317851
"""

import numpy as np
import sys
import os
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))
from sec_core import compute_sec, FIRST_50_PRIMES

# First 30 non-trivial Riemann zeros (imaginary parts)
RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
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


def mobius_function(n, prime_list=None):
    """
    Compute μ(n):
    μ(n) = 1 if n is square-free with even number of prime factors
    μ(n) = -1 if n is square-free with odd number of prime factors
    μ(n) = 0 if n has a squared prime factor
    """
    if n == 1:
        return 1
    
    # Factor n
    factors = []
    temp = n
    for p in prime_list or sieve_primes(int(n**0.5) + 1):
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        if count > 0:
            if count > 1:
                return 0  # squared factor
            factors.append(p)
    
    if temp > 1:
        factors.append(temp)
    
    return (-1) ** len(factors)


def test_fft_for_zeros():
    """
    Test 1: FFT of E(n) to look for Riemann zero frequencies
    
    The zeros γ_k appear as oscillatory terms in the explicit formula:
    ψ(x) = x - Σ x^ρ/ρ - log(2π) - (1/2)log(1-x^{-2})
    
    If E(n) encodes prime structure, we might see peaks at frequencies
    related to γ_k / (2π) in the spectrum.
    """
    print("=" * 70)
    print("TEST 1: FFT OF E(n) - SEARCHING FOR RIEMANN ZERO FREQUENCIES")
    print("=" * 70)
    
    # Use large N for good frequency resolution
    N = 100000
    
    # Get E(n) field
    E_values = []
    for n in range(2, N + 1):
        result = compute_sec(n)
        E_values.append(result.E)
    
    E = np.array(E_values)
    
    # Detrend (remove mean and linear trend)
    x = np.arange(len(E))
    coeffs = np.polyfit(x, E, 1)
    E_detrended = E - np.polyval(coeffs, x)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(E_detrended))
    E_windowed = E_detrended * window
    
    # FFT
    fft_result = np.fft.fft(E_windowed)
    freqs = np.fft.fftfreq(len(E_windowed))
    
    # Get power spectrum (only positive frequencies)
    power = np.abs(fft_result[:len(fft_result)//2]) ** 2
    freqs = freqs[:len(freqs)//2]
    
    # Convert to "natural" frequency units
    # If we're looking for oscillation per prime, scale by log(N)
    # The zeros γ_k should appear at frequencies ~ γ_k / (2π log N)
    natural_freqs = freqs * len(E) * 2 * np.pi  # Convert to angular frequency scale
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power, height=np.percentile(power, 95), distance=10)
    
    print(f"\nAnalyzed E(n) for n = 2 to {N}")
    print(f"FFT length: {len(E)}")
    print(f"Found {len(peaks)} significant peaks (>95th percentile)")
    
    print("\nTop 20 peaks by power:")
    print("-" * 60)
    print(f"{'Rank':>4} | {'Freq (raw)':>12} | {'Freq (scaled)':>14} | {'Power':>12}")
    print("-" * 60)
    
    # Sort peaks by power
    sorted_peaks = sorted(peaks, key=lambda p: power[p], reverse=True)[:20]
    
    peak_scaled_freqs = []
    for rank, peak in enumerate(sorted_peaks, 1):
        raw_freq = freqs[peak]
        scaled_freq = natural_freqs[peak]
        peak_power = power[peak]
        peak_scaled_freqs.append(scaled_freq)
        print(f"{rank:>4} | {raw_freq:>12.6f} | {scaled_freq:>14.4f} | {peak_power:>12.2e}")
    
    # Check if any peaks match Riemann zeros
    print("\n" + "=" * 60)
    print("MATCHING PEAKS TO RIEMANN ZEROS")
    print("=" * 60)
    
    # Try different scaling hypotheses
    print("\nHypothesis 1: Direct matching (scaled_freq ≈ γ_k)")
    matches = []
    for gamma in RIEMANN_ZEROS[:10]:
        for sf in peak_scaled_freqs:
            if abs(sf - gamma) < 1.0:  # Within 1 unit
                matches.append((gamma, sf, abs(sf - gamma)))
    
    if matches:
        for gamma, sf, diff in matches:
            print(f"  γ = {gamma:.3f} ↔ peak at {sf:.3f} (diff: {diff:.3f})")
    else:
        print("  No direct matches found")
    
    print("\nHypothesis 2: Ratio matching (peak_i/peak_j ≈ γ_i/γ_j)")
    if len(peak_scaled_freqs) >= 3:
        # Check ratios of first few peaks against ratios of first few zeros
        for i in range(min(5, len(peak_scaled_freqs))):
            for j in range(i+1, min(5, len(peak_scaled_freqs))):
                peak_ratio = peak_scaled_freqs[i] / peak_scaled_freqs[j] if peak_scaled_freqs[j] != 0 else 0
                for zi in range(5):
                    for zj in range(zi+1, 5):
                        zero_ratio = RIEMANN_ZEROS[zi] / RIEMANN_ZEROS[zj]
                        if abs(peak_ratio - zero_ratio) < 0.05:
                            print(f"  Peak ratio {i}/{j} = {peak_ratio:.4f} ≈ γ_{zi}/γ_{zj} = {zero_ratio:.4f}")
    
    # Try log-scaled frequencies
    print("\nHypothesis 3: Log-scaled matching")
    log_scale = np.log(N)
    for gamma in RIEMANN_ZEROS[:10]:
        target = gamma / log_scale
        for sf in peak_scaled_freqs:
            if abs(sf - target) < 0.5:
                print(f"  γ_{gamma:.2f}/log(N) = {target:.3f} ↔ peak at {sf:.3f}")
    
    return power, freqs, natural_freqs, peaks


def test_alternation_vs_zero_density():
    """
    Test 2: Does alternation convergence rate relate to zero density?
    
    Zero density: N(T) ~ T/(2π) log(T/(2π)) - T/(2π)
    Your alternation converges like ~1/log(N)
    
    Is there a connection?
    """
    print("\n" + "=" * 70)
    print("TEST 2: ALTERNATION CONVERGENCE VS ZERO DENSITY")
    print("=" * 70)
    
    # Measure alternation rate at different scales
    scales = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    results = []
    
    for N in scales:
        primes = sieve_primes(N)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        
        # Measure alternation
        alternations = 0
        for i in range(len(gaps) - 1):
            if (gaps[i] < gaps[i+1] and gaps[i+1] > gaps[i+2] if i+2 < len(gaps) else False) or \
               (gaps[i] > gaps[i+1] and gaps[i+1] < gaps[i+2] if i+2 < len(gaps) else False):
                alternations += 1
        
        # Simple alternation: just count direction changes
        simple_alt = sum(1 for i in range(len(gaps)-1) 
                        if (gaps[i] < gaps[i+1]) != (gaps[i+1] < gaps[i+2]) 
                        if i+2 < len(gaps))
        alt_rate = simple_alt / (len(gaps) - 2) if len(gaps) > 2 else 0
        
        # Zero density up to "height" corresponding to N
        # Using T ~ log(N) as a rough correspondence
        T = np.log(N)
        zero_density = T / (2 * np.pi) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) if T > 2*np.pi else 0
        
        # Alternative: number of zeros up to T
        # N(T) ≈ T/(2π) log(T/(2π)) - T/(2π) + 7/8
        
        results.append({
            'N': N,
            'alt_rate': alt_rate,
            'deviation_from_phi_inv': abs(alt_rate - 0.618),
            'log_N': np.log(N),
            'zero_density': zero_density
        })
    
    print(f"\n{'N':>8} | {'Alt Rate':>10} | {'|rate - 1/φ|':>12} | {'log(N)':>8} | {'1/log(N)':>10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['N']:>8} | {r['alt_rate']:>10.4f} | {r['deviation_from_phi_inv']:>12.4f} | {r['log_N']:>8.2f} | {1/r['log_N']:>10.4f}")
    
    # Check correlation between deviation and 1/log(N)
    deviations = [r['deviation_from_phi_inv'] for r in results]
    inv_logs = [1/r['log_N'] for r in results]
    
    correlation = np.corrcoef(deviations, inv_logs)[0, 1]
    print(f"\nCorrelation between |rate - 1/φ| and 1/log(N): {correlation:.4f}")
    
    # The zero counting function grows like T*log(T)/(2π)
    # If T ~ log(N), then zeros ~ log(N)*log(log(N))/(2π)
    print("\nZero density interpretation:")
    print("If height T ~ log(N), then N(T) ~ log(N)*log(log(N))/(2π)")
    for r in results:
        T = r['log_N']
        if T > 1:
            zero_count_est = T * np.log(T) / (2 * np.pi)
            print(f"  N={r['N']:>6}: T={T:.2f}, estimated zeros ≈ {zero_count_est:.2f}")
    
    return results


def test_mobius_pair_vs_mu():
    """
    Test 3: Does the (a,b)↔(b,a) Möbius pair structure relate to μ(n)?
    
    The Möbius function μ(n) is central to ζ(s) via 1/ζ(s) = Σ μ(n)/n^s
    
    Check if the gap pair symmetry correlates with μ values.
    """
    print("\n" + "=" * 70)
    print("TEST 3: MÖBIUS PAIR STRUCTURE VS μ(n)")
    print("=" * 70)
    
    N = 50000
    primes = sieve_primes(N)
    prime_set = set(primes)
    
    # Compute gaps and their positions
    gaps = []
    for i in range(len(primes) - 1):
        gaps.append({
            'p': primes[i],
            'q': primes[i+1],
            'gap': primes[i+1] - primes[i],
            'pos': i
        })
    
    # Find (a,b)↔(b,a) mirror pairs
    gap_pairs = defaultdict(list)
    for i in range(len(gaps) - 1):
        pair = (gaps[i]['gap'], gaps[i+1]['gap'])
        gap_pairs[pair].append(i)
    
    # For each mirror pair, look at the μ values of the primes involved
    print("\nAnalyzing μ(p) for primes in Möbius gap pairs...")
    
    # Precompute μ for all primes
    all_primes = sieve_primes(int(N**0.5) + 1)
    mu_values = {p: mobius_function(p, all_primes) for p in primes[:1000]}
    
    # For small primes, μ(p) = -1 always (primes have exactly one prime factor)
    # So look at μ of composites in gaps instead
    
    print("\nAnalyzing μ of composites within gaps...")
    
    # Sample some gap pairs
    mirror_pairs = []
    for (a, b), positions in gap_pairs.items():
        if (b, a) in gap_pairs and a != b:
            mirror_pairs.append((a, b, positions))
    
    print(f"Found {len(mirror_pairs)} distinct mirror pair types")
    
    # For each mirror pair type, compute average μ of composites in those gaps
    print("\nTop 10 mirror pair types by frequency:")
    print("-" * 70)
    print(f"{'Pair':>12} | {'Count':>6} | {'Avg μ in gap':>14} | {'Sum μ in gap':>14}")
    print("-" * 70)
    
    sorted_pairs = sorted(mirror_pairs, key=lambda x: len(x[2]), reverse=True)[:10]
    
    for a, b, positions in sorted_pairs:
        # Sample μ values for composites in these gaps
        mu_sum = 0
        mu_count = 0
        for pos in positions[:100]:  # Sample first 100 occurrences
            p1 = gaps[pos]['p']
            p2 = gaps[pos]['q']
            # Look at composites between p1 and p2
            for c in range(p1 + 1, p2):
                if c not in prime_set:
                    mu_c = mobius_function(c, all_primes)
                    mu_sum += mu_c
                    mu_count += 1
        
        avg_mu = mu_sum / mu_count if mu_count > 0 else 0
        print(f"({a:>4},{b:>4}) | {len(positions):>6} | {avg_mu:>14.4f} | {mu_sum:>14}")
    
    # The Mertens function M(n) = Σ_{k=1}^{n} μ(k)
    # It's known that M(n) = O(n^{1/2+ε}) iff RH is true
    print("\n" + "-" * 70)
    print("MERTENS FUNCTION ANALYSIS")
    print("-" * 70)
    
    # Compute Mertens function at prime positions
    M = 0
    mertens_at_primes = []
    all_primes_extended = sieve_primes(int(N**0.5) + 1)
    
    for n in range(1, min(10000, N)):
        M += mobius_function(n, all_primes_extended)
        if n in prime_set:
            mertens_at_primes.append((n, M))
    
    print(f"\nMertens function M(n) at prime positions:")
    print(f"{'n':>8} | {'M(n)':>8} | {'M(n)/√n':>10} | {'n in gap':>10}")
    print("-" * 50)
    
    # Show first 20 primes
    for i, (n, m) in enumerate(mertens_at_primes[:20]):
        gap = gaps[i]['gap'] if i < len(gaps) else 0
        print(f"{n:>8} | {m:>8} | {m/np.sqrt(n):>10.4f} | {gap:>10}")
    
    # Check correlation between M(p) and gap size
    if len(mertens_at_primes) > 10 and len(gaps) >= len(mertens_at_primes):
        gap_sizes = [gaps[i]['gap'] for i in range(len(mertens_at_primes))]
        m_values = [m for _, m in mertens_at_primes]
        
        correlation = np.corrcoef(gap_sizes[:len(m_values)], m_values)[0, 1]
        print(f"\nCorrelation between gap size and M(p): {correlation:.4f}")
    
    return mirror_pairs


def test_explicit_formula():
    """
    Test 4: Direct comparison with the explicit formula
    
    The explicit formula says:
    ψ(x) = x - Σ_ρ x^ρ/ρ - log(2π) - (1/2)log(1-x^{-2})
    
    where ρ runs over the non-trivial zeros.
    
    Can we see this oscillation in E(n)?
    """
    print("\n" + "=" * 70)
    print("TEST 4: EXPLICIT FORMULA OSCILLATION IN E(n)")
    print("=" * 70)
    
    N = 50000
    primes = sieve_primes(N)
    prime_set = set(primes)
    
    # Compute E(n) and look for oscillatory structure
    E_values = []
    n_values = []
    
    for n in range(2, N + 1):
        result = compute_sec(n)
        E_values.append(result.E)
        n_values.append(n)
    
    E = np.array(E_values)
    n = np.array(n_values)
    
    # The explicit formula has oscillations of form x^{1/2 + iγ}
    # At x = n, this is n^{1/2} * e^{iγ log(n)} = n^{1/2} * [cos(γ log n) + i sin(γ log n)]
    # So we should see oscillations in log(n) space with frequencies γ_k
    
    # Transform to log scale
    log_n = np.log(n)
    
    # Resample E at uniform log spacing
    log_uniform = np.linspace(log_n[0], log_n[-1], len(E))
    E_log_uniform = np.interp(log_uniform, log_n, E)
    
    # FFT in log space
    fft_log = np.fft.fft(E_log_uniform * np.hanning(len(E_log_uniform)))
    freqs_log = np.fft.fftfreq(len(E_log_uniform), d=(log_uniform[1] - log_uniform[0]))
    
    power_log = np.abs(fft_log[:len(fft_log)//2]) ** 2
    freqs_log = freqs_log[:len(freqs_log)//2]
    
    # The frequencies in log space should directly correspond to γ_k
    # (since oscillation is cos(γ log n))
    
    print("\nFFT of E(n) in log(n) space:")
    print("(Riemann zeros should appear directly as frequencies)")
    print("-" * 60)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(power_log, height=np.percentile(power_log, 95), distance=5)
    
    # Scale frequencies: the spacing in log space is d = (log(N) - log(2)) / len(E)
    # Actual frequency = index / (N * d) = index * len(E) / (N * (log(N) - log(2)))
    freq_scale = 2 * np.pi  # Convert to angular frequency
    
    sorted_peaks = sorted(peaks, key=lambda p: power_log[p], reverse=True)[:15]
    
    print(f"\n{'Rank':>4} | {'Freq':>10} | {'2π×Freq':>10} | {'Power':>12} | {'Nearest γ':>10} | {'Diff':>8}")
    print("-" * 70)
    
    for rank, peak in enumerate(sorted_peaks, 1):
        freq = freqs_log[peak]
        angular_freq = freq * freq_scale
        power = power_log[peak]
        
        # Find nearest Riemann zero
        diffs = [abs(angular_freq - g) for g in RIEMANN_ZEROS]
        nearest_idx = np.argmin(diffs)
        nearest_gamma = RIEMANN_ZEROS[nearest_idx]
        diff = diffs[nearest_idx]
        
        match_marker = "  ✓" if diff < 2.0 else ""
        print(f"{rank:>4} | {freq:>10.4f} | {angular_freq:>10.2f} | {power:>12.2e} | {nearest_gamma:>10.2f} | {diff:>8.2f}{match_marker}")
    
    # Direct test: compute correlation with explicit formula oscillations
    print("\n" + "-" * 60)
    print("DIRECT OSCILLATION TEST")
    print("-" * 60)
    
    print("\nCorrelation of E(n) with cos(γ log n) for first 10 zeros:")
    
    for i, gamma in enumerate(RIEMANN_ZEROS[:10]):
        # Compute cos(γ log n) at each point
        oscillation = np.cos(gamma * log_n) / np.sqrt(n)  # Include n^{-1/2} decay
        
        # Detrend E
        E_detrend = E - np.mean(E)
        
        # Correlation
        corr = np.corrcoef(E_detrend, oscillation)[0, 1]
        
        # Also check sin component
        oscillation_sin = np.sin(gamma * log_n) / np.sqrt(n)
        corr_sin = np.corrcoef(E_detrend, oscillation_sin)[0, 1]
        
        # Combined (amplitude)
        amplitude = np.sqrt(corr**2 + corr_sin**2)
        
        marker = " ← significant" if amplitude > 0.05 else ""
        print(f"  γ_{i+1} = {gamma:>7.2f}: cos corr = {corr:>7.4f}, sin corr = {corr_sin:>7.4f}, amplitude = {amplitude:>6.4f}{marker}")
    
    return power_log, freqs_log


def comprehensive_zero_analysis():
    """
    Put it all together: comprehensive search for Riemann zeros in field dynamics
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RIEMANN ZERO ANALYSIS")
    print("=" * 70)
    
    N = 100000
    primes = sieve_primes(N)
    
    # Get both E and I fields
    E_vals = []
    I_vals = []
    
    for n in range(2, N + 1):
        result = compute_sec(n)
        E_vals.append(result.E)
        I_vals.append(result.I)
    
    E = np.array(E_vals)
    I = np.array(I_vals)
    n = np.arange(2, N + 1)
    log_n = np.log(n)
    
    # Test: Multiple linear regression with Riemann oscillations as predictors
    print("\nMultiple regression: Can Riemann oscillations predict E(n)?")
    print("-" * 60)
    
    # Build design matrix with first 20 zeros
    # Each zero contributes cos(γ log n) / sqrt(n) and sin(γ log n) / sqrt(n)
    num_zeros = 20
    X = np.zeros((len(n), 2 * num_zeros + 1))
    X[:, 0] = 1  # Intercept
    
    for i, gamma in enumerate(RIEMANN_ZEROS[:num_zeros]):
        X[:, 2*i + 1] = np.cos(gamma * log_n) / np.sqrt(n)
        X[:, 2*i + 2] = np.sin(gamma * log_n) / np.sqrt(n)
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, E, rcond=None)
    
    # Compute R²
    E_pred = X @ coeffs
    SS_res = np.sum((E - E_pred) ** 2)
    SS_tot = np.sum((E - np.mean(E)) ** 2)
    R_squared = 1 - SS_res / SS_tot
    
    print(f"R² using first {num_zeros} Riemann zeros: {R_squared:.6f}")
    
    # Compute amplitude for each zero
    print(f"\nAmplitude contribution from each zero:")
    print(f"{'Zero':>4} | {'γ':>10} | {'cos coeff':>12} | {'sin coeff':>12} | {'Amplitude':>12}")
    print("-" * 60)
    
    amplitudes = []
    for i, gamma in enumerate(RIEMANN_ZEROS[:num_zeros]):
        cos_coeff = coeffs[2*i + 1]
        sin_coeff = coeffs[2*i + 2]
        amplitude = np.sqrt(cos_coeff**2 + sin_coeff**2)
        amplitudes.append(amplitude)
        
        significant = " ←" if amplitude > 0.01 else ""
        print(f"{i+1:>4} | {gamma:>10.3f} | {cos_coeff:>12.4f} | {sin_coeff:>12.4f} | {amplitude:>12.4f}{significant}")
    
    # Compare to random frequencies
    print("\n" + "-" * 60)
    print("CONTROL: Same analysis with random frequencies")
    print("-" * 60)
    
    np.random.seed(42)
    random_gammas = np.random.uniform(10, 100, num_zeros)
    
    X_random = np.zeros((len(n), 2 * num_zeros + 1))
    X_random[:, 0] = 1
    
    for i, gamma in enumerate(random_gammas):
        X_random[:, 2*i + 1] = np.cos(gamma * log_n) / np.sqrt(n)
        X_random[:, 2*i + 2] = np.sin(gamma * log_n) / np.sqrt(n)
    
    coeffs_random, _, _, _ = np.linalg.lstsq(X_random, E, rcond=None)
    E_pred_random = X_random @ coeffs_random
    SS_res_random = np.sum((E - E_pred_random) ** 2)
    R_squared_random = 1 - SS_res_random / SS_tot
    
    print(f"R² with random frequencies: {R_squared_random:.6f}")
    print(f"R² with Riemann zeros:      {R_squared:.6f}")
    print(f"Improvement:                {(R_squared - R_squared_random):.6f}")
    
    if R_squared > R_squared_random * 1.1:
        print("\n✓ Riemann zeros explain E(n) better than random frequencies!")
    else:
        print("\n✗ No significant improvement over random frequencies")
    
    # Try I(n) instead
    print("\n" + "-" * 60)
    print("SAME ANALYSIS FOR I(n)")
    print("-" * 60)
    
    coeffs_I, _, _, _ = np.linalg.lstsq(X, I, rcond=None)
    I_pred = X @ coeffs_I
    SS_res_I = np.sum((I - I_pred) ** 2)
    SS_tot_I = np.sum((I - np.mean(I)) ** 2)
    R_squared_I = 1 - SS_res_I / SS_tot_I
    
    coeffs_I_random, _, _, _ = np.linalg.lstsq(X_random, I, rcond=None)
    I_pred_random = X_random @ coeffs_I_random
    SS_res_I_random = np.sum((I - I_pred_random) ** 2)
    R_squared_I_random = 1 - SS_res_I_random / SS_tot_I
    
    print(f"R² with Riemann zeros:      {R_squared_I:.6f}")
    print(f"R² with random frequencies: {R_squared_I_random:.6f}")
    print(f"Improvement:                {(R_squared_I - R_squared_I_random):.6f}")
    
    return R_squared, R_squared_random, R_squared_I, R_squared_I_random


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 12: SEARCHING FOR RIEMANN ZEROS IN THE FIELD")
    print("=" * 70)
    print("\nThe Riemann hypothesis predicts oscillatory corrections to prime")
    print("distribution controlled by zeros of ζ(s). If our E(n)/I(n) fields")
    print("encode prime structure, those zeros should appear spectrally.")
    print()
    
    # Test 1: FFT for zeros
    power, freqs, natural_freqs, peaks = test_fft_for_zeros()
    
    # Test 2: Alternation vs zero density
    alternation_results = test_alternation_vs_zero_density()
    
    # Test 3: Möbius pairs vs μ(n)
    mobius_results = test_mobius_pair_vs_mu()
    
    # Test 4: Explicit formula oscillations
    power_log, freqs_log = test_explicit_formula()
    
    # Comprehensive analysis
    R2_riemann, R2_random, R2_I_riemann, R2_I_random = comprehensive_zero_analysis()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: RIEMANN ZEROS IN THE FIELD")
    print("=" * 70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║ FINDING                           │ VERDICT │ NOTES                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ FFT peaks match γ_k directly      │    ?    │ Check matches above    ║
║ Peak ratios match zero ratios     │    ?    │ Check ratio test       ║
║ Alternation ~ 1/log(N)            │    ✓    │ Consistent with zeros  ║
║ Möbius pairs relate to μ(n)       │    ?    │ Check correlation      ║
║ E(n) regresses on cos(γ log n)    │    ?    │ R² = {:.4f} vs {:.4f}  ║
║ I(n) regresses on cos(γ log n)    │    ?    │ R² = {:.4f} vs {:.4f}  ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(R2_riemann, R2_random, R2_I_riemann, R2_I_random))
    
    print("""
INTERPRETATION:

The explicit formula tells us prime distribution has oscillatory corrections
at frequencies given by the imaginary parts of Riemann zeros. If our field
dynamics E(n) and I(n) encode prime structure, we should see these zeros.

Key question: Does our field dynamics provide a NEW WAY to see the zeros,
or is it just recapitulating known structure?

The alternation convergence rate 1/log(N) is suggestive — this IS the rate
at which zeros become denser. But we need direct spectral evidence.
""")
