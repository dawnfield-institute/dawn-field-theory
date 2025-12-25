"""
Oscillation Engine — Zero-Crossing Detection and Attractor Dynamics
====================================================================

Builds on SEC core to test the hypothesis that primes represent
zero-crossings in an oscillatory system collapsing toward attractors.

Key components:
- detect_zero_crossings: Find sign changes in stress field E(n)
- oscillation_envelope: Extract amplitude envelope via Hilbert transform
- damping_rate: Measure decay rate of oscillation amplitude
- crossing_prime_correlation: Test if crossings correlate with primes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import json
from datetime import datetime
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sec_prime_manifold', 'core'))
from sec_core import compute_sec, SECResult, PHI, FIRST_50_PRIMES, create_trace, get_timestamp


# ============================================================================
# CONSTANTS
# ============================================================================

XI = 1.0571428571428572  # Balance operator from CA experiments
PI_SQUARED_INV = 1 / (np.pi ** 2)  # 0.10132... - decay rate from PHM


# ============================================================================
# RESULT CONTAINERS
# ============================================================================

@dataclass
class ZeroCrossingResult:
    """Container for zero-crossing analysis."""
    crossing_indices: np.ndarray      # Positions where E(n) crosses zero
    crossing_directions: np.ndarray   # +1 for positive-going, -1 for negative-going
    prime_crossings: int              # Count of crossings at or near primes
    total_crossings: int              # Total crossing count
    crossing_fraction: float          # Fraction of crossings near primes
    expected_fraction: float          # Expected fraction by chance
    enrichment: float                 # Observed / Expected
    p_value: float                    # Statistical significance


@dataclass
class OscillationResult:
    """Container for oscillation characterization."""
    envelope: np.ndarray              # Amplitude envelope
    instantaneous_freq: np.ndarray    # Local frequency
    damping_rate: float               # Decay exponent
    damping_rate_error: float         # Fit uncertainty
    characteristic_freq: float        # Dominant frequency
    r_squared: float                  # Fit quality


# ============================================================================
# ZERO-CROSSING DETECTION
# ============================================================================

def detect_zero_crossings(E: np.ndarray, 
                          start_idx: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect indices where E(n) crosses zero.
    
    Args:
        E: Stress field array
        start_idx: Skip initial transient
        
    Returns:
        crossing_indices: Array of crossing positions
        crossing_directions: +1 for positive-going, -1 for negative-going
    """
    E_valid = E[start_idx:]
    
    # Find sign changes
    signs = np.sign(E_valid)
    # Handle exact zeros: treat as positive for continuity
    signs[signs == 0] = 1
    
    # Detect transitions
    sign_changes = np.diff(signs)
    crossing_mask = sign_changes != 0
    
    # Get indices (adjusted for start_idx offset)
    crossing_indices = np.where(crossing_mask)[0] + start_idx + 1
    
    # Determine direction: +2 = negative→positive, -2 = positive→negative
    crossing_directions = (sign_changes[crossing_mask] / 2).astype(int)
    
    return crossing_indices, crossing_directions


def crossing_prime_correlation(crossing_indices: np.ndarray,
                                prime_mask: np.ndarray,
                                proximity: int = 1) -> ZeroCrossingResult:
    """
    Test if zero-crossings are correlated with prime positions.
    
    Args:
        crossing_indices: Positions of zero-crossings
        prime_mask: Boolean array where True = prime
        proximity: How close crossing must be to prime (±proximity)
        
    Returns:
        ZeroCrossingResult with statistics
    """
    n_max = len(prime_mask) - 1
    total_crossings = len(crossing_indices)
    
    if total_crossings == 0:
        return ZeroCrossingResult(
            crossing_indices=crossing_indices,
            crossing_directions=np.array([]),
            prime_crossings=0,
            total_crossings=0,
            crossing_fraction=0.0,
            expected_fraction=0.0,
            enrichment=0.0,
            p_value=1.0
        )
    
    # Count crossings near primes
    prime_crossings = 0
    for idx in crossing_indices:
        # Check proximity window
        lo = max(2, idx - proximity)
        hi = min(n_max, idx + proximity)
        if np.any(prime_mask[lo:hi+1]):
            prime_crossings += 1
    
    crossing_fraction = prime_crossings / total_crossings
    
    # Expected fraction: P(random position is near prime)
    # For proximity=1, each position has 2*proximity+1 chances to hit
    prime_count = np.sum(prime_mask)
    coverage_per_prime = 2 * proximity + 1
    # Approximate: expected fraction ≈ min(1, prime_density * coverage)
    prime_density = prime_count / n_max
    expected_fraction = min(1.0, prime_density * coverage_per_prime)
    
    enrichment = crossing_fraction / expected_fraction if expected_fraction > 0 else 0
    
    # Binomial test for significance
    from scipy.stats import binomtest
    result = binomtest(prime_crossings, total_crossings, expected_fraction,
                      alternative='greater')
    p_value = result.pvalue
    
    return ZeroCrossingResult(
        crossing_indices=crossing_indices,
        crossing_directions=np.array([]),  # Filled by caller
        prime_crossings=prime_crossings,
        total_crossings=total_crossings,
        crossing_fraction=crossing_fraction,
        expected_fraction=expected_fraction,
        enrichment=enrichment,
        p_value=p_value
    )


# ============================================================================
# OSCILLATION CHARACTERIZATION
# ============================================================================

def extract_envelope(E: np.ndarray, start_idx: int = 100) -> np.ndarray:
    """
    Extract amplitude envelope using Hilbert transform.
    
    Args:
        E: Stress field array
        start_idx: Skip initial transient
        
    Returns:
        Amplitude envelope array
    """
    E_valid = E[start_idx:]
    
    # Hilbert transform gives analytic signal
    analytic = signal.hilbert(E_valid)
    envelope = np.abs(analytic)
    
    # Pad to match original size
    full_envelope = np.zeros_like(E)
    full_envelope[start_idx:] = envelope
    
    return full_envelope


def measure_damping(envelope: np.ndarray, 
                    start_idx: int = 100,
                    sample_points: int = 1000) -> Tuple[float, float, float]:
    """
    Measure damping rate from envelope decay.
    
    Fits: envelope(n) = A * n^(-α)
    
    Args:
        envelope: Amplitude envelope
        start_idx: Skip initial transient
        sample_points: Number of points for fitting
        
    Returns:
        (damping_exponent, error, r_squared)
    """
    E_valid = envelope[start_idx:]
    n_valid = len(E_valid)
    
    # Sample evenly in log space for robust fit
    sample_step = max(1, n_valid // sample_points)
    indices = np.arange(0, n_valid, sample_step)
    
    n_values = indices + start_idx  # Actual n values
    amp_values = E_valid[indices]
    
    # Filter out zeros/negatives for log fit
    mask = amp_values > 0
    n_values = n_values[mask]
    amp_values = amp_values[mask]
    
    if len(n_values) < 10:
        return 0.0, 1.0, 0.0
    
    # Log-log fit: log(amp) = log(A) - α*log(n)
    log_n = np.log(n_values)
    log_amp = np.log(amp_values)
    
    # Linear regression
    coeffs = np.polyfit(log_n, log_amp, 1)
    damping_exponent = -coeffs[0]  # Negative of slope
    
    # R² calculation
    predicted = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_amp - predicted) ** 2)
    ss_tot = np.sum((log_amp - np.mean(log_amp)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Error estimate from residuals
    residuals = log_amp - predicted
    error = np.std(residuals) / np.sqrt(len(residuals))
    
    return damping_exponent, error, r_squared


def compute_instantaneous_frequency(E: np.ndarray, 
                                     start_idx: int = 100) -> np.ndarray:
    """
    Compute instantaneous frequency via phase derivative.
    
    Args:
        E: Stress field array
        start_idx: Skip initial transient
        
    Returns:
        Instantaneous frequency array
    """
    E_valid = E[start_idx:]
    
    # Hilbert transform
    analytic = signal.hilbert(E_valid)
    phase = np.unwrap(np.angle(analytic))
    
    # Frequency = d(phase)/dn / (2π)
    inst_freq = np.diff(phase) / (2 * np.pi)
    
    # Pad to match original size
    full_freq = np.zeros_like(E)
    full_freq[start_idx:-1] = inst_freq
    
    return full_freq


# ============================================================================
# INTERVAL ANALYSIS
# ============================================================================

def analyze_crossing_intervals(crossing_indices: np.ndarray,
                                prime_mask: np.ndarray) -> Dict:
    """
    Analyze intervals between zero-crossings and compare to prime gaps.
    
    Args:
        crossing_indices: Positions of zero-crossings
        prime_mask: Boolean array where True = prime
        
    Returns:
        Dict with interval statistics and correlation measures
    """
    if len(crossing_indices) < 2:
        return {"error": "insufficient crossings"}
    
    # Crossing intervals
    crossing_gaps = np.diff(crossing_indices)
    
    # Prime gaps for comparison
    primes = np.where(prime_mask)[0]
    prime_gaps = np.diff(primes)
    
    # Match scales by looking at gaps in same range
    max_cross = np.max(crossing_indices)
    prime_gaps_in_range = prime_gaps[primes[:-1] < max_cross]
    
    # Statistics
    result = {
        "crossing_gap_mean": float(np.mean(crossing_gaps)),
        "crossing_gap_std": float(np.std(crossing_gaps)),
        "crossing_gap_median": float(np.median(crossing_gaps)),
        "prime_gap_mean": float(np.mean(prime_gaps_in_range)),
        "prime_gap_std": float(np.std(prime_gaps_in_range)),
        "prime_gap_median": float(np.median(prime_gaps_in_range)),
        "gap_ratio": float(np.mean(crossing_gaps) / np.mean(prime_gaps_in_range)),
        "n_crossings": len(crossing_indices),
        "n_primes_in_range": len(prime_gaps_in_range) + 1
    }
    
    # Histogram comparison (KS test)
    from scipy.stats import ks_2samp
    # Normalize gaps for comparison
    norm_cross = crossing_gaps / np.mean(crossing_gaps)
    norm_prime = prime_gaps_in_range / np.mean(prime_gaps_in_range)
    ks_stat, ks_pvalue = ks_2samp(norm_cross, norm_prime)
    
    result["ks_statistic"] = float(ks_stat)
    result["ks_pvalue"] = float(ks_pvalue)
    
    return result


# ============================================================================
# FULL ANALYSIS PIPELINE
# ============================================================================

def full_oscillation_analysis(n_max: int = 100000,
                               factor_base_size: int = 9,
                               window: int = 13,
                               lam: float = 0.99,
                               proximity: int = 2) -> Dict:
    """
    Run complete oscillation analysis pipeline.
    
    Args:
        n_max: Range to analyze
        factor_base_size: Number of primes in factor base
        window: SEC window parameter
        lam: SEC decay parameter
        proximity: Prime proximity threshold for crossing correlation
        
    Returns:
        Dict with all analysis results
    """
    # Compute SEC
    factor_base = FIRST_50_PRIMES[:factor_base_size]
    sec = compute_sec(n_max=n_max, factor_base=factor_base, window=window, lam=lam)
    
    # Zero-crossing detection
    crossings, directions = detect_zero_crossings(sec.E, start_idx=100)
    
    # Prime correlation
    cross_result = crossing_prime_correlation(crossings, sec.prime_mask, proximity=proximity)
    
    # Envelope and damping
    envelope = extract_envelope(sec.E, start_idx=100)
    damping, damping_err, r_sq = measure_damping(envelope, start_idx=100)
    
    # Interval analysis
    interval_stats = analyze_crossing_intervals(crossings, sec.prime_mask)
    
    # Compile results
    results = {
        "parameters": {
            "n_max": n_max,
            "factor_base_size": factor_base_size,
            "window": window,
            "lambda": lam,
            "proximity": proximity
        },
        "zero_crossings": {
            "total_crossings": cross_result.total_crossings,
            "prime_crossings": cross_result.prime_crossings,
            "crossing_fraction": cross_result.crossing_fraction,
            "expected_fraction": cross_result.expected_fraction,
            "enrichment": cross_result.enrichment,
            "p_value": cross_result.p_value
        },
        "oscillation": {
            "damping_exponent": damping,
            "damping_error": damping_err,
            "r_squared": r_sq,
            "damping_vs_phi": abs(damping - (1/PHI)) / (1/PHI),
            "damping_vs_xi_inv": abs(damping - (1/XI)) / (1/XI),
            "damping_vs_pi2_inv": abs(damping - PI_SQUARED_INV) / PI_SQUARED_INV
        },
        "intervals": interval_stats,
        "timestamp": get_timestamp()
    }
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_results(results: Dict, experiment_name: str, results_dir: str = "../results"):
    """Save results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = get_timestamp()
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {filepath}")
    return filepath
