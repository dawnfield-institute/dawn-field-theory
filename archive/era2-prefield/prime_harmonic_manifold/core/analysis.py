"""
Analysis utilities for Prime Harmonic Manifold experiments.
"""

import numpy as np
import sympy as sp
from collections import Counter
from typing import List, Tuple, Dict, Optional

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def gap_autocorrelation(gaps: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation of gap sequence.
    
    Returns array of ACF values for lags 0 to max_lag-1.
    """
    n = len(gaps)
    mean = np.mean(gaps)
    var = np.var(gaps)
    
    acf = []
    for lag in range(max_lag):
        if lag == 0:
            acf.append(1.0)
        else:
            cov = np.mean((gaps[:-lag] - mean) * (gaps[lag:] - mean))
            acf.append(cov / var if var > 0 else 0)
    
    return np.array(acf)


def find_decorrelation_length(acf: np.ndarray, threshold: float = None) -> int:
    """
    Find decorrelation length (where ACF drops below threshold).
    
    Default threshold is 1/e ≈ 0.368.
    """
    if threshold is None:
        threshold = 1 / np.e
    
    below = np.where(acf < threshold)[0]
    return below[0] if len(below) > 0 else len(acf)


def compute_local_entropy(gaps: np.ndarray, window: int = 50, step: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local chord entropy in sliding windows.
    
    Returns (entropies, positions) arrays.
    """
    if step is None:
        step = window // 2
    
    entropies = []
    positions = []
    
    for i in range(0, len(gaps) - window - 1, step):
        window_chords = [(gaps[j], gaps[j+1]) for j in range(i, i + window)]
        chord_counts = Counter(window_chords)
        total = sum(chord_counts.values())
        probs = np.array(list(chord_counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(entropy)
        positions.append(i + window // 2)
    
    return np.array(entropies), np.array(positions)


def rolling_std(data: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute rolling standard deviation (local curvature proxy)."""
    return np.array([
        data[i:i+window].std() for i in range(len(data) - window + 1)
    ])


def check_fibonacci_gaps(gaps: np.ndarray) -> Dict:
    """
    Check what fraction of gaps are Fibonacci numbers.
    """
    fibs = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610}
    
    gap_is_fib = [g in fibs for g in gaps]
    fib_fraction = sum(gap_is_fib) / len(gaps)
    
    gap_counts = Counter(gaps)
    expected_fib = sum(c for g, c in gap_counts.items() if g in fibs) / sum(gap_counts.values())
    
    return {
        'fib_fraction': fib_fraction,
        'expected_fraction': expected_fib,
        'enrichment': fib_fraction / expected_fib if expected_fib > 0 else 0,
    }


def check_consecutive_ratios_near_phi(gaps: np.ndarray, tolerance: float = 0.1) -> Dict:
    """
    Check how many consecutive gap ratios are near φ or 1/φ.
    """
    ratios = []
    for i in range(len(gaps) - 1):
        if gaps[i] > 0:
            ratios.append(gaps[i+1] / gaps[i])
    
    near_phi = sum(1 for r in ratios 
                   if abs(r - PHI) < tolerance or abs(r - PHI_INV) < tolerance)
    
    return {
        'n_ratios': len(ratios),
        'near_phi_count': near_phi,
        'near_phi_fraction': near_phi / len(ratios) if ratios else 0,
        'tolerance': tolerance,
    }


def eigenvalue_phi_analysis(eigenvalues: np.ndarray) -> List[Dict]:
    """
    Analyze eigenvalues for φ-related structure.
    """
    phi_targets = [
        (PHI, 'φ'),
        (PHI_INV, '1/φ'),
        (PHI**2, 'φ²'),
        (1/PHI**2, '1/φ²'),
        (1/PHI**3, '1/φ³'),
        (1.0, '1'),
        (0.5, '1/2'),
    ]
    
    results = []
    for i, ev in enumerate(eigenvalues[:10]):
        ev_mag = np.abs(ev)
        best_dist = float('inf')
        best_match = ''
        
        for target, label in phi_targets:
            dist = abs(ev_mag - target)
            if dist < best_dist:
                best_dist = dist
                best_match = label
        
        results.append({
            'index': i + 1,
            'value': ev_mag,
            'closest_phi': best_match,
            'distance': best_dist,
        })
    
    return results


def compute_consecutive_eigenvalue_ratios(eigenvalues: np.ndarray, min_val: float = 0.01) -> List[float]:
    """
    Compute ratios between consecutive eigenvalue magnitudes.
    """
    mags = np.sort(np.abs(eigenvalues))[::-1]
    
    ratios = []
    for i in range(len(mags) - 1):
        if mags[i] > min_val:
            ratios.append(mags[i+1] / mags[i])
    
    return ratios


def pac_depth_estimate(primes: np.ndarray) -> float:
    """
    Estimate PAC tree depth as log_φ(median_prime).
    """
    median_prime = np.median(primes)
    return np.log(median_prime) / np.log(PHI)
