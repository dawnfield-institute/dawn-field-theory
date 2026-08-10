"""
Prime Harmonic Manifold - Core Module

Golden ratio eigenvalue emergence in prime chord dynamics.
"""

import numpy as np
import sympy as sp
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def get_primes(limit: int) -> np.ndarray:
    """Generate primes up to limit."""
    return np.array(list(sp.primerange(2, limit)), dtype=float)


def compute_gaps(primes: np.ndarray) -> np.ndarray:
    """Compute consecutive prime gaps."""
    return np.diff(primes)


def extract_chords(gaps: np.ndarray, n_gaps: int = 2) -> List[Tuple]:
    """
    Extract n-gap chord motifs from gap sequence.
    
    Args:
        gaps: Array of prime gaps
        n_gaps: Number of consecutive gaps per chord (2 or 3)
    
    Returns:
        List of chord tuples
    """
    chords = []
    for i in range(len(gaps) - n_gaps + 1):
        chord = tuple(gaps[i:i + n_gaps])
        chords.append(chord)
    return chords


def build_transition_matrix(chords: List[Tuple], top_k: int = 25) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Build Markov transition matrix from chord sequence.
    
    Args:
        chords: List of chord tuples
        top_k: Number of top chord types to track (rest go to "other")
    
    Returns:
        Tuple of (transition matrix P, list of tracked chord types)
    """
    # Count chord frequencies
    counts = Counter(chords)
    top_chords = [c for c, _ in counts.most_common(top_k)]
    chord_to_idx = {c: i for i, c in enumerate(top_chords)}
    other_idx = top_k
    
    # Map chord sequence to indices
    seq_idx = [chord_to_idx.get(c, other_idx) for c in chords]
    
    # Build transition count matrix
    T = np.zeros((top_k + 1, top_k + 1), dtype=int)
    for a, b in zip(seq_idx[:-1], seq_idx[1:]):
        T[a, b] += 1
    
    # Normalize to probabilities
    P = T.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums
    
    return P, top_chords


def compute_eigenvalues(P: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalue magnitudes of transition matrix.
    
    Returns sorted array of |eigenvalues| in descending order.
    """
    eigenvals = np.linalg.eigvals(P)
    return np.sort(np.abs(eigenvals))[::-1]


def phi_distance(value: float) -> Tuple[float, str]:
    """
    Compute distance to nearest φ-related value.
    
    Returns (distance, label) tuple.
    """
    targets = [
        (PHI, 'φ'),
        (PHI_INV, '1/φ'),
        (PHI**2, 'φ²'),
        (1/PHI**2, '1/φ²'),
        (1/PHI**3, '1/φ³'),
        (1.0, '1'),
        (0.5, '1/2'),
    ]
    
    best_dist = float('inf')
    best_label = ''
    for target, label in targets:
        dist = abs(value - target)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    
    return best_dist, best_label


def analyze_prime_range(prime_limit: int, top_k: int = 25) -> Dict:
    """
    Full analysis pipeline for primes up to limit.
    
    Returns dictionary with all computed metrics.
    """
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    chords = extract_chords(gaps, n_gaps=2)
    P, top_chords = build_transition_matrix(chords, top_k=top_k)
    
    # Compute on the tracked states only (exclude "other")
    eigenvals = compute_eigenvalues(P[:top_k, :top_k])
    
    lambda1 = eigenvals[0] if len(eigenvals) > 0 else 0
    lambda1_dist, lambda1_match = phi_distance(lambda1)
    
    return {
        'prime_limit': prime_limit,
        'n_primes': len(primes),
        'n_chords': len(chords),
        'top_k': top_k,
        'eigenvalues': eigenvals[:10].tolist(),
        'lambda1': lambda1,
        'lambda1_vs_phi_inv': lambda1 - PHI_INV,
        'lambda1_match': lambda1_match,
        'top_chords': [(c, Counter(chords)[c]) for c in top_chords[:10]],
    }


def motif_enrichment(gaps: np.ndarray, motifs: List[Tuple], n_shuffle: int = 10) -> Dict:
    """
    Compare motif frequencies in real vs shuffled gap sequences.
    
    Returns enrichment ratios for each motif.
    """
    # Real counts
    chords_real = extract_chords(gaps, n_gaps=len(motifs[0]))
    counts_real = Counter(chords_real)
    
    # Shuffled counts (average over n_shuffle trials)
    rng = np.random.default_rng(42)
    counts_shuffled = defaultdict(list)
    
    for _ in range(n_shuffle):
        gaps_shuf = gaps.copy()
        rng.shuffle(gaps_shuf)
        chords_shuf = extract_chords(gaps_shuf, n_gaps=len(motifs[0]))
        counts_shuf = Counter(chords_shuf)
        for m in motifs:
            counts_shuffled[m].append(counts_shuf.get(m, 0))
    
    # Compute enrichment
    results = {}
    for m in motifs:
        real = counts_real.get(m, 0)
        shuffled_mean = np.mean(counts_shuffled[m])
        enrichment = real / shuffled_mean if shuffled_mean > 0 else float('inf')
        results[m] = {
            'real': real,
            'shuffled_mean': shuffled_mean,
            'enrichment': enrichment,
        }
    
    return results


# Constants for external use
GOLDEN_RATIO = PHI
GOLDEN_RATIO_INV = PHI_INV
