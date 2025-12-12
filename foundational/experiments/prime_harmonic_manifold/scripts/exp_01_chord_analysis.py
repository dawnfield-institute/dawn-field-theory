"""
Experiment 01: Baseline Chord Analysis

Validates the core finding that prime gap chords form structured
Markov dynamics with λ₁ ≈ 1/φ at coarse-grained scale.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import (
    get_primes, compute_gaps, extract_chords,
    build_transition_matrix, compute_eigenvalues,
    phi_distance, motif_enrichment, PHI_INV
)
import numpy as np


def run_experiment(prime_limit: int = 200000, top_k: int = 25):
    """Run baseline chord analysis."""
    
    print("=" * 60)
    print("PRIME HARMONIC MANIFOLD: Baseline Chord Analysis")
    print("=" * 60)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    print(f"  Primes: {len(primes):,}")
    print(f"  Gaps: {len(gaps):,}")
    
    # Extract chords
    chords_2gap = extract_chords(gaps, n_gaps=2)
    chords_3gap = extract_chords(gaps, n_gaps=3)
    print(f"  2-gap chords: {len(chords_2gap):,}")
    print(f"  3-gap chords: {len(chords_3gap):,}")
    
    # Build transition matrix
    print(f"\nBuilding transition matrix (top_k={top_k})...")
    P, top_chords = build_transition_matrix(chords_2gap, top_k=top_k)
    
    # Compute eigenvalues
    eigenvals = compute_eigenvalues(P[:top_k, :top_k])
    
    print("\n" + "-" * 60)
    print("EIGENVALUE ANALYSIS")
    print("-" * 60)
    
    for i, ev in enumerate(eigenvals[:10]):
        dist, match = phi_distance(ev)
        print(f"  λ_{i+1} = {ev:.6f}  (closest: {match}, dist={dist:.4f})")
    
    lambda1 = eigenvals[0]
    print(f"\n  λ₁ = {lambda1:.6f}")
    print(f"  1/φ = {PHI_INV:.6f}")
    print(f"  Error: {abs(lambda1 - PHI_INV):.6f} ({abs(lambda1 - PHI_INV)/PHI_INV*100:.2f}%)")
    
    # Motif enrichment
    print("\n" + "-" * 60)
    print("MOTIF ENRICHMENT (vs shuffled)")
    print("-" * 60)
    
    test_motifs = [
        (6.0, 4.0, 6.0),
        (4.0, 2.0, 4.0),
        (2.0, 10.0, 2.0),
        (6.0, 8.0, 6.0),
    ]
    
    enrichment = motif_enrichment(gaps, test_motifs, n_shuffle=20)
    for motif, data in enrichment.items():
        print(f"  {motif}: {data['real']:4d} real vs {data['shuffled_mean']:.1f} shuffled = {data['enrichment']:.2f}x enrichment")
    
    # Save results
    results = {
        'experiment': 'exp_01_chord_analysis',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'prime_limit': prime_limit,
            'top_k': top_k,
        },
        'results': {
            'n_primes': len(primes),
            'n_gaps': len(gaps),
            'n_chords_2gap': len(chords_2gap),
            'lambda1': lambda1,
            'phi_inv': PHI_INV,
            'lambda1_error': abs(lambda1 - PHI_INV),
            'lambda1_error_pct': abs(lambda1 - PHI_INV)/PHI_INV*100,
            'eigenvalues_top10': eigenvals[:10].tolist(),
            'motif_enrichment': {str(k): v for k, v in enrichment.items()},
        },
        'conclusion': 'CONFIRMED' if abs(lambda1 - PHI_INV) < 0.05 else 'PARTIAL'
    }
    
    # Write to results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_01_chord_analysis_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
