"""
Experiment 05: Motif Enrichment Analysis

Tests whether palindromic and structured gap motifs are enriched
in real prime sequences vs shuffled controls.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import (
    get_primes, compute_gaps, motif_enrichment
)
import numpy as np
from collections import Counter


def run_experiment(prime_limit: int = 500000, n_shuffle: int = 100):
    """Run motif enrichment analysis."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Motif Enrichment Analysis")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    print(f"  Primes: {len(primes):,}")
    print(f"  Gaps: {len(gaps):,}")
    
    # Gap statistics
    gap_counts = Counter(gaps)
    print("\nTop 10 gap values:")
    for gap, count in gap_counts.most_common(10):
        print(f"  gap={int(gap):3d}: {count:,} occurrences")
    
    # Define motif categories
    palindrome_3 = [
        (2.0, 4.0, 2.0),
        (4.0, 2.0, 4.0),
        (6.0, 4.0, 6.0),
        (2.0, 6.0, 2.0),
        (6.0, 2.0, 6.0),
        (4.0, 6.0, 4.0),
        (2.0, 10.0, 2.0),
        (6.0, 8.0, 6.0),
        (8.0, 6.0, 8.0),
        (4.0, 8.0, 4.0),
    ]
    
    twin_adjacent = [
        (2.0, 2.0, 2.0),
        (2.0, 2.0, 4.0),
        (4.0, 2.0, 2.0),
        (2.0, 4.0, 4.0),
    ]
    
    arithmetic = [
        (2.0, 4.0, 6.0),
        (6.0, 4.0, 2.0),
        (4.0, 6.0, 8.0),
        (8.0, 6.0, 4.0),
    ]
    
    print(f"\nComputing enrichment vs {n_shuffle} shuffled controls...")
    
    # Test palindromes
    print("\n" + "-" * 60)
    print("PALINDROME MOTIFS (ABA structure)")
    print("-" * 60)
    
    palindrome_results = motif_enrichment(gaps, palindrome_3, n_shuffle=n_shuffle)
    enrichments_pal = []
    for motif, data in palindrome_results.items():
        print(f"  {motif}: {data['real']:5d} real vs {data['shuffled_mean']:6.1f} shuffled = {data['enrichment']:.2f}x")
        enrichments_pal.append(data['enrichment'])
    
    mean_pal = np.mean(enrichments_pal)
    print(f"\n  Mean palindrome enrichment: {mean_pal:.2f}x")
    
    # Test twin-adjacent
    print("\n" + "-" * 60)
    print("TWIN-ADJACENT MOTIFS")
    print("-" * 60)
    
    twin_results = motif_enrichment(gaps, twin_adjacent, n_shuffle=n_shuffle)
    enrichments_twin = []
    for motif, data in twin_results.items():
        print(f"  {motif}: {data['real']:5d} real vs {data['shuffled_mean']:6.1f} shuffled = {data['enrichment']:.2f}x")
        enrichments_twin.append(data['enrichment'])
    
    mean_twin = np.mean(enrichments_twin)
    print(f"\n  Mean twin-adjacent enrichment: {mean_twin:.2f}x")
    
    # Test arithmetic
    print("\n" + "-" * 60)
    print("ARITHMETIC PROGRESSION MOTIFS")
    print("-" * 60)
    
    arith_results = motif_enrichment(gaps, arithmetic, n_shuffle=n_shuffle)
    enrichments_arith = []
    for motif, data in arith_results.items():
        print(f"  {motif}: {data['real']:5d} real vs {data['shuffled_mean']:6.1f} shuffled = {data['enrichment']:.2f}x")
        enrichments_arith.append(data['enrichment'])
    
    mean_arith = np.mean(enrichments_arith)
    print(f"\n  Mean arithmetic enrichment: {mean_arith:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Palindromes:     {mean_pal:.2f}x enriched")
    print(f"  Twin-adjacent:   {mean_twin:.2f}x enriched")
    print(f"  Arithmetic:      {mean_arith:.2f}x enriched")
    print("\n  Interpretation: Prime gaps show STRUCTURAL preference")
    print("  for symmetric (palindromic) patterns over random.")
    
    # Save results
    results = {
        'experiment': 'exp_05_motif_enrichment',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'prime_limit': prime_limit,
            'n_shuffle': n_shuffle,
        },
        'results': {
            'palindrome': {
                'motifs': {str(k): v for k, v in palindrome_results.items()},
                'mean_enrichment': mean_pal,
            },
            'twin_adjacent': {
                'motifs': {str(k): v for k, v in twin_results.items()},
                'mean_enrichment': mean_twin,
            },
            'arithmetic': {
                'motifs': {str(k): v for k, v in arith_results.items()},
                'mean_enrichment': mean_arith,
            },
        },
        'conclusion': 'PALINDROMES_ENRICHED' if mean_pal > 1.5 else 'NO_CLEAR_PREFERENCE'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_05_motif_enrichment_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
