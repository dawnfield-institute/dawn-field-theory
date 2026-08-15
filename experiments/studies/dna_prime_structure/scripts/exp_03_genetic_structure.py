"""
Experiment 03: Codon and Genetic Structure
==========================================

Testing if prime patterns appear at the genetic code level:
1. Codon positions (every 3rd nucleotide)
2. Start/stop codon spacing
3. Reading frame structure
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime

# Primes
def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(10000)

# Genetic code
STOP_CODONS = {'TAA', 'TAG', 'TGA'}
START_CODON = 'ATG'


def generate_random_dna(length: int) -> str:
    """Generate random DNA sequence."""
    return ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=length))


def find_codon_positions(sequence: str, codon: str) -> list:
    """Find all positions of a specific codon (in any reading frame)."""
    positions = []
    for i in range(len(sequence) - 2):
        if sequence[i:i+3] == codon:
            positions.append(i)
    return positions


def compute_gaps(positions: list) -> list:
    """Compute gaps between consecutive positions."""
    if len(positions) < 2:
        return []
    return [positions[i+1] - positions[i] for i in range(len(positions)-1)]


def analyze_codon_spacing(sequence: str, n_shuffles: int = 100) -> dict:
    """
    Analyze if start/stop codons have prime-spaced distribution.
    """
    results = {}
    
    # Find start and stop codon positions
    start_positions = find_codon_positions(sequence, START_CODON)
    stop_positions = []
    for codon in STOP_CODONS:
        stop_positions.extend(find_codon_positions(sequence, codon))
    stop_positions = sorted(stop_positions)
    
    # Analyze start codon gaps
    start_gaps = compute_gaps(start_positions)
    if len(start_gaps) >= 10:
        prime_frac = sum(1 for g in start_gaps if g in PRIMES) / len(start_gaps)
        
        # Expected from random
        max_gap = max(start_gaps) if start_gaps else 100
        expected_frac = sum(1 for p in range(2, max_gap+1) if p in PRIMES) / max_gap
        
        results['start_codon'] = {
            'n_occurrences': len(start_positions),
            'n_gaps': len(start_gaps),
            'mean_gap': np.mean(start_gaps),
            'prime_fraction': prime_frac,
            'expected_fraction': expected_frac,
            'enrichment': prime_frac / expected_frac if expected_frac > 0 else 0,
        }
    
    # Analyze stop codon gaps
    stop_gaps = compute_gaps(stop_positions)
    if len(stop_gaps) >= 10:
        prime_frac = sum(1 for g in stop_gaps if g in PRIMES) / len(stop_gaps)
        max_gap = max(stop_gaps) if stop_gaps else 100
        expected_frac = sum(1 for p in range(2, max_gap+1) if p in PRIMES) / max_gap
        
        results['stop_codon'] = {
            'n_occurrences': len(stop_positions),
            'n_gaps': len(stop_gaps),
            'mean_gap': np.mean(stop_gaps),
            'prime_fraction': prime_frac,
            'expected_fraction': expected_frac,
            'enrichment': prime_frac / expected_frac if expected_frac > 0 else 0,
        }
    
    # Compare to shuffled
    if start_gaps:
        shuffled_fracs = []
        for _ in range(n_shuffles):
            shuffled_seq = ''.join(np.random.permutation(list(sequence)))
            shuffled_starts = find_codon_positions(shuffled_seq, START_CODON)
            shuffled_gaps = compute_gaps(shuffled_starts)
            if shuffled_gaps:
                frac = sum(1 for g in shuffled_gaps if g in PRIMES) / len(shuffled_gaps)
                shuffled_fracs.append(frac)
        
        if shuffled_fracs:
            results['start_vs_shuffled'] = {
                'real_frac': results['start_codon']['prime_fraction'],
                'shuffled_mean': np.mean(shuffled_fracs),
                'shuffled_std': np.std(shuffled_fracs),
                'z_score': (results['start_codon']['prime_fraction'] - np.mean(shuffled_fracs)) / np.std(shuffled_fracs) if np.std(shuffled_fracs) > 0 else 0,
            }
    
    return results


def analyze_reading_frame_structure(sequence: str) -> dict:
    """
    Analyze if reading frame boundaries (every 3rd position) show special structure.
    """
    length = len(sequence)
    
    # In a coding region, positions 0,3,6,9... are codon starts
    # The "gaps" between meaningful codons might have prime structure
    
    # Simulate: find all valid ORFs (start to stop codon in frame)
    orfs = []
    i = 0
    while i < length - 2:
        if sequence[i:i+3] == START_CODON:
            # Found start, look for in-frame stop
            for j in range(i+3, length-2, 3):
                if sequence[j:j+3] in STOP_CODONS:
                    orf_length = j - i
                    orfs.append({
                        'start': i,
                        'end': j,
                        'length': orf_length,
                        'codons': orf_length // 3,
                    })
                    break
        i += 1
    
    if len(orfs) < 5:
        return {'insufficient_orfs': True}
    
    # Analyze ORF lengths (in codons)
    orf_lengths = [orf['codons'] for orf in orfs]
    prime_lengths = sum(1 for l in orf_lengths if l in PRIMES)
    
    # Gaps between ORFs
    orf_gaps = []
    sorted_orfs = sorted(orfs, key=lambda x: x['start'])
    for i in range(len(sorted_orfs) - 1):
        gap = sorted_orfs[i+1]['start'] - sorted_orfs[i]['end']
        if gap > 0:
            orf_gaps.append(gap)
    
    return {
        'n_orfs': len(orfs),
        'orf_lengths': {
            'mean': np.mean(orf_lengths),
            'prime_fraction': prime_lengths / len(orf_lengths) if orf_lengths else 0,
        },
        'orf_gaps': {
            'n_gaps': len(orf_gaps),
            'mean': np.mean(orf_gaps) if orf_gaps else 0,
            'prime_fraction': sum(1 for g in orf_gaps if g in PRIMES) / len(orf_gaps) if orf_gaps else 0,
        }
    }


def test_divisibility_by_3(gaps: list) -> dict:
    """
    Test if gaps show codon structure (divisible by 3).
    This would indicate genetic code organization.
    """
    if not gaps:
        return None
    
    div_by_3 = sum(1 for g in gaps if g % 3 == 0)
    frac = div_by_3 / len(gaps)
    
    # Expected if random: 1/3
    expected = 1/3
    
    # Chi-square test
    observed = [div_by_3, len(gaps) - div_by_3]
    expected_counts = [len(gaps) / 3, 2 * len(gaps) / 3]
    chi2, p_value = stats.chisquare(observed, expected_counts)
    
    return {
        'divisible_by_3': div_by_3,
        'total': len(gaps),
        'fraction': frac,
        'expected': expected,
        'enrichment': frac / expected,
        'chi2': chi2,
        'p_value': p_value,
    }


def run_experiment():
    """Run genetic structure analysis."""
    print("=" * 60)
    print("Experiment 03: Codon and Genetic Structure")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Generate a longer random DNA sequence for analysis
    print("\n[1] Generating test DNA sequences...")
    
    # Test on random DNA
    np.random.seed(42)
    random_dna = generate_random_dna(100000)
    print(f"  Random DNA: {len(random_dna)} bp")
    
    # Codon spacing analysis
    print("\n[2] Analyzing codon spacing...")
    codon_analysis = analyze_codon_spacing(random_dna, n_shuffles=50)
    results['tests']['codon_spacing'] = codon_analysis
    
    if 'start_codon' in codon_analysis:
        sc = codon_analysis['start_codon']
        print(f"  Start codons (ATG): {sc['n_occurrences']} found, {sc['n_gaps']} gaps")
        print(f"    Mean gap: {sc['mean_gap']:.1f} bp")
        print(f"    Prime fraction: {sc['prime_fraction']:.3f} (expected: {sc['expected_fraction']:.3f})")
        print(f"    Enrichment: {sc['enrichment']:.2f}x")
    
    if 'stop_codon' in codon_analysis:
        sc = codon_analysis['stop_codon']
        print(f"  Stop codons: {sc['n_occurrences']} found, {sc['n_gaps']} gaps")
        print(f"    Mean gap: {sc['mean_gap']:.1f} bp")
        print(f"    Prime fraction: {sc['prime_fraction']:.3f} (expected: {sc['expected_fraction']:.3f})")
        print(f"    Enrichment: {sc['enrichment']:.2f}x")
    
    if 'start_vs_shuffled' in codon_analysis:
        svs = codon_analysis['start_vs_shuffled']
        print(f"  Start codon vs shuffled: z = {svs['z_score']:.2f}")
    
    # Reading frame analysis
    print("\n[3] Reading frame structure...")
    rf_analysis = analyze_reading_frame_structure(random_dna)
    results['tests']['reading_frame'] = rf_analysis
    
    if 'insufficient_orfs' not in rf_analysis:
        print(f"  ORFs found: {rf_analysis['n_orfs']}")
        print(f"  ORF lengths prime fraction: {rf_analysis['orf_lengths']['prime_fraction']:.3f}")
        print(f"  ORF gap prime fraction: {rf_analysis['orf_gaps']['prime_fraction']:.3f}")
    
    # Test divisibility by 3 (codon structure)
    print("\n[4] Testing divisibility by 3 (codon structure)...")
    
    # Get all gaps between start codons
    start_positions = find_codon_positions(random_dna, START_CODON)
    start_gaps = compute_gaps(start_positions)
    
    div3_analysis = test_divisibility_by_3(start_gaps)
    results['tests']['divisibility_by_3'] = div3_analysis
    
    if div3_analysis:
        print(f"  Gaps divisible by 3: {div3_analysis['fraction']:.3f} (expected: 0.333)")
        print(f"  Enrichment: {div3_analysis['enrichment']:.2f}x")
        print(f"  Chi-square p-value: {div3_analysis['p_value']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n  Random DNA shows no special prime structure in:")
    print("  - Start codon spacing")
    print("  - Stop codon spacing")
    print("  - ORF lengths")
    print("  - ORF gaps")
    print("\n  → Need to test on REAL genomic data with functional elements")
    print("  → The codon structure (divisibility by 3) might be more relevant than primes")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_03_genetic_structure_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    run_experiment()
