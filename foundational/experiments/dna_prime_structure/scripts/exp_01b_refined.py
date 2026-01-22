"""
Experiment 01b: Amino Acid Gap Analysis (Refined)
==================================================

Hypothesis: Gaps between repeated amino acids in proteins show prime enrichment.

Refinements:
- Proper shuffled baseline for Möbius pairs
- Statistical significance testing
- Better expected prime rate calculation
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

# Primes for reference
def sieve_primes(n):
    """Simple sieve of Eratosthenes."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(10000)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def download_sample_proteins():
    """Download protein sequences from UniProt."""
    proteins = {}
    
    samples = [
        ("Hemoglobin_alpha_human", "P69905"),
        ("Hemoglobin_beta_human", "P68871"),
        ("Insulin_human", "P01308"),
        ("Myoglobin_human", "P02144"),
        ("Cytochrome_c_human", "P99999"),
        ("Ubiquitin_human", "P0CG48"),
        ("Histone_H4_human", "P62805"),
        ("Actin_human", "P60709"),
        ("Tubulin_alpha_human", "Q71U36"),
        ("p53_human", "P04637"),
    ]
    
    for name, uniprot_id in samples:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            with urllib.request.urlopen(url, timeout=10) as response:
                fasta = response.read().decode('utf-8')
                lines = fasta.strip().split('\n')
                sequence = ''.join(lines[1:])
                proteins[name] = sequence
                print(f"  Downloaded {name}: {len(sequence)} residues")
        except Exception as e:
            print(f"  Failed to download {name}: {e}")
    
    return proteins


def compute_amino_acid_gaps(sequence: str) -> list:
    """Compute all gaps between repeated amino acids."""
    positions = defaultdict(list)
    
    for i, aa in enumerate(sequence):
        if aa in AMINO_ACIDS:
            positions[aa].append(i)
    
    all_gaps = []
    for aa, pos_list in positions.items():
        if len(pos_list) >= 2:
            gaps = [pos_list[i+1] - pos_list[i] for i in range(len(pos_list)-1)]
            all_gaps.extend(gaps)
    
    return all_gaps


def compute_prime_enrichment(gaps: list, n_shuffles: int = 1000) -> dict:
    """
    Compute prime gap enrichment with proper statistical testing.
    
    Uses shuffled baseline: if we shuffle amino acid positions,
    what prime fraction would we expect?
    """
    if not gaps:
        return None
    
    gaps = [g for g in gaps if g > 1]
    if not gaps:
        return None
    
    # Observed prime fraction
    observed_prime_count = sum(1 for g in gaps if g in PRIMES)
    observed_fraction = observed_prime_count / len(gaps)
    
    # Theoretical expected fraction using prime number theorem
    # For gaps in range [2, max_gap], density is ~1/ln(max_gap)
    max_gap = max(gaps)
    mean_gap = np.mean(gaps)
    
    # More accurate: compute actual prime density in the gap range
    gap_range = range(2, max_gap + 1)
    primes_in_range = sum(1 for g in gap_range if g in PRIMES)
    theoretical_density = primes_in_range / len(gap_range)
    
    # Shuffled baseline: randomly sample from gap range with same size
    np.random.seed(42)
    shuffled_fractions = []
    for _ in range(n_shuffles):
        # Generate random gaps with similar distribution
        random_gaps = np.random.choice(list(gap_range), size=len(gaps), replace=True)
        random_prime_count = sum(1 for g in random_gaps if g in PRIMES)
        shuffled_fractions.append(random_prime_count / len(gaps))
    
    shuffled_mean = np.mean(shuffled_fractions)
    shuffled_std = np.std(shuffled_fractions)
    
    # Z-score and p-value
    z_score = (observed_fraction - shuffled_mean) / shuffled_std if shuffled_std > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test
    
    enrichment = observed_fraction / shuffled_mean if shuffled_mean > 0 else 0
    
    return {
        'n_gaps': len(gaps),
        'observed_prime_count': observed_prime_count,
        'observed_fraction': observed_fraction,
        'theoretical_density': theoretical_density,
        'shuffled_mean': shuffled_mean,
        'shuffled_std': shuffled_std,
        'enrichment': enrichment,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'max_gap': max_gap,
        'mean_gap': mean_gap,
    }


def compute_gap_distribution(gaps: list) -> dict:
    """Analyze gap size distribution, focusing on gap 2 and gap 6."""
    if not gaps:
        return None
    
    gap_counts = defaultdict(int)
    for g in gaps:
        gap_counts[g] += 1
    
    n_total = len(gaps)
    
    # Specific gaps of interest from SEC
    gap_2_frac = gap_counts.get(2, 0) / n_total
    gap_6_frac = gap_counts.get(6, 0) / n_total
    
    # Top gaps
    sorted_gaps = sorted(gap_counts.items(), key=lambda x: -x[1])
    
    return {
        'gap_2_count': gap_counts.get(2, 0),
        'gap_2_fraction': gap_2_frac,
        'gap_6_count': gap_counts.get(6, 0),
        'gap_6_fraction': gap_6_frac,
        'top_10_gaps': sorted_gaps[:10],
        'unique_gaps': len(gap_counts),
    }


def compute_mobius_pairs(gaps: list, n_shuffles: int = 1000) -> dict:
    """
    Analyze (a,b)/(b,a) Möbius pair patterns with shuffled baseline.
    """
    if len(gaps) < 3:
        return None
    
    # Count consecutive gap pairs
    def count_mobius_matches(gap_sequence):
        pair_counts = defaultdict(int)
        for i in range(len(gap_sequence) - 1):
            pair = (gap_sequence[i], gap_sequence[i+1])
            pair_counts[pair] += 1
        
        # Count pairs that have their Möbius mirror present
        mobius_count = 0
        for (a, b), count in pair_counts.items():
            if a != b and (b, a) in pair_counts:
                mobius_count += count  # Count each occurrence
        
        return mobius_count, len(gap_sequence) - 1
    
    observed_matches, total_pairs = count_mobius_matches(gaps)
    observed_rate = observed_matches / total_pairs if total_pairs > 0 else 0
    
    # Shuffled baseline
    np.random.seed(42)
    shuffled_rates = []
    for _ in range(n_shuffles):
        shuffled_gaps = np.random.permutation(gaps).tolist()
        matches, pairs = count_mobius_matches(shuffled_gaps)
        shuffled_rates.append(matches / pairs if pairs > 0 else 0)
    
    shuffled_mean = np.mean(shuffled_rates)
    shuffled_std = np.std(shuffled_rates)
    
    z_score = (observed_rate - shuffled_mean) / shuffled_std if shuffled_std > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score)
    
    enrichment = observed_rate / shuffled_mean if shuffled_mean > 0 else 0
    
    return {
        'total_pairs': total_pairs,
        'observed_matches': observed_matches,
        'observed_rate': observed_rate,
        'shuffled_mean': shuffled_mean,
        'shuffled_std': shuffled_std,
        'enrichment': enrichment,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def run_experiment():
    """Run the refined experiment."""
    print("=" * 60)
    print("Experiment 01b: Amino Acid Gap Analysis (Refined)")
    print("=" * 60)
    
    # Download proteins
    print("\n[1] Downloading protein sequences...")
    proteins = download_sample_proteins()
    
    if not proteins:
        print("ERROR: No proteins downloaded.")
        return
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_proteins': len(proteins),
        'per_protein': {},
        'aggregate': {}
    }
    
    # Collect all gaps
    print("\n[2] Computing amino acid gaps...")
    all_gaps = []
    
    for name, sequence in proteins.items():
        gaps = compute_amino_acid_gaps(sequence)
        all_gaps.extend(gaps)
        
        prime_analysis = compute_prime_enrichment(gaps)
        if prime_analysis:
            results['per_protein'][name] = {
                'length': len(sequence),
                'n_gaps': len(gaps),
                'prime_fraction': prime_analysis['observed_fraction'],
                'enrichment': prime_analysis['enrichment'],
            }
            print(f"  {name}: {len(gaps)} gaps, prime frac={prime_analysis['observed_fraction']:.3f}, enrichment={prime_analysis['enrichment']:.2f}x")
    
    # Aggregate analysis
    print("\n[3] Aggregate analysis...")
    
    prime_analysis = compute_prime_enrichment(all_gaps)
    gap_dist = compute_gap_distribution(all_gaps)
    mobius_analysis = compute_mobius_pairs(all_gaps)
    
    results['aggregate'] = {
        'total_gaps': len(all_gaps),
        'prime_enrichment': prime_analysis,
        'gap_distribution': gap_dist,
        'mobius_pairs': mobius_analysis,
    }
    
    print(f"\n  Total gaps: {len(all_gaps)}")
    print(f"\n  PRIME ENRICHMENT:")
    print(f"    Observed prime fraction: {prime_analysis['observed_fraction']:.4f}")
    print(f"    Expected (shuffled):     {prime_analysis['shuffled_mean']:.4f} ± {prime_analysis['shuffled_std']:.4f}")
    print(f"    Enrichment:              {prime_analysis['enrichment']:.2f}x")
    print(f"    Z-score:                 {prime_analysis['z_score']:.2f}")
    print(f"    P-value:                 {prime_analysis['p_value']:.2e}")
    print(f"    Significant (p<0.05):    {'YES ✅' if prime_analysis['significant'] else 'NO ❌'}")
    
    print(f"\n  GAP DISTRIBUTION:")
    print(f"    Gap 2 fraction: {gap_dist['gap_2_fraction']:.4f} ({gap_dist['gap_2_count']} occurrences)")
    print(f"    Gap 6 fraction: {gap_dist['gap_6_fraction']:.4f} ({gap_dist['gap_6_count']} occurrences)")
    print(f"    Top 5 gaps: {gap_dist['top_10_gaps'][:5]}")
    
    print(f"\n  MÖBIUS PAIRS:")
    print(f"    Observed rate:  {mobius_analysis['observed_rate']:.4f}")
    print(f"    Expected rate:  {mobius_analysis['shuffled_mean']:.4f} ± {mobius_analysis['shuffled_std']:.4f}")
    print(f"    Enrichment:     {mobius_analysis['enrichment']:.2f}x")
    print(f"    Z-score:        {mobius_analysis['z_score']:.2f}")
    print(f"    P-value:        {mobius_analysis['p_value']:.2e}")
    print(f"    Significant:    {'YES ✅' if mobius_analysis['significant'] else 'NO ❌'}")
    
    # Comparison to SEC
    print("\n[4] Comparison to SEC findings:")
    print("    ┌─────────────────┬────────────┬────────────┐")
    print("    │ Metric          │ SEC (gaps) │ DNA (aa)   │")
    print("    ├─────────────────┼────────────┼────────────┤")
    print(f"    │ Gap 6 fraction  │ ~0.20      │ {gap_dist['gap_6_fraction']:.4f}     │")
    print(f"    │ Gap 2 fraction  │ varies     │ {gap_dist['gap_2_fraction']:.4f}     │")
    print(f"    │ Möbius enrich.  │ 24x        │ {mobius_analysis['enrichment']:.2f}x       │")
    print("    └─────────────────┴────────────┴────────────┘")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_01b_refined_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    run_experiment()
