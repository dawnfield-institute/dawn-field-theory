"""
Experiment 01: Amino Acid Gap Analysis
=======================================

Hypothesis: Gaps between repeated amino acids in proteins show prime enrichment.

Method:
1. Download protein sequences from UniProt
2. For each amino acid type, find gaps between occurrences
3. Check if prime gaps are enriched vs random
4. Compare to SEC gap signatures (gap 6 hub, gap 2 anchor)
"""

import numpy as np
from collections import defaultdict
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

# Standard amino acids
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def download_sample_proteins():
    """
    Download a few well-known protein sequences from UniProt.
    Returns dict of {name: sequence}
    """
    proteins = {}
    
    # Some well-studied proteins with known UniProt IDs
    # Format: (name, uniprot_id)
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
                # Parse FASTA - skip header line, join sequence lines
                lines = fasta.strip().split('\n')
                sequence = ''.join(lines[1:])
                proteins[name] = sequence
                print(f"Downloaded {name}: {len(sequence)} residues")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    
    return proteins


def compute_amino_acid_gaps(sequence: str) -> dict:
    """
    For each amino acid, compute gaps between consecutive occurrences.
    Returns {amino_acid: [gaps]}
    """
    positions = defaultdict(list)
    
    # Find positions of each amino acid
    for i, aa in enumerate(sequence):
        if aa in AMINO_ACIDS:
            positions[aa].append(i)
    
    # Compute gaps
    gaps = {}
    for aa, pos_list in positions.items():
        if len(pos_list) >= 2:
            gaps[aa] = [pos_list[i+1] - pos_list[i] for i in range(len(pos_list)-1)]
    
    return gaps


def analyze_gap_primeness(gaps: list) -> dict:
    """
    Analyze what fraction of gaps are prime.
    Compare to expected random rate.
    """
    if not gaps:
        return None
    
    gaps = [g for g in gaps if g > 1]  # Exclude gap 1 (adjacent)
    if not gaps:
        return None
    
    prime_gaps = [g for g in gaps if g in PRIMES]
    n_prime = len(prime_gaps)
    n_total = len(gaps)
    
    # Expected prime density at this scale (using prime number theorem)
    max_gap = max(gaps)
    expected_density = 1 / np.log(max_gap) if max_gap > 2 else 0.5
    
    observed_density = n_prime / n_total
    enrichment = observed_density / expected_density if expected_density > 0 else 0
    
    # Gap frequency distribution
    gap_counts = defaultdict(int)
    for g in gaps:
        gap_counts[g] += 1
    
    # Check for gap 6 and gap 2 signatures
    gap_6_count = gap_counts.get(6, 0)
    gap_2_count = gap_counts.get(2, 0)
    
    return {
        'n_gaps': n_total,
        'n_prime_gaps': n_prime,
        'observed_prime_density': observed_density,
        'expected_prime_density': expected_density,
        'enrichment': enrichment,
        'gap_6_count': gap_6_count,
        'gap_2_count': gap_2_count,
        'gap_6_fraction': gap_6_count / n_total if n_total > 0 else 0,
        'gap_2_fraction': gap_2_count / n_total if n_total > 0 else 0,
        'most_common_gaps': sorted(gap_counts.items(), key=lambda x: -x[1])[:10],
        'mean_gap': np.mean(gaps),
        'std_gap': np.std(gaps),
    }


def analyze_mobius_pairs(gaps: list) -> dict:
    """
    Check for (a,b)/(b,a) Möbius pair patterns in consecutive gaps.
    From SEC: these appear at 24x random rate.
    """
    if len(gaps) < 2:
        return None
    
    # Count consecutive gap pairs
    pair_counts = defaultdict(int)
    for i in range(len(gaps) - 1):
        pair = (gaps[i], gaps[i+1])
        pair_counts[pair] += 1
    
    # Find Möbius mirrors
    mobius_matches = 0
    total_pairs = len(gaps) - 1
    
    for (a, b), count in pair_counts.items():
        if a != b and (b, a) in pair_counts:
            mobius_matches += min(count, pair_counts[(b, a)])
    
    # Expected random rate (rough approximation)
    unique_gaps = len(set(gaps))
    expected_match_rate = 1 / (unique_gaps * unique_gaps) if unique_gaps > 0 else 0
    observed_match_rate = mobius_matches / total_pairs if total_pairs > 0 else 0
    
    return {
        'total_pairs': total_pairs,
        'mobius_matches': mobius_matches,
        'observed_rate': observed_match_rate,
        'expected_rate': expected_match_rate,
        'enrichment': observed_match_rate / expected_match_rate if expected_match_rate > 0 else 0,
        'top_pairs': sorted(pair_counts.items(), key=lambda x: -x[1])[:10],
    }


def run_experiment():
    """Run the full experiment."""
    print("=" * 60)
    print("Experiment 01: Amino Acid Gap Analysis")
    print("=" * 60)
    
    # Download proteins
    print("\n[1] Downloading protein sequences...")
    proteins = download_sample_proteins()
    
    if not proteins:
        print("ERROR: No proteins downloaded. Check internet connection.")
        return
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_proteins': len(proteins),
        'proteins': {},
        'aggregate': {}
    }
    
    # Analyze each protein
    print("\n[2] Analyzing amino acid gaps...")
    all_gaps = []
    
    for name, sequence in proteins.items():
        print(f"\n  {name}:")
        gaps_by_aa = compute_amino_acid_gaps(sequence)
        
        protein_result = {
            'length': len(sequence),
            'amino_acids': {}
        }
        
        protein_all_gaps = []
        
        for aa, gaps in gaps_by_aa.items():
            analysis = analyze_gap_primeness(gaps)
            if analysis:
                protein_result['amino_acids'][aa] = analysis
                protein_all_gaps.extend(gaps)
                all_gaps.extend(gaps)
        
        # Möbius pair analysis for this protein
        if protein_all_gaps:
            mobius = analyze_mobius_pairs(protein_all_gaps)
            protein_result['mobius_pairs'] = mobius
        
        results['proteins'][name] = protein_result
        
        # Summary for this protein
        if protein_all_gaps:
            prime_frac = sum(1 for g in protein_all_gaps if g in PRIMES) / len(protein_all_gaps)
            print(f"    Total gaps: {len(protein_all_gaps)}, Prime fraction: {prime_frac:.3f}")
    
    # Aggregate analysis
    print("\n[3] Aggregate analysis across all proteins...")
    
    if all_gaps:
        agg_analysis = analyze_gap_primeness(all_gaps)
        agg_mobius = analyze_mobius_pairs(all_gaps)
        
        results['aggregate'] = {
            'total_gaps': len(all_gaps),
            'primeness': agg_analysis,
            'mobius_pairs': agg_mobius,
        }
        
        print(f"\n  Total gaps analyzed: {len(all_gaps)}")
        print(f"  Prime gap enrichment: {agg_analysis['enrichment']:.2f}x")
        print(f"  Gap 6 fraction: {agg_analysis['gap_6_fraction']:.3f}")
        print(f"  Gap 2 fraction: {agg_analysis['gap_2_fraction']:.3f}")
        
        if agg_mobius:
            print(f"  Möbius pair enrichment: {agg_mobius['enrichment']:.2f}x")
    
    # Comparison to SEC findings
    print("\n[4] Comparison to SEC findings:")
    print("  SEC gap 6 fraction: ~0.20 (Möbius hub)")
    print(f"  DNA gap 6 fraction: {agg_analysis['gap_6_fraction']:.3f}")
    print("  SEC Möbius pair enrichment: 24x")
    print(f"  DNA Möbius pair enrichment: {agg_mobius['enrichment']:.2f}x" if agg_mobius else "  DNA Möbius: N/A")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_01_amino_acid_gaps_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    run_experiment()
