"""
Experiment 02: Deep Prime Analysis
===================================

Investigating whether prime enrichment in proteins is:
1. Real or artifact of small-number bias
2. Scale-dependent
3. Robust to controls
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

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
        except Exception as e:
            print(f"  Failed: {name}: {e}")
    
    return proteins


def compute_gaps(sequence: str) -> list:
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


def analyze_by_gap_range(gaps: list) -> dict:
    """
    Analyze prime enrichment in different gap size ranges.
    This tests whether enrichment is just a small-number artifact.
    """
    results = {}
    
    ranges = [
        (2, 10, "small"),
        (11, 30, "medium"),
        (31, 100, "large"),
        (101, 500, "very_large"),
    ]
    
    for low, high, name in ranges:
        range_gaps = [g for g in gaps if low <= g <= high]
        if len(range_gaps) < 20:
            continue
        
        # Count primes in this range
        prime_count = sum(1 for g in range_gaps if g in PRIMES)
        observed_frac = prime_count / len(range_gaps)
        
        # Expected: what fraction of integers in [low, high] are prime?
        primes_in_range = sum(1 for i in range(low, high+1) if i in PRIMES)
        expected_frac = primes_in_range / (high - low + 1)
        
        enrichment = observed_frac / expected_frac if expected_frac > 0 else 0
        
        results[name] = {
            'range': (low, high),
            'n_gaps': len(range_gaps),
            'prime_count': prime_count,
            'observed_frac': observed_frac,
            'expected_frac': expected_frac,
            'enrichment': enrichment,
        }
    
    return results


def compare_to_shuffled_protein(sequence: str, n_shuffles: int = 100) -> dict:
    """
    Compare real protein gaps to shuffled protein sequences.
    This tests if the pattern requires real protein structure.
    """
    real_gaps = compute_gaps(sequence)
    real_prime_frac = sum(1 for g in real_gaps if g in PRIMES) / len(real_gaps) if real_gaps else 0
    
    shuffled_fracs = []
    for _ in range(n_shuffles):
        # Shuffle the sequence
        shuffled_seq = ''.join(np.random.permutation(list(sequence)))
        shuffled_gaps = compute_gaps(shuffled_seq)
        if shuffled_gaps:
            frac = sum(1 for g in shuffled_gaps if g in PRIMES) / len(shuffled_gaps)
            shuffled_fracs.append(frac)
    
    if not shuffled_fracs:
        return None
    
    shuffled_mean = np.mean(shuffled_fracs)
    shuffled_std = np.std(shuffled_fracs)
    
    z_score = (real_prime_frac - shuffled_mean) / shuffled_std if shuffled_std > 0 else 0
    
    return {
        'real_prime_frac': real_prime_frac,
        'shuffled_mean': shuffled_mean,
        'shuffled_std': shuffled_std,
        'z_score': z_score,
        'enrichment_vs_shuffled': real_prime_frac / shuffled_mean if shuffled_mean > 0 else 0,
    }


def analyze_amino_acid_frequency_effect(sequence: str) -> dict:
    """
    Test if prime enrichment is driven by amino acid frequency distribution.
    Some amino acids are rare, creating inherently larger gaps.
    """
    # Count amino acid frequencies
    aa_counts = defaultdict(int)
    for aa in sequence:
        if aa in AMINO_ACIDS:
            aa_counts[aa] += 1
    
    total = sum(aa_counts.values())
    aa_freqs = {aa: count/total for aa, count in aa_counts.items()}
    
    # For each amino acid, compute average gap
    positions = defaultdict(list)
    for i, aa in enumerate(sequence):
        if aa in AMINO_ACIDS:
            positions[aa].append(i)
    
    aa_gap_stats = {}
    for aa, pos_list in positions.items():
        if len(pos_list) >= 2:
            gaps = [pos_list[i+1] - pos_list[i] for i in range(len(pos_list)-1)]
            prime_frac = sum(1 for g in gaps if g in PRIMES) / len(gaps)
            aa_gap_stats[aa] = {
                'frequency': aa_freqs.get(aa, 0),
                'n_gaps': len(gaps),
                'mean_gap': np.mean(gaps),
                'prime_fraction': prime_frac,
            }
    
    # Correlation between frequency and prime fraction
    freqs = [v['frequency'] for v in aa_gap_stats.values()]
    prime_fracs = [v['prime_fraction'] for v in aa_gap_stats.values()]
    
    if len(freqs) >= 3:
        corr, p_val = stats.pearsonr(freqs, prime_fracs)
    else:
        corr, p_val = 0, 1
    
    return {
        'aa_stats': aa_gap_stats,
        'freq_primefrac_correlation': corr,
        'correlation_pvalue': p_val,
    }


def analyze_random_sequences(length: int = 400, n_sequences: int = 100) -> dict:
    """
    Generate random amino acid sequences and compute prime enrichment.
    This is the ultimate null hypothesis test.
    """
    # Use natural amino acid frequencies (approximate)
    aa_freqs = {
        'A': 0.074, 'R': 0.042, 'N': 0.044, 'D': 0.059, 'C': 0.033,
        'Q': 0.037, 'E': 0.058, 'G': 0.074, 'H': 0.029, 'I': 0.038,
        'L': 0.076, 'K': 0.072, 'M': 0.018, 'F': 0.040, 'P': 0.050,
        'S': 0.081, 'T': 0.062, 'W': 0.013, 'Y': 0.033, 'V': 0.068,
    }
    
    aas = list(aa_freqs.keys())
    probs = [aa_freqs[aa] for aa in aas]
    probs = np.array(probs) / sum(probs)  # Normalize
    
    random_prime_fracs = []
    
    for _ in range(n_sequences):
        # Generate random sequence with natural frequencies
        random_seq = ''.join(np.random.choice(aas, size=length, p=probs))
        gaps = compute_gaps(random_seq)
        if gaps:
            frac = sum(1 for g in gaps if g in PRIMES) / len(gaps)
            random_prime_fracs.append(frac)
    
    return {
        'n_sequences': n_sequences,
        'sequence_length': length,
        'mean_prime_frac': np.mean(random_prime_fracs),
        'std_prime_frac': np.std(random_prime_fracs),
        'min_prime_frac': np.min(random_prime_fracs),
        'max_prime_frac': np.max(random_prime_fracs),
    }


def run_experiment():
    """Run deep analysis."""
    print("=" * 60)
    print("Experiment 02: Deep Prime Analysis")
    print("=" * 60)
    
    # Download proteins
    print("\n[1] Downloading proteins...")
    proteins = download_sample_proteins()
    print(f"  Downloaded {len(proteins)} proteins")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Collect all gaps
    all_gaps = []
    for seq in proteins.values():
        all_gaps.extend(compute_gaps(seq))
    
    print(f"  Total gaps: {len(all_gaps)}")
    
    # Test 1: Gap range analysis
    print("\n[2] Analyzing by gap size range...")
    range_analysis = analyze_by_gap_range(all_gaps)
    results['tests']['gap_ranges'] = range_analysis
    
    print("\n  Gap Range Analysis:")
    print("  ┌─────────────┬─────────┬──────────┬──────────┬────────────┐")
    print("  │ Range       │ N gaps  │ Observed │ Expected │ Enrichment │")
    print("  ├─────────────┼─────────┼──────────┼──────────┼────────────┤")
    for name, data in range_analysis.items():
        lo, hi = data['range']
        print(f"  │ {lo:3d}-{hi:3d}     │ {data['n_gaps']:7d} │ {data['observed_frac']:.4f}   │ {data['expected_frac']:.4f}   │ {data['enrichment']:.2f}x       │")
    print("  └─────────────┴─────────┴──────────┴──────────┴────────────┘")
    
    # Test 2: Shuffled protein comparison
    print("\n[3] Comparing to shuffled proteins...")
    shuffled_results = {}
    for name, seq in list(proteins.items())[:3]:  # Just first 3 for speed
        result = compare_to_shuffled_protein(seq, n_shuffles=50)
        if result:
            shuffled_results[name] = result
            print(f"  {name}: real={result['real_prime_frac']:.3f}, shuffled={result['shuffled_mean']:.3f}, z={result['z_score']:.2f}")
    
    results['tests']['shuffled_comparison'] = shuffled_results
    
    # Test 3: Random sequence baseline
    print("\n[4] Random sequence baseline...")
    random_analysis = analyze_random_sequences(length=400, n_sequences=100)
    results['tests']['random_sequences'] = random_analysis
    
    real_overall = sum(1 for g in all_gaps if g in PRIMES) / len(all_gaps)
    
    print(f"  Random sequences (natural AA freq):")
    print(f"    Mean prime fraction: {random_analysis['mean_prime_frac']:.4f} ± {random_analysis['std_prime_frac']:.4f}")
    print(f"    Range: [{random_analysis['min_prime_frac']:.4f}, {random_analysis['max_prime_frac']:.4f}]")
    print(f"  Real proteins:")
    print(f"    Prime fraction: {real_overall:.4f}")
    
    z_vs_random = (real_overall - random_analysis['mean_prime_frac']) / random_analysis['std_prime_frac']
    print(f"  Z-score vs random: {z_vs_random:.2f}")
    
    results['tests']['real_vs_random'] = {
        'real_prime_frac': real_overall,
        'random_mean': random_analysis['mean_prime_frac'],
        'random_std': random_analysis['std_prime_frac'],
        'z_score': z_vs_random,
    }
    
    # Test 4: Amino acid frequency effect
    print("\n[5] Amino acid frequency effects...")
    # Use largest protein for this
    largest_protein = max(proteins.items(), key=lambda x: len(x[1]))
    aa_analysis = analyze_amino_acid_frequency_effect(largest_protein[1])
    results['tests']['aa_frequency_effect'] = aa_analysis
    
    print(f"  Correlation (AA frequency vs prime fraction): {aa_analysis['freq_primefrac_correlation']:.3f}")
    print(f"  P-value: {aa_analysis['correlation_pvalue']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n  Q: Is prime enrichment just a small-number artifact?")
    if range_analysis.get('medium') and range_analysis.get('large'):
        med_enrich = range_analysis['medium']['enrichment']
        large_enrich = range_analysis.get('large', {}).get('enrichment', 0)
        print(f"  A: Medium gaps (11-30): {med_enrich:.2f}x enrichment")
        if large_enrich > 0:
            print(f"     Large gaps (31-100): {large_enrich:.2f}x enrichment")
        if med_enrich > 1.1:
            print("     → Enrichment persists at larger scales ✅")
        else:
            print("     → Enrichment may be small-number artifact ⚠️")
    
    print("\n  Q: Does shuffling destroy the pattern?")
    if shuffled_results:
        avg_z = np.mean([r['z_score'] for r in shuffled_results.values()])
        print(f"  A: Average z-score vs shuffled: {avg_z:.2f}")
        if avg_z > 2:
            print("     → Sequence order matters ✅")
        else:
            print("     → Pattern survives shuffling (composition-driven) ⚠️")
    
    print("\n  Q: Do random sequences show the same pattern?")
    print(f"  A: Real proteins: {real_overall:.4f}, Random: {random_analysis['mean_prime_frac']:.4f}")
    print(f"     Z-score: {z_vs_random:.2f}")
    if z_vs_random > 2:
        print("     → Real proteins are special ✅")
    else:
        print("     → Pattern may be composition artifact ⚠️")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_02_deep_analysis_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    run_experiment()
