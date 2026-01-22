"""
Experiment 07: Protein Decomposition - Hunting for Primals
============================================================

Exploratory mission: decompose proteins in every way possible
and look for prime/Fibonacci/φ signatures.

Decomposition angles:
1. Domain boundaries (where do folds separate?)
2. Hydrophobic/hydrophilic runs (like 0s and 1s)
3. Charge patterns (+ - neutral runs)
4. Contact distances in 3D (if we can get them)
5. Conservation patterns (which positions are conserved?)
6. Codon usage (back to DNA, but structured differently)
7. Motif lengths and spacings
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

# Constants
PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(1000)

# Amino acid properties
HYDROPHOBIC = set('AILMFVPWG')
HYDROPHILIC = set('RKDENQHSTY')
CHARGED_POS = set('RKH')
CHARGED_NEG = set('DE')
NEUTRAL = set('STYCNQMWFAILGPV')

# Size classes (by molecular weight roughly)
SMALL = set('GASC')
MEDIUM = set('TVPDN')
LARGE = set('QEIMKRLHFYW')


def fetch_proteins():
    """Fetch a set of well-characterized proteins."""
    proteins = [
        ("Hemoglobin_alpha", "P69905"),
        ("Myoglobin", "P02144"),
        ("Lysozyme", "P61626"),
        ("Cytochrome_c", "P99999"),
        ("Ubiquitin", "P0CG48"),
        ("Insulin", "P01308"),
        ("Actin", "P60709"),
        ("Tubulin_alpha", "Q71U36"),
        ("Calmodulin", "P0DP23"),
        ("Ferritin", "P02794"),
    ]
    
    all_proteins = {}
    
    for name, uniprot_id in proteins:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            with urllib.request.urlopen(url, timeout=10) as response:
                fasta = response.read().decode('utf-8')
            
            lines = fasta.strip().split('\n')
            seq = ''.join(lines[1:])
            all_proteins[name] = seq
            print(f"  {name}: {len(seq)} aa")
            
        except Exception as e:
            print(f"  Failed {name}: {e}")
    
    return all_proteins


def analyze_runs(seq, property_set, property_name):
    """
    Analyze runs of amino acids with a given property.
    Returns run lengths and gaps between runs.
    """
    runs = []
    gaps = []
    in_run = False
    run_start = 0
    last_run_end = None
    
    for i, aa in enumerate(seq):
        if aa in property_set:
            if not in_run:
                in_run = True
                run_start = i
                if last_run_end is not None:
                    gaps.append(i - last_run_end - 1)
        else:
            if in_run:
                run_length = i - run_start
                runs.append(run_length)
                last_run_end = i - 1
                in_run = False
    
    # Handle run at end
    if in_run:
        runs.append(len(seq) - run_start)
    
    return runs, gaps


def analyze_property_runs(proteins):
    """Analyze runs of different amino acid properties."""
    results = {}
    
    properties = [
        ('hydrophobic', HYDROPHOBIC),
        ('hydrophilic', HYDROPHILIC),
        ('charged_pos', CHARGED_POS),
        ('charged_neg', CHARGED_NEG),
        ('small', SMALL),
        ('large', LARGE),
    ]
    
    for prop_name, prop_set in properties:
        all_runs = []
        all_gaps = []
        
        for name, seq in proteins.items():
            runs, gaps = analyze_runs(seq, prop_set, prop_name)
            all_runs.extend(runs)
            all_gaps.extend(gaps)
        
        if all_runs:
            # Prime analysis
            prime_runs = sum(1 for r in all_runs if r in PRIMES)
            fib_runs = sum(1 for r in all_runs if r in FIBONACCI)
            
            # Expected fractions
            max_run = max(all_runs) if all_runs else 10
            expected_prime = len([p for p in PRIMES if p <= max_run]) / max_run if max_run > 0 else 0.2
            expected_fib = len([f for f in FIBONACCI if f <= max_run]) / max_run if max_run > 0 else 0.1
            
            results[prop_name] = {
                'n_runs': len(all_runs),
                'mean_run': np.mean(all_runs),
                'mode_run': int(stats.mode(all_runs, keepdims=True).mode[0]) if all_runs else None,
                'prime_frac': prime_runs / len(all_runs),
                'prime_expected': expected_prime,
                'prime_enrichment': (prime_runs / len(all_runs)) / expected_prime if expected_prime > 0 else 0,
                'fib_frac': fib_runs / len(all_runs),
                'fib_expected': expected_fib,
                'fib_enrichment': (fib_runs / len(all_runs)) / expected_fib if expected_fib > 0 else 0,
                'n_gaps': len(all_gaps),
                'mean_gap': np.mean(all_gaps) if all_gaps else 0,
            }
    
    return results


def analyze_domain_like_segments(proteins):
    """
    Look for domain-like segments using hydrophobicity transitions.
    Domains often have hydrophobic cores > look for HH...H patterns.
    """
    all_segment_lengths = []
    
    for name, seq in proteins.items():
        # Simple domain detection: long hydrophobic runs
        runs, gaps = analyze_runs(seq, HYDROPHOBIC, 'hydrophobic')
        
        # Segments between major hydrophobic regions
        for gap in gaps:
            if gap > 5:  # Non-trivial segment
                all_segment_lengths.append(gap)
    
    if not all_segment_lengths:
        return None
    
    prime_count = sum(1 for s in all_segment_lengths if s in PRIMES)
    fib_count = sum(1 for s in all_segment_lengths if s in FIBONACCI)
    
    return {
        'n_segments': len(all_segment_lengths),
        'mean_length': np.mean(all_segment_lengths),
        'prime_frac': prime_count / len(all_segment_lengths),
        'fib_frac': fib_count / len(all_segment_lengths),
        'length_distribution': sorted(set(all_segment_lengths))[:20],
    }


def analyze_charge_patterns(proteins):
    """
    Analyze charge sign runs: +++ --- or 000
    Salt bridges often occur at specific distances.
    """
    all_distances = []
    
    for name, seq in proteins.items():
        pos_positions = [i for i, aa in enumerate(seq) if aa in CHARGED_POS]
        neg_positions = [i for i, aa in enumerate(seq) if aa in CHARGED_NEG]
        
        # Distances between opposite charges (potential salt bridges)
        for p in pos_positions:
            for n in neg_positions:
                dist = abs(p - n)
                if 1 < dist < 50:  # Reasonable salt bridge range
                    all_distances.append(dist)
    
    if not all_distances:
        return None
    
    prime_count = sum(1 for d in all_distances if d in PRIMES)
    fib_count = sum(1 for d in all_distances if d in FIBONACCI)
    
    # Distribution analysis
    dist_counts = defaultdict(int)
    for d in all_distances:
        dist_counts[d] += 1
    
    top_distances = sorted(dist_counts.items(), key=lambda x: -x[1])[:15]
    
    return {
        'n_pairs': len(all_distances),
        'mean_distance': np.mean(all_distances),
        'prime_frac': prime_count / len(all_distances),
        'fib_frac': fib_count / len(all_distances),
        'top_distances': top_distances,
        'primes_in_top10': [d for d, c in top_distances[:10] if d in PRIMES],
        'fibs_in_top10': [d for d, c in top_distances[:10] if d in FIBONACCI],
    }


def analyze_repeat_patterns(proteins):
    """
    Look for repeated motifs and their lengths/spacings.
    """
    results = {}
    
    for name, seq in proteins.items():
        # Find 3-6 aa repeats
        for motif_len in [3, 4, 5, 6]:
            motifs = defaultdict(list)
            for i in range(len(seq) - motif_len + 1):
                motif = seq[i:i+motif_len]
                motifs[motif].append(i)
            
            # Find repeated motifs
            repeated = {m: pos for m, pos in motifs.items() if len(pos) >= 2}
            
            if repeated:
                # Analyze spacings between repeats
                spacings = []
                for motif, positions in repeated.items():
                    for i in range(len(positions) - 1):
                        spacing = positions[i+1] - positions[i]
                        spacings.append(spacing)
                
                if spacings:
                    key = f"{name}_motif{motif_len}"
                    results[key] = {
                        'n_repeated_motifs': len(repeated),
                        'n_spacings': len(spacings),
                        'mean_spacing': np.mean(spacings),
                        'prime_spacings': sum(1 for s in spacings if s in PRIMES),
                        'fib_spacings': sum(1 for s in spacings if s in FIBONACCI),
                    }
    
    return results


def analyze_periodicity(proteins):
    """
    Look for periodic patterns (like alpha helix 3.6 residue period).
    Use autocorrelation.
    """
    results = {}
    
    for name, seq in proteins.items():
        # Convert to hydrophobicity signal
        signal = [1 if aa in HYDROPHOBIC else 0 for aa in seq]
        signal = np.array(signal) - np.mean(signal)
        
        if len(signal) < 20:
            continue
        
        # Autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks (periods)
        peaks = []
        for i in range(2, min(50, len(autocorr) - 1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.1:  # Significant peak
                    peaks.append((i, autocorr[i]))
        
        if peaks:
            top_periods = sorted(peaks, key=lambda x: -x[1])[:5]
            results[name] = {
                'top_periods': top_periods,
                'prime_periods': [p for p, v in top_periods if p in PRIMES],
                'fib_periods': [p for p, v in top_periods if p in FIBONACCI],
            }
    
    return results


def run_exploration():
    """Run the exploratory decomposition."""
    print("=" * 60)
    print("Experiment 07: Protein Decomposition Hunt")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'goal': 'Hunt for prime/Fibonacci signatures in protein structure',
    }
    
    # Fetch proteins
    print("\n[1] Fetching proteins...")
    proteins = fetch_proteins()
    results['n_proteins'] = len(proteins)
    results['total_residues'] = sum(len(s) for s in proteins.values())
    
    # Property runs
    print("\n[2] Analyzing property runs (hydrophobic, charged, etc.)...")
    run_results = analyze_property_runs(proteins)
    results['property_runs'] = run_results
    
    print("\n  Property       | N runs | Mode | Prime enrich | Fib enrich")
    print("  " + "-" * 60)
    for prop, data in run_results.items():
        print(f"  {prop:14} | {data['n_runs']:6} | {data['mode_run']:4} | {data['prime_enrichment']:11.2f}x | {data['fib_enrichment']:.2f}x")
    
    # Domain-like segments
    print("\n[3] Analyzing domain-like segments...")
    domain_results = analyze_domain_like_segments(proteins)
    results['domain_segments'] = domain_results
    
    if domain_results:
        print(f"  Found {domain_results['n_segments']} segments")
        print(f"  Mean length: {domain_results['mean_length']:.1f}")
        print(f"  Prime fraction: {domain_results['prime_frac']:.2%}")
        print(f"  Fibonacci fraction: {domain_results['fib_frac']:.2%}")
    
    # Charge patterns
    print("\n[4] Analyzing charge patterns (salt bridge distances)...")
    charge_results = analyze_charge_patterns(proteins)
    results['charge_patterns'] = charge_results
    
    if charge_results:
        print(f"  Found {charge_results['n_pairs']} +/- pairs")
        print(f"  Mean distance: {charge_results['mean_distance']:.1f}")
        print(f"  Top distances: {[d for d, c in charge_results['top_distances'][:10]]}")
        print(f"  Primes in top 10: {charge_results['primes_in_top10']}")
        print(f"  Fibonacci in top 10: {charge_results['fibs_in_top10']}")
    
    # Repeat patterns
    print("\n[5] Analyzing repeat motif spacings...")
    repeat_results = analyze_repeat_patterns(proteins)
    results['repeat_patterns'] = repeat_results
    
    # Aggregate repeat stats
    if repeat_results:
        total_spacings = sum(r['n_spacings'] for r in repeat_results.values())
        total_prime = sum(r['prime_spacings'] for r in repeat_results.values())
        total_fib = sum(r['fib_spacings'] for r in repeat_results.values())
        print(f"  Total spacings analyzed: {total_spacings}")
        print(f"  Prime spacings: {total_prime} ({100*total_prime/total_spacings:.1f}%)" if total_spacings > 0 else "")
        print(f"  Fibonacci spacings: {total_fib} ({100*total_fib/total_spacings:.1f}%)" if total_spacings > 0 else "")
    
    # Periodicity
    print("\n[6] Analyzing hydrophobicity periodicity...")
    period_results = analyze_periodicity(proteins)
    results['periodicity'] = period_results
    
    if period_results:
        print(f"\n  Protein         | Top periods        | Primes | Fibonacci")
        print("  " + "-" * 60)
        for prot, data in period_results.items():
            periods = [p for p, v in data['top_periods']]
            print(f"  {prot:15} | {str(periods)[:18]:18} | {data['prime_periods']} | {data['fib_periods']}")
    
    # Summary: What patterns emerged?
    print("\n" + "=" * 60)
    print("PATTERN SUMMARY: What showed enrichment?")
    print("=" * 60)
    
    signals = []
    
    # Check property runs
    for prop, data in run_results.items():
        if data['fib_enrichment'] > 1.5:
            signals.append(f"{prop} runs: Fibonacci {data['fib_enrichment']:.1f}x enriched ✅")
        if data['prime_enrichment'] > 1.5:
            signals.append(f"{prop} runs: Prime {data['prime_enrichment']:.1f}x enriched ✅")
    
    # Check charge patterns
    if charge_results:
        n_fib = len(charge_results['fibs_in_top10'])
        n_prime = len(charge_results['primes_in_top10'])
        if n_fib >= 3:
            signals.append(f"Salt bridge distances: {n_fib}/10 top distances are Fibonacci ✅")
        if n_prime >= 3:
            signals.append(f"Salt bridge distances: {n_prime}/10 top distances are prime ✅")
    
    # Check periodicity
    if period_results:
        all_prime_periods = []
        all_fib_periods = []
        for data in period_results.values():
            all_prime_periods.extend(data['prime_periods'])
            all_fib_periods.extend(data['fib_periods'])
        
        if len(all_fib_periods) >= 3:
            signals.append(f"Hydrophobicity periods: {len(all_fib_periods)} Fibonacci periods found ✅")
        if len(all_prime_periods) >= 3:
            signals.append(f"Hydrophobicity periods: {len(all_prime_periods)} prime periods found ✅")
    
    if signals:
        for s in signals:
            print(f"  {s}")
    else:
        print("  No strong enrichment patterns found")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_07_decomposition_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_exploration()
