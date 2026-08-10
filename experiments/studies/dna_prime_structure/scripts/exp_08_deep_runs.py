"""
Experiment 08: Deeper Run Analysis
===================================

The mode of 1 in property runs is hiding the signal.
Let's filter to structurally meaningful runs (length >= 3)
and see if primes/Fibonacci become clearer.

Also: look at the BOUNDARY positions of runs
(where runs start/end might be more informative than length)
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(500)

HYDROPHOBIC = set('AILMFVPWG')
HYDROPHILIC = set('RKDENQHSTY')


def fetch_proteins():
    """Fetch proteins."""
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
        ("Collagen_I", "P02452"),
        ("Keratin", "P04264"),
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
            
        except:
            pass
    
    return all_proteins


def get_runs_with_positions(seq, property_set):
    """
    Get runs with their START and END positions.
    """
    runs = []
    in_run = False
    run_start = 0
    
    for i, aa in enumerate(seq):
        if aa in property_set:
            if not in_run:
                in_run = True
                run_start = i
        else:
            if in_run:
                runs.append({
                    'start': run_start,
                    'end': i - 1,
                    'length': i - run_start
                })
                in_run = False
    
    if in_run:
        runs.append({
            'start': run_start,
            'end': len(seq) - 1,
            'length': len(seq) - run_start
        })
    
    return runs


def analyze_structural_runs(proteins, min_length=3):
    """Focus on runs of length >= min_length (structurally meaningful)."""
    
    hydro_runs = []
    hydro_starts = []
    hydro_ends = []
    
    for name, seq in proteins.items():
        runs = get_runs_with_positions(seq, HYDROPHOBIC)
        for r in runs:
            if r['length'] >= min_length:
                hydro_runs.append(r['length'])
                hydro_starts.append(r['start'] + 1)  # 1-indexed
                hydro_ends.append(r['end'] + 1)
    
    return hydro_runs, hydro_starts, hydro_ends


def prime_fib_analysis(values, label):
    """Analyze a set of values for prime/Fibonacci content."""
    if not values:
        return None
    
    n = len(values)
    max_v = max(values)
    
    # Actual counts
    n_prime = sum(1 for v in values if v in PRIMES)
    n_fib = sum(1 for v in values if v in FIBONACCI)
    
    # Expected under uniform
    primes_in_range = len([p for p in PRIMES if 1 <= p <= max_v])
    fibs_in_range = len([f for f in FIBONACCI if 1 <= f <= max_v])
    
    expected_prime = n * primes_in_range / max_v if max_v > 0 else 0
    expected_fib = n * fibs_in_range / max_v if max_v > 0 else 0
    
    # Enrichment
    prime_enrich = n_prime / expected_prime if expected_prime > 0 else 0
    fib_enrich = n_fib / expected_fib if expected_fib > 0 else 0
    
    # Monte Carlo significance
    np.random.seed(42)
    mc_prime = []
    mc_fib = []
    for _ in range(1000):
        random_vals = np.random.randint(1, max_v + 1, size=n)
        mc_prime.append(sum(1 for v in random_vals if v in PRIMES))
        mc_fib.append(sum(1 for v in random_vals if v in FIBONACCI))
    
    z_prime = (n_prime - np.mean(mc_prime)) / np.std(mc_prime) if np.std(mc_prime) > 0 else 0
    z_fib = (n_fib - np.mean(mc_fib)) / np.std(mc_fib) if np.std(mc_fib) > 0 else 0
    
    return {
        'n': n,
        'max': max_v,
        'mean': np.mean(values),
        'n_prime': n_prime,
        'expected_prime': expected_prime,
        'prime_enrichment': prime_enrich,
        'z_prime': z_prime,
        'n_fib': n_fib,
        'expected_fib': expected_fib,
        'fib_enrichment': fib_enrich,
        'z_fib': z_fib,
    }


def analyze_boundary_gaps(proteins):
    """
    Look at gaps between run boundaries.
    If structure is recursive, boundaries might fall at special positions.
    """
    all_boundaries = []
    
    for name, seq in proteins.items():
        runs = get_runs_with_positions(seq, HYDROPHOBIC)
        # Collect all boundary positions
        for r in runs:
            all_boundaries.append(r['start'] + 1)
            all_boundaries.append(r['end'] + 1)
    
    all_boundaries.sort()
    
    # Gaps between consecutive boundaries
    gaps = []
    for i in range(len(all_boundaries) - 1):
        gap = all_boundaries[i+1] - all_boundaries[i]
        if gap > 0:
            gaps.append(gap)
    
    return gaps


def run_deep_analysis():
    """Run deeper structural analysis."""
    print("=" * 60)
    print("Experiment 08: Deeper Structural Run Analysis")
    print("=" * 60)
    
    print("\n[1] Fetching proteins...")
    proteins = fetch_proteins()
    print(f"  Got {len(proteins)} proteins, {sum(len(s) for s in proteins.values())} total residues")
    
    results = {'timestamp': datetime.now().isoformat()}
    
    # Analyze runs of length >= 3
    print("\n[2] Hydrophobic runs (length >= 3)...")
    runs, starts, ends = analyze_structural_runs(proteins, min_length=3)
    
    print(f"  Found {len(runs)} structural hydrophobic runs")
    
    # Analyze run lengths
    print("\n  RUN LENGTHS:")
    run_analysis = prime_fib_analysis(runs, "run_lengths")
    results['run_lengths'] = run_analysis
    
    print(f"    Mean length: {run_analysis['mean']:.1f}")
    print(f"    Prime: {run_analysis['n_prime']}/{run_analysis['n']} = {run_analysis['prime_enrichment']:.2f}x (z={run_analysis['z_prime']:.2f})")
    print(f"    Fibonacci: {run_analysis['n_fib']}/{run_analysis['n']} = {run_analysis['fib_enrichment']:.2f}x (z={run_analysis['z_fib']:.2f})")
    
    # Length distribution
    length_dist = defaultdict(int)
    for r in runs:
        length_dist[r] += 1
    print(f"    Top lengths: {sorted(length_dist.items(), key=lambda x: -x[1])[:10]}")
    
    # Analyze start positions
    print("\n  START POSITIONS:")
    start_analysis = prime_fib_analysis(starts, "start_positions")
    results['start_positions'] = start_analysis
    
    print(f"    Prime: {start_analysis['n_prime']}/{start_analysis['n']} = {start_analysis['prime_enrichment']:.2f}x (z={start_analysis['z_prime']:.2f})")
    print(f"    Fibonacci: {start_analysis['n_fib']}/{start_analysis['n']} = {start_analysis['fib_enrichment']:.2f}x (z={start_analysis['z_fib']:.2f})")
    
    # Analyze boundary gaps
    print("\n[3] Gaps between hydrophobic region boundaries...")
    gaps = analyze_boundary_gaps(proteins)
    gap_analysis = prime_fib_analysis(gaps, "boundary_gaps")
    results['boundary_gaps'] = gap_analysis
    
    print(f"  Found {len(gaps)} boundary gaps")
    print(f"    Mean gap: {gap_analysis['mean']:.1f}")
    print(f"    Prime: {gap_analysis['n_prime']}/{gap_analysis['n']} = {gap_analysis['prime_enrichment']:.2f}x (z={gap_analysis['z_prime']:.2f})")
    print(f"    Fibonacci: {gap_analysis['n_fib']}/{gap_analysis['n']} = {gap_analysis['fib_enrichment']:.2f}x (z={gap_analysis['z_fib']:.2f})")
    
    # Gap distribution
    gap_dist = defaultdict(int)
    for g in gaps:
        gap_dist[g] += 1
    print(f"    Top gaps: {sorted(gap_dist.items(), key=lambda x: -x[1])[:15]}")
    
    # Mark primes and Fibonacci in top gaps
    top_gaps = sorted(gap_dist.items(), key=lambda x: -x[1])[:15]
    print(f"    Primes in top 15: {[g for g, c in top_gaps if g in PRIMES]}")
    print(f"    Fibonacci in top 15: {[g for g, c in top_gaps if g in FIBONACCI]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINDINGS")
    print("=" * 60)
    
    findings = []
    
    if run_analysis['z_fib'] > 2:
        findings.append(f"Run lengths Fibonacci enriched: z={run_analysis['z_fib']:.1f} ✅")
    elif run_analysis['z_fib'] > 1:
        findings.append(f"Run lengths Fibonacci trend: z={run_analysis['z_fib']:.1f} (weak)")
    
    if gap_analysis['z_fib'] > 2:
        findings.append(f"Boundary gaps Fibonacci enriched: z={gap_analysis['z_fib']:.1f} ✅")
    elif gap_analysis['z_fib'] > 1:
        findings.append(f"Boundary gaps Fibonacci trend: z={gap_analysis['z_fib']:.1f} (weak)")
    
    if gap_analysis['z_prime'] > 2:
        findings.append(f"Boundary gaps Prime enriched: z={gap_analysis['z_prime']:.1f} ✅")
    
    if findings:
        for f in findings:
            print(f"  {f}")
    else:
        print("  No significant enrichment found")
    
    # Check for φ in length ratios
    print("\n[4] Checking for φ in consecutive run length ratios...")
    ratios = []
    for name, seq in proteins.items():
        runs_data = get_runs_with_positions(seq, HYDROPHOBIC)
        runs_len = [r['length'] for r in runs_data if r['length'] >= 3]
        for i in range(len(runs_len) - 1):
            if runs_len[i] > 0 and runs_len[i+1] > 0:
                ratio = max(runs_len[i], runs_len[i+1]) / min(runs_len[i], runs_len[i+1])
                ratios.append(ratio)
    
    if ratios:
        mean_ratio = np.mean(ratios)
        phi_dist = np.mean([abs(r - PHI) for r in ratios])
        near_phi = sum(1 for r in ratios if abs(r - PHI) < 0.2)
        
        print(f"  Mean consecutive ratio: {mean_ratio:.3f} (φ = 1.618)")
        print(f"  Mean distance from φ: {phi_dist:.3f}")
        print(f"  Ratios within 0.2 of φ: {near_phi}/{len(ratios)} ({100*near_phi/len(ratios):.1f}%)")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_08_deep_runs_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_deep_analysis()
