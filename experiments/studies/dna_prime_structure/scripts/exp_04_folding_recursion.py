"""
Experiment 04: Protein Folding as PAC Recursion
================================================

Hypothesis: Protein folding is recursive (PAC-like), so fold structure
should show Fibonacci/φ signatures rather than linear prime gaps.

Test:
1. Secondary structure element lengths (helix, sheet, coil)
2. φ ratios in structural element size distributions
3. Fibonacci clustering in element lengths
4. Contact distance distributions in 3D structure
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

# Constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618...
PHI_INV = 1 / PHI  # 0.618...
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# Primes for comparison
def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(1000)


def download_dssp_annotations():
    """
    Download secondary structure assignments for several proteins.
    Using UniProt feature annotations.
    
    Returns dict of {protein: [(start, end, type), ...]}
    """
    # Well-characterized proteins with known secondary structure
    # Format: (name, uniprot_id)
    proteins = [
        ("Hemoglobin_alpha", "P69905"),
        ("Myoglobin", "P02144"),
        ("Lysozyme", "P61626"),
        ("Cytochrome_c", "P99999"),
        ("Ubiquitin", "P0CG48"),
        ("Insulin", "P01308"),
        ("Actin", "P60709"),
        ("Tubulin_alpha", "Q71U36"),
    ]
    
    all_structures = {}
    
    for name, uniprot_id in proteins:
        try:
            # Fetch UniProt entry in text format to get features
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode('utf-8')
            
            # Parse FT (feature) lines for secondary structure
            structures = []
            for line in text.split('\n'):
                if line.startswith('FT   HELIX') or line.startswith('FT   STRAND') or line.startswith('FT   TURN'):
                    parts = line.split()
                    if len(parts) >= 3:
                        # Parse range like "1..10"
                        try:
                            struct_type = parts[1]
                            range_str = parts[2]
                            if '..' in range_str:
                                start, end = range_str.split('..')
                                start = int(start)
                                end = int(end)
                                length = end - start + 1
                                structures.append({
                                    'type': struct_type,
                                    'start': start,
                                    'end': end,
                                    'length': length
                                })
                        except:
                            continue
            
            if structures:
                all_structures[name] = structures
                print(f"  {name}: {len(structures)} secondary structure elements")
            
        except Exception as e:
            print(f"  Failed {name}: {e}")
    
    return all_structures


def analyze_element_lengths(structures: dict) -> dict:
    """
    Analyze if secondary structure element lengths show Fibonacci/φ patterns.
    """
    all_lengths = []
    by_type = defaultdict(list)
    
    for protein, elements in structures.items():
        for elem in elements:
            all_lengths.append(elem['length'])
            by_type[elem['type']].append(elem['length'])
    
    if not all_lengths:
        return None
    
    results = {
        'n_elements': len(all_lengths),
        'mean_length': np.mean(all_lengths),
        'std_length': np.std(all_lengths),
        'by_type': {},
    }
    
    # Analyze Fibonacci proximity
    def nearest_fibonacci(n):
        """Find nearest Fibonacci number."""
        for i, f in enumerate(FIBONACCI):
            if f >= n:
                if i == 0:
                    return f, abs(n - f)
                prev = FIBONACCI[i-1]
                if abs(n - prev) < abs(n - f):
                    return prev, abs(n - prev)
                return f, abs(n - f)
        return FIBONACCI[-1], abs(n - FIBONACCI[-1])
    
    # Compute distances to nearest Fibonacci for real data
    fib_distances = [nearest_fibonacci(l)[1] for l in all_lengths]
    mean_fib_dist = np.mean(fib_distances)
    
    # Compare to random baseline
    np.random.seed(42)
    random_lengths = np.random.randint(3, max(all_lengths)+1, size=len(all_lengths)*100)
    random_fib_distances = [nearest_fibonacci(l)[1] for l in random_lengths]
    random_mean_fib_dist = np.mean(random_fib_distances)
    random_std_fib_dist = np.std([np.mean([nearest_fibonacci(l)[1] for l in np.random.choice(random_lengths, len(all_lengths))]) for _ in range(100)])
    
    z_score = (mean_fib_dist - random_mean_fib_dist) / random_std_fib_dist if random_std_fib_dist > 0 else 0
    
    results['fibonacci_analysis'] = {
        'mean_distance_to_fibonacci': mean_fib_dist,
        'random_mean_distance': random_mean_fib_dist,
        'z_score': z_score,
        'closer_than_random': mean_fib_dist < random_mean_fib_dist,
    }
    
    # Count exact Fibonacci matches
    exact_fib = sum(1 for l in all_lengths if l in FIBONACCI)
    expected_fib = len(all_lengths) * len([f for f in FIBONACCI if f <= max(all_lengths)]) / max(all_lengths)
    
    results['fibonacci_analysis']['exact_matches'] = exact_fib
    results['fibonacci_analysis']['expected_matches'] = expected_fib
    results['fibonacci_analysis']['enrichment'] = exact_fib / expected_fib if expected_fib > 0 else 0
    
    # Length distribution
    results['length_distribution'] = {
        'min': min(all_lengths),
        'max': max(all_lengths),
        'median': np.median(all_lengths),
        'mode': int(stats.mode(all_lengths, keepdims=True).mode[0]),
    }
    
    # Top lengths
    length_counts = defaultdict(int)
    for l in all_lengths:
        length_counts[l] += 1
    results['top_lengths'] = sorted(length_counts.items(), key=lambda x: -x[1])[:10]
    
    # By structure type
    for stype, lengths in by_type.items():
        if lengths:
            results['by_type'][stype] = {
                'count': len(lengths),
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'mode': int(stats.mode(lengths, keepdims=True).mode[0]) if lengths else None,
            }
    
    return results


def analyze_phi_ratios(structures: dict) -> dict:
    """
    Look for φ ratios in consecutive element lengths.
    """
    all_ratios = []
    
    for protein, elements in structures.items():
        # Sort by position
        sorted_elems = sorted(elements, key=lambda x: x['start'])
        
        # Compute ratios of consecutive element lengths
        for i in range(len(sorted_elems) - 1):
            l1 = sorted_elems[i]['length']
            l2 = sorted_elems[i+1]['length']
            if l1 > 0 and l2 > 0:
                ratio = max(l1, l2) / min(l1, l2)
                all_ratios.append(ratio)
    
    if not all_ratios:
        return None
    
    # Distance from φ
    phi_distances = [abs(r - PHI) for r in all_ratios]
    mean_phi_dist = np.mean(phi_distances)
    
    # Compare to random
    np.random.seed(42)
    random_ratios = np.random.uniform(1, 5, size=len(all_ratios)*100)
    random_phi_distances = [abs(r - PHI) for r in random_ratios]
    random_mean = np.mean(random_phi_distances)
    random_std = np.std([np.mean([abs(r - PHI) for r in np.random.choice(random_ratios, len(all_ratios))]) for _ in range(100)])
    
    z_score = (mean_phi_dist - random_mean) / random_std if random_std > 0 else 0
    
    # Count ratios close to φ (within 10%)
    phi_close = sum(1 for r in all_ratios if abs(r - PHI) / PHI < 0.1)
    
    return {
        'n_ratios': len(all_ratios),
        'mean_ratio': np.mean(all_ratios),
        'std_ratio': np.std(all_ratios),
        'mean_distance_from_phi': mean_phi_dist,
        'random_mean_distance': random_mean,
        'z_score': z_score,
        'closer_to_phi_than_random': mean_phi_dist < random_mean,
        'ratios_within_10pct_of_phi': phi_close,
        'fraction_near_phi': phi_close / len(all_ratios),
    }


def analyze_gap_structure(structures: dict) -> dict:
    """
    Analyze gaps between secondary structure elements.
    These represent loop/coil regions - the "boundaries" in the fold.
    """
    all_gaps = []
    
    for protein, elements in structures.items():
        sorted_elems = sorted(elements, key=lambda x: x['start'])
        
        for i in range(len(sorted_elems) - 1):
            gap = sorted_elems[i+1]['start'] - sorted_elems[i]['end'] - 1
            if gap > 0:
                all_gaps.append(gap)
    
    if not all_gaps:
        return None
    
    # Prime analysis
    prime_gaps = sum(1 for g in all_gaps if g in PRIMES)
    prime_frac = prime_gaps / len(all_gaps)
    
    # Expected prime density
    max_gap = max(all_gaps)
    expected_frac = sum(1 for p in range(2, max_gap+1) if p in PRIMES) / max(1, max_gap-1)
    
    # Fibonacci analysis
    fib_gaps = sum(1 for g in all_gaps if g in FIBONACCI)
    expected_fib = len(all_gaps) * len([f for f in FIBONACCI if f <= max_gap]) / max_gap
    
    return {
        'n_gaps': len(all_gaps),
        'mean_gap': np.mean(all_gaps),
        'std_gap': np.std(all_gaps),
        'prime_fraction': prime_frac,
        'expected_prime': expected_frac,
        'prime_enrichment': prime_frac / expected_frac if expected_frac > 0 else 0,
        'fibonacci_matches': fib_gaps,
        'expected_fibonacci': expected_fib,
        'fibonacci_enrichment': fib_gaps / expected_fib if expected_fib > 0 else 0,
    }


def run_experiment():
    """Run protein folding analysis."""
    print("=" * 60)
    print("Experiment 04: Protein Folding as PAC Recursion")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Protein folding is recursive (PAC), showing Fibonacci/φ in structure',
        'tests': {}
    }
    
    # Download structure data
    print("\n[1] Downloading secondary structure annotations...")
    structures = download_dssp_annotations()
    
    if not structures:
        print("ERROR: No structure data obtained")
        return
    
    print(f"\n  Total proteins: {len(structures)}")
    total_elements = sum(len(v) for v in structures.values())
    print(f"  Total structure elements: {total_elements}")
    
    # Analyze element lengths
    print("\n[2] Analyzing secondary structure element lengths...")
    length_analysis = analyze_element_lengths(structures)
    results['tests']['element_lengths'] = length_analysis
    
    if length_analysis:
        print(f"\n  Mean element length: {length_analysis['mean_length']:.1f} residues")
        print(f"  Mode: {length_analysis['length_distribution']['mode']}")
        print(f"  Top 5 lengths: {length_analysis['top_lengths'][:5]}")
        
        fib = length_analysis['fibonacci_analysis']
        print(f"\n  FIBONACCI ANALYSIS:")
        print(f"    Mean distance to nearest Fib: {fib['mean_distance_to_fibonacci']:.2f}")
        print(f"    Random baseline distance:     {fib['random_mean_distance']:.2f}")
        print(f"    Z-score: {fib['z_score']:.2f}")
        print(f"    Closer than random: {'YES ✅' if fib['closer_than_random'] else 'NO ❌'}")
        print(f"    Exact Fibonacci matches: {fib['exact_matches']} (expected: {fib['expected_matches']:.1f})")
        print(f"    Fibonacci enrichment: {fib['enrichment']:.2f}x")
    
    # Analyze φ ratios
    print("\n[3] Analyzing φ ratios between consecutive elements...")
    phi_analysis = analyze_phi_ratios(structures)
    results['tests']['phi_ratios'] = phi_analysis
    
    if phi_analysis:
        print(f"\n  Mean ratio: {phi_analysis['mean_ratio']:.3f} (φ = 1.618)")
        print(f"  Mean distance from φ: {phi_analysis['mean_distance_from_phi']:.3f}")
        print(f"  Random baseline:      {phi_analysis['random_mean_distance']:.3f}")
        print(f"  Z-score: {phi_analysis['z_score']:.2f}")
        print(f"  Closer to φ than random: {'YES ✅' if phi_analysis['closer_to_phi_than_random'] else 'NO ❌'}")
        print(f"  Ratios within 10% of φ: {phi_analysis['ratios_within_10pct_of_phi']} ({phi_analysis['fraction_near_phi']:.1%})")
    
    # Analyze gaps (loop regions)
    print("\n[4] Analyzing loop/coil gaps (recursion boundaries)...")
    gap_analysis = analyze_gap_structure(structures)
    results['tests']['gap_structure'] = gap_analysis
    
    if gap_analysis:
        print(f"\n  Mean gap (loop length): {gap_analysis['mean_gap']:.1f}")
        print(f"  Prime enrichment: {gap_analysis['prime_enrichment']:.2f}x")
        print(f"  Fibonacci enrichment: {gap_analysis['fibonacci_enrichment']:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Is protein folding PAC-like?")
    print("=" * 60)
    
    signals = []
    
    if length_analysis and length_analysis['fibonacci_analysis']['closer_than_random']:
        signals.append("Element lengths cluster near Fibonacci ✅")
    else:
        signals.append("Element lengths NOT closer to Fibonacci ❌")
    
    if phi_analysis and phi_analysis['closer_to_phi_than_random']:
        signals.append("Consecutive element ratios approach φ ✅")
    else:
        signals.append("Consecutive ratios NOT closer to φ ❌")
    
    if gap_analysis and gap_analysis['fibonacci_enrichment'] > 1.2:
        signals.append(f"Loop lengths show Fibonacci enrichment ({gap_analysis['fibonacci_enrichment']:.2f}x) ✅")
    else:
        signals.append("Loop lengths show no Fibonacci enrichment ❌")
    
    for s in signals:
        print(f"  {s}")
    
    positive = sum(1 for s in signals if '✅' in s)
    print(f"\n  PAC signals: {positive}/{len(signals)}")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_04_folding_recursion_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    run_experiment()
