"""
Experiment 12: Contact Distance Analysis
=========================================

The 3D folded structure is where recursion should really show.
In a folded protein, amino acids that are far apart in sequence
can be close in 3D space (contacts).

Hypothesis: Contact distances should show Fibonacci/φ patterns
because the folding recursion creates these spatial relationships.

Using AlphaFold predicted structures via UniProt.
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request
import gzip
from io import BytesIO

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


def fetch_alphafold_structure(uniprot_id):
    """
    Fetch AlphaFold predicted structure from EBI.
    Returns list of CA (alpha carbon) coordinates.
    """
    try:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        # Parse CA atoms
        ca_coords = []
        for line in pdb_data.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append(np.array([x, y, z]))
                except:
                    continue
        
        return ca_coords
    
    except Exception as e:
        print(f"    Failed to fetch {uniprot_id}: {e}")
        return None


def compute_contact_map(coords, threshold=8.0):
    """
    Compute contact map: which residues are within threshold distance in 3D?
    Returns list of (sequence_distance, 3d_distance) for contacts.
    """
    n = len(coords)
    contacts = []
    
    for i in range(n):
        for j in range(i + 5, n):  # Skip nearby residues (always close)
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                seq_dist = j - i
                contacts.append({
                    'seq_distance': seq_dist,
                    '3d_distance': dist_3d,
                    'i': i,
                    'j': j
                })
    
    return contacts


def analyze_contact_distances(contacts):
    """Analyze sequence distances of 3D contacts."""
    if not contacts:
        return None
    
    seq_dists = [c['seq_distance'] for c in contacts]
    
    # Prime analysis
    prime_count = sum(1 for d in seq_dists if d in PRIMES)
    
    # Fibonacci analysis
    fib_count = sum(1 for d in seq_dists if d in FIBONACCI)
    
    # Expected fractions
    max_dist = max(seq_dists)
    primes_in_range = len([p for p in PRIMES if 5 <= p <= max_dist])
    fibs_in_range = len([f for f in FIBONACCI if 5 <= f <= max_dist])
    
    n_possible = max_dist - 4  # Distances 5 to max
    expected_prime = len(seq_dists) * primes_in_range / n_possible
    expected_fib = len(seq_dists) * fibs_in_range / n_possible
    
    # Distribution
    dist_counts = defaultdict(int)
    for d in seq_dists:
        dist_counts[d] += 1
    
    top_dists = sorted(dist_counts.items(), key=lambda x: -x[1])[:20]
    
    return {
        'n_contacts': len(contacts),
        'mean_seq_distance': np.mean(seq_dists),
        'median_seq_distance': np.median(seq_dists),
        'prime_count': prime_count,
        'expected_prime': expected_prime,
        'prime_enrichment': prime_count / expected_prime if expected_prime > 0 else 0,
        'fib_count': fib_count,
        'expected_fib': expected_fib,
        'fib_enrichment': fib_count / expected_fib if expected_fib > 0 else 0,
        'top_distances': top_dists,
        'primes_in_top20': [d for d, c in top_dists if d in PRIMES],
        'fibs_in_top20': [d for d, c in top_dists if d in FIBONACCI],
    }


def analyze_contact_ratios(contacts):
    """Look for φ ratios in contact distance relationships."""
    if len(contacts) < 10:
        return None
    
    seq_dists = sorted(set(c['seq_distance'] for c in contacts))
    
    # Compute ratios of consecutive popular distances
    ratios = []
    for i in range(len(seq_dists) - 1):
        d1 = seq_dists[i]
        d2 = seq_dists[i+1]
        if d1 > 0:
            ratios.append(d2 / d1)
    
    phi_distances = [abs(r - PHI) for r in ratios]
    
    return {
        'n_ratios': len(ratios),
        'mean_ratio': np.mean(ratios),
        'mean_phi_distance': np.mean(phi_distances),
        'near_phi_count': sum(1 for d in phi_distances if d < 0.2),
    }


def run_experiment():
    """Analyze contact distances in 3D protein structures."""
    print("=" * 60)
    print("Experiment 12: 3D Contact Distance Analysis")
    print("=" * 60)
    
    results = {'timestamp': datetime.now().isoformat()}
    
    # Proteins to analyze
    proteins = [
        ("Ubiquitin", "P0CG48"),
        ("Lysozyme", "P61626"),
        ("Myoglobin", "P02144"),
        ("Cytochrome_c", "P99999"),
        ("Calmodulin", "P0DP23"),
        ("Insulin", "P01308"),
    ]
    
    all_contacts = []
    protein_results = {}
    
    for name, uniprot_id in proteins:
        print(f"\n[{name}] Fetching structure...")
        coords = fetch_alphafold_structure(uniprot_id)
        
        if coords and len(coords) > 20:
            print(f"  Got {len(coords)} residues")
            
            contacts = compute_contact_map(coords, threshold=8.0)
            print(f"  Found {len(contacts)} contacts (8Å threshold)")
            
            all_contacts.extend(contacts)
            
            analysis = analyze_contact_distances(contacts)
            if analysis:
                protein_results[name] = analysis
                print(f"  Mean seq distance: {analysis['mean_seq_distance']:.1f}")
                print(f"  Fibonacci enrichment: {analysis['fib_enrichment']:.2f}x")
                print(f"  Prime enrichment: {analysis['prime_enrichment']:.2f}x")
    
    if not all_contacts:
        print("\nNo contacts found!")
        return
    
    # Aggregate analysis
    print("\n" + "=" * 60)
    print("AGGREGATE ANALYSIS (all proteins)")
    print("=" * 60)
    
    agg_analysis = analyze_contact_distances(all_contacts)
    results['aggregate'] = agg_analysis
    
    print(f"\nTotal contacts: {agg_analysis['n_contacts']}")
    print(f"Mean sequence distance: {agg_analysis['mean_seq_distance']:.1f}")
    print(f"Median sequence distance: {agg_analysis['median_seq_distance']:.1f}")
    
    print(f"\nPrime enrichment: {agg_analysis['prime_enrichment']:.2f}x")
    print(f"  ({agg_analysis['prime_count']} observed, {agg_analysis['expected_prime']:.0f} expected)")
    
    print(f"\nFibonacci enrichment: {agg_analysis['fib_enrichment']:.2f}x")
    print(f"  ({agg_analysis['fib_count']} observed, {agg_analysis['expected_fib']:.0f} expected)")
    
    print(f"\nTop 20 contact distances:")
    for d, count in agg_analysis['top_distances']:
        markers = []
        if d in PRIMES:
            markers.append("P")
        if d in FIBONACCI:
            markers.append("F")
        marker_str = f" ← {','.join(markers)}" if markers else ""
        print(f"  {d:3d}: {count:4d}{marker_str}")
    
    print(f"\nPrimes in top 20: {agg_analysis['primes_in_top20']}")
    print(f"Fibonacci in top 20: {agg_analysis['fibs_in_top20']}")
    
    # Statistical test
    print("\n" + "=" * 60)
    print("STATISTICAL TEST")
    print("=" * 60)
    
    seq_dists = [c['seq_distance'] for c in all_contacts]
    
    # Monte Carlo: random distances
    np.random.seed(42)
    n_mc = 1000
    max_dist = max(seq_dists)
    
    random_fib_counts = []
    random_prime_counts = []
    for _ in range(n_mc):
        random_dists = np.random.randint(5, max_dist + 1, size=len(seq_dists))
        random_fib_counts.append(sum(1 for d in random_dists if d in FIBONACCI))
        random_prime_counts.append(sum(1 for d in random_dists if d in PRIMES))
    
    # Z-scores
    z_fib = (agg_analysis['fib_count'] - np.mean(random_fib_counts)) / np.std(random_fib_counts)
    z_prime = (agg_analysis['prime_count'] - np.mean(random_prime_counts)) / np.std(random_prime_counts)
    
    print(f"\nFibonacci: observed={agg_analysis['fib_count']}, random={np.mean(random_fib_counts):.0f}±{np.std(random_fib_counts):.0f}")
    print(f"  Z-score: {z_fib:.2f}")
    
    print(f"\nPrime: observed={agg_analysis['prime_count']}, random={np.mean(random_prime_counts):.0f}±{np.std(random_prime_counts):.0f}")
    print(f"  Z-score: {z_prime:.2f}")
    
    results['z_scores'] = {'fibonacci': z_fib, 'prime': z_prime}
    
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if z_fib > 2:
        print(f"  Contact distances show Fibonacci enrichment ✅ (z={z_fib:.1f})")
    else:
        print(f"  No significant Fibonacci enrichment (z={z_fib:.1f})")
    
    if z_prime > 2:
        print(f"  Contact distances show Prime enrichment ✅ (z={z_prime:.1f})")
    else:
        print(f"  No significant Prime enrichment (z={z_prime:.1f})")
    
    # Save
    results['protein_results'] = {k: v for k, v in protein_results.items()}
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_12_contacts_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_experiment()
