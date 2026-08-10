"""
Experiment 12b: Contact Distance Analysis (PDB approach)
=========================================================

Using PDB structures directly instead of AlphaFold.
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


def fetch_pdb_structure(pdb_id):
    """
    Fetch structure from RCSB PDB.
    Returns list of CA (alpha carbon) coordinates.
    """
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        # Parse CA atoms (first chain only)
        ca_coords = []
        seen_resnum = set()
        
        for line in pdb_data.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                try:
                    chain = line[21]
                    resnum = int(line[22:26])
                    
                    # Only first chain
                    if chain != 'A':
                        continue
                    
                    # Skip duplicates
                    if resnum in seen_resnum:
                        continue
                    seen_resnum.add(resnum)
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append(np.array([x, y, z]))
                except:
                    continue
        
        return ca_coords
    
    except Exception as e:
        print(f"    Failed: {e}")
        return None


def compute_contact_map(coords, threshold=8.0):
    """Compute contact map."""
    n = len(coords)
    contacts = []
    
    for i in range(n):
        for j in range(i + 5, n):
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                seq_dist = j - i
                contacts.append({
                    'seq_distance': seq_dist,
                    '3d_distance': dist_3d,
                })
    
    return contacts


def analyze_contact_distances(contacts):
    """Analyze sequence distances of 3D contacts."""
    if not contacts:
        return None
    
    seq_dists = [c['seq_distance'] for c in contacts]
    
    prime_count = sum(1 for d in seq_dists if d in PRIMES)
    fib_count = sum(1 for d in seq_dists if d in FIBONACCI)
    
    max_dist = max(seq_dists)
    primes_in_range = len([p for p in PRIMES if 5 <= p <= max_dist])
    fibs_in_range = len([f for f in FIBONACCI if 5 <= f <= max_dist])
    
    n_possible = max_dist - 4
    expected_prime = len(seq_dists) * primes_in_range / n_possible if n_possible > 0 else 0
    expected_fib = len(seq_dists) * fibs_in_range / n_possible if n_possible > 0 else 0
    
    dist_counts = defaultdict(int)
    for d in seq_dists:
        dist_counts[d] += 1
    
    top_dists = sorted(dist_counts.items(), key=lambda x: -x[1])[:20]
    
    return {
        'n_contacts': len(contacts),
        'mean_seq_distance': np.mean(seq_dists),
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


def run_experiment():
    """Analyze contact distances from PDB structures."""
    print("=" * 60)
    print("Experiment 12b: 3D Contact Distance Analysis (PDB)")
    print("=" * 60)
    
    # Well-characterized protein structures
    proteins = [
        ("Ubiquitin", "1UBQ"),
        ("Lysozyme", "1LYZ"),
        ("Myoglobin", "1MBN"),
        ("Cytochrome_c", "1HRC"),
        ("Insulin", "4INS"),
        ("Hemoglobin", "1HHO"),
        ("Ribonuclease", "7RSA"),
        ("Crambin", "1CRN"),
    ]
    
    all_contacts = []
    results = {'timestamp': datetime.now().isoformat()}
    
    for name, pdb_id in proteins:
        print(f"\n[{name}] Fetching {pdb_id}...")
        coords = fetch_pdb_structure(pdb_id)
        
        if coords and len(coords) > 20:
            print(f"  Got {len(coords)} residues")
            
            contacts = compute_contact_map(coords, threshold=8.0)
            print(f"  Found {len(contacts)} contacts")
            
            all_contacts.extend(contacts)
            
            analysis = analyze_contact_distances(contacts)
            if analysis:
                print(f"  Fibonacci enrichment: {analysis['fib_enrichment']:.2f}x")
    
    if not all_contacts:
        print("\nNo contacts found!")
        return
    
    print("\n" + "=" * 60)
    print("AGGREGATE ANALYSIS")
    print("=" * 60)
    
    agg = analyze_contact_distances(all_contacts)
    results['aggregate'] = agg
    
    print(f"\nTotal contacts: {agg['n_contacts']}")
    print(f"Mean sequence distance: {agg['mean_seq_distance']:.1f}")
    
    print(f"\nFibonacci: {agg['fib_count']}/{agg['n_contacts']} = {agg['fib_enrichment']:.2f}x enrichment")
    print(f"Prime: {agg['prime_count']}/{agg['n_contacts']} = {agg['prime_enrichment']:.2f}x enrichment")
    
    print(f"\nTop 20 contact distances:")
    for d, count in agg['top_distances']:
        markers = []
        if d in PRIMES:
            markers.append("P")
        if d in FIBONACCI:
            markers.append("F")
        marker_str = f" ← {','.join(markers)}" if markers else ""
        print(f"  {d:3d}: {count:4d}{marker_str}")
    
    # Statistical test
    seq_dists = [c['seq_distance'] for c in all_contacts]
    max_dist = max(seq_dists)
    
    np.random.seed(42)
    random_fib = []
    random_prime = []
    for _ in range(1000):
        rand_dists = np.random.randint(5, max_dist + 1, size=len(seq_dists))
        random_fib.append(sum(1 for d in rand_dists if d in FIBONACCI))
        random_prime.append(sum(1 for d in rand_dists if d in PRIMES))
    
    z_fib = (agg['fib_count'] - np.mean(random_fib)) / np.std(random_fib)
    z_prime = (agg['prime_count'] - np.mean(random_prime)) / np.std(random_prime)
    
    print(f"\nZ-scores:")
    print(f"  Fibonacci: {z_fib:.2f}")
    print(f"  Prime: {z_prime:.2f}")
    
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if z_fib > 2:
        print(f"  Fibonacci enrichment SIGNIFICANT ✅ (z={z_fib:.1f})")
    elif z_fib > 1:
        print(f"  Fibonacci trend (weak) (z={z_fib:.1f})")
    else:
        print(f"  No Fibonacci signal (z={z_fib:.1f})")
    
    if z_prime > 2:
        print(f"  Prime enrichment SIGNIFICANT ✅ (z={z_prime:.1f})")
    else:
        print(f"  No Prime signal (z={z_prime:.1f})")
    
    # Save
    results['z_scores'] = {'fibonacci': z_fib, 'prime': z_prime}
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_12b_contacts_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_experiment()
