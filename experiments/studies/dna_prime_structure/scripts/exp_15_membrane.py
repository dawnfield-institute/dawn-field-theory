"""
Experiment 15: Membrane Proteins Control
=========================================

Membrane proteins fold in a lipid bilayer environment with different
constraints than soluble proteins:
- Transmembrane helices are ~20 residues (hydrophobic core)
- Different packing geometry
- Lateral pressure from lipids

If Fibonacci enrichment persists here, it's not specific to soluble protein physics.
"""

import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime
import urllib.request
import time

FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
FAR_FIBONACCI = set([21, 34, 55, 89])

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(500)


def fetch_membrane_proteins(max_proteins=200):
    """Fetch membrane protein PDB IDs from OPM database or RCSB query."""
    print("Fetching membrane protein list...")
    
    # Query RCSB for membrane proteins
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct_keywords.pdbx_keywords",
                        "operator": "contains_words",
                        "value": "MEMBRANE PROTEIN"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 3.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_proteins * 3}
        }
    }
    
    try:
        import json as json_mod
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        req = urllib.request.Request(
            url, data=json_mod.dumps(query).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json_mod.loads(response.read().decode('utf-8'))
            pdb_ids = [hit['identifier'] for hit in data.get('result_set', [])]
            print(f"  Found {len(pdb_ids)} membrane proteins")
            return pdb_ids
    except Exception as e:
        print(f"  Query failed: {e}")
        # Fallback: well-known membrane proteins
        return [
            "1BL8", "1C3W", "1EHK", "1FX8", "1GZM", "1J4N", "1JB0", "1KPL",
            "1MSL", "1OCC", "1OKC", "1P7B", "1PV6", "1RC2", "1RH5", "1U19",
            "1XIO", "2BG9", "2NWL", "2OAR", "2QTS", "2R9R", "2UUI", "2VL0",
            "2WIE", "2X2V", "2ZW3", "3B9W", "3DDL", "3EAM", "3EMN", "3G5U",
            "3KCU", "3KDP", "3M9I", "3MP7", "3ND0", "3QAP", "3RKO", "3SYA",
            "3TIJ", "3UKM", "3V5U", "3WME", "3ZUX", "4B4A", "4DJK", "4DVE",
            "4EIY", "4F35", "4HYO", "4IU8", "4J7C", "4KJS", "4M48", "4MBS",
            "4NTJ", "4O9R", "4OR2", "4PHZ", "4Q2E", "4QND", "4RNG", "4TQU",
            "4U4T", "4UVM", "4XES", "4YAY", "4ZTL", "5AYN", "5C78", "5D0Y",
            "5DGY", "5EE7", "5HK1", "5I6C", "5IWK", "5JQH", "5LWE", "5MRW",
        ]


def fetch_pdb_structure(pdb_id):
    """Fetch PDB structure."""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        ca_coords = []
        seen_resnum = set()
        
        for line in pdb_data.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                try:
                    chain = line[21]
                    resnum = int(line[22:26])
                    
                    if chain != 'A' or resnum in seen_resnum:
                        continue
                    seen_resnum.add(resnum)
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append(np.array([x, y, z]))
                except:
                    continue
        
        return ca_coords if len(ca_coords) >= 30 else None
    except:
        return None


def compute_contacts(coords, threshold=8.0):
    """Compute contact distances."""
    n = len(coords)
    distances = []
    
    for i in range(n):
        for j in range(i + 5, n):
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                distances.append(j - i)
    
    return distances


def analyze_enrichment(distances, target_set, name, max_dist=200):
    """Calculate enrichment."""
    if len(distances) < 50:
        return None
    
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 50:
        return None
    
    observed = sum(1 for d in distances if d in target_set)
    targets_in_range = len([t for t in target_set if 5 <= t <= max_dist])
    expected = len(distances) * targets_in_range / (max_dist - 4)
    enrichment = observed / expected if expected > 0 else 0
    
    np.random.seed(42)
    null_counts = []
    for _ in range(500):
        rand_dists = np.random.randint(5, max_dist + 1, size=len(distances))
        null_counts.append(sum(1 for d in rand_dists if d in target_set))
    
    z_score = (observed - np.mean(null_counts)) / (np.std(null_counts) + 1e-10)
    
    return {
        'name': name,
        'n_distances': len(distances),
        'observed': observed,
        'expected': float(expected),
        'enrichment': float(enrichment),
        'z_score': float(z_score),
    }


def run_experiment(max_proteins=150):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 15: Membrane Proteins Fibonacci Analysis")
    print("=" * 70)
    
    pdb_ids = fetch_membrane_proteins(max_proteins)
    
    all_distances = []
    processed = 0
    
    print("\nProcessing membrane proteins...")
    for i, pdb_id in enumerate(pdb_ids):
        if processed >= max_proteins:
            break
        
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}] Processed {processed}")
        
        coords = fetch_pdb_structure(pdb_id)
        if coords is None:
            continue
        
        if len(coords) < 50 or len(coords) > 800:
            continue
        
        distances = compute_contacts(coords)
        if len(distances) >= 20:
            all_distances.extend(distances)
            processed += 1
        
        time.sleep(0.1)
    
    print(f"\nProcessed {processed} membrane proteins")
    print(f"Total contacts: {len(all_distances)}")
    
    print("\n" + "=" * 70)
    print("ENRICHMENT ANALYSIS")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'protein_type': 'membrane',
        'n_proteins': processed,
        'n_contacts': len(all_distances),
        'analyses': {},
    }
    
    targets = [
        ('Fibonacci (all)', FIBONACCI),
        ('Far-Fibonacci', FAR_FIBONACCI),
        ('Primes', PRIMES),
    ]
    
    for name, target in targets:
        analysis = analyze_enrichment(all_distances, target, name)
        if analysis:
            sig = "✅" if analysis['z_score'] > 3 else "❌"
            print(f"  {name:25s}: {analysis['enrichment']:5.2f}x (z={analysis['z_score']:6.1f}) {sig}")
            results['analyses'][name] = analysis
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    fib_result = results['analyses'].get('Fibonacci (all)', {})
    fib_sig = fib_result.get('z_score', 0) > 3
    
    if fib_sig:
        print(f"\n✅ MEMBRANE PROTEINS: Fibonacci enrichment CONFIRMED")
        print(f"   z = {fib_result.get('z_score', 0):.1f}, enrichment = {fib_result.get('enrichment', 0):.2f}x")
        print("   → Pattern persists despite different folding environment!")
    else:
        print(f"\n❌ Membrane proteins: No significant Fibonacci enrichment")
    
    results['verdict'] = {'fib_significant': fib_sig}
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_15_membrane_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {filepath}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=150)
    args = parser.parse_args()
    run_experiment(max_proteins=args.proteins)
