"""
Experiment 18: RNA Secondary Structure
=======================================

RNA folds differently than protein:
- Base pairing (A-U, G-C, G-U wobble)
- Helices are A-form (not α-helix)
- Tertiary interactions (pseudoknots, kissing loops)

If Fibonacci patterns appear in RNA contact distances,
it suggests the pattern is about information organization,
not protein-specific physics.

We analyze:
1. rRNA structures (ribosomal RNA)
2. tRNA structures
3. Ribozymes (catalytic RNA)
4. Riboswitches (regulatory RNA)
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
RNA_HELIX = set([11, 22, 33, 44])  # A-form helix ~11 bp per turn

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(500)


def fetch_rna_structures(rna_type='all', max_structures=150):
    """Fetch RNA PDB structures."""
    print(f"Fetching RNA structures ({rna_type})...")
    
    keywords = {
        'all': 'RNA',
        'rrna': 'RIBOSOMAL RNA',
        'trna': 'TRANSFER RNA',
        'ribozyme': 'RIBOZYME',
        'riboswitch': 'RIBOSWITCH',
    }
    
    keyword = keywords.get(rna_type, 'RNA')
    
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
                        "value": keyword
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "greater",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 4.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_structures * 3}
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
            print(f"  Found {len(pdb_ids)} RNA structures")
            return pdb_ids
    except Exception as e:
        print(f"  Query failed: {e}")
        # Fallback
        return [
            "1EHZ", "1EVV", "1FFK", "1GID", "1HR2", "1J5E", "1JJ2", "1KXK",
            "1L8V", "1M82", "1NKW", "1Q9A", "1S72", "1U8D", "1VQO", "1X8W",
            "1YIJ", "2AW4", "2GDI", "2GIS", "2HO7", "2OEU", "2QBZ", "2Y9A",
            "3DIG", "3G78", "3IWN", "3OFC", "3PDR", "3U5F", "4GXY", "4P5J",
            "4TNA", "4V42", "4V88", "4Y4O", "5AJ3", "5IT9", "5JUO", "5TBW",
            "6C4H", "6GWT", "6QZP", "6TNU", "6UES", "6YHS", "7A09", "7K00",
        ]


def fetch_rna_pdb(pdb_id):
    """
    Fetch RNA structure - look for phosphate atoms (P) as backbone marker.
    Returns coordinates of P atoms for each nucleotide.
    """
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        p_coords = []
        seen_resnum = set()
        
        for line in pdb_data.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                
                # Phosphate backbone atom
                if atom_name == 'P':
                    try:
                        chain = line[21]
                        resnum = int(line[22:26])
                        resname = line[17:20].strip()
                        
                        # Check it's an RNA nucleotide
                        if resname not in ['A', 'U', 'G', 'C', 'DA', 'DU', 'DG', 'DC', 
                                           'ADE', 'URA', 'GUA', 'CYT', 'RA', 'RU', 'RG', 'RC']:
                            continue
                        
                        key = (chain, resnum)
                        if key in seen_resnum:
                            continue
                        seen_resnum.add(key)
                        
                        # Only first chain with substantial length
                        if p_coords and chain != p_coords[0].get('chain', chain):
                            continue
                        
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        p_coords.append({
                            'chain': chain,
                            'resnum': resnum,
                            'coords': np.array([x, y, z])
                        })
                    except:
                        continue
        
        if len(p_coords) < 20:
            return None
        
        return [p['coords'] for p in sorted(p_coords, key=lambda x: x['resnum'])]
    
    except Exception as e:
        return None


def compute_contacts(coords, threshold=15.0):
    """
    Compute contacts in RNA. 
    Using larger threshold (15Å) because RNA is less compact than protein.
    """
    n = len(coords)
    distances = []
    
    for i in range(n):
        for j in range(i + 4, n):  # Minimum 4 nucleotides apart
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                distances.append(j - i)
    
    return distances


def analyze_enrichment(distances, target_set, name, max_dist=200):
    """Calculate enrichment."""
    if len(distances) < 30:
        return None
    
    distances = [d for d in distances if 4 <= d <= max_dist]
    if len(distances) < 30:
        return None
    
    observed = sum(1 for d in distances if d in target_set)
    targets_in_range = len([t for t in target_set if 4 <= t <= max_dist])
    expected = len(distances) * targets_in_range / (max_dist - 3)
    enrichment = observed / expected if expected > 0 else 0
    
    np.random.seed(42)
    null_counts = []
    for _ in range(500):
        rand_dists = np.random.randint(4, max_dist + 1, size=len(distances))
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


def run_experiment(max_structures=120):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 18: RNA Secondary Structure Analysis")
    print("=" * 70)
    
    pdb_ids = fetch_rna_structures('all', max_structures)
    
    all_distances = []
    processed = 0
    
    print("\nProcessing RNA structures...")
    for i, pdb_id in enumerate(pdb_ids):
        if processed >= max_structures:
            break
        
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}] Processed {processed}")
        
        coords = fetch_rna_pdb(pdb_id)
        if coords is None:
            continue
        
        if len(coords) < 20 or len(coords) > 1000:
            continue
        
        distances = compute_contacts(coords)
        if len(distances) >= 10:
            all_distances.extend(distances)
            processed += 1
        
        time.sleep(0.1)
    
    print(f"\nProcessed {processed} RNA structures")
    print(f"Total contacts: {len(all_distances)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'molecule_type': 'RNA',
        'n_structures': processed,
        'n_contacts': len(all_distances),
        'analyses': {},
    }
    
    print("\n" + "=" * 70)
    print("ENRICHMENT ANALYSIS")
    print("=" * 70)
    
    targets = [
        ('Fibonacci (all)', FIBONACCI),
        ('Far-Fibonacci', FAR_FIBONACCI),
        ('RNA A-helix (~11bp)', RNA_HELIX),
        ('Primes', PRIMES),
    ]
    
    for name, target in targets:
        analysis = analyze_enrichment(all_distances, target, name)
        if analysis:
            sig = "✅" if analysis['z_score'] > 3 else "❌"
            print(f"  {name:25s}: {analysis['enrichment']:5.2f}x (z={analysis['z_score']:6.1f}) {sig}")
            results['analyses'][name] = analysis
        else:
            print(f"  {name:25s}: insufficient data")
    
    # Distance distribution
    print("\n  Top 20 contact distances:")
    dist_counts = defaultdict(int)
    for d in all_distances:
        dist_counts[d] += 1
    
    top_dists = sorted(dist_counts.items(), key=lambda x: -x[1])[:20]
    for d, count in top_dists:
        markers = []
        if d in FIBONACCI:
            markers.append("F")
        if d in PRIMES:
            markers.append("P")
        marker_str = f" ← {','.join(markers)}" if markers else ""
        print(f"    {d:3d}: {count:5d}{marker_str}")
    
    results['top_distances'] = [[d, count] for d, count in top_dists]
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    fib_result = results['analyses'].get('Fibonacci (all)', {})
    fib_sig = fib_result.get('z_score', 0) > 3
    
    far_fib_result = results['analyses'].get('Far-Fibonacci', {})
    far_fib_sig = far_fib_result.get('z_score', 0) > 3
    
    if fib_sig:
        print(f"\n✅ RNA: Fibonacci enrichment CONFIRMED")
        print(f"   z = {fib_result.get('z_score', 0):.1f}, enrichment = {fib_result.get('enrichment', 0):.2f}x")
        
        if far_fib_sig:
            print(f"   Far-Fibonacci also significant (z={far_fib_result.get('z_score', 0):.1f})")
        
        print("\n   → Pattern exists in RNA (different chemistry than protein)!")
        print("   → Suggests information-level organization, not chemistry-specific")
    else:
        print(f"\n❌ RNA: No significant Fibonacci enrichment")
        print("   → Pattern may be protein-specific")
    
    results['verdict'] = {
        'fib_significant': fib_sig,
        'far_fib_significant': far_fib_sig,
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_18_rna_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {filepath}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--structures', type=int, default=120)
    args = parser.parse_args()
    run_experiment(max_structures=args.structures)
