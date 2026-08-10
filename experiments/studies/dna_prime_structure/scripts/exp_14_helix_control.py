"""
Experiment 14: Fibonacci vs Helix Periodicity Control
======================================================

The exp_13 result showed Fibonacci enrichment (7.2x, z=391) but helix
periodicity showed similar enrichment (8.5x). This experiment directly
tests whether Fibonacci is doing something BEYOND helix periodicity.

Key tests:
1. Contacts in COIL regions only (no helix/sheet bias)
2. Contacts in SHEET regions only (different periodicity)
3. Fibonacci numbers not near helix multiples (21, 34, 55, 89)
4. Non-Fibonacci helix numbers (4, 7, 11, 14, 18) vs pure Fibonacci

If Fibonacci is just proxying for helix periodicity, then:
- Coil regions should show NO Fibonacci enrichment
- Sheet regions should show NO Fibonacci enrichment  
- Far-Fibonacci (21, 34, 55) should show NO enrichment
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request
import time

PHI = (1 + np.sqrt(5)) / 2

# Fibonacci numbers
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])

# Fibonacci far from helix multiples of 3.6
# Helix: 4, 7, 11, 14, 18, 22, 25, 29, 32, 36...
FAR_FIBONACCI = set([21, 34, 55, 89])  # Not close to 3.6*n

# Helix-related but NOT Fibonacci
HELIX_ONLY = set([4, 7, 11, 14, 18, 22, 25, 29, 32, 36, 40, 43, 47, 50])

# Near-Fibonacci helix numbers (Fib ∩ Helix)
OVERLAP = set([8, 13])  # 8 ≈ 3.6*2.2, 13 ≈ 3.6*3.6

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(500)


def fetch_pdb_list(max_proteins=500):
    """Fetch representative PDB list."""
    print("Fetching PDB list...")
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 2.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_proteins * 2}
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
            return [hit['identifier'] for hit in data.get('result_set', [])][:max_proteins * 2]
    except Exception as e:
        print(f"  Query failed: {e}")
        return []


def fetch_pdb_structure(pdb_id):
    """Fetch PDB with secondary structure annotation."""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        ca_coords = []
        residue_info = []
        seen_resnum = set()
        helix_ranges = []
        sheet_ranges = []
        
        for line in pdb_data.split('\n'):
            if line.startswith('HELIX'):
                try:
                    start = int(line[21:25])
                    end = int(line[33:37])
                    helix_ranges.append((start, end))
                except:
                    pass
            elif line.startswith('SHEET'):
                try:
                    start = int(line[22:26])
                    end = int(line[33:37])
                    sheet_ranges.append((start, end))
                except:
                    pass
            
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
                    
                    ss = 'C'
                    for start, end in helix_ranges:
                        if start <= resnum <= end:
                            ss = 'H'
                            break
                    for start, end in sheet_ranges:
                        if start <= resnum <= end:
                            ss = 'E'
                            break
                    
                    ca_coords.append(np.array([x, y, z]))
                    residue_info.append({'resnum': resnum, 'ss': ss})
                except:
                    continue
        
        if len(ca_coords) < 30:
            return None
        
        return {'coords': ca_coords, 'residues': residue_info}
    except:
        return None


def compute_contacts_by_ss(structure, threshold=8.0):
    """Compute contacts categorized by secondary structure."""
    coords = structure['coords']
    residues = structure['residues']
    n = len(coords)
    
    contacts_by_type = {
        'all': [],
        'coil_coil': [],      # Both in coil
        'helix_helix': [],    # Both in helix
        'sheet_sheet': [],    # Both in sheet
    }
    
    for i in range(n):
        for j in range(i + 5, n):
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                seq_dist = j - i
                ss_i, ss_j = residues[i]['ss'], residues[j]['ss']
                
                contacts_by_type['all'].append(seq_dist)
                
                if ss_i == 'C' and ss_j == 'C':
                    contacts_by_type['coil_coil'].append(seq_dist)
                elif ss_i == 'H' and ss_j == 'H':
                    contacts_by_type['helix_helix'].append(seq_dist)
                elif ss_i == 'E' and ss_j == 'E':
                    contacts_by_type['sheet_sheet'].append(seq_dist)
    
    return contacts_by_type


def analyze_enrichment(distances, target_set, name, max_dist=200):
    """Calculate enrichment and z-score for a target set."""
    if len(distances) < 50:
        return None
    
    # Filter to reasonable range
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 50:
        return None
    
    observed = sum(1 for d in distances if d in target_set)
    
    # Expected under uniform null
    targets_in_range = len([t for t in target_set if 5 <= t <= max_dist])
    expected = len(distances) * targets_in_range / (max_dist - 4)
    
    enrichment = observed / expected if expected > 0 else 0
    
    # Z-score via permutation
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


def run_experiment(max_proteins=300):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 14: Fibonacci vs Helix Periodicity Control")
    print("=" * 70)
    
    pdb_ids = fetch_pdb_list(max_proteins)
    print(f"Retrieved {len(pdb_ids)} PDB IDs")
    
    all_contacts = {
        'all': [],
        'coil_coil': [],
        'helix_helix': [],
        'sheet_sheet': [],
    }
    
    processed = 0
    print("\nProcessing structures...")
    
    for i, pdb_id in enumerate(pdb_ids):
        if processed >= max_proteins:
            break
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] Processed {processed}")
        
        structure = fetch_pdb_structure(pdb_id)
        if structure is None:
            continue
        
        if len(structure['coords']) < 50 or len(structure['coords']) > 500:
            continue
        
        contacts = compute_contacts_by_ss(structure)
        for key in all_contacts:
            all_contacts[key].extend(contacts[key])
        
        processed += 1
        time.sleep(0.1)
    
    print(f"\nProcessed {processed} structures")
    for key, dists in all_contacts.items():
        print(f"  {key}: {len(dists)} contacts")
    
    # Analyze each contact type for each target set
    target_sets = [
        ('Fibonacci (all)', FIBONACCI),
        ('Far-Fibonacci (21,34,55,89)', FAR_FIBONACCI),
        ('Helix-only (not Fib)', HELIX_ONLY),
        ('Overlap (Fib ∩ Helix)', OVERLAP),
        ('Primes', PRIMES),
    ]
    
    contact_types = ['all', 'coil_coil', 'helix_helix', 'sheet_sheet']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_proteins': processed,
        'contact_counts': {k: len(v) for k, v in all_contacts.items()},
        'analyses': {},
    }
    
    print("\n" + "=" * 70)
    print("ENRICHMENT ANALYSIS")
    print("=" * 70)
    
    for contact_type in contact_types:
        distances = all_contacts[contact_type]
        print(f"\n[{contact_type.upper()}] ({len(distances)} contacts)")
        
        results['analyses'][contact_type] = {}
        
        for name, target in target_sets:
            analysis = analyze_enrichment(distances, target, name)
            if analysis:
                sig = "✅" if analysis['z_score'] > 3 else "❌"
                print(f"  {name:30s}: {analysis['enrichment']:5.2f}x (z={analysis['z_score']:6.1f}) {sig}")
                results['analyses'][contact_type][name] = analysis
            else:
                print(f"  {name:30s}: insufficient data")
    
    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)
    
    # Test 1: Does Far-Fibonacci show enrichment in coil regions?
    coil_far_fib = results['analyses'].get('coil_coil', {}).get('Far-Fibonacci (21,34,55,89)')
    if coil_far_fib:
        if coil_far_fib['z_score'] > 3:
            print("\n✅ FAR-FIBONACCI in COIL: SIGNIFICANT enrichment")
            print("   → Fibonacci signal exists BEYOND helix periodicity!")
        else:
            print("\n❌ Far-Fibonacci in coil: NOT significant")
            print("   → Fibonacci signal may be explained by helix periodicity")
    
    # Test 2: Far-Fibonacci vs Helix-only in helix regions
    helix_far_fib = results['analyses'].get('helix_helix', {}).get('Far-Fibonacci (21,34,55,89)')
    helix_helix_only = results['analyses'].get('helix_helix', {}).get('Helix-only (not Fib)')
    
    if helix_far_fib and helix_helix_only:
        if helix_far_fib['enrichment'] > helix_helix_only['enrichment'] * 0.5:
            print("\n✅ In helix regions: Far-Fibonacci comparable to helix-specific")
            print(f"   Far-Fib: {helix_far_fib['enrichment']:.2f}x vs Helix-only: {helix_helix_only['enrichment']:.2f}x")
        else:
            print("\n❌ In helix regions: Helix periodicity dominates")
    
    # Test 3: Sheet regions - should have no helix bias
    sheet_fib = results['analyses'].get('sheet_sheet', {}).get('Fibonacci (all)')
    sheet_helix = results['analyses'].get('sheet_sheet', {}).get('Helix-only (not Fib)')
    
    if sheet_fib and sheet_helix:
        print(f"\n📊 SHEET REGIONS (no helix bias):")
        print(f"   Fibonacci: {sheet_fib['enrichment']:.2f}x (z={sheet_fib['z_score']:.1f})")
        print(f"   Helix-only: {sheet_helix['enrichment']:.2f}x (z={sheet_helix['z_score']:.1f})")
        
        if sheet_fib['z_score'] > 3 and sheet_fib['enrichment'] > sheet_helix['enrichment']:
            print("   → ✅ Fibonacci enriched in sheets MORE than helix numbers!")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    coil_all_fib = results['analyses'].get('coil_coil', {}).get('Fibonacci (all)', {})
    sheet_all_fib = results['analyses'].get('sheet_sheet', {}).get('Fibonacci (all)', {})
    
    coil_sig = coil_all_fib.get('z_score', 0) > 3 if coil_all_fib else False
    sheet_sig = sheet_all_fib.get('z_score', 0) > 3 if sheet_all_fib else False
    far_fib_coil_sig = coil_far_fib and coil_far_fib['z_score'] > 3
    
    if coil_sig or sheet_sig or far_fib_coil_sig:
        print("\n🔬 FIBONACCI SIGNAL EXCEEDS HELIX PERIODICITY EXPLANATION")
        if coil_sig:
            print(f"   - Coil-coil contacts: z={coil_all_fib.get('z_score', 0):.1f}")
        if sheet_sig:
            print(f"   - Sheet-sheet contacts: z={sheet_all_fib.get('z_score', 0):.1f}")
        if far_fib_coil_sig:
            print(f"   - Far-Fibonacci in coils: z={coil_far_fib['z_score']:.1f}")
    else:
        print("\n⚠️  FIBONACCI SIGNAL MAY BE EXPLAINED BY HELIX PERIODICITY")
        print("   Further investigation needed.")
    
    results['verdict'] = {
        'coil_fib_significant': coil_sig,
        'sheet_fib_significant': sheet_sig,
        'far_fib_coil_significant': far_fib_coil_sig,
        'exceeds_helix_explanation': coil_sig or sheet_sig or far_fib_coil_sig,
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_14_helix_control_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=300)
    args = parser.parse_args()
    
    run_experiment(max_proteins=args.proteins)
