#!/usr/bin/env python3
"""
exp_24_sequence_vs_structure.py

Test whether Fibonacci pattern is in SEQUENCE separation or 3D DISTANCE.

The permutation test (exp_13-19) showed Fibonacci enrichment when comparing
observed contacts to shuffled residue labels. The residual test (exp_23) 
showed Fibonacci distances are at baseline relative to smoothed distribution.

Hypothesis: The Fibonacci pattern is in which residues contact each other,
not in the distances being preferred. Specifically: residue pairs at 
Fibonacci SEQUENCE separations preferentially form 3D contacts.

Method:
1. For each contact, record both sequence separation and 3D distance
2. Check if sequence separations are Fibonacci-enriched
3. Check if 3D distances at non-Fibonacci sequence separations are Fibonacci
"""

import argparse
import json
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

FIBONACCI = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
FIB_SET = set(FIBONACCI)


def fetch_protein_ids(n_proteins=200):
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                               "operator": "less", "value": 2.0}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                               "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                               "operator": "range", "value": {"from": 100, "to": 500}}}
            ]
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"],
                          "sort": [{"sort_by": "score", "direction": "desc"}],
                          "paginate": {"start": 0, "rows": n_proteins * 2}}
    }
    
    try:
        response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                                json=query, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return [r['identifier'] for r in data.get('result_set', [])][:n_proteins]
    except:
        pass
    return []


def fetch_structure(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        lines = response.text.split('\n')
        atoms = []
        
        for line in lines:
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 12 and parts[3] == 'CA':
                    try:
                        atoms.append({
                            'res_name': parts[5],
                            'chain': parts[6],
                            'res_seq': int(parts[8]),
                            'coords': np.array([float(parts[10]), float(parts[11]), float(parts[12])])
                        })
                    except:
                        continue
        
        return atoms if len(atoms) > 50 else None
    except:
        return None


def compute_contacts(atoms, max_dist=20, min_seq_sep=4):
    """Compute contacts with both sequence separation and 3D distance."""
    contacts = []
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            
            dist_3d = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist_3d <= max_dist:
                seq_sep = j - i  # Sequence separation in residues
                contacts.append({
                    'seq_sep': seq_sep,
                    'dist_3d': int(round(dist_3d)),
                })
    
    return contacts


def calc_enrichment(values, target_set, range_max=100):
    """Calculate enrichment of target_set in values."""
    if not values:
        return {'enrichment': 0, 'z_score': 0}
    
    observed = sum(1 for v in values if v in target_set)
    total = len(values)
    
    # Expected under uniform
    expected_rate = len([v for v in range(1, range_max + 1) if v in target_set]) / range_max
    expected = expected_rate * total
    
    # Permutation null
    null_counts = []
    for _ in range(500):
        shuffled = np.random.choice(range(1, range_max + 1), size=total, replace=True)
        null_count = sum(1 for v in shuffled if v in target_set)
        null_counts.append(null_count)
    
    null_mean = np.mean(null_counts)
    null_std = np.std(null_counts)
    
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0
    enrichment = observed / expected if expected > 0 else 0
    
    return {
        'enrichment': enrichment,
        'z_score': z_score,
        'observed': observed,
        'expected': expected,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 24: SEQUENCE SEPARATION vs 3D DISTANCE")
    print("=" * 70)
    print("\nQuestion: Is the Fibonacci pattern in SEQUENCE or 3D GEOMETRY?")
    
    # Get proteins
    print(f"\nFetching {args.proteins} proteins...")
    pdb_ids = fetch_protein_ids(args.proteins)
    
    all_contacts = []
    processed = 0
    
    for pdb_id in pdb_ids:
        atoms = fetch_structure(pdb_id)
        if atoms:
            all_contacts.extend(compute_contacts(atoms, max_dist=20))
            processed += 1
            if processed % 20 == 0:
                print(f"  Processed {processed} proteins...")
    
    print(f"\nAnalyzed {processed} proteins, {len(all_contacts):,} contacts")
    
    # Test 1: Are SEQUENCE separations Fibonacci-enriched?
    print("\n" + "=" * 70)
    print("TEST 1: SEQUENCE SEPARATION (residue spacing)")
    print("=" * 70)
    
    seq_seps = [c['seq_sep'] for c in all_contacts]
    seq_stats = calc_enrichment(seq_seps, FIB_SET, range_max=100)
    
    print(f"\nSequence separations at Fibonacci values:")
    print(f"  Observed: {seq_stats['observed']:,} / {seq_stats['total']:,}")
    print(f"  Expected: {seq_stats['expected']:,.0f}")
    print(f"  Enrichment: {seq_stats['enrichment']:.2f}x")
    print(f"  Z-score: {seq_stats['z_score']:.1f}")
    
    if seq_stats['z_score'] > 3:
        print("  ✅ Sequence separations ARE Fibonacci-enriched")
    else:
        print("  ≈ Sequence separations are NOT significantly Fibonacci-enriched")
    
    # Histogram of sequence separations
    seq_counts = defaultdict(int)
    for c in all_contacts:
        seq_counts[c['seq_sep']] += 1
    
    print("\nSequence separation distribution (top 15):")
    for sep, count in sorted(seq_counts.items(), key=lambda x: -x[1])[:15]:
        marker = "◆ FIB" if sep in FIB_SET else ""
        print(f"  {sep:>3} residues: {count:>6,} {marker}")
    
    # Test 2: Are 3D DISTANCES Fibonacci-enriched?
    print("\n" + "=" * 70)
    print("TEST 2: 3D DISTANCE (Ångströms)")
    print("=" * 70)
    
    dists_3d = [c['dist_3d'] for c in all_contacts]
    dist_stats = calc_enrichment(dists_3d, FIB_SET, range_max=20)
    
    print(f"\n3D distances at Fibonacci values:")
    print(f"  Observed: {dist_stats['observed']:,} / {dist_stats['total']:,}")
    print(f"  Expected: {dist_stats['expected']:,.0f}")
    print(f"  Enrichment: {dist_stats['enrichment']:.2f}x")
    print(f"  Z-score: {dist_stats['z_score']:.1f}")
    
    if dist_stats['z_score'] > 3:
        print("  ✅ 3D distances ARE Fibonacci-enriched")
    else:
        print("  ≈ 3D distances are NOT significantly Fibonacci-enriched")
    
    # Test 3: Cross-analysis - do Fib sequence separations give Fib distances?
    print("\n" + "=" * 70)
    print("TEST 3: CROSS-ANALYSIS (does Fib seq give Fib dist?)")
    print("=" * 70)
    
    # Contacts at Fibonacci sequence separations
    fib_seq_contacts = [c for c in all_contacts if c['seq_sep'] in FIB_SET]
    non_fib_seq_contacts = [c for c in all_contacts if c['seq_sep'] not in FIB_SET]
    
    fib_seq_dists = [c['dist_3d'] for c in fib_seq_contacts]
    non_fib_seq_dists = [c['dist_3d'] for c in non_fib_seq_contacts]
    
    fib_in_fib = sum(1 for d in fib_seq_dists if d in FIB_SET)
    fib_in_non_fib = sum(1 for d in non_fib_seq_dists if d in FIB_SET)
    
    pct_fib_in_fib = 100 * fib_in_fib / len(fib_seq_dists) if fib_seq_dists else 0
    pct_fib_in_non_fib = 100 * fib_in_non_fib / len(non_fib_seq_dists) if non_fib_seq_dists else 0
    
    print(f"\nContacts at Fibonacci sequence separations:")
    print(f"  Total: {len(fib_seq_contacts):,}")
    print(f"  3D distances at Fibonacci: {fib_in_fib:,} ({pct_fib_in_fib:.1f}%)")
    
    print(f"\nContacts at non-Fibonacci sequence separations:")
    print(f"  Total: {len(non_fib_seq_contacts):,}")
    print(f"  3D distances at Fibonacci: {fib_in_non_fib:,} ({pct_fib_in_non_fib:.1f}%)")
    
    print(f"\nFibonacci 3D rate ratio: {pct_fib_in_fib / pct_fib_in_non_fib:.2f}x")
    
    if pct_fib_in_fib > pct_fib_in_non_fib * 1.1:
        print("  ✅ Fibonacci sequence separations correlate with Fibonacci 3D distances")
    else:
        print("  ≈ No correlation between sequence and 3D Fibonacci patterns")
    
    # Test 4: Helix periodicity check (3, 4 residues)
    print("\n" + "=" * 70)
    print("TEST 4: HELIX PERIODICITY (3-4 residue patterns)")
    print("=" * 70)
    
    helix_seps = {3, 4, 7, 10, 11, 14}  # Helix-related separations
    helix_contacts = [c for c in all_contacts if c['seq_sep'] in helix_seps]
    
    print(f"\nHelix-related sequence separations (3, 4, 7, 10, 11, 14):")
    print(f"  Total: {len(helix_contacts):,} ({100*len(helix_contacts)/len(all_contacts):.1f}% of all)")
    
    helix_3d_stats = calc_enrichment([c['dist_3d'] for c in helix_contacts], FIB_SET, range_max=20)
    print(f"  3D Fibonacci enrichment: {helix_3d_stats['enrichment']:.2f}x (z={helix_3d_stats['z_score']:.1f})")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    print(f"\nSequence Fibonacci enrichment: {seq_stats['enrichment']:.2f}x (z={seq_stats['z_score']:.1f})")
    print(f"3D Distance Fibonacci enrichment: {dist_stats['enrichment']:.2f}x (z={dist_stats['z_score']:.1f})")
    
    if seq_stats['z_score'] > dist_stats['z_score'] + 10:
        print("\n✅ Fibonacci pattern is primarily in SEQUENCE, not 3D geometry")
        print("   → Amino acids at Fibonacci sequence positions form contacts")
        print("   → This is a SEQUENCE ORGANIZATION principle, not geometric")
    elif dist_stats['z_score'] > seq_stats['z_score'] + 10:
        print("\n✅ Fibonacci pattern is primarily in 3D GEOMETRY, not sequence")
        print("   → Contacts at Fibonacci distances are preferred regardless of sequence")
    else:
        print("\n≈ Fibonacci pattern exists in BOTH sequence and 3D geometry")
        print("   → Complex relationship between sequence and structure")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'proteins': processed,
        'contacts': len(all_contacts),
        'sequence_enrichment': seq_stats['enrichment'],
        'sequence_z': seq_stats['z_score'],
        'distance_enrichment': dist_stats['enrichment'],
        'distance_z': dist_stats['z_score'],
        'fib_seq_to_fib_dist_rate': pct_fib_in_fib,
        'non_fib_seq_to_fib_dist_rate': pct_fib_in_non_fib,
    }
    
    out_path = Path(__file__).parent.parent / 'results' / f'exp_24_seq_vs_struct_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
