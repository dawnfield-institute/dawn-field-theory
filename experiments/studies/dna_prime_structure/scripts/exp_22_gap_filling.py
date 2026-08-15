#!/usr/bin/env python3
"""
exp_22_gap_filling.py

Analyze what distances FILL THE GAPS between Fibonacci numbers.
If Fibonacci are attractors, what are the valleys/gaps?

Key questions:
1. Are non-Fibonacci distances random or structured?
2. Do gaps follow a pattern (e.g., midpoints, 1/φ ratios)?
3. Which amino acid pairs fill which gaps?
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

# Fibonacci and gaps
FIBONACCI = [3, 5, 8, 13, 21, 34, 55]
GAPS = {
    (3, 5): [4],
    (5, 8): [6, 7],
    (8, 13): [9, 10, 11, 12],
    (13, 21): [14, 15, 16, 17, 18, 19, 20],
    (21, 34): list(range(22, 34)),
    (34, 55): list(range(35, 55)),
}

# Amino acid categories
AA_CATEGORIES = {
    'aromatic': {'PHE', 'TYR', 'TRP', 'HIS'},
    'charged': {'LYS', 'ARG', 'ASP', 'GLU'},
    'hydrophobic': {'ALA', 'VAL', 'LEU', 'ILE', 'MET'},
    'polar': {'SER', 'THR', 'ASN', 'GLN'},
    'special': {'CYS', 'PRO', 'GLY'},
}


def fetch_protein_ids(n_proteins=200):
    """Fetch high-quality protein PDB IDs."""
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
            ids = [r['identifier'] for r in data.get('result_set', [])]
            return ids[:n_proteins]
    except Exception as e:
        print(f"Search failed: {e}")
    return []


def fetch_structure(pdb_id):
    """Fetch structure and extract CA coordinates with residue info."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        lines = response.text.split('\n')
        atoms = []
        
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                parts = line.split()
                if len(parts) >= 12:
                    atom_name = parts[3]
                    if atom_name == 'CA':
                        try:
                            res_name = parts[5]
                            chain = parts[6]
                            res_seq = int(parts[8])
                            x = float(parts[10])
                            y = float(parts[11])
                            z = float(parts[12])
                            atoms.append({
                                'res_name': res_name,
                                'chain': chain,
                                'res_seq': res_seq,
                                'coords': np.array([x, y, z])
                            })
                        except (ValueError, IndexError):
                            continue
        
        return atoms if len(atoms) > 50 else None
    except Exception:
        return None


def get_aa_category(res_name):
    """Get the category of an amino acid."""
    for cat, residues in AA_CATEGORIES.items():
        if res_name in residues:
            return cat
    return 'other'


def compute_all_distances(atoms, max_dist=60, seq_sep=4):
    """Compute all pairwise distances with amino acid info."""
    contacts = []
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + seq_sep, n):
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist <= max_dist:
                cat_i = get_aa_category(atoms[i]['res_name'])
                cat_j = get_aa_category(atoms[j]['res_name'])
                pair = tuple(sorted([cat_i, cat_j]))
                
                contacts.append({
                    'dist': dist,
                    'dist_int': int(round(dist)),
                    'pair': pair,
                    'res_i': atoms[i]['res_name'],
                    'res_j': atoms[j]['res_name'],
                })
    
    return contacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 22: GAP FILLING ANALYSIS")
    print("=" * 70)
    print("\nQuestion: What distances fill the gaps between Fibonacci numbers?")
    print("          Do certain amino acid pairs prefer gaps vs Fibonacci?\n")
    
    # Get proteins
    print(f"Fetching {args.proteins} protein structures...")
    pdb_ids = fetch_protein_ids(args.proteins)
    print(f"Found {len(pdb_ids)} proteins")
    
    # Collect all contacts
    all_contacts = []
    processed = 0
    
    for i, pdb_id in enumerate(pdb_ids):
        atoms = fetch_structure(pdb_id)
        if atoms is None:
            continue
        
        contacts = compute_all_distances(atoms, max_dist=60)
        all_contacts.extend(contacts)
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed} proteins...")
    
    print(f"\nAnalyzed {processed} proteins, {len(all_contacts):,} contacts")
    
    # Build distance histogram
    print("\n" + "=" * 70)
    print("FULL DISTANCE DISTRIBUTION (4-40Å)")
    print("=" * 70)
    
    dist_counts = defaultdict(int)
    for c in all_contacts:
        if 4 <= c['dist_int'] <= 40:
            dist_counts[c['dist_int']] += 1
    
    max_count = max(dist_counts.values()) if dist_counts else 1
    
    print(f"\n{'Dist':>4}  {'Count':>8}  {'Bar':<40}  {'Type'}")
    print("-" * 70)
    
    fib_set = set(FIBONACCI)
    
    for d in range(4, 41):
        count = dist_counts.get(d, 0)
        bar_len = int(40 * count / max_count)
        bar = "█" * bar_len
        
        if d in fib_set:
            marker = "◆ FIB"
        else:
            # Find which gap this is in
            gap_marker = ""
            for (a, b), gap_vals in GAPS.items():
                if d in gap_vals:
                    gap_marker = f"  gap({a},{b})"
                    break
            marker = gap_marker
        
        print(f"{d:>4}Å {count:>8,}  {bar:<40}  {marker}")
    
    # Analyze gap filling by amino acid type
    print("\n" + "=" * 70)
    print("GAP FILLING BY AMINO ACID PAIR TYPE")
    print("=" * 70)
    
    # For each gap, what pairs fill it most?
    print("\nWhich amino acid pairs prefer gaps vs Fibonacci?")
    
    pair_fib = defaultdict(int)
    pair_gap = defaultdict(int)
    pair_total = defaultdict(int)
    
    for c in all_contacts:
        pair = c['pair']
        d = c['dist_int']
        pair_total[pair] += 1
        
        if d in fib_set:
            pair_fib[pair] += 1
        else:
            # Check if in a gap
            for (a, b), gap_vals in GAPS.items():
                if d in gap_vals:
                    pair_gap[pair] += 1
                    break
    
    print(f"\n{'Pair Type':<30} {'Fib %':>8} {'Gap %':>8} {'Fib/Gap':>8}")
    print("-" * 60)
    
    pair_analysis = []
    for pair in pair_total:
        total = pair_total[pair]
        if total < 1000:
            continue
        
        fib_pct = 100 * pair_fib[pair] / total
        gap_pct = 100 * pair_gap[pair] / total
        ratio = fib_pct / gap_pct if gap_pct > 0 else 0
        
        pair_name = f"{pair[0]}-{pair[1]}"
        print(f"{pair_name:<30} {fib_pct:>7.1f}% {gap_pct:>7.1f}% {ratio:>8.2f}")
        pair_analysis.append({
            'pair': pair_name,
            'fib_pct': fib_pct,
            'gap_pct': gap_pct,
            'ratio': ratio
        })
    
    # Analyze specific gaps
    print("\n" + "=" * 70)
    print("SPECIFIC GAP ANALYSIS")
    print("=" * 70)
    
    for (fib_a, fib_b), gap_vals in GAPS.items():
        if fib_b > 35:
            continue
            
        print(f"\n--- Gap between {fib_a}Å and {fib_b}Å ---")
        
        gap_contacts = [c for c in all_contacts if c['dist_int'] in gap_vals]
        fib_contacts = [c for c in all_contacts if c['dist_int'] in [fib_a, fib_b]]
        
        if not gap_contacts or not fib_contacts:
            continue
        
        # What distances within the gap are most populated?
        gap_dist_counts = defaultdict(int)
        for c in gap_contacts:
            gap_dist_counts[c['dist_int']] += 1
        
        print(f"\nDistance distribution within gap:")
        total_gap = sum(gap_dist_counts.values())
        for d in sorted(gap_dist_counts.keys()):
            count = gap_dist_counts[d]
            pct = 100 * count / total_gap
            bar = "█" * int(pct * 2)
            print(f"  {d:>2}Å: {count:>6} ({pct:>5.1f}%) {bar}")
        
        # Is there a pattern? Check midpoint and φ-scaled positions
        midpoint = (fib_a + fib_b) / 2
        phi_pos = fib_a + (fib_b - fib_a) / 1.618  # Golden section
        
        print(f"\nSpecial positions:")
        print(f"  Midpoint: {midpoint:.1f}Å")
        print(f"  φ-section: {phi_pos:.1f}Å")
        
        # Find peak in gap
        if gap_dist_counts:
            peak_dist = max(gap_dist_counts, key=gap_dist_counts.get)
            print(f"  Actual peak: {peak_dist}Å")
    
    # Key insight: What's special about the gap-filling pattern?
    print("\n" + "=" * 70)
    print("VERDICT: GAP FILLING PATTERN")
    print("=" * 70)
    
    # Calculate overall Fib vs Gap density
    all_dists = [c['dist_int'] for c in all_contacts if 4 <= c['dist_int'] <= 40]
    
    fib_count = sum(1 for d in all_dists if d in fib_set)
    fib_positions = len([f for f in FIBONACCI if 4 <= f <= 40])
    fib_density = fib_count / fib_positions if fib_positions > 0 else 0
    
    gap_count = sum(1 for d in all_dists if d not in fib_set)
    gap_positions = 37 - fib_positions  # 4-40 range minus Fib positions
    gap_density = gap_count / gap_positions if gap_positions > 0 else 0
    
    print(f"\nFibonacci positions: {fib_count:,} contacts / {fib_positions} positions = {fib_density:,.0f} per position")
    print(f"Gap positions: {gap_count:,} contacts / {gap_positions} positions = {gap_density:,.0f} per position")
    print(f"\nFib/Gap density ratio: {fib_density/gap_density:.2f}x")
    
    if fib_density > gap_density:
        print("\n✅ Fibonacci positions have HIGHER contact density than gaps")
    else:
        print("\n❌ Gap positions have higher contact density")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'proteins_analyzed': processed,
        'total_contacts': len(all_contacts),
        'distance_histogram': {str(k): v for k, v in dist_counts.items()},
        'pair_analysis': pair_analysis,
        'fib_density': fib_density,
        'gap_density': gap_density,
        'ratio': fib_density / gap_density if gap_density > 0 else 0
    }
    
    out_path = Path(__file__).parent.parent / 'results' / f'exp_22_gap_filling_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
