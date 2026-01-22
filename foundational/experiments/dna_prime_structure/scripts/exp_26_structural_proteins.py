#!/usr/bin/env python3
"""
exp_26_structural_proteins.py

Deep dive into structural proteins: collagen, keratin, cytoskeleton.
If Fibonacci is about structural stability, these should show strongest signal.
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

FIBONACCI = [3, 5, 8, 13, 21, 34, 55, 89]
FIB_SET = set(FIBONACCI)

STRUCTURAL_TYPES = {
    'collagen': 'collagen',
    'keratin': 'keratin', 
    'actin': 'actin cytoskeleton',
    'tubulin': 'tubulin microtubule',
    'fibronectin': 'fibronectin',
    'elastin': 'elastin',
    'myosin': 'myosin motor',
}


def fetch_proteins_by_keyword(keyword, n_proteins=30):
    """Fetch proteins matching keyword."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                               "operator": "less", "value": 2.5}},
                {"type": "terminal", "service": "full_text",
                 "parameters": {"value": keyword}},
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": n_proteins * 2}}
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
    """Fetch structure coordinates."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        atoms = []
        for line in response.text.split('\n'):
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 12 and parts[3] == 'CA':
                    try:
                        atoms.append({
                            'chain': parts[6],
                            'res_seq': int(parts[8]),
                            'coords': np.array([float(parts[10]), float(parts[11]), float(parts[12])])
                        })
                    except:
                        continue
        
        return atoms if len(atoms) > 30 else None
    except:
        return None


def compute_contacts(atoms, max_dist=15, min_seq_sep=4):
    """Compute contacts with sequence separation."""
    contacts = []
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist <= max_dist:
                contacts.append({'seq_sep': j - i})
    
    return contacts


def calc_enrichment(contacts):
    """Calculate Fibonacci sequence enrichment."""
    if len(contacts) < 50:
        return None
    
    seq_seps = [c['seq_sep'] for c in contacts]
    observed = sum(1 for s in seq_seps if s in FIB_SET)
    total = len(seq_seps)
    
    max_sep = max(seq_seps) if seq_seps else 100
    expected_rate = len([s for s in range(1, max_sep + 1) if s in FIB_SET]) / max_sep
    expected = expected_rate * total
    
    null_counts = []
    for _ in range(300):
        shuffled = np.random.choice(range(4, max_sep + 1), size=total, replace=True)
        null_count = sum(1 for s in shuffled if s in FIB_SET)
        null_counts.append(null_count)
    
    z_score = (observed - np.mean(null_counts)) / np.std(null_counts) if np.std(null_counts) > 0 else 0
    
    return {
        'enrichment': observed / expected if expected > 0 else 0,
        'z_score': z_score,
        'fib_rate': 100 * observed / total,
        'n_contacts': total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 26: STRUCTURAL PROTEIN DEEP DIVE")
    print("=" * 70)
    print("\nTesting: collagen, keratin, actin, tubulin, fibronectin, elastin, myosin")
    
    results = {}
    
    print(f"\n{'Type':<15} {'Proteins':>10} {'Contacts':>12} {'Enrichment':>12} {'Z-score':>10}")
    print("-" * 65)
    
    for struct_type, keyword in STRUCTURAL_TYPES.items():
        pdb_ids = fetch_proteins_by_keyword(keyword, args.proteins)
        
        all_contacts = []
        processed = 0
        
        for pdb_id in pdb_ids:
            atoms = fetch_structure(pdb_id)
            if atoms:
                contacts = compute_contacts(atoms)
                all_contacts.extend(contacts)
                processed += 1
        
        stats = calc_enrichment(all_contacts)
        
        if stats:
            print(f"{struct_type:<15} {processed:>10} {stats['n_contacts']:>12,} {stats['enrichment']:>12.2f}x {stats['z_score']:>10.1f}")
            results[struct_type] = stats
        else:
            print(f"{struct_type:<15} {processed:>10} {'(insufficient data)':<35}")
    
    # Find highest
    print("\n" + "=" * 70)
    print("RANKING BY FIBONACCI ENRICHMENT")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['enrichment'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Type':<15} {'Enrichment':>12}")
    print("-" * 35)
    for i, (name, stats) in enumerate(sorted_results, 1):
        marker = "◆" if stats['enrichment'] > 4 else ""
        print(f"{i:<6} {name:<15} {stats['enrichment']:>12.2f}x {marker}")
    
    if sorted_results:
        best = sorted_results[0]
        print(f"\n✅ {best[0].upper()} shows strongest Fibonacci pattern ({best[1]['enrichment']:.2f}x)")
    
    # Save
    out_path = Path(__file__).parent.parent / 'results' / f'exp_26_structural_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
