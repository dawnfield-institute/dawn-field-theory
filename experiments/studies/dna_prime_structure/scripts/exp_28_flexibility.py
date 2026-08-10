#!/usr/bin/env python3
"""
exp_28_flexibility.py

Compare Fibonacci patterns in FLEXIBLE vs RIGID regions of proteins.
Uses B-factors (temperature factors) as proxy for flexibility.

Hypothesis: If Fibonacci relates to stability, rigid regions should
show stronger pattern. If Fibonacci relates to dynamics, flexible
regions might show different patterns.
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


def fetch_high_quality_proteins(n_proteins=100):
    """Fetch high-resolution proteins for B-factor analysis."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                               "operator": "less", "value": 1.5}},  # Very high res for good B-factors
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                               "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                               "operator": "range", "value": {"from": 100, "to": 400}}}
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


def fetch_structure_with_bfactors(pdb_id):
    """Fetch structure with B-factors."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        atoms = []
        for line in response.text.split('\n'):
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 15 and parts[3] == 'CA':
                    try:
                        bfactor = float(parts[14])  # B-factor column
                        atoms.append({
                            'chain': parts[6],
                            'res_seq': int(parts[8]),
                            'coords': np.array([float(parts[10]), float(parts[11]), float(parts[12])]),
                            'bfactor': bfactor
                        })
                    except:
                        continue
        
        if len(atoms) < 50:
            return None
        
        # Normalize B-factors within protein
        bfactors = [a['bfactor'] for a in atoms]
        mean_b = np.mean(bfactors)
        std_b = np.std(bfactors)
        
        for atom in atoms:
            atom['bfactor_norm'] = (atom['bfactor'] - mean_b) / std_b if std_b > 0 else 0
        
        return atoms
    except:
        return None


def compute_contacts_by_flexibility(atoms, max_dist=15, min_seq_sep=4):
    """Compute contacts classified by flexibility of residues."""
    
    # Classify residues by B-factor percentile
    bfactors = [a['bfactor'] for a in atoms]
    p25 = np.percentile(bfactors, 25)
    p75 = np.percentile(bfactors, 75)
    
    rigid_contacts = []    # Both residues low B-factor
    flexible_contacts = [] # Both residues high B-factor
    mixed_contacts = []    # One rigid, one flexible
    
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist <= max_dist:
                seq_sep = j - i
                
                # Classify by B-factor
                rigid_i = atoms[i]['bfactor'] < p25
                rigid_j = atoms[j]['bfactor'] < p25
                flex_i = atoms[i]['bfactor'] > p75
                flex_j = atoms[j]['bfactor'] > p75
                
                contact = {'seq_sep': seq_sep}
                
                if rigid_i and rigid_j:
                    rigid_contacts.append(contact)
                elif flex_i and flex_j:
                    flexible_contacts.append(contact)
                elif (rigid_i and flex_j) or (flex_i and rigid_j):
                    mixed_contacts.append(contact)
                # Middle-B contacts are ignored for cleaner comparison
    
    return rigid_contacts, flexible_contacts, mixed_contacts


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
    parser.add_argument('--proteins', type=int, default=80)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 28: FLEXIBLE vs RIGID REGIONS")
    print("=" * 70)
    print("\nUsing B-factors to identify flexible/rigid regions")
    print("(High-resolution structures only: < 1.5Å)")
    
    pdb_ids = fetch_high_quality_proteins(args.proteins)
    print(f"\nFound {len(pdb_ids)} high-resolution proteins")
    
    all_rigid = []
    all_flexible = []
    all_mixed = []
    processed = 0
    
    for pdb_id in pdb_ids:
        atoms = fetch_structure_with_bfactors(pdb_id)
        if atoms is None:
            continue
        
        rigid, flexible, mixed = compute_contacts_by_flexibility(atoms)
        all_rigid.extend(rigid)
        all_flexible.extend(flexible)
        all_mixed.extend(mixed)
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed} proteins...")
    
    print(f"\nAnalyzed {processed} proteins")
    print(f"  Rigid-rigid contacts: {len(all_rigid):,}")
    print(f"  Flexible-flexible contacts: {len(all_flexible):,}")
    print(f"  Rigid-flexible contacts: {len(all_mixed):,}")
    
    # Calculate enrichment
    print("\n" + "=" * 70)
    print("FIBONACCI ENRICHMENT BY FLEXIBILITY")
    print("=" * 70)
    
    results = {}
    
    print(f"\n{'Region':<20} {'Contacts':>10} {'Enrichment':>12} {'Z-score':>10} {'Fib Rate':>10}")
    print("-" * 65)
    
    for name, contacts in [('Rigid (low B)', all_rigid), 
                           ('Flexible (high B)', all_flexible),
                           ('Rigid-Flex mixed', all_mixed)]:
        stats = calc_enrichment(contacts)
        if stats:
            print(f"{name:<20} {stats['n_contacts']:>10,} {stats['enrichment']:>12.2f}x {stats['z_score']:>10.1f} {stats['fib_rate']:>9.1f}%")
            results[name.split()[0].lower()] = stats
        else:
            print(f"{name:<20} {'(insufficient data)':<45}")
    
    # Sequence separation distribution by flexibility
    print("\n" + "=" * 70)
    print("SEQUENCE SEPARATION DISTRIBUTION")
    print("=" * 70)
    
    for name, contacts in [('Rigid', all_rigid), ('Flexible', all_flexible)]:
        if len(contacts) < 100:
            continue
        
        seq_seps = [c['seq_sep'] for c in contacts]
        
        print(f"\n{name} contacts - top sequence separations:")
        sep_counts = defaultdict(int)
        for s in seq_seps:
            sep_counts[s] += 1
        
        for sep, count in sorted(sep_counts.items(), key=lambda x: -x[1])[:8]:
            marker = "◆ FIB" if sep in FIB_SET else ""
            pct = 100 * count / len(seq_seps)
            print(f"  {sep:>3} residues: {count:>6} ({pct:>4.1f}%) {marker}")
    
    # Compare
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if 'rigid' in results and 'flexible' in results:
        rigid_e = results['rigid']['enrichment']
        flex_e = results['flexible']['enrichment']
        ratio = rigid_e / flex_e if flex_e > 0 else 0
        
        print(f"\nRigid regions enrichment: {rigid_e:.2f}x")
        print(f"Flexible regions enrichment: {flex_e:.2f}x")
        print(f"Rigid/Flexible ratio: {ratio:.2f}x")
        
        if ratio > 1.15:
            print("\n✅ RIGID regions show ENHANCED Fibonacci pattern")
            print("   → Fibonacci sequence organization relates to structural stability")
        elif ratio < 0.85:
            print("\n⬇ FLEXIBLE regions show stronger Fibonacci pattern")
            print("   → Fibonacci may relate to dynamic function")
        else:
            print("\n≈ Similar patterns in rigid and flexible regions")
            print("   → Fibonacci is flexibility-independent")
    
    # Save
    out_path = Path(__file__).parent.parent / 'results' / f'exp_28_flexibility_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': results, 'proteins': processed}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
