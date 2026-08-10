#!/usr/bin/env python3
"""
exp_27_active_sites.py

Deep analysis of enzyme active sites vs non-active regions.
Uses UniProt annotations for precise active site locations.
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


def fetch_enzymes_with_annotations(n_proteins=50):
    """Fetch enzymes with UniProt annotations."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                               "operator": "less", "value": 2.0}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                               "operator": "exists"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                               "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                               "operator": "range", "value": {"from": 150, "to": 400}}}
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": n_proteins * 3}}
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


def fetch_structure_with_sites(pdb_id):
    """Fetch structure and active site annotations."""
    # Get structure
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
        
        if len(atoms) < 50:
            return None
    except:
        return None
    
    # Get annotations
    active_sites = set()
    binding_sites = set()
    catalytic_sites = set()
    
    try:
        # Try to get UniProt features via PDB API
        api_url = f"https://data.rcsb.org/rest/v1/core/uniprot/{pdb_id}/1"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for entry in data if isinstance(data, list) else [data]:
                features = entry.get('rcsb_uniprot_feature', [])
                for feat in features:
                    feat_type = feat.get('type', '').lower()
                    positions = feat.get('feature_positions', [])
                    
                    for pos in positions:
                        start = pos.get('beg_seq_id')
                        end = pos.get('end_seq_id', start)
                        if start:
                            for i in range(start, (end or start) + 1):
                                if 'active' in feat_type or 'act_site' in feat_type:
                                    active_sites.add(i)
                                elif 'binding' in feat_type:
                                    binding_sites.add(i)
                                elif 'catalytic' in feat_type:
                                    catalytic_sites.add(i)
    except:
        pass
    
    # Also try polymer entity features
    try:
        api_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get('rcsb_polymer_entity_feature', [])
            for feat in features:
                feat_type = feat.get('type', '').lower()
                positions = feat.get('feature_positions', [])
                
                for pos in positions:
                    start = pos.get('beg_seq_id')
                    end = pos.get('end_seq_id', start)
                    if start:
                        for i in range(start, (end or start) + 1):
                            if 'active' in feat_type:
                                active_sites.add(i)
                            elif 'binding' in feat_type:
                                binding_sites.add(i)
    except:
        pass
    
    functional_sites = active_sites | binding_sites | catalytic_sites
    
    return {
        'atoms': atoms,
        'active_sites': active_sites,
        'binding_sites': binding_sites,
        'catalytic_sites': catalytic_sites,
        'all_functional': functional_sites,
        'pdb_id': pdb_id
    }


def compute_contacts_by_site(data, max_dist=15, min_seq_sep=4):
    """Compute contacts classified by functional site involvement."""
    atoms = data['atoms']
    functional = data['all_functional']
    
    active_contacts = []  # Both residues in functional sites
    mixed_contacts = []   # One residue in functional site
    structural_contacts = []  # Neither in functional site
    
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist <= max_dist:
                seq_sep = j - i
                
                in_func_i = atoms[i]['res_seq'] in functional
                in_func_j = atoms[j]['res_seq'] in functional
                
                contact = {'seq_sep': seq_sep}
                
                if in_func_i and in_func_j:
                    active_contacts.append(contact)
                elif in_func_i or in_func_j:
                    mixed_contacts.append(contact)
                else:
                    structural_contacts.append(contact)
    
    return active_contacts, mixed_contacts, structural_contacts


def calc_enrichment(contacts):
    """Calculate Fibonacci sequence enrichment."""
    if len(contacts) < 30:
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
    parser.add_argument('--proteins', type=int, default=60)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 27: ENZYME ACTIVE SITES vs NON-ACTIVE REGIONS")
    print("=" * 70)
    print("\nFetching enzymes with active site annotations...")
    
    pdb_ids = fetch_enzymes_with_annotations(args.proteins)
    print(f"Found {len(pdb_ids)} enzymes")
    
    all_active = []
    all_mixed = []
    all_structural = []
    proteins_with_sites = 0
    
    for i, pdb_id in enumerate(pdb_ids):
        data = fetch_structure_with_sites(pdb_id)
        if data is None:
            continue
        
        if len(data['all_functional']) > 0:
            proteins_with_sites += 1
        
        active, mixed, structural = compute_contacts_by_site(data)
        all_active.extend(active)
        all_mixed.extend(mixed)
        all_structural.extend(structural)
        
        if (i + 1) % 15 == 0:
            print(f"  Processed {i + 1} proteins... ({proteins_with_sites} with annotations)")
    
    print(f"\nAnalyzed {len(pdb_ids)} enzymes")
    print(f"  With functional annotations: {proteins_with_sites}")
    print(f"  Active site contacts: {len(all_active):,}")
    print(f"  Mixed contacts: {len(all_mixed):,}")
    print(f"  Structural contacts: {len(all_structural):,}")
    
    # Calculate enrichment
    print("\n" + "=" * 70)
    print("FIBONACCI ENRICHMENT BY REGION TYPE")
    print("=" * 70)
    
    results = {}
    
    print(f"\n{'Region':<20} {'Contacts':>10} {'Enrichment':>12} {'Z-score':>10} {'Fib Rate':>10}")
    print("-" * 65)
    
    for name, contacts in [('Active sites', all_active), 
                           ('Mixed', all_mixed), 
                           ('Structural', all_structural)]:
        stats = calc_enrichment(contacts)
        if stats:
            print(f"{name:<20} {stats['n_contacts']:>10,} {stats['enrichment']:>12.2f}x {stats['z_score']:>10.1f} {stats['fib_rate']:>9.1f}%")
            results[name.lower().replace(' ', '_')] = stats
        else:
            print(f"{name:<20} {'(insufficient data)':<45}")
    
    # Compare active vs structural
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if 'active_sites' in results and 'structural' in results:
        active_e = results['active_sites']['enrichment']
        struct_e = results['structural']['enrichment']
        ratio = active_e / struct_e if struct_e > 0 else 0
        
        print(f"\nActive sites enrichment: {active_e:.2f}x")
        print(f"Structural enrichment: {struct_e:.2f}x")
        print(f"Active/Structural ratio: {ratio:.2f}x")
        
        if ratio > 1.15:
            print("\n✅ ACTIVE SITES show ENHANCED Fibonacci pattern")
            print("   → Fibonacci sequence organization relates to enzyme function")
        elif ratio < 0.85:
            print("\n⬇ STRUCTURAL regions show stronger Fibonacci pattern")
            print("   → Fibonacci relates to stability, not catalysis")
        else:
            print("\n≈ Similar patterns in active and structural regions")
            print("   → Fibonacci is function-independent")
    
    # Save
    out_path = Path(__file__).parent.parent / 'results' / f'exp_27_active_sites_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
