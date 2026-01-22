#!/usr/bin/env python3
"""
exp_25_function_correlation.py

Test whether Fibonacci sequence patterns correlate with protein FUNCTION.

Hypotheses:
1. Active site residues have enhanced Fibonacci sequence contacts
2. Binding sites show different patterns than structural regions
3. Enzymes vs structural proteins show different Fibonacci profiles
4. Conserved residues preferentially at Fibonacci separations

Data sources:
- UniProt: Function annotations, active sites, binding sites
- PDB: Structure + functional annotations
- EC numbers: Enzyme classification
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


def fetch_proteins_by_function(function_type, n_proteins=50):
    """Fetch proteins with specific function annotations."""
    
    if function_type == 'enzyme':
        # Enzymes with EC numbers
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
                                   "operator": "equals", "value": 1}}
                ]
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": n_proteins * 2}}
        }
    elif function_type == 'structural':
        # Structural proteins (collagen, keratin, etc)
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                                   "operator": "less", "value": 2.0}},
                    {"type": "terminal", "service": "full_text",
                     "parameters": {"value": "structural protein OR cytoskeleton OR collagen"}},
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                                   "operator": "equals", "value": 1}}
                ]
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": n_proteins * 2}}
        }
    elif function_type == 'binding':
        # Proteins with ligand binding
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                                   "operator": "less", "value": 2.0}},
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "rcsb_entry_info.nonpolymer_entity_count",
                                   "operator": "greater", "value": 0}},
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                                   "operator": "equals", "value": 1}}
                ]
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": n_proteins * 2}}
        }
    else:
        return []
    
    try:
        response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                                json=query, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return [r['identifier'] for r in data.get('result_set', [])][:n_proteins]
    except Exception as e:
        print(f"  Query failed: {e}")
    return []


def fetch_structure_with_annotations(pdb_id):
    """Fetch structure and functional annotations."""
    
    # Get structure
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
        
        if len(atoms) < 50:
            return None
            
    except:
        return None
    
    # Get functional annotations from PDB API
    annotations = {
        'active_sites': set(),
        'binding_sites': set(),
        'catalytic_sites': set()
    }
    
    try:
        # Get polymer entity annotations
        api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Check for ligand info
            if data.get('rcsb_entry_info', {}).get('nonpolymer_entity_count', 0) > 0:
                annotations['has_ligand'] = True
    except:
        pass
    
    try:
        # Get binding site annotations
        api_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Check for feature annotations
            features = data.get('rcsb_polymer_entity_feature', [])
            for feat in features:
                feat_type = feat.get('type', '')
                positions = feat.get('feature_positions', [])
                
                for pos in positions:
                    start = pos.get('beg_seq_id')
                    end = pos.get('end_seq_id', start)
                    if start:
                        for i in range(start, (end or start) + 1):
                            if 'active' in feat_type.lower():
                                annotations['active_sites'].add(i)
                            elif 'binding' in feat_type.lower():
                                annotations['binding_sites'].add(i)
                            elif 'catalytic' in feat_type.lower():
                                annotations['catalytic_sites'].add(i)
    except:
        pass
    
    return {'atoms': atoms, 'annotations': annotations, 'pdb_id': pdb_id}


def compute_contacts_with_function(data, max_dist=15, min_seq_sep=4):
    """Compute contacts and annotate with functional information."""
    atoms = data['atoms']
    annotations = data['annotations']
    
    # Combine all functional sites
    functional_sites = (annotations['active_sites'] | 
                       annotations['binding_sites'] | 
                       annotations['catalytic_sites'])
    
    contacts = []
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist <= max_dist:
                seq_sep = j - i
                
                # Check if either residue is functional
                res_i = atoms[i]['res_seq']
                res_j = atoms[j]['res_seq']
                
                is_functional_i = res_i in functional_sites
                is_functional_j = res_j in functional_sites
                
                contact_type = 'structural'
                if is_functional_i and is_functional_j:
                    contact_type = 'functional_pair'
                elif is_functional_i or is_functional_j:
                    contact_type = 'functional_one'
                
                contacts.append({
                    'seq_sep': seq_sep,
                    'dist_3d': int(round(dist)),
                    'contact_type': contact_type,
                    'is_functional': is_functional_i or is_functional_j
                })
    
    return contacts, len(functional_sites)


def calc_fib_enrichment(contacts, filter_type=None):
    """Calculate Fibonacci sequence enrichment for contacts."""
    if filter_type:
        seq_seps = [c['seq_sep'] for c in contacts if c['contact_type'] == filter_type]
    else:
        seq_seps = [c['seq_sep'] for c in contacts]
    
    if len(seq_seps) < 100:
        return None
    
    observed = sum(1 for s in seq_seps if s in FIB_SET)
    total = len(seq_seps)
    
    # Expected under distribution of observed separations
    max_sep = max(seq_seps) if seq_seps else 100
    expected_rate = len([s for s in range(1, max_sep + 1) if s in FIB_SET]) / max_sep
    expected = expected_rate * total
    
    # Permutation test
    null_counts = []
    for _ in range(500):
        shuffled = np.random.choice(range(4, max_sep + 1), size=total, replace=True)
        null_count = sum(1 for s in shuffled if s in FIB_SET)
        null_counts.append(null_count)
    
    null_mean = np.mean(null_counts)
    null_std = np.std(null_counts)
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0
    
    return {
        'enrichment': observed / expected if expected > 0 else 0,
        'z_score': z_score,
        'observed': observed,
        'total': total,
        'fib_rate': 100 * observed / total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 25: FIBONACCI SEQUENCE PATTERN vs PROTEIN FUNCTION")
    print("=" * 70)
    print("\nHypothesis: Functional sites show enhanced Fibonacci sequence patterns")
    
    results = {}
    
    # Test 1: Enzymes vs Structural proteins
    print("\n" + "=" * 70)
    print("TEST 1: ENZYMES vs STRUCTURAL PROTEINS")
    print("=" * 70)
    
    for func_type in ['enzyme', 'structural', 'binding']:
        print(f"\nFetching {func_type} proteins...")
        pdb_ids = fetch_proteins_by_function(func_type, args.proteins)
        print(f"  Found {len(pdb_ids)} proteins")
        
        all_contacts = []
        total_functional_sites = 0
        processed = 0
        
        for pdb_id in pdb_ids[:args.proteins]:
            data = fetch_structure_with_annotations(pdb_id)
            if data:
                contacts, n_func = compute_contacts_with_function(data)
                all_contacts.extend(contacts)
                total_functional_sites += n_func
                processed += 1
                if processed % 20 == 0:
                    print(f"  Processed {processed} proteins...")
        
        print(f"  Analyzed {processed} proteins, {len(all_contacts):,} contacts")
        print(f"  Total functional sites: {total_functional_sites}")
        
        # Calculate enrichment
        overall = calc_fib_enrichment(all_contacts)
        
        if overall:
            print(f"\n  Overall Fibonacci sequence enrichment:")
            print(f"    Enrichment: {overall['enrichment']:.2f}x")
            print(f"    Z-score: {overall['z_score']:.1f}")
            print(f"    Fib rate: {overall['fib_rate']:.1f}%")
            
            results[func_type] = {
                'overall': overall,
                'n_proteins': processed,
                'n_contacts': len(all_contacts),
                'n_functional': total_functional_sites
            }
        
        # Compare functional vs structural contacts
        func_pairs = calc_fib_enrichment(all_contacts, 'functional_pair')
        func_one = calc_fib_enrichment(all_contacts, 'functional_one')
        structural = calc_fib_enrichment(all_contacts, 'structural')
        
        if func_pairs and structural:
            print(f"\n  By contact type:")
            print(f"    Functional pairs: {func_pairs['enrichment']:.2f}x (z={func_pairs['z_score']:.1f}, n={func_pairs['total']})")
            print(f"    Functional-one: {func_one['enrichment']:.2f}x (z={func_one['z_score']:.1f}, n={func_one['total']})" if func_one else "")
            print(f"    Structural: {structural['enrichment']:.2f}x (z={structural['z_score']:.1f}, n={structural['total']})")
            
            results[func_type]['functional_pairs'] = func_pairs
            results[func_type]['structural'] = structural
    
    # Compare function types
    print("\n" + "=" * 70)
    print("COMPARISON: FUNCTION TYPES")
    print("=" * 70)
    
    print(f"\n{'Type':<15} {'Enrichment':>12} {'Z-score':>10} {'Fib Rate':>10}")
    print("-" * 50)
    
    for func_type in ['enzyme', 'structural', 'binding']:
        if func_type in results and results[func_type].get('overall'):
            r = results[func_type]['overall']
            print(f"{func_type:<15} {r['enrichment']:>12.2f}x {r['z_score']:>10.1f} {r['fib_rate']:>9.1f}%")
    
    # Test 2: Active sites specifically
    print("\n" + "=" * 70)
    print("TEST 2: ACTIVE SITE ENRICHMENT ANALYSIS")
    print("=" * 70)
    
    # Collect all contacts from enzymes
    if 'enzyme' in results:
        print("\nAnalyzing enzyme active site contacts...")
        
        # Re-analyze with focus on functional sites
        pdb_ids = fetch_proteins_by_function('enzyme', args.proteins)
        
        active_contacts = []
        non_active_contacts = []
        
        for pdb_id in pdb_ids[:50]:  # Smaller set for detailed analysis
            data = fetch_structure_with_annotations(pdb_id)
            if data:
                contacts, _ = compute_contacts_with_function(data)
                for c in contacts:
                    if c['contact_type'] == 'functional_pair':
                        active_contacts.append(c)
                    elif c['contact_type'] == 'structural':
                        non_active_contacts.append(c)
        
        active_enrich = calc_fib_enrichment(active_contacts) if active_contacts else None
        non_active_enrich = calc_fib_enrichment(non_active_contacts) if non_active_contacts else None
        
        if active_enrich and non_active_enrich:
            print(f"\nActive site pairs: {len(active_contacts):,} contacts")
            print(f"  Fibonacci enrichment: {active_enrich['enrichment']:.2f}x (z={active_enrich['z_score']:.1f})")
            
            print(f"\nNon-active pairs: {len(non_active_contacts):,} contacts")
            print(f"  Fibonacci enrichment: {non_active_enrich['enrichment']:.2f}x (z={non_active_enrich['z_score']:.1f})")
            
            ratio = active_enrich['enrichment'] / non_active_enrich['enrichment'] if non_active_enrich['enrichment'] > 0 else 0
            print(f"\nActive/Non-active ratio: {ratio:.2f}x")
            
            if ratio > 1.1:
                print("✅ Active sites show ENHANCED Fibonacci pattern")
            elif ratio < 0.9:
                print("❌ Active sites show REDUCED Fibonacci pattern")
            else:
                print("≈ No significant difference")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT: FUNCTION CORRELATION")
    print("=" * 70)
    
    if results:
        enzyme_enrich = results.get('enzyme', {}).get('overall', {}).get('enrichment', 0)
        struct_enrich = results.get('structural', {}).get('overall', {}).get('enrichment', 0)
        
        if enzyme_enrich > struct_enrich * 1.1:
            print("\n✅ ENZYMES show stronger Fibonacci sequence pattern than structural proteins")
            print("   → Fibonacci may be related to catalytic function")
        elif struct_enrich > enzyme_enrich * 1.1:
            print("\n⬇ STRUCTURAL proteins show stronger Fibonacci pattern")
            print("   → Fibonacci may be related to structural stability")
        else:
            print("\n≈ Similar Fibonacci pattern across function types")
            print("   → Pattern is function-independent")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'proteins_per_type': args.proteins,
        'results': {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, dict))} 
                   for k, v in results.items()}
    }
    
    out_path = Path(__file__).parent.parent / 'results' / f'exp_25_function_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
