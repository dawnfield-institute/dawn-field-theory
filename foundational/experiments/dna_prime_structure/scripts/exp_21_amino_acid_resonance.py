#!/usr/bin/env python3
"""
exp_21_amino_acid_resonance.py

Test whether amino acid chemical properties affect Fibonacci enrichment.
Focus on:
1. Aromatic residues (Phe, Tyr, Trp, His) - π-electron resonance/stacking
2. Charged residues (Asp, Glu, Lys, Arg) - electrostatic interactions
3. Hydrophobic residues (Ala, Val, Leu, Ile, Met) - van der Waals
4. Polar residues (Ser, Thr, Asn, Gln) - hydrogen bonding
5. Special (Cys, Pro, Gly) - disulfides, rigidity, flexibility

Hypothesis: Aromatic contacts (π-stacking) may show enhanced Fibonacci 
due to resonance-mediated distance preferences.
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

# Fibonacci numbers up to 300Å
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

# Amino acid categories by chemical property
AA_CATEGORIES = {
    'aromatic': {'PHE', 'TYR', 'TRP', 'HIS'},  # π-electron systems
    'charged_pos': {'LYS', 'ARG'},  # Positive charge
    'charged_neg': {'ASP', 'GLU'},  # Negative charge
    'hydrophobic': {'ALA', 'VAL', 'LEU', 'ILE', 'MET'},  # Nonpolar
    'polar': {'SER', 'THR', 'ASN', 'GLN'},  # Polar uncharged
    'special': {'CYS', 'PRO', 'GLY'},  # Special roles
}

# Combine charged
AA_CATEGORIES['charged'] = AA_CATEGORIES['charged_pos'] | AA_CATEGORIES['charged_neg']

# All standard amino acids
ALL_AA = set()
for cat in AA_CATEGORIES.values():
    ALL_AA.update(cat)


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


def compute_contacts_by_type(atoms, threshold=8.0, seq_sep=4):
    """
    Compute contacts and categorize by amino acid pair types.
    
    Returns dict of {pair_type: [distances]}
    """
    contacts = defaultdict(list)
    n = len(atoms)
    
    for i in range(n):
        for j in range(i + seq_sep, n):
            # Same chain check
            if atoms[i]['chain'] != atoms[j]['chain']:
                continue
            
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            if dist <= threshold:
                cat_i = get_aa_category(atoms[i]['res_name'])
                cat_j = get_aa_category(atoms[j]['res_name'])
                
                # Create sorted pair key
                pair = tuple(sorted([cat_i, cat_j]))
                contacts[pair].append(int(round(dist)))
                
                # Also track specific pairs for aromatic
                if cat_i == 'aromatic' and cat_j == 'aromatic':
                    res_pair = tuple(sorted([atoms[i]['res_name'], atoms[j]['res_name']]))
                    contacts[('aromatic_specific', res_pair)].append(int(round(dist)))
    
    return contacts


def calc_enrichment(distances, target_set, n_permutations=500):
    """Calculate enrichment and z-score for target distances."""
    if not distances:
        return {'enrichment': 0, 'z_score': 0, 'observed': 0, 'expected': 0}
    
    observed = sum(1 for d in distances if d in target_set)
    total = len(distances)
    
    if total == 0:
        return {'enrichment': 0, 'z_score': 0, 'observed': 0, 'expected': 0}
    
    # Expected under uniform distribution
    dist_range = range(1, max(distances) + 1) if distances else range(1, 100)
    expected_rate = len([d for d in dist_range if d in target_set]) / len(dist_range)
    expected = expected_rate * total
    
    # Permutation test
    null_counts = []
    for _ in range(n_permutations):
        shuffled = np.random.choice(list(dist_range), size=total, replace=True)
        null_count = sum(1 for d in shuffled if d in target_set)
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
    parser.add_argument('--threshold', type=float, default=8.0)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 21: AMINO ACID RESONANCE AND FIBONACCI ENRICHMENT")
    print("=" * 70)
    print(f"\nHypothesis: Aromatic contacts (π-stacking) show enhanced Fibonacci")
    print(f"            due to resonance-mediated distance preferences\n")
    
    # Get proteins
    print(f"Fetching {args.proteins} protein structures...")
    pdb_ids = fetch_protein_ids(args.proteins)
    print(f"Found {len(pdb_ids)} proteins")
    
    # Collect all contacts by pair type
    all_contacts = defaultdict(list)
    processed = 0
    
    for i, pdb_id in enumerate(pdb_ids):
        atoms = fetch_structure(pdb_id)
        if atoms is None:
            continue
        
        contacts = compute_contacts_by_type(atoms, args.threshold)
        for pair_type, dists in contacts.items():
            all_contacts[pair_type].extend(dists)
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed} proteins...")
    
    print(f"\nAnalyzed {processed} proteins successfully")
    
    # Target sets
    FIB_SET = set(FIBONACCI)
    
    # Analyze by category pair
    print("\n" + "=" * 70)
    print("FIBONACCI ENRICHMENT BY AMINO ACID PAIR TYPE")
    print("=" * 70)
    
    # Main category pairs
    main_pairs = [
        ('aromatic', 'aromatic'),
        ('charged', 'charged'),
        ('hydrophobic', 'hydrophobic'),
        ('polar', 'polar'),
        ('aromatic', 'charged'),
        ('aromatic', 'hydrophobic'),
        ('charged', 'hydrophobic'),
    ]
    
    results = {}
    
    print(f"\n{'Pair Type':<30} {'Contacts':>10} {'Fib Enrich':>12} {'Z-score':>10}")
    print("-" * 70)
    
    for pair in main_pairs:
        key = tuple(sorted(pair))
        dists = all_contacts.get(key, [])
        
        if len(dists) < 100:
            continue
        
        stats = calc_enrichment(dists, FIB_SET)
        
        pair_name = f"{pair[0]}-{pair[1]}"
        print(f"{pair_name:<30} {stats['total']:>10} {stats['enrichment']:>12.2f}x {stats['z_score']:>10.1f}")
        
        results[pair_name] = stats
    
    # Aromatic-aromatic specific breakdown
    print("\n" + "-" * 70)
    print("AROMATIC-AROMATIC BREAKDOWN (π-stacking)")
    print("-" * 70)
    
    aromatic_specific = {}
    for key, dists in all_contacts.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] == 'aromatic_specific':
            res_pair = key[1]
            aromatic_specific[res_pair] = dists
    
    print(f"\n{'Residue Pair':<20} {'Contacts':>10} {'Fib Enrich':>12} {'Z-score':>10}")
    print("-" * 60)
    
    for pair, dists in sorted(aromatic_specific.items(), key=lambda x: -len(x[1])):
        if len(dists) < 50:
            continue
        
        stats = calc_enrichment(dists, FIB_SET)
        pair_name = f"{pair[0]}-{pair[1]}"
        print(f"{pair_name:<20} {stats['total']:>10} {stats['enrichment']:>12.2f}x {stats['z_score']:>10.1f}")
        
        results[f"aromatic_{pair_name}"] = stats
    
    # Distance distribution analysis
    print("\n" + "=" * 70)
    print("DISTANCE DISTRIBUTION BY PAIR TYPE")
    print("=" * 70)
    
    for pair in [('aromatic', 'aromatic'), ('hydrophobic', 'hydrophobic'), ('charged', 'charged')]:
        key = tuple(sorted(pair))
        dists = all_contacts.get(key, [])
        
        if len(dists) < 100:
            continue
        
        print(f"\n{pair[0]}-{pair[1]}:")
        
        # Count at each Fibonacci distance
        for fib in [3, 5, 8, 13, 21]:
            count = sum(1 for d in dists if d == fib)
            pct = 100 * count / len(dists)
            bar = "█" * int(pct * 2)
            print(f"  {fib:>2}Å: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Compare aromatic to baseline
    print("\n" + "=" * 70)
    print("AROMATIC vs BASELINE COMPARISON")
    print("=" * 70)
    
    aromatic_dists = all_contacts.get(('aromatic', 'aromatic'), [])
    hydrophobic_dists = all_contacts.get(('hydrophobic', 'hydrophobic'), [])
    
    if aromatic_dists and hydrophobic_dists:
        aromatic_stats = calc_enrichment(aromatic_dists, FIB_SET)
        hydrophobic_stats = calc_enrichment(hydrophobic_dists, FIB_SET)
        
        print(f"\nAromatic-Aromatic:    {aromatic_stats['enrichment']:.2f}x (z={aromatic_stats['z_score']:.1f})")
        print(f"Hydrophobic-Hydrophobic: {hydrophobic_stats['enrichment']:.2f}x (z={hydrophobic_stats['z_score']:.1f})")
        
        ratio = aromatic_stats['enrichment'] / hydrophobic_stats['enrichment'] if hydrophobic_stats['enrichment'] > 0 else 0
        print(f"\nAromatic/Hydrophobic ratio: {ratio:.2f}x")
        
        if ratio > 1.2:
            print("✅ Aromatic contacts show ENHANCED Fibonacci enrichment")
            print("   → π-electron resonance may influence distance preferences")
        elif ratio < 0.8:
            print("❌ Aromatic contacts show REDUCED Fibonacci enrichment")
        else:
            print("≈ Aromatic and hydrophobic show similar Fibonacci enrichment")
    
    # Gap analysis - which Fibonacci numbers are most enriched?
    print("\n" + "=" * 70)
    print("FIBONACCI GAP FILLING ANALYSIS")
    print("=" * 70)
    
    print("\nWhich Fibonacci distances are over/under-represented?")
    
    for pair in [('aromatic', 'aromatic'), ('hydrophobic', 'hydrophobic')]:
        key = tuple(sorted(pair))
        dists = all_contacts.get(key, [])
        
        if len(dists) < 100:
            continue
        
        print(f"\n{pair[0]}-{pair[1]}:")
        
        # Expected uniform distribution
        total = len(dists)
        max_dist = max(dists) if dists else 100
        expected_per = total / max_dist
        
        for fib in FIBONACCI:
            if fib > max_dist:
                break
            observed = sum(1 for d in dists if d == fib)
            ratio = observed / expected_per if expected_per > 0 else 0
            
            bar = "█" * min(int(ratio * 3), 30)
            status = "✅" if ratio > 2 else "≈" if ratio > 0.5 else "❌"
            print(f"  {fib:>3}Å: {ratio:>5.1f}x expected {status} {bar}")
    
    # Save results
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    # Determine if aromatic shows enhanced enrichment
    aromatic_key = ('aromatic', 'aromatic')
    hydro_key = ('hydrophobic', 'hydrophobic')
    
    if aromatic_key in all_contacts and hydro_key in all_contacts:
        aromatic_stats = calc_enrichment(all_contacts[aromatic_key], FIB_SET)
        hydro_stats = calc_enrichment(all_contacts[hydro_key], FIB_SET)
        
        verdict = {
            'aromatic_enrichment': aromatic_stats['enrichment'],
            'aromatic_z': aromatic_stats['z_score'],
            'hydrophobic_enrichment': hydro_stats['enrichment'],
            'hydrophobic_z': hydro_stats['z_score'],
            'aromatic_enhanced': aromatic_stats['enrichment'] > hydro_stats['enrichment'] * 1.2
        }
        
        if verdict['aromatic_enhanced']:
            print("\n✅ AROMATIC CONTACTS SHOW ENHANCED FIBONACCI ENRICHMENT")
            print("   π-electron resonance appears to influence distance preferences")
        else:
            print("\n≈ No significant difference between aromatic and hydrophobic")
            print("   Fibonacci enrichment is chemistry-independent")
    
    # Save full results
    output = {
        'timestamp': datetime.now().isoformat(),
        'proteins_analyzed': processed,
        'contact_threshold': args.threshold,
        'pair_results': {str(k): v for k, v in results.items()},
        'fibonacci_set': list(FIB_SET),
        'contact_counts': {str(k): len(v) for k, v in all_contacts.items() 
                          if not isinstance(k[0], tuple)}
    }
    
    out_path = Path(__file__).parent.parent / 'results' / f'exp_21_aa_resonance_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
