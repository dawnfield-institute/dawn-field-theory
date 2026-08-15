#!/usr/bin/env python3
"""
exp_23_residual_analysis.py

Proper statistical analysis: compare to expected bell curve, not uniform.
The question: After accounting for the natural distance distribution,
are Fibonacci distances STILL enriched?

Method:
1. Fit a smooth curve to the distance histogram
2. Calculate residuals (observed - expected)
3. Check if Fibonacci positions have positive residuals
"""

import argparse
import json
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

FIBONACCI = [5, 8, 13, 21, 34, 55]
GAP6_HUB = [5, 11, 17, 23, 29, 35, 41, 47, 53]  # Primes with gap 6

AA_CATEGORIES = {
    'aromatic': {'PHE', 'TYR', 'TRP', 'HIS'},
    'charged': {'LYS', 'ARG', 'ASP', 'GLU'},
    'hydrophobic': {'ALA', 'VAL', 'LEU', 'ILE', 'MET'},
    'polar': {'SER', 'THR', 'ASN', 'GLN'},
}


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


def get_aa_category(res_name):
    for cat, residues in AA_CATEGORIES.items():
        if res_name in residues:
            return cat
    return 'other'


def compute_distances(atoms, max_dist=60, seq_sep=4):
    """Compute all distances with amino acid categories."""
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
                contacts.append({
                    'dist': round(dist, 1),
                    'dist_int': int(round(dist)),
                    'pair': tuple(sorted([cat_i, cat_j]))
                })
    
    return contacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXP 23: RESIDUAL ANALYSIS - FIBONACCI AFTER BASELINE CORRECTION")
    print("=" * 70)
    print("\nMethod: Compare to smoothed baseline, not uniform distribution")
    
    # Get proteins
    print(f"\nFetching {args.proteins} proteins...")
    pdb_ids = fetch_protein_ids(args.proteins)
    
    all_contacts = []
    processed = 0
    
    for pdb_id in pdb_ids:
        atoms = fetch_structure(pdb_id)
        if atoms:
            all_contacts.extend(compute_distances(atoms, max_dist=60))
            processed += 1
            if processed % 20 == 0:
                print(f"  Processed {processed} proteins...")
    
    print(f"\nAnalyzed {processed} proteins, {len(all_contacts):,} contacts")
    
    # Build histogram at 0.1Å resolution
    dist_counts_fine = defaultdict(int)
    for c in all_contacts:
        dist_counts_fine[c['dist']] += 1
    
    # Integer histogram
    dist_counts = defaultdict(int)
    for c in all_contacts:
        dist_counts[c['dist_int']] += 1
    
    # Create smooth baseline using Gaussian filter
    distances = list(range(4, 56))
    observed = np.array([dist_counts.get(d, 0) for d in distances])
    
    # Smooth baseline (sigma=2 ~ 2Å smoothing)
    baseline = gaussian_filter1d(observed.astype(float), sigma=2)
    
    # Calculate residuals
    residuals = observed - baseline
    
    # Normalize residuals by baseline (percentage deviation)
    pct_deviation = 100 * residuals / np.maximum(baseline, 1)
    
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS: DEVIATION FROM SMOOTH BASELINE")
    print("=" * 70)
    print(f"\n{'Dist':>4} {'Observed':>10} {'Baseline':>10} {'Residual':>10} {'% Dev':>8} {'Type'}")
    print("-" * 65)
    
    fib_set = set(FIBONACCI)
    gap6_set = set(GAP6_HUB)
    
    fib_residuals = []
    gap6_residuals = []
    other_residuals = []
    
    for i, d in enumerate(distances):
        obs = observed[i]
        base = baseline[i]
        res = residuals[i]
        pct = pct_deviation[i]
        
        if d in fib_set:
            marker = "◆ FIB"
            fib_residuals.append(pct)
        elif d in gap6_set:
            marker = "◇ GAP6"
            gap6_residuals.append(pct)
        else:
            marker = ""
            other_residuals.append(pct)
        
        bar = ""
        if pct > 0:
            bar = "+" + "█" * min(int(pct / 2), 20)
        else:
            bar = "-" + "░" * min(int(-pct / 2), 20)
        
        if d in fib_set or d in gap6_set or abs(pct) > 5:
            print(f"{d:>4}Å {obs:>10,} {base:>10,.0f} {res:>+10,.0f} {pct:>+7.1f}% {marker:>8} {bar}")
    
    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    
    print(f"\nFibonacci positions ({len(fib_residuals)} points):")
    print(f"  Mean % deviation: {np.mean(fib_residuals):+.2f}%")
    print(f"  Std: {np.std(fib_residuals):.2f}%")
    
    print(f"\nGap 6 hub positions ({len(gap6_residuals)} points):")
    print(f"  Mean % deviation: {np.mean(gap6_residuals):+.2f}%")
    print(f"  Std: {np.std(gap6_residuals):.2f}%")
    
    print(f"\nOther positions ({len(other_residuals)} points):")
    print(f"  Mean % deviation: {np.mean(other_residuals):+.2f}%")
    print(f"  Std: {np.std(other_residuals):.2f}%")
    
    # T-test: Are Fibonacci residuals different from other?
    if len(fib_residuals) > 2 and len(other_residuals) > 10:
        t_stat, p_val = stats.ttest_ind(fib_residuals, other_residuals)
        print(f"\nFibonacci vs Other: t={t_stat:.2f}, p={p_val:.4f}")
        if p_val < 0.05 and np.mean(fib_residuals) > np.mean(other_residuals):
            print("  ✅ Fibonacci positions are significantly ABOVE baseline")
        elif p_val < 0.05:
            print("  ❌ Fibonacci positions are significantly BELOW baseline")
        else:
            print("  ≈ No significant difference from baseline")
    
    # By amino acid pair type
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS BY AMINO ACID PAIR TYPE")
    print("=" * 70)
    
    for pair_type in [('aromatic', 'aromatic'), ('hydrophobic', 'hydrophobic'), 
                      ('charged', 'charged'), ('polar', 'polar')]:
        key = tuple(sorted(pair_type))
        pair_contacts = [c for c in all_contacts if c['pair'] == key]
        
        if len(pair_contacts) < 1000:
            continue
        
        # Build histogram for this pair type
        pair_counts = defaultdict(int)
        for c in pair_contacts:
            pair_counts[c['dist_int']] += 1
        
        pair_obs = np.array([pair_counts.get(d, 0) for d in distances])
        pair_base = gaussian_filter1d(pair_obs.astype(float), sigma=2)
        pair_res = pair_obs - pair_base
        pair_pct = 100 * pair_res / np.maximum(pair_base, 1)
        
        # Fibonacci residuals for this pair
        pair_fib_res = [pair_pct[i] for i, d in enumerate(distances) if d in fib_set]
        pair_other_res = [pair_pct[i] for i, d in enumerate(distances) if d not in fib_set and d not in gap6_set]
        
        print(f"\n{pair_type[0]}-{pair_type[1]} ({len(pair_contacts):,} contacts):")
        print(f"  Fibonacci mean deviation: {np.mean(pair_fib_res):+.2f}%")
        print(f"  Other mean deviation: {np.mean(pair_other_res):+.2f}%")
        print(f"  Fib - Other: {np.mean(pair_fib_res) - np.mean(pair_other_res):+.2f}%")
        
        # Highlight if aromatic shows stronger Fibonacci preference
        if pair_type == ('aromatic', 'aromatic'):
            aromatic_fib_advantage = np.mean(pair_fib_res) - np.mean(pair_other_res)
        elif pair_type == ('hydrophobic', 'hydrophobic'):
            hydrophobic_fib_advantage = np.mean(pair_fib_res) - np.mean(pair_other_res)
    
    # Compare aromatic vs hydrophobic
    print("\n" + "=" * 70)
    print("AROMATIC vs HYDROPHOBIC COMPARISON")
    print("=" * 70)
    
    try:
        print(f"\nAromatic Fibonacci advantage: {aromatic_fib_advantage:+.2f}%")
        print(f"Hydrophobic Fibonacci advantage: {hydrophobic_fib_advantage:+.2f}%")
        
        if aromatic_fib_advantage > hydrophobic_fib_advantage + 1:
            print("\n✅ Aromatic pairs show STRONGER Fibonacci preference")
            print("   → π-electron resonance may contribute to distance selection")
        elif aromatic_fib_advantage < hydrophobic_fib_advantage - 1:
            print("\n⬇ Hydrophobic pairs show stronger Fibonacci preference")
        else:
            print("\n≈ Similar Fibonacci preference across pair types")
            print("   → Fibonacci pattern is chemistry-independent")
    except:
        pass
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'proteins': processed,
        'contacts': len(all_contacts),
        'fibonacci_mean_deviation': float(np.mean(fib_residuals)),
        'gap6_mean_deviation': float(np.mean(gap6_residuals)),
        'other_mean_deviation': float(np.mean(other_residuals)),
        'distance_histogram': {str(d): int(dist_counts.get(d, 0)) for d in distances},
        'residuals': {str(d): float(pct_deviation[i]) for i, d in enumerate(distances)}
    }
    
    out_path = Path(__file__).parent.parent / 'results' / f'exp_23_residuals_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
