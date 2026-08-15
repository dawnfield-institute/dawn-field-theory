"""
Experiment 06: Fold Hierarchy φ Analysis
=========================================

If protein folding is PAC recursion, the HIERARCHY should show φ:
- Domain contains N subdomains
- Subdomain contains M motifs
- Ratios should approach φ

Also test: Do consecutive element lengths within a protein show φ ratios?
(Not between proteins, but WITHIN the same folding unit)
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

PHI = (1 + np.sqrt(5)) / 2


def download_protein_structures():
    """
    Fetch secondary structure for multiple proteins.
    Returns organized by protein.
    """
    proteins = [
        ("Hemoglobin_alpha", "P69905"),
        ("Myoglobin", "P02144"),
        ("Lysozyme", "P61626"),
        ("Cytochrome_c", "P99999"),
        ("Ubiquitin", "P0CG48"),
        ("Insulin", "P01308"),
        ("Actin", "P60709"),
        ("Tubulin_alpha", "Q71U36"),
    ]
    
    all_structures = {}
    
    for name, uniprot_id in proteins:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode('utf-8')
            
            structures = []
            for line in text.split('\n'):
                if line.startswith('FT   HELIX') or line.startswith('FT   STRAND') or line.startswith('FT   TURN'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            struct_type = parts[1]
                            range_str = parts[2]
                            if '..' in range_str:
                                start, end = range_str.split('..')
                                start = int(start)
                                end = int(end)
                                length = end - start + 1
                                structures.append({
                                    'type': struct_type,
                                    'start': start,
                                    'end': end,
                                    'length': length
                                })
                        except:
                            continue
            
            if structures:
                all_structures[name] = sorted(structures, key=lambda x: x['start'])
                
        except Exception as e:
            print(f"  Failed {name}: {e}")
    
    return all_structures


def analyze_within_protein_phi(structures: dict):
    """
    Analyze φ ratios WITHIN each protein's fold sequence.
    
    If folding is PAC recursion, consecutive elements within
    the same protein should show φ relationships.
    """
    all_ratios = []
    protein_results = {}
    
    for protein, elements in structures.items():
        ratios = []
        for i in range(len(elements) - 1):
            l1 = elements[i]['length']
            l2 = elements[i+1]['length']
            if l1 > 0 and l2 > 0:
                ratio = max(l1, l2) / min(l1, l2)
                ratios.append(ratio)
        
        if ratios:
            phi_distances = [abs(r - PHI) for r in ratios]
            protein_results[protein] = {
                'n_ratios': len(ratios),
                'mean_ratio': np.mean(ratios),
                'mean_phi_distance': np.mean(phi_distances),
                'min_phi_distance': min(phi_distances),
                'near_phi_count': sum(1 for d in phi_distances if d < 0.2),
            }
            all_ratios.extend(ratios)
    
    return all_ratios, protein_results


def analyze_helix_sheet_ratio(structures: dict):
    """
    In each protein, compute helix_count / sheet_count.
    PAC predicts this should approach φ or 1/φ.
    """
    ratios = []
    protein_data = {}
    
    for protein, elements in structures.items():
        n_helix = sum(1 for e in elements if e['type'] == 'HELIX')
        n_strand = sum(1 for e in elements if e['type'] == 'STRAND')
        
        if n_helix > 0 and n_strand > 0:
            ratio = max(n_helix, n_strand) / min(n_helix, n_strand)
            ratios.append(ratio)
            protein_data[protein] = {
                'n_helix': n_helix,
                'n_strand': n_strand,
                'ratio': ratio,
                'distance_from_phi': abs(ratio - PHI)
            }
    
    return ratios, protein_data


def analyze_total_length_ratios(structures: dict):
    """
    Compare total helix length vs total strand length.
    """
    ratios = []
    protein_data = {}
    
    for protein, elements in structures.items():
        helix_len = sum(e['length'] for e in elements if e['type'] == 'HELIX')
        strand_len = sum(e['length'] for e in elements if e['type'] == 'STRAND')
        
        if helix_len > 0 and strand_len > 0:
            ratio = max(helix_len, strand_len) / min(helix_len, strand_len)
            ratios.append(ratio)
            protein_data[protein] = {
                'helix_length': helix_len,
                'strand_length': strand_len,
                'ratio': ratio,
                'distance_from_phi': abs(ratio - PHI)
            }
    
    return ratios, protein_data


def run_experiment():
    """Run fold hierarchy analysis."""
    print("=" * 60)
    print("Experiment 06: Fold Hierarchy φ Analysis")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Protein fold hierarchy shows φ relationships',
        'tests': {}
    }
    
    print("\n[1] Downloading protein structures...")
    structures = download_protein_structures()
    print(f"  Got {len(structures)} proteins")
    
    # Test 1: Within-protein consecutive element ratios
    print("\n[2] Within-protein consecutive element ratios...")
    ratios, protein_phi = analyze_within_protein_phi(structures)
    
    if ratios:
        mean_ratio = np.mean(ratios)
        mean_phi_dist = np.mean([abs(r - PHI) for r in ratios])
        
        print(f"\n  Total consecutive pairs: {len(ratios)}")
        print(f"  Mean ratio: {mean_ratio:.3f} (φ = 1.618)")
        print(f"  Mean distance from φ: {mean_phi_dist:.3f}")
        
        # Control: random ratio pairs
        np.random.seed(42)
        random_ratios = np.random.uniform(1, 4, size=10000)
        random_phi_dist = np.mean([abs(r - PHI) for r in random_ratios])
        random_std = np.std([np.mean([abs(r - PHI) for r in np.random.uniform(1, 4, len(ratios))]) for _ in range(1000)])
        
        z_score = (mean_phi_dist - random_phi_dist) / random_std
        
        print(f"  Random baseline φ distance: {random_phi_dist:.3f}")
        print(f"  Z-score vs random: {z_score:.2f}")
        print(f"  Closer to φ: {'YES ✅' if mean_phi_dist < random_phi_dist else 'NO ❌'}")
        
        results['tests']['consecutive_ratios'] = {
            'n_pairs': len(ratios),
            'mean_ratio': mean_ratio,
            'mean_phi_distance': mean_phi_dist,
            'z_score': z_score,
            'significant': z_score < -2
        }
        
        # Per-protein breakdown
        print("\n  Per-protein breakdown:")
        for prot, data in protein_phi.items():
            near_frac = data['near_phi_count'] / data['n_ratios'] if data['n_ratios'] > 0 else 0
            print(f"    {prot}: {data['n_ratios']} pairs, mean φ-dist={data['mean_phi_distance']:.2f}, {near_frac:.0%} near φ")
    
    # Test 2: Helix/Strand count ratios
    print("\n[3] Helix vs Strand COUNT ratios...")
    count_ratios, count_data = analyze_helix_sheet_ratio(structures)
    
    if count_ratios:
        mean_count_ratio = np.mean(count_ratios)
        mean_count_phi_dist = np.mean([abs(r - PHI) for r in count_ratios])
        
        print(f"\n  Proteins with both: {len(count_ratios)}")
        print(f"  Mean H/S ratio: {mean_count_ratio:.3f}")
        print(f"  Mean distance from φ: {mean_count_phi_dist:.3f}")
        
        for prot, data in count_data.items():
            print(f"    {prot}: H={data['n_helix']}, S={data['n_strand']}, ratio={data['ratio']:.2f}, φ-dist={data['distance_from_phi']:.2f}")
        
        results['tests']['helix_strand_count_ratio'] = {
            'n_proteins': len(count_ratios),
            'mean_ratio': mean_count_ratio,
            'mean_phi_distance': mean_count_phi_dist,
            'per_protein': count_data
        }
    
    # Test 3: Helix/Strand TOTAL LENGTH ratios
    print("\n[4] Helix vs Strand TOTAL LENGTH ratios...")
    len_ratios, len_data = analyze_total_length_ratios(structures)
    
    if len_ratios:
        mean_len_ratio = np.mean(len_ratios)
        mean_len_phi_dist = np.mean([abs(r - PHI) for r in len_ratios])
        
        print(f"\n  Mean total-length ratio: {mean_len_ratio:.3f}")
        print(f"  Mean distance from φ: {mean_len_phi_dist:.3f}")
        
        for prot, data in len_data.items():
            print(f"    {prot}: H-len={data['helix_length']}, S-len={data['strand_length']}, ratio={data['ratio']:.2f}, φ-dist={data['distance_from_phi']:.2f}")
        
        results['tests']['helix_strand_length_ratio'] = {
            'n_proteins': len(len_ratios),
            'mean_ratio': mean_len_ratio,
            'mean_phi_distance': mean_len_phi_dist,
            'per_protein': len_data
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Does fold hierarchy show φ?")
    print("=" * 60)
    
    signals = []
    
    if results['tests'].get('consecutive_ratios', {}).get('significant', False):
        signals.append("Consecutive ratios approach φ ✅")
    else:
        signals.append("Consecutive ratios NOT significantly near φ ⚠️")
    
    # Check if any protein has ratio very close to φ
    close_to_phi = []
    for prot, data in count_data.items():
        if data['distance_from_phi'] < 0.1:
            close_to_phi.append(f"{prot} (ratio={data['ratio']:.3f})")
    
    if close_to_phi:
        signals.append(f"Proteins with H/S near φ: {', '.join(close_to_phi)} ✅")
    else:
        signals.append("No protein has H/S ratio very close to φ ⚠️")
    
    for s in signals:
        print(f"  {s}")
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_06_hierarchy_phi_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_experiment()
