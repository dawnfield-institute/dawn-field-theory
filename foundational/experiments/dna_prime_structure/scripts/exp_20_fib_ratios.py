"""
Experiment 20: Fibonacci Ratio Analysis in Contact Distances
=============================================================

Deep dive into whether consecutive contact distances show
the characteristic Fibonacci ratios:

KEY RATIOS FROM DAWN FIELD THEORY:
==================================
1. φ = 1.618... (golden ratio, lim F_{n+1}/F_n)
2. 1/φ = 0.618... (inverse golden)
3. 2/3 = F₃/F₄ (Koide, quarks, turbulence)
4. 3/5 = F₄/F₅ (early Fibonacci)
5. 5/8 = F₅/F₆ 
6. 8/13 = F₆/F₇
7. 13/21 = F₇/F₈

From SEC Prime Manifold:
- φ-threshold at 0.6184 (0.04% from 1/φ)
- Fibonacci cascade: 2/3 → 1/φ → 3/5

From Oscillation Attractor Dynamics:
- Transition probability ratios → φ
- Mean gap ratio 1.466 → φ
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request
import time

# =============================================================================
# KEY CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
INV_PHI = 1 / PHI  # 0.618033988749895

# Key Fibonacci ratios (approaching φ as n increases)
FIB_RATIOS = {
    '2/3': 2/3,           # 0.6667 - F₃/F₄
    '3/5': 3/5,           # 0.6000 - F₄/F₅
    '5/8': 5/8,           # 0.6250 - F₅/F₆
    '8/13': 8/13,         # 0.6154 - F₆/F₇
    '13/21': 13/21,       # 0.6190 - F₇/F₈
    '21/34': 21/34,       # 0.6176 - F₈/F₉
    '34/55': 34/55,       # 0.6182 - F₉/F₁₀
    '1/φ': INV_PHI,       # 0.6180 - limit
}

FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])


def fetch_pdb_list(max_proteins=300):
    """Fetch representative PDB list."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less", "value": 2.0}}
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_proteins * 2}}
    }
    
    try:
        import json as json_mod
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        req = urllib.request.Request(url, data=json_mod.dumps(query).encode('utf-8'),
                                      headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json_mod.loads(response.read().decode('utf-8'))
            return [hit['identifier'] for hit in data.get('result_set', [])]
    except:
        return []


def fetch_pdb_structure(pdb_id):
    """Fetch CA coordinates from PDB."""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        ca_coords = []
        seen_resnum = set()
        
        for line in pdb_data.split('\n'):
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
                    ca_coords.append(np.array([x, y, z]))
                except:
                    continue
        
        return ca_coords if len(ca_coords) >= 30 else None
    except:
        return None


def compute_contacts(coords, threshold=8.0):
    """Compute all contact distances."""
    n = len(coords)
    contacts = []
    
    for i in range(n):
        for j in range(i + 5, n):
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                seq_dist = j - i
                contacts.append({
                    'i': i, 'j': j, 'seq_dist': seq_dist, 'dist_3d': dist_3d
                })
    
    return contacts


def analyze_consecutive_ratios(contacts):
    """
    Analyze ratios between consecutive contact distances.
    Consecutive means contacts sorted by sequence position.
    """
    if len(contacts) < 100:
        return None
    
    # Sort by first residue position
    sorted_contacts = sorted(contacts, key=lambda c: (c['i'], c['j']))
    
    # Get consecutive distance ratios
    ratios = []
    for k in range(len(sorted_contacts) - 1):
        d1 = sorted_contacts[k]['seq_dist']
        d2 = sorted_contacts[k + 1]['seq_dist']
        if d1 > 0 and d2 > 0:
            # Always take smaller/larger to get ratio < 1
            ratio = min(d1, d2) / max(d1, d2)
            ratios.append(ratio)
    
    return np.array(ratios)


def analyze_all_pair_ratios(distances):
    """
    Analyze ratios between all unique distance pairs.
    """
    unique_dists = sorted(set(distances))
    if len(unique_dists) < 10:
        return None
    
    ratios = []
    for i in range(len(unique_dists)):
        for j in range(i + 1, len(unique_dists)):
            d1, d2 = unique_dists[i], unique_dists[j]
            if d2 > 0:
                ratio = d1 / d2
                ratios.append(ratio)
    
    return np.array(ratios)


def count_near_target(ratios, target, tolerance=0.02):
    """Count ratios near a target value."""
    return sum(1 for r in ratios if abs(r - target) < tolerance)


def test_ratio_enrichment(ratios, target_name, target_value, tolerance=0.02):
    """Test if ratios cluster near a target value more than random."""
    if len(ratios) < 100:
        return None
    
    observed = count_near_target(ratios, target_value, tolerance)
    
    # Expected under uniform distribution [0, 1]
    expected = len(ratios) * (2 * tolerance)  # fraction of [0,1] within tolerance
    
    # Z-score via permutation
    np.random.seed(42)
    null_counts = []
    for _ in range(500):
        rand_ratios = np.random.uniform(0, 1, size=len(ratios))
        null_counts.append(count_near_target(rand_ratios, target_value, tolerance))
    
    z = (observed - np.mean(null_counts)) / (np.std(null_counts) + 1e-10)
    
    return {
        'target': target_name,
        'target_value': target_value,
        'observed': observed,
        'expected': expected,
        'enrichment': observed / expected if expected > 0 else 0,
        'z': z,
    }


def analyze_ratio_distribution(ratios):
    """Analyze the full ratio distribution for peaks."""
    if len(ratios) < 100:
        return None
    
    # Histogram
    bins = np.linspace(0, 1, 51)
    hist, edges = np.histogram(ratios, bins=bins)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks_idx, properties = find_peaks(hist, height=np.mean(hist) + np.std(hist))
    
    peaks = []
    for idx in peaks_idx:
        bin_center = (edges[idx] + edges[idx + 1]) / 2
        count = hist[idx]
        peaks.append({'ratio': bin_center, 'count': int(count)})
    
    # Sort by count
    peaks = sorted(peaks, key=lambda x: -x['count'])[:10]
    
    return {
        'histogram': hist.tolist(),
        'bin_edges': edges.tolist(),
        'peaks': peaks,
        'mean': float(np.mean(ratios)),
        'median': float(np.median(ratios)),
        'std': float(np.std(ratios)),
    }


def run_experiment(max_proteins=300):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 20: Fibonacci Ratio Analysis in Contact Distances")
    print("=" * 70)
    print("\nKey ratios being tested:")
    for name, value in FIB_RATIOS.items():
        print(f"  {name}: {value:.4f}")
    print()
    
    pdb_ids = fetch_pdb_list(max_proteins)
    print(f"Fetched {len(pdb_ids)} PDB IDs")
    
    all_contacts = []
    all_distances = []
    processed = 0
    
    print("\nProcessing structures...")
    for i, pdb_id in enumerate(pdb_ids):
        if processed >= max_proteins:
            break
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] Processed {processed}")
        
        coords = fetch_pdb_structure(pdb_id)
        if coords is None:
            continue
        
        if len(coords) < 50 or len(coords) > 500:
            continue
        
        contacts = compute_contacts(coords)
        if len(contacts) >= 20:
            all_contacts.extend(contacts)
            all_distances.extend([c['seq_dist'] for c in contacts])
            processed += 1
        
        time.sleep(0.1)
    
    print(f"\nProcessed {processed} proteins, {len(all_contacts)} contacts")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_proteins': processed,
        'n_contacts': len(all_contacts),
        'analyses': {},
    }
    
    # Analyze consecutive ratios
    print("\n" + "=" * 70)
    print("CONSECUTIVE CONTACT RATIO ANALYSIS")
    print("=" * 70)
    
    cons_ratios = analyze_consecutive_ratios(all_contacts)
    
    if cons_ratios is not None and len(cons_ratios) > 100:
        print(f"\nAnalyzing {len(cons_ratios)} consecutive contact ratios...")
        
        results['analyses']['consecutive'] = {'n_ratios': len(cons_ratios), 'tests': {}}
        
        print("\nFibonacci Ratio Enrichment:")
        print("-" * 60)
        
        for name, target in FIB_RATIOS.items():
            test = test_ratio_enrichment(cons_ratios, name, target, tolerance=0.02)
            if test:
                sig = "✅" if test['z'] > 3 else ("⚠️" if test['z'] > 2 else "❌")
                print(f"  {name:12s} ({target:.4f}): {test['enrichment']:5.2f}x (z={test['z']:6.1f}) {sig}")
                results['analyses']['consecutive']['tests'][name] = test
        
        # Distribution analysis
        dist_analysis = analyze_ratio_distribution(cons_ratios)
        if dist_analysis:
            print(f"\nDistribution:")
            print(f"  Mean: {dist_analysis['mean']:.4f}")
            print(f"  Median: {dist_analysis['median']:.4f}")
            print(f"  Std: {dist_analysis['std']:.4f}")
            print(f"\nTop peaks:")
            for peak in dist_analysis['peaks'][:5]:
                # Check if near any Fibonacci ratio
                near_fib = None
                for name, target in FIB_RATIOS.items():
                    if abs(peak['ratio'] - target) < 0.03:
                        near_fib = name
                        break
                marker = f" ← {near_fib}" if near_fib else ""
                print(f"    {peak['ratio']:.3f}: {peak['count']} contacts{marker}")
            
            results['analyses']['consecutive']['distribution'] = dist_analysis
    
    # Analyze all pair ratios
    print("\n" + "=" * 70)
    print("ALL UNIQUE DISTANCE PAIR RATIOS")
    print("=" * 70)
    
    pair_ratios = analyze_all_pair_ratios(all_distances)
    
    if pair_ratios is not None and len(pair_ratios) > 100:
        print(f"\nAnalyzing {len(pair_ratios)} unique distance pair ratios...")
        
        results['analyses']['all_pairs'] = {'n_ratios': len(pair_ratios), 'tests': {}}
        
        print("\nFibonacci Ratio Enrichment:")
        print("-" * 60)
        
        for name, target in FIB_RATIOS.items():
            test = test_ratio_enrichment(pair_ratios, name, target, tolerance=0.01)
            if test:
                sig = "✅" if test['z'] > 3 else ("⚠️" if test['z'] > 2 else "❌")
                print(f"  {name:12s} ({target:.4f}): {test['enrichment']:5.2f}x (z={test['z']:6.1f}) {sig}")
                results['analyses']['all_pairs']['tests'][name] = test
    
    # Test specific Fibonacci distance ratios
    print("\n" + "=" * 70)
    print("EXACT FIBONACCI DISTANCE PAIR ANALYSIS")
    print("=" * 70)
    
    dist_counts = defaultdict(int)
    for d in all_distances:
        dist_counts[d] += 1
    
    print("\nFibonacci distance pair co-occurrence:")
    fib_pairs = [
        (5, 8), (8, 13), (13, 21), (21, 34), (34, 55), (55, 89),
    ]
    
    results['analyses']['fib_pairs'] = {}
    
    for a, b in fib_pairs:
        count_a = dist_counts.get(a, 0)
        count_b = dist_counts.get(b, 0)
        ratio = a / b
        
        if count_a > 0 and count_b > 0:
            # Correlation: do proteins with high a also have high b?
            cooccur = min(count_a, count_b)
            print(f"  ({a:2d}, {b:2d}) ratio={ratio:.3f}: a={count_a:5d}, b={count_b:5d}, cooccur={cooccur}")
            results['analyses']['fib_pairs'][f'{a}/{b}'] = {
                'a': a, 'b': b, 'count_a': count_a, 'count_b': count_b, 
                'ratio': ratio, 'cooccur': cooccur
            }
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT: FIBONACCI RATIOS IN PROTEIN STRUCTURE")
    print("=" * 70)
    
    significant_ratios = []
    
    if 'consecutive' in results['analyses']:
        for name, test in results['analyses']['consecutive'].get('tests', {}).items():
            if test['z'] > 3:
                significant_ratios.append((name, test['z'], 'consecutive'))
    
    if 'all_pairs' in results['analyses']:
        for name, test in results['analyses']['all_pairs'].get('tests', {}).items():
            if test['z'] > 3:
                significant_ratios.append((name, test['z'], 'all_pairs'))
    
    if significant_ratios:
        print("\n🔬 SIGNIFICANT FIBONACCI RATIOS DETECTED:")
        for name, z, source in sorted(significant_ratios, key=lambda x: -x[1]):
            print(f"   {name}: z={z:.1f} ({source})")
        
        # Check if 1/φ is enriched
        inv_phi_results = [r for r in significant_ratios if r[0] == '1/φ']
        if inv_phi_results:
            print(f"\n   ✅ Golden ratio inverse (1/φ = 0.618) significantly enriched!")
            print("      This is the theoretical limit of F_{n}/F_{n+1}")
    else:
        print("\n   No significant Fibonacci ratio enrichment detected.")
    
    results['significant_ratios'] = [{'name': n, 'z': z, 'source': s} for n, z, s in significant_ratios]
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_20_fib_ratios_{timestamp}.json')
    
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
