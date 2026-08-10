"""
Experiment 16: Intrinsically Disordered Proteins (IDPs)
========================================================

IDPs lack stable 3D structure - they're "natively unfolded."
If Fibonacci patterns appear in contact maps of disordered proteins,
it would be surprising and suggest sequence-level organization.

However, IDPs by definition don't have stable contacts, so we test:
1. Transient contacts in NMR ensembles of IDPs
2. Sequence-based patterns (repeat spacing without 3D structure)
"""

import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime
import urllib.request
import time

FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
FAR_FIBONACCI = set([21, 34, 55, 89])

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(500)


def fetch_disordered_proteins(max_proteins=100):
    """Fetch IDP PDB IDs - these are NMR structures with high B-factors."""
    print("Fetching intrinsically disordered proteins...")
    
    # Query for NMR structures (often capture disorder)
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "SOLUTION NMR"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct_keywords.pdbx_keywords",
                        "operator": "contains_words",
                        "value": "INTRINSICALLY DISORDERED"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_proteins * 3}
        }
    }
    
    try:
        import json as json_mod
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        req = urllib.request.Request(
            url, data=json_mod.dumps(query).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json_mod.loads(response.read().decode('utf-8'))
            pdb_ids = [hit['identifier'] for hit in data.get('result_set', [])]
            print(f"  Found {len(pdb_ids)} IDPs")
            if pdb_ids:
                return pdb_ids
    except Exception as e:
        print(f"  IDP query failed: {e}")
    
    # Fallback: known IDPs
    print("  Using fallback IDP list...")
    return [
        # Known IDPs with some structural data
        "1CD3", "1F0R", "2FFT", "2K0P", "2KJ3", "2L3S", "2LEA", "2LHC",
        "2LMN", "2LPL", "2M0J", "2M10", "2M2F", "2M3B", "2M55", "2MOM",
        "2MPZ", "2MXU", "2N0A", "2N1F", "5UGO", "5WHN", "6CU7", "6GWP",
    ]


def fetch_pdb_nmr_ensemble(pdb_id):
    """Fetch NMR ensemble - returns multiple models."""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url, timeout=30) as response:
            pdb_data = response.read().decode('utf-8')
        
        models = []
        current_model = []
        seen_resnum = set()
        
        for line in pdb_data.split('\n'):
            if line.startswith('MODEL'):
                current_model = []
                seen_resnum = set()
            elif line.startswith('ENDMDL'):
                if len(current_model) >= 20:
                    models.append(current_model)
            elif line.startswith('ATOM') and ' CA ' in line:
                try:
                    chain = line[21]
                    resnum = int(line[22:26])
                    
                    if chain != 'A' or resnum in seen_resnum:
                        continue
                    seen_resnum.add(resnum)
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_model.append(np.array([x, y, z]))
                except:
                    continue
        
        # If no MODEL records, treat whole file as one model
        if not models and current_model:
            models = [current_model]
        
        return models if models else None
    except:
        return None


def compute_ensemble_contacts(models, threshold=8.0):
    """
    Compute contacts that appear in multiple NMR models.
    Returns distance and frequency across ensemble.
    """
    if not models:
        return []
    
    n = len(models[0])
    contact_freq = defaultdict(int)
    
    for coords in models:
        if len(coords) != n:
            continue
        for i in range(n):
            for j in range(i + 5, n):
                dist_3d = np.linalg.norm(coords[i] - coords[j])
                if dist_3d < threshold:
                    contact_freq[(i, j)] += 1
    
    # Only keep contacts present in >50% of models
    n_models = len(models)
    stable_distances = []
    transient_distances = []
    
    for (i, j), count in contact_freq.items():
        seq_dist = j - i
        freq = count / n_models
        if freq > 0.5:
            stable_distances.append(seq_dist)
        elif freq > 0.1:
            transient_distances.append(seq_dist)
    
    return stable_distances, transient_distances


def analyze_sequence_repeats(sequence):
    """Analyze repeat spacing in amino acid sequence."""
    if len(sequence) < 20:
        return []
    
    distances = []
    for aa in set(sequence):
        positions = [i for i, c in enumerate(sequence) if c == aa]
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            if 5 <= gap <= 100:
                distances.append(gap)
    
    return distances


def analyze_enrichment(distances, target_set, name, max_dist=200):
    """Calculate enrichment."""
    if len(distances) < 30:
        return None
    
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 30:
        return None
    
    observed = sum(1 for d in distances if d in target_set)
    targets_in_range = len([t for t in target_set if 5 <= t <= max_dist])
    expected = len(distances) * targets_in_range / (max_dist - 4)
    enrichment = observed / expected if expected > 0 else 0
    
    np.random.seed(42)
    null_counts = []
    for _ in range(500):
        rand_dists = np.random.randint(5, max_dist + 1, size=len(distances))
        null_counts.append(sum(1 for d in rand_dists if d in target_set))
    
    z_score = (observed - np.mean(null_counts)) / (np.std(null_counts) + 1e-10)
    
    return {
        'name': name,
        'n_distances': len(distances),
        'observed': observed,
        'expected': float(expected),
        'enrichment': float(enrichment),
        'z_score': float(z_score),
    }


def run_experiment(max_proteins=80):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 16: Intrinsically Disordered Proteins")
    print("=" * 70)
    
    pdb_ids = fetch_disordered_proteins(max_proteins)
    
    all_stable = []
    all_transient = []
    processed = 0
    
    print("\nProcessing IDPs...")
    for i, pdb_id in enumerate(pdb_ids):
        if processed >= max_proteins:
            break
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}] Processed {processed}")
        
        models = fetch_pdb_nmr_ensemble(pdb_id)
        if models is None:
            continue
        
        if len(models[0]) < 30 or len(models[0]) > 500:
            continue
        
        stable, transient = compute_ensemble_contacts(models)
        
        if stable or transient:
            all_stable.extend(stable)
            all_transient.extend(transient)
            processed += 1
        
        time.sleep(0.15)
    
    print(f"\nProcessed {processed} IDPs")
    print(f"Stable contacts: {len(all_stable)}")
    print(f"Transient contacts: {len(all_transient)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'protein_type': 'intrinsically_disordered',
        'n_proteins': processed,
        'n_stable_contacts': len(all_stable),
        'n_transient_contacts': len(all_transient),
        'analyses': {},
    }
    
    print("\n" + "=" * 70)
    print("STABLE CONTACTS (>50% of ensemble)")
    print("=" * 70)
    
    targets = [
        ('Fibonacci (all)', FIBONACCI),
        ('Far-Fibonacci', FAR_FIBONACCI),
        ('Primes', PRIMES),
    ]
    
    results['analyses']['stable'] = {}
    for name, target in targets:
        analysis = analyze_enrichment(all_stable, target, name)
        if analysis:
            sig = "✅" if analysis['z_score'] > 3 else "❌"
            print(f"  {name:25s}: {analysis['enrichment']:5.2f}x (z={analysis['z_score']:6.1f}) {sig}")
            results['analyses']['stable'][name] = analysis
        else:
            print(f"  {name:25s}: insufficient data")
    
    print("\n" + "=" * 70)
    print("TRANSIENT CONTACTS (10-50% of ensemble)")
    print("=" * 70)
    
    results['analyses']['transient'] = {}
    for name, target in targets:
        analysis = analyze_enrichment(all_transient, target, name)
        if analysis:
            sig = "✅" if analysis['z_score'] > 3 else "❌"
            print(f"  {name:25s}: {analysis['enrichment']:5.2f}x (z={analysis['z_score']:6.1f}) {sig}")
            results['analyses']['transient'][name] = analysis
        else:
            print(f"  {name:25s}: insufficient data")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    stable_fib = results['analyses'].get('stable', {}).get('Fibonacci (all)', {})
    stable_sig = stable_fib.get('z_score', 0) > 3 if stable_fib else False
    
    transient_fib = results['analyses'].get('transient', {}).get('Fibonacci (all)', {})
    transient_sig = transient_fib.get('z_score', 0) > 3 if transient_fib else False
    
    if stable_sig or transient_sig:
        print(f"\n✅ IDPs: Fibonacci pattern DETECTED")
        if stable_sig:
            print(f"   Stable contacts: z={stable_fib.get('z_score', 0):.1f}")
        if transient_sig:
            print(f"   Transient contacts: z={transient_fib.get('z_score', 0):.1f}")
        print("   → Pattern exists even in disordered proteins!")
    else:
        print(f"\n❌ IDPs: No significant Fibonacci enrichment")
        print("   → Pattern may require stable 3D structure")
    
    results['verdict'] = {
        'stable_fib_significant': stable_sig,
        'transient_fib_significant': transient_sig,
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_16_idp_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {filepath}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=80)
    args = parser.parse_args()
    run_experiment(max_proteins=args.proteins)
