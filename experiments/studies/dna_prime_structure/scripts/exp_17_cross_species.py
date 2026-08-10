"""
Experiment 17: Cross-Species Conservation
==========================================

Test if Fibonacci enrichment in protein contacts is conserved
across evolutionary distant species:
1. Human proteins
2. E. coli (bacteria)
3. Yeast (fungi)
4. Archaea
5. Thermophiles (extreme environment)

If pattern is universal across all life, it suggests fundamental constraint.
If pattern varies, it may be lineage-specific optimization.
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

# Organism-specific queries
ORGANISMS = {
    'human': {
        'name': 'Homo sapiens',
        'tax_id': '9606',
    },
    'ecoli': {
        'name': 'Escherichia coli',
        'tax_id': '562',
    },
    'yeast': {
        'name': 'Saccharomyces cerevisiae',
        'tax_id': '4932',
    },
    'archaea': {
        'name': 'Archaea (various)',
        'tax_ids': ['2157'],  # Archaea superkingdom
    },
    'thermophile': {
        'name': 'Thermus thermophilus',
        'tax_id': '274',
    },
}


def fetch_organism_proteins(organism_key, max_proteins=100):
    """Fetch PDB IDs for a specific organism."""
    org = ORGANISMS[organism_key]
    print(f"  Fetching {org['name']}...")
    
    tax_id = org.get('tax_id', org.get('tax_ids', [''])[0])
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                        "operator": "exact_match",
                        "value": tax_id
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 2.5
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_proteins * 2}
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
            print(f"    Found {len(pdb_ids)} structures")
            return pdb_ids[:max_proteins * 2]
    except Exception as e:
        print(f"    Query failed: {e}")
        return []


def fetch_pdb_structure(pdb_id):
    """Fetch PDB structure."""
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
    """Compute contact distances."""
    n = len(coords)
    distances = []
    
    for i in range(n):
        for j in range(i + 5, n):
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                distances.append(j - i)
    
    return distances


def analyze_enrichment(distances, target_set, max_dist=200):
    """Calculate enrichment."""
    if len(distances) < 50:
        return None
    
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 50:
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
        'n_distances': len(distances),
        'observed': observed,
        'expected': float(expected),
        'enrichment': float(enrichment),
        'z_score': float(z_score),
    }


def run_experiment(proteins_per_org=60):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 17: Cross-Species Conservation")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'proteins_per_organism': proteins_per_org,
        'organisms': {},
    }
    
    all_enrichments = []
    
    for org_key in ORGANISMS.keys():
        org = ORGANISMS[org_key]
        print(f"\n[{org['name'].upper()}]")
        
        pdb_ids = fetch_organism_proteins(org_key, proteins_per_org)
        
        all_distances = []
        processed = 0
        
        for pdb_id in pdb_ids:
            if processed >= proteins_per_org:
                break
            
            coords = fetch_pdb_structure(pdb_id)
            if coords is None:
                continue
            
            if len(coords) < 50 or len(coords) > 500:
                continue
            
            distances = compute_contacts(coords)
            if len(distances) >= 20:
                all_distances.extend(distances)
                processed += 1
            
            time.sleep(0.1)
        
        print(f"  Processed {processed} proteins, {len(all_distances)} contacts")
        
        org_results = {
            'n_proteins': processed,
            'n_contacts': len(all_distances),
        }
        
        # Analyze
        fib_analysis = analyze_enrichment(all_distances, FIBONACCI)
        far_fib_analysis = analyze_enrichment(all_distances, FAR_FIBONACCI)
        prime_analysis = analyze_enrichment(all_distances, PRIMES)
        
        if fib_analysis:
            sig = "✅" if fib_analysis['z_score'] > 3 else "❌"
            print(f"  Fibonacci: {fib_analysis['enrichment']:.2f}x (z={fib_analysis['z_score']:.1f}) {sig}")
            org_results['fibonacci'] = fib_analysis
            all_enrichments.append((org_key, fib_analysis['enrichment'], fib_analysis['z_score']))
        
        if far_fib_analysis:
            print(f"  Far-Fib:   {far_fib_analysis['enrichment']:.2f}x (z={far_fib_analysis['z_score']:.1f})")
            org_results['far_fibonacci'] = far_fib_analysis
        
        if prime_analysis:
            print(f"  Primes:    {prime_analysis['enrichment']:.2f}x (z={prime_analysis['z_score']:.1f})")
            org_results['primes'] = prime_analysis
        
        results['organisms'][org_key] = org_results
    
    # Summary
    print("\n" + "=" * 70)
    print("CROSS-SPECIES COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Organism':<20} {'Enrichment':>12} {'Z-score':>10} {'Significant':>12}")
    print("-" * 56)
    
    for org_key, enrichment, z in all_enrichments:
        org_name = ORGANISMS[org_key]['name'][:18]
        sig = "✅" if z > 3 else "❌"
        print(f"{org_name:<20} {enrichment:>12.2f}x {z:>10.1f} {sig:>12}")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    sig_count = sum(1 for _, _, z in all_enrichments if z > 3)
    
    if sig_count == len(all_enrichments):
        print(f"\n✅ UNIVERSAL CONSERVATION: All {len(all_enrichments)} organisms show significant Fibonacci enrichment")
        print("   → Pattern is fundamental across all life!")
    elif sig_count > len(all_enrichments) / 2:
        print(f"\n✅ MAJORITY CONSERVATION: {sig_count}/{len(all_enrichments)} organisms show significant enrichment")
        print("   → Pattern is widespread but not universal")
    else:
        print(f"\n❌ LIMITED CONSERVATION: Only {sig_count}/{len(all_enrichments)} organisms significant")
    
    # Check variance in enrichment
    if all_enrichments:
        enrichments = [e for _, e, _ in all_enrichments]
        mean_e = np.mean(enrichments)
        std_e = np.std(enrichments)
        cv = std_e / mean_e if mean_e > 0 else 0
        
        print(f"\n   Mean enrichment: {mean_e:.2f}x (CV={cv:.2f})")
        if cv < 0.3:
            print("   → Low variation suggests fundamental constraint")
        else:
            print("   → High variation suggests lineage-specific adaptation")
    
    results['summary'] = {
        'organisms_tested': len(all_enrichments),
        'organisms_significant': sig_count,
        'universal': sig_count == len(all_enrichments),
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_17_cross_species_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {filepath}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=60)
    args = parser.parse_args()
    run_experiment(proteins_per_org=args.proteins)
