"""
Experiment 19: Dawn Field Theory Constants in Biomolecular Structures
======================================================================

Tests whether the key constants from Dawn Field Theory experiments
appear in biomolecular 3D contact distances:

KEY CONSTANTS TO TEST:
======================
1. Fibonacci numbers: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144
2. F₇ = 13 (Gauge closure, appears in DOF sum 1+3+8+1=13)
3. F₁₀ = 55 (EM recursion depth, edge-of-chaos)
4. Ξ-related: distances where d/φ or d*φ ≈ integer
5. 2/3-related: F₃/F₄ = 2/3 ratios in consecutive distances
6. Möbius pairs: (a,b)↔(b,a) symmetry in contact pairs
7. Gap 6 hub (from oscillation_attractor_dynamics)
8. φ-scaled triplets: distances in ratio φ:1:1/φ

From milestone1/constants.py:
- PHI = 1.618033988749895
- XI = 1.0571081... (1 + π/55)
- F₇ = 13 (gauge closure)
- F₁₀ = 55 (EM recursion)

From oscillation_attractor_dynamics:
- Gap 6 is the Möbius hub (31 connections)
- (a,b)↔(b,a) pairs at 24x random
- 70.4% alternation in gaps
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
# DAWN FIELD THEORY CONSTANTS (from milestone1)
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PSI = PHI - 1  # 0.618033988749895 = 1/φ
XI = 1 + np.pi / 55  # 1.0571081...

# Fibonacci numbers
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
FIB_SET = set(FIBONACCI)

# Key Fibonacci (from pac_confluence_xi)
F7 = 13   # Gauge closure, DOF sum 1+3+8+1=13
F10 = 55  # EM recursion depth

# Numbers that are distance*φ or distance/φ from a Fibonacci
# These would indicate φ-scaling in structure
PHI_SCALED = set()
for f in FIBONACCI:
    phi_up = round(f * PHI)
    phi_down = round(f / PHI)
    if phi_up not in FIB_SET:
        PHI_SCALED.add(phi_up)
    if phi_down > 0 and phi_down not in FIB_SET:
        PHI_SCALED.add(phi_down)

# Möbius hub distances (from oscillation_attractor_dynamics)
# Gap 6 is the hub, (4,6) and (6,4) are strongest pairs
MOBIUS_HUB = set([6, 4, 10])  # 4+6=10

# Prime gaps that show Möbius pairing
MOBIUS_PAIRS = [(2, 4), (4, 2), (2, 6), (6, 2), (4, 6), (6, 4), (6, 8), (8, 6)]

# 2/3 related: consecutive distances with ratio 2/3 or 3/2
TWO_THIRDS = 2/3
THREE_HALVES = 3/2

def sieve_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return set(i for i in range(n + 1) if is_prime[i])

PRIMES = sieve_primes(500)

# =============================================================================
# PDB FETCHING
# =============================================================================

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
    distances = []
    contact_pairs = []  # (i, j, dist) for pair analysis
    
    for i in range(n):
        for j in range(i + 5, n):
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            if dist_3d < threshold:
                seq_dist = j - i
                distances.append(seq_dist)
                contact_pairs.append((i, j, seq_dist))
    
    return distances, contact_pairs

# =============================================================================
# DAWN FIELD THEORY TESTS
# =============================================================================

def test_fibonacci_enrichment(distances, max_dist=200):
    """Standard Fibonacci enrichment test."""
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 100:
        return None
    
    observed = sum(1 for d in distances if d in FIB_SET)
    targets_in_range = len([f for f in FIBONACCI if 5 <= f <= max_dist])
    expected = len(distances) * targets_in_range / (max_dist - 4)
    
    np.random.seed(42)
    null_counts = [sum(1 for d in np.random.randint(5, max_dist + 1, size=len(distances)) if d in FIB_SET) for _ in range(500)]
    z = (observed - np.mean(null_counts)) / (np.std(null_counts) + 1e-10)
    
    return {'observed': observed, 'expected': expected, 'enrichment': observed/expected if expected > 0 else 0, 'z': z}


def test_f7_f10_enrichment(distances, max_dist=200):
    """Test enrichment at F₇=13 and F₁₀=55 specifically."""
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 100:
        return None
    
    count_13 = sum(1 for d in distances if d == 13)
    count_55 = sum(1 for d in distances if d == 55)
    
    # Expected if uniform
    expected_per_value = len(distances) / (max_dist - 4)
    
    return {
        'F7_13': {'count': count_13, 'expected': expected_per_value, 'enrichment': count_13 / expected_per_value if expected_per_value > 0 else 0},
        'F10_55': {'count': count_55, 'expected': expected_per_value, 'enrichment': count_55 / expected_per_value if expected_per_value > 0 else 0},
    }


def test_phi_scaled_distances(distances, max_dist=200):
    """Test enrichment at φ-scaled distances."""
    distances = [d for d in distances if 5 <= d <= max_dist]
    if len(distances) < 100:
        return None
    
    observed = sum(1 for d in distances if d in PHI_SCALED)
    targets_in_range = len([p for p in PHI_SCALED if 5 <= p <= max_dist])
    expected = len(distances) * targets_in_range / (max_dist - 4)
    
    np.random.seed(42)
    null_counts = [sum(1 for d in np.random.randint(5, max_dist + 1, size=len(distances)) if d in PHI_SCALED) for _ in range(500)]
    z = (observed - np.mean(null_counts)) / (np.std(null_counts) + 1e-10)
    
    return {'observed': observed, 'expected': expected, 'enrichment': observed/expected if expected > 0 else 0, 'z': z}


def test_gap_6_hub(distances):
    """Test if distance 6 is a 'hub' in the contact network."""
    dist_counts = defaultdict(int)
    for d in distances:
        dist_counts[d] += 1
    
    if len(dist_counts) < 10:
        return None
    
    count_6 = dist_counts.get(6, 0)
    total = sum(dist_counts.values())
    fraction_6 = count_6 / total if total > 0 else 0
    
    # Expected fraction if uniform across observed range
    n_values = len(dist_counts)
    expected_frac = 1 / n_values if n_values > 0 else 0
    
    return {
        'count_6': count_6,
        'fraction': fraction_6,
        'expected_fraction': expected_frac,
        'enrichment': fraction_6 / expected_frac if expected_frac > 0 else 0,
    }


def test_mobius_pair_symmetry(contact_pairs):
    """
    Test for Möbius (a,b)↔(b,a) symmetry in consecutive contacts.
    From oscillation_attractor_dynamics: 24x random rate.
    """
    if len(contact_pairs) < 100:
        return None
    
    # Sort by first residue position
    sorted_pairs = sorted(contact_pairs, key=lambda x: (x[0], x[1]))
    
    # Look at consecutive contact distances
    distances = [p[2] for p in sorted_pairs]
    
    # Count (a,b) followed by (b,a) patterns
    mobius_count = 0
    total_pairs = 0
    
    for i in range(len(distances) - 1):
        a, b = distances[i], distances[i+1]
        if a != b:
            total_pairs += 1
            # Check if (a,b) and (b,a) both exist in the sequence
            for j in range(len(distances) - 1):
                if distances[j] == b and distances[j+1] == a:
                    mobius_count += 1
                    break
    
    # Also check direct consecutive mirror
    direct_mirror = 0
    for i in range(len(distances) - 3):
        a, b = distances[i], distances[i+1]
        c, d = distances[i+2], distances[i+3]
        if a == d and b == c and a != b:
            direct_mirror += 1
    
    return {
        'mobius_patterns': mobius_count,
        'total_pairs': total_pairs,
        'direct_mirrors': direct_mirror,
    }


def test_two_thirds_ratio(distances):
    """
    Test for 2/3 ratio in consecutive distance pairs.
    From pac_confluence_xi: F₃/F₄ = 2/3 is universal.
    """
    if len(distances) < 100:
        return None
    
    # Count consecutive pairs with ratio near 2/3 or 3/2
    near_two_thirds = 0
    near_three_halves = 0
    total_pairs = 0
    
    sorted_dists = sorted(distances)
    for i in range(len(sorted_dists) - 1):
        a, b = sorted_dists[i], sorted_dists[i+1]
        if a > 0 and b > 0:
            ratio = a / b
            total_pairs += 1
            if 0.60 <= ratio <= 0.70:
                near_two_thirds += 1
            if 1.40 <= ratio <= 1.55:
                near_three_halves += 1
    
    # Also check exact Fibonacci pairs that form 2/3: (2,3), (8,12)≈(8,13), etc.
    fib_23_pairs = 0
    dist_counts = defaultdict(int)
    for d in distances:
        dist_counts[d] += 1
    
    # Check (2,3), (3,5), (5,8), (8,13), (13,21), (21,34), (34,55)
    fib_adjacent = [(2,3), (3,5), (5,8), (8,13), (13,21), (21,34), (34,55)]
    for a, b in fib_adjacent:
        if dist_counts[a] > 0 and dist_counts[b] > 0:
            fib_23_pairs += min(dist_counts[a], dist_counts[b])
    
    return {
        'near_two_thirds': near_two_thirds,
        'near_three_halves': near_three_halves,
        'fib_adjacent_pairs': fib_23_pairs,
        'total_pairs': total_pairs,
    }


def test_xi_modulation(distances):
    """
    Test if distances cluster around values related to Ξ = 1.057.
    Check if d/Ξ or d*Ξ produces integers or Fibonacci numbers.
    """
    if len(distances) < 100:
        return None
    
    xi_hits = 0
    for d in distances:
        # Check if d/Ξ is near integer
        ratio = d / XI
        if abs(ratio - round(ratio)) < 0.05:
            xi_hits += 1
        # Check if d*Ξ is near Fibonacci
        scaled = d * XI
        for f in FIBONACCI:
            if abs(scaled - f) < 0.5:
                xi_hits += 1
                break
    
    return {
        'xi_related_hits': xi_hits,
        'total': len(distances),
        'fraction': xi_hits / len(distances) if distances else 0,
    }


def test_phi_triplets(contact_pairs):
    """
    Test for triplets of contacts with distances in φ:1:1/φ ratio.
    This would indicate φ-scaling in 3D structure.
    """
    if len(contact_pairs) < 100:
        return None
    
    distances = sorted(set(p[2] for p in contact_pairs))
    
    phi_triplets = 0
    for d in distances:
        # Look for d, d*φ, d/φ (or close approximations)
        d_phi = d * PHI
        d_psi = d * PSI
        
        has_phi = any(abs(x - d_phi) < 1 for x in distances)
        has_psi = any(abs(x - d_psi) < 1 for x in distances)
        
        if has_phi and has_psi:
            phi_triplets += 1
    
    return {
        'phi_triplets': phi_triplets,
        'unique_distances': len(distances),
        'triplet_fraction': phi_triplets / len(distances) if distances else 0,
    }


def run_experiment(max_proteins=250):
    """Main experiment."""
    print("=" * 70)
    print("Experiment 19: Dawn Field Theory Constants in Biomolecules")
    print("=" * 70)
    print(f"\nTesting constants from milestone1 and arithmetic experiments:")
    print(f"  PHI = {PHI:.6f}")
    print(f"  XI = {XI:.6f}")
    print(f"  F7 = {F7}, F10 = {F10}")
    print(f"  Fibonacci: {FIBONACCI[:10]}...")
    print()
    
    pdb_ids = fetch_pdb_list(max_proteins)
    print(f"Fetched {len(pdb_ids)} PDB IDs")
    
    all_distances = []
    all_pairs = []
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
        
        distances, pairs = compute_contacts(coords)
        if len(distances) >= 20:
            all_distances.extend(distances)
            all_pairs.extend(pairs)
            processed += 1
        
        time.sleep(0.1)
    
    print(f"\nProcessed {processed} proteins, {len(all_distances)} contacts")
    
    # Run all tests
    print("\n" + "=" * 70)
    print("DAWN FIELD THEORY CONSTANT TESTS")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_proteins': processed,
        'n_contacts': len(all_distances),
        'tests': {},
    }
    
    # Test 1: Fibonacci
    print("\n[1] FIBONACCI ENRICHMENT")
    fib = test_fibonacci_enrichment(all_distances)
    if fib:
        sig = "✅" if fib['z'] > 3 else "❌"
        print(f"    Enrichment: {fib['enrichment']:.2f}x (z={fib['z']:.1f}) {sig}")
        results['tests']['fibonacci'] = fib
    
    # Test 2: F7=13 and F10=55
    print("\n[2] KEY FIBONACCI: F₇=13, F₁₀=55")
    f7f10 = test_f7_f10_enrichment(all_distances)
    if f7f10:
        print(f"    F₇=13: {f7f10['F7_13']['enrichment']:.2f}x enrichment ({f7f10['F7_13']['count']} contacts)")
        print(f"    F₁₀=55: {f7f10['F10_55']['enrichment']:.2f}x enrichment ({f7f10['F10_55']['count']} contacts)")
        results['tests']['f7_f10'] = f7f10
    
    # Test 3: φ-scaled
    print("\n[3] φ-SCALED DISTANCES")
    phi_scaled = test_phi_scaled_distances(all_distances)
    if phi_scaled:
        sig = "✅" if phi_scaled['z'] > 3 else "❌"
        print(f"    Enrichment: {phi_scaled['enrichment']:.2f}x (z={phi_scaled['z']:.1f}) {sig}")
        results['tests']['phi_scaled'] = phi_scaled
    
    # Test 4: Gap 6 hub
    print("\n[4] GAP 6 HUB (from oscillation_attractor_dynamics)")
    gap6 = test_gap_6_hub(all_distances)
    if gap6:
        sig = "✅" if gap6['enrichment'] > 1.5 else "❌"
        print(f"    Gap 6 fraction: {gap6['fraction']:.3f} ({gap6['enrichment']:.2f}x expected) {sig}")
        results['tests']['gap_6_hub'] = gap6
    
    # Test 5: Möbius pairs
    print("\n[5] MÖBIUS PAIR SYMMETRY")
    mobius = test_mobius_pair_symmetry(all_pairs)
    if mobius:
        print(f"    Möbius patterns: {mobius['mobius_patterns']}")
        print(f"    Direct mirrors: {mobius['direct_mirrors']}")
        results['tests']['mobius_pairs'] = mobius
    
    # Test 6: 2/3 ratio
    print("\n[6] 2/3 = F₃/F₄ RATIO")
    twothirds = test_two_thirds_ratio(all_distances)
    if twothirds:
        print(f"    Near 2/3 pairs: {twothirds['near_two_thirds']}")
        print(f"    Fib adjacent pairs: {twothirds['fib_adjacent_pairs']}")
        results['tests']['two_thirds'] = twothirds
    
    # Test 7: Ξ modulation
    print("\n[7] Ξ = 1 + π/55 MODULATION")
    xi = test_xi_modulation(all_distances)
    if xi:
        print(f"    Ξ-related hits: {xi['xi_related_hits']} ({xi['fraction']*100:.1f}%)")
        results['tests']['xi_modulation'] = xi
    
    # Test 8: φ triplets
    print("\n[8] φ:1:1/φ TRIPLETS")
    triplets = test_phi_triplets(all_pairs)
    if triplets:
        print(f"    φ-triplets found: {triplets['phi_triplets']} ({triplets['triplet_fraction']*100:.1f}%)")
        results['tests']['phi_triplets'] = triplets
    
    # Distance distribution
    print("\n" + "=" * 70)
    print("DISTANCE DISTRIBUTION (Top 25)")
    print("=" * 70)
    
    dist_counts = defaultdict(int)
    for d in all_distances:
        dist_counts[d] += 1
    
    top_dists = sorted(dist_counts.items(), key=lambda x: -x[1])[:25]
    
    for d, count in top_dists:
        markers = []
        if d in FIB_SET:
            markers.append("FIB")
        if d in PRIMES:
            markers.append("P")
        if d == 13:
            markers.append("F₇")
        if d == 55:
            markers.append("F₁₀")
        if d == 6:
            markers.append("HUB")
        if d in PHI_SCALED:
            markers.append("φ")
        marker_str = f" ← {', '.join(markers)}" if markers else ""
        print(f"  {d:3d}: {count:6d}{marker_str}")
    
    results['top_distances'] = [[d, count] for d, count in top_dists]
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT: DAWN FIELD THEORY CONSTANTS IN BIOLOGY")
    print("=" * 70)
    
    findings = []
    
    if fib and fib['z'] > 3:
        findings.append(f"✅ Fibonacci enrichment: {fib['enrichment']:.2f}x (z={fib['z']:.1f})")
    
    if f7f10:
        if f7f10['F7_13']['enrichment'] > 1.5:
            findings.append(f"✅ F₇=13 (gauge closure): {f7f10['F7_13']['enrichment']:.2f}x enrichment")
        if f7f10['F10_55']['enrichment'] > 1.5:
            findings.append(f"✅ F₁₀=55 (EM recursion): {f7f10['F10_55']['enrichment']:.2f}x enrichment")
    
    if phi_scaled and phi_scaled['z'] > 2:
        findings.append(f"✅ φ-scaled distances: {phi_scaled['enrichment']:.2f}x (z={phi_scaled['z']:.1f})")
    
    if gap6 and gap6['enrichment'] > 1.3:
        findings.append(f"✅ Gap 6 hub pattern: {gap6['enrichment']:.2f}x enrichment")
    
    if findings:
        print("\n🔬 SIGNIFICANT FINDINGS:")
        for f in findings:
            print(f"   {f}")
        print("\n   Dawn Field Theory constants appear in biomolecular structure!")
    else:
        print("\n   No significant patterns detected.")
    
    results['findings'] = findings
    results['dft_constants_present'] = len(findings) > 0
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_19_dft_constants_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {filepath}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', type=int, default=250)
    args = parser.parse_args()
    run_experiment(max_proteins=args.proteins)
