"""
Experiment 05: φ-Emergence on Odd Manifold
============================================

Tests whether removing p=2 (the even-manifold sieve wave) prevents
φ from emerging in prime k-tuple distributions.

Phase model prediction: p=2 creates the strongest single-prime
interference wave. Removing it should destroy the manifold where
φ-clustering occurs, since φ requires the full SEC collapse cascade.

Success criterion: φ-clustering disappears or degrades significantly
when p=2 is excluded from the sieve model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def compute_phi_clustering(primes_base, limit=100000):
    """
    Measure φ-clustering in gaps between numbers surviving a sieve
    using only the given prime base.

    Returns normalized gap ratio statistics — how close gap ratios
    approach φ.
    """
    # Sieve with only the given primes
    is_candidate = np.ones(limit, dtype=bool)
    is_candidate[0] = False
    if limit > 1:
        is_candidate[1] = False

    for p in primes_base:
        if p < limit:
            is_candidate[p::p] = False  # Remove multiples (but not p itself)
            is_candidate[p] = True       # Restore the prime itself

    survivors = np.where(is_candidate)[0]
    if len(survivors) < 100:
        return None

    # Compute consecutive gaps
    gaps = np.diff(survivors).astype(float)
    if len(gaps) < 50:
        return None

    # Compute gap ratios (consecutive gaps)
    ratios = gaps[1:] / gaps[:-1]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    if len(ratios) < 20:
        return None

    # φ-clustering: how many ratios are near φ or 1/φ?
    phi_window = 0.05
    near_phi = np.sum(np.abs(ratios - PHI) < phi_window * PHI) / len(ratios)
    near_inv_phi = np.sum(np.abs(ratios - 1/PHI) < phi_window / PHI) / len(ratios)

    # Also measure the histogram peak near φ
    hist, bin_edges = np.histogram(ratios, bins=100, range=(0, 3))
    phi_bin = int(PHI / 3.0 * 100)
    inv_phi_bin = int((1/PHI) / 3.0 * 100)

    peak_near_phi = max(hist[max(0,phi_bin-2):min(100,phi_bin+3)]) if phi_bin < 100 else 0
    peak_near_inv = max(hist[max(0,inv_phi_bin-2):min(100,inv_phi_bin+3)]) if inv_phi_bin < 100 else 0

    return {
        'n_survivors': len(survivors),
        'n_gaps': len(gaps),
        'n_ratios': len(ratios),
        'fraction_near_phi': float(near_phi),
        'fraction_near_inv_phi': float(near_inv_phi),
        'total_phi_fraction': float(near_phi + near_inv_phi),
        'mean_ratio': float(np.mean(ratios)),
        'std_ratio': float(np.std(ratios)),
        'median_ratio': float(np.median(ratios)),
        'peak_phi': int(peak_near_phi),
        'peak_inv_phi': int(peak_near_inv),
    }


def run():
    print("=" * 70)
    print("EXP 05: φ-Emergence on Odd Manifold")
    print("=" * 70)

    primes_list = sieve(200)
    limit = 200000

    # ================================================================
    # Test 1: Full sieve vs odd-only sieve
    # ================================================================
    print("\n--- Test 1: Full sieve vs odd-only sieve ---")

    # Full sieve: use first k primes {2, 3, 5, 7, 11, ...}
    # Odd sieve: skip p=2, use {3, 5, 7, 11, ...}
    test_k_values = [3, 5, 7, 9, 11]

    full_results = {}
    odd_results = {}

    for k in test_k_values:
        full_base = primes_list[:k]
        odd_base = [p for p in primes_list[:k+1] if p != 2][:k]

        print(f"\n  k={k}:")
        print(f"    Full base: {full_base}")
        print(f"    Odd base:  {odd_base}")

        full_data = compute_phi_clustering(full_base, limit)
        odd_data = compute_phi_clustering(odd_base, limit)

        full_results[k] = full_data
        odd_results[k] = odd_data

        if full_data and odd_data:
            print(f"    Full: φ-fraction = {full_data['total_phi_fraction']:.4f}, "
                  f"mean_ratio = {full_data['mean_ratio']:.4f}")
            print(f"    Odd:  φ-fraction = {odd_data['total_phi_fraction']:.4f}, "
                  f"mean_ratio = {odd_data['mean_ratio']:.4f}")
            ratio = odd_data['total_phi_fraction'] / full_data['total_phi_fraction'] \
                    if full_data['total_phi_fraction'] > 0 else float('inf')
            print(f"    Degradation: {(1-ratio)*100:.1f}%")

    # ================================================================
    # Test 2: Progressive prime removal
    # ================================================================
    print("\n--- Test 2: Remove individual small primes ---")
    print(f"  Base: first 10 primes = {primes_list[:10]}")

    base10 = primes_list[:10]
    removal_results = {}

    # Full base
    full = compute_phi_clustering(base10, limit)
    removal_results['full'] = full
    print(f"  Full base:     φ-fraction = {full['total_phi_fraction']:.4f}" if full else "  Full base: no data")

    # Remove each prime one at a time
    for remove_p in primes_list[:5]:  # Remove 2, 3, 5, 7, 11
        reduced = [p for p in base10 if p != remove_p]
        result = compute_phi_clustering(reduced, limit)
        removal_results[f'no_{remove_p}'] = result
        if result and full:
            degradation = (1 - result['total_phi_fraction'] / full['total_phi_fraction']) * 100
            print(f"  Without p={remove_p:2d}: φ-fraction = {result['total_phi_fraction']:.4f}  "
                  f"(degradation = {degradation:+.1f}%)")
        elif result:
            print(f"  Without p={remove_p:2d}: φ-fraction = {result['total_phi_fraction']:.4f}")

    # ================================================================
    # Test 3: Odd-only gap distribution shape
    # ================================================================
    print("\n--- Test 3: Gap distribution analysis ---")

    # Compare gap distributions for full vs odd sieve
    full_sieve = sieve(limit)
    odd_primes = [p for p in full_sieve if p > 2]

    full_gaps = np.diff(full_sieve[:5000]).astype(float)
    odd_gaps = np.diff(odd_primes[:5000]).astype(float)

    print(f"  Full prime gaps: mean={np.mean(full_gaps):.2f}, "
          f"std={np.std(full_gaps):.2f}, median={np.median(full_gaps):.1f}")
    print(f"  Odd prime gaps:  mean={np.mean(odd_gaps):.2f}, "
          f"std={np.std(odd_gaps):.2f}, median={np.median(odd_gaps):.1f}")

    # Gap=6 analysis (should be the hub in full model)
    gap6_full = np.sum(full_gaps == 6) / len(full_gaps)
    gap6_odd = np.sum(odd_gaps == 6) / len(odd_gaps)
    print(f"  Gap=6 fraction: full={gap6_full:.4f}, odd={gap6_odd:.4f}")

    # Gap=2 analysis (twin primes)
    gap2_full = np.sum(full_gaps == 2) / len(full_gaps)
    gap2_odd = np.sum(odd_gaps == 2) / len(odd_gaps)
    print(f"  Gap=2 fraction: full={gap2_full:.4f}, odd={gap2_odd:.4f}")

    # ================================================================
    # Results
    # ================================================================
    degradation_detected = False
    if full_results.get(9) and odd_results.get(9):
        if odd_results[9]['total_phi_fraction'] < full_results[9]['total_phi_fraction'] * 0.8:
            degradation_detected = True

    data = {
        'experiment': 'exp_05_phi_odd_manifold',
        'hypothesis': 'Removing p=2 destroys φ-clustering',
        'full_vs_odd': {
            str(k): {
                'full': full_results.get(k),
                'odd': odd_results.get(k),
            } for k in test_k_values
        },
        'removal_analysis': {k: v for k, v in removal_results.items() if v},
        'gap_analysis': {
            'full_gaps_mean': float(np.mean(full_gaps)),
            'odd_gaps_mean': float(np.mean(odd_gaps)),
            'gap6_full_fraction': float(gap6_full),
            'gap6_odd_fraction': float(gap6_odd),
        },
        'degradation_detected': degradation_detected,
        'success': degradation_detected,
        'success_criterion': 'φ-clustering degrades > 20% when p=2 removed',
    }

    print(f"\n{'='*70}")
    print(f"φ-DEGRADATION: {'YES' if degradation_detected else 'NO'}")
    print(f"SUCCESS: {'YES — p=2 removal degrades φ-clustering' if degradation_detected else 'INCONCLUSIVE'}")
    print(f"{'='*70}")

    save_results(data, 'exp_05_phi_odd_manifold')
    return data


if __name__ == '__main__':
    run()
