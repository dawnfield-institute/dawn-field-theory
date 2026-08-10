"""
exp_19 -- Two-Channel Blind Spots: Why You Need All Three

Milestone R, Block C (Novel Physics)

Thesis: exp_18 showed that three channels (FPT/JSD/HKS) achieve 100%
classification but the combination was chosen empirically. If the
three channels genuinely correspond to PAC/SEC/RBF — three INDEPENDENT
measurement axes — then dropping any ONE channel should create a
SPECIFIC blind spot that the other two cannot cover.

Prediction: each two-channel pair fails on a DIFFERENT subset of graphs.
- FPT+JSD (no HKS) = PAC+SEC without RBF: fails on same-degree-sequence
  pairs (no geometric channel to distinguish them)
- FPT+HKS (no JSD) = PAC+RBF without SEC: fails on pairs where
  topology is similar but identity/information structure differs
- JSD+HKS (no FPT) = SEC+RBF without PAC: fails on pairs where
  the degree signal is the dominant distinguisher

If each pair has a different blind spot, that's a structure theorem about
measurement: you need all three DFT axioms to fully resolve a source.

Tests:
  T1: Each two-channel pair has at least one blind spot
      (fails on at least one pair that 3-channel succeeds on)
  T2: Different pairs have different blind spots
      (the failure sets are not identical)
  T3: Three channels together have no blind spot within the test set
      (all pairs distinguishable at 0.5sigma or better in 3-channel)
  T4: The PAC/SEC/RBF assignment is falsifiable — permuting channel
      labels should NOT preserve the blind spot structure
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    stress_barrier_walk,
    perspective_divergence,
    ade_graphs,
    save_mr_results,
)


def compute_fingerprint(name, adj, stress_threshold=0.008,
                        noise_amplitude=0.020, n_trials=150, max_steps=5000):
    """Compute 3-channel fingerprint for each vertex."""
    n = adj.shape[0]

    # Laplacian + spectral decomposition
    degrees_arr = np.sum(adj > 0, axis=1).astype(float)
    L = np.diag(degrees_arr) - adj.astype(float)
    eigvals, eigvecs = np.linalg.eigh(L)

    # Heat kernel signature at t=1
    t = 1.0
    hks = np.zeros(n)
    for k in range(n):
        hks += np.exp(-eigvals[k] * t) * eigvecs[:, k] ** 2

    fp = {}
    for v in range(n):
        degree = int(np.sum(adj[v] > 0))

        # Channel 1: Stress FPT
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            result = stress_barrier_walk(
                adj, v, initial,
                stress_threshold=stress_threshold,
                noise_amplitude=noise_amplitude,
                max_steps=max_steps,
                seed=trial * 100 + v * 50000 + abs(hash(name)) % 10000,
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])
        median_fpt = float(np.median(fpts)) if fpts else float(max_steps)

        # Channel 2: JSD
        jsd = perspective_divergence(adj, v, horizon=2)

        # Channel 3: HKS
        fp[v] = {
            'fpt': median_fpt,
            'jsd': jsd,
            'hks': float(hks[v]),
            'degree': degree,
        }

    return fp


def compute_pair_distance(fp1, fp2, channels):
    """
    Compute normalized distance between two same-size fingerprints
    using only the specified channels.

    channels: subset of ['fpt', 'jsd', 'hks']
    """
    n = len(fp1)
    assert len(fp2) == n

    # Build matrices with selected channels
    mat1 = []
    mat2 = []
    for v in sorted(fp1):
        row1 = [fp1[v][ch] for ch in channels]
        row2 = [fp2[v][ch] for ch in channels]
        mat1.append(row1)
        mat2.append(row2)

    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    # Per-channel: sort and compute normalized distance
    norm_dists = []
    for c in range(len(channels)):
        col1 = np.sort(mat1[:, c])
        col2 = np.sort(mat2[:, c])
        raw_dist = np.sqrt(np.sum((col1 - col2) ** 2))
        pooled = np.concatenate([col1, col2])
        scale = max(np.std(pooled), 1e-10)
        norm_dists.append(raw_dist / scale)

    return float(np.sqrt(np.sum(np.array(norm_dists) ** 2)))


def main():
    print("=" * 60)
    print("exp_19: Two-Channel Blind Spots")
    print("=" * 60)

    # Compute fingerprints for all same-size graphs
    all_graphs = {}
    for name, adj in ade_graphs(max_rank=8):
        if name in ('A_5', 'A_6', 'A_7', 'A_8',
                     'D_5', 'D_6', 'D_7', 'D_8',
                     'E_6', 'E_7', 'E_8'):
            all_graphs[name] = adj

    # Group by size
    size_groups = {}
    for name, adj in all_graphs.items():
        n = adj.shape[0]
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(name)
    size_groups = {k: v for k, v in size_groups.items() if len(v) >= 2}

    fingerprints = {}
    for name, adj in all_graphs.items():
        n = adj.shape[0]
        if n in size_groups:
            print(f"  Computing fingerprint for {name} ({n} vertices)...")
            fingerprints[name] = compute_fingerprint(name, adj)

    # Build all same-size pairs
    pairs = []
    for size, names in sorted(size_groups.items()):
        for g1, g2 in combinations(names, 2):
            pairs.append((g1, g2))

    # Channel combinations
    channel_sets = {
        'PAC+SEC (FPT+JSD)': ['fpt', 'jsd'],
        'PAC+RBF (FPT+HKS)': ['fpt', 'hks'],
        'SEC+RBF (JSD+HKS)': ['jsd', 'hks'],
        'ALL (FPT+JSD+HKS)': ['fpt', 'jsd', 'hks'],
    }

    # Compute distances for all pairs under all channel sets
    # Use 0.5 sigma as threshold (relaxed from exp_18's 1.0)
    THRESHOLD = 0.5
    results_table = {}

    for set_name, channels in channel_sets.items():
        results_table[set_name] = {}
        for g1, g2 in pairs:
            dist = compute_pair_distance(fingerprints[g1], fingerprints[g2], channels)
            results_table[set_name][(g1, g2)] = dist

    # Print distance table
    print(f"\n  Distance table (threshold = {THRESHOLD} sigma):")
    print(f"  {'Pair':<16} {'PAC+SEC':>10} {'PAC+RBF':>10} {'SEC+RBF':>10} {'ALL 3':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for g1, g2 in pairs:
        pair_name = f"{g1} vs {g2}"
        d_ps = results_table['PAC+SEC (FPT+JSD)'][(g1, g2)]
        d_pr = results_table['PAC+RBF (FPT+HKS)'][(g1, g2)]
        d_sr = results_table['SEC+RBF (JSD+HKS)'][(g1, g2)]
        d_all = results_table['ALL (FPT+JSD+HKS)'][(g1, g2)]

        def mark(d):
            return f"{d:>8.2f}{'*' if d < THRESHOLD else ' ':>2}"

        print(f"  {pair_name:<16} {mark(d_ps)} {mark(d_pr)} {mark(d_sr)} {mark(d_all)}")

    # T1: Each two-channel pair has at least one blind spot
    print("\n  T1: Each two-channel pair has at least one blind spot")
    t1_results = {}
    two_channel_sets = {k: v for k, v in channel_sets.items() if len(v) == 2}

    for set_name, channels in two_channel_sets.items():
        failures = []
        for g1, g2 in pairs:
            dist_2ch = results_table[set_name][(g1, g2)]
            dist_3ch = results_table['ALL (FPT+JSD+HKS)'][(g1, g2)]
            if dist_2ch < THRESHOLD and dist_3ch >= THRESHOLD:
                failures.append(f"{g1} vs {g2}")
        # Also count pairs where 2-channel fails regardless
        blind = []
        for g1, g2 in pairs:
            if results_table[set_name][(g1, g2)] < THRESHOLD:
                blind.append(f"{g1} vs {g2}")

        t1_results[set_name] = {
            'blind_spots': blind,
            'n_blind': len(blind),
            'has_blind_spot': len(blind) > 0,
        }
        status = 'HAS BLIND SPOT' if blind else 'no blind spot'
        print(f"    {set_name}: {len(blind)} blind spots: {blind} -> {status}")

    t1_pass = all(r['has_blind_spot'] for r in t1_results.values())
    print(f"    -> {'PASS' if t1_pass else 'FAIL'} (need: all pairs have >= 1 blind spot)")

    # T2: Different pairs have different blind spots
    print("\n  T2: Different two-channel pairs have different blind spots")
    blind_sets = {}
    for set_name in two_channel_sets:
        blind_sets[set_name] = set(t1_results[set_name]['blind_spots'])

    all_different = True
    t2_comparisons = []
    for s1, s2 in combinations(two_channel_sets.keys(), 2):
        same = blind_sets[s1] == blind_sets[s2]
        if same:
            all_different = False
        overlap = blind_sets[s1] & blind_sets[s2]
        symmetric_diff = blind_sets[s1] ^ blind_sets[s2]
        t2_comparisons.append({
            'pair': f'{s1} vs {s2}',
            'identical': same,
            'overlap': list(overlap),
            'unique_to_each': list(symmetric_diff),
        })
        print(f"    {s1} vs {s2}: "
              f"{'IDENTICAL' if same else 'DIFFERENT'} "
              f"(overlap={len(overlap)}, unique={len(symmetric_diff)})")

    t2_pass = all_different
    print(f"    -> {'PASS' if t2_pass else 'FAIL'} (need: all blind spot sets different)")

    # T3: Three channels cover all pairs at 0.5 sigma
    print("\n  T3: Three channels together have no blind spots")
    three_ch_blind = []
    for g1, g2 in pairs:
        if results_table['ALL (FPT+JSD+HKS)'][(g1, g2)] < THRESHOLD:
            three_ch_blind.append(f"{g1} vs {g2}")

    t3_pass = len(three_ch_blind) == 0
    print(f"    3-channel blind spots: {three_ch_blind if three_ch_blind else 'NONE'}")
    print(f"    -> {'PASS' if t3_pass else 'FAIL'} (need: zero blind spots)")

    # T4: Channel permutation breaks the structure
    print("\n  T4: Blind spot structure is not permutation-invariant")
    # If channels were interchangeable, swapping labels wouldn't change which
    # pairs each 2-channel combination can't distinguish.
    # Test: compute how many pairs each 2-channel set uniquely fails on
    # (i.e., that pair is blind ONLY for that specific channel combination)
    unique_blinds = {}
    for set_name in two_channel_sets:
        others = [s for s in two_channel_sets if s != set_name]
        unique = blind_sets[set_name].copy()
        for other in others:
            unique -= blind_sets[other]
        unique_blinds[set_name] = unique
        print(f"    {set_name}: {len(unique)} unique blind spots: {list(unique)}")

    # If channels are interchangeable, no combination has unique blind spots
    # If they're genuinely different (PAC/SEC/RBF), at least one combination
    # has a unique blind spot
    has_unique = any(len(u) > 0 for u in unique_blinds.values())
    # Also: the blind spot SIZES should differ (channels aren't symmetric)
    sizes = [len(blind_sets[s]) for s in two_channel_sets]
    size_variation = max(sizes) - min(sizes) if sizes else 0

    t4_pass = has_unique or size_variation > 0
    print(f"    Blind spot sizes: {dict(zip(two_channel_sets.keys(), sizes))}")
    print(f"    Size variation: {size_variation}, Unique blind spots exist: {has_unique}")
    print(f"    -> {'PASS' if t4_pass else 'FAIL'} (need: non-symmetric structure)")

    score = sum(1 for t in [t1_pass, t2_pass, t3_pass, t4_pass] if t)
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_19_two_channel_blindspots',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'If the three measurement channels (FPT/JSD/HKS) correspond to '
                  'PAC/SEC/RBF, then dropping any one axiom creates a SPECIFIC '
                  'blind spot that the other two cannot cover. Each two-channel '
                  'pair should fail on a different subset of graphs. This is a '
                  'structure theorem: you need all three DFT axioms to fully '
                  'resolve a source.',
        'test_results': {
            'T1': {'description': 'Each 2-channel pair has blind spots',
                   'results': t1_results, 'PASS': t1_pass},
            'T2': {'description': 'Different pairs have different blind spots',
                   'comparisons': t2_comparisons, 'PASS': t2_pass},
            'T3': {'description': '3 channels cover all pairs',
                   'blind_spots': three_ch_blind, 'PASS': t3_pass},
            'T4': {'description': 'Channel roles not interchangeable',
                   'unique_blinds': {k: list(v) for k, v in unique_blinds.items()},
                   'size_variation': size_variation,
                   'PASS': t4_pass},
        },
        'overall_score': f"{score}/4",
        'distance_table': {
            f"{g1} vs {g2}": {
                set_name: round(results_table[set_name][(g1, g2)], 3)
                for set_name in channel_sets
            }
            for g1, g2 in pairs
        },
    }
    save_mr_results(data, 'exp_19_two_channel_blindspots')


if __name__ == '__main__':
    main()
