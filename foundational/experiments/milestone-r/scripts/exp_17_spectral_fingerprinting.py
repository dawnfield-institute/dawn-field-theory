"""
exp_17 -- Spectral Fingerprinting: Reading Topology from Light

Milestone R, Block C (Novel Physics)

Thesis: The stress barrier FPT distribution across all vertices of a graph
IS the graph's "emission spectrum." If this spectrum uniquely identifies
the graph topology, then measuring light = measuring information structure.
This turns spectroscopy from a lookup table into a topology readout.

The practical claim: given an emission pattern (collection of FPTs), you
can read back WHAT SYSTEM produced it — not just "which element" but what
its CONNECTION STRUCTURE looks like.

Key test: same-size graphs from different ADE families (A_6/D_6/E_6,
A_7/D_7/E_7, A_8/D_8/E_8). They have the same number of "energy levels"
(vertices) but different topologies. Can the FPT spectrum tell them apart?

Tests:
  T1: Same-size graph discrimination — FPT spectra distinguish A/D/E
      at same vertex count (KS test p < 0.05 for all pairs)
  T2: Degree recovery — cluster FPTs to recover degree distribution
      (adjusted Rand index > 0.7 for majority)
  T3: Beyond-degree information — same-degree vertices at different
      positions have detectably different FPTs (CV > 0.05)
  T4: Noise robustness — discrimination survives across noise levels
      (majority of same-size pairs distinguishable at 3+ noise levels)
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
    ade_graphs,
    save_mr_results,
)


def compute_fpt_spectrum(adj, stress_threshold=0.008, noise_amplitude=0.020,
                         n_trials=150, max_steps=5000):
    """
    Compute the FPT spectrum of a graph: median FPT for each vertex.

    Returns dict: {vertex_index: {'median_fpt': float, 'degree': int, 'fpts': list}}
    """
    n = adj.shape[0]
    spectrum = {}

    for v in range(n):
        degree = int(np.sum(adj[v] > 0))
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            result = stress_barrier_walk(
                adj, v, initial,
                stress_threshold=stress_threshold,
                noise_amplitude=noise_amplitude,
                max_steps=max_steps,
                seed=trial * 100 + v * 50000,
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        median_fpt = float(np.median(fpts)) if fpts else float(max_steps)
        spectrum[v] = {
            'median_fpt': median_fpt,
            'degree': degree,
            'fpts': fpts,
            'n_converged': len(fpts),
        }

    return spectrum


def spectrum_to_sorted_fpts(spectrum):
    """Extract sorted median FPT vector from a spectrum."""
    return sorted([s['median_fpt'] for s in spectrum.values()])


def test_T1_same_size_discrimination():
    """T1: FPT spectra distinguish same-size graphs from different families."""
    print("\n  T1: Same-size graph discrimination")
    results = {'description': 'KS test p < 0.05 for all same-size pairs'}

    stress_threshold = 0.008
    noise = 0.020

    # Collect all ADE graphs, group by size
    all_graphs = {}
    for name, adj in ade_graphs(max_rank=8):
        if name in ('A_5', 'A_6', 'A_7', 'A_8',
                     'D_5', 'D_6', 'D_7', 'D_8',
                     'E_6', 'E_7', 'E_8'):
            all_graphs[name] = adj

    # Group by vertex count
    size_groups = {}
    for name, adj in all_graphs.items():
        n = adj.shape[0]
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(name)

    # Only test groups with 2+ graphs
    size_groups = {k: v for k, v in size_groups.items() if len(v) >= 2}

    # Compute spectra
    spectra = {}
    for name, adj in all_graphs.items():
        n = adj.shape[0]
        if n in size_groups:
            print(f"    Computing spectrum for {name} ({n} vertices)...")
            spectra[name] = compute_fpt_spectrum(
                adj, stress_threshold=stress_threshold,
                noise_amplitude=noise, n_trials=150)

    # For each same-size group, KS test all pairs
    pair_results = []
    for size, names in sorted(size_groups.items()):
        for g1, g2 in combinations(names, 2):
            # Pool all FPTs for each graph (not just medians)
            fpts1 = []
            for v_data in spectra[g1].values():
                fpts1.extend(v_data['fpts'])
            fpts2 = []
            for v_data in spectra[g2].values():
                fpts2.extend(v_data['fpts'])

            if len(fpts1) < 10 or len(fpts2) < 10:
                pair_results.append({
                    'pair': f'{g1} vs {g2}', 'size': size,
                    'ks_stat': 0, 'p_value': 1.0, 'distinguishable': False,
                })
                continue

            ks_stat, p_val = stats.ks_2samp(fpts1, fpts2)

            # Also compare sorted median vectors
            med1 = spectrum_to_sorted_fpts(spectra[g1])
            med2 = spectrum_to_sorted_fpts(spectra[g2])
            l2_dist = np.sqrt(np.sum((np.array(med1) - np.array(med2)) ** 2))

            distinguishable = p_val < 0.05
            pair_results.append({
                'pair': f'{g1} vs {g2}',
                'size': size,
                'ks_stat': round(float(ks_stat), 4),
                'p_value': float(p_val),
                'l2_distance': round(float(l2_dist), 2),
                'distinguishable': distinguishable,
            })
            print(f"    {g1} vs {g2}: KS={ks_stat:.3f}, p={p_val:.2e}, "
                  f"L2={l2_dist:.1f} -> {'distinct' if distinguishable else 'SAME'}")

    n_distinguishable = sum(1 for pr in pair_results if pr['distinguishable'])
    n_total = len(pair_results)
    passed = n_total > 0 and n_distinguishable == n_total

    results['pair_results'] = pair_results
    results['n_distinguishable'] = n_distinguishable
    results['n_total'] = n_total
    results['PASS'] = passed
    print(f"    Distinguishable: {n_distinguishable}/{n_total}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: all pairs)")
    return results, spectra


def test_T2_degree_recovery(spectra):
    """T2: Cluster FPTs to recover degree distribution."""
    print("\n  T2: Degree recovery from FPT spectrum")
    results = {'description': 'FPT clusters match degree classes (ARI > 0.7)'}

    per_graph = []

    for name, spectrum in sorted(spectra.items()):
        degrees = [spectrum[v]['degree'] for v in sorted(spectrum)]
        fpts = [spectrum[v]['median_fpt'] for v in sorted(spectrum)]
        n_distinct_degrees = len(set(degrees))

        if n_distinct_degrees < 2:
            per_graph.append({
                'graph': name, 'n_degrees': n_distinct_degrees,
                'ari': 1.0, 'pass': True, 'note': 'only 1 degree class',
            })
            continue

        # Simple clustering: sort FPTs and assign to k groups using
        # natural gaps (largest gaps between consecutive sorted values)
        indexed = sorted(enumerate(fpts), key=lambda x: x[1])
        sorted_fpts = [f for _, f in indexed]
        sorted_indices = [i for i, _ in indexed]

        # Find gaps
        gaps = []
        for i in range(1, len(sorted_fpts)):
            gaps.append((sorted_fpts[i] - sorted_fpts[i-1], i))
        gaps.sort(reverse=True)

        # Take top (k-1) gaps to split into k clusters
        k = n_distinct_degrees
        split_points = sorted([g[1] for g in gaps[:k-1]])

        # Assign cluster labels
        predicted_labels = np.zeros(len(fpts), dtype=int)
        cluster = 0
        for i, orig_idx in enumerate(sorted_indices):
            if i in split_points:
                cluster += 1
            predicted_labels[orig_idx] = cluster

        # Compute Adjusted Rand Index
        true_labels = np.array(degrees)
        # Map degrees to sequential labels
        unique_degrees = sorted(set(degrees))
        degree_map = {d: i for i, d in enumerate(unique_degrees)}
        true_labels = np.array([degree_map[d] for d in degrees])

        # ARI computation
        from scipy.special import comb as sp_comb

        def adjusted_rand_index(labels_true, labels_pred):
            """Compute ARI without sklearn."""
            n = len(labels_true)
            # Contingency table
            classes = sorted(set(labels_true))
            clusters = sorted(set(labels_pred))
            table = np.zeros((len(classes), len(clusters)), dtype=int)
            class_map = {c: i for i, c in enumerate(classes)}
            clust_map = {c: i for i, c in enumerate(clusters)}
            for i in range(n):
                table[class_map[labels_true[i]], clust_map[labels_pred[i]]] += 1

            sum_comb_c = sum(sp_comb(table[i, j], 2) for i in range(table.shape[0])
                            for j in range(table.shape[1]))
            sum_comb_r = sum(sp_comb(table[i, :].sum(), 2) for i in range(table.shape[0]))
            sum_comb_s = sum(sp_comb(table[:, j].sum(), 2) for j in range(table.shape[1]))
            comb_n = sp_comb(n, 2)

            if comb_n == 0:
                return 1.0
            expected = sum_comb_r * sum_comb_s / comb_n
            max_index = 0.5 * (sum_comb_r + sum_comb_s)
            denom = max_index - expected
            if abs(denom) < 1e-10:
                return 1.0
            return float((sum_comb_c - expected) / denom)

        ari = adjusted_rand_index(true_labels, predicted_labels)
        graph_pass = ari > 0.7

        per_graph.append({
            'graph': name,
            'n_degrees': n_distinct_degrees,
            'ari': round(float(ari), 4),
            'true_degrees': degrees,
            'predicted_clusters': predicted_labels.tolist(),
            'pass': graph_pass,
        })
        print(f"    {name}: {n_distinct_degrees} degree classes, ARI={ari:.3f} "
              f"-> {'pass' if graph_pass else 'fail'}")

    n_pass = sum(1 for pg in per_graph if pg['pass'])
    n_total = len(per_graph)
    passed = n_total > 0 and n_pass >= n_total * 0.6  # Majority

    results['per_graph'] = per_graph
    results['n_pass'] = n_pass
    results['n_total'] = n_total
    results['PASS'] = passed
    print(f"    Recovery: {n_pass}/{n_total}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: majority > 0.7 ARI)")
    return results


def test_T3_beyond_degree(spectra):
    """T3: Same-degree vertices at different positions have different FPTs."""
    print("\n  T3: Beyond-degree positional information")
    results = {'description': 'Within-degree CV > 0.05 for majority of graphs'}

    per_graph = []

    for name, spectrum in sorted(spectra.items()):
        # Group vertices by degree
        degree_groups = {}
        for v in sorted(spectrum):
            d = spectrum[v]['degree']
            if d not in degree_groups:
                degree_groups[d] = []
            degree_groups[d].append(spectrum[v]['median_fpt'])

        # Look at degree classes with 2+ vertices
        within_cvs = []
        for d, fpts in degree_groups.items():
            if len(fpts) >= 2:
                mean_fpt = np.mean(fpts)
                std_fpt = np.std(fpts)
                cv = std_fpt / mean_fpt if mean_fpt > 0 else 0
                within_cvs.append({
                    'degree': d,
                    'n_vertices': len(fpts),
                    'mean_fpt': round(float(mean_fpt), 1),
                    'cv': round(float(cv), 4),
                })

        if not within_cvs:
            per_graph.append({
                'graph': name, 'max_cv': 0.0, 'pass': False,
                'note': 'no multi-vertex degree classes',
            })
            continue

        max_cv = max(wc['cv'] for wc in within_cvs)
        mean_cv = np.mean([wc['cv'] for wc in within_cvs])
        graph_pass = max_cv > 0.05

        per_graph.append({
            'graph': name,
            'within_degree_cvs': within_cvs,
            'max_cv': round(float(max_cv), 4),
            'mean_cv': round(float(mean_cv), 4),
            'pass': graph_pass,
        })
        status = 'POSITION MATTERS' if graph_pass else 'degree only'
        print(f"    {name}: max within-degree CV={max_cv:.4f}, "
              f"mean CV={mean_cv:.4f} -> {status}")

    n_pass = sum(1 for pg in per_graph if pg['pass'])
    n_total = len(per_graph)
    passed = n_total > 0 and n_pass >= n_total * 0.6  # Majority

    results['per_graph'] = per_graph
    results['n_pass'] = n_pass
    results['n_total'] = n_total
    results['PASS'] = passed
    print(f"    Positional info detected: {n_pass}/{n_total}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: majority with CV > 0.05)")
    return results


def test_T4_noise_robustness():
    """T4: Discrimination survives across multiple noise levels."""
    print("\n  T4: Noise robustness of spectral fingerprints")
    results = {'description': 'Discrimination at 3+ noise levels for majority of pairs'}

    stress_threshold = 0.008
    noise_levels = [0.012, 0.020, 0.035]

    # Test on size-7 triple: A_7 vs D_7 vs E_7
    target_graphs = {}
    for name, adj in ade_graphs(max_rank=8):
        if name in ('A_7', 'D_7', 'E_7'):
            target_graphs[name] = adj

    pair_noise_results = {}
    pairs = list(combinations(sorted(target_graphs.keys()), 2))

    for noise in noise_levels:
        print(f"    noise={noise:.3f}:")
        spectra_at_noise = {}
        for name, adj in target_graphs.items():
            spectra_at_noise[name] = compute_fpt_spectrum(
                adj, stress_threshold=stress_threshold,
                noise_amplitude=noise, n_trials=120, max_steps=5000)

        for g1, g2 in pairs:
            fpts1 = []
            for v_data in spectra_at_noise[g1].values():
                fpts1.extend(v_data['fpts'])
            fpts2 = []
            for v_data in spectra_at_noise[g2].values():
                fpts2.extend(v_data['fpts'])

            if len(fpts1) < 10 or len(fpts2) < 10:
                distinguishable = False
                ks_stat, p_val = 0.0, 1.0
            else:
                ks_stat, p_val = stats.ks_2samp(fpts1, fpts2)
                distinguishable = p_val < 0.05

            key = f'{g1} vs {g2}'
            if key not in pair_noise_results:
                pair_noise_results[key] = []
            pair_noise_results[key].append({
                'noise': noise,
                'ks_stat': round(float(ks_stat), 4),
                'p_value': float(p_val),
                'distinguishable': distinguishable,
            })
            print(f"      {key}: KS={ks_stat:.3f}, p={p_val:.2e} "
                  f"-> {'distinct' if distinguishable else 'SAME'}")

    # Check: for each pair, how many noise levels give discrimination?
    per_pair = []
    for pair_name, noise_results in pair_noise_results.items():
        n_distinct = sum(1 for nr in noise_results if nr['distinguishable'])
        per_pair.append({
            'pair': pair_name,
            'n_noise_levels_distinct': n_distinct,
            'total_noise_levels': len(noise_levels),
            'robust': n_distinct >= len(noise_levels),  # All noise levels
        })
        print(f"    {pair_name}: distinct at {n_distinct}/{len(noise_levels)} noise levels")

    n_robust = sum(1 for pp in per_pair if pp['robust'])
    passed = len(per_pair) > 0 and n_robust >= len(per_pair) * 0.6  # Majority

    results['pair_noise_results'] = pair_noise_results
    results['per_pair'] = per_pair
    results['n_robust'] = n_robust
    results['n_pairs'] = len(per_pair)
    results['PASS'] = passed
    print(f"    Robust across noise: {n_robust}/{len(per_pair)} pairs")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: majority robust)")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_17: Spectral Fingerprinting — Reading Topology from Light")
    print("=" * 60)

    t1, spectra = test_T1_same_size_discrimination()
    t2 = test_T2_degree_recovery(spectra)
    t3 = test_T3_beyond_degree(spectra)
    t4 = test_T4_noise_robustness()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_17_spectral_fingerprinting',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'The stress barrier FPT distribution across all vertices of a graph '
                  'IS the graph\'s emission spectrum. This spectrum uniquely identifies '
                  'graph topology: same-size graphs from different ADE families produce '
                  'distinct FPT distributions. The spectrum encodes not just degree '
                  '(barrier height) but positional information (centrality, neighborhood). '
                  'Spectroscopy = topology readout.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_17_spectral_fingerprinting')
