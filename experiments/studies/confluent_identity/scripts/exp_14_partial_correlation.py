"""
exp_14_partial_correlation.py -- Confluent Identity Phase 5

PURPOSE:
    Prove that the coupling~natural weight correlation survives when region
    size is controlled for. The natural~size confound (rho=0.77) means bigger
    regions have larger natural weights. This experiment uses partial Spearman
    correlation, size-stratified analysis, and permutation testing to isolate
    the true coupling signal.

METHODS:
    1. Partial Spearman: rho(coupling, natural | size) via rank residuals
    2. Size-stratified: coupling~natural within each size tercile
    3. Permutation test: 10,000 shuffles, empirical p-value
    4. Log-transform: partial Spearman with log(size) as confound

VERIFICATION:
    - partial rho(coupling, natural | size) > 0.2 AND p < 0.05
    - At least 2/3 size terciles show rho > 0
    - Permutation p < 0.01
    - partial rho(coupling, natural | log(size)) > 0.15

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy.stats import spearmanr, rankdata

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    RESULTS_DIR, K_MODES, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices, get_parent_children_data,
)
from exp_08_gradient_coupling import (
    compute_coupling_weights_uniform, compute_natural_weights,
)


def partial_spearman(x, y, z):
    """
    Partial Spearman correlation rho(x, y | z).

    Ranks x, y, z; regresses rank(x) and rank(y) on rank(z) via OLS;
    correlates the residuals.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    # Regress rx on rz
    rz_centered = rz - rz.mean()
    denom = np.dot(rz_centered, rz_centered)
    if denom < 1e-15:
        return float(spearmanr(x, y)[0]), float(spearmanr(x, y)[1])

    beta_x = np.dot(rz_centered, rx - rx.mean()) / denom
    beta_y = np.dot(rz_centered, ry - ry.mean()) / denom

    resid_x = rx - (rx.mean() + beta_x * rz_centered)
    resid_y = ry - (ry.mean() + beta_y * rz_centered)

    rho, p = spearmanr(resid_x, resid_y)
    return float(rho), float(p)


def stratified_correlation(coupling, natural, size, n_bins=3):
    """
    Bin regions into n_bins size terciles, compute Spearman rho within each.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(size, percentiles)
    results = []

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            mask = (size >= lo) & (size < hi)
        else:
            mask = (size >= lo) & (size <= hi)

        n_in_bin = mask.sum()
        if n_in_bin < 5:
            results.append({
                'bin': b, 'n': int(n_in_bin),
                'size_range': [float(lo), float(hi)],
                'rho': None, 'p': None,
                'note': 'insufficient data',
            })
            continue

        rho, p = spearmanr(coupling[mask], natural[mask])
        results.append({
            'bin': b, 'n': int(n_in_bin),
            'size_range': [float(lo), float(hi)],
            'rho': float(rho), 'p': float(p),
        })

    return results


def permutation_test(coupling, natural, size, n_perms=10000, seed=42):
    """
    Shuffle coupling 10000 times, compute partial_spearman each time,
    return empirical p-value and null distribution summary.
    """
    rng = np.random.RandomState(seed)
    observed_rho, _ = partial_spearman(coupling, natural, size)

    null_rhos = []
    for _ in range(n_perms):
        coupling_shuffled = rng.permutation(coupling)
        rho_null, _ = partial_spearman(coupling_shuffled, natural, size)
        null_rhos.append(rho_null)

    null_rhos = np.array(null_rhos)
    # Two-sided p-value
    p_value = float(np.mean(np.abs(null_rhos) >= abs(observed_rho)))

    return {
        'observed_rho': float(observed_rho),
        'p_value': p_value,
        'null_mean': float(null_rhos.mean()),
        'null_std': float(null_rhos.std()),
        'null_95_ci': [float(np.percentile(null_rhos, 2.5)),
                       float(np.percentile(null_rhos, 97.5))],
    }


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 5, Experiment 14")
    print("Partial Correlation: Size Deconfound")
    print("=" * 70)

    # Load data
    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # Collect (coupling_w, natural_w, size_frac) per child
    all_coupling = []
    all_natural = []
    all_size = []

    n_parents = 0
    for (level, pid), parent_indices, children_list, L_parent, state_parent in \
            get_parent_children_data(labels_by_level, hierarchy, adjacency, state_flat):

        n_parents += 1
        identity_parent = compute_spectral_identity(L_parent, state_parent)
        eigvecs_parent = identity_parent.get('eigenvectors')
        if eigvecs_parent is None:
            continue

        natural_weights, size_fractions = compute_natural_weights(
            state_flat, parent_indices, children_list, eigvecs_parent
        )
        coupling_weights = compute_coupling_weights_uniform(
            adjacency, state_flat, parent_indices, children_list
        )

        for child_id, _ in children_list:
            cid = child_id
            if cid in coupling_weights and cid in natural_weights and cid in size_fractions:
                all_coupling.append(coupling_weights[cid])
                all_natural.append(natural_weights[cid])
                all_size.append(size_fractions[cid])

    coupling = np.array(all_coupling)
    natural = np.array(all_natural)
    size = np.array(all_size)
    n = len(coupling)

    print(f"\nCollected {n} child measurements from {n_parents} parents")

    # Raw correlations (baseline)
    rho_cn_raw, p_cn_raw = spearmanr(coupling, natural)
    rho_ns_raw, p_ns_raw = spearmanr(natural, size)
    rho_cs_raw, p_cs_raw = spearmanr(coupling, size)
    print(f"\nRaw correlations:")
    print(f"  rho(coupling, natural) = {rho_cn_raw:.4f}  p={p_cn_raw:.2e}")
    print(f"  rho(natural, size)     = {rho_ns_raw:.4f}  p={p_ns_raw:.2e}")
    print(f"  rho(coupling, size)    = {rho_cs_raw:.4f}  p={p_cs_raw:.2e}")

    # 1. Partial Spearman correlation
    print(f"\n{'=' * 70}")
    print("1. Partial Spearman Correlation")
    print(f"{'=' * 70}")
    partial_rho, partial_p = partial_spearman(coupling, natural, size)
    print(f"  rho(coupling, natural | size) = {partial_rho:.4f}  p={partial_p:.2e}")

    # Also with log(size)
    log_size = np.log(size + 1e-15)
    partial_rho_log, partial_p_log = partial_spearman(coupling, natural, log_size)
    print(f"  rho(coupling, natural | log(size)) = {partial_rho_log:.4f}  p={partial_p_log:.2e}")

    # 2. Size-stratified analysis
    print(f"\n{'=' * 70}")
    print("2. Size-Stratified Analysis")
    print(f"{'=' * 70}")
    strat_results = stratified_correlation(coupling, natural, size, n_bins=3)
    for sr in strat_results:
        rho_str = f"{sr['rho']:.4f}" if sr['rho'] is not None else "N/A"
        print(f"  Tercile {sr['bin']}: n={sr['n']}, "
              f"size=[{sr['size_range'][0]:.4f}, {sr['size_range'][1]:.4f}], "
              f"rho={rho_str}")

    # 3. Permutation test
    print(f"\n{'=' * 70}")
    print("3. Permutation Test (10,000 shuffles)")
    print(f"{'=' * 70}")
    perm_results = permutation_test(coupling, natural, size)
    print(f"  Observed partial rho: {perm_results['observed_rho']:.4f}")
    print(f"  Null distribution: mean={perm_results['null_mean']:.4f}, "
          f"std={perm_results['null_std']:.4f}")
    print(f"  95% CI: [{perm_results['null_95_ci'][0]:.4f}, "
          f"{perm_results['null_95_ci'][1]:.4f}]")
    print(f"  Empirical p-value: {perm_results['p_value']:.4f}")

    # Verification
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = partial_rho > 0.2 and partial_p < 0.05
    print(f"\n  Test 1: partial rho > 0.2 AND p < 0.05?")
    print(f"    rho={partial_rho:.4f}, p={partial_p:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    # Test 2: At least 2/3 terciles show rho > 0
    positive_terciles = sum(1 for sr in strat_results
                           if sr['rho'] is not None and sr['rho'] > 0)
    valid_terciles = sum(1 for sr in strat_results if sr['rho'] is not None)
    test2 = positive_terciles >= 2
    print(f"\n  Test 2: At least 2/3 size terciles show rho > 0?")
    print(f"    {positive_terciles}/{valid_terciles} terciles positive")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = perm_results['p_value'] < 0.01
    print(f"\n  Test 3: Permutation p < 0.01?")
    print(f"    p={perm_results['p_value']:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    test4 = partial_rho_log > 0.15
    print(f"\n  Test 4: partial rho(coupling, natural | log(size)) > 0.15?")
    print(f"    rho={partial_rho_log:.4f}")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 deconfound tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_14_partial_correlation',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Size deconfound via partial Spearman correlation',
        'n_parents': n_parents,
        'n_measurements': n,
        'raw_correlations': {
            'rho_coupling_natural': float(rho_cn_raw),
            'p_coupling_natural': float(p_cn_raw),
            'rho_natural_size': float(rho_ns_raw),
            'rho_coupling_size': float(rho_cs_raw),
        },
        'partial_correlation': {
            'rho_partial_size': float(partial_rho),
            'p_partial_size': float(partial_p),
            'rho_partial_logsize': float(partial_rho_log),
            'p_partial_logsize': float(partial_p_log),
        },
        'stratified': strat_results,
        'permutation': perm_results,
        'verification': {
            'test1_partial_rho_significant': bool(test1),
            'test2_terciles_positive': bool(test2),
            'test3_permutation_significant': bool(test3),
            'test4_logsize_deconfound': bool(test4),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_14_partial_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
