"""
exp_18_fiedler_deconfound.py -- Confluent Identity Phase 9

PURPOSE:
    Resolve the Fiedler-size confound (rho=-0.90) from exp_16. The 2/4 verified
    result (Fiedler predicts noise sensitivity) might be entirely mediated by
    region size. Also investigate the entropy reversal (rho=+0.43 when expected
    negative).

METHODS:
    1. Partial Spearman: rho(Fiedler, noise_sensitivity | size) and | log(size)
    2. Size-stratified: Fiedler vs noise_sensitivity within each size tercile
    3. Permutation test: 10,000 shuffles on partial correlation
    4. Entropy investigation: partial_rho(Fiedler, entropy | size) and
       normalized entropy = spectral_entropy / log(k_actual)

VERIFICATION:
    - partial_rho(Fiedler, noise_sensitivity | size) < -0.2 AND p < 0.05
    - >= 2/3 size terciles show negative rho(Fiedler, noise_sensitivity)
    - Permutation p < 0.01
    - partial_rho(Fiedler, entropy | size) NOT significantly positive

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
    RESULTS_DIR, load_baseline, build_lattice_adjacency,
    graph_laplacian_subgraph, compute_spectral_identity,
    get_region_indices,
)
from exp_14_partial_correlation import (
    partial_spearman, stratified_correlation, permutation_test,
)


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 9, Experiment 18")
    print("Fiedler-Size Deconfound + Entropy Reversal")
    print("=" * 70)

    P, A, C, stone_mask, labels_by_level, hierarchy = load_baseline()
    N = C.shape[0]
    state_flat = C.ravel()
    print(f"\nLoaded: {N}x{N} field, {len(labels_by_level)} levels")

    print("Building adjacency...")
    adjacency = build_lattice_adjacency(C)

    # Reproduce exp_16's region analysis for level-0 regions
    labels0 = labels_by_level[0]
    region_ids = sorted(np.unique(labels0).tolist())
    print(f"Level 0: {len(region_ids)} regions")

    fiedlers = []
    noise_sensitivities = []
    uniform_sensitivities = []
    entropies = []
    sizes = []
    k_actuals = []  # for normalized entropy

    for rid in region_ids:
        indices = get_region_indices(labels_by_level, 0, rid)
        n_cells = len(indices)
        if n_cells < 10:
            continue

        L, _ = graph_laplacian_subgraph(adjacency, indices)
        state_region = state_flat[indices]
        I_baseline = compute_spectral_identity(L, state_region)

        fiedler = I_baseline['fiedler_value']
        entropy = I_baseline['spectral_entropy']
        coeffs_baseline = np.array(I_baseline['state_coefficients'])
        coeff_norm = float(np.linalg.norm(coeffs_baseline))

        if coeff_norm < 1e-15:
            continue

        # Count nonzero eigenvalues for normalized entropy
        eigs = np.array(I_baseline['eigenvalues'])
        k_actual = int(np.sum(eigs > 1e-10))

        # Noise sensitivity (same as exp_16)
        rng = np.random.RandomState(42 + rid)
        noise = rng.randn(n_cells) * 0.1 * np.mean(state_region)
        state_noisy = state_region + noise
        I_noisy = compute_spectral_identity(L, state_noisy)
        coeffs_noisy = np.array(I_noisy['state_coefficients'])
        min_len = min(len(coeffs_baseline), len(coeffs_noisy))
        noise_shift = float(np.linalg.norm(
            coeffs_noisy[:min_len] - coeffs_baseline[:min_len]))
        noise_sens = noise_shift / (coeff_norm + 1e-15)

        # Uniform sensitivity at eps=0.1
        epsilon = 0.1 * np.mean(state_region)
        state_uniform = state_region + epsilon
        I_uniform = compute_spectral_identity(L, state_uniform)
        coeffs_uniform = np.array(I_uniform['state_coefficients'])
        min_len2 = min(len(coeffs_baseline), len(coeffs_uniform))
        uniform_shift = float(np.linalg.norm(
            coeffs_uniform[:min_len2] - coeffs_baseline[:min_len2]))
        uniform_sens = uniform_shift / (epsilon * coeff_norm + 1e-15)

        fiedlers.append(fiedler)
        noise_sensitivities.append(noise_sens)
        uniform_sensitivities.append(uniform_sens)
        entropies.append(entropy)
        sizes.append(float(n_cells))
        k_actuals.append(k_actual)

    fiedlers = np.array(fiedlers)
    noise_sens_arr = np.array(noise_sensitivities)
    uniform_sens_arr = np.array(uniform_sensitivities)
    entropies_arr = np.array(entropies)
    sizes_arr = np.array(sizes)
    k_actuals_arr = np.array(k_actuals)
    n_regions = len(fiedlers)

    print(f"\nAnalyzed {n_regions} regions (>= 10 cells)")

    # --- Raw confound confirmation ---
    print(f"\n{'=' * 70}")
    print("Raw Correlations (confirming exp_16)")
    print(f"{'=' * 70}")

    rho_fs, _ = spearmanr(fiedlers, sizes_arr)
    rho_fn, p_fn = spearmanr(fiedlers, noise_sens_arr)
    rho_fe, p_fe = spearmanr(fiedlers, entropies_arr)
    print(f"  rho(Fiedler, size) = {rho_fs:.4f}")
    print(f"  rho(Fiedler, noise_sensitivity) = {rho_fn:.4f}, p={p_fn:.2e}")
    print(f"  rho(Fiedler, entropy) = {rho_fe:.4f}, p={p_fe:.2e}")

    # --- 1. Partial Spearman: Fiedler vs noise_sensitivity | size ---
    print(f"\n{'=' * 70}")
    print("1. Partial Spearman Correlation")
    print(f"{'=' * 70}")

    partial_rho, partial_p = partial_spearman(fiedlers, noise_sens_arr, sizes_arr)
    print(f"  rho(Fiedler, noise_sens | size) = {partial_rho:.4f}, p={partial_p:.2e}")

    log_sizes = np.log(sizes_arr + 1e-15)
    partial_rho_log, partial_p_log = partial_spearman(
        fiedlers, noise_sens_arr, log_sizes)
    print(f"  rho(Fiedler, noise_sens | log(size)) = {partial_rho_log:.4f}, p={partial_p_log:.2e}")

    # Also for uniform sensitivity
    partial_rho_u, partial_p_u = partial_spearman(
        fiedlers, uniform_sens_arr, sizes_arr)
    print(f"  rho(Fiedler, uniform_sens | size) = {partial_rho_u:.4f}, p={partial_p_u:.2e}")

    # --- 2. Size-stratified analysis ---
    print(f"\n{'=' * 70}")
    print("2. Size-Stratified Analysis: Fiedler vs noise_sensitivity")
    print(f"{'=' * 70}")

    # Custom stratification (negative correlations, so use existing func but
    # interpret differently)
    strat_results = []
    percentiles = np.linspace(0, 100, 4)
    edges = np.percentile(sizes_arr, percentiles)

    for b in range(3):
        lo, hi = edges[b], edges[b + 1]
        if b < 2:
            mask = (sizes_arr >= lo) & (sizes_arr < hi)
        else:
            mask = (sizes_arr >= lo) & (sizes_arr <= hi)

        n_in_bin = mask.sum()
        if n_in_bin < 5:
            strat_results.append({
                'bin': b, 'n': int(n_in_bin),
                'size_range': [float(lo), float(hi)],
                'rho': None, 'note': 'insufficient data',
            })
            continue

        rho, p = spearmanr(fiedlers[mask], noise_sens_arr[mask])
        strat_results.append({
            'bin': b, 'n': int(n_in_bin),
            'size_range': [float(lo), float(hi)],
            'rho': float(rho), 'p': float(p),
        })
        print(f"  Tercile {b}: n={n_in_bin}, "
              f"size=[{lo:.0f}, {hi:.0f}], rho={rho:.4f}")

    # --- 3. Permutation test ---
    print(f"\n{'=' * 70}")
    print("3. Permutation Test (10,000 shuffles)")
    print(f"{'=' * 70}")

    # Custom permutation test for negative correlation
    rng = np.random.RandomState(42)
    observed_partial, _ = partial_spearman(fiedlers, noise_sens_arr, sizes_arr)
    null_rhos = []
    for _ in range(10000):
        fiedlers_shuffled = rng.permutation(fiedlers)
        rho_null, _ = partial_spearman(fiedlers_shuffled, noise_sens_arr, sizes_arr)
        null_rhos.append(rho_null)

    null_rhos = np.array(null_rhos)
    # One-sided p-value for negative correlation
    p_perm = float(np.mean(null_rhos <= observed_partial))

    print(f"  Observed partial rho: {observed_partial:.4f}")
    print(f"  Null: mean={null_rhos.mean():.4f}, std={null_rhos.std():.4f}")
    print(f"  95% CI: [{np.percentile(null_rhos, 2.5):.4f}, {np.percentile(null_rhos, 97.5):.4f}]")
    print(f"  Empirical p-value (one-sided): {p_perm:.4f}")

    # --- 4. Entropy reversal investigation ---
    print(f"\n{'=' * 70}")
    print("4. Entropy Reversal Investigation")
    print(f"{'=' * 70}")

    partial_rho_ent, partial_p_ent = partial_spearman(
        fiedlers, entropies_arr, sizes_arr)
    print(f"  partial_rho(Fiedler, entropy | size) = {partial_rho_ent:.4f}, p={partial_p_ent:.2e}")

    # Normalized entropy: entropy / log(k_actual)
    norm_entropy = np.array([
        e / np.log(max(k, 2)) for e, k in zip(entropies_arr, k_actuals_arr)
    ])
    rho_fn_norm, p_fn_norm = spearmanr(fiedlers, norm_entropy)
    print(f"  rho(Fiedler, normalized_entropy) = {rho_fn_norm:.4f}, p={p_fn_norm:.2e}")

    partial_rho_norm, partial_p_norm = partial_spearman(
        fiedlers, norm_entropy, sizes_arr)
    print(f"  partial_rho(Fiedler, norm_entropy | size) = {partial_rho_norm:.4f}, p={partial_p_norm:.2e}")

    # Explain: small regions have fewer eigenvalues -> higher entropy / log(k)
    print(f"\n  Size vs k_actual: rho={spearmanr(sizes_arr, k_actuals_arr)[0]:.4f}")
    print(f"  Size vs entropy:  rho={spearmanr(sizes_arr, entropies_arr)[0]:.4f}")

    # --- Verification ---
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = partial_rho < -0.2 and partial_p < 0.05
    print(f"\n  Test 1: partial_rho(Fiedler, noise_sens | size) < -0.2 AND p < 0.05?")
    print(f"    rho={partial_rho:.4f}, p={partial_p:.2e}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    negative_terciles = sum(1 for sr in strat_results
                            if sr['rho'] is not None and sr['rho'] < 0)
    valid_terciles = sum(1 for sr in strat_results if sr['rho'] is not None)
    test2 = negative_terciles >= 2
    print(f"\n  Test 2: >= 2/3 size terciles show negative rho(Fiedler, noise_sens)?")
    print(f"    {negative_terciles}/{valid_terciles} terciles negative")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = p_perm < 0.01
    print(f"\n  Test 3: Permutation p < 0.01?")
    print(f"    p={p_perm:.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    # Test 4: entropy reversal is a size artifact
    # Pass if partial rho is not significantly positive (p > 0.05 or rho <= 0)
    test4 = (partial_p_ent > 0.05) or (partial_rho_ent <= 0)
    print(f"\n  Test 4: partial_rho(Fiedler, entropy | size) NOT significantly positive?")
    print(f"    rho={partial_rho_ent:.4f}, p={partial_p_ent:.2e}")
    if partial_rho_ent <= 0:
        print(f"    Sign flipped (negative after deconfounding)")
    elif partial_p_ent > 0.05:
        print(f"    Not significant after deconfounding")
    print(f"    {'[VERIFIED]' if test4 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3, test4])
    print(f"\n  OVERALL: {n_verified}/4 Fiedler deconfound tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_18_fiedler_deconfound',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Fiedler-size deconfound + entropy reversal investigation',
        'n_regions': n_regions,
        'raw_correlations': {
            'rho_fiedler_size': float(rho_fs),
            'rho_fiedler_noise_sensitivity': float(rho_fn),
            'rho_fiedler_entropy': float(rho_fe),
        },
        'partial_correlations': {
            'fiedler_noise_sens_given_size': {
                'rho': float(partial_rho), 'p': float(partial_p)},
            'fiedler_noise_sens_given_logsize': {
                'rho': float(partial_rho_log), 'p': float(partial_p_log)},
            'fiedler_uniform_sens_given_size': {
                'rho': float(partial_rho_u), 'p': float(partial_p_u)},
            'fiedler_entropy_given_size': {
                'rho': float(partial_rho_ent), 'p': float(partial_p_ent)},
            'fiedler_norm_entropy_given_size': {
                'rho': float(partial_rho_norm), 'p': float(partial_p_norm)},
        },
        'stratified': strat_results,
        'permutation': {
            'observed_rho': float(observed_partial),
            'p_value': float(p_perm),
            'null_mean': float(null_rhos.mean()),
            'null_std': float(null_rhos.std()),
        },
        'entropy_investigation': {
            'rho_fiedler_norm_entropy': float(rho_fn_norm),
            'p_fiedler_norm_entropy': float(p_fn_norm),
        },
        'verification': {
            'test1_partial_rho_significant': bool(test1),
            'test2_terciles_negative': bool(test2),
            'test3_permutation_significant': bool(test3),
            'test4_entropy_artifact': bool(test4),
            'n_verified': n_verified,
        },
    }

    output_file = RESULTS_DIR / f'exp_18_fiedler_deconfound_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
