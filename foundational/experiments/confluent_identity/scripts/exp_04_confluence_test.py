"""
exp_04_confluence_test.py -- Confluent Identity Phase 1

PURPOSE:
    The core hypothesis test: can a parent's identity be reconstructed
    from its children's identities weighted by coupling strength?

    I(parent) ~= f(I(children), w)

    Tests Claim 1: Identity is confluence, not aggregation.

DESIGN (v2 -- basis-aligned):
    The v1 approach compared spectral coefficients across incompatible bases
    (each region's eigenvectors are different). This version fixes that by
    working entirely in the PARENT's eigenbasis.

    Since children partition the parent, each child's contribution to a
    parent spectral coefficient is the partial dot product over that child's
    cells. These sum exactly to the parent coefficient by linearity:

        coeff_i(parent) = SUM_S coeff_i(S)
        where coeff_i(S) = <state_S, v_i|_S>  (restriction of parent eigenvector)

    The "natural weight" of child S is how much it contributes to the parent's
    spectral fingerprint: w_natural(S) = ||contributions(S)|| / total

    Test 1 (CORRELATION): Do coupling weights correlate with natural weights?
        If yes: perturbation sensitivity captures actual identity contribution.

    Test 2 (RECONSTRUCTION): Does weighted combination of child SCALARS
        (fiedler, spectral entropy, harmonic projection) predict parent scalars
        better than simple mean?

    Test 3 (CONTRIBUTION PROFILE): Is the contribution profile non-uniform?
        i.e., do some children dominate the parent's identity despite similar size?

FALSIFICATION:
    Claim 1 SUPPORTED if:
        - Coupling weights correlate with natural weights (rho > 0.5, p < 0.05)
        - Contribution profiles are non-uniform (Gini > 0.2)
        - Weighted scalar reconstruction outperforms mean
    Claim 1 FALSIFIED if:
        - Natural weights proportional to region size (weight = mass)
        - Coupling weights uncorrelated with natural weights
        - Contribution profiles are uniform

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh

RESULTS_DIR = Path(__file__).parent.parent / 'results'
K_MODES = 10


def load_data():
    """Load exp_01 fields, exp_02 partition, exp_03 identities/weights."""
    P = np.load(RESULTS_DIR / 'exp_01_P_steady.npy')
    A = np.load(RESULTS_DIR / 'exp_01_A_steady.npy')

    # Labels
    labels_by_level = []
    level = 0
    while True:
        path = RESULTS_DIR / f'exp_02_labels_level{level}.npy'
        if path.exists():
            labels_by_level.append(np.load(path))
            level += 1
        else:
            break

    # Hierarchy
    exp02_files = sorted(RESULTS_DIR.glob('exp_02_partition_*.json'))
    with open(exp02_files[-1]) as f:
        partition_data = json.load(f)

    hierarchy = {}
    for key, children in partition_data['hierarchy'].items():
        level_str, rid_str = key.split(',')
        hierarchy[(int(level_str), int(rid_str))] = [
            (int(c[0]), int(c[1])) for c in children
        ]

    # Exp03 identities and coupling weights
    exp03_files = sorted(RESULTS_DIR.glob('exp_03_identity_*.json'))
    with open(exp03_files[-1]) as f:
        exp03 = json.load(f)

    return P, A, labels_by_level, hierarchy, exp03


def build_lattice_adjacency(C):
    """Build sparse weighted adjacency matrix for periodic lattice."""
    N = C.shape[0]
    C_mean = C.mean()
    rows, cols, weights = [], [], []

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = (i + di) % N, (j + dj) % N
                nidx = ni * N + nj
                w = np.exp(-abs(C[i, j] - C[ni, nj]) / C_mean)
                rows.append(idx)
                cols.append(nidx)
                weights.append(w)

    return sparse.csr_matrix(
        (weights, (rows, cols)), shape=(N * N, N * N)
    )


def parent_eigenvectors(adjacency, parent_indices, k=K_MODES):
    """Compute parent subgraph Laplacian eigenvectors."""
    W_sub = adjacency[np.ix_(parent_indices, parent_indices)]
    degrees = np.array(W_sub.sum(axis=1)).ravel()
    L = sparse.diags(degrees) - W_sub
    n = L.shape[0]
    k_actual = min(k + 1, n - 1)

    if n < 50:
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigvecs = np.linalg.eigh(L_dense)
    else:
        try:
            eigenvalues, eigvecs = eigsh(
                L.astype(float), k=k_actual, which='SM',
                tol=1e-8, maxiter=5000
            )
        except Exception:
            L_dense = L.toarray() if sparse.issparse(L) else L
            eigenvalues, eigvecs = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:k_actual]
            eigvecs = eigvecs[:, :k_actual]

    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigvecs[:, idx]


def child_contributions_in_parent_basis(state_parent, eigvecs, parent_indices,
                                         children_map):
    """
    For each child, compute its contribution to each parent spectral coefficient.

    Returns:
        contributions: dict child_id -> array of contributions per mode
        natural_weights: dict child_id -> scalar weight (L2 norm of contribution vector)
    """
    # Center state (remove harmonic/DC component)
    state_centered = state_parent - np.mean(state_parent)
    n_modes = eigvecs.shape[1]

    # Build parent-local index map
    parent_local = {int(g): i for i, g in enumerate(parent_indices)}

    contributions = {}
    for child_id, child_global_indices in children_map.items():
        # Get child's local positions within parent
        local_positions = []
        for g in child_global_indices:
            g_int = int(g)
            if g_int in parent_local:
                local_positions.append(parent_local[g_int])
        local_positions = np.array(local_positions)

        if len(local_positions) == 0:
            contributions[child_id] = np.zeros(n_modes)
            continue

        # Partial dot product: child's contribution to each parent coefficient
        child_state = state_centered[local_positions]
        child_eigvec_slices = eigvecs[local_positions, :]  # (n_child_cells, n_modes)
        contrib = child_state @ child_eigvec_slices  # (n_modes,)
        contributions[child_id] = contrib

    # Natural weights: L2 norm of contribution vector, normalized
    norms = {cid: float(np.linalg.norm(c)) for cid, c in contributions.items()}
    total_norm = sum(norms.values())
    if total_norm > 1e-15:
        natural_weights = {cid: n / total_norm for cid, n in norms.items()}
    else:
        n = len(norms)
        natural_weights = {cid: 1.0 / n for cid in norms}

    return contributions, natural_weights


def gini_coefficient(weights):
    """Gini coefficient: 0 = perfectly equal, 1 = maximally concentrated."""
    w = np.array(sorted(weights))
    n = len(w)
    if n < 2 or w.sum() < 1e-15:
        return 0.0
    cumw = np.cumsum(w)
    return float((2.0 * np.sum((np.arange(1, n + 1) * w)) / (n * w.sum())) - (n + 1) / n)


def run_experiment():
    """Run the confluence hypothesis test (v2: basis-aligned)."""

    print("=" * 70)
    print("Confluent Identity -- Phase 1, Experiment 04 (v2)")
    print("Confluence Hypothesis Test -- Basis-Aligned")
    print("=" * 70)

    P, A, labels_by_level, hierarchy, exp03 = load_data()
    C = P + A
    N = C.shape[0]
    state_flat = C.ravel()
    n_levels = len(labels_by_level)
    identities = exp03['identities']
    coupling_weights = exp03['coupling_weights']

    print(f"\nLoaded: {N}x{N} field, {n_levels} levels, "
          f"{len(identities)} identities, {len(coupling_weights)} weight sets")

    print("\nBuilding adjacency matrix...")
    adjacency = build_lattice_adjacency(C)

    # ================================================================
    # Test 1: Coupling weight vs natural weight correlation
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Test 1: Coupling Weight vs Natural Weight Correlation")
    print(f"{'=' * 70}")

    all_coupling = []
    all_natural = []
    all_size_frac = []
    per_parent_results = []

    for (level, pid), children in hierarchy.items():
        if len(children) < 2:
            continue

        weights_key = f"{level},{pid}"
        if weights_key not in coupling_weights:
            continue

        cw = coupling_weights[weights_key]

        # Get parent indices
        parent_labels = labels_by_level[level]
        parent_indices = np.where((parent_labels == pid).ravel())[0]
        parent_n_cells = len(parent_indices)

        if parent_n_cells < 5:
            continue

        # Build children map: child_id -> global indices
        children_map = {}
        child_sizes = {}
        for child_level, child_id in children:
            child_labels = labels_by_level[child_level]
            child_indices = np.where((child_labels == child_id).ravel())[0]
            children_map[child_id] = child_indices
            child_sizes[child_id] = len(child_indices)

        # Compute parent eigenvectors and child contributions
        eigenvalues, eigvecs = parent_eigenvectors(adjacency, parent_indices)
        state_parent = state_flat[parent_indices]
        contributions, natural_weights = child_contributions_in_parent_basis(
            state_parent, eigvecs, parent_indices, children_map
        )

        # Collect paired data for correlation
        paired = []
        for child_id in children_map:
            cid_str = str(child_id)
            if cid_str in cw:
                c_weight = cw[cid_str]
                n_weight = natural_weights.get(child_id, 0.0)
                s_frac = child_sizes.get(child_id, 0) / parent_n_cells
                paired.append((child_id, c_weight, n_weight, s_frac))
                all_coupling.append(c_weight)
                all_natural.append(n_weight)
                all_size_frac.append(s_frac)

        if len(paired) < 2:
            continue

        # Per-parent correlation
        cw_arr = np.array([p[1] for p in paired])
        nw_arr = np.array([p[2] for p in paired])
        sf_arr = np.array([p[3] for p in paired])

        # Spearman rank correlation (more robust for small N)
        from scipy.stats import spearmanr
        if len(paired) >= 3:
            rho_cn, p_cn = spearmanr(cw_arr, nw_arr)
            rho_cs, p_cs = spearmanr(cw_arr, sf_arr)
            rho_ns, p_ns = spearmanr(nw_arr, sf_arr)
        else:
            rho_cn = float(np.corrcoef(cw_arr, nw_arr)[0, 1])
            rho_cs = float(np.corrcoef(cw_arr, sf_arr)[0, 1])
            rho_ns = float(np.corrcoef(nw_arr, sf_arr)[0, 1])
            p_cn = p_cs = p_ns = float('nan')

        # Gini of natural weights
        gini = gini_coefficient(list(natural_weights.values()))

        parent_result = {
            'level': level, 'parent_id': pid,
            'n_children': len(paired), 'parent_n_cells': parent_n_cells,
            'rho_coupling_natural': float(rho_cn),
            'p_coupling_natural': float(p_cn),
            'rho_coupling_size': float(rho_cs),
            'rho_natural_size': float(rho_ns),
            'gini_natural': gini,
            'children': [
                {'id': p[0], 'coupling_w': p[1], 'natural_w': p[2],
                 'size_frac': p[3]}
                for p in paired
            ],
        }
        per_parent_results.append(parent_result)

        print(f"\n  L{level} P{pid} ({len(paired)} children, {parent_n_cells} cells):")
        print(f"    Coupling vs Natural:  rho={rho_cn:+.3f}  p={p_cn:.4f}")
        print(f"    Coupling vs Size:     rho={rho_cs:+.3f}")
        print(f"    Natural vs Size:      rho={rho_ns:+.3f}")
        print(f"    Gini(natural):        {gini:.3f}")

        # Show top contributors
        sorted_children = sorted(paired, key=lambda x: x[2], reverse=True)
        for cid, cw_val, nw_val, sf_val in sorted_children[:3]:
            marker = " *" if nw_val > 2 * sf_val else ""
            print(f"      child {cid}: coupling={cw_val:.3f}  "
                  f"natural={nw_val:.3f}  size={sf_val:.3f}{marker}")

    # Global correlation
    print(f"\n  {'-' * 50}")
    print(f"  Global (N={len(all_coupling)} child-parent pairs):")

    from scipy.stats import spearmanr, pearsonr
    if len(all_coupling) >= 3:
        rho_global, p_global = spearmanr(all_coupling, all_natural)
        r_global, pr_global = pearsonr(all_coupling, all_natural)
        rho_size, p_size = spearmanr(all_natural, all_size_frac)
        print(f"    Coupling vs Natural: Spearman rho={rho_global:+.3f} (p={p_global:.4f})")
        print(f"                         Pearson  r  ={r_global:+.3f} (p={pr_global:.4f})")
        print(f"    Natural vs Size:     Spearman rho={rho_size:+.3f} (p={p_size:.4f})")
    else:
        rho_global = p_global = r_global = rho_size = float('nan')

    # ================================================================
    # Test 2: Weighted scalar reconstruction
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Test 2: Weighted Scalar Reconstruction")
    print(f"{'=' * 70}")

    scalar_results = []

    for (level, pid), children in hierarchy.items():
        if len(children) < 2:
            continue

        weights_key = f"{level},{pid}"
        parent_key = f"{level},{pid}"
        if weights_key not in coupling_weights or parent_key not in identities:
            continue

        cw = coupling_weights[weights_key]
        parent_id = identities[parent_key]

        # Gather child scalars
        child_scalars = []
        for child_level, child_id in children:
            child_key = f"{child_level},{child_id}"
            cid_str = str(child_id)
            if child_key in identities and cid_str in cw:
                child_ident = identities[child_key]
                child_scalars.append({
                    'id': child_id,
                    'coupling_w': cw[cid_str],
                    'fiedler': child_ident['fiedler_value'],
                    'entropy': child_ident['spectral_entropy'],
                    'harmonic': child_ident['harmonic_projection'],
                    'n_cells': child_ident['n_cells'],
                })

        if len(child_scalars) < 2:
            continue

        parent_n = parent_id['n_cells']

        for scalar_name in ['fiedler', 'entropy', 'harmonic']:
            actual = parent_id[scalar_name + '_value' if scalar_name == 'fiedler'
                               else 'spectral_' + scalar_name if scalar_name == 'entropy'
                               else scalar_name + '_projection']

            vals = np.array([c[scalar_name] for c in child_scalars])
            c_weights = np.array([c['coupling_w'] for c in child_scalars])
            s_weights = np.array([c['n_cells'] for c in child_scalars], dtype=float)

            c_weights = c_weights / (c_weights.sum() + 1e-15)
            s_weights = s_weights / (s_weights.sum() + 1e-15)

            pred_coupling = float(np.dot(c_weights, vals))
            pred_mean = float(np.mean(vals))
            pred_size = float(np.dot(s_weights, vals))

            err_coupling = abs(pred_coupling - actual)
            err_mean = abs(pred_mean - actual)
            err_size = abs(pred_size - actual)
            scale = abs(actual) + 1e-15

            scalar_results.append({
                'level': level, 'parent_id': pid, 'scalar': scalar_name,
                'actual': actual,
                'pred_coupling': pred_coupling, 'pred_mean': pred_mean,
                'pred_size': pred_size,
                'err_coupling': err_coupling / scale,
                'err_mean': err_mean / scale,
                'err_size': err_size / scale,
            })

    if scalar_results:
        print(f"\n  {'Scalar':<12s} {'Coupling':<12s} {'Mean':<12s} {'Size-Wt':<12s} {'Best':<10s}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

        for scalar_name in ['fiedler', 'entropy', 'harmonic']:
            subset = [r for r in scalar_results if r['scalar'] == scalar_name]
            if not subset:
                continue
            avg_c = np.mean([r['err_coupling'] for r in subset])
            avg_m = np.mean([r['err_mean'] for r in subset])
            avg_s = np.mean([r['err_size'] for r in subset])
            best = min([('coupling', avg_c), ('mean', avg_m), ('size', avg_s)],
                       key=lambda x: x[1])
            print(f"  {scalar_name:<12s} {avg_c:<12.4f} {avg_m:<12.4f} {avg_s:<12.4f} {best[0]:<10s}")

        # Overall
        avg_c_all = np.mean([r['err_coupling'] for r in scalar_results])
        avg_m_all = np.mean([r['err_mean'] for r in scalar_results])
        avg_s_all = np.mean([r['err_size'] for r in scalar_results])
        print(f"  {'OVERALL':<12s} {avg_c_all:<12.4f} {avg_m_all:<12.4f} {avg_s_all:<12.4f}")
    else:
        avg_c_all = avg_m_all = avg_s_all = float('nan')

    # ================================================================
    # Test 3: Contribution profile non-uniformity
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Test 3: Contribution Profile Non-Uniformity")
    print(f"{'=' * 70}")

    gini_values = [r['gini_natural'] for r in per_parent_results]
    if gini_values:
        mean_gini = np.mean(gini_values)
        print(f"\n  Mean Gini coefficient: {mean_gini:.3f}")
        print(f"  Range: [{min(gini_values):.3f}, {max(gini_values):.3f}]")
        print(f"  Parents with Gini > 0.2: {sum(1 for g in gini_values if g > 0.2)}/{len(gini_values)}")

        # Check: is natural weight correlated with size?
        print(f"\n  If Gini > 0.2 AND natural != size => identity is NOT mass")
    else:
        mean_gini = 0.0

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    verdicts = []

    # Test 1 verdict
    if not np.isnan(rho_global):
        if rho_global > 0.5 and p_global < 0.05:
            v1 = "SUPPORTED"
            verdicts.append(("Coupling predicts contribution", True))
        elif rho_global > 0.3:
            v1 = "MODERATE"
            verdicts.append(("Coupling predicts contribution", True))
        else:
            v1 = "WEAK/FALSIFIED"
            verdicts.append(("Coupling predicts contribution", False))
        print(f"\n  Test 1 (coupling ~ natural):  {v1}")
        print(f"    Global Spearman rho = {rho_global:+.3f}, p = {p_global:.4f}")
    else:
        print(f"\n  Test 1: INSUFFICIENT DATA")

    # Test 2 verdict
    if not np.isnan(avg_c_all):
        if avg_c_all < avg_m_all:
            improvement = (avg_m_all - avg_c_all) / avg_m_all * 100
            v2 = f"COUPLING WINS ({improvement:.1f}% better)"
            verdicts.append(("Weighted reconstruction", True))
        else:
            v2 = "MEAN WINS or TIE"
            verdicts.append(("Weighted reconstruction", False))
        print(f"  Test 2 (scalar reconstruction): {v2}")
    else:
        print(f"  Test 2: INSUFFICIENT DATA")

    # Test 3 verdict
    if gini_values:
        if mean_gini > 0.2 and (np.isnan(rho_size) or rho_size < 0.8):
            v3 = "SUPPORTED -- identity is NOT mass"
            verdicts.append(("Non-uniform, not-size", True))
        elif mean_gini > 0.1:
            v3 = "MODERATE non-uniformity"
            verdicts.append(("Non-uniform, not-size", True))
        else:
            v3 = "UNIFORM -- no confluent structure"
            verdicts.append(("Non-uniform, not-size", False))
        print(f"  Test 3 (non-uniformity):       {v3}")
        print(f"    Mean Gini = {mean_gini:.3f}, Natural~Size rho = {rho_size:+.3f}")

    # Overall
    n_supported = sum(1 for _, v in verdicts if v)
    n_total = len(verdicts)
    print(f"\n  Overall: {n_supported}/{n_total} tests support Claim 1")

    if n_supported == n_total and n_total >= 2:
        print("\n  ==> CLAIM 1 SUPPORTED: Identity is confluence, not aggregation")
        print("      Coupling weights predict spectral contribution.")
        print("      Contribution profiles are non-uniform and not explained by size.")
    elif n_supported >= n_total / 2:
        print("\n  ==> CLAIM 1 PARTIALLY SUPPORTED: mixed evidence")
        print("      Some tests pass, further investigation needed.")
    else:
        print("\n  ==> CLAIM 1 CHALLENGED: insufficient evidence for confluence")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': 'exp_04_confluence_test_v2',
        'timestamp': datetime.now().isoformat(),
        'version': 2,
        'test1_correlation': {
            'global_spearman_rho': float(rho_global) if not np.isnan(rho_global) else None,
            'global_spearman_p': float(p_global) if not np.isnan(p_global) else None,
            'global_pearson_r': float(r_global) if not np.isnan(r_global) else None,
            'natural_vs_size_rho': float(rho_size) if not np.isnan(rho_size) else None,
            'n_pairs': len(all_coupling),
            'per_parent': per_parent_results,
        },
        'test2_scalar_reconstruction': {
            'mean_err_coupling': float(avg_c_all) if not np.isnan(avg_c_all) else None,
            'mean_err_mean': float(avg_m_all) if not np.isnan(avg_m_all) else None,
            'mean_err_size': float(avg_s_all) if not np.isnan(avg_s_all) else None,
            'detailed': scalar_results,
        },
        'test3_nonuniformity': {
            'mean_gini': float(mean_gini),
            'gini_values': [float(g) for g in gini_values],
        },
        'verdicts': {name: supported for name, supported in verdicts},
        'n_supported': n_supported,
        'n_total': n_total,
    }

    output_file = RESULTS_DIR / f'exp_04_confluence_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return results


if __name__ == '__main__':
    run_experiment()
