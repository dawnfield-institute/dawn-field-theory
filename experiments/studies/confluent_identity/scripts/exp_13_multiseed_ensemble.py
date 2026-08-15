"""
exp_13_multiseed_ensemble.py -- Confluent Identity Phase 4

PURPOSE:
    Verify that key results are robust across random seeds, not artifacts
    of seed=42. Runs the Phase 1 pipeline (lattice -> partition -> identity ->
    confluence test) for 5 seeds and reports mean +/- std.

SEEDS: [42, 137, 256, 314, 999]

METRICS TRACKED:
    - rho(coupling, natural): correlation between coupling and natural weights
    - rho(natural, size): size confound strength
    - Gini coefficient: non-uniformity of natural weights
    - Conservation error: PAC conservation quality

VERIFICATION:
    - std(rho_coupling_natural) < 0.15 across seeds
    - All seeds produce rho > 0.2

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent))

from exp_01_lattice_fluid_baseline import PeriodicLatticeFluid

RESULTS_DIR = Path(__file__).parent.parent / 'results'
K_MODES = 10
SEEDS = [42, 137, 256, 314, 999]


def build_adjacency(C):
    """Build weighted adjacency matrix."""
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
    return sparse.csr_matrix((weights, (rows, cols)), shape=(N*N, N*N))


def watershed_partition(C, sigma=0.5, min_filter_size=3):
    """Simplified watershed (mirrors exp_02 logic)."""
    from scipy.ndimage import gaussian_filter, minimum_filter, label
    import heapq

    N = C.shape[0]
    C_smooth = gaussian_filter(C, sigma=sigma, mode='wrap')

    # Local minima
    local_min = minimum_filter(C_smooth, size=min_filter_size, mode='wrap')
    is_min = (C_smooth == local_min)
    labeled_mins, n_seeds = label(is_min)

    # Priority queue flood fill
    labels = np.zeros_like(C, dtype=int) - 1
    heap = []

    for seed_id in range(1, n_seeds + 1):
        seed_cells = np.argwhere(labeled_mins == seed_id)
        for cell in seed_cells:
            i, j = cell
            labels[i, j] = seed_id - 1
            heapq.heappush(heap, (C_smooth[i, j], i, j, seed_id - 1))

    while heap:
        val, i, j, lid = heapq.heappop(heap)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % N, (j + dj) % N
            if labels[ni, nj] == -1:
                labels[ni, nj] = lid
                heapq.heappush(heap, (C_smooth[ni, nj], ni, nj, lid))

    # Assign any remaining
    if (labels == -1).any():
        from scipy.ndimage import distance_transform_edt
        for i in range(N):
            for j in range(N):
                if labels[i, j] == -1:
                    labels[i, j] = 0

    return labels


def compute_identities_and_weights(C, labels):
    """Run identity and coupling weight computation for one level."""
    N = C.shape[0]
    state_flat = C.ravel()
    adjacency = build_adjacency(C)

    region_ids = sorted(np.unique(labels).tolist())

    # Compute parent identity (full lattice)
    degrees = np.array(adjacency.sum(axis=1)).ravel()
    L_full = sparse.diags(degrees) - adjacency
    k_actual = min(K_MODES + 1, N * N - 1)

    try:
        eigenvalues, eigvecs = eigsh(L_full.astype(float), k=k_actual,
                                      which='SM', tol=1e-8, maxiter=5000)
    except Exception:
        return None

    idx = np.argsort(eigenvalues)
    eigvecs = eigvecs[:, idx]

    harmonic = np.mean(state_flat)
    state_centered = state_flat - harmonic

    # Natural contribution weights per region
    natural_weights = {}
    size_fractions = {}
    total_cells = N * N

    for rid in region_ids:
        mask = (labels == rid).ravel()
        indices = np.where(mask)[0]
        n_cells = len(indices)

        if n_cells < 3:
            continue

        child_state = state_centered[indices]
        child_eigvec = eigvecs[indices, :]
        contrib = child_state @ child_eigvec
        natural_weights[rid] = float(np.linalg.norm(contrib))
        size_fractions[rid] = n_cells / total_cells

    # Normalize natural weights
    total_norm = sum(natural_weights.values())
    if total_norm > 1e-15:
        natural_weights = {r: w / total_norm for r, w in natural_weights.items()}

    # Coupling weights via perturbation
    epsilon = 0.01 * np.mean(state_flat)
    parent_coeffs = np.array([float(np.dot(state_centered, eigvecs[:, i]))
                               for i in range(eigvecs.shape[1])])

    coupling_weights = {}
    for rid in region_ids:
        mask = (labels == rid).ravel()
        indices = np.where(mask)[0]
        if len(indices) < 3:
            continue

        state_perturbed = state_flat.copy()
        state_perturbed[indices] += epsilon
        perturbed_centered = state_perturbed - np.mean(state_perturbed)
        perturbed_coeffs = np.array([
            float(np.dot(perturbed_centered, eigvecs[:, i]))
            for i in range(eigvecs.shape[1])
        ])
        delta = np.linalg.norm(perturbed_coeffs - parent_coeffs) / epsilon
        coupling_weights[rid] = delta

    # Normalize coupling
    total_coupling = sum(coupling_weights.values())
    if total_coupling > 1e-15:
        coupling_weights = {r: w / total_coupling for r, w in coupling_weights.items()}

    return natural_weights, coupling_weights, size_fractions


def gini_coefficient(weights_dict):
    """Gini coefficient from weight dictionary."""
    w = np.array(sorted(weights_dict.values()))
    n = len(w)
    if n < 2 or w.sum() < 1e-15:
        return 0.0
    return float((2.0 * np.sum((np.arange(1, n + 1) * w)) / (n * w.sum())) - (n + 1) / n)


def run_single_seed(seed):
    """Run Phase 1 pipeline for a single seed, return key metrics."""
    print(f"\n  --- Seed {seed} ---")

    # Phase 1: Lattice fluid
    fluid = PeriodicLatticeFluid(
        N=128, total_value=100.0, seed=seed,
        n_large_stones=12, n_small_stones=40, gravity=0.005
    )

    for step in range(2000):
        fluid.fluid_step(dt=0.005, viscosity=0.05, sec_threshold=0.1)

    conservation_error = fluid.conservation_error()
    C = fluid.C
    print(f"    Conservation: {conservation_error:.2e}")

    # Phase 2: Partition
    labels = watershed_partition(C)
    n_regions = len(np.unique(labels))
    print(f"    Regions: {n_regions}")

    # Phase 3: Identities and weights
    result = compute_identities_and_weights(C, labels)
    if result is None:
        print(f"    FAILED: eigendecomposition error")
        return None

    natural_weights, coupling_weights, size_fractions = result

    # Compute correlations (only for regions present in all three dicts)
    common_rids = set(natural_weights) & set(coupling_weights) & set(size_fractions)
    if len(common_rids) < 5:
        print(f"    FAILED: only {len(common_rids)} common regions")
        return None

    common_rids = sorted(common_rids)
    nat = np.array([natural_weights[r] for r in common_rids])
    coup = np.array([coupling_weights[r] for r in common_rids])
    size = np.array([size_fractions[r] for r in common_rids])

    rho_cn, p_cn = spearmanr(coup, nat)
    rho_ns, p_ns = spearmanr(nat, size)
    gini = gini_coefficient(natural_weights)

    print(f"    rho(coupling, natural) = {rho_cn:.4f}  p={p_cn:.2e}")
    print(f"    rho(natural, size)     = {rho_ns:.4f}")
    print(f"    Gini                   = {gini:.4f}")

    return {
        'seed': seed,
        'conservation_error': float(conservation_error),
        'n_regions': n_regions,
        'n_measurements': len(common_rids),
        'rho_coupling_natural': float(rho_cn),
        'p_coupling_natural': float(p_cn),
        'rho_natural_size': float(rho_ns),
        'gini': float(gini),
    }


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 4, Experiment 13")
    print("Multi-Seed Ensemble Robustness Test")
    print("=" * 70)
    print(f"\nSeeds: {SEEDS}")

    results = []
    for seed in SEEDS:
        r = run_single_seed(seed)
        if r is not None:
            results.append(r)

    if len(results) < 2:
        print("\n  FAILED: fewer than 2 seeds produced results")
        return

    # Aggregate
    print(f"\n{'=' * 70}")
    print(f"Aggregate ({len(results)} seeds)")
    print(f"{'=' * 70}")

    rho_cn = np.array([r['rho_coupling_natural'] for r in results])
    rho_ns = np.array([r['rho_natural_size'] for r in results])
    gini = np.array([r['gini'] for r in results])

    print(f"\n  rho(coupling, natural):")
    print(f"    mean={rho_cn.mean():.4f}, std={rho_cn.std():.4f}")
    print(f"    range=[{rho_cn.min():.4f}, {rho_cn.max():.4f}]")
    print(f"    per seed: {[f'{r:.3f}' for r in rho_cn]}")

    print(f"\n  rho(natural, size):")
    print(f"    mean={rho_ns.mean():.4f}, std={rho_ns.std():.4f}")

    print(f"\n  Gini:")
    print(f"    mean={gini.mean():.4f}, std={gini.std():.4f}")

    # Verification
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    test1 = rho_cn.std() < 0.15
    print(f"\n  Test 1: std(rho_coupling_natural) < 0.15?")
    print(f"    std = {rho_cn.std():.4f}")
    print(f"    {'[VERIFIED]' if test1 else '[FAILED]'}")

    test2 = all(r > 0.2 for r in rho_cn)
    print(f"\n  Test 2: All seeds rho > 0.2?")
    print(f"    min = {rho_cn.min():.4f}")
    print(f"    {'[VERIFIED]' if test2 else '[FAILED]'}")

    test3 = all(r > 0.1 for r in gini)
    print(f"\n  Test 3: All seeds Gini > 0.1 (non-uniform)?")
    print(f"    min = {gini.min():.4f}")
    print(f"    {'[VERIFIED]' if test3 else '[FAILED]'}")

    n_verified = sum([test1, test2, test3])
    print(f"\n  OVERALL: {n_verified}/3 robustness tests verified")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_13_multiseed_ensemble',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Multi-seed robustness verification',
        'seeds': SEEDS,
        'n_successful': len(results),
        'aggregate': {
            'rho_coupling_natural': {'mean': float(rho_cn.mean()),
                                     'std': float(rho_cn.std()),
                                     'min': float(rho_cn.min()),
                                     'max': float(rho_cn.max())},
            'rho_natural_size': {'mean': float(rho_ns.mean()),
                                 'std': float(rho_ns.std())},
            'gini': {'mean': float(gini.mean()), 'std': float(gini.std())},
        },
        'verification': {
            'test1_std_below_015': bool(test1),
            'test2_all_rho_above_02': bool(test2),
            'test3_all_gini_above_01': bool(test3),
            'n_verified': n_verified,
        },
        'per_seed': results,
    }

    output_file = RESULTS_DIR / f'exp_13_ensemble_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
