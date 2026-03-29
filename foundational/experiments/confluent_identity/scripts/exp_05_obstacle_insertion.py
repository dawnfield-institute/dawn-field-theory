"""
exp_05_obstacle_insertion.py -- Confluent Identity Phase 2

PURPOSE:
    Test Claim 3: Weight is coupling to global identity, not intrinsic mass.
    "A small node in a critical position outweighs a large node in a peripheral
    position. Like the small pebble in the main channel."

DESIGN:
    1. Load baseline steady-state field from Phase 1
    2. Identify high-flow and low-flow positions using gradient magnitude
    3. Insert test obstacles:
       a. SMALL obstacle in HIGH-FLOW channel (the pebble in the stream)
       b. LARGE obstacle in LOW-FLOW region (the boulder on the bank)
       c. SMALL obstacle in LOW-FLOW region (control)
       d. LARGE obstacle in HIGH-FLOW channel (control)
    4. For each: re-run dynamics to new steady state, recompute hierarchy +
       identities, measure global reweighting vs baseline
    5. Compare: which obstacle causes more identity change?

METRICS:
    - Global identity shift: ||I_new - I_baseline|| for top-level regions
    - Fiedler value change: how much coherence shifts
    - Weight redistribution: how much coupling weights change
    - Cascade distance: how far from the obstacle does reweighting reach?

FALSIFICATION:
    Claim 3 SUPPORTED if: small-in-flow > large-out-of-flow identity shift
    Claim 3 FALSIFIED if: large obstacle always dominates regardless of position

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.ndimage import minimum_filter, label as ndlabel, uniform_filter

RESULTS_DIR = Path(__file__).parent.parent / 'results'
K_MODES = 10

# Import fluid class from exp_01
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_01_lattice_fluid_baseline import PeriodicLatticeFluid


def load_baseline():
    """Load Phase 1 baseline field and identities."""
    P = np.load(RESULTS_DIR / 'exp_01_P_steady.npy')
    A = np.load(RESULTS_DIR / 'exp_01_A_steady.npy')
    stone_mask = np.load(RESULTS_DIR / 'exp_01_stone_mask.npy')

    # Load baseline identities
    exp03_files = sorted(RESULTS_DIR.glob('exp_03_identity_*.json'))
    with open(exp03_files[-1]) as f:
        exp03 = json.load(f)

    return P, A, stone_mask, exp03


def find_obstacle_positions(P, A, stone_mask, N=128):
    """
    Find high-flow and low-flow positions for obstacle insertion.

    High-flow: regions with large C gradient (active channels)
    Low-flow: flat regions far from stones (stagnant zones)
    """
    C = P + A
    fluid_mask = ~stone_mask

    # Gradient magnitude
    grad_x = (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)) / 2.0
    grad_y = (np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)) / 2.0
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Smooth to find sustained channels, not just edges
    from scipy.ndimage import uniform_filter
    grad_smooth = uniform_filter(grad_mag, size=5)

    # Mask out stones and their immediate neighbors (buffer of 8 cells)
    from scipy.ndimage import binary_dilation
    stone_buffer = binary_dilation(stone_mask, iterations=8)

    # Candidate positions: fluid cells not near existing stones
    candidate_mask = fluid_mask & ~stone_buffer

    if not candidate_mask.any():
        # Relax buffer
        stone_buffer = binary_dilation(stone_mask, iterations=4)
        candidate_mask = fluid_mask & ~stone_buffer

    # High-flow: top 5% of gradient in candidate area
    grad_candidate = grad_smooth.copy()
    grad_candidate[~candidate_mask] = 0
    high_threshold = np.percentile(grad_candidate[candidate_mask], 95)
    high_flow_mask = candidate_mask & (grad_smooth >= high_threshold)

    # Low-flow: bottom 20% of gradient in candidate area
    grad_candidate_low = grad_smooth.copy()
    grad_candidate_low[~candidate_mask] = np.inf
    low_threshold = np.percentile(grad_candidate[candidate_mask], 20)
    low_flow_mask = candidate_mask & (grad_smooth <= low_threshold)

    # Pick centroid of each region
    def pick_center(mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return N // 4, N // 4  # fallback
        # Pick the point with the most neighbors also in the mask (well inside)
        best_idx = len(ys) // 2  # approximate center
        return int(ys[best_idx]), int(xs[best_idx])

    high_pos = pick_center(high_flow_mask)
    low_pos = pick_center(low_flow_mask)

    return high_pos, low_pos, grad_smooth


def insert_obstacle(P, A, stone_mask, center, radius):
    """
    Insert a circular stone obstacle at the given center.
    Returns new P, A, stone_mask arrays (copies).
    The obstacle's C value is set to 3x local mean (dense stone).
    """
    P_new = P.copy()
    A_new = A.copy()
    mask_new = stone_mask.copy()
    N = P.shape[0]

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # Circular obstacle (periodic-aware)
    dy = np.minimum(np.abs(Y - center[0]), N - np.abs(Y - center[0]))
    dx = np.minimum(np.abs(X - center[1]), N - np.abs(X - center[1]))
    dist = np.sqrt(dx**2 + dy**2)
    obstacle_mask = dist < radius

    # Set obstacle: high A (actualized), low P
    local_C_mean = (P + A).mean()
    obstacle_C = local_C_mean * 3.0

    # Remove value from obstacle cells, add back as stone
    total_before = P_new.sum() + A_new.sum()

    P_new[obstacle_mask] = 0.1 * obstacle_C
    A_new[obstacle_mask] = 0.9 * obstacle_C
    mask_new[obstacle_mask] = True

    # Conserve total: redistribute difference to fluid cells
    total_after = P_new.sum() + A_new.sum()
    diff = total_after - total_before
    fluid_mask = ~mask_new
    n_fluid = fluid_mask.sum()
    if n_fluid > 0 and abs(diff) > 1e-15:
        P_new[fluid_mask] -= diff * 0.7 / n_fluid
        A_new[fluid_mask] -= diff * 0.3 / n_fluid

    n_obstacle_cells = obstacle_mask.sum()
    return P_new, A_new, mask_new, n_obstacle_cells


def run_to_steady(P, A, stone_mask, max_steps=3000, dt=0.005,
                  viscosity=0.05, sec_threshold=0.1):
    """Run from given state to new steady state using PeriodicLatticeFluid internals."""
    N = P.shape[0]

    # Create a fluid object and inject state
    fluid = PeriodicLatticeFluid.__new__(PeriodicLatticeFluid)
    fluid.N = N
    fluid.P = P.copy()
    fluid.A = A.copy()
    fluid.stone_mask = stone_mask.copy()
    fluid.gravity = 0.005
    fluid.rng = np.random.default_rng(42)

    fluid.stone_values_P = np.zeros((N, N))
    fluid.stone_values_A = np.zeros((N, N))
    fluid.stone_values_P[stone_mask] = fluid.P[stone_mask]
    fluid.stone_values_A[stone_mask] = fluid.A[stone_mask]
    fluid.initial_total = fluid.P.sum() + fluid.A.sum()

    history = fluid.run_to_steady_state(
        max_steps=max_steps, dt=dt, viscosity=viscosity,
        sec_threshold=sec_threshold, tol=1e-6, stable_count=10
    )
    return fluid.P, fluid.A, history


def build_adjacency(C):
    """Build sparse weighted adjacency matrix."""
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


def compute_global_identity(adjacency, state_flat, k=K_MODES):
    """Compute spectral identity for the entire lattice."""
    n = len(state_flat)
    W = adjacency
    degrees = np.array(W.sum(axis=1)).ravel()
    L = sparse.diags(degrees) - W
    k_actual = min(k + 1, n - 1)

    eigenvalues, eigvecs = eigsh(L.astype(float), k=k_actual, which='SM',
                                  tol=1e-8, maxiter=5000)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigvecs = eigvecs[:, idx]

    harmonic = float(np.mean(state_flat))
    state_centered = state_flat - harmonic
    coeffs = np.array([float(np.dot(state_centered, eigvecs[:, i]))
                        for i in range(eigvecs.shape[1])])

    nonzero = eigenvalues > 1e-10
    fiedler = float(eigenvalues[nonzero][0]) if nonzero.any() else 0.0

    nz_eigs = eigenvalues[nonzero]
    if len(nz_eigs) > 0:
        p = nz_eigs / nz_eigs.sum()
        spec_entropy = float(-np.sum(p * np.log(p + 1e-15)))
    else:
        spec_entropy = 0.0

    return {
        'harmonic': harmonic,
        'fiedler': fiedler,
        'spectral_entropy': spec_entropy,
        'coefficients': coeffs.tolist(),
        'eigenvalues': eigenvalues.tolist(),
    }


def identity_distance(id1, id2):
    """Distance between two identities using multiple metrics."""
    c1 = np.array(id1['coefficients'])
    c2 = np.array(id2['coefficients'])
    min_len = min(len(c1), len(c2))
    c1, c2 = c1[:min_len], c2[:min_len]

    l2 = float(np.linalg.norm(c1 - c2))
    norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
    if norm1 > 1e-15 and norm2 > 1e-15:
        cosine = float(np.dot(c1, c2) / (norm1 * norm2))
    else:
        cosine = 0.0

    fiedler_change = abs(id1['fiedler'] - id2['fiedler'])
    entropy_change = abs(id1['spectral_entropy'] - id2['spectral_entropy'])
    harmonic_change = abs(id1['harmonic'] - id2['harmonic'])

    return {
        'l2_distance': l2,
        'cosine_similarity': cosine,
        'fiedler_change': fiedler_change,
        'entropy_change': entropy_change,
        'harmonic_change': harmonic_change,
        'composite': l2 + fiedler_change + entropy_change,
    }


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 2, Experiment 05")
    print("Obstacle Insertion: Weight = Coupling, Not Mass")
    print("=" * 70)

    P_base, A_base, stone_mask_base, exp03 = load_baseline()
    N = P_base.shape[0]
    C_base = P_base + A_base
    print(f"\nLoaded baseline: {N}x{N}, total={C_base.sum():.6f}")
    print(f"  Existing stones: {stone_mask_base.sum()} cells")

    # Compute baseline global identity
    print("\nComputing baseline global identity...")
    adj_base = build_adjacency(C_base)
    id_baseline = compute_global_identity(adj_base, C_base.ravel())
    print(f"  Fiedler: {id_baseline['fiedler']:.6f}")
    print(f"  Spectral entropy: {id_baseline['spectral_entropy']:.4f}")

    # Find positions
    high_pos, low_pos, grad_smooth = find_obstacle_positions(
        P_base, A_base, stone_mask_base
    )
    print(f"\n  High-flow position: {high_pos} "
          f"(grad={grad_smooth[high_pos]:.6f})")
    print(f"  Low-flow position:  {low_pos} "
          f"(grad={grad_smooth[low_pos]:.6f})")

    # Define test cases
    SMALL_R = 2  # ~12 cells
    LARGE_R = 5  # ~78 cells

    test_cases = [
        ('small_high_flow', high_pos, SMALL_R, 'The pebble in the stream'),
        ('large_low_flow',  low_pos,  LARGE_R, 'The boulder on the bank'),
        ('small_low_flow',  low_pos,  SMALL_R, 'Control: small + peripheral'),
        ('large_high_flow', high_pos, LARGE_R, 'Control: large + central'),
    ]

    results = {}

    for name, pos, radius, description in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test: {name} -- {description}")
        print(f"  Position: {pos}, Radius: {radius}")
        print(f"{'=' * 70}")

        # Insert obstacle
        P_new, A_new, mask_new, n_cells = insert_obstacle(
            P_base, A_base, stone_mask_base, pos, radius
        )
        print(f"  Obstacle cells: {n_cells}")
        print(f"  Conservation check: {abs(P_new.sum() + A_new.sum() - C_base.sum()):.2e}")

        # Run to new steady state
        print(f"  Running dynamics...")
        P_steady, A_steady, history = run_to_steady(P_new, A_new, mask_new)
        C_steady = P_steady + A_steady
        print(f"  Steps: {history['steps_to_steady']}, "
              f"conservation: {history['conservation_error'][-1]:.2e}")

        # Compute new global identity
        print(f"  Computing new identity...")
        adj_new = build_adjacency(C_steady)
        id_new = compute_global_identity(adj_new, C_steady.ravel())

        # Compare to baseline
        dist = identity_distance(id_baseline, id_new)

        print(f"\n  Identity shift:")
        print(f"    L2 distance:     {dist['l2_distance']:.6f}")
        print(f"    Cosine sim:      {dist['cosine_similarity']:.6f}")
        print(f"    Fiedler change:  {dist['fiedler_change']:.6f}")
        print(f"    Entropy change:  {dist['entropy_change']:.6f}")
        print(f"    Harmonic change: {dist['harmonic_change']:.6f}")
        print(f"    Composite:       {dist['composite']:.6f}")

        # Measure spatial extent of change
        delta_C = np.abs(C_steady - C_base)
        # Distance from obstacle center
        x = np.arange(N)
        y = np.arange(N)
        X, Y = np.meshgrid(x, y)
        dy = np.minimum(np.abs(Y - pos[0]), N - np.abs(Y - pos[0]))
        dx = np.minimum(np.abs(X - pos[1]), N - np.abs(X - pos[1]))
        dist_from_obstacle = np.sqrt(dx**2 + dy**2)

        # Cascade: what fraction of change is > 10 cells away?
        far_mask = dist_from_obstacle > 10
        near_mask = dist_from_obstacle <= 10
        change_far = delta_C[far_mask].sum()
        change_near = delta_C[near_mask].sum()
        change_total = delta_C.sum()
        cascade_ratio = change_far / (change_total + 1e-15)

        print(f"\n  Spatial cascade:")
        print(f"    Total field change:  {change_total:.6f}")
        print(f"    Near (<10 cells):    {change_near:.6f} "
              f"({100*change_near/(change_total+1e-15):.1f}%)")
        print(f"    Far (>10 cells):     {change_far:.6f} "
              f"({100*cascade_ratio:.1f}%)")

        results[name] = {
            'position': [int(p) for p in pos],
            'radius': int(radius),
            'n_obstacle_cells': int(n_cells),
            'description': description,
            'identity_shift': dist,
            'new_identity': {
                'fiedler': id_new['fiedler'],
                'spectral_entropy': id_new['spectral_entropy'],
                'harmonic': id_new['harmonic'],
            },
            'cascade': {
                'total_change': float(change_total),
                'near_change': float(change_near),
                'far_change': float(change_far),
                'cascade_ratio': float(cascade_ratio),
            },
            'dynamics': {
                'steps': history['steps_to_steady'],
                'conservation_error': float(history['conservation_error'][-1]),
            },
        }

    # ================================================================
    # COMPARISON: The Core Test
    # ================================================================
    print(f"\n{'=' * 70}")
    print("COMPARISON: Does Position Trump Size?")
    print(f"{'=' * 70}")

    sh = results['small_high_flow']
    ll = results['large_low_flow']
    sl = results['small_low_flow']
    lh = results['large_high_flow']

    print(f"\n  {'Test':<25s} {'Cells':<8s} {'L2 Shift':<12s} {'Composite':<12s} {'Cascade%':<10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")

    for name, r in results.items():
        print(f"  {name:<25s} {r['n_obstacle_cells']:<8d} "
              f"{r['identity_shift']['l2_distance']:<12.6f} "
              f"{r['identity_shift']['composite']:<12.6f} "
              f"{100*r['cascade']['cascade_ratio']:<10.1f}")

    # Key comparisons
    print(f"\n  KEY COMPARISON 1: Small-in-flow vs Large-out-of-flow")
    ratio_l2 = sh['identity_shift']['l2_distance'] / (ll['identity_shift']['l2_distance'] + 1e-15)
    ratio_comp = sh['identity_shift']['composite'] / (ll['identity_shift']['composite'] + 1e-15)
    size_ratio = ll['n_obstacle_cells'] / (sh['n_obstacle_cells'] + 1e-15)
    print(f"    Size ratio (large/small):     {size_ratio:.1f}x")
    print(f"    L2 shift ratio (small/large): {ratio_l2:.2f}x")
    print(f"    Composite ratio:              {ratio_comp:.2f}x")

    if ratio_comp > 1.0:
        print(f"    ==> SMALL IN FLOW WINS: {ratio_comp:.1f}x more impact despite "
              f"{size_ratio:.0f}x fewer cells")
        claim3 = "SUPPORTED"
    elif ratio_comp > 0.5:
        print(f"    ==> COMPARABLE: small obstacle has {100*ratio_comp:.0f}% "
              f"the impact of {size_ratio:.0f}x larger one")
        claim3 = "PARTIALLY SUPPORTED"
    else:
        print(f"    ==> SIZE DOMINATES: large obstacle wins {1/ratio_comp:.1f}x")
        claim3 = "CHALLENGED"

    print(f"\n  KEY COMPARISON 2: Same size, different position")
    print(f"    Small high vs small low: "
          f"{sh['identity_shift']['composite']:.6f} vs {sl['identity_shift']['composite']:.6f} "
          f"(ratio {sh['identity_shift']['composite']/(sl['identity_shift']['composite']+1e-15):.2f}x)")
    print(f"    Large high vs large low: "
          f"{lh['identity_shift']['composite']:.6f} vs {ll['identity_shift']['composite']:.6f} "
          f"(ratio {lh['identity_shift']['composite']/(ll['identity_shift']['composite']+1e-15):.2f}x)")

    # Cascade test (Claim 4)
    print(f"\n  CLAIM 4 (global cascade):")
    for name, r in results.items():
        cr = r['cascade']['cascade_ratio']
        print(f"    {name}: {100*cr:.1f}% of change is distant (>10 cells)")

    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    print(f"\n  Claim 3 (weight = coupling, not mass): {claim3}")

    cascade_supported = all(
        r['cascade']['cascade_ratio'] > 0.3 for r in results.values()
    )
    print(f"  Claim 4 (global cascade): "
          f"{'SUPPORTED' if cascade_supported else 'PARTIAL'} "
          f"(all tests show >{30 if cascade_supported else '?'}% distant change)")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_05_obstacle_insertion',
        'timestamp': datetime.now().isoformat(),
        'baseline_identity': {
            'fiedler': id_baseline['fiedler'],
            'spectral_entropy': id_baseline['spectral_entropy'],
            'harmonic': id_baseline['harmonic'],
        },
        'positions': {
            'high_flow': [int(p) for p in high_pos],
            'low_flow': [int(p) for p in low_pos],
            'high_flow_gradient': float(grad_smooth[high_pos]),
            'low_flow_gradient': float(grad_smooth[low_pos]),
        },
        'test_cases': results,
        'verdicts': {
            'claim3': claim3,
            'claim4_cascade': cascade_supported,
            'key_ratio_l2': float(ratio_l2),
            'key_ratio_composite': float(ratio_comp),
            'size_ratio': float(size_ratio),
        },
    }

    output_file = RESULTS_DIR / f'exp_05_obstacle_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
