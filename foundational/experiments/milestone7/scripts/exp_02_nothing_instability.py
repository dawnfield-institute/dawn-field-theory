"""
Milestone 7 -- Exp 02: "Nothing" Is Unstable Under Multi-Scale Drive

Block A: Foundations

HYPOTHESIS: A uniform field ("nothing") is unstable when a MULTI-SCALE
symmetry drive operates under PAC conservation.

From exp_01: phi emerges from cross-scale relational self-reference.
The drive toward phi operates at EVERY scale simultaneously — it wants
the dominant/subordinate split to be phi at scale 1, phi within each
half at scale 2, phi within each quarter at scale 3, etc.

A uniform state cannot satisfy this multi-scale drive under conservation.
The incompatibility forces structure formation.

Key distinction: a FLAT drive (same scale everywhere) cannot destabilize
a uniform state on a regular graph. The instability requires HIERARCHY —
the drive acting at multiple scales.

Tests:
  1. Multi-scale drive + conservation: perturbations grow (instability)
  2. Single-scale drive + conservation: perturbations decay (stable)
  3. Multi-scale drive WITHOUT conservation: converges to uniform (no structure)
  4. Emergent structure depth scales with drive range (more scales = more structure)
"""

import sys
import numpy as np
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
M7_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M7_ROOT))

from core.symmetry import PHI, INV_PHI, save_results

RESULTS_DIR = M7_ROOT / "results"


def spectral_partition(L, n_levels):
    """
    Hierarchical partition of a graph using spectral bisection.
    Returns a list of (level, partition_indices) pairs describing
    which nodes belong to which group at each level.
    """
    eigs, vecs = np.linalg.eigh(L)

    partitions = []
    # Level 0: whole system
    all_nodes = list(range(L.shape[0]))
    current_groups = [all_nodes]

    for level in range(1, n_levels + 1):
        new_groups = []
        for group in current_groups:
            if len(group) < 4:
                new_groups.append(group)
                continue
            # Bisect using Fiedler vector restricted to this group
            sub_L = L[np.ix_(group, group)]
            sub_eigs, sub_vecs = np.linalg.eigh(sub_L)
            fiedler = sub_vecs[:, 1]
            half1 = [group[i] for i in range(len(group)) if fiedler[i] >= 0]
            half2 = [group[i] for i in range(len(group)) if fiedler[i] < 0]
            if half1 and half2:
                new_groups.extend([half1, half2])
            else:
                new_groups.append(group)
        partitions.append((level, new_groups))
        current_groups = new_groups

    return partitions


def multi_scale_drive(state, L, n_levels=4):
    """
    Multi-scale drive: at each partition level, push the dominant/subordinate
    split of each group toward phi.

    At level k, for each group:
      - Compute dominant half sum D and subordinate half sum S
      - The drive pushes toward D/S = phi
      - This is applied to individual nodes proportionally

    The total drive on each node is the SUM across all scales.
    """
    n = len(state)
    partitions = spectral_partition(L, n_levels)
    total_drive = np.zeros(n)

    for level, groups in partitions:
        # Weight decreases with level (coarser scales matter more)
        weight = 1.0 / level

        for group in groups:
            if len(group) < 2:
                continue

            group_vals = state[group]
            group_sum = np.sum(group_vals)
            if group_sum < 1e-15:
                continue

            # Split group into dominant and subordinate halves by value
            sorted_idx = np.argsort(group_vals)
            mid = len(group) // 2
            sub_nodes = [group[sorted_idx[i]] for i in range(mid)]
            dom_nodes = [group[sorted_idx[i]] for i in range(mid, len(group))]

            S = np.sum(state[sub_nodes])
            D = np.sum(state[dom_nodes])

            if S < 1e-15 or D < 1e-15:
                continue

            # Current ratio
            R = (D + S) / D  # parent/dominant
            # Target: R = phi (from exp_01 cross-scale constraint)
            # Target dominant: group_sum / phi
            # Target subordinate: group_sum * (1 - 1/phi)

            target_D = group_sum / PHI
            target_S = group_sum - target_D

            # Drive: proportional redistribution toward target
            if D > 1e-15:
                dom_factor = target_D / D
            else:
                dom_factor = 1.0
            if S > 1e-15:
                sub_factor = target_S / S
            else:
                sub_factor = 1.0

            for node in dom_nodes:
                total_drive[node] += weight * (state[node] * dom_factor - state[node])
            for node in sub_nodes:
                total_drive[node] += weight * (state[node] * sub_factor - state[node])

    return total_drive


def single_scale_drive(state, A):
    """
    Single-scale (flat) drive: push each node toward phi * mean(neighbors).
    From the failed version — this should NOT destabilize uniform states.
    """
    n = len(state)
    drive = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        if len(neighbors) > 0:
            mean_nb = np.mean(state[neighbors])
            drive[i] = PHI * mean_nb - state[i]
    return drive


def evolve(state, drive_fn, conserve=True, n_steps=500, alpha=0.05):
    """
    Evolve state under a drive function with optional conservation.
    Returns state, CV history.
    """
    total = np.sum(state)
    cv_history = []

    for step in range(n_steps):
        drive = drive_fn(state)
        state = state + alpha * drive
        state = np.maximum(state, 1e-10)

        if conserve:
            state *= total / np.sum(state)

        cv = np.std(state) / np.mean(state) if np.mean(state) > 0 else 0
        cv_history.append(cv)

    return state, cv_history


def build_graph_laplacian(graph_type, n):
    """Build adjacency matrix and Laplacian for a given graph type."""
    from core.symmetry import build_ring, build_torus, build_random_regular

    if graph_type == 'ring':
        A = build_ring(n).toarray()
    elif graph_type == 'torus':
        side = int(np.sqrt(n))
        A = build_torus(side, side).toarray()
        n = side * side
    elif graph_type == 'random_regular':
        A = build_random_regular(n, k=4).toarray()
    else:
        A = build_ring(n).toarray()

    D = np.diag(A.sum(axis=1))
    L = D - A
    return A, L, n


def main():
    print("=" * 70)
    print("MILESTONE 7 - EXP 02: NOTHING IS UNSTABLE")
    print("Block A: Foundations")
    print("=" * 70)

    configs = [
        ("Ring N=50", 'ring', 50),
        ("Torus 7x7", 'torus', 49),
        ("Random Regular k=4 N=50", 'random_regular', 50),
    ]

    # ============================================================
    # Test 1: Multi-scale drive + conservation => instability
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: MULTI-SCALE DRIVE + CONSERVATION")
    print("(Does near-uniform state develop structure?)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    test1_results = []

    for name, gtype, n in configs:
        A, L, n = build_graph_laplacian(gtype, n)

        # Near-uniform start
        state_init = np.ones(n) + rng.randn(n) * 1e-6

        drive_fn = lambda s, L=L: multi_scale_drive(s, L, n_levels=4)
        final, cv_hist = evolve(state_init.copy(), drive_fn,
                                conserve=True, n_steps=500, alpha=0.1)

        cv_start = cv_hist[0]
        cv_end = cv_hist[-1]
        cv_max = max(cv_hist)
        growth = cv_end / max(cv_start, 1e-15)

        print(f"\n  {name}:")
        print(f"    CV start: {cv_start:.8f}")
        print(f"    CV end:   {cv_end:.8f}")
        print(f"    CV max:   {cv_max:.8f}")
        print(f"    Growth factor: {growth:.1f}x")
        print(f"    Structure formed: {cv_end > 0.01}")

        test1_results.append({
            'name': name,
            'cv_start': float(cv_start),
            'cv_end': float(cv_end),
            'cv_max': float(cv_max),
            'growth': float(growth),
            'structured': cv_end > 0.01,
        })

    # ============================================================
    # Test 2: Single-scale drive + conservation => stable
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: SINGLE-SCALE DRIVE + CONSERVATION (CONTROL)")
    print("(Flat drive should NOT destabilize uniform state)")
    print("=" * 60)

    test2_results = []

    for name, gtype, n in configs:
        A, L, n = build_graph_laplacian(gtype, n)

        state_init = np.ones(n) + rng.randn(n) * 1e-6

        drive_fn = lambda s, A=A: single_scale_drive(s, A)
        final, cv_hist = evolve(state_init.copy(), drive_fn,
                                conserve=True, n_steps=500, alpha=0.05)

        cv_end = cv_hist[-1]
        decayed = cv_end < cv_hist[0]

        print(f"\n  {name}:")
        print(f"    CV start: {cv_hist[0]:.8f}")
        print(f"    CV end:   {cv_end:.8f}")
        print(f"    Perturbation decayed: {decayed}")

        test2_results.append({
            'name': name,
            'cv_end': float(cv_end),
            'decayed': decayed,
        })

    # ============================================================
    # Test 3: Multi-scale drive WITHOUT conservation => no structure
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: MULTI-SCALE DRIVE, NO CONSERVATION (CONTROL)")
    print("(Without conservation, drive should homogenize)")
    print("=" * 60)

    test3_results = []

    for name, gtype, n in configs:
        A, L, n = build_graph_laplacian(gtype, n)

        # Start STRUCTURED (non-uniform)
        state_init = rng.exponential(1.0, size=n)
        cv_init = np.std(state_init) / np.mean(state_init)

        drive_fn = lambda s, L=L: multi_scale_drive(s, L, n_levels=4)
        final, cv_hist = evolve(state_init.copy(), drive_fn,
                                conserve=False, n_steps=500, alpha=0.1)

        cv_end = cv_hist[-1]
        destroyed = cv_end < cv_init * 0.5

        print(f"\n  {name}:")
        print(f"    CV start: {cv_init:.4f}")
        print(f"    CV end:   {cv_end:.4f}")
        print(f"    Structure destroyed: {destroyed}")

        test3_results.append({
            'name': name,
            'cv_start': float(cv_init),
            'cv_end': float(cv_end),
            'destroyed': destroyed,
        })

    # ============================================================
    # Test 4: More scales => more structure
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: STRUCTURE DEPTH vs NUMBER OF DRIVE SCALES")
    print("=" * 60)

    A, L, n = build_graph_laplacian('ring', 64)
    scale_vs_structure = []

    for n_levels in [1, 2, 3, 4, 5]:
        state_init = np.ones(n) + rng.randn(n) * 1e-6

        drive_fn = lambda s, L=L, nl=n_levels: multi_scale_drive(s, L, n_levels=nl)
        final, cv_hist = evolve(state_init.copy(), drive_fn,
                                conserve=True, n_steps=500, alpha=0.1)

        cv_end = cv_hist[-1]
        scale_vs_structure.append((n_levels, cv_end))
        print(f"  n_levels={n_levels}: CV_final={cv_end:.6f}")

    # Check monotonicity
    cvs = [s[1] for s in scale_vs_structure]
    monotonic = sum(1 for i in range(len(cvs)-1) if cvs[i+1] >= cvs[i] - 1e-10)
    mono_frac = monotonic / max(len(cvs) - 1, 1)
    print(f"\n  Monotonic: {monotonic}/{len(cvs)-1} ({mono_frac:.0%})")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test 1: Multi-scale + conservation => structure
    n_structured = sum(1 for r in test1_results if r['structured'])
    test1 = n_structured >= 2
    print(f"\n  Test 1: Multi-scale drive + conservation => structure")
    for r in test1_results:
        tag = '[structured]' if r['structured'] else '[uniform]'
        print(f"    {r['name']}: CV={r['cv_end']:.6f}, growth={r['growth']:.1f}x {tag}")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    # Test 2: Single-scale + conservation => stable
    n_decayed = sum(1 for r in test2_results if r['decayed'])
    test2 = n_decayed >= 2
    print(f"\n  Test 2: Single-scale drive + conservation => perturbation decays")
    for r in test2_results:
        print(f"    {r['name']}: decayed={r['decayed']}")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    # Test 3: Multi-scale WITHOUT conservation => homogenizes
    n_destroyed = sum(1 for r in test3_results if r['destroyed'])
    test3 = n_destroyed >= 2
    print(f"\n  Test 3: Multi-scale drive without conservation => homogenizes")
    for r in test3_results:
        print(f"    {r['name']}: {r['cv_start']:.4f} -> {r['cv_end']:.4f}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    # Test 4: More scales => more structure
    test4 = mono_frac >= 0.6
    print(f"\n  Test 4: More drive scales => more structure")
    print(f"    Monotonic: {mono_frac:.0%}")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    results = {
        'experiment': 'exp_02_nothing_instability',
        'milestone': 7,
        'block': 'A',
        'test1_multiscale_conservation': test1_results,
        'test2_singlescale_control': test2_results,
        'test3_no_conservation_control': test3_results,
        'test4_scales_vs_structure': {
            'data': [{'levels': l, 'cv': c} for l, c in scale_vs_structure],
            'monotonic_frac': float(mono_frac),
        },
        'verification': {
            'test1_instability': test1,
            'test2_flat_stable': test2,
            'test3_conservation_needed': test3,
            'test4_scale_dependence': test4,
            'verified_count': verified,
        },
    }
    save_results(results, 'exp_02_nothing_instability', RESULTS_DIR)


if __name__ == '__main__':
    main()
