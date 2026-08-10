"""
Milestone 7 -- Exp 05: Global Symmetry Requires Local Asymmetry

Block C: Consequences

HYPOTHESIS: As a system evolves toward global phi-balance (multi-scale
symmetry), it necessarily develops local asymmetry. You cannot achieve
phi-balance without local differences.

From exp_01: D/S = phi at every level (the cross-scale constraint).
From exp_02: multi-scale drive + conservation forces structure.

If all nodes are equal, D/S = 1 (not phi). So achieving D/S = phi at
any partition level REQUIRES nodes to differ — local asymmetry is the
necessary mechanism for global phi-balance.

Tests:
  1. Evolved states have BETTER phi-balance than uniform AND non-zero LA
     (PB_evolved > PB_uniform and LA_evolved > 0.3, across >= 2/3 graphs)
  2. Uniform state has LA = 0 (the only way to have zero local asymmetry
     is to have all nodes equal, which gives D/S = 1, not phi)
  3. Final evolved states: phi-balance > 0.80 AND LA > 0.3 simultaneously
  4. Holds across graph topologies (>= 2/3)
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

from core.symmetry import (PHI, INV_PHI, save_results, local_asymmetry,
                            build_ring, build_torus, build_random_regular)

RESULTS_DIR = M7_ROOT / "results"


def phi_balance_score(state, L, n_levels=3):
    """
    Measure how well the state achieves phi-balance at each hierarchical level.
    Score = 1 - mean(|R - phi| / phi) where R = (D+S)/D at each partition.
    """
    all_nodes = list(range(len(state)))
    current_groups = [all_nodes]
    deviations = []

    for level in range(1, n_levels + 1):
        new_groups = []
        for group in current_groups:
            if len(group) < 4:
                new_groups.append(group)
                continue

            sub_L = L[np.ix_(group, group)]
            eigs, vecs = np.linalg.eigh(sub_L)
            fiedler = vecs[:, 1]

            half1 = [group[i] for i in range(len(group)) if fiedler[i] >= 0]
            half2 = [group[i] for i in range(len(group)) if fiedler[i] < 0]

            if not half1 or not half2:
                new_groups.append(group)
                continue

            s1 = np.sum(state[half1])
            s2 = np.sum(state[half2])
            D = max(s1, s2)
            S = min(s1, s2)

            if S > 1e-15:
                R = (D + S) / D
                dev = abs(R - PHI) / PHI
            else:
                dev = 1.0

            deviations.append(dev)
            new_groups.extend([half1, half2])
        current_groups = new_groups

    return 1.0 - np.mean(deviations) if deviations else 0.0


def evolve_and_track(state, L, A, n_steps=500, alpha=0.1, sample_every=20):
    """
    Evolve under multi-scale drive, tracking phi-balance and local asymmetry.
    """
    from scripts.exp_02_nothing_instability import multi_scale_drive

    total = np.sum(state)
    trajectory = []

    for step in range(n_steps):
        drive = multi_scale_drive(state, L, n_levels=4)
        state = state + alpha * drive
        state = np.maximum(state, 1e-10)
        state *= total / np.sum(state)

        if step % sample_every == 0:
            pb = phi_balance_score(state, L, n_levels=3)
            la = local_asymmetry(state, A)
            trajectory.append({'step': step, 'phi_balance': pb, 'local_asym': la})

    return state, trajectory


def main():
    print("=" * 70)
    print("MILESTONE 7 - EXP 05: GLOBAL SYMMETRY REQUIRES LOCAL ASYMMETRY")
    print("Block C: Consequences")
    print("=" * 70)

    configs = [
        ("Ring N=50", 'ring', 50),
        ("Torus 7x7", 'torus', 49),
        ("Random Regular k=4", 'random_regular', 50),
    ]

    test1_results = []  # Co-increase
    test3_results = []  # Both high at end
    all_trajectories = {}

    for name, gtype, n in configs:
        print(f"\n{'=' * 60}")
        print(f"GRAPH: {name}")
        print("=" * 60)

        if gtype == 'ring':
            A = build_ring(n).toarray()
        elif gtype == 'torus':
            side = int(np.sqrt(n))
            A = build_torus(side, side).toarray()
            n = side * side
        elif gtype == 'random_regular':
            A = build_random_regular(n, k=4).toarray()
        D_mat = np.diag(A.sum(axis=1))
        L = D_mat - A

        # Uniform baseline
        uniform = np.ones(n)
        pb_uniform = phi_balance_score(uniform, L, n_levels=3)
        la_uniform = local_asymmetry(uniform, A)
        print(f"\n  Uniform state: PB={pb_uniform:.4f}, LA={la_uniform:.6f}")

        # Evolve from near-uniform
        rng = np.random.RandomState(42)
        state_init = np.ones(n) + rng.randn(n) * 1e-4

        state_final, traj = evolve_and_track(state_init.copy(), L, A,
                                              n_steps=500, alpha=0.1)

        all_trajectories[name] = traj

        # Trajectory analysis
        pb_start = traj[0]['phi_balance']
        la_start = traj[0]['local_asym']
        pb_end = traj[-1]['phi_balance']
        la_end = traj[-1]['local_asym']

        pb_better = pb_end > pb_uniform
        la_nonzero = la_end > 0.3

        print(f"\n  Evolved vs uniform:")
        print(f"    Phi-balance: {pb_uniform:.4f} (uniform) -> {pb_end:.4f} (evolved) "
              f"({'BETTER' if pb_better else 'worse'})")
        print(f"    Local asym:  {la_uniform:.4f} (uniform) -> {la_end:.4f} (evolved) "
              f"({'NON-ZERO' if la_nonzero else 'low'})")
        print(f"    Better PB + non-zero LA: {pb_better and la_nonzero}")

        test1_results.append(pb_better and la_nonzero)

        # Show a few trajectory points
        print(f"\n  Sample trajectory:")
        for t in traj[::5]:  # Every 5th sample
            print(f"    step {t['step']:3d}: PB={t['phi_balance']:.4f}, "
                  f"LA={t['local_asym']:.4f}")

        # Test 3: Both high at end
        both_high = pb_end > 0.80 and la_end > 0.3
        test3_results.append(both_high)
        print(f"\n  Final: PB={pb_end:.4f} (>0.80?), LA={la_end:.4f} (>0.30?)")
        print(f"  Both high: {both_high}")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    n_co = sum(1 for x in test1_results if x)
    test1 = n_co >= 2
    print(f"\n  Test 1: Evolved has better PB than uniform + non-zero LA (>= 2/3)")
    for (name, _, _), co in zip(configs, test1_results):
        print(f"    {name}: {co}")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    test2 = la_uniform < 0.01
    print(f"\n  Test 2: Uniform state has LA ~ 0")
    print(f"    LA(uniform) = {la_uniform:.6f}")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    n_both = sum(1 for x in test3_results if x)
    test3 = n_both >= 2
    print(f"\n  Test 3: Evolved states have PB > 0.80 AND LA > 0.30")
    for (name, _, _), bh in zip(configs, test3_results):
        print(f"    {name}: {bh}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    test4 = n_co >= 2 and n_both >= 2
    print(f"\n  Test 4: Cross-topology (T1 and T3 pass for >= 2/3)")
    print(f"    T1 pass: {n_co}/3, T3 pass: {n_both}/3")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    results = {
        'experiment': 'exp_05_global_local_asymmetry',
        'milestone': 7,
        'block': 'C',
        'test1_co_increase': [bool(x) for x in test1_results],
        'test3_both_high': [bool(x) for x in test3_results],
        'uniform_la': float(la_uniform),
        'verification': {
            'test1_co_increase': test1,
            'test2_uniform': test2,
            'test3_both_high': test3,
            'test4_cross_topology': test4,
            'verified_count': verified,
        },
    }
    save_results(results, 'exp_05_global_local_asymmetry', RESULTS_DIR)


if __name__ == '__main__':
    main()
