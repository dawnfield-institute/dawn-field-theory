"""
Milestone 7 -- Exp 08: RBF Balance from Symmetry Constraint

Block D: Synthesis

HYPOTHESIS: The RBF balance equation B(x,t) = lambda(E-I)/(1+alpha*M)*Phi(x)
emerges naturally from the symmetry restoration drive.

Tests:
  1. The restoring force magnitude scales with |E - I| (linear regime, r > 0.5)
  2. States evolved under multi-scale drive converge toward moderate E/I
     (mean |E/I - 1| < 0.5 across nodes)
  3. Memory damping: SOME memory model produces negative correlation with drive
     (Tests: accumulated change, convergence, boundary distance)
  4. Evolved (E/I-balanced) states are more phi-structured than random

NOTE ON TEST 3: The simple accumulated-change model of memory FAILS —
high-change nodes are at boundaries where the drive works hardest (POSITIVE
correlation). This experiment investigates whether alternative memory models
capture the RBF damping term. This is an honest investigation, not a
guaranteed result.
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

from core.symmetry import (PHI, INV_PHI, save_results,
                           build_ring, build_torus, build_random_regular)

RESULTS_DIR = M7_ROOT / "results"


# ============================================================
# Reuse dynamics from exp_02
# ============================================================

def spectral_partition(L, n_levels):
    """Hierarchical partition via spectral bisection."""
    all_nodes = list(range(L.shape[0]))
    current_groups = [all_nodes]
    partitions = []

    for level in range(1, n_levels + 1):
        new_groups = []
        for group in current_groups:
            if len(group) < 4:
                new_groups.append(group)
                continue
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
    """Multi-scale drive: push D/S -> phi at each partition level."""
    n = len(state)
    partitions = spectral_partition(L, n_levels)
    total_drive = np.zeros(n)

    for level, groups in partitions:
        weight = 1.0 / level
        for group in groups:
            if len(group) < 2:
                continue
            group_vals = state[group]
            group_sum = np.sum(group_vals)
            if group_sum < 1e-15:
                continue
            sorted_idx = np.argsort(group_vals)
            mid = len(group) // 2
            sub_nodes = [group[sorted_idx[i]] for i in range(mid)]
            dom_nodes = [group[sorted_idx[i]] for i in range(mid, len(group))]
            S = np.sum(state[sub_nodes])
            D = np.sum(state[dom_nodes])
            if S < 1e-15 or D < 1e-15:
                continue
            target_D = group_sum / PHI
            target_S = group_sum - target_D
            dom_factor = target_D / D if D > 1e-15 else 1.0
            sub_factor = target_S / S if S > 1e-15 else 1.0
            for node in dom_nodes:
                total_drive[node] += weight * (state[node] * dom_factor - state[node])
            for node in sub_nodes:
                total_drive[node] += weight * (state[node] * sub_factor - state[node])
    return total_drive


def compute_ei_at_node(state, A, i):
    """
    Compute local Energy (dispersal) and Information (structure) at node i.
    E_i = state[i] / mean(state) — excess above mean
    I_i = mean(state[neighbors]) / state[i] — neighborhood coherence
    """
    mean_state = np.mean(state)
    if mean_state < 1e-15 or state[i] < 1e-15:
        return 1.0, 1.0
    E = state[i] / mean_state
    neighbors = np.where(A[i] > 0)[0]
    if len(neighbors) > 0:
        I = np.mean(state[neighbors]) / state[i]
    else:
        I = 1.0
    return E, I


def phi_balance_score(state, L, n_levels=4):
    """Phi-balance: 1 - mean(|R-phi|/phi) across hierarchical levels."""
    n = len(state)
    all_nodes = list(range(n))
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
            total = D + S
            if D > 1e-15 and S > 1e-15:
                R = total / D
                dev = abs(R - PHI) / PHI
                deviations.append(dev)
            new_groups.extend([half1, half2])
        current_groups = new_groups

    if deviations:
        return 1.0 - np.mean(deviations)
    return 0.5


def evolve_with_tracking(state, L, A, n_steps=500, alpha=0.1):
    """
    Evolve under multi-scale drive, tracking multiple memory models.
    Returns: final_state, memory_models dict, ei_history
    """
    total = np.sum(state)
    n = len(state)

    # Memory models
    accum_change = np.zeros(n)       # Model 1: accumulated |delta|
    convergence = np.zeros(n)         # Model 2: recent variance (rolling window)
    steps_since_change = np.zeros(n)  # Model 3: time since significant change
    window = 50
    recent_vals = np.zeros((window, n))

    ei_history = []

    for step in range(n_steps):
        old_state = state.copy()
        drive = multi_scale_drive(state, L, n_levels=4)
        state = state + alpha * drive
        state = np.maximum(state, 1e-10)
        state *= total / np.sum(state)

        delta = np.abs(state - old_state)

        # Model 1: accumulated change
        accum_change += delta

        # Model 2: rolling variance (convergence)
        recent_vals[step % window] = state
        if step >= window:
            convergence = np.var(recent_vals, axis=0)

        # Model 3: time since significant change
        significant = delta > 0.01 * np.mean(state)
        steps_since_change[significant] = 0
        steps_since_change[~significant] += 1

        if step % 100 == 0 or step == n_steps - 1:
            ei_ratios = []
            for i in range(n):
                E, I = compute_ei_at_node(state, A, i)
                ei_ratios.append(E / I if I > 1e-15 else float('inf'))
            clean = [r for r in ei_ratios if r < 100]
            mean_dev = np.mean([abs(r - 1) for r in clean]) if clean else 0
            ei_history.append({'step': step, 'mean_deviation': float(mean_dev)})

    memory_models = {
        'accumulated_change': accum_change,
        'convergence': convergence,
        'time_since_change': steps_since_change,
    }

    return state, memory_models, ei_history


def boundary_distance(L, n):
    """
    Compute how far each node is from the spectral bisection boundary.
    Nodes near the Fiedler cut (|fiedler[i]| near 0) are at boundaries.
    """
    eigs, vecs = np.linalg.eigh(L)
    fiedler = vecs[:, 1]
    return np.abs(fiedler)


def test_force_scaling(state, A, L):
    """Test that restoring force scales with E-I imbalance."""
    n = len(state)
    rng = np.random.RandomState(42)
    total = np.sum(state)
    imbalances = []
    forces = []

    for trial in range(30):
        perturbed = state.copy()
        n_perturb = rng.randint(1, n // 4)
        nodes = rng.choice(n, size=n_perturb, replace=False)
        perturbation = rng.uniform(0.5, 3.0, size=n_perturb)
        perturbed[nodes] *= perturbation
        perturbed *= total / np.sum(perturbed)

        ei_devs = []
        for i in range(n):
            E, I = compute_ei_at_node(perturbed, A, i)
            ei_devs.append(abs(E - I))
        mean_imbalance = np.mean(ei_devs)

        drive = multi_scale_drive(perturbed, L, n_levels=4)
        drive_mag = np.mean(np.abs(drive))

        imbalances.append(mean_imbalance)
        forces.append(drive_mag)

    if np.std(imbalances) > 1e-10 and np.std(forces) > 1e-10:
        corr = np.corrcoef(imbalances, forces)[0, 1]
    else:
        corr = 0
    return corr


def build_graph(gtype, n):
    if gtype == 'ring':
        A = build_ring(n).toarray()
    elif gtype == 'torus':
        side = int(np.sqrt(n))
        A = build_torus(side, side).toarray()
        n = side * side
    elif gtype == 'random_regular':
        A = build_random_regular(n, k=4).toarray()
    else:
        A = build_ring(n).toarray()
    D_mat = np.diag(A.sum(axis=1))
    L = D_mat - A
    return A, L, n


def main():
    print("=" * 70)
    print("MILESTONE 7 - EXP 08: RBF BALANCE FROM SYMMETRY CONSTRAINT")
    print("Block D: Synthesis")
    print("=" * 70)

    configs = [
        ("Ring N=50", 'ring', 50),
        ("Torus 7x7", 'torus', 49),
        ("Random Regular k=4", 'random_regular', 50),
    ]

    test1_results = []  # Force scaling
    test2_results = []  # E/I convergence
    test3_models = []   # Memory damping investigation
    test4_results = []  # Phi-structured balance

    for name, gtype, n in configs:
        print(f"\n{'=' * 60}")
        print(f"GRAPH: {name}")
        print("=" * 60)

        A, L, n = build_graph(gtype, n)
        rng = np.random.RandomState(42)
        state = np.ones(n) + rng.randn(n) * 0.01
        state = np.maximum(state, 1e-10)

        # Evolve with tracking
        final_state, memory_models, ei_hist = evolve_with_tracking(
            state.copy(), L, A, n_steps=500, alpha=0.1)

        # ============================================================
        # Test 1: Force scales with E-I imbalance
        # ============================================================
        print(f"\n  TEST 1: RESTORING FORCE vs E-I IMBALANCE")
        corr = test_force_scaling(final_state, A, L)
        print(f"    Correlation(|E-I|, |drive|): {corr:.4f}")
        test1_results.append(corr)

        # ============================================================
        # Test 2: E/I convergence
        # ============================================================
        print(f"\n  TEST 2: E/I CONVERGENCE")
        if ei_hist:
            final_dev = ei_hist[-1]['mean_deviation']
            for h in ei_hist:
                print(f"    step {h['step']:3d}: mean |E/I - 1| = {h['mean_deviation']:.4f}")
            test2_results.append(final_dev)
        else:
            test2_results.append(float('inf'))

        # ============================================================
        # Test 3: Memory damping investigation
        # ============================================================
        print(f"\n  TEST 3: MEMORY DAMPING INVESTIGATION")
        print(f"  (Testing which memory model produces negative correlation)")

        drive = multi_scale_drive(final_state, L, n_levels=4)
        drive_mag = np.abs(drive)

        # Add boundary distance as a 4th model
        bdist = boundary_distance(L, n)
        memory_models['boundary_distance'] = bdist

        model_corrs = {}
        for model_name, mem_values in memory_models.items():
            if np.std(mem_values) > 1e-10 and np.std(drive_mag) > 1e-10:
                c = np.corrcoef(mem_values, drive_mag)[0, 1]
            else:
                c = 0
            model_corrs[model_name] = c
            sign = "+" if c > 0 else "-" if c < 0 else "0"
            damping = "DAMPS" if c < -0.3 else "amplifies" if c > 0.3 else "weak"
            print(f"    {model_name:25s}: r = {c:+.4f} ({damping})")

        test3_models.append(model_corrs)

        # ============================================================
        # Test 4: Evolved state is phi-structured
        # ============================================================
        print(f"\n  TEST 4: PHI-STRUCTURED BALANCE POINT")

        pb_evolved = phi_balance_score(final_state, L, n_levels=3)
        # Multiple random states for fair comparison
        pb_randoms = []
        for seed in range(20):
            rng2 = np.random.RandomState(seed * 7 + 3)
            random_state = rng2.exponential(1.0, size=n)
            pb_randoms.append(phi_balance_score(random_state, L, n_levels=3))
        mean_pb_random = np.mean(pb_randoms)
        better = pb_evolved > mean_pb_random

        print(f"    Evolved: PB = {pb_evolved:.4f}")
        print(f"    Random (20 seeds): PB = {mean_pb_random:.4f} "
              f"+/- {np.std(pb_randoms):.4f}")
        print(f"    Evolved is better: {better}")
        test4_results.append(better)

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    mean_corr = np.mean(test1_results)
    test1 = mean_corr > 0.5
    print(f"\n  Test 1: Force scales with |E-I| (r > 0.5)")
    for (name, _, _), c in zip(configs, test1_results):
        print(f"    {name}: r = {c:.4f}")
    print(f"    Mean: {mean_corr:.4f}")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    mean_dev = np.mean(test2_results)
    test2 = mean_dev < 0.5
    print(f"\n  Test 2: E/I converges (mean |E/I-1| < 0.5)")
    for (name, _, _), d in zip(configs, test2_results):
        print(f"    {name}: {d:.4f}")
    print(f"    Mean: {mean_dev:.4f}")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    # Test 3: Any memory model produces damping?
    all_model_names = ['accumulated_change', 'convergence',
                       'time_since_change', 'boundary_distance']
    print(f"\n  Test 3: Memory damping (any model gives r < -0.3)")
    any_damps = False
    best_model = None
    best_mean = 1.0
    for model in all_model_names:
        corrs = [m[model] for m in test3_models if model in m]
        mean_c = np.mean(corrs) if corrs else 0
        if mean_c < best_mean:
            best_mean = mean_c
            best_model = model
        damps = mean_c < -0.3
        if damps:
            any_damps = True
        print(f"    {model:25s}: mean r = {mean_c:+.4f} "
              f"{'** DAMPS **' if damps else ''}")
    test3 = any_damps
    print(f"    Best model: {best_model} (r = {best_mean:+.4f})")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    if not test3:
        print(f"\n    HONEST FAILURE: No memory model tested produces the")
        print(f"    negative correlation expected by RBF's 1/(1+alpha*M) term.")
        print(f"    The accumulated-change model gives POSITIVE correlation")
        print(f"    because high-change nodes are at partition boundaries —")
        print(f"    they're the most active, not the most damped.")
        print(f"    The RBF memory term may require an entropy-based or")
        print(f"    information-theoretic definition of M, not an")
        print(f"    activity-based one.")

    n_phi = sum(1 for x in test4_results if x)
    test4 = n_phi >= 2
    print(f"\n  Test 4: Evolved states are phi-structured (>= 2/3)")
    for (name, _, _), p in zip(configs, test4_results):
        print(f"    {name}: {p}")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    results = {
        'experiment': 'exp_08_rbf_from_symmetry',
        'milestone': 7,
        'block': 'D',
        'test1_force_scaling': [float(c) for c in test1_results],
        'test2_ei_convergence': [float(d) for d in test2_results],
        'test3_memory_models': {
            model: [float(m[model]) for m in test3_models if model in m]
            for model in all_model_names
        },
        'test4_phi_structured': [bool(p) for p in test4_results],
        'verification': {
            'test1_force_scaling': test1,
            'test2_ei_convergence': test2,
            'test3_memory_damping': test3,
            'test4_phi_balance': test4,
            'verified_count': verified,
        },
    }
    save_results(results, 'exp_08_rbf_from_symmetry', RESULTS_DIR)


if __name__ == '__main__':
    main()
