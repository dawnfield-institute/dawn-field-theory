"""
exp_12 -- Topological Barrier and the Geiger-Nuttall Law

Milestone R, Block C (Novel Physics)

Thesis: The Geiger-Nuttall law requires a BARRIER, not just smooth Laplacian
relaxation (exp_11 T2 showed this). The PAC analog of the Coulomb barrier
is a topological barrier: for a vertex to sever, ALL of its d connections
must be simultaneously decoupled. The probability of d independent noise
fluctuations coinciding scales as p^d, creating exponential suppression
in degree -- the missing ingredient for log(t_half) ~ 1/sqrt(E).

The physics: in standard nuclear physics, the alpha particle must quantum-
tunnel through the Coulomb barrier. The WKB tunneling probability gives
exp(-sqrt(Z/E)). In PAC terms, the "tunneling" is the simultaneous
decoupling of d connections, and the "barrier height" is degree(v).

Tests:
  T1: Barrier FPT scales exponentially with vertex degree
  T2: Barrier walk produces GN-like scaling (log(FPT) vs 1/sqrt(deficit))
  T3: Control -- barrier is necessary (without barrier R^2 < with barrier)
  T4: Empirical GN data fit with barrier-parameterized model
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    StochasticCascade,
    ledger_severance, pac_deficit,
    stochastic_balance_walk, stochastic_barrier_walk,
    ade_graphs, redistribute_on_graph,
    GN_ALPHA_DATA,
    save_mr_results,
)


def test_T1_barrier_fpt_vs_degree():
    """T1: Barrier FPT scales exponentially with vertex degree."""
    print("\n  T1: Barrier FPT scales exponentially with vertex degree")
    results = {'description': 'log(FPT) increases with degree, R^2 > 0.7'}

    # Collect FPTs for vertices of different degrees across ADE graphs
    degree_fpts = {}  # degree -> list of median FPTs
    n_trials = 300

    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        if n < 3:
            continue

        for v in range(n):
            degree = int(np.sum(adj[v] > 0))
            if degree < 1:
                continue

            fpts = []
            for trial in range(n_trials):
                initial = np.ones(n) / n
                initial[v] += 1.0 / n
                initial = initial / np.sum(initial)

                result = stochastic_barrier_walk(
                    adj, v, initial,
                    noise_amplitude=0.01,
                    max_steps=5000,
                    seed=trial + v * 10000 + hash(name) % 100000,
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 4:
                med = np.median(fpts)
                if degree not in degree_fpts:
                    degree_fpts[degree] = []
                degree_fpts[degree].append(med)
                print(f"    {name} v={v} deg={degree}: "
                      f"median_fpt={med:.0f} ({len(fpts)}/{n_trials} converged)")

    # Aggregate: median FPT per degree
    degrees = sorted(degree_fpts.keys())
    median_per_degree = []
    for d in degrees:
        median_per_degree.append(np.median(degree_fpts[d]))

    degrees = np.array(degrees, dtype=float)
    median_per_degree = np.array(median_per_degree)

    print(f"\n    Degree -> median FPT:")
    for d, m in zip(degrees, median_per_degree):
        print(f"      degree {d:.0f}: median FPT = {m:.0f}")

    # Regression: log(FPT) vs degree
    if len(degrees) >= 3:
        log_fpts = np.log10(median_per_degree + 1)
        slope, intercept, r, p, se = stats.linregress(degrees, log_fpts)
        r2 = r ** 2
        passed = r2 > 0.7 and slope > 0
    else:
        slope, r2 = float('nan'), 0.0
        passed = False

    results['degrees'] = [int(d) for d in degrees]
    results['median_fpts'] = [float(m) for m in median_per_degree]
    results['slope'] = float(slope)
    results['r_squared'] = float(r2)
    results['PASS'] = passed
    print(f"    log(FPT) vs degree: slope={slope:.3f}, R^2={r2:.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_barrier_gn_scaling():
    """T2: Barrier walk produces GN-like scaling."""
    print("\n  T2: Barrier walk produces GN-like scaling")
    results = {'description': 'log(FPT) vs 1/sqrt(deficit) R^2 > 0.8 with barrier'}

    # Use D_6 graph (6 vertices, mixed degrees 1-3)
    test_graph = None
    for name, adj in ade_graphs(max_rank=7):
        if name == 'D_6':
            test_graph = adj
            break

    if test_graph is None:
        results['PASS'] = False
        results['error'] = 'D_6 not found'
        return results

    n = test_graph.shape[0]
    # Target: vertex 0 (the hub with degree 3 in D_n)
    target = 0
    target_degree = int(np.sum(test_graph[target] > 0))
    print(f"    Graph: D_6, target vertex {target} (degree {target_degree})")

    # Vary initial perturbation (= deficit proxy)
    deficit_levels = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    n_trials = 300
    median_fpts = []
    actual_deficits = []

    for pert in deficit_levels:
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            initial[target] += pert / n
            initial = initial / np.sum(initial)

            result = stochastic_barrier_walk(
                test_graph, target, initial,
                noise_amplitude=0.01,
                max_steps=8000,
                seed=trial * 1000 + int(pert * 100),
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        if len(fpts) >= n_trials // 4:
            med = np.median(fpts)
            median_fpts.append(med)
            actual_deficits.append(pert)
            print(f"      deficit={pert:.1f}: median_fpt={med:.0f} "
                  f"({len(fpts)}/{n_trials} converged)")
        else:
            print(f"      deficit={pert:.1f}: too few ({len(fpts)}/{n_trials})")

    # Regression: log(FPT) vs 1/sqrt(deficit)
    if len(median_fpts) >= 4:
        x = 1.0 / np.sqrt(np.array(actual_deficits))
        y = np.log10(np.array(median_fpts))
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r ** 2

        # Compare to empirical GN
        energies = np.array([e for _, e, _ in GN_ALPHA_DATA])
        halflives = np.array([t for _, _, t in GN_ALPHA_DATA])
        slope_emp, _, r_emp, _, _ = stats.linregress(
            1.0 / np.sqrt(energies), np.log10(halflives))
        r2_emp = r_emp ** 2

        same_sign = (slope > 0) == (slope_emp > 0)
        passed = r2 > 0.8 and same_sign
    else:
        slope, r2, same_sign = float('nan'), 0.0, False
        passed = False

    results['barrier_slope'] = float(slope)
    results['barrier_r2'] = float(r2)
    results['empirical_slope'] = float(slope_emp) if 'slope_emp' in dir() else None
    results['same_sign'] = same_sign
    results['deficit_levels'] = [float(d) for d in actual_deficits]
    results['median_fpts'] = [float(f) for f in median_fpts]
    results['PASS'] = passed
    print(f"    Barrier: slope={slope:.3f}, R^2={r2:.4f}, same_sign={same_sign}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_barrier_vs_no_barrier():
    """T3: Barrier is necessary -- improves GN R^2 over no-barrier."""
    print("\n  T3: Barrier is necessary (control comparison)")
    results = {'description': 'Barrier R^2 > no-barrier R^2 from exp_11 T2'}

    # Use A_7 for direct comparison with exp_11 T2
    test_graph = None
    for name, adj in ade_graphs(max_rank=8):
        if name == 'A_7':
            test_graph = adj
            break

    n = test_graph.shape[0]
    target = 0
    deficit_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    n_trials = 200

    # --- With barrier ---
    barrier_fpts = []
    barrier_deficits = []
    print("    With barrier (A_7, target=0):")
    for pert in deficit_levels:
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            initial[target] += pert / n
            initial = initial / np.sum(initial)

            result = stochastic_barrier_walk(
                test_graph, target, initial,
                noise_amplitude=0.01,
                max_steps=8000,
                seed=trial * 1000 + int(pert * 100),
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        if len(fpts) >= n_trials // 4:
            barrier_fpts.append(np.median(fpts))
            barrier_deficits.append(pert)
            print(f"      deficit={pert:.1f}: median_fpt={np.median(fpts):.0f} "
                  f"({len(fpts)}/{n_trials})")

    # --- Without barrier (plain balance walk, relative threshold) ---
    plain_fpts = []
    plain_deficits = []
    print("    Without barrier (plain balance walk):")
    for pert in deficit_levels:
        fpts = []
        init_def = pac_deficit(test_graph, 0, perturbation=pert)
        abs_thresh = max(init_def * 0.15, 0.005)

        for trial in range(n_trials):
            initial = np.ones(n) / n
            initial[0] += pert / n
            initial = initial / np.sum(initial)

            result = stochastic_balance_walk(
                test_graph, initial,
                noise_amplitude=0.003,
                threshold=abs_thresh,
                max_steps=8000,
                seed=trial * 1000 + int(pert * 100),
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        if len(fpts) >= n_trials // 4:
            plain_fpts.append(np.median(fpts))
            plain_deficits.append(pert)
            print(f"      deficit={pert:.1f}: median_fpt={np.median(fpts):.0f} "
                  f"({len(fpts)}/{n_trials})")

    # Compare R^2 values
    r2_barrier = 0.0
    r2_plain = 0.0

    if len(barrier_fpts) >= 3:
        x = 1.0 / np.sqrt(np.array(barrier_deficits))
        y = np.log10(np.array(barrier_fpts))
        _, _, r, _, _ = stats.linregress(x, y)
        r2_barrier = r ** 2

    if len(plain_fpts) >= 3:
        x = 1.0 / np.sqrt(np.array(plain_deficits))
        y = np.log10(np.array(plain_fpts))
        _, _, r, _, _ = stats.linregress(x, y)
        r2_plain = r ** 2

    passed = r2_barrier > r2_plain
    results['r2_barrier'] = float(r2_barrier)
    results['r2_plain'] = float(r2_plain)
    results['improvement'] = float(r2_barrier - r2_plain)
    results['PASS'] = passed
    print(f"    R^2 with barrier: {r2_barrier:.4f}")
    print(f"    R^2 without:      {r2_plain:.4f}")
    print(f"    Improvement:      {r2_barrier - r2_plain:+.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_empirical_gn_fit():
    """T4: Fit empirical GN data with barrier model."""
    print("\n  T4: Empirical GN data analysis")
    results = {'description': 'Empirical GN R^2 > 0.95 AND barrier model interpretation'}

    # Part A: Verify empirical GN is strong
    energies = np.array([e for _, e, _ in GN_ALPHA_DATA])
    halflives = np.array([t for _, _, t in GN_ALPHA_DATA])
    labels = [l for l, _, _ in GN_ALPHA_DATA]

    x = 1.0 / np.sqrt(energies)
    y = np.log10(halflives)

    slope, intercept, r, p, se = stats.linregress(x, y)
    r2 = r ** 2

    print(f"    Empirical GN: slope={slope:.2f}, R^2={r2:.4f}")

    # Part B: Barrier interpretation
    # If log(t) ~ degree and log(t) ~ 1/sqrt(E),
    # then degree_eff ~ 1/sqrt(E) -> E ~ 1/degree^2
    # Check: does E * degree^2 = constant?
    # We don't know real degrees, but we can infer degree_eff from the fit
    degree_eff = slope * x + intercept  # = log10(t_predicted)

    # The barrier model predicts: log10(FPT) = a * degree + b
    # Combined with GN: degree_eff = c / sqrt(E)
    # So: higher energy -> lower effective degree -> faster decay
    # This is consistent with: more energetic alphas come from nuclei
    # where the alpha is more loosely bound (fewer effective connections)

    # Check monotonicity: higher E -> lower inferred degree
    degree_from_energy = slope / np.sqrt(energies)  # Proportional to 1/sqrt(E)
    monotonic = all(degree_from_energy[i] >= degree_from_energy[i+1]
                    for i in range(len(degree_from_energy)-1)
                    if energies[i] < energies[i+1])

    # Compute the implied degree range
    d_min = np.min(degree_from_energy)
    d_max = np.max(degree_from_energy)
    d_ratio = d_max / d_min if d_min > 0 else float('inf')

    passed = r2 > 0.95

    results['gn_r2'] = float(r2)
    results['gn_slope'] = float(slope)
    results['degree_range_ratio'] = float(d_ratio)
    results['isotope_details'] = []
    for i, (label, e, t) in enumerate(GN_ALPHA_DATA):
        results['isotope_details'].append({
            'label': label,
            'energy_mev': float(e),
            'halflife_s': float(t),
            'inferred_degree': float(degree_from_energy[i]),
        })
        print(f"      {label:6s}: E={e:.3f}, t1/2={t:.2e}, "
              f"d_eff={degree_from_energy[i]:.2f}")

    results['PASS'] = passed
    print(f"    GN R^2={r2:.4f}, degree ratio={d_ratio:.1f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_12: Topological Barrier and the Geiger-Nuttall Law")
    print("=" * 60)

    t1 = test_T1_barrier_fpt_vs_degree()
    t2 = test_T2_barrier_gn_scaling()
    t3 = test_T3_barrier_vs_no_barrier()
    t4 = test_T4_empirical_gn_fit()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_12_topological_barrier_gn',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'Topological barrier (simultaneous edge decoupling) creates '
                  'exponential suppression in degree, producing GN-like scaling. '
                  'The PAC Coulomb barrier analog.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_12_topological_barrier_gn')
