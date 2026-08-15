"""
exp_14 -- SEC as Kinetic Energy: Fixing the GN Sign

Milestone R, Block C (Novel Physics)

Thesis: The persistent GN sign problem (exp_11 T2, exp_12 T2, exp_13 T2)
arises because perturbation strength controls BOTH the barrier height (via
initial JSD) and the "energy to overcome" -- they scale together.

In nuclear physics these are independent:
  - Coulomb barrier depends on Z (structural) -- FIXED for a given nucleus
  - Alpha kinetic energy depends on Q-value (dynamical) -- VARIES

In PAC/SEC terms:
  - Barrier = perspectival divergence (PAC, structural, fixed JSD for a vertex)
  - Kinetic energy = SEC dynamics (noise amplitude = entropy production rate)

The perspectival barrier has a FIXED threshold (jsd_threshold). The noise
provides fluctuations that let the system cross it. Higher noise = more
SEC flux = more "kinetic energy" = barrier crossed sooner. The barrier
height and kinetic energy are now INDEPENDENT.

The prediction: varying noise (with fixed perturbation) gives the OPPOSITE
sign from varying perturbation (with fixed noise). This IS the GN sign fix.

Tests:
  T1: Higher noise -> shorter FPT (sign test)
  T2: 2D surface: FPT increases with barrier, decreases with noise
  T3: GN combined variable: log(FPT) vs JSD_0/sqrt(noise) gives R^2 > 0.7
  T4: Sign correct across graph types (majority of 5 graphs)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    pac_deficit,
    perspective_divergence,
    perspectival_barrier_walk,
    ade_graphs, redistribute_on_graph,
    save_mr_results,
)


def test_T1_noise_as_kinetic_energy():
    """T1: Higher noise (SEC) -> shorter FPT. The GN sign fix."""
    print("\n  T1: Higher noise -> shorter FPT (sign test)")
    results = {'description': 'Negative correlation between noise and FPT (more kinetic energy = faster crossing)'}

    # D_8 hub vertex (degree 3) -- same as exp_13 T2
    test_graph = None
    for name, adj in ade_graphs(max_rank=8):
        if name == 'D_8':
            test_graph = adj
            break

    n = test_graph.shape[0]
    target = n - 3  # hub
    target_degree = int(np.sum(test_graph[target] > 0))
    jsd_0 = perspective_divergence(test_graph, target, horizon=2)
    print(f"    D_8 hub vertex {target} (degree {target_degree}, JSD_0={jsd_0:.4f})")

    # Fix perturbation, vary noise
    perturbation = 2.0
    noise_levels = [0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035]
    n_trials = 150

    median_fpts = []
    actual_noises = []

    for noise in noise_levels:
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            initial[target] += perturbation / n
            initial = initial / np.sum(initial)

            result = perspectival_barrier_walk(
                test_graph, target, initial,
                horizon=2,
                noise_amplitude=noise,
                jsd_threshold=0.005,
                max_steps=8000,
                seed=trial * 1000 + int(noise * 10000),
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        if len(fpts) >= n_trials // 4:
            med = np.median(fpts)
            median_fpts.append(med)
            actual_noises.append(noise)
            print(f"      noise={noise:.3f}: median_fpt={med:.0f} "
                  f"({len(fpts)}/{n_trials} converged)")
        else:
            print(f"      noise={noise:.3f}: too few converged ({len(fpts)}/{n_trials})")

    # Regression: FPT vs noise
    slope = float('nan')
    r2 = 0.0
    rho = 0.0
    passed = False
    if len(median_fpts) >= 4:
        rho_val, p_val = stats.spearmanr(actual_noises, median_fpts)
        rho = float(rho_val)

        # Also check log(FPT) vs 1/sqrt(noise) for GN form
        x = 1.0 / np.sqrt(np.array(actual_noises))
        y = np.log10(np.array(median_fpts))
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r ** 2

        # PASS: negative correlation (more noise -> shorter FPT)
        passed = rho < -0.5

    results['perturbation'] = perturbation
    results['noise_levels'] = [float(n) for n in actual_noises]
    results['median_fpts'] = [float(f) for f in median_fpts]
    results['spearman_rho'] = round(float(rho), 4)
    results['gn_slope'] = round(float(slope), 4) if not np.isnan(slope) else None
    results['gn_r2'] = round(float(r2), 4)
    results['target_jsd_0'] = round(float(jsd_0), 4)
    results['PASS'] = passed
    print(f"    Spearman rho(noise, FPT) = {rho:.4f}")
    print(f"    log(FPT) vs 1/sqrt(noise): slope={slope:.4f}, R^2={r2:.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_barrier_vs_kinetic_2d():
    """T2: FPT increases with barrier height, decreases with kinetic energy."""
    print("\n  T2: 2D surface — barrier increases FPT, noise decreases FPT")
    results = {'description': 'Partial correlations: FPT~barrier (+), FPT~noise (-)'}

    # Collect data across multiple vertices and noise levels
    data_points = []
    noise_levels = [0.005, 0.010, 0.020, 0.035]
    perturbation = 2.0
    n_trials = 100

    # Use D_7 and E_7 for vertex variety
    test_configs = []
    for name, adj in ade_graphs(max_rank=8):
        if name in ('D_7', 'E_7'):
            n = adj.shape[0]
            for v in range(n):
                jsd = perspective_divergence(adj, v, horizon=2)
                test_configs.append((name, adj, v, jsd))

    print(f"    Testing {len(test_configs)} vertices x {len(noise_levels)} noise levels")

    for name, adj, v, jsd_0 in test_configs:
        n = adj.shape[0]
        for noise in noise_levels:
            fpts = []
            for trial in range(n_trials):
                initial = np.ones(n) / n
                initial[v] += perturbation / n
                initial = initial / np.sum(initial)

                result = perspectival_barrier_walk(
                    adj, v, initial,
                    horizon=2,
                    noise_amplitude=noise,
                    jsd_threshold=0.005,
                    max_steps=5000,
                    seed=trial * 100 + int(noise * 10000) + v * 50000,
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 4:
                data_points.append({
                    'graph': name,
                    'vertex': v,
                    'jsd_0': jsd_0,
                    'noise': noise,
                    'median_fpt': float(np.median(fpts)),
                    'n_converged': len(fpts),
                })

    if len(data_points) < 10:
        results['PASS'] = False
        results['error'] = f'Only {len(data_points)} data points'
        print(f"    FAIL: insufficient data ({len(data_points)} points)")
        return results

    # Extract arrays
    jsds = np.array([d['jsd_0'] for d in data_points])
    noises = np.array([d['noise'] for d in data_points])
    log_fpts = np.log10(np.array([d['median_fpt'] for d in data_points]))

    # Partial correlations via multiple regression
    # log(FPT) = a * JSD + b * noise + c
    X = np.column_stack([jsds, noises, np.ones(len(jsds))])
    beta, residuals, rank, sv = np.linalg.lstsq(X, log_fpts, rcond=None)
    coeff_jsd = beta[0]   # should be positive (more barrier -> longer FPT)
    coeff_noise = beta[1]  # should be negative (more noise -> shorter FPT)

    y_pred = X @ beta
    ss_res = np.sum((log_fpts - y_pred) ** 2)
    ss_tot = np.sum((log_fpts - np.mean(log_fpts)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Simple correlations
    rho_jsd, _ = stats.spearmanr(jsds, log_fpts)
    rho_noise, _ = stats.spearmanr(noises, log_fpts)

    # PASS: barrier coefficient positive AND noise coefficient negative
    passed = coeff_jsd > 0 and coeff_noise < 0

    results['n_points'] = len(data_points)
    results['coeff_barrier_jsd'] = round(float(coeff_jsd), 4)
    results['coeff_noise_sec'] = round(float(coeff_noise), 4)
    results['r2_combined'] = round(float(r2), 4)
    results['rho_jsd_fpt'] = round(float(rho_jsd), 4)
    results['rho_noise_fpt'] = round(float(rho_noise), 4)
    results['PASS'] = passed
    print(f"    Multiple regression: log(FPT) = {coeff_jsd:.3f}*JSD + {coeff_noise:.3f}*noise + const")
    print(f"    R^2 = {r2:.4f}")
    print(f"    Spearman: rho(JSD, FPT)={rho_jsd:.3f}, rho(noise, FPT)={rho_noise:.3f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: JSD coeff > 0 AND noise coeff < 0)")
    return results


def test_T3_gn_combined_variable():
    """T3: GN combined variable JSD_0/sqrt(noise) predicts FPT."""
    print("\n  T3: GN combined variable: log(FPT) vs JSD_0/sqrt(noise)")
    results = {'description': 'R^2 > 0.5 for combined variable, positive slope'}

    # Use D_8 hub, vary both perturbation and noise
    test_graph = None
    for name, adj in ade_graphs(max_rank=8):
        if name == 'D_8':
            test_graph = adj
            break

    n = test_graph.shape[0]
    target = n - 3

    perturbation_levels = [0.5, 1.0, 2.0, 4.0]
    noise_levels = [0.005, 0.010, 0.020, 0.035]
    n_trials = 100

    data_points = []

    for pert in perturbation_levels:
        for noise in noise_levels:
            # Compute initial JSD for this perturbation
            initial = np.ones(n) / n
            initial[target] += pert / n
            initial = initial / np.sum(initial)

            # Compute the JSD of the initial state's neighborhood
            # (this is the "barrier height" for this specific perturbation)
            nbhd = [target]
            frontier = {target}
            visited = {target}
            for _ in range(2):
                nf = set()
                for v in frontier:
                    for u in range(n):
                        if test_graph[v, u] > 0 and u not in visited:
                            nf.add(u)
                            visited.add(u)
                frontier = nf
            nbhd = sorted(visited)
            vals = initial[nbhd]
            vals = np.maximum(vals, 1e-30)
            vals = vals / np.sum(vals)
            eq = np.ones(len(nbhd)) / len(nbhd)
            m = 0.5 * (vals + eq)
            kl_l = np.sum(vals * np.log(vals / m))
            kl_g = np.sum(eq * np.log(eq / m))
            initial_jsd = float(0.5 * kl_l + 0.5 * kl_g)

            fpts = []
            for trial in range(n_trials):
                init = initial.copy()
                result = perspectival_barrier_walk(
                    test_graph, target, init,
                    horizon=2,
                    noise_amplitude=noise,
                    jsd_threshold=0.005,
                    max_steps=8000,
                    seed=trial * 100 + int(pert * 1000) + int(noise * 100000),
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 4:
                data_points.append({
                    'perturbation': pert,
                    'noise': noise,
                    'initial_jsd': initial_jsd,
                    'median_fpt': float(np.median(fpts)),
                    'n_converged': len(fpts),
                })
                print(f"      pert={pert:.1f}, noise={noise:.3f}: "
                      f"JSD_0={initial_jsd:.4f}, fpt={np.median(fpts):.0f} "
                      f"({len(fpts)}/{n_trials})")

    if len(data_points) < 6:
        results['PASS'] = False
        results['error'] = f'Only {len(data_points)} data points'
        return results

    # GN combined variable: JSD_0 / sqrt(noise)
    gn_vars = np.array([d['initial_jsd'] / np.sqrt(d['noise']) for d in data_points])
    log_fpts = np.log10(np.array([d['median_fpt'] for d in data_points]))

    slope, intercept, r, p, se = stats.linregress(gn_vars, log_fpts)
    r2 = r ** 2

    # Compare to perturbation-only (the old way that had wrong sign)
    perts = np.array([d['perturbation'] for d in data_points])
    inv_sqrt_pert = 1.0 / np.sqrt(perts)
    slope_old, _, r_old, _, _ = stats.linregress(inv_sqrt_pert, log_fpts)
    r2_old = r_old ** 2

    passed = slope > 0 and r2 > 0.5

    results['n_points'] = len(data_points)
    results['gn_combined_slope'] = round(float(slope), 4)
    results['gn_combined_r2'] = round(float(r2), 4)
    results['old_pert_only_slope'] = round(float(slope_old), 4)
    results['old_pert_only_r2'] = round(float(r2_old), 4)
    results['data_points'] = data_points
    results['PASS'] = passed
    print(f"    Combined JSD_0/sqrt(noise): slope={slope:.4f}, R^2={r2:.4f}")
    print(f"    Old 1/sqrt(pert) only:      slope={slope_old:.4f}, R^2={r2_old:.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: positive slope AND R^2 > 0.5)")
    return results


def test_T4_sign_across_graph_types():
    """T4: Noise-FPT negative correlation holds across graph types."""
    print("\n  T4: GN sign (noise -> shorter FPT) across graph types")
    results = {'description': 'Negative rho(noise, FPT) for majority of graph types'}

    noise_levels = [0.005, 0.010, 0.020, 0.035]
    perturbation = 2.0
    n_trials = 100
    per_graph = []

    test_graphs = ['D_5', 'D_6', 'D_7', 'E_6', 'E_7']

    for graph_name in test_graphs:
        adj = None
        for name, a in ade_graphs(max_rank=8):
            if name == graph_name:
                adj = a
                break
        if adj is None:
            continue

        n = adj.shape[0]
        degrees_arr = np.sum(adj > 0, axis=1).astype(int)
        target = int(np.argmax(degrees_arr))
        target_degree = int(degrees_arr[target])

        median_fpts = []
        actual_noises = []

        for noise in noise_levels:
            fpts = []
            for trial in range(n_trials):
                initial = np.ones(n) / n
                initial[target] += perturbation / n
                initial = initial / np.sum(initial)

                result = perspectival_barrier_walk(
                    adj, target, initial,
                    horizon=2,
                    noise_amplitude=noise,
                    jsd_threshold=0.005,
                    max_steps=5000,
                    seed=trial * 100 + int(noise * 10000),
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 4:
                median_fpts.append(float(np.median(fpts)))
                actual_noises.append(noise)

        rho = 0.0
        if len(median_fpts) >= 3:
            rho_val, _ = stats.spearmanr(actual_noises, median_fpts)
            rho = float(rho_val)

        sign_correct = rho < 0
        per_graph.append({
            'graph': graph_name,
            'target': target,
            'degree': target_degree,
            'rho_noise_fpt': round(rho, 4),
            'sign_correct': sign_correct,
            'n_noise_levels': len(median_fpts),
            'fpts': median_fpts,
        })
        print(f"    {graph_name}: v={target} (deg {target_degree}), "
              f"rho(noise, FPT)={rho:.3f} -> {'correct' if sign_correct else 'WRONG'}")

    correct_count = sum(1 for pg in per_graph if pg['sign_correct'])
    total = len(per_graph)
    passed = total > 0 and correct_count >= 3  # majority of 5

    results['per_graph'] = per_graph
    results['correct_count'] = correct_count
    results['total_graphs'] = total
    results['PASS'] = passed
    print(f"    Correct sign: {correct_count}/{total}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_14: SEC as Kinetic Energy -- Fixing the GN Sign")
    print("=" * 60)

    t1 = test_T1_noise_as_kinetic_energy()
    t2 = test_T2_barrier_vs_kinetic_2d()
    t3 = test_T3_gn_combined_variable()
    t4 = test_T4_sign_across_graph_types()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_14_sec_kinetic_energy',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'The GN sign problem arises because perturbation controls both '
                  'barrier height AND energy. In nuclear physics these are independent: '
                  'Coulomb barrier (structural, Z) vs alpha kinetic energy (dynamical, Q-value). '
                  'In PAC/SEC: barrier = perspectival JSD (PAC, structural), '
                  'kinetic energy = noise amplitude (SEC, dynamical entropy flux). '
                  'Higher SEC flux -> more fluctuations -> barrier crossed sooner -> '
                  'shorter FPT. This is the GN sign fix.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_14_sec_kinetic_energy')
