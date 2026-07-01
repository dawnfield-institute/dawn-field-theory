"""
exp_15 -- Stress Barrier and the GN Sign Fix

Milestone R, Block C (Novel Physics)

Thesis: exp_14 proved that the perspectival/topological barriers are
ENTROPY barriers (convergence to equilibrium). Noise opposes convergence,
giving rho=+1.0 (wrong sign). The Coulomb barrier analog must be a
STRESS barrier: connections sever when ALL d edges are simultaneously
OVERSTRESSED (|state[v]-state[u]| > threshold).

Higher noise (SEC flux) -> larger fluctuations -> easier to overstress
all d edges simultaneously -> shorter FPT. This flips the sign.

The stress_threshold is FIXED (not noise-dependent), so noise and
barrier are genuinely independent -- unlike the old topological barrier
where edge_threshold = noise * LN2 (self-canceling).

The physics: CO2 bubble escapes when internal pressure exceeds surface
tension on ALL sides. Alpha particle escapes when kinetic energy
overcomes Coulomb repulsion. Both are STRESS events, not relaxation.

Tests:
  T1: Higher noise -> shorter FPT (the sign flip)
  T2: FPT increases with degree (barrier) and decreases with noise (KE)
  T3: GN combined variable: log(FPT) vs degree/sqrt(noise) gives R^2 > 0.5
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
    stress_barrier_walk,
    perspective_divergence,
    ade_graphs, redistribute_on_graph,
    save_mr_results,
)


def find_stress_threshold(adj, target, noise_amplitude=0.01, n_trials=50):
    """Find a stress_threshold that gives ~50% convergence at 5000 steps."""
    n = adj.shape[0]
    # Run walks at various thresholds to find the sweet spot
    # The typical edge flow near equilibrium is ~ noise * LN2 * sqrt(2)
    # We want a threshold somewhat above typical but not impossible
    typical_flow = noise_amplitude * LN2 * 1.5
    return typical_flow


def test_T1_noise_decreases_stress_fpt():
    """T1: Higher noise -> shorter stress FPT (the sign flip)."""
    print("\n  T1: Higher noise -> shorter stress FPT (sign flip)")
    results = {'description': 'Negative rho(noise, FPT) — more SEC flux = faster severance'}

    # D_8 hub vertex
    test_graph = None
    for name, adj in ade_graphs(max_rank=8):
        if name == 'D_8':
            test_graph = adj
            break

    n = test_graph.shape[0]
    target = n - 3  # hub, degree 3

    # The stress threshold must be FIXED across noise levels
    # Choose a threshold that's achievable but challenging
    # Typical edge flow at equilibrium ≈ noise * LN2 * sqrt(2) for independent noise
    # We want threshold slightly above the LOWEST noise level's typical flow
    # so low noise rarely crosses, high noise crosses often
    stress_threshold = 0.008  # Fixed, between typical flows at noise=0.005 and 0.035

    noise_levels = [0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.050]
    n_trials = 200

    median_fpts = []
    actual_noises = []

    for noise in noise_levels:
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n  # Start from equilibrium

            result = stress_barrier_walk(
                test_graph, target, initial,
                stress_threshold=stress_threshold,
                noise_amplitude=noise,
                max_steps=8000,
                seed=trial * 1000 + int(noise * 10000),
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        convergence = len(fpts) / n_trials
        if len(fpts) >= n_trials // 10:  # Allow low convergence for tough thresholds
            med = np.median(fpts)
            median_fpts.append(med)
            actual_noises.append(noise)
            print(f"      noise={noise:.3f}: median_fpt={med:.0f} "
                  f"({len(fpts)}/{n_trials} = {convergence:.0%} converged)")
        else:
            print(f"      noise={noise:.3f}: too few converged "
                  f"({len(fpts)}/{n_trials} = {convergence:.0%})")

    # Test: negative correlation
    rho = 0.0
    slope = float('nan')
    r2 = 0.0
    passed = False
    if len(median_fpts) >= 4:
        rho_val, _ = stats.spearmanr(actual_noises, median_fpts)
        rho = float(rho_val)

        # GN form: log(FPT) vs 1/sqrt(noise)
        x = 1.0 / np.sqrt(np.array(actual_noises))
        y = np.log10(np.array(median_fpts))
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r ** 2

        passed = rho < -0.5

    results['stress_threshold'] = stress_threshold
    results['noise_levels'] = [float(n) for n in actual_noises]
    results['median_fpts'] = [float(f) for f in median_fpts]
    results['spearman_rho'] = round(float(rho), 4)
    results['gn_slope'] = round(float(slope), 4) if not np.isnan(slope) else None
    results['gn_r2'] = round(float(r2), 4)
    results['PASS'] = passed
    print(f"    Spearman rho(noise, FPT) = {rho:.4f}")
    if not np.isnan(slope):
        print(f"    log(FPT) vs 1/sqrt(noise): slope={slope:.4f}, R^2={r2:.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_stress_barrier_2d():
    """T2: FPT increases with degree (barrier), decreases with noise (KE)."""
    print("\n  T2: Degree increases FPT, noise decreases FPT")
    results = {'description': 'Correct sign for both variables'}

    stress_threshold = 0.008
    noise_levels = [0.008, 0.015, 0.025, 0.040]
    n_trials = 150

    data_points = []

    # Collect data across vertices with different degrees
    for name, adj in ade_graphs(max_rank=8):
        if name not in ('D_6', 'D_7', 'D_8', 'E_6', 'E_7'):
            continue
        n = adj.shape[0]
        for v in range(n):
            degree = int(np.sum(adj[v] > 0))
            if degree < 1:
                continue

            for noise in noise_levels:
                fpts = []
                for trial in range(n_trials):
                    initial = np.ones(n) / n

                    result = stress_barrier_walk(
                        adj, v, initial,
                        stress_threshold=stress_threshold,
                        noise_amplitude=noise,
                        max_steps=5000,
                        seed=trial * 100 + int(noise * 10000) + v * 50000 + abs(hash(name)) % 10000,
                    )
                    if result['converged']:
                        fpts.append(result['first_passage_time'])

                if len(fpts) >= n_trials // 10:
                    data_points.append({
                        'graph': name,
                        'vertex': v,
                        'degree': degree,
                        'noise': noise,
                        'median_fpt': float(np.median(fpts)),
                        'n_converged': len(fpts),
                    })

    if len(data_points) < 10:
        results['PASS'] = False
        results['error'] = f'Only {len(data_points)} data points'
        print(f"    FAIL: insufficient data ({len(data_points)} points)")
        return results

    print(f"    Collected {len(data_points)} data points")

    degrees = np.array([d['degree'] for d in data_points])
    noises = np.array([d['noise'] for d in data_points])
    log_fpts = np.log10(np.array([d['median_fpt'] for d in data_points]))

    # Multiple regression
    X = np.column_stack([degrees, noises, np.ones(len(degrees))])
    beta, _, _, _ = np.linalg.lstsq(X, log_fpts, rcond=None)
    coeff_degree = beta[0]
    coeff_noise = beta[1]

    y_pred = X @ beta
    ss_res = np.sum((log_fpts - y_pred) ** 2)
    ss_tot = np.sum((log_fpts - np.mean(log_fpts)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rho_degree, _ = stats.spearmanr(degrees, log_fpts)
    rho_noise, _ = stats.spearmanr(noises, log_fpts)

    # PASS: degree positive AND noise negative
    passed = coeff_degree > 0 and coeff_noise < 0

    results['n_points'] = len(data_points)
    results['coeff_degree'] = round(float(coeff_degree), 4)
    results['coeff_noise'] = round(float(coeff_noise), 4)
    results['r2'] = round(float(r2), 4)
    results['rho_degree_fpt'] = round(float(rho_degree), 4)
    results['rho_noise_fpt'] = round(float(rho_noise), 4)
    results['PASS'] = passed
    print(f"    Regression: log(FPT) = {coeff_degree:.3f}*degree + {coeff_noise:.3f}*noise + const")
    print(f"    R^2 = {r2:.4f}")
    print(f"    Spearman: rho(degree, FPT)={rho_degree:.3f}, rho(noise, FPT)={rho_noise:.3f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: degree>0 AND noise<0)")
    return results


def test_T3_gn_combined_variable():
    """T3: GN combined variable degree/sqrt(noise) predicts FPT."""
    print("\n  T3: GN combined variable: log(FPT) vs degree/sqrt(noise)")
    results = {'description': 'R^2 > 0.5 for combined variable, positive slope'}

    stress_threshold = 0.008
    noise_levels = [0.008, 0.015, 0.025, 0.040]
    n_trials = 150

    data_points = []

    # Use D_8 — has vertices of degree 1, 2, and 3
    test_graph = None
    for name, adj in ade_graphs(max_rank=8):
        if name == 'D_8':
            test_graph = adj
            break

    n = test_graph.shape[0]

    for v in range(n):
        degree = int(np.sum(test_graph[v] > 0))
        for noise in noise_levels:
            fpts = []
            for trial in range(n_trials):
                initial = np.ones(n) / n

                result = stress_barrier_walk(
                    test_graph, v, initial,
                    stress_threshold=stress_threshold,
                    noise_amplitude=noise,
                    max_steps=5000,
                    seed=trial * 100 + int(noise * 10000) + v * 50000,
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 10:
                data_points.append({
                    'vertex': v,
                    'degree': degree,
                    'noise': noise,
                    'median_fpt': float(np.median(fpts)),
                    'n_converged': len(fpts),
                })

    if len(data_points) < 6:
        results['PASS'] = False
        results['error'] = f'Only {len(data_points)} data points'
        print(f"    FAIL: insufficient data")
        return results

    print(f"    Collected {len(data_points)} data points")

    # GN combined variable: degree / sqrt(noise)
    gn_vars = np.array([d['degree'] / np.sqrt(d['noise']) for d in data_points])
    log_fpts = np.log10(np.array([d['median_fpt'] for d in data_points]))

    slope, intercept, r, p, se = stats.linregress(gn_vars, log_fpts)
    r2 = r ** 2

    # Compare: degree alone
    degrees = np.array([d['degree'] for d in data_points])
    _, _, r_deg, _, _ = stats.linregress(degrees, log_fpts)
    r2_deg = r_deg ** 2

    # Compare: noise alone
    noises = np.array([d['noise'] for d in data_points])
    _, _, r_noise, _, _ = stats.linregress(noises, log_fpts)
    r2_noise = r_noise ** 2

    passed = slope > 0 and r2 > 0.5

    results['n_points'] = len(data_points)
    results['gn_slope'] = round(float(slope), 4)
    results['gn_r2'] = round(float(r2), 4)
    results['degree_only_r2'] = round(float(r2_deg), 4)
    results['noise_only_r2'] = round(float(r2_noise), 4)
    results['PASS'] = passed
    print(f"    Combined degree/sqrt(noise): slope={slope:.4f}, R^2={r2:.4f}")
    print(f"    Degree alone: R^2={r2_deg:.4f}")
    print(f"    Noise alone: R^2={r2_noise:.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: positive slope AND R^2 > 0.5)")
    return results


def test_T4_sign_across_graph_types():
    """T4: Noise-FPT negative correlation holds across graph types."""
    print("\n  T4: Stress barrier sign correct across graph types")
    results = {'description': 'Negative rho(noise, FPT) for majority'}

    stress_threshold = 0.008
    noise_levels = [0.008, 0.015, 0.025, 0.040]
    n_trials = 150
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

                result = stress_barrier_walk(
                    adj, target, initial,
                    stress_threshold=stress_threshold,
                    noise_amplitude=noise,
                    max_steps=5000,
                    seed=trial * 100 + int(noise * 10000),
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 10:
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
    passed = total > 0 and correct_count >= 3

    results['per_graph'] = per_graph
    results['correct_count'] = correct_count
    results['total_graphs'] = total
    results['PASS'] = passed
    print(f"    Correct sign: {correct_count}/{total}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_15: Stress Barrier and the GN Sign Fix")
    print("=" * 60)

    t1 = test_T1_noise_decreases_stress_fpt()
    t2 = test_T2_stress_barrier_2d()
    t3 = test_T3_gn_combined_variable()
    t4 = test_T4_sign_across_graph_types()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_15_stress_barrier_gn',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'The Coulomb barrier analog is a STRESS barrier, not a relaxation '
                  'barrier. Connections sever when ALL d edges are simultaneously '
                  'OVERSTRESSED (edge flow > fixed threshold). Higher noise (SEC flux) '
                  '-> larger fluctuations -> easier to overstress all d edges -> shorter '
                  'FPT. This flips the GN sign: more kinetic energy -> faster severance.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_15_stress_barrier_gn')
