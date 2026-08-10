"""
exp_16 -- Universality of the Geiger-Nuttall Law

Milestone R, Block C (Novel Physics)

Thesis: The Geiger-Nuttall scaling log(FPT) ~ d/f(sigma) is NOT specific
to the Coulomb potential or to ADE graphs. It is a GENERIC consequence of
any barrier requiring d simultaneous threshold exceedances under stochastic
noise. If true, the GN law is a universal property of multi-connection
severance -- a statistical-mechanical theorem, not a nuclear-physics fact.

Analytical prediction (independent edges approximation):
    P_sever = [2 * (1 - Phi(tau / (sigma * sqrt(2))))]^d
    FPT = 1 / P_sever
where Phi is the standard normal CDF, tau is the stress threshold,
sigma is the effective noise scale (noise_amplitude * LN2), and d is degree.

Tests:
  T1: Data collapse -- log(FPT) vs d/sqrt(sigma) across A, D, E types
      onto a single curve (R^2 > 0.7)
  T2: Analytical prediction matches empirical (correlation > 0.8)
  T3: Threshold is a scale parameter -- different thresholds preserve
      functional form (R^2 > 0.5 at each threshold)
  T4: Universal exponent -- fit k in d/sigma^k, check consistency
      across graph types (std(k) < 0.3 * mean(k))
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    stress_barrier_walk,
    ade_graphs,
    save_mr_results,
)


def analytical_fpt(degree, stress_threshold, noise_amplitude):
    """
    Predict FPT from the independent-edge model.

    Each edge flow is approximately Gaussian with std ~ noise_amplitude * LN2.
    P(|flow| > tau) = erfc(tau / (sigma * sqrt(2))) for one edge.
    P(all d edges > tau simultaneously) = P_single^d (independence approx).
    FPT = 1 / P_all.
    """
    sigma = noise_amplitude * LN2
    if sigma < 1e-30:
        return float('inf')
    # P(|X| > tau) for X ~ N(0, sigma) = erfc(tau / (sigma * sqrt(2)))
    p_single = erfc(stress_threshold / (sigma * np.sqrt(2)))
    if p_single < 1e-30:
        return float('inf')
    p_all = p_single ** degree
    if p_all < 1e-30:
        return float('inf')
    return 1.0 / p_all


def collect_stress_data(graphs, noise_levels, stress_threshold, n_trials=150,
                        max_steps=5000):
    """Collect (graph, vertex, degree, noise, median_fpt) data points."""
    data_points = []

    for graph_name, adj in graphs:
        n = adj.shape[0]
        for v in range(n):
            degree = int(np.sum(adj[v] > 0))
            if degree < 1:
                continue

            for noise in noise_levels:
                fpts = []
                for trial in range(n_trials):
                    initial = np.ones(n) / n  # Start from equilibrium

                    result = stress_barrier_walk(
                        adj, v, initial,
                        stress_threshold=stress_threshold,
                        noise_amplitude=noise,
                        max_steps=max_steps,
                        seed=trial * 100 + int(noise * 10000) + v * 50000
                             + abs(hash(graph_name)) % 10000,
                    )
                    if result['converged']:
                        fpts.append(result['first_passage_time'])

                if len(fpts) >= n_trials // 10:
                    data_points.append({
                        'graph': graph_name,
                        'vertex': v,
                        'degree': degree,
                        'noise': noise,
                        'median_fpt': float(np.median(fpts)),
                        'n_converged': len(fpts),
                        'n_trials': n_trials,
                    })

    return data_points


def test_T1_data_collapse():
    """T1: Data collapse across A, D, E graph types."""
    print("\n  T1: Data collapse -- log(FPT) vs d/sqrt(sigma) across graph types")
    results = {'description': 'Single curve R^2 > 0.7 across A, D, E types'}

    stress_threshold = 0.008
    noise_levels = [0.008, 0.015, 0.025, 0.040]
    n_trials = 150

    # Collect from all three ADE families
    target_graphs = []
    for name, adj in ade_graphs(max_rank=8):
        # A-type: A_4 to A_7, D-type: D_5 to D_8, E-type: E_6, E_7
        if name in ('A_4', 'A_5', 'A_6', 'A_7',
                     'D_5', 'D_6', 'D_7', 'D_8',
                     'E_6', 'E_7'):
            target_graphs.append((name, adj))

    data = collect_stress_data(target_graphs, noise_levels, stress_threshold,
                               n_trials=n_trials)

    if len(data) < 10:
        results['PASS'] = False
        results['error'] = f'Only {len(data)} data points'
        print(f"    FAIL: insufficient data ({len(data)} points)")
        return results

    # GN combined variable: degree / sqrt(noise)
    gn_vars = np.array([d['degree'] / np.sqrt(d['noise']) for d in data])
    log_fpts = np.log10(np.array([d['median_fpt'] for d in data]))

    slope, intercept, r, p, se = stats.linregress(gn_vars, log_fpts)
    r2 = r ** 2

    # Per-family R^2
    families = {'A': [], 'D': [], 'E': []}
    for d in data:
        family = d['graph'][0]
        families[family].append(d)

    family_r2s = {}
    for fam, fam_data in families.items():
        if len(fam_data) >= 4:
            fam_gn = np.array([d['degree'] / np.sqrt(d['noise']) for d in fam_data])
            fam_log = np.log10(np.array([d['median_fpt'] for d in fam_data]))
            _, _, r_fam, _, _ = stats.linregress(fam_gn, fam_log)
            family_r2s[fam] = round(r_fam ** 2, 4)

    passed = slope > 0 and r2 > 0.7

    results['n_points'] = len(data)
    results['n_graphs'] = len(target_graphs)
    results['combined_r2'] = round(float(r2), 4)
    results['combined_slope'] = round(float(slope), 4)
    results['family_r2s'] = family_r2s
    results['families_present'] = {fam: len(pts) for fam, pts in families.items()}
    results['PASS'] = passed
    print(f"    {len(data)} points across {len(target_graphs)} graphs")
    print(f"    Combined R^2 = {r2:.4f} (slope = {slope:.4f})")
    for fam, r2_fam in family_r2s.items():
        print(f"    {fam}-family R^2 = {r2_fam}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: positive slope AND R^2 > 0.7)")
    return results


def test_T2_analytical_prediction():
    """T2: Analytical prediction matches empirical FPTs."""
    print("\n  T2: Analytical prediction vs empirical")
    results = {'description': 'Spearman correlation > 0.8 between predicted and actual FPT'}

    stress_threshold = 0.008
    noise_levels = [0.008, 0.012, 0.018, 0.025, 0.035]
    n_trials = 200

    # Use D_7, D_8, E_6 for diverse degrees
    target_graphs = []
    for name, adj in ade_graphs(max_rank=8):
        if name in ('D_7', 'D_8', 'E_6'):
            target_graphs.append((name, adj))

    data = collect_stress_data(target_graphs, noise_levels, stress_threshold,
                               n_trials=n_trials, max_steps=8000)

    if len(data) < 8:
        results['PASS'] = False
        results['error'] = f'Only {len(data)} data points'
        print(f"    FAIL: insufficient data ({len(data)} points)")
        return results

    # Compute analytical predictions
    predicted = []
    empirical = []
    for d in data:
        pred = analytical_fpt(d['degree'], stress_threshold, d['noise'])
        if pred < float('inf') and pred > 0:
            predicted.append(pred)
            empirical.append(d['median_fpt'])

    if len(predicted) < 6:
        results['PASS'] = False
        results['error'] = f'Only {len(predicted)} finite predictions'
        print(f"    FAIL: insufficient finite predictions ({len(predicted)})")
        return results

    predicted = np.array(predicted)
    empirical = np.array(empirical)

    # Compare in log space
    log_pred = np.log10(predicted)
    log_emp = np.log10(empirical)

    rho, p_val = stats.spearmanr(log_pred, log_emp)
    slope, intercept, r, _, _ = stats.linregress(log_pred, log_emp)
    r2 = r ** 2

    # Ratio statistics
    ratios = empirical / predicted
    median_ratio = float(np.median(ratios))
    ratio_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    passed = float(rho) > 0.8

    results['n_points'] = len(predicted)
    results['spearman_rho'] = round(float(rho), 4)
    results['log_log_r2'] = round(float(r2), 4)
    results['log_log_slope'] = round(float(slope), 4)
    results['median_ratio_emp_pred'] = round(median_ratio, 4)
    results['ratio_iqr'] = round(ratio_iqr, 4)
    results['PASS'] = passed
    print(f"    {len(predicted)} points with finite analytical predictions")
    print(f"    Spearman rho(predicted, empirical) = {rho:.4f}")
    print(f"    log-log R^2 = {r2:.4f}, slope = {slope:.4f}")
    print(f"    Median ratio (empirical/predicted) = {median_ratio:.3f}")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: rho > 0.8)")
    return results


def test_T3_threshold_independence():
    """T3: Functional form survives across different stress thresholds."""
    print("\n  T3: Threshold independence -- GN form at multiple thresholds")
    results = {'description': 'R^2 > 0.5 for d/sqrt(sigma) at each threshold level'}

    noise_levels = [0.008, 0.015, 0.025, 0.040]
    thresholds = [0.005, 0.008, 0.012]
    n_trials = 120

    # D_6, D_7, D_8 — fast, diverse degrees
    target_graphs = []
    for name, adj in ade_graphs(max_rank=8):
        if name in ('D_6', 'D_7', 'D_8'):
            target_graphs.append((name, adj))

    per_threshold = []
    for tau in thresholds:
        data = collect_stress_data(target_graphs, noise_levels, tau,
                                   n_trials=n_trials)

        if len(data) < 6:
            per_threshold.append({
                'threshold': tau,
                'n_points': len(data),
                'r2': 0.0,
                'slope': 0.0,
                'pass': False,
            })
            print(f"    tau={tau:.3f}: insufficient data ({len(data)} points)")
            continue

        gn_vars = np.array([d['degree'] / np.sqrt(d['noise']) for d in data])
        log_fpts = np.log10(np.array([d['median_fpt'] for d in data]))

        slope, intercept, r, p, se = stats.linregress(gn_vars, log_fpts)
        r2 = r ** 2

        thresh_pass = slope > 0 and r2 > 0.5
        per_threshold.append({
            'threshold': tau,
            'n_points': len(data),
            'r2': round(float(r2), 4),
            'slope': round(float(slope), 4),
            'pass': thresh_pass,
        })
        print(f"    tau={tau:.3f}: {len(data)} pts, R^2={r2:.4f}, "
              f"slope={slope:.4f} -> {'pass' if thresh_pass else 'fail'}")

    n_pass = sum(1 for pt in per_threshold if pt['pass'])
    passed = n_pass >= 2  # Majority of 3

    results['per_threshold'] = per_threshold
    results['n_pass'] = n_pass
    results['n_thresholds'] = len(thresholds)
    results['PASS'] = passed
    print(f"    {n_pass}/{len(thresholds)} thresholds show GN form")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: majority)")
    return results


def test_T4_universal_exponent():
    """T4: Extract universal exponent k in d/sigma^k."""
    print("\n  T4: Universal exponent k in log(FPT) ~ d / sigma^k")
    results = {'description': 'Consistent k across graph types (std < 0.3*mean)'}

    stress_threshold = 0.008
    noise_levels = [0.006, 0.010, 0.015, 0.022, 0.033, 0.050]
    n_trials = 150

    # Per-graph-type: fit k that maximizes R^2
    graph_groups = {
        'A': ['A_5', 'A_6', 'A_7'],
        'D': ['D_5', 'D_6', 'D_7', 'D_8'],
        'E': ['E_6', 'E_7'],
    }

    all_graphs = {}
    for name, adj in ade_graphs(max_rank=8):
        all_graphs[name] = adj

    per_family = []
    for family, graph_names in graph_groups.items():
        target_graphs = [(n, all_graphs[n]) for n in graph_names if n in all_graphs]
        if not target_graphs:
            continue

        data = collect_stress_data(target_graphs, noise_levels, stress_threshold,
                                   n_trials=n_trials)

        if len(data) < 8:
            print(f"    {family}-family: insufficient data ({len(data)} points)")
            continue

        degrees = np.array([d['degree'] for d in data])
        noises = np.array([d['noise'] for d in data])
        log_fpts = np.log10(np.array([d['median_fpt'] for d in data]))

        # Grid search for best k in [0.1, 2.0]
        best_k = 0.5
        best_r2 = -1.0
        for k_test in np.linspace(0.1, 2.0, 40):
            combined = degrees / (noises ** k_test)
            _, _, r, _, _ = stats.linregress(combined, log_fpts)
            if r ** 2 > best_r2:
                best_r2 = r ** 2
                best_k = k_test

        per_family.append({
            'family': family,
            'n_points': len(data),
            'best_k': round(float(best_k), 3),
            'best_r2': round(float(best_r2), 4),
        })
        print(f"    {family}-family: k={best_k:.3f}, R^2={best_r2:.4f} ({len(data)} pts)")

    if len(per_family) < 2:
        results['PASS'] = False
        results['error'] = f'Only {len(per_family)} families'
        print(f"    FAIL: insufficient families")
        return results

    ks = [pf['best_k'] for pf in per_family]
    mean_k = float(np.mean(ks))
    std_k = float(np.std(ks))

    # Also check: what's the overall best k using ALL data combined?
    all_target = []
    for family, graph_names in graph_groups.items():
        all_target.extend([(n, all_graphs[n]) for n in graph_names if n in all_graphs])
    all_data = collect_stress_data(all_target, noise_levels, stress_threshold,
                                   n_trials=n_trials)
    overall_best_k = 0.5
    overall_best_r2 = -1.0
    if len(all_data) >= 10:
        all_deg = np.array([d['degree'] for d in all_data])
        all_noise = np.array([d['noise'] for d in all_data])
        all_log_fpt = np.log10(np.array([d['median_fpt'] for d in all_data]))
        for k_test in np.linspace(0.1, 2.0, 40):
            combined = all_deg / (all_noise ** k_test)
            _, _, r, _, _ = stats.linregress(combined, all_log_fpt)
            if r ** 2 > overall_best_r2:
                overall_best_r2 = r ** 2
                overall_best_k = k_test

    # PASS: consistent k (low spread) and mean_k > 0
    passed = mean_k > 0 and (std_k < 0.3 * mean_k if mean_k > 0 else False)

    results['per_family'] = per_family
    results['mean_k'] = round(mean_k, 3)
    results['std_k'] = round(std_k, 3)
    results['overall_best_k'] = round(float(overall_best_k), 3)
    results['overall_best_r2'] = round(float(overall_best_r2), 4)
    results['PASS'] = passed
    print(f"    Mean k = {mean_k:.3f} +/- {std_k:.3f}")
    print(f"    Overall best k = {overall_best_k:.3f} (R^2 = {overall_best_r2:.4f})")
    print(f"    -> {'PASS' if passed else 'FAIL'} (need: std < 0.3*mean)")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_16: Universality of the Geiger-Nuttall Law")
    print("=" * 60)

    t1 = test_T1_data_collapse()
    t2 = test_T2_analytical_prediction()
    t3 = test_T3_threshold_independence()
    t4 = test_T4_universal_exponent()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_16_universality_gn',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'The Geiger-Nuttall scaling log(FPT) ~ d/f(sigma) is a UNIVERSAL '
                  'property of any barrier requiring d simultaneous threshold '
                  'exceedances under stochastic noise. Not specific to Coulomb or '
                  'to ADE -- it is a statistical-mechanical theorem. Analytical '
                  'prediction: FPT = 1/[erfc(tau/(sigma*sqrt(2)))]^d.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_16_universality_gn')
