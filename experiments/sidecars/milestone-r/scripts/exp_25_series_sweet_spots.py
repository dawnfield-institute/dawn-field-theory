"""
exp_25 -- Per-Series Convergence Sweet Spots

Milestone R, Block D (Consolidation)

From exp_20 and exp_23: Laplacian inverse eigenvalues of A_n path graphs
converge to the hydrogen 1/k^2 pattern, but convergence is per-series.
Lyman (m=1) converges fastest, Paschen (m=3) slowest. Each series has a
"sweet spot" -- the minimum graph rank where it achieves target precision.

This resolves 6-8 series-selective and resolution-limit failures by showing
the failures occurred below each series' sweet spot rank.

Tests:
  T1: Sweet spot ordering -- Lyman < Balmer < Paschen
  T2: Sweet spot precision -- alpha transition < 3% at sweet spot
  T3: Exponent predicts sweet spot -- log(n_sweet) vs 1/p_m, R^2 > 0.8
  T4: Branch departure at sweet spots -- ordered by series sensitivity
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, save_mr_results,
)


def path_graph(n):
    """Build adjacency matrix for A_n path graph."""
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def dynkin_d(n):
    """Build adjacency matrix for D_n graph (n >= 4)."""
    adj = np.zeros((n, n), dtype=float)
    # Chain: 0-1-2-..-(n-3)
    for i in range(n - 3):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    # Branch: (n-3) connects to both (n-2) and (n-1)
    hub = n - 3
    adj[hub, n - 2] = 1.0
    adj[n - 2, hub] = 1.0
    adj[hub, n - 1] = 1.0
    adj[n - 1, hub] = 1.0
    return adj


def laplacian_energy_levels(adj):
    """Compute hydrogen-like energy levels from Laplacian inverse eigenvalues."""
    degrees = np.sum(adj > 0, axis=1).astype(float)
    L = np.diag(degrees) - adj.astype(float)
    eigvals = np.linalg.eigvalsh(L)
    pos = eigvals[eigvals > 1e-10]
    if len(pos) == 0:
        return np.array([])
    return np.sort(1.0 / pos)[::-1]  # E_1 > E_2 > ...


def hydrogen_ratio(k, kp, m):
    """Hydrogen transition energy ratio: dE(k->m) / dE(kp->m)."""
    de_k = 1.0 / m ** 2 - 1.0 / k ** 2
    de_kp = 1.0 / m ** 2 - 1.0 / kp ** 2
    if abs(de_kp) < 1e-15:
        return float('inf')
    return de_k / de_kp


def series_mean_error(E, series_m):
    """Mean relative error of consecutive transition ratios in series m."""
    m = series_m
    errors = []
    max_k = len(E)
    for k in range(m + 1, max_k - 1):
        kp = k + 1
        if kp >= max_k or m >= max_k:
            break
        # Graph ratio: (E_m - E_k) / (E_m - E_kp)
        de_graph_k = E[m - 1] - E[k - 1] if m - 1 < len(E) and k - 1 < len(E) else None
        de_graph_kp = E[m - 1] - E[kp - 1] if m - 1 < len(E) and kp - 1 < len(E) else None
        if de_graph_k is None or de_graph_kp is None:
            continue
        if abs(de_graph_kp) < 1e-15:
            continue
        graph_r = de_graph_k / de_graph_kp
        hydro_r = hydrogen_ratio(k, kp, m)
        if abs(hydro_r) < 1e-15 or hydro_r == float('inf'):
            continue
        errors.append(abs(graph_r - hydro_r) / abs(hydro_r))
    return np.mean(errors) if errors else float('inf')


def alpha_transition_error(E, series_m):
    """Error of the first (alpha) transition ratio in series m."""
    m = series_m
    k1 = m + 1
    k2 = m + 2
    if k2 > len(E):
        return float('inf')
    de1 = E[m - 1] - E[k1 - 1]
    de2 = E[m - 1] - E[k2 - 1]
    if abs(de2) < 1e-15:
        return float('inf')
    graph_r = de1 / de2
    hydro_r = hydrogen_ratio(k1, k2, m)
    if hydro_r == float('inf'):
        return float('inf')
    return abs(graph_r - hydro_r) / abs(hydro_r)


def test_T1_sweet_spot_ordering():
    """T1: Sweet spot rank increases with series index."""
    print("\n  T1: Sweet spot ordering (Lyman < Balmer < Paschen)")
    results = {'description': 'n_sweet(m=1) < n_sweet(m=2) < n_sweet(m=3)'}

    tolerance = 0.05  # 5% mean error threshold
    max_n = 50
    series_labels = ['Lyman', 'Balmer', 'Paschen']
    sweet_spots = {}

    for m_idx, m in enumerate([1, 2, 3]):
        found = None
        for n in range(2 * m + 2, max_n + 1):
            E = laplacian_energy_levels(path_graph(n))
            err = series_mean_error(E, m)
            if err < tolerance:
                found = n
                break
        sweet_spots[m] = found
        label = series_labels[m_idx]
        if found:
            print(f"    {label} (m={m}): sweet spot at A_{found}")
        else:
            print(f"    {label} (m={m}): not converged by n={max_n}")

    ss_vals = [sweet_spots[m] for m in [1, 2, 3]]
    all_found = all(v is not None for v in ss_vals)

    if all_found:
        ordered = ss_vals[0] < ss_vals[1] < ss_vals[2]
    else:
        ordered = False

    passed = all_found and ordered

    results['sweet_spots'] = {f'm={m}': sweet_spots[m] for m in [1, 2, 3]}
    results['all_found'] = all_found
    results['strictly_ordered'] = ordered if all_found else False
    results['PASS'] = passed
    print(f"    Ordered: {ordered if all_found else 'N/A (missing)'} -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_sweet_spot_precision():
    """T2: Alpha transition error < 3% at each sweet spot."""
    print("\n  T2: Alpha transition precision at sweet spots")
    results = {'description': 'Alpha transition error < 3% at sweet spot for all 3 series'}

    tolerance = 0.05
    max_n = 50
    series_labels = ['Lyman', 'Balmer', 'Paschen']
    details = []
    all_precise = True

    for m_idx, m in enumerate([1, 2, 3]):
        # Find sweet spot
        sweet_n = None
        for n in range(2 * m + 2, max_n + 1):
            E = laplacian_energy_levels(path_graph(n))
            if series_mean_error(E, m) < tolerance:
                sweet_n = n
                break

        if sweet_n is None:
            details.append({
                'series': series_labels[m_idx],
                'm': m,
                'sweet_n': None,
                'alpha_error': None,
                'precise': False,
            })
            all_precise = False
            print(f"    {series_labels[m_idx]}: no sweet spot found")
            continue

        E = laplacian_energy_levels(path_graph(sweet_n))
        alpha_err = alpha_transition_error(E, m)
        precise = alpha_err < 0.03

        if not precise:
            all_precise = False

        details.append({
            'series': series_labels[m_idx],
            'm': m,
            'sweet_n': sweet_n,
            'alpha_error': float(alpha_err),
            'precise': precise,
        })
        print(f"    {series_labels[m_idx]} at A_{sweet_n}: alpha error = {alpha_err:.4f} "
              f"({'< 3%' if precise else '>= 3%'})")

    passed = all_precise
    results['details'] = details
    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_exponent_predicts_sweet_spot():
    """T3: Convergence exponents predict sweet spots via log(n_sweet) ~ 1/p_m."""
    print("\n  T3: Convergence exponents predict sweet spots")
    results = {'description': 'R^2 > 0.8 for log(n_sweet) vs 1/p_m'}

    tolerance = 0.05
    max_n = 50
    series_labels = ['Lyman', 'Balmer', 'Paschen']

    # Compute convergence exponents (error ~ n^(-p))
    exponents = {}
    sweet_spots = {}

    for m_idx, m in enumerate([1, 2, 3]):
        # Sweet spot
        sweet_n = None
        for n in range(2 * m + 2, max_n + 1):
            E = laplacian_energy_levels(path_graph(n))
            if series_mean_error(E, m) < tolerance:
                sweet_n = n
                break
        sweet_spots[m] = sweet_n

        # Convergence exponent: fit log(error) vs log(n) for n = 2m+2 .. min(sweet_n+10, 30)
        ns = []
        errs = []
        for n in range(2 * m + 2, min(35, max_n + 1)):
            E = laplacian_energy_levels(path_graph(n))
            err = series_mean_error(E, m)
            if 0 < err < float('inf'):
                ns.append(n)
                errs.append(err)

        if len(ns) >= 3:
            log_n = np.log(ns)
            log_e = np.log(errs)
            slope, intercept, r_val, p_val, std_err = stats.linregress(log_n, log_e)
            exponents[m] = -slope  # p = -slope (error decreases with n)
        else:
            exponents[m] = None

        print(f"    {series_labels[m_idx]}: p={exponents[m]:.3f}" if exponents[m] else
              f"    {series_labels[m_idx]}: insufficient data")

    # Fit log(n_sweet) vs 1/p_m
    valid = [(m, sweet_spots[m], exponents[m])
             for m in [1, 2, 3]
             if sweet_spots[m] is not None and exponents[m] is not None and exponents[m] > 0]

    if len(valid) >= 3:
        inv_p = [1.0 / v[2] for v in valid]
        log_ns = [np.log(v[1]) for v in valid]
        slope, intercept, r_val, p_val, std_err = stats.linregress(inv_p, log_ns)
        r2 = r_val ** 2
        passed = r2 > 0.8
        print(f"    Fit: log(n_sweet) = {slope:.3f}/p + {intercept:.3f}, R^2 = {r2:.4f}")
    else:
        r2 = 0.0
        passed = False
        print(f"    Insufficient valid data points ({len(valid)}/3)")

    results['exponents'] = {f'm={m}': float(exponents[m]) if exponents[m] else None for m in [1, 2, 3]}
    results['sweet_spots'] = {f'm={m}': sweet_spots[m] for m in [1, 2, 3]}
    results['r_squared'] = float(r2) if r2 else 0.0
    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_branch_departure_scaling():
    """T4: Branch departure at sweet spots ordered by series sensitivity."""
    print("\n  T4: Branch departure scaling at sweet spots")
    results = {'description': 'D_n departure ordered: Paschen > Balmer > Lyman for >= 3/4 sizes'}

    tolerance = 0.05
    max_n = 50
    series_labels = ['Lyman', 'Balmer', 'Paschen']

    # Find sweet spots
    sweet_spots = {}
    for m in [1, 2, 3]:
        for n in range(2 * m + 2, max_n + 1):
            E = laplacian_energy_levels(path_graph(n))
            if series_mean_error(E, m) < tolerance:
                sweet_spots[m] = n
                break
        if m not in sweet_spots:
            sweet_spots[m] = None

    # Test sizes: sweet spots rounded up to valid D_n sizes (>= 4)
    test_sizes = sorted(set(
        max(n, 4) for n in
        [sweet_spots.get(m, 8) or 8 for m in [1, 2, 3]]
        + [8, 10, 12, 15]
    ))
    # Filter to sizes where all 3 series can be tested
    test_sizes = [n for n in test_sizes if n >= 8][:4]

    n_ordered = 0
    details = []

    for n in test_sizes:
        E_path = laplacian_energy_levels(path_graph(n))
        E_branch = laplacian_energy_levels(dynkin_d(n))

        departures = {}
        for m_idx, m in enumerate([1, 2, 3]):
            err_path = series_mean_error(E_path, m)
            err_branch = series_mean_error(E_branch, m)
            departures[m] = err_branch - err_path if err_path < float('inf') and err_branch < float('inf') else float('inf')

        ordered = (departures[3] > departures[2] > departures[1])
        if ordered:
            n_ordered += 1

        details.append({
            'n': n,
            'departures': {f'm={m}': float(departures[m]) for m in [1, 2, 3]},
            'ordered': ordered,
        })
        print(f"    n={n}: Lyman={departures[1]:.4f}, Balmer={departures[2]:.4f}, "
              f"Paschen={departures[3]:.4f} {'ordered' if ordered else ''}")

    passed = n_ordered >= 3 if len(test_sizes) >= 4 else n_ordered >= len(test_sizes) * 0.75

    results['test_sizes'] = test_sizes
    results['n_ordered'] = n_ordered
    results['n_tested'] = len(test_sizes)
    results['details'] = details
    results['PASS'] = bool(passed)
    print(f"    {n_ordered}/{len(test_sizes)} ordered -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_25: Per-Series Convergence Sweet Spots")
    print("=" * 60)

    t1 = test_T1_sweet_spot_ordering()
    t2 = test_T2_sweet_spot_precision()
    t3 = test_T3_exponent_predicts_sweet_spot()
    t4 = test_T4_branch_departure_scaling()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_25_series_sweet_spots',
        'timestamp': datetime.now().isoformat(),
        'block': 'D',
        'thesis': 'Each hydrogen series has a sweet spot rank where Laplacian '
                  'inverse eigenvalues converge. Sweet spots are ordered by '
                  'series index and predicted by convergence exponents.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_25_series_sweet_spots')
