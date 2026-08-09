"""
exp_22 -- The Branching Sweet Spot

Milestone R, Block C (Novel Physics)

exp_21 T4 revealed that at w=alpha_EM, weighted D_n graphs are MORE
hydrogen-like than pure A_n paths for all 5 sizes tested. The A_n->D_n
deformation landscape has a minimum: an optimal branch weight w* where
the graph best approximates hydrogen energy level ratios.

The physics: A_n discretization error in Lyman-alpha ratio is O(n^-4).
A small branch perturbation shifts eigenvalues in a way that partially
cancels this error, creating a w* closer to the 1/k^2 continuum limit
than the unperturbed path. Analogous to how electron correlations
correct pure Coulomb levels in multi-electron atoms.

Key question: does w* follow a clean pattern? If w* ~ n^(-p), the
exponent p reveals how the correction scales with graph size. If p ~ 4,
the sweet spot exactly compensates the O(n^-4) discretization error.

Tests:
  T1: w* exists for D_4..D_12 and follows power law w*(n) ~ c*n^(-p)
      (PASS: R^2 > 0.9, >= 7/9 sizes have sweet spot)
  T2: At w*, improvement spans multiple series (not just Lyman-alpha)
      (PASS: >= 2/3 series improve for >= 4/5 testable sizes)
  T3: E-family sweet spots exist for E_6, E_7, E_8
      (PASS: all 3 have error below A_n baseline at some w*)
  T4: Branch position matters: w*(D_n) vs w*(E_n) differ consistently
      for n=6,7,8 (D branches at n-3, E branches deeper at n-4)
      (PASS: consistent direction for all 3 comparable sizes)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    ade_graphs,
    save_mr_results,
)


def path_graph(n):
    """A_n path graph adjacency matrix."""
    adj = np.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def weighted_dn(n, w):
    """
    D_n with branch edge weight w. Continuous A_n -> D_n deformation.
    At w=0: A_n (path). At w=1: D_n (branch at vertex n-3).
    Interpolation: edge (n-2)-(n-1) weight (1-w), edge (n-3)-(n-1) weight w.
    """
    adj = np.zeros((n, n))
    for i in range(n - 2):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    adj[n - 2, n - 1] = 1.0 - w
    adj[n - 1, n - 2] = 1.0 - w
    adj[n - 3, n - 1] = w
    adj[n - 1, n - 3] = w
    return adj


def weighted_en(n, w):
    """
    E_n with branch edge weight w. Continuous A_n -> E_n deformation.
    At w=0: A_n (path). At w=1: E_n (branch at vertex n-4).
    Requires n >= 6.
    """
    adj = np.zeros((n, n))
    for i in range(n - 2):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    adj[n - 2, n - 1] = 1.0 - w
    adj[n - 1, n - 2] = 1.0 - w
    adj[n - 4, n - 1] = w
    adj[n - 1, n - 4] = w
    return adj


def graph_eigendata(adj):
    """Laplacian eigenvalues and inverse energy levels."""
    weight_sums = np.sum(adj, axis=1)
    L = np.diag(weight_sums) - adj
    eigvals = np.sort(np.linalg.eigvalsh(L))
    pos = eigvals[eigvals > 1e-10]
    E = np.sort(1.0 / pos)[::-1] if len(pos) > 0 else np.array([])
    return pos, E


def hydrogen_ratio(k, kp, m):
    """Hydrogen dE(k->m)/dE(kp->m)."""
    de_k = 1.0 / m**2 - 1.0 / k**2
    de_kp = 1.0 / m**2 - 1.0 / kp**2
    if abs(de_kp) < 1e-15:
        return None
    return de_k / de_kp


def lyman_alpha_error(E):
    """Error of Lyman-alpha ratio dE(2->1)/dE(3->1) vs hydrogen."""
    if len(E) < 3:
        return None
    de_21 = E[0] - E[1]
    de_31 = E[0] - E[2]
    if abs(de_31) < 1e-15:
        return None
    g_r = de_21 / de_31
    h_r = 27.0 / 32.0
    return abs(g_r - h_r) / h_r


def series_errors_by_name(E):
    """Per-series mean ratio errors vs hydrogen. Returns dict."""
    result = {}
    for name, m in [('Lyman', 1), ('Balmer', 2), ('Paschen', 3)]:
        errs = []
        k = m + 1
        while k + 1 <= len(E):
            kp = k + 1
            h_r = hydrogen_ratio(k, kp, m)
            if h_r is not None and max(k, kp) <= len(E):
                de_km = E[m - 1] - E[k - 1]
                de_kpm = E[m - 1] - E[kp - 1]
                if abs(de_kpm) > 1e-15:
                    g_r = de_km / de_kpm
                    errs.append(abs(g_r - h_r) / max(abs(h_r), 1e-15))
            k += 1
        if errs:
            result[name] = float(np.mean(errs))
    return result


def find_sweet_spot(n, family='D'):
    """
    Find optimal branch weight minimizing Lyman-alpha error.
    Two-phase search: log scan then fine refinement.
    Returns (w_star, min_error, baseline_error).
    """
    _, E_base = graph_eigendata(path_graph(n))
    base_err = lyman_alpha_error(E_base)
    if base_err is None:
        return None, None, None

    build = weighted_dn if family == 'D' else weighted_en

    # Phase 1: logarithmic scan from 1e-6 to 0.5
    log_ws = np.logspace(-6, np.log10(0.5), 500)
    best_w, best_err = 0.0, base_err

    for w in log_ws:
        adj = build(n, w)
        _, E = graph_eigendata(adj)
        err = lyman_alpha_error(E)
        if err is not None and err < best_err:
            best_err = err
            best_w = w

    if best_w < 1e-7:
        return 0.0, float(base_err), float(base_err)

    # Phase 2: fine linear scan around the best
    lo = max(1e-7, best_w / 3)
    hi = min(0.5, best_w * 3)
    fine_ws = np.linspace(lo, hi, 1000)

    for w in fine_ws:
        adj = build(n, w)
        _, E = graph_eigendata(adj)
        err = lyman_alpha_error(E)
        if err is not None and err < best_err:
            best_err = err
            best_w = w

    return float(best_w), float(best_err), float(base_err)


def main():
    print("=" * 60)
    print("exp_22: The Branching Sweet Spot")
    print("=" * 60)
    print()
    print("  The A_n->D_n/E_n deformation has an optimal branch weight w*")
    print("  where the graph best approximates hydrogen energy levels.")
    print()

    # ===== T1: Find w* for D_4..D_12 =====
    print("  T1: Sweet spot w* for D_4 through D_12")
    print()

    d_results = {}
    for n in range(4, 13):
        w_star, min_err, base_err = find_sweet_spot(n, family='D')
        has_spot = (w_star is not None and w_star > 1e-7
                    and min_err < base_err)
        if has_spot:
            improvement = (base_err - min_err) / base_err * 100
            d_results[n] = {
                'w_star': w_star,
                'min_error': min_err,
                'base_error': base_err,
                'improvement_pct': improvement,
            }
            print(f"    D_{n:>2}: w*={w_star:.6f}  err={min_err*100:.6f}%"
                  f"  (A_{n}: {base_err*100:.6f}%)"
                  f"  improvement: {improvement:.1f}%")
        else:
            d_results[n] = None
            print(f"    D_{n:>2}: no sweet spot found")

    # Fit power law w*(n) = c * n^(-p)
    valid_ns = [n for n in d_results if d_results[n] is not None]
    n_with_spots = len(valid_ns)

    if len(valid_ns) >= 3:
        ns_arr = np.array(valid_ns, dtype=float)
        ws_arr = np.array([d_results[n]['w_star'] for n in valid_ns])
        log_n = np.log(ns_arr)
        log_w = np.log(ws_arr)
        slope, intercept, r_val, _, _ = stats.linregress(log_n, log_w)
        t1_power = -slope
        t1_c = np.exp(intercept)
        t1_r2 = r_val**2
        print(f"\n    Power law: w* = {t1_c:.4f} * n^(-{t1_power:.2f}),"
              f" R^2={t1_r2:.4f}")
    else:
        t1_power, t1_c, t1_r2 = 0.0, 0.0, 0.0

    t1_pass = t1_r2 > 0.9 and n_with_spots >= 7
    print(f"    Sweet spots found: {n_with_spots}/9")
    print(f"    -> {'PASS' if t1_pass else 'FAIL'}"
          f" (need: R^2 > 0.9, >= 7/9 with sweet spot)")

    # ===== T2: Per-series improvement at w* =====
    print(f"\n  T2: Per-series improvement at optimal w*")

    # Only test sizes with >= 2 computable series (n >= 5)
    testable = [n for n in valid_ns if n >= 5]
    series_data = {}

    for n in testable:
        w_star = d_results[n]['w_star']

        _, E_base = graph_eigendata(path_graph(n))
        base_s = series_errors_by_name(E_base)

        adj_star = weighted_dn(n, w_star)
        _, E_star = graph_eigendata(adj_star)
        star_s = series_errors_by_name(E_star)

        improved = []
        total_series = 0
        details = []
        for sname in ['Lyman', 'Balmer', 'Paschen']:
            if sname in base_s and sname in star_s:
                total_series += 1
                better = star_s[sname] < base_s[sname]
                if better:
                    improved.append(sname)
                details.append(
                    f"{sname[0]}:{base_s[sname]*100:.3f}->"
                    f"{star_s[sname]*100:.3f}%"
                    f"{'v' if better else 'x'}")

        frac = len(improved) / total_series if total_series > 0 else 0
        series_data[n] = {
            'improved': improved,
            'total': total_series,
            'fraction': frac,
        }
        print(f"    D_{n:>2} (w*={w_star:.6f}): [{', '.join(details)}]"
              f"  {len(improved)}/{total_series}")

    good_sizes = sum(1 for n in series_data
                     if series_data[n]['fraction'] >= 2 / 3)
    t2_pass = (good_sizes >= len(testable) * 4 / 5
               if len(testable) >= 3 else False)
    print(f"    Sizes with >= 2/3 series improved:"
          f" {good_sizes}/{len(testable)}")
    print(f"    -> {'PASS' if t2_pass else 'FAIL'}"
          f" (need: >= 4/5 of testable sizes)")

    # ===== T3: E-family sweet spots =====
    print(f"\n  T3: E-family sweet spots (E_6, E_7, E_8)")

    e_results = {}
    for n in [6, 7, 8]:
        w_star, min_err, base_err = find_sweet_spot(n, family='E')
        has_spot = (w_star is not None and w_star > 1e-7
                    and min_err < base_err)
        if has_spot:
            improvement = (base_err - min_err) / base_err * 100
            e_results[n] = {
                'w_star': w_star,
                'min_error': min_err,
                'base_error': base_err,
                'improvement_pct': improvement,
            }
            print(f"    E_{n}: w*={w_star:.6f}  err={min_err*100:.6f}%"
                  f"  (A_{n}: {base_err*100:.6f}%)"
                  f"  improvement: {improvement:.1f}%")
        else:
            e_results[n] = None
            if min_err is not None and base_err is not None:
                print(f"    E_{n}: no sweet spot"
                      f" (min={min_err*100:.6f}%"
                      f" vs base={base_err*100:.6f}%)")
            else:
                print(f"    E_{n}: no sweet spot found")

    e_spots = sum(1 for n in [6, 7, 8] if e_results.get(n) is not None)
    t3_pass = e_spots == 3
    print(f"    Sweet spots found: {e_spots}/3")
    print(f"    -> {'PASS' if t3_pass else 'FAIL'}"
          f" (need: all 3 E-types have sweet spot)")

    # ===== T4: Branch position matters =====
    print(f"\n  T4: D vs E branch position -- w* comparison")

    comparisons = []
    for n in [6, 7, 8]:
        d_data = d_results.get(n)
        e_data = e_results.get(n)
        if d_data is not None and e_data is not None:
            d_w = d_data['w_star']
            e_w = e_data['w_star']
            d_imp = d_data['improvement_pct']
            e_imp = e_data['improvement_pct']
            comparisons.append({
                'n': n,
                'd_w_star': d_w,
                'e_w_star': e_w,
                'd_improvement': d_imp,
                'e_improvement': e_imp,
                'd_bigger': d_w > e_w,
            })
            print(f"    n={n}: D w*={d_w:.6f} ({d_imp:.1f}%)"
                  f"  E w*={e_w:.6f} ({e_imp:.1f}%)"
                  f"  D {'>' if d_w > e_w else '<='} E")
        else:
            missing = []
            if d_data is None:
                missing.append('D')
            if e_data is None:
                missing.append('E')
            print(f"    n={n}: missing {'+'.join(missing)}")

    if len(comparisons) >= 3:
        all_d_bigger = all(c['d_bigger'] for c in comparisons)
        all_e_bigger = all(not c['d_bigger'] for c in comparisons)
        consistent = all_d_bigger or all_e_bigger
        if all_d_bigger:
            direction = 'D > E (shallower branch needs larger w*)'
        elif all_e_bigger:
            direction = 'E > D (deeper branch needs larger w*)'
        else:
            direction = 'inconsistent'
    else:
        consistent = False
        direction = 'insufficient data'

    t4_pass = consistent and len(comparisons) >= 3
    print(f"    Direction: {direction}")
    print(f"    -> {'PASS' if t4_pass else 'FAIL'}"
          f" (need: consistent direction for all 3 sizes)")

    # ===== Summary =====
    score = sum(1 for t in [t1_pass, t2_pass, t3_pass, t4_pass] if t)
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_22_branching_sweet_spot',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'The A_n->D_n/E_n deformation has an optimal branch '
                  'weight w* where the graph best approximates hydrogen '
                  'energy levels. w* compensates O(n^-4) discretization '
                  'error via eigenvalue perturbation. Branch position '
                  '(D at n-3, E at n-4) shifts the sweet spot.',
        'test_results': {
            'T1': {
                'description': 'w* exists for D_4..D_12, power law fit',
                'sweet_spots': {
                    str(n): d_results[n] for n in range(4, 13)
                    if d_results[n] is not None
                },
                'n_found': n_with_spots,
                'power_law_p': round(t1_power, 3),
                'power_law_c': round(t1_c, 6),
                'r_squared': round(t1_r2, 4),
                'PASS': t1_pass,
            },
            'T2': {
                'description': 'Per-series improvement at w*',
                'series_data': {
                    str(n): series_data[n] for n in testable
                },
                'good_sizes': good_sizes,
                'total_testable': len(testable),
                'PASS': t2_pass,
            },
            'T3': {
                'description': 'E-family sweet spots',
                'sweet_spots': {
                    str(n): e_results[n] for n in [6, 7, 8]
                    if e_results[n] is not None
                },
                'n_found': e_spots,
                'PASS': t3_pass,
            },
            'T4': {
                'description': 'D vs E branch position comparison',
                'comparisons': comparisons,
                'direction': direction,
                'consistent': consistent,
                'PASS': t4_pass,
            },
        },
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_22_branching_sweet_spot')


if __name__ == '__main__':
    main()
