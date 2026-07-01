"""
exp_23 -- Confluence Convergence Ordering

Milestone R, Block C (Novel Physics)

The confluence operator (PAC arithmetic) is non-commutative and stateful:
each output depends on accumulated memory from prior steps. In the spectral
context, each eigenvalue ratio is a different position in the confluence
stream. Low-k modes (Lyman) are early -- less accumulated memory, more
robust. High-k modes (Paschen) are late -- more memory, more sensitive
to perturbation.

exp_22 showed branching corrections are series-selective. The confluence
arithmetic predicts this: you CAN'T have a universal correction because
the memory state m_t differs at each stream position. The only point
where all series agree is the confluent attractor (n -> infinity, the
1/k^2 pattern).

This experiment tests three quantitative predictions:

1. Per-series convergence rates are ORDERED by series index:
   Lyman converges fastest (earliest in stream), Paschen slowest.

2. The convergence exponents relate to each other through the
   series index -- not arbitrary, but structured by the stream position.

3. Branching sensitivity (how much D_n departure exceeds A_n) is
   also ordered by series: Paschen > Balmer > Lyman.

Tests:
  T1: Per-series convergence exponents ordered: p_Lyman > p_Balmer > p_Paschen
      (PASS: strictly ordered for A_n path graphs)
  T2: Convergence exponents relate to series index m: p_m ~ f(m)
      with R^2 > 0.95 for some functional form
      (PASS: clean functional relationship)
  T3: Branching sensitivity ordered: D_n departure(Paschen) > departure(Balmer)
      > departure(Lyman) for majority of sizes
      (PASS: ordering holds for >= 4/5 sizes where all 3 series computable)
  T4: At the confluent point (large n), per-series errors converge to
      the SAME value -- the attractor is unique
      (PASS: ratio of max/min series error < 2.0 at n=50)
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


def alpha_ratio_error(E, series_m):
    """
    Error of the FIRST (alpha) ratio in a given series.
    Series m: dE(m+1 -> m) / dE(m+2 -> m) vs hydrogen.
    Returns None if insufficient energy levels.
    """
    k = series_m + 1
    kp = series_m + 2
    if max(k, kp) > len(E) or series_m > len(E):
        return None
    h_r = hydrogen_ratio(k, kp, series_m)
    if h_r is None:
        return None
    de_km = E[series_m - 1] - E[k - 1]
    de_kpm = E[series_m - 1] - E[kp - 1]
    if abs(de_kpm) < 1e-15:
        return None
    g_r = de_km / de_kpm
    return abs(g_r - h_r) / abs(h_r)


def series_mean_error(E, series_m):
    """Mean error of ALL consecutive ratios in a given series."""
    errs = []
    k = series_m + 1
    while k + 1 <= len(E):
        kp = k + 1
        h_r = hydrogen_ratio(k, kp, series_m)
        if h_r is not None and max(k, kp) <= len(E):
            de_km = E[series_m - 1] - E[k - 1]
            de_kpm = E[series_m - 1] - E[kp - 1]
            if abs(de_kpm) > 1e-15:
                g_r = de_km / de_kpm
                errs.append(abs(g_r - h_r) / max(abs(h_r), 1e-15))
        k += 1
    return float(np.mean(errs)) if errs else None


def main():
    print("=" * 60)
    print("exp_23: Confluence Convergence Ordering")
    print("=" * 60)
    print()
    print("  Confluence arithmetic predicts: per-series convergence")
    print("  rates are ordered by stream position (series index).")
    print()

    # ===== T1: Per-series convergence exponents =====
    print("  T1: Per-series convergence exponents for A_n path graphs")
    print()

    series_names = {1: 'Lyman', 2: 'Balmer', 3: 'Paschen', 4: 'Brackett'}
    # Use alpha ratio (first ratio in each series) for clean comparison
    # Need n >= 2*m + 2 for series m to have at least one ratio
    # Lyman (m=1): n >= 4. Balmer (m=2): n >= 6. Paschen (m=3): n >= 8.
    # Brackett (m=4): n >= 10.

    ranks = list(range(4, 51))
    series_errors = {}  # {m: [(n, error), ...]}

    for m in [1, 2, 3, 4]:
        series_errors[m] = []
        for n in ranks:
            _, E = graph_eigendata(path_graph(n))
            err = alpha_ratio_error(E, m)
            if err is not None and err > 0:
                series_errors[m].append((n, err))

    # Fit power law for each series
    exponents = {}
    for m in [1, 2, 3, 4]:
        data = series_errors[m]
        if len(data) < 5:
            continue
        ns = np.array([d[0] for d in data], dtype=float)
        es = np.array([d[1] for d in data])
        valid = es > 0
        if np.sum(valid) < 5:
            continue
        log_n = np.log(ns[valid])
        log_e = np.log(es[valid])
        slope, intercept, r_val, _, _ = stats.linregress(log_n, log_e)
        p = -slope
        r2 = r_val**2
        exponents[m] = {'p': p, 'r2': r2, 'n_points': int(np.sum(valid))}
        sname = series_names.get(m, f"m={m}")
        print(f"    {sname:<10} (m={m}): error ~ n^(-{p:.3f}),"
              f" R^2={r2:.5f}  ({int(np.sum(valid))} points)")

    # Check ordering: p_1 > p_2 > p_3 > p_4
    ordered_ms = sorted(exponents.keys())
    if len(ordered_ms) >= 3:
        ps = [exponents[m]['p'] for m in ordered_ms]
        strictly_decreasing = all(ps[i] > ps[i+1]
                                  for i in range(len(ps)-1))
        print(f"\n    Exponents: {' > '.join(f'{p:.3f}' for p in ps)}")
        print(f"    Strictly decreasing: {strictly_decreasing}")
        t1_pass = strictly_decreasing
    else:
        strictly_decreasing = False
        t1_pass = False

    print(f"    -> {'PASS' if t1_pass else 'FAIL'}"
          f" (need: p_Lyman > p_Balmer > p_Paschen)")

    # ===== T2: Functional relationship p_m vs m =====
    print(f"\n  T2: Convergence exponent vs series index")

    if len(exponents) >= 3:
        ms = np.array(sorted(exponents.keys()), dtype=float)
        ps = np.array([exponents[int(m)]['p'] for m in ms])

        # Try p(m) = a / m^b (power law in series index)
        log_m = np.log(ms)
        log_p = np.log(ps)
        sl_pm, int_pm, r_pm, _, _ = stats.linregress(log_m, log_p)
        r2_power = r_pm**2

        # Try p(m) = a - b*m (linear)
        sl_lin, int_lin, r_lin, _, _ = stats.linregress(ms, ps)
        r2_linear = r_lin**2

        # Try p(m) = a - b*ln(m) (log)
        sl_log, int_log, r_log, _, _ = stats.linregress(np.log(ms), ps)
        r2_log = r_log**2

        best_r2 = max(r2_power, r2_linear, r2_log)
        if best_r2 == r2_power:
            form = f"p(m) ~ m^({sl_pm:.3f})"
            best_label = "power"
        elif best_r2 == r2_linear:
            form = f"p(m) = {int_lin:.3f} - {-sl_lin:.3f}*m"
            best_label = "linear"
        else:
            form = f"p(m) = {int_log:.3f} + {sl_log:.3f}*ln(m)"
            best_label = "logarithmic"

        print(f"    Power law in m:   R^2 = {r2_power:.4f}")
        print(f"    Linear in m:      R^2 = {r2_linear:.4f}")
        print(f"    Logarithmic in m: R^2 = {r2_log:.4f}")
        print(f"    Best fit: {form} ({best_label}, R^2={best_r2:.4f})")

        t2_pass = best_r2 > 0.95
    else:
        best_r2 = 0
        best_label = 'insufficient'
        form = 'N/A'
        t2_pass = False

    print(f"    -> {'PASS' if t2_pass else 'FAIL'}"
          f" (need: R^2 > 0.95)")

    # ===== T3: Branching sensitivity ordered by series =====
    print(f"\n  T3: Branching sensitivity ordered by series")

    # For each size where all 3 series are computable (n >= 8),
    # compare D_n departure vs A_n departure per series
    all_graphs = {}
    for name, adj in ade_graphs(max_rank=8):
        all_graphs[name] = adj

    ordering_holds = 0
    ordering_total = 0
    branching_data = {}

    for n in range(8, 13):
        a_name = f"A_{n}"
        d_name = f"D_{n}"
        if a_name not in all_graphs or d_name not in all_graphs:
            # Build them if not in ade_graphs
            a_adj = path_graph(n)
        else:
            a_adj = all_graphs[a_name]

        if d_name not in all_graphs:
            continue
        d_adj = all_graphs[d_name]

        _, E_a = graph_eigendata(a_adj)
        _, E_d = graph_eigendata(d_adj)

        departures = {}
        for m, sname in [(1, 'Lyman'), (2, 'Balmer'), (3, 'Paschen')]:
            err_a = series_mean_error(E_a, m)
            err_d = series_mean_error(E_d, m)
            if err_a is not None and err_d is not None:
                departures[sname] = err_d - err_a

        if len(departures) == 3:
            ordering_total += 1
            L = departures['Lyman']
            B = departures['Balmer']
            P = departures['Paschen']
            ordered = P > B > L
            if ordered:
                ordering_holds += 1
            branching_data[n] = departures
            print(f"    n={n}: Lyman={L*100:+.3f}%"
                  f"  Balmer={B*100:+.3f}%"
                  f"  Paschen={P*100:+.3f}%"
                  f"  {'P>B>L' if ordered else 'NOT ordered'}")

    t3_pass = (ordering_holds >= ordering_total * 4 / 5
               if ordering_total >= 3 else False)
    print(f"    Ordered: {ordering_holds}/{ordering_total}")
    print(f"    -> {'PASS' if t3_pass else 'FAIL'}"
          f" (need: >= 4/5 sizes)")

    # ===== T4: Confluence -- errors converge at large n =====
    print(f"\n  T4: Confluence at large n -- per-series errors converge")

    # At n=50, compute per-series alpha ratio errors
    # If the confluent attractor is unique, all series should have
    # similar error magnitudes
    convergence_data = {}
    for n_test in [10, 20, 30, 50]:
        _, E = graph_eigendata(path_graph(n_test))
        errs = {}
        for m, sname in [(1, 'Lyman'), (2, 'Balmer'), (3, 'Paschen')]:
            err = alpha_ratio_error(E, m)
            if err is not None:
                errs[sname] = err
        if len(errs) >= 3:
            vals = list(errs.values())
            ratio = max(vals) / min(vals) if min(vals) > 0 else float('inf')
            convergence_data[n_test] = {
                'errors': errs,
                'max_min_ratio': ratio,
            }
            err_str = ', '.join(f"{s}:{e*100:.6f}%"
                                for s, e in errs.items())
            print(f"    n={n_test:>2}: [{err_str}]"
                  f"  max/min={ratio:.2f}")

    if 50 in convergence_data:
        t4_ratio = convergence_data[50]['max_min_ratio']
        t4_pass = t4_ratio < 2.0
    else:
        t4_ratio = float('inf')
        t4_pass = False

    # Also check that the ratio DECREASES with n (convergence to attractor)
    if len(convergence_data) >= 3:
        ns_conv = sorted(convergence_data.keys())
        ratios_conv = [convergence_data[n]['max_min_ratio'] for n in ns_conv]
        rho_conv, _ = stats.spearmanr(ns_conv, ratios_conv)
        print(f"\n    Ratio trend with n: Spearman rho = {rho_conv:.3f}"
              f" ({'decreasing' if rho_conv < 0 else 'NOT decreasing'})")

    print(f"    -> {'PASS' if t4_pass else 'FAIL'}"
          f" (need: max/min ratio < 2.0 at n=50)")

    # ===== Summary =====
    score = sum(1 for t in [t1_pass, t2_pass, t3_pass, t4_pass] if t)
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_23_confluence_convergence',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'Confluence arithmetic predicts per-series convergence '
                  'rates are ordered by stream position: Lyman (earliest, '
                  'most robust) converges fastest, Paschen (latest, most '
                  'sensitive) converges slowest. The 1/k^2 pattern is the '
                  'confluent attractor where all streams agree.',
        'test_results': {
            'T1': {
                'description': 'Per-series convergence exponents ordered',
                'exponents': {
                    series_names.get(m, f"m={m}"): exponents[m]
                    for m in sorted(exponents.keys())
                },
                'strictly_decreasing': strictly_decreasing,
                'PASS': t1_pass,
            },
            'T2': {
                'description': 'Exponents follow clean f(m)',
                'best_fit': form,
                'best_label': best_label,
                'r2_power': round(r2_power, 4) if 'r2_power' in dir() else None,
                'r2_linear': round(r2_linear, 4) if 'r2_linear' in dir() else None,
                'r2_log': round(r2_log, 4) if 'r2_log' in dir() else None,
                'best_r2': round(best_r2, 4),
                'PASS': t2_pass,
            },
            'T3': {
                'description': 'Branching sensitivity Paschen > Balmer > Lyman',
                'branching_data': {
                    str(k): {s: round(v * 100, 4) for s, v in d.items()}
                    for k, d in branching_data.items()
                },
                'ordering_holds': ordering_holds,
                'ordering_total': ordering_total,
                'PASS': t3_pass,
            },
            'T4': {
                'description': 'Per-series errors converge at large n',
                'convergence_data': {
                    str(k): {
                        'errors_pct': {s: round(e * 100, 6)
                                       for s, e in v['errors'].items()},
                        'max_min_ratio': round(v['max_min_ratio'], 3),
                    }
                    for k, v in convergence_data.items()
                },
                'ratio_at_50': round(t4_ratio, 3),
                'PASS': t4_pass,
            },
        },
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_23_confluence_convergence')


if __name__ == '__main__':
    main()
