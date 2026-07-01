"""
exp_21 -- Spectral Consolidation

Milestone R, Block D (Hardening)

Pulls three threads to strengthen the Milestone R foundations:

Thread 1 (T1): exp_20 T2 was confounded -- mixing different numbers of
series across ranks. Fix: track the SAME Lyman-alpha ratio from n=4 to
n=50. Verify clean O(1/n^2) convergence to exact hydrogen.

Thread 2 (T2): exp_18 (multi-channel fingerprints) and exp_20
(eigenvalue ratios) never talk to each other. Test: does eigenvalue
distance predict JSD+HKS fingerprint distance? If yes, confirms exp_19
finding that all channels read the same Laplacian.

Thread 3 (T3/T4): ADE branching departure is full-strength (6-18%).
Real fine structure is O(alpha^2). Continuously deform A_n -> D_n by
varying branch edge weight from 0 to 1. Measure how the departure
from hydrogen ratios scales with branch weight w. At w=alpha_EM, what
is the departure? This determines whether branching = fine structure
requires w=alpha (linear) or w=alpha^2 (quadratic).

Tests:
  T1: Lyman-alpha ratio error decreases monotonically n=4..50,
      power law error ~ n^(-p) with p > 1.5
  T2: Eigenvalue distance correlates with JSD+HKS distance (rho > 0.7)
  T3: Departure scales as w^p consistently across D_5..D_8 (CV < 30%)
  T4: Departure at w=alpha_EM is consistent across sizes and quantified
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    perspective_divergence,
    ade_graphs,
    save_mr_results,
)

# M6 alpha_EM: F3/(F4*phi*F10) * (1 - F10/(4*pi*F7^2))
ALPHA_EM = 2.0 / (3.0 * PHI * 55.0) * (1.0 - 55.0 / (4.0 * PI * 169.0))


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

    At w=0: A_n (path). At w=1: D_n (branched).
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
    h_r = 27.0 / 32.0  # hydrogen_ratio(2, 3, 1)
    return abs(g_r - h_r) / h_r


def compute_series_errors(E):
    """Mean series ratio error vs hydrogen (from exp_20)."""
    errors = []
    for _, m in [('Lyman', 1), ('Balmer', 2), ('Paschen', 3)]:
        k = m + 1
        while k + 1 <= len(E):
            kp = k + 1
            h_r = hydrogen_ratio(k, kp, m)
            if h_r is None or max(k, kp, m) > len(E):
                k += 1
                continue
            de_km = E[m - 1] - E[k - 1]
            de_kpm = E[m - 1] - E[kp - 1]
            if abs(de_kpm) < 1e-15:
                k += 1
                continue
            g_r = de_km / de_kpm
            errors.append(abs(g_r - h_r) / max(abs(h_r), 1e-15))
            k += 1
    return errors


def hks_values(adj, t=1.0):
    """Heat kernel signature per vertex."""
    n = adj.shape[0]
    weight_sums = np.sum(adj, axis=1)
    L = np.diag(weight_sums) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    hks = np.zeros(n)
    for k in range(n):
        hks += np.exp(-eigvals[k] * t) * eigvecs[:, k]**2
    return hks


def main():
    print("=" * 60)
    print("exp_21: Spectral Consolidation")
    print("=" * 60)
    print(f"  alpha_EM = {ALPHA_EM:.6f} = 1/{1.0/ALPHA_EM:.1f}")

    # ===== T1: Clean Lyman-alpha convergence =====
    print(f"\n  T1: Lyman-alpha ratio convergence (A_4 to A_50)")

    H_LYMAN = 27.0 / 32.0
    ranks = list(range(4, 51))
    lyman_errs = []

    for n in ranks:
        _, E = graph_eigendata(path_graph(n))
        err = lyman_alpha_error(E)
        lyman_errs.append(err if err is not None else np.nan)

    ns = np.array(ranks, dtype=float)
    es = np.array(lyman_errs)
    valid = ~np.isnan(es)
    ns_v, es_v = ns[valid], es[valid]

    for n_show in [4, 5, 6, 7, 8, 10, 15, 20, 30, 50]:
        idx = np.where(ns_v == n_show)[0]
        if len(idx) > 0:
            print(f"    A_{n_show:>2}: Lyman-alpha error = {es_v[idx[0]]*100:.5f}%")

    monotonic = bool(np.all(np.diff(es_v) <= 0))
    log_n = np.log(ns_v)
    log_e = np.log(es_v)
    slope, intercept, r_val, _, _ = stats.linregress(log_n, log_e)
    power = -slope

    print(f"    Monotonic decrease: {monotonic}")
    print(f"    Power law: error ~ n^(-{power:.3f}), R^2={r_val**2:.5f}")
    t1_pass = monotonic and power > 1.5
    print(f"    -> {'PASS' if t1_pass else 'FAIL'}"
          f" (need: monotonic AND power > 1.5)")

    # ===== T2: Eigenvalue distance vs fingerprint distance =====
    print(f"\n  T2: Eigenvalue distance vs JSD+HKS fingerprint distance")

    all_graphs = {}
    for name, adj in ade_graphs(max_rank=8):
        all_graphs[name] = adj

    size_groups = {}
    for name, adj in all_graphs.items():
        n = adj.shape[0]
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(name)
    size_groups = {k: v for k, v in size_groups.items() if len(v) >= 2}

    eig_dists = []
    fp_dists = []
    pair_labels = []

    for size, names in sorted(size_groups.items()):
        for g1, g2 in combinations(names, 2):
            adj1, adj2 = all_graphs[g1], all_graphs[g2]

            eig1, _ = graph_eigendata(adj1)
            eig2, _ = graph_eigendata(adj2)
            eig_d = float(np.linalg.norm(eig1 - eig2))

            jsd1 = np.sort([perspective_divergence(adj1, v) for v in range(size)])
            jsd2 = np.sort([perspective_divergence(adj2, v) for v in range(size)])
            jsd_d = float(np.linalg.norm(jsd1 - jsd2))

            hks1 = np.sort(hks_values(adj1))
            hks2 = np.sort(hks_values(adj2))
            hks_d = float(np.linalg.norm(hks1 - hks2))

            fp_d = float(np.sqrt(jsd_d**2 + hks_d**2))

            eig_dists.append(eig_d)
            fp_dists.append(fp_d)
            pair_labels.append(f"{g1} vs {g2}")

            print(f"    {g1:>5} vs {g2:<5}: eig={eig_d:.4f}"
                  f"  jsd={jsd_d:.4f}  hks={hks_d:.4f}"
                  f"  fp={fp_d:.4f}")

    if len(eig_dists) >= 3:
        rho_t2, p_t2 = stats.spearmanr(eig_dists, fp_dists)
    else:
        rho_t2, p_t2 = 0.0, 1.0
    t2_pass = rho_t2 > 0.7
    print(f"    Spearman rho = {rho_t2:.3f} (p={p_t2:.4f})")
    print(f"    -> {'PASS' if t2_pass else 'FAIL'} (need: rho > 0.7)")

    # ===== T3: Weighted branching departure scaling =====
    print(f"\n  T3: Weighted branching -- departure scaling")

    weights = np.array([
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.3, 0.5, 0.7, 1.0,
    ])
    power_exponents = {}

    for n in [4, 5, 6, 7, 8]:
        departures = []
        valid_w = []

        for w in weights:
            adj_w = weighted_dn(n, w)
            _, E = graph_eigendata(adj_w)
            errs = compute_series_errors(E)
            if errs:
                departures.append(float(np.mean(errs)))
                valid_w.append(w)

        if len(departures) >= 5:
            lw = np.log(np.array(valid_w))
            ld = np.log(np.array(departures))
            sl, _, r, _, _ = stats.linregress(lw, ld)
            power_exponents[n] = sl
            print(f"    D_{n}: departure ~ w^{sl:.3f} (R^2={r**2:.3f})")
            for w_show in [0.01, 0.1, 1.0]:
                if w_show in valid_w:
                    idx = valid_w.index(w_show)
                    print(f"      w={w_show}: {departures[idx]*100:.3f}%")

    if power_exponents:
        exps = list(power_exponents.values())
        mean_p = float(np.mean(exps))
        std_p = float(np.std(exps))
        cv = std_p / abs(mean_p) if abs(mean_p) > 1e-10 else float('inf')
    else:
        mean_p, std_p, cv = 0.0, 0.0, float('inf')

    t3_pass = cv < 0.30 and len(power_exponents) >= 3
    print(f"    Exponent: {mean_p:.3f} +/- {std_p:.3f} (CV={cv:.2f})")
    print(f"    -> {'PASS' if t3_pass else 'FAIL'}"
          f" (need: CV < 0.30, >= 3 sizes)")

    # ===== T4: Fine structure at w = alpha_EM =====
    print(f"\n  T4: Departure at w = alpha_EM")

    alpha_deps = {}
    baseline_deps = {}

    for n in [4, 5, 6, 7, 8]:
        # Baseline: A_n error
        _, E_an = graph_eigendata(path_graph(n))
        base_errs = compute_series_errors(E_an)
        baseline = float(np.mean(base_errs)) if base_errs else 0

        # At w = alpha_EM
        adj_a = weighted_dn(n, ALPHA_EM)
        _, E_a = graph_eigendata(adj_a)
        alpha_errs = compute_series_errors(E_a)
        if alpha_errs:
            dep = float(np.mean(alpha_errs))
            additional = dep - baseline
            alpha_deps[n] = additional
            baseline_deps[n] = baseline
            print(f"    D_{n}(w=alpha): total={dep*100:.5f}%"
                  f"  baseline={baseline*100:.5f}%"
                  f"  additional={additional*100:.5f}%")

    alpha_sq = ALPHA_EM**2
    if alpha_deps:
        adds = np.array(list(alpha_deps.values()))
        mean_add = float(np.mean(np.abs(adds)))
        ratio_alpha = mean_add / ALPHA_EM if ALPHA_EM > 0 else 0
        ratio_alpha_sq = mean_add / alpha_sq if alpha_sq > 0 else 0

        if ratio_alpha > 0 and ratio_alpha_sq > 0:
            closer_to = ('alpha' if abs(np.log10(ratio_alpha)) <
                         abs(np.log10(ratio_alpha_sq)) else 'alpha^2')
        else:
            closer_to = 'undetermined'

        dep_cv = float(np.std(np.abs(adds)) / mean_add) if mean_add > 0 else float('inf')
        t4_pass = dep_cv < 0.5 and len(alpha_deps) >= 3

        print(f"\n    alpha = {ALPHA_EM:.6f}, alpha^2 = {alpha_sq:.2e}")
        print(f"    Mean |additional departure| = {mean_add*100:.5f}%")
        print(f"    Ratio to alpha: {ratio_alpha:.4f}")
        print(f"    Ratio to alpha^2: {ratio_alpha_sq:.1f}")
        print(f"    Scaling closer to: {closer_to}")
    else:
        t4_pass = False
        mean_add = 0
        ratio_alpha = 0
        ratio_alpha_sq = 0
        closer_to = 'undetermined'
        dep_cv = float('inf')

    print(f"    -> {'PASS' if t4_pass else 'FAIL'}"
          f" (need: consistent across >= 3 sizes)")

    # ===== Summary =====
    score = sum(1 for t in [t1_pass, t2_pass, t3_pass, t4_pass] if t)
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_21_spectral_consolidation',
        'timestamp': datetime.now().isoformat(),
        'block': 'D',
        'thesis': 'Consolidation pulling three threads: (1) clean '
                  'Lyman-alpha convergence to n=50, (2) eigenvalue '
                  'distance predicts fingerprint distance, (3) weighted '
                  'branching departure scaling determines if fine '
                  'structure is O(alpha) or O(alpha^2).',
        'test_results': {
            'T1': {
                'description': 'Lyman-alpha converges monotonically',
                'power': round(power, 3),
                'r_squared': round(r_val**2, 5),
                'monotonic': monotonic,
                'error_at_n50_pct': round(float(es_v[-1]) * 100, 6),
                'PASS': t1_pass,
            },
            'T2': {
                'description': 'Eigenvalue dist correlates with fingerprint dist',
                'spearman_rho': round(float(rho_t2), 3),
                'p_value': round(float(p_t2), 4),
                'n_pairs': len(eig_dists),
                'PASS': t2_pass,
            },
            'T3': {
                'description': 'Branching departure scales consistently',
                'exponents': {str(k): round(v, 3)
                              for k, v in power_exponents.items()},
                'mean_exponent': round(mean_p, 3),
                'cv': round(cv, 3),
                'PASS': t3_pass,
            },
            'T4': {
                'description': 'Departure at alpha_EM consistent and quantified',
                'alpha_em': round(ALPHA_EM, 8),
                'alpha_sq': round(alpha_sq, 10),
                'additional_departures_pct': {
                    str(k): round(v * 100, 6)
                    for k, v in alpha_deps.items()
                },
                'mean_additional_pct': round(mean_add * 100, 6),
                'scaling_closer_to': closer_to,
                'ratio_to_alpha': round(ratio_alpha, 4),
                'ratio_to_alpha_sq': round(ratio_alpha_sq, 1),
                'PASS': t4_pass,
            },
        },
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_21_spectral_consolidation')


if __name__ == '__main__':
    main()
