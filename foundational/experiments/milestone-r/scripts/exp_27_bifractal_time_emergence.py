"""
exp_27 -- Bifractal Time Emergence from Laplacian Collapse

Milestone R, Block D (Consolidation)

The bifractal time paper defines M(t) = Collapse(R_b(t), R_f(t)) -- the present
is the intersection of backward ancestry (R_b) and forward constraint (R_f).
exp_26 validated the prerequisites: the Laplacian (R_b) determines all three
measurement channels (FPT, JSD, HKS), but they carry genuinely non-redundant
information (different R_f directions).

Key mapping:
  R_b = Laplacian spectrum (static graph structure, PAC ancestry)
  R_f = noise amplitude (dynamic SEC flux, forward constraint)
  M(t) = measurement (FPT, JSD, HKS -- three collapse projections)

Tests:
  T1: Dual scaling regimes -- crossover from R_b-dominated to R_f-dominated
  T2: Collapse dimensionality from eigenvalue structure
  T3: Non-commutative collapse -- channels produce different vertex orderings
  T4: Time is not privileged -- no single channel dominates classification
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, LN2,
    stress_barrier_walk,
    perspective_divergence,
    ade_graphs,
    save_mr_results,
)


# ============================================================
# Utility functions (inline, from exp_26)
# ============================================================

def gap_split_classify(values):
    """Classify vertices by gap-splitting on sorted values."""
    n = len(values)
    if n <= 1:
        return list(range(n))

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    labels = [0] * n
    current_label = 0
    labels[indexed[0][0]] = current_label

    for i in range(1, n):
        gap = indexed[i][1] - indexed[i - 1][1]
        spread = indexed[-1][1] - indexed[0][1]
        if spread > 0 and gap / spread > 0.3:
            current_label += 1
        labels[indexed[i][0]] = current_label

    return labels


def adjusted_rand_index(labels_true, labels_pred):
    """Compute ARI between two label assignments."""
    from collections import Counter
    n = len(labels_true)
    if n <= 1:
        return 1.0

    pairs_true = Counter()
    pairs_pred = Counter()
    pairs_both = Counter()

    for i in range(n):
        pairs_true[labels_true[i]] += 1
        pairs_pred[labels_pred[i]] += 1
        pairs_both[(labels_true[i], labels_pred[i])] += 1

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_comb_both = sum(comb2(v) for v in pairs_both.values())
    sum_comb_true = sum(comb2(v) for v in pairs_true.values())
    sum_comb_pred = sum(comb2(v) for v in pairs_pred.values())
    total_comb = comb2(n)

    if total_comb == 0:
        return 1.0

    expected = sum_comb_true * sum_comb_pred / total_comb
    max_index = (sum_comb_true + sum_comb_pred) / 2.0
    denom = max_index - expected

    if abs(denom) < 1e-15:
        return 1.0

    return (sum_comb_both - expected) / denom


# ============================================================
# Channel computation with variable noise
# ============================================================

def compute_channels(name, adj, n_trials=50, stress_threshold=0.008,
                     noise_amplitude=0.020, max_steps=3000):
    """Compute FPT, JSD, HKS for each vertex at a given noise level."""
    n = adj.shape[0]

    # Laplacian spectral decomposition
    degrees = np.sum(adj > 0, axis=1).astype(float)
    L = np.diag(degrees) - adj.astype(float)
    eigvals, eigvecs = np.linalg.eigh(L)

    # HKS at t=1.0
    hks = np.zeros(n)
    for k in range(n):
        hks += np.exp(-eigvals[k] * 1.0) * eigvecs[:, k] ** 2

    channels = {'fpt': [], 'jsd': [], 'hks': [], 'degree': []}

    for v in range(n):
        degree = int(np.sum(adj[v] > 0))

        # FPT: median stress barrier first-passage time
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            result = stress_barrier_walk(
                adj, v, initial,
                stress_threshold=stress_threshold,
                noise_amplitude=noise_amplitude,
                max_steps=max_steps,
                seed=trial * 100 + v * 50000 + abs(hash(name)) % 10000,
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])
        median_fpt = float(np.median(fpts)) if fpts else float(max_steps)

        # JSD
        jsd = perspective_divergence(adj, v, horizon=2)

        channels['fpt'].append(median_fpt)
        channels['jsd'].append(jsd)
        channels['hks'].append(float(hks[v]))
        channels['degree'].append(degree)

    return channels


def spectral_gap(adj):
    """Smallest nonzero Laplacian eigenvalue (Fiedler value)."""
    n = adj.shape[0]
    degrees = np.sum(adj > 0, axis=1).astype(float)
    L = np.diag(degrees) - adj.astype(float)
    eigvals = np.linalg.eigvalsh(L)
    # Find smallest eigenvalue > tolerance
    nonzero = eigvals[eigvals > 1e-10]
    return float(nonzero[0]) if len(nonzero) > 0 else 0.0


def eigenvalue_clusters(adj):
    """Count Laplacian eigenvalue clusters by gap-splitting."""
    n = adj.shape[0]
    degrees = np.sum(adj > 0, axis=1).astype(float)
    L = np.diag(degrees) - adj.astype(float)
    eigvals = np.linalg.eigvalsh(L)
    labels = gap_split_classify(list(eigvals))
    return len(set(labels))


# ============================================================
# Tests
# ============================================================

def test_T1_dual_scaling():
    """T1: Dual scaling regimes -- R_b vs R_f dominance with crossover."""
    print("\n  T1: Dual Scaling Regimes (R_b vs R_f Dominance)")
    results = {'description': 'Crossover exists for >=5/7 graphs AND |rho(sigma*, 1/gap)| > 0.5'}

    noise_levels = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    # Use D_5..D_8, E_6, E_7, E_8 for this test (7 graphs)
    target_graphs = ['D_5', 'D_6', 'D_7', 'D_8', 'E_6', 'E_7', 'E_8']

    crossover_sigmas = []
    inv_gaps = []
    details = []

    for name, adj in ade_graphs(max_rank=8):
        if name not in target_graphs:
            continue

        n = adj.shape[0]
        print(f"    {name} (n={n})...", end="", flush=True)

        gap = spectral_gap(adj)
        degrees = np.sum(adj > 0, axis=1).astype(float)

        # For each noise level, compute median FPT per vertex
        fpt_by_noise = {}
        for sigma in noise_levels:
            ch = compute_channels(name, adj, n_trials=30,
                                  noise_amplitude=sigma, max_steps=2000)
            fpt_by_noise[sigma] = np.array(ch['fpt'])

        # R_b model: at each noise level, R^2 of log(FPT) vs degree
        r2_struct_by_sigma = {}
        for sigma in noise_levels:
            fpt_arr = fpt_by_noise[sigma]
            log_fpt = np.log(fpt_arr + 1)
            if np.std(degrees) < 1e-10 or np.std(log_fpt) < 1e-10:
                r2_struct_by_sigma[sigma] = 0.0
                continue
            r = np.corrcoef(degrees, log_fpt)[0, 1]
            r2_struct_by_sigma[sigma] = r ** 2

        # R_f model: for each degree class, R^2 of log(FPT) vs log(1/sigma)
        # Aggregate: mean R^2 across degree classes
        unique_degrees = sorted(set(int(d) for d in degrees))
        r2_noise_by_sigma = {}
        if len(noise_levels) >= 3:
            log_inv_sigma = np.log(1.0 / np.array(noise_levels))
            for sigma_idx, sigma in enumerate(noise_levels):
                # At this sigma, how well does noise predict FPT across degree classes?
                r2_per_deg = []
                for deg in unique_degrees:
                    deg_mask = np.array([int(d) == deg for d in degrees])
                    if np.sum(deg_mask) < 1:
                        continue
                    # Get FPT for this degree class across all noise levels
                    fpts_across_noise = [np.mean(fpt_by_noise[s][deg_mask])
                                         for s in noise_levels]
                    log_fpts = np.log(np.array(fpts_across_noise) + 1)
                    if np.std(log_fpts) < 1e-10 or np.std(log_inv_sigma) < 1e-10:
                        continue
                    r = np.corrcoef(log_inv_sigma, log_fpts)[0, 1]
                    r2_per_deg.append(r ** 2)
                r2_noise_by_sigma[sigma] = np.mean(r2_per_deg) if r2_per_deg else 0.0

        # Find crossover: sigma where R_f model starts to dominate R_b model
        crossover = None
        for sigma in noise_levels:
            r2_s = r2_struct_by_sigma.get(sigma, 0)
            r2_n = r2_noise_by_sigma.get(sigma, 0)
            if r2_n > r2_s:
                crossover = sigma
                break

        if crossover is not None:
            crossover_sigmas.append(crossover)
            inv_gaps.append(1.0 / gap if gap > 0 else float('inf'))

        details.append({
            'graph': name, 'spectral_gap': gap,
            'crossover_sigma': crossover,
            'r2_struct': r2_struct_by_sigma,
            'r2_noise': r2_noise_by_sigma,
        })
        status = f"sigma*={crossover}" if crossover else "no crossover"
        print(f" gap={gap:.3f}, {status}")

    n_with_crossover = len(crossover_sigmas)
    n_tested = len(details)

    # Spearman correlation between sigma* and 1/spectral_gap
    if len(crossover_sigmas) >= 3:
        rho, p_val = spearmanr(crossover_sigmas, inv_gaps)
    else:
        rho, p_val = 0.0, 1.0

    passed = n_with_crossover >= 5 and abs(rho) > 0.5
    results['n_with_crossover'] = n_with_crossover
    results['n_tested'] = n_tested
    results['spearman_rho'] = float(rho)
    results['spearman_p'] = float(p_val)
    results['crossover_sigmas'] = crossover_sigmas
    results['inv_gaps'] = inv_gaps
    results['details'] = details
    results['PASS'] = passed

    print(f"    {n_with_crossover}/{n_tested} have crossover, "
          f"rho={rho:.3f} -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_collapse_dimensionality():
    """T2: Effective rank of channel matrix = min(3, n_eigenvalue_clusters - 1)."""
    print("\n  T2: Collapse Dimensionality from Eigenvalue Structure")
    results = {'description': 'Prediction correct for >=6/11 ADE graphs'}

    n_correct = 0
    n_tested = 0
    details = []

    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        if n < 5:
            continue

        print(f"    {name} (n={n})...", end="", flush=True)

        ch = compute_channels(name, adj, n_trials=30, max_steps=2000)

        # 3-channel matrix
        M = np.column_stack([ch['fpt'], ch['jsd'], ch['hks']])
        # Normalize
        for col in range(3):
            std = np.std(M[:, col])
            if std > 1e-15:
                M[:, col] = (M[:, col] - np.mean(M[:, col])) / std

        # Effective rank: PCA components for 95% variance
        sv = np.linalg.svd(M, compute_uv=False)
        total_var = np.sum(sv ** 2)
        cumvar = np.cumsum(sv ** 2) / total_var if total_var > 0 else np.ones(3)
        eff_rank = int(np.searchsorted(cumvar, 0.95) + 1)

        # Predicted: min(3, n_clusters - 1)
        n_clusters = eigenvalue_clusters(adj)
        predicted = min(3, max(1, n_clusters - 1))

        correct = eff_rank == predicted
        if correct:
            n_correct += 1
        n_tested += 1

        details.append({
            'graph': name, 'n_clusters': n_clusters,
            'predicted_rank': predicted, 'actual_rank': eff_rank,
            'singular_values': [float(s) for s in sv],
            'correct': correct,
        })
        print(f" clusters={n_clusters}, predicted={predicted}, "
              f"actual={eff_rank} -> {'ok' if correct else 'miss'}")

    passed = n_correct >= 6
    results['n_correct'] = n_correct
    results['n_tested'] = n_tested
    results['details'] = details
    results['PASS'] = passed
    print(f"    {n_correct}/{n_tested} correct -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_noncommutative_collapse():
    """T3: Pairwise Kendall tau between channel rankings in (0.1, 0.9)."""
    print("\n  T3: Non-Commutative Collapse (Kendall tau)")
    results = {'description': 'All pairwise tau in (0.1, 0.9) for >=8/11 graphs'}

    channel_names = ['fpt', 'jsd', 'hks']
    pairs = [('fpt', 'jsd'), ('fpt', 'hks'), ('jsd', 'hks')]
    n_in_range = 0
    n_tested = 0
    details = []

    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        if n < 5:
            continue

        print(f"    {name} (n={n})...", end="", flush=True)

        ch = compute_channels(name, adj, n_trials=30, max_steps=2000)

        all_ok = True
        taus = {}
        for c1, c2 in pairs:
            v1 = np.array(ch[c1])
            v2 = np.array(ch[c2])
            tau, p_val = kendalltau(v1, v2)
            taus[f'{c1}-{c2}'] = float(tau)
            if not (0.1 < abs(tau) < 0.9):
                all_ok = False

        if all_ok:
            n_in_range += 1
        n_tested += 1

        details.append({'graph': name, 'taus': taus, 'all_in_range': all_ok})
        tau_str = ", ".join(f"{k}={v:.3f}" for k, v in taus.items())
        print(f" {tau_str} -> {'ok' if all_ok else 'miss'}")

    passed = n_in_range >= 8
    results['n_in_range'] = n_in_range
    results['n_tested'] = n_tested
    results['details'] = details
    results['PASS'] = passed
    print(f"    {n_in_range}/{n_tested} have all tau in (0.1,0.9) "
          f"-> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_time_not_privileged():
    """T4: No channel is privileged -- CV of ARI-loss < 0.5 across removals."""
    print("\n  T4: Time Is Not Privileged (leave-one-channel-out)")
    results = {'description': 'Mean CV of ARI-loss < 0.5 across graphs'}

    channel_names = ['fpt', 'jsd', 'hks']
    cvs = []
    details = []

    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        if n < 5:
            continue

        print(f"    {name} (n={n})...", end="", flush=True)

        ch = compute_channels(name, adj, n_trials=30, max_steps=2000)
        true_labels = ch['degree']

        # Full 3-channel ARI
        full_labels = gap_split_classify(ch['fpt'])  # Best single channel
        # Actually use best of 3 as baseline
        aris_single = {}
        for cname in channel_names:
            pred = gap_split_classify(ch[cname])
            aris_single[cname] = adjusted_rand_index(true_labels, pred)
        full_ari = max(aris_single.values())

        # Leave-one-out: remove each channel, classify with remaining 2
        ari_losses = {}
        for remove_ch in channel_names:
            remaining = [c for c in channel_names if c != remove_ch]
            # Use the better of the two remaining channels
            ari_remaining = max(
                adjusted_rand_index(true_labels, gap_split_classify(ch[c]))
                for c in remaining
            )
            loss = max(0, full_ari - ari_remaining)
            ari_losses[remove_ch] = loss

        losses = list(ari_losses.values())
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        cv = std_loss / mean_loss if mean_loss > 1e-10 else 0.0

        cvs.append(cv)
        details.append({
            'graph': name, 'ari_losses': ari_losses,
            'cv': float(cv), 'full_ari': float(full_ari),
            'aris_single': aris_single,
        })
        loss_str = ", ".join(f"-{k}={v:.3f}" for k, v in ari_losses.items())
        print(f" CV={cv:.3f} ({loss_str})")

    mean_cv = float(np.mean(cvs)) if cvs else 1.0
    passed = mean_cv < 0.5
    results['mean_cv'] = mean_cv
    results['per_graph_cvs'] = cvs
    results['details'] = details
    results['PASS'] = passed
    print(f"    Mean CV = {mean_cv:.3f} -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_27: Bifractal Time Emergence from Laplacian Collapse")
    print("=" * 60)

    t1 = test_T1_dual_scaling()
    t2 = test_T2_collapse_dimensionality()
    t3 = test_T3_noncommutative_collapse()
    t4 = test_T4_time_not_privileged()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_27_bifractal_time_emergence',
        'timestamp': datetime.now().isoformat(),
        'block': 'D',
        'thesis': 'Bifractal time M(t) = Collapse(R_b(t), R_f(t)) predicts: '
                  'dual scaling regimes (R_b/R_f crossover at spectral gap), '
                  'collapse dimensionality from eigenvalue clusters, '
                  'non-commutative channel orderings, and no privileged time channel.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_27_bifractal_time_emergence')
