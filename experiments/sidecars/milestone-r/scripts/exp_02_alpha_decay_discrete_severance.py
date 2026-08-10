"""
exp_02 -- Alpha Decay as Discrete Ledger Severance

Milestone R, Block A (Ledger Severance Mechanics)

Hypothesis: Alpha decay is the severance of a settled PAC ledger at a specific
Fibonacci depth. The discrete energy spectrum reflects the finite number of
structurally distinct severance sites. Alpha energies should be integer
multiples of Xi * E_scale(d) for some Fibonacci depth d.

Tests:
  T1: Alpha energies as integer multiples of Xi * E_scale at depth d
  T2: Pairwise alpha energy ratios cluster near phi powers vs random
  T3: Daughter graph relaxation timescale correlates with graph size
  T4: Alpha vs proton emission energy ratio ~ phi^(delta_d)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import percentileofscore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI,
    scope_boundary_count, severance_energy,
    ledger_severance, equilibration_energy,
    build_pac_tree, ade_graphs,
    U238_CHAIN_ALPHAS, U238_CHAIN_LABELS, PLANCK_ENERGY_MEV,
    save_mr_results, _convert_numpy,
)


def test_T1_integer_multiples():
    """T1: Alpha energies as integer multiples of Xi * E_scale."""
    print("\n  T1: Alpha energies as integer multiples of Xi * E_scale(d)")
    results = {'description': 'Test if n = E_alpha / (Xi * E_Planck * phi^(-d)) is near-integer'}

    best_depth = None
    best_score = 0
    depth_results = []

    for d in range(3, 15):
        e_scale = PLANCK_ENERGY_MEV * PHI ** (-d)
        e_unit = XI_BALANCE * e_scale
        if e_unit <= 0:
            continue

        n_values = [E / e_unit for E in U238_CHAIN_ALPHAS]
        residuals = [abs(n - round(n)) for n in n_values]
        n_near_integer = sum(1 for r in residuals if r < 0.2)

        depth_results.append({
            'depth': d,
            'e_unit_mev': float(e_unit),
            'n_values': [float(n) for n in n_values],
            'residuals': [float(r) for r in residuals],
            'n_near_integer': n_near_integer,
        })

        if n_near_integer > best_score:
            best_score = n_near_integer
            best_depth = d

        if n_near_integer >= 6:
            print(f"    depth={d}: e_unit={e_unit:.3e} MeV, {n_near_integer}/8 near integer")

    passed = best_score >= 6  # >= 6 of 8 within 0.2 of integer
    results['depth_results'] = depth_results
    results['best_depth'] = best_depth
    results['best_score'] = best_score
    results['PASS'] = passed

    if best_depth is not None:
        print(f"    Best: depth={best_depth}, {best_score}/8 near integer")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_phi_ratio_clustering():
    """T2: Pairwise alpha energy ratios cluster near phi powers."""
    print("\n  T2: Pairwise alpha energy ratios cluster near phi powers vs random")
    results = {'description': 'Alpha energy ratios cluster near phi^k more than random'}

    alphas = U238_CHAIN_ALPHAS
    n = len(alphas)

    # Phi powers to check: phi^{-3} to phi^{3}
    phi_powers = [PHI ** k for k in range(-3, 4)]

    def phi_clustering_metric(energies):
        """Average minimum distance to nearest phi power for all pairwise ratios."""
        ratios = []
        for i in range(len(energies)):
            for j in range(i + 1, len(energies)):
                r = energies[i] / energies[j]
                ratios.append(r)
                ratios.append(1.0 / r)  # Include inverse
        min_dists = []
        for r in ratios:
            dists = [abs(np.log(r) - np.log(p)) for p in phi_powers if p > 0]
            min_dists.append(min(dists))
        return np.mean(min_dists)

    alpha_metric = phi_clustering_metric(alphas)

    # Random comparison: 10000 sets of 8 energies from [3.5, 9.0] MeV
    rng = np.random.RandomState(42)
    random_metrics = []
    for _ in range(10000):
        random_energies = rng.uniform(3.5, 9.0, size=n)
        random_metrics.append(phi_clustering_metric(random_energies))

    percentile = 100 - percentileofscore(random_metrics, alpha_metric)
    passed = percentile > 95  # Alpha clustering better than 95% of random

    results['alpha_metric'] = float(alpha_metric)
    results['random_mean'] = float(np.mean(random_metrics))
    results['random_std'] = float(np.std(random_metrics))
    results['percentile'] = float(percentile)
    results['PASS'] = passed
    print(f"    Alpha metric: {alpha_metric:.4f}, Random mean: {np.mean(random_metrics):.4f}")
    print(f"    Percentile (higher=better clustering): {percentile:.1f}%")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_daughter_relaxation():
    """T3: Daughter graph relaxation timescale correlates with graph size."""
    print("\n  T3: Daughter graph relaxation timescale vs graph size")
    results = {'description': 'Relaxation steps correlate with graph size (|r| > 0.8)'}

    sizes = []
    relaxation_times = []

    for depth in range(2, 7):
        A = build_pac_tree(depth)
        n = A.shape[0]

        # Sever root
        sev = ledger_severance(A, 0)
        daughter = sev['daughter_adj']
        nd = daughter.shape[0]
        if nd < 2:
            continue

        # Initialize uniform state, measure steps to equilibrium
        state = np.ones(nd) / nd
        target_entropy = -np.sum(state * np.log(state + 1e-30))

        for step in range(1, 501):
            new_state = np.zeros(nd)
            for v in range(nd):
                neighbors = np.where(daughter[v] > 0)[0]
                if len(neighbors) > 0:
                    share = state[v] * INV_PHI / len(neighbors)
                    new_state[v] += state[v] * (1 - INV_PHI)
                    for u in neighbors:
                        new_state[u] += share
                else:
                    new_state[v] += state[v]
            state = new_state

            entropy = -np.sum(state * np.log(state + 1e-30))
            if abs(entropy - target_entropy) / abs(target_entropy) > 0.01:
                target_entropy = entropy
            else:
                break

        sizes.append(nd)
        relaxation_times.append(step)
        print(f"    depth={depth}: daughter_size={nd}, relaxation_steps={step}")

    if len(sizes) >= 3:
        r = np.corrcoef(sizes, relaxation_times)[0, 1]
    else:
        r = 0.0

    passed = abs(r) > 0.8
    results['sizes'] = sizes
    results['relaxation_times'] = relaxation_times
    results['pearson_r'] = float(r)
    results['PASS'] = passed
    print(f"    Pearson r = {r:.4f} -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_alpha_proton_ratio():
    """T4: Alpha vs proton emission energy ratio ~ phi^(delta_d)."""
    print("\n  T4: Alpha vs proton emission energy ratio ~ phi^(delta_d)")
    results = {'description': 'E_alpha / E_proton within 30% of phi^(delta_d) for some integer delta_d'}

    # Alpha particle binding energy: 28.3 MeV
    # Typical alpha decay energy: ~5 MeV (average of chain)
    # Proton separation energy for heavy nuclei: ~6-8 MeV (higher due to Coulomb)
    alpha_avg = np.mean(U238_CHAIN_ALPHAS)  # ~5.7 MeV
    proton_sep_u238 = 7.0  # MeV (approximate proton separation energy for heavy nuclei)

    ratio = alpha_avg / proton_sep_u238

    # Check against phi^k for k in {-3, ..., 3}
    best_k = None
    best_error = float('inf')
    for k in range(-3, 4):
        phi_k = PHI ** k
        error = abs(ratio - phi_k) / phi_k
        if error < best_error:
            best_error = error
            best_k = k

    passed = best_error < 0.30  # Within 30%
    results['alpha_avg_mev'] = float(alpha_avg)
    results['proton_sep_mev'] = float(proton_sep_u238)
    results['ratio'] = float(ratio)
    results['best_k'] = best_k
    results['best_phi_k'] = float(PHI ** best_k) if best_k is not None else None
    results['best_error'] = float(best_error)
    results['PASS'] = passed
    print(f"    ratio = {ratio:.4f}, best phi^{best_k} = {PHI**best_k:.4f}, error = {best_error:.1%}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_02: Alpha Decay as Discrete Ledger Severance")
    print("=" * 60)

    t1 = test_T1_integer_multiples()
    t2 = test_T2_phi_ratio_clustering()
    t3 = test_T3_daughter_relaxation()
    t4 = test_T4_alpha_proton_ratio()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_02_alpha_decay_discrete_severance',
        'timestamp': datetime.now().isoformat(),
        'block': 'A',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_02_alpha_decay_discrete_severance')
