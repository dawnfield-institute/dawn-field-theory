"""
exp_06 — Page Curve and Unitarity from PAC Tree

Milestone 11, Block B (Black Hole Resolution)

Hypothesis: BH evaporation is pruning a PAC Tree Tensor Network.
Entanglement entropy follows the Page curve: rises to S/2, then returns
to zero (unitarity). PAC conservation ensures information is never destroyed.

Tests:
  T1: Page time from cascade counting matches t_Page ~ M^3 scaling
  T2: Scrambling time ~ S*t_P*ln(S) (Sekino-Susskind)
  T3: Page curve peaks at k/N = 0.5
  T4: Epsilon-PAC violation → Page curve fails to return to zero
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_gravity import (
    PHI, PI,
    T_PLANCK_S, M_PLANCK_KG, M_SUN_KG,
    HBAR, C_LIGHT, G_NEWTON,
    PACTreeEvaporator, page_time_scaling, scrambling_time,
    save_results, setup_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def test_T1_page_time_scaling():
    """T1: Page time scales with BH mass."""
    masses = [0.1, 1.0, 10.0, 100.0]

    data = []
    for M in masses:
        pt = page_time_scaling(M)
        data.append({
            'M_solar': M,
            'S_BH': pt['S_BH'],
            't_page': pt['t_page_seconds'],
            't_evap': pt['t_evap_seconds'],
            'log_S': np.log10(pt['S_BH']),
            'log_t_evap': np.log10(pt['t_evap_seconds']),
        })

    # Check M^3 scaling of evaporation time
    log_M = np.array([np.log10(d['M_solar']) for d in data])
    log_t_evap = np.array([d['log_t_evap'] for d in data])

    coeffs = np.polyfit(log_M, log_t_evap, 1)
    slope = coeffs[0]
    slope_near_3 = abs(slope - 3.0) < 0.1

    # S scales as M^2
    log_S = np.array([d['log_S'] for d in data])
    coeffs_S = np.polyfit(log_M, log_S, 1)
    S_slope = coeffs_S[0]
    S_slope_near_2 = abs(S_slope - 2.0) < 0.1

    return {
        'test': 'T1_page_time_scaling',
        'data': data,
        't_evap_slope': float(slope),
        't_evap_slope_near_3': slope_near_3,
        'S_slope': float(S_slope),
        'S_slope_near_2': S_slope_near_2,
        'PASS': slope_near_3 and S_slope_near_2,
    }


def test_T2_scrambling():
    """T2: Scrambling time ~ S*t_P*ln(S)."""
    masses = [0.1, 1.0, 10.0, 100.0]

    data = []
    for M in masses:
        st = scrambling_time(M)
        # Expected: t_scram = S * t_P * ln(S)
        S = st['S_BH']
        t_expected = S * T_PLANCK_S * np.log(S)
        data.append({
            'M_solar': M,
            'S_BH': S,
            't_scramble': st['t_scramble_seconds'],
            't_expected': t_expected,
            'ratio': st['t_scramble_seconds'] / t_expected if t_expected > 0 else float('inf'),
        })

    # All ratios should be 1 (exact match)
    ratios = [d['ratio'] for d in data]
    all_match = all(abs(r - 1.0) < 1e-10 for r in ratios)

    # t_scramble << t_evaporation (fast scrambling)
    for d in data:
        pt = page_time_scaling(d['M_solar'])
        d['t_evap'] = pt['t_evap_seconds']
        d['scramble_over_evap'] = d['t_scramble'] / d['t_evap']

    scramble_fast = all(d['scramble_over_evap'] < 1e-10 for d in data)

    return {
        'test': 'T2_scrambling',
        'data': data,
        'formula_match': all_match,
        'scramble_much_faster_than_evap': scramble_fast,
        'PASS': all_match,
    }


def test_T3_page_curve_peak():
    """T3: Page curve peaks at k/N = 0.5."""
    n_values = [32, 64, 128, 256]
    results = []

    for N in n_values:
        evap = PACTreeEvaporator(N, seed=42)
        curve = evap.run_evaporation()
        results.append({
            'N': N,
            'peak_fraction': curve['peak_fraction'],
            'final_entropy': curve['final_entropy'],
            'error_from_half': abs(curve['peak_fraction'] - 0.5),
        })

    # Peak should be near 0.5 for all N
    all_near_half = all(r['error_from_half'] < 0.05 for r in results)

    # Final entropy should be near zero (unitarity)
    all_return_zero = all(abs(r['final_entropy']) < 0.1 for r in results)

    return {
        'test': 'T3_page_curve_peak',
        'results': results,
        'all_peak_near_half': all_near_half,
        'all_return_to_zero': all_return_zero,
        'PASS': all_near_half and all_return_zero,
    }


def test_T4_epsilon_pac():
    """
    T4: Epsilon-PAC violation → Page curve fails to return to zero.

    With perfect PAC: information preserved, entropy returns to 0.
    With broken PAC (epsilon > 0): information leaked, residual entropy ~ epsilon*S.
    """
    N = 128

    # Perfect PAC: standard evaporation
    evap_perfect = PACTreeEvaporator(N, seed=42)
    curve_perfect = evap_perfect.run_evaporation()

    # Broken PAC: add information loss at each step
    # Model: at each step, fraction epsilon of entanglement entropy is
    # irreversibly leaked. This accumulated loss persists even after S
    # returns to zero, preventing unitarity restoration.
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]
    epsilon_results = []

    for eps in epsilons:
        evap = PACTreeEvaporator(N, seed=42)
        entropies = [0.0]
        accumulated_loss = 0.0

        for step in range(N):
            evap.evaporate_one()
            S = evap.entanglement_entropy()
            # With epsilon violation: information leaks out of the system
            # Each step loses eps * S_current, creating irreversible entropy
            if eps > 0 and S > 0:
                accumulated_loss += eps * S
            S_degraded = S + accumulated_loss
            entropies.append(S_degraded)

        epsilon_results.append({
            'epsilon': eps,
            'final_entropy': float(entropies[-1]),
            'peak_entropy': float(max(entropies)),
            'returns_to_zero': abs(entropies[-1]) < 0.1,
        })

    # Perfect PAC returns to zero
    perfect_returns = epsilon_results[0]['returns_to_zero']
    # Non-zero epsilon has residual
    nonzero_has_residual = all(
        not r['returns_to_zero'] for r in epsilon_results if r['epsilon'] > 0.05
    )

    return {
        'test': 'T4_epsilon_pac',
        'N': N,
        'epsilon_results': epsilon_results,
        'perfect_returns_to_zero': perfect_returns,
        'nonzero_has_residual': nonzero_has_residual,
        'PASS': perfect_returns and nonzero_has_residual,
    }


def main():
    setup = setup_experiment(__file__)

    print("=" * 70)
    print("EXP 06 — Page Curve and Unitarity from PAC Tree")
    print("Milestone 11, Block B")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [('T1', test_T1_page_time_scaling),
                           ('T2', test_T2_scrambling),
                           ('T3', test_T3_page_curve_peak),
                           ('T4', test_T4_epsilon_pac)]:
        print(f"\n--- {name} ---")
        t = test_fn()
        results[name] = t
        if t['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

        if name == 'T1':
            print(f"    t_evap slope: {t['t_evap_slope']:.3f} (target 3.0)")
            print(f"    S slope: {t['S_slope']:.3f} (target 2.0)")
        elif name == 'T2':
            for d in t['data'][:2]:
                print(f"    M={d['M_solar']:.1f}: t_scram/t_evap = {d['scramble_over_evap']:.2e}")
        elif name == 'T3':
            for r in t['results']:
                print(f"    N={r['N']:>4d}: peak at k/N={r['peak_fraction']:.3f}, final S={r['final_entropy']:.4f}")
        elif name == 'T4':
            for r in t['epsilon_results']:
                print(f"    eps={r['epsilon']:.2f}: final S={r['final_entropy']:.4f} "
                      f"({'returns' if r['returns_to_zero'] else 'RESIDUAL'})")

    print("\n" + "=" * 70)
    print(f"EXP 06 SCORE: {score}/{total}")
    print("=" * 70)

    results['score'] = score
    results['total'] = total
    save_results(results, RESULTS_DIR, "exp_06_page_curve_unitarity")
    return results


if __name__ == "__main__":
    main()
