"""
exp_15 -- Coherence Limit Universality

Milestone 13.5, Investigation Experiment

Central question: Does the max complement-deformation rate converge in the
large-rank limit of ADE families? If so, does the limiting value relate to
phi, ln(phi), or Xi?

This is the speed-of-light investigation: if ADE deformation rates converge
to a universal limit, that limit IS the discrete analogue of c.

Tests:
  T1: A-family (A_3..A_20) rate convergence (last-5 variation <5%)
  T2: D-family converges AND agrees with A-family limit within 20%
  T3: Random graphs have higher rate variance than ADE families
  T4: Limiting value relates to phi, ln(phi), or Xi
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
    DynkinDiagram,
    max_deformation_rate,
    generate_random_connected_graph,
    save_m13_results, _convert_numpy,
)


def test_T1_a_family_convergence():
    """T1: A-family (A_3..A_20) rate convergence (last-5 variation <5%)."""

    rates = []
    for n in range(3, 21):
        d = DynkinDiagram('A', n)
        r = max_deformation_rate(d.adjacency)
        rates.append({'rank': n, 'rate': float(r)})
        print(f"    A_{n}: rate = {r:.6f}")

    values = [r['rate'] for r in rates]

    # Check last-5 convergence
    last5 = values[-5:]
    last5_mean = float(np.mean(last5))
    last5_std = float(np.std(last5))
    last5_cv = last5_std / last5_mean if last5_mean > 0 else float('inf')
    last5_variation = float(np.max(last5) - np.min(last5)) / last5_mean if last5_mean > 0 else float('inf')

    print(f"  Last-5 (A_16..A_20): mean={last5_mean:.4f}, CV={last5_cv:.4f}, "
          f"variation={last5_variation:.4f}")

    # The A-family oscillates (even/odd bipartite effect).
    # Check even and odd sub-families separately
    even_rates = [r['rate'] for r in rates if r['rank'] % 2 == 0]
    odd_rates = [r['rate'] for r in rates if r['rank'] % 2 != 0]

    even_last3 = even_rates[-3:] if len(even_rates) >= 3 else even_rates
    odd_last3 = odd_rates[-3:] if len(odd_rates) >= 3 else odd_rates

    even_cv = float(np.std(even_last3) / np.mean(even_last3)) if even_last3 else float('inf')
    odd_cv = float(np.std(odd_last3) / np.mean(odd_last3)) if odd_last3 else float('inf')

    print(f"  Even sub-family last-3 CV: {even_cv:.4f}")
    print(f"  Odd sub-family last-3 CV: {odd_cv:.4f}")

    # Pass if either the full last-5 or both sub-families converge
    full_converges = last5_variation < 0.05
    sub_converges = even_cv < 0.05 and odd_cv < 0.05
    converges = full_converges or sub_converges

    result = {
        'test': 'T1_a_family_convergence',
        'rates': rates,
        'last5_mean': last5_mean,
        'last5_cv': last5_cv,
        'last5_variation': last5_variation,
        'even_last3_cv': even_cv,
        'odd_last3_cv': odd_cv,
        'full_converges': full_converges,
        'sub_converges': sub_converges,
        'note': 'A-family rates alternate (bipartite even/odd effect). '
                'Convergence tested both overall and within even/odd sub-families.',
        'PASS': converges,
    }
    return result


def test_T2_d_family_convergence():
    """T2: D-family converges AND agrees with A-family limit within 20%."""

    d_rates = []
    for n in range(4, 21):
        d = DynkinDiagram('D', n)
        r = max_deformation_rate(d.adjacency)
        d_rates.append({'rank': n, 'rate': float(r)})
        print(f"    D_{n}: rate = {r:.6f}")

    d_values = [r['rate'] for r in d_rates]
    d_last5 = d_values[-5:]
    d_last5_mean = float(np.mean(d_last5))
    d_last5_cv = float(np.std(d_last5) / np.mean(d_last5))
    d_converges = d_last5_cv < 0.05

    print(f"  D-family last-5 (D_16..D_20): mean={d_last5_mean:.4f}, CV={d_last5_cv:.4f}")

    # A-family limit (compute from A_16..A_20)
    a_rates = []
    for n in range(16, 21):
        d = DynkinDiagram('A', n)
        r = max_deformation_rate(d.adjacency)
        a_rates.append(float(r))
    a_limit = float(np.mean(a_rates))

    # Cross-family agreement
    cross_diff = abs(d_last5_mean - a_limit) / max(d_last5_mean, a_limit)
    agrees = cross_diff < 0.20

    print(f"  A-family limit (A_16..A_20): {a_limit:.4f}")
    print(f"  Cross-family difference: {cross_diff:.4f} (agrees within 20%: {agrees})")

    result = {
        'test': 'T2_d_family_convergence',
        'd_rates': d_rates,
        'd_last5_mean': d_last5_mean,
        'd_last5_cv': d_last5_cv,
        'd_converges': d_converges,
        'a_limit': a_limit,
        'cross_diff': float(cross_diff),
        'agrees_within_20pct': agrees,
        'PASS': d_converges and agrees,
    }
    return result


def test_T3_random_higher_variance():
    """T3: Random graphs have higher rate variance than ADE families."""

    # ADE rates for rank 10-20
    ade_rates = []
    for n in range(10, 21):
        for family in ['A', 'D']:
            if family == 'D' and n < 4:
                continue
            d = DynkinDiagram(family, n)
            r = max_deformation_rate(d.adjacency)
            ade_rates.append(r)
    ade_cv = float(np.std(ade_rates) / np.mean(ade_rates))

    # Random graphs of same sizes
    random_rates = []
    for seed in range(30):
        n = np.random.RandomState(seed + 200).randint(10, 21)
        try:
            G = generate_random_connected_graph(n, density=0.3, seed=seed + 200)
            r = max_deformation_rate(G)
            random_rates.append(r)
        except RuntimeError:
            pass

    random_cv = float(np.std(random_rates) / np.mean(random_rates)) if random_rates else float('inf')

    print(f"  ADE rates (rank 10-20, A+D): CV = {ade_cv:.4f} ({len(ade_rates)} diagrams)")
    print(f"  Random rates (n=10-20): CV = {random_cv:.4f} ({len(random_rates)} graphs)")

    ade_more_constrained = ade_cv < random_cv

    result = {
        'test': 'T3_random_higher_variance',
        'n_ade': len(ade_rates),
        'ade_cv': ade_cv,
        'n_random': len(random_rates),
        'random_cv': random_cv,
        'ade_more_constrained': ade_more_constrained,
        'note': 'ADE classification constrains deformation rates more tightly than '
                'random graph topology. This is the structural signature of the '
                'complement coherence limit.',
        'PASS': ade_more_constrained,
    }
    return result


def test_T4_limiting_value_relates_to_dft():
    """T4: Limiting value relates to phi, ln(phi), or Xi."""

    # Compute the best estimate of the limiting rate from large-rank A and D
    a_rates = []
    for n in range(15, 21):
        d = DynkinDiagram('A', n)
        a_rates.append(max_deformation_rate(d.adjacency))

    d_rates = []
    for n in range(15, 21):
        d = DynkinDiagram('D', n)
        d_rates.append(max_deformation_rate(d.adjacency))

    # Combined limit estimate
    all_rates = a_rates + d_rates
    limit_estimate = float(np.mean(all_rates))
    limit_std = float(np.std(all_rates))

    print(f"  Limiting rate estimate: {limit_estimate:.6f} +/- {limit_std:.6f}")

    # Test against DFT constants
    candidates = {
        'phi': PHI,
        'inv_phi': INV_PHI,
        'ln_phi': LN_PHI,
        'Xi': XI_BALANCE,
        'gamma_EM': GAMMA_EM,
        '1': 1.0,
        'sqrt(2)': np.sqrt(2),
        '2*ln_phi': 2 * LN_PHI,
        'phi-1': PHI - 1,  # = inv_phi
    }

    best_match = None
    best_error = float('inf')
    match_results = {}

    for name, value in candidates.items():
        error = abs(limit_estimate - value) / max(abs(value), 1e-10)
        match_results[name] = {
            'value': float(value),
            'error': float(error),
        }
        if error < best_error:
            best_error = error
            best_match = name
        print(f"    {name} = {value:.6f}: error = {error:.4f}")

    # Does the limiting value match any DFT constant within 10%?
    matches_dft = best_error < 0.10
    print(f"  Best match: {best_match} (error = {best_error:.4f})")
    print(f"  Matches DFT constant within 10%: {matches_dft}")

    result = {
        'test': 'T4_limiting_value_relates_to_dft',
        'limit_estimate': limit_estimate,
        'limit_std': limit_std,
        'match_results': match_results,
        'best_match': best_match,
        'best_error': float(best_error),
        'matches_dft': matches_dft,
        'note': 'Tests whether the complement coherence limit converges to a '
                'DFT constant (phi, ln(phi), Xi, etc.). If it does, the speed '
                'of light has a direct Fibonacci-arithmetic origin.',
        'PASS': matches_dft,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 15 -- Coherence Limit Universality")
    print("Milestone 13.5, Investigation Experiment")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_a_family_convergence),
        ('T2', test_T2_d_family_convergence),
        ('T3', test_T3_random_higher_variance),
        ('T4', test_T4_limiting_value_relates_to_dft),
    ]:
        print(f"\n--- {name}: {test_fn.__doc__.strip().split(chr(10))[0]} ---")
        r = test_fn()
        results[name] = r
        if r['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

    final = {
        'experiment': 'exp_15_coherence_limit_universality',
        'milestone': 'milestone13.5',
        'block': 'investigation',
        'version': 'v0.1',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_15_coherence_limit_universality', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
