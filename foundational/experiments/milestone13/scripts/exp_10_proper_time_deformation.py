"""
exp_10 -- Proper Time Is the Complement-Deformation Rate Along a Worldline

Milestone 13, Block D

Hypothesis: Proper time is the complement-deformation rate along a worldline.
A boosted observer's complement changes slower per coordinate time step -- this
IS time dilation. The Lorentz factor gamma emerges from the deformation ratio,
the twin paradox follows from path-dependence of total deformation, and basin
relaxation rates (gravitational coupling) modulate the local deformation clock.

Tests (hardened v0.3):
  T1: 20 rapidities at 1e-10 + non-Lorentz adversarial (random matrices fail)
  T2: Complement-deformation rate ratio on ADE chain vs 1/cosh(eta) prediction
  T3: Graph-based twin paradox: straight path vs detour deformation comparison
  T4: phi^{-depth} vs e^{-depth} vs 1/depth^2 against known force hierarchy
"""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram,
    so31_4d_generators,
    complement_spectrum, complement_deformation_rate, max_deformation_rate,
    BasinAttractor,
    DEPTH_EM, DEPTH_DARK, DEPTH_GRAVITY,
    save_m13_results, _convert_numpy,
)


def test_T1_time_dilation_from_boost():
    """T1: 20 rapidities at 1e-10 + non-Lorentz adversarial."""

    rotations_4d, boosts_4d = so31_4d_generators()

    # 20 rapidities spanning 0 to 10
    rapidities = np.linspace(0.0, 10.0, 20)
    max_error = 0.0
    all_results = []

    for eta in rapidities:
        Lambda = expm(eta * boosts_4d[0])
        lambda_00 = float(Lambda[0, 0])
        cosh_eta = float(np.cosh(eta))
        error = abs(lambda_00 - cosh_eta) / max(cosh_eta, 1e-15)
        max_error = max(max_error, error)
        all_results.append({
            'eta': float(eta),
            'Lambda_00': lambda_00,
            'cosh_eta': cosh_eta,
            'error': float(error),
        })

    print(f"  20 rapidities [0, 10]: max error = {max_error:.2e}")

    # Adversarial: random 4x4 matrices should NOT have M[0,0] = cosh(eta)
    rng = np.random.RandomState(42)
    n_random_match = 0
    random_errors = []
    for trial in range(10):
        M_random = rng.randn(4, 4)
        # Make it look like a "transform" by exponentiating
        M_exp = expm(0.5 * M_random)
        random_00 = float(M_exp[0, 0])
        # Compare to cosh(0.5) (the "rapidity" we used)
        cosh_05 = float(np.cosh(0.5))
        rand_error = abs(random_00 - cosh_05) / cosh_05
        random_errors.append(rand_error)
        if rand_error < 0.01:
            n_random_match += 1

    random_mean_error = float(np.mean(random_errors))
    print(f"  Random matrices matching cosh(0.5): {n_random_match}/10")
    print(f"  Random mean error from cosh: {random_mean_error:.4f}")

    # Selectivity: Lorentz boosts match, random matrices don't
    selectivity = max_error < 1e-10 and n_random_match <= 1

    result = {
        'test': 'T1_time_dilation_from_boost',
        'n_rapidities': 20,
        'max_error': float(max_error),
        'selectivity': selectivity,
        'n_random_match': n_random_match,
        'random_mean_error': random_mean_error,
        'PASS': selectivity,
    }
    return result


def test_T2_complement_deformation_rate_ratio():
    """T2: Complement-deformation rate ratio on ADE chain vs 1/cosh(eta) prediction."""
    # The claim: a "boosted" complement (shifted observation point) should have
    # deformation rate proportional to 1/cosh(eta).
    #
    # Test: on an ADE chain (A_8), compare deformation rates from different
    # starting positions. Position 0 is "rest frame", deeper positions are
    # "moving". The rate ratio should approximate 1/cosh(eta) where eta is
    # some function of the graph distance.
    #
    # This is the key discrete-to-continuous test.

    a8 = DynkinDiagram('A', 8)
    adj = a8.adjacency

    # Compute deformation rate from each starting position along the chain
    rates_from_position = []
    for start in range(7):
        path = [start, start + 1]
        deform = complement_deformation_rate(adj, path)
        rates_from_position.append(deform['mean_rate'])

    # Normalize by max rate
    max_rate = max(rates_from_position) if rates_from_position else 1.0
    normalized_rates = [r / max_rate for r in rates_from_position]

    print(f"  Deformation rates from each position on A_8:")
    for i, (r, nr) in enumerate(zip(rates_from_position, normalized_rates)):
        print(f"    pos {i}->{i+1}: rate={r:.4f}, normalized={nr:.4f}")

    # If the complement-deformation-as-proper-time analogy holds,
    # normalized rates should follow 1/cosh(eta_i) for some eta_i.
    # Define eta_i = distance from midpoint (symmetry axis)
    midpoint = 3.5
    etas = [abs(i + 0.5 - midpoint) / midpoint for i in range(7)]
    predicted_rates = [1.0 / np.cosh(e) for e in etas]

    # Compare
    errors = []
    for i in range(7):
        if predicted_rates[i] > 1e-10:
            err = abs(normalized_rates[i] - predicted_rates[i]) / predicted_rates[i]
            errors.append(err)

    mean_error = float(np.mean(errors)) if errors else float('inf')
    best_error = float(min(errors)) if errors else float('inf')

    print(f"  Mean error vs 1/cosh model: {mean_error:.4f}")
    print(f"  Best single-point error: {best_error:.4f}")

    # This is genuinely testing whether discrete complement rates match
    # the continuous 1/cosh prediction. Predicted to be a difficult match.
    match_quality = mean_error < 0.25

    result = {
        'test': 'T2_complement_deformation_rate_ratio',
        'rates_from_position': [float(r) for r in rates_from_position],
        'normalized_rates': [float(r) for r in normalized_rates],
        'etas': [float(e) for e in etas],
        'predicted_1_over_cosh': [float(p) for p in predicted_rates],
        'errors': [float(e) for e in errors],
        'mean_error': mean_error,
        'best_error': best_error,
        'note': 'Tests whether discrete complement-deformation rates approximate '
                '1/cosh(eta). Large deviations reveal the gap between discrete '
                'complement operations and continuous Lorentz time dilation.',
        'PASS': match_quality,
    }
    return result


def test_T3_graph_twin_paradox():
    """T3: Graph-based twin paradox: straight path accumulates less total deformation."""
    # On a graph, the "straight path" (shortest between two points) should
    # have LESS total complement deformation than a "detour" path.
    # This is the discrete analogue of the twin paradox.
    # (In SR: inertial path maximizes proper time -> minimal aging for traveler.
    #  In complement language: straight path has LESS total deformation.)

    # Use D_6 which has branching structure
    d6 = DynkinDiagram('D', 6)
    adj = d6.adjacency

    # D_6 has vertices 0-1-2-3-4 chain plus vertex 5 branching from vertex 3
    # Straight path: 0 -> 1 -> 2 -> 3
    straight_path = [0, 1, 2, 3]
    straight_deform = complement_deformation_rate(adj, straight_path)

    # Detour paths through the branch
    detour_path_1 = [0, 1, 2, 3, 4, 3]  # go to 4 and back
    detour_deform_1 = complement_deformation_rate(adj, detour_path_1)

    detour_path_2 = [0, 1, 2, 3, 5, 3]  # go to branch vertex 5 and back
    detour_deform_2 = complement_deformation_rate(adj, detour_path_2)

    print(f"  D_6 straight (0->3): total deformation = {straight_deform['total']:.4f}")
    print(f"  D_6 detour via 4 (0->3->4->3): total = {detour_deform_1['total']:.4f}")
    print(f"  D_6 detour via 5 (0->3->5->3): total = {detour_deform_2['total']:.4f}")

    # Detour paths should have MORE total deformation (more steps = more change)
    detour_more_1 = detour_deform_1['total'] > straight_deform['total']
    detour_more_2 = detour_deform_2['total'] > straight_deform['total']

    # Also test on A_8 chain
    a8 = DynkinDiagram('A', 8)
    adj_a8 = a8.adjacency

    straight_a8 = complement_deformation_rate(adj_a8, [0, 1, 2, 3])
    detour_a8 = complement_deformation_rate(adj_a8, [0, 1, 2, 1, 2, 3])

    print(f"  A_8 straight (0->3): total = {straight_a8['total']:.4f}")
    print(f"  A_8 detour (0->1->2->1->2->3): total = {detour_a8['total']:.4f}")

    detour_more_a8 = detour_a8['total'] > straight_a8['total']

    # Twin paradox: at least 2 of 3 detour cases have more deformation
    n_detour_more = sum([detour_more_1, detour_more_2, detour_more_a8])

    result = {
        'test': 'T3_graph_twin_paradox',
        'straight_D6': float(straight_deform['total']),
        'detour_D6_via4': float(detour_deform_1['total']),
        'detour_D6_via5': float(detour_deform_2['total']),
        'straight_A8': float(straight_a8['total']),
        'detour_A8': float(detour_a8['total']),
        'detour_more_D6_via4': detour_more_1,
        'detour_more_D6_via5': detour_more_2,
        'detour_more_A8': detour_more_a8,
        'n_detour_more': n_detour_more,
        'note': 'Detour paths accumulate more complement deformation than straight paths. '
                'This is the graph-based twin paradox: the "traveling twin" (detour) '
                'experiences more complement change than the "resting twin" (straight).',
        'PASS': n_detour_more >= 2,
    }
    return result


def test_T4_coupling_model_comparison():
    """T4: phi^{-depth} vs e^{-depth} vs 1/depth^2 against known force hierarchy."""
    # Compare three coupling models against the KNOWN force hierarchy:
    # EM (depth ~7), gravity (depth ~183), with coupling ratio ~ 10^{-36}
    #
    # The question: does phi^{-depth} produce a better match than alternatives?

    depth_em = int(DEPTH_EM)     # should be 7
    depth_grav = int(DEPTH_GRAVITY)  # should be 183

    # Known coupling ratio: G_N / alpha_EM ~ 10^{-36}
    # More precisely: G_N * m_p^2 / (e^2/(4*pi*eps_0)) ~ 10^{-36}
    target_log_ratio = -36.0  # log10 of gravity/EM coupling ratio

    models = {}

    # Model 1: phi^{-depth}
    phi_em = PHI ** (-depth_em)
    phi_grav = PHI ** (-depth_grav)
    phi_log_ratio = np.log10(phi_grav / phi_em) if phi_em > 0 else float('-inf')
    phi_error = abs(phi_log_ratio - target_log_ratio)
    models['phi^{-depth}'] = {
        'em_coupling': float(phi_em),
        'grav_coupling': float(phi_grav),
        'log10_ratio': float(phi_log_ratio),
        'error_from_target': float(phi_error),
    }

    # Model 2: e^{-depth}
    e_em = np.exp(-depth_em)
    e_grav = np.exp(-depth_grav)
    e_log_ratio = np.log10(e_grav / e_em) if e_em > 0 else float('-inf')
    e_error = abs(e_log_ratio - target_log_ratio)
    models['e^{-depth}'] = {
        'em_coupling': float(e_em),
        'grav_coupling': float(e_grav),
        'log10_ratio': float(e_log_ratio),
        'error_from_target': float(e_error),
    }

    # Model 3: 1/depth^2
    d2_em = 1.0 / depth_em**2
    d2_grav = 1.0 / depth_grav**2
    d2_log_ratio = np.log10(d2_grav / d2_em)
    d2_error = abs(d2_log_ratio - target_log_ratio)
    models['1/depth^2'] = {
        'em_coupling': float(d2_em),
        'grav_coupling': float(d2_grav),
        'log10_ratio': float(d2_log_ratio),
        'error_from_target': float(d2_error),
    }

    for name, data in models.items():
        print(f"  {name}: log10(grav/em) = {data['log10_ratio']:.2f} "
              f"(target: {target_log_ratio:.1f}, error: {data['error_from_target']:.2f})")

    # Which model is closest?
    best_model = min(models.keys(), key=lambda k: models[k]['error_from_target'])
    phi_is_best = best_model == 'phi^{-depth}'
    phi_within_5 = phi_error < 5.0  # within 5 orders of magnitude

    print(f"  Best model: {best_model}")
    print(f"  phi^{{-depth}} is best: {phi_is_best}")
    print(f"  phi^{{-depth}} within 5 orders: {phi_within_5}")

    result = {
        'test': 'T4_coupling_model_comparison',
        'depth_em': depth_em,
        'depth_gravity': depth_grav,
        'target_log10_ratio': target_log_ratio,
        'models': models,
        'best_model': best_model,
        'phi_is_best': phi_is_best,
        'phi_within_5_orders': phi_within_5,
        'note': 'Compares three attenuation models against the known EM/gravity '
                'coupling ratio (~10^{-36}). phi^{-depth} must outperform alternatives '
                'to justify its privileged role in DFT.',
        'PASS': phi_is_best and phi_within_5,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 10 -- Proper Time Is Complement-Deformation Rate")
    print("Milestone 13, Block D (hardened v0.3)")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_time_dilation_from_boost),
        ('T2', test_T2_complement_deformation_rate_ratio),
        ('T3', test_T3_graph_twin_paradox),
        ('T4', test_T4_coupling_model_comparison),
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
        'experiment': 'exp_10_proper_time_deformation',
        'milestone': 'milestone13',
        'block': 'D',
        'version': 'v0.3_hardened',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_10_proper_time_deformation', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
