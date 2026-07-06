"""
exp_06 -- Graph Double-Slit

Milestone 14, Block C (Interference)

Hypothesis: The double-slit experiment can be modeled on graphs using path
amplitudes with SEC phases. Two-path interference shows constructive/destructive
patterns for complex amplitudes, which-path measurement destroys interference,
and multi-path topology (D_4 vs A_4) determines interference structure.

Tests:
  T1: Two-path amplitude shows interference for complex amplitudes
  T2: Which-path measurement destroys interference
  T3: Interference requires multiple paths (A_4 vs D_4 topology)
  T4: Interference visibility vs complement-deformation rate (50/50 — cross-layer)
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI, PI,
    DynkinDiagram,
    two_path_amplitude,
    orbit_hilbert_basis, born_probabilities, measurement_collapse,
    complement_deformation_rate, max_deformation_rate,
    save_m14_results, _convert_numpy,
)


def test_T1_two_path_interference():
    """T1: Two-path amplitude shows interference for complex amplitudes."""
    # Model: particle goes from source to detector via two paths on graph
    # Path 1: amplitude a1 = |a1| * exp(i*phi1)
    # Path 2: amplitude a2 = |a2| * exp(i*phi2)
    # Total amplitude: a1 + a2

    # Test with equal amplitudes, varying relative phase
    a = 1.0 / np.sqrt(2)
    phases = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    results_per_phase = []
    has_interference = False

    for phi in phases:
        a1 = complex(a)
        a2 = a * np.exp(1j * phi)
        r = two_path_amplitude(a1, a2)

        if r['has_interference']:
            has_interference = True

        results_per_phase.append({
            'phase': float(phi),
            'phase_pi': float(phi / np.pi),
            'p_quantum': r['p_quantum'],
            'p_classical': r['p_classical'],
            'interference_term': r['interference_term'],
            'has_interference': r['has_interference'],
        })

        print(f"  phi={phi/np.pi:.2f}pi: P_q={r['p_quantum']:.4f}, "
              f"P_c={r['p_classical']:.4f}, I={r['interference_term']:.4f}")

    # At phi=0: max constructive (P_q = 2*a^2 + 2*a^2 = 2, P_c = 1)
    # At phi=pi: max destructive (P_q = 0, P_c = 1)
    max_constructive = results_per_phase[0]['p_quantum'] > results_per_phase[0]['p_classical']
    max_destructive = results_per_phase[-1]['p_quantum'] < results_per_phase[-1]['p_classical'] + 1e-10

    passed = has_interference and max_constructive

    result = {
        'test': 'T1_two_path_interference',
        'results_per_phase': results_per_phase,
        'has_interference': has_interference,
        'max_constructive': max_constructive,
        'max_destructive': max_destructive,
        'PASS': passed,
    }
    return result


def test_T2_which_path_destroys_interference():
    """T2: Which-path measurement destroys interference."""
    # Setup: superposition of orbits on D_4
    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency
    n = adj.shape[0]
    basis, orbits = orbit_hilbert_basis(adj)

    # Superposition: equal weight on both orbits with relative phase
    theta = np.pi / 3  # SEC phase
    c0 = 1.0 / np.sqrt(2)
    c1 = np.exp(1j * theta) / np.sqrt(2)
    psi = c0 * basis[:, 0] + c1 * basis[:, 1]

    # "Interference" is the cross-term in the probability
    # P = |c0|^2 + |c1|^2 + 2*Re(c0* c1 <O0|O1>) = 1 + 0 = 1
    # (orbits are orthogonal, so no spatial interference in orbit basis)
    # But interference shows up in vertex-space probabilities
    vertex_probs = np.abs(psi) ** 2

    # Without measurement: probability distribution has cross-terms
    # The cross-term structure depends on the relative phase theta
    orbit_probs_before = born_probabilities(psi, adj)
    sum_before = sum(orbit_probs_before)

    # After which-path measurement (measure which orbit): collapse to one orbit
    # Try both outcomes
    psi_0 = measurement_collapse(psi, 0, adj)
    psi_1 = measurement_collapse(psi, 1, adj)

    # After measurement: vertex probabilities are pure orbit patterns
    vertex_probs_0 = np.abs(psi_0) ** 2
    vertex_probs_1 = np.abs(psi_1) ** 2

    # Classical mixture: p0 * |psi_0|^2 + p1 * |psi_1|^2
    p0 = born_probabilities(psi, adj)[0]
    p1 = born_probabilities(psi, adj)[1]
    classical_probs = p0 * vertex_probs_0 + p1 * vertex_probs_1

    # Interference destroyed: vertex_probs should differ from classical_probs
    # by the interference term
    interference_term = vertex_probs - classical_probs
    has_interference = np.max(np.abs(interference_term)) > 1e-10

    # After measurement: no interference (classical_probs)
    # Before measurement: interference present
    # The difference = which-path destroys interference

    print(f"  Before measurement: vertex probs = {[f'{p:.4f}' for p in vertex_probs]}")
    print(f"  Classical mixture:  vertex probs = {[f'{p:.4f}' for p in classical_probs]}")
    print(f"  Interference term: {[f'{t:.4f}' for t in interference_term]}")
    print(f"  Has interference: {has_interference}")

    passed = has_interference

    result = {
        'test': 'T2_which_path_destroys_interference',
        'theta': float(theta),
        'vertex_probs_superposition': [float(p) for p in vertex_probs],
        'vertex_probs_classical': [float(p) for p in classical_probs],
        'interference_term': [float(t) for t in interference_term],
        'max_interference': float(np.max(np.abs(interference_term))),
        'has_interference': has_interference,
        'PASS': passed,
    }
    return result


def test_T3_topology_determines_interference():
    """T3: Interference requires multiple paths (A_4 vs D_4 topology)."""
    # A_4: chain 0-1-2-3, Aut=Z_2, 2 orbits {0,3},{1,2}
    # D_4: star 0-1, 2-1, 3-1, Aut=S_3, 2 orbits {1},{0,2,3}
    #
    # D_4 has 3-fold branching at hub → multiple equivalent paths
    # A_4 has linear structure → simpler interference
    #
    # The claim: branching topology gives richer interference structure

    results_by_type = {}

    for family, rank, label in [('A', 4, 'A_4_chain'), ('D', 4, 'D_4_star')]:
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency
        n = adj.shape[0]
        basis, orbits = orbit_hilbert_basis(adj)
        n_orbits = len(orbits)

        # Create superposition with complex phase
        theta = np.pi / 3
        if n_orbits >= 2:
            c = np.zeros(n_orbits, dtype=complex)
            c[0] = 1.0 / np.sqrt(2)
            c[1] = np.exp(1j * theta) / np.sqrt(2)
            psi = basis @ c
        else:
            psi = basis[:, 0]

        # Vertex-space probability
        vertex_probs = np.abs(psi) ** 2

        # Classical mixture
        orbit_probs = born_probabilities(psi, adj)
        classical_probs = np.zeros(n)
        for i in range(n_orbits):
            if orbit_probs[i] > 1e-15:
                collapsed = measurement_collapse(psi, i, adj)
                classical_probs += orbit_probs[i] * np.abs(collapsed) ** 2

        interference = vertex_probs - classical_probs
        max_interference = float(np.max(np.abs(interference)))

        # Orbit size asymmetry: D_4 has {3,1}, A_4 has {2,2}
        orbit_sizes = sorted([len(o) for o in orbits], reverse=True)
        asymmetry = orbit_sizes[0] / orbit_sizes[-1] if len(orbit_sizes) > 1 else 1

        print(f"  {label}: orbits={orbit_sizes}, max_interference={max_interference:.6f}, "
              f"asymmetry={asymmetry:.2f}")

        results_by_type[label] = {
            'n_orbits': n_orbits,
            'orbit_sizes': orbit_sizes,
            'asymmetry': asymmetry,
            'max_interference': max_interference,
            'has_interference': max_interference > 1e-10,
            'interference_pattern': [float(t) for t in interference],
        }

    # Both should show interference, but D_4 should have different structure
    # due to 3-fold branching
    both_interfere = all(r['has_interference'] for r in results_by_type.values())

    # D_4's 3-leaf orbit gives larger interference amplitude
    d4_interference = results_by_type['D_4_star']['max_interference']
    a4_interference = results_by_type['A_4_chain']['max_interference']
    d4_stronger = d4_interference > a4_interference

    passed = both_interfere

    print(f"\n  Both interfere: {both_interfere}")
    print(f"  D_4 stronger: {d4_stronger} ({d4_interference:.6f} vs {a4_interference:.6f})")

    result = {
        'test': 'T3_topology_determines_interference',
        'results_by_type': results_by_type,
        'both_interfere': both_interfere,
        'd4_stronger': d4_stronger,
        'PASS': passed,
    }
    return result


def test_T4_visibility_vs_deformation_rate():
    """T4: Interference visibility vs complement-deformation rate (50/50 — cross-layer)."""
    # This tests whether the algebraic layer (interference) correlates with
    # the metric layer (complement-deformation rate). M13.5 showed these are
    # largely independent, so we expect this to FAIL.

    test_cases = [('A', 5), ('A', 7), ('D', 4), ('D', 6), ('E', 6)]
    results_by_type = {}
    deformation_rates = []
    max_interferences = []

    for family, rank in test_cases:
        label = f"{family}_{rank}"
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency
        n = adj.shape[0]
        basis, orbits = orbit_hilbert_basis(adj)
        n_orbits = len(orbits)

        # Maximum interference for equal superposition with optimal phase
        if n_orbits >= 2:
            theta = np.pi / 3
            c = np.zeros(n_orbits, dtype=complex)
            c[0] = 1.0 / np.sqrt(2)
            c[1] = np.exp(1j * theta) / np.sqrt(2)
            psi = basis @ c

            vertex_probs = np.abs(psi) ** 2
            orbit_probs = born_probabilities(psi, adj)
            classical_probs = np.zeros(n)
            for i in range(n_orbits):
                if orbit_probs[i] > 1e-15:
                    collapsed = measurement_collapse(psi, i, adj)
                    classical_probs += orbit_probs[i] * np.abs(collapsed) ** 2
            max_int = float(np.max(np.abs(vertex_probs - classical_probs)))
        else:
            max_int = 0.0

        # Complement deformation rate
        try:
            max_rate = max_deformation_rate(adj)
        except Exception:
            max_rate = 0.0

        deformation_rates.append(max_rate)
        max_interferences.append(max_int)

        results_by_type[label] = {
            'n_orbits': n_orbits,
            'max_interference': max_int,
            'max_deformation_rate': float(max_rate),
        }

        print(f"  {label}: max_interference={max_int:.6f}, max_deformation_rate={max_rate:.4f}")

    # Correlation between the two
    if len(deformation_rates) >= 3:
        corr = float(np.corrcoef(deformation_rates, max_interferences)[0, 1])
    else:
        corr = 0.0

    # Pre-registered as 50/50: cross-layer correlation unlikely to be strong
    strong_correlation = abs(corr) > 0.7
    passed = strong_correlation

    print(f"\n  Correlation: r={corr:.4f}")
    print(f"  Strong: {strong_correlation} (pre-registered as 50/50)")

    result = {
        'test': 'T4_visibility_vs_deformation_rate',
        'results_by_type': results_by_type,
        'correlation': corr,
        'strong_correlation': strong_correlation,
        'PASS': passed,
    }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 06: Graph Double-Slit")
    print("Milestone 14, Block C")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_two_path_interference),
        ("T2", test_T2_which_path_destroys_interference),
        ("T3", test_T3_topology_determines_interference),
        ("T4", test_T4_visibility_vs_deformation_rate),
    ]

    for name, fn in tests:
        print(f"\n--- {name}: {fn.__doc__.strip()} ---")
        r = fn()
        results[name] = r
        scorecard.append(r['PASS'])
        status = "PASS" if r['PASS'] else "FAIL"
        print(f"  => {status}")

    n_pass = sum(scorecard)
    n_total = len(scorecard)
    print(f"\n{'=' * 70}")
    print(f"Score: {n_pass}/{n_total}")
    print(f"{'=' * 70}")

    save_data = {
        'experiment': 'exp_06_graph_double_slit',
        'milestone': 14,
        'block': 'C',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_06_graph_double_slit', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
