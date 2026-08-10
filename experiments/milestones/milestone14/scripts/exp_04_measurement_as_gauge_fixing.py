"""
exp_04 -- Measurement as Gauge Fixing

Milestone 14, Block B (Born Rule & Measurement)

Hypothesis: Measurement in the orbit framework is gauge fixing — projecting onto
a specific orbit collapses gauge freedom. Measurement is idempotent (P^2 = P),
superposition ensembles have positive entropy, and two-stage measurement fully
resolves gauge freedom.

Tests:
  T1: Measurement is idempotent (P^2 = P)
  T2: Ensemble entropy > 0 for superpositions
  T3: Two-stage measurement resolves gauge freedom
  T4: SEC arrow: complex collapse loses phase info (50/50 — orbit projectors are real)
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    orbit_hilbert_basis, orbit_projector, all_orbit_projectors,
    measurement_collapse, born_probabilities, born_probability,
    von_neumann_entropy, vertex_orbits,
    save_m14_results, _convert_numpy,
)


def test_T1_measurement_idempotent():
    """T1: Measurement is idempotent (P^2 = P) for all ADE types."""
    diagrams = all_ade_diagrams(max_rank=8)
    all_pass = True
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency
        projectors, orbits = all_orbit_projectors(adj)

        max_error = 0.0
        for i, P in enumerate(projectors):
            # P^2 should equal P
            P2 = P @ P
            error = float(np.max(np.abs(P2 - P)))
            max_error = max(max_error, error)

        is_idempotent = max_error < 1e-10
        all_pass = all_pass and is_idempotent

        print(f"  {label}: {len(projectors)} projectors, max_error={max_error:.2e}, "
              f"idempotent={is_idempotent}")

        results_by_type[label] = {
            'n_projectors': len(projectors),
            'max_idempotency_error': max_error,
            'is_idempotent': is_idempotent,
            'PASS': is_idempotent,
        }

    result = {
        'test': 'T1_measurement_idempotent',
        'n_diagrams': len(diagrams),
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T2_ensemble_entropy():
    """T2: Ensemble entropy > 0 for superpositions of orbits."""
    test_cases = [('A', 5), ('D', 4), ('E', 6)]
    all_pass = True
    results_by_type = {}

    for family, rank in test_cases:
        label = f"{family}_{rank}"
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency
        n = adj.shape[0]
        basis, orbits = orbit_hilbert_basis(adj)
        n_orbits = len(orbits)

        if n_orbits < 2:
            results_by_type[label] = {'skipped': True, 'reason': 'only 1 orbit', 'PASS': True}
            continue

        # Create a superposition state across orbits
        coeffs = np.ones(n_orbits) / np.sqrt(n_orbits)
        super_state = basis @ coeffs

        # Born probabilities give the ensemble
        probs = born_probabilities(super_state, adj)

        # Construct the ensemble density matrix: rho = sum_i p_i |O_i><O_i|
        rho_ensemble = np.zeros((n, n))
        for i in range(n_orbits):
            v = basis[:, i]
            rho_ensemble += probs[i] * np.outer(v, v)

        # Von Neumann entropy should be > 0 for genuine superposition
        S = von_neumann_entropy(rho_ensemble)

        # Compare to pure state entropy (should be 0)
        rho_pure = np.outer(super_state, super_state)
        S_pure = von_neumann_entropy(rho_pure)

        # Entropy should be positive for ensemble, ~0 for pure
        entropy_positive = S > 1e-10
        pure_zero = S_pure < 1e-10

        # For equal superposition: S = log(n_orbits)
        S_expected = np.log(n_orbits)
        entropy_matches = abs(S - S_expected) < 0.01

        passed = entropy_positive and pure_zero
        all_pass = all_pass and passed

        print(f"  {label}: S_ensemble={S:.4f} (expected={S_expected:.4f}), "
              f"S_pure={S_pure:.2e}, positive={entropy_positive}")

        results_by_type[label] = {
            'n_orbits': n_orbits,
            'ensemble_entropy': float(S),
            'expected_entropy': float(S_expected),
            'pure_state_entropy': float(S_pure),
            'entropy_positive': entropy_positive,
            'entropy_matches_log_m': entropy_matches,
            'pure_zero': pure_zero,
            'PASS': passed,
        }

    result = {
        'test': 'T2_ensemble_entropy',
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T3_two_stage_measurement():
    """T3: Two-stage measurement resolves gauge freedom."""
    # D_4 has 2 orbits. After measuring orbit, state is gauge-fixed.
    # Second measurement on same orbit should give same state (idempotent collapse).
    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency
    n = adj.shape[0]
    basis, orbits = orbit_hilbert_basis(adj)

    # Start with superposition
    coeffs = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    psi = basis @ coeffs

    # First measurement: collapse to orbit 0
    psi_1 = measurement_collapse(psi, 0, adj)

    # Check: state is now purely in orbit 0
    probs_after_1 = born_probabilities(psi_1, adj)
    orbit_0_prob = probs_after_1[0]
    in_orbit_0 = abs(orbit_0_prob - 1.0) < 1e-10

    # Second measurement: same orbit — should be idempotent
    psi_2 = measurement_collapse(psi_1, 0, adj)

    # States should be identical
    state_identical = np.allclose(psi_1, psi_2)

    # Also test with A_5 (3 orbits): sequential measurements
    diag_a5 = DynkinDiagram('A', 5)
    adj_a5 = diag_a5.adjacency
    basis_a5, orbits_a5 = orbit_hilbert_basis(adj_a5)

    # Start with uniform superposition over 3 orbits
    coeffs_a5 = np.ones(3) / np.sqrt(3)
    psi_a5 = basis_a5 @ coeffs_a5

    # Measure orbit 1
    psi_a5_1 = measurement_collapse(psi_a5, 1, adj_a5)
    probs_a5_1 = born_probabilities(psi_a5_1, adj_a5)

    # After measurement: only orbit 1 has probability
    only_orbit_1 = abs(probs_a5_1[1] - 1.0) < 1e-10

    # Gauge freedom resolved: state is fully determined
    passed = in_orbit_0 and state_identical and only_orbit_1

    print(f"  D_4: first measure -> orbit_0 prob={orbit_0_prob:.6f}, in_orbit={in_orbit_0}")
    print(f"  D_4: second measure -> identical={state_identical}")
    print(f"  A_5: measure orbit 1 -> prob={probs_a5_1[1]:.6f}, only_orbit_1={only_orbit_1}")

    result = {
        'test': 'T3_two_stage_measurement',
        'd4_orbit_0_prob_after_first': float(orbit_0_prob),
        'd4_in_orbit_0': in_orbit_0,
        'd4_state_identical_after_second': state_identical,
        'a5_orbit_1_prob_after_measure': float(probs_a5_1[1]),
        'a5_only_orbit_1': only_orbit_1,
        'PASS': passed,
    }
    return result


def test_T4_sec_arrow_phase_loss():
    """T4: SEC arrow: complex collapse loses phase info (50/50 — orbit projectors are real)."""
    # The SEC arrow (second law) requires that measurement is irreversible.
    # Complex states carry phase information. After orbit projection, is phase lost?
    #
    # Key issue: orbit projectors P_i = |O_i><O_i| are REAL matrices.
    # So they project out the imaginary part of states.
    # A complex state |psi> = sum c_i |v_i> with c_i complex
    # after projection becomes a REAL state (proportional to |O_i>).
    # This IS phase loss!
    #
    # But: the orbit basis vectors themselves are real. So there's no
    # way to encode phase IN the orbit Hilbert space. Phase lives in the
    # gauge-variant sector (the complement of orbit space in full vertex space).
    #
    # Pre-registered as 50/50: the SEC arrow enters through Weyl irrep projectors,
    # not orbit projectors.

    diag = DynkinDiagram('D', 4)
    adj = diag.adjacency
    n = adj.shape[0]
    basis, orbits = orbit_hilbert_basis(adj)

    # Create complex state
    psi_complex = np.array([1.0 + 0.5j, 0.3 - 0.2j, 0.4 + 0.1j, 0.2 - 0.3j], dtype=complex)
    psi_complex = psi_complex / np.linalg.norm(psi_complex)

    # Measure orbit 0
    try:
        psi_collapsed = measurement_collapse(psi_complex, 0, adj)
    except ValueError:
        # Zero overlap with orbit 0
        psi_collapsed = measurement_collapse(psi_complex, 1, adj)

    # Check: is collapsed state real (up to global phase)?
    # Remove global phase: multiply by conj of first nonzero element
    first_nonzero = psi_collapsed[np.abs(psi_collapsed) > 1e-10][0]
    psi_dephased = psi_collapsed * np.conj(first_nonzero) / np.abs(first_nonzero)
    is_real_after = np.max(np.abs(np.imag(psi_dephased))) < 1e-10

    # Phase information: initial state had relative phases between vertices
    initial_phases = np.angle(psi_complex)
    collapsed_phases = np.angle(psi_collapsed)

    # In collapsed state, all nonzero components should have same phase
    nonzero_mask = np.abs(psi_collapsed) > 1e-10
    if np.sum(nonzero_mask) > 1:
        phase_variance = np.var(collapsed_phases[nonzero_mask])
        phase_lost = phase_variance < 1e-10
    else:
        phase_lost = True

    # Check: orbit projectors are real matrices
    projectors, _ = all_orbit_projectors(adj)
    all_real = all(np.max(np.abs(np.imag(P))) < 1e-10 for P in projectors)

    # The SEC arrow question: does collapse IRREVERSIBLY lose information?
    # Real projectors destroy imaginary components → yes, phase is lost
    # But this is because orbit basis is real, not because of SEC per se
    passed = is_real_after and phase_lost and all_real

    print(f"  Collapsed state real (up to global phase): {is_real_after}")
    print(f"  Phase variance after collapse: {phase_variance if np.sum(nonzero_mask) > 1 else 0:.2e}")
    print(f"  Phase lost: {phase_lost}")
    print(f"  All projectors real: {all_real}")
    print(f"  NOTE: phase loss is from real orbit basis, not SEC mechanism per se")

    result = {
        'test': 'T4_sec_arrow_phase_loss',
        'is_real_after_collapse': is_real_after,
        'phase_lost': phase_lost,
        'all_projectors_real': all_real,
        'note': 'Orbit projectors are real -> phase loss is structural, SEC arrow enters through Weyl irreps',
        'PASS': passed,
    }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 04: Measurement as Gauge Fixing")
    print("Milestone 14, Block B")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_measurement_idempotent),
        ("T2", test_T2_ensemble_entropy),
        ("T3", test_T3_two_stage_measurement),
        ("T4", test_T4_sec_arrow_phase_loss),
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
        'experiment': 'exp_04_measurement_as_gauge_fixing',
        'milestone': 14,
        'block': 'B',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_04_measurement_as_gauge_fixing', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
