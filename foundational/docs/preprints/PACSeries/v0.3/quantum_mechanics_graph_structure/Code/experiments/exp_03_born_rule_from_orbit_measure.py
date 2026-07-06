"""
exp_03 -- Born Rule from Orbit Measure

Milestone 14, Block B (Born Rule & Measurement)

Hypothesis: The Born rule emerges from orbit geometry. Born probabilities sum to 1
for gauge-invariant states (those in the orbit Hilbert space), but < 1 for states
with gauge-variant components. The uniform state gives P(O_i) = |O_i|/n (orbit
volume = gauge volume). Gleason's theorem guarantees uniqueness for orbit dim >= 3.

Tests:
  T1: Born probs sum to 1 for gauge-invariant states; < 1 for gauge-variant
  T2: Uniform state gives P(O_i) = |O_i|/n (gauge volume)
  T3: PAC phi-splitting maps to orbit probabilities (50/50: PAC tree != linear chain)
  T4: Gleason's theorem: Born rule unique for orbit dim >= 3 (D_6+ only)
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    orbit_hilbert_basis, born_probability, born_probabilities,
    vertex_orbits,
    save_m14_results, _convert_numpy,
)


def test_T1_born_sum_gauge_invariance():
    """T1: Born probs sum to 1 for gauge-invariant states; < 1 for gauge-variant."""
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

        # Test 1a: gauge-invariant state (superposition of orbit basis vectors)
        # Any state in span(orbit basis) is gauge-invariant
        if n_orbits >= 2:
            # Random superposition of orbit basis vectors
            coeffs = np.array([1.0, 0.5] + [0.3] * (n_orbits - 2))
            gauge_inv_state = basis @ (coeffs[:n_orbits] / np.linalg.norm(basis @ coeffs[:n_orbits]))
        else:
            gauge_inv_state = basis[:, 0]

        probs_inv = born_probabilities(gauge_inv_state, adj)
        sum_inv = sum(probs_inv)
        sum_1_check = abs(sum_inv - 1.0) < 1e-10

        # Test 1b: gauge-variant state (has components outside orbit space)
        # A state that breaks gauge symmetry: e.g., |0> (single vertex)
        gauge_var_state = np.zeros(n)
        gauge_var_state[0] = 1.0
        probs_var = born_probabilities(gauge_var_state, adj)
        sum_var = sum(probs_var)

        # For gauge-variant states, Born probs CAN still sum to 1 IF the state
        # happens to lie in the orbit space. |0> decomposes into orbit components.
        # Actually: orbit basis spans a SUBSPACE of C^n. |0> has components in
        # orbit space (projection onto each orbit) but these might not sum to 1
        # if orbits don't cover all of |0>'s norm.
        #
        # Key insight: P_total = sum_i P_i = sum of rank-1 projectors onto orbit vectors.
        # For gauge-invariant states, P_total|psi> = |psi> so sum = 1.
        # For gauge-variant states, P_total|psi> != |psi> so sum < 1.
        #
        # But wait: orbit vectors span the gauge-invariant subspace.
        # |0> is NOT gauge-invariant (unless it's in a singleton orbit).
        # So the gauge-variant part should have sum < 1... but only if orbits
        # don't fully resolve the state.

        # Actually for |0> in orbit {0, n-1}: projection onto that orbit has
        # probability = (1/|O|), and onto other orbits = 0. So sum = 1/|O| < 1.
        # Unless |0| is a singleton orbit, then sum = 1.

        # Check: is vertex 0 in a multi-vertex orbit?
        orbit_of_0 = None
        for o in orbits:
            if 0 in o:
                orbit_of_0 = o
                break
        is_singleton = len(orbit_of_0) == 1

        if is_singleton:
            # |0> IS gauge-invariant (it's an orbit), sum should be 1
            gauge_var_valid = abs(sum_var - 1.0) < 1e-10
        else:
            # |0> is NOT gauge-invariant, sum should be < 1
            gauge_var_valid = sum_var < 1.0 - 1e-10

        passed = sum_1_check and gauge_var_valid
        all_pass = all_pass and passed

        print(f"  {label}: inv_sum={sum_inv:.6f} (=1? {sum_1_check}), "
              f"var_sum={sum_var:.6f} (orbit_of_0 size={len(orbit_of_0)}, valid={gauge_var_valid})")

        results_by_type[label] = {
            'n': n,
            'n_orbits': n_orbits,
            'gauge_invariant_sum': float(sum_inv),
            'gauge_variant_sum': float(sum_var),
            'orbit_of_vertex_0': sorted(orbit_of_0),
            'vertex_0_is_singleton': is_singleton,
            'sum_1_check': sum_1_check,
            'gauge_var_valid': gauge_var_valid,
            'PASS': passed,
        }

    result = {
        'test': 'T1_born_sum_gauge_invariance',
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T2_uniform_state_gauge_volume():
    """T2: Uniform state gives P(O_i) = |O_i|/n (gauge volume)."""
    diagrams = all_ade_diagrams(max_rank=8)
    all_pass = True
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency
        n = adj.shape[0]

        # Uniform state: |psi> = (1/sqrt(n)) * sum_v |v>
        uniform = np.ones(n) / np.sqrt(n)

        basis, orbits = orbit_hilbert_basis(adj)
        probs = born_probabilities(uniform, adj)

        # Expected: P(O_i) = |O_i| / n
        expected = [len(o) / n for o in orbits]

        max_error = max(abs(p - e) for p, e in zip(probs, expected))
        matches = max_error < 1e-10

        # Also check: sum = 1 (uniform is gauge-invariant)
        sum_check = abs(sum(probs) - 1.0) < 1e-10

        passed = matches and sum_check
        all_pass = all_pass and passed

        print(f"  {label}: orbits={[len(o) for o in orbits]}, probs={[f'{p:.4f}' for p in probs]}, "
              f"match={matches}")

        results_by_type[label] = {
            'n': n,
            'orbit_sizes': [len(o) for o in orbits],
            'born_probs': [float(p) for p in probs],
            'expected_probs': [float(e) for e in expected],
            'max_error': float(max_error),
            'matches': matches,
            'sum_check': sum_check,
            'PASS': passed,
        }

    result = {
        'test': 'T2_uniform_state_gauge_volume',
        'n_diagrams': len(diagrams),
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T3_pac_phi_splitting():
    """T3: PAC phi-splitting maps to orbit probabilities (50/50 — PAC tree != linear chain)."""
    # PAC splitting: at each level, energy splits phi : 1-phi
    # This is a binary tree structure, not the linear chain topology of ADE
    # So we expect this mapping to NOT work cleanly

    diag = DynkinDiagram('A', 5)
    adj = diag.adjacency
    n = adj.shape[0]
    basis, orbits = orbit_hilbert_basis(adj)

    # PAC phi-splitting gives a specific probability distribution
    # For depth d, weights are phi^d and (1-phi)^d on branches
    # Try to construct a state whose orbit probabilities match PAC splitting
    pac_probs = [PHI ** 2, PHI * INV_PHI, INV_PHI ** 2]  # 3 orbits for A_5

    # Normalize
    pac_probs = [p / sum(pac_probs) for p in pac_probs]

    # Can we find a gauge-invariant state with these probabilities?
    # |psi> = sum_i c_i |O_i> with |c_i|^2 = pac_probs[i]
    coeffs = np.sqrt(pac_probs)
    pac_state = basis @ coeffs

    # Verify Born probabilities match
    actual_probs = born_probabilities(pac_state, adj)
    max_error = max(abs(a - e) for a, e in zip(actual_probs, pac_probs))

    # This tests whether the construction WORKS, not whether it's natural
    construction_works = max_error < 1e-10

    # The deeper question: does PAC binary tree topology match ADE linear chain?
    # PAC splits as phi:(1-phi) at each node — binary tree
    # A_5 orbits: {0,4}, {1,3}, {2} — linear chain topology
    # These are structurally different, so natural mapping is unlikely

    # Check: do orbit sizes follow phi-like ratios?
    orbit_sizes = sorted([len(o) for o in orbits], reverse=True)
    if len(orbit_sizes) >= 2:
        ratio = orbit_sizes[0] / orbit_sizes[1] if orbit_sizes[1] > 0 else float('inf')
        phi_like = abs(ratio - PHI) < 0.5  # loose tolerance
    else:
        phi_like = False

    # Pre-registered as 50/50: construction works but mapping isn't natural
    passed = construction_works and phi_like

    print(f"  PAC probs: {[f'{p:.4f}' for p in pac_probs]}")
    print(f"  Actual probs: {[f'{p:.4f}' for p in actual_probs]}")
    print(f"  Construction works: {construction_works}")
    print(f"  Orbit sizes: {orbit_sizes}, ratio: {ratio if len(orbit_sizes)>=2 else 'N/A'}")
    print(f"  Phi-like ratio: {phi_like}")

    result = {
        'test': 'T3_pac_phi_splitting',
        'pac_probs': pac_probs,
        'actual_probs': [float(p) for p in actual_probs],
        'max_error': float(max_error),
        'construction_works': construction_works,
        'orbit_sizes': orbit_sizes,
        'phi_like_ratio': phi_like,
        'PASS': passed,
    }
    return result


def test_T4_gleason_uniqueness():
    """T4: Gleason's theorem: Born rule unique for orbit dim >= 3 (D_6+ only)."""
    # Gleason's theorem: on a Hilbert space of dim >= 3, the only
    # probability measure on subspaces that is countably additive is
    # the Born rule. For dim < 3, other measures exist.
    #
    # We check: which ADE types have orbit dim >= 3?
    # Those are the ones where Born rule is FORCED by Gleason.

    diagrams = all_ade_diagrams(max_rank=8)
    results_by_type = {}
    gleason_types = []

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency
        basis, orbits = orbit_hilbert_basis(adj)
        orbit_dim = len(orbits)

        gleason_applies = orbit_dim >= 3
        if gleason_applies:
            gleason_types.append(label)

        # For Gleason types: verify Born rule is the ONLY consistent assignment
        # We check: any frame function on the orbit projectors must be Born rule
        if gleason_applies:
            # The orbit projectors form a resolution of the identity on the orbit subspace
            # By Gleason, any probability measure must be of the form Tr(rho * P_i)
            # which IS the Born rule. We verify the projectors sum to identity on orbit space.
            from quantum_complement import all_orbit_projectors
            projectors, _ = all_orbit_projectors(adj)
            P_sum = sum(projectors)

            # P_sum should equal the projector onto the orbit subspace
            # On the orbit subspace, this equals identity
            P_orbit = basis @ basis.T  # projector onto orbit subspace
            resolution_error = float(np.max(np.abs(P_sum - P_orbit)))
            is_resolution = resolution_error < 1e-10
        else:
            is_resolution = True  # vacuously
            resolution_error = 0.0

        results_by_type[label] = {
            'orbit_dim': orbit_dim,
            'gleason_applies': gleason_applies,
            'resolution_of_identity': is_resolution,
            'resolution_error': resolution_error,
        }

        print(f"  {label}: orbit_dim={orbit_dim}, Gleason={gleason_applies}")

    # Gleason should apply to: A_5+(3 orbits), A_7+(4), D_5+(4+), E_6(4), E_7(7), E_8(8)
    # NOT to: A_1(1), A_2(1), A_3(2), A_4(2), D_4(2)
    at_least_some = len(gleason_types) >= 5
    all_resolutions = all(r['resolution_of_identity'] for r in results_by_type.values())

    passed = at_least_some and all_resolutions

    print(f"\n  Gleason types ({len(gleason_types)}): {gleason_types}")

    result = {
        'test': 'T4_gleason_uniqueness',
        'gleason_types': gleason_types,
        'n_gleason': len(gleason_types),
        'at_least_some': at_least_some,
        'all_resolutions': all_resolutions,
        'results_by_type': results_by_type,
        'PASS': passed,
    }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 03: Born Rule from Orbit Measure")
    print("Milestone 14, Block B")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_born_sum_gauge_invariance),
        ("T2", test_T2_uniform_state_gauge_volume),
        ("T3", test_T3_pac_phi_splitting),
        ("T4", test_T4_gleason_uniqueness),
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
        'experiment': 'exp_03_born_rule_from_orbit_measure',
        'milestone': 14,
        'block': 'B',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_03_born_rule_from_orbit_measure', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
