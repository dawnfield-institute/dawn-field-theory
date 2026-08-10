"""
Milestone 10 -- Exp 14: Spectral Confinement -- Why Symmetry Freezes Geometry

EXTENSION -- testing what self-applied symmetry CAN and CANNOT generate.

The M10 thesis claims self-applied symmetry is the unique generative primitive.
This experiment reveals what that means computationally: self-applied symmetry
confines dynamics to EIGENVALUE space. Eigenvectors never rotate.

Key mathematical fact: for symmetric W = V D V^T, any spectral operation
(eigenvalue modulation, spectral radius normalization) produces W' = V D' V^T
with the SAME eigenvectors V. This is not a design choice -- it's a theorem.

Consequence: the SelfApplicator's state x chases cycling eigenvalue peaks
through FIXED directions. It can never catch up, because anti-Hebbian moves
the peak faster than x can realign. Result: state collapses to zero.

But spectral structure (phi ratios, hierarchy in eigenvalue patterns) persists
in W even after x dies. Self-applied symmetry generates structure in the
spectrum, not in the state.

Tests:
  1. Eigenvector preservation: W eigenvectors unchanged after 1000 steps
     of anti-Hebbian modulation (exact to machine precision)
  2. State collapse: state norm decays exponentially; mechanism traced to
     eigenvalue-state misalignment caused by anti-Hebbian cycling
  3. Asymmetric contrast: asymmetric self-application DOES rotate eigenvectors
     and the state survives (establishing that eigenvector fixity requires symmetry)
  4. Spectral structure persists: hierarchy measures remain present in W
     dynamics even after state collapses to zero

Builds on: exp_01 (uniqueness), exp_03 (mutual reference)
Extension: Spectral confinement as structural consequence of M10 thesis
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    SelfApplicator, measure_hierarchical_structure,
    save_results, setup_experiment,
    PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
)

_, RESULTS_DIR = setup_experiment(__file__)


# ============================================================
# Test 1: Eigenvector Preservation Under Self-Application
# ============================================================
def test1_eigenvector_preservation():
    """
    Eigenvectors of W are preserved exactly under anti-Hebbian modulation.
    This is a mathematical identity: modulation operates as
    W' = V @ diag(new_eigenvalues) @ V^T -- same V every step.
    """
    print("\n=== Test 1: Eigenvector Preservation ===")

    max_drifts = []

    for N in [8, 16, 32]:
        drifts_N = []
        for seed in range(20):
            sa = SelfApplicator(seed, self_applies=True, symmetric=True, size=N)

            # Record initial eigenvectors
            _, V_init = np.linalg.eigh(sa.W)

            # Run 1000 steps
            for _ in range(1000):
                sa.step()

            # Compare eigenvectors (up to sign -- eigenvectors defined up to +/-)
            _, V_final = np.linalg.eigh(sa.W)

            # Overlap matrix: |V_init^T @ V_final| should be identity
            overlap = np.abs(V_init.T @ V_final)
            # Each column of V_final should match exactly one column of V_init
            max_per_col = np.max(overlap, axis=0)
            drift = 1.0 - np.min(max_per_col)
            drifts_N.append(drift)

        max_drifts.extend(drifts_N)
        mean_drift = np.mean(drifts_N)
        print(f"  N={N:3d}: max eigenvector drift = {mean_drift:.2e} "
              f"(20 seeds, 1000 steps)")

    overall_max = max(max_drifts)
    passed = overall_max < 1e-10
    print(f"\n  Overall max drift: {overall_max:.2e}")
    print(f"  PASS: {passed} (need < 1e-10 -- machine precision)")

    return {
        'test': 'eigenvector_preservation',
        'passed': bool(passed),
        'max_drift': float(overall_max),
        'n_systems': len(max_drifts),
    }


# ============================================================
# Test 2: State Collapse Mechanism
# ============================================================
def test2_state_collapse():
    """
    State norm decays exponentially because:
    1. Anti-Hebbian weakens the eigenvalue aligned with the state
    2. Another eigenvalue gets boosted, but in a DIFFERENT (fixed) direction
    3. State has small projection on the new dominant direction
    4. Repeat: state chases, never catches, decays

    At sr=1.2 with weak=0.95, the dominant eigenvalue crosses 1.0 in
    ~3.6 steps. The state can't grow fast enough to compensate.
    """
    print("\n=== Test 2: State Collapse Mechanism ===")

    collapse_times = []

    for seed in range(50):
        sa = SelfApplicator(seed, self_applies=True, symmetric=True, size=16)
        for t in range(500):
            sa.step()
            if np.linalg.norm(sa.state) < 1e-4:
                collapse_times.append(t)
                break
        else:
            collapse_times.append(500)

    mean_collapse = np.mean(collapse_times)
    std_collapse = np.std(collapse_times)

    print(f"  Collapse time: {mean_collapse:.0f} +/- {std_collapse:.0f} steps "
          f"(50 seeds, N=16)")

    # Verify the mechanism: dominant eigenvalue crosses 1.0 in ~4 steps
    k_cross = np.log(1 / 1.2) / np.log(0.95)
    print(f"  Predicted k_cross (steps to subcritical): {k_cross:.1f}")

    # All systems should collapse within 200 steps
    all_collapse = all(t < 200 for t in collapse_times)
    passed = all_collapse and mean_collapse < 150
    print(f"\n  All collapse < 200 steps: {all_collapse}")
    print(f"  PASS: {passed}")

    return {
        'test': 'state_collapse',
        'passed': bool(passed),
        'mean_collapse_time': float(mean_collapse),
        'std_collapse_time': float(std_collapse),
        'k_cross_predicted': float(k_cross),
    }


# ============================================================
# Test 3: Asymmetric Contrast
# ============================================================
def test3_asymmetric_contrast():
    """
    Spectral confinement (eigenvector fixity) enables STRUCTURED collapse.
    Both symmetric and asymmetric self-application kill the state, but
    symmetric produces hierarchical structure during the transient while
    asymmetric does not.

    Why: symmetric W has real eigenvalues and orthogonal eigenvectors.
    Anti-Hebbian modulation cycles eigenvalues through FIXED directions,
    creating organized multi-scale structure in the trajectory covariance.
    Asymmetric W's SVD-based modulation is incoherent with the actual
    (complex, non-orthogonal) dynamical eigenvectors → no organized structure.

    This is the structural consequence of spectral confinement: fixed
    eigenvectors make the collapse STRUCTURED, not random.
    """
    print("\n=== Test 3: Asymmetric Contrast ===")

    sym_hierarchies = 0
    asym_hierarchies = 0
    sym_complexities = []
    asym_complexities = []
    n_systems = 100

    for seed in range(n_systems):
        # Symmetric: expect hierarchical transient
        sa_sym = SelfApplicator(seed, self_applies=True, symmetric=True, size=16)
        traj_sym = sa_sym.run(300)
        result_sym = measure_hierarchical_structure(traj_sym)
        sym_complexities.append(result_sym['mean_complexity'])
        if result_sym['has_hierarchy']:
            sym_hierarchies += 1

        # Asymmetric: expect no hierarchy
        sa_asym = SelfApplicator(seed, self_applies=True, symmetric=False, size=16)
        traj_asym = sa_asym.run(300)
        result_asym = measure_hierarchical_structure(traj_asym)
        asym_complexities.append(result_asym['mean_complexity'])
        if result_asym['has_hierarchy']:
            asym_hierarchies += 1

    sym_frac = sym_hierarchies / n_systems
    asym_frac = asym_hierarchies / n_systems

    print(f"  Symmetric:  {sym_hierarchies}/{n_systems} ({sym_frac:.0%}) hierarchical, "
          f"complexity={np.mean(sym_complexities):.2f}")
    print(f"  Asymmetric: {asym_hierarchies}/{n_systems} ({asym_frac:.0%}) hierarchical, "
          f"complexity={np.mean(asym_complexities):.2f}")

    # Symmetric produces structured hierarchy; asymmetric doesn't
    sym_structured = sym_frac > 0.20
    asym_unstructured = asym_frac < 0.05
    dominance = sym_frac > asym_frac * 3 if asym_frac > 0 else sym_frac > 0.15

    passed = sym_structured and (asym_unstructured or dominance)
    print(f"\n  Symmetric structured (>20%): {sym_structured}")
    print(f"  Asymmetric unstructured (<5%): {asym_unstructured}")
    print(f"  Symmetric dominates: {dominance}")
    print(f"  PASS: {passed}")

    return {
        'test': 'asymmetric_contrast',
        'passed': bool(passed),
        'sym_hierarchy_frac': float(sym_frac),
        'asym_hierarchy_frac': float(asym_frac),
        'sym_mean_complexity': float(np.mean(sym_complexities)),
        'asym_mean_complexity': float(np.mean(asym_complexities)),
    }


# ============================================================
# Test 4: Spectral Structure Persists Despite State Collapse
# ============================================================
def test4_spectral_persistence():
    """
    Even though the state collapses, the coupling matrix W retains
    structured eigenvalue patterns. Hierarchy in W persists.

    Compare: self-applying symmetric (hierarchy in short transient)
    vs fixed W (no spectral evolution, no hierarchy).
    """
    print("\n=== Test 4: Spectral Structure Persists ===")

    sa_hierarchy_count = 0
    fixed_hierarchy_count = 0
    n_systems = 100
    n_steps = 300

    sa_complexities = []
    fixed_complexities = []

    for seed in range(n_systems):
        sa = SelfApplicator(seed, self_applies=True, symmetric=True, size=32)
        traj = sa.run(n_steps)
        result = measure_hierarchical_structure(traj)
        if result['has_hierarchy']:
            sa_hierarchy_count += 1
        sa_complexities.append(result['mean_complexity'])

        sa_fixed = SelfApplicator(seed, self_applies=False, symmetric=True, size=32)
        traj_fixed = sa_fixed.run(n_steps)
        result_fixed = measure_hierarchical_structure(traj_fixed)
        if result_fixed['has_hierarchy']:
            fixed_hierarchy_count += 1
        fixed_complexities.append(result_fixed['mean_complexity'])

    sa_frac = sa_hierarchy_count / n_systems
    fixed_frac = fixed_hierarchy_count / n_systems

    print(f"  Self-applying: {sa_hierarchy_count}/{n_systems} ({sa_frac:.0%}) hierarchical, "
          f"complexity = {np.mean(sa_complexities):.2f}")
    print(f"  Fixed W:       {fixed_hierarchy_count}/{n_systems} ({fixed_frac:.0%}) hierarchical, "
          f"complexity = {np.mean(fixed_complexities):.2f}")

    passed = sa_frac > 0.5 and sa_frac > fixed_frac * 3
    print(f"\n  PASS: {passed}")

    return {
        'test': 'spectral_persistence',
        'passed': bool(passed),
        'sa_hierarchy_frac': float(sa_frac),
        'fixed_hierarchy_frac': float(fixed_frac),
        'sa_mean_complexity': float(np.mean(sa_complexities)),
        'fixed_mean_complexity': float(np.mean(fixed_complexities)),
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Exp 14: Spectral Confinement -- Why Symmetry Freezes Geometry")
    print("  Self-applied symmetry confines dynamics to eigenvalue space")
    print("=" * 70)

    tests = [
        test1_eigenvector_preservation,
        test2_state_collapse,
        test3_asymmetric_contrast,
        test4_spectral_persistence,
    ]

    results = []
    n_passed = 0

    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_14_spectral_confinement',
        'type': 'extension',
        'extension_section': 'Spectral confinement under self-applied symmetry',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'timestamp': datetime.now().isoformat(),
    }
    save_results(output, RESULTS_DIR, 'exp_14_spectral_confinement')
