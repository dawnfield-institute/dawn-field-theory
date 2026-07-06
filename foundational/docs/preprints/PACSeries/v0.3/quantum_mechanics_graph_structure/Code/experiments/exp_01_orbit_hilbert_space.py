"""
exp_01 -- Orbit Quotient as Hilbert Space

Milestone 14, Block A (Orbit Hilbert Space)

Hypothesis: The orbit quotient V/Aut(G) gives a positive-definite Hilbert space
for every ADE graph. This is the resolution of M13's PSD problem: the full vertex
space has degenerate inner products (gauge equivalence), but the orbit quotient
is always positive definite because each orbit collapses to a single basis vector.

Tests:
  T1: Orbit basis orthonormal on A_5, D_4, E_6
  T2: Orbit Gram matrix positive DEFINITE (resolves M13's PSD!)
  T3: Orbit dimension = Aut(G) orbits, matches complement-spectrum orbits across all ADE
  T4: Trivial Aut (E_7, E_8) gives dim = n (classical limit, no gauge structure)
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    complement_spectrum, vertex_orbits,
    graph_automorphisms, orbit_hilbert_basis, all_orbit_projectors,
    save_m14_results, _convert_numpy,
)


def test_T1_orbit_basis_orthonormal():
    """T1: Orbit basis is orthonormal on A_5, D_4, E_6."""
    test_cases = [('A', 5), ('D', 4), ('E', 6)]
    results_by_type = {}
    all_pass = True

    for family, rank in test_cases:
        label = f"{family}_{rank}"
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency
        basis, orbits = orbit_hilbert_basis(adj)
        n_vertices, n_orbits = basis.shape

        # Check orthonormality: B^T @ B should be identity
        gram = basis.T @ basis
        identity = np.eye(n_orbits)
        error = np.max(np.abs(gram - identity))
        is_orthonormal = error < 1e-10

        # Check each column has unit norm
        norms = [np.linalg.norm(basis[:, i]) for i in range(n_orbits)]
        all_unit = all(abs(n - 1.0) < 1e-10 for n in norms)

        # Check columns are orthogonal
        max_dot = 0.0
        for i in range(n_orbits):
            for j in range(i + 1, n_orbits):
                dot = abs(np.dot(basis[:, i], basis[:, j]))
                max_dot = max(max_dot, dot)
        is_orthogonal = max_dot < 1e-10

        passed = is_orthonormal and all_unit and is_orthogonal
        all_pass = all_pass and passed

        print(f"  {label}: {n_vertices} vertices, {n_orbits} orbits, "
              f"orthonormal={is_orthonormal}, max_error={error:.2e}")

        results_by_type[label] = {
            'n_vertices': n_vertices,
            'n_orbits': n_orbits,
            'gram_max_error': float(error),
            'is_orthonormal': is_orthonormal,
            'all_unit_norm': all_unit,
            'is_orthogonal': is_orthogonal,
            'max_off_diagonal_dot': float(max_dot),
            'norms': [float(n) for n in norms],
            'PASS': passed,
        }

    result = {
        'test': 'T1_orbit_basis_orthonormal',
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T2_orbit_gram_positive_definite():
    """T2: Orbit Gram matrix is positive DEFINITE for all ADE types."""
    diagrams = all_ade_diagrams(max_rank=8)
    all_pass = True
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency
        basis, orbits = orbit_hilbert_basis(adj)

        # Gram matrix of orbit basis
        gram = basis.T @ basis
        eigenvalues = np.linalg.eigvalsh(gram)
        min_eig = float(np.min(eigenvalues))
        max_eig = float(np.max(eigenvalues))

        # Positive definite means ALL eigenvalues > 0 (not just >= 0)
        is_pd = min_eig > 1e-10

        # Also check it's actually the identity (since basis should be orthonormal)
        identity_error = float(np.max(np.abs(gram - np.eye(len(orbits)))))
        is_identity = identity_error < 1e-10

        passed = is_pd
        all_pass = all_pass and passed

        print(f"  {label}: {len(orbits)} orbits, min_eig={min_eig:.6f}, "
              f"PD={is_pd}, identity={is_identity}")

        results_by_type[label] = {
            'n_orbits': len(orbits),
            'eigenvalues': [float(e) for e in eigenvalues],
            'min_eigenvalue': min_eig,
            'max_eigenvalue': max_eig,
            'is_positive_definite': is_pd,
            'is_identity': is_identity,
            'identity_error': identity_error,
            'PASS': passed,
        }

    result = {
        'test': 'T2_orbit_gram_positive_definite',
        'n_diagrams': len(diagrams),
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T3_orbit_dimension_matches():
    """T3: Orbit dimension = Aut(G) orbits, matches complement-spectrum orbits."""
    diagrams = all_ade_diagrams(max_rank=8)
    all_pass = True
    results_by_type = {}

    for diag in diagrams:
        label = diag.name
        adj = diag.adjacency
        n = adj.shape[0]

        # Method 1: orbit_hilbert_basis (from automorphisms via vertex_orbits)
        basis, orbits_aut = orbit_hilbert_basis(adj)
        n_orbits_aut = len(orbits_aut)

        # Method 2: complement-spectrum orbits
        # Vertices with same complement spectrum are in the same orbit
        spectra = {}
        for v in range(n):
            spec = complement_spectrum(adj, v)
            key = tuple(np.round(spec, decimals=10))
            if key not in spectra:
                spectra[key] = []
            spectra[key].append(v)
        n_orbits_spec = len(spectra)

        # Method 3: Hilbert space dimension from basis matrix
        n_orbits_hilbert = basis.shape[1]

        # All three should match
        all_match = (n_orbits_aut == n_orbits_spec == n_orbits_hilbert)
        passed = all_match
        all_pass = all_pass and passed

        print(f"  {label}: aut_orbits={n_orbits_aut}, spec_orbits={n_orbits_spec}, "
              f"hilbert_dim={n_orbits_hilbert}, match={all_match}")

        results_by_type[label] = {
            'n_vertices': n,
            'n_orbits_automorphism': n_orbits_aut,
            'n_orbits_complement_spectrum': n_orbits_spec,
            'n_orbits_hilbert_dim': n_orbits_hilbert,
            'all_match': all_match,
            'aut_orbit_members': [sorted(o) for o in orbits_aut],
            'spec_orbit_members': [sorted(v) for v in spectra.values()],
            'PASS': passed,
        }

    result = {
        'test': 'T3_orbit_dimension_matches',
        'n_diagrams': len(diagrams),
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


def test_T4_trivial_aut_classical_limit():
    """T4: Trivial Aut (E_7, E_8) gives dim = n (classical limit)."""
    # E_7 and E_8 have no nontrivial automorphisms (Aut = trivial group)
    # So every vertex is its own orbit, and orbit Hilbert space = full vertex space
    test_cases = [('E', 7), ('E', 8)]
    results_by_type = {}
    all_pass = True

    for family, rank in test_cases:
        label = f"{family}_{rank}"
        diag = DynkinDiagram(family, rank)
        adj = diag.adjacency
        n = adj.shape[0]

        # Compute automorphisms
        auts = graph_automorphisms(adj)
        n_auts = len(auts)

        # Orbit structure
        basis, orbits = orbit_hilbert_basis(adj)
        n_orbits = len(orbits)

        # Check: trivial Aut means only identity
        is_trivial = n_auts == 1

        # Check: dim = n (each vertex is its own orbit)
        dim_equals_n = n_orbits == n

        # Check: all orbits are singletons
        all_singleton = all(len(o) == 1 for o in orbits)

        # Classical limit: orbit basis = standard basis
        identity_error = float(np.max(np.abs(basis @ basis.T - np.eye(n))))

        passed = is_trivial and dim_equals_n and all_singleton
        all_pass = all_pass and passed

        print(f"  {label}: n={n}, |Aut|={n_auts}, orbits={n_orbits}, "
              f"trivial={is_trivial}, dim=n: {dim_equals_n}")

        results_by_type[label] = {
            'n_vertices': n,
            'n_automorphisms': n_auts,
            'n_orbits': n_orbits,
            'is_trivial_aut': is_trivial,
            'dim_equals_n': dim_equals_n,
            'all_singleton_orbits': all_singleton,
            'basis_completeness_error': identity_error,
            'PASS': passed,
        }

    result = {
        'test': 'T4_trivial_aut_classical_limit',
        'results_by_type': results_by_type,
        'PASS': all_pass,
    }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 01: Orbit Quotient as Hilbert Space")
    print("Milestone 14, Block A")
    print("=" * 70)

    results = {}
    scorecard = []

    tests = [
        ("T1", test_T1_orbit_basis_orthonormal),
        ("T2", test_T2_orbit_gram_positive_definite),
        ("T3", test_T3_orbit_dimension_matches),
        ("T4", test_T4_trivial_aut_classical_limit),
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
        'experiment': 'exp_01_orbit_hilbert_space',
        'milestone': 14,
        'block': 'A',
        'results': results,
        'scorecard': {f"T{i+1}": s for i, s in enumerate(scorecard)},
        'score': f"{n_pass}/{n_total}",
        'n_pass': n_pass,
        'n_total': n_total,
    }

    save_m14_results('exp_01_orbit_hilbert_space', _convert_numpy(save_data))
    return n_pass, n_total


if __name__ == "__main__":
    main()
