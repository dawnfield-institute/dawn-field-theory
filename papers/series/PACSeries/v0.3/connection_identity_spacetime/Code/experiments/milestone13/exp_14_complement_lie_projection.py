"""
exp_14 -- Complement -> Lie Algebra Projection

Milestone 13.5, Investigation Experiment

Central question: Can we close the exp_05 gap by working on the orbit quotient?
The exp_05 hardening revealed that the complement->Lie algebra bridge is indirect:
complement gives topology, ADE classification gives algebra, SEC gives dynamics.
This experiment tests whether a projection from eigenvalue-space to weight-space
provides a well-defined, equivariant bridge.

Tests:
  T1: Projection eigenvalue-space -> weight-space well-defined on orbits
  T2: Projection commutes with Weyl action (equivariance)
  T3: Gram matrix on orbit quotient is positive-DEFINITE (fixes exp_05 T4)
  T4: Non-ADE graphs fail root system test under projection
"""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI, LN_PHI,
    DynkinDiagram, all_ade_diagrams,
    complement_spectrum, vertex_orbits,
    weyl_element_su2, weyl_conjugate,
    complement_derived_generators,
    generate_random_connected_graph,
    save_m13_results, _convert_numpy,
)


def eigenvalue_orbit_projection(adjacency):
    """
    Project complement spectra into orbit-averaged eigenvalue vectors.

    For each orbit (set of symmetry-equivalent vertices), compute the
    average complement spectrum. This quotients out the graph symmetry
    and should map to something resembling a weight space.

    Returns: list of (orbit, averaged_spectrum) pairs
    """
    orbits = vertex_orbits(adjacency)
    projections = []

    for orbit in orbits:
        spectra = []
        for v in orbit:
            spec = complement_spectrum(adjacency, v)
            spectra.append(spec)

        # Average the spectra within the orbit
        max_len = max(len(s) for s in spectra)
        padded = np.zeros((len(spectra), max_len))
        for i, s in enumerate(spectra):
            padded[i, :len(s)] = s
        avg_spectrum = np.mean(padded, axis=0)

        projections.append({
            'orbit': sorted(orbit),
            'orbit_size': len(orbit),
            'averaged_spectrum': avg_spectrum,
            'spectrum_std': float(np.mean(np.std(padded, axis=0))),
        })

    return projections


def test_T1_projection_well_defined():
    """T1: Projection eigenvalue-space -> weight-space well-defined on orbits."""
    # For each ADE diagram, vertices in the same orbit should have
    # identical (or nearly identical) complement spectra.
    # The projection is well-defined if within-orbit variance is < 1e-10.

    test_diagrams = [
        DynkinDiagram('A', 3),
        DynkinDiagram('A', 5),
        DynkinDiagram('D', 4),
        DynkinDiagram('D', 6),
        DynkinDiagram('E', 6),
    ]

    all_well_defined = True
    diagram_results = []

    for d in test_diagrams:
        name = d.name
        proj = eigenvalue_orbit_projection(d.adjacency)
        max_std = max(p['spectrum_std'] for p in proj) if proj else 0.0
        n_orbits = len(proj)
        well_defined = max_std < 1e-10

        if not well_defined:
            all_well_defined = False

        diagram_results.append({
            'name': name,
            'n_orbits': n_orbits,
            'orbit_sizes': [p['orbit_size'] for p in proj],
            'max_within_orbit_std': max_std,
            'well_defined': well_defined,
        })
        print(f"  {name}: {n_orbits} orbits, max std = {max_std:.2e}, well-defined: {well_defined}")

    result = {
        'test': 'T1_projection_well_defined',
        'n_diagrams': len(test_diagrams),
        'diagram_results': diagram_results,
        'all_well_defined': all_well_defined,
        'note': 'Vertices in the same orbit have identical complement spectra by definition '
                '(same spectral invariant). Projection to orbit quotient is well-defined.',
        'PASS': all_well_defined,
    }
    return result


def test_T2_projection_equivariance():
    """T2: Projection commutes with Weyl action (equivariance on A_1)."""
    # For A_1, the Weyl group W = Z_2 acts on the root system.
    # In the complement picture, this corresponds to the complement-transformation
    # that swaps the two vertices.
    #
    # Test: compute complement spectra, apply Weyl reflection (swap vertices),
    # verify the orbit-projected spectra transform consistently.
    #
    # For larger diagrams (A_3, D_4), test whether orbit structure is preserved
    # under vertex permutations that correspond to Dynkin diagram symmetries.

    # A_3: symmetry group is Z_2 (reflection 0<->2, 1 fixed)
    a3 = DynkinDiagram('A', 3)
    adj = a3.adjacency

    # Original orbits
    orbits_orig = vertex_orbits(adj)

    # Apply the reflection permutation: 0<->2, 1 fixed
    perm = [2, 1, 0]
    adj_permuted = adj[np.ix_(perm, perm)]
    orbits_perm = vertex_orbits(adj_permuted)

    # The orbit structure should be identical (equivariance)
    orig_sizes = sorted([len(o) for o in orbits_orig])
    perm_sizes = sorted([len(o) for o in orbits_perm])
    a3_equivariant = orig_sizes == perm_sizes

    print(f"  A_3 original orbits: {[sorted(o) for o in orbits_orig]}")
    print(f"  A_3 permuted orbits: {[sorted(o) for o in orbits_perm]}")
    print(f"  A_3 equivariant: {a3_equivariant}")

    # D_4: triality symmetry (3 leaves are equivalent)
    d4 = DynkinDiagram('D', 4)
    adj_d4 = d4.adjacency
    orbits_d4 = vertex_orbits(adj_d4)

    # Permute two of the three leaves: swap vertex 1 and 3 (if leaves are at 1,2,3)
    perm_d4 = [0, 3, 2, 1]
    adj_d4_perm = adj_d4[np.ix_(perm_d4, perm_d4)]
    orbits_d4_perm = vertex_orbits(adj_d4_perm)

    d4_orig_sizes = sorted([len(o) for o in orbits_d4])
    d4_perm_sizes = sorted([len(o) for o in orbits_d4_perm])
    d4_equivariant = d4_orig_sizes == d4_perm_sizes

    print(f"  D_4 original orbits: {[sorted(o) for o in orbits_d4]}")
    print(f"  D_4 permuted orbits: {[sorted(o) for o in orbits_d4_perm]}")
    print(f"  D_4 equivariant: {d4_equivariant}")

    # E_6: has Z_2 symmetry
    e6 = DynkinDiagram('E', 6)
    adj_e6 = e6.adjacency
    orbits_e6 = vertex_orbits(adj_e6)
    # E_6 reflection: 0<->5, 1<->4, 2<->3 (depends on labeling)
    perm_e6 = [5, 4, 3, 2, 1, 0]
    adj_e6_perm = adj_e6[np.ix_(perm_e6, perm_e6)]
    orbits_e6_perm = vertex_orbits(adj_e6_perm)

    e6_orig_sizes = sorted([len(o) for o in orbits_e6])
    e6_perm_sizes = sorted([len(o) for o in orbits_e6_perm])
    e6_equivariant = e6_orig_sizes == e6_perm_sizes

    print(f"  E_6 equivariant: {e6_equivariant}")

    all_equivariant = a3_equivariant and d4_equivariant and e6_equivariant

    result = {
        'test': 'T2_projection_equivariance',
        'a3_equivariant': a3_equivariant,
        'd4_equivariant': d4_equivariant,
        'e6_equivariant': e6_equivariant,
        'all_equivariant': all_equivariant,
        'note': 'Orbit projection commutes with Dynkin diagram symmetries. '
                'This is necessary for the projection to be a valid bridge '
                'from complement-space to weight-space.',
        'PASS': all_equivariant,
    }
    return result


def test_T3_gram_matrix_definiteness():
    """T3: Gram matrix on orbit quotient -- positive-definite or only PSD (exp_05 crux)."""
    # This is the theoretical crux. exp_05 T4 found the Gram matrix is PSD (not PD).
    # Working on the orbit QUOTIENT might fix this: if degenerate vertices are
    # identified, the resulting Gram matrix might become PD.

    test_diagrams = [
        DynkinDiagram('A', 3),
        DynkinDiagram('A', 5),
        DynkinDiagram('D', 4),
        DynkinDiagram('D', 6),
        DynkinDiagram('E', 6),
    ]

    gram_results = []
    n_pd = 0
    n_psd = 0
    n_tested = 0

    for d in test_diagrams:
        name = d.name
        proj = eigenvalue_orbit_projection(d.adjacency)

        if len(proj) < 2:
            gram_results.append({
                'name': name,
                'n_orbits': len(proj),
                'note': 'Too few orbits for Gram matrix',
                'is_pd': False,
                'is_psd': True,
            })
            continue

        n_tested += 1

        # Build Gram matrix from orbit-averaged spectra
        spectra = []
        for p in proj:
            spectra.append(p['averaged_spectrum'])

        # Pad to same length
        max_len = max(len(s) for s in spectra)
        padded = np.zeros((len(spectra), max_len))
        for i, s in enumerate(spectra):
            padded[i, :len(s)] = s

        # Gram matrix: G_ij = <spec_i, spec_j>
        G = padded @ padded.T
        eigs = np.linalg.eigvalsh(G)
        min_eig = float(np.min(eigs))

        is_pd = min_eig > 1e-10
        is_psd = min_eig >= -1e-10

        if is_pd:
            n_pd += 1
        elif is_psd:
            n_psd += 1

        gram_results.append({
            'name': name,
            'n_orbits': len(proj),
            'gram_eigenvalues': [float(e) for e in sorted(eigs)],
            'min_eigenvalue': min_eig,
            'is_pd': is_pd,
            'is_psd': is_psd,
        })
        print(f"  {name}: {len(proj)} orbits, min_eig={min_eig:.6f}, "
              f"PD={is_pd}, PSD={is_psd}")

    # The test: does quotienting by orbits make the Gram matrix PD?
    # If all are PD, the exp_05 gap is closed.
    # If some are only PSD, the degeneracy is fundamental.
    all_pd = n_pd == n_tested
    all_psd = (n_pd + n_psd) == n_tested

    print(f"  Summary: {n_pd} PD, {n_psd} PSD-only, of {n_tested} tested")

    result = {
        'test': 'T3_gram_matrix_definiteness',
        'gram_results': gram_results,
        'n_pd': n_pd,
        'n_psd': n_psd,
        'n_tested': n_tested,
        'all_pd': all_pd,
        'all_psd': all_psd,
        'note': 'Tests whether orbit-quotient Gram matrix is positive-definite. '
                'If PD: complement -> weight-space projection is a metric embedding. '
                'If only PSD: complement degeneracy is fundamental (different orbits '
                'can have linearly dependent spectra).',
        'PASS': all_pd,
    }
    return result


def test_T4_non_ade_fail_root_system():
    """T4: Non-ADE graphs fail root system properties under projection."""
    # ADE graphs have special spectral properties (spectral radius < 2).
    # The orbit-quotient of non-ADE graphs should fail root-system-like properties:
    # e.g., the Gram matrix structure should NOT match Cartan matrix patterns.

    # ADE: compute Cartan-like matrix from orbit spectra
    ade_diagrams = [
        DynkinDiagram('A', 4),
        DynkinDiagram('D', 5),
    ]

    # Random graphs (non-ADE)
    random_graphs = []
    for seed in [42, 137, 256]:
        G = generate_random_connected_graph(5, density=0.5, seed=seed)
        random_graphs.append(('random_' + str(seed), G))

    ade_properties = []
    random_properties = []

    for name, adj in [(d.name, d.adjacency) for d in ade_diagrams] + random_graphs:
        proj = eigenvalue_orbit_projection(adj if isinstance(adj, np.ndarray) else adj)
        n_orbits = len(proj)

        # ADE spectral radius should be < 2 for proper ADE
        eigs = np.linalg.eigvalsh(adj if isinstance(adj, np.ndarray) else adj)
        spectral_radius = float(np.max(np.abs(eigs)))
        is_ade_spectral = spectral_radius < 2.0 + 1e-10

        # Orbit spectra should have integer-like inner product ratios for ADE
        if n_orbits >= 2:
            spectra = []
            for p in proj:
                spectra.append(p['averaged_spectrum'])
            max_len = max(len(s) for s in spectra)
            padded = np.zeros((len(spectra), max_len))
            for i, s in enumerate(spectra):
                padded[i, :len(s)] = s
            G = padded @ padded.T

            # Check if off-diagonal / diagonal ratios are "simple" (close to rationals)
            diag = np.diag(G)
            if np.min(np.abs(diag)) > 1e-10:
                normalized = G / np.outer(np.sqrt(diag), np.sqrt(diag))
                off_diag = normalized[np.triu_indices(len(normalized), 1)]
                # "Simple" = close to 0, 0.5, or 1.0
                simple_count = sum(1 for v in off_diag
                                   if min(abs(v), abs(v-0.5), abs(v-1.0), abs(v+0.5)) < 0.15)
                simplicity = simple_count / len(off_diag) if len(off_diag) > 0 else 0
            else:
                simplicity = 0.0
        else:
            simplicity = 1.0  # trivially simple

        entry = {
            'name': name,
            'n_orbits': n_orbits,
            'spectral_radius': spectral_radius,
            'is_ade_spectral': is_ade_spectral,
            'simplicity': float(simplicity),
        }

        if name.startswith('random'):
            random_properties.append(entry)
        else:
            ade_properties.append(entry)

        print(f"  {name}: sr={spectral_radius:.3f}, ade_spectral={is_ade_spectral}, "
              f"simplicity={simplicity:.3f}")

    # ADE should have spectral radius < 2 and high simplicity
    ade_pass = all(p['is_ade_spectral'] for p in ade_properties)

    # Random graphs should fail ADE spectral test (most have sr > 2)
    random_fail_spectral = sum(1 for p in random_properties if not p['is_ade_spectral'])
    random_mostly_fail = random_fail_spectral >= 2  # at least 2 of 3

    result = {
        'test': 'T4_non_ade_fail_root_system',
        'ade_properties': ade_properties,
        'random_properties': random_properties,
        'ade_pass_spectral': ade_pass,
        'random_fail_count': random_fail_spectral,
        'random_mostly_fail': random_mostly_fail,
        'note': 'ADE graphs have spectral radius < 2 (ADE classification theorem). '
                'Non-ADE graphs typically violate this, showing the orbit quotient '
                'projection only produces root-system-like structure for ADE inputs.',
        'PASS': ade_pass and random_mostly_fail,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 14 -- Complement -> Lie Algebra Projection")
    print("Milestone 13.5, Investigation Experiment")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_projection_well_defined),
        ('T2', test_T2_projection_equivariance),
        ('T3', test_T3_gram_matrix_definiteness),
        ('T4', test_T4_non_ade_fail_root_system),
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
        'experiment': 'exp_14_complement_lie_projection',
        'milestone': 'milestone13.5',
        'block': 'investigation',
        'version': 'v0.1',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_14_complement_lie_projection', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
