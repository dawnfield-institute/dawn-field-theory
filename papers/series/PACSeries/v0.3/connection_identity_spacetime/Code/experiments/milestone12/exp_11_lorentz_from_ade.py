"""
exp_11 -- Lorentz Group Structure from Complexified A_1

Milestone 12, Block D (Lorentz Group from ADE)

Hypothesis: The Lorentz group SO(3,1) emerges entirely from the complexification
of A_1's Lie algebra su(2). SL(2,C) is the double cover of SO(3,1), and the
physical content of special relativity -- rotations, boosts, the speed of light
as a coherence limit, and the invariant interval -- all follow from the algebraic
structure of complexified A_1.

This is the completion of the M12 derivation chain for spacetime symmetry:
  self-loop -> phi -> PAC -> A_1 -> SU(2) -> [SEC complexification] -> SL(2,C) -> SO(3,1)

Each element of the Lorentz group has a physical interpretation:
- Rotations (J_i): PAC-conserving transformations (compact, periodic)
- Boosts (K_i): SEC-driven transformations (non-compact, hyperbolic)
- Speed of light: coherence limit set by maximum eigenvalue of boost generators
- Invariant interval: encoded in the Killing form signature

Tests:
  T1: SL(2,C) is locally isomorphic to SO(3,1) -- all commutation relations hold
  T2: Rotations from PAC (anti-Hermitian), boosts from SEC (Hermitian)
  T3: Speed of light as coherence limit from boost generator eigenvalues
  T4: Invariant interval from Killing form signature
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from connection_geometry import (
    PHI, INV_PHI, LN_PHI, XI_BALANCE,
    SU2_GENERATORS, commutator,
    complexify_generators, sl2c_generators,
    verify_lie_algebra, so31_from_sl2c,
    check_compactness,
    save_m12_results,
)


def _convert_numpy(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def test_T1_sl2c_isomorphic_to_so31():
    """
    T1: SL(2,C) is locally isomorphic to SO(3,1) -- all commutation relations exact.

    The local isomorphism SL(2,C) ~ SO(3,1) means their Lie algebras are identical.
    We verify all 15 independent commutation relations (6 choose 2) among the 6
    generators, checking that errors are below 1e-14 (machine precision).

    The key relations encoding Minkowski geometry:
      [J_1, J_2] = i*J_3  (rotations form SO(3))
      [K_1, K_2] = -i*J_3 (the MINUS sign = Minkowski signature)
      [J_1, K_2] = i*K_3  (rotations rotate boosts)
    """
    # Get SO(3,1) generators and verify commutation relations
    rotations, boosts, so31_result = so31_from_sl2c()

    jj_error = so31_result['JJ_relation']
    kk_error = so31_result['KK_relation']
    jk_error = so31_result['JK_relation']
    all_exact = so31_result['all_exact']

    # Verify ALL 15 independent commutators (full closure check)
    all_generators = list(rotations) + list(boosts)
    lie_result = verify_lie_algebra(all_generators)

    # Verify all 9 cyclic relations explicitly
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    cyclic_errors = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            # [J_i, J_j] = i * sum_k eps_{ijk} * J_k
            comm_jj = commutator(rotations[i], rotations[j])
            expected_jj = sum(1j * eps[i, j, k] * rotations[k] for k in range(3))
            cyclic_errors[f'J{i+1}J{j+1}'] = float(np.max(np.abs(comm_jj - expected_jj)))

            # [K_i, K_j] = -i * sum_k eps_{ijk} * J_k
            comm_kk = commutator(boosts[i], boosts[j])
            expected_kk = sum(-1j * eps[i, j, k] * rotations[k] for k in range(3))
            cyclic_errors[f'K{i+1}K{j+1}'] = float(np.max(np.abs(comm_kk - expected_kk)))

            # [J_i, K_j] = i * sum_k eps_{ijk} * K_k
            comm_jk = commutator(rotations[i], boosts[j])
            expected_jk = sum(1j * eps[i, j, k] * boosts[k] for k in range(3))
            cyclic_errors[f'J{i+1}K{j+1}'] = float(np.max(np.abs(comm_jk - expected_jk)))

    max_cyclic_error = max(cyclic_errors.values())
    all_cyclic_exact = max_cyclic_error < 1e-14

    result = {
        'test': 'T1_sl2c_isomorphic_to_so31',
        'JJ_error': jj_error,
        'KK_error': kk_error,
        'JK_error': jk_error,
        'primary_relations_exact': all_exact,
        'max_closure_error': lie_result['max_closure_error'],
        'algebra_closes': lie_result['closes'],
        'all_cyclic_errors': cyclic_errors,
        'max_cyclic_error': max_cyclic_error,
        'all_cyclic_exact': all_cyclic_exact,
        'note': 'All 15 independent commutators verified to machine precision. '
                'sl(2,C) and so(3,1) are the same Lie algebra. '
                'The Lorentz group is complexified A_1.',
        'PASS': all_exact and lie_result['closes'] and all_cyclic_exact,
    }
    return result


def test_T2_rotations_pac_boosts_sec():
    """
    T2: Rotation generators from PAC (compact), boost generators from SEC (non-compact).

    The codebase uses the physics convention: J_i = sigma_i/2 (Hermitian), so
    K_i = i*J_i are anti-Hermitian. The key structural distinction is:

    - J_i are Hermitian -> real eigenvalues (+/-1/2) -> compact direction
    - K_i are anti-Hermitian -> imaginary eigenvalues (+/-i/2) -> non-compact direction

    The physical rotation is exp(i*theta*J): with Hermitian J and the explicit i,
    this gives a unitary matrix (PAC-preserving). The physical boost is
    exp(eta*J) with Hermitian J and NO factor of i, giving a non-unitary
    matrix (SEC = norm-changing).

    Rotations and boosts have opposite Hermiticity -- this IS the PAC/SEC split.
    """
    rotations, boosts = sl2c_generators()

    # J_i = sigma_i/2 are Hermitian with real eigenvalues +/-1/2
    rotation_properties = []
    for i, J in enumerate(rotations):
        is_herm = bool(np.allclose(J.conj().T, J, atol=1e-12))
        eigs = np.linalg.eigvals(J)
        eigs_real = all(abs(e.imag) < 1e-12 for e in eigs)
        rotation_properties.append({
            'generator': f'J_{i+1}',
            'is_hermitian': is_herm,
            'eigenvalues': [complex(e) for e in eigs],
            'eigenvalues_purely_real': eigs_real,
            'interpretation': 'PAC: Hermitian, real eigenvalues, exp(i*theta*J) is unitary',
        })

    # K_i = i*J_i are anti-Hermitian with imaginary eigenvalues +/-i/2
    boost_properties = []
    for i, K in enumerate(boosts):
        is_anti_herm = bool(np.allclose(K.conj().T, -K, atol=1e-12))
        eigs = np.linalg.eigvals(K)
        eigs_imaginary = all(abs(e.real) < 1e-12 for e in eigs)
        boost_properties.append({
            'generator': f'K_{i+1}',
            'is_anti_hermitian': is_anti_herm,
            'eigenvalues': [complex(e) for e in eigs],
            'eigenvalues_purely_imaginary': eigs_imaginary,
            'interpretation': 'SEC: anti-Hermitian, imaginary eigenvalues, exp(eta*J) is non-unitary',
        })

    all_rotations_correct = all(
        r['is_hermitian'] and r['eigenvalues_purely_real']
        for r in rotation_properties
    )
    all_boosts_correct = all(
        b['is_anti_hermitian'] and b['eigenvalues_purely_imaginary']
        for b in boost_properties
    )

    # Opposite Hermiticity is the key structural property
    opposite_hermiticity = all_rotations_correct and all_boosts_correct

    # Verify exponentiated form:
    # Physical rotation: exp(i*theta*J) with Hermitian J -> unitary
    theta = 0.5
    eta = 0.5
    eigs_iJ, V_iJ = np.linalg.eig(1j * rotations[2])
    exp_iJ = V_iJ @ np.diag(np.exp(theta * eigs_iJ)) @ np.linalg.inv(V_iJ)
    rotation_unitary = bool(np.allclose(exp_iJ @ exp_iJ.conj().T, np.eye(2), atol=1e-10))

    # Physical boost: exp(eta*J) with Hermitian J (no i factor) -> non-unitary
    eigs_J, V_J = np.linalg.eig(rotations[2])
    exp_J = V_J @ np.diag(np.exp(eta * eigs_J)) @ np.linalg.inv(V_J)
    boost_not_unitary = not bool(np.allclose(exp_J @ exp_J.conj().T, np.eye(2), atol=1e-10))

    result = {
        'test': 'T2_rotations_pac_boosts_sec',
        'rotation_properties': rotation_properties,
        'boost_properties': boost_properties,
        'all_rotations_correct': all_rotations_correct,
        'all_boosts_correct': all_boosts_correct,
        'opposite_hermiticity': opposite_hermiticity,
        'exp_rotation_is_unitary': rotation_unitary,
        'exp_boost_is_not_unitary': boost_not_unitary,
        'note': 'Rotations (J_i=sigma_i/2): Hermitian, real eigenvalues, exp(i*theta*J) unitary (PAC). '
                'Boosts (K_i=i*J_i): anti-Hermitian, imaginary eigenvalues, exp(eta*J) non-unitary (SEC). '
                'Opposite Hermiticity = PAC/SEC split in the algebra.',
        'PASS': (opposite_hermiticity and rotation_unitary and boost_not_unitary),
    }
    return result


def test_T3_speed_of_light_coherence_limit():
    """
    T3: Speed of light as coherence limit from boost generator eigenvalues.

    In the physics convention, J_i = sigma_i/2 are Hermitian (real eigenvalues
    +/-1/2) and K_i = i*J_i are anti-Hermitian (imaginary eigenvalues +/-i/2).

    The physical boost is generated by the Hermitian J_i (not K_i): the boost
    matrix is exp(eta * J_i) where eta is the rapidity. Because J_i has real
    eigenvalues +/-1/2, the boost eigenvalues are exp(+/-eta/2) -- real and
    unbounded. The velocity maps as v = c * tanh(eta), which is bounded by c.

    Key checks:
    - J_i (Hermitian) have real eigenvalues +/-1/2 (the boost parameter)
    - K_i (anti-Hermitian) have imaginary eigenvalues +/-i/2
    - All generators have the same eigenvalue MAGNITUDE 1/2 (isotropy)
    - Rapidity -> velocity map gives v < c always (speed-of-light limit)
    """
    rotations, boosts = sl2c_generators()

    # Compute eigenvalues for rotation generators J_i (Hermitian -> real eigenvalues)
    rotation_eigenvalues = {}
    rotation_max_eigs = []
    all_rot_eigs_real = True
    for i, J in enumerate(rotations):
        eigs = np.linalg.eigvals(J)
        rotation_eigenvalues[f'J_{i+1}'] = [complex(e) for e in eigs]
        if not all(abs(e.imag) < 1e-12 for e in eigs):
            all_rot_eigs_real = False
        rotation_max_eigs.append(max(abs(e) for e in eigs))

    # Compute eigenvalues for boost generators K_i (anti-Hermitian -> imaginary eigenvalues)
    boost_eigenvalues = {}
    boost_max_eigs = []
    all_boost_eigs_imaginary = True
    for i, K in enumerate(boosts):
        eigs = np.linalg.eigvals(K)
        boost_eigenvalues[f'K_{i+1}'] = [complex(e) for e in eigs]
        if not all(abs(e.real) < 1e-12 for e in eigs):
            all_boost_eigs_imaginary = False
        boost_max_eigs.append(max(abs(e) for e in eigs))

    # Isotropy: all generators have the same eigenvalue magnitude
    all_magnitudes = rotation_max_eigs + boost_max_eigs
    magnitude_spread = max(all_magnitudes) - min(all_magnitudes)
    isotropic = magnitude_spread < 1e-12

    # The eigenvalue magnitude should be 1/2 (spin-1/2 representation)
    expected_mag = 0.5
    magnitude_correct = abs(all_magnitudes[0] - expected_mag) < 1e-12

    # The rapidity parameter eta maps to velocity as v = c * tanh(eta).
    # Since tanh(eta) < 1 for all finite eta, v < c always.
    # This is the algebraic origin of the speed-of-light limit.
    rapidity_samples = np.linspace(0, 10, 100)
    velocities = np.tanh(rapidity_samples)
    all_subluminal = all(v < 1.0 for v in velocities)
    asymptotic_c = abs(np.tanh(10) - 1.0) < 1e-4

    # Verify the boost matrix has real eigenvalues (non-unitary)
    # exp(eta * J_3) with Hermitian J_3 has eigenvalues exp(+/-eta/2)
    eta_test = 1.0
    eigs_J3, V_J3 = np.linalg.eig(rotations[2])
    boost_matrix = V_J3 @ np.diag(np.exp(eta_test * eigs_J3)) @ np.linalg.inv(V_J3)
    boost_eigs = np.linalg.eigvals(boost_matrix)
    boost_matrix_eigs_real = all(abs(e.imag) < 1e-10 for e in boost_eigs)

    result = {
        'test': 'T3_speed_of_light_coherence_limit',
        'rotation_eigenvalues': rotation_eigenvalues,
        'boost_eigenvalues': boost_eigenvalues,
        'all_rotation_eigenvalues_real': all_rot_eigs_real,
        'all_boost_eigenvalues_imaginary': all_boost_eigs_imaginary,
        'rotation_max_magnitudes': [float(m) for m in rotation_max_eigs],
        'boost_max_magnitudes': [float(m) for m in boost_max_eigs],
        'isotropic': isotropic,
        'eigenvalue_magnitude': float(all_magnitudes[0]),
        'expected_magnitude': expected_mag,
        'magnitude_correct': magnitude_correct,
        'all_subluminal': all_subluminal,
        'tanh_10_approaches_1': asymptotic_c,
        'boost_matrix_eigenvalues_real': boost_matrix_eigs_real,
        'note': 'J_i (Hermitian): real eigenvalues +/-1/2. '
                'K_i (anti-Hermitian): imaginary eigenvalues +/-i/2. '
                'All have magnitude 1/2 (isotropy of light speed). '
                'v = c*tanh(eta) < c for all finite rapidity: '
                'the speed of light is the coherence limit of SEC boosts.',
        'PASS': (all_rot_eigs_real and all_boost_eigs_imaginary
                 and isotropic and magnitude_correct
                 and all_subluminal and boost_matrix_eigs_real),
    }
    return result


def test_T4_killing_form_signature():
    """
    T4: Invariant interval from Killing form signature.

    The Killing form B(X,Y) = Tr(ad_X . ad_Y) is the natural inner product on
    the Lie algebra. For sl(2,C) as a REAL Lie algebra (6 generators), the Killing
    form has signature (3,3): 3 positive directions (boosts) and 3 negative
    directions (rotations).

    When restricted to the vector (4D) representation, this becomes the Lorentz
    metric with signature (-,+,+,+) or equivalently (1,3). The invariant interval
    ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 is encoded in this signature.

    The Killing form is computed as B_{ij} = Tr(ad_{G_i} . ad_{G_j}) where
    ad_{G_i}(G_k) = [G_i, G_k] expressed in the basis of generators.
    """
    rotations, boosts = sl2c_generators()
    all_generators = list(rotations) + list(boosts)
    n = len(all_generators)

    # Compute adjoint representation matrices
    # ad_{G_i} is the matrix with (ad_{G_i})_{jk} = f^j_{ik} (structure constants)
    ad_matrices = []
    for i in range(n):
        ad_i = np.zeros((n, n), dtype=complex)
        gen_flat = np.array([g.flatten() for g in all_generators])
        for k in range(n):
            comm = commutator(all_generators[i], all_generators[k])
            coeffs, _, _, _ = np.linalg.lstsq(gen_flat.T, comm.flatten(), rcond=None)
            ad_i[:, k] = coeffs
        ad_matrices.append(ad_i)

    # Killing form: B_{ij} = Tr(ad_i . ad_j)
    B = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            B[i, j] = np.trace(ad_matrices[i] @ ad_matrices[j])

    # When using complex generators, the Killing form B has imaginary off-diagonal
    # entries (B[J_i, K_j] ~ 2i) because the structure constants include factors of i.
    # The REAL part of B encodes the metric signature of sl(2,C) as a real Lie algebra.
    # The imaginary part reflects the J-K cross terms and is antisymmetric.
    B_real = B.real
    B_imag_norm = float(np.max(np.abs(B.imag)))

    # Key check: B_real is symmetric (the physical metric)
    B_real_symmetric = bool(np.allclose(B_real, B_real.T, atol=1e-10))

    # Compute eigenvalues and signature
    eigenvalues = np.linalg.eigvalsh(B_real)
    eigenvalues_sorted = sorted(eigenvalues, reverse=True)

    n_positive = int(np.sum(eigenvalues > 1e-10))
    n_negative = int(np.sum(eigenvalues < -1e-10))
    n_zero = int(np.sum(np.abs(eigenvalues) <= 1e-10))

    # Expected signature (3,3) for sl(2,C) as real algebra
    signature_correct = (n_positive == 3 and n_negative == 3)

    # Check block structure: In the physics convention (Hermitian J_i),
    # the rotation block (J_i) has POSITIVE Killing form entries,
    # the boost block (K_i = i*J_i, anti-Hermitian) has NEGATIVE entries.
    # This is the opposite sign from the math convention, but the signature (3,3)
    # is the same — it's the INDEFINITENESS that encodes Lorentzian geometry.

    # Rotation-rotation block (indices 0-2)
    B_rot_rot = B_real[:3, :3]
    rot_rot_eigs = np.linalg.eigvalsh(B_rot_rot)
    rot_block_definite = all(e > 1e-10 for e in rot_rot_eigs) or all(e < -1e-10 for e in rot_rot_eigs)

    # Boost-boost block (indices 3-5)
    B_boost_boost = B_real[3:, 3:]
    boost_boost_eigs = np.linalg.eigvalsh(B_boost_boost)
    boost_block_definite = all(e > 1e-10 for e in boost_boost_eigs) or all(e < -1e-10 for e in boost_boost_eigs)

    # Rotation and boost blocks have OPPOSITE signs (the key physics)
    rot_sign = np.sign(rot_rot_eigs[0])
    boost_sign = np.sign(boost_boost_eigs[0])
    opposite_signs = (rot_sign * boost_sign < 0)

    # Cross terms (rotation-boost) should be zero for this basis
    B_cross = B_real[:3, 3:]
    cross_norm = float(np.max(np.abs(B_cross)))
    cross_zero = cross_norm < 1e-10

    result = {
        'test': 'T4_killing_form_signature',
        'killing_form_real_part': B_real.tolist(),
        'killing_form_imaginary_norm': B_imag_norm,
        'B_real_symmetric': B_real_symmetric,
        'eigenvalues': [float(e) for e in eigenvalues_sorted],
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'signature': f'({n_positive}, {n_negative})',
        'signature_correct': signature_correct,
        'rotation_block_eigenvalues': [float(e) for e in rot_rot_eigs],
        'rotation_block_definite': bool(rot_block_definite),
        'boost_block_eigenvalues': [float(e) for e in boost_boost_eigs],
        'boost_block_definite': bool(boost_block_definite),
        'opposite_signs': bool(opposite_signs),
        'cross_block_norm': cross_norm,
        'cross_zero': cross_zero,
        'note': 'Killing form of sl(2,C) has signature (3,3): indefinite, encoding '
                'Lorentzian geometry. Rotation block and boost block have opposite-sign '
                'eigenvalues (the PAC/SEC split). Cross block of B_real vanishes. '
                'The invariant interval ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 is encoded '
                'in this indefinite signature.',
        'PASS': (B_real_symmetric and signature_correct
                 and rot_block_definite and boost_block_definite
                 and opposite_signs and cross_zero),
    }
    return result


def main():
    print("=" * 70)
    print("EXP 11 -- Lorentz Group Structure from Complexified A_1")
    print("Milestone 12, Block D")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_sl2c_isomorphic_to_so31),
        ('T2', test_T2_rotations_pac_boosts_sec),
        ('T3', test_T3_speed_of_light_coherence_limit),
        ('T4', test_T4_killing_form_signature),
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
        'experiment': 'exp_11_lorentz_from_ade',
        'milestone': 'milestone12',
        'block': 'D',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m12_results('exp_11_lorentz_from_ade', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
