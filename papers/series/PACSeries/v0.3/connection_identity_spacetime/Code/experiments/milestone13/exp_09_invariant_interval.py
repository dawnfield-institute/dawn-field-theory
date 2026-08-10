"""
exp_09 -- ds^2 Is the Unique Complement-Transformation-Invariant Bilinear Form

Milestone 13, Block C

Hypothesis: The spacetime interval ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 is the
UNIQUE (up to scale) bilinear form preserved by all complement-transformations.
The Killing form of sl(2,C) has signature (3,3); the 4D vector representation
yields signature (1,3) or (3,1) -- the Minkowski metric. By Schur's lemma, this
invariant form spans a 1-dimensional space: it is unique.

Tests (hardened v0.3):
  T1: Selectivity: sl(2,C) indefinite (3,3), su(2) definite (0,3), su(3) definite (0,8)
  T2: 4D signature (1,3) + causal structure preservation (timelike/spacelike/lightlike)
  T3: ds^2 preserved under 20+ transforms including large-eta, lightlike, cumulative
  T4: Invariant form proportional to Minkowski + alternative metrics NOT preserved
"""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, INV_PHI, LN_PHI,
    SU2_GENERATORS, commutator, sl2c_generators,
    killing_form, killing_form_for_algebra,
    so31_4d_generators, lorentz_invariant_form,
    save_m13_results, _convert_numpy,
)


def test_T1_killing_form_selectivity():
    """T1: sl(2,C) indefinite (3,3); su(2) and su(3) definite (selectivity)."""

    # sl(2,C): should be (3,3) -- indefinite
    rotations, boosts = sl2c_generators()
    sl2c_gens = list(rotations) + list(boosts)
    kf_sl2c = killing_form(sl2c_gens)

    # su(2): should be (0,3) -- negative definite (compact)
    kf_su2 = killing_form(list(SU2_GENERATORS))

    # su(3): build Gell-Mann generators / 2i
    # 8 generators of su(3)
    lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex) / (2j)
    lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex) / (2j)
    lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex) / (2j)
    lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex) / (2j)
    lambda5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex) / (2j)
    lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex) / (2j)
    lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex) / (2j)
    lambda8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / (2j * np.sqrt(3))
    su3_gens = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]
    B_su3 = killing_form_for_algebra(su3_gens)
    su3_eigs = np.linalg.eigvalsh(B_su3.real)
    su3_n_pos = int(np.sum(su3_eigs > 1e-10))
    su3_n_neg = int(np.sum(su3_eigs < -1e-10))

    print(f"  sl(2,C) Killing form: signature ({kf_sl2c['n_positive']},{kf_sl2c['n_negative']})")
    print(f"  su(2) Killing form: signature ({kf_su2['n_positive']},{kf_su2['n_negative']})")
    print(f"  su(3) Killing form: signature ({su3_n_pos},{su3_n_neg})")

    # Selectivity: only sl(2,C) is indefinite
    sl2c_indefinite = kf_sl2c['n_positive'] > 0 and kf_sl2c['n_negative'] > 0
    su2_definite = kf_su2['n_positive'] == 0 or kf_su2['n_negative'] == 0
    su3_definite = su3_n_pos == 0 or su3_n_neg == 0

    print(f"  sl(2,C) indefinite: {sl2c_indefinite}")
    print(f"  su(2) definite: {su2_definite}")
    print(f"  su(3) definite: {su3_definite}")

    result = {
        'test': 'T1_killing_form_selectivity',
        'sl2c_signature': f"({kf_sl2c['n_positive']},{kf_sl2c['n_negative']})",
        'su2_signature': f"({kf_su2['n_positive']},{kf_su2['n_negative']})",
        'su3_signature': f"({su3_n_pos},{su3_n_neg})",
        'sl2c_indefinite': sl2c_indefinite,
        'su2_definite': su2_definite,
        'su3_definite': su3_definite,
        'selectivity': 'SEC complexification A_1 -> sl(2,C) breaks the definite (compact) '
                        'Killing form into indefinite (non-compact). This is the algebraic '
                        'origin of Minkowski signature: boosts make the form indefinite.',
        'PASS': (kf_sl2c['n_positive'] == 3 and kf_sl2c['n_negative'] == 3 and
                 su2_definite and su3_definite),
    }
    return result


def test_T2_4d_signature_and_causality():
    """T2: 4D signature (1,3) + causal preservation (timelike/spacelike/lightlike)."""

    generators_4d = so31_4d_generators()
    inv_form = lorentz_invariant_form(generators_4d)

    eigs = inv_form['eigenvalues']
    n_pos = sum(1 for e in eigs if e > 1e-10)
    n_neg = sum(1 for e in eigs if e < -1e-10)

    # Extract the metric (normalized to have eigenvalues +/-1)
    form = inv_form['form']
    # Normalize: scale so largest abs eigenvalue is 1
    scale = np.max(np.abs(eigs))
    eta = form / scale if scale > 0 else form

    print(f"  4D invariant form signature: ({n_pos},{n_neg})")
    print(f"  Normalized form eigenvalues: {np.linalg.eigvalsh(eta)}")

    # Causal structure preservation test
    rotations_4d, boosts_4d = generators_4d

    # Define test vectors of each causal type
    eta_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    test_vectors = {
        'timelike': np.array([3.0, 1.0, 0.0, 0.0]),   # s^2 = -9+1 = -8
        'spacelike': np.array([1.0, 3.0, 0.0, 0.0]),   # s^2 = -1+9 = +8
        'lightlike': np.array([1.0, 1.0, 0.0, 0.0]),   # s^2 = -1+1 = 0
    }

    # Apply a Lorentz boost and check causal type is preserved
    Lambda = expm(0.8 * boosts_4d[0])  # moderate x-boost

    causal_results = {}
    all_preserved = True
    for vtype, v in test_vectors.items():
        s2_orig = float(v @ eta_metric @ v)
        v_prime = Lambda @ v
        s2_prime = float(v_prime @ eta_metric @ v_prime)

        # Causal type from sign of s^2
        orig_type = 'timelike' if s2_orig < -1e-10 else ('spacelike' if s2_orig > 1e-10 else 'lightlike')
        prime_type = 'timelike' if s2_prime < -1e-10 else ('spacelike' if s2_prime > 1e-10 else 'lightlike')
        preserved = orig_type == prime_type

        causal_results[vtype] = {
            's2_original': s2_orig,
            's2_transformed': s2_prime,
            'original_type': orig_type,
            'transformed_type': prime_type,
            'preserved': preserved,
        }
        if not preserved:
            all_preserved = False
        print(f"  {vtype}: s2={s2_orig:.4f} -> {s2_prime:.4f} ({prime_type}) preserved={preserved}")

    is_minkowski = (n_pos == 1 and n_neg == 3) or (n_pos == 3 and n_neg == 1)

    result = {
        'test': 'T2_4d_signature_and_causality',
        'signature': f'({n_pos},{n_neg})',
        'is_minkowski': is_minkowski,
        'causal_results': causal_results,
        'all_causal_preserved': all_preserved,
        'PASS': is_minkowski and all_preserved,
    }
    return result


def test_T3_ds2_preserved_extensive():
    """T3: ds^2 preserved under 20+ transforms (rotations, boosts, large-eta, lightlike)."""

    rotations_4d, boosts_4d = so31_4d_generators()
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])

    # Multiple test vectors
    test_vectors = [
        np.array([2.0, 1.0, 0.0, 0.0]),    # timelike
        np.array([1.0, 3.0, 2.0, 0.0]),     # spacelike
        np.array([1.0, 1.0, 0.0, 0.0]),     # lightlike
        np.array([5.0, 3.0, 4.0, 0.0]),     # lightlike-ish
        np.array([10.0, 1.0, 1.0, 1.0]),    # deeply timelike
    ]

    # 25 Lorentz transformations
    transforms = []
    # 3 pure rotations at various angles
    for i, angle in enumerate([0.3, 1.0, 2.5]):
        for axis in range(3):
            transforms.append((f'rot_{axis}_a{angle}', expm(angle * rotations_4d[axis])))

    # 3 pure boosts at various rapidities
    for eta_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for axis in range(3):
            transforms.append((f'boost_{axis}_eta{eta_val}', expm(eta_val * boosts_4d[axis])))

    # Large rapidity (ultra-relativistic)
    transforms.append(('boost_x_eta10', expm(10.0 * boosts_4d[0])))

    # Cumulative: rotation then boost then rotation
    L_cumulative = expm(0.5 * rotations_4d[2]) @ expm(1.5 * boosts_4d[0]) @ expm(0.3 * rotations_4d[1])
    transforms.append(('cumulative_RBR', L_cumulative))

    # Double boost (non-collinear)
    L_double = expm(0.8 * boosts_4d[0]) @ expm(0.6 * boosts_4d[1])
    transforms.append(('double_boost_xy', L_double))

    max_error = 0.0
    n_tests = 0
    n_pass = 0
    n_tight_pass = 0

    for x in test_vectors:
        s2_orig = float(x @ eta @ x)
        for name, Lambda in transforms:
            x_prime = Lambda @ x
            s2_prime = float(x_prime @ eta @ x_prime)

            if abs(s2_orig) > 1e-15:
                rel_err = abs(s2_prime - s2_orig) / abs(s2_orig)
            else:
                rel_err = abs(s2_prime - s2_orig)

            max_error = max(max_error, rel_err)
            n_tests += 1
            if rel_err < 1e-5:
                n_pass += 1
            if rel_err < 1e-10:
                n_tight_pass += 1

    print(f"  Tested {n_tests} (vector, transform) pairs")
    print(f"  Passed (tol 1e-5): {n_pass}/{n_tests}")
    print(f"  Passed (tol 1e-10): {n_tight_pass}/{n_tests}")
    print(f"  Max relative error: {max_error:.2e}")

    result = {
        'test': 'T3_ds2_preserved_extensive',
        'n_vectors': len(test_vectors),
        'n_transforms': len(transforms),
        'n_total_tests': n_tests,
        'n_passed_1e5': n_pass,
        'n_passed_1e10': n_tight_pass,
        'max_relative_error': float(max_error),
        'note': 'Large-rapidity transforms (eta=10) have ~1e-6 numerical error from '
                'matrix exponentiation. This is numerical precision, not physics failure.',
        'PASS': n_pass == n_tests and max_error < 1e-5,
    }
    return result


def test_T4_metric_uniqueness():
    """T4: Invariant form proportional to Minkowski + alternative metrics NOT preserved."""

    generators_4d = so31_4d_generators()
    rotations_4d, boosts_4d = generators_4d
    inv_form = lorentz_invariant_form(generators_4d)

    null_dim = inv_form['null_space_dimension']
    is_unique = inv_form['is_unique']

    # Verify the invariant form IS proportional to Minkowski
    form = inv_form['form']
    eta_minkowski = np.diag([-1.0, 1.0, 1.0, 1.0])

    # Check proportionality: form = alpha * eta for some scalar alpha
    if np.max(np.abs(eta_minkowski)) > 0:
        # Find the proportionality constant
        nonzero_idx = np.argmax(np.abs(eta_minkowski.flatten()))
        alpha = form.flatten()[nonzero_idx] / eta_minkowski.flatten()[nonzero_idx]
        residual = np.max(np.abs(form - alpha * eta_minkowski))
    else:
        alpha = 0.0
        residual = float('inf')

    proportional_to_minkowski = residual < 1e-10
    print(f"  Null space dimension: {null_dim}")
    print(f"  Proportionality constant alpha: {alpha:.4f}")
    print(f"  Residual |form - alpha*eta|: {residual:.2e}")
    print(f"  Proportional to Minkowski: {proportional_to_minkowski}")

    # Alternative metrics that should NOT be preserved
    alternatives = {
        'Euclidean': np.diag([1.0, 1.0, 1.0, 1.0]),
        'signature_22': np.diag([-1.0, -1.0, 1.0, 1.0]),
        'random_sym': np.array([[1, 0.5, 0, 0], [0.5, 2, 0.3, 0],
                                [0, 0.3, 1, 0.1], [0, 0, 0.1, 3]]),
    }

    alt_results = {}
    n_alternatives_fail = 0
    for name, alt_metric in alternatives.items():
        # Check if Lambda^T alt Lambda = alt for a Lorentz boost
        Lambda = expm(0.7 * boosts_4d[0])
        transformed = Lambda.T @ alt_metric @ Lambda
        error = float(np.max(np.abs(transformed - alt_metric)))
        preserved = error < 1e-8
        alt_results[name] = {'error': error, 'preserved': preserved}
        if not preserved:
            n_alternatives_fail += 1
        print(f"  Alternative '{name}': error={error:.4f}, preserved={preserved}")

    all_alternatives_broken = n_alternatives_fail == len(alternatives)

    result = {
        'test': 'T4_metric_uniqueness',
        'null_space_dimension': null_dim,
        'is_unique': is_unique,
        'proportionality_constant': float(alpha),
        'proportionality_residual': float(residual),
        'proportional_to_minkowski': proportional_to_minkowski,
        'alternative_results': alt_results,
        'all_alternatives_broken': all_alternatives_broken,
        'PASS': null_dim == 1 and proportional_to_minkowski and all_alternatives_broken,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 09 -- ds^2 Is the Unique Complement-Transformation Invariant")
    print("Milestone 13, Block C (hardened v0.3)")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_killing_form_selectivity),
        ('T2', test_T2_4d_signature_and_causality),
        ('T3', test_T3_ds2_preserved_extensive),
        ('T4', test_T4_metric_uniqueness),
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
        'experiment': 'exp_09_invariant_interval',
        'milestone': 'milestone13',
        'block': 'C',
        'version': 'v0.3_hardened',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_09_invariant_interval', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()
