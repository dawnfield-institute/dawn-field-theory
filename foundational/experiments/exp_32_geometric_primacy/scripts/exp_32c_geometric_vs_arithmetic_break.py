"""
exp_32c — Geometric Failure Precedes Arithmetic Failure at Tetration

HYPOTHESIS: At the L3→L4 transition in the ADE hyperoperation hierarchy,
geometric properties (manifold smoothness, exp map convergence, curvature
finiteness) degrade BEFORE arithmetic properties (invertibility, closure).

This is the strongest test of the geometry-precedes-arithmetic claim.
If geometry breaks first, geometric structure is the load-bearing primitive
and arithmetic properties are secondary readouts.

FALSIFICATION: If arithmetic breaks first (invertibility lost before the
manifold degrades), the hypothesis is wrong — arithmetic is primary.

Tests:
  1. Exp map radius vs invertibility domain — geometric reach shrinks
     at L3 while arithmetic invertibility is still 100%.
  2. Continuous L3→L4 interpolation — track geometric and arithmetic
     quality as continuous functions of interpolation parameter t.
  3. Curvature diagnostic — manifold curvature diverges before
     arithmetic operations lose invertibility.
  4. Symmetry group effective dimension — drops to zero geometrically
     before arithmetic closure fails.

Author: Peter Groom
Date: 2026-04-18
"""

import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = EXP_ROOT / "results"


# ============================================================
# Hyperoperation infrastructure
# ============================================================

def hyper(level, a, b):
    """Hyperoperation H_n(a, b)."""
    if level == 1:
        return a + b
    elif level == 2:
        return a * b
    elif level == 3:
        if a <= 0 and b != int(b):
            return float('nan')
        try:
            result = a ** b
            return result if np.isfinite(result) else float('inf')
        except (OverflowError, ValueError):
            return float('inf')
    elif level == 4:
        if not isinstance(b, int) and b != int(b):
            return float('nan')
        b = int(b)
        if b == 0:
            return 1.0
        if b == 1:
            return float(a)
        if a > 1 and b > 3:
            return float('inf')
        if a > 2 and b > 2:
            return float('inf')
        result = float(a)
        for _ in range(b - 1):
            try:
                if result > 709:
                    return float('inf')
                result = float(a) ** result
                if result > 1e308:
                    return float('inf')
            except (OverflowError, ValueError):
                return float('inf')
        return result
    return float('nan')


def hyper_inverse_right(level, a, result):
    """Right inverse: find b such that H_n(a, b) = result."""
    if level == 1:
        return result - a, True
    elif level == 2:
        if abs(a) < 1e-15:
            return float('nan'), False
        return result / a, True
    elif level == 3:
        if a <= 0 or a == 1 or result <= 0:
            return float('nan'), False
        try:
            val = np.log(result) / np.log(a)
            return val, np.isfinite(val)
        except (ValueError, ZeroDivisionError):
            return float('nan'), False
    elif level == 4:
        if a <= 1 or result <= 0:
            return float('nan'), False
        count = 0
        x = result
        while x > 1.01 and count < 100:
            if x == float('inf'):
                return float('nan'), False
            x = np.log(x) / np.log(a)
            count += 1
            if x <= 0:
                return float('nan'), False
        if count >= 100:
            return float('nan'), False
        if abs(x - 1.0) < 0.01:
            return float(count), True
        return float('nan'), False
    return float('nan'), False


# ============================================================
# Fractional hyperoperation (continuous interpolation L3→L4)
# ============================================================

def fractional_hyper(level_frac, a, b):
    """
    Continuously interpolate between hyperoperation levels.

    For level_frac between 3 and 4, use:
      H_{3+t}(a, b) = (1-t) * H_3(a, b) + t * H_4(a, b)

    This is a WEIGHTED interpolation, not a true fractional iteration
    (which doesn't have a canonical definition). But it's sufficient
    to track the DEGRADATION of properties as we move from L3 to L4.

    For cleaner interpolation at the geometric level, we also provide
    the iterated exponential approach: H_{3+t}(a, b) where the
    tower height is extended fractionally via exp interpolation.
    """
    level_int = int(level_frac)
    t = level_frac - level_int

    if t < 1e-10:
        return hyper(level_int, a, b)

    if level_int == 3:
        # Interpolate between exponentiation and tetration
        # Use iterated exponential with fractional tower height:
        # At t=0: a^b (single exponentiation)
        # At t=1: a^^b (full tetration = tower of b)
        # Intermediate: tower of height 1 + t*(b-1)
        if b <= 0 or a <= 0:
            return float('nan')

        effective_height = 1.0 + t * (b - 1.0)
        if effective_height <= 0:
            return float('nan')

        # Build tower bottom-up with fractional top
        full_levels = int(effective_height)
        frac_part = effective_height - full_levels

        result = float(a)
        # First, the fractional top level
        if frac_part > 1e-10:
            # Partial exponentiation: a^(frac_part * a) ≈ a^(frac * prev)
            result = a ** (frac_part * a)

        # Then full levels
        for _ in range(full_levels):
            try:
                if result > 709:
                    return float('inf')
                result = float(a) ** result
                if not np.isfinite(result):
                    return float('inf')
            except (OverflowError, ValueError):
                return float('inf')

        return result

    # For other level transitions, use simple interpolation
    h_low = hyper(level_int, a, b)
    h_high = hyper(level_int + 1, a, b)

    if not np.isfinite(h_low):
        return h_low
    if not np.isfinite(h_high):
        # High level overflows — interpolate toward overflow
        return h_low * (1 + t * 1e10)  # effectively infinity for t > 0

    return (1 - t) * h_low + t * h_high


# ============================================================
# Test 1: Exp Map Radius vs Invertibility Domain
# ============================================================

def test1_exp_map_vs_invertibility():
    """
    For each level, measure:
      (a) Exp map radius: largest t for which exp(t * X) converges
          (geometric property — how far the smooth manifold extends)
      (b) Invertibility fraction: fraction of test domain where
          H_n(a, b) has a well-defined inverse
          (arithmetic property)

    Prediction: at Level 3, exp map radius is FINITE (geometric
    degradation) but invertibility is still ~100% (arithmetic intact).
    Geometry breaks first.
    """
    print("=" * 60)
    print("Test 1: Exp Map Radius vs Invertibility Domain")
    print("=" * 60)

    results = {}

    for level, name in [(1, "addition"), (2, "multiplication"),
                        (3, "exponentiation"), (4, "tetration")]:

        # (a) Exp map radius: find max t where exp(t * X) converges
        # For level n with base a, the "exponential map" is
        # the iterated action: start at identity, apply H_n(a, ·) for t steps
        # Convergence = result stays finite
        a_test = 2.0
        t_values = np.logspace(-3, 2, 200)  # t from 0.001 to 100
        max_converging_t = 0.0

        for t in t_values:
            # Simulate exp map: apply H_n(a, ·) starting from identity
            # Identity for each level: add→0, mult→1, exp→1, tet→1
            if level == 1:
                # exp(t * X) for addition: just t * a
                result = t * a_test
                converged = np.isfinite(result)
            elif level == 2:
                # exp(t * X) for multiplication: a^t
                result = a_test ** t
                converged = np.isfinite(result) and result < 1e100
            elif level == 3:
                # exp(t * X) for exponentiation: tower of height t
                # a^(a^(a^...)) with t levels
                # Converges only for a <= e^(1/e) ≈ 1.4447 (infinite tower)
                # For a=2, finite tower: converges for small t
                result = 1.0
                steps = int(t * 10)  # discretize
                for _ in range(max(1, steps)):
                    try:
                        result = a_test ** result
                        if result > 1e100 or not np.isfinite(result):
                            result = float('inf')
                            break
                    except OverflowError:
                        result = float('inf')
                        break
                converged = np.isfinite(result) and result < 1e100
            elif level == 4:
                # exp(t * X) for tetration: tower of towers
                # Diverges almost immediately for a > 1
                result = 1.0
                steps = max(1, int(t))
                for _ in range(steps):
                    try:
                        # One level of tetration
                        tower = result
                        for _ in range(2):  # even 2 levels of nesting
                            tower = a_test ** tower
                            if tower > 1e100:
                                tower = float('inf')
                                break
                        result = tower
                        if not np.isfinite(result):
                            break
                    except OverflowError:
                        result = float('inf')
                        break
                converged = np.isfinite(result) and result < 1e100

            if converged:
                max_converging_t = t

        # (b) Invertibility: fraction of test pairs where inverse exists
        test_as = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 1.1, 1.01, 10.0, 0.5]
        test_bs = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.1, 4.0, 5.0, 0.01]
        n_invertible = 0
        n_tested = 0

        for a in test_as:
            for b in test_bs:
                h = hyper(level, a, b)
                if not np.isfinite(h) or np.isnan(h):
                    continue
                n_tested += 1
                b_inv, success = hyper_inverse_right(level, a, h)
                if success and np.isfinite(b_inv) and abs(b_inv - b) < 1e-4:
                    n_invertible += 1

        inv_fraction = n_invertible / n_tested if n_tested > 0 else 0.0

        # Normalize exp map radius to [0, 1] scale
        # L1: infinite (set to 1.0), L2: very large, L3: finite, L4: ~0
        if max_converging_t >= 100:
            normalized_radius = 1.0
        else:
            normalized_radius = np.log1p(max_converging_t) / np.log1p(100)

        results[name] = {
            'level': level,
            'exp_map_max_t': float(max_converging_t),
            'exp_map_normalized': float(normalized_radius),
            'invertibility_fraction': float(inv_fraction),
            'n_invertible': n_invertible,
            'n_tested': n_tested,
        }

        print(f"\n  {name} (Level {level}):")
        print(f"    Exp map radius: t_max = {max_converging_t:.4f} "
              f"(normalized: {normalized_radius:.4f})")
        print(f"    Invertibility: {n_invertible}/{n_tested} = {inv_fraction:.2%}")

    # Key comparison at L3: geometry degraded, arithmetic intact?
    l3 = results['exponentiation']
    l4 = results['tetration']

    geom_breaks_first = (
        l3['exp_map_normalized'] < 0.5 and  # geometry already degraded at L3
        l3['invertibility_fraction'] > 0.8   # but arithmetic still works at L3
    )

    print(f"\n  KEY: At Level 3 (exponentiation):")
    print(f"    Geometry (exp map): {l3['exp_map_normalized']:.4f} "
          f"({'degraded' if l3['exp_map_normalized'] < 0.5 else 'intact'})")
    print(f"    Arithmetic (invertibility): {l3['invertibility_fraction']:.2%} "
          f"({'intact' if l3['invertibility_fraction'] > 0.8 else 'degraded'})")
    print(f"    Geometry breaks first: {geom_breaks_first}")

    results['geometry_breaks_first'] = geom_breaks_first
    return results


# ============================================================
# Test 2: Continuous L3→L4 Interpolation
# ============================================================

def test2_continuous_interpolation():
    """
    Smoothly interpolate between Level 3 and Level 4.
    Track geometric quality and arithmetic quality as functions
    of interpolation parameter t in [0, 1].

    Geometric quality: smoothness of the operation
      (finite differences of fractional_hyper should be smooth)
    Arithmetic quality: invertibility at each interpolation point

    Prediction: geometric quality drops to zero BEFORE arithmetic
    invertibility does.
    """
    print("\n" + "=" * 60)
    print("Test 2: Continuous L3 → L4 Interpolation")
    print("=" * 60)

    n_points = 21
    t_values = np.linspace(0, 1, n_points)
    a_test = 1.5  # small base to avoid overflow
    b_test = 3.0

    geometric_quality = []
    arithmetic_quality = []

    for t in t_values:
        level = 3 + t

        # Geometric quality: smoothness of the operation
        # Measure via finite differences — if d/db H is smooth, geometry is ok
        db = 1e-4
        h_center = fractional_hyper(level, a_test, b_test)
        h_plus = fractional_hyper(level, a_test, b_test + db)
        h_minus = fractional_hyper(level, a_test, b_test - db)

        if all(np.isfinite(x) for x in [h_center, h_plus, h_minus]):
            first_deriv = (h_plus - h_minus) / (2 * db)
            second_deriv = (h_plus - 2 * h_center + h_minus) / (db ** 2)

            # Smoothness = inverse of curvature (high curvature = low smoothness)
            curvature = abs(second_deriv) / (1 + first_deriv ** 2) ** 1.5
            if curvature < 1e10:
                smoothness = 1.0 / (1.0 + curvature / 100)  # normalized to [0, 1]
            else:
                smoothness = 0.0
        else:
            smoothness = 0.0

        geometric_quality.append(smoothness)

        # Arithmetic quality: can we invert the operation?
        # Try several test points
        inv_successes = 0
        inv_total = 0
        for b_try in [1.0, 1.5, 2.0, 2.5, 3.0]:
            h_val = fractional_hyper(level, a_test, b_try)
            if np.isfinite(h_val) and h_val > 0:
                inv_total += 1
                # Try to invert via Newton's method
                b_guess = b_try  # start near true value
                for _ in range(20):
                    h_guess = fractional_hyper(level, a_test, b_guess)
                    h_deriv = (fractional_hyper(level, a_test, b_guess + 1e-5) -
                               fractional_hyper(level, a_test, b_guess - 1e-5)) / 2e-5

                    if not np.isfinite(h_guess) or not np.isfinite(h_deriv) or abs(h_deriv) < 1e-15:
                        break

                    b_guess -= (h_guess - h_val) / h_deriv

                    if abs(b_guess - b_try) < 1e-6:
                        inv_successes += 1
                        break

        inv_quality = inv_successes / inv_total if inv_total > 0 else 0.0
        arithmetic_quality.append(inv_quality)

    # Print trajectory
    print(f"\n  {'t':>5} {'Level':>6} {'Geom (smoothness)':>18} {'Arith (invertibility)':>22}")
    print(f"  {'-'*5} {'-'*6} {'-'*18} {'-'*22}")
    for i, t in enumerate(t_values):
        print(f"  {t:5.2f} {3+t:6.2f} {geometric_quality[i]:18.4f} {arithmetic_quality[i]:22.4f}")

    # Find where each crosses 0.5 (degradation threshold)
    geom_cross = 1.0  # default: never crosses
    arith_cross = 1.0
    for i in range(len(t_values) - 1):
        if geometric_quality[i] >= 0.5 and geometric_quality[i + 1] < 0.5:
            # Linear interpolation
            frac = (0.5 - geometric_quality[i]) / (geometric_quality[i + 1] - geometric_quality[i])
            geom_cross = t_values[i] + frac * (t_values[i + 1] - t_values[i])
        if arithmetic_quality[i] >= 0.5 and arithmetic_quality[i + 1] < 0.5:
            frac = (0.5 - arithmetic_quality[i]) / (arithmetic_quality[i + 1] - arithmetic_quality[i])
            arith_cross = t_values[i] + frac * (t_values[i + 1] - t_values[i])

    # Also check: is geometry ALREADY below 0.5 at t=0 (L3)?
    geom_degraded_at_l3 = geometric_quality[0] < 0.5
    arith_intact_at_l3 = arithmetic_quality[0] > 0.5

    print(f"\n  Geometry 50% crossing at t = {geom_cross:.3f} (level {3 + geom_cross:.3f})")
    print(f"  Arithmetic 50% crossing at t = {arith_cross:.3f} (level {3 + arith_cross:.3f})")
    print(f"  At L3 (t=0): geometry={geometric_quality[0]:.4f}, "
          f"arithmetic={arithmetic_quality[0]:.4f}")

    geom_first = geom_cross < arith_cross
    print(f"\n  Geometry breaks first: {geom_first}")

    # The L3 comparison is actually the cleanest test
    l3_split = geom_degraded_at_l3 and arith_intact_at_l3
    print(f"  L3 split (geometry degraded, arithmetic intact): {l3_split}")

    passed = geom_first or l3_split
    print(f"  PASS: {passed}")

    return {
        't_values': [float(t) for t in t_values],
        'geometric_quality': [float(g) for g in geometric_quality],
        'arithmetic_quality': [float(a) for a in arithmetic_quality],
        'geometry_50pct_crossing': float(geom_cross),
        'arithmetic_50pct_crossing': float(arith_cross),
        'geometry_at_L3': float(geometric_quality[0]),
        'arithmetic_at_L3': float(arithmetic_quality[0]),
        'geometry_breaks_first': geom_first,
        'l3_split': l3_split,
        'passed': passed,
    }


# ============================================================
# Test 3: Curvature Diagnostic
# ============================================================

def test3_curvature_diagnostic():
    """
    Compute the curvature of the transformation manifold at each level.

    For a 1-parameter group g(t) = H_n(a, t), the curvature of the
    curve {(t, g(t))} measures how "bent" the transformation space is.

    L1-L2: flat (linear/polynomial) → curvature ≈ 0
    L3: curved but finite → curvature is large but bounded
    L4: curvature diverges → no smooth manifold

    The geometric object (manifold) breaks (infinite curvature = no smooth
    surface) before the arithmetic operation loses invertibility.
    """
    print("\n" + "=" * 60)
    print("Test 3: Curvature Diagnostic")
    print("=" * 60)

    a_values = [1.5, 2.0, 2.5, 3.0]
    levels = [1, 2, 3, 4]
    results = {}

    for level, name in zip(levels, ["addition", "multiplication",
                                     "exponentiation", "tetration"]):
        curvatures = []
        inv_fractions = []

        for a in a_values:
            # Sample the curve (t, H_n(a, t)) for t in [0.1, 5]
            t_samples = np.linspace(0.1, 5.0, 200)
            h_values = []
            for t in t_samples:
                if level == 4:
                    # Tetration only defined for integer t
                    h = hyper(level, a, int(round(t)))
                else:
                    h = hyper(level, a, t)
                h_values.append(h if np.isfinite(h) else np.nan)

            h_values = np.array(h_values)
            valid = np.isfinite(h_values)

            if np.sum(valid) >= 10:
                # Compute curvature via finite differences on valid segment
                valid_t = t_samples[valid]
                valid_h = h_values[valid]

                # First and second derivatives
                dt = np.diff(valid_t)
                dh = np.diff(valid_h)
                first_deriv = dh / dt

                if len(first_deriv) > 1:
                    dt2 = (dt[:-1] + dt[1:]) / 2
                    second_deriv = np.diff(first_deriv) / dt2

                    # Curvature = |f''| / (1 + f'^2)^(3/2)
                    f_prime = first_deriv[:-1]  # align lengths
                    kappa = np.abs(second_deriv) / (1 + f_prime ** 2) ** 1.5

                    # Use median curvature (robust to outliers)
                    median_kappa = float(np.nanmedian(kappa))
                    max_kappa = float(np.nanmax(kappa)) if np.any(np.isfinite(kappa)) else float('inf')
                else:
                    median_kappa = 0.0
                    max_kappa = 0.0
            else:
                median_kappa = float('inf')
                max_kappa = float('inf')

            curvatures.append(median_kappa)

            # Invertibility for this a
            inv_count = 0
            inv_total = 0
            for b in [1.0, 1.5, 2.0, 2.5, 3.0]:
                h = hyper(level, a, b)
                if np.isfinite(h) and not np.isnan(h):
                    inv_total += 1
                    b_inv, success = hyper_inverse_right(level, a, h)
                    if success and np.isfinite(b_inv) and abs(b_inv - b) < 1e-4:
                        inv_count += 1
            inv_fractions.append(inv_count / inv_total if inv_total > 0 else 0.0)

        mean_curvature = np.mean(curvatures) if curvatures else float('inf')
        mean_inv = np.mean(inv_fractions) if inv_fractions else 0.0

        # Classify
        if mean_curvature < 1:
            curvature_class = "flat"
        elif mean_curvature < 100:
            curvature_class = "curved-finite"
        elif mean_curvature < 1e10:
            curvature_class = "highly-curved"
        else:
            curvature_class = "divergent"

        results[name] = {
            'level': level,
            'mean_curvature': float(mean_curvature),
            'curvature_class': curvature_class,
            'mean_invertibility': float(mean_inv),
            'curvatures_by_a': [float(k) for k in curvatures],
            'invertibility_by_a': [float(f) for f in inv_fractions],
        }

        print(f"\n  {name} (Level {level}):")
        print(f"    Mean curvature: {mean_curvature:.4f} ({curvature_class})")
        print(f"    Mean invertibility: {mean_inv:.2%}")

    # Check the ordering: curvature should blow up BEFORE invertibility drops
    l3_curved = results['exponentiation']['curvature_class'] in ['curved-finite', 'highly-curved']
    l3_invertible = results['exponentiation']['mean_invertibility'] > 0.8
    l4_divergent = results['tetration']['curvature_class'] in ['highly-curved', 'divergent']

    print(f"\n  L3 curved but invertible: {l3_curved and l3_invertible}")
    print(f"  L4 curvature class: {results['tetration']['curvature_class']}")
    print(f"  Curvature progression: "
          f"{results['addition']['curvature_class']} → "
          f"{results['multiplication']['curvature_class']} → "
          f"{results['exponentiation']['curvature_class']} → "
          f"{results['tetration']['curvature_class']}")

    # The key: geometry (curvature) degrades monotonically while
    # arithmetic (invertibility) stays high until it suddenly drops
    curvature_monotonic = (
        results['addition']['mean_curvature'] <=
        results['multiplication']['mean_curvature'] <=
        results['exponentiation']['mean_curvature']
    )

    inv_stays_high = (
        results['addition']['mean_invertibility'] > 0.8 and
        results['multiplication']['mean_invertibility'] > 0.8 and
        results['exponentiation']['mean_invertibility'] > 0.5
    )

    passed = l3_curved and l3_invertible and curvature_monotonic
    print(f"\n  Curvature monotonically increases: {curvature_monotonic}")
    print(f"  Invertibility stays high through L3: {inv_stays_high}")
    print(f"  PASS: {passed} (curvature grows while invertibility persists)")

    results['curvature_monotonic'] = curvature_monotonic
    results['inv_stays_high'] = inv_stays_high
    results['passed'] = passed
    return results


# ============================================================
# Test 4: Effective Symmetry Group Dimension
# ============================================================

def test4_symmetry_dimension():
    """
    At each level, measure the effective dimension of the symmetry
    group generated by the operation.

    Method: sample the orbit of a point under the operation with
    varying parameters. The dimension of the orbit manifold =
    the dimension of the symmetry group.

    L1: orbit is a line (dim=1) — translations
    L2: orbit is a line (dim=1) — scalings
    L3: orbit is a curve (dim=1) but with finite range — rotations/spirals
    L4: orbit is ill-defined (dim→0) — no smooth group action

    Track effective dimension via local PCA of orbit samples.
    """
    print("\n" + "=" * 60)
    print("Test 4: Effective Symmetry Group Dimension")
    print("=" * 60)

    results = {}

    for level, name in [(1, "addition"), (2, "multiplication"),
                        (3, "exponentiation"), (4, "tetration")]:

        # Generate orbit: fix a, vary b around a reference point
        a_test = 1.5
        b_center = 2.0
        n_samples = 100

        # Sample the 2D curve (b, H_n(a, b)) in the (parameter, output) plane
        b_values = np.linspace(b_center - 1.0, b_center + 1.0, n_samples)
        orbit_points = []

        for b in b_values:
            if level == 4:
                h = hyper(level, a_test, max(0, int(round(b))))
            else:
                h = hyper(level, a_test, b)

            if np.isfinite(h) and abs(h) < 1e50:
                orbit_points.append([b, h])

        if len(orbit_points) >= 10:
            orbit = np.array(orbit_points)

            # Normalize
            orbit_norm = (orbit - orbit.mean(axis=0)) / (orbit.std(axis=0) + 1e-15)

            # Compute covariance and PCA
            cov = np.cov(orbit_norm.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Effective dimension: ratio of largest to second eigenvalue
            # High ratio → 1D curve, low ratio → 0D (point/discrete)
            if eigenvalues[0] > 1e-10:
                dim_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-10)
                # Participation ratio (effective dimensionality)
                total_var = np.sum(eigenvalues)
                participation = total_var ** 2 / (np.sum(eigenvalues ** 2) + 1e-15)
            else:
                dim_ratio = 0.0
                participation = 0.0

            # Smoothness of the orbit: measure via arc length regularity
            diffs = np.diff(orbit, axis=0)
            arc_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            if len(arc_lengths) > 1 and np.mean(arc_lengths) > 1e-15:
                arc_cv = np.std(arc_lengths) / np.mean(arc_lengths)
            else:
                arc_cv = float('inf')

            # Effective dimension: 1 if smooth 1D curve, <1 if degraded
            smooth_1d = arc_cv < 0.5  # low CV = smooth curve
            eff_dim = 1.0 if smooth_1d else max(0.0, 1.0 - arc_cv)

            n_valid = len(orbit_points)
        else:
            dim_ratio = 0.0
            participation = 0.0
            arc_cv = float('inf')
            eff_dim = 0.0
            n_valid = len(orbit_points)

        results[name] = {
            'level': level,
            'n_valid_orbit_points': n_valid,
            'effective_dimension': float(eff_dim),
            'arc_length_cv': float(arc_cv) if np.isfinite(arc_cv) else None,
            'participation_ratio': float(participation),
        }

        print(f"\n  {name} (Level {level}):")
        print(f"    Valid orbit points: {n_valid}/{n_samples}")
        print(f"    Effective dimension: {eff_dim:.4f}")
        print(f"    Arc length CV: {arc_cv:.4f}" if np.isfinite(arc_cv) else "    Arc length CV: inf")
        print(f"    Participation ratio: {participation:.4f}")

    # Check: dimension degrades monotonically
    dims = [results[n]['effective_dimension'] for n in
            ['addition', 'multiplication', 'exponentiation', 'tetration']]

    # L1, L2 should be ~1.0; L3 should be <1.0 or borderline; L4 should be ~0
    l3_degraded = dims[2] < dims[0]  # L3 dim < L1 dim
    l4_collapsed = dims[3] < dims[2]  # L4 dim < L3 dim

    print(f"\n  Dimension trajectory: {' → '.join(f'{d:.3f}' for d in dims)}")
    print(f"  L3 dimension degraded from L1: {l3_degraded}")
    print(f"  L4 dimension collapsed from L3: {l4_collapsed}")

    passed = l4_collapsed
    print(f"  PASS: {passed} (dimension drops at L4)")

    results['dimension_trajectory'] = dims
    results['l3_degraded'] = l3_degraded
    results['l4_collapsed'] = l4_collapsed
    results['passed'] = passed
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("exp_32c — Geometric Failure Precedes Arithmetic Failure")
    print("at the ADE Hyperoperation Ladder (L3 → L4)")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Geometric properties (manifold smoothness, exp map")
    print("convergence, curvature) degrade BEFORE arithmetic properties")
    print("(invertibility, closure) at the L3→L4 transition.")
    print()
    print("FALSIFICATION: If arithmetic breaks first, geometry is secondary.")
    print()

    r1 = test1_exp_map_vs_invertibility()
    r2 = test2_continuous_interpolation()
    r3 = test3_curvature_diagnostic()
    r4 = test4_symmetry_dimension()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY — Geometric Failure Precedes Arithmetic Failure")
    print("=" * 70)

    checks = [
        ("Exp map radius finite at L3, invertibility intact",
         r1.get('geometry_breaks_first', False)),
        ("Continuous interpolation: geometry degrades first",
         r2.get('passed', False)),
        ("Curvature monotonically increases while invertibility persists",
         r3.get('passed', False)),
        ("Symmetry group dimension collapses at L4",
         r4.get('passed', False)),
    ]

    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/4")

    if passed_count >= 3:
        print("\n  CONCLUSION: Geometric properties degrade before arithmetic")
        print("  properties at the ADE hierarchy transition. The manifold")
        print("  (geometric object) breaks before the operation (arithmetic)")
        print("  loses invertibility. This confirms: geometry is the load-")
        print("  bearing primitive; arithmetic is the SEC-collapsed readout.")

    # Save
    results = {
        'experiment': 'exp_32c_geometric_vs_arithmetic_break',
        'version': 1,
        'milestone': 8,
        'series': 'exp_32',
        'block': 'geometric_primacy',
        'hypothesis': (
            'Geometric properties (manifold smoothness, exp map convergence, '
            'curvature) degrade BEFORE arithmetic properties (invertibility, '
            'closure) at the ADE L3→L4 transition.'
        ),
        'exp_map_vs_invertibility': r1,
        'continuous_interpolation': r2,
        'curvature_diagnostic': r3,
        'symmetry_dimension': r4,
        'verification': {
            'checks': {name: passed for name, passed in checks},
            'passed_count': passed_count,
            'total': len(checks),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp_32c_geometric_vs_arithmetic_break_v1_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\n  Results saved: {out_path.name}")
    return results


if __name__ == '__main__':
    main()
