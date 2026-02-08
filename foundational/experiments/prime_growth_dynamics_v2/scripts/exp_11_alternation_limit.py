"""
Experiment 11: Alternation Limit Analysis
============================================

Tests whether the alternation limit ≈ 0.68 from oscillation_attractor_
dynamics is exactly 2/3 = F₃/F₄, or relates to 1/φ via MED constraints.

From oscillation_attractor_dynamics: alternation limit approaches ≈ 0.68
but was not derived. Two candidates:
  - 2/3 = F₃/F₄ = 0.6667 (the simplest MED-constrained ratio)
  - Some function of 1/φ = 0.6180 with γ-correction

The gap: 0.68 - 2/3 ≈ 0.013, vs 0.68 - 1/φ ≈ 0.062

Success criterion: Identify the alternation limit to < 0.5% accuracy
from phase constants.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def compute_alternation_limit(n_terms=10000):
    """
    Compute the alternation limit from oscillation attractor dynamics.

    Uses the sum: Σ (-1)^(n+1) / n for prime n, which converges
    to a specific value related to the prime distribution.
    """
    primes = sieve(n_terms * 20)  # Ensure enough primes
    primes = primes[:n_terms]

    # Alternating sum over reciprocals of primes
    total = 0.0
    for i, p in enumerate(primes):
        if i % 2 == 0:
            total += 1.0 / p
        else:
            total -= 1.0 / p

    return total, primes


def compute_mertens_alternating(limit=100000):
    """
    Compute alternating version of Mertens' theorem:
    Σ (-1)^k · 1/p_k for k = 1, 2, 3, ...
    """
    primes = sieve(limit)
    partial_sums = []
    total = 0.0

    for k, p in enumerate(primes):
        sign = 1 if k % 2 == 0 else -1
        total += sign / p
        partial_sums.append(total)

    return partial_sums, primes


def run():
    print("=" * 70)
    print("EXP 11: Alternation Limit Analysis")
    print("=" * 70)

    # Target value from oscillation_attractor_dynamics
    target = 0.68  # approximate
    inv_phi = 1 / PHI
    two_thirds = 2.0 / 3.0

    print(f"  Target alternation limit: ≈ {target}")
    print(f"  1/φ = {inv_phi:.10f}")
    print(f"  2/3 = {two_thirds:.10f}")

    # ================================================================
    # Test 1: Compute actual alternation limit
    # ================================================================
    print("\n--- Test 1: Compute alternation limit ---")

    partial_sums, primes = compute_mertens_alternating(limit=500000)

    print(f"  Primes computed: {len(primes)}")
    print(f"  Convergence trajectory:")
    checkpoints = [10, 50, 100, 500, 1000, 5000, 10000, len(partial_sums)-1]
    for cp in checkpoints:
        if cp < len(partial_sums):
            print(f"    After {cp+1:6d} primes: {partial_sums[cp]:.10f}")

    actual_limit = partial_sums[-1]
    print(f"\n  Best estimate: {actual_limit:.10f}")

    # ================================================================
    # Test 2: How close to known constants?
    # ================================================================
    print("\n--- Test 2: Distance from known constants ---")

    candidates_direct = [
        ("2/3 = F₃/F₄", two_thirds),
        ("1/φ", inv_phi),
        ("ln(2)", math.log(2)),
        ("γ + ln(2)/π", GAMMA + math.log(2)/math.pi),
        ("1 - 1/e", 1 - 1/math.e),
        ("1/φ + γ²/4", inv_phi + GAMMA**2/4),
        ("2/3 + γ/55", two_thirds + GAMMA/55),
        ("ln(φ) + 1/φ", LN_PHI + inv_phi),
    ]

    for name, val in sorted(candidates_direct, key=lambda x: abs(x[1] - actual_limit)):
        err = abs(val - actual_limit) / actual_limit * 100
        if err < 10:
            print(f"  {name:<25} = {val:.10f}  (error = {err:.4f}%)")

    # ================================================================
    # Test 3: Systematic formula search
    # ================================================================
    print("\n--- Test 3: Systematic formula search ---")

    results = phase_formula_search(actual_limit, max_depth=3)
    print(f"  Found {len(results)} candidates within 2%:")
    for m in results[:20]:
        print(f"    {m['expression']} = {m['value']:.10f}  (error = {m['error_pct']:.4f}%)")

    # ================================================================
    # Test 4: Convergence rate analysis
    # ================================================================
    print("\n--- Test 4: Convergence rate ---")

    # How fast does it converge? Compare to 1/ln(p_n) rate
    n_check = [100, 500, 1000, 5000, 10000, 40000]
    for n in n_check:
        if n < len(partial_sums):
            deviation = abs(partial_sums[n] - actual_limit)
            p_n = primes[n]
            rate_ln = 1 / math.log(p_n)
            rate_sqrt_ln = 1 / (math.sqrt(n) * math.log(p_n))
            print(f"  n={n:5d}: |S_n - S*| = {deviation:.8f}, "
                  f"1/ln(p) = {rate_ln:.8f}, "
                  f"ratio = {deviation/rate_ln:.4f}" if rate_ln > 0 else "")

    # ================================================================
    # Test 5: Decomposition attempt
    # ================================================================
    print("\n--- Test 5: Decomposition into phase components ---")

    # Is it γ · f(φ) + ln(φ) · g(φ)?
    # actual = a·γ + b·ln(φ) + c
    # Try to solve the linear system

    # Several decomposition attempts
    decompositions = [
        ("γ·X + ln(φ)·Y", GAMMA, LN_PHI),
        ("γ·X + (1/φ)·Y", GAMMA, inv_phi),
        ("ln(2)·X + γ·Y", math.log(2), GAMMA),
    ]

    for name, a, b in decompositions:
        # actual = x·a + y·b, try integer/simple rational x, y
        best_match = None
        best_err = float('inf')
        for x_num in range(-5, 6):
            for x_den in range(1, 6):
                x = x_num / x_den
                remainder = actual_limit - x * a
                if abs(b) > 1e-10:
                    y = remainder / b
                    # Check if y is a simple fraction
                    for y_den in range(1, 10):
                        y_rounded = round(y * y_den) / y_den
                        recon = x * a + y_rounded * b
                        err = abs(recon - actual_limit) / actual_limit * 100
                        if err < best_err:
                            best_err = err
                            best_match = (x_num, x_den, round(y * y_den), y_den, recon)

        if best_match and best_err < 1:
            xn, xd, yn, yd = best_match[0], best_match[1], best_match[2], best_match[3]
            recon = best_match[4]
            print(f"  {name}: ({xn}/{xd})·{a:.4f} + ({yn}/{yd})·{b:.4f} "
                  f"= {recon:.10f} (err={best_err:.4f}%)")

    # ================================================================
    # Test 6: Is alternation limit a Mertens variant?
    # ================================================================
    print("\n--- Test 6: Mertens connection ---")

    # Standard Mertens: Σ 1/p = ln(ln(N)) + M₁ where M₁ ≈ 0.2615
    # Our alternating: Σ (-1)^k/p_k — what's the theoretical value?
    # This relates to the prime race between primes ≡ 1 vs 3 mod 4

    # Compute non-alternating sum for comparison
    mertens_sum = sum(1/p for p in primes)
    print(f"  Non-alternating Mertens sum (up to p={primes[-1]}): {mertens_sum:.6f}")
    print(f"  Alternating sum: {actual_limit:.6f}")
    print(f"  Ratio: {actual_limit/mertens_sum:.6f}")
    print(f"  Difference: {mertens_sum - actual_limit:.6f}")

    # ================================================================
    # Results
    # ================================================================
    # Which constant is closest?
    best = results[0] if results else None

    success = best is not None and best['error'] < 0.005  # < 0.5%

    data = {
        'experiment': 'exp_11_alternation_limit',
        'hypothesis': 'Alternation limit = 2/3 = F₃/F₄ or derived from 1/φ + γ-correction',
        'actual_limit': float(actual_limit),
        'n_primes': len(primes),
        'convergence_trajectory': {
            str(cp+1): float(partial_sums[cp])
            for cp in checkpoints if cp < len(partial_sums)
        },
        'distances': {
            '2/3': float(abs(actual_limit - two_thirds)),
            '1/phi': float(abs(actual_limit - inv_phi)),
            'ln(2)': float(abs(actual_limit - math.log(2))),
        },
        'formula_candidates_top5': [{'expr': m['expression'], 'val': m['value'], 'err': m['error']} for m in (results[:5] if results else [])],
        'best_formula': best['expression'] if best else None,
        'best_error_pct': float(best['error_pct']) if best else None,
        'success': success,
        'success_criterion': '< 0.5% error from phase constants',
    }

    print(f"\n{'='*70}")
    print(f"ALTERNATION LIMIT: {actual_limit:.10f}")
    if best:
        print(f"BEST FORMULA: {best['expression']} = {best['value']:.10f} (error={best['error_pct']:.4f}%)")
    print(f"SUCCESS: {'YES' if success else 'INCONCLUSIVE'}")
    print(f"{'='*70}")

    save_results(data, 'exp_11_alternation_limit')
    return data


if __name__ == '__main__':
    run()
