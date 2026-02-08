"""
Experiment 04: Forbidden k Prediction
======================================

Tests H2: Forbidden k values {5, 12, 13, 14, 15} in sec_prime_manifold
are Phase III resonance gaps — predictable from sieve wave interference.

Success criterion: Predict ≥ 4/5 forbidden k values from wave interference.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 04: Forbidden k Prediction from Wave Interference")
    print("=" * 70)

    # Known data from sec_prime_manifold
    # k values that CAN reach φ (error < 0.001)
    working_k = {3, 4, 6, 7, 8, 9, 10, 11}
    # k values that CANNOT reach φ
    forbidden_k = {5, 12, 13, 14, 15}
    # All tested
    all_k = sorted(working_k | forbidden_k)

    # ================================================================
    # Test 1: Wave interference strength at each k
    # ================================================================
    print("\n--- Test 1: Destructive interference by k ---")
    print(f"  Computing wave interference for k = {min(all_k)}..{max(all_k)} "
          f"(limit=100000)...")

    interference_data = {}
    for k in all_k:
        interf = wave_destructive_interference(k, limit=100000)
        status = "FORBIDDEN" if k in forbidden_k else "working"
        interference_data[k] = interf
        print(f"  k={k:2d}: interference = {interf:.4f}  [{status}]")

    # ================================================================
    # Test 2: Is there a threshold that separates forbidden from working?
    # ================================================================
    print("\n--- Test 2: Threshold analysis ---")

    working_vals = [interference_data[k] for k in working_k if k in interference_data]
    forbidden_vals = [interference_data[k] for k in forbidden_k if k in interference_data]

    if working_vals and forbidden_vals:
        print(f"  Working k interference:   mean={np.mean(working_vals):.4f}, "
              f"range=[{min(working_vals):.4f}, {max(working_vals):.4f}]")
        print(f"  Forbidden k interference: mean={np.mean(forbidden_vals):.4f}, "
              f"range=[{min(forbidden_vals):.4f}, {max(forbidden_vals):.4f}]")

        # Check separation
        overlap = (min(forbidden_vals) < max(working_vals)) if forbidden_vals and working_vals else True
        print(f"  Overlap: {'YES (not separated)' if overlap else 'NO (cleanly separated)'}")

    # ================================================================
    # Test 3: Prime-specific interference patterns
    # ================================================================
    print("\n--- Test 3: Detailed wave analysis for each k ---")

    primes_list = sieve(200)  # More than enough
    for k in all_k:
        base = primes_list[:k]
        # Check specific features of this base
        # Does the base include primes that multiply to create
        # strong destructive interference?

        # Product of (1-1/p) for this base
        prod = 1.0
        for p in base:
            prod *= (1 - 1/p)

        # Mertens-corrected estimate
        mertens_est = math.exp(-GAMMA) / math.log(base[-1]) if base[-1] > 1 else 0

        # Ratio of actual Mertens to naive
        ratio = prod / mertens_est if mertens_est > 0 else 0

        status = "FORBIDDEN" if k in forbidden_k else "working"
        print(f"  k={k:2d}: base_max={base[-1]:3d}, "
              f"prod(1-1/p)={prod:.6f}, "
              f"mertens_est={mertens_est:.6f}, "
              f"ratio={ratio:.4f}  [{status}]")

    # ================================================================
    # Test 4: Fibonacci structure in forbidden k
    # ================================================================
    print("\n--- Test 4: Fibonacci analysis of forbidden k ---")

    # k=5 = F5
    # k=12 = not Fibonacci
    # k=13 = F7
    # k=14 = not Fibonacci
    # k=15 = not Fibonacci

    fib_set = set(FIBS[:15])
    for k in sorted(forbidden_k):
        is_fib = k in fib_set
        # Zeckendorf representation
        zeck = []
        remaining = k
        for f in reversed(FIBS[:15]):
            if f <= remaining:
                zeck.append(f)
                remaining -= f
            if remaining == 0:
                break
        print(f"  k={k:2d}: Fibonacci={is_fib}, Zeckendorf={'+'.join(map(str,zeck))}")

    # Check: forbidden k might be where the k-th prime creates
    # a specific kind of interference
    print(f"\n  k-th primes at forbidden k:")
    for k in sorted(forbidden_k):
        if k <= len(primes_list):
            p = primes_list[k-1]
            print(f"  k={k:2d}: p(k)={p}, p/F7={p/F7:.3f}, p/F10={p/F10:.3f}")

    # ================================================================
    # Test 5: Does the Phase model predict forbidden k?
    # ================================================================
    print("\n--- Test 5: Phase prediction attempt ---")

    # Hypothesis: Forbidden k are where the base primes' LCM
    # creates a resonance with the Fibonacci structure
    predicted_forbidden = set()
    for k in all_k:
        base = primes_list[:k]
        # Compute LCM of base
        from math import gcd
        lcm = base[0]
        for p in base[1:]:
            lcm = lcm * p // gcd(lcm, p)

        # Check if LCM shares factors with F10 or F7
        lcm_mod_55 = lcm % F10
        lcm_mod_13 = lcm % F7

        # Also check if k-th prime is "too close" to a Fibonacci number
        pk = primes_list[k-1]
        nearest_fib = min(FIBS[:15], key=lambda f: abs(f - pk))
        fib_distance = abs(pk - nearest_fib)

        status = "FORBIDDEN" if k in forbidden_k else "working"
        print(f"  k={k:2d}: p(k)={pk:3d}, LCM%55={lcm_mod_55:2d}, "
              f"LCM%13={lcm_mod_13:2d}, nearest_fib={nearest_fib}, "
              f"fib_dist={fib_distance}  [{status}]")

    # ================================================================
    # Results
    # ================================================================
    # Count how many forbidden k we can correctly identify
    # (Using the best separation metric found)
    correct_predictions = 0
    total_forbidden = len(forbidden_k)

    data = {
        'experiment': 'exp_04_forbidden_k_prediction',
        'forbidden_k': sorted(forbidden_k),
        'working_k': sorted(working_k),
        'interference_by_k': {str(k): v for k, v in interference_data.items()},
        'working_interference': {
            'mean': float(np.mean(working_vals)) if working_vals else None,
            'std': float(np.std(working_vals)) if working_vals else None,
        },
        'forbidden_interference': {
            'mean': float(np.mean(forbidden_vals)) if forbidden_vals else None,
            'std': float(np.std(forbidden_vals)) if forbidden_vals else None,
        },
        'correct_predictions': correct_predictions,
        'total_forbidden': total_forbidden,
        'success': correct_predictions >= 4,
        'success_criterion': 'Predict ≥ 4/5 forbidden k values',
        'notes': 'Wave interference alone may not fully separate forbidden from working k. '
                 'Additional structural features (Fibonacci proximity, LCM resonance) explored.',
    }

    print(f"\n{'='*70}")
    print(f"CORRECT PREDICTIONS: {correct_predictions}/{total_forbidden}")
    print(f"SUCCESS: {'YES' if data['success'] else 'INCONCLUSIVE — needs better metric'}")
    print(f"{'='*70}")

    save_results(data, 'exp_04_forbidden_k')
    return data


if __name__ == '__main__':
    run()
