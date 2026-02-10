"""
Experiment 08: Falsification Suite for Landauer-Gauge Connection

Claims to falsify:
1. F₁₀ = 55 is uniquely special (not arbitrary Fibonacci choice)
2. The α formula isn't curve-fitting with free parameters
3. sin²θ_W = 3/13 isn't coincidence
4. Ξ - 1 = π/55 connection is meaningful, not numerology

Falsification tests:
A. Parameter substitution: Do other Fibonacci indices work?
B. Non-Fibonacci test: Do nearby integers work equally well?
C. Random baseline: What precision is achievable by chance?
D. Degrees of freedom: Is the formula over-parameterized?
E. Cross-validation: Does α formula predict other constants?
"""

import numpy as np
import json
from datetime import datetime
from itertools import combinations

np.random.seed(42)

# ===========================================================================
# CONSTANTS
# ===========================================================================

PHI = (1 + np.sqrt(5)) / 2

# Fibonacci sequence (1-indexed: F_1=1, F_2=1, F_3=2, ...)
FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

# Measured values
ALPHA_MEASURED = 0.0072973525643
SIN2_THETA_W_MEASURED = 0.23122
ALPHA_S_MEASURED = 0.1180

print("=" * 70)
print("FALSIFICATION SUITE: Landauer-Gauge Connection")
print("=" * 70)
print()

# ===========================================================================
# TEST A: Parameter Substitution
# ===========================================================================

print("TEST A: Does F₁₀ = 55 work better than other Fibonacci indices?")
print("-" * 70)
print()

def alpha_formula(F_a, F_b, F_c, F_d):
    """
    α = [F_a/(F_b × φ × F_c)] × [1 - F_c/(4π × F_d²)]
    Original: F_a=2(F_3), F_b=3(F_4), F_c=55(F_10), F_d=13(F_7)
    """
    if F_b == 0 or F_c == 0 or F_d == 0:
        return float('inf')
    base = F_a / (F_b * PHI * F_c)
    correction = 1 - F_c / (4 * np.pi * F_d**2)
    if correction <= 0:
        return float('inf')
    return base * correction

# Test all Fibonacci substitutions for F_c (the "55" position)
print("Varying F_c (hierarchy depth) with F_a=2, F_b=3, F_d=13:")
print()
print(f"{'F_c index':<10} {'F_c value':<10} {'α_calc':<14} {'error (ppm)':<12}")
print("-" * 50)

results_a = []
for i in range(5, 15):  # F_5 to F_14
    F_c = FIB[i]
    alpha = alpha_formula(2, 3, F_c, 13)
    if alpha != float('inf'):
        error_ppm = abs(alpha - ALPHA_MEASURED) / ALPHA_MEASURED * 1e6
        results_a.append((i, F_c, alpha, error_ppm))
        marker = " <-- ORIGINAL" if i == 10 else ""
        print(f"F_{i:<8} {F_c:<10} {alpha:<14.10f} {error_ppm:<12.1f}{marker}")

print()
best_a = min(results_a, key=lambda x: x[3])
print(f"Best fit: F_{best_a[0]} = {best_a[1]} with {best_a[3]:.1f} ppm error")
print(f"F₁₀ = 55 is {'OPTIMAL' if best_a[0] == 10 else 'NOT optimal'}")
print()

# ===========================================================================
# TEST B: Non-Fibonacci Test
# ===========================================================================

print("TEST B: Do non-Fibonacci integers near 55 work equally well?")
print("-" * 70)
print()

print("Testing integers from 45 to 65:")
print()
print(f"{'Value':<10} {'α_calc':<14} {'error (ppm)':<12}")
print("-" * 40)

results_b = []
for n in range(45, 66):
    alpha = alpha_formula(2, 3, n, 13)
    if alpha != float('inf'):
        error_ppm = abs(alpha - ALPHA_MEASURED) / ALPHA_MEASURED * 1e6
        results_b.append((n, alpha, error_ppm))
        marker = " <-- F₁₀" if n == 55 else ""
        if error_ppm < 100 or n == 55 or n in [50, 54, 56, 60]:
            print(f"{n:<10} {alpha:<14.10f} {error_ppm:<12.1f}{marker}")

print()
best_b = min(results_b, key=lambda x: x[2])
print(f"Best non-constrained integer: {best_b[0]} with {best_b[2]:.1f} ppm error")

# Count how many integers beat 55
better_than_55 = [r for r in results_b if r[2] < 5.7 and r[0] != 55]
print(f"Integers with error < 5.7 ppm (excluding 55): {len(better_than_55)}")
if better_than_55:
    for r in better_than_55:
        print(f"  {r[0]}: {r[2]:.1f} ppm")
print()

# ===========================================================================
# TEST C: Random Baseline
# ===========================================================================

print("TEST C: What precision is achievable by random parameter selection?")
print("-" * 70)
print()

# With 4 integer parameters (a, b, c, d) in range 1-100, what's the chance
# of getting < 10 ppm accuracy on α?

N_RANDOM = 100000
hits_10ppm = 0
hits_100ppm = 0
best_random = (None, None, None, None, float('inf'))

for _ in range(N_RANDOM):
    a, b, c, d = np.random.randint(1, 101, size=4)
    alpha = alpha_formula(a, b, c, d)
    if alpha != float('inf') and alpha > 0:
        error_ppm = abs(alpha - ALPHA_MEASURED) / ALPHA_MEASURED * 1e6
        if error_ppm < 10:
            hits_10ppm += 1
        if error_ppm < 100:
            hits_100ppm += 1
        if error_ppm < best_random[4]:
            best_random = (a, b, c, d, error_ppm)

print(f"Random trials: {N_RANDOM:,}")
print(f"Hits < 10 ppm:  {hits_10ppm} ({hits_10ppm/N_RANDOM*100:.3f}%)")
print(f"Hits < 100 ppm: {hits_100ppm} ({hits_100ppm/N_RANDOM*100:.3f}%)")
print(f"Best random: a={best_random[0]}, b={best_random[1]}, c={best_random[2]}, d={best_random[3]}")
print(f"             error = {best_random[4]:.1f} ppm")
print()

# ===========================================================================
# TEST D: Degrees of Freedom Analysis
# ===========================================================================

print("TEST D: Is the formula over-parameterized?")
print("-" * 70)
print()

# The formula has structure: α = (a/b) × (1/φ) × (1/c) × (1 - c/(4πd²))
# Effective DOF = 4 integers (a, b, c, d)
# Target = 1 real number (α)
# Constraint: we also claim sin²θ_W = b/d

print("The formula attempts to predict TWO constants with FOUR integers:")
print("  1. α = [F_a/(F_b × φ × F_c)] × [1 - F_c/(4π × F_d²)]")
print("  2. sin²θ_W = F_b/F_d")
print()

# Test: do F_3=2, F_4=3, F_7=13, F_10=55 satisfy BOTH constraints?
sin2_calc = 3 / 13
sin2_error = abs(sin2_calc - SIN2_THETA_W_MEASURED) / SIN2_THETA_W_MEASURED * 100
alpha_calc = alpha_formula(2, 3, 55, 13)
alpha_error = abs(alpha_calc - ALPHA_MEASURED) / ALPHA_MEASURED * 1e6

print(f"With (F_3, F_4, F_7, F_10) = (2, 3, 13, 55):")
print(f"  α error:        {alpha_error:.1f} ppm")
print(f"  sin²θ_W error:  {sin2_error:.2f}%")
print()

# Now test: how many 4-tuples (a,b,c,d) can match BOTH?
print("Searching for 4-tuples matching BOTH α (< 100 ppm) AND sin²θ_W (< 1%):")
print()

matches_both = []
for a in range(1, 20):
    for b in range(1, 20):
        for c in range(10, 200):
            for d in range(5, 50):
                alpha = alpha_formula(a, b, c, d)
                if alpha != float('inf') and alpha > 0:
                    alpha_err = abs(alpha - ALPHA_MEASURED) / ALPHA_MEASURED * 1e6
                    sin2 = b / d
                    sin2_err = abs(sin2 - SIN2_THETA_W_MEASURED) / SIN2_THETA_W_MEASURED * 100
                    
                    if alpha_err < 100 and sin2_err < 1:
                        matches_both.append((a, b, c, d, alpha_err, sin2_err))

print(f"Total 4-tuples searched: {19 * 19 * 190 * 45:,}")
print(f"Matches both constraints: {len(matches_both)}")
print()

if matches_both:
    # Sort by combined error
    matches_both.sort(key=lambda x: x[4] + x[5]*100)
    print("Top 5 matches:")
    print(f"{'(a,b,c,d)':<20} {'α error (ppm)':<15} {'sin²θ_W error (%)':<15}")
    print("-" * 50)
    for m in matches_both[:5]:
        print(f"({m[0]},{m[1]},{m[2]},{m[3]}){'':<10} {m[4]:<15.1f} {m[5]:<15.2f}")
    print()
    
    # Check if (2,3,55,13) is among them
    fib_match = [m for m in matches_both if m[0]==2 and m[1]==3 and m[2]==55 and m[3]==13]
    print(f"Is (2,3,55,13) among the matches? {'YES' if fib_match else 'NO'}")

print()

# ===========================================================================
# TEST E: Cross-Validation with α_s
# ===========================================================================

print("TEST E: Does the framework predict α_s correctly?")
print("-" * 70)
print()

# Claimed: α_s = F_4/(2φ × F_6) = 3/(2 × 1.618 × 8)
alpha_s_calc = 3 / (2 * PHI * 8)
alpha_s_error = abs(alpha_s_calc - ALPHA_S_MEASURED) / ALPHA_S_MEASURED * 100

print(f"α_s = F_4/(2φ × F_6) = 3/(2 × {PHI:.4f} × 8)")
print(f"    = {alpha_s_calc:.6f}")
print(f"Measured: {ALPHA_S_MEASURED:.4f}")
print(f"Error: {alpha_s_error:.2f}%")
print()

# Test alternative Fibonacci combinations for α_s
print("Testing alternative formulas for α_s:")
print()

def alpha_s_formula(F_num, F_den, extra_factor=1):
    return F_num / (extra_factor * PHI * F_den)

# Try different numerator/denominator combinations
best_alpha_s = (None, None, None, float('inf'))
for num_idx in range(3, 10):
    for den_idx in range(3, 10):
        for factor in [1, 2, 3, 4]:
            F_num = FIB[num_idx]
            F_den = FIB[den_idx]
            calc = alpha_s_formula(F_num, F_den, factor)
            if 0.01 < calc < 1.0:
                err = abs(calc - ALPHA_S_MEASURED) / ALPHA_S_MEASURED * 100
                if err < best_alpha_s[3]:
                    best_alpha_s = (num_idx, den_idx, factor, err)

print(f"Best α_s formula: F_{best_alpha_s[0]}/(factor × φ × F_{best_alpha_s[1]})")
print(f"  where factor = {best_alpha_s[2]}")
print(f"  F_{best_alpha_s[0]} = {FIB[best_alpha_s[0]]}, F_{best_alpha_s[1]} = {FIB[best_alpha_s[1]]}")
print(f"  error = {best_alpha_s[3]:.2f}%")
print()

# Check if the claimed formula (F_4, F_6, factor=2) is optimal
claimed_is_optimal = (best_alpha_s[0] == 4 and best_alpha_s[1] == 6 and best_alpha_s[2] == 2)
print(f"Is claimed formula (F_4, 2, F_6) optimal? {'YES' if claimed_is_optimal else 'NO'}")
print()

# ===========================================================================
# TEST F: The Ξ Connection
# ===========================================================================

print("TEST F: Is the Ξ - 1 = π/55 connection special?")
print("-" * 70)
print()

# If Ξ - 1 = π/F_n, does n = 10 have special properties?
print("Testing Ξ - 1 = π/F_n for different n:")
print()

print(f"{'n':<5} {'F_n':<8} {'π/F_n':<12} {'Interpretation'}")
print("-" * 50)

for n in range(6, 14):
    F_n = FIB[n]
    xi_minus_1 = np.pi / F_n
    interp = ""
    if n == 10:
        interp = "<-- Ξ - 1 = 0.0571"
    elif n == 7:
        interp = "(F_7 = 13 = gauge closure)"
    print(f"{n:<5} {F_n:<8} {xi_minus_1:<12.6f} {interp}")

print()

# The KEY question: why F_10 and not F_7 or F_8?
# F_10 = 55 = 5 × 11 = hierarchical structure
# 55 appears in EM hierarchy (1/137 ≈ related to 55)

print("Follow-up: Why F_10 specifically?")
print()
print("1/137 ≈ 0.00730")
print("F_10 = 55")
print("55 × 137 = 7535")
print("8 × 55 = 440 (concert pitch A)")
print("55 = F_10 = number of EM hierarchy levels?")
print()

# ===========================================================================
# SUMMARY
# ===========================================================================

print("=" * 70)
print("FALSIFICATION SUMMARY")
print("=" * 70)
print()

results = {
    'test_a': {
        'description': 'F_10 uniqueness among Fibonacci indices',
        'result': 'PASS' if best_a[0] == 10 else 'FAIL',
        'detail': f'F_{best_a[0]} gives best fit ({best_a[3]:.1f} ppm)'
    },
    'test_b': {
        'description': 'F_10 vs nearby non-Fibonacci integers',
        'result': 'PASS' if best_b[0] == 55 else 'PARTIAL',
        'detail': f'Best integer: {best_b[0]} ({best_b[2]:.1f} ppm), 55 = {[r[2] for r in results_b if r[0]==55][0]:.1f} ppm'
    },
    'test_c': {
        'description': 'Random baseline precision',
        'result': f'{hits_10ppm/N_RANDOM*100:.3f}% random trials beat 10 ppm',
        'detail': f'Best random found {best_random[4]:.1f} ppm'
    },
    'test_d': {
        'description': 'Joint constraint (α AND sin²θ_W)',
        'result': f'{len(matches_both)} tuples satisfy both',
        'detail': 'Strong constraint if few matches'
    },
    'test_e': {
        'description': 'α_s cross-validation',
        'result': 'PASS' if claimed_is_optimal else 'PARTIAL',
        'detail': f'Claimed formula error: {alpha_s_error:.2f}%'
    }
}

for test, data in results.items():
    print(f"{test.upper()}: {data['description']}")
    print(f"  Result: {data['result']}")
    print(f"  Detail: {data['detail']}")
    print()

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'results': results,
    'parameters': {
        'alpha_measured': ALPHA_MEASURED,
        'sin2_theta_w_measured': SIN2_THETA_W_MEASURED,
        'alpha_s_measured': ALPHA_S_MEASURED,
    },
    'fibonacci_used': FIB,
    'random_trials': N_RANDOM,
    'matches_both_constraints': len(matches_both),
}

output_path = f'../results/exp_08_falsification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Results saved to {output_path}")
