"""
Experiment 18b: Refined Closed-Form Search

Best candidates from exp_18:
1. (δ-4)/(α × L₁₀ × 4.05) → error 1.53e-06
2. 1/(δ × 400) → error 2.83e-06
3. (δ-4)/(L₁₀ × 10) → error 5.81e-06

Let's find the EXACT closed form.
"""

from mpmath import mp, mpf, sqrt, pi, fib
import numpy as np

mp.dps = 100  # Very high precision

# Lucas numbers
def lucas(n):
    PHI_f = (1 + sqrt(5)) / 2
    return int(round(float(PHI_f**n + (-1/PHI_f)**n)))

# Constants
PHI = (1 + sqrt(5)) / 2
PHI_INV = 1 / PHI
DELTA = mpf('4.6692016091029906718532038204662016172581855774757686327456513430')
ALPHA = mpf('2.5029078750958928222839028732182157863812713767271499773361920567')
R_INF = mpf('3.5699456718709449018420051513864989367638369115148323781388011418')

F = [fib(n) for n in range(20)]
L = [lucas(n) for n in range(20)]

# Exact offset
target = R_INF / pi
z_exact = (34 * target - 55) / (89 - 55 * target)
delta_z = z_exact - (-PHI_INV)

print("=" * 70)
print("EXPERIMENT 18b: Refined Closed-Form Search")
print("=" * 70)
print(f"\nΔz = {delta_z}")
print(f"1/Δz = {1/delta_z}")

# ============================================================
# PART 1: The 4.05 coefficient
# ============================================================
print("\n### PART 1: The mysterious 4.05")

# (δ-4)/(α × L₁₀ × c) = Δz
# So c = (δ-4)/(α × L₁₀ × Δz)
c_exact = (DELTA - 4) / (ALPHA * L[10] * delta_z)
print(f"\nExact coefficient c = (δ-4)/(α × L₁₀ × Δz)")
print(f"c = {c_exact}")

# Is c related to something simple?
print(f"\nc / π = {c_exact / pi}")
print(f"c / φ = {c_exact / PHI}")
print(f"c × φ = {c_exact * PHI}")
print(f"c - 4 = {c_exact - 4}")
print(f"c / (φ-1) = {c_exact / (PHI - 1)}")  # φ-1 = 1/φ
print(f"c × (φ-1) = {c_exact * (PHI - 1)}")

# ============================================================
# PART 2: 1/(δ × k) form
# ============================================================
print("\n### PART 2: 1/(δ × k) form")

# 1/(δ × k) = Δz → k = 1/(δ × Δz)
k_exact = 1 / (DELTA * delta_z)
print(f"\nExact k = 1/(δ × Δz)")
print(f"k = {k_exact}")
print(f"\nk / F₁₀ = {k_exact / F[10]}")
print(f"k / F₉ = {k_exact / F[9]}")
print(f"k / L₅ = {k_exact / L[5]}")

# Check if k is close to a nice product
print(f"\nk / (F₉ × 11.77) = {k_exact / (F[9] * 11.77)}")
print(f"F₉ × L₅ = {F[9] * L[5]}")  # 34 × 11 = 374
print(f"F₁₀ × L₄ = {F[10] * L[4]}")  # 55 × 7 = 385

# ============================================================
# PART 3: 1/(F₁₀ × F₉) form
# ============================================================
print("\n### PART 3: 1/(F₁₀ × F₉) correction")

# We have 1/(F₁₀ × F₉) = 1/1870 ≈ 0.000535 vs Δz ≈ 0.000538
# Need a correction factor

base = 1 / (F[10] * F[9])
correction = delta_z / base
print(f"\nΔz / (1/(F₁₀×F₉)) = {correction}")
print(f"1/correction = {1/correction}")

# The correction is close to 1.0065...
print(f"\ncorrection - 1 = {correction - 1}")
print(f"(correction - 1) × 1000 = {(correction - 1) * 1000}")

# Check if correction relates to known constants
print(f"\n(correction - 1) × F₁₀ = {(correction - 1) * F[10]}")
print(f"(correction - 1) × L₁₀ = {(correction - 1) * L[10]}")
print(f"(correction - 1) × δ = {(correction - 1) * DELTA}")

# ============================================================
# PART 4: Quadratic Fibonacci relation
# ============================================================
print("\n### PART 4: Quadratic Relations")

# F₁₀ × F₉ = F₁₀² - F₁₀ × F₈ = F₁₀ × (F₁₀ - F₈) = 55 × 21 ≠ 1870
# Actually F₁₀ × F₉ = 55 × 34 = 1870
# And F₁₀² - F₉² = (F₁₀ + F₉)(F₁₀ - F₉) = 89 × 21 = 1869
print(f"F₁₀² - F₉² = {F[10]**2 - F[9]**2}")  # 3025 - 1156 = 1869
print(f"F₁₀ × F₉ = {F[10] * F[9]}")  # 1870

# Difference!
print(f"\n1/Δz = {1/delta_z}")
print(f"F₁₀² - F₉² = {F[10]**2 - F[9]**2}")
print(f"F₁₀ × F₉ = {F[10] * F[9]}")
print(f"(F₁₀² - F₉² + F₁₀×F₉)/2 = {(F[10]**2 - F[9]**2 + F[10]*F[9])/2}")

# ============================================================
# PART 5: Connection to 1857.85
# ============================================================
print("\n### PART 5: 1/Δz ≈ 1857.85")

reciprocal = 1/delta_z
print(f"\n1/Δz = {reciprocal}")

# Factor analysis
print(f"\n1857.85 / 123 (L₁₀) = {reciprocal / L[10]}")
print(f"1857.85 / 55 (F₁₀) = {reciprocal / F[10]}")
print(f"1857.85 / 89 (F₁₁) = {reciprocal / 89}")

# 1857.85 ≈ 1870 × 0.9935
print(f"\n1/Δz / (F₁₀ × F₉) = {reciprocal / (F[10] * F[9])}")
print(f"That's ≈ 1 - 0.0065")

# The deviation from F₁₀ × F₉
deviation = reciprocal - F[10] * F[9]
print(f"\n1/Δz - F₁₀×F₉ = {deviation}")
print(f"This is about -12.15")

# Is -12.15 related to something?
print(f"\nDeviation / π = {deviation / pi}")
print(f"Deviation / φ = {deviation / PHI}")
print(f"Deviation / (δ-4) = {deviation / (DELTA - 4)}")
print(f"Deviation / L₄ = {deviation / L[4]}")  # L₄ = 7
print(f"Deviation + F₇ = {deviation + F[7]}")  # F₇ = 13

# ============================================================
# PART 6: Most elegant formula
# ============================================================
print("\n" + "=" * 70)
print("### PART 6: Most Elegant Formula")
print("=" * 70)

# Test: Δz = 1 / (F₁₀ × F₉ - F₇ + correction)
# 1870 - 13 = 1857
test_base = F[10] * F[9] - F[7]  # 1870 - 13 = 1857
print(f"\nF₁₀ × F₉ - F₇ = {test_base}")
print(f"1/Δz = {reciprocal}")
print(f"Difference = {reciprocal - test_base}")

# Very close! Need ~0.85 more
print(f"\n1/(F₁₀×F₉ - F₇) = {1/test_base}")
print(f"Δz = {delta_z}")
print(f"Error = {abs(1/test_base - delta_z)}")

# Add small correction
for a, b in [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (5, 3), (8, 5)]:
    correction_term = a/b
    denom = test_base + correction_term
    val = 1/denom
    err = abs(val - delta_z)
    print(f"1/(F₁₀×F₉ - F₇ + {a}/{b}): {float(val):.10f}, error = {float(err):.2e}")

# Try with Feigenbaum constants
print("\nWith Feigenbaum constants:")
for mult in [1, 2, 4, 10]:
    denom = test_base + (DELTA - 4)/mult
    val = 1/denom
    err = abs(val - delta_z)
    print(f"1/(1857 + (δ-4)/{mult}): {float(val):.10f}, error = {float(err):.2e}")

# ============================================================
# PART 7: Ultimate formula search
# ============================================================
print("\n" + "=" * 70)
print("### PART 7: Ultimate Formula")
print("=" * 70)

# The exact denominator for 1/Δz
exact_denom = 1/delta_z
print(f"\nExact 1/Δz = {exact_denom}")
print(f"= {F[10] * F[9]} - {F[7]} + {exact_denom - test_base}")

# The fractional part
frac_part = exact_denom - test_base  # Should be ~0.85
print(f"\nFractional part = {frac_part}")
print(f"Fractional × φ = {frac_part * PHI}")
print(f"Fractional × δ = {frac_part * DELTA}")
print(f"Fractional × α = {frac_part * ALPHA}")

# Is it (δ-4)/something or (φ-1)/something?
print(f"\n(δ-4) / frac = {(DELTA-4) / frac_part}")
print(f"(φ-1) / frac = {(PHI-1) / frac_part}")
print(f"(α-2) / frac = {(ALPHA-2) / frac_part}")
print(f"(π-3) / frac = {(pi-3) / frac_part}")

# CANDIDATE: frac ≈ 1 - (π-3)/(δ-4)?
test_frac = 1 - (pi-3)/(DELTA-4)
print(f"\n1 - (π-3)/(δ-4) = {test_frac}")
print(f"Actual frac = {frac_part}")
print(f"Error = {abs(test_frac - frac_part)}")

# CANDIDATE: frac ≈ π/(4 - α)?
# No, let's try simpler

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Best Closed Forms")
print("=" * 70)

print("""
1. FIBONACCI PRODUCT FORM:
   Δz ≈ 1 / (F₁₀ × F₉ - F₇ + ε)
   where ε ≈ 0.85 is a small correction
   
   This gives: Δz ≈ 1/1857.85

2. LUCAS FORM:
   Δz ≈ (δ - 4) / (α × L₁₀ × 4.04)
   Error: ~10⁻⁶

3. PURE FEIGENBAUM:
   Δz ≈ 1 / (δ × 398.2)
   Error: ~10⁻⁶

4. INTERPRETATION:
   The base 1857 = F₁₀×F₉ - F₇ = 1870 - 13 is Fibonacci
   The correction 0.85 encodes the logistic map's deviation from φ
""")

# Final best candidate
print("\n*** FINAL BEST CANDIDATE ***")
# Try: 1/(F₁₀×F₉ - F₇ + (δ-4)/0.79)
for c in [0.78, 0.785, 0.79, 0.795, 0.8]:
    denom = test_base + (DELTA - 4)/c
    val = 1/denom
    err = abs(val - delta_z)
    print(f"1/(1857 + (δ-4)/{c}): {float(val):.12f}, error = {float(err):.2e}")
