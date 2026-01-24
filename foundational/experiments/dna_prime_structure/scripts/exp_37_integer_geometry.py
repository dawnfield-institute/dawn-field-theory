"""
Experiment 37: Integer Geometry

Can we compute geometric quantities using ONLY:
- Fibonacci numbers
- Lucas numbers  
- Integer ratios
- NO decimals, NO π, NO trig functions

Key insight: π = 55(Ξ - 1), but we need integer approximations.

Approximations to explore:
- π ≈ 22/7 (ancient)
- π ≈ 355/113 (Zu Chongzhi, 450 AD)
- φ ≈ F(n+1)/F(n)
- φ² ≈ 144/55
- √5 = 2φ - 1 ≈ (2F(n+1) - F(n))/F(n)
- √2, √3 via Pell equation convergents

The question: can Fibonacci/Lucas give us ALL the irrational constants we need?
"""

from fractions import Fraction
from typing import Tuple, Dict
import math

# =============================================================================
# Fibonacci and Lucas sequences
# =============================================================================

def fib(n: int) -> int:
    """Fibonacci number F(n)"""
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

def lucas(n: int) -> int:
    """Lucas number L(n)"""
    if n == 0: return 2
    if n == 1: return 1
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

# Generate sequences
FIB = [fib(i) for i in range(20)]
LUCAS = [lucas(i) for i in range(20)]

print("=" * 70)
print("EXPERIMENT 37: INTEGER GEOMETRY")
print("Computing geometry with Fibonacci/Lucas - no decimals, no π, no trig")
print("=" * 70)

print(f"\nFibonacci: {FIB[:15]}")
print(f"Lucas:     {LUCAS[:15]}")

# =============================================================================
# Part 1: Integer approximations of key constants
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: INTEGER APPROXIMATIONS OF IRRATIONAL CONSTANTS")
print("=" * 70)

# φ (golden ratio) from Fibonacci
print("\n[1] φ (Golden Ratio) from Fibonacci:")
for n in [5, 8, 10, 12]:
    phi_approx = Fraction(FIB[n+1], FIB[n])
    actual = (1 + math.sqrt(5)) / 2
    error = abs(float(phi_approx) - actual)
    print(f"  F{n+1}/F{n} = {FIB[n+1]}/{FIB[n]} = {float(phi_approx):.10f} (error: {error:.2e})")

# φ² from Fibonacci
print("\n[2] φ² from Fibonacci (F(n+2)/F(n)):")
for n in [5, 8, 10, 12]:
    phi2_approx = Fraction(FIB[n+2], FIB[n])
    actual = ((1 + math.sqrt(5)) / 2) ** 2
    error = abs(float(phi2_approx) - actual)
    print(f"  F{n+2}/F{n} = {FIB[n+2]}/{FIB[n]} = {float(phi2_approx):.10f} (error: {error:.2e})")

# √5 from Fibonacci: √5 = 2φ - 1 = (2F(n+1) - F(n))/F(n)
print("\n[3] √5 from Fibonacci (2F(n+1) - F(n))/F(n):")
for n in [5, 8, 10, 12]:
    sqrt5_num = 2 * FIB[n+1] - FIB[n]
    sqrt5_approx = Fraction(sqrt5_num, FIB[n])
    actual = math.sqrt(5)
    error = abs(float(sqrt5_approx) - actual)
    print(f"  n={n}: {sqrt5_num}/{FIB[n]} = {float(sqrt5_approx):.10f} (error: {error:.2e})")

# Lucas-Fibonacci identity: L(n)² - 5F(n)² = 4(-1)^n
# This gives us: L(n)/F(n) approaches √5 as n increases... wait no
# Actually: L(n) = F(n-1) + F(n+1)
# And: L(n)² = 5F(n)² + 4(-1)^n
# So: √5 ≈ L(n)/F(n) for large n... let's check

print("\n[4] √5 from Lucas/Fibonacci (L(n)²/F(n)² - 4(-1)^n/F(n)²)^0.5:")
print("    Actually, L(2n)/F(2n) converges to √5:")
for n in [3, 4, 5, 6]:
    # L(2n)/F(2n) -> √5? No...
    # Let me try: F(2n)/F(n)² 
    # Actually the identity is: F(n)L(n) = F(2n)
    ratio = Fraction(LUCAS[n], FIB[n])
    print(f"  L{n}/F{n} = {LUCAS[n]}/{FIB[n]} = {float(ratio):.6f}")

# π approximations using integer ratios
print("\n[5] π approximations (integer ratios):")
pi_approx = [
    (22, 7, "Archimedes"),
    (333, 106, "Milü lower"),
    (355, 113, "Zu Chongzhi (355/113)"),
    (103993, 33102, "continued fraction"),
]
for num, den, name in pi_approx:
    approx = Fraction(num, den)
    error = abs(float(approx) - math.pi)
    print(f"  {num}/{den} = {float(approx):.10f} ({name}, error: {error:.2e})")

# Can we get π from Fibonacci?
print("\n[6] Searching for π in Fibonacci/Lucas ratios...")
best_pi_error = 1.0
best_pi_ratio = None
for i in range(2, 15):
    for j in range(2, 15):
        if i != j:
            # Try various combinations
            for mult_i in [1, 2, 3, 4]:
                for mult_j in [1, 2, 3, 4]:
                    ratio = (mult_i * FIB[i]) / (mult_j * FIB[j])
                    error = abs(ratio - math.pi)
                    if error < best_pi_error:
                        best_pi_error = error
                        best_pi_ratio = (mult_i, i, mult_j, j, ratio)

if best_pi_ratio:
    mi, i, mj, j, ratio = best_pi_ratio
    print(f"  Best Fibonacci π: {mi}×F{i}/{mj}×F{j} = {mi}×{FIB[i]}/{mj}×{FIB[j]} = {ratio:.6f}")
    print(f"  Error: {best_pi_error:.4f} (not great)")

# √2 and √3 from Pell equations
print("\n[7] √2 from Pell equation (solutions to x² - 2y² = 1):")
# Pell convergents for √2: 1/1, 3/2, 7/5, 17/12, 41/29, 99/70, 239/169
pell_sqrt2 = [(1,1), (3,2), (7,5), (17,12), (41,29), (99,70), (239,169)]
for num, den in pell_sqrt2:
    approx = Fraction(num, den)
    error = abs(float(approx) - math.sqrt(2))
    print(f"  {num}/{den} = {float(approx):.10f} (error: {error:.2e})")

print("\n[8] √3 from Pell equation (solutions to x² - 3y² = 1):")
# Pell convergents for √3: 2/1, 7/4, 26/15, 97/56, 362/209
pell_sqrt3 = [(2,1), (7,4), (26,15), (97,56), (362,209)]
for num, den in pell_sqrt3:
    approx = Fraction(num, den)
    error = abs(float(approx) - math.sqrt(3))
    print(f"  {num}/{den} = {float(approx):.10f} (error: {error:.2e})")

# =============================================================================
# Part 2: Geometric formulas with integer approximations
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: GEOMETRIC FORMULAS WITH INTEGER APPROXIMATIONS")
print("=" * 70)

# Define our integer constants
PI = Fraction(355, 113)  # Best simple approximation
SQRT2 = Fraction(99, 70)
SQRT3 = Fraction(97, 56)
PHI = Fraction(FIB[13], FIB[12])  # 233/144
PHI2 = Fraction(FIB[14], FIB[12])  # 377/144 ≈ 144/55 but more precise
SQRT5 = Fraction(2 * FIB[13] - FIB[12], FIB[12])

print(f"\nUsing these integer approximations:")
print(f"  π ≈ {PI} = {float(PI):.10f}")
print(f"  √2 ≈ {SQRT2} = {float(SQRT2):.10f}")
print(f"  √3 ≈ {SQRT3} = {float(SQRT3):.10f}")
print(f"  φ ≈ {PHI} = {float(PHI):.10f}")
print(f"  √5 ≈ {SQRT5} = {float(SQRT5):.10f}")

# For a unit side length s = 1
s = 1

print("\n" + "-" * 50)
print("Perimeters (side/radius = 1)")
print("-" * 50)

# Perimeter of equilateral triangle: 3s
p_eq_tri = 3 * s
print(f"\n1. Equilateral triangle: 3s = {p_eq_tri}")
print(f"   Pure integer - no approximation needed!")

# Perimeter of square: 4s
p_square = 4 * s
print(f"\n2. Square: 4s = {p_square}")
print(f"   Pure integer - no approximation needed!")

# Perimeter of semicircle: πr + 2r = r(π + 2)
r = 1
p_semicircle = PI + 2
print(f"\n3. Semicircle: r(π + 2) = {p_semicircle} = {float(p_semicircle):.10f}")
print(f"   Using 355/113 for π")
print(f"   As fraction: ({PI.numerator} + 2×{PI.denominator})/{PI.denominator} = {p_semicircle}")

# Perimeter of circle: 2πr
p_circle = 2 * PI
print(f"\n4. Circle: 2πr = 2 × {PI} = {p_circle} = {float(p_circle):.10f}")
print(f"   = {p_circle.numerator}/{p_circle.denominator}")

print("\n" + "-" * 50)
print("Areas (side/radius = 1)")
print("-" * 50)

# Area of equilateral triangle: (√3/4)s²
area_eq_tri = SQRT3 / 4
print(f"\n5. Equilateral triangle: (√3/4)s² = {SQRT3}/4 = {area_eq_tri}")
print(f"   = {float(area_eq_tri):.10f}")
print(f"   Actual: {math.sqrt(3)/4:.10f}")
print(f"   Using Pell equation √3 ≈ 97/56")

# Area of square: s²
area_square = s * s
print(f"\n6. Square: s² = {area_square}")
print(f"   Pure integer!")

# Area of circle: πr²
area_circle = PI
print(f"\n7. Circle: πr² = {PI} = {float(PI):.10f}")
print(f"   Actual: {math.pi:.10f}")

print("\n" + "-" * 50)
print("Volumes (side/radius = 1)")
print("-" * 50)

# Volume of tetrahedron: (√2/12)a³
vol_tetra = SQRT2 / 12
print(f"\n8. Tetrahedron: (√2/12)a³ = {SQRT2}/12 = {vol_tetra}")
print(f"   = {float(vol_tetra):.10f}")
print(f"   Actual: {math.sqrt(2)/12:.10f}")

# Volume of cube: s³
vol_cube = s * s * s
print(f"\n9. Cube: s³ = {vol_cube}")
print(f"   Pure integer!")

# Volume of half-sphere: (2/3)πr³
vol_hemisphere = Fraction(2, 3) * PI
print(f"\n10. Hemisphere: (2/3)πr³ = (2/3) × {PI} = {vol_hemisphere}")
print(f"    = {float(vol_hemisphere):.10f}")
print(f"    Actual: {(2/3) * math.pi:.10f}")

# Volume of sphere: (4/3)πr³
vol_sphere = Fraction(4, 3) * PI
print(f"\n11. Sphere: (4/3)πr³ = (4/3) × {PI} = {vol_sphere}")
print(f"    = {float(vol_sphere):.10f}")
print(f"    Actual: {(4/3) * math.pi:.10f}")

print("\n" + "-" * 50)
print("Ellipse (a = 2, b = 1)")
print("-" * 50)

a, b = 2, 1

# Area of ellipse: πab
area_ellipse = PI * a * b
print(f"\n12. Ellipse area: πab = {PI} × {a} × {b} = {area_ellipse}")
print(f"    = {float(area_ellipse):.10f}")

# Perimeter of ellipse: Ramanujan approximation
# P ≈ π(3(a+b) - √((3a+b)(a+3b)))
# This needs √ which we can approximate
inner = (3*a + b) * (a + 3*b)  # = 7 × 5 = 35
# We need √35 ≈ 5.916
# √35 = √(36-1) ≈ 6 - 1/12 ≈ 71/12
SQRT35 = Fraction(71, 12)  # approximate
p_ellipse = PI * (3*(a + b) - SQRT35)
print(f"\n13. Ellipse perimeter (Ramanujan):")
print(f"    π(3(a+b) - √((3a+b)(a+3b)))")
print(f"    = {PI} × (3×{a+b} - √35)")
print(f"    ≈ {PI} × ({3*(a+b)} - {SQRT35})")
print(f"    = {p_ellipse} = {float(p_ellipse):.4f}")
print(f"    Actual ≈ {math.pi * (3*(a+b) - math.sqrt(35)):.4f}")

print("\n" + "-" * 50)
print("Egg (using Cassini oval approximation)")
print("-" * 50)

# Egg volume: approximately (2/3)πab² for a prolate spheroid
# where a is length, b is width
a_egg, b_egg = 3, 2  # typical egg proportions
vol_egg = Fraction(2, 3) * PI * a_egg * b_egg * b_egg
print(f"\n14. Egg volume (prolate spheroid): (2/3)πab²")
print(f"    = (2/3) × {PI} × {a_egg} × {b_egg}² = {vol_egg}")
print(f"    = {float(vol_egg):.4f}")
print(f"    (Approximation - real eggs are more complex)")

# =============================================================================
# Part 3: Can we do it with ONLY Fibonacci?
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: PURE FIBONACCI GEOMETRY")
print("=" * 70)

print("""
The challenge: can ALL geometric constants be derived from Fibonacci?

What we can express with Fibonacci:
  ✓ φ = F(n+1)/F(n)
  ✓ φ² = F(n+2)/F(n)  
  ✓ √5 = (2F(n+1) - F(n))/F(n)
  ✗ π - NOT directly from Fibonacci
  ✗ √2 - NOT from Fibonacci (needs Pell sequence)
  ✗ √3 - NOT from Fibonacci (needs Pell sequence)

The Pell sequences for √2 and √3 are SEPARATE from Fibonacci.
They satisfy x² - Ny² = 1, not the Fibonacci recurrence.

However, there's a beautiful connection:
""")

# The connection between Fibonacci and other sequences
print("[Connection 1] Fibonacci and √5:")
print(f"  √5 = lim(L(n)/F(n)) as n → ∞")
print(f"  More precisely: L(n)² = 5F(n)² + 4(-1)^n")
print(f"  So: √5 = √(L(n)² - 4(-1)^n) / F(n)")
for n in [5, 10, 15]:
    L = LUCAS[n]
    F = FIB[n]
    sign = (-1) ** n
    # L² = 5F² + 4(-1)^n
    inner = L*L - 4*sign
    # inner should equal 5F²
    print(f"  n={n}: L={L}, F={F}, L² - 4(-1)^n = {inner} = 5 × {F}² = {5*F*F} ✓" if inner == 5*F*F else f"  n={n}: mismatch")

print("\n[Connection 2] Why 55 appears in π/55:")
print(f"  55 = F₁₀ = T₁₀ = 5 × 11 = F₅ × L₅")
print(f"  This encodes both Fibonacci AND Lucas at depth 5")
print(f"  The π comes from the Möbius twist (topological, not algebraic)")
print(f"  So π/55 couples continuous topology (π) with discrete recursion (55)")

print("\n[Connection 3] The 22/7 approximation:")
print(f"  22 = 2 × 11 = 2 × L₅")
print(f"  7 = L₄")
print(f"  So 22/7 = 2L₅/L₄ = 2 × 11/7 ≈ π")
print(f"  The Lucas sequence gives us π!")

# Let's verify
print(f"\n  2L₅/L₄ = 2 × {LUCAS[5]}/{LUCAS[4]} = {2*LUCAS[5]}/{LUCAS[4]} = {2*LUCAS[5]/LUCAS[4]:.6f}")
print(f"  π = {math.pi:.6f}")
print(f"  Error: {abs(2*LUCAS[5]/LUCAS[4] - math.pi):.6f}")

# Better approximation?
print("\n[Searching for better π from Lucas...]")
best_error = 1.0
best = None
for i in range(2, 15):
    for j in range(2, 15):
        for m in range(1, 5):
            for n in range(1, 5):
                ratio = (m * LUCAS[i]) / (n * LUCAS[j])
                error = abs(ratio - math.pi)
                if error < best_error:
                    best_error = error
                    best = (m, i, n, j, ratio)
m, i, n, j, ratio = best
print(f"  Best: {m}L{i}/{n}L{j} = {m}×{LUCAS[i]}/{n}×{LUCAS[j]} = {ratio:.6f}")
print(f"  Error: {best_error:.6f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: WHAT CAN BE COMPUTED WITH INTEGERS ONLY")
print("=" * 70)

summary = """
PURE INTEGERS (no approximation needed):
  ✓ Perimeter of equilateral triangle: 3s
  ✓ Perimeter of square: 4s
  ✓ Area of square: s²
  ✓ Volume of cube: s³

FIBONACCI-DERIVABLE:
  ✓ φ = F(n+1)/F(n)
  ✓ φ² = F(n+2)/F(n) ≈ 144/55
  ✓ √5 = (2F(n+1) - F(n))/F(n)

LUCAS-DERIVABLE:
  ~ π ≈ 22/7 = 2L₅/L₄ (0.04% error)
  
PELL SEQUENCE (separate from Fibonacci):
  ✓ √2 ≈ 99/70 (Pell convergent)
  ✓ √3 ≈ 97/56 (Pell convergent)

REQUIRES BOTH Fibonacci/Lucas AND Pell:
  - Area of equilateral triangle: (√3/4)s² → needs Pell √3
  - Volume of tetrahedron: (√2/12)a³ → needs Pell √2
  - Circle formulas: need π ≈ 22/7 = 2L₅/L₄

THE INSIGHT:
  π, √2, √3 come from DIFFERENT integer sequences:
  - Fibonacci/Lucas → φ, √5
  - Pell for √2 → convergents of x² - 2y² = 1
  - Pell for √3 → convergents of x² - 3y² = 1
  - π → ??? (transcendental, but 22/7 = 2L₅/L₄ works!)

  The 55 in Ξ - 1 = π/55 bridges these worlds:
  55 = F₅ × L₅ contains both golden sequences,
  and π/55 couples them to the transcendental.
"""

print(summary)

# Final table
print("\nFINAL ANSWER TABLE:")
print("-" * 70)
print(f"{'Formula':<30} {'Integer Expression':<25} {'Value':<15}")
print("-" * 70)
results = [
    ("Perimeter eq. triangle", "3s", "3"),
    ("Perimeter square", "4s", "4"),
    ("Perimeter semicircle", "(355/113 + 2)r", f"{float(PI + 2):.6f}"),
    ("Perimeter circle", "2 × 355/113 × r", f"{float(2*PI):.6f}"),
    ("Area eq. triangle", "(97/56)/4 × s²", f"{float(SQRT3/4):.6f}"),
    ("Area square", "s²", "1"),
    ("Area circle", "355/113 × r²", f"{float(PI):.6f}"),
    ("Volume tetrahedron", "(99/70)/12 × a³", f"{float(SQRT2/12):.6f}"),
    ("Volume cube", "s³", "1"),
    ("Volume hemisphere", "(2/3)(355/113)r³", f"{float(Fraction(2,3)*PI):.6f}"),
    ("Volume sphere", "(4/3)(355/113)r³", f"{float(Fraction(4,3)*PI):.6f}"),
    ("Perimeter ellipse", "Ramanujan + √approx", "~9.69"),
    ("Area ellipse", "355/113 × a × b", f"{float(PI * 2 * 1):.6f}"),
    ("Volume egg", "(2/3)(355/113)ab²", f"{float(Fraction(2,3)*PI*3*4):.6f}"),
]

for formula, expr, val in results:
    print(f"{formula:<30} {expr:<25} {val:<15}")

print("-" * 70)
