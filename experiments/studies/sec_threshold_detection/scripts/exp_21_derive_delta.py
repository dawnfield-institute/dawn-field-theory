"""
Experiment 21: Derive δ from First Principles

Can we DERIVE δ from Fibonacci/π structure rather than using it as input?

The goal: Find an equation involving only:
- Fibonacci numbers (F_n)
- Lucas numbers (L_n)
- π
- φ (golden ratio)

That predicts δ = 4.6692016091...

Key relationships we've found:
1. r∞ = π × M₁₀(z) where z ≈ -1/φ + Δz
2. Δz ≈ π / (1857π + 4(δ-4))
3. 1857 = F₁₀ × F₉ - F₇

If we could express Δz without δ, we could solve for δ!
"""

from mpmath import mp, mpf, sqrt, pi, fib
import numpy as np

mp.dps = 100

def lucas(n):
    PHI = (1 + sqrt(5)) / 2
    return int(round(float(PHI**n + (-1/PHI)**n)))

# Constants
PHI = (1 + sqrt(5)) / 2
PHI_INV = 1 / PHI
DELTA = mpf('4.6692016091029906718532038204662016172581855774757686327456513430')
ALPHA = mpf('2.5029078750958928222839028732182157863812713767271499773361920567')
R_INF = mpf('3.5699456718709449018420051513864989367638369115148323781388011418')

F = [int(fib(n)) for n in range(25)]
L = [lucas(n) for n in range(25)]

print("=" * 70)
print("EXPERIMENT 21: Derive δ from First Principles")
print("=" * 70)

# ============================================================
# APPROACH 1: Self-consistency equation
# ============================================================
print("\n### APPROACH 1: Self-Consistency Equation")

print("""
We have:
  r∞ = π × M₁₀(z)  where M₁₀ = [[89, 55], [55, 34]]
  z = -1/φ + Δz
  Δz = π / (1857π + 4(δ-4))

The bifurcation points satisfy:
  r_{n+1} - r_n ≈ (r_∞ - r_n) / δ

At the limit, r∞ is the fixed point of the renormalization operator.
""")

# If we knew r∞ independently, we could solve for δ
# r∞ is defined as the accumulation point of bifurcations

# The self-consistency: r∞ depends on δ (via convergence), δ depends on r∞
# But our formula gives r∞ in terms of δ... can we invert?

# ============================================================
# APPROACH 2: Invert the Δz formula
# ============================================================
print("\n### APPROACH 2: Invert the Δz Formula")

print("""
Given:
  Δz = π / (1857π + 4(δ-4))
  
Solve for δ:
  Δz × (1857π + 4(δ-4)) = π
  1857π×Δz + 4(δ-4)×Δz = π
  4(δ-4)×Δz = π - 1857π×Δz
  4(δ-4)×Δz = π(1 - 1857×Δz)
  δ - 4 = π(1 - 1857×Δz) / (4×Δz)
  δ = 4 + π(1 - 1857×Δz) / (4×Δz)
""")

# The exact Δz
target = R_INF / pi
z_exact = (34 * target - 55) / (89 - 55 * target)
delta_z = z_exact - (-PHI_INV)

# Check the formula
delta_calc = 4 + pi * (1 - 1857 * delta_z) / (4 * delta_z)
print(f"Calculated δ = {delta_calc}")
print(f"Known δ     = {DELTA}")
print(f"Error       = {abs(delta_calc - DELTA)}")

# ============================================================
# APPROACH 3: Express Δz without δ
# ============================================================
print("\n### APPROACH 3: Express Δz Without δ")

print("""
We know:
  z_exact = (34 × r∞/π - 55) / (89 - 55 × r∞/π)
  Δz = z_exact - (-1/φ)

If we could express r∞ using only Fibonacci/π...

Let's try: r∞ ≈ 3 + something
""")

# r∞ - 3 = 0.5699456718709449...
diff_from_3 = R_INF - 3
print(f"\nr∞ - 3 = {diff_from_3}")
print(f"(r∞ - 3) / φ = {diff_from_3 / PHI}")
print(f"(r∞ - 3) × φ = {diff_from_3 * PHI}")
print(f"(r∞ - 3) / (1/φ) = {diff_from_3 / PHI_INV}")
print(f"(r∞ - 3) × π = {diff_from_3 * pi}")

# Is (r∞ - 3) related to something simple?
print(f"\n(r∞ - 3) / (φ - 1) = {diff_from_3 / (PHI - 1)}")  # φ-1 = 1/φ
print(f"(r∞ - 3) - 1/φ = {diff_from_3 - PHI_INV}")

# Hmm, (r∞ - 3) ≈ 0.57, and 1/φ ≈ 0.618
# The difference is about -0.048

# ============================================================
# APPROACH 4: δ as a function of Möbius fixed point geometry
# ============================================================
print("\n### APPROACH 4: Geometric Derivation")

print("""
The Möbius transformation M₁₀ has fixed points at φ and -1/φ.

The seed z = -1/φ + Δz is perturbed from the unstable fixed point.

Hypothesis: δ measures the "expansion rate" at the fixed point.

M'₁₀(-1/φ) = 1 / (55×(-1/φ) + 34)² = 1 / (34 - 55/φ)²
""")

# Derivative at fixed point
denom = 55 * (-PHI_INV) + 34
M_prime = 1 / denom**2

print(f"\nM'₁₀(-1/φ) = {M_prime}")
print(f"√(M'₁₀) = {sqrt(M_prime)}")
print(f"M'₁₀ / δ = {M_prime / DELTA}")
print(f"M'₁₀ / δ² = {M_prime / DELTA**2}")
print(f"√(M'₁₀) / δ = {sqrt(M_prime) / DELTA}")

# The denominator 34 - 55/φ = 34 - 34 = 0? No wait...
# 55/φ = 55 × (φ-1) = 55φ - 55 = 89 - 55 = 34. So denom = 0!
# Actually -1/φ IS the fixed point, so M(-1/φ) = -1/φ
# The derivative formula needs care

print(f"\n55 × (-1/φ) + 34 = {55 * (-PHI_INV) + 34}")
print("(This is very close to zero - near the fixed point!)")

# ============================================================
# APPROACH 5: δ from the recursion relation
# ============================================================
print("\n### APPROACH 5: δ from Recursion")

print("""
Key insight: δ is the ratio of successive "widths" in parameter space:
  δ = lim (r_n - r_{n-1}) / (r_{n+1} - r_n)

In Möbius terms, this might relate to the contraction ratio.
""")

# The Möbius contraction at each level
# For Fibonacci Möbius, the trace is L_n (Lucas)
print(f"\nFibonacci Möbius traces (= Lucas numbers):")
for n in range(5, 15):
    Fn1 = F[n+1]
    Fn_1 = F[n-1]
    trace = Fn1 + Fn_1
    print(f"  trace(M_{n}) = F_{n+1} + F_{n-1} = {Fn1} + {Fn_1} = {trace} = L_{n}")

print(f"\nRatio of successive traces:")
for n in range(6, 14):
    ratio = L[n] / L[n-1]
    print(f"  L_{n}/L_{n-1} = {L[n]}/{L[n-1]} = {ratio:.6f}")

print(f"\nThis converges to φ = {float(PHI):.6f}, not δ!")

# ============================================================
# APPROACH 6: The exact relation
# ============================================================
print("\n### APPROACH 6: Exact Bootstrap")

print("""
We have an exact identity:
  r∞ = π × (89z + 55) / (55z + 34)  where z = -1/φ + Δz

And:
  Δz = π / (1857π + 4(δ-4))

Can we find a SECOND equation relating r∞ and δ?
""")

# From universality theory:
# r_n ≈ r∞ - c × δ^{-n}
# So r∞ = r_n + c × δ^{-n}

# At n=1 (first bifurcation): r_1 = 3
# r∞ = 3 + c₁ × δ^{-1}
# c₁ = (r∞ - 3) × δ

c1 = (R_INF - 3) * DELTA
print(f"\nc₁ = (r∞ - 3) × δ = {c1}")

# At n=2: r_2 = 3.449...
r2 = mpf('3.44948974278')
c2 = (R_INF - r2) * DELTA**2
print(f"c₂ = (r∞ - r₂) × δ² = {c2}")

# If the scaling is exact, c₁ = c₂
print(f"\nc₁/c₂ = {c1/c2}")
print("(Should be ~1 for perfect scaling)")

# ============================================================
# APPROACH 7: Pure Fibonacci formula for δ
# ============================================================
print("\n### APPROACH 7: Fibonacci Formula Search")

print("\nSearching for δ ≈ f(F_n, L_n, π, φ)...")

candidates = [
    ("4 + 2/3", 4 + mpf(2)/3),
    ("4 + (φ-1)", 4 + (PHI-1)),
    ("4 + 1/φ", 4 + PHI_INV),
    ("4 + π/5", 4 + pi/5),
    ("4 + 2/π", 4 + 2/pi),
    ("4 + L₃/5", 4 + mpf(L[3])/5),
    ("3 + φ", 3 + PHI),
    ("π + 1", pi + 1),
    ("2π - 1.6", 2*pi - mpf('1.6')),
    ("φ³", PHI**3),
    ("φ² + φ", PHI**2 + PHI),
    ("1 + φ + φ²", 1 + PHI + PHI**2),
    ("L₅/√5 + 3", L[5]/sqrt(5) + 3),
    ("F₇/F₄ + 3", mpf(F[7])/F[4] + 3),
]

print(f"\n{'Formula':<25} {'Value':<15} {'Error':<15}")
print("-" * 55)
for name, val in candidates:
    err = abs(val - DELTA)
    print(f"{name:<25} {float(val):<15.8f} {float(err):<15.2e}")

# ============================================================
# APPROACH 8: The constraint equation
# ============================================================
print("\n### APPROACH 8: The Constraint Equation")

print("""
KEY INSIGHT:

We have two expressions for r∞:
1. r∞ = π × M₁₀(z)  [Möbius representation]
2. r∞ = 3 + c/δ + O(1/δ²)  [Universality theory]

And z depends on δ via:
  z = -1/φ + π/(1857π + 4(δ-4))

This gives a self-consistent equation for δ!
""")

# Let's write: r∞(δ) = π × M₁₀(-1/φ + π/(1857π + 4(δ-4)))
# And set this equal to r∞ from bifurcation theory

def r_inf_from_delta(d):
    """Compute r∞ given δ using our formula."""
    delta_z = pi / (1857*pi + 4*(d - 4))
    z = -PHI_INV + delta_z
    return pi * (89*z + 55) / (55*z + 34)

print("\nChecking self-consistency:")
for d_test in [4.5, 4.6, 4.669, 4.6692, 4.66920]:
    r_test = r_inf_from_delta(mpf(str(d_test)))
    print(f"  δ = {d_test}: r∞ = {float(r_test):.10f}")

print(f"\nActual r∞ = {float(R_INF):.10f}")

# Newton's method to find δ that gives the right r∞
print("\n--- Newton's Method to Derive δ ---")
# We need ANOTHER constraint. The Möbius formula alone doesn't pin down δ.

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Deriving δ")
print("=" * 70)
print("""
FINDING: We CANNOT derive δ from our formula alone!

The formula r∞ = π × M₁₀(z) with z = -1/φ + π/(1857π + 4(δ-4))
is a REPRESENTATION of r∞ in terms of δ, not a derivation.

To derive δ, we need additional input:
1. The logistic map itself (x → rx(1-x))
2. Renormalization group theory
3. The period-doubling cascade dynamics

HOWEVER, the formula DOES predict:
- If we measure r∞ experimentally, we can compute δ
- The structure (Fibonacci Möbius + perturbation) is meaningful
- δ ≈ 4 + (something involving φ and π)

BEST CANDIDATE from search:
  δ ≈ 3 + φ = 4.618... (error ~0.05)
  δ ≈ 4 + 2/3 = 4.667... (error ~0.002)
  δ ≈ 4 + 1/φ² = 4.382... (error ~0.29)

None of these are exact. δ appears to be a TRANSCENDENTAL number
that cannot be expressed in closed form using elementary constants.
""")
