"""
Experiment 11: Identify the Cross-Ratio Limit

The cross-ratio of bifurcation points converges to ~1.1699...
What is this number? Does it have closed form in terms of δ, φ, or π?
"""

from mpmath import mp, mpf, sqrt, pi, phi, log, e
mp.dps = 50

# Observed cross-ratio limit
CR = mpf('1.1699484686990637087650493725228734171933803793274')

# Feigenbaum constants
delta = mpf('4.6692016091029906718532038204662016172581855774757686327456513430041343302113')
alpha = mpf('2.5029078750958928222839028732182157863812713767271499773361920567792354')

# Key integers
F10 = 55
F8 = 21
L5 = 11

print("=" * 60)
print("CROSS-RATIO LIMIT IDENTITY SEARCH")
print("=" * 60)
print(f"\nCR_limit = {CR}")
print(f"CR - 1   = {CR - 1}")

# Test various relationships
print("\n" + "-" * 60)
print("TESTING δ RELATIONSHIPS")
print("-" * 60)

print(f"\n1 + 1/δ = {1 + 1/delta}")
print(f"  diff: {CR - (1 + 1/delta)}")

print(f"\n(δ-1)/(δ+1) = {(delta-1)/(delta+1)}")
print(f"  diff: {CR - (delta-1)/(delta+1)}")

print(f"\n1 + 1/δ² = {1 + 1/delta**2}")
print(f"  diff: {CR - (1 + 1/delta**2)}")

print(f"\nδ/(δ+1) = {delta/(delta+1)}")
print(f"  × 2: {2*delta/(delta+1)}")
print(f"  diff from 2×: {CR - 2*delta/(delta+1)}")

print(f"\n(δ+1)/(δ-1) × (1/δ) = {(delta+1)/(delta-1) * (1/delta)}")

# δ - 3 = 1.669... and CR - 1 = 0.169...
print(f"\n(δ-3)/10 = {(delta-3)/10}")
print(f"  diff from CR-1: {(CR-1) - (delta-3)/10}")

print(f"\nδ/4 = {delta/4}")
print(f"  diff: {CR - delta/4}")

print("\n" + "-" * 60)
print("TESTING φ RELATIONSHIPS")
print("-" * 60)

print(f"\n1 + 1/φ² = {1 + 1/phi**2}")
print(f"  diff: {CR - (1 + 1/phi**2)}")

print(f"\n1/φ² = {1/phi**2}")
print(f"CR-1 = {CR-1}")
print(f"  diff: {(CR-1) - 1/phi**2}")

print(f"\nφ-1 = 1/φ = {phi-1}")
print(f"(CR-1)/(φ-1) = {(CR-1)/(phi-1)}")

print(f"\n2-φ = {2-phi}")
print(f"  diff: {(CR-1) - (2-phi)}")

print(f"\nφ/√5 = {phi/sqrt(5)}")
print(f"  diff: {CR - phi/sqrt(5)}")

print("\n" + "-" * 60)
print("TESTING COMBINED δ-φ RELATIONSHIPS")
print("-" * 60)

print(f"\n1 + 1/(δφ) = {1 + 1/(delta*phi)}")
print(f"  diff: {CR - (1 + 1/(delta*phi))}")

print(f"\n1 + φ/δ² = {1 + phi/delta**2}")
print(f"  diff: {CR - (1 + phi/delta**2)}")

print(f"\n1 + (δ-3)/10 = {1 + (delta-3)/10}")
print(f"  diff: {CR - (1 + (delta-3)/10)}")

# The key discovery from before: δ uses 55, 17, etc.
# Maybe CR does too?

print("\n" + "-" * 60)
print("TESTING INTEGER RELATIONSHIPS")
print("-" * 60)

print(f"\n1 + 1/6 = {mpf(1) + mpf(1)/6}")
print(f"  diff: {CR - (1 + mpf(1)/6)}")

print(f"\n7/6 = {mpf(7)/6}")
print(f"  diff: {CR - mpf(7)/6}")

print(f"\n1 + 9/55 = {1 + mpf(9)/55}")
print(f"  diff: {CR - (1 + mpf(9)/55)}")

print(f"\n1 + 17/100 = {1 + mpf(17)/100}")
print(f"  diff: {CR - (1 + mpf(17)/100)}")

# Try continued fraction
print("\n" + "-" * 60)
print("CONTINUED FRACTION OF CR")
print("-" * 60)

def to_continued_fraction(x, n=15):
    """Extract continued fraction coefficients."""
    cf = []
    for _ in range(n):
        a = int(x)
        cf.append(a)
        x = x - a
        if x < mpf('1e-30'):
            break
        x = 1/x
    return cf

cf = to_continued_fraction(CR)
print(f"CR = {cf}")

cf_minus_1 = to_continued_fraction(CR - 1)
print(f"CR-1 = {cf_minus_1}")

# Try 1/(CR-1)
print(f"\n1/(CR-1) = {1/(CR-1)}")
cf_inv = to_continued_fraction(1/(CR-1))
print(f"CF of 1/(CR-1) = {cf_inv}")

# Maybe related to log?
print("\n" + "-" * 60)
print("TESTING LOGARITHMIC RELATIONSHIPS")
print("-" * 60)

print(f"\nln(δ)/ln(φ) = {log(delta)/log(phi)}")
print(f"  diff: {CR - log(delta)/log(phi)}")

print(f"\nln(δ)/π = {log(delta)/pi}")
print(f"  diff: {CR - log(delta)/pi}")

print(f"\nδ^(1/10) = {delta**(mpf(1)/10)}")
print(f"  diff: {CR - delta**(mpf(1)/10)}")

print(f"\ne^(CR-1) = {e**(CR-1)}")
print(f"  δ/4? diff: {e**(CR-1) - delta/4}")

# Key insight: CR converges to ~1.17, which might be (δ-3)/10 + 1
# δ - 3 = 1.6692...
# (δ-3)/10 = 0.16692...
# CR - 1 = 0.16994...

print("\n" + "-" * 60)
print("REFINED SEARCH AROUND (δ-3)/10")
print("-" * 60)

base = (delta - 3) / 10
residual = (CR - 1) - base
print(f"\n(δ-3)/10 = {base}")
print(f"CR-1 = {CR-1}")
print(f"residual = {residual}")
print(f"residual × 1000 = {residual * 1000}")
print(f"residual × δ = {residual * delta}")
print(f"residual / (1/55) = {residual / (mpf(1)/55)}")
print(f"residual × 55 = {residual * 55}")

# So CR ≈ 1 + (δ-3)/10 + 0.003/55?
correction = mpf('0.166')  
print(f"\n0.166 + 1 = {1 + correction}")
print(f"  diff: {CR - (1 + correction)}")

# Try (δ-3)/10 + correction term with Fibonacci
approx = 1 + (delta - 3)/10 + mpf(1)/330
print(f"\n1 + (δ-3)/10 + 1/330 = {approx}")
print(f"  diff: {CR - approx}")

approx2 = 1 + (delta - 3)/10 + mpf(1)/300
print(f"\n1 + (δ-3)/10 + 1/300 = {approx2}")
print(f"  diff: {CR - approx2}")

# What if it's exactly 1 + (δ-3)/10 + 1/(10δ)?
approx3 = 1 + (delta-3)/10 + 1/(10*delta)
print(f"\n1 + (δ-3)/10 + 1/(10δ) = {approx3}")
print(f"  diff: {CR - approx3}")

# Simpler: maybe it's just 1 + (δ-3)/(δ+17)?
approx4 = 1 + (delta-3)/(delta+17)
print(f"\n1 + (δ-3)/(δ+17) = {approx4}")
print(f"  diff: {CR - approx4}")

# Or 1 + δ/(δ² - 1)?
approx5 = 1 + delta/(delta**2 - 1)
print(f"\n1 + δ/(δ²-1) = {approx5}")
print(f"  diff: {CR - approx5}")

print("\n" + "=" * 60)
print("BEST CANDIDATE SO FAR")
print("=" * 60)

# Find the best
candidates = [
    ("1 + (δ-3)/10", 1 + (delta-3)/10),
    ("1 + (δ-3)/(δ+17)", 1 + (delta-3)/(delta+17)),
    ("1 + δ/(δ²-1)", 1 + delta/(delta**2 - 1)),
    ("1 + 1/δ", 1 + 1/delta),
]

for name, val in candidates:
    diff = abs(CR - val)
    print(f"{name}: diff = {diff:.6e}")
