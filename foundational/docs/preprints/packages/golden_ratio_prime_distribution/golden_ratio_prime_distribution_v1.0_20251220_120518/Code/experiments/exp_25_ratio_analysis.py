"""
Experiment 25: Is the Run-Length Ratio Related to ξ?
=====================================================

Hypothesis: The run-length ratio L+/L- might be expressible as a 
function of ξ (the 2-component amplitude) or other fundamental quantities.

Key quantities:
- ξ = 1/(2k) = 1/18 ≈ 0.0556 (2-component amplitude)
- π_odd ≈ 0.19 (prime density on odds)
- I_prime ≈ 0.166
- I_composite ≈ 0.029
- λ = 0.99
"""

import numpy as np
from fractions import Fraction

# Fundamental quantities
k = 9  # factor base size
xi = 1 / (2 * k)  # = 1/18
lambda_ = 0.99
phi = (1 + np.sqrt(5)) / 2

# Observed quantities
L_plus = 2.9477
L_minus = 1.8402
run_ratio = L_plus / L_minus  # ≈ 1.60

I_prime = 0.166
I_composite = 0.029
prime_kick = I_prime - I_composite  # ≈ 0.137

pi_odd = 0.1918  # prime density on odds (for n up to 100k)

print("=" * 70)
print("EXPERIMENT 25: RATIO ANALYSIS")
print("=" * 70)
print()

print("Fundamental quantities:")
print(f"  k (factor base size) = {k}")
print(f"  ξ = 1/(2k) = 1/{2*k} = {xi:.6f}")
print(f"  λ = {lambda_}")
print(f"  φ = {phi:.6f}")
print()

print("Observed quantities:")
print(f"  L+ = {L_plus:.4f}")
print(f"  L- = {L_minus:.4f}")
print(f"  Run ratio L+/L- = {run_ratio:.4f}")
print(f"  I_prime = {I_prime:.4f}")
print(f"  I_composite = {I_composite:.4f}")
print(f"  Prime kick = {prime_kick:.4f}")
print(f"  π_odd (prime density) = {pi_odd:.4f}")
print()

print("=" * 70)
print("HYPOTHESIS TESTING: Is run_ratio a function of ξ?")
print("=" * 70)
print()

# Test various relationships
print("Simple ratios involving ξ:")
print(f"  1/ξ = {1/xi:.4f}")
print(f"  1/(2ξ) = {1/(2*xi):.4f} = k = {k}")
print(f"  ξ * k = {xi * k:.4f}")
print()

print("Ratios involving φ and ξ:")
print(f"  φ = {phi:.4f}")
print(f"  φ * ξ = {phi * xi:.6f}")
print(f"  φ / ξ = {phi / xi:.4f}")
print(f"  φ * (1-ξ) = {phi * (1-xi):.4f}")
print(f"  φ / (1+ξ) = {phi / (1+xi):.4f}")
print()

# What if run_ratio = φ * f(ξ)?
# run_ratio ≈ 1.60, φ ≈ 1.618
# So f(ξ) ≈ 1.60/1.618 ≈ 0.989
correction_factor = run_ratio / phi
print(f"If run_ratio = φ * f(ξ), then f(ξ) = {correction_factor:.6f}")
print(f"  Compare to: 1 - ξ = {1 - xi:.6f}")
print(f"  Compare to: 1 - 2ξ = {1 - 2*xi:.6f}")
print(f"  Compare to: λ = {lambda_:.6f}")
print()

# What if φ emerges from a ratio involving ξ?
print("Searching for φ in ratios:")
print()

# The key relationship: on odds, I_mean ≈ ξ = 1/(2k)
# What produces φ from this?

# Test: (1 + something) / something = φ?
# This is the defining property: φ = 1 + 1/φ

print("Key observation: L+/L- defines frac = L+/(L+ + L-)")
print(f"  frac = {L_plus/(L_plus + L_minus):.6f}")
print(f"  1/φ = {1/phi:.6f}")
print()

# If frac = 1/φ, then:
# L+/(L+ + L-) = 1/φ
# φ*L+ = L+ + L-
# (φ-1)*L+ = L-
# L+/L- = 1/(φ-1) = φ (since φ-1 = 1/φ)
print("Mathematical identity:")
print(f"  If frac = 1/φ, then L+/L- = 1/(φ-1) = φ")
print(f"  Because φ - 1 = 1/φ (defining property of φ)")
print(f"  So L+/L- = φ is EQUIVALENT to frac = 1/φ")
print()

# Now the question: why does prime structure give frac = 1/φ?
print("=" * 70)
print("THE REAL QUESTION: Why does prime structure give frac = 1/φ?")
print("=" * 70)
print()

# What quantities could combine to give φ?
print("Candidate relationships:")
print()

# 1. Prime density and kick
print("1. Prime injection parameters:")
print(f"   Prime rate on odds: π = {pi_odd:.4f}")
print(f"   Prime kick: Δ = {prime_kick:.4f}")
print(f"   Product π*Δ = {pi_odd * prime_kick:.6f}")
print(f"   Compare to ξ = {xi:.6f}")
print(f"   Compare to 1-λ = {1-lambda_:.6f}")
print()

# 2. The crossing asymmetry
# At positive transitions: prime rate ≈ 0.367
# At negative transitions: prime rate ≈ 0.046
prime_at_pos = 0.367
prime_at_neg = 0.046
print("2. Transition asymmetry:")
print(f"   Prime rate at + transitions: {prime_at_pos:.3f}")
print(f"   Prime rate at - transitions: {prime_at_neg:.3f}")
print(f"   Ratio: {prime_at_pos/prime_at_neg:.2f}")
print(f"   Ratio to overall: {prime_at_pos/pi_odd:.2f} (at +), {prime_at_neg/pi_odd:.2f} (at -)")
print()

# 3. Could φ come from log(prime density)?
print("3. Logarithmic relationships:")
print(f"   log(1/π_odd) = {np.log(1/pi_odd):.4f}")
print(f"   log(φ) = {np.log(phi):.4f}")
print(f"   1/log(φ) = {1/np.log(phi):.4f}")
print()

# 4. What if it's about the effective decay?
print("4. Effective dynamics:")
effective_decay = 1 - lambda_
time_scale = 1 / effective_decay  # E-folding time
print(f"   Decay rate: 1-λ = {effective_decay:.4f}")
print(f"   E-folding time: {time_scale:.1f} steps")
print(f"   Prime gap on odds: ~{1/pi_odd:.1f} steps")
print(f"   Ratio: {time_scale * pi_odd:.2f}")
print()

# 5. What if φ = (1 + √5)/2 relates to the quadratic nature of divisibility?
print("5. Quadratic structure:")
print(f"   φ satisfies: x² - x - 1 = 0")
print(f"   Or: x = 1 + 1/x")
print(f"   This is self-referential, like...")
print(f"   ...the relationship between E and I?")
print()

# Let's compute what ratio of I_prime to I_composite would give exactly φ
# frac = 1/φ means L+/L- = φ
print("=" * 70)
print("INVESTIGATING: What parameters would give EXACT φ?")
print("=" * 70)
print()

# The observed run ratio is 1.60, but φ = 1.618
# The gap is about 1%
gap = phi - run_ratio
print(f"Gap from φ: {gap:.4f} ({100*gap/phi:.2f}%)")
print()

# If we had more data (larger N), would it converge to φ?
print("Convergence question:")
print("  At N=100k, run_ratio = 1.60")
print("  Need to test larger N to see if it converges to φ")
print()

# What's the standard error on the run ratio?
n_runs = 10443  # number of runs
se_L_plus = L_plus / np.sqrt(n_runs)  # rough estimate
se_L_minus = L_minus / np.sqrt(n_runs)
# Delta method for ratio
se_ratio = run_ratio * np.sqrt((se_L_plus/L_plus)**2 + (se_L_minus/L_minus)**2)
print(f"Rough standard error on ratio: ±{se_ratio:.3f}")
print(f"95% CI: [{run_ratio - 2*se_ratio:.3f}, {run_ratio + 2*se_ratio:.3f}]")
print(f"φ = {phi:.3f} is {'within' if abs(phi - run_ratio) < 2*se_ratio else 'outside'} 95% CI")
print()

# ===================================================================
# KEY INSIGHT SECTION
# ===================================================================
print("=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print()

print("""
The run ratio L+/L- ≈ 1.60 vs φ ≈ 1.618.

The question is: is this φ, or just "approximately 1.6"?

Possible interpretations:

A) EXACT φ (within measurement error)
   - Need larger N to verify convergence
   - Would suggest deep connection to prime structure

B) APPROXIMATE φ (coincidence)
   - Just happens to be near 1.6
   - No special significance

C) RELATED TO φ (through ξ or other parameters)
   - run_ratio = φ * (1 - ε) where ε depends on parameters
   - The correction might vanish as N → ∞ or k → ∞

The fact that frac = 1/φ IMPLIES L+/L- = φ (by the math identity)
means the question reduces to: why is frac = 1/φ?

And that question might relate to ξ = 1/(2k) in some way we
haven't identified yet.
""")

# Final comparison
print()
print("=" * 70)
print("SUMMARY OF RELEVANT RATIOS")
print("=" * 70)
print()
print(f"φ = {phi:.6f}")
print(f"1/φ = {1/phi:.6f}")
print(f"φ-1 = 1/φ = {phi-1:.6f}")
print(f"")
print(f"ξ = 1/(2k) = {xi:.6f}")
print(f"k*ξ = {k*xi:.6f}")
print(f"1/ξ = {1/xi:.6f}")
print(f"")
print(f"run_ratio = {run_ratio:.6f}")
print(f"frac(E>0) = {L_plus/(L_plus+L_minus):.6f}")
print(f"")
print(f"φ * (1-ξ) = {phi*(1-xi):.6f}  ← close to run_ratio?")
print(f"φ * (1-1/k) = {phi*(1-1/k):.6f}  ← close to run_ratio?")
print(f"φ * λ = {phi*lambda_:.6f}  ← close to run_ratio?")
