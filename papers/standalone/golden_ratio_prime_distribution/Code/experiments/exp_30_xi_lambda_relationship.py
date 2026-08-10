"""
Experiment 30: Is there a ξ-λ relationship?
==========================================

Observation: 
- ξ = 1/(2k) = 1/18 ≈ 0.0556 (the 2-component amplitude)
- λ* = 0.9816 (the critical λ)
- 1 - λ* = 0.0184

Is there a relationship between ξ and (1-λ*)?
"""

import numpy as np

# The key quantities
k = 9  # factor base size
xi = 1 / (2 * k)  # = 1/18 ≈ 0.0556
lambda_star = 0.9816  # critical λ from exp_28
one_minus_lambda = 1 - lambda_star

PHI = (1 + np.sqrt(5)) / 2

print("=" * 60)
print("EXPERIMENT 30: THE ξ-λ RELATIONSHIP")
print("=" * 60)
print()

print("Key quantities:")
print(f"  k = {k}")
print(f"  ξ = 1/(2k) = {xi:.6f}")
print(f"  λ* = {lambda_star:.6f}")
print(f"  1 - λ* = {one_minus_lambda:.6f}")
print()

print("=" * 60)
print("TESTING RELATIONSHIPS")
print("=" * 60)
print()

# Direct comparison
print("Direct comparison:")
print(f"  ξ = {xi:.6f}")
print(f"  1 - λ* = {one_minus_lambda:.6f}")
print(f"  Ratio ξ / (1-λ*) = {xi / one_minus_lambda:.4f}")
print()

# Is ξ ≈ 3(1-λ*)?
print("Is ξ ≈ c(1-λ*) for some constant c?")
c = xi / one_minus_lambda
print(f"  c = ξ / (1-λ*) = {c:.4f}")
print(f"  Compare to: 3 = {3:.4f}")
print(f"  Compare to: φ² = {PHI**2:.4f}")
print(f"  Compare to: e = {np.e:.4f}")
print()

# Is 1-λ* ≈ ξ/φ?
print("Is 1-λ* ≈ ξ/φ?")
print(f"  ξ/φ = {xi/PHI:.6f}")
print(f"  1-λ* = {one_minus_lambda:.6f}")
print(f"  Ratio = {one_minus_lambda / (xi/PHI):.4f}")
print()

# Is 1-λ* ≈ 1/k × something?
print("Is 1-λ* related to 1/k?")
print(f"  1/k = {1/k:.6f}")
print(f"  1-λ* = {one_minus_lambda:.6f}")
print(f"  (1-λ*) / (1/k) = {one_minus_lambda * k:.4f}")
print(f"  (1-λ*) × k = {one_minus_lambda * k:.6f}")
print()

# What about ξ × (1-λ*)?
print("Product ξ × (1-λ*):")
product = xi * one_minus_lambda
print(f"  ξ × (1-λ*) = {product:.6f}")
print(f"  Compare to 1/k² = {1/k**2:.6f}")
print(f"  Compare to 1/(4k²) = {1/(4*k**2):.6f}")
print()

# The window relationship
window = 101
print("Window relationship:")
print(f"  Window = {window}")
print(f"  1/window = {1/window:.6f}")
print(f"  (1-λ*) / (1/window) = {one_minus_lambda * window:.4f}")
print(f"  So 1-λ* ≈ {one_minus_lambda * window:.2f}/window")
print()

# Is there a ξ-window-λ relationship?
print("=" * 60)
print("THREE-WAY RELATIONSHIP")
print("=" * 60)
print()

print("ξ × window:")
print(f"  ξ × window = {xi * window:.4f}")
print(f"  Compare to k = {k}")
print()

print("(1-λ*) × window:")
print(f"  (1-λ*) × window = {one_minus_lambda * window:.4f}")
print()

print("If λ* is tuned so that 1-λ* ∝ 1/window, then:")
print(f"  E-folding time = 1/(1-λ*) = {1/one_minus_lambda:.1f}")
print(f"  This is about {1/one_minus_lambda/window:.2f}× the window")
print()

# Key insight check
print("=" * 60)
print("KEY INSIGHT CHECK")
print("=" * 60)
print()

# The I_mean on odds is approximately ξ
I_mean_odd = 0.055  # from our experiments
print(f"I_mean on odds ≈ {I_mean_odd:.4f}")
print(f"ξ = 1/(2k) = {xi:.4f}")
print(f"Match: {abs(I_mean_odd - xi) < 0.001}")
print()

# At critical λ*, the system balances...
# The decay rate (1-λ*) vs the drift rate (ξ)
print("Balance hypothesis:")
print(f"  Drift rate (ξ) = {xi:.6f}")
print(f"  Decay rate (1-λ*) = {one_minus_lambda:.6f}")
print(f"  Ratio drift/decay = {xi/one_minus_lambda:.4f}")
print()

# If drift/decay = φ, that would be interesting
print(f"Is drift/decay = φ?")
print(f"  ξ/(1-λ*) = {xi/one_minus_lambda:.4f}")
print(f"  φ = {PHI:.4f}")
print(f"  φ² = {PHI**2:.4f}")
print()

# What if (1-λ*) = ξ/φ²?
print("Testing (1-λ*) = ξ/φ²:")
predicted_one_minus_lambda = xi / PHI**2
print(f"  Predicted 1-λ* = ξ/φ² = {predicted_one_minus_lambda:.6f}")
print(f"  Actual 1-λ* = {one_minus_lambda:.6f}")
print(f"  Error = {abs(predicted_one_minus_lambda - one_minus_lambda):.6f}")
print()

# What if (1-λ*) = ξ/3?
print("Testing (1-λ*) = ξ/3:")
predicted_1 = xi / 3
print(f"  Predicted 1-λ* = ξ/3 = {predicted_1:.6f}")
print(f"  Actual 1-λ* = {one_minus_lambda:.6f}")
print(f"  Error = {abs(predicted_1 - one_minus_lambda):.6f}")
print()

# What if λ* = 1 - 2/(k×window)?
print("Testing λ* = 1 - c/(k×window) for various c:")
for c_test in [1, 2, 3, np.e, PHI, PHI**2]:
    pred = 1 - c_test / (k * window)
    err = abs(pred - lambda_star)
    print(f"  c={c_test:.3f}: λ* = {pred:.6f}, error = {err:.6f}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

print(f"""
Key observations:

1. ξ = 1/(2k) = {xi:.4f} (the 2-component bias on odds)
2. 1-λ* = {one_minus_lambda:.4f} (the critical decay rate)
3. Ratio ξ/(1-λ*) = {xi/one_minus_lambda:.2f} ≈ 3

This means:
- The drift rate (ξ) is about 3× the decay rate (1-λ*)
- At criticality, accumulated drift over ~3 decay times balances
- This might explain why the system finds φ at this balance point

The relationship:
  1 - λ* ≈ ξ/3 = 1/(6k)

If this holds, then λ* is determined by k alone:
  λ* = 1 - 1/(6k) = 1 - 1/{6*k} = {1 - 1/(6*k):.6f}
  (Actual λ* = {lambda_star:.6f})
""")
