"""
PAC Tree vs SEC Dark Matter Parameters
======================================

The SEC dark matter simulation uses α = 0.005857 (empirically tuned).
Can we derive this from the PAC tree?

If yes, this validates PAC structure in an INDEPENDENT cosmological context.
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2

def fib(n):
    if n <= 0: return 0
    if n <= 2: return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

print("=" * 70)
print("PAC TREE vs SEC DARK MATTER PARAMETERS")
print("=" * 70)

# The empirically tuned SEC dark matter parameter
alpha_SEC = 0.005857
xi_SEC = 1.0571  # balance operator

# Our PAC visible sector alpha
alpha_visible = 0.0072973526

print(f"\n   SEC dark matter α = {alpha_SEC}")
print(f"   PAC visible α     = {alpha_visible}")
print(f"   Ratio: α_SEC/α_visible = {alpha_SEC/alpha_visible:.6f}")

# =============================================================================
# Part 1: Try Simple Fibonacci Ratios
# =============================================================================
print("\n" + "=" * 70)
print("1. SEARCHING FOR FIBONACCI EXPRESSION")
print("=" * 70)

target_ratio = alpha_SEC / alpha_visible
print(f"\n   Target ratio: {target_ratio:.6f}")

print("\n   Trying Fibonacci ratios:")
candidates = []

# Single ratios
for n1 in range(1, 15):
    for n2 in range(1, 15):
        ratio = fib(n1) / fib(n2)
        err = abs(ratio - target_ratio) / target_ratio
        if err < 0.15:
            candidates.append((f"F_{n1}/F_{n2}", ratio, err))

# With phi powers
for n1 in range(1, 12):
    for n2 in range(1, 12):
        for p in range(-3, 4):
            if p == 0:
                continue
            ratio = fib(n1) / fib(n2) * PHI**p
            err = abs(ratio - target_ratio) / target_ratio
            if err < 0.10:
                candidates.append((f"F_{n1}/F_{n2} × φ^{p}", ratio, err))

# Differences and sums
for n1 in range(1, 12):
    for n2 in range(1, 12):
        if fib(n1) > fib(n2):
            ratio = (fib(n1) - fib(n2)) / fib(n1)
            err = abs(ratio - target_ratio) / target_ratio
            if err < 0.10:
                candidates.append((f"(F_{n1}-F_{n2})/F_{n1}", ratio, err))

# 1 - F_a/F_b forms
for n1 in range(1, 12):
    for n2 in range(n1+1, 15):
        ratio = 1 - fib(n1)/fib(n2)
        err = abs(ratio - target_ratio) / target_ratio
        if err < 0.10:
            candidates.append((f"1 - F_{n1}/F_{n2}", ratio, err))

# Sort by error
candidates.sort(key=lambda x: x[2])
print("\n   Best matches:")
for name, val, err in candidates[:10]:
    print(f"   {name:25s} = {val:.6f} (error: {err*100:.2f}%)")

# =============================================================================
# Part 2: Physical Interpretation
# =============================================================================
print("\n" + "=" * 70)
print("2. PHYSICAL INTERPRETATION")
print("=" * 70)

# Best candidate analysis
best = candidates[0] if candidates else None

print(f"""
   The SEC dark matter α = 0.005857 appears to be:
   
   α_dark ≈ α_visible × {target_ratio:.4f}
   
   Best Fibonacci matches:
""")

for name, val, err in candidates[:5]:
    alpha_pred = alpha_visible * val
    print(f"   {name}: α_dark = {alpha_pred:.6f} (vs {alpha_SEC}, err {err*100:.2f}%)")

# =============================================================================
# Part 3: Tree Branch Interpretation
# =============================================================================
print("\n" + "=" * 70)
print("3. TREE BRANCH INTERPRETATION")
print("=" * 70)

print("""
   PAC Tree at F_7 = 13:
   
                    13 (root)
                   /        \\
                  8          5
               (LEFT)     (RIGHT)
               
   LEFT branch (visible):  8/13 = 0.615
   RIGHT branch (dark):    5/13 = 0.385
   
   If α_dark = α_visible × (some tree ratio):
""")

# Check branch ratios
left = 8/13
right = 5/13
print(f"   α × (5/13) = {alpha_visible * right:.6f} (vs {alpha_SEC})")
print(f"   α × (8/13) = {alpha_visible * left:.6f}")
print(f"   α × (5/8)  = {alpha_visible * 5/8:.6f}")
print(f"   α × (8/13)² = {alpha_visible * left**2:.6f}")

# The actual ratio
print(f"\n   Actual ratio needed: {target_ratio:.6f}")
print(f"   (F_6-F_4)/F_6 = (8-3)/8 = {5/8:.6f}")
print(f"   F_5/F_6 = 5/8 = {5/8:.6f}")
print(f"   (F_7-F_5)/F_7 = 8/13 = {8/13:.6f}")

# =============================================================================
# Part 4: The ξ Parameter
# =============================================================================
print("\n" + "=" * 70)
print("4. THE BALANCE OPERATOR ξ = 1.0571")
print("=" * 70)

print(f"\n   SEC uses ξ = {xi_SEC}")
print(f"   Is this a Fibonacci ratio?")

xi_candidates = []
for n1 in range(1, 15):
    for n2 in range(1, 15):
        ratio = fib(n1) / fib(n2)
        err = abs(ratio - xi_SEC) / xi_SEC
        if err < 0.10:
            xi_candidates.append((f"F_{n1}/F_{n2}", ratio, err))

# With phi
for p in np.arange(-2, 2, 0.5):
    ratio = PHI**p
    err = abs(ratio - xi_SEC) / xi_SEC
    if err < 0.10:
        xi_candidates.append((f"φ^{p}", ratio, err))

# Check 1 + small Fibonacci ratio
for n1 in range(1, 10):
    for n2 in range(n1+1, 15):
        ratio = 1 + fib(n1)/fib(n2)
        err = abs(ratio - xi_SEC) / xi_SEC
        if err < 0.05:
            xi_candidates.append((f"1 + F_{n1}/F_{n2}", ratio, err))

xi_candidates.sort(key=lambda x: x[2])
print("\n   Best ξ matches:")
for name, val, err in xi_candidates[:5]:
    print(f"   {name:25s} = {val:.6f} (error: {err*100:.2f}%)")

# =============================================================================
# Part 5: Predicted Dark Matter α from PAC
# =============================================================================
print("\n" + "=" * 70)
print("5. PAC PREDICTION FOR DARK MATTER α")
print("=" * 70)

# From tree structure, the most natural dark α would be:
# α_dark = α_visible × (RIGHT/LEFT) = α × (5/8)
alpha_dark_pred1 = alpha_visible * 5/8
print(f"\n   Option 1: α_dark = α × (F_5/F_6) = α × (5/8)")
print(f"   Predicted: {alpha_dark_pred1:.6f}")
print(f"   SEC uses:  {alpha_SEC:.6f}")
print(f"   Error: {abs(alpha_dark_pred1 - alpha_SEC)/alpha_SEC*100:.1f}%")

# Or: α_dark = α × (1 - 1/F_5) = α × 4/5
alpha_dark_pred2 = alpha_visible * 4/5
print(f"\n   Option 2: α_dark = α × (1 - 1/F_5) = α × (4/5)")
print(f"   Predicted: {alpha_dark_pred2:.6f}")
print(f"   SEC uses:  {alpha_SEC:.6f}")
print(f"   Error: {abs(alpha_dark_pred2 - alpha_SEC)/alpha_SEC*100:.1f}%")

# Or: the target ratio 0.8025... is close to 8/10 = 4/5
# Or 13/16 = 0.8125
alpha_dark_pred3 = alpha_visible * 13/16
print(f"\n   Option 3: α_dark = α × (F_7/16) = α × (13/16)")
print(f"   Predicted: {alpha_dark_pred3:.6f}")
print(f"   SEC uses:  {alpha_SEC:.6f}")
print(f"   Error: {abs(alpha_dark_pred3 - alpha_SEC)/alpha_SEC*100:.1f}%")

# Direct from tree: dark uses φ scaling differently
alpha_dark_pred4 = alpha_visible * (1 - 1/PHI**2)
print(f"\n   Option 4: α_dark = α × (1 - 1/φ²) = α × (1 - 0.382)")
print(f"   Predicted: {alpha_dark_pred4:.6f}")
print(f"   SEC uses:  {alpha_SEC:.6f}")
print(f"   Error: {abs(alpha_dark_pred4 - alpha_SEC)/alpha_SEC*100:.1f}%")

# =============================================================================
# Part 6: Best Prediction
# =============================================================================
print("\n" + "=" * 70)
print("6. BEST PAC PREDICTION FOR DARK SECTOR")
print("=" * 70)

# Find the absolute best match
all_predictions = [
    ("α × (5/8)", alpha_visible * 5/8),
    ("α × (4/5)", alpha_visible * 4/5),
    ("α × (8/10)", alpha_visible * 8/10),
    ("α × (13/16)", alpha_visible * 13/16),
    ("α × (1 - 1/φ²)", alpha_visible * (1 - 1/PHI**2)),
    ("α × (φ - 1)", alpha_visible * (PHI - 1)),
    ("α × (2/φ²)", alpha_visible * (2/PHI**2)),
    ("α × (F_4/F_5)", alpha_visible * 3/5),
    ("α × (F_5/F_7)", alpha_visible * 5/13),
]

all_predictions.sort(key=lambda x: abs(x[1] - alpha_SEC))

print(f"\n   SEC empirical α_dark = {alpha_SEC}")
print(f"\n   Ranked PAC predictions:")
for name, val in all_predictions:
    err = abs(val - alpha_SEC)/alpha_SEC * 100
    print(f"   {name:20s} = {val:.6f} (error: {err:.1f}%)")

best_name, best_val = all_predictions[0]
best_err = abs(best_val - alpha_SEC)/alpha_SEC * 100

print(f"""
   
   RESULT:
   -------
   Best PAC prediction: {best_name}
   Predicted α_dark = {best_val:.6f}
   SEC empirical    = {alpha_SEC:.6f}
   Error: {best_err:.1f}%
   
   INTERPRETATION:
   The SEC dark matter simulation's empirically-tuned α ≈ 0.00586
   is close to α_visible × (4/5) = 0.00584
   
   In tree terms: (4/5) = (F_5 - 1)/F_5 = (5-1)/5
   Or: dark sector "loses" 1/F_5 of coupling strength
   
   This suggests the dark sector operates with ~80% of visible coupling,
   which could explain why dark matter is weakly self-interacting.
""")

# =============================================================================
# Part 7: Recommendation for SEC Simulation
# =============================================================================
print("\n" + "=" * 70)
print("7. RECOMMENDATION: TEST PAC-PREDICTED α IN SEC SIMULATION")
print("=" * 70)

print(f"""
   Current SEC α = {alpha_SEC} (empirically tuned, 63% similarity)
   
   PAC predictions to test:
   
   1. α = {alpha_visible * 4/5:.6f}  [α × (4/5)]
      Rationale: Dark loses 1/F_5 of coupling
      
   2. α = {alpha_visible * 5/8:.6f}  [α × (F_5/F_6)]
      Rationale: RIGHT/LEFT branch ratio
      
   3. α = {alpha_visible * (PHI-1):.6f}  [α × (φ-1)]
      Rationale: Golden ratio complement
   
   If any of these IMPROVES the 63% similarity to observations,
   it validates PAC structure in cosmological simulations.
   
   ACTION: Re-run darkmatter_SEC_WIP with these α values.
""")
