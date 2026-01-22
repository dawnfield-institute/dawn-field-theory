"""
Exploring Andy's observation about special Fibonacci numbers:

- 55 (F10): Largest number that is BOTH Fibonacci AND Triangular
- 144 (F12): Largest Fibonacci that is a perfect SQUARE (12²)
- 144/55 ≈ φ² ?

And from our oscillation work: Ξ - 1 = π/55

Is there something linking π, φ², and these special positions?
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi

print("=" * 60)
print("ANDY'S OBSERVATION: Special Fibonacci Numbers")
print("=" * 60)

# The claim: 144/55 = φ²
print("\n[1] 144/55 vs φ²")
ratio = 144 / 55
phi_sq = PHI ** 2
print(f"  144/55     = {ratio}")
print(f"  φ²         = {phi_sq}")
print(f"  Difference = {abs(ratio - phi_sq):.10f}")
print(f"  Relative   = {100 * abs(ratio - phi_sq) / phi_sq:.6f}%")

# This is actually a known result: consecutive Fibonacci ratios approach φ
# F(n+2)/F(n) approaches φ² as n increases
print("\n[2] Fibonacci ratio convergence to φ²")
fib = [1, 1]
for i in range(20):
    fib.append(fib[-1] + fib[-2])

print("  F(n+2)/F(n) converges to φ²:")
for n in range(1, 15):
    ratio = fib[n+2] / fib[n]
    print(f"    F({n+2:2d})/F({n:2d}) = {fib[n+2]:5d}/{fib[n]:3d} = {ratio:.6f} (φ² = {phi_sq:.6f})")

# Now the interesting part: π/55 connection
print("\n[3] The π/55 connection (from oscillation attractor work)")
print(f"  Ξ - 1 = π/55 = {PI/55:.6f}")
print(f"  Ξ = 1 + π/55 = {1 + PI/55:.6f}")
print(f"  At depth 55, net twist = π (Möbius half-twist)")

# Is there a relationship between φ², π, and 55?
print("\n[4] Looking for π-φ² relationships")

# Some known relationships
print("\n  Known/approximate identities:")
print(f"  φ² = φ + 1 = {PHI + 1:.10f}")
print(f"  φ² = {phi_sq:.10f}")
print(f"  π/φ² = {PI/phi_sq:.10f}")
print(f"  π·φ = {PI * PHI:.10f}")
print(f"  (π + φ)/e = {(PI + PHI)/np.e:.10f}")

# What about 144, 55, and π?
print("\n[5] Combinations with 144, 55, π")
print(f"  144/π = {144/PI:.6f}")
print(f"  55/π  = {55/PI:.6f}")
print(f"  π²/55 = {PI**2/55:.6f}")
print(f"  144/π² = {144/PI**2:.6f}")
print(f"  (144-55)/π = 89/π = {89/PI:.6f}")

# The gap between 144 and 55 is 89 (also Fibonacci!)
print("\n[6] 144 - 55 = 89 (F11, also Fibonacci!)")
print(f"  F10 = 55")
print(f"  F11 = 89")
print(f"  F12 = 144")
print(f"  This is just the Fibonacci recurrence: 55 + 89 = 144")

# Check if there's something special about F10, F11, F12
print("\n[7] Special properties of F10, F11, F12")
print(f"  55 = T10 (10th triangular number: 1+2+...+10 = 55)")
print(f"  55 = F10 (10th Fibonacci)")
print(f"  144 = 12² (perfect square)")
print(f"  144 = F12 (12th Fibonacci)")

# Is there a pattern with triangular + square?
print("\n[8] Triangular numbers near Fibonacci")
triangular = [n*(n+1)//2 for n in range(1, 20)]
squares = [n**2 for n in range(1, 15)]
print(f"  Triangular: {triangular[:15]}")
print(f"  Squares:    {squares}")
print(f"  Fibonacci:  {fib[1:15]}")

# Intersections
fib_set = set(fib[:20])
tri_set = set(triangular)
sq_set = set(squares)

print(f"\n  Fib ∩ Triangular: {sorted(fib_set & tri_set)}")
print(f"  Fib ∩ Square:     {sorted(fib_set & sq_set)}")

# Now the key question: does 144/55 or related ratios appear in our Ξ framework?
print("\n[9] Connections to Ξ = 1 + π/55")
XI = 1 + PI/55
print(f"  Ξ = {XI:.10f}")
print(f"  Ξ² = {XI**2:.10f}")
print(f"  φ²/Ξ = {phi_sq/XI:.10f}")
print(f"  Ξ·φ = {XI * PHI:.10f}")
print(f"  Ξ·φ² = {XI * phi_sq:.10f}")

# What about π²?
print("\n[10] π² relationships")
print(f"  π² = {PI**2:.10f}")
print(f"  π²/φ² = {PI**2/phi_sq:.10f}")
print(f"  π²/Ξ² = {PI**2/XI**2:.10f}")
print(f"  55·Ξ/π = {55*XI/PI:.10f}")
print(f"  144·Ξ/π² = {144*XI/PI**2:.10f}")

# The c² = πφ/Ξ relationship from last session
print("\n[11] From previous session: c² = πφ/Ξ")
c_squared = PI * PHI / XI
print(f"  πφ/Ξ = {c_squared:.10f}")
print(f"  √(πφ/Ξ) = {np.sqrt(c_squared):.10f}")

# Check: is there a clean relationship between 144/55 and π/55?
print("\n[12] Relationship between 144/55 and π/55")
print(f"  144/55 = {144/55:.10f}")
print(f"  π/55   = {PI/55:.10f}")
print(f"  (144/55)/(π/55) = 144/π = {144/PI:.10f}")
print(f"  φ²/(π/55) = 55φ²/π = {55*phi_sq/PI:.10f}")

# That last one is close to something...
print("\n[13] Looking for near-integer or near-simple relationships")
print(f"  55φ²/π = {55*phi_sq/PI:.6f}")
print(f"  55φ/π = {55*PHI/PI:.6f}")
print(f"  89/π = {89/PI:.6f}")  # F11/π
print(f"  144/(π·φ) = {144/(PI*PHI):.6f}")

# Check some of these against simple numbers
print("\n[14] Near-integer checks")
for expr, val in [
    ("55φ²/π", 55*phi_sq/PI),
    ("55φ/π", 55*PHI/PI),
    ("144/π", 144/PI),
    ("89φ/π²", 89*PHI/PI**2),
    ("144φ/π²", 144*PHI/PI**2),
]:
    nearest_int = round(val)
    diff = abs(val - nearest_int)
    print(f"  {expr:12} = {val:.6f} (nearest int: {nearest_int}, diff: {diff:.6f})")

print("\n" + "=" * 60)
print("SUMMARY: What jumps out")
print("=" * 60)
print("""
1. 144/55 = φ² is exact in the limit (Fibonacci property)
2. 55 is uniquely both F10 AND T10 (Fibonacci and Triangular)
3. 144 = F12 = 12² (Fibonacci and Square)
4. 89 = F11 bridges them (55 + 89 = 144)
5. Ξ - 1 = π/55 from our oscillation work
6. The 55 appearing in both contexts is suggestive

The question: is there a deeper structure linking:
  - Triangular/Square intersections with Fibonacci
  - The π/55 ratio in collapse dynamics (Ξ)
  - The φ² limit of F(n+2)/F(n)
""")
