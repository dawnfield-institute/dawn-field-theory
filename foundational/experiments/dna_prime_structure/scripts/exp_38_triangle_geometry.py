"""
Experiment 38: Triangle-First Geometry

Andy's challenge: what if equilateral triangle is the "boss" instead of square?

In square-first geometry:
- Square area = s² (integer)
- Triangle area = (√3/4)s² (needs √3)

In triangle-first geometry:
- Triangle area = 1 (the unit)
- Square area = ??? (needs irrational?)

This explores which irrationals appear in each basis.
"""

from fractions import Fraction
import math

print("=" * 70)
print("EXPERIMENT 38: TRIANGLE-FIRST VS SQUARE-FIRST GEOMETRY")
print("=" * 70)

# =============================================================================
# Part 1: Square-First (Standard Western Math)
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: SQUARE-FIRST GEOMETRY (standard)")
print("=" * 70)

print("""
Define: unit = side of square = 1
        unit area = 1² = 1

In this system:
  Square area         = s² = 1           (integer)
  Equilateral triangle = (√3/4)s²        (needs √3)
  Circle              = πr²              (needs π)
  Cube volume         = s³ = 1           (integer)
  Tetrahedron         = (√2/12)a³        (needs √2)
""")

s = 1
print("Areas with s=1:")
print(f"  Square:              {s**2} (exact integer)")
print(f"  Equilateral triangle: {math.sqrt(3)/4:.6f} = √3/4 (irrational)")
print(f"  Circle (r=1):        {math.pi:.6f} = π (transcendental)")

print("\nVolumes with s=1:")
print(f"  Cube:                {s**3} (exact integer)")
print(f"  Tetrahedron:         {math.sqrt(2)/12:.6f} = √2/12 (irrational)")
print(f"  Sphere:              {4/3 * math.pi:.6f} = 4π/3 (transcendental)")

# =============================================================================
# Part 2: Triangle-First Geometry
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: TRIANGLE-FIRST GEOMETRY (Andy's challenge)")
print("=" * 70)

print("""
Define: unit = equilateral triangle with area = 1
        What is the side length?
        
        If triangle area = (√3/4)s² = 1
        Then s² = 4/√3
        So s = 2/√(√3) = 2/3^(1/4) ≈ 1.5197
""")

# Side length when triangle area = 1
s_tri = 2 / (3 ** 0.25)
print(f"Triangle unit side: s = 2/3^(1/4) = {s_tri:.6f}")

# Now express square in terms of this unit
# A square with the same side length as our unit triangle:
square_area_same_side = s_tri ** 2
print(f"\nSquare with same side length:")
print(f"  Area = s² = (2/3^(1/4))² = 4/√3 = {square_area_same_side:.6f}")
print(f"       = 4/√3 = 4√3/3 (needs √3!)")

# But what if we want a square with area = 1 in triangle units?
# We need to find what side length gives area = 1
# That's just s = 1, but now our "unit" is the triangle

print("""
The key insight: in triangle-first geometry, a SQUARE becomes the 
shape that needs √3 to express cleanly.

Square area (in triangle units) = √3/4 × (triangle areas)
Triangle area (in triangle units) = 1

The irrational has MOVED from triangle to square!
""")

# =============================================================================
# Part 3: Why do these irrationals appear?
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: WHY THESE SPECIFIC IRRATIONALS?")
print("=" * 70)

print("""
RIGHT ANGLES = INTEGERS

When you tile a plane with squares, everything lines up:
- Pythagorean theorem: a² + b² = c² (integer solutions exist: 3,4,5)
- Grid is orthogonal, coordinates are additive
- No irrationals needed for lengths parallel to axes

The 90° angle is special because sin(90°) = 1, cos(90°) = 0
Both rational! This is why squares give integers.
""")

print("""
60° ANGLES = √3 (PELL SEQUENCE)

For equilateral triangles:
- Height = (√3/2) × base
- This comes from: sin(60°) = √3/2, cos(60°) = 1/2

The √3 appears because of the 60° angle specifically.
In the Pell equation x² - 3y² = 1, the convergents give √3:
  2/1, 7/4, 26/15, 97/56, ...

Why 3? Because 60° = π/3, and 3 is the denominator.
The angle π/3 encodes √3.
""")

print("""
GOLDEN ANGLES = FIBONACCI

The golden angle is 137.5° = 360°/φ²
Pentagon has interior angle 108° = 3π/5

These angles encode φ because:
- cos(36°) = φ/2
- cos(72°) = (φ-1)/2 = 1/(2φ)

The Fibonacci sequence converges to φ because:
F(n+1)/F(n) → φ solves x² = x + 1

Pentagons and golden spirals live in Fibonacci-land.
""")

print("""
CIRCLES = π (TRANSCENDENTAL)

π appears because a circle is the limit of regular n-gons as n → ∞

Each n-gon has its own algebraic irrationals:
- Triangle (n=3): √3
- Square (n=4): √2 (for diagonal)
- Pentagon (n=5): φ
- Hexagon (n=6): √3 again
- ...

As n → ∞, you leave the algebraic world entirely.
π is transcendental = not the root of any polynomial.

The Lucas approximation 22/7 = 2L₅/L₄ is a coincidence (?)
or maybe reflects that regular polygons with Fibonacci sides
converge to circles in a special way?
""")

# =============================================================================
# Part 4: The Angle-Irrational Correspondence
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: ANGLE-IRRATIONAL CORRESPONDENCE")
print("=" * 70)

angles = [
    (90, "π/2", "1, 0", "integers (1, 0)", "Pell-2 (x²-2y²=1)"),
    (60, "π/3", "√3/2, 1/2", "√3", "Pell-3 (x²-3y²=1)"),
    (45, "π/4", "√2/2, √2/2", "√2", "Pell-2 (x²-2y²=1)"),
    (36, "π/5", "see below", "φ", "Fibonacci"),
    (30, "π/6", "1/2, √3/2", "√3", "Pell-3 (x²-3y²=1)"),
]

print(f"{'Angle':<8} {'Radians':<8} {'sin, cos':<15} {'Irrational':<10} {'Sequence':<20}")
print("-" * 70)
for angle, rad, sincos, irr, seq in angles:
    print(f"{angle}°{'':<5} {rad:<8} {sincos:<15} {irr:<10} {seq:<20}")

print(f"\n  cos(36°) = (1 + √5)/4 = φ/2 = {math.cos(math.radians(36)):.6f}")
print(f"  cos(72°) = (√5 - 1)/4 = 1/2φ = {math.cos(math.radians(72)):.6f}")

print("""
The pattern:
  π/N angle → needs √(related to N) or special algebraic number
  
  N=2 (180°, straight): trivial
  N=3 (60°): √3
  N=4 (90°): integers (special case!)
  N=5 (72°): φ
  N=6 (60°): √3 again
  
  90° is the ONLY angle where sin and cos are both rational.
  That's why square geometry is "simple" - it's the exception!
""")

# =============================================================================
# Part 5: Triangle-First Reformulation
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: REFORMULATING ALL GEOMETRY WITH TRIANGLE AS UNIT")
print("=" * 70)

# Define triangle area = 1
# Side of unit triangle: s where (√3/4)s² = 1 → s = 2/3^(1/4)
s_unit = 2 / (3 ** 0.25)

print(f"Unit definition: equilateral triangle with area = 1")
print(f"Unit side length: s = 2/3^(1/4) = {s_unit:.6f}")
print(f"Unit height: h = √3/2 × s = {math.sqrt(3)/2 * s_unit:.6f}")

# Express other shapes in triangle-area units
print(f"\nShapes expressed in 'triangle units' (where triangle = 1):")

# Square with same side as unit triangle
sq_same_side = s_unit ** 2
print(f"\n  Square (same side as unit triangle):")
print(f"    Area = s² = 4/√3 = {sq_same_side:.6f} triangle-units")
print(f"         = 4/√3 = 4√3/3 ≈ 2.309 triangle-units")
print(f"    NOTE: Square now needs √3!")

# Square with area = 1 (in normal units)
# Its side is 1, what's that in triangle-side units?
sq_unit_side_ratio = 1 / s_unit
print(f"\n  Square with area = 1 (standard unit):")
print(f"    Side = 1 = {sq_unit_side_ratio:.6f} × (triangle side)")
print(f"    In triangle units: still 1, but side is irrational multiple")

# Circle
print(f"\n  Circle (radius = triangle side):")
print(f"    Area = π × s² = π × 4/√3 = {math.pi * sq_same_side:.6f} triangle-units")
print(f"    Still needs π (transcendental doesn't care about basis)")

# Hexagon - interesting because it's made of 6 equilateral triangles!
print(f"\n  Regular hexagon (side = triangle side):")
print(f"    Area = 6 × (triangle area) = 6 triangle-units")
print(f"    PURE INTEGER in triangle basis!")

print("""
BEAUTIFUL RESULT:

In triangle-first geometry, HEXAGONS become integer!

Hexagon area = 6 triangles (exact)
Square area = 4/√3 triangles (irrational)
Circle area = 4π/√3 triangles (still transcendental)

The hexagon is the "natural" shape for triangle geometry,
just as the square is natural for Cartesian geometry.

This matches nature: honeycombs are hexagonal, not square!
""")

# =============================================================================
# Part 6: The Deep Pattern
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: THE DEEP PATTERN")
print("=" * 70)

print("""
The irrationals don't disappear - they MOVE depending on your basis.

SQUARE-FIRST (Western standard):
  Integer: square, cube, right angles
  √3:      equilateral triangle, tetrahedron, 60° angles
  √2:      diagonal of square, face diagonal of cube
  φ:       pentagon, dodecahedron, 72°/36° angles
  π:       circle, sphere

TRIANGLE-FIRST (Andy's challenge):
  Integer: triangle, hexagon, 60° angles
  √3:      SQUARE, cube becomes weird
  √2:      still appears in 3D
  φ:       still pentagon, dodecahedron
  π:       still circle, sphere (transcendental is basis-independent)

The KEY insight:
  √3 is the "conversion factor" between square and triangle geometries.
  
  In square-basis: triangles need √3
  In triangle-basis: squares need √3
  
  It's symmetric! The √3 is the interface between the two worlds.
""")

# Verify the symmetry
print(f"\nVerifying symmetry:")
print(f"  Triangle in square units: √3/4 = {math.sqrt(3)/4:.6f}")
print(f"  Square in triangle units: 4/√3 = {4/math.sqrt(3):.6f}")
print(f"  Product: (√3/4) × (4/√3) = {(math.sqrt(3)/4) * (4/math.sqrt(3)):.6f} = 1")
print(f"  They're reciprocals (up to factor of 4)!")

# The actual reciprocal relationship
print(f"\n  More precisely:")
print(f"  (√3/4)^(-1) = 4/√3 = 4√3/3 = {4*math.sqrt(3)/3:.6f}")
print(f"  The √3 appears in BOTH - just in numerator vs denominator")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Why specific angles map to specific irrationals:

90° → integers
  Because sin(90°)=1, cos(90°)=0 are both rational
  The ONLY angle with this property
  Makes squares the "simple" basis in standard math

60° → √3  
  Because sin(60°)=√3/2
  Comes from π/3 angle, solved by Pell equation x²-3y²=1

45° → √2
  Because sin(45°)=cos(45°)=√2/2
  The square's diagonal, solved by Pell equation x²-2y²=1

36°/72° → φ
  Because cos(36°)=φ/2, the pentagon angle
  Solved by Fibonacci recurrence F(n+1)=F(n)+F(n-1)

0° → π (as limit)
  Circle = limit of n-gon as n→∞
  Leaves algebraic world entirely

ANDY'S INSIGHT:
  The irrationals aren't intrinsic - they're RELATIVE.
  Switch basis from square to triangle, and √3 moves
  from the "complicated" column to the "simple" column.
  
  Nature uses hexagons (honeycomb) because in the 
  triangle-first world, hexagons ARE the integers.
""")
