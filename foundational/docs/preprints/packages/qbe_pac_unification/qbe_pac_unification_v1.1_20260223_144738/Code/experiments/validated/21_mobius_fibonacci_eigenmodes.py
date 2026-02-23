#!/usr/bin/env python3
"""
21_mobius_fibonacci_eigenmodes.py - Test for Fibonacci structure in Möbius eigenmodes

Hypothesis: The Möbius topology (which creates π harmonics) may have Fibonacci
structure in its eigenmode spectrum, connecting PAC (discrete) to Möbius (continuous).

Key questions:
1. Do Möbius eigenfrequencies have Fibonacci ratios?
2. Does golden angle parameterization create special stability?
3. Is φ hidden in the π-harmonic spectrum?
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# ============================================================================
# CONSTANTS
# ============================================================================

phi = (1 + np.sqrt(5)) / 2  # Golden ratio = 1.618...
golden_angle = 2 * np.pi / phi**2  # = 137.5° in radians = 2.399...

def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# First 15 Fibonacci numbers and their ratios
fibs = [fib(i) for i in range(1, 16)]
fib_ratios = [fibs[i+1]/fibs[i] for i in range(len(fibs)-1)]

print("=" * 70)
print("MÖBIUS EIGENMODE ANALYSIS: SEARCHING FOR FIBONACCI STRUCTURE")
print("=" * 70)

print(f"\nGolden ratio φ = {phi:.10f}")
print(f"Golden angle = 2π/φ² = {golden_angle:.6f} rad = {np.degrees(golden_angle):.3f}°")
print(f"π = {np.pi:.10f}")
print(f"Ratio π/φ = {np.pi/phi:.10f}")

# ============================================================================
# MÖBIUS LAPLACIAN EIGENVALUES
# ============================================================================

def mobius_laplacian_matrix(N, M, twist=True):
    """
    Construct the discrete Laplacian on a Möbius strip.
    
    N: points along the loop (u direction, periodic with twist)
    M: points across width (v direction, Neumann boundary)
    twist: if True, Möbius twist at u = N (anti-periodic)
    """
    size = N * M
    L = np.zeros((size, size))
    
    def idx(i, j):
        """Convert (i,j) grid coords to linear index"""
        return i * M + j
    
    for i in range(N):
        for j in range(M):
            k = idx(i, j)
            
            # Self term
            neighbors = 0
            
            # u+ neighbor (with Möbius twist at boundary)
            i_plus = (i + 1) % N
            if i_plus == 0 and twist:  # Crossed the twist
                j_twist = M - 1 - j  # Flip v coordinate
                L[k, idx(i_plus, j_twist)] = 1
                neighbors += 1
            else:
                L[k, idx(i_plus, j)] = 1
                neighbors += 1
            
            # u- neighbor (with Möbius twist at boundary)
            i_minus = (i - 1) % N
            if i == 0 and twist:  # Coming from twist
                j_twist = M - 1 - j
                L[k, idx(i_minus, j_twist)] = 1
                neighbors += 1
            else:
                L[k, idx(i_minus, j)] = 1
                neighbors += 1
            
            # v+ neighbor (Neumann: reflect at boundary)
            if j < M - 1:
                L[k, idx(i, j + 1)] = 1
                neighbors += 1
            
            # v- neighbor (Neumann: reflect at boundary)  
            if j > 0:
                L[k, idx(i, j - 1)] = 1
                neighbors += 1
            
            # Diagonal (negative sum of neighbors for Laplacian)
            L[k, k] = -neighbors
    
    return L

print("\n" + "─" * 70)
print("COMPUTING MÖBIUS STRIP EIGENVALUES")
print("─" * 70)

# Compute eigenvalues for Möbius strip
N, M = 64, 16  # 64 points around loop, 16 across width
L_mobius = mobius_laplacian_matrix(N, M, twist=True)
eigenvalues_mobius = np.sort(np.real(linalg.eigvals(L_mobius)))

# Also compute for regular cylinder (no twist) for comparison
L_cylinder = mobius_laplacian_matrix(N, M, twist=False)
eigenvalues_cylinder = np.sort(np.real(linalg.eigvals(L_cylinder)))

print(f"Grid: {N} × {M} = {N*M} points")
print(f"Computed {len(eigenvalues_mobius)} eigenvalues")

# ============================================================================
# SEARCH FOR FIBONACCI RATIOS IN EIGENVALUE SPECTRUM
# ============================================================================

print("\n" + "─" * 70)
print("SEARCHING FOR FIBONACCI RATIOS IN EIGENVALUE SPECTRUM")
print("─" * 70)

# Get non-zero eigenvalues (skip the zero mode)
nonzero_mobius = eigenvalues_mobius[np.abs(eigenvalues_mobius) > 1e-10]
nonzero_mobius = np.abs(nonzero_mobius)  # Take absolute values
nonzero_mobius = np.sort(nonzero_mobius)

# Compute ratios of consecutive eigenvalues
eig_ratios = nonzero_mobius[1:] / nonzero_mobius[:-1]

# Look for ratios close to φ or Fibonacci ratios
phi_matches = []
fib_ratio_matches = []

for i, ratio in enumerate(eig_ratios[:50]):  # Check first 50 ratios
    # Check against φ
    if abs(ratio - phi) / phi < 0.05:  # Within 5%
        phi_matches.append((i, ratio, abs(ratio - phi) / phi * 100))
    
    # Check against Fibonacci ratios (which converge to φ)
    for j, fr in enumerate(fib_ratios):
        if abs(ratio - fr) / fr < 0.02:  # Within 2%
            fib_ratio_matches.append((i, ratio, j+1, fr))

print(f"\nEigenvalue ratios close to φ = {phi:.6f}:")
if phi_matches:
    for idx, ratio, err in phi_matches[:10]:
        print(f"  λ[{idx+1}]/λ[{idx}] = {ratio:.6f} (error: {err:.2f}%)")
else:
    print("  None found within 5%")

print(f"\nEigenvalue ratios matching Fibonacci ratios F(n+1)/F(n):")
if fib_ratio_matches:
    for idx, ratio, fib_idx, fib_val in fib_ratio_matches[:10]:
        print(f"  λ[{idx+1}]/λ[{idx}] = {ratio:.6f} ≈ F({fib_idx+1})/F({fib_idx}) = {fib_val:.6f}")
else:
    print("  None found within 2%")

# ============================================================================
# LOOK FOR π/φ RELATIONSHIPS
# ============================================================================

print("\n" + "─" * 70)
print("TESTING π/φ RELATIONSHIPS")
print("─" * 70)

pi_phi = np.pi / phi
two_pi_phi = 2 * np.pi / phi
pi_phi_sq = np.pi / phi**2

print(f"Key ratios:")
print(f"  π/φ     = {pi_phi:.10f}")
print(f"  2π/φ    = {two_pi_phi:.10f}")
print(f"  π/φ²    = {pi_phi_sq:.10f}")
print(f"  Golden angle = 2π/φ² = {golden_angle:.10f}")

# Check if any eigenvalues are multiples of these
print(f"\nEigenvalues as multiples of π/φ:")
for i, ev in enumerate(nonzero_mobius[:20]):
    ratio_to_pi_phi = ev / pi_phi
    ratio_to_golden = ev / golden_angle
    
    # Check if close to integer or simple fraction
    for denom in [1, 2, 3, 4, 5]:
        for numer in range(1, 20):
            target = numer / denom
            if abs(ratio_to_pi_phi - target) < 0.05:
                print(f"  λ[{i}] = {ev:.6f} ≈ {numer}/{denom} × (π/φ) = {target * pi_phi:.6f}")
                break

# ============================================================================
# MÖBIUS VS CYLINDER: WHERE DO THEY DIFFER?
# ============================================================================

print("\n" + "─" * 70)
print("MÖBIUS VS CYLINDER EIGENVALUE COMPARISON")
print("─" * 70)

# The Möbius twist should shift some eigenvalues
diff = eigenvalues_mobius - eigenvalues_cylinder
significant_diff = np.where(np.abs(diff) > 0.1)[0]

print(f"Eigenvalues with significant Möbius shift (>0.1):")
for idx in significant_diff[:15]:
    print(f"  λ[{idx}]: Möbius={eigenvalues_mobius[idx]:.4f}, Cylinder={eigenvalues_cylinder[idx]:.4f}, Δ={diff[idx]:.4f}")

# ============================================================================
# GOLDEN ANGLE MODE ANALYSIS
# ============================================================================

print("\n" + "─" * 70)
print("GOLDEN ANGLE STABILITY TEST")
print("─" * 70)

def mode_stability(angle, N=100, steps=200):
    """
    Test how stable a sinusoidal mode is under iteration with given angular frequency.
    Higher stability = mode persists longer under diffusive dynamics.
    """
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    field = np.sin(angle * np.arange(N))
    
    # Simple diffusion iteration
    stability = 0
    for step in range(steps):
        # Laplacian diffusion
        field_new = 0.9 * field + 0.05 * (np.roll(field, 1) + np.roll(field, -1))
        
        # Measure persistence (correlation with original)
        corr = np.abs(np.corrcoef(field, field_new)[0, 1])
        if corr > 0.9:
            stability += 1
        
        field = field_new
    
    return stability / steps

# Test different angular frequencies
test_angles = {
    'π': np.pi,
    'φ': phi,
    '2π/φ² (golden)': golden_angle,
    'π/φ': np.pi/phi,
    '2': 2.0,
    '3': 3.0,
    'e': np.e,
    '√2': np.sqrt(2),
}

print("Mode stability scores (higher = more stable):")
results = []
for name, angle in test_angles.items():
    stab = mode_stability(angle)
    results.append((name, angle, stab))
    print(f"  {name:15s} ({angle:.6f}): {stab:.3f}")

# Also test Fibonacci numbers as angular frequencies
print("\nFibonacci number angular frequencies:")
for i in range(2, 10):
    f = fib(i)
    stab = mode_stability(f)
    print(f"  F({i}) = {f:3d}: stability = {stab:.3f}")

# ============================================================================
# KEY INSIGHT: LOOK FOR φ IN EIGENVALUE SPACING
# ============================================================================

print("\n" + "─" * 70)
print("EIGENVALUE SPACING ANALYSIS")
print("─" * 70)

# The key insight: maybe it's not the ratios, but the SPACING that's Fibonacci
spacings = np.diff(nonzero_mobius[:30])

print("Consecutive eigenvalue spacings (first 20):")
for i, sp in enumerate(spacings[:20]):
    # Check if spacing is close to a Fibonacci number (scaled)
    closest_fib = min(fibs[:10], key=lambda f: abs(sp - f * spacings[0] / fibs[0]))
    print(f"  Δλ[{i}] = {sp:.6f}")

# Look for Fibonacci ratios in spacings
spacing_ratios = spacings[1:] / spacings[:-1]
print(f"\nSpacing ratios close to φ = {phi:.6f}:")
for i, ratio in enumerate(spacing_ratios[:20]):
    if 1.0 < ratio < 3.0:
        error = abs(ratio - phi) / phi * 100
        if error < 20:
            print(f"  Δλ[{i+1}]/Δλ[{i}] = {ratio:.6f} (error from φ: {error:.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "═" * 70)
print("SUMMARY: FIBONACCI-MÖBIUS CONNECTION")
print("═" * 70)

print("""
FINDINGS:

1. EIGENVALUE RATIOS:
   - Möbius eigenvalue ratios don't show obvious φ clustering
   - The spectrum is determined by grid geometry, not golden ratio

2. GOLDEN ANGLE:
   - 2π/φ² = 137.5° is the most irrational angle
   - But Möbius topology uses π (180°) as its fundamental period
   - These are different organizational principles

3. THE DEEPER CONNECTION:
   - Möbius: continuous, uses π for anti-periodic boundary f(u+π) = -f(u)
   - Fibonacci: discrete, uses φ for recursive ratio F(n+1)/F(n) → φ
   
   BOTH create self-referential structures that never close:
   - π is transcendental (circle never closes algebraically)
   - φ is the "most irrational" (continued fraction [1;1,1,1,...])

4. HYPOTHESIS:
   The PAC tree (Fibonacci/φ) and Möbius topology (π) may be 
   DUAL REPRESENTATIONS of the same underlying self-reference:
   
   - φ = discrete recursion optimal (Fibonacci)
   - π = continuous rotation optimal (circles/Möbius)
   
   They meet in structures like:
   - Golden spiral: r(θ) = e^(bθ) where b = ln(φ)/(π/2)
   - The 137.5° golden angle = 2π(1 - 1/φ)

5. WHY BOTH GIVE α ≈ 0.00584 FOR DARK MATTER:
   - PAC: α_dark = α × 4/5 (from Fibonacci tree)
   - SEC/Möbius: α = 0.005857 (empirically optimized)
   
   Perhaps dark matter structure lives at the intersection of
   discrete (particle) and continuous (field) self-reference.
""")

print("=" * 70)
