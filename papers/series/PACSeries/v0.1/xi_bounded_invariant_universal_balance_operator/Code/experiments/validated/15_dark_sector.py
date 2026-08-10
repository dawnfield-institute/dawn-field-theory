"""
Dark Sector Analysis from Fractal PAC Tree
==========================================

The PAC tree at F_7 = 13:

                13 (root)
               /        \
              8          5
             / \        / \
            5   3      3   2
           /\ /\      /\ /\
          3 2 2 1    2 1 1 1

Key observation: The tree has TWO primary branches:
- LEFT branch: F_6 = 8 (visible sector? - contains gauge structure)
- RIGHT branch: F_5 = 5 (dark sector?)

Let's explore this systematically.
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
print("DARK SECTOR ANALYSIS FROM PAC TREE")
print("=" * 70)

# =============================================================================
# Part 1: The Two Branches
# =============================================================================
print("\n1. THE TWO PRIMARY BRANCHES")
print("-" * 50)

print("""
   The PAC tree splits at the root:
   
                    13
                   /  \\
                  8    5
                 /\\   /\\
               ... ...
   
   LEFT (F_6 = 8):  Contains 8, 5, 3, 2, 1
                    This is the visible gauge sector
                    8 gluons + SU(2) + U(1)
   
   RIGHT (F_5 = 5): Contains 5, 3, 2, 1
                    What is this?
""")

# Branch fractions
left_frac = 8/13
right_frac = 5/13
print(f"   LEFT branch fraction:  {left_frac:.4f} = {8}/13 = {left_frac*100:.1f}%")
print(f"   RIGHT branch fraction: {right_frac:.4f} = {5}/13 = {right_frac*100:.1f}%")

# =============================================================================
# Part 2: Cosmological Abundances
# =============================================================================
print("\n\n2. COSMOLOGICAL ABUNDANCES")
print("-" * 50)

# Planck 2018 values
Omega_b = 0.0493      # Baryonic matter
Omega_DM = 0.265      # Dark matter
Omega_DE = 0.685      # Dark energy
Omega_total = 1.0

print(f"\n   Planck 2018 cosmic abundances:")
print(f"   Ω_baryon     = {Omega_b:.3f} ({Omega_b*100:.1f}%)")
print(f"   Ω_dark_matter = {Omega_DM:.3f} ({Omega_DM*100:.1f}%)")
print(f"   Ω_dark_energy = {Omega_DE:.3f} ({Omega_DE*100:.1f}%)")
print(f"   Ω_total       = {Omega_total:.3f} ({Omega_total*100:.1f}%)")

print("\n   Visible vs Dark:")
visible = Omega_b
dark = Omega_DM + Omega_DE
print(f"   Visible (baryon): {visible:.3f} ({visible*100:.1f}%)")
print(f"   Dark (DM + DE):   {dark:.3f} ({dark*100:.1f}%)")

print("\n   Tree branch comparison:")
print(f"   LEFT branch (8/13):  {left_frac*100:.1f}%")
print(f"   RIGHT branch (5/13): {right_frac*100:.1f}%")
print(f"   Visible sector:      {visible*100:.1f}%")
print(f"   Dark sector:         {dark*100:.1f}%")

print("""
   MISMATCH: 8/13 ≈ 61.5% vs 5% visible
             5/13 ≈ 38.5% vs 95% dark
   
   This doesn't match directly. Let's think differently...
""")

# =============================================================================
# Part 3: Alternative Interpretation - Mass vs Number
# =============================================================================
print("\n3. ALTERNATIVE: MASS vs DEGREE OF FREEDOM COUNT")
print("-" * 50)

print("""
   The tree counts DEGREES OF FREEDOM, not masses.
   
   Consider: Most visible mass is in protons/neutrons
   Most of THAT mass is from QCD binding (~99%)
   
   QCD binding energy comes from gluon field (F_6 = 8)
   Fundamental quark masses are tiny
   
   If dark matter is weakly interacting:
   - Fewer DoF contribute to binding energy
   - More mass remains "dark" (not in QCD)
""")

# =============================================================================
# Part 4: Dark Matter as Hidden Branch Structure
# =============================================================================
print("\n\n4. DARK MATTER AS HIDDEN BRANCH")
print("-" * 50)

print("""
   Hypothesis: The RIGHT branch (F_5 = 5) represents
   degrees of freedom that DON'T couple to QCD.
   
   In the visible sector:
   - SU(3)_color couples matter to itself via gluons
   - This creates strong binding → visible mass
   
   In the dark sector:
   - Different gauge structure (if any)
   - Weak or no self-coupling → less binding
   - Mass stays "hidden"
   
   What gauge structure fits F_5 = 5?
   - SU(2) has dim = 3 ✗
   - U(1) has dim = 1 ✗
   - No simple group has dim = 5
   
   BUT: 5 = 3 + 2 (appears in the RIGHT branch!)
   Could be: SU(2)_dark × U(1)_dark?
   Or: A different structure entirely
""")

# =============================================================================
# Part 5: Dark Energy from Tree Geometry
# =============================================================================
print("\n\n5. DARK ENERGY FROM TREE GEOMETRY")
print("-" * 50)

print("""
   Dark energy (Λ): The cosmological constant problem
   
   Measured: Λ ~ 10^(-122) in Planck units
   QFT prediction: Λ ~ 1 (in Planck units)
   
   This is the "worst prediction in physics"
   
   Tree interpretation: Dark energy is NOT a particle DoF
   It's the GEOMETRY of the tree itself.
   
   Consider: The tree has 4 levels (depths 0-3)
   Each level sums to 13 (conservation)
   
   "Vacuum energy" = energy of the tree structure
   = something related to F_7² or F_7/φ^k ?
""")

# Attempt: cosmological constant from Fibonacci
Lambda_Planck = 10**(-122)  # measured
print(f"\n   Measured Λ (Planck units): {Lambda_Planck:.0e}")
print(f"   φ^(-122) = {PHI**(-122):.2e}")
print(f"   1/F_58 ≈ ? (F_58 ≈ 10^12)")

# Find which Fibonacci gives right order
for n in range(50, 300):
    f_n = fib(n)
    if f_n > 0:
        log_fn = np.log10(float(f_n))
        if abs(log_fn - 122) < 5:
            print(f"   1/F_{n} ≈ 1/{f_n:.2e} ≈ {1/float(f_n):.2e}")

# =============================================================================
# Part 6: The 5/13 Dark Fraction Revisited
# =============================================================================
print("\n\n6. THE 5/13 FRACTION REVISITED")
print("-" * 50)

print("""
   Let's reconsider what 5/13 could represent:
   
   If the tree represents ALL structure (visible + dark),
   then both branches contribute to mass/energy.
   
   LEFT (8):  Gauge DoF that bind strongly → visible mass
   RIGHT (5): DoF that don't bind → dark matter/energy?
   
   Actually: Maybe the split is about COUPLING STRENGTH
   - 8/13 of structure couples strongly
   - 5/13 couples weakly
   
   This would predict:
   Dark sector coupling / Visible coupling ~ 5/8 = 0.625
""")

# Let's check: if visible sector has α_s ~ 0.12, 
# dark sector might have α_dark ~ 0.12 * 5/8 = 0.075
alpha_s = 0.118
alpha_dark_predicted = alpha_s * 5/8
print(f"\n   If dark coupling = (5/8) × α_s:")
print(f"   α_s = {alpha_s}")
print(f"   α_dark ~ {alpha_dark_predicted:.4f}")
print(f"\n   This is close to g'/g_Z = 1/13 = {1/13:.4f} × g_Z")
print(f"   Interesting coincidence...")

# =============================================================================
# Part 7: Dark Matter Candidate from Tree
# =============================================================================
print("\n\n7. DARK MATTER CANDIDATE FROM TREE")
print("-" * 50)

print("""
   The tree predicts specific structures:
   
   At depth 3 (stable): {3, 2, 2, 1, 2, 1, 1, 1}
   
   The THREE F_3=2 are fermion generations
   The FIVE F_2=1 and F_3=2 in RIGHT branch could be:
   - Dark fermions?
   - Sterile neutrinos?
   - WIMPs?
   
   Mass estimate for dark particle:
   If visible leptons have masses m_e, m_μ, m_τ
   Dark leptons might have masses ~ m × φ^k
   
   For dark electron analog:
   m_dark_e ~ m_e × φ^5 = 0.511 MeV × 11.09 = 5.7 MeV?
   Or: m_dark_e ~ m_e × F_7 = 0.511 × 13 = 6.6 MeV?
""")

m_e = 0.511  # MeV
print(f"\n   Dark electron mass candidates:")
print(f"   m_e × φ⁵ = {m_e * PHI**5:.2f} MeV")
print(f"   m_e × F₇ = {m_e * 13:.2f} MeV")
print(f"   m_e × F₁₀ = {m_e * 55:.2f} MeV = {m_e * 55 / 1000:.3f} GeV")

# What about WIMP scale?
print(f"\n   WIMP-scale dark matter (~100 GeV)?")
print(f"   m_e × F₇² = {m_e * 169:.1f} MeV = {m_e * 169 / 1000:.2f} GeV")
print(f"   m_e × F₁₀² = {m_e * 55**2:.0f} MeV = {m_e * 55**2 / 1000:.1f} GeV")
print(f"   m_e × F₇ × F₁₀ = {m_e * 13 * 55:.0f} MeV = {m_e * 13 * 55 / 1000:.1f} GeV")

# =============================================================================
# Part 8: Predictions for Dark Sector
# =============================================================================
print("\n\n" + "=" * 70)
print("PREDICTIONS FOR DARK SECTOR")
print("=" * 70)

print("""
   From the PAC tree RIGHT branch, we predict:
   
   1. DARK SECTOR GAUGE STRUCTURE
      DoF count: F_5 = 5
      Possible: SU(2)_dark + U(1)_dark' (3+1 = 4... close)
      Or: New gauge group with dim(adj) = 4 or 5
      
   2. DARK SECTOR COUPLING
      α_dark ~ (5/8) × α_s ≈ 0.074
      Or: α_dark ~ 1/F_7 = 1/13 ≈ 0.077
      This is the Z' coupling we predicted!
      
   3. DARK PARTICLE SPECTRUM
      - Dark leptons at MeV-GeV scale
      - Dark hadrons (if SU(2)_dark exists) at GeV scale
      - Possible dark Z' at 395 GeV (already predicted)
      
   4. DARK-VISIBLE COUPLING
      Z' mediates between sectors
      g'/g_Z = 1/13 (weak mixing)
      Explains why dark matter is "dark" (weakly coupled)
      
   5. COSMOLOGICAL SIGNATURES
      Dark sector temperature: T_dark/T_visible ~ 5/8?
      Or: Different thermal history
      
   KEY INSIGHT:
   The Z' boson at 395 GeV may be the PORTAL 
   between visible (LEFT) and dark (RIGHT) branches!
""")
