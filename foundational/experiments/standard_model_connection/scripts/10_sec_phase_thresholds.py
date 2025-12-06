#!/usr/bin/env python3
"""
10_sec_phase_thresholds.py - Magic numbers as SEC phase thresholds

HYPOTHESIS:
  Magic numbers aren't Fibonacci directly - they're SEC PHASE THRESHOLDS
  where structure becomes stable against entropy dissolution.
  
  Fibonacci ratios (3/13, 5/3, 3/2) appear BETWEEN thresholds.
  π harmonics might decode the threshold positions themselves.

Key insight:
  - Fibonacci = the dynamics (flow, ratios, cascade)
  - Magic numbers = the phase boundaries (where things stabilize)
  - π = the harmonic structure connecting them
"""

import numpy as np
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi

def fib(n: int) -> int:
    if n <= 0: return 0
    if n <= 2: return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

F = {i: fib(i) for i in range(1, 25)}

MAGIC = [2, 8, 20, 28, 50, 82, 126]

print("=" * 70)
print("SEC PHASE THRESHOLDS AND π HARMONICS")
print("=" * 70)

# ============================================================================
# TEST 1: MAGIC NUMBERS AS π HARMONICS
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: MAGIC NUMBERS AND π")
print("=" * 70)

print("""
If magic numbers are phase thresholds, they might relate to π harmonics.
Let's check various π-based formulas.
""")

# Check if magic numbers relate to π
print("Magic numbers vs π multiples:")
for m in MAGIC:
    ratio_pi = m / PI
    ratio_pi2 = m / (PI**2)
    ratio_2pi = m / (2*PI)
    print(f"  {m:3d}: m/π = {ratio_pi:.3f}, m/π² = {ratio_pi2:.3f}, m/2π = {ratio_2pi:.3f}")

# Check if differences relate to π
print(f"\nDifferences between magic numbers vs π:")
diffs = [MAGIC[i+1] - MAGIC[i] for i in range(len(MAGIC)-1)]
for i, d in enumerate(diffs):
    ratio = d / PI
    print(f"  {MAGIC[i+1]} - {MAGIC[i]} = {d}: d/π = {ratio:.3f}")

# Check cumulative sums
print(f"\nCumulative pattern:")
cumsum = 0
for i, m in enumerate(MAGIC):
    cumsum += m
    ratio = cumsum / (PI * (i+1)**2)
    print(f"  Σ({m}) = {cumsum}: Σ/(π(n+1)²) = {ratio:.4f}")

# ============================================================================
# TEST 2: MAGIC NUMBERS AS n(n+1) TYPE FORMULA
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: ALGEBRAIC STRUCTURE OF MAGIC NUMBERS")
print("=" * 70)

print("""
Nuclear shell model gives magic numbers from:
  - Harmonic oscillator: 2, 8, 20, 40, 70, 112...
  - With spin-orbit coupling: 2, 8, 20, 28, 50, 82, 126...

The spin-orbit correction shifts the higher shells.
Is there a π or φ pattern in the corrections?
""")

# Harmonic oscillator magic numbers
HO_magic = [2, 8, 20, 40, 70, 112, 168]  # 2(n+1)(n+2)/2 cumulative

print("Harmonic oscillator vs actual magic:")
for ho, actual in zip(HO_magic, MAGIC):
    diff = actual - ho
    print(f"  HO: {ho:3d}, Actual: {actual:3d}, Diff: {diff:+3d}")

# The corrections
corrections = [m - ho for m, ho in zip(MAGIC, HO_magic[:len(MAGIC)])]
print(f"\nCorrections from spin-orbit: {corrections}")
print(f"These are: 0, 0, 0, -12, -20, -30, -42")
print(f"Differences in corrections: {[corrections[i+1]-corrections[i] for i in range(len(corrections)-1)]}")
print(f"Pattern: 0, 0, -12, -8, -10, -12")

# Check if corrections follow pattern
print(f"\nCorrection pattern analysis:")
corr_diffs = [-12, -8, -10, -12]
for c in corr_diffs:
    print(f"  {c}: |c|/π = {abs(c)/PI:.3f}, |c|/φ = {abs(c)/PHI:.3f}")

# ============================================================================
# TEST 3: RATIOS BETWEEN MAGIC NUMBERS AND π
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3: MAGIC NUMBER RATIOS AND π")
print("=" * 70)

print("Ratios of consecutive magic numbers:")
for i in range(len(MAGIC)-1):
    ratio = MAGIC[i+1] / MAGIC[i]
    # Check against π-related values
    pi_frac = ratio / PI
    phi_diff = abs(ratio - PHI)
    print(f"  {MAGIC[i+1]:3d}/{MAGIC[i]:3d} = {ratio:.4f}  (ratio/π = {pi_frac:.4f}, |ratio-φ| = {phi_diff:.4f})")

# ============================================================================
# TEST 4: π × φ COMBINATIONS
# ============================================================================

print("\n" + "=" * 70)
print("TEST 4: π × φ COMBINATIONS")
print("=" * 70)

print("""
What if magic numbers come from π × φ combinations?
Let's search for patterns.
""")

# Key combinations
combos = {
    'π': PI,
    'φ': PHI,
    'π×φ': PI * PHI,
    'π/φ': PI / PHI,
    'π²': PI**2,
    'φ²': PHI**2,
    'π+φ': PI + PHI,
    'π×φ²': PI * PHI**2,
    '2π': 2*PI,
    'π²/φ': PI**2 / PHI,
    'eφ': np.e * PHI,
    'e×π': np.e * PI,
}

print("Searching for magic = n × constant patterns:")
for name, const in combos.items():
    print(f"\n  Testing {name} = {const:.4f}:")
    for m in MAGIC:
        n = m / const
        if abs(n - round(n)) < 0.15:  # Close to integer
            print(f"    {m} ≈ {round(n)} × {name} (error: {abs(n-round(n)):.3f})")

# ============================================================================
# TEST 5: PHASE TRANSITION INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("TEST 5: SEC PHASE TRANSITION INTERPRETATION")
print("=" * 70)

print("""
HYPOTHESIS: Magic numbers are where SEC (entropy) reaches a threshold
that makes the structure stable. Between thresholds, Fibonacci dynamics
govern the transitions.

Phase interpretation:
  - Below magic: structure unstable, seeks equilibrium via weak decay
  - At magic: structure stable, closed shell
  - Fibonacci ratios: describe the DYNAMICS between phases
""")

# Analyze the "between" regions
print("\nBetween-threshold analysis:")
print("-" * 50)

for i in range(len(MAGIC)-1):
    m1, m2 = MAGIC[i], MAGIC[i+1]
    span = m2 - m1
    midpoint = (m1 + m2) / 2
    
    # Check Fibonacci near midpoint
    closest_fib = min(F.values(), key=lambda x: abs(x - midpoint))
    fib_idx = [k for k, v in F.items() if v == closest_fib][0]
    
    # Ratio of span to previous span
    if i > 0:
        prev_span = MAGIC[i] - MAGIC[i-1]
        span_ratio = span / prev_span
    else:
        span_ratio = None
    
    print(f"  [{m1} → {m2}]: span={span}, midpoint={midpoint:.1f}")
    print(f"    Closest Fibonacci: F_{fib_idx} = {closest_fib}")
    if span_ratio:
        print(f"    Span ratio to previous: {span_ratio:.4f} (φ = {PHI:.4f})")
    
    # The "Fibonacci reading" between thresholds
    ratio_fib = span / closest_fib
    print(f"    span/F_{fib_idx} = {ratio_fib:.4f}")
    print()

# ============================================================================
# TEST 6: π HARMONICS IN SHELL STRUCTURE
# ============================================================================

print("\n" + "=" * 70)
print("TEST 6: π HARMONICS IN NUCLEAR SHELLS")
print("=" * 70)

print("""
Nuclear shells have angular momentum quantum numbers.
Angular momentum involves 2π periodicity.
Let's see if magic numbers encode π harmonics.
""")

# Shell closures and angular momentum
# Magic number 2: 1s (l=0)
# Magic number 8: 1s + 1p (l=0,1)
# Magic number 20: + 1d + 2s (l=2,0)
# etc.

# Total angular momentum capacity per shell
shells = [
    (2, '1s', 0),      # 2(2×0+1) = 2
    (6, '1p', 1),      # 2(2×1+1) = 6, cumulative = 8
    (10, '1d', 2),     # 2(2×2+1) = 10
    (2, '2s', 0),      # 2, cumulative = 20
    (14, '1f', 3),     # 2(2×3+1) = 14
    (6, '2p', 1),      # 6
    (8, '1f5/2', None),  # Spin-orbit split
]

print("Shell structure and 2l+1 pattern:")
cumulative = 0
for capacity, name, l in shells[:6]:
    cumulative += capacity
    if l is not None:
        print(f"  {name}: 2(2×{l}+1) = {capacity}, cumulative = {cumulative}")

# Key insight: 2l+1 is the magnetic degeneracy
# Total with spin: 2(2l+1)
# Sum over shells: Σ 2(2l+1) = 2 × Σ(2l+1)

print(f"\n2l+1 sequence for l = 0,1,2,3,4,5: {[2*l+1 for l in range(6)]}")
print(f"Cumulative: {[sum([2*i+1 for i in range(l+1)]) for l in range(6)]}")
print(f"These are perfect squares! (l+1)² = {[(l+1)**2 for l in range(6)]}")
print(f"With spin factor 2: {[2*(l+1)**2 for l in range(6)]}")

# So without spin-orbit: magic = 2n² for n = 1,2,3,...
# n=1: 2, n=2: 8, n=3: 18, n=4: 32, n=5: 50, n=6: 72, n=7: 98

print(f"\nSimple 2n² formula: {[2*n**2 for n in range(1, 8)]}")
print(f"Actual magic:       {MAGIC}")

# The deviation tells us about spin-orbit!
deviations = [m - 2*n**2 for n, m in enumerate(MAGIC, 1)]
print(f"Deviations (actual - 2n²): {deviations}")

# ============================================================================
# TEST 7: FIBONACCI BETWEEN π-PHASES
# ============================================================================

print("\n" + "=" * 70)
print("TEST 7: FIBONACCI AS INTER-PHASE DYNAMICS")
print("=" * 70)

print("""
CORE IDEA:
  - π harmonics set the PHASE BOUNDARIES (shell closures)
  - Fibonacci ratios govern TRANSITIONS BETWEEN phases
  - The 2n² structure comes from angular momentum (2π periodicity)
  - Deviations from 2n² come from spin-orbit (a Fibonacci-like splitting)
""")

# The spin-orbit splitting follows j = l ± 1/2
# This creates pairs that could have Fibonacci structure

print("Spin-orbit splitting pattern:")
print("  For l > 0, shell splits into j = l+1/2 and j = l-1/2")
print("  Capacities: 2(l+1/2)+1 = 2l+2 and 2(l-1/2)+1 = 2l")
print("  Ratio: (2l+2)/(2l) = (l+1)/l")
print()

for l in range(1, 7):
    j_high = 2*l + 2  # j = l + 1/2
    j_low = 2*l       # j = l - 1/2
    ratio = j_high / j_low
    print(f"  l={l}: j_high={j_high}, j_low={j_low}, ratio = {ratio:.4f}")

print(f"\nThese ratios (l+1)/l approach 1 for large l")
print(f"But for small l: 2/1=2, 3/2=1.5=F₄/F₃, 4/3={4/3:.4f}, 5/4={5/4:.4f}")
print(f"Wait: 3/2 = F₄/F₃ appears naturally in spin-orbit!")

# ============================================================================
# TEST 8: THE SYNTHESIS
# ============================================================================

print("\n" + "=" * 70)
print("TEST 8: SYNTHESIS - π PHASES + FIBONACCI DYNAMICS")
print("=" * 70)

print("""
EMERGING PICTURE:

1. ANGULAR MOMENTUM (π periodicity):
   - Gives base structure 2n² for shells
   - This is the "clock" or phase structure
   - Magic numbers without spin-orbit: 2, 8, 18, 32, 50, 72, 98...

2. SPIN-ORBIT COUPLING (Fibonacci dynamics):
   - Splits shells with ratio (l+1)/l
   - For l=1: ratio = 2/1 = F₃/F₁
   - For l=2: ratio = 3/2 = F₄/F₃ ← Fibonacci!
   - For l=3: ratio = 4/3
   - This is the "flow" between phase boundaries

3. THE CORRECTION:
   - Spin-orbit lowers some levels, raises others
   - The pattern: certain j = l - 1/2 levels drop into lower shell
   - Creates actual magic: 2, 8, 20, 28, 50, 82, 126

4. THE FIBONACCI CONNECTION:
   - 3/2 = F₄/F₃ in spin-orbit splitting
   - This same ratio appears in valley of stability N/Z
   - Not coincidence: both involve angular momentum!

5. WHY sin²θ_W = 3/13 = F₄/F₇:
   - Weak force couples to spin (left-handed only!)
   - Spin is angular momentum = π periodicity
   - The weak mixing angle might encode the
     Fibonacci-modulated π harmonic structure!
""")

# ============================================================================
# TEST 9: PREDICTIONS
# ============================================================================

print("\n" + "=" * 70)
print("TEST 9: TESTABLE PREDICTIONS")
print("=" * 70)

print("""
If this picture is correct:

PREDICTION 1: Superheavy magic numbers
  - Next magic after 126 should follow the pattern
  - 2n² for n=8 gives 128, plus spin-orbit correction
  - Expected: around 126 + 34 = 160? Or following ratio pattern?
""")

# Extrapolate magic numbers
ratios = [MAGIC[i+1]/MAGIC[i] for i in range(len(MAGIC)-1)]
avg_ratio = np.mean(ratios[-3:])  # Use recent ratios
predicted_next = int(MAGIC[-1] * avg_ratio)

print(f"  Recent ratios: {[f'{r:.3f}' for r in ratios[-3:]]}")
print(f"  Average: {avg_ratio:.3f}")
print(f"  Predicted next magic: {MAGIC[-1]} × {avg_ratio:.3f} ≈ {predicted_next}")
print(f"  Compare to standard prediction: 184 (protons) or 126, 184 (neutrons)")

print(f"""
PREDICTION 2: Fibonacci in nuclear transition rates
  - Transitions BETWEEN magic numbers should show F ratios
  - Gamma ray energies in nuclear cascades
  - Transition probabilities between shells

PREDICTION 3: π appears in nuclear form factors
  - The charge distribution of nuclei
  - Should show π-periodic structure at magic numbers

PREDICTION 4: Weak decay rates near magic numbers
  - Should show DISCONTINUITIES at magic numbers
  - The "phase transition" interpretation
  - Decay rates jump when crossing threshold
""")

# ============================================================================
# FINAL SYNTHESIS
# ============================================================================

print("\n" + "═" * 70)
print("FINAL SYNTHESIS")
print("═" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│         π PHASES + FIBONACCI DYNAMICS = NUCLEAR STRUCTURE           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  THE TWO LAYERS:                                                    │
│                                                                     │
│    π (PHASE STRUCTURE):          FIBONACCI (DYNAMICS):             │
│    ────────────────────          ──────────────────────             │
│    Angular momentum L            Spin-orbit splitting               │
│    2π periodicity                (l+1)/l ratios                     │
│    Shell closures at 2n²         → 3/2 = F₄/F₃ for l=2             │
│    The "clock"                   The "flow"                         │
│                                                                     │
│  HOW THEY COMBINE:                                                  │
│                                                                     │
│    Base magic (2n²):     2, 8, 18, 32, 50, 72, 98...               │
│    Fibonacci correction: (spin-orbit with F ratios)                 │
│    Actual magic:         2, 8, 20, 28, 50, 82, 126                 │
│                                                                     │
│  THE WEAK FORCE CONNECTION:                                         │
│                                                                     │
│    sin²θ_W = 3/13 = F₄/F₇                                          │
│    Weak force couples to SPIN (left-handed)                         │
│    Spin = angular momentum = π periodicity                          │
│    The Fibonacci ratio IN the π structure!                          │
│                                                                     │
│  INTERPRETATION:                                                    │
│                                                                     │
│    Magic numbers = SEC phase thresholds (where entropy stabilizes) │
│    Fibonacci = the dynamics between thresholds                      │
│    π = the harmonic structure that sets threshold positions        │
│    Together: π × F structure = observable physics                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")
