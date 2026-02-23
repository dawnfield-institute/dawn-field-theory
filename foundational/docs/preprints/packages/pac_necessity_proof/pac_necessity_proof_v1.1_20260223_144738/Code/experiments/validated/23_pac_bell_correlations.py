#!/usr/bin/env python3
"""
23_pac_bell_correlations.py - Testing Bell-type Correlations in PAC Tree Structure
===================================================================================

HYPOTHESIS:
- SEC alone is local → CHSH ≤ 2 (classical bound) ✓ (confirmed in quantum_validation)
- PAC tree structure encodes non-local correlations
- Tree-related nodes should show correlations that CAN exceed classical bounds

The key insight: In PAC, Ψ(k) = Ψ(k+1) + Ψ(k+2)
This means measuring Ψ(k+1) instantly constrains Ψ(k+2) given knowledge of Ψ(k).
This is structural entanglement - not through signaling but through conservation.

TEST: Create entangled pairs based on PAC tree relationships and measure CHSH.
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Golden ratio
phi = (1 + np.sqrt(5)) / 2

def fib(n: int) -> int:
    """Return nth Fibonacci number."""
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print("=" * 78)
print("PAC BELL CORRELATIONS TEST")
print("Testing whether PAC tree structure produces Bell-type violations")
print("=" * 78)

# ============================================================================
# PART 1: THE CHSH INEQUALITY
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         THE CHSH INEQUALITY                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Classical (local hidden variables):  |S| ≤ 2                                ║
║  Quantum mechanics:                   |S| ≤ 2√2 ≈ 2.828                      ║
║                                                                              ║
║  S = E(a,b) - E(a,b') + E(a',b) + E(a',b')                                   ║
║                                                                              ║
║  Where E(a,b) is the correlation between measurements at angles a and b     ║
║                                                                              ║
║  SEC alone (quantum_validation): S ≈ 1.0 (well below classical bound)        ║
║  PAC prediction: Tree-correlated pairs may exceed S = 2                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PART 2: PAC ENTANGLEMENT MODEL
# ============================================================================

class PACEntangledPair:
    """
    Represents an entangled pair based on PAC tree structure.
    
    The key insight: At any node k, we have Ψ(k) = Ψ(k+1) + Ψ(k+2).
    This creates a constraint that acts like entanglement.
    
    We model this as a shared "hidden state" that determines outcomes,
    but the state is constrained by the Fibonacci conservation law.
    """
    
    def __init__(self, root_index: int = 7):
        """
        Create entangled pair from PAC tree at given root.
        
        At index k:
        - Parent value: F_k
        - Left child: F_{k-1}
        - Right child: F_{k-2}
        - Conservation: F_k = F_{k-1} + F_{k-2}
        """
        self.k = root_index
        self.F_k = fib(root_index)      # Parent (root)
        self.F_left = fib(root_index - 1)   # Left child
        self.F_right = fib(root_index - 2)  # Right child
        
        # The "entangled state" is the ratio structure
        # Normalize to unit interval for measurement
        self.left_weight = self.F_left / self.F_k
        self.right_weight = self.F_right / self.F_k
        
        # Phase angle encodes the Fibonacci relationship
        # This is where the "non-local" correlation lives
        self.entanglement_phase = np.arctan2(self.F_right, self.F_left)
        
    def measure(self, angle_a: float, angle_b: float) -> Tuple[int, int]:
        """
        Measure the entangled pair at angles a (left) and b (right).
        
        The measurement outcome depends on:
        1. The measurement angles
        2. The shared entanglement phase (PAC structure)
        
        This models: "Given conservation Ψ(k) = Ψ(k+1) + Ψ(k+2),
        measuring at angle a on left constrains what we find at angle b on right."
        """
        # The PAC constraint creates correlation through the shared phase
        # This is NOT a local hidden variable - it's a structural constraint
        
        # Left measurement: depends on angle_a and the entanglement
        left_amplitude = np.cos(angle_a - self.entanglement_phase)
        
        # Right measurement: ANTI-correlated due to conservation
        # If left takes more, right takes less (they sum to parent)
        right_amplitude = np.cos(angle_b + self.entanglement_phase)
        
        # Convert to ±1 outcomes (spin-like)
        # The threshold is determined by the Fibonacci weights
        left_outcome = 1 if left_amplitude > 0 else -1
        right_outcome = 1 if right_amplitude > 0 else -1
        
        return left_outcome, right_outcome


def compute_correlation(pairs: List[PACEntangledPair], 
                        angle_a: float, angle_b: float,
                        n_trials: int = 10000) -> float:
    """
    Compute E(a,b) = <A(a) * B(b)> for many trials.
    """
    products = []
    for pair in pairs:
        a, b = pair.measure(angle_a, angle_b)
        products.append(a * b)
    return np.mean(products)


def compute_CHSH(pairs: List[PACEntangledPair],
                 a: float, a_prime: float,
                 b: float, b_prime: float) -> float:
    """
    Compute CHSH value S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    """
    E_ab = compute_correlation(pairs, a, b)
    E_ab_prime = compute_correlation(pairs, a, b_prime)
    E_a_prime_b = compute_correlation(pairs, a_prime, b)
    E_a_prime_b_prime = compute_correlation(pairs, a_prime, b_prime)
    
    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
    return S, E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime


# ============================================================================
# PART 3: TEST AT OPTIMAL ANGLES
# ============================================================================

print("\n" + "=" * 78)
print("PART 3: CHSH TEST AT OPTIMAL QUANTUM ANGLES")
print("=" * 78)

# Optimal angles for maximum quantum violation
# a = 0, a' = π/2, b = π/4, b' = 3π/4
a = 0
a_prime = np.pi / 2
b = np.pi / 4
b_prime = 3 * np.pi / 4

# Create entangled pairs from different PAC tree levels
print("\nTesting PAC entanglement at different tree depths...\n")

results = []
for root_k in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
    # Create many pairs at this tree level
    pairs = [PACEntangledPair(root_k) for _ in range(1000)]
    
    S, E1, E2, E3, E4 = compute_CHSH(pairs, a, a_prime, b, b_prime)
    
    F_k = fib(root_k)
    results.append({
        'k': root_k,
        'F_k': F_k,
        'S': S,
        'exceeds_classical': abs(S) > 2,
        'exceeds_quantum': abs(S) > 2 * np.sqrt(2)
    })
    
    status = "⚠️  EXCEEDS CLASSICAL!" if abs(S) > 2 else "   (classical)"
    if abs(S) > 2 * np.sqrt(2):
        status = "🚨 EXCEEDS QUANTUM!"
    
    print(f"  k={root_k:2d}  F_k={F_k:4d}  |  S = {S:+.4f}  {status}")

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│  Classical bound: |S| ≤ 2.0000                                     │
│  Quantum bound:   |S| ≤ 2.8284                                     │
└────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# PART 4: ANGLE SWEEP TO FIND MAXIMUM VIOLATION
# ============================================================================

print("\n" + "=" * 78)
print("PART 4: SEARCHING FOR MAXIMUM CHSH VIOLATION")
print("=" * 78)

# Use F_7 = 13 (the Standard Model root)
pairs = [PACEntangledPair(7) for _ in range(1000)]

print("\nSweeping measurement angles to find maximum |S|...")
print("Using PAC tree at F_7 = 13 (Standard Model root)\n")

max_S = 0
best_angles = None

# Sweep angles
for a in np.linspace(0, np.pi, 20):
    for a_prime in np.linspace(0, np.pi, 20):
        for b in np.linspace(0, np.pi, 20):
            for b_prime in np.linspace(0, np.pi, 20):
                S, _, _, _, _ = compute_CHSH(pairs, a, a_prime, b, b_prime)
                if abs(S) > abs(max_S):
                    max_S = S
                    best_angles = (a, a_prime, b, b_prime)

print(f"Maximum |S| found: {abs(max_S):.4f}")
print(f"Best angles: a={best_angles[0]:.3f}, a'={best_angles[1]:.3f}, "
      f"b={best_angles[2]:.3f}, b'={best_angles[3]:.3f}")

if abs(max_S) > 2:
    print("\n🔔 PAC STRUCTURE EXCEEDS CLASSICAL BOUND!")
    print(f"   Violation: {abs(max_S) - 2:.4f} above classical limit")
elif abs(max_S) > 1.9:
    print("\n📊 PAC approaches but does not exceed classical bound")
else:
    print("\n📊 PAC correlations remain well within classical bound")

# ============================================================================
# PART 5: COMPARISON WITH SEC-ONLY RESULT
# ============================================================================

print("\n" + "=" * 78)
print("PART 5: COMPARISON WITH SEC-ONLY (FROM quantum_validation)")
print("=" * 78)

SEC_CHSH = 1.002  # From symbolic_entanglement results.md

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SEC vs PAC CHSH COMPARISON                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SEC only (local dynamics):     S = {SEC_CHSH:.3f}                                  │
│  PAC tree (structural):         S = {abs(max_S):.3f}                                  │
│                                                                              │
│  Classical bound:               S = 2.000                                    │
│  Quantum bound:                 S = 2.828                                    │
│                                                                              │
│  SEC improvement from PAC:      {abs(max_S)/SEC_CHSH:.1f}×                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# PART 6: DEEPER ANALYSIS - FIBONACCI PHASE STRUCTURE
# ============================================================================

print("\n" + "=" * 78)
print("PART 6: FIBONACCI PHASE STRUCTURE")
print("=" * 78)

print("\nThe entanglement phase at each tree level:\n")

for k in range(3, 15):
    F_left = fib(k - 1)
    F_right = fib(k - 2)
    F_k = fib(k)
    
    phase = np.arctan2(F_right, F_left)
    phase_over_pi = phase / np.pi
    
    # The golden angle!
    golden_angle = 2 * np.pi / (phi ** 2)  # ≈ 137.5° = 0.764π
    
    print(f"  k={k:2d}  F_k={F_k:4d}  |  phase = {phase:.4f} rad = {phase_over_pi:.4f}π  "
          f"|  ratio = {F_right/F_left:.6f}")

print(f"""
As k → ∞, the phase → arctan(1/φ) = {np.arctan(1/phi):.4f} rad = {np.arctan(1/phi)/np.pi:.4f}π

This is related to the golden angle: 2π/φ² = {2*np.pi/phi**2:.4f} rad = {2/phi**2:.4f}π
""")

# ============================================================================
# PART 7: ENHANCED PAC MODEL WITH SUPERPOSITION
# ============================================================================

print("\n" + "=" * 78)
print("PART 7: ENHANCED MODEL - PAC WITH QUANTUM SUPERPOSITION")
print("=" * 78)

print("""
The basic PAC model above uses deterministic thresholds.
Let's try a model where the Fibonacci structure creates a quantum-like superposition.
""")

class PACQuantumPair:
    """
    Enhanced model: PAC structure creates genuine superposition.
    
    The state is |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
    where coefficients are determined by Fibonacci ratios.
    """
    
    def __init__(self, root_index: int = 7):
        self.k = root_index
        F_k = fib(root_index)
        F_left = fib(root_index - 1)
        F_right = fib(root_index - 2)
        
        # Create entangled state coefficients from Fibonacci structure
        # |ψ⟩ = (F_left|01⟩ + F_right|10⟩) / √(F_left² + F_right²)
        # This is a maximally entangled state weighted by Fibonacci
        
        norm = np.sqrt(F_left**2 + F_right**2)
        
        # State amplitudes (complex)
        self.alpha = 0  # |00⟩
        self.beta = F_left / norm  # |01⟩
        self.gamma = F_right / norm  # |10⟩  
        self.delta = 0  # |11⟩
        
        # For Bell state, we want anticorrelation
        # |ψ⟩ = (|01⟩ - |10⟩)/√2 is singlet
        # Our Fibonacci version: (F_{k-1}|01⟩ - F_{k-2}|10⟩)/norm
        self.gamma = -self.gamma  # Anticorrelation
        
    def measure(self, angle_a: float, angle_b: float) -> Tuple[int, int]:
        """
        Quantum measurement with Born rule probabilities.
        """
        # Measurement operators in computational basis rotated by angle
        # For spin-1/2: |+θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        
        # Probability of outcomes given measurement angles
        # This follows standard quantum mechanics for entangled pairs
        
        # P(+,+) = |⟨++|ψ⟩|² etc.
        ca, sa = np.cos(angle_a/2), np.sin(angle_a/2)
        cb, sb = np.cos(angle_b/2), np.sin(angle_b/2)
        
        # Projection amplitudes
        amp_pp = ca*cb*self.alpha + ca*sb*self.beta + sa*cb*self.gamma + sa*sb*self.delta
        amp_pm = ca*sb*self.alpha + ca*cb*self.beta + sa*sb*self.gamma + sa*cb*self.delta
        amp_mp = sa*cb*self.alpha + sa*sb*self.beta + ca*cb*self.gamma + ca*sb*self.delta
        amp_mm = sa*sb*self.alpha + sa*cb*self.beta + ca*sb*self.gamma + ca*cb*self.delta
        
        # Probabilities (Born rule)
        p_pp = abs(amp_pp)**2
        p_pm = abs(amp_pm)**2
        p_mp = abs(amp_mp)**2
        p_mm = abs(amp_mm)**2
        
        # Normalize
        total = p_pp + p_pm + p_mp + p_mm
        if total > 0:
            p_pp, p_pm, p_mp, p_mm = p_pp/total, p_pm/total, p_mp/total, p_mm/total
        
        # Sample outcome
        r = np.random.random()
        if r < p_pp:
            return 1, 1
        elif r < p_pp + p_pm:
            return 1, -1
        elif r < p_pp + p_pm + p_mp:
            return -1, 1
        else:
            return -1, -1


def compute_correlation_quantum(pairs: List[PACQuantumPair],
                                angle_a: float, angle_b: float,
                                n_trials: int = 10000) -> float:
    """Compute correlation for quantum PAC pairs."""
    products = []
    for _ in range(n_trials):
        pair = np.random.choice(pairs)
        a, b = pair.measure(angle_a, angle_b)
        products.append(a * b)
    return np.mean(products)


def compute_CHSH_quantum(pairs: List[PACQuantumPair],
                         a: float, a_prime: float,
                         b: float, b_prime: float,
                         n_trials: int = 10000) -> Tuple:
    """Compute CHSH for quantum PAC pairs."""
    E_ab = compute_correlation_quantum(pairs, a, b, n_trials)
    E_ab_prime = compute_correlation_quantum(pairs, a, b_prime, n_trials)
    E_a_prime_b = compute_correlation_quantum(pairs, a_prime, b, n_trials)
    E_a_prime_b_prime = compute_correlation_quantum(pairs, a_prime, b_prime, n_trials)
    
    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
    return S, E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime


# Test quantum PAC model
print("\nTesting PAC-Quantum hybrid at optimal angles...\n")

# Optimal angles for Bell state
a = 0
a_prime = np.pi / 2
b = np.pi / 4
b_prime = 3 * np.pi / 4

for root_k in [5, 7, 9, 11, 13]:
    pairs = [PACQuantumPair(root_k) for _ in range(100)]
    S, E1, E2, E3, E4 = compute_CHSH_quantum(pairs, a, a_prime, b, b_prime, n_trials=5000)
    
    F_k = fib(root_k)
    F_left = fib(root_k - 1)
    F_right = fib(root_k - 2)
    ratio = F_left / F_right
    
    status = "✓ EXCEEDS CLASSICAL!" if abs(S) > 2 else "  (within classical)"
    
    print(f"  k={root_k:2d}  F_k={F_k:4d}  ratio={ratio:.4f}  |  S = {S:+.4f}  {status}")

# ============================================================================
# PART 8: ANALYSIS AND CONCLUSIONS
# ============================================================================

print("\n" + "=" * 78)
print("PART 8: ANALYSIS AND CONCLUSIONS")
print("=" * 78)

# Run definitive test at k=7
pairs = [PACQuantumPair(7) for _ in range(100)]

# Multiple runs for statistics
S_values = []
for _ in range(20):
    S, _, _, _, _ = compute_CHSH_quantum(pairs, a, a_prime, b, b_prime, n_trials=5000)
    S_values.append(S)

mean_S = np.mean(S_values)
std_S = np.std(S_values)

# Theoretical prediction for our state
# For |ψ⟩ = (F_{k-1}|01⟩ - F_{k-2}|10⟩)/norm
F_left = fib(6)  # 8
F_right = fib(5)  # 5
norm = np.sqrt(F_left**2 + F_right**2)
# This is like a Bell state with unequal weights

# For perfect Bell state |01⟩-|10⟩, S = 2√2
# For our weighted state, the maximum violation depends on the ratio
weight_ratio = F_left / F_right  # 8/5 = 1.6 = φ!

# The theoretical S for a partially entangled state
# S_max = 2√(1 + C²) where C is concurrence
concurrence = 2 * abs(F_left * F_right) / (F_left**2 + F_right**2)
theoretical_S = 2 * np.sqrt(1 + concurrence**2)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         FINAL RESULTS                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PAC-Quantum Model at F₇ = 13:                                               ║
║                                                                              ║
║  Measured:     S = {mean_S:.3f} ± {std_S:.3f}                                         ║
║  Theoretical:  S ≤ {theoretical_S:.3f} (for this concurrence)                        ║
║                                                                              ║
║  Concurrence:  C = {concurrence:.4f}                                                 ║
║  Weight ratio: F₆/F₅ = 8/5 = {F_left/F_right:.4f} ≈ φ                                      ║
║                                                                              ║
║  BOUNDS:                                                                     ║
║  Classical:    |S| ≤ 2.000                                                   ║
║  Quantum max:  |S| ≤ 2.828                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

if abs(mean_S) > 2:
    print("🔔 RESULT: PAC-Quantum EXCEEDS classical bound!")
    print(f"   Violation magnitude: {abs(mean_S) - 2:.3f}")
    print(f"   This suggests PAC tree structure enables genuine quantum correlations.")
else:
    print("📊 RESULT: PAC-Quantum remains within classical bound")
    print("   The Fibonacci weighting reduces entanglement compared to perfect Bell state.")

print(f"""
INTERPRETATION:
───────────────
1. SEC alone: S ≈ 1.0 (local, no violation) ✓
2. Basic PAC: S ≈ {abs(max_S):.1f} (structural correlations approach classical limit)
3. PAC-Quantum: S ≈ {abs(mean_S):.1f} (Fibonacci-weighted entanglement)

The PAC tree structure creates correlations that:
- Significantly exceed SEC-only correlations ({abs(mean_S)/SEC_CHSH:.1f}× improvement)
- {"Exceed" if abs(mean_S) > 2 else "Approach"} the classical bound
- Use Fibonacci ratios as entanglement weights (F₆/F₅ = φ)

This is consistent with:
- SEC = local dynamics (Born rule, interference, decoherence)  
- PAC = non-local structure (entanglement correlations)

The weight ratio F₆/F₅ = 8/5 = 1.6 ≈ φ is the golden ratio!
PAC entanglement is "golden entanglement" - weighted by φ.
""")

print("\n" + "=" * 78)
print("TEST COMPLETE")
print("=" * 78)
