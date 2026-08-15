#!/usr/bin/env python3
"""
28_pac_full_tree_entanglement.py - Multi-Level Tree Entanglement
=================================================================

KEY INSIGHT: Entanglement isn't just between siblings (8:5).
The PAC conservation Ψ(k) = Ψ(k+1) + Ψ(k+2) applies at EVERY level.

This means correlations CASCADE through the entire tree.
A measurement at any node affects ALL connected nodes.

The question: Does multi-level entanglement give S > 2.68?
Could it reach the full QM maximum S = 2.83?
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize

phi = (1 + np.sqrt(5)) / 2

def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

print("=" * 78)
print("PAC FULL TREE ENTANGLEMENT")
print("Multi-level correlations in the Fibonacci tree")
print("=" * 78)

# ============================================================================
# THE KEY INSIGHT
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    THE MULTI-LEVEL INSIGHT                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Previous analysis: Only considered sibling entanglement (F_6 : F_5 = 8:5)
This gave: S_max = 2.68

But the tree structure means EVERY level is entangled:

Level 0:                    F_7 = 13
                           /        \\
Level 1:              F_6 = 8      F_5 = 5
                     /      \\      /      \\
Level 2:         F_5=5   F_4=3  F_4=3   F_3=2

Conservation at EACH node:
  F_7 = F_6 + F_5  (root)
  F_6 = F_5 + F_4  (left branch)
  F_5 = F_4 + F_3  (right branch)

If we measure F_5 in the left branch, we constrain:
  - F_4 (sibling) via F_6 = F_5 + F_4
  - F_6 (parent) 
  - F_7 (grandparent) via F_7 = F_6 + F_5
  - F_5 on the right (cousin) via F_7 conservation

The entanglement FLOWS through the whole tree!
""")

# ============================================================================
# MODEL: TREE STATE WITH MULTI-LEVEL CORRELATIONS
# ============================================================================

print("\n" + "=" * 78)
print("MODELING MULTI-LEVEL ENTANGLEMENT")
print("=" * 78)

class PACTreeState:
    """
    A quantum state defined by the PAC tree structure.
    
    The state lives in a Hilbert space where each level contributes.
    Conservation laws create entanglement between ALL levels.
    """
    
    def __init__(self, root_k: int = 7, depth: int = 3):
        """
        Create tree state rooted at F_k, going down 'depth' levels.
        """
        self.root_k = root_k
        self.depth = depth
        
        # Build the tree structure
        self.tree = self._build_tree(root_k, depth)
        
        # Calculate the full entangled state
        self.state = self._build_state()
        
    def _build_tree(self, k, d):
        """Recursively build tree of Fibonacci numbers."""
        if d == 0 or k < 3:
            return {'k': k, 'F': fib(k), 'left': None, 'right': None}
        return {
            'k': k,
            'F': fib(k),
            'left': self._build_tree(k-1, d-1),
            'right': self._build_tree(k-2, d-1)
        }
    
    def _build_state(self):
        """
        Build the entangled state from tree structure.
        
        The state is a superposition where each branch configuration
        has amplitude proportional to the Fibonacci product.
        
        For a tree with conservation at each node, the state is:
        |ψ⟩ = Σ (product of F weights) |config⟩
        
        The key insight: conservation at each level COMPOUNDS.
        """
        # For simplicity, focus on the root split
        # but now include NESTED conservation
        
        F_root = fib(self.root_k)
        F_left = fib(self.root_k - 1)
        F_right = fib(self.root_k - 2)
        
        # First level: |ψ_1⟩ = (F_left|L⟩ - F_right|R⟩)/N_1
        # This is what we had before
        
        # But now: each branch ALSO has internal structure
        # |L⟩ itself is entangled: |L⟩ = (F_{k-2}|LL⟩ - F_{k-3}|LR⟩)/N_L
        
        # The FULL state is a tensor product with nested entanglement
        
        return {
            'root': F_root,
            'branches': [(F_left, 'L'), (F_right, 'R')],
            'nested_left': (fib(self.root_k-2), fib(self.root_k-3)) if self.root_k > 4 else None,
            'nested_right': (fib(self.root_k-3), fib(self.root_k-4)) if self.root_k > 5 else None,
        }


# ============================================================================
# MULTI-PARTICLE ENTANGLEMENT
# ============================================================================

print("""
MULTI-PARTICLE BELL STATES
──────────────────────────
Standard 2-particle entanglement: |ψ⟩ = α|01⟩ + β|10⟩
Our single-level result: S = 2.68 for α:β = 8:5

But the tree has MORE than 2 particles!
At depth 2, we have 4 "leaves": (5, 3, 3, 2)

Multi-particle entanglement can give HIGHER Bell violations
via GHZ states or cluster states.

Question: Does the PAC tree structure naturally create
multi-particle entanglement that boosts S?
""")

# ============================================================================
# GHZ-LIKE STATE FROM TREE
# ============================================================================

print("\n" + "=" * 78)
print("GHZ-LIKE STATES FROM PAC TREE")
print("=" * 78)

print("""
A GHZ state: |GHZ⟩ = (|000⟩ + |111⟩)/√2

The PAC tree might create something like:
|ψ_tree⟩ = (F_L|LLL⟩ + F_R|RRR⟩)/N  (if branches align)

Or more complex:
|ψ_tree⟩ = Σ F_path |path⟩  (over all valid paths)

Let's compute the entanglement for the full tree structure.
""")

def compute_tree_correlation(k_root, theta_angles, depth=2):
    """
    Compute correlation for tree-structured entanglement.
    
    Each leaf gets a measurement angle.
    Conservation constraints couple all measurements.
    """
    n_leaves = 2**depth
    
    if len(theta_angles) != n_leaves:
        raise ValueError(f"Need {n_leaves} angles for depth {depth}")
    
    # Build weights for each path through tree
    # Path is sequence of L/R choices
    
    def path_weight(path, k):
        """Weight for path through tree starting at F_k"""
        w = 1.0
        current_k = k
        for choice in path:
            if choice == 'L':
                w *= fib(current_k - 1) / fib(current_k)
                current_k = current_k - 1
            else:  # R
                w *= fib(current_k - 2) / fib(current_k)
                current_k = current_k - 2
        return w
    
    # Generate all paths
    from itertools import product
    paths = list(product(['L', 'R'], repeat=depth))
    
    # Weights for each path
    weights = [path_weight(p, k_root) for p in paths]
    weights = np.array(weights)
    weights = weights / np.linalg.norm(weights)  # Normalize
    
    # Now compute correlation
    # For each path, assign +1 or -1 based on measurement
    
    correlation = 0
    for i, (path, w) in enumerate(zip(paths, weights)):
        # Measurement outcome is product of cos(theta) for each angle
        outcome = 1
        for j, choice in enumerate(path):
            if choice == 'L':
                outcome *= np.cos(theta_angles[j])
            else:
                outcome *= -np.cos(theta_angles[j])  # Anti-correlated for R
        correlation += w**2 * outcome
    
    return correlation


# Compute for depth-2 tree (4 leaves)
print("\nDepth-2 tree (4 leaves):")
print("-" * 40)

# Try various angle configurations
test_angles = [
    [0, 0, 0, 0],
    [0, np.pi/4, np.pi/2, 3*np.pi/4],
    [np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8],
]

for angles in test_angles:
    corr = compute_tree_correlation(7, angles, depth=2)
    print(f"  Angles: {[f'{a:.3f}' for a in angles]}")
    print(f"  Correlation: {corr:.4f}")
    print()

# ============================================================================
# THE CRUCIAL CALCULATION: EFFECTIVE 2-PARTICLE STATE
# ============================================================================

print("\n" + "=" * 78)
print("EFFECTIVE 2-PARTICLE STATE FROM FULL TREE")
print("=" * 78)

print("""
Key insight: When we "trace out" the internal structure of each branch,
we get an EFFECTIVE 2-particle state. But the effective amplitudes
are modified by the internal correlations!

For tree rooted at F_7 = 13:
  Level 1: |ψ⟩ ∝ F_6|L⟩ - F_5|R⟩ = 8|L⟩ - 5|R⟩

But |L⟩ itself has structure:
  |L⟩ ∝ F_5|LL⟩ - F_4|LR⟩ = 5|LL⟩ - 3|LR⟩

And |R⟩:
  |R⟩ ∝ F_4|RL⟩ - F_3|RR⟩ = 3|RL⟩ - 2|RR⟩

When we project to 2-particle (L vs R at top level), 
the EFFECTIVE weight includes contributions from all levels.
""")

def effective_2particle_state(k_root, depth):
    """
    Compute effective 2-particle state after tracing internal structure.
    
    Returns (alpha_eff, beta_eff) for |ψ_eff⟩ = α_eff|0⟩ - β_eff|1⟩
    """
    # The effective weight for "left" is the sum over all left-starting paths
    # weighted by their probabilities
    
    def path_amplitude(k, path):
        """Amplitude for path through tree"""
        amp = 1.0
        sign = 1
        current_k = k
        for choice in path:
            if choice == 'L':
                amp *= fib(current_k - 1)
                current_k = current_k - 1
            else:
                amp *= fib(current_k - 2)
                sign *= -1  # Anti-correlation
                current_k = current_k - 2
        return amp * sign
    
    from itertools import product
    
    # All paths of given depth
    all_paths = list(product(['L', 'R'], repeat=depth))
    
    # Separate into L-starting and R-starting
    L_paths = [p for p in all_paths if p[0] == 'L']
    R_paths = [p for p in all_paths if p[0] == 'R']
    
    # Sum amplitudes (with proper signs from anti-correlation)
    alpha = sum(path_amplitude(k_root, p) for p in L_paths)
    beta = sum(path_amplitude(k_root, p) for p in R_paths)
    
    # Normalize
    norm = np.sqrt(alpha**2 + beta**2)
    
    return alpha/norm, beta/norm


print("\nEffective 2-particle state at different tree depths:")
print("-" * 60)
print(f"{'Depth':<8} {'α_eff':<12} {'β_eff':<12} {'2αβ':<12} {'Ratio α/β':<12}")
print("-" * 60)

for depth in range(1, 6):
    alpha, beta = effective_2particle_state(7, depth)
    two_ab = 2 * alpha * beta
    ratio = abs(alpha / beta) if beta != 0 else float('inf')
    print(f"{depth:<8} {alpha:<12.6f} {beta:<12.6f} {two_ab:<12.6f} {ratio:<12.4f}")

# ============================================================================
# BELL VIOLATION FROM EFFECTIVE STATE
# ============================================================================

print("\n" + "=" * 78)
print("BELL VIOLATION FROM EFFECTIVE MULTI-LEVEL STATE")
print("=" * 78)

def compute_bell_S_for_state(alpha, beta):
    """
    Compute maximum CHSH S for state |ψ⟩ = α|01⟩ + β|10⟩
    """
    def correlation(theta_a, theta_b):
        ca, cb = np.cos(theta_a), np.cos(theta_b)
        sa, sb = np.sin(theta_a), np.sin(theta_b)
        return -(alpha**2 + beta**2) * ca * cb + 2 * alpha * beta * sa * sb
    
    def CHSH(angles):
        a, ap, b, bp = angles
        return -(correlation(a, b) - correlation(a, bp) + correlation(ap, b) + correlation(ap, bp))
    
    # Optimize
    from scipy.optimize import minimize
    best_S = 0
    for _ in range(10):
        x0 = np.random.uniform(0, np.pi, 4)
        result = minimize(lambda x: -abs(CHSH(x)), x0, method='Nelder-Mead')
        if abs(CHSH(result.x)) > abs(best_S):
            best_S = CHSH(result.x)
    
    return abs(best_S)


print("\nBell S for effective multi-level states:")
print("-" * 60)
print(f"{'Depth':<8} {'2αβ_eff':<12} {'S_max':<12} {'vs Single':<15}")
print("-" * 60)

# Single level result
alpha_1, beta_1 = effective_2particle_state(7, 1)
S_single = compute_bell_S_for_state(alpha_1, beta_1)
print(f"{'1':<8} {2*alpha_1*beta_1:<12.6f} {S_single:<12.4f} {'(baseline)':<15}")

for depth in range(2, 6):
    alpha, beta = effective_2particle_state(7, depth)
    S = compute_bell_S_for_state(alpha, beta)
    improvement = (S - S_single) / S_single * 100
    print(f"{depth:<8} {2*alpha*beta:<12.6f} {S:<12.4f} {improvement:+.2f}%")

print(f"\nQuantum maximum: S = {2*np.sqrt(2):.4f}")
print(f"Classical bound: S = 2.0000")

# ============================================================================
# THE CONVERGENCE
# ============================================================================

print("\n" + "=" * 78)
print("ANALYZING THE CONVERGENCE")
print("=" * 78)

print("""
What happens as tree depth → ∞?

The effective entanglement parameter 2αβ should converge to something.
Let's see if it converges to a value that gives S = 2.83 (full QM)
or stays at S = 2.68 (single-level Fibonacci).
""")

alphas = []
betas = []
two_abs = []
S_values = []

for depth in range(1, 10):
    try:
        alpha, beta = effective_2particle_state(7, depth)
        S = compute_bell_S_for_state(alpha, beta)
        alphas.append(alpha)
        betas.append(beta)
        two_abs.append(2*alpha*beta)
        S_values.append(S)
    except:
        break

print(f"\nConvergence analysis:")
print(f"  2αβ approaches: {two_abs[-1]:.6f}")
print(f"  S approaches:   {S_values[-1]:.4f}")
print(f"  QM maximum:     {2*np.sqrt(2):.4f}")
print(f"  Single-level:   {S_values[0]:.4f}")

# The limit of 2αβ
# As depth → ∞, the ratio α/β → ?
# This depends on how the Fibonacci recursion compounds

print(f"\nRatio α/β approaches: {abs(alphas[-1]/betas[-1]):.6f}")
print(f"Compare to φ = {phi:.6f}")
print(f"Compare to 1 (equal weights) = 1.000000")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "=" * 78)
print("INTERPRETATION")
print("=" * 78)

final_S = S_values[-1] if S_values else 2.68

if final_S > 2.78:
    print(f"""
RESULT: Multi-level tree entanglement INCREASES S toward QM maximum!

Single level:  S = {S_values[0]:.4f}
Multi-level:   S = {final_S:.4f}
QM maximum:    S = {2*np.sqrt(2):.4f}

The cascading correlations through the tree BOOST the Bell violation!
This could explain why experiments achieve S ≈ 2.79.
""")
elif final_S > 2.70:
    print(f"""
RESULT: Multi-level effects give modest increase.

Single level:  S = {S_values[0]:.4f}
Multi-level:   S = {final_S:.4f}
QM maximum:    S = {2*np.sqrt(2):.4f}

Some improvement but not enough to fully explain Storz (S = 2.79).
""")
else:
    print(f"""
RESULT: Multi-level effects don't significantly change S.

Single level:  S = {S_values[0]:.4f}
Multi-level:   S = {final_S:.4f}
QM maximum:    S = {2*np.sqrt(2):.4f}

The tree structure converges to a fixed entanglement.
The Bell tension remains.
""")

print("\n" + "=" * 78)
print("ANALYSIS COMPLETE")
print("=" * 78)
