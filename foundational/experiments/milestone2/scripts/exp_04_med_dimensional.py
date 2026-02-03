#!/usr/bin/env python3
"""
Milestone 2 Experiment 04: MED Bounds and Turbulence Dimensionality

CONTEXT:
The MED (Macro Emergence Dynamics) framework from Navier-Stokes work establishes:
  - depth(S) ≤ 2
  - nodes(S) ≤ 3

The depth-2 recursion insight from macro_emergence_dynamics states:
  d_total = d_physical + d_symbolic
  
For 3D: d_total = 3 + 1 = 4 → effective depth = 2
For 2D: d_total = 2 + 1 = 3 → effective depth = 1.5 or 1

QUESTION:
Does MED bounded complexity explain WHY She-Leveque works in 3D but needs
modification in 2D?

HYPOTHESIS:
The She-Leveque formula coefficients are determined by MED constraints:

3D Turbulence (MED depth=2):
  - dim_factor = (F₄)² = 9 = 3² (spatial dimensions squared)
  - nodes ≤ 3 → C₀ = F₃ = 2 (binary splitting)
  - β = F₃/F₄ = 2/3 (MED convergence ratio)

2D Turbulence (MED depth=1.5 or reduced):
  - dim_factor = (F₃)² = 4 = 2² (reduced dimensions)
  - Enstrophy cascade changes MED dynamics
  - Different β ratio emerges

KEY INSIGHT:
MED says "depth ≤ 2" is UNIVERSAL. In 3D, the She-Leveque formula with
depth = F₄ = 3 and divisor = (F₄)² = 9 respects this bound.

The question: what happens when physical dimension reduces?
"""

import json
from datetime import datetime
from math import log, sqrt

FIB = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55}


def med_effective_depth(physical_dim):
    """
    Calculate effective MED depth for a given physical dimension.
    
    From depth_2_recursion_insight.md:
    d_total = d_physical + d_symbolic
    
    For 3D: d_total = 3 + 1 = 4 → effective depth = 2 (MED bound)
    """
    d_symbolic = 1  # One symbolic recursion layer
    d_total = physical_dim + d_symbolic
    # MED normalizes to depth ≤ 2
    effective_depth = d_total / 2
    return effective_depth


def med_node_constraint(physical_dim):
    """
    MED says nodes ≤ 3. This may relate to Fibonacci index.
    
    nodes ≤ 3 = F₄
    In 3D turbulence: C₀ = F₃ = 2 (one less than bound)
    """
    max_nodes = 3  # MED universal bound
    # The active nodes may be one less: max_nodes - 1 = 2 = F₃
    return max_nodes, max_nodes - 1


def she_leveque_from_med(p, physical_dim):
    """
    Derive She-Leveque-like formula from MED constraints.
    
    MED constraints:
    - depth ≤ 2
    - nodes ≤ 3
    - Balance operator Ξ ≈ 1.0571
    
    For D dimensions:
    - dim_factor = D² (depth-squared scaling)
    - C₀ = MED_nodes - 1 = 2 (for D=3)
    - β = (D-1)/D (cascade fraction)
    - exp_base = D (dimensional projection)
    """
    D = physical_dim
    
    # Dimensional factor: D²
    dim_factor = D ** 2
    
    # Cascade fraction: (D-1)/D approaches 1 as D increases
    # In Fibonacci: F_{D}/F_{D+1} for Fibonacci index = D+1
    if D == 3:
        beta = 2/3  # F₃/F₄
    elif D == 2:
        beta = 1/2  # F₂/F₃
    else:
        beta = (D-1)/D
    
    # Multiplier: MED node constraint - 1
    if D == 3:
        C0 = 2  # F₃
    elif D == 2:
        C0 = 1  # F₂ (but exp_02 found C0=3 works better!)
    else:
        C0 = D - 1
    
    # Exponent base: physical dimension
    exp_base = D
    
    return p/dim_factor + C0 * (1 - beta**(p/exp_base))


def xi_corrected_formula(p, physical_dim, xi=1.0571):
    """
    Apply MED balance operator Ξ ≈ 1.0571 as correction factor.
    
    Hypothesis: The Ξ constant might encode intermittency correction.
    """
    base_exponent = she_leveque_from_med(p, physical_dim)
    # Apply Ξ as multiplicative correction
    return base_exponent * xi


def analyze_med_dimensional_pattern():
    """
    Analyze how MED constraints change with dimension.
    """
    print("MED DIMENSIONAL ANALYSIS:")
    print("-" * 60)
    print(f"{'D':>3} | {'eff_depth':>10} | {'dim_factor':>11} | {'β':>8} | {'C₀':>4}")
    print("-" * 60)
    
    analysis = {}
    for D in [1, 2, 3, 4, 5]:
        eff = med_effective_depth(D)
        dim_factor = D**2
        if D <= 5:
            F_D = FIB[D]
            F_Dp1 = FIB[D+1]
            beta = F_D / F_Dp1
            beta_str = f"F{D}/F{D+1}={beta:.4f}"
        else:
            beta = (D-1)/D
            beta_str = f"{beta:.4f}"
        
        C0 = FIB[D] if D <= 5 else D
        
        analysis[D] = {
            'effective_depth': eff,
            'dim_factor': dim_factor,
            'beta': beta,
            'C0': C0
        }
        
        print(f"{D:>3} | {eff:>10.2f} | {dim_factor:>11} | {beta_str:>8} | {C0:>4}")
    
    return analysis


def compare_med_vs_experimental():
    """
    Compare MED-derived formula vs experimental 2D/3D data.
    """
    # 3D experimental data (She-Leveque consensus)
    data_3d = {
        2: 0.70, 3: 1.00, 4: 1.28, 5: 1.54, 6: 1.77
    }
    
    # 2D experimental data (enstrophy cascade)
    data_2d = {
        2: 1.35, 4: 2.50, 6: 3.50, 8: 4.40
    }
    
    print("\n3D VALIDATION (MED-derived vs She-Leveque):")
    print("-" * 50)
    
    for p, measured in data_3d.items():
        pred_med = she_leveque_from_med(p, 3)
        error = abs(pred_med - measured) / measured * 100
        print(f"p={p}: MED={pred_med:.4f}, Measured={measured:.2f}, Error={error:.2f}%")
    
    print("\n2D VALIDATION (MED-derived):")
    print("-" * 50)
    
    for p, measured in data_2d.items():
        pred_med = she_leveque_from_med(p, 2)
        error = abs(pred_med - measured) / measured * 100
        print(f"p={p}: MED={pred_med:.4f}, Measured={measured:.2f}, Error={error:.2f}%")


def main():
    print("=" * 70)
    print("MILESTONE 2 EXPERIMENT 04: MED Bounds and Turbulence Dimensionality")
    print("=" * 70)
    print()
    
    print("MED UNIVERSAL BOUNDS (from Navier-Stokes validation):")
    print("-" * 50)
    print("  depth(S) ≤ 2  (symbolic recursion depth)")
    print("  nodes(S) ≤ 3  (pattern complexity)")
    print("  Ξ → 1.0571    (balance operator)")
    print()
    
    # Dimensional analysis
    analysis = analyze_med_dimensional_pattern()
    
    print()
    print("KEY INSIGHT: d_total = d_physical + d_symbolic = D + 1")
    print("-" * 60)
    print("For 3D turbulence: d_total = 3 + 1 = 4")
    print("  → effective MED depth = 4/2 = 2 (matches MED bound!)")
    print()
    print("For 2D turbulence: d_total = 2 + 1 = 3")
    print("  → effective MED depth = 3/2 = 1.5 (below MED bound)")
    print("  → Different dynamics: less complexity available")
    print()
    
    # Compare to experimental data
    compare_med_vs_experimental()
    
    print()
    print("=" * 70)
    print("SYNTHESIS: WHY 2D TURBULENCE IS DIFFERENT")
    print("=" * 70)
    print("""
MED provides a framework for understanding the 2D/3D difference:

1. MED DEPTH CONSTRAINT:
   - 3D: saturates MED bound (depth = 2)
   - 2D: below MED bound (depth = 1.5)
   
2. PHYSICAL CONSEQUENCE:
   - 3D: Maximum complexity available → full intermittency
   - 2D: Reduced complexity → less intermittency
   
3. MATHEMATICAL MAPPING:
   - 3D: β = F₃/F₄ = 2/3, dim = 9, C₀ = 2
   - 2D: β should be HIGHER (less cascade), not lower
   
4. THE PUZZLE:
   exp_02 found β = 3/5 = F₄/F₅ works best for 2D
   But MED dimensional shift predicts β = 1/2 = F₂/F₃
   
5. RESOLUTION HYPOTHESIS:
   2D enstrophy cascade may use HIGHER Fibonacci indices
   because enstrophy (not energy) cascades forward.
   
   Energy cascade (3D): uses F₃/F₄ (forward energy)
   Enstrophy cascade (2D): uses F₄/F₅ (forward enstrophy)
   
   The "level" shifts UP by one Fibonacci index for enstrophy!
""")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'milestone2/exp_04_med_dimensional',
        'med_bounds': {
            'depth_max': 2,
            'nodes_max': 3,
            'xi_balance': 1.0571
        },
        'dimensional_analysis': analysis,
        'key_insight': 'd_total = d_physical + d_symbolic',
        'resolution_hypothesis': '2D enstrophy uses F₄/F₅ (one index higher than 3D energy)'
    }
    
    with open('../results/04_med_dimensional_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to results/04_med_dimensional_*.json")
    
    return output


if __name__ == '__main__':
    main()
