#!/usr/bin/env python3
"""
Experiment 20: Koide Formula Q = F₃/F₄ = 2/3

The Koide formula relates charged lepton masses:
Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3

This is one of the most precise empirical relations in particle physics.
"""

import numpy as np
from constants import F3, F4, print_header, print_result

# Lepton masses (MeV/c²)
M_ELECTRON = 0.51099895  
M_MUON = 105.6583755
M_TAU = 1776.86

def koide_calculation():
    """Calculate Koide ratio from measured masses."""
    masses = [M_ELECTRON, M_MUON, M_TAU]
    
    sum_masses = sum(masses)
    sum_sqrt_masses = sum(np.sqrt(m) for m in masses)
    
    Q = sum_masses / (sum_sqrt_masses ** 2)
    
    return {
        'masses_MeV': masses,
        'sum_masses': sum_masses,
        'sum_sqrt_masses': sum_sqrt_masses,
        'Q_measured': Q,
        'Q_predicted': F3/F4,
        'error_pct': 100 * abs(Q - F3/F4) / (F3/F4)
    }

def fibonacci_interpretation():
    """Why Q = F₃/F₄ = 2/3?"""
    return {
        'F3': F3,
        'F4': F4,
        'ratio': F3/F4,
        'interpretation': 'MED bounds: depth=2, nodes=3 gives fundamental ratio',
        'geometric_meaning': '2/3 is the balance point of binary depth in ternary structure'
    }

def main():
    print_header("Experiment 20: Koide Formula Q = 2/3")
    
    k = koide_calculation()
    interp = fibonacci_interpretation()
    
    print(f"\nLepton masses (MeV): e={k['masses_MeV'][0]}, μ={k['masses_MeV'][1]:.2f}, τ={k['masses_MeV'][2]:.2f}")
    print(f"\nKoide ratio Q = (Σm)/(Σ√m)²")
    print(f"Q measured:  {k['Q_measured']:.10f}")
    print(f"Q predicted: {k['Q_predicted']:.10f} = F₃/F₄ = {F3}/{F4}")
    print(f"Error: {k['error_pct']:.4f}%")
    
    print(f"\nInterpretation: {interp['interpretation']}")
    print_result("Koide Q = 2/3", k['error_pct'] < 0.01)

if __name__ == "__main__":
    main()
