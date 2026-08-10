#!/usr/bin/env python3
"""
Parameter Investigation Framework

Testing extreme parameter values to understand their physical meaning.
Based on v3.0 arithmetic iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_kappa_scaling():
    """Test kappa vs scale relationship"""
    print("Testing kappa scaling with system size...")
    
    scales = np.logspace(0, 3, 20)  # 1 to 1000 kpc
    kappa_values = []
    
    for scale in scales:
        # Find minimum kappa for structure formation at this scale
        # This is a placeholder - implement actual test
        kappa_min = 5e46 * (scale / 30)**1.5  # Hypothesis: kappa scales with scale^1.5
        kappa_values.append(kappa_min)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(scales, kappa_values, 'bo-', alpha=0.7)
    plt.xlabel('System Scale (kpc)')
    plt.ylabel('Required kappa')
    plt.title('kappa Scaling Investigation')
    plt.grid(True, alpha=0.3)
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/kappa_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"kappa scaling plot saved to results/kappa_scaling.png")
    return True

def test_beta_reduction():
    """Test quantum floor reduction"""
    print("Testing beta reduction while maintaining dark matter...")
    
    beta_values = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    dark_matter_fractions = []
    
    for beta in beta_values:
        # Simplified test - implement full simulation
        # Lower beta should reduce dark matter unless compensated
        dm_fraction = 0.5 * (beta / 3.0)  # Placeholder relationship
        dark_matter_fractions.append(dm_fraction)
    
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, dark_matter_fractions, 'ro-', alpha=0.7, linewidth=2)
    plt.axhline(y=0.27, color='k', linestyle='--', alpha=0.5, label='Observed DM fraction')
    plt.xlabel('Quantum Floor beta')
    plt.ylabel('Dark Matter Fraction')
    plt.title('Quantum Floor vs Dark Matter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/beta_investigation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"beta investigation plot saved to results/beta_investigation.png")
    return True

def dimensional_analysis():
    """Analyze dimensional structure of parameters"""
    print("Performing dimensional analysis...")
    
    # Physical constants
    c = 2.998e8  # m/s
    G = 6.674e-11  # m³/kg/s²
    hbar = 1.055e-34  # J⋅s
    k_B = 1.381e-23  # J/K
    
    # Scales
    planck_length = np.sqrt(hbar * G / c**3)  # ~1.6e-35 m
    planck_force = c**4 / G  # ~1.2e44 N
    cosmic_scale = 1e26  # ~observable universe diameter
    
    print(f"Planck length: {planck_length:.2e} m")
    print(f"Planck force: {planck_force:.2e} N")
    print(f"Cosmic scale: {cosmic_scale:.2e} m")
    
    # Scale ratio
    scale_ratio = cosmic_scale / planck_length
    print(f"Scale ratio (cosmic/Planck): {scale_ratio:.2e}")
    
    # Information energy scale at CMB temperature
    T_cmb = 2.7  # K
    info_energy = k_B * T_cmb * np.log(2)
    print(f"Information bit energy at CMB: {info_energy:.2e} J")
    
    # Dimensional estimate for kappa
    kappa_estimate = planck_force * scale_ratio**2 / info_energy
    print(f"Dimensional kappa estimate: {kappa_estimate:.2e}")
    print(f"Actual kappa used: 5e46")
    print(f"Ratio: {5e46 / kappa_estimate:.2f}")
    return True

if __name__ == "__main__":
    print("Parameter Investigation Suite")
    print("=" * 50)
    
    success = True
    
    try:
        test_kappa_scaling()
        print()
    except Exception as e:
        print(f"Kappa scaling test failed: {e}")
        success = False
    
    try:
        test_beta_reduction()
        print()
    except Exception as e:
        print(f"Beta reduction test failed: {e}")
        success = False
    
    try:
        dimensional_analysis()
        print()
    except Exception as e:
        print(f"Dimensional analysis failed: {e}")
        success = False
    
    if success:
        print("Investigation complete! Check results/ for plots.")
    else:
        print("Investigation completed with some errors.")
