#!/usr/bin/env python3
"""
Infodynamic Gravity - Master Runner

Unified interface for running all infodynamic gravity experiments and tests.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_validation_suite():
    """Run the complete validation test suite"""
    print("=" * 60)
    print("INFODYNAMIC GRAVITY VALIDATION SUITE")
    print("=" * 60)
    
    os.system("python tests/validation_tests.py")

def run_cosmic_web_experiment():
    """Run SEC-enhanced cosmic web formation experiment"""
    print("=" * 60) 
    print("SEC-ENHANCED COSMIC WEB EXPERIMENT")
    print("=" * 60)
    
    os.system("python experiments/sec_enhanced_cosmic_web.py")

def run_hierarchical_test():
    """Run hierarchical information field test"""
    print("=" * 60)
    print("HIERARCHICAL INFORMATION FIELD TEST") 
    print("=" * 60)
    
    os.system("python tests/test_hierarchical_information.py")

def run_scale_summary():
    """Run scale-dependent parameter summary"""
    print("=" * 60)
    print("SCALE-DEPENDENT PARAMETER SUMMARY")
    print("=" * 60)
    
    os.system("python tests/test_scale_dependent_summary.py")

def run_parameter_investigation():
    """Run parameter investigation experiments"""
    print("=" * 60)
    print("PARAMETER INVESTIGATION SUITE")
    print("=" * 60)
    print("Testing scale dependence, quantum floor reduction, and dimensional analysis")
    print()
    
    # Create parameter investigation script if it doesn't exist
    import os
    if not os.path.exists("experiments/parameter_investigation.py"):
        print("Creating parameter investigation framework...")
        with open("experiments/parameter_investigation.py", "w", encoding='utf-8') as f:
            f.write('''#!/usr/bin/env python3
"""
Parameter Investigation Framework

Testing extreme parameter values to understand their physical meaning.
Based on v3.0 arithmetic iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infodynamic_gravity import InfoGravityConfig

def test_kappa_scaling():
    """Test κ vs scale relationship"""
    print("Testing κ scaling with system size...")
    
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
    plt.savefig('../results/kappa_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"kappa scaling plot saved to results/kappa_scaling.png")

def test_beta_reduction():
    """Test quantum floor reduction"""
    print("Testing beta reduction while maintaining dark matter...")
    
    beta_values = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    dark_matter_fractions = []
    
    for beta in beta_values:
        config = InfoGravityConfig(
            kappa=5e46,
            beta_floor=beta,
            alpha_info=0.005857,
            lambda_0=30 * 3.086e19  # 30 kpc
        )
        
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
    plt.savefig('../results/beta_investigation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"beta investigation plot saved to results/beta_investigation.png")

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

if __name__ == "__main__":
    print("Parameter Investigation Suite")
    print("=" * 50)
    
    test_kappa_scaling()
    print()
    
    test_beta_reduction()
    print()
    
    dimensional_analysis()
    print()
    
    print("Investigation complete! Check results/ for plots.")
''')
    
    os.system("python experiments/parameter_investigation.py")

def run_all():
    """Run all experiments and tests"""
    print("🚀 Running complete infodynamic gravity experiment suite...")
    print()
    
    # Core validation
    run_validation_suite()
    print()
    
    # Scale-dependent summary  
    run_scale_summary()
    print()
    
    # Cosmic web formation
    run_cosmic_web_experiment()
    print()
    
    # Hierarchical information
    run_hierarchical_test()
    print()
    
    # Parameter investigation
    run_parameter_investigation()
    print()
    
    print("✅ All experiments completed!")

def show_menu():
    """Show interactive menu"""
    print("=" * 60)
    print("INFODYNAMIC GRAVITY EXPERIMENT RUNNER v3.0")
    print("=" * 60)
    print()
    print("1. Run Validation Suite")
    print("2. Run Cosmic Web Experiment") 
    print("3. Run Hierarchical Information Test")
    print("4. Run Scale-Dependent Summary")
    print("5. Run Parameter Investigation (NEW)")
    print("6. Run All Experiments")
    print("7. Exit")
    print()
    print("Note: v3.0 focuses on parameter iteration and investigation")

def main():
    """Main runner interface"""
    if len(sys.argv) > 1:
        # Command line mode
        arg = sys.argv[1].lower()
        if arg == "validation":
            run_validation_suite()
        elif arg == "cosmic":
            run_cosmic_web_experiment()
        elif arg == "hierarchical":
            run_hierarchical_test()
        elif arg == "scale":
            run_scale_summary()
        elif arg == "investigate" or arg == "params":
            run_parameter_investigation()
        elif arg == "all":
            run_all()
        else:
            print(f"Unknown command: {arg}")
            print("Usage: python run.py [validation|cosmic|hierarchical|scale|investigate|all]")
    else:
        # Interactive mode
        while True:
            show_menu()
            choice = input("Select option (1-7): ").strip()
            
            if choice == "1":
                run_validation_suite()
            elif choice == "2":
                run_cosmic_web_experiment()
            elif choice == "3":
                run_hierarchical_test()
            elif choice == "4":
                run_scale_summary()
            elif choice == "5":
                run_parameter_investigation()
            elif choice == "6":
                run_all()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-7.")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
