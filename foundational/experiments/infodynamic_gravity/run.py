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
    
    print("✅ All experiments completed!")

def show_menu():
    """Show interactive menu"""
    print("=" * 60)
    print("INFODYNAMIC GRAVITY EXPERIMENT RUNNER")
    print("=" * 60)
    print()
    print("1. Run Validation Suite")
    print("2. Run Cosmic Web Experiment") 
    print("3. Run Hierarchical Information Test")
    print("4. Run Scale-Dependent Summary")
    print("5. Run All Experiments")
    print("6. Exit")
    print()

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
        elif arg == "all":
            run_all()
        else:
            print(f"Unknown command: {arg}")
            print("Usage: python run.py [validation|cosmic|hierarchical|scale|all]")
    else:
        # Interactive mode
        while True:
            show_menu()
            choice = input("Select option (1-6): ").strip()
            
            if choice == "1":
                run_validation_suite()
            elif choice == "2":
                run_cosmic_web_experiment()
            elif choice == "3":
                run_hierarchical_test()
            elif choice == "4":
                run_scale_summary()
            elif choice == "5":
                run_all()
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-6.")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
