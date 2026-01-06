#!/usr/bin/env python3
"""
PAC Cosmology JWST Validation - Reproduction Script

Run all experiments or specific ones to reproduce the paper's results.

Usage:
    python reproduce.py          # Run all experiments
    python reproduce.py 3        # Run experiment 3
    python reproduce.py --list   # List available experiments
"""

import sys
import os
import subprocess
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
RESULTS_DIR = SCRIPT_DIR.parent / "Data" / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    1: ("exp_01_recursion_test.py", "PAC Recursion φ-Necessity Test"),
    2: ("exp_02_qbe_constraint.py", "QBE Constraint Validation"),
    3: ("exp_03_jwst_comparison.py", "JWST Mass Comparison (Original)"),
    "3b": ("exp_03b_jwst_redesigned.py", "JWST Forward Predictions (Redesigned)"),
    4: ("exp_04_predictions.py", "Future Predictions"),
    5: ("exp_05_eddington_comparison.py", "Eddington Limit Comparison"),
    6: ("exp_06_sec_dynamics.py", "SEC Dynamics Verification"),
    7: ("exp_07_lcdm_comparison.py", "ΛCDM vs PAC Comparison"),
    8: ("exp_08_falsification_criteria.py", "Falsification Criteria"),
    9: ("exp_09_expanded_analysis.py", "Expanded 10-Object Analysis"),
    10: ("exp_10_uhz1_heavy_seeds.py", "UHZ-1 Heavy Seed Analysis"),
}


def list_experiments():
    """Print available experiments."""
    print("\nAvailable experiments:")
    print("-" * 60)
    for key, (script, description) in sorted(EXPERIMENTS.items(), key=lambda x: str(x[0])):
        print(f"  {key:>3}: {description}")
        print(f"       ({script})")
    print("-" * 60)
    print("\nKey experiments for paper:")
    print("  3b - Forward predictions (main validation)")
    print("  7  - ΛCDM comparison (core result)")
    print("  8  - Falsification criteria")
    print("  9  - Expanded sample analysis")
    print("  10 - UHZ-1 heavy seed analysis")


def run_experiment(key):
    """Run a specific experiment."""
    if key not in EXPERIMENTS:
        print(f"Error: Experiment {key} not found.")
        list_experiments()
        return False
    
    script, description = EXPERIMENTS[key]
    script_path = EXPERIMENTS_DIR / script
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        print("Note: You may need to copy scripts from the source experiment folder.")
        print("See trace.yaml for source locations.")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running Experiment {key}: {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(SCRIPT_DIR))
    return result.returncode == 0


def run_all():
    """Run all experiments in order."""
    print("\nRunning all PAC Cosmology experiments...")
    print("="*60)
    
    # Run in logical order
    order = [1, 2, 6, "3b", 7, 8, 9, 10]
    
    results = {}
    for key in order:
        success = run_experiment(key)
        results[key] = "✓" if success else "✗"
    
    print("\n" + "="*60)
    print("Summary:")
    print("-"*60)
    for key, status in results.items():
        _, description = EXPERIMENTS[key]
        print(f"  {status} Experiment {key}: {description}")
    print("="*60)


def main():
    if len(sys.argv) < 2:
        run_all()
    elif sys.argv[1] == "--list":
        list_experiments()
    else:
        try:
            key = sys.argv[1]
            # Try to convert to int if it's a number
            try:
                key = int(key)
            except ValueError:
                pass  # Keep as string (e.g., "3b")
            run_experiment(key)
        except Exception as e:
            print(f"Error: {e}")
            list_experiments()


if __name__ == "__main__":
    main()
