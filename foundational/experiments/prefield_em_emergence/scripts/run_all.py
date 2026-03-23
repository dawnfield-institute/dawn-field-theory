#!/usr/bin/env python3
"""
Run All Experiments
===================

Executes all pre-field EM emergence experiments in sequence.

Usage:
    python run_all.py           # Run all experiments
    python run_all.py --quick   # Run quick validation only (exp_01)
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Run pre-field EM emergence experiments")
    parser.add_argument('--quick', action='store_true', help="Run only basic validation")
    parser.add_argument('--exp', type=int, help="Run specific experiment (1-4)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PRE-FIELD → 3D ELECTROMAGNETIC EMERGENCE")
    print("Full Experimental Suite")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().isoformat()}")
    
    results = {}
    
    if args.exp:
        # Run specific experiment
        experiments = {args.exp}
    elif args.quick:
        experiments = {1}
    else:
        experiments = {1, 2, 3, 4}
    
    if 1 in experiments:
        print("\n" + "=" * 70)
        from experiments.exp_01_basic_validation import run_experiment as exp_01
        results['exp_01'] = exp_01()
    
    if 2 in experiments:
        print("\n" + "=" * 70)
        from experiments.exp_02_parameter_sweep import run_experiment as exp_02
        results['exp_02'] = exp_02()
    
    if 3 in experiments:
        print("\n" + "=" * 70)
        from experiments.exp_03_deep_dive import run_experiment as exp_03
        results['exp_03'] = exp_03()
    
    if 4 in experiments:
        print("\n" + "=" * 70)
        from experiments.exp_04_long_evolution import run_experiment as exp_04
        results['exp_04'] = exp_04()
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 70)
    print(f"\nFinished: {datetime.now().isoformat()}")
    print(f"Experiments run: {len(results)}")
    
    if 'exp_02' in results:
        pl = results['exp_02'].get('power_law', {})
        if pl:
            print(f"\nKey Result - Power Law:")
            print(f"  {pl.get('formula', 'N/A')}")
            print(f"  R² = {pl.get('r_squared', 'N/A')}")
    
    return results


if __name__ == "__main__":
    main()
