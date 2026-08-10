#!/usr/bin/env python3
"""
WorldSeed Evolutionary Architecture - Master Reproduction Script

This script reproduces all experiments from the paper:
"WorldSeed: Evolutionary Software Architecture Through PAC/SEC Dynamics"

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py --quick      # Quick validation run
    python reproduce.py --exp 1      # Run specific experiment

Author: Peter Lorne Groom
Date: 2026-01-24
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent paths for imports
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "core"))
sys.path.insert(0, str(SCRIPT_DIR / "experiments"))


def run_experiment_1_basic():
    """Experiment 1: Basic evolution with mock fitness."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Basic Evolution Engine")
    print("="*60)
    
    from exp_01_basic_evolution import main as exp1_main
    results = exp1_main()
    
    save_results("exp_01_basic_evolution", results)
    return results


def run_experiment_2_real_gaia():
    """Experiment 2: Real GAIA integration."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Real GAIA Integration")
    print("="*60)
    
    from exp_02_real_gaia_integration import main as exp2_main
    results = exp2_main()
    
    save_results("exp_02_real_gaia_integration", results)
    return results


def run_experiment_3_wikitext2():
    """Experiment 3: WikiText-2 evolution."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: WikiText-2 Evolution")
    print("="*60)
    
    from exp_03_wikitext2_evolution import main as exp3_main
    results = exp3_main()
    
    save_results("exp_03_wikitext2_evolution", results)
    return results


def save_results(experiment_name: str, results: dict):
    """Save experiment results to JSON."""
    results_dir = ROOT_DIR / "Data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {filepath}")


def generate_figures():
    """Generate all figures for the paper."""
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    from generate_figures import main as gen_figures
    gen_figures()
    
    print("✓ Figures saved to: Figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce WorldSeed experiments"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick validation run (reduced generations)"
    )
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3],
        help="Run specific experiment only"
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="Generate figures only (requires existing results)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("WORLDSEED EVOLUTIONARY ARCHITECTURE")
    print("Reproduction Script")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    if args.quick:
        os.environ["WORLDSEED_QUICK_MODE"] = "1"
    
    if args.figures:
        generate_figures()
        return
    
    if args.exp:
        experiments = {
            1: run_experiment_1_basic,
            2: run_experiment_2_real_gaia,
            3: run_experiment_3_wikitext2,
        }
        experiments[args.exp]()
    else:
        # Run all experiments
        run_experiment_1_basic()
        run_experiment_2_real_gaia()
        run_experiment_3_wikitext2()
        generate_figures()
    
    print("\n" + "="*60)
    print("REPRODUCTION COMPLETE")
    print("="*60)
    print("See Data/results/ for experiment outputs")
    print("See Figures/ for visualizations")


if __name__ == "__main__":
    main()
