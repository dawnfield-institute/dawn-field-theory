#!/usr/bin/env python3
"""
ML Validation Reproduction Script
=================================

Run all experiments to reproduce paper results:
  - exp_01: Pythia phi-convergence (p=0.0014)
  - exp_02: GPT-2 inference dynamics (entropy ratio ≈ 1.0)
  - exp_03: GPT-2 generation dynamics (entropy < 1/φ)

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 1            # Run exp_01 only
    python reproduce.py --list       # List available experiments
    
NOTE: exp_01 (Pythia) downloads ~30GB of model checkpoints.
      Recommend running exp_02 and exp_03 first for quick validation.
"""

import sys
import subprocess
from pathlib import Path


def main():
    scripts_dir = Path(__file__).parent / "experiments"
    
    if not scripts_dir.exists():
        print("No experiments directory found.")
        return
    
    scripts = sorted(scripts_dir.glob("exp_*.py"))
    
    if not scripts:
        print("No experiment scripts found.")
        return
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            print("Available experiments:")
            print("  exp_01 - Pythia phi-convergence (requires ~30GB download)")
            print("  exp_02 - GPT-2 inference entropy dynamics")
            print("  exp_03 - GPT-2 generation SEC dynamics")
            return
        
        # Run specific experiment
        exp_num = sys.argv[1].zfill(2)
        scripts = [s for s in scripts if f"exp_{exp_num}" in s.name or f"exp_0{exp_num}" in s.name]
        
        if not scripts:
            print(f"No experiment {exp_num} found.")
            return
    
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running: {script.name}")
        print('='*60)
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"\n[WARNING] {script.name} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
