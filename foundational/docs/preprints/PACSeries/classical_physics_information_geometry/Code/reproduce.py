#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 5: Classical Physics from Information Geometry
=====================================================================================

Run all experiments to reproduce paper results.

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 1            # Run experiment 01 only
    python reproduce.py --list       # List available experiments

Experiments map to paper sections as follows:
    exp_01  → Section 2       (SEC wave equation → speed of light)
    exp_02  → Section 3       (five independent paths to D=3)
    exp_03  → Section 4       (curl from depth-2 projection)
    exp_04  → Section 5       (charge quantization as winding number)
    exp_05  → Section 6       (SEC–Navier–Stokes equivalence)
    exp_06  → Section 7       (Casimir 240 and Mersenne dimensions)
    exp_07  → Section 8       (speculative gravity extension)
    exp_08  → Section 9       (Ξ = 1 + π/55 derivation)
    exp_09  → Section 12      (falsification conditions)
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
            for s in scripts:
                print(f"  - {s.name}")
            return

        # Run specific experiment
        exp_num = sys.argv[1].zfill(2)
        scripts = [s for s in scripts if f"exp_{exp_num}" in s.name]

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
