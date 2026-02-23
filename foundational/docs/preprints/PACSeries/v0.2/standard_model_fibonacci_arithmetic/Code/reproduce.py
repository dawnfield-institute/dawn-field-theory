#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 4: Standard Model Parameters from Fibonacci Arithmetic
==============================================================================================

Run all experiments to reproduce paper results.

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 1            # Run experiment 01 only
    python reproduce.py --list       # List available experiments

Experiments map to paper sections as follows:
    exp_01  → Section 3       (F₇=13 gauge group closure)
    exp_02  → Section 3.3     (SU(4)+ forbidden by Fibonacci filter)
    exp_03  → Section 4.1–4.2 (fine structure constant α to 5.7 ppm)
    exp_04  → Section 4.3     (Weinberg angle sin²θ_W = 3/13)
    exp_05  → Section 5       (Koide formula Q = 2/3 to 0.5 ppm)
    exp_06  → Section 6       (mass ratios from Fibonacci arithmetic)
    exp_07  → Section 8       (Bell correlation from PAC tree)
    exp_08  → Section 9       (She–Lévêque k = d × F_{d+1})
    exp_09  → Section 11      (gravity hierarchy F₁₈₃ ≈ 10³⁸)
    exp_10  → Section 14      (Z' boson prediction at 395 GeV)
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
