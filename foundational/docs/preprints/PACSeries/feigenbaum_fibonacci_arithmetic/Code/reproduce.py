#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 3: Feigenbaum Constants from Fibonacci Arithmetic
========================================================================================

Run all experiments to reproduce paper results.

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 1            # Run experiment 01 only
    python reproduce.py --list       # List available experiments

Experiments map to paper sections as follows:
    exp_01  → §2      (closed-form formulas — r∞, δ, α)
    exp_02  → §4      (exhaustive search — 1 in 280 billion)
    exp_03  → §5      (Möbius transformation structure of δ)
    exp_04  → §5.2    (cross-ratios of bifurcation cascade)
    exp_05  → §6      (high-precision Möbius perturbation series)
    exp_06  → §6      (M₁₀ theoretical framework, 1857)
    exp_07  → §7      (RBF self-closing formula for δ)
    exp_08  → §8      (universality across quadratic maps)
    exp_09  → §9      (cross-domain validation — 1 in 120 billion)

Dependencies: numpy, scipy, mpmath, matplotlib
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
