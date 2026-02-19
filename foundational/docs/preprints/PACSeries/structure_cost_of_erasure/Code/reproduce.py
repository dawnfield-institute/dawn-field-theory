#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 1: The Structure Cost of Erasure
======================================================================

Run all experiments to reproduce paper results.

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 1            # Run experiment 01 only
    python reproduce.py --list       # List available experiments

Experiments map to paper sections as follows:
    exp_01  → Sections 3–4   (core Landauer erasure)
    exp_02  → Section 4.2    (temperature independence)
    exp_03  → Section 6      (ratio analysis → ln(φ))
    exp_04  → Section 6      (robustness across seeds)
    exp_05  → Section 6      (decay rate sweep)
    exp_06  → Section 5      (gauge topologies)
    exp_07  → Section 5      (Lie algebra entropy)
    exp_08  → Section 6, 8   (falsification suite)
    exp_09  → Section 12.3   (conservative RBF binding)
    exp_10  → Section 10     (thermodynamic cascade)
    exp_11  → Section 11     (time as computation)
    exp_12  → Section 9.2    (causal lag test)
    exp_13  → Section 9.2    (causal lag falsification)
    exp_14  → Section 9.2    (PAC ratio vs magnitude)
    exp_15  → Section 5, 15  (gauge hierarchy prediction)
    exp_16  → Section 6, 15  (ln(φ) derivation)
    exp_19  → Section 6, 15  (coupling efficiency sweep → ln(φ) proximity)
    exp_23  → Section 15.3   (precision tightening: N=5M → 0.15% from ln(φ))
    exp_25  → Section 15.3   (full stack validation: 6 layers, thermal init)
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
