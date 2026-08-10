#!/usr/bin/env python3
"""
Reproduction Script — PACSeries Paper 2: The Balance Constant and Its Decomposition
=====================================================================================

Run all experiments to reproduce paper results.

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 1            # Run experiment 01 only
    python reproduce.py --list       # List available experiments

Experiments map to paper sections as follows:
    exp_01  → §4.1    (Ξ derivation contest — three candidates)
    exp_02  → §4.2    (exact k = 10.0121 derivation)
    exp_03  → §9.1    (γ as emergence surplus — decomposition)
    exp_04  → §9.2    (γ falsification — alternative constants)
    exp_05  → §3.2    (universal decomposition — p = 0.00376)
    exp_06  → §5      (cellular automata Class IV clustering)
    exp_07  → §6.1    (prime sieve PAC conservation)
    exp_08  → §6.3    (three-phase decomposition)
    exp_09  → §6.3    (p=3 reconciliation — 82.1% φ-carrier)
    exp_10  → §8      (base invariance across 11 bases)
    exp_11  → §8      (Zeckendorf validation — F₁₀ significance)
    exp_12  → §9.3    (PAC→MED theorem — depth ≤ 2, milestone3)
    exp_13  → §9.3    (MED depth criticality — d ≈ 3.25, milestone3)
    exp_14  → §9.3    (Fibonacci–MED complementarity, milestone3)
    exp_15  → §3, §7.1 (Möbius field dynamics — Ξ_L2 = 1.0581, reality-engine)
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
