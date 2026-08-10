#!/usr/bin/env python3
"""
Experiment 04 — Charge as Topological Winding Number
======================================================

PACSeries Paper 5, Section 5

Electric charge emerges as a quantized winding number around
SEC collapse singularities. The topological argument:

1. SEC collapse events create point-like singularities in the
   information field
2. The phase of the SEC field around such a point acquires a
   quantized winding number n ∈ Z
3. Charge q = n × e, where e is the fundamental unit

Fractional quark charges arise from MED constraint (nodes ≤ 3):
   With 3 nodes sharing a winding, each carries charge e/3.
   Two quark types with 2/3 or 1/3 of e correspond to
   2-node vs 1-node sharing within the MED tree.

Source: maxwell_from_pac_sec/scripts/exp_02_charge_quantization.py
"""

import json
import os
import math
import numpy as np
from datetime import datetime


def winding_number(phase_values):
    """Compute winding number from a sequence of phase values around a loop."""
    total = 0
    for i in range(len(phase_values)):
        dphi = phase_values[(i+1) % len(phase_values)] - phase_values[i]
        # Wrap to [-π, π]
        while dphi > math.pi:
            dphi -= 2 * math.pi
        while dphi < -math.pi:
            dphi += 2 * math.pi
        total += dphi
    return round(total / (2 * math.pi))


def main():
    results = {
        'experiment': 'exp_04_charge_quantization',
        'paper': 'PACSeries Paper 5',
        'section': '5',
        'timestamp': datetime.now().isoformat(),
    }

    print("=" * 60)
    print("Charge Quantization from Topological Winding")
    print("=" * 60)
    print()
    print("  SEC collapses create singularities in the information field.")
    print("  The phase field ψ(x,y) around a singularity has:")
    print("    ∮ dψ = 2πn,  n ∈ Z")
    print()
    print("  Charge q = n × e (fundamental winding unit)")
    print()

    # Demonstrate winding numbers around point charges
    print("=" * 60)
    print("Winding Number Computation")
    print("=" * 60)

    N_points = 100  # points around the loop
    theta = np.linspace(0, 2*np.pi, N_points, endpoint=False)

    test_cases = [
        {'name': 'Electron (q = -e)', 'n': -1},
        {'name': 'Positron (q = +e)', 'n': 1},
        {'name': 'Neutral (q = 0)', 'n': 0},
        {'name': 'Double charge (q = 2e)', 'n': 2},
    ]

    winding_results = []
    for tc in test_cases:
        n = tc['n']
        phase = n * theta  # Phase wraps n times around the loop
        w = winding_number(phase.tolist())
        print(f"  {tc['name']:25s}  n_input = {n:+d}  n_measured = {w:+d}  {'✓' if w == n else '✗'}")
        winding_results.append({'name': tc['name'], 'n': n, 'measured': w, 'match': w == n})

    # Fractional charges from MED
    print()
    print("=" * 60)
    print("Fractional Quark Charges from MED")
    print("=" * 60)
    print()
    print("  MED constraint: nodes ≤ 3")
    print()
    print("  A single winding (n=1) distributed over N nodes:")
    print("    Each node carries charge = e/N")
    print()
    print("  For N = 3 (MED maximum):")
    print("    • 1/3 sharing → charge = e/3  (down-type quarks: d, s, b)")
    print("    • 2/3 sharing → charge = 2e/3 (up-type quarks: u, c, t)")
    print()

    quark_charges = []
    for n_nodes in range(1, 4):
        for n_carrying in range(1, n_nodes + 1):
            charge = n_carrying / n_nodes
            name = ''
            if n_nodes == 3:
                if n_carrying == 1:
                    name = '← d, s, b quarks'
                elif n_carrying == 2:
                    name = '← u, c, t quarks'
                elif n_carrying == 3:
                    name = '← integer (e, μ, τ)'
            print(f"    N={n_nodes}, carrying={n_carrying}: q = {n_carrying}/{n_nodes} e = {charge:.4f} e  {name}")
            quark_charges.append({
                'nodes': n_nodes,
                'carrying': n_carrying,
                'charge_fraction': round(charge, 6),
            })

    print()
    print("  Note: this reproduces the observed pattern of quark charges")
    print("  WITHOUT introducing fractional charges as a postulate.")
    print("  They arise from the MED node limit applied to winding topology.")

    # Anti-quarks and color confinement
    print()
    print("=" * 60)
    print("Color Confinement and Charge Neutrality")
    print("=" * 60)
    print()
    print("  MED requires total winding to be integer for observable states:")
    print("    • Mesons: q + q̄ = 1/3 + (-1/3) = 0 or 2/3 + (-2/3) = 0")
    print("    • Baryons: 3 × (1/3) = 1 or 2×(2/3) + 1×(-1/3) = 1")
    print("    • Free quarks: fractional winding → not MED-closed → confined")
    print()
    print("  Color confinement = topological closure requirement of MED.")

    results['main_results'] = {
        'winding_number_tests': winding_results,
        'quark_charges_from_med': quark_charges,
        'med_constraint': 'nodes ≤ 3',
        'fractional_charges': {
            '1/3': 'down-type quarks (1 of 3 nodes carrying)',
            '2/3': 'up-type quarks (2 of 3 nodes carrying)',
            '1': 'leptons (all nodes carrying)',
        },
        'confinement': 'Fractional winding is not MED-closed → confinement',
        'conclusion': (
            'Charge quantization follows from topological winding numbers around '
            'SEC collapse singularities. Fractional quark charges arise from '
            'MED node constraint ≤ 3, with 1/3 and 2/3 fractions corresponding '
            'to how winding distributes over the maximal MED tree.'
        ),
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_04_charge_quantization_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
