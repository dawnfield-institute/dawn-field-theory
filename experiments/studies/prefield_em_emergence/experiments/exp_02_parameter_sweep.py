#!/usr/bin/env python3
"""
Experiment 02: Parameter Sweep
==============================

Purpose:
    Sweep Möbius geometry parameters to discover the relationship
    between w/R ratio and E/B ratio.

Key Finding:
    E/B = φ^(-4.42 × w/R + 2.34)
    R² = 0.9764

Author: Peter Lorne Groom, Claude (Anthropic)
Date: February 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import MobiusField, SECOperator, EMProjector
from core.constants import PHI, PHI_SQ, XI


def run_experiment():
    """Run parameter sweep experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 02: Parameter Sweep")
    print("=" * 70)
    print("\nSweeping w/R ratio to find E/B = f(w/R) relationship")
    
    # Configuration
    R = 2.0
    w_values = np.linspace(0.2, 1.2, 21)
    n_iterations = 150
    
    proj = EMProjector(n=20, L=3.0)
    results = []
    
    print(f"\n{'w/R':<8} {'w':<6} {'E/B':<10} {'φ-power':<10} {'Closest':<8}")
    print("-" * 50)
    
    for w in w_values:
        # Initialize and evolve
        field = MobiusField(n_u=48, n_v=24, R=R, w=w)
        sec = SECOperator(damping=0.98, pi_coupling=0.05)
        
        for _ in range(n_iterations):
            sec.step(field)
        
        # Project and analyze
        em = proj.project(field)
        wR = w / R
        
        # Compute φ-power
        phi_power = np.log(em['EB_ratio']) / np.log(PHI)
        
        results.append({
            'w': float(w),
            'w_R': float(wR),
            'EB_ratio': em['EB_ratio'],
            'phi_power': float(phi_power),
            'closest': em['closest_match'],
            'deviation': em['closest_deviation'],
            'pac_final': float(field.pac_residual())
        })
        
        print(f"{wR:<8.3f} {w:<6.2f} {em['EB_ratio']:<10.4f} {phi_power:<10.2f} {em['closest_match']:<8}")
    
    # Fit power law
    wR_values = np.array([r['w_R'] for r in results])
    powers = np.array([r['phi_power'] for r in results])
    
    # Linear regression: power = slope * wR + intercept
    A = np.vstack([wR_values, np.ones(len(wR_values))]).T
    slope, intercept = np.linalg.lstsq(A, powers, rcond=None)[0]
    
    # R² calculation
    powers_fit = slope * wR_values + intercept
    ss_res = ((powers - powers_fit) ** 2).sum()
    ss_tot = ((powers - powers.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot
    
    # Find optimal w/R for E/B = φ (power = 1)
    optimal_wR = (1 - intercept) / slope
    
    print(f"\n{'='*50}")
    print("POWER LAW FIT")
    print(f"{'='*50}")
    print(f"\n  E/B = φ^({slope:.2f} × w/R + {intercept:.2f})")
    print(f"  R² = {r_squared:.4f}")
    print(f"\n  E/B = φ when w/R = {optimal_wR:.3f}")
    
    # Find best match to φ
    best = min(results, key=lambda x: abs(x['EB_ratio'] - PHI))
    print(f"\n  Best experimental match to φ:")
    print(f"    w/R = {best['w_R']:.3f}")
    print(f"    E/B = {best['EB_ratio']:.4f}")
    print(f"    Deviation = {abs(best['EB_ratio'] - PHI) / PHI * 100:.2f}%")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_02_parameter_sweep',
        'config': {'R': R, 'n_iterations': n_iterations, 'n_points': len(w_values)},
        'power_law': {
            'formula': f"E/B = φ^({slope:.2f} × w/R + {intercept:.2f})",
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'optimal_wR': float(optimal_wR)
        },
        'best_phi_match': {
            'w_R': best['w_R'],
            'EB_ratio': best['EB_ratio'],
            'deviation_pct': abs(best['EB_ratio'] - PHI) / PHI * 100
        },
        'sweep_results': results
    }
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"exp_02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_experiment()
