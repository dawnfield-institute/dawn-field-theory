#!/usr/bin/env python3
"""
Experiment 04: Long Evolution
=============================

Purpose:
    Test whether E/B ratio converges to φ with sufficient evolution time.

Question:
    Does the system reach true equilibrium where E/B = φ exactly?

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
from core.constants import PHI


def run_experiment():
    """Run long evolution experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 04: Long Evolution to Equilibrium")
    print("=" * 70)
    print("\nQuestion: Does E/B converge to φ with enough time?")
    
    # Use near-optimal geometry
    R, w = 2.0, 0.55
    field = MobiusField(n_u=64, n_v=32, R=R, w=w)
    sec = SECOperator(damping=0.99, pi_coupling=0.03)  # Gentler for stability
    proj = EMProjector(n=20, L=3.0)
    
    print(f"\nGeometry: R={R}, w={w}, w/R={w/R:.3f}")
    
    checkpoints = [100, 250, 500, 750, 1000, 1500, 2000, 3000]
    log = []
    
    print(f"\n{'Iter':<8} {'PAC':<12} {'E/B':<12} {'φ-dev%':<10} {'Δ(E/B)':<10}")
    print("-" * 55)
    
    prev_eb = None
    
    for target in checkpoints:
        # Evolve to checkpoint
        while sec.iteration < target:
            sec.step(field)
        
        # Project and measure
        em = proj.project(field)
        eb = em['EB_ratio']
        phi_dev = abs(eb - PHI) / PHI * 100
        pac = field.pac_residual()
        
        delta = f"{eb - prev_eb:+.4f}" if prev_eb else "---"
        prev_eb = eb
        
        log.append({
            'iteration': target,
            'pac': float(pac),
            'EB_ratio': float(eb),
            'phi_deviation': float(phi_dev)
        })
        
        print(f"{target:<8} {pac:<12.6f} {eb:<12.4f} {phi_dev:<10.2f} {delta:<10}")
    
    # Analyze convergence
    eb_values = [l['EB_ratio'] for l in log]
    eb_late = eb_values[-3:]
    eb_std = np.std(eb_late)
    
    print(f"\n{'='*55}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*55}")
    print(f"\nFinal E/B: {eb_values[-1]:.4f}")
    print(f"Target φ: {PHI:.4f}")
    print(f"Final deviation: {log[-1]['phi_deviation']:.2f}%")
    print(f"\nLate-stage stability (std of last 3): {eb_std:.4f}")
    
    if eb_std < 0.02:
        print("→ CONVERGED")
        if log[-1]['phi_deviation'] < 5:
            print(f"→ Converged NEAR φ (within {log[-1]['phi_deviation']:.1f}%)")
        else:
            print(f"→ Converged but NOT at φ")
    else:
        # Check trend
        trend = "rising toward φ" if eb_values[-1] > eb_values[-2] else "falling away from φ"
        print(f"→ Still evolving ({trend})")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_04_long_evolution',
        'config': {'R': R, 'w': w, 'w_R': w/R, 'max_iterations': checkpoints[-1]},
        'final': {
            'EB_ratio': eb_values[-1],
            'phi_deviation_pct': log[-1]['phi_deviation'],
            'pac': log[-1]['pac'],
            'converged': eb_std < 0.02
        },
        'evolution_log': log
    }
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"exp_04_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_experiment()
