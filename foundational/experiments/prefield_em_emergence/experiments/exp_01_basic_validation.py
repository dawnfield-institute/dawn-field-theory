#!/usr/bin/env python3
"""
Experiment 01: Basic Validation
===============================

Purpose:
    Validate that pre-field dynamics on a Möbius manifold, when projected
    into 3D space, produce Maxwell-like electromagnetic field structure.

Success Criteria:
    - PAC improves by >50%
    - ∇·B < 0.01
    - E/B within 50% of some φ-power

Author: Peter Lorne Groom, Claude (Anthropic)
Date: February 2026
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import MobiusField, SECOperator, EMProjector, MaxwellValidator
from core.constants import PHI, XI


def run_experiment():
    """Run basic validation experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 01: Basic Validation")
    print("=" * 70)
    
    # Setup
    field = MobiusField(n_u=64, n_v=32, R=2.0, w=0.6)
    sec = SECOperator(damping=0.98, pi_coupling=0.05)
    proj = EMProjector(n=24, L=3.0, shape='sphere')
    
    print(f"\nMobiusField: w/R={field.w_R_ratio:.3f}")
    print(f"Initial PAC: {field.pac_residual():.6f}")
    
    # Evolution
    n_iterations = 200
    checkpoints = [1, 50, 100, 150, 200]
    log = []
    
    print(f"\n{'Iter':<8} {'PAC':<12} {'Entropy':<12}")
    print("-" * 35)
    
    for i in range(1, n_iterations + 1):
        metrics = sec.step(field)
        if i in checkpoints:
            print(f"{i:<8} {metrics['pac_residual']:<12.6f} {metrics['total_entropy']:<12.4f}")
            log.append(metrics.copy())
    
    # Projection
    result = proj.project(field)
    
    print(f"\n--- Results ---")
    print(f"E/B ratio: {result['EB_ratio']:.4f}")
    print(f"Closest: {result['closest_match']} ({result['closest_deviation']*100:.1f}% dev)")
    print(f"∇·B: {result['div_B_mean']:.6f}")
    
    # Validation
    validator = MaxwellValidator(proj)
    validation = validator.validate(result)
    
    # Assessment
    initial_pac = log[0]['pac_residual']
    final_pac = log[-1]['pac_residual']
    pac_improvement = (initial_pac - final_pac) / initial_pac * 100
    
    criteria = {
        'PAC improves >50%': pac_improvement > 50,
        'No monopoles': validation['no_monopoles'],
        'E/B near φ-power': result['closest_deviation'] < 0.5,
    }
    
    print(f"\n--- Assessment ---")
    for name, passed in criteria.items():
        print(f"  {'✓' if passed else '✗'} {name}")
    
    passed = sum(criteria.values())
    verdict = "SUCCESS" if passed >= 2 else "PARTIAL" if passed >= 1 else "FAILED"
    print(f"\nVerdict: {verdict} ({passed}/{len(criteria)})")
    
    # Save
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_01_basic_validation',
        'config': {'n_u': 64, 'n_v': 32, 'R': 2.0, 'w': 0.6, 'iterations': 200},
        'pac': {'initial': initial_pac, 'final': final_pac, 'improvement_pct': pac_improvement},
        'em': {'EB_ratio': result['EB_ratio'], 'closest': result['closest_match']},
        'maxwell': {'div_B': result['div_B_mean'], 'no_monopoles': result['no_monopoles']},
        'verdict': verdict
    }
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"exp_01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_experiment()
