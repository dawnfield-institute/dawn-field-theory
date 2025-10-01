#!/usr/bin/env python3
"""
Pre-Field Recursion - Main Entry Point

Current Version: 2.2.0 (Resonance-Aware Convergence)

This is the primary interface for running pre-field recursion experiments.
For version comparisons and benchmarks, use test_suite.py instead.

Usage:
    python main.py                      # Run v2.2 with defaults
    python main.py --iterations 1000    # Extended run
    python main.py --seed 123           # Different initial state
    python main.py --version v20        # Run older version

Author: Dawn Field Institute
Date: October 1, 2025
Version: 2.2.0
"""

import numpy as np
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Ensure core is in path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core import (
    create_initial_state,
    RecursionOperator,
    AdaptiveRecursionOperator,
    PreFieldTransition
)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Pre-Field Recursion v2.2 - Resonance-Aware Convergence',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--version', choices=['v20', 'v21', 'v22'], default='v22',
                       help='Version to run (default: v22)')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of iterations (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--topology', default='mobius',
                       choices=['mobius', 'torus', 'klein'],
                       help='Topology type (default: mobius)')
    parser.add_argument('--size', type=int, default=100,
                       help='Field size (default: 100)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRE-FIELD RECURSION - DAWN FIELD INSTITUTE")
    print(f"Version: {args.version.upper()}")
    print("="*70)
    
    # Create initial state
    print(f"\n[Setup]")
    print(f"  Topology: {args.topology}")
    print(f"  Size: {args.size}")
    print(f"  Seed: {args.seed}")
    print(f"  Iterations: {args.iterations}")
    
    state = create_initial_state(
        size=args.size,
        topology=args.topology,
        seed=args.seed
    )
    
    print(f"  Initial PAC: {state.pac_residual:.6f}")
    
    # Create operator based on version
    if args.version == 'v20':
        operator = RecursionOperator(twist_rate=np.pi/2)
        print("\n[Operator] v2.0 Fixed-Rate Baseline")
    elif args.version == 'v21':
        operator = AdaptiveRecursionOperator(
            initial_twist=np.pi/2,
            resonance_aware=False
        )
        print("\n[Operator] v2.1 Adaptive (No Resonance)")
    else:  # v22
        operator = AdaptiveRecursionOperator(
            initial_twist=np.pi/2,
            resonance_aware=True
        )
        print("\n[Operator] v2.2 Resonance-Aware (Current)")
    
    # Run evolution
    print("\n[Running]")
    pac_history = []
    
    for i in range(args.iterations):
        state = operator.apply(state)
        pac_history.append(state.pac_residual)
        
        if args.verbose or (i + 1) % 100 == 0:
            if hasattr(operator, 'resonance_locked') and operator.resonance_locked:
                print(f"  Iteration {i+1}: PAC = {state.pac_residual:.6f} "
                      f"(locked, period={operator.detected_period:.1f})")
            else:
                print(f"  Iteration {i+1}: PAC = {state.pac_residual:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Final PAC: {state.pac_residual:.6f}")
    print(f"Initial PAC: {pac_history[0]:.6f}")
    print(f"Improvement: {(pac_history[0] - state.pac_residual) / pac_history[0] * 100:.1f}%")
    
    if hasattr(operator, 'resonance_locked'):
        print(f"\nResonance Status:")
        print(f"  Locked: {operator.resonance_locked}")
        if operator.resonance_locked:
            print(f"  Period: {operator.detected_period:.1f} iterations")
            print(f"  Frequency: {operator.detected_frequency:.4f} cycles/iter")
            print(f"  Twist Rate: {operator.twist_rate:.4f} rad")
    
    if hasattr(operator, 'adaptation_history'):
        print(f"\nAdaptation Statistics:")
        print(f"  Total adaptations: {len(operator.adaptation_history)}")
        print(f"  Final acceleration: {operator.acceleration_factor:.2f}x")
    
    print("\n✅ Run complete!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
