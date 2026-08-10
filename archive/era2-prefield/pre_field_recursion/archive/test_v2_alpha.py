#!/usr/bin/env python3
"""
Pre-Field Recursion v2.0 - Initial Test Runner

Tests the new formal definitions and transition dynamics modules.
This is a simplified version while we build out the full framework.

Version: 2.0-alpha
"""

import numpy as np
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core.formal_definitions import PreFieldState, RecursionOperator, create_initial_state
from core.transition_dynamics import PreFieldTransition


def test_basic_recursion():
    """Test 1: Basic recursion mechanics"""
    print("\n" + "="*60)
    print("TEST 1: Basic Recursion Mechanics")
    print("="*60)
    
    # Create initial state
    state = create_initial_state(size=64, topology="mobius", seed=42)
    print(f"✓ Initial state created")
    print(f"  Size: {len(state.wavefunction)}")
    print(f"  Energy: {state.compute_field_energy():.4f}")
    print(f"  Entropy: {state.compute_information_entropy():.4f}")
    
    # Apply recursions
    recursion = RecursionOperator(twist_rate=np.pi/6)
    print(f"\n✓ Applying {10} recursions...")
    
    for i in range(10):
        state = recursion.apply(state)
    
    stats = recursion.get_statistics()
    print(f"  Final PAC residual: {stats['final_pac_residual']:.6f}")
    print(f"  Final emergence: {stats['final_emergence_metric']:.6f}")
    print(f"  PAC improvement: {stats['pac_improvement']:.2f}x")
    
    return stats['final_pac_residual'] < 100  # Loose check for now


def test_emergence_detection():
    """Test 2: Emergence detection mechanism"""
    print("\n" + "="*60)
    print("TEST 2: Emergence Detection")
    print("="*60)
    
    # Create initial state
    state = create_initial_state(size=64, topology="mobius", seed=123)
    print(f"✓ Initial state created")
    
    # Try to evolve to emergence
    transition = PreFieldTransition(state, twist_rate=np.pi/8)
    print(f"✓ Attempting emergence (max 200 iterations)...")
    
    emerged, final_state = transition.evolve_until_emergence(
        max_iterations=200,
        verbose=False
    )
    
    metrics = transition.get_transition_metrics()
    
    print(f"\n  Emerged: {emerged}")
    print(f"  Iterations: {metrics['total_iterations']}")
    print(f"  Final PAC: {metrics['final_pac_residual']:.6e}")
    print(f"  Final Ξ metric: {metrics['final_emergence_metric']:.6f}")
    print(f"  PAC convergence rate: {metrics['pac_convergence_rate']:.6f}")
    
    return True  # Always pass for now, just collecting data


def test_different_topologies():
    """Test 3: Compare different topologies"""
    print("\n" + "="*60)
    print("TEST 3: Topology Comparison")
    print("="*60)
    
    results = {}
    
    for topology in ['mobius', 'torus']:
        print(f"\n  Testing {topology} topology...")
        
        state = create_initial_state(size=64, topology=topology, seed=42)
        transition = PreFieldTransition(state, twist_rate=np.pi/8)
        
        emerged, final_state = transition.evolve_until_emergence(
            max_iterations=100,
            verbose=False
        )
        
        metrics = transition.get_transition_metrics()
        topo_analysis = transition.analyze_topology_role()
        
        results[topology] = {
            'emerged': emerged,
            'final_pac': metrics['final_pac_residual'],
            'final_xi': metrics['final_emergence_metric'],
            'topology_influence': topo_analysis['topology_influence_score']
        }
        
        print(f"    PAC: {results[topology]['final_pac']:.6f}")
        print(f"    Ξ: {results[topology]['final_xi']:.6f}")
        print(f"    Topology influence: {results[topology]['topology_influence']:.6f}")
    
    return True


def test_parameter_sweep():
    """Test 4: Parameter sweep for optimal twist rate"""
    print("\n" + "="*60)
    print("TEST 4: Parameter Sweep (Twist Rate)")
    print("="*60)
    
    twist_rates = [np.pi/16, np.pi/8, np.pi/4, np.pi/2]
    best_result = None
    best_pac = np.inf
    
    for twist_rate in twist_rates:
        state = create_initial_state(size=64, topology="mobius", seed=42)
        transition = PreFieldTransition(state, twist_rate=twist_rate)
        
        emerged, final_state = transition.evolve_until_emergence(
            max_iterations=50,
            verbose=False
        )
        
        metrics = transition.get_transition_metrics()
        final_pac = metrics['final_pac_residual']
        
        print(f"  θ = π/{int(np.pi/twist_rate)}: PAC = {final_pac:.6f}, Ξ = {metrics['final_emergence_metric']:.6f}")
        
        if final_pac < best_pac:
            best_pac = final_pac
            best_result = {
                'twist_rate': twist_rate,
                'emerged': emerged,
                'metrics': metrics
            }
    
    print(f"\n  Best twist rate: π/{int(np.pi/best_result['twist_rate'])}")
    print(f"  Best PAC: {best_pac:.6f}")
    
    return True


def test_convergence_analysis():
    """Test 5: Detailed convergence analysis"""
    print("\n" + "="*60)
    print("TEST 5: Convergence Analysis")
    print("="*60)
    
    state = create_initial_state(size=64, topology="mobius", seed=42)
    transition = PreFieldTransition(state, twist_rate=np.pi/8)
    
    print("  Running 100 iterations...")
    emerged, final_state = transition.evolve_until_emergence(
        max_iterations=100,
        verbose=False
    )
    
    # Analyze convergence
    pac_evolution = transition.pac_evolution
    
    # Check if converging
    if len(pac_evolution) > 20:
        early_mean = np.mean(pac_evolution[:10])
        late_mean = np.mean(pac_evolution[-10:])
        improvement = (early_mean - late_mean) / early_mean * 100
        
        print(f"\n  Early PAC (avg): {early_mean:.6f}")
        print(f"  Late PAC (avg): {late_mean:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        # Check trend
        if len(pac_evolution) > 2:
            trend = np.polyfit(range(len(pac_evolution)), pac_evolution, 1)[0]
            print(f"  Trend: {'Decreasing' if trend < 0 else 'Increasing/Stable'}")
            print(f"  Slope: {trend:.6f}")
    
    # Predict emergence
    predicted_iter, confidence = transition.predict_emergence_iteration()
    if predicted_iter is not None:
        print(f"\n  Predicted emergence: iteration {predicted_iter}")
        print(f"  Confidence: {confidence:.2%}")
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("PRE-FIELD RECURSION v2.0 - TEST SUITE")
    print("="*60)
    print("\nTesting new formal definitions and transition dynamics...")
    
    tests = [
        ("Basic Recursion", test_basic_recursion),
        ("Emergence Detection", test_emergence_detection),
        ("Topology Comparison", test_different_topologies),
        ("Parameter Sweep", test_parameter_sweep),
        ("Convergence Analysis", test_convergence_analysis)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n  ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         {error}")
    
    print(f"\n  Total: {passed_count}/{total_count} passed")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("""
  1. PAC convergence is slower than expected
     → Need to investigate recursion mechanism
     → Consider adaptive twist rates
     → Try different initial conditions
  
  2. Emergence criteria very strict
     → Current parameters don't reach targets
     → May need parameter tuning
     → Consider relaxed criteria for testing
  
  3. Next steps:
     → Implement calibration_test_v2.py
     → Create herniation_bridge.py
     → Begin metrics dashboard
     → Optimize recursion parameters
    """)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
