"""
Experiment 07: Falsification Tests

PURPOSE:
    Define and test conditions that would falsify the asymmetric conservation hypothesis.

FALSIFICATION CONDITIONS:
    1. Δ grows unbounded (conservation actually violated)
    2. Event-indexed gives different final state than time-indexed
    3. P + A + Δ ≠ C at any point (strict violation)
    4. Reconciliation doesn't restore Δ → 0

If ANY of these occur, the model is falsified.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from async_pac import AsyncPACTree
from pac_tensors import PACTensor, NodeState, XI
from event_system import ReconciliationBoundary
from constants import print_header, print_subheader, save_results


def run_experiment():
    """Run falsification tests."""
    print_header("EXPERIMENT 07: FALSIFICATION TESTS")
    
    results = {
        'experiment': 'exp_07_falsification',
        'tests': [],
        'falsified': False,
    }
    
    # ==========================================================================
    # Test F1: Δ bounded (doesn't grow forever)
    # ==========================================================================
    print_subheader("F1: Δ Remains Bounded")
    
    # No reconciliation - let Δ accumulate
    tree = AsyncPACTree(embed_dim=32, theta=0.2)
    tree.boundary.delta_threshold = float('inf')  # Never auto-reconcile
    
    np.random.seed(42)
    embeddings = np.random.randn(10, 32) * 0.5
    tree.graft_embeddings(embeddings, 10)
    
    delta_values = []
    for step in range(200):
        tree.step_async()
        status = tree.check_global_conservation()
        delta_values.append(abs(status['total_delta']))
        
        # Inject to keep system active
        if step % 30 == 0:
            tree.inject_potential(step % 10 + 1, 0.3)
    
    max_delta = max(delta_values)
    final_delta = delta_values[-1]
    
    # Δ should be bounded by total value in system
    total_C = tree.check_global_conservation()['total_C']
    is_bounded = max_delta <= total_C * 1.1  # Allow 10% margin
    
    print(f"Δ growth test (no reconciliation):")
    print(f"  Max Δ observed: {max_delta:.4f}")
    print(f"  Total C in system: {total_C:.4f}")
    print(f"  Δ bounded by C: {'✓ PASS' if is_bounded else '✗ FAIL'}")
    
    f1_passed = is_bounded
    results['tests'].append({
        'name': 'F1_delta_bounded',
        'max_delta': max_delta,
        'total_C': total_C,
        'passed': f1_passed,
    })
    
    if not f1_passed:
        results['falsified'] = True
    
    # ==========================================================================
    # Test F2: Sync and Async give same final state
    # ==========================================================================
    print_subheader("F2: Sync ≡ Async Final State")
    
    np.random.seed(123)
    embeddings2 = np.random.randn(8, 32) * 0.5
    
    # Sync execution (immediate reconcile)
    sync_tree = AsyncPACTree(embed_dim=32, theta=0.2)
    sync_tree.boundary.delta_threshold = 0.0
    sync_tree.graft_embeddings(embeddings2.copy(), 8)
    sync_tree.run_until_stable(max_steps=100)
    sync_tree.force_reconcile_all()
    sync_status = sync_tree.check_global_conservation()
    
    # Async execution (delayed reconcile)
    async_tree = AsyncPACTree(embed_dim=32, theta=0.2)
    async_tree.boundary.delta_threshold = 0.5
    async_tree.graft_embeddings(embeddings2.copy(), 8)
    async_tree.run_until_stable(max_steps=100)
    async_tree.force_reconcile_all()
    async_status = async_tree.check_global_conservation()
    
    P_match = abs(sync_status['total_P'] - async_status['total_P']) < 1e-6
    A_match = abs(sync_status['total_A'] - async_status['total_A']) < 1e-6
    
    print(f"Sync vs Async comparison:")
    print(f"  Sync:  P={sync_status['total_P']:.6f}, A={sync_status['total_A']:.6f}")
    print(f"  Async: P={async_status['total_P']:.6f}, A={async_status['total_A']:.6f}")
    print(f"  P matches: {'✓' if P_match else '✗'}")
    print(f"  A matches: {'✓' if A_match else '✗'}")
    
    f2_passed = P_match and A_match
    print(f"  Result: {'✓ PASS' if f2_passed else '✗ FAIL'}")
    
    results['tests'].append({
        'name': 'F2_sync_async_equivalent',
        'sync_P': sync_status['total_P'],
        'sync_A': sync_status['total_A'],
        'async_P': async_status['total_P'],
        'async_A': async_status['total_A'],
        'passed': f2_passed,
    })
    
    if not f2_passed:
        results['falsified'] = True
    
    # ==========================================================================
    # Test F3: P + A + Δ = C always
    # ==========================================================================
    print_subheader("F3: Conservation P + A + Δ = C Always Holds")
    
    tree3 = AsyncPACTree(embed_dim=32, theta=0.2)
    tree3.boundary.delta_threshold = 1.0  # Accumulate Δ
    
    np.random.seed(456)
    embeddings3 = np.random.randn(10, 32) * 0.6
    tree3.graft_embeddings(embeddings3, 10)
    
    violations = []
    for step in range(150):
        tree3.step_async()
        status = tree3.check_global_conservation()
        
        if not status['is_conserved']:
            violations.append({
                'step': step,
                'error': status['conservation_error'],
                'P': status['total_P'],
                'A': status['total_A'],
                'delta': status['total_delta'],
                'C': status['total_C'],
            })
        
        # Inject periodically
        if step % 25 == 0:
            tree3.inject_potential(step % 10 + 1, 0.25)
    
    f3_passed = len(violations) == 0
    
    print(f"Conservation check over 150 steps:")
    print(f"  Violations: {len(violations)}")
    if violations:
        print(f"  First violation at step {violations[0]['step']}: error={violations[0]['error']:.2e}")
    print(f"  Result: {'✓ PASS' if f3_passed else '✗ FAIL'}")
    
    results['tests'].append({
        'name': 'F3_conservation_always',
        'n_violations': len(violations),
        'violations': violations[:5],  # First 5
        'passed': f3_passed,
    })
    
    if not f3_passed:
        results['falsified'] = True
    
    # ==========================================================================
    # Test F4: Reconciliation restores Δ → 0
    # ==========================================================================
    print_subheader("F4: Reconciliation Restores Δ → 0")
    
    tree4 = AsyncPACTree(embed_dim=32, theta=0.2)
    tree4.boundary.delta_threshold = float('inf')  # Manual reconcile only
    
    np.random.seed(789)
    embeddings4 = np.random.randn(8, 32) * 0.5
    tree4.graft_embeddings(embeddings4, 8)
    
    # Accumulate Δ
    for _ in range(50):
        tree4.step_async()
    
    status_before = tree4.check_global_conservation()
    delta_before = status_before['total_delta']
    
    # Force reconcile
    tree4.force_reconcile_all()
    
    status_after = tree4.check_global_conservation()
    delta_after = status_after['total_delta']
    
    f4_passed = delta_after < 1e-10
    
    print(f"Reconciliation test:")
    print(f"  Δ before reconcile: {delta_before:.6f}")
    print(f"  Δ after reconcile: {delta_after:.2e}")
    print(f"  Result: {'✓ PASS' if f4_passed else '✗ FAIL'}")
    
    results['tests'].append({
        'name': 'F4_reconcile_clears_delta',
        'delta_before': delta_before,
        'delta_after': delta_after,
        'passed': f4_passed,
    })
    
    if not f4_passed:
        results['falsified'] = True
    
    # ==========================================================================
    # Test F5: Extreme conditions
    # ==========================================================================
    print_subheader("F5: Extreme Conditions Stress Test")
    
    # Test with many nodes, high injection rate
    tree5 = AsyncPACTree(embed_dim=64, theta=0.1)
    tree5.boundary.delta_threshold = 2.0
    
    np.random.seed(999)
    embeddings5 = np.random.randn(50, 64) * 1.0  # High initial values
    tree5.graft_embeddings(embeddings5, 50)
    
    # Aggressive injection
    for step in range(300):
        tree5.step_async()
        if step % 5 == 0:
            tree5.inject_potential(np.random.randint(1, 51), np.random.uniform(0.1, 1.0))
    
    tree5.force_reconcile_all()
    final_status = tree5.check_global_conservation()
    
    f5_passed = final_status['is_conserved']
    
    print(f"Extreme conditions (50 nodes, 60 injections):")
    print(f"  Final conservation error: {final_status['conservation_error']:.2e}")
    print(f"  Result: {'✓ PASS' if f5_passed else '✗ FAIL'}")
    
    results['tests'].append({
        'name': 'F5_extreme_conditions',
        'n_nodes': 50,
        'n_injections': 60,
        'final_error': final_status['conservation_error'],
        'passed': f5_passed,
    })
    
    if not f5_passed:
        results['falsified'] = True
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_subheader("FALSIFICATION SUMMARY")
    
    all_passed = all(t['passed'] for t in results['tests'])
    n_passed = sum(1 for t in results['tests'] if t['passed'])
    n_total = len(results['tests'])
    
    print(f"""
    Falsification Test Results: {n_passed}/{n_total} passed
    
    {'='*50}
    """)
    
    for t in results['tests']:
        status = '✓ PASS' if t['passed'] else '✗ FAIL'
        print(f"    {t['name']}: {status}")
    
    print(f"""
    {'='*50}
    
    MODEL STATUS: {'✓ NOT FALSIFIED' if all_passed else '✗ FALSIFIED'}
    
    """)
    
    if all_passed:
        print("""
    The asymmetric conservation model survives all falsification tests:
    
    1. Δ remains bounded by system total
    2. Sync and async produce equivalent final states
    3. P + A + Δ = C holds at all times
    4. Reconciliation properly clears Δ
    5. Model survives extreme conditions
    
    CONCLUSION: Asymmetric conservation is a valid execution model for PAC.
    """)
    else:
        failed = [t for t in results['tests'] if not t['passed']]
        print(f"""
    FALSIFICATION DETECTED:
    
    Failed tests: {[t['name'] for t in failed]}
    
    The model requires revision or the implementation has bugs.
    """)
        results['falsified'] = True
    
    results['summary'] = {
        'all_passed': all_passed,
        'n_passed': n_passed,
        'n_total': n_total,
        'falsified': results['falsified'],
        'conclusion': 'Model validated' if all_passed else 'Model falsified',
    }
    
    save_results(results, 'exp_07')
    return results


if __name__ == '__main__':
    run_experiment()
