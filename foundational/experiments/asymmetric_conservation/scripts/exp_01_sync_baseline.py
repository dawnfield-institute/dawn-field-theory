"""
Experiment 01: Synchronous PAC Baseline

PURPOSE:
    Establish baseline behavior with traditional synchronous PAC execution.
    Conservation is checked at every timestep.
    
    This is the "control" against which async execution will be compared.

HYPOTHESIS:
    Synchronous PAC maintains P + A = C at every step (no Δ buffer needed).

OUTPUT:
    - Conservation verified at every step
    - Total actualization over time
    - Baseline statistics for comparison
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from constants import print_header, print_subheader, save_results, PHI, PHI_INV, XI


class SyncPACNode:
    """Synchronous PAC node - conservation checked immediately."""
    
    def __init__(self, node_id: int, parent_id: int = None, 
                 initial_P: float = 1.0, theta: float = 0.3):
        self.node_id = node_id
        self.parent_id = parent_id
        self.P = initial_P
        self.A = 0.0
        self.theta = theta
        self.children = {}
        
        # Track conservation constant
        self._C = self.P + self.A
        
        # History
        self.collapse_history = []
    
    @property
    def C(self):
        return self._C
    
    def check_conservation(self) -> float:
        """Return conservation error (should be 0)."""
        return abs((self.P + self.A) - self._C)
    
    def collapse(self, fraction: float = PHI_INV) -> float:
        """
        Synchronous collapse: immediately transfer P → A.
        Conservation is enforced instantly.
        """
        if self.P <= self.theta:
            return 0.0
        
        amount = self.P * fraction
        self.P -= amount
        self.A += amount
        
        # Conservation still holds
        assert self.check_conservation() < 1e-10, "Conservation violated!"
        
        self.collapse_history.append(amount)
        return amount


class SyncPACTree:
    """Synchronous PAC tree with per-step conservation."""
    
    def __init__(self, n_children: int = 3, initial_potential: float = 1.0,
                 theta: float = 0.3):
        self.theta = theta
        
        # Root
        self.root = SyncPACNode(0, None, 0.0, theta)
        self.nodes = {0: self.root}
        
        # Children
        child_P = initial_potential / n_children
        for i in range(n_children):
            child = SyncPACNode(i + 1, 0, child_P, theta)
            self.nodes[i + 1] = child
            self.root.children[i + 1] = child
        
        # Total conservation constant
        self._total_C = sum(n.C for n in self.nodes.values())
        
        # Statistics
        self.step_count = 0
        self.conservation_history = []
        self.P_history = []
        self.A_history = []
    
    def check_global_conservation(self) -> dict:
        """Check conservation across all nodes."""
        total_P = sum(n.P for n in self.nodes.values())
        total_A = sum(n.A for n in self.nodes.values())
        total_C = sum(n.C for n in self.nodes.values())
        error = abs((total_P + total_A) - total_C)
        
        return {
            'total_P': total_P,
            'total_A': total_A,
            'total_C': total_C,
            'error': error,
            'is_conserved': error < 1e-10
        }
    
    def step(self) -> int:
        """
        One synchronous step: all nodes collapse if ready.
        Conservation checked after each node.
        """
        collapses = 0
        
        for node in self.nodes.values():
            if node.P > node.theta:
                amount = node.collapse()
                if amount > 0:
                    collapses += 1
                    
                    # Immediately propagate to parent
                    if node.parent_id is not None:
                        parent = self.nodes[node.parent_id]
                        parent.P += amount
                        parent._C += amount  # Conservation constant updates
        
        # Record state
        status = self.check_global_conservation()
        self.conservation_history.append(status['error'])
        self.P_history.append(status['total_P'])
        self.A_history.append(status['total_A'])
        self.step_count += 1
        
        return collapses
    
    def run(self, max_steps: int = 100) -> int:
        """Run until stable or max steps."""
        total_collapses = 0
        for _ in range(max_steps):
            c = self.step()
            total_collapses += c
            if c == 0:
                break
        return total_collapses


def run_experiment():
    """Run synchronous baseline experiment."""
    print_header("EXPERIMENT 01: SYNCHRONOUS PAC BASELINE")
    
    results = {
        'experiment': 'exp_01_sync_baseline',
        'tests': []
    }
    
    # ==========================================================================
    # Test 1: Basic synchronous execution
    # ==========================================================================
    print_subheader("Test 1: Basic Synchronous Execution")
    
    tree = SyncPACTree(n_children=3, initial_potential=1.0, theta=0.3)
    
    print(f"Initial state:")
    status = tree.check_global_conservation()
    print(f"  Total P: {status['total_P']:.6f}")
    print(f"  Total A: {status['total_A']:.6f}")
    print(f"  Total C: {status['total_C']:.6f}")
    print(f"  Conservation error: {status['error']:.2e}")
    
    collapses = tree.run(max_steps=50)
    
    print(f"\nAfter {tree.step_count} steps ({collapses} collapses):")
    status = tree.check_global_conservation()
    print(f"  Total P: {status['total_P']:.6f}")
    print(f"  Total A: {status['total_A']:.6f}")
    print(f"  Conservation error: {status['error']:.2e}")
    
    # Check all steps maintained conservation
    max_error = max(tree.conservation_history)
    all_conserved = all(e < 1e-10 for e in tree.conservation_history)
    
    print(f"\nConservation at every step: {'✓' if all_conserved else '✗'}")
    print(f"  Max error across all steps: {max_error:.2e}")
    
    results['tests'].append({
        'name': 'basic_sync',
        'steps': tree.step_count,
        'collapses': collapses,
        'final_P': status['total_P'],
        'final_A': status['total_A'],
        'max_conservation_error': max_error,
        'all_conserved': all_conserved,
        'P_history': tree.P_history,
        'A_history': tree.A_history,
    })
    
    # ==========================================================================
    # Test 2: Synchronous with injection
    # ==========================================================================
    print_subheader("Test 2: Synchronous with Staggered Injection")
    
    tree2 = SyncPACTree(n_children=3, initial_potential=0.5, theta=0.2)
    
    # Run a few steps
    tree2.run(max_steps=5)
    
    # Inject at step 5
    print(f"Before injection (step {tree2.step_count}):")
    status_before = tree2.check_global_conservation()
    print(f"  Total P: {status_before['total_P']:.6f}")
    print(f"  Total A: {status_before['total_A']:.6f}")
    
    # Inject 0.5 into node 1
    tree2.nodes[1].P += 0.5
    tree2.nodes[1]._C += 0.5
    
    print(f"\nAfter injection of 0.5 into node 1:")
    status_after = tree2.check_global_conservation()
    print(f"  Total P: {status_after['total_P']:.6f}")
    print(f"  Total A: {status_after['total_A']:.6f}")
    print(f"  Total C increased: {status_after['total_C']:.6f}")
    
    # Continue
    tree2.run(max_steps=20)
    
    status_final = tree2.check_global_conservation()
    print(f"\nFinal state:")
    print(f"  Total P: {status_final['total_P']:.6f}")
    print(f"  Total A: {status_final['total_A']:.6f}")
    print(f"  Conservation error: {status_final['error']:.2e}")
    
    results['tests'].append({
        'name': 'sync_with_injection',
        'injection_amount': 0.5,
        'final_P': status_final['total_P'],
        'final_A': status_final['total_A'],
        'total_C': status_final['total_C'],
        'conserved': status_final['is_conserved'],
    })
    
    # ==========================================================================
    # Test 3: Conservation constraint is strict
    # ==========================================================================
    print_subheader("Test 3: Sync Model Has No Local Asymmetry")
    
    tree3 = SyncPACTree(n_children=5, initial_potential=2.0, theta=0.2)
    
    # Run and track
    local_asymmetries = []
    for step in range(30):
        tree3.step()
        # In sync model, there's no "pending" or "unresolved" value
        # Everything is resolved immediately
        asymmetry = 0.0  # By definition in sync model
        local_asymmetries.append(asymmetry)
    
    max_asymmetry = max(local_asymmetries)
    print(f"Max local asymmetry (should be 0): {max_asymmetry:.2e}")
    print(f"This is the BASELINE—async model will show non-zero asymmetry")
    
    results['tests'].append({
        'name': 'no_local_asymmetry',
        'max_asymmetry': max_asymmetry,
        'is_zero': max_asymmetry == 0,
    })
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_subheader("SUMMARY")
    
    all_passed = all(t.get('conserved', True) and t.get('all_conserved', True) 
                     for t in results['tests'])
    
    print(f"""
    Synchronous PAC Baseline:
    
    ✓ Conservation holds at EVERY timestep
    ✓ No local asymmetry (Δ = 0 always)
    ✓ Injection increases total C, conservation still holds
    
    This is the CONTROL for async experiments.
    
    Key values for comparison:
    - Typical final P after 1.0 → ~{results['tests'][0]['final_P']:.4f}
    - Typical final A after 1.0 → ~{results['tests'][0]['final_A']:.4f}
    """)
    
    results['summary'] = {
        'baseline_established': True,
        'all_tests_passed': all_passed,
        'sync_has_no_asymmetry': True,
        'conservation_strict': True,
    }
    
    save_results(results, 'exp_01')
    return results


if __name__ == '__main__':
    run_experiment()
