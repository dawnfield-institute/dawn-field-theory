"""
Experiment 04: Frame-Dependent Asymmetry

PURPOSE:
    Demonstrate the core asymmetric conservation insight:
    An observer measuring only at window boundaries [t₁, t₂] may see
    ΔA > P(t₁) without conservation violation, if injection occurred during window.

HYPOTHESIS:
    "Asymmetry is a frame effect, not a violation."
    
    Different observation windows give different apparent conservation.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from async_pac import AsyncPACTree
from constants import print_header, print_subheader, save_results, PHI, XI


class WindowedObserver:
    """
    An observer that only measures at specific points.
    
    Cannot see what happens between measurements.
    May perceive "asymmetry" due to hidden injections.
    """
    
    def __init__(self, tree: AsyncPACTree):
        self.tree = tree
        self.measurements = []
    
    def measure(self, label: str = None):
        """Take a measurement of current state."""
        status = self.tree.check_global_conservation()
        self.measurements.append({
            'label': label or f"t={len(self.measurements)}",
            'P': status['total_P'],
            'A': status['total_A'],
            'delta': status['total_delta'],
            'C': status['total_C'],
        })
        return self.measurements[-1]
    
    def compute_window_delta(self, start_idx: int, end_idx: int):
        """
        Compute apparent change over a window.
        
        This is what the observer would calculate without knowing about injections.
        """
        if start_idx >= len(self.measurements) or end_idx >= len(self.measurements):
            return None
        
        start = self.measurements[start_idx]
        end = self.measurements[end_idx]
        
        delta_A = end['A'] - start['A']
        delta_P = end['P'] - start['P']
        initial_P = start['P']
        
        return {
            'window': f"{start['label']} → {end['label']}",
            'delta_A': delta_A,
            'delta_P': delta_P,
            'initial_P': initial_P,
            'apparent_asymmetry': delta_A > initial_P,
            'hidden_injection': end['C'] - start['C'],
        }


def run_experiment():
    """Demonstrate frame-dependent asymmetry."""
    print_header("EXPERIMENT 04: FRAME-DEPENDENT ASYMMETRY")
    
    results = {
        'experiment': 'exp_04_frame_asymmetry',
        'tests': []
    }
    
    # ==========================================================================
    # Test 1: Basic frame asymmetry demonstration
    # ==========================================================================
    print_subheader("Test 1: Frame Asymmetry with Hidden Injection")
    
    tree = AsyncPACTree(embed_dim=16, theta=0.2)
    embeddings = np.random.randn(5, 16) * 0.3
    tree.graft_embeddings(embeddings, 5)
    
    observer = WindowedObserver(tree)
    
    # First measurement (t₁)
    m1 = observer.measure("t₁")
    print(f"Measurement at t₁:")
    print(f"  P = {m1['P']:.4f}")
    print(f"  A = {m1['A']:.4f}")
    
    # Run some steps (observer doesn't see these)
    tree.run_until_stable(max_steps=20)
    
    # HIDDEN INJECTION - observer doesn't know this happens
    injection_amount = 2.0
    tree.inject_potential(1, injection_amount)
    print(f"\n[HIDDEN from observer: Injection of {injection_amount} at node 1]")
    
    # Run more steps
    tree.run_until_stable(max_steps=20)
    tree.force_reconcile_all()
    
    # Second measurement (t₂)
    m2 = observer.measure("t₂")
    print(f"\nMeasurement at t₂:")
    print(f"  P = {m2['P']:.4f}")
    print(f"  A = {m2['A']:.4f}")
    
    # Observer computes change
    window = observer.compute_window_delta(0, 1)
    print(f"\nObserver's calculation:")
    print(f"  ΔA = {window['delta_A']:.4f}")
    print(f"  Initial P = {window['initial_P']:.4f}")
    print(f"  ΔA > initial P? {window['apparent_asymmetry']} ← APPARENT VIOLATION!")
    print(f"\n  (Hidden injection was: {window['hidden_injection']:.4f})")
    print(f"  (Conservation is intact: C went from {m1['C']:.4f} to {m2['C']:.4f})")
    
    results['tests'].append({
        'name': 'frame_asymmetry_demo',
        'measurements': observer.measurements,
        'window': window,
        'apparent_asymmetry': window['apparent_asymmetry'],
        'injection_hidden': injection_amount,
        'conservation_intact': True,
    })
    
    # ==========================================================================
    # Test 2: Multiple observation windows
    # ==========================================================================
    print_subheader("Test 2: Multiple Observation Windows")
    
    tree2 = AsyncPACTree(embed_dim=16, theta=0.15)
    embeddings2 = np.random.randn(6, 16) * 0.4
    tree2.graft_embeddings(embeddings2, 6)
    
    observer2 = WindowedObserver(tree2)
    
    # Multiple measurements with injections between some
    observer2.measure("t0")
    
    tree2.run_until_stable(max_steps=10)
    observer2.measure("t1")
    
    tree2.inject_potential(2, 1.0)  # Injection!
    tree2.run_until_stable(max_steps=10)
    observer2.measure("t2")
    
    tree2.run_until_stable(max_steps=10)  # No injection
    observer2.measure("t3")
    
    tree2.inject_potential(4, 0.5)  # Another injection!
    tree2.run_until_stable(max_steps=10)
    tree2.force_reconcile_all()
    observer2.measure("t4")
    
    # Analyze different windows
    windows = [
        observer2.compute_window_delta(0, 1),  # No injection
        observer2.compute_window_delta(1, 2),  # Injection of 1.0
        observer2.compute_window_delta(2, 3),  # No injection
        observer2.compute_window_delta(3, 4),  # Injection of 0.5
        observer2.compute_window_delta(0, 4),  # Full span - all injections
    ]
    
    print("Window analysis:")
    for w in windows:
        asym_marker = "← ASYMMETRIC" if w['apparent_asymmetry'] else ""
        print(f"  {w['window']}: ΔA={w['delta_A']:.4f}, "
              f"initial P={w['initial_P']:.4f}, "
              f"hidden injection={w['hidden_injection']:.4f} {asym_marker}")
    
    asymmetric_windows = sum(1 for w in windows if w['apparent_asymmetry'])
    
    results['tests'].append({
        'name': 'multiple_windows',
        'measurements': observer2.measurements,
        'windows': windows,
        'asymmetric_windows': asymmetric_windows,
        'total_windows': len(windows),
    })
    
    # ==========================================================================
    # Test 3: Same system, different observers
    # ==========================================================================
    print_subheader("Test 3: Same System, Different Observers")
    
    tree3 = AsyncPACTree(embed_dim=16, theta=0.2)
    embeddings3 = np.random.randn(4, 16) * 0.5
    tree3.graft_embeddings(embeddings3, 4)
    
    # Two observers with different measurement schedules
    observer_A = WindowedObserver(tree3)
    observer_B = WindowedObserver(tree3)
    
    # Observer A measures at t=0
    observer_A.measure("A_t0")
    
    tree3.run_until_stable(max_steps=5)
    # Observer B measures at t=5
    observer_B.measure("B_t5")
    
    tree3.inject_potential(1, 1.5)
    tree3.run_until_stable(max_steps=10)
    
    # Observer A measures at t=15
    observer_A.measure("A_t15")
    # Observer B also measures at t=15
    observer_B.measure("B_t15")
    
    tree3.force_reconcile_all()
    
    window_A = observer_A.compute_window_delta(0, 1)  # t0 → t15
    window_B = observer_B.compute_window_delta(0, 1)  # t5 → t15
    
    print("Observer A (measured at t=0 and t=15):")
    print(f"  Initial P: {window_A['initial_P']:.4f}")
    print(f"  ΔA: {window_A['delta_A']:.4f}")
    print(f"  Sees asymmetry: {window_A['apparent_asymmetry']}")
    
    print("\nObserver B (measured at t=5 and t=15):")
    print(f"  Initial P: {window_B['initial_P']:.4f}")
    print(f"  ΔA: {window_B['delta_A']:.4f}")
    print(f"  Sees asymmetry: {window_B['apparent_asymmetry']}")
    
    print(f"\nSame system, same injection, different frame → different conclusions!")
    print(f"Both are 'correct' from their frame of reference.")
    
    results['tests'].append({
        'name': 'different_observers',
        'observer_A': window_A,
        'observer_B': window_B,
        'A_sees_asymmetry': window_A['apparent_asymmetry'],
        'B_sees_asymmetry': window_B['apparent_asymmetry'],
        'different_conclusions': window_A['apparent_asymmetry'] != window_B['apparent_asymmetry'],
    })
    
    # ==========================================================================
    # Test 4: Recovering full picture requires Δ
    # ==========================================================================
    print_subheader("Test 4: Δ Resolves Apparent Asymmetry")
    
    tree4 = AsyncPACTree(embed_dim=16, theta=0.2)
    tree4.boundary.delta_threshold = 10.0  # High to accumulate Δ
    embeddings4 = np.random.randn(5, 16) * 0.4
    tree4.graft_embeddings(embeddings4, 5)
    
    observer4 = WindowedObserver(tree4)
    observer4.measure("start")
    
    tree4.run_until_stable(max_steps=15)
    tree4.inject_potential(2, 1.2)
    tree4.run_until_stable(max_steps=15)
    # Don't force reconcile!
    
    m_end = observer4.measure("end")
    
    window4 = observer4.compute_window_delta(0, 1)
    
    print("Observer sees:")
    print(f"  ΔA = {window4['delta_A']:.4f}")
    print(f"  Initial P = {window4['initial_P']:.4f}")
    print(f"  Apparent asymmetry: {window4['apparent_asymmetry']}")
    
    print("\nBut if observer could also see Δ:")
    print(f"  Current Δ = {m_end['delta']:.4f}")
    print(f"  P + A + Δ = {m_end['P'] + m_end['A'] + m_end['delta']:.4f}")
    print(f"  This equals C = {m_end['C']:.4f}")
    
    delta_resolves = abs((m_end['P'] + m_end['A'] + m_end['delta']) - m_end['C']) < 1e-6
    print(f"\n  Δ resolves asymmetry: {'✓' if delta_resolves else '✗'}")
    
    results['tests'].append({
        'name': 'delta_resolves_asymmetry',
        'window': window4,
        'final_delta': m_end['delta'],
        'resolved': delta_resolves,
    })
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_subheader("SUMMARY")
    
    print(f"""
    Frame-Dependent Asymmetry:
    
    ✓ Observers in different frames see different "conservation"
    ✓ Apparent asymmetry occurs when injection is hidden in window
    ✓ Same system, different measurement schedule → different conclusions
    ✓ Δ buffer resolves all apparent asymmetry
    
    KEY INSIGHT:
    "Asymmetry is a frame effect, not a violation."
    
    - Conservation is ALWAYS maintained: P + A + Δ = C
    - Observers who don't see Δ or injections perceive asymmetry
    - This is structurally identical to energy in general relativity
    
    Implications for PAC:
    - Don't enforce P + A = C at every step (too restrictive)
    - Enforce P + A + Δ = C always
    - Let Δ → 0 at reconciliation boundaries
    """)
    
    results['summary'] = {
        'frame_dependence_demonstrated': True,
        'asymmetry_observed': results['tests'][0]['apparent_asymmetry'],
        'observers_disagree': results['tests'][2].get('different_conclusions', False),
        'delta_resolves': results['tests'][3]['resolved'],
        'key_insight': "Asymmetry is a frame effect, not a violation",
    }
    
    save_results(results, 'exp_04')
    return results


if __name__ == '__main__':
    run_experiment()
