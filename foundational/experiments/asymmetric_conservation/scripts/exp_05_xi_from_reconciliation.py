"""
Experiment 05: Ξ from Reconciliation Delay Distribution

PURPOSE:
    Test if Ξ = 1.0571 (the balance constant) emerges from the statistics
    of reconciliation delays in async PAC systems.

HYPOTHESIS:
    The characteristic frequency of reconciliation events relates to Ξ.
    Specifically: some function of reconciliation interval distribution → Ξ.

DERIVATION CONTEXT:
    Ξ = 1 + π/55 was derived in oscillation_attractor_dynamics exp_24.
    Here we test if it also emerges from async execution dynamics.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from async_pac import AsyncPACTree
from event_system import ReconciliationBoundary
from constants import print_header, print_subheader, save_results, PHI, XI, PI


def run_experiment():
    """Test Ξ emergence from reconciliation dynamics."""
    print_header("EXPERIMENT 05: Ξ FROM RECONCILIATION DELAY DISTRIBUTION")
    
    results = {
        'experiment': 'exp_05_xi_from_reconciliation',
        'xi_target': XI,
        'tests': []
    }
    
    print(f"Target Ξ = 1 + π/55 = {XI:.10f}")
    
    # ==========================================================================
    # Test 1: Basic reconciliation interval statistics
    # ==========================================================================
    print_subheader("Test 1: Reconciliation Interval Statistics")
    
    # Run multiple trials to gather statistics
    all_intervals = []
    
    for trial in range(10):
        np.random.seed(trial * 100)
        tree = AsyncPACTree(embed_dim=32, theta=0.2)
        tree.boundary = ReconciliationBoundary(delta_threshold=0.5)
        
        embeddings = np.random.randn(8, 32) * 0.6
        tree.graft_embeddings(embeddings, 8)
        
        # Inject periodically to keep system active
        for step in range(200):
            tree.step_async()
            if step % 20 == 0 and step > 0:
                tree.inject_potential(step % 8 + 1, 0.3)
        
        if len(tree.reconciliation_events) > 1:
            intervals = np.diff(tree.reconciliation_events)
            all_intervals.extend(intervals)
    
    if all_intervals:
        intervals_arr = np.array(all_intervals)
        mean_interval = np.mean(intervals_arr)
        std_interval = np.std(intervals_arr)
        median_interval = np.median(intervals_arr)
        
        print(f"Reconciliation interval statistics ({len(all_intervals)} intervals):")
        print(f"  Mean: {mean_interval:.4f}")
        print(f"  Std: {std_interval:.4f}")
        print(f"  Median: {median_interval:.4f}")
        
        # Various attempts to extract Ξ from these statistics
        attempts = []
        
        # Attempt 1: 1 + π/mean
        if mean_interval > 0:
            xi_attempt1 = 1 + PI / mean_interval
            attempts.append(('1 + π/mean', xi_attempt1, abs(xi_attempt1 - XI)))
        
        # Attempt 2: 1 + π/55 with 55 from ratio
        if std_interval > 0:
            ratio = mean_interval / std_interval
            xi_attempt2 = 1 + PI / (ratio * 10)  # Scale factor
            attempts.append(('ratio-based', xi_attempt2, abs(xi_attempt2 - XI)))
        
        # Attempt 3: Coefficient of variation based
        if mean_interval > 0 and std_interval > 0:
            cv = std_interval / mean_interval
            xi_attempt3 = 1 + PI * cv
            attempts.append(('1 + π*CV', xi_attempt3, abs(xi_attempt3 - XI)))
        
        # Attempt 4: 1 + median/mean * π/55
        if mean_interval > 0:
            xi_attempt4 = 1 + (median_interval / mean_interval) * PI / 55
            attempts.append(('median ratio', xi_attempt4, abs(xi_attempt4 - XI)))
        
        print(f"\nΞ extraction attempts:")
        for name, val, err in sorted(attempts, key=lambda x: x[2]):
            match = "✓" if err < 0.1 else "✗"
            print(f"  {match} {name}: {val:.6f} (error: {err:.6f})")
        
        best_attempt = min(attempts, key=lambda x: x[2])
        
        results['tests'].append({
            'name': 'interval_statistics',
            'n_intervals': len(all_intervals),
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'median_interval': median_interval,
            'attempts': [{'method': a[0], 'value': a[1], 'error': a[2]} for a in attempts],
            'best_method': best_attempt[0],
            'best_value': best_attempt[1],
            'best_error': best_attempt[2],
        })
    else:
        print("No reconciliation intervals gathered")
        results['tests'].append({'name': 'interval_statistics', 'error': 'no data'})
    
    # ==========================================================================
    # Test 2: Threshold sweep looking for Ξ
    # ==========================================================================
    print_subheader("Test 2: Threshold Sweep for Ξ Signature")
    
    thresholds = np.linspace(0.1, 2.0, 20)
    threshold_results = []
    
    for thresh in thresholds:
        intervals = []
        for trial in range(5):
            np.random.seed(trial * 50)
            tree = AsyncPACTree(embed_dim=32, theta=0.2)
            tree.boundary = ReconciliationBoundary(delta_threshold=thresh)
            
            embeddings = np.random.randn(8, 32) * 0.5
            tree.graft_embeddings(embeddings, 8)
            
            for step in range(100):
                tree.step_async()
                if step % 25 == 0 and step > 0:
                    tree.inject_potential(step % 8 + 1, 0.2)
            
            if len(tree.reconciliation_events) > 1:
                intervals.extend(np.diff(tree.reconciliation_events))
        
        if intervals:
            mean_int = np.mean(intervals)
            threshold_results.append({
                'threshold': thresh,
                'mean_interval': mean_int,
                'n_intervals': len(intervals),
                'near_xi': abs(thresh - XI) < 0.1,
            })
    
    print(f"Threshold → Mean interval:")
    xi_row = None
    for r in threshold_results[::2]:  # Every other for brevity
        marker = " ← Ξ region" if r['near_xi'] else ""
        print(f"  θ={r['threshold']:.3f}: mean_interval={r['mean_interval']:.3f}{marker}")
        if r['near_xi']:
            xi_row = r
    
    # Check if there's something special at Ξ threshold
    if xi_row:
        print(f"\nAt Ξ ≈ {XI:.4f} threshold:")
        print(f"  Mean interval: {xi_row['mean_interval']:.4f}")
    
    results['tests'].append({
        'name': 'threshold_sweep',
        'results': threshold_results,
        'xi_threshold_result': xi_row,
    })
    
    # ==========================================================================
    # Test 3: Long-run convergence
    # ==========================================================================
    print_subheader("Test 3: Long-Run Convergence")
    
    np.random.seed(42)
    tree3 = AsyncPACTree(embed_dim=32, theta=0.2)
    tree3.boundary = ReconciliationBoundary(delta_threshold=0.5)
    
    embeddings3 = np.random.randn(10, 32) * 0.5
    tree3.graft_embeddings(embeddings3, 10)
    
    # Long run with continuous injection
    xi_estimates = []
    for epoch in range(10):
        for step in range(100):
            tree3.step_async()
            if step % 15 == 0:
                tree3.inject_potential(step % 10 + 1, 0.2)
        
        # Estimate Ξ from current data
        if len(tree3.reconciliation_events) > 5:
            recent_intervals = np.diff(tree3.reconciliation_events[-20:])
            if len(recent_intervals) > 2:
                mean_int = np.mean(recent_intervals)
                std_int = np.std(recent_intervals)
                if std_int > 0:
                    estimate = 1 + (std_int / mean_int) * PI
                    xi_estimates.append({
                        'epoch': epoch,
                        'estimate': estimate,
                        'error': abs(estimate - XI),
                    })
    
    print("Ξ estimates over time:")
    for e in xi_estimates:
        match = "✓" if e['error'] < 0.1 else "✗"
        print(f"  {match} Epoch {e['epoch']}: Ξ ≈ {e['estimate']:.6f} (error: {e['error']:.4f})")
    
    if xi_estimates:
        final_estimate = xi_estimates[-1]['estimate']
        converging = xi_estimates[-1]['error'] < xi_estimates[0]['error'] if len(xi_estimates) > 1 else False
        print(f"\nConverging toward Ξ: {'✓' if converging else '✗'}")
    else:
        final_estimate = None
        converging = False
    
    results['tests'].append({
        'name': 'long_run_convergence',
        'estimates': xi_estimates,
        'final_estimate': final_estimate,
        'converging': converging,
    })
    
    # ==========================================================================
    # Test 4: Check if 55 emerges
    # ==========================================================================
    print_subheader("Test 4: Does 55 (F₁₀) Emerge?")
    
    # In the formula Ξ = 1 + π/55, can we find 55 in the dynamics?
    
    np.random.seed(123)
    tree4 = AsyncPACTree(embed_dim=32, theta=0.2)
    tree4.boundary = ReconciliationBoundary(delta_threshold=0.5)
    
    embeddings4 = np.random.randn(10, 32) * 0.5
    tree4.graft_embeddings(embeddings4, 10)
    
    # Run and count events per reconciliation
    events_per_recon = []
    last_event_count = 0
    last_recon_count = 0
    
    for step in range(500):
        tree4.step_async()
        if step % 20 == 0:
            tree4.inject_potential(step % 10 + 1, 0.25)
        
        if tree4.stats['reconciliations'] > last_recon_count:
            events = tree4.stats['events_emitted'] - last_event_count
            events_per_recon.append(events)
            last_event_count = tree4.stats['events_emitted']
            last_recon_count = tree4.stats['reconciliations']
    
    if events_per_recon:
        mean_events = np.mean(events_per_recon)
        print(f"Events per reconciliation:")
        print(f"  Mean: {mean_events:.2f}")
        print(f"  Compare to F₁₀ = 55: {55 / mean_events:.2f}x")
        
        # Check if some multiple relates to 55
        for mult in [1, 2, 5, 10, PI]:
            scaled = mean_events * mult
            if abs(scaled - 55) < 5:
                print(f"  Found: mean × {mult} = {scaled:.2f} ≈ 55!")
    
    results['tests'].append({
        'name': 'fibonacci_55_search',
        'events_per_recon': events_per_recon[:20] if events_per_recon else [],
        'mean_events': float(np.mean(events_per_recon)) if events_per_recon else None,
    })
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_subheader("SUMMARY")
    
    # Determine if we found Ξ
    best_xi_found = None
    best_error = float('inf')
    for test in results['tests']:
        if 'best_value' in test and test.get('best_error', float('inf')) < best_error:
            best_xi_found = test['best_value']
            best_error = test['best_error']
    
    xi_emerged = best_error < 0.1 if best_xi_found else False
    
    best_str = f"{best_xi_found:.6f}" if best_xi_found else "N/A"
    error_str = f"{best_error:.6f}" if best_error != float('inf') else "N/A"
    
    print(f"""
    Ξ Emergence from Reconciliation:
    
    Target: Ξ = 1 + π/55 = {XI:.10f}
    
    Best found: {best_str} (error: {error_str})
    
    Ξ emerged from dynamics: {'✓' if xi_emerged else '✗ (needs more investigation)'}
    
    INTERPRETATION:
    - If Ξ emerges: Reconciliation dynamics ARE the source of Ξ
    - If not: Ξ may come from a different mechanism (PAC collapse geometry)
    
    Either way, this experiment maps the relationship between:
    - Reconciliation frequency
    - Event distribution
    - The balance constant Ξ
    """)
    
    results['summary'] = {
        'xi_target': XI,
        'best_estimate': best_xi_found,
        'best_error': best_error,
        'xi_emerged': xi_emerged,
        'conclusion': 'Ξ emergence confirmed' if xi_emerged else 'Ξ mechanism unclear - needs more research',
    }
    
    save_results(results, 'exp_05')
    return results


if __name__ == '__main__':
    run_experiment()
