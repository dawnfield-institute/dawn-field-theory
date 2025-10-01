#!/usr/bin/env python3
"""
Convergence Comparison Test: v2.0 vs v2.1

Compares fixed-rate recursion (v2.0) with adaptive recursion (v2.1)
to validate 10x improvement target.

Version: 2.1
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core import PreFieldState, RecursionOperator, AdaptiveRecursionOperator, create_initial_state


def compare_convergence(iterations=500, size=100, seed=42):
    """
    Compare v2.0 baseline vs v2.1 adaptive convergence
    
    Args:
        iterations: Number of recursion steps
        size: Field size
        seed: Random seed for reproducibility
        
    Returns:
        improvement_percentage: How much better v2.1 is
    """
    
    print("="*70)
    print("CONVERGENCE COMPARISON: v2.0 (Fixed) vs v2.1 (Adaptive)")
    print("="*70)
    
    # Create identical initial state for both
    print(f"\n[Setup] Creating initial state (size={size}, seed={seed})...")
    initial = create_initial_state(size=size, topology="mobius", seed=seed)
    print(f"  Initial energy: {initial.compute_field_energy():.4f}")
    print(f"  Initial entropy: {initial.compute_information_entropy():.4f}")
    
    # v2.0: Fixed twist rate (baseline)
    print("\n[1] Running v2.0 BASELINE (fixed twist rate)...")
    baseline_op = RecursionOperator(twist_rate=np.pi/2)
    baseline_state = initial.copy()
    baseline_pac_history = []
    baseline_emergence_history = []
    
    for i in range(iterations):
        baseline_state = baseline_op.apply(baseline_state)
        baseline_pac_history.append(baseline_state.pac_residual)
        baseline_emergence_history.append(baseline_state.compute_emergence_metric())
        
        if (i + 1) % 100 == 0:
            print(f"    Iteration {i+1:3d}: PAC = {baseline_state.pac_residual:.6f}")
    
    # v2.1: Adaptive (new)
    print("\n[2] Running v2.1 ADAPTIVE (dynamic parameters)...")
    adaptive_op = AdaptiveRecursionOperator(initial_twist=np.pi/2)
    adaptive_state = initial.copy()
    adaptive_pac_history = []
    adaptive_emergence_history = []
    
    for i in range(iterations):
        adaptive_state = adaptive_op.apply(adaptive_state)
        adaptive_pac_history.append(adaptive_state.pac_residual)
        adaptive_emergence_history.append(adaptive_state.compute_emergence_metric())
        
        if (i + 1) % 100 == 0:
            accel = adaptive_op.acceleration_factor
            print(f"    Iteration {i+1:3d}: PAC = {adaptive_state.pac_residual:.6f} "
                  f"(accel={accel:.2f}x)")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    baseline_final = baseline_pac_history[-1]
    adaptive_final = adaptive_pac_history[-1]
    
    # Calculate metrics
    if baseline_final > adaptive_final:
        improvement = (baseline_final - adaptive_final) / baseline_final * 100
        speedup = baseline_final / adaptive_final
        better = "v2.1 (ADAPTIVE)"
    else:
        improvement = (adaptive_final - baseline_final) / adaptive_final * 100
        speedup = adaptive_final / baseline_final
        better = "v2.0 (BASELINE)"
    
    print(f"\nFinal PAC Residuals:")
    print(f"  v2.0 (baseline):  {baseline_final:.6f}")
    print(f"  v2.1 (adaptive):  {adaptive_final:.6f}")
    print(f"\n  Better: {better}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Speedup factor: {speedup:.2f}x")
    
    # Convergence rate analysis
    def calculate_convergence_rate(history):
        if len(history) < 50:
            return 0.0
        early = np.mean(history[:25])
        late = np.mean(history[-25:])
        return (early - late) / early * 100
    
    baseline_rate = calculate_convergence_rate(baseline_pac_history)
    adaptive_rate = calculate_convergence_rate(adaptive_pac_history)
    
    print(f"\nConvergence Rates:")
    print(f"  v2.0: {baseline_rate:.2f}% improvement")
    print(f"  v2.1: {adaptive_rate:.2f}% improvement")
    
    # Adaptation statistics
    adapt_stats = adaptive_op.get_adaptation_statistics()
    print(f"\nAdaptation Statistics (v2.1):")
    print(f"  Total parameter changes: {adapt_stats['total_adaptations']}")
    print(f"  Final acceleration: {adapt_stats['current_acceleration']:.2f}x")
    if 'max_acceleration' in adapt_stats:
        print(f"  Max acceleration reached: {adapt_stats['max_acceleration']:.2f}x")
    
    # Visualization
    print("\n[3] Generating comparison plots...")
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: PAC Residual (Linear)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(baseline_pac_history, label='v2.0 Fixed', linewidth=2, alpha=0.7, color='blue')
    ax1.plot(adaptive_pac_history, label='v2.1 Adaptive', linewidth=2, color='red')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('PAC Residual')
    ax1.set_title('PAC Convergence (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PAC Residual (Log)
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(baseline_pac_history, label='v2.0 Fixed', linewidth=2, alpha=0.7, color='blue')
    ax2.semilogy(adaptive_pac_history, label='v2.1 Adaptive', linewidth=2, color='red')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PAC Residual (log)')
    ax2.set_title('PAC Convergence (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Emergence Metric
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(baseline_emergence_history, label='v2.0 Fixed', linewidth=2, alpha=0.7, color='blue')
    ax3.plot(adaptive_emergence_history, label='v2.1 Adaptive', linewidth=2, color='red')
    ax3.axhline(y=1.0571, color='green', linestyle='--', label='Ξ threshold', alpha=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Emergence Metric')
    ax3.set_title('Emergence Metric Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Acceleration Factor (v2.1 only)
    ax4 = plt.subplot(2, 3, 4)
    accel_history = [h['acceleration'] for h in adaptive_op.transformation_history]
    ax4.plot(accel_history, linewidth=2, color='orange')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Acceleration Factor')
    ax4.set_title('v2.1 Dynamic Acceleration')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Convergence Rate Comparison
    ax5 = plt.subplot(2, 3, 5)
    window = 50
    baseline_rolling = np.convolve(baseline_pac_history, np.ones(window)/window, mode='valid')
    adaptive_rolling = np.convolve(adaptive_pac_history, np.ones(window)/window, mode='valid')
    ax5.plot(baseline_rolling, label='v2.0 Fixed', linewidth=2, alpha=0.7, color='blue')
    ax5.plot(adaptive_rolling, label='v2.1 Adaptive', linewidth=2, color='red')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel(f'PAC (moving avg, window={window})')
    ax5.set_title('Smoothed Convergence Trends')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary Bar Chart
    ax6 = plt.subplot(2, 3, 6)
    metrics = ['Final PAC', 'Convergence\nRate (%)', 'Emergence\nMetric']
    v20_values = [baseline_final, baseline_rate, baseline_emergence_history[-1]]
    v21_values = [adaptive_final, adaptive_rate, adaptive_emergence_history[-1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax6.bar(x - width/2, v20_values, width, label='v2.0', alpha=0.7, color='blue')
    ax6.bar(x + width/2, v21_values, width, label='v2.1', color='red')
    ax6.set_ylabel('Value')
    ax6.set_title('Metrics Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/convergence_v21_comparison_{timestamp}.png'
    Path('results').mkdir(exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plot saved: {filename}")
    plt.close()
    
    # Success criteria
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK")
    print("="*70)
    
    success = True
    target_improvement = 10  # Target: 10x improvement
    
    if speedup >= target_improvement:
        print(f"✅ PASS: {speedup:.1f}x speedup exceeds {target_improvement}x target")
    elif speedup >= 2.0:
        print(f"⚠️  PARTIAL: {speedup:.1f}x speedup (target: {target_improvement}x)")
        success = False
    else:
        print(f"❌ FAIL: Only {speedup:.1f}x speedup (target: {target_improvement}x)")
        success = False
    
    return success, improvement


def main():
    """Run convergence comparison"""
    
    success, improvement = compare_convergence(iterations=500, size=100, seed=42)
    
    print("\n" + "="*70)
    if success:
        print("✅ v2.1 VALIDATION SUCCESSFUL")
        print("Ready for physical constant validation and herniation detection!")
    else:
        print("⚠️  v2.1 SHOWS IMPROVEMENT BUT NOT TARGET")
        print("Additional tuning recommended before proceeding.")
    print("="*70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
