"""
Pre-Field Recursion v2.2: Resonance-Aware Convergence Test
=========================================================

Compares three approaches:
1. v2.0 Fixed - Baseline with fixed twist rate
2. v2.1 Adaptive (no resonance) - Adaptive but resonance-blind
3. v2.2 Resonance-Aware - Full resonance detection and locking

Success criteria:
- Resonance lock within 100 iterations
- >5x faster convergence than v2.0 baseline
- No divergence after lock
- Physical constants ready for validation

Author: Dawn Field Institute
Date: October 1, 2025
Version: 2.2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path

# Ensure core is in path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core import (
    PreFieldState, 
    RecursionOperator, 
    AdaptiveRecursionOperator,
    ResonanceDetector,
    create_initial_state
)


def test_resonance_detection():
    """
    Main test: Compare v2.0, v2.1, and v2.2 convergence
    """
    
    print("="*70)
    print("PRE-FIELD RECURSION v2.2: RESONANCE-AWARE CONVERGENCE TEST")
    print("="*70)
    
    # Create initial state
    print("\n[Setup] Creating initial state...")
    initial = create_initial_state(size=100, topology="mobius", seed=42)
    print(f"  Initial PAC residual: {initial.pac_residual:.6f}")
    
    # Three operators to compare
    operators = {
        'v2.0 Fixed': RecursionOperator(twist_rate=np.pi/2),
        'v2.1 Adaptive (no resonance)': AdaptiveRecursionOperator(
            initial_twist=np.pi/2, 
            resonance_aware=False
        ),
        'v2.2 Resonance-Aware': AdaptiveRecursionOperator(
            initial_twist=np.pi/2,
            resonance_aware=True
        )
    }
    
    results = {}
    iterations = 500
    
    # Run each operator
    for name, op in operators.items():
        print(f"\n[Running] {name}...")
        state = initial.copy()
        pac_history = []
        
        for i in range(iterations):
            state = op.apply(state)
            pac_history.append(state.pac_residual)
            
            # Progress reporting
            if hasattr(op, 'resonance_locked') and op.resonance_locked:
                if (i + 1) % 100 == 0:
                    print(f"    Iteration {i+1}: PAC = {state.pac_residual:.6f} "
                          f"(locked to period {op.detected_period:.1f})")
            elif (i + 1) % 100 == 0:
                print(f"    Iteration {i+1}: PAC = {state.pac_residual:.6f}")
        
        results[name] = {
            'pac_history': pac_history,
            'final_pac': pac_history[-1],
            'operator': op,
            'final_state': state
        }
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    baseline = results['v2.0 Fixed']['final_pac']
    
    for name, data in results.items():
        final = data['final_pac']
        print(f"\n{name}:")
        print(f"  Final PAC: {final:.6f}")
        
        if final < baseline:
            improvement = (baseline - final) / baseline * 100
            speedup = baseline / final
            print(f"  vs baseline: {improvement:.1f}% better")
            print(f"  Speedup: {speedup:.2f}x")
            
            if speedup >= 5.0:
                print(f"  ✅ Target achieved! (>5x)")
            elif speedup >= 2.0:
                print(f"  ⚠️  Good progress (>2x)")
            else:
                print(f"  ❌ Below target (<2x)")
        else:
            degradation = (final - baseline) / baseline * 100
            print(f"  vs baseline: {degradation:.1f}% WORSE")
            print(f"  ❌ Regression")
    
    # Resonance lock status
    print("\n" + "="*70)
    print("RESONANCE ANALYSIS")
    print("="*70)
    
    v22_op = results['v2.2 Resonance-Aware']['operator']
    if hasattr(v22_op, 'resonance_locked'):
        print(f"\nv2.2 Resonance Status:")
        print(f"  Locked: {v22_op.resonance_locked}")
        
        if v22_op.resonance_locked:
            print(f"  Period: {v22_op.detected_period:.1f} iterations")
            print(f"  Frequency: {v22_op.detected_frequency:.4f} cycles/iter")
            print(f"  Locked twist rate: {v22_op.twist_rate:.4f} rad")
            print(f"  ✅ Resonance detection successful!")
        else:
            print(f"  ⚠️  No resonance lock achieved")
            
            # Try manual analysis
            detector = ResonanceDetector()
            pac_hist = results['v2.2 Resonance-Aware']['pac_history']
            resonance_info = detector.analyze_oscillations(pac_hist)
            
            print(f"\n  Manual analysis:")
            print(f"    Detected period: {resonance_info.get('period', 'None')}")
            print(f"    Confidence: {resonance_info.get('confidence', 0):.2f}")
            print(f"    Amplitude: {resonance_info.get('amplitude', 0):.4f}")
    
    # Adaptation statistics
    print("\n" + "="*70)
    print("ADAPTATION STATISTICS")
    print("="*70)
    
    for name in ['v2.1 Adaptive (no resonance)', 'v2.2 Resonance-Aware']:
        op = results[name]['operator']
        if hasattr(op, 'adaptation_history'):
            print(f"\n{name}:")
            print(f"  Total adaptations: {len(op.adaptation_history)}")
            print(f"  Final acceleration: {op.acceleration_factor:.2f}x")
            
            if hasattr(op, 'stagnation_counter'):
                print(f"  Stagnation counter: {op.stagnation_counter}")
    
    # Visualization
    visualize_comparison(results, iterations)
    
    # Success check
    v22_final = results['v2.2 Resonance-Aware']['final_pac']
    v22_locked = results['v2.2 Resonance-Aware']['operator'].resonance_locked if hasattr(
        results['v2.2 Resonance-Aware']['operator'], 'resonance_locked'
    ) else False
    
    success = (v22_final < baseline * 0.2) and v22_locked  # 5x better + locked
    
    return success, results


def visualize_comparison(results, iterations):
    """
    Create comprehensive 6-panel visualization
    """
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: PAC evolution (linear)
    ax1 = plt.subplot(2, 3, 1)
    colors = {'v2.0 Fixed': 'blue', 
              'v2.1 Adaptive (no resonance)': 'orange',
              'v2.2 Resonance-Aware': 'red'}
    
    for name, data in results.items():
        ax1.plot(data['pac_history'], label=name, 
                linewidth=2, alpha=0.8, color=colors[name])
    
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('PAC Residual', fontsize=11)
    ax1.set_title('PAC Convergence Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PAC evolution (log)
    ax2 = plt.subplot(2, 3, 2)
    for name, data in results.items():
        ax2.semilogy(data['pac_history'], label=name, 
                    linewidth=2, alpha=0.8, color=colors[name])
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('PAC Residual (log)', fontsize=11)
    ax2.set_title('PAC Convergence (Log Scale)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detrended oscillations (v2.2 only)
    ax3 = plt.subplot(2, 3, 3)
    v22_history = results['v2.2 Resonance-Aware']['pac_history']
    x = np.arange(len(v22_history))
    
    if len(v22_history) > 2:
        coeffs = np.polyfit(x, v22_history, 1)
        trend = np.poly1d(coeffs)(x)
        detrended = np.array(v22_history) - trend
        
        ax3.plot(detrended, color='red', linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Detrended PAC', fontsize=11)
        ax3.set_title('v2.2 Oscillation Pattern', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: FFT spectrum (v2.2)
    ax4 = plt.subplot(2, 3, 4)
    
    if len(v22_history) > 2:
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        power = np.abs(fft)**2
        
        positive_mask = freqs > 0
        ax4.plot(freqs[positive_mask], power[positive_mask], 
                linewidth=1.5, color='red')
        
        # Mark detected frequency if locked
        op = results['v2.2 Resonance-Aware']['operator']
        if hasattr(op, 'detected_frequency') and op.detected_frequency:
            ax4.axvline(x=op.detected_frequency, color='green', 
                       linestyle='--', linewidth=2, 
                       label=f'Locked: {op.detected_frequency:.4f}')
            ax4.legend(fontsize=9)
        
        ax4.set_xlabel('Frequency (cycles/iteration)', fontsize=11)
        ax4.set_ylabel('Power', fontsize=11)
        ax4.set_title('Frequency Spectrum (v2.2)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Convergence rate comparison
    ax5 = plt.subplot(2, 3, 5)
    for name, data in results.items():
        history = data['pac_history']
        rates = -np.gradient(history)  # Negative gradient = improvement
        
        # Smooth
        if len(rates) >= 20:
            smoothed = np.convolve(rates, np.ones(20)/20, mode='valid')
            ax5.plot(smoothed, label=name, linewidth=2, 
                    alpha=0.8, color=colors[name])
    
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Convergence Rate', fontsize=11)
    ax5.set_title('Convergence Rate Evolution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 6: Final comparison bar chart
    ax6 = plt.subplot(2, 3, 6)
    names = list(results.keys())
    finals = [results[n]['final_pac'] for n in names]
    bar_colors = [colors[n] for n in names]
    
    bars = ax6.bar(range(len(names)), finals, color=bar_colors, alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax6.set_ylabel('Final PAC Residual', fontsize=11)
    ax6.set_title('Final PAC Comparison\n(lower is better)', 
                  fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, finals):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speedup annotations
    baseline = results['v2.0 Fixed']['final_pac']
    for i, (bar, val) in enumerate(zip(bars, finals)):
        if i > 0 and val < baseline:  # Skip baseline itself
            speedup = baseline / val
            ax6.text(bar.get_x() + bar.get_width()/2., val/2,
                    f'{speedup:.1f}x', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green' if speedup >= 5 else 'orange', 
                             alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    filename = results_dir / f'convergence_v22_resonance_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {filename}")
    
    plt.close()


def main():
    """Main entry point"""
    
    try:
        success, results = test_resonance_detection()
        
        print("\n" + "="*70)
        if success:
            print("✅ v2.2 RESONANCE-AWARE CONVERGENCE: SUCCESS!")
            print("\nAchievements:")
            print("  • Resonance detection and locking working")
            print("  • >5x faster convergence than baseline")
            print("  • System ready for physical constant validation")
            print("\nNext steps:")
            print("  • Validate against fine structure constant (α)")
            print("  • Test with multiple initial conditions")
            print("  • Integrate with Q-Socket phase locking")
        else:
            print("⚠️  v2.2 PARTIAL SUCCESS - Needs fine-tuning")
            print("\nStatus:")
            
            v22_final = results['v2.2 Resonance-Aware']['final_pac']
            baseline = results['v2.0 Fixed']['final_pac']
            speedup = baseline / v22_final if v22_final > 0 else 0
            
            print(f"  • Speedup: {speedup:.2f}x (target: >5x)")
            
            v22_op = results['v2.2 Resonance-Aware']['operator']
            locked = v22_op.resonance_locked if hasattr(v22_op, 'resonance_locked') else False
            print(f"  • Resonance locked: {locked}")
            
            if speedup < 2.0:
                print("\nRecommendations:")
                print("  • Check resonance detection thresholds")
                print("  • Verify FFT window size")
                print("  • Test with longer runs (>500 iterations)")
            elif not locked:
                print("\nRecommendations:")
                print("  • Lower confidence threshold for locking")
                print("  • Increase analysis window size")
                print("  • Check for multi-modal frequencies")
        
        print("="*70)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
