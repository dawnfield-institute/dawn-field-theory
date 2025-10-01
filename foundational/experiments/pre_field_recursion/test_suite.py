"""
Pre-Field Recursion Test Suite
==============================

Unified test suite with version selection for comparing different
recursion operator implementations.

Versions:
- v2.0: Fixed-rate Möbius transformation (baseline)
- v2.1: Adaptive acceleration (momentum-based)
- v2.2: Resonance-aware convergence (FFT-based locking) [CURRENT]

Usage:
    python test_suite.py --version v22              # Run v2.2 only
    python test_suite.py --compare                  # Compare all versions
    python test_suite.py --iterations 1000          # Extended run
    python test_suite.py --seed 42                  # Specific seed

Author: Dawn Field Institute
Date: October 1, 2025
Version: 2.2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Ensure core is in path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core import (
    PreFieldState, 
    RecursionOperator,           # v2.0 baseline
    AdaptiveRecursionOperator,   # v2.1/v2.2
    ResonanceDetector,
    create_initial_state
)


class TestConfig:
    """Configuration for test runs"""
    def __init__(self):
        self.iterations = 500
        self.seed = 42
        self.topology = "mobius"
        self.size = 100
        self.initial_twist = np.pi / 2
        self.output_dir = Path('results')
        self.plot_format = 'png'
        self.dpi = 150


def run_version(version: str, initial_state: PreFieldState, 
                iterations: int, verbose: bool = True) -> Dict:
    """
    Run a specific version of the recursion operator
    
    Args:
        version: 'v20', 'v21', or 'v22'
        initial_state: Starting pre-field state
        iterations: Number of iterations to run
        verbose: Print progress updates
        
    Returns:
        Dictionary with results
    """
    # Create appropriate operator
    if version == 'v20':
        op = RecursionOperator(twist_rate=np.pi/2)
        name = "v2.0 Fixed (Baseline)"
    elif version == 'v21':
        op = AdaptiveRecursionOperator(
            initial_twist=np.pi/2,
            resonance_aware=False
        )
        name = "v2.1 Adaptive (No Resonance)"
    elif version == 'v22':
        op = AdaptiveRecursionOperator(
            initial_twist=np.pi/2,
            resonance_aware=True
        )
        name = "v2.2 Resonance-Aware"
    else:
        raise ValueError(f"Unknown version: {version}")
    
    if verbose:
        print(f"\n[Running] {name}...")
    
    # Run iterations
    state = initial_state.copy()
    pac_history = []
    
    for i in range(iterations):
        state = op.apply(state)
        pac_history.append(state.pac_residual)
        
        # Progress reporting
        if verbose:
            if hasattr(op, 'resonance_locked') and op.resonance_locked:
                if (i + 1) % 100 == 0:
                    print(f"    Iteration {i+1}: PAC = {state.pac_residual:.6f} "
                          f"(locked to period {op.detected_period:.1f})")
            elif (i + 1) % 100 == 0:
                print(f"    Iteration {i+1}: PAC = {state.pac_residual:.6f}")
    
    # Collect results
    results = {
        'version': version,
        'name': name,
        'pac_history': pac_history,
        'final_pac': pac_history[-1],
        'final_state': state,
        'operator': op
    }
    
    # Add version-specific metadata
    if hasattr(op, 'resonance_locked'):
        results['resonance_locked'] = op.resonance_locked
        results['detected_period'] = op.detected_period
        results['detected_frequency'] = op.detected_frequency
    
    if hasattr(op, 'adaptation_history'):
        results['adaptations'] = len(op.adaptation_history)
        results['acceleration_factor'] = op.acceleration_factor
    
    return results


def compare_versions(config: TestConfig, versions: List[str] = None) -> Dict[str, Dict]:
    """
    Compare multiple versions side-by-side
    
    Args:
        config: Test configuration
        versions: List of versions to compare (default: all)
        
    Returns:
        Dictionary mapping version to results
    """
    if versions is None:
        versions = ['v20', 'v21', 'v22']
    
    print("="*70)
    print("PRE-FIELD RECURSION TEST SUITE - VERSION COMPARISON")
    print("="*70)
    
    # Create initial state
    print(f"\n[Setup] Creating initial state...")
    print(f"  Topology: {config.topology}")
    print(f"  Size: {config.size}")
    print(f"  Seed: {config.seed}")
    
    initial = create_initial_state(
        size=config.size,
        topology=config.topology,
        seed=config.seed
    )
    print(f"  Initial PAC residual: {initial.pac_residual:.6f}")
    
    # Run each version
    results = {}
    for version in versions:
        results[version] = run_version(
            version, 
            initial, 
            config.iterations,
            verbose=True
        )
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    baseline = results['v20']['final_pac']
    
    for version in versions:
        data = results[version]
        final = data['final_pac']
        
        print(f"\n{data['name']}:")
        print(f"  Final PAC: {final:.6f}")
        
        if version != 'v20':
            if final < baseline:
                improvement = (baseline - final) / baseline * 100
                speedup = baseline / final
                print(f"  vs baseline: {improvement:.1f}% better")
                print(f"  Speedup: {speedup:.2f}x")
                
                if speedup >= 5.0:
                    print(f"  ✅ Excellent (>5x)")
                elif speedup >= 2.0:
                    print(f"  ⚠️  Good (>2x)")
                else:
                    print(f"  ❌ Needs improvement (<2x)")
            else:
                degradation = (final - baseline) / baseline * 100
                print(f"  vs baseline: {degradation:.1f}% WORSE")
                print(f"  ❌ Regression")
        
        # Version-specific details
        if 'resonance_locked' in data:
            print(f"  Resonance locked: {data['resonance_locked']}")
            if data['resonance_locked']:
                print(f"  Period: {data['detected_period']:.1f} iterations")
        
        if 'adaptations' in data:
            print(f"  Adaptations: {data['adaptations']}")
            print(f"  Final acceleration: {data['acceleration_factor']:.2f}x")
    
    return results


def visualize_comparison(results: Dict[str, Dict], config: TestConfig):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(18, 12))
    
    colors = {
        'v20': 'blue',
        'v21': 'orange', 
        'v22': 'red'
    }
    
    versions = list(results.keys())
    
    # Plot 1: Linear PAC evolution
    ax1 = plt.subplot(2, 3, 1)
    for version in versions:
        data = results[version]
        ax1.plot(data['pac_history'], label=data['name'],
                linewidth=2, alpha=0.8, color=colors[version])
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('PAC Residual', fontsize=11)
    ax1.set_title('PAC Convergence Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-scale PAC evolution
    ax2 = plt.subplot(2, 3, 2)
    for version in versions:
        data = results[version]
        ax2.semilogy(data['pac_history'], label=data['name'],
                    linewidth=2, alpha=0.8, color=colors[version])
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('PAC Residual (log)', fontsize=11)
    ax2.set_title('PAC Convergence (Log Scale)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detrended oscillations (v2.2 if available)
    ax3 = plt.subplot(2, 3, 3)
    if 'v22' in results:
        history = results['v22']['pac_history']
        x = np.arange(len(history))
        
        if len(history) > 2:
            coeffs = np.polyfit(x, history, 1)
            trend = np.poly1d(coeffs)(x)
            detrended = np.array(history) - trend
            
            ax3.plot(detrended, color='red', linewidth=1.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax3.set_xlabel('Iteration', fontsize=11)
            ax3.set_ylabel('Detrended PAC', fontsize=11)
            ax3.set_title('v2.2 Oscillation Pattern', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: FFT spectrum (v2.2 if available)
    ax4 = plt.subplot(2, 3, 4)
    if 'v22' in results and len(results['v22']['pac_history']) > 2:
        history = results['v22']['pac_history']
        x = np.arange(len(history))
        coeffs = np.polyfit(x, history, 1)
        trend = np.poly1d(coeffs)(x)
        detrended = np.array(history) - trend
        
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        power = np.abs(fft)**2
        
        positive_mask = freqs > 0
        ax4.plot(freqs[positive_mask], power[positive_mask],
                linewidth=1.5, color='red')
        
        # Mark detected frequency
        op = results['v22']['operator']
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
    for version in versions:
        data = results[version]
        history = data['pac_history']
        rates = -np.gradient(history)
        
        if len(rates) >= 20:
            smoothed = np.convolve(rates, np.ones(20)/20, mode='valid')
            ax5.plot(smoothed, label=data['name'], linewidth=2,
                    alpha=0.8, color=colors[version])
    
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Convergence Rate', fontsize=11)
    ax5.set_title('Convergence Rate Evolution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 6: Final comparison bar chart
    ax6 = plt.subplot(2, 3, 6)
    names = [results[v]['name'] for v in versions]
    finals = [results[v]['final_pac'] for v in versions]
    bar_colors = [colors[v] for v in versions]
    
    bars = ax6.bar(range(len(names)), finals, color=bar_colors,
                   alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax6.set_ylabel('Final PAC Residual', fontsize=11)
    ax6.set_title('Final PAC Comparison\n(lower is better)',
                  fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and speedups
    baseline = results['v20']['final_pac']
    for i, (bar, val, version) in enumerate(zip(bars, finals, versions)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        
        if version != 'v20' and val < baseline:
            speedup = baseline / val
            ax6.text(bar.get_x() + bar.get_width()/2., val/2,
                    f'{speedup:.1f}x', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round',
                             facecolor='green' if speedup >= 5 else 'orange',
                             alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir.mkdir(exist_ok=True)
    
    filename = config.output_dir / f'comparison_{timestamp}.{config.plot_format}'
    plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {filename}")
    
    plt.close()


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Pre-Field Recursion Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_suite.py                    # Compare all versions
  python test_suite.py --version v22      # Run v2.2 only
  python test_suite.py --iterations 1000  # Extended run
  python test_suite.py --seed 123         # Different initial state
        """
    )
    
    parser.add_argument('--version', choices=['v20', 'v21', 'v22'],
                       help='Run specific version only')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Compare all versions (default)')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of iterations (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--topology', default='mobius',
                       choices=['mobius', 'torus', 'klein'],
                       help='Topology type (default: mobius)')
    parser.add_argument('--size', type=int, default=100,
                       help='Field size (default: 100)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = TestConfig()
    config.iterations = args.iterations
    config.seed = args.seed
    config.topology = args.topology
    config.size = args.size
    
    try:
        if args.version:
            # Run single version
            print("="*70)
            print(f"PRE-FIELD RECURSION TEST - {args.version.upper()}")
            print("="*70)
            
            initial = create_initial_state(
                size=config.size,
                topology=config.topology,
                seed=config.seed
            )
            
            result = run_version(args.version, initial, config.iterations)
            results = {args.version: result}
            
            print(f"\n✅ Final PAC: {result['final_pac']:.6f}")
        else:
            # Compare all versions
            results = compare_versions(config)
        
        # Visualization
        if not args.no_plot and len(results) > 1:
            # Only visualize comparison if multiple versions
            visualize_comparison(results, config)
        
        # Success determination
        if 'v22' in results and 'v20' in results:
            v22_final = results['v22']['final_pac']
            baseline = results['v20']['final_pac']
            speedup = baseline / v22_final
            
            success = speedup >= 5.0
        else:
            # Single version run - just check it completed
            success = True
        
        print("\n" + "="*70)
        if success:
            print("✅ TEST SUITE COMPLETE - All targets met!")
        else:
            print("⚠️  TEST SUITE COMPLETE - Some targets missed")
        print("="*70)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
