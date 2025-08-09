"""
SCBF Visualization Module
===================        print("No adaptation data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Live Adaptation Dynamics & Available Metrics', fontsize=16, fontweight='bold')

Plotting and visualization functions for SCBF analysis results.
Provides real-time and post-experiment visualization of all SCBF metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

def plot_live_adaptation_dynamics_fallback(logs: List[Dict], save_path: Optional[str] = None):
    """
    Fallback plot showing basic live adaptation dynamics when SCBF data is limited.
    Always works with basic log data.
    """
    steps = []
    losses = []
    outputs = []
    targets = []
    quantum_kl = []
    quantum_js = []
    quantum_wd = []
    quantum_qwcs = []
    
    for log in logs:
        steps.append(log.get('step', len(steps)))
        losses.append(log.get('loss', 0))
        outputs.append(log.get('output', 0))
        targets.append(log.get('target', 0))
        
        # Extract quantum metrics if available
        if 'quantum_metrics' in log:
            qm = log['quantum_metrics']
            quantum_kl.append(qm.get('KL-Divergence', 0))
            quantum_js.append(qm.get('Jensen-Shannon', 0))
            quantum_wd.append(qm.get('Wasserstein Distance', 0))
            quantum_qwcs.append(qm.get('QWCS', 0))
        else:
            quantum_kl.append(0)
            quantum_js.append(0)
            quantum_wd.append(0)
            quantum_qwcs.append(0)
    
    if not steps:
        print("No training data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Live Adaptation Dynamics & Available Metrics', fontsize=16, fontweight='bold')
    
    # Loss curve
    axes[0, 0].plot(steps, losses, 'purple', linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_title('Live Adaptation Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Predictions vs Targets
    axes[0, 1].plot(steps, outputs, 'b-', label='Predictions', linewidth=2)
    axes[0, 1].plot(steps, targets, 'r-', label='Targets', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Predictions vs Targets')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction Error
    errors = np.array(outputs) - np.array(targets)
    axes[0, 2].plot(steps, errors, 'orange', linewidth=2, marker='s', markersize=3)
    axes[0, 2].set_title('Prediction Error')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Error')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Quantum Metrics (if available)
    if any(quantum_kl) or any(quantum_js):
        axes[1, 0].plot(steps, quantum_kl, 'red', linewidth=2, label='KL-Divergence')
        axes[1, 0].plot(steps, quantum_js, 'blue', linewidth=2, label='Jensen-Shannon')
        axes[1, 0].set_title('Quantum Divergence Metrics')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        if min(quantum_kl + quantum_js) > 0:
            axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'Quantum Divergence\nMetrics Not Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Quantum Divergence Metrics')
    
    # Wasserstein & QWCS
    if any(quantum_wd) or any(quantum_qwcs):
        axes[1, 1].plot(steps, quantum_wd, 'green', linewidth=2, label='Wasserstein Dist')
        axes[1, 1].plot(steps, quantum_qwcs, 'purple', linewidth=2, label='QWCS')
        axes[1, 1].set_title('Quantum Coherence Metrics')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Quantum Coherence\nMetrics Not Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Quantum Coherence Metrics')
    
    # Live Adaptation Statistics
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    final_loss = losses[-1] if losses else 0
    
    stats_text = f"""Live Adaptation Statistics:
    
Final Loss: {final_loss:.6f}
MSE: {mse:.6f}
MAE: {mae:.6f}
Steps: {len(steps)}

Loss Trend: {"↓ Decreasing" if len(losses) > 1 and losses[-1] < losses[0] else "→ Stable/Increasing"}
"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    axes[1, 2].set_title('Live Adaptation Summary')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_entropy_collapse_timeline(logs: List[Dict], save_path: Optional[str] = None):
    """Plot entropy collapse metrics over time with robust error handling."""
    steps = []
    collapse_magnitudes = []
    collapse_rates = []
    symbolic_states = []
    
    for log in logs:
        if 'scbf' in log and 'entropy_collapse' in log['scbf']:
            steps.append(log.get('step', len(steps)))
            entropy_data = log['scbf']['entropy_collapse']
            collapse_magnitudes.append(entropy_data.get('magnitude', 0))
            collapse_rates.append(entropy_data.get('rate', 0))
            symbolic_states.append(entropy_data.get('symbolic_states', 0))
    
    if not steps:
        print("No entropy collapse data found - creating placeholder plot")
        # Create a placeholder plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Entropy Collapse Data Available\n\nThis may occur when:\n• Insufficient live adaptation steps\n• Model activations too small\n• Analysis errors during computation', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax.set_title('SCBF Entropy Collapse Analysis')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SCBF Entropy Collapse Analysis', fontsize=14, fontweight='bold')
    
    # Collapse magnitude
    axes[0, 0].plot(steps, collapse_magnitudes, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Collapse Magnitude')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Magnitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Collapse rate
    axes[0, 1].plot(steps, collapse_rates, 'r-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Collapse Rate')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Symbolic states
    axes[1, 0].plot(steps, symbolic_states, 'g-', linewidth=2)
    axes[1, 0].set_title('Symbolic States')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].text(0.1, 0.8, f'Mean Magnitude: {np.mean(collapse_magnitudes):.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.7, f'Max Magnitude: {np.max(collapse_magnitudes):.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.6, f'Mean Rate: {np.mean(collapse_rates):.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.5, f'Mean States: {np.mean(symbolic_states):.1f}', 
                    transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_bifractal_evolution(logs: List[Dict], save_path: Optional[str] = None):
    """Plot bifractal dimension evolution over time."""
    steps = []
    fractal_dims = []
    entropies = []
    similarities = []
    
    for log in logs:
        if 'scbf' in log and 'lineage' in log['scbf']:
            steps.append(log.get('step', len(steps)))
            lineage_data = log['scbf']['lineage']
            fractal_dims.append(lineage_data['fractal_dimension'])
            entropies.append(lineage_data['entropy'])
            similarities.append(lineage_data.get('similarity', 0))
    
    if not steps:
        print("No bifractal lineage data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SCBF Bifractal Lineage Evolution', fontsize=14, fontweight='bold')
    
    # Fractal dimension
    axes[0, 0].plot(steps, fractal_dims, 'purple', linewidth=2)
    axes[0, 0].set_title('Fractal Dimension')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Dimension')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Lineage entropy
    axes[0, 1].plot(steps, entropies, 'orange', linewidth=2)
    axes[0, 1].set_title('Lineage Entropy')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Structural similarity
    if similarities and any(s > 0 for s in similarities):
        axes[1, 0].plot(steps, similarities, 'teal', linewidth=2)
        axes[1, 0].set_title('Structural Similarity')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Similarity')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No similarity data', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Structural Similarity')
    
    # Evolution summary
    if len(fractal_dims) > 1:
        evolution = fractal_dims[-1] - fractal_dims[0]
        axes[1, 1].text(0.1, 0.8, f'Dimension Evolution: {evolution:.3f}', 
                        transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.7, f'Final Dimension: {fractal_dims[-1]:.3f}', 
                        transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.6, f'Mean Entropy: {np.mean(entropies):.3f}', 
                        transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Evolution Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_activation_ancestry(logs: List[Dict], save_path: Optional[str] = None):
    """Plot activation ancestry metrics."""
    steps = []
    strengths = []
    stabilities = []
    dimensions = []
    
    for log in logs:
        if 'scbf' in log and 'ancestry' in log['scbf']:
            steps.append(log.get('step', len(steps)))
            ancestry_data = log['scbf']['ancestry']
            strengths.append(ancestry_data['strength'])
            stabilities.append(ancestry_data['stability'])
            dimensions.append(ancestry_data.get('dimension', 0))
    
    if not steps:
        print("No activation ancestry data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('SCBF Activation Ancestry Analysis', fontsize=14, fontweight='bold')
    
    # Ancestry strength
    axes[0].plot(steps, strengths, 'navy', linewidth=2)
    axes[0].set_title('Ancestry Strength')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Strength')
    axes[0].grid(True, alpha=0.3)
    
    # Lineage stability
    axes[1].plot(steps, stabilities, 'darkred', linewidth=2)
    axes[1].set_title('Lineage Stability')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Stability')
    axes[1].grid(True, alpha=0.3)
    
    # Bifractal dimension
    if dimensions and any(d > 0 for d in dimensions):
        axes[2].plot(steps, dimensions, 'darkgreen', linewidth=2)
        axes[2].set_title('Bifractal Dimension')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Dimension')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No dimension data', 
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Bifractal Dimension')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_semantic_attractors(logs: List[Dict], save_path: Optional[str] = None):
    """Plot semantic attractor analysis."""
    steps = []
    counts = []
    densities = []
    stabilities = []
    
    for log in logs:
        if 'scbf' in log and 'attractors' in log['scbf']:
            steps.append(log.get('step', len(steps)))
            attractor_data = log['scbf']['attractors']
            counts.append(attractor_data['count'])
            densities.append(attractor_data['density'])
            stabilities.append(attractor_data['stability'])
    
    if not steps:
        print("No semantic attractor data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('SCBF Semantic Attractor Analysis', fontsize=14, fontweight='bold')
    
    # Attractor count
    axes[0].plot(steps, counts, 'magenta', linewidth=2, marker='o', markersize=3)
    axes[0].set_title('Attractor Count')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Attractor density
    axes[1].plot(steps, densities, 'cyan', linewidth=2)
    axes[1].set_title('Attractor Density')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Density')
    axes[1].grid(True, alpha=0.3)
    
    # Attractor stability
    axes[2].plot(steps, stabilities, 'brown', linewidth=2)
    axes[2].set_title('Attractor Stability')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Stability')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_complete_scbf_dashboard(logs: List[Dict], save_path: Optional[str] = None):
    """Create a comprehensive dashboard of all SCBF metrics."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Complete SCBF Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Extract all metrics
    steps = []
    entropy_mags = []
    fractal_dims = []
    ancestry_strengths = []
    attractor_counts = []
    
    for log in logs:
        if 'scbf' in log:
            steps.append(log.get('step', len(steps)))
            scbf = log['scbf']
            
            entropy_mags.append(scbf.get('entropy_collapse', {}).get('magnitude', 0))
            fractal_dims.append(scbf.get('lineage', {}).get('fractal_dimension', 0))
            ancestry_strengths.append(scbf.get('ancestry', {}).get('strength', 0))
            attractor_counts.append(scbf.get('attractors', {}).get('count', 0))
    
    if not steps:
        print("No SCBF data found for dashboard")
        return
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main timeline plots
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, entropy_mags, 'b-', label='Entropy Collapse', linewidth=2)
    ax1.set_title('Primary SCBF Metrics Timeline')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Metric Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Individual metric plots
    ax2 = fig.add_subplot(gs[1, 0])
    if fractal_dims:
        ax2.plot(steps, fractal_dims, 'purple', linewidth=2)
    ax2.set_title('Fractal Dimension')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    if ancestry_strengths:
        ax3.plot(steps, ancestry_strengths, 'navy', linewidth=2)
    ax3.set_title('Ancestry Strength')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2])
    if attractor_counts:
        ax4.plot(steps, attractor_counts, 'magenta', linewidth=2, marker='o', markersize=3)
    ax4.set_title('Attractor Count')
    ax4.grid(True, alpha=0.3)
    
    # Summary statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_text = f"""
    SCBF Analysis Summary ({len(steps)} steps analyzed)
    
    Entropy Collapse: Mean = {np.mean(entropy_mags):.3f}, Max = {np.max(entropy_mags):.3f}
    Fractal Dimension: Mean = {np.mean(fractal_dims):.3f}, Range = {np.max(fractal_dims) - np.min(fractal_dims):.3f}
    Ancestry Strength: Mean = {np.mean(ancestry_strengths):.3f}, Std = {np.std(ancestry_strengths):.3f}
    Attractor Count: Mean = {np.mean(attractor_counts):.1f}, Max = {np.max(attractor_counts)}
    
    Learning Detected: {'Yes' if np.mean(entropy_mags) > 0.001 or np.std(fractal_dims) > 0.001 else 'No'}
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_all_plots(logs: List[Dict], output_dir: str = "scbf_plots"):
    """Save all SCBF plots to a directory with robust error handling."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving SCBF plots to {output_dir}/")
    
    # Always create live adaptation dynamics fallback (this always works)
    try:
        plot_live_adaptation_dynamics_fallback(logs, f"{output_dir}/live_adaptation_dynamics.png")
        print("✓ Live adaptation dynamics plot saved")
    except Exception as e:
        print(f"✗ Live adaptation dynamics plot failed: {e}")
    
    # Quantum-aware metrics analysis (usually works)
    try:
        plot_quantum_metrics_analysis(logs, f"{output_dir}/quantum_metrics_analysis.png")
        print("✓ Quantum metrics analysis saved")
    except Exception as e:
        print(f"✗ Quantum metrics analysis failed: {e}")
    
    # Enhanced analysis plots (usually work)
    try:
        plot_predictions_vs_actual(logs, f"{output_dir}/predictions_vs_actual.png")
        print("✓ Predictions vs actual plot saved")
    except Exception as e:
        print(f"✗ Predictions vs actual plot failed: {e}")
    
    # SCBF-specific plots (may fail if insufficient data)
    scbf_plots = [
        ("entropy_collapse.png", plot_entropy_collapse_timeline, "Entropy collapse"),
        ("bifractal_evolution.png", plot_bifractal_evolution, "Bifractal evolution"), 
        ("activation_ancestry.png", plot_activation_ancestry, "Activation ancestry"),
        ("semantic_attractors.png", plot_semantic_attractors, "Semantic attractors"),
        ("weight_analysis.png", plot_weight_analysis, "Weight analysis"),
        ("neural_growth.png", plot_neural_growth_analysis, "Neural growth"),
        ("complete_dashboard.png", plot_complete_scbf_dashboard, "Complete dashboard")
    ]
    
    for filename, plot_func, description in scbf_plots:
        try:
            plot_func(logs, f"{output_dir}/{filename}")
            print(f"✓ {description} plot saved")
        except Exception as e:
            print(f"✗ {description} plot failed: {e}")
    
    print("✓ All available plots saved successfully!")

def plot_predictions_vs_actual(logs: List[Dict], save_path: Optional[str] = None):
    """Plot model predictions vs actual targets over time."""
    steps = []
    predictions = []
    targets = []
    losses = []
    
    for log in logs:
        if 'output' in log and 'target' in log:
            steps.append(log.get('step', len(steps)))
            predictions.append(log['output'])
            targets.append(log['target'])
            losses.append(log.get('loss', 0))
    
    if not steps:
        print("No prediction data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold')
    
    # Predictions vs Targets timeline
    axes[0, 0].plot(steps, predictions, 'b-', label='Predictions', linewidth=2)
    axes[0, 0].plot(steps, targets, 'r-', label='Targets', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Predictions vs Targets')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curve with quantum metrics
    axes[0, 1].plot(steps, losses, 'purple', linewidth=2, label='Composite Loss')
    axes[0, 1].set_title('Quantum-Aware Loss Curve')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Scatter plot: Predictions vs Targets
    axes[1, 0].scatter(targets, predictions, alpha=0.6, s=20)
    axes[1, 0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', alpha=0.5)
    axes[1, 0].set_title('Prediction Accuracy')
    axes[1, 0].set_xlabel('Target Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error analysis
    errors = np.array(predictions) - np.array(targets)
    axes[1, 1].hist(errors, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Error (Prediction - Target)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add performance statistics
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    axes[1, 1].text(0.05, 0.95, f'MSE: {mse:.6f}\nMAE: {mae:.6f}', 
                    transform=axes[1, 1].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_quantum_metrics_analysis(logs: List[Dict], save_path: Optional[str] = None):
    """Plot quantum-aware loss metrics analysis."""
    steps = []
    kl_divs = []
    js_divs = []
    wasserstein_dists = []
    qwcs_scores = []
    composite_losses = []
    
    for log in logs:
        if 'quantum_metrics' in log:
            steps.append(log.get('step', len(steps)))
            metrics = log['quantum_metrics']
            kl_divs.append(metrics.get('KL-Divergence', 0))
            js_divs.append(metrics.get('Jensen-Shannon', 0))
            wasserstein_dists.append(metrics.get('Wasserstein Distance', 0))
            qwcs_scores.append(metrics.get('QWCS', 0))
            composite_losses.append(log.get('loss', 0))
    
    if not steps:
        print("No quantum metrics data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum-Aware Loss Metrics Analysis', fontsize=16, fontweight='bold')
    
    # KL-Divergence
    axes[0, 0].plot(steps, kl_divs, 'red', linewidth=2)
    axes[0, 0].set_title('KL-Divergence')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('KL-Div')
    axes[0, 0].grid(True, alpha=0.3)
    if min(kl_divs) > 0:  # Only use log scale if all values are positive
        axes[0, 0].set_yscale('log')
    
    # Jensen-Shannon Divergence
    axes[0, 1].plot(steps, js_divs, 'blue', linewidth=2)
    axes[0, 1].set_title('Jensen-Shannon Divergence')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('JS-Div')
    axes[0, 1].grid(True, alpha=0.3)
    if min(js_divs) > 0:  # Only use log scale if all values are positive
        axes[0, 1].set_yscale('log')
    
    # Wasserstein Distance
    axes[0, 2].plot(steps, wasserstein_dists, 'green', linewidth=2)
    axes[0, 2].set_title('Wasserstein Distance')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('W-Distance')
    axes[0, 2].grid(True, alpha=0.3)
    if min(wasserstein_dists) > 0:  # Only use log scale if all values are positive
        axes[0, 2].set_yscale('log')
    
    # QWCS (Quantum Wave Coherence Score)
    axes[1, 0].plot(steps, qwcs_scores, 'purple', linewidth=2)
    axes[1, 0].set_title('Quantum Wave Coherence Score')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('QWCS')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Composite Loss
    axes[1, 1].plot(steps, composite_losses, 'orange', linewidth=2)
    axes[1, 1].set_title('Composite Quantum Loss')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Correlation matrix of metrics
    if len(steps) > 1:
        metrics_matrix = np.array([kl_divs, js_divs, wasserstein_dists, qwcs_scores]).T
        # Check for valid correlation computation
        if metrics_matrix.shape[0] > 1 and np.any(np.std(metrics_matrix, axis=0) > 1e-6):
            try:
                corr_matrix = np.corrcoef(metrics_matrix.T)
                # Replace NaN values with zeros for visualization
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                
                im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 2].set_title('Metrics Correlation')
                axes[1, 2].set_xticks(range(4))
                axes[1, 2].set_yticks(range(4))
                axes[1, 2].set_xticklabels(['KL-Div', 'JS-Div', 'W-Dist', 'QWCS'], rotation=45)
                axes[1, 2].set_yticklabels(['KL-Div', 'JS-Div', 'W-Dist', 'QWCS'])
                
                # Add correlation values as text
                for i in range(4):
                    for j in range(4):
                        axes[1, 2].text(j, i, f'{corr_matrix[i, j]:.2f}', 
                                       ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
                
                plt.colorbar(im, ax=axes[1, 2])
            except:
                # Fallback: show a simple text message
                axes[1, 2].text(0.5, 0.5, 'Insufficient variance\nfor correlation analysis', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Metrics Correlation')
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Metrics Correlation')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved quantum metrics analysis: {save_path}")
    plt.show()

def plot_weight_analysis(logs: List[Dict], save_path: Optional[str] = None):
    """Plot weight evolution and distribution analysis."""
    steps = []
    fractal_dims = []
    entropies = []
    similarities = []
    
    for log in logs:
        if 'scbf' in log and 'lineage' in log['scbf']:
            steps.append(log.get('step', len(steps)))
            lineage = log['scbf']['lineage']
            fractal_dims.append(lineage.get('fractal_dimension', 0))
            entropies.append(lineage.get('entropy', 0))
            similarities.append(lineage.get('similarity', 0))
    
    if not steps:
        print("No weight analysis data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Neural Weight Evolution Analysis', fontsize=14, fontweight='bold')
    
    # Fractal dimension evolution
    axes[0, 0].plot(steps, fractal_dims, 'purple', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Weight Fractal Dimension')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Fractal Dimension')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Weight entropy evolution
    axes[0, 1].plot(steps, entropies, 'orange', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Weight Entropy')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Structural similarity
    if similarities and any(s > 0 for s in similarities):
        axes[0, 2].plot(steps, similarities, 'teal', linewidth=2, marker='^', markersize=4)
        axes[0, 2].set_title('Weight Structure Similarity')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Similarity')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No similarity data\n(First step = baseline)', 
                        ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Weight Structure Similarity')
    
    # Weight complexity growth
    if len(fractal_dims) > 1:
        complexity_growth = np.diff(fractal_dims)
        axes[1, 0].plot(steps[1:], complexity_growth, 'darkred', linewidth=2)
        axes[1, 0].set_title('Complexity Growth Rate')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('∆ Fractal Dimension')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'Need more data\nfor growth analysis', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Complexity Growth Rate')
    
    # Entropy vs Dimension correlation
    if len(fractal_dims) > 1 and len(entropies) > 1:
        axes[1, 1].scatter(fractal_dims, entropies, c=steps, cmap='viridis', s=50)
        axes[1, 1].set_title('Dimension vs Entropy Correlation')
        axes[1, 1].set_xlabel('Fractal Dimension')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Step')
    else:
        axes[1, 1].text(0.5, 0.5, 'Need more data\nfor correlation', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Dimension vs Entropy Correlation')
    
    # Summary statistics
    axes[1, 2].axis('off')
    if fractal_dims and entropies:
        stats_text = f"""Weight Evolution Summary:
        
Initial Fractal Dim: {fractal_dims[0]:.4f}
Final Fractal Dim: {fractal_dims[-1]:.4f}
Total Change: {fractal_dims[-1] - fractal_dims[0]:.4f}

Initial Entropy: {entropies[0]:.3f}
Final Entropy: {entropies[-1]:.3f}
Total Change: {entropies[-1] - entropies[0]:.3f}

Complexity Trend: {"Increasing" if fractal_dims[-1] > fractal_dims[0] else "Decreasing"}
Entropy Trend: {"Increasing" if entropies[-1] > entropies[0] else "Decreasing"}
        """
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                         verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_neural_growth_analysis(logs: List[Dict], save_path: Optional[str] = None):
    """Plot neural network growth and adaptation patterns."""
    steps = []
    ancestry_strengths = []
    ancestry_stabilities = []
    attractor_counts = []
    attractor_densities = []
    attractor_stabilities = []
    
    for log in logs:
        if 'scbf' in log:
            steps.append(log.get('step', len(steps)))
            scbf = log['scbf']
            
            # Ancestry metrics
            ancestry = scbf.get('ancestry', {})
            ancestry_strengths.append(ancestry.get('strength', 0))
            ancestry_stabilities.append(ancestry.get('stability', 0))
            
            # Attractor metrics
            attractors = scbf.get('attractors', {})
            attractor_counts.append(attractors.get('count', 0))
            attractor_densities.append(attractors.get('density', 0))
            attractor_stabilities.append(attractors.get('stability', 0))
    
    if not steps:
        print("No neural growth data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Neural Growth & Adaptation Analysis', fontsize=14, fontweight='bold')
    
    # Activation ancestry strength
    axes[0, 0].plot(steps, ancestry_strengths, 'navy', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Activation Ancestry Strength')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Ancestry Strength')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Lineage stability
    axes[0, 1].plot(steps, ancestry_stabilities, 'darkred', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Neural Lineage Stability')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Stability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Semantic attractor formation
    axes[0, 2].plot(steps, attractor_counts, 'magenta', linewidth=2, marker='^', markersize=4)
    axes[0, 2].set_title('Semantic Attractor Count')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Attractor Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Attractor density
    axes[1, 0].plot(steps, attractor_densities, 'green', linewidth=2, marker='D', markersize=4)
    axes[1, 0].set_title('Attractor Density')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Attractor stability
    axes[1, 1].plot(steps, attractor_stabilities, 'darkorange', linewidth=2, marker='v', markersize=4)
    axes[1, 1].set_title('Attractor Stability')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Stability')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Growth summary
    axes[1, 2].axis('off')
    if ancestry_strengths and attractor_counts:
        growth_text = f"""Neural Growth Summary:
        
Ancestry Evolution:
• Initial Strength: {ancestry_strengths[0]:.3f}
• Final Strength: {ancestry_strengths[-1]:.3f}
• Stability Range: {min(ancestry_stabilities):.3f} - {max(ancestry_stabilities):.3f}

Attractor Formation:
• Max Attractors: {max(attractor_counts)}
• Mean Density: {np.mean(attractor_densities):.3f}
• Mean Stability: {np.mean(attractor_stabilities):.3f}

Growth Pattern: {"Expanding" if max(attractor_counts) > min(attractor_counts) else "Stable"}
Network Adaptation: {"Active" if np.std(ancestry_strengths) > 0.01 else "Minimal"}
        """
        axes[1, 2].text(0.1, 0.9, growth_text, transform=axes[1, 2].transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                         verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
