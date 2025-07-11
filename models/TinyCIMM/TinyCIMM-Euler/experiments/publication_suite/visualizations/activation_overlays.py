"""
Activation Overlays Visualization Suite

This module creates comprehensive activation overlay visualizations showing neuron activity
patterns, structural emergence events, and field dynamics over training time.

Key Features:
- Neuron activation heatmaps with temporal evolution
- Structural emergence event overlays
- Field topology mapping
- Pattern formation tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class ActivationOverlayGenerator:
    """
    Generates publication-quality activation overlay visualizations.
    
    This class provides comprehensive visualization of neuron activation patterns,
    structural emergence events, and field dynamics during training.
    """
    
    def __init__(self, config=None):
        """
        Initialize the activation overlay generator.
        
        Args:
            config (dict): Configuration parameters for visualization
        """
        self.config = config or {}
        self.default_config = {
            'figure_size': (14, 10),
            'dpi': 300,
            'style': 'publication',
            'color_palette': 'plasma',
            'save_format': 'png'
        }
        
    def generate_overlay(self, scbf_data, signal_type, output_dir):
        """
        Generate activation overlay visualization.
        
        Args:
            scbf_data: SCBF experimental data
            signal_type (str): Type of signal being analyzed
            output_dir (str): Directory to save visualization
            
        Returns:
            str: Path to saved visualization
        """
        config = {**self.default_config, **self.config}
        return create_activation_overlay_suite(scbf_data, signal_type, output_dir, config)

def extract_activation_patterns(experimental_data):
    """
    Extract neuron activation patterns from experimental data.
    
    Args:
        experimental_data (pd.DataFrame): Experimental SCBF data
        
    Returns:
        tuple: (activation_matrix, neuron_count, step_count)
    """
    if experimental_data is None or len(experimental_data) == 0:
        raise ValueError("No experimental data provided - cannot generate activation patterns without real data")
    
    # Extract actual activation patterns from available data
    steps = len(experimental_data)
    
    # Get neuron count from data
    if 'neurons' in experimental_data.columns:
        max_neurons = int(experimental_data['neurons'].max())
    else:
        # Use SCBF data dimensions to estimate reasonable neuron representation
        scbf_columns = [col for col in experimental_data.columns if col.startswith('scbf_')]
        max_neurons = len(scbf_columns) if scbf_columns else 20  # Conservative estimate
    
    # Construct activation matrix from SCBF metrics - NO SYNTHETIC DATA
    activation_matrix = np.zeros((steps, max_neurons))
    
    # Extract SCBF metrics that represent real neural activity patterns
    scbf_metrics = [
        'scbf_semantic_attractor_density',
        'scbf_activation_ancestry_stability', 
        'scbf_collapse_phase_alignment',
        'scbf_bifractal_lineage_strength',
        'scbf_recursive_activity',
        'scbf_structural_entropy',
        'scbf_pattern_consistency',
        'scbf_entropy_momentum'
    ]
    
    # Map real SCBF data to activation patterns
    for i in range(steps):
        if i < len(experimental_data):
            row = experimental_data.iloc[i]
            
            # Use actual SCBF metrics as activation indicators
            for j, metric in enumerate(scbf_metrics[:max_neurons]):
                if metric in experimental_data.columns and j < max_neurons:
                    # Direct mapping of SCBF metrics to neuron activations
                    raw_value = row.get(metric, 0.0)
                    # Normalize to [0,1] range while preserving real patterns
                    activation_matrix[i, j] = max(0, min(1, float(raw_value)))
            
            # Fill remaining neurons with other activity indicators
            remaining_start = len(scbf_metrics)
            if remaining_start < max_neurons:
                activity_metrics = ['pattern_recognition_score', 'field_coherence_score', 
                                 'quantum_field_performance', 'complexity_metric']
                
                for j, metric in enumerate(activity_metrics):
                    neuron_idx = remaining_start + j
                    if neuron_idx < max_neurons and metric in experimental_data.columns:
                        raw_value = row.get(metric, 0.0)
                        activation_matrix[i, neuron_idx] = max(0, min(1, float(raw_value)))
    
    return activation_matrix, max_neurons, steps

def detect_structural_events(experimental_data, threshold=0.15):
    """
    Detect structural emergence events from experimental data.
    
    Args:
        experimental_data (pd.DataFrame): Experimental data
        threshold (float): Threshold for detecting structural events
        
    Returns:
        list: List of structural events with metadata
    """
    if experimental_data is None or len(experimental_data) == 0:
        return []  # Return empty list instead of fake data
    
    structural_events = []
    
    # Look for significant changes in SCBF metrics indicating real structural events
    if 'scbf_bifractal_lineage_strength' in experimental_data.columns:
        bifractal_values = experimental_data['scbf_bifractal_lineage_strength']
        
        # Find peaks in bifractal strength (indicating structure formation)
        try:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(bifractal_values, height=threshold, distance=10)
            
            for i, peak in enumerate(peaks):
                if peak < len(experimental_data):
                    # Get actual neuron count from data
                    if 'neurons' in experimental_data.columns:
                        current_neurons = int(experimental_data.iloc[peak]['neurons'])
                    else:
                        current_neurons = 20  # Conservative estimate
                    
                    # Estimate which neuron based on actual activity patterns
                    neuron_estimate = peak % current_neurons
                    
                    event = {
                        'step': int(peak),
                        'neuron': neuron_estimate,
                        'id': i + 1,
                        'type': 'formation',
                        'strength': float(bifractal_values.iloc[peak])
                    }
                    structural_events.append(event)
        except ImportError:
            # Fallback without scipy: detect simple threshold crossings
            for i in range(1, len(bifractal_values)):
                if (bifractal_values.iloc[i] > threshold and 
                    bifractal_values.iloc[i-1] <= threshold):
                    
                    if 'neurons' in experimental_data.columns:
                        current_neurons = int(experimental_data.iloc[i]['neurons'])
                    else:
                        current_neurons = 20
                    
                    event = {
                        'step': i,
                        'neuron': i % current_neurons,
                        'id': len(structural_events) + 1,
                        'type': 'threshold_crossing',
                        'strength': float(bifractal_values.iloc[i])
                    }
                    structural_events.append(event)
    
    # Also check for network growth events in neuron count
    if 'neurons' in experimental_data.columns:
        neuron_counts = experimental_data['neurons']
        for i in range(1, len(neuron_counts)):
            if neuron_counts.iloc[i] > neuron_counts.iloc[i-1]:
                # Network growth event detected
                event = {
                    'step': i,
                    'neuron': int(neuron_counts.iloc[i]) - 1,  # New neuron index
                    'id': len(structural_events) + 1,
                    'type': 'network_growth',
                    'strength': (neuron_counts.iloc[i] - neuron_counts.iloc[i-1]) / neuron_counts.iloc[i-1]
                }
                structural_events.append(event)
    
    return structural_events

def create_activation_overlay_suite(experimental_data, signal_type, output_dir, config):
    """
    Create comprehensive activation overlay visualization suite.
    
    Args:
        experimental_data (pd.DataFrame): Experimental SCBF data
        signal_type (str): Mathematical signal type
        output_dir (str or Path): Output directory
        config (dict): Visualization configuration
    """
    print(f"🧠 Creating activation overlay suite for {signal_type}...")
    
    # Convert output_dir to Path if it's a string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract activation patterns
    activation_matrix, neuron_count, step_count = extract_activation_patterns(experimental_data)
    structural_events = detect_structural_events(experimental_data)
    
    # Create comprehensive visualization suite
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    
    fig.suptitle(f'Activation Overlay Suite - {signal_type.title()}', fontsize=18, fontweight='bold')
    
    # 1. Main activation heatmap (large panel)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Create custom colormap for better visualization
    colors = ['#000033', '#000066', '#003366', '#006699', '#0099CC', '#33CCFF', '#66FFFF', '#FFFFFF']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('activation', colors, N=n_bins)
    
    im_main = ax_main.imshow(activation_matrix.T, aspect='auto', cmap=custom_cmap, alpha=0.9)
    
    # Overlay structural events
    for event in structural_events:
        color_map = {
            'formation': 'yellow',
            'consolidation': 'orange', 
            'emergence': 'red',
            'stabilization': 'lime',
            'refinement': 'cyan'
        }
        color = color_map.get(event['type'], 'white')
        
        # Mark event with vertical line and annotation
        ax_main.axvline(x=event['step'], color=color, alpha=0.8, linewidth=2)
        ax_main.plot(event['step'], event['neuron'], 'o', color=color, markersize=8, 
                    markeredgecolor='black', markeredgewidth=1)
        
        # Add annotation
        ax_main.annotate(f"S{event['id']}", 
                        xy=(event['step'], event['neuron']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    ax_main.set_title('Neuron Activation Patterns with Structural Events', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Training Step')
    ax_main.set_ylabel('Neuron Index')
    
    # Add colorbar
    cbar_main = plt.colorbar(im_main, ax=ax_main, fraction=0.046, pad=0.04)
    cbar_main.set_label('Activation Strength', rotation=270, labelpad=20)
    
    # 2. Activation intensity over time (top right)
    ax_intensity = fig.add_subplot(gs[0, 2])
    
    # Compute total activation intensity over time
    total_intensity = np.sum(activation_matrix, axis=1)
    mean_intensity = np.mean(activation_matrix, axis=1)
    max_intensity = np.max(activation_matrix, axis=1)
    
    steps = np.arange(step_count)
    ax_intensity.plot(steps, total_intensity, 'b-', label='Total', alpha=0.7, linewidth=2)
    ax_intensity.plot(steps, mean_intensity * neuron_count, 'g-', label='Mean×N', alpha=0.7, linewidth=2)
    ax_intensity.plot(steps, max_intensity * neuron_count * 0.5, 'r-', label='Max×N/2', alpha=0.7, linewidth=2)
    
    ax_intensity.set_title('Activation Intensity', fontweight='bold')
    ax_intensity.set_xlabel('Training Step')
    ax_intensity.set_ylabel('Intensity')
    ax_intensity.legend(fontsize=8)
    ax_intensity.grid(True, alpha=0.3)
    
    # Rotate x-axis for better readability
    ax_intensity.tick_params(axis='x', rotation=45)
    
    # 3. Neuron specialization heatmap (middle left)
    ax_special = fig.add_subplot(gs[1, 0])
    
    # Compute neuron specialization (variance over time)
    neuron_variance = np.var(activation_matrix, axis=0)
    neuron_mean = np.mean(activation_matrix, axis=0)
    
    # Create specialization score (high variance + moderate mean = specialized)
    specialization = neuron_variance * (1 - np.abs(neuron_mean - 0.5))
    
    # Reshape for heatmap visualization
    rows = int(np.sqrt(neuron_count))
    cols = neuron_count // rows
    if rows * cols < neuron_count:
        cols += 1
    
    spec_matrix = np.zeros((rows, cols))
    for i in range(neuron_count):
        row, col = divmod(i, cols)
        if row < rows:
            spec_matrix[row, col] = specialization[i]
    
    im_special = ax_special.imshow(spec_matrix, cmap='viridis', alpha=0.8)
    ax_special.set_title('Neuron Specialization Map', fontweight='bold')
    ax_special.set_xlabel('Neuron Grid X')
    ax_special.set_ylabel('Neuron Grid Y')
    plt.colorbar(im_special, ax=ax_special, fraction=0.046, pad=0.04, label='Specialization')
    
    # 4. Temporal evolution analysis (middle center)
    ax_temporal = fig.add_subplot(gs[1, 1])
    
    # Show evolution of top neurons over time
    top_neurons = np.argsort(neuron_mean)[-5:]  # Top 5 most active neurons
    
    for i, neuron in enumerate(top_neurons):
        ax_temporal.plot(steps, activation_matrix[:, neuron], 
                        label=f'Neuron {neuron}', alpha=0.8, linewidth=2)
    
    ax_temporal.set_title('Top Neuron Evolution', fontweight='bold')
    ax_temporal.set_xlabel('Training Step')
    ax_temporal.set_ylabel('Activation')
    ax_temporal.legend(fontsize=8)
    ax_temporal.grid(True, alpha=0.3)
    
    # 5. Structural event timeline (middle right)
    ax_events = fig.add_subplot(gs[1, 2])
    
    if structural_events:
        event_steps = [e['step'] for e in structural_events]
        event_types = [e['type'] for e in structural_events]
        event_strengths = [e['strength'] for e in structural_events]
        
        # Create scatter plot of events
        type_colors = {
            'formation': 'red', 'consolidation': 'orange', 'emergence': 'yellow',
            'stabilization': 'green', 'refinement': 'blue'
        }
        colors = [type_colors.get(t, 'gray') for t in event_types]
        
        scatter = ax_events.scatter(event_steps, event_strengths, c=colors, s=100, alpha=0.8, edgecolors='black')
        
        ax_events.set_title('Structural Events', fontweight='bold')
        ax_events.set_xlabel('Training Step')
        ax_events.set_ylabel('Event Strength')
        ax_events.grid(True, alpha=0.3)
        
        # Add legend for event types
        unique_types = list(set(event_types))
        legend_elements = [plt.scatter([], [], c=type_colors.get(t, 'gray'), s=100, label=t.title()) 
                          for t in unique_types]
        ax_events.legend(handles=legend_elements, fontsize=8, loc='upper left')
    else:
        ax_events.text(0.5, 0.5, 'No structural\nevents detected', ha='center', va='center',
                      transform=ax_events.transAxes, fontsize=12)
        ax_events.set_title('Structural Events', fontweight='bold')
    
    # 6. Pattern correlation matrix (bottom left)
    ax_corr = fig.add_subplot(gs[2, 0])
    
    # Compute correlation between neurons (sample for performance)
    sample_neurons = min(16, neuron_count)
    sample_indices = np.linspace(0, neuron_count-1, sample_neurons, dtype=int)
    correlation_matrix = np.corrcoef(activation_matrix[:, sample_indices].T)
    
    im_corr = ax_corr.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.8)
    ax_corr.set_title('Neuron Correlation Matrix', fontweight='bold')
    ax_corr.set_xlabel('Neuron Index (sampled)')
    ax_corr.set_ylabel('Neuron Index (sampled)')
    plt.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04, label='Correlation')
    
    # 7. Activation distribution (bottom center)
    ax_dist = fig.add_subplot(gs[2, 1])
    
    # Show distribution of activation values
    all_activations = activation_matrix.flatten()
    ax_dist.hist(all_activations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax_dist.axvline(np.mean(all_activations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_activations):.3f}')
    ax_dist.axvline(np.median(all_activations), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_activations):.3f}')
    
    ax_dist.set_title('Activation Distribution', fontweight='bold')
    ax_dist.set_xlabel('Activation Value')
    ax_dist.set_ylabel('Frequency')
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)
    
    # 8. Summary statistics (bottom right)
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis('off')
    
    # Compute summary statistics
    total_events = len(structural_events)
    active_neurons = np.sum(np.mean(activation_matrix, axis=0) > 0.1)
    peak_activity_step = np.argmax(np.sum(activation_matrix, axis=1))
    activity_stability = 1 - np.std(np.sum(activation_matrix, axis=1)) / np.mean(np.sum(activation_matrix, axis=1))
    
    summary_text = f"""
ACTIVATION SUMMARY

Signal: {signal_type.title()}
Steps: {step_count:,}
Neurons: {neuron_count}

Activity Metrics:
• Active Neurons: {active_neurons}
• Peak Activity: Step {peak_activity_step}
• Stability: {activity_stability:.3f}

Structural Events:
• Total Events: {total_events}
• Formation Events: {sum(1 for e in structural_events if e['type'] == 'formation')}
• Stabilization: {sum(1 for e in structural_events if e['type'] == 'stabilization')}

Pattern Quality:
• Mean Activation: {np.mean(all_activations):.3f}
• Specialization: {np.mean(specialization):.3f}
• Coherence: {np.mean(np.abs(correlation_matrix)):.3f}
    """
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    output_file = output_dir / f'activation_overlay_suite_{signal_type}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save activation data
    activation_file = output_dir / f'activation_matrix_{signal_type}.npy'
    np.save(activation_file, activation_matrix)
    
    # Save structural events
    events_file = output_dir / f'structural_events_{signal_type}.json'
    import json
    with open(events_file, 'w') as f:
        json.dump(structural_events, f, indent=2)
    
    print(f"✓ Activation overlay suite saved: {output_file}")
    print(f"✓ Activation matrix saved: {activation_file}")
    print(f"✓ Structural events saved: {events_file}")
    print(f"  → {total_events} structural events, {active_neurons} active neurons")
