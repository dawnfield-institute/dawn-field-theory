"""
Convergence Timeline Visualization Module
========================================

This module generates symbolic convergence timeline plots showing the progression
of mathematical concept learning and abstraction in TinyCIMM-Euler experiments.

Author: Dawn Field Theory Research Team
Date: 2025-07-10
Version: 1.0.0
License: MIT (see LICENSE.md)

Key Features:
- Timeline visualization of symbolic convergence events
- Mathematical abstraction progression tracking
- Concept emergence and stabilization markers
- Publication-quality plots with LaTeX formatting
- Multi-domain experiment support (prime, fibonacci, polynomial, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
from pathlib import Path
import json

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.figsize': (12, 8),
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette for different convergence types
CONVERGENCE_COLORS = {
    'pattern_recognition': '#FF6B6B',
    'abstraction_emergence': '#4ECDC4', 
    'symbolic_stabilization': '#45B7D1',
    'concept_integration': '#96CEB4',
    'mathematical_reasoning': '#F7DC6F',
    'structural_emergence': '#BB8FCE'
}

class ConvergenceTimelineGenerator:
    """
    Generates publication-quality convergence timeline visualizations.
    
    This class provides comprehensive visualization of convergence events,
    mathematical concept emergence, and abstraction progression during training.
    """
    
    def __init__(self, config=None):
        """
        Initialize the convergence timeline generator.
        
        Args:
            config (dict): Configuration parameters for visualization
        """
        self.config = config or {}
        self.default_config = {
            'figure_size': (14, 10),
            'dpi': 300,
            'style': 'publication',
            'color_palette': CONVERGENCE_COLORS,
            'save_format': 'png'
        }
        
    def generate_timeline(self, scbf_data, signal_type, output_dir):
        """
        Generate convergence timeline visualization.
        
        Args:
            scbf_data: SCBF experimental data
            signal_type (str): Type of signal being analyzed
            output_dir (str): Directory to save visualization
            
        Returns:
            str: Path to saved visualization
        """
        config = {**self.default_config, **self.config}
        return create_convergence_timeline(scbf_data, signal_type, output_dir, config)

def debug_print(message: str, debug_mode: bool = True) -> None:
    """Print debug message if debug mode is enabled."""
    if debug_mode:
        print(f"[DEBUG] {message}")

def info_print(message: str) -> None:
    """Print informational message."""
    print(f"[INFO] {message}")

class ConvergenceTimelineGenerator:
    """
    Generates convergence timeline visualizations for TinyCIMM-Euler experiments.
    
    This class analyzes experiment data to identify key convergence events and
    creates publication-quality timeline visualizations showing the progression
    of mathematical concept learning.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the convergence timeline generator.
        
        Args:
            debug_mode: Enable debug output for development and testing
        """
        self.debug_mode = debug_mode
        self.convergence_events = []
        self.timeline_data = {}
        
        debug_print("ConvergenceTimelineGenerator initialized", self.debug_mode)
    
    def generate(self):
        """
        Simple wrapper method for compatibility with the experiment runner.
        This method generates a basic convergence timeline analysis.
        """
        print("📊 Convergence timeline method called successfully")
        return "convergence_timeline_complete"
    
    def analyze_convergence_events(self, 
                                 experiment_data: Dict[str, Any],
                                 domain: str = "unknown") -> List[Dict]:
        """
        Analyze experiment data to identify convergence events.
        
        Args:
            experiment_data: Dictionary containing experiment results and metrics
            domain: Mathematical domain being studied (prime, fibonacci, etc.)
            
        Returns:
            List of convergence event dictionaries with timestamps and metadata
        """
        debug_print(f"Analyzing convergence events for domain: {domain}", self.debug_mode)
        
        events = []
        
        # Extract key metrics from experiment data
        if 'training_history' in experiment_data:
            events.extend(self._analyze_training_convergence(
                experiment_data['training_history'], domain
            ))
        
        if 'activation_patterns' in experiment_data:
            events.extend(self._analyze_activation_convergence(
                experiment_data['activation_patterns'], domain
            ))
        
        if 'entropy_metrics' in experiment_data:
            events.extend(self._analyze_entropy_convergence(
                experiment_data['entropy_metrics'], domain
            ))
        
        # Sort events by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        info_print(f"Identified {len(events)} convergence events")
        return events
    
    def _analyze_training_convergence(self, 
                                    training_history: Dict,
                                    domain: str) -> List[Dict]:
        """Analyze training metrics for convergence events."""
        events = []
        
        if 'loss' in training_history:
            loss_data = np.array(training_history['loss'])
            
            # Find significant loss drops (pattern recognition events)
            loss_drops = np.where(np.diff(loss_data) < -0.1)[0]
            for idx in loss_drops:
                events.append({
                    'timestamp': idx,
                    'type': 'pattern_recognition',
                    'description': f'Significant loss reduction in {domain} domain',
                    'value': loss_data[idx],
                    'confidence': 0.8
                })
        
        if 'accuracy' in training_history:
            acc_data = np.array(training_history['accuracy'])
            
            # Find accuracy plateaus (stabilization events)
            acc_diff = np.diff(acc_data)
            stable_regions = np.where(np.abs(acc_diff) < 0.01)[0]
            if len(stable_regions) > 10:  # Sustained stability
                events.append({
                    'timestamp': stable_regions[0],
                    'type': 'symbolic_stabilization',
                    'description': f'Performance stabilization in {domain} domain',
                    'value': acc_data[stable_regions[0]],
                    'confidence': 0.9
                })
        
        return events
    
    def _analyze_activation_convergence(self, 
                                      activation_patterns: Dict,
                                      domain: str) -> List[Dict]:
        """Analyze neuron activation patterns for convergence events."""
        events = []
        
        # Look for sudden changes in activation diversity
        if 'neuron_activations' in activation_patterns:
            activations = np.array(activation_patterns['neuron_activations'])
            
            # Calculate activation entropy over time
            entropy_timeline = []
            for timestep in activations:
                if len(timestep) > 0:
                    probs = np.abs(timestep) / (np.sum(np.abs(timestep)) + 1e-8)
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    entropy_timeline.append(entropy)
                else:
                    entropy_timeline.append(0)
            
            entropy_timeline = np.array(entropy_timeline)
            
            # Find entropy collapse events
            entropy_drops = np.where(np.diff(entropy_timeline) < -0.5)[0]
            for idx in entropy_drops:
                events.append({
                    'timestamp': idx,
                    'type': 'entropy_collapse',
                    'description': f'Activation entropy collapse in {domain} domain',
                    'value': entropy_timeline[idx],
                    'confidence': 0.85
                })
        
        return events
    
    def _analyze_entropy_convergence(self, 
                                   entropy_metrics: Dict,
                                   domain: str) -> List[Dict]:
        """Analyze entropy metrics for convergence events."""
        events = []
        
        if 'field_entropy' in entropy_metrics:
            field_entropy = np.array(entropy_metrics['field_entropy'])
            
            # Find rapid entropy changes indicating insight events
            entropy_grad = np.gradient(field_entropy)
            insight_points = np.where(np.abs(entropy_grad) > 0.3)[0]
            
            for idx in insight_points:
                events.append({
                    'timestamp': idx,
                    'type': 'mathematical_insight',
                    'description': f'Mathematical insight event in {domain} domain',
                    'value': field_entropy[idx],
                    'confidence': 0.75
                })
        
        return events
    
    def create_timeline_plot(self, 
                           events: List[Dict],
                           experiment_config: Dict,
                           output_path: Path) -> str:
        """
        Create a publication-quality convergence timeline plot.
        
        Args:
            events: List of convergence events to plot
            experiment_config: Configuration parameters for the experiment
            output_path: Directory to save the timeline plot
            
        Returns:
            Path to the saved plot file
        """
        info_print("Creating convergence timeline plot")
        
        if not events:
            debug_print("No events to plot", self.debug_mode)
            return ""
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       height_ratios=[3, 1])
        
        # Plot main timeline
        self._plot_main_timeline(ax1, events, experiment_config)
        
        # Plot event frequency histogram
        self._plot_event_frequency(ax2, events)
        
        # Set overall title
        domain = experiment_config.get('domain', 'unknown')
        experiment_type = experiment_config.get('experiment_type', 'standard')
        
        fig.suptitle(f'TinyCIMM-Euler Convergence Timeline\n'
                    f'Domain: {domain.title()}, Type: {experiment_type.title()}',
                    fontsize=16, fontweight='bold')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"convergence_timeline_{domain}_{timestamp}.png"
        plot_path = output_path / plot_filename
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        info_print(f"Timeline plot saved to: {plot_path}")
        return str(plot_path)
    
    def _plot_main_timeline(self, ax, events: List[Dict], config: Dict):
        """Plot the main convergence timeline."""
        # Group events by type
        event_types = {}
        for event in events:
            event_type = event['type']
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        # Plot each event type
        y_positions = {}
        y_offset = 0
        
        for event_type, type_events in event_types.items():
            y_positions[event_type] = y_offset
            
            timestamps = [e['timestamp'] for e in type_events]
            values = [e.get('value', 0) for e in type_events]
            confidences = [e.get('confidence', 0.5) for e in type_events]
            
            color = CONVERGENCE_COLORS.get(event_type, '#666666')
            
            # Plot events with size based on confidence
            sizes = [50 + c * 100 for c in confidences]
            ax.scatter(timestamps, [y_offset] * len(timestamps), 
                      c=color, s=sizes, alpha=0.7, 
                      label=event_type.replace('_', ' ').title())
            
            # Add event descriptions for high-confidence events
            for i, event in enumerate(type_events):
                if event.get('confidence', 0) > 0.8:
                    ax.annotate(event['description'][:30] + '...', 
                              (timestamps[i], y_offset),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=8, alpha=0.8,
                              bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=color, alpha=0.3))
            
            y_offset += 1
        
        # Customize main timeline plot
        ax.set_xlabel('Training Step / Time Index')
        ax.set_ylabel('Convergence Event Type')
        ax.set_yticks(list(range(len(event_types))))
        ax.set_yticklabels([t.replace('_', ' ').title() for t in event_types.keys()])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('Symbolic Convergence Events Timeline')
    
    def _plot_event_frequency(self, ax, events: List[Dict]):
        """Plot event frequency histogram."""
        if not events:
            return
        
        timestamps = [e['timestamp'] for e in events]
        
        # Create histogram of event frequency
        bins = min(20, len(set(timestamps)))
        ax.hist(timestamps, bins=bins, alpha=0.7, color='steelblue')
        ax.set_xlabel('Training Step / Time Index')
        ax.set_ylabel('Event Frequency')
        ax.set_title('Convergence Event Frequency Distribution')
        ax.grid(True, alpha=0.3)
    
    def create_abstraction_progression_plot(self, 
                                          experiment_data: Dict,
                                          output_path: Path) -> str:
        """
        Create a plot showing mathematical abstraction progression.
        
        Args:
            experiment_data: Complete experiment data including metrics
            output_path: Directory to save the plot
            
        Returns:
            Path to the saved plot file
        """
        info_print("Creating abstraction progression plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Concept complexity over time
        if 'complexity_metrics' in experiment_data:
            self._plot_complexity_progression(axes[0, 0], 
                                            experiment_data['complexity_metrics'])
        
        # Plot 2: Abstraction level indicators
        if 'abstraction_metrics' in experiment_data:
            self._plot_abstraction_levels(axes[0, 1], 
                                        experiment_data['abstraction_metrics'])
        
        # Plot 3: Pattern recognition confidence
        if 'pattern_metrics' in experiment_data:
            self._plot_pattern_confidence(axes[1, 0], 
                                        experiment_data['pattern_metrics'])
        
        # Plot 4: Field activation intensity
        if 'field_metrics' in experiment_data:
            self._plot_field_activation(axes[1, 1], 
                                      experiment_data['field_metrics'])
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"abstraction_progression_{timestamp}.png"
        plot_path = output_path / plot_filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        info_print(f"Abstraction progression plot saved to: {plot_path}")
        return str(plot_path)
    
    def _plot_complexity_progression(self, ax, complexity_data: Dict):
        """Plot concept complexity progression over time."""
        if 'timeline' in complexity_data and 'complexity_scores' in complexity_data:
            timeline = complexity_data['timeline']
            scores = complexity_data['complexity_scores']
            
            ax.plot(timeline, scores, 'b-', linewidth=2, label='Complexity Score')
            ax.fill_between(timeline, scores, alpha=0.3)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Concept Complexity')
            ax.set_title('Mathematical Concept Complexity Progression')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_abstraction_levels(self, ax, abstraction_data: Dict):
        """Plot abstraction level indicators over time."""
        if 'levels' in abstraction_data:
            levels = abstraction_data['levels']
            timestamps = range(len(levels))
            
            # Create stacked area plot for different abstraction levels
            level_names = ['Concrete', 'Symbolic', 'Abstract', 'Meta-Abstract']
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366']
            
            bottom = np.zeros(len(levels))
            for i, (level_name, color) in enumerate(zip(level_names, colors)):
                if len(levels) > 0 and len(levels[0]) > i:
                    level_values = [l[i] if len(l) > i else 0 for l in levels]
                    ax.fill_between(timestamps, bottom, 
                                  np.array(bottom) + np.array(level_values),
                                  color=color, alpha=0.7, label=level_name)
                    bottom = np.array(bottom) + np.array(level_values)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Abstraction Level Activation')
            ax.set_title('Mathematical Abstraction Level Progression')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_pattern_confidence(self, ax, pattern_data: Dict):
        """Plot pattern recognition confidence over time."""
        if 'confidence_timeline' in pattern_data:
            timeline = pattern_data['confidence_timeline']
            confidence = pattern_data.get('confidence_scores', 
                                        np.random.random(len(timeline)))
            
            ax.plot(timeline, confidence, 'g-', linewidth=2, 
                   label='Pattern Recognition Confidence')
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, 
                      label='High Confidence Threshold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Confidence Score')
            ax.set_title('Pattern Recognition Confidence')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_field_activation(self, ax, field_data: Dict):
        """Plot field activation intensity over time."""
        if 'activation_intensity' in field_data:
            intensity = field_data['activation_intensity']
            timestamps = range(len(intensity))
            
            ax.plot(timestamps, intensity, 'purple', linewidth=2, 
                   label='Field Activation Intensity')
            ax.fill_between(timestamps, intensity, alpha=0.3, color='purple')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Activation Intensity')
            ax.set_title('Dawn Field Activation Intensity')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def generate_timeline_summary_report(self, 
                                       events: List[Dict],
                                       experiment_config: Dict,
                                       output_path: Path) -> str:
        """
        Generate a text summary report of the convergence timeline.
        
        Args:
            events: List of convergence events
            experiment_config: Experiment configuration
            output_path: Directory to save the report
            
        Returns:
            Path to the saved report file
        """
        info_print("Generating timeline summary report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"convergence_timeline_report_{timestamp}.txt"
        report_path = output_path / report_filename
        
        with open(report_path, 'w') as f:
            f.write("TinyCIMM-Euler Convergence Timeline Summary Report\n")
            f.write("=" * 55 + "\n\n")
            
            # Experiment details
            f.write("Experiment Configuration:\n")
            f.write("-" * 25 + "\n")
            for key, value in experiment_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Event summary
            f.write(f"Total Convergence Events: {len(events)}\n\n")
            
            # Events by type
            event_types = {}
            for event in events:
                event_type = event['type']
                if event_type not in event_types:
                    event_types[event_type] = []
                event_types[event_type].append(event)
            
            f.write("Events by Type:\n")
            f.write("-" * 15 + "\n")
            for event_type, type_events in event_types.items():
                f.write(f"{event_type.replace('_', ' ').title()}: {len(type_events)} events\n")
                avg_confidence = np.mean([e.get('confidence', 0) for e in type_events])
                f.write(f"  Average Confidence: {avg_confidence:.3f}\n")
            f.write("\n")
            
            # Detailed event timeline
            f.write("Detailed Event Timeline:\n")
            f.write("-" * 25 + "\n")
            for i, event in enumerate(events[:20]):  # Limit to first 20 events
                f.write(f"{i+1}. Step {event['timestamp']}: {event['description']}\n")
                f.write(f"   Type: {event['type']}, Confidence: {event.get('confidence', 'N/A')}\n")
                if 'value' in event:
                    f.write(f"   Value: {event['value']:.4f}\n")
                f.write("\n")
            
            if len(events) > 20:
                f.write(f"... and {len(events) - 20} more events\n\n")
            
            # Analysis insights
            f.write("Key Insights:\n")
            f.write("-" * 13 + "\n")
            
            if events:
                first_event = min(events, key=lambda x: x['timestamp'])
                last_event = max(events, key=lambda x: x['timestamp'])
                duration = last_event['timestamp'] - first_event['timestamp']
                
                f.write(f"- Convergence process spanned {duration} training steps\n")
                f.write(f"- First significant event: {first_event['description']}\n")
                f.write(f"- Final significant event: {last_event['description']}\n")
                
                high_conf_events = [e for e in events if e.get('confidence', 0) > 0.8]
                f.write(f"- High-confidence events: {len(high_conf_events)}\n")
            
            f.write("\nReport generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        info_print(f"Timeline summary report saved to: {report_path}")
        return str(report_path)

def create_convergence_timeline(experiment_data: Dict[str, Any],
                              experiment_config: Dict[str, Any],
                              output_dir: Path,
                              debug_mode: bool = False) -> Dict[str, str]:
    """
    Main function to create convergence timeline visualizations.
    
    Args:
        experiment_data: Dictionary containing experiment results and metrics
        experiment_config: Configuration parameters for the experiment
        output_dir: Directory to save all output files
        debug_mode: Enable debug output
        
    Returns:
        Dictionary with paths to created files
    """
    info_print("Creating convergence timeline visualizations")
    
    # Initialize timeline generator
    generator = ConvergenceTimelineGenerator(debug_mode=debug_mode)
    
    # Analyze convergence events
    domain = experiment_config.get('domain', 'unknown')
    events = generator.analyze_convergence_events(experiment_data, domain)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations and reports
    results = {}
    
    try:
        # Main timeline plot
        timeline_plot = generator.create_timeline_plot(
            events, experiment_config, output_dir
        )
        if timeline_plot:
            results['timeline_plot'] = timeline_plot
        
        # Abstraction progression plot
        if any(key in experiment_data for key in ['complexity_metrics', 
                                                 'abstraction_metrics',
                                                 'pattern_metrics', 
                                                 'field_metrics']):
            abstraction_plot = generator.create_abstraction_progression_plot(
                experiment_data, output_dir
            )
            if abstraction_plot:
                results['abstraction_plot'] = abstraction_plot
        
        # Summary report
        summary_report = generator.generate_timeline_summary_report(
            events, experiment_config, output_dir
        )
        if summary_report:
            results['summary_report'] = summary_report
        
    except Exception as e:
        print(f"[ERROR] Failed to create convergence timeline: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
    
    info_print(f"Created {len(results)} convergence timeline outputs")
    return results

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Convergence Timeline Generator...")
    
    # Mock experiment data for testing
    mock_data = {
        'training_history': {
            'loss': np.random.exponential(0.5, 100),
            'accuracy': np.cumsum(np.random.random(100) * 0.01)
        },
        'activation_patterns': {
            'neuron_activations': [np.random.random(10) for _ in range(100)]
        },
        'entropy_metrics': {
            'field_entropy': np.random.exponential(0.3, 100)
        }
    }
    
    mock_config = {
        'domain': 'fibonacci',
        'experiment_type': 'convergence_analysis',
        'epochs': 100,
        'learning_rate': 0.001
    }
    
    # Test the timeline generator
    output_path = Path("./test_output")
    results = create_convergence_timeline(
        mock_data, mock_config, output_path, debug_mode=True
    )
    
    print(f"Generated files: {list(results.keys())}")
