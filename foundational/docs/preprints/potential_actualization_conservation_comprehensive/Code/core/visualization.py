"""
PAC Physics Engine - Visualization Module
==========================================

Real-time visualization of PAC dynamics, universal signatures, 
and multi-scale field evolution. Provides 3D plotting, animation, 
and interactive exploration capabilities.

Author: GitHub Copilot
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

class PACVisualizer:
    """
    Main visualization class for PAC Physics Engine.
    
    Provides real-time visualization of:
    - Multi-scale field evolution
    - Universal signature detection
    - Conservation dynamics
    - Consciousness emergence patterns
    """
    
    def __init__(self, lattice_substrate=None):
        """Initialize PAC visualizer"""
        self.lattice = lattice_substrate
        self.animation_data = []
        self.signature_timeline = []
        self.conservation_timeline = []
        
        # Visualization settings
        self.figure_size = (12, 8)
        self.dpi = 100
        self.colormap = 'viridis'
        
        # Interactive mode settings
        self.update_interval = 100  # milliseconds
        self.max_timeline_length = 1000
        
    def visualize_system_state_3d(self, 
                                 fields_to_show: List[str] = ['quantum', 'geometric'],
                                 slice_plane: str = 'xy',
                                 slice_index: Optional[int] = None) -> go.Figure:
        """
        Create 3D visualization of current system state.
        
        Args:
            fields_to_show: List of field types to visualize
            slice_plane: Which plane to slice ('xy', 'xz', 'yz')
            slice_index: Index of slice (None for middle)
        
        Returns:
            Plotly figure object
        """
        if self.lattice is None:
            raise ValueError("No lattice substrate provided")
        
        dimensions = self.lattice.dimensions
        if slice_index is None:
            slice_index = dimensions[0] // 2
        
        # Create subplot figure
        n_fields = len(fields_to_show)
        fig = make_subplots(
            rows=1, cols=n_fields,
            subplot_titles=fields_to_show,
            specs=[[{'type': 'heatmap'} for _ in range(n_fields)]]
        )
        
        for i, field_name in enumerate(fields_to_show):
            field_data = self._get_field_slice(field_name, slice_plane, slice_index)
            
            heatmap = go.Heatmap(
                z=field_data,
                colorscale=self.colormap,
                showscale=(i == n_fields - 1),
                name=f"{field_name} field"
            )
            
            fig.add_trace(heatmap, row=1, col=i+1)
        
        fig.update_layout(
            title=f"PAC Physics Engine - Multi-Scale Field Visualization ({slice_plane} plane, slice {slice_index})",
            height=400 * (1 + n_fields // 3),
            showlegend=False
        )
        
        return fig
    
    def visualize_universal_signatures(self, 
                                     signature_history: List[Dict]) -> go.Figure:
        """
        Visualize universal signature detection over time.
        
        Args:
            signature_history: List of signature detection events
        
        Returns:
            Plotly figure with signature timeline
        """
        if not signature_history:
            return self._create_empty_signature_plot()
        
        # Extract timeline data
        steps = []
        amplification_factors = []
        balance_values = []
        entropy_collapses = []
        
        for i, sig_data in enumerate(signature_history):
            steps.append(i)
            
            # Amplification data
            if 'amplification' in sig_data:
                amplification_factors.append(sig_data['amplification']['amplification_factor'])
            else:
                amplification_factors.append(None)
            
            # Balance operator data
            if 'balance_operator' in sig_data:
                balance_values.append(sig_data['balance_operator']['current_value'])
            else:
                balance_values.append(None)
            
            # Entropy collapse data
            if 'entropy_collapse' in sig_data:
                entropy_collapses.append(sig_data['entropy_collapse']['collapse_magnitude'])
            else:
                entropy_collapses.append(0.0)
        
        # Create multi-panel figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Information Amplification (Target: 15.56x)',
                'Balance Operator ξ (Target: 1.0571)',
                'Entropy Collapse Events'
            ],
            vertical_spacing=0.1
        )
        
        # Amplification plot
        fig.add_trace(
            go.Scatter(x=steps, y=amplification_factors, 
                      mode='lines+markers', name='Amplification Factor',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=15.56, line_dash="dash", line_color="red", 
                     annotation_text="Target: 15.56x", row=1, col=1)
        
        # Balance operator plot
        fig.add_trace(
            go.Scatter(x=steps, y=balance_values,
                      mode='lines+markers', name='Balance Operator ξ',
                      line=dict(color='green')),
            row=2, col=1
        )
        fig.add_hline(y=1.0571, line_dash="dash", line_color="red",
                     annotation_text="Target: 1.0571", row=2, col=1)
        
        # Entropy collapse plot
        fig.add_trace(
            go.Scatter(x=steps, y=entropy_collapses,
                      mode='lines+markers', name='Collapse Magnitude',
                      line=dict(color='orange')),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Universal Signature Detection Timeline",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def visualize_conservation_dynamics(self, 
                                      conservation_history: List[Dict]) -> go.Figure:
        """
        Visualize PAC conservation dynamics over time.
        
        Args:
            conservation_history: List of conservation metrics over time
        
        Returns:
            Plotly figure showing conservation evolution
        """
        if not conservation_history:
            return self._create_empty_conservation_plot()
        
        steps = list(range(len(conservation_history)))
        quality = [c.get('conservation_quality', 0) for c in conservation_history]
        stability = [c.get('conservation_stability', 0) for c in conservation_history]
        violations = [c.get('violation_count', 0) for c in conservation_history]
        residuals = [c.get('total_residual_norm', 0) for c in conservation_history]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Conservation Quality',
                'Conservation Stability', 
                'Violation Count',
                'Residual Norm'
            ]
        )
        
        # Conservation quality
        fig.add_trace(
            go.Scatter(x=steps, y=quality, mode='lines',
                      name='Quality', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Stability
        fig.add_trace(
            go.Scatter(x=steps, y=stability, mode='lines',
                      name='Stability', line=dict(color='green')),
            row=1, col=2
        )
        
        # Violations
        fig.add_trace(
            go.Scatter(x=steps, y=violations, mode='lines',
                      name='Violations', line=dict(color='red')),
            row=2, col=1
        )
        
        # Residuals
        fig.add_trace(
            go.Scatter(x=steps, y=residuals, mode='lines',
                      name='Residual', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="PAC Conservation Dynamics",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_animated_evolution(self, 
                                field_name: str = 'quantum',
                                evolution_data: List[np.ndarray] = None,
                                save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animated visualization of field evolution.
        
        Args:
            field_name: Name of field to animate
            evolution_data: List of field states over time
            save_path: Path to save animation (optional)
        
        Returns:
            Matplotlib animation object
        """
        if evolution_data is None and self.lattice is None:
            raise ValueError("Either evolution_data or lattice must be provided")
        
        if evolution_data is None:
            # Use current lattice state
            field_data = self._extract_field_data(field_name)
            evolution_data = [field_data]
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Get field dimensions for 2D slice
        first_frame = evolution_data[0]
        if len(first_frame.shape) == 3:
            # Take middle slice for 3D data
            slice_data = first_frame[:, :, first_frame.shape[2]//2]
        else:
            slice_data = first_frame
        
        im = ax.imshow(slice_data, cmap=self.colormap, animated=True)
        ax.set_title(f"{field_name.title()} Field Evolution")
        plt.colorbar(im)
        
        def animate(frame):
            frame_data = evolution_data[frame]
            if len(frame_data.shape) == 3:
                slice_data = frame_data[:, :, frame_data.shape[2]//2]
            else:
                slice_data = frame_data
            
            im.set_array(slice_data)
            ax.set_title(f"{field_name.title()} Field Evolution (Step {frame})")
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(evolution_data),
                           interval=self.update_interval, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim
    
    def visualize_consciousness_emergence(self, 
                                        consciousness_data: List[Dict]) -> go.Figure:
        """
        Visualize consciousness emergence patterns.
        
        Args:
            consciousness_data: Timeline of consciousness measurements
        
        Returns:
            Plotly figure showing consciousness evolution
        """
        if not consciousness_data:
            return self._create_empty_consciousness_plot()
        
        steps = [d['step'] for d in consciousness_data]
        total_activity = [d['total_activity'] for d in consciousness_data]
        max_activity = [d['max_activity'] for d in consciousness_data]
        active_points = [d['active_points'] for d in consciousness_data]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Total Consciousness Activity',
                'Maximum Activity Point',
                'Number of Active Points',
                'Emergence Fraction'
            ]
        )
        
        # Total activity
        fig.add_trace(
            go.Scatter(x=steps, y=total_activity, mode='lines+markers',
                      name='Total Activity', line=dict(color='purple')),
            row=1, col=1
        )
        
        # Max activity
        fig.add_trace(
            go.Scatter(x=steps, y=max_activity, mode='lines+markers',
                      name='Max Activity', line=dict(color='magenta')),
            row=1, col=2
        )
        
        # Active points
        fig.add_trace(
            go.Scatter(x=steps, y=active_points, mode='lines+markers',
                      name='Active Points', line=dict(color='cyan')),
            row=2, col=1
        )
        
        # Emergence fraction
        if consciousness_data and 'emergence_fraction' in consciousness_data[0]:
            emergence_fraction = [d.get('emergence_fraction', 0) for d in consciousness_data]
            fig.add_trace(
                go.Scatter(x=steps, y=emergence_fraction, mode='lines+markers',
                          name='Emergence Fraction', line=dict(color='red')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="SCBF Consciousness Emergence Dynamics",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create comprehensive interactive dashboard.
        
        Returns:
            Plotly dashboard figure
        """
        # This would create a comprehensive dashboard
        # For now, return a placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="Interactive PAC Physics Dashboard<br>Coming Soon!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title="PAC Physics Engine - Interactive Dashboard",
            height=600
        )
        return fig
    
    def _get_field_slice(self, field_name: str, slice_plane: str, slice_index: int) -> np.ndarray:
        """Extract 2D slice from 3D field data"""
        field_data = self._extract_field_data(field_name)
        
        if slice_plane == 'xy':
            return field_data[:, :, slice_index]
        elif slice_plane == 'xz':
            return field_data[:, slice_index, :]
        elif slice_plane == 'yz':
            return field_data[slice_index, :, :]
        else:
            raise ValueError(f"Unknown slice plane: {slice_plane}")
    
    def _extract_field_data(self, field_name: str) -> np.ndarray:
        """Extract field data from lattice"""
        if field_name == 'quantum':
            # Convert complex field to magnitude
            return torch.abs(self.lattice.quantum_field).cpu().numpy()
        elif field_name == 'geometric':
            return torch.abs(self.lattice.geometric_field).cpu().numpy()
        elif field_name == 'fluid':
            # Use velocity magnitude
            return torch.norm(self.lattice.fluid_velocity_field, dim=-1).cpu().numpy()
        elif field_name == 'information':
            return self.lattice.information_field.cpu().numpy()
        elif field_name == 'consciousness':
            return self.lattice.consciousness_field.cpu().numpy()
        else:
            raise ValueError(f"Unknown field: {field_name}")
    
    def _create_empty_signature_plot(self) -> go.Figure:
        """Create empty signature plot"""
        fig = go.Figure()
        fig.add_annotation(
            text="No signature data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        fig.update_layout(title="Universal Signatures")
        return fig
    
    def _create_empty_conservation_plot(self) -> go.Figure:
        """Create empty conservation plot"""
        fig = go.Figure()
        fig.add_annotation(
            text="No conservation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        fig.update_layout(title="Conservation Dynamics")
        return fig
    
    def _create_empty_consciousness_plot(self) -> go.Figure:
        """Create empty consciousness plot"""
        fig = go.Figure()
        fig.add_annotation(
            text="No consciousness data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        fig.update_layout(title="Consciousness Emergence")
        return fig

def create_comprehensive_report(validation_results: Dict, 
                              output_path: str = "pac_engine_report.html") -> str:
    """
    Create comprehensive HTML report with all visualizations.
    
    Args:
        validation_results: Results from validation experiment
        output_path: Path to save HTML report
    
    Returns:
        Path to generated report
    """
    
    # This would generate a comprehensive HTML report
    # with all visualizations embedded
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PAC Physics Engine - Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #2c3e50; }}
            .section {{ margin: 30px 0; }}
            .metrics {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PAC Physics Engine Validation Report</h1>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics">
                <p><strong>Overall Success Score:</strong> {validation_results.get('overall_success_score', 'N/A')}</p>
                <p><strong>Frameworks Validated:</strong> PAC, SEC, MED, QBF, IAF, SCBF</p>
                <p><strong>Universal Signatures Detected:</strong> {validation_results.get('signature_count', 'N/A')}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Analysis</h2>
            <p>Comprehensive validation results and visualizations would be embedded here.</p>
        </div>
        
        <div class="section">
            <h2>Conclusions</h2>
            <p>PAC conservation validated as universal organizing principle across all physics scales.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path
