"""
PAC Flow Visualization

Real-time visualization of PAC conservation flows across all physics scales.
Provides 4D visualization capabilities for understanding the dynamics of
parent-children conservation relationships.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
from typing import Dict, List, Tuple, Optional, Any
import time

class PACFlowVisualizer:
    """Visualizes PAC conservation flows in real-time"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Visualization parameters
        self.colormap = 'viridis'
        self.flow_alpha = 0.7
        self.conservation_threshold = 1e-6
        
        # Plotly configuration
        self.plotly_config = {
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
        
    def visualize_pac_flows_4d(self, states_history: List[Dict[str, torch.Tensor]], 
                             save_path: Optional[str] = None) -> go.Figure:
        """Create 4D visualization of PAC flows over time"""
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Conservation Flows', 'Energy Distribution', 
                          'Scale Interactions', 'Temporal Evolution'),
            specs=[[{'type': 'scatter3d'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Process temporal data
        time_points = len(states_history)
        
        # Extract flow data
        flow_data = self._extract_flow_data(states_history)
        
        # 1. 3D Conservation Flow Visualization
        self._add_3d_flow_plot(fig, flow_data, row=1, col=1)
        
        # 2. Energy Distribution Heatmap
        self._add_energy_heatmap(fig, flow_data, row=1, col=2)
        
        # 3. Scale Interaction Network
        self._add_scale_interactions(fig, flow_data, row=2, col=1)
        
        # 4. Temporal Evolution
        self._add_temporal_evolution(fig, flow_data, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="PAC Conservation Flow Visualization",
            height=800,
            showlegend=True,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _extract_flow_data(self, states_history: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Extract flow data from states history"""
        
        flow_data = {
            'time_points': list(range(len(states_history))),
            'conservation_flows': [],
            'energy_distributions': [],
            'scale_interactions': [],
            'conservation_quality': []
        }
        
        for t, state_dict in enumerate(states_history):
            # Calculate conservation flows
            flows = self._calculate_conservation_flows(state_dict)
            flow_data['conservation_flows'].append(flows)
            
            # Energy distribution
            energy_dist = self._calculate_energy_distribution(state_dict)
            flow_data['energy_distributions'].append(energy_dist)
            
            # Scale interactions
            interactions = self._calculate_scale_interactions(state_dict)
            flow_data['scale_interactions'].append(interactions)
            
            # Conservation quality
            quality = self._assess_conservation_quality(state_dict)
            flow_data['conservation_quality'].append(quality)
        
        return flow_data
    
    def _calculate_conservation_flows(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate conservation flows between parent and children states"""
        
        flows = {}
        
        for scale_name, state in state_dict.items():
            if torch.is_tensor(state) and state.numel() > 1:
                # Calculate flow magnitude
                flow_magnitude = torch.norm(state).item()
                flows[f"{scale_name}_magnitude"] = flow_magnitude
                
                # Calculate flow direction (gradient)
                if state.dim() >= 2:
                    grad_x = torch.diff(state, dim=-1).norm().item()
                    grad_y = torch.diff(state, dim=-2).norm().item()
                    flows[f"{scale_name}_grad_x"] = grad_x
                    flows[f"{scale_name}_grad_y"] = grad_y
                
                # Conservation residual
                total_sum = state.sum().item()
                flows[f"{scale_name}_sum"] = total_sum
        
        return flows
    
    def _calculate_energy_distribution(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Calculate energy distribution across scales"""
        
        energy_matrix = np.zeros((len(state_dict), len(state_dict)))
        
        scale_names = list(state_dict.keys())
        
        for i, scale1 in enumerate(scale_names):
            for j, scale2 in enumerate(scale_names):
                if scale1 in state_dict and scale2 in state_dict:
                    state1 = state_dict[scale1]
                    state2 = state_dict[scale2]
                    
                    if torch.is_tensor(state1) and torch.is_tensor(state2):
                        # Calculate energy coupling
                        energy1 = torch.norm(state1).item()
                        energy2 = torch.norm(state2).item()
                        coupling = min(energy1, energy2) / (max(energy1, energy2) + 1e-6)
                        energy_matrix[i, j] = coupling
        
        return energy_matrix
    
    def _calculate_scale_interactions(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate interactions between different scales"""
        
        interactions = {}
        scale_names = list(state_dict.keys())
        
        for i, scale1 in enumerate(scale_names):
            for j, scale2 in enumerate(scale_names[i+1:], i+1):
                if scale1 in state_dict and scale2 in state_dict:
                    state1 = state_dict[scale1]
                    state2 = state_dict[scale2]
                    
                    if torch.is_tensor(state1) and torch.is_tensor(state2):
                        # Calculate correlation
                        if state1.numel() == state2.numel():
                            correlation = torch.corrcoef(torch.stack([
                                state1.flatten(), state2.flatten()
                            ]))[0, 1].item()
                        else:
                            # Use norm correlation for different sizes
                            norm1 = torch.norm(state1).item()
                            norm2 = torch.norm(state2).item()
                            correlation = min(norm1, norm2) / (max(norm1, norm2) + 1e-6)
                        
                        interactions[f"{scale1}_{scale2}"] = correlation
        
        return interactions
    
    def _assess_conservation_quality(self, state_dict: Dict[str, torch.Tensor]) -> float:
        """Assess overall conservation quality"""
        
        total_energy = 0.0
        total_residual = 0.0
        
        for scale_name, state in state_dict.items():
            if torch.is_tensor(state):
                energy = torch.norm(state).item()
                total_energy += energy
                
                # Calculate local conservation residual
                if state.dim() >= 2:
                    # Simple conservation check: sum of gradients should be small
                    grad_sum = (torch.diff(state, dim=-1).sum() + 
                               torch.diff(state, dim=-2).sum()).abs().item()
                    total_residual += grad_sum
        
        # Quality metric (lower residual = higher quality)
        quality = 1.0 / (1.0 + total_residual / (total_energy + 1e-6))
        return quality
    
    def _add_3d_flow_plot(self, fig: go.Figure, flow_data: Dict[str, Any], row: int, col: int):
        """Add 3D flow visualization"""
        
        # Create 3D scatter plot of conservation flows
        time_points = flow_data['time_points']
        flows = flow_data['conservation_flows']
        
        if flows:
            # Extract representative flow vectors
            x_flows = [f.get('quantum_state_grad_x', 0) for f in flows]
            y_flows = [f.get('geometric_state_grad_y', 0) for f in flows]
            z_flows = [f.get('fluid_state_magnitude', 0) for f in flows]
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_flows,
                    y=y_flows,
                    z=z_flows,
                    mode='markers+lines',
                    marker=dict(
                        size=8,
                        color=time_points,
                        colorscale='Viridis',
                        colorbar=dict(title="Time"),
                        opacity=0.8
                    ),
                    line=dict(width=4, color='rgba(100,100,100,0.5)'),
                    name="Flow Trajectory"
                ),
                row=row, col=col
            )
    
    def _add_energy_heatmap(self, fig: go.Figure, flow_data: Dict[str, Any], row: int, col: int):
        """Add energy distribution heatmap"""
        
        energy_dists = flow_data['energy_distributions']
        
        if energy_dists:
            # Average energy distribution
            avg_energy = np.mean(energy_dists, axis=0)
            
            scale_names = ['Quantum', 'Geometric', 'Fluid', 'Information', 'Consciousness']
            n_scales = min(len(scale_names), avg_energy.shape[0])
            
            fig.add_trace(
                go.Heatmap(
                    z=avg_energy[:n_scales, :n_scales],
                    x=scale_names[:n_scales],
                    y=scale_names[:n_scales],
                    colorscale='Viridis',
                    colorbar=dict(title="Energy Coupling"),
                    name="Energy Distribution"
                ),
                row=row, col=col
            )
    
    def _add_scale_interactions(self, fig: go.Figure, flow_data: Dict[str, Any], row: int, col: int):
        """Add scale interaction plot"""
        
        interactions = flow_data['scale_interactions']
        time_points = flow_data['time_points']
        
        if interactions:
            # Plot interaction strengths over time
            for interaction_name in interactions[0].keys():
                interaction_values = [inter.get(interaction_name, 0) for inter in interactions]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=interaction_values,
                        mode='lines',
                        name=interaction_name.replace('_', ' → '),
                        opacity=0.7
                    ),
                    row=row, col=col
                )
    
    def _add_temporal_evolution(self, fig: go.Figure, flow_data: Dict[str, Any], row: int, col: int):
        """Add temporal evolution plot"""
        
        time_points = flow_data['time_points']
        quality = flow_data['conservation_quality']
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=quality,
                mode='lines+markers',
                marker=dict(size=8, color='red'),
                line=dict(width=3, color='red'),
                name="Conservation Quality"
            ),
            row=row, col=col
        )
        
        # Add reference line for perfect conservation
        fig.add_hline(
            y=1.0, 
            line_dash="dash", 
            line_color="green",
            annotation_text="Perfect Conservation",
            row=row, col=col
        )
    
    def create_animated_flow_plot(self, states_history: List[Dict[str, torch.Tensor]], 
                                interval: int = 100) -> animation.FuncAnimation:
        """Create animated matplotlib visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PAC Conservation Flow Animation', fontsize=16)
        
        # Initialize plots
        flow_data = self._extract_flow_data(states_history)
        
        def animate(frame):
            # Clear all axes
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()
            
            # Current frame data
            if frame < len(flow_data['conservation_flows']):
                current_flows = flow_data['conservation_flows'][frame]
                current_energy = flow_data['energy_distributions'][frame]
                current_interactions = flow_data['scale_interactions'][frame]
                current_quality = flow_data['conservation_quality'][frame]
                
                # Plot 1: Flow vectors
                ax = axes[0, 0]
                flow_names = list(current_flows.keys())
                flow_values = list(current_flows.values())
                
                ax.bar(range(len(flow_names)), flow_values, alpha=0.7)
                ax.set_title(f'Conservation Flows (t={frame})')
                ax.set_xticks(range(len(flow_names)))
                ax.set_xticklabels([name.split('_')[0] for name in flow_names], rotation=45)
                
                # Plot 2: Energy distribution
                ax = axes[0, 1]
                im = ax.imshow(current_energy, cmap='viridis', aspect='auto')
                ax.set_title(f'Energy Distribution (t={frame})')
                
                # Plot 3: Scale interactions
                ax = axes[1, 0]
                if current_interactions:
                    interaction_names = list(current_interactions.keys())
                    interaction_values = list(current_interactions.values())
                    ax.bar(range(len(interaction_names)), interaction_values, alpha=0.7)
                    ax.set_title(f'Scale Interactions (t={frame})')
                    ax.set_xticks(range(len(interaction_names)))
                    ax.set_xticklabels([name.replace('_', '→') for name in interaction_names], rotation=45)
                
                # Plot 4: Conservation quality over time
                ax = axes[1, 1]
                quality_history = flow_data['conservation_quality'][:frame+1]
                time_history = list(range(len(quality_history)))
                ax.plot(time_history, quality_history, 'r-', linewidth=2)
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
                ax.set_title('Conservation Quality')
                ax.set_xlabel('Time')
                ax.set_ylabel('Quality')
                ax.legend()
                ax.set_ylim(0, 1.1)
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(states_history), 
            interval=interval, blit=False
        )
        
        plt.tight_layout()
        return anim
    
    def create_real_time_dashboard(self, port: int = 8050) -> dash.Dash:
        """Create real-time dashboard for PAC flow monitoring"""
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("PAC Conservation Flow Dashboard", style={'textAlign': 'center'}),
            
            dcc.Graph(id='flow-plot'),
            dcc.Graph(id='conservation-quality'),
            
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            ),
            
            html.Div(id='conservation-status', style={'textAlign': 'center', 'fontSize': 20})
        ])
        
        @app.callback(
            [Output('flow-plot', 'figure'),
             Output('conservation-quality', 'figure'),
             Output('conservation-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Generate sample data (in real application, this would come from the engine)
            flow_fig = self._create_sample_flow_plot()
            quality_fig = self._create_sample_quality_plot()
            status = self._get_conservation_status()
            
            return flow_fig, quality_fig, status
        
        return app
    
    def _create_sample_flow_plot(self) -> go.Figure:
        """Create sample flow plot for dashboard"""
        
        fig = go.Figure()
        
        # Sample flow data
        scales = ['Quantum', 'Geometric', 'Fluid', 'Information', 'Consciousness']
        flows = np.random.exponential(1.0, len(scales))
        
        fig.add_trace(go.Bar(
            x=scales,
            y=flows,
            marker_color='viridis',
            name='Flow Magnitude'
        ))
        
        fig.update_layout(
            title="Current PAC Flow Magnitudes",
            xaxis_title="Physics Scale",
            yaxis_title="Flow Magnitude",
            showlegend=False
        )
        
        return fig
    
    def _create_sample_quality_plot(self) -> go.Figure:
        """Create sample conservation quality plot"""
        
        fig = go.Figure()
        
        # Sample quality history
        time_points = list(range(50))
        quality = 0.95 + 0.05 * np.sin(np.array(time_points) * 0.1) + np.random.normal(0, 0.01, len(time_points))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=quality,
            mode='lines',
            line=dict(color='red', width=2),
            name='Conservation Quality'
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", 
                     annotation_text="Perfect Conservation")
        
        fig.update_layout(
            title="Conservation Quality Over Time",
            xaxis_title="Time Step",
            yaxis_title="Quality",
            yaxis=dict(range=[0.8, 1.05]),
            showlegend=False
        )
        
        return fig
    
    def _get_conservation_status(self) -> str:
        """Get current conservation status"""
        
        # Sample status (in real application, would check actual conservation)
        quality = 0.95 + np.random.normal(0, 0.02)
        
        if quality > 0.99:
            return "🟢 EXCELLENT CONSERVATION"
        elif quality > 0.95:
            return "🟡 GOOD CONSERVATION"
        elif quality > 0.90:
            return "🟠 FAIR CONSERVATION"
        else:
            return "🔴 POOR CONSERVATION"
    
    def save_flow_animation(self, states_history: List[Dict[str, torch.Tensor]], 
                          filename: str = "pac_flow_animation.gif"):
        """Save animated flow visualization"""
        
        anim = self.create_animated_flow_plot(states_history)
        anim.save(filename, writer='pillow', fps=10)
        print(f"Animation saved as {filename}")
    
    def run_dashboard(self, port: int = 8050):
        """Run the real-time dashboard"""
        
        app = self.create_real_time_dashboard(port)
        print(f"🚀 Starting PAC Flow Dashboard on port {port}")
        print(f"📊 Visit http://localhost:{port} to view the dashboard")
        app.run_server(debug=True, port=port)
