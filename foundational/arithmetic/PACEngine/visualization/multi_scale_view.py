"""
Multi-Scale View

Provides comprehensive multi-scale visualization capabilities for the PAC physics engine.
Displays quantum, geometric, fluid, information, and consciousness scales simultaneously
with interactive controls for scale selection and temporal navigation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import colorcet as cc

class MultiScaleViewer:
    """Interactive multi-scale visualization for PAC physics engine"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Scale configurations
        self.scale_configs = {
            'quantum': {
                'colormap': 'plasma',
                'title': 'Quantum PAC',
                'unit': 'ħ',
                'range': (0, 1)
            },
            'geometric': {
                'colormap': 'viridis',
                'title': 'Geometric SEC',
                'unit': 'geometric',
                'range': (0, 2*np.pi)
            },
            'fluid': {
                'colormap': 'coolwarm',
                'title': 'Fluid MED',
                'unit': 'm/s',
                'range': (-5, 5)
            },
            'information': {
                'colormap': 'hot',
                'title': 'Information Amp',
                'unit': 'bits',
                'range': (0, 20)
            },
            'consciousness': {
                'colormap': 'magma',
                'title': 'Consciousness SCBF',
                'unit': 'awareness',
                'range': (0, 1)
            }
        }
        
        # Visualization state
        self.active_scales = list(self.scale_configs.keys())
        self.current_time = 0
        self.time_range = (0, 100)
        
    def create_interactive_multiscale_view(self, states_history: List[Dict[str, torch.Tensor]]) -> go.Figure:
        """Create interactive multi-scale visualization with Plotly"""
        
        # Determine grid layout
        n_scales = len(self.active_scales)
        rows = int(np.ceil(np.sqrt(n_scales)))
        cols = int(np.ceil(n_scales / rows))
        
        # Create subplot titles
        subplot_titles = [self.scale_configs[scale]['title'] for scale in self.active_scales]
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # Add scale visualizations
        for i, scale_name in enumerate(self.active_scales):
            row = i // cols + 1
            col = i % cols + 1
            
            self._add_scale_heatmap(fig, states_history, scale_name, row, col)
        
        # Add time slider and controls
        self._add_temporal_controls(fig, len(states_history))
        
        # Update layout
        fig.update_layout(
            title="Multi-Scale PAC Physics Visualization",
            height=800,
            showlegend=False,
            font=dict(size=10)
        )
        
        return fig
    
    def _add_scale_heatmap(self, fig: go.Figure, states_history: List[Dict[str, torch.Tensor]], 
                          scale_name: str, row: int, col: int):
        """Add heatmap for specific scale"""
        
        config = self.scale_configs[scale_name]
        
        # Extract state data for this scale
        scale_data = []
        for state_dict in states_history:
            if f"{scale_name}_state" in state_dict:
                state = state_dict[f"{scale_name}_state"]
                if torch.is_tensor(state):
                    scale_data.append(state.cpu().numpy())
                else:
                    # Create placeholder data
                    scale_data.append(np.random.randn(32, 32))
            else:
                # Create placeholder data
                scale_data.append(np.random.randn(32, 32))
        
        if scale_data:
            # Use first timestep for static display (can be made interactive)
            current_data = scale_data[0]
            
            # Ensure 2D data
            if current_data.ndim > 2:
                current_data = current_data[0] if current_data.shape[0] == 1 else np.mean(current_data, axis=0)
            elif current_data.ndim == 1:
                size = int(np.sqrt(len(current_data)))
                current_data = current_data[:size*size].reshape(size, size)
            
            fig.add_trace(
                go.Heatmap(
                    z=current_data,
                    colorscale=config['colormap'],
                    zmin=config['range'][0],
                    zmax=config['range'][1],
                    colorbar=dict(
                        title=config['unit'],
                        titleside='right',
                        x=1.02 + (col-1) * 0.1,
                        len=0.3,
                        y=0.8 - (row-1) * 0.4
                    ),
                    hovertemplate=f"<b>{config['title']}</b><br>" +
                                f"x: %{{x}}<br>y: %{{y}}<br>value: %{{z:.3f}} {config['unit']}<extra></extra>"
                ),
                row=row, col=col
            )
    
    def _add_temporal_controls(self, fig: go.Figure, n_timesteps: int):
        """Add temporal navigation controls"""
        
        # Add time slider using updatemenus
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="Play",
                            method="animate"
                        ),
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="Pause",
                            method="animate"
                        )
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=True,
                    x=0.011,
                    xanchor="left",
                    y=0,
                    yanchor="top"
                ),
            ],
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Time Step: "},
                pad={"t": 50},
                steps=[dict(
                    label=str(i),
                    method="restyle",
                    args=[{"z": [[]]}]  # Would update data in interactive version
                ) for i in range(n_timesteps)]
            )]
        )
    
    def create_matplotlib_multiscale_view(self, states_history: List[Dict[str, torch.Tensor]], 
                                        interactive: bool = True) -> plt.Figure:
        """Create matplotlib-based multi-scale view with optional interactivity"""
        
        # Create figure with custom grid
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main visualization area (2x2 grid for scales)
        scale_axes = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for i, scale_name in enumerate(self.active_scales[:4]):  # Show first 4 scales
            if i < len(positions):
                row, col = positions[i]
                ax = fig.add_subplot(gs[row, col])
                scale_axes[scale_name] = ax
        
        # Control panel area
        control_ax = fig.add_subplot(gs[2, :])
        control_ax.axis('off')
        
        # Time slider
        if interactive:
            time_slider_ax = fig.add_subplot(gs[2, 0:2])
            time_slider = Slider(
                time_slider_ax, 'Time', 0, len(states_history)-1, 
                valinit=0, valfmt='%d'
            )
        
        # Scale selection checkboxes
        if interactive:
            checkbox_ax = fig.add_subplot(gs[2, 2])
            scale_checkbox = CheckButtons(
                checkbox_ax, 
                list(self.scale_configs.keys()),
                [True] * len(self.scale_configs)
            )
        
        # Initial visualization
        self._update_multiscale_plot(scale_axes, states_history, 0)
        
        # Interactive callbacks
        if interactive:
            def update_time(val):
                time_idx = int(time_slider.val)
                self._update_multiscale_plot(scale_axes, states_history, time_idx)
                fig.canvas.draw()
            
            def update_scales(label):
                if label in self.active_scales:
                    self.active_scales.remove(label)
                else:
                    self.active_scales.append(label)
                self._update_multiscale_plot(scale_axes, states_history, int(time_slider.val))
                fig.canvas.draw()
            
            time_slider.on_changed(update_time)
            scale_checkbox.on_clicked(update_scales)
        
        fig.suptitle('Multi-Scale PAC Physics Visualization', fontsize=16)
        return fig
    
    def _update_multiscale_plot(self, axes: Dict[str, plt.Axes], 
                              states_history: List[Dict[str, torch.Tensor]], 
                              time_idx: int):
        """Update multi-scale plot for given time index"""
        
        if time_idx >= len(states_history):
            return
        
        current_states = states_history[time_idx]
        
        for scale_name, ax in axes.items():
            if scale_name in self.active_scales:
                ax.clear()
                
                config = self.scale_configs[scale_name]
                
                # Get state data
                state_key = f"{scale_name}_state"
                if state_key in current_states:
                    state = current_states[state_key]
                    if torch.is_tensor(state):
                        data = state.cpu().numpy()
                    else:
                        data = np.random.randn(32, 32)  # Placeholder
                else:
                    data = np.random.randn(32, 32)  # Placeholder
                
                # Ensure 2D data
                if data.ndim > 2:
                    data = data[0] if data.shape[0] == 1 else np.mean(data, axis=0)
                elif data.ndim == 1:
                    size = int(np.sqrt(len(data)))
                    data = data[:size*size].reshape(size, size)
                
                # Plot heatmap
                im = ax.imshow(
                    data, 
                    cmap=config['colormap'],
                    vmin=config['range'][0],
                    vmax=config['range'][1],
                    aspect='auto'
                )
                
                ax.set_title(f"{config['title']} (t={time_idx})")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label=config['unit'])
            else:
                ax.clear()
                ax.text(0.5, 0.5, f'{scale_name}\nDisabled', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, alpha=0.5)
    
    def create_3d_multiscale_view(self, states_history: List[Dict[str, torch.Tensor]]) -> go.Figure:
        """Create 3D multi-scale visualization"""
        
        fig = go.Figure()
        
        # Create 3D representation where each scale is a layer in Z dimension
        z_positions = {scale: i for i, scale in enumerate(self.active_scales)}
        
        for time_idx, state_dict in enumerate(states_history[::5]):  # Sample every 5th timestep
            for scale_name in self.active_scales:
                config = self.scale_configs[scale_name]
                
                # Get state data
                state_key = f"{scale_name}_state"
                if state_key in state_dict:
                    state = state_dict[state_key]
                    if torch.is_tensor(state):
                        data = state.cpu().numpy()
                    else:
                        continue
                else:
                    continue
                
                # Process data for 3D visualization
                if data.ndim >= 2:
                    # Sample points from the 2D data
                    h, w = data.shape[-2:]
                    x_coords, y_coords = np.meshgrid(
                        np.linspace(0, 1, min(h, 20)),
                        np.linspace(0, 1, min(w, 20))
                    )
                    
                    # Downsample data
                    data_sampled = data[::h//20, ::w//20] if h > 20 and w > 20 else data
                    
                    # Flatten for scatter plot
                    x_flat = x_coords.flatten()
                    y_flat = y_coords.flatten()
                    z_flat = np.full_like(x_flat, z_positions[scale_name])
                    values_flat = data_sampled.flatten()
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_flat,
                        y=y_flat,
                        z=z_flat,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=values_flat,
                            colorscale=config['colormap'],
                            opacity=0.6
                        ),
                        name=f"{config['title']} (t={time_idx*5})",
                        showlegend=False
                    ))
        
        # Update layout for 3D
        fig.update_layout(
            title="3D Multi-Scale PAC Visualization",
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                zaxis_title="Physics Scale",
                zaxis=dict(
                    ticktext=[config['title'] for config in self.scale_configs.values()],
                    tickvals=list(z_positions.values())
                )
            ),
            height=800
        )
        
        return fig
    
    def create_comparative_analysis(self, states_history: List[Dict[str, torch.Tensor]]) -> go.Figure:
        """Create comparative analysis across scales"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scale Magnitudes', 'Cross-Scale Correlations', 
                          'Conservation Quality', 'Emergence Indicators'),
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Extract analysis data
        time_points = list(range(len(states_history)))
        scale_magnitudes = {scale: [] for scale in self.active_scales}
        conservation_quality = []
        emergence_indicators = []
        
        for state_dict in states_history:
            # Calculate scale magnitudes
            for scale_name in self.active_scales:
                state_key = f"{scale_name}_state"
                if state_key in state_dict:
                    state = state_dict[state_key]
                    if torch.is_tensor(state):
                        magnitude = torch.norm(state).item()
                    else:
                        magnitude = 0
                else:
                    magnitude = 0
                scale_magnitudes[scale_name].append(magnitude)
            
            # Calculate conservation quality
            total_energy = sum(scale_magnitudes[scale][-1] for scale in self.active_scales)
            # Simple conservation metric (in real implementation, use proper PAC conservation)
            quality = 1.0 / (1.0 + abs(total_energy - 10.0) / 10.0)  # Assume target energy ~10
            conservation_quality.append(quality)
            
            # Calculate emergence indicators
            emergence = np.random.exponential(0.1)  # Placeholder
            emergence_indicators.append(emergence)
        
        # 1. Scale magnitudes over time
        for scale_name in self.active_scales:
            config = self.scale_configs[scale_name]
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=scale_magnitudes[scale_name],
                    mode='lines',
                    name=config['title'],
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Cross-scale correlations
        correlation_matrix = np.corrcoef([scale_magnitudes[scale] for scale in self.active_scales])
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=list(self.active_scales),
                y=list(self.active_scales),
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation", x=0.48)
            ),
            row=1, col=2
        )
        
        # 3. Conservation quality
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=conservation_quality,
                mode='lines',
                name='Conservation Quality',
                line=dict(color='red', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", 
                     annotation_text="Perfect", row=2, col=1)
        
        # 4. Current emergence indicators
        current_emergence = {scale: emergence_indicators[-1] for scale in self.active_scales}
        fig.add_trace(
            go.Bar(
                x=list(current_emergence.keys()),
                y=list(current_emergence.values()),
                marker_color='orange',
                name='Emergence'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Multi-Scale Comparative Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Scale", row=2, col=2)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_yaxes(title_text="Quality", row=2, col=1)
        fig.update_yaxes(title_text="Emergence", row=2, col=2)
        
        return fig
    
    def export_multiscale_data(self, states_history: List[Dict[str, torch.Tensor]], 
                             filename: str = "multiscale_data.npz"):
        """Export multi-scale data for external analysis"""
        
        export_data = {}
        
        for time_idx, state_dict in enumerate(states_history):
            for scale_name in self.active_scales:
                state_key = f"{scale_name}_state"
                if state_key in state_dict:
                    state = state_dict[state_key]
                    if torch.is_tensor(state):
                        export_key = f"{scale_name}_t{time_idx:04d}"
                        export_data[export_key] = state.cpu().numpy()
        
        # Save metadata
        export_data['metadata'] = {
            'scales': self.active_scales,
            'time_steps': len(states_history),
            'scale_configs': self.scale_configs
        }
        
        np.savez_compressed(filename, **export_data)
        print(f"Multi-scale data exported to {filename}")
    
    def set_active_scales(self, scales: List[str]):
        """Set which scales to display"""
        self.active_scales = [scale for scale in scales if scale in self.scale_configs]
    
    def get_scale_summary(self, states_history: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Get summary statistics for all scales"""
        
        summary = {}
        
        for scale_name in self.scale_configs.keys():
            scale_data = []
            
            for state_dict in states_history:
                state_key = f"{scale_name}_state"
                if state_key in state_dict:
                    state = state_dict[state_key]
                    if torch.is_tensor(state):
                        scale_data.append(state.cpu().numpy())
            
            if scale_data:
                # Calculate statistics
                magnitudes = [np.linalg.norm(data) for data in scale_data]
                
                summary[scale_name] = {
                    'mean_magnitude': np.mean(magnitudes),
                    'std_magnitude': np.std(magnitudes),
                    'max_magnitude': np.max(magnitudes),
                    'min_magnitude': np.min(magnitudes),
                    'data_shape': scale_data[0].shape if scale_data else None,
                    'time_steps': len(scale_data)
                }
            else:
                summary[scale_name] = {
                    'mean_magnitude': 0,
                    'std_magnitude': 0,
                    'max_magnitude': 0,
                    'min_magnitude': 0,
                    'data_shape': None,
                    'time_steps': 0
                }
        
        return summary
