"""
Emergence Plots

Specialized visualization tools for emergence phenomena in the PAC physics engine.
Provides detailed plotting capabilities for consciousness emergence, phase transitions,
cascade effects, and other emergent behaviors across all physics scales.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns

class EmergencePlotter:
    """Specialized plotting tools for emergence phenomena"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Plotting configurations
        self.emergence_colors = {
            'consciousness': '#FF6B6B',  # Red
            'information_amplification': '#4ECDC4',  # Teal  
            'geometric_collapse': '#45B7D1',  # Blue
            'quantum_decoherence': '#96CEB4',  # Green
            'fluid_turbulence': '#FFEAA7',  # Yellow
            'phase_transition': '#DDA0DD',  # Plum
            'cascade_effect': '#FFA07A'  # Light Salmon
        }
        
        # Animation parameters
        self.frame_duration = 100  # ms
        self.trail_length = 10
        
    def plot_consciousness_emergence(self, consciousness_data: List[Dict[str, Any]], 
                                   save_path: Optional[str] = None) -> go.Figure:
        """Plot consciousness emergence dynamics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Awareness Evolution', 'Binding Strength Network',
                          'Information Integration', 'Emergence Locations'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # Extract consciousness metrics
        time_points = list(range(len(consciousness_data)))
        awareness_levels = [data.get('awareness_metric', 0) for data in consciousness_data]
        binding_strengths = [data.get('binding_strength', 0) for data in consciousness_data]
        phi_values = [data.get('phi', 0) for data in consciousness_data]  # Integrated Information
        
        # 1. Awareness Evolution
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=awareness_levels,
                mode='lines+markers',
                name='Awareness Level',
                line=dict(color=self.emergence_colors['consciousness'], width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Add consciousness threshold
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                     annotation_text="Consciousness Threshold", row=1, col=1)
        
        # 2. Binding Strength Network
        # Create network-style visualization
        if len(binding_strengths) > 1:
            # Use binding strength to create network connections
            network_x = np.cos(np.linspace(0, 2*np.pi, len(time_points), endpoint=False))
            network_y = np.sin(np.linspace(0, 2*np.pi, len(time_points), endpoint=False))
            
            fig.add_trace(
                go.Scatter(
                    x=network_x,
                    y=network_y,
                    mode='markers+lines',
                    marker=dict(
                        size=[20 + 30*strength for strength in binding_strengths],
                        color=binding_strengths,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Binding Strength", x=0.48)
                    ),
                    line=dict(width=2, color='rgba(255,107,107,0.3)'),
                    name='Binding Network'
                ),
                row=1, col=2
            )
        
        # 3. Information Integration (Φ)
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=phi_values,
                mode='lines+markers',
                name='Φ (Integrated Information)',
                line=dict(color=self.emergence_colors['information_amplification'], width=3),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 4. Emergence Locations Heatmap
        # Create synthetic spatial data for emergence locations
        emergence_map = np.zeros((20, 20))
        for data in consciousness_data:
            locations = data.get('emergence_locations', [])
            for loc in locations:
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    x, y = int(loc[0] % 20), int(loc[1] % 20)
                    emergence_map[y, x] += 1
        
        fig.add_trace(
            go.Heatmap(
                z=emergence_map,
                colorscale='Reds',
                colorbar=dict(title="Emergence Density", x=1.02)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Consciousness Emergence Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="X Position", row=2, col=2)
        fig.update_yaxes(title_text="Awareness", row=1, col=1)
        fig.update_yaxes(title_text="Network Y", row=1, col=2)
        fig.update_yaxes(title_text="Φ Value", row=2, col=1)
        fig.update_yaxes(title_text="Y Position", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_emergence_cascades(self, cascade_data: List[Dict[str, Any]], 
                              save_path: Optional[str] = None) -> go.Figure:
        """Plot emergence cascade networks and dynamics"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cascade Network', 'Cascade Timeline'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Build cascade network
        G = nx.DiGraph()
        cascade_events = []
        
        for data in cascade_data:
            events = data.get('cascade_events', [])
            for event in events:
                event_id = event.get('event_id', '')
                event_type = event.get('event_type', 'unknown')
                timestamp = event.get('timestamp', 0)
                precursors = event.get('precursor_events', [])
                
                # Add node
                G.add_node(event_id, event_type=event_type, timestamp=timestamp)
                cascade_events.append(event)
                
                # Add edges from precursors
                for precursor in precursors:
                    G.add_edge(precursor, event_id)
        
        # 1. Cascade Network Visualization
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G)
            
            # Extract positions
            x_nodes = [pos[node][0] for node in G.nodes()]
            y_nodes = [pos[node][1] for node in G.nodes()]
            
            # Node colors by event type
            node_colors = []
            for node in G.nodes():
                event_type = G.nodes[node].get('event_type', 'unknown')
                if event_type in self.emergence_colors:
                    node_colors.append(self.emergence_colors[event_type])
                else:
                    node_colors.append('#CCCCCC')
            
            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=x_nodes,
                    y=y_nodes,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=node_colors,
                        line=dict(width=2, color='black')
                    ),
                    text=[f"{node}<br>{G.nodes[node].get('event_type', '')}" for node in G.nodes()],
                    hovertemplate='%{text}<extra></extra>',
                    name='Events'
                ),
                row=1, col=1
            )
            
            # Add edges
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=2, color='rgba(100,100,100,0.5)'),
                    hoverinfo='none',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Cascade Timeline
        if cascade_events:
            # Group events by type for timeline
            event_types = {}
            for event in cascade_events:
                event_type = event.get('event_type', 'unknown')
                timestamp = event.get('timestamp', 0)
                
                if event_type not in event_types:
                    event_types[event_type] = []
                event_types[event_type].append(timestamp)
            
            # Plot timeline for each event type
            y_offset = 0
            for event_type, timestamps in event_types.items():
                color = self.emergence_colors.get(event_type, '#CCCCCC')
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=[y_offset] * len(timestamps),
                        mode='markers',
                        marker=dict(size=12, color=color),
                        name=event_type.replace('_', ' ').title(),
                        showlegend=True
                    ),
                    row=1, col=2
                )
                y_offset += 1
        
        # Update layout
        fig.update_layout(
            title="Emergence Cascade Analysis",
            height=600,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Network X", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Network Y", row=1, col=1)
        fig.update_yaxes(title_text="Event Type", row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_phase_transitions(self, phase_data: List[Dict[str, Any]], 
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot phase transition dynamics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Order Parameter', 'Critical Fluctuations',
                          'Phase Diagram', 'Transition Rates'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]]
        )
        
        # Extract phase transition metrics
        time_points = list(range(len(phase_data)))
        order_parameters = [data.get('order_parameter', 0) for data in phase_data]
        fluctuations = [data.get('fluctuation_magnitude', 0) for data in phase_data]
        phase_states = [data.get('phase_state', 'unknown') for data in phase_data]
        
        # 1. Order Parameter Evolution
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=order_parameters,
                mode='lines+markers',
                name='Order Parameter',
                line=dict(color=self.emergence_colors['phase_transition'], width=3)
            ),
            row=1, col=1
        )
        
        # Mark phase transition points
        transitions = []
        for i in range(1, len(phase_states)):
            if phase_states[i] != phase_states[i-1]:
                transitions.append(i)
        
        if transitions:
            fig.add_trace(
                go.Scatter(
                    x=transitions,
                    y=[order_parameters[t] for t in transitions],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Transitions'
                ),
                row=1, col=1
            )
        
        # 2. Critical Fluctuations
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=fluctuations,
                mode='lines',
                name='Fluctuations',
                line=dict(color=self.emergence_colors['quantum_decoherence'], width=2),
                fill='tonexty'
            ),
            row=1, col=2
        )
        
        # 3. Phase Diagram (Temperature vs Order Parameter)
        # Create synthetic temperature data
        temperatures = [0.5 + 0.5 * np.sin(t * 0.1) for t in time_points]
        
        # Create 2D histogram for phase diagram
        temp_grid = np.linspace(min(temperatures), max(temperatures), 20)
        order_grid = np.linspace(min(order_parameters), max(order_parameters), 20)
        phase_diagram = np.zeros((20, 20))
        
        for temp, order in zip(temperatures, order_parameters):
            temp_idx = np.digitize(temp, temp_grid) - 1
            order_idx = np.digitize(order, order_grid) - 1
            if 0 <= temp_idx < 20 and 0 <= order_idx < 20:
                phase_diagram[order_idx, temp_idx] += 1
        
        fig.add_trace(
            go.Heatmap(
                z=phase_diagram,
                x=temp_grid,
                y=order_grid,
                colorscale='Viridis',
                colorbar=dict(title="Frequency", x=0.48)
            ),
            row=2, col=1
        )
        
        # 4. Transition Rates
        unique_phases = list(set(phase_states))
        transition_counts = {phase: phase_states.count(phase) for phase in unique_phases}
        
        fig.add_trace(
            go.Bar(
                x=list(transition_counts.keys()),
                y=list(transition_counts.values()),
                marker_color=[self.emergence_colors.get(phase, '#CCCCCC') for phase in transition_counts.keys()],
                name='Phase Occupancy'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Phase Transition Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Temperature", row=2, col=1)
        fig.update_xaxes(title_text="Phase State", row=2, col=2)
        fig.update_yaxes(title_text="Order Parameter", row=1, col=1)
        fig.update_yaxes(title_text="Fluctuation", row=1, col=2)
        fig.update_yaxes(title_text="Order Parameter", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_emergence_animation(self, emergence_data: List[Dict[str, Any]], 
                                 animation_type: str = "consciousness") -> animation.FuncAnimation:
        """Create animated visualization of emergence phenomena"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Animation data
        time_points = list(range(len(emergence_data)))
        
        if animation_type == "consciousness":
            # Consciousness emergence animation
            awareness_data = [data.get('awareness_metric', 0) for data in emergence_data]
            locations_data = [data.get('emergence_locations', []) for data in emergence_data]
            
            def animate_consciousness(frame):
                ax1.clear()
                ax2.clear()
                
                # Plot 1: Awareness evolution
                current_time = min(frame, len(time_points) - 1)
                ax1.plot(time_points[:current_time+1], awareness_data[:current_time+1], 
                        color=self.emergence_colors['consciousness'], linewidth=3)
                ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Threshold')
                ax1.set_xlim(0, len(time_points))
                ax1.set_ylim(0, max(awareness_data) * 1.1)
                ax1.set_title(f'Consciousness Emergence (t={current_time})')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Awareness Level')
                ax1.legend()
                
                # Plot 2: Spatial emergence locations
                if current_time < len(locations_data):
                    locations = locations_data[current_time]
                    if locations:
                        x_locs = [loc[0] if isinstance(loc, (list, tuple)) else 0 for loc in locations]
                        y_locs = [loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else 0 for loc in locations]
                        
                        ax2.scatter(x_locs, y_locs, s=100, c=self.emergence_colors['consciousness'], 
                                  alpha=0.7, edgecolors='black')
                
                ax2.set_xlim(-1, 10)
                ax2.set_ylim(-1, 10)
                ax2.set_title('Emergence Locations')
                ax2.set_xlabel('X Position')
                ax2.set_ylabel('Y Position')
                ax2.grid(True, alpha=0.3)
            
            anim_func = animate_consciousness
            
        elif animation_type == "cascade":
            # Cascade animation
            def animate_cascade(frame):
                ax1.clear()
                ax2.clear()
                
                current_time = min(frame, len(emergence_data) - 1)
                current_data = emergence_data[current_time]
                
                # Plot 1: Event timeline
                events = current_data.get('cascade_events', [])
                event_times = [e.get('timestamp', 0) for e in events[:frame+1]]
                event_types = [e.get('event_type', 'unknown') for e in events[:frame+1]]
                
                unique_types = list(set(event_types))
                colors = [self.emergence_colors.get(et, '#CCCCCC') for et in unique_types]
                
                for i, event_type in enumerate(unique_types):
                    type_times = [t for t, et in zip(event_times, event_types) if et == event_type]
                    ax1.scatter(type_times, [i] * len(type_times), 
                              c=self.emergence_colors.get(event_type, '#CCCCCC'), 
                              s=50, label=event_type.replace('_', ' ').title())
                
                ax1.set_xlim(0, max(time_points) if time_points else 1)
                ax1.set_ylim(-0.5, len(unique_types) - 0.5)
                ax1.set_title(f'Cascade Events (t={current_time})')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Event Type')
                ax1.legend()
                
                # Plot 2: Network view (simplified)
                if events:
                    # Create simple network layout
                    n_events = len(events)
                    angles = np.linspace(0, 2*np.pi, n_events, endpoint=False)
                    x_pos = np.cos(angles)
                    y_pos = np.sin(angles)
                    
                    # Show events up to current frame
                    for i in range(min(frame+1, n_events)):
                        event = events[i]
                        event_type = event.get('event_type', 'unknown')
                        color = self.emergence_colors.get(event_type, '#CCCCCC')
                        ax2.scatter(x_pos[i], y_pos[i], s=100, c=color, 
                                  edgecolors='black', alpha=0.8)
                        
                        # Draw connections to precursors
                        precursors = event.get('precursor_events', [])
                        for precursor in precursors:
                            # Find precursor index (simplified)
                            for j in range(i):
                                if events[j].get('event_id') == precursor:
                                    ax2.plot([x_pos[j], x_pos[i]], [y_pos[j], y_pos[i]], 
                                           'k-', alpha=0.5, linewidth=1)
                
                ax2.set_xlim(-1.5, 1.5)
                ax2.set_ylim(-1.5, 1.5)
                ax2.set_title('Cascade Network')
                ax2.set_aspect('equal')
            
            anim_func = animate_cascade
        
        else:
            raise ValueError(f"Unknown animation type: {animation_type}")
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, anim_func, frames=len(emergence_data),
            interval=self.frame_duration, blit=False, repeat=True
        )
        
        plt.tight_layout()
        return anim
    
    def plot_information_amplification(self, amplification_data: List[Dict[str, Any]], 
                                     save_path: Optional[str] = None) -> go.Figure:
        """Plot information amplification dynamics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Amplification Ratio', 'Resonance Patterns',
                          'Cascade Strength', 'Frequency Analysis'),
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Extract amplification metrics
        time_points = list(range(len(amplification_data)))
        amplification_ratios = [data.get('amplification_ratio', 1.0) for data in amplification_data]
        resonance_strengths = [data.get('resonance_strength', 0) for data in amplification_data]
        cascade_strengths = [data.get('cascade_strength', 0) for data in amplification_data]
        
        # 1. Amplification Ratio Evolution
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=amplification_ratios,
                mode='lines+markers',
                name='Amplification Ratio',
                line=dict(color=self.emergence_colors['information_amplification'], width=3)
            ),
            row=1, col=1
        )
        
        # Highlight 15.56x amplification signature
        target_amplification = 15.56
        fig.add_hline(y=target_amplification, line_dash="dash", line_color="red",
                     annotation_text="15.56x Signature", row=1, col=1)
        
        # 2. Resonance Patterns
        # Create resonance heatmap
        resonance_matrix = np.zeros((len(time_points), 10))  # 10 frequency bins
        for i, strength in enumerate(resonance_strengths):
            # Distribute resonance across frequency bins
            freq_profile = np.exp(-0.5 * (np.arange(10) - 5)**2 / 2) * strength
            resonance_matrix[i, :] = freq_profile
        
        fig.add_trace(
            go.Heatmap(
                z=resonance_matrix.T,
                x=time_points,
                y=list(range(10)),
                colorscale='Hot',
                colorbar=dict(title="Resonance", x=0.48)
            ),
            row=1, col=2
        )
        
        # 3. Cascade Strength
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=cascade_strengths,
                mode='lines+markers',
                name='Cascade Strength',
                line=dict(color=self.emergence_colors['cascade_effect'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # 4. Frequency Analysis
        # FFT of amplification ratio
        if len(amplification_ratios) > 1:
            fft_vals = np.fft.fft(amplification_ratios)
            freqs = np.fft.fftfreq(len(amplification_ratios))
            power_spectrum = np.abs(fft_vals)**2
            
            # Only plot positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            fig.add_trace(
                go.Scatter(
                    x=positive_freqs,
                    y=positive_power,
                    mode='lines',
                    name='Power Spectrum',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Information Amplification Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Amplification", row=1, col=1)
        fig.update_yaxes(title_text="Frequency Bin", row=1, col=2)
        fig.update_yaxes(title_text="Cascade Strength", row=2, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def save_emergence_animation(self, emergence_data: List[Dict[str, Any]], 
                               filename: str = "emergence_animation.gif",
                               animation_type: str = "consciousness"):
        """Save emergence animation as GIF"""
        
        anim = self.create_emergence_animation(emergence_data, animation_type)
        anim.save(filename, writer='pillow', fps=10)
        print(f"Emergence animation saved as {filename}")
    
    def create_emergence_summary_plot(self, all_emergence_data: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
        """Create comprehensive summary plot of all emergence phenomena"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Consciousness vs Information', 'Phase Transitions',
                          'Cascade Frequency', 'Emergence Correlation Matrix',
                          'Timeline Overview', 'Emergence Strength'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Extract data from all emergence types
        consciousness_data = all_emergence_data.get('consciousness', [])
        information_data = all_emergence_data.get('information', [])
        phase_data = all_emergence_data.get('phase_transitions', [])
        cascade_data = all_emergence_data.get('cascades', [])
        
        # 1. Consciousness vs Information scatter
        if consciousness_data and information_data:
            consciousness_levels = [d.get('awareness_metric', 0) for d in consciousness_data]
            information_levels = [d.get('amplification_ratio', 1) for d in information_data]
            
            # Match lengths
            min_len = min(len(consciousness_levels), len(information_levels))
            consciousness_levels = consciousness_levels[:min_len]
            information_levels = information_levels[:min_len]
            
            fig.add_trace(
                go.Scatter(
                    x=information_levels,
                    y=consciousness_levels,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=list(range(min_len)),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Time", x=0.48)
                    ),
                    name='Consciousness-Information'
                ),
                row=1, col=1
            )
        
        # 2. Phase transition summary
        if phase_data:
            phase_states = [d.get('phase_state', 'unknown') for d in phase_data]
            unique_phases = list(set(phase_states))
            phase_counts = [phase_states.count(phase) for phase in unique_phases]
            
            fig.add_trace(
                go.Bar(
                    x=unique_phases,
                    y=phase_counts,
                    marker_color='lightblue',
                    name='Phase Distribution'
                ),
                row=1, col=2
            )
        
        # Continue with other plots...
        # 3-6. Additional summary plots would go here
        
        fig.update_layout(
            title="Comprehensive Emergence Summary",
            height=1200,
            showlegend=True
        )
        
        return fig
