"""
SCBF Core Integration Module
============================

Foundational SCBF debugging and metrics layer for GAIA runtime.
Provides hooks, trackers, and dashboard generation capabilities.

Architecture:
- SCBFTracker: Core debugging hooks and metrics collection
- SCBFDashboard: Visualization and reporting system  
- SCBFMetrics: Fallback implementations when external SCBF unavailable
- Integration with any runtime through standardized hooks

Based on TinyCIMM SCBF integration patterns.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

# Try to import full SCBF framework
try:
    from scbf.metrics.entropy_collapse import compute_symbolic_entropy_collapse
    from scbf.metrics.activation_ancestry import compute_activation_ancestry
    from scbf.metrics.bifractal_lineage import compute_bifractal_lineage
    from scbf.metrics.semantic_attractors import compute_semantic_attractor_density
    from scbf.visualization.scbf_plots import plot_complete_scbf_dashboard
    SCBF_AVAILABLE = True
    print("✓ Full SCBF framework available")
except ImportError:
    SCBF_AVAILABLE = False
    print("⚠️ SCBF framework not available, using fallback implementations")


class SCBFMetrics:
    """
    Enhanced SCBF Metrics computation with meaningful neural dynamics analysis.
    Provides deep insights into field dynamics, pattern formation, and cognitive processes.
    """
    
    @staticmethod
    def compute_symbolic_entropy_collapse(data: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute meaningful symbolic entropy collapse metrics"""
        if SCBF_AVAILABLE:
            return compute_symbolic_entropy_collapse(data, **kwargs)
        else:
            # Enhanced fallback with actual information theory analysis
            data_flat = data.flatten()
            if len(data_flat) == 0:
                return {'entropy_initial': 0.0, 'entropy_final': 0.0, 'collapse_magnitude': 0.0}
            
            # Compute actual Shannon entropy
            data_norm = torch.abs(data_flat) + 1e-8  # Avoid log(0)
            data_prob = data_norm / torch.sum(data_norm)
            shannon_entropy = -torch.sum(data_prob * torch.log2(data_prob + 1e-8))
            
            # Compute Von Neumann entropy approximation for quantum-like analysis
            if len(data.shape) > 1 and data.shape[-1] > 1 and data.numel() > 4:
                try:
                    # Safer correlation matrix computation with robust eigenvalue handling
                    data_reshaped = data.view(-1, data.shape[-1])
                    if data_reshaped.shape[0] > 1 and data_reshaped.shape[1] > 1:
                        # Normalize data to prevent extreme values
                        data_norm = torch.nn.functional.normalize(data_reshaped, p=2, dim=1)
                        
                        # Use SVD instead of eigendecomposition for better numerical stability
                        U, S, Vt = torch.linalg.svd(data_norm, full_matrices=False)
                        
                        # Convert singular values to probability distribution
                        eigenvals = S * S  # Squared singular values approximate eigenvalues
                        eigenvals = eigenvals + 1e-12  # Prevent zeros
                        eigenvals = eigenvals / torch.sum(eigenvals)  # Normalize
                        
                        # Filter out near-zero eigenvalues to prevent log issues
                        eigenvals = eigenvals[eigenvals > 1e-10]
                        
                        if len(eigenvals) > 0:
                            von_neumann_entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-12))
                        else:
                            von_neumann_entropy = shannon_entropy
                    else:
                        von_neumann_entropy = shannon_entropy
                except Exception as e:
                    # Silently use fallback to reduce console noise
                    von_neumann_entropy = shannon_entropy  # Safe fallback
            else:
                von_neumann_entropy = shannon_entropy  # For 1D or small data
            
            # Information integration measure (inspired by IIT)
            mutual_info = shannon_entropy - von_neumann_entropy
            
            # Collapse magnitude based on entropy reduction and coherence
            coherence = 1.0 - (shannon_entropy / torch.log2(torch.tensor(len(data_flat), dtype=torch.float32)))
            collapse_magnitude = float(coherence * torch.abs(mutual_info))
            
            return {
                'entropy_initial': float(shannon_entropy),
                'entropy_final': float(von_neumann_entropy),
                'collapse_magnitude': collapse_magnitude,
                'mutual_information': float(mutual_info),
                'coherence_factor': float(coherence),
                'information_integration': float(torch.abs(mutual_info)),
                'dimensionality_reduction': float(shannon_entropy - von_neumann_entropy)
            }
    
    @staticmethod
    def compute_activation_ancestry(data: torch.Tensor, prev_data: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, float]:
        """Compute meaningful activation ancestry and lineage metrics"""
        if SCBF_AVAILABLE and prev_data is not None:
            return compute_activation_ancestry(data, prev_data, **kwargs)
        else:
            if prev_data is not None:
                # Cosine similarity for pattern preservation
                if len(data.flatten()) == len(prev_data.flatten()) and len(data.flatten()) > 0:
                    similarity = float(torch.cosine_similarity(data.flatten(), prev_data.flatten(), dim=0))
                else:
                    similarity = 0.0
                
                # Pattern drift analysis
                if data.shape == prev_data.shape:
                    l2_drift = float(torch.norm(data - prev_data))
                    relative_drift = l2_drift / (torch.norm(prev_data) + 1e-8)
                else:
                    relative_drift = 1.0  # Maximum drift for shape mismatch
                
                # Information flow analysis
                data_entropy = -torch.sum(torch.softmax(data.flatten(), dim=0) * torch.log_softmax(data.flatten(), dim=0))
                prev_entropy = -torch.sum(torch.softmax(prev_data.flatten(), dim=0) * torch.log_softmax(prev_data.flatten(), dim=0))
                entropy_flow = float(data_entropy - prev_entropy)
                
                # Structural coherence
                structural_coherence = 1.0 / (1.0 + relative_drift)
                
                # Memory persistence (how much of previous pattern is retained)
                memory_persistence = max(0.0, similarity)
                
                # Innovation factor (how much new information is introduced)
                innovation_factor = float(torch.abs(torch.tensor(entropy_flow)))
                
                return {
                    'ancestry_strength': abs(similarity),
                    'lineage_stability': structural_coherence,
                    'pattern_drift': float(relative_drift),
                    'information_flow': entropy_flow,
                    'memory_persistence': memory_persistence,
                    'innovation_factor': innovation_factor,
                    'structural_coherence': structural_coherence
                }
            else:
                return {
                    'ancestry_strength': 0.0,
                    'lineage_stability': 1.0,
                    'pattern_drift': 0.0,
                    'information_flow': 0.0,
                    'memory_persistence': 0.0,
                    'innovation_factor': 0.0,
                    'structural_coherence': 1.0
                }
    
    @staticmethod
    def compute_bifractal_lineage(data: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute meaningful bifractal and complexity metrics"""
        if SCBF_AVAILABLE:
            return compute_bifractal_lineage(data, **kwargs)
        else:
            data_flat = data.flatten()
            if len(data_flat) < 4:
                return {'fractal_dimension': 1.0, 'bifractal_strength': 0.0}
            
            # Box-counting fractal dimension approximation
            data_abs = torch.abs(data_flat)
            sorted_data = torch.sort(data_abs)[0]
            
            # Compute Higuchi fractal dimension
            def higuchi_fd(signal, k_max=10):
                N = len(signal)
                L = []
                for k in range(1, min(k_max, N//2)):
                    Lk = 0
                    for m in range(k):
                        idxs = torch.arange(m, N, k)
                        if len(idxs) > 1:
                            diffs = torch.abs(torch.diff(signal[idxs]))
                            Lk += torch.sum(diffs) * (N - 1) / (((N - m) // k) * k)
                    L.append(float(Lk / k))
                
                if len(L) < 2:
                    return 1.5
                
                # Linear regression on log-log plot
                k_vals = torch.arange(1, len(L) + 1, dtype=torch.float32)
                log_k = torch.log(k_vals)
                log_L = torch.log(torch.tensor(L) + 1e-8)
                
                # Simple linear regression
                n = len(log_k)
                slope = (n * torch.sum(log_k * log_L) - torch.sum(log_k) * torch.sum(log_L)) / \
                       (n * torch.sum(log_k ** 2) - torch.sum(log_k) ** 2)
                
                return float(-slope)  # Negative because we expect negative slope
            
            fractal_dim = higuchi_fd(data_flat)
            
            # Multifractal analysis approximation
            # Compute local scaling exponents
            windows = []
            window_size = max(4, len(data_flat) // 8)
            for i in range(0, len(data_flat) - window_size, window_size // 2):
                window = data_flat[i:i + window_size]
                windows.append(torch.std(window))
            
            if len(windows) > 1:
                windows_tensor = torch.stack(windows)
                multifractal_spectrum = torch.std(windows_tensor) / (torch.mean(windows_tensor) + 1e-8)
                bifractal_strength = float(multifractal_spectrum)
            else:
                bifractal_strength = 0.0
            
            # Complexity measures
            lempel_ziv_complexity = len(set(tuple(data_flat[i:i+3].tolist()) for i in range(len(data_flat)-2)))
            normalized_complexity = lempel_ziv_complexity / max(1, len(data_flat) - 2)
            
            # Self-similarity measure using convolution instead of correlate
            if len(data_flat) > 3:
                # Use conv1d for autocorrelation approximation
                data_normalized = (data_flat - torch.mean(data_flat)) / (torch.std(data_flat) + 1e-8)
                # Simple autocorrelation approximation
                mid_idx = len(data_normalized) // 2
                if mid_idx > 1:
                    lag1_corr = float(torch.dot(data_normalized[:-1], data_normalized[1:]) / (len(data_normalized) - 1))
                    self_similarity = abs(lag1_corr)
                else:
                    self_similarity = 0.0
            else:
                self_similarity = 0.0
            
            return {
                'fractal_dimension': min(3.0, max(1.0, fractal_dim)),
                'bifractal_strength': min(1.0, bifractal_strength),
                'complexity_measure': normalized_complexity,
                'self_similarity': min(1.0, max(0.0, self_similarity)),
                'multifractal_spectrum': float(multifractal_spectrum) if len(windows) > 1 else 0.0,
                'pattern_regularity': 1.0 - normalized_complexity
            }
    
    @staticmethod
    def compute_semantic_attractors(data: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Compute meaningful semantic attractor and phase space metrics"""
        if SCBF_AVAILABLE:
            return compute_semantic_attractor_density(data, **kwargs)
        else:
            data_flat = data.flatten()
            if len(data_flat) == 0:
                return {'num_attractors': 0, 'attractor_strength': 0.0, 'density': 0.0}
            
            # K-means clustering for attractor identification
            data_2d = data.view(-1, data.shape[-1]) if len(data.shape) > 1 else data.unsqueeze(1)
            
            # Determine optimal number of clusters using elbow method approximation
            max_k = min(10, len(data_2d) // 2)
            if max_k < 2:
                return {'num_attractors': 1, 'attractor_strength': 1.0, 'density': 1.0}
            
            inertias = []
            for k in range(1, max_k + 1):
                # Simple k-means approximation
                centroids = data_2d[torch.randperm(len(data_2d))[:k]]
                distances = torch.cdist(data_2d, centroids)
                closest_centroids = torch.argmin(distances, dim=1)
                inertia = 0
                for i in range(k):
                    cluster_points = data_2d[closest_centroids == i]
                    if len(cluster_points) > 0:
                        inertia += torch.sum((cluster_points - centroids[i]) ** 2)
                inertias.append(float(inertia))
            
            # Find elbow point
            if len(inertias) > 2:
                diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                optimal_k = diffs.index(max(diffs)) + 1
            else:
                optimal_k = 2
            
            # Attractor strength based on cluster separation
            centroids = data_2d[torch.randperm(len(data_2d))[:optimal_k]]
            inter_cluster_distances = torch.cdist(centroids, centroids)
            avg_separation = float(torch.mean(inter_cluster_distances[inter_cluster_distances > 0]))
            
            # Attractor basin analysis
            distances_to_centroids = torch.cdist(data_2d, centroids)
            closest_centroids = torch.argmin(distances_to_centroids, dim=1)
            
            # Calculate basin stability
            basin_stabilities = []
            for i in range(optimal_k):
                cluster_points = data_2d[closest_centroids == i]
                if len(cluster_points) > 1:
                    intra_cluster_var = float(torch.var(cluster_points))
                    basin_stabilities.append(1.0 / (1.0 + intra_cluster_var))
                else:
                    basin_stabilities.append(1.0)
            
            avg_basin_stability = sum(basin_stabilities) / len(basin_stabilities)
            
            # Phase space density
            phase_space_volume = float(torch.prod(torch.max(data_2d, dim=0)[0] - torch.min(data_2d, dim=0)[0]))
            density = len(data_2d) / max(phase_space_volume, 1e-8)
            
            # Information bottleneck principle
            cluster_entropies = []
            for i in range(optimal_k):
                cluster_points = data_2d[closest_centroids == i]
                if len(cluster_points) > 1:
                    cluster_flat = cluster_points.flatten()
                    probs = torch.softmax(cluster_flat, dim=0)
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
                    cluster_entropies.append(float(entropy))
            
            avg_cluster_entropy = sum(cluster_entropies) / max(len(cluster_entropies), 1)
            
            # Safe entropy calculation with division by zero protection
            global_entropy = -torch.sum(torch.softmax(data_flat, dim=0) * torch.log_softmax(data_flat, dim=0))
            safe_global_entropy = max(float(global_entropy), 1e-8)  # Prevent division by zero
            information_compression = 1.0 - (avg_cluster_entropy / safe_global_entropy)
            
            return {
                'num_attractors': optimal_k,
                'attractor_strength': avg_basin_stability,
                'density': min(1.0, density / len(data_2d)),
                'basin_stability': avg_basin_stability,
                'cluster_separation': avg_separation,
                'phase_space_density': density,
                'information_compression': max(0.0, information_compression),
                'attractor_diversity': len(set(closest_centroids.tolist())) / len(data_2d)
            }


class SCBFTracker:
    """
    Core SCBF debugging and metrics tracker.
    Provides hooks into runtime operations for comprehensive analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get('scbf_enabled', True)
        
        # Tracking data
        self.metrics_history = []
        self.collapse_events = []
        self.ancestry_traces = []
        self.attractor_evolution = []
        self.operation_logs = []
        
        # State tracking
        self.step_count = 0
        self.previous_state = None
        self.current_experiment = None
        
        # Performance tracking
        self.start_time = time.time()
        self.operation_times = {}
        
        print(f"🧠 SCBF Tracker initialized (enabled: {self.enabled})")
    
    def start_experiment(self, experiment_name: str, metadata: Dict[str, Any] = None):
        """Start tracking a new experiment"""
        if not self.enabled:
            return
        
        self.current_experiment = {
            'name': experiment_name,
            'start_time': time.time(),
            'metadata': metadata or {},
            'step_count': 0
        }
        
        print(f"📝 SCBF: Started experiment '{experiment_name}'")
    
    def track_operation(self, operation_name: str, input_data: torch.Tensor, 
                       output_data: torch.Tensor = None, metadata: Dict[str, Any] = None):
        """
        Enhanced SCBF operation tracking with comprehensive neural dynamics analysis.
        Captures meaningful patterns, information flow, and cognitive processes.
        """
        if not self.enabled:
            return {}
        
        operation_start = time.time()
        
        # Comprehensive SCBF metrics with enhanced fallback implementations
        metrics = {}
        
        # 1. Enhanced Entropy Collapse Analysis
        entropy_metrics = SCBFMetrics.compute_symbolic_entropy_collapse(input_data)
        metrics['entropy_collapse'] = entropy_metrics
        
        # 2. Enhanced Activation Ancestry with Pattern Tracking
        if self.previous_state is not None:
            ancestry_metrics = SCBFMetrics.compute_activation_ancestry(input_data, self.previous_state)
            metrics['ancestry'] = ancestry_metrics
            self.ancestry_traces.append(ancestry_metrics)
            
            # Additional temporal dynamics
            if len(input_data.flatten()) == len(self.previous_state.flatten()) and len(input_data.flatten()) > 1:
                try:
                    temporal_coherence = float(torch.corrcoef(torch.stack([
                        input_data.flatten(), 
                        self.previous_state.flatten()
                    ]))[0, 1])
                except Exception:
                    temporal_coherence = 0.0
            else:
                temporal_coherence = 0.0
            metrics['temporal_coherence'] = temporal_coherence
            
        # 3. Enhanced Bifractal and Complexity Analysis
        bifractal_metrics = SCBFMetrics.compute_bifractal_lineage(input_data)
        metrics['bifractal'] = bifractal_metrics
        
        # 4. Enhanced Semantic Attractors and Phase Space Analysis
        attractor_metrics = SCBFMetrics.compute_semantic_attractors(input_data)
        metrics['attractors'] = attractor_metrics
        self.attractor_evolution.append(attractor_metrics)
        
        # 5. Neural Dynamics Analysis
        neural_dynamics = self._compute_operation_neural_dynamics(input_data, output_data)
        metrics['neural_dynamics'] = neural_dynamics
        
        # 6. Information Integration Analysis
        if output_data is not None:
            integration_metrics = self._compute_operation_information_flow(input_data, output_data)
            metrics['information_integration'] = integration_metrics
        
        # 7. Cognitive Load and Processing Analysis
        cognitive_metrics = self._compute_operation_cognitive_metrics(input_data, output_data, metadata)
        metrics['cognitive_analysis'] = cognitive_metrics
        
        # 8. Enhanced Collapse Event Detection
        collapse_magnitude = entropy_metrics.get('collapse_magnitude', 0.0)
        information_integration = metrics.get('information_integration', {}).get('integrated_information', 0.0)
        
        # More sophisticated collapse detection
        is_significant_collapse = (
            collapse_magnitude > 0.3 or  # Lower threshold for entropy collapse
            information_integration > 0.7 or  # High information integration
            neural_dynamics.get('criticality', 0.0) > 0.8  # High criticality
        )
        
        if is_significant_collapse:
            collapse_event = {
                'step': self.step_count,
                'operation': operation_name,
                'entropy_magnitude': collapse_magnitude,
                'integration_level': information_integration,
                'criticality': neural_dynamics.get('criticality', 0.0),
                'timestamp': time.time() - self.start_time,
                'significance_score': collapse_magnitude + information_integration + neural_dynamics.get('criticality', 0.0)
            }
            self.collapse_events.append(collapse_event)
        
        # Enhanced Operation Logging
        operation_log = {
            'step': self.step_count,
            'operation': operation_name,
            'timestamp': time.time() - self.start_time,
            'execution_time': time.time() - operation_start,
            'metrics': metrics,
            'metadata': metadata or {},
            'data_characteristics': {
                'input_shape': list(input_data.shape),
                'input_norm': float(torch.norm(input_data)),
                'input_sparsity': float(torch.sum(torch.abs(input_data) < 1e-6)) / input_data.numel(),
                'output_shape': list(output_data.shape) if output_data is not None else None,
                'output_norm': float(torch.norm(output_data)) if output_data is not None else None
            }
        }
        
        self.operation_logs.append(operation_log)
        self.metrics_history.append(metrics)
        
        # Update state with enhanced tracking
        self.previous_state = input_data.clone() if input_data is not None else None
        self.step_count += 1
        
        # Enhanced operation timing and performance tracking
        execution_time = time.time() - operation_start
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(execution_time)
        
        # Track performance trends
        if not hasattr(self, 'performance_trends'):
            self.performance_trends = {}
        
        if operation_name not in self.performance_trends:
            self.performance_trends[operation_name] = {
                'avg_time': execution_time,
                'avg_complexity': cognitive_metrics.get('processing_complexity', 0.0),
                'avg_efficiency': cognitive_metrics.get('cognitive_efficiency', 0.0),
                'count': 1
            }
        else:
            trend = self.performance_trends[operation_name]
            count = trend['count']
            alpha = 1.0 / (count + 1)  # Simple moving average
            trend['avg_time'] = (1 - alpha) * trend['avg_time'] + alpha * execution_time
            trend['avg_complexity'] = (1 - alpha) * trend['avg_complexity'] + alpha * cognitive_metrics.get('processing_complexity', 0.0)
            trend['avg_efficiency'] = (1 - alpha) * trend['avg_efficiency'] + alpha * cognitive_metrics.get('cognitive_efficiency', 0.0)
            trend['count'] += 1
        
        return metrics
    
    def _compute_operation_neural_dynamics(self, input_data: torch.Tensor, output_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute neural dynamics specific to this operation"""
        metrics = {}
        
        input_flat = input_data.flatten()
        
        # Input characteristics
        sparsity = float(torch.sum(torch.abs(input_flat) < 1e-6)) / len(input_flat)
        metrics['input_sparsity'] = sparsity
        
        dynamic_range = float(torch.max(input_flat) - torch.min(input_flat))
        metrics['input_dynamic_range'] = dynamic_range
        
        # Signal characteristics
        signal_power = float(torch.mean(input_flat ** 2))
        noise_estimate = float(torch.var(input_flat))
        snr = signal_power / (noise_estimate + 1e-8)
        metrics['signal_to_noise_ratio'] = snr
        
        # Processing transformation (if output available)
        if output_data is not None:
            output_flat = output_data.flatten()
            
            # Information preservation
            if len(input_flat) == len(output_flat) and len(input_flat) > 1:
                try:
                    preservation = float(torch.corrcoef(torch.stack([input_flat, output_flat]))[0, 1])
                except Exception:
                    preservation = 0.0
            else:
                preservation = 0.0
            metrics['information_preservation'] = preservation
            
            # Transformation magnitude
            transformation = float(torch.norm(output_flat - input_flat)) / (torch.norm(input_flat) + 1e-8)
            metrics['transformation_magnitude'] = transformation
            
            # Amplification/Attenuation
            gain = float(torch.norm(output_flat)) / (float(torch.norm(input_flat)) + 1e-8)
            metrics['processing_gain'] = gain
            
            # Nonlinearity measure
            linear_prediction = input_flat * gain
            nonlinearity = float(torch.norm(output_flat - linear_prediction)) / (float(torch.norm(output_flat)) + 1e-8)
            metrics['nonlinearity'] = nonlinearity
        
        # Criticality indicators
        if len(input_data.shape) > 1:
            # Lyapunov-like exponent approximation
            jacobian_approx = torch.diff(input_data, dim=-1)
            if jacobian_approx.numel() > 0:
                lyapunov_approx = float(torch.mean(torch.log(torch.abs(jacobian_approx) + 1e-8)))
                criticality = 1.0 / (1.0 + abs(lyapunov_approx))
                metrics['criticality'] = criticality
        
        return metrics
    
    def _compute_operation_information_flow(self, input_data: torch.Tensor, output_data: torch.Tensor) -> Dict[str, float]:
        """Compute information flow and integration metrics"""
        metrics = {}
        
        input_flat = input_data.flatten()
        output_flat = output_data.flatten()
        
        # Information transfer efficiency
        if len(input_flat) == len(output_flat) and len(input_flat) > 1:
            try:
                mutual_info_approx = float(torch.corrcoef(torch.stack([input_flat, output_flat]))[0, 1] ** 2)
            except Exception:
                mutual_info_approx = 0.0
            metrics['information_transfer'] = mutual_info_approx
        else:
            metrics['information_transfer'] = 0.0
        
        # Entropy change
        input_entropy = -torch.sum(torch.softmax(input_flat, dim=0) * torch.log_softmax(input_flat, dim=0))
        output_entropy = -torch.sum(torch.softmax(output_flat, dim=0) * torch.log_softmax(output_flat, dim=0))
        entropy_change = float(output_entropy - input_entropy)
        metrics['entropy_change'] = entropy_change
        
        # Information compression/expansion
        compression_ratio = len(output_flat) / len(input_flat)
        metrics['compression_ratio'] = compression_ratio
        
        # Integrated information approximation
        if len(input_data.shape) > 1 and len(output_data.shape) > 1:
            # Split into regions and compute integration
            mid_input = input_data.shape[-1] // 2
            mid_output = output_data.shape[-1] // 2
            
            input_region1 = input_data[..., :mid_input].flatten()
            input_region2 = input_data[..., mid_input:].flatten()
            output_region1 = output_data[..., :mid_output].flatten()
            output_region2 = output_data[..., mid_output:].flatten()
            
            # Cross-region integration
            cross_integration = 0.0
            try:
                if len(input_region1) > 0 and len(output_region2) > 0 and len(input_region1) == len(output_region2):
                    cross_integration += abs(float(torch.corrcoef(torch.stack([input_region1, output_region2]))[0, 1]))
                if len(input_region2) > 0 and len(output_region1) > 0 and len(input_region2) == len(output_region1):
                    cross_integration += abs(float(torch.corrcoef(torch.stack([input_region2, output_region1]))[0, 1]))
            except Exception:
                cross_integration = 0.0
            
            metrics['integrated_information'] = cross_integration / 2.0
        
        return metrics
    
    def _compute_operation_cognitive_metrics(self, input_data: torch.Tensor, output_data: Optional[torch.Tensor], metadata: Optional[Dict]) -> Dict[str, float]:
        """Compute cognitive load and processing efficiency for this operation"""
        metrics = {}
        
        input_flat = input_data.flatten()
        
        # Cognitive load based on processing complexity
        min_val = torch.min(input_flat)
        max_val = torch.max(input_flat)
        
        # Safe histogram calculation with equal min/max protection
        if torch.abs(max_val - min_val) < 1e-8:
            # All values are essentially the same, create uniform distribution
            histogram = torch.ones(20) / 20.0
        else:
            histogram = torch.histc(input_flat, bins=20, min=float(min_val), max=float(max_val))
            
        histogram_norm = histogram / (torch.sum(histogram) + 1e-8)
        entropy = -torch.sum(histogram_norm * torch.log2(histogram_norm + 1e-8))
        processing_complexity = float(entropy / torch.log2(torch.tensor(20.0)))
        metrics['processing_complexity'] = processing_complexity
        
        # Processing efficiency
        total_activation = float(torch.sum(torch.abs(input_flat)))
        if total_activation > 0:
            information_density = entropy / total_activation
            metrics['cognitive_efficiency'] = float(information_density)
        else:
            metrics['cognitive_efficiency'] = 0.0
        
        # Attention and focus metrics
        max_activation = float(torch.max(torch.abs(input_flat)))
        mean_activation = float(torch.mean(torch.abs(input_flat)))
        attention_focus = max_activation / (mean_activation + 1e-8)
        metrics['attention_focus'] = min(10.0, attention_focus)
        
        # Working memory load
        significant_threshold = torch.std(input_flat) * 2
        significant_activations = torch.abs(input_flat) > significant_threshold
        working_memory_load = float(torch.sum(significant_activations)) / len(input_flat)
        metrics['working_memory_load'] = working_memory_load
        
        # Task-specific metrics from metadata
        if metadata:
            if 'energy_level' in metadata:
                energy_efficiency = float(metadata['energy_level']) / (total_activation + 1e-8)
                metrics['energy_efficiency'] = energy_efficiency
            
            if 'stimulus_strength' in metadata:
                response_sensitivity = total_activation / (float(metadata['stimulus_strength']) + 1e-8)
                metrics['response_sensitivity'] = response_sensitivity
        
        # Output-based metrics
        if output_data is not None:
            output_flat = output_data.flatten()
            output_activation = float(torch.sum(torch.abs(output_flat)))
            
            # Processing amplification
            amplification = output_activation / (total_activation + 1e-8)
            metrics['processing_amplification'] = amplification
            
            # Selectivity (how focused the output is)
            output_max = float(torch.max(torch.abs(output_flat)))
            output_mean = float(torch.mean(torch.abs(output_flat)))
            selectivity = output_max / (output_mean + 1e-8)
            metrics['output_selectivity'] = min(10.0, selectivity)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive SCBF tracking summary"""
        if not self.enabled:
            return {'enabled': False}
        
        total_time = time.time() - self.start_time
        
        # Compute aggregated metrics
        avg_entropy_collapse = np.mean([m['entropy_collapse']['collapse_magnitude'] 
                                      for m in self.metrics_history]) if self.metrics_history else 0.0
        
        avg_ancestry_strength = np.mean([a['ancestry_strength'] 
                                       for a in self.ancestry_traces]) if self.ancestry_traces else 0.0
        
        total_attractors = sum([a['num_attractors'] 
                              for a in self.attractor_evolution]) if self.attractor_evolution else 0
        
        return {
            'enabled': True,
            'experiment': self.current_experiment,
            'total_runtime': total_time,
            'total_steps': self.step_count,
            'total_operations': len(self.operation_logs),
            'collapse_events': len(self.collapse_events),
            'metrics_summary': {
                'avg_entropy_collapse': avg_entropy_collapse,
                'avg_ancestry_strength': avg_ancestry_strength,
                'total_attractors': total_attractors,
                'operations_tracked': list(self.operation_times.keys())
            },
            'performance': {
                'avg_operation_times': {op: np.mean(times) 
                                      for op, times in self.operation_times.items()}
            }
        }


class SCBFDashboard:
    """
    SCBF Dashboard data provider for GAIA OutputManager.
    Provides raw data and visualization components instead of generating standalone dashboards.
    """
    
    def __init__(self, tracker: SCBFTracker):
        self.tracker = tracker
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive SCBF data for dashboard integration"""
        if not self.tracker.enabled or not self.tracker.metrics_history:
            return {'enabled': False, 'message': 'No SCBF data available'}
        
        # Extract metrics time series
        steps = list(range(len(self.tracker.metrics_history)))
        
        # Entropy collapse data
        entropy_data = {
            'steps': steps,
            'collapse_magnitudes': [m['entropy_collapse']['collapse_magnitude'] 
                                   for m in self.tracker.metrics_history],
            'collapse_events': [{'step': e['step'], 'magnitude': e.get('entropy_magnitude', e.get('magnitude', 0))} 
                              for e in self.tracker.collapse_events]
        }
        
        # Ancestry data
        ancestry_data = {
            'steps': list(range(len(self.tracker.ancestry_traces))),
            'ancestry_strengths': [a['ancestry_strength'] for a in self.tracker.ancestry_traces],
            'lineage_stability': [a['lineage_stability'] for a in self.tracker.ancestry_traces]
        } if self.tracker.ancestry_traces else {'steps': [], 'ancestry_strengths': [], 'lineage_stability': []}
        
        # Bifractal data
        bifractal_data = {
            'steps': steps,
            'fractal_dimensions': [m['bifractal']['fractal_dimension'] for m in self.tracker.metrics_history],
            'bifractal_strengths': [m['bifractal']['bifractal_strength'] for m in self.tracker.metrics_history]
        }
        
        # Attractor data
        attractor_data = {
            'steps': list(range(len(self.tracker.attractor_evolution))),
            'num_attractors': [a['num_attractors'] for a in self.tracker.attractor_evolution],
            'attractor_densities': [a['density'] for a in self.tracker.attractor_evolution]
        } if self.tracker.attractor_evolution else {'steps': [], 'num_attractors': [], 'attractor_densities': []}
        
        # Performance data
        performance_data = {
            'operations': list(self.tracker.operation_times.keys()),
            'avg_times': [np.mean(times) for times in self.tracker.operation_times.values()]
        } if self.tracker.operation_times else {'operations': [], 'avg_times': []}
        
        # Summary statistics
        summary = self.tracker.get_summary()
        
        return {
            'enabled': True,
            'summary': summary,
            'time_series': {
                'entropy': entropy_data,
                'ancestry': ancestry_data,
                'bifractal': bifractal_data,
                'attractors': attractor_data,
                'performance': performance_data
            },
            'raw_data': {
                'metrics_history': self.tracker.metrics_history,
                'operation_logs': self.tracker.operation_logs
            }
        }
    
    def get_plot_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get plot configuration data for dashboard visualization"""
        return {
            'entropy_collapse': {
                'type': 'line',
                'title': 'Symbolic Entropy Collapse',
                'x_label': 'Step',
                'y_label': 'Collapse Magnitude',
                'data_key': 'time_series.entropy.collapse_magnitudes'
            },
            'ancestry_evolution': {
                'type': 'multi_line',
                'title': 'Activation Ancestry Evolution',
                'x_label': 'Step',
                'y_label': 'Strength',
                'lines': [
                    {'data_key': 'time_series.ancestry.ancestry_strengths', 'label': 'Ancestry Strength'},
                    {'data_key': 'time_series.ancestry.lineage_stability', 'label': 'Lineage Stability'}
                ]
            },
            'bifractal_dynamics': {
                'type': 'multi_line',
                'title': 'Bifractal Dynamics',
                'x_label': 'Step', 
                'y_label': 'Value',
                'lines': [
                    {'data_key': 'time_series.bifractal.fractal_dimensions', 'label': 'Fractal Dimension'},
                    {'data_key': 'time_series.bifractal.bifractal_strengths', 'label': 'Bifractal Strength'}
                ]
            },
            'attractor_formation': {
                'type': 'dual_axis',
                'title': 'Semantic Attractor Formation',
                'x_label': 'Step',
                'y1_label': 'Number of Attractors',
                'y2_label': 'Attractor Density',
                'y1_data_key': 'time_series.attractors.num_attractors',
                'y2_data_key': 'time_series.attractors.attractor_densities'
            },
            'operation_performance': {
                'type': 'bar',
                'title': 'Average Operation Times',
                'x_label': 'Operation',
                'y_label': 'Time (seconds)',
                'data_key': 'time_series.performance'
            }
        }
    
    def create_plots_for_dashboard(self, save_dir: Path) -> List[str]:
        """Create enhanced plot files for dashboard inclusion with meaningful neural dynamics insights"""
        if not self.tracker.enabled or not self.tracker.metrics_history:
            return []
        
        # Set style for scientific visualization
        plt.style.use('default')
        sns.set_palette("husl")
        
        created_plots = []
        
        # 1. Create comprehensive neural dynamics plot
        if self.tracker.metrics_history:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('SCBF Neural Dynamics Analysis', fontsize=16, fontweight='bold')
            
            steps = list(range(len(self.tracker.metrics_history)))
            
            # Enhanced Entropy and Information Analysis
            entropy_initial = [m['entropy_collapse'].get('entropy_initial', 0) for m in self.tracker.metrics_history]
            entropy_final = [m['entropy_collapse'].get('entropy_final', 0) for m in self.tracker.metrics_history]
            mutual_info = [m['entropy_collapse'].get('mutual_information', 0) for m in self.tracker.metrics_history]
            
            ax1.plot(steps, entropy_initial, 'b-', alpha=0.8, linewidth=2, label='Shannon Entropy')
            ax1.plot(steps, entropy_final, 'r-', alpha=0.8, linewidth=2, label='Von Neumann Entropy')
            ax1.plot(steps, mutual_info, 'g-', alpha=0.8, linewidth=2, label='Mutual Information')
            ax1.fill_between(steps, entropy_initial, entropy_final, alpha=0.2, color='cyan', label='Entropy Gap')
            
            # Mark significant collapse events
            for event in self.tracker.collapse_events:
                ax1.axvline(x=event['step'], color='darkred', linestyle='--', alpha=0.6, 
                           label='Collapse Event' if event == self.tracker.collapse_events[0] else "")
            
            ax1.set_title('Information Dynamics & Entropy Evolution', fontweight='bold')
            ax1.set_ylabel('Entropy (bits)')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Cognitive Processing Metrics
            if any('cognitive_analysis' in m for m in self.tracker.metrics_history):
                complexity = [m.get('cognitive_analysis', {}).get('processing_complexity', 0) for m in self.tracker.metrics_history]
                efficiency = [m.get('cognitive_analysis', {}).get('cognitive_efficiency', 0) for m in self.tracker.metrics_history]
                attention = [m.get('cognitive_analysis', {}).get('attention_focus', 0) for m in self.tracker.metrics_history]
                
                ax2_twin = ax2.twinx()
                line1 = ax2.plot(steps, complexity, 'purple', alpha=0.8, linewidth=2, label='Processing Complexity')
                line2 = ax2.plot(steps, efficiency, 'orange', alpha=0.8, linewidth=2, label='Cognitive Efficiency')
                line3 = ax2_twin.plot(steps, attention, 'red', alpha=0.8, linewidth=2, label='Attention Focus')
                
                ax2.set_title('Cognitive Processing Dynamics', fontweight='bold')
                ax2.set_ylabel('Complexity / Efficiency')
                ax2_twin.set_ylabel('Attention Focus')
                
                # Combine legends
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, fontsize=8, loc='upper left')
                ax2.grid(True, alpha=0.3)
            
            # Neural Network State Dynamics
            if any('neural_dynamics' in m for m in self.tracker.metrics_history):
                sparsity = [m.get('neural_dynamics', {}).get('input_sparsity', 0) for m in self.tracker.metrics_history]
                snr = [m.get('neural_dynamics', {}).get('signal_to_noise_ratio', 0) for m in self.tracker.metrics_history]
                criticality = [m.get('neural_dynamics', {}).get('criticality', 0) for m in self.tracker.metrics_history]
                
                ax3.plot(steps, sparsity, 'cyan', alpha=0.8, linewidth=2, label='Sparsity')
                ax3.plot(steps, criticality, 'magenta', alpha=0.8, linewidth=2, label='Criticality')
                
                ax3_twin = ax3.twinx()
                ax3_twin.plot(steps, snr, 'brown', alpha=0.8, linewidth=2, label='SNR')
                
                ax3.set_title('Neural State & Criticality', fontweight='bold')
                ax3.set_ylabel('Sparsity / Criticality')
                ax3_twin.set_ylabel('Signal-to-Noise Ratio')
                ax3.set_xlabel('Step')
                
                # Combine legends
                lines1 = ax3.get_lines()
                lines2 = ax3_twin.get_lines()
                ax3.legend(lines1 + lines2, [l.get_label() for l in lines1 + lines2], fontsize=8)
                ax3.grid(True, alpha=0.3)
            
            # Information Integration & Flow
            if any('information_integration' in m for m in self.tracker.metrics_history):
                integration = [m.get('information_integration', {}).get('integrated_information', 0) for m in self.tracker.metrics_history]
                entropy_change = [m.get('information_integration', {}).get('entropy_change', 0) for m in self.tracker.metrics_history]
                info_transfer = [m.get('information_integration', {}).get('information_transfer', 0) for m in self.tracker.metrics_history]
                
                ax4.plot(steps, integration, 'blue', alpha=0.8, linewidth=2, label='Integrated Information (Φ)')
                ax4.plot(steps, info_transfer, 'green', alpha=0.8, linewidth=2, label='Information Transfer')
                
                ax4_twin = ax4.twinx()
                ax4_twin.plot(steps, entropy_change, 'red', alpha=0.8, linewidth=2, label='Entropy Change')
                
                ax4.set_title('Information Integration & Flow', fontweight='bold')
                ax4.set_ylabel('Integration / Transfer')
                ax4_twin.set_ylabel('Entropy Change')
                ax4.set_xlabel('Step')
                
                # Combine legends
                lines1 = ax4.get_lines()
                lines2 = ax4_twin.get_lines()
                ax4.legend(lines1 + lines2, [l.get_label() for l in lines1 + lines2], fontsize=8)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            neural_dynamics_plot = save_dir / "scbf_neural_dynamics.png"
            plt.savefig(neural_dynamics_plot, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(neural_dynamics_plot.name))
        
        # 2. Create semantic attractor evolution plot
        if self.tracker.attractor_evolution:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Semantic Attractor Dynamics', fontsize=14, fontweight='bold')
            
            attractor_steps = list(range(len(self.tracker.attractor_evolution)))
            num_attractors = [a.get('num_attractors', 0) for a in self.tracker.attractor_evolution]
            attractor_strength = [a.get('attractor_strength', 0) for a in self.tracker.attractor_evolution]
            basin_stability = [a.get('basin_stability', 0) for a in self.tracker.attractor_evolution]
            phase_density = [a.get('phase_space_density', 0) for a in self.tracker.attractor_evolution]
            
            # Attractor count and strength
            ax1_twin = ax1.twinx()
            line1 = ax1.plot(attractor_steps, num_attractors, 'b-', alpha=0.8, linewidth=2, 
                            marker='o', markersize=4, label='Number of Attractors')
            line2 = ax1_twin.plot(attractor_steps, attractor_strength, 'r-', alpha=0.8, linewidth=2,
                                 marker='s', markersize=4, label='Attractor Strength')
            
            ax1.set_title('Attractor Count & Strength Evolution')
            ax1.set_ylabel('Number of Attractors', color='blue')
            ax1_twin.set_ylabel('Attractor Strength', color='red')
            ax1.set_xlabel('Step')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Basin stability and phase space
            ax2.plot(attractor_steps, basin_stability, 'green', alpha=0.8, linewidth=2,
                    marker='^', markersize=4, label='Basin Stability')
            ax2.plot(attractor_steps, phase_density, 'purple', alpha=0.8, linewidth=2,
                    marker='v', markersize=4, label='Phase Space Density')
            
            ax2.set_title('Attractor Basin Dynamics')
            ax2.set_ylabel('Stability / Density')
            ax2.set_xlabel('Step')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            attractor_plot = save_dir / "scbf_attractor_dynamics.png"
            plt.savefig(attractor_plot, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(attractor_plot.name))
        
        # 3. Create complexity and fractal analysis plot
        if self.tracker.metrics_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Complexity & Bifractal Analysis', fontsize=14, fontweight='bold')
            
            # Fractal and complexity metrics
            fractal_dims = [m['bifractal'].get('fractal_dimension', 1.0) for m in self.tracker.metrics_history]
            bifractal_strengths = [m['bifractal'].get('bifractal_strength', 0) for m in self.tracker.metrics_history]
            complexity_measures = [m['bifractal'].get('complexity_measure', 0) for m in self.tracker.metrics_history]
            self_similarity = [m['bifractal'].get('self_similarity', 0) for m in self.tracker.metrics_history]
            
            ax1.plot(steps, fractal_dims, 'purple', alpha=0.8, linewidth=2, marker='o', 
                    markersize=4, label='Fractal Dimension')
            ax1.plot(steps, bifractal_strengths, 'orange', alpha=0.8, linewidth=2, marker='s',
                    markersize=4, label='Bifractal Strength')
            
            ax1.set_title('Fractal Dynamics')
            ax1.set_ylabel('Dimension / Strength')
            ax1.set_xlabel('Step')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(steps, complexity_measures, 'red', alpha=0.8, linewidth=2, marker='^',
                    markersize=4, label='Complexity Measure')
            ax2.plot(steps, self_similarity, 'blue', alpha=0.8, linewidth=2, marker='v',
                    markersize=4, label='Self-Similarity')
            
            ax2.set_title('Pattern Complexity')
            ax2.set_ylabel('Complexity / Similarity')
            ax2.set_xlabel('Step')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            complexity_plot = save_dir / "scbf_complexity_analysis.png"
            plt.savefig(complexity_plot, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(complexity_plot.name))
        
        # 4. Create performance and efficiency analysis
        if hasattr(self.tracker, 'performance_trends') and self.tracker.performance_trends:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('SCBF Performance & Efficiency Analysis', fontsize=16, fontweight='bold')
            
            operations = list(self.tracker.performance_trends.keys())
            avg_times = [self.tracker.performance_trends[op]['avg_time'] * 1000 for op in operations]  # Convert to ms
            avg_complexities = [self.tracker.performance_trends[op]['avg_complexity'] for op in operations]
            avg_efficiencies = [self.tracker.performance_trends[op]['avg_efficiency'] for op in operations]
            counts = [self.tracker.performance_trends[op]['count'] for op in operations]
            
            # Processing times
            bars1 = ax1.bar(operations, avg_times, alpha=0.7, color='skyblue')
            ax1.set_title('Average Processing Time per Operation')
            ax1.set_ylabel('Time (ms)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time in zip(bars1, avg_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(avg_times),
                        f'{time:.2f}ms', ha='center', va='bottom', fontsize=8)
            
            # Complexity analysis
            bars2 = ax2.bar(operations, avg_complexities, alpha=0.7, color='lightcoral')
            ax2.set_title('Average Processing Complexity')
            ax2.set_ylabel('Complexity')
            ax2.tick_params(axis='x', rotation=45)
            
            # Efficiency analysis
            bars3 = ax3.bar(operations, avg_efficiencies, alpha=0.7, color='lightgreen')
            ax3.set_title('Average Cognitive Efficiency')
            ax3.set_ylabel('Efficiency')
            ax3.tick_params(axis='x', rotation=45)
            
            # Operation frequency
            bars4 = ax4.bar(operations, counts, alpha=0.7, color='gold')
            ax4.set_title('Operation Frequency')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            performance_plot = save_dir / "scbf_performance_analysis.png"
            plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(performance_plot.name))
        
        # 5. Create legacy entropy collapse plot for compatibility
        if self.tracker.metrics_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            collapse_magnitudes = [m['entropy_collapse'].get('collapse_magnitude', 0) 
                                 for m in self.tracker.metrics_history]
            
            ax.plot(steps, collapse_magnitudes, 'r-', alpha=0.7, linewidth=2, label='Collapse Magnitude')
            ax.fill_between(steps, collapse_magnitudes, alpha=0.3, color='red')
            
            # Mark significant collapse events
            for event in self.tracker.collapse_events:
                ax.axvline(x=event['step'], color='darkred', linestyle='--', alpha=0.8)
            
            ax.set_title('SCBF Symbolic Entropy Collapse (Enhanced)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Collapse Magnitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            entropy_plot_path = save_dir / "scbf_entropy_collapse.png"
            plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            if str(entropy_plot_path.name) not in created_plots:
                created_plots.append(str(entropy_plot_path.name))
        
        return created_plots
    
    def save_metrics_report(self, output_path: str) -> str:
        """Save detailed metrics report as JSON (kept for compatibility)"""
        dashboard_data = self.get_dashboard_data()
        
        report = {
            'scbf_dashboard_data': dashboard_data,
            'generation_time': datetime.now().isoformat(),
            'plot_configs': self.get_plot_configs()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 SCBF Dashboard data saved to: {output_path}")
        return output_path


# Convenience function for easy integration
def create_scbf_system(config: Dict[str, Any] = None) -> Tuple[SCBFTracker, SCBFDashboard]:
    """Create complete SCBF tracking and dashboard system"""
    tracker = SCBFTracker(config)
    dashboard = SCBFDashboard(tracker)
    return tracker, dashboard


# Export main classes
__all__ = ['SCBFTracker', 'SCBFDashboard', 'SCBFMetrics', 'create_scbf_system']
