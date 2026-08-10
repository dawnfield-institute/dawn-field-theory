import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class QBEController:
    """
    Quantum Bifractal Equilibrium (QBE) Controller for dynamic adaptation.
    Manages momentum, error bands, and energy balance for real-time learning.
    """
    def __init__(self, initial_momentum=0.8, error_band=0.1):
        self.momentum = initial_momentum
        self.error_band = error_band
        self.energy_balance = 1.0

    def update(self, error, entropy):
        """Update QBE metrics based on error and entropy."""
        self.momentum = 0.9 * self.momentum + 0.1 * abs(error)
        self.error_band = max(0.05, min(0.2, self.error_band + 0.01 * entropy))
        self.energy_balance = self.momentum + self.error_band

    def get_status(self):
        """Return equilibrium status based on energy balance."""
        if self.energy_balance < 1.5:
            return "Near Equilibrium"
        elif self.energy_balance < 2.0:
            return "Moderate Equilibrium"
        else:
            return "Far from Equilibrium"
            
    def detect_pattern_type(self, recent_values):
        """Detect pattern type based on recent values."""
        if len(recent_values) > 10:
            variance = torch.var(torch.tensor(recent_values[-10:])).item()
            if variance < 0.01:
                return "convergence"
            elif variance > 0.5:
                return "chaotic"
        return "unknown"

    def adjust_for_pattern(self, pattern_type):
        """Adjust QBE settings based on detected pattern type."""
        if pattern_type == "convergence":
            self.error_band = max(0.05, self.error_band - 0.01)
            self.momentum = min(0.9, self.momentum + 0.05)
        elif pattern_type == "chaotic":
            self.error_band = min(0.2, self.error_band + 0.01)
            self.momentum = max(0.7, self.momentum - 0.05)

# --- TinyCIMM-Euler: Higher-Order Mathematical Reasoning Utilities ---

def safe_item(value):
    """Safely extract scalar from tensor or return float value"""
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        elif value.numel() > 1:
            return float(torch.mean(value))
        else:
            return 0.0
    else:
        return float(value)

# Unified SCBF (Symbolic Collapse Benchmarking Framework) Support
class UnifiedSymbolicCollapseTracker:
    """
    Unified SCBF tracker that consolidates all symbolic cognition and collapse-aware metrics
    for interpretable AI (XAI) in mathematical reasoning contexts.
    
    Tracks all key SCBF metrics:
    - Symbolic Entropy Collapse (SEC): emergence of minimal entropy configuration
    - Activation Ancestry Trace: stability of neuron identity over time  
    - Collapse Phase Alignment: temporal coherence of activation collapse
    - Bifractal Lineage: recursive reactivation patterns
    - Semantic Attractor Density: clustering of activation attractors
    - Weight Drift Entropy (ΔW): structural evolution interpretability
    """
    def __init__(self, memory_window=30, sensitivity_factor=2.0):
        self.memory_window = memory_window
        self.sensitivity_factor = sensitivity_factor
        
        # === Symbolic Entropy Collapse Tracking ===
        self.raw_entropies = []
        self.smoothed_entropies = []
        self.collapse_events = []
        self.entropy_momentum = 0.0
        
        # === Activation Ancestry Tracking ===
        self.activation_signatures = []
        self.pattern_stability_scores = []
        self.top_neuron_consistency = []
        self.ancestry_traces = []
        
        # === Phase Alignment Tracking ===
        self.phase_vectors = []
        self.coherence_scores = []
        self.phase_alignment_history = []
        
        # === Bifractal Pattern Tracking ===
        self.pattern_fingerprints = []
        self.recursion_scores = []
        self.mathematical_memory = []
        self.recursive_lineage = []
        
        # === Semantic Attractor Analysis ===
        self.activation_centroids = []
        self.cluster_densities = []
        self.attractor_evolution = []
        self.semantic_attractors = []
        
        # === Weight Evolution Tracking ===
        self.weight_norms = []
        self.weight_changes = []
        self.structural_entropy = []
        self.adaptation_signals = []
        self.gradient_alignment_history = []
        self.prev_weights = None
    
    def compute_symbolic_entropy_collapse(self, activations):
        """
        Compute Symbolic Entropy Collapse (SEC) - the emergence of minimal entropy 
        configuration indicating mathematical pattern learning and consolidation.
        """
        activation_flat = activations.flatten()
        
        # Multi-level entropy analysis for mathematical patterns
        if len(activation_flat) > 1:
            # 1. Raw activation entropy
            abs_activations = torch.abs(activation_flat) + 1e-9
            norm_activations = abs_activations / torch.sum(abs_activations)
            raw_entropy = -torch.sum(norm_activations * torch.log(norm_activations + 1e-9))
            
            # 2. Gradient-based entropy (pattern sharpness)
            if len(activation_flat) > 2:
                gradients = torch.diff(activation_flat)
                grad_magnitude = torch.abs(gradients) + 1e-9
                grad_probs = grad_magnitude / torch.sum(grad_magnitude)
                gradient_entropy = -torch.sum(grad_probs * torch.log(grad_probs + 1e-9))
            else:
                gradient_entropy = raw_entropy
            
            # 3. Order-based entropy (mathematical structure)
            sorted_vals, _ = torch.sort(torch.abs(activation_flat), descending=True)
            if torch.sum(sorted_vals) > 1e-9:
                order_probs = sorted_vals / torch.sum(sorted_vals)
                order_entropy = -torch.sum(order_probs * torch.log(order_probs + 1e-9))
            else:
                order_entropy = raw_entropy
            
            # Combine entropies with mathematical weighting
            combined_entropy = 0.5 * raw_entropy + 0.3 * gradient_entropy + 0.2 * order_entropy
        else:
            combined_entropy = torch.tensor(3.0)  # High entropy for single activation
        
        entropy_val = safe_item(combined_entropy)
        self.raw_entropies.append(entropy_val)
        
        # Smoothed entropy with momentum for collapse detection
        if len(self.raw_entropies) > 1:
            self.entropy_momentum = 0.7 * self.entropy_momentum + 0.3 * (entropy_val - self.raw_entropies[-2])
            smoothed = 0.8 * entropy_val + 0.2 * (self.raw_entropies[-2] if len(self.raw_entropies) > 1 else entropy_val)
        else:
            smoothed = entropy_val
            self.entropy_momentum = 0.0
        
        self.smoothed_entropies.append(smoothed)
        
        # Enhanced collapse detection
        if len(self.smoothed_entropies) > 3:
            # Multi-scale collapse detection
            immediate_change = abs(self.smoothed_entropies[-1] - self.smoothed_entropies[-2])
            short_term_trend = abs(np.mean(self.smoothed_entropies[-2:]) - np.mean(self.smoothed_entropies[-4:-2])) if len(self.smoothed_entropies) >= 4 else 0.0
            
            # Adaptive threshold based on recent variance
            recent_variance = safe_item(torch.var(torch.tensor(self.smoothed_entropies[-5:]))) if len(self.smoothed_entropies) >= 5 else 0.1
            dynamic_threshold = max(0.005, min(0.2, recent_variance * self.sensitivity_factor))
            
            # Detect both immediate and trend-based collapses
            if immediate_change > dynamic_threshold or short_term_trend > dynamic_threshold * 0.7:
                self.collapse_events.append({
                    'step': len(self.smoothed_entropies),
                    'immediate_change': immediate_change,
                    'trend_change': short_term_trend,
                    'magnitude': max(immediate_change, short_term_trend),
                    'threshold_used': dynamic_threshold,
                    'entropy_momentum': self.entropy_momentum
                })
        
        # Cleanup
        if len(self.raw_entropies) > self.memory_window:
            self.raw_entropies.pop(0)
        if len(self.smoothed_entropies) > self.memory_window:
            self.smoothed_entropies.pop(0)
        
        return entropy_val
    
    def track_activation_ancestry(self, activations):
        """
        Track Activation Ancestry Trace - stability of neuron identity and patterns over time.
        This reveals how consistently certain neurons are activated for specific mathematical patterns.
        """
        activation_flat = activations.flatten()
        
        # Create comprehensive pattern signature
        if len(activation_flat) >= 2:
            # 1. Magnitude-based ranking (which neurons are most active)
            activation_magnitude = torch.abs(activation_flat)
            magnitude_ranking = torch.argsort(activation_magnitude, descending=True)
            
            # 2. Value-based signature (actual activation patterns)
            normalized_activations = activation_flat / (torch.norm(activation_flat) + 1e-9)
            
            # 3. Top-k neuron consistency tracking
            top_k = min(5, len(activation_flat))
            current_top_neurons = set(magnitude_ranking[:top_k].tolist())
            
            # Store pattern signature
            pattern_signature = {
                'magnitude_ranking': magnitude_ranking,
                'normalized_pattern': normalized_activations,
                'top_neurons': current_top_neurons,
                'activation_sum': torch.sum(torch.abs(activation_flat))
            }
            self.activation_signatures.append(pattern_signature)
            
            # Compute stability metrics
            if len(self.activation_signatures) > 1:
                prev_signature = self.activation_signatures[-2]
                
                # Top neuron consistency
                prev_top = prev_signature['top_neurons']
                top_consistency = len(current_top_neurons.intersection(prev_top)) / top_k
                self.top_neuron_consistency.append(top_consistency)
                
                # Pattern correlation stability
                try:
                    # Ensure same size for correlation
                    min_size = min(len(prev_signature['normalized_pattern']), len(normalized_activations))
                    if min_size > 1:
                        prev_pattern = prev_signature['normalized_pattern'][:min_size]
                        curr_pattern = normalized_activations[:min_size]
                        
                        pattern_correlation = torch.corrcoef(torch.stack([prev_pattern, curr_pattern]))[0, 1]
                        if not torch.isnan(pattern_correlation):
                            stability_score = safe_item(pattern_correlation)
                        else:
                            stability_score = 0.0
                    else:
                        stability_score = 1.0
                except:
                    stability_score = 0.0
                
                self.pattern_stability_scores.append(stability_score)
                
                # Ancestry trace (how patterns evolve)
                ancestry_trace = {
                    'step': len(self.activation_signatures),
                    'top_consistency': top_consistency,
                    'pattern_stability': stability_score,
                    'activation_intensity_change': abs(safe_item(pattern_signature['activation_sum']) - safe_item(prev_signature['activation_sum']))
                }
                self.ancestry_traces.append(ancestry_trace)
        else:
            # Handle single activation case
            self.top_neuron_consistency.append(1.0)
            self.pattern_stability_scores.append(1.0)
        
        # Cleanup
        if len(self.activation_signatures) > self.memory_window:
            self.activation_signatures.pop(0)
        if len(self.pattern_stability_scores) > self.memory_window:
            self.pattern_stability_scores.pop(0)
        if len(self.top_neuron_consistency) > self.memory_window:
            self.top_neuron_consistency.pop(0)
        
        return self.pattern_stability_scores[-1] if self.pattern_stability_scores else 1.0
    
    def compute_collapse_phase_alignment(self, activations):
        """
        Compute Collapse Phase Alignment - temporal coherence of activation collapse patterns.
        This measures how synchronized the collapse events are across different parts of the network.
        """
        if len(self.activation_signatures) < 3:
            self.phase_alignment_history.append(0.0)
            return 0.0
        
        try:
            # Extract phase information from recent activations
            recent_signatures = self.activation_signatures[-3:]
            
            # Compute phase vectors based on activation patterns
            phase_vectors = []
            for signature in recent_signatures:
                if 'normalized_pattern' in signature:
                    pattern = signature['normalized_pattern']
                    # Create phase vector from top activations
                    top_indices = signature['magnitude_ranking'][:min(4, len(pattern))]
                    phase_vector = torch.zeros(4)
                    for i, idx in enumerate(top_indices):
                        if i < len(phase_vector) and idx < len(pattern):
                            phase_vector[i] = pattern[idx]
                    phase_vectors.append(phase_vector)
            
            if len(phase_vectors) >= 2:
                # Compute phase coherence across time
                correlations = []
                for i in range(len(phase_vectors) - 1):
                    corr = torch.corrcoef(torch.stack([phase_vectors[i], phase_vectors[i+1]]))[0, 1]
                    if not torch.isnan(corr):
                        correlations.append(safe_item(corr))
                
                phase_alignment = np.mean(correlations) if correlations else 0.0
            else:
                phase_alignment = 0.0
                
        except Exception as e:
            phase_alignment = 0.0
        
        self.phase_alignment_history.append(phase_alignment)
        
        # Cleanup
        if len(self.phase_alignment_history) > self.memory_window:
            self.phase_alignment_history.pop(0)
        
        return phase_alignment
    
    def track_bifractal_lineage(self, activations):
        """
        Track Bifractal Lineage - recursive reactivation patterns and mathematical memory.
        This captures how the network reuses learned mathematical patterns recursively.
        """
        activation_flat = activations.flatten()
        
        # Create pattern fingerprint for recursive detection
        signature_size = min(6, activation_flat.numel())
        if signature_size > 0:
            # Multi-faceted pattern fingerprint
            activation_signature = activation_flat[:signature_size]
            
            # Pad to consistent size
            if len(activation_signature) < 6:
                padding = torch.zeros(6 - len(activation_signature), device=activation_signature.device)
                activation_signature = torch.cat([activation_signature, padding])
                
            # Add derivative information for richer patterns
            if len(activation_flat) > 1:
                derivative_info = torch.diff(activation_flat[:min(3, len(activation_flat))])
                # Pad derivative info if needed
                if len(derivative_info) < 2:
                    deriv_padding = torch.zeros(2 - len(derivative_info), device=derivative_info.device)
                    derivative_info = torch.cat([derivative_info, deriv_padding])
                else:
                    derivative_info = derivative_info[:2]
                
                # Combine activation and derivative signatures
                full_signature = torch.cat([activation_signature, derivative_info])
            else:
                full_signature = torch.cat([activation_signature, torch.zeros(2, device=activation_signature.device)])
        else:
            full_signature = torch.zeros(8)
        
        self.pattern_fingerprints.append(full_signature)
        
        # Detect recursive patterns
        if len(self.pattern_fingerprints) >= 5:
            recent_patterns = torch.stack(self.pattern_fingerprints[-5:])
            
            # Multi-scale recursive detection
            recursive_scores = []
            
            # Look for exact and approximate recursive matches
            for i in range(len(recent_patterns) - 2):
                for j in range(i + 2, len(recent_patterns)):
                    try:
                        # Correlation-based similarity
                        pattern_corr = torch.corrcoef(torch.stack([recent_patterns[i], recent_patterns[j]]))[0, 1]
                        if not torch.isnan(pattern_corr) and pattern_corr > 0.7:
                            recursive_scores.append(safe_item(pattern_corr))
                        
                        # Distance-based similarity
                        pattern_distance = torch.norm(recent_patterns[i] - recent_patterns[j])
                        if pattern_distance < 0.5:  # Close patterns
                            recursive_scores.append(1.0 - safe_item(pattern_distance))
                    except:
                        continue
            
            # Compute recursive strength
            if recursive_scores:
                recursive_strength = np.mean(recursive_scores)
                # Add temporal weighting (more recent recursions are more significant)
                temporal_weight = 1.0 + 0.2 * (len(self.pattern_fingerprints) % 10) / 10
                recursive_strength *= temporal_weight
            else:
                recursive_strength = 0.0
                
            self.recursion_scores.append(recursive_strength)
            
            # Mathematical memory: track patterns that recur multiple times
            if recursive_strength > 0.8:  # Strong recursive pattern
                memory_entry = {
                    'pattern': full_signature.clone(),
                    'strength': recursive_strength,
                    'first_seen': len(self.pattern_fingerprints) - 5,
                    'recurrence_count': 1
                }
                
                # Check if this pattern already exists in memory
                found_match = False
                for memory_item in self.mathematical_memory:
                    memory_corr = torch.corrcoef(torch.stack([memory_item['pattern'], full_signature]))[0, 1]
                    if not torch.isnan(memory_corr) and memory_corr > 0.85:
                        memory_item['recurrence_count'] += 1
                        memory_item['strength'] = max(memory_item['strength'], recursive_strength)
                        found_match = True
                        break
                
                if not found_match:
                    self.mathematical_memory.append(memory_entry)
        
        # Cleanup
        if len(self.pattern_fingerprints) > self.memory_window:
            self.pattern_fingerprints.pop(0)
        if len(self.recursion_scores) > self.memory_window:
            self.recursion_scores.pop(0)
        
        return self.recursion_scores[-1] if self.recursion_scores else 0.0
    
    def compute_semantic_attractor_density(self, activations):
        """
        Compute Semantic Attractor Density - clustering of activation attractors in semantic space.
        This measures how the network organizes mathematical concepts into distinct attractors.
        """
        activation_flat = activations.flatten()
        
        if len(activation_flat) >= 2:
            # Compute activation centroid for this step
            activation_mean = torch.mean(activation_flat)
            activation_std = torch.std(activation_flat)
            activation_centroid = torch.tensor([activation_mean, activation_std, torch.max(activation_flat), torch.min(activation_flat)])
            
            self.activation_centroids.append(activation_centroid)
            
            # Compute clustering density
            if len(self.activation_centroids) >= 3:
                recent_centroids = torch.stack(self.activation_centroids[-3:])
                
                # Compute pairwise distances between centroids
                distances = []
                for i in range(len(recent_centroids)):
                    for j in range(i + 1, len(recent_centroids)):
                        dist = torch.norm(recent_centroids[i] - recent_centroids[j])
                        distances.append(safe_item(dist))
                
                if distances:
                    # Density is inverse of average distance (closer centroids = higher density)
                    avg_distance = np.mean(distances)
                    base_density = 1.0 / (1.0 + avg_distance)
                    
                    # Adjust density based on centroid stability
                    if len(self.activation_centroids) > 3:
                        prev_centroid = self.activation_centroids[-4]
                        current_centroid = self.activation_centroids[-1]
                        centroid_drift = torch.norm(current_centroid - prev_centroid)
                        stability_factor = 1.0 / (1.0 + safe_item(centroid_drift))
                        adjusted_density = base_density * 0.7 + stability_factor * 0.3
                    else:
                        adjusted_density = base_density
                else:
                    adjusted_density = 0.5
            else:
                adjusted_density = 0.5
        else:
            adjusted_density = 0.5
        
        density_val = safe_item(adjusted_density)
        self.cluster_densities.append(density_val)
        
        # Track attractor evolution (how attractors change over time)
        if len(self.cluster_densities) > 1:
            density_change = abs(self.cluster_densities[-1] - self.cluster_densities[-2])
            evolution_entry = {
                'step': len(self.cluster_densities),
                'density': density_val,
                'change_magnitude': density_change
            }
            self.attractor_evolution.append(evolution_entry)
        
        # Cleanup
        if len(self.activation_centroids) > self.memory_window:
            self.activation_centroids.pop(0)
        if len(self.cluster_densities) > self.memory_window:
            self.cluster_densities.pop(0)
        if len(self.attractor_evolution) > self.memory_window:
            self.attractor_evolution.pop(0)
        
        return density_val
    
    def track_weight_drift_entropy(self, weights):
        """
        Track Weight Drift Entropy (ΔW) - structural evolution interpretability.
        This captures how the network's structure evolves during learning.
        """
        current_weights = weights.clone().detach()
        
        # 1. Weight norm evolution
        weight_norm = torch.norm(current_weights)
        self.weight_norms.append(safe_item(weight_norm))
        
        # 2. Detailed weight change analysis
        if self.prev_weights is not None:
            # Handle dynamic network size changes
            min_rows = min(self.prev_weights.shape[0], current_weights.shape[0])
            min_cols = min(self.prev_weights.shape[1], current_weights.shape[1])
            
            if min_rows > 0 and min_cols > 0:
                prev_slice = self.prev_weights[:min_rows, :min_cols]
                curr_slice = current_weights[:min_rows, :min_cols]
                
                # Change magnitude
                change_magnitude = torch.norm(curr_slice - prev_slice)
                self.weight_changes.append(safe_item(change_magnitude))
                
                # Structural entropy (organization of weight changes)
                change_matrix = torch.abs(curr_slice - prev_slice)
                if torch.sum(change_matrix) > 1e-9:
                    change_probs = change_matrix / torch.sum(change_matrix)
                    change_entropy = -torch.sum(change_probs * torch.log(change_probs + 1e-9))
                    self.structural_entropy.append(safe_item(change_entropy))
                else:
                    self.structural_entropy.append(0.0)
                
                # Adaptation signal (combination of magnitude and entropy)
                adaptation_signal = 0.6 * safe_item(change_magnitude) + 0.4 * self.structural_entropy[-1]
                self.adaptation_signals.append(adaptation_signal)
                
                # Gradient alignment (entropy change vs weight change alignment)
                if len(self.smoothed_entropies) > 1:
                    entropy_change = self.smoothed_entropies[-1] - self.smoothed_entropies[-2]
                    weight_change = adaptation_signal - (self.adaptation_signals[-2] if len(self.adaptation_signals) > 1 else 0.0)
                    # Negative because we want entropy to decrease as weights adapt
                    alignment = -entropy_change * weight_change
                    self.gradient_alignment_history.append(alignment)
                else:
                    self.gradient_alignment_history.append(0.0)
            else:
                self.weight_changes.append(0.0)
                self.structural_entropy.append(0.0)
                self.adaptation_signals.append(0.0)
                self.gradient_alignment_history.append(0.0)
        else:
            self.weight_changes.append(0.0)
            self.structural_entropy.append(0.0)
            self.adaptation_signals.append(0.0)
            self.gradient_alignment_history.append(0.0)
        
        # Store for next iteration
        self.prev_weights = current_weights.clone()
        
        # Cleanup
        if len(self.weight_norms) > self.memory_window:
            self.weight_norms.pop(0)
        if len(self.weight_changes) > self.memory_window:
            self.weight_changes.pop(0)
        if len(self.structural_entropy) > self.memory_window:
            self.structural_entropy.pop(0)
        if len(self.adaptation_signals) > self.memory_window:
            self.adaptation_signals.pop(0)
        if len(self.gradient_alignment_history) > self.memory_window:
            self.gradient_alignment_history.pop(0)
        
        return self.adaptation_signals[-1] if self.adaptation_signals else 0.0
    
    def get_scbf_metrics(self, activations, weights):
        """
        Get comprehensive SCBF interpretability metrics for XAI analysis.
        Returns all key symbolic cognition and collapse-aware metrics.
        """
        # Compute all core metrics
        entropy_collapse = self.compute_symbolic_entropy_collapse(activations)
        ancestry_stability = self.track_activation_ancestry(activations)
        phase_alignment = self.compute_collapse_phase_alignment(activations)
        bifractal_strength = self.track_bifractal_lineage(activations)
        attractor_density = self.compute_semantic_attractor_density(activations)
        weight_drift = self.track_weight_drift_entropy(weights)
        
        # Derived metrics
        total_collapse_events = len(self.collapse_events)
        recent_collapse_magnitude = self.collapse_events[-1]['magnitude'] if self.collapse_events else 0.0
        entropy_gradient_alignment = self.gradient_alignment_history[-1] if self.gradient_alignment_history else 0.0
        
        # Enhanced mathematical interpretability metrics
        entropy_variance = safe_item(torch.var(torch.tensor(self.smoothed_entropies[-10:]))) if len(self.smoothed_entropies) >= 10 else 0.0
        pattern_consistency = np.mean(self.pattern_stability_scores[-5:]) if len(self.pattern_stability_scores) >= 5 else 1.0
        recursive_activity = np.mean(self.recursion_scores[-5:]) if len(self.recursion_scores) >= 5 else 0.0
        mathematical_memory_size = len(self.mathematical_memory)
        top_neuron_consistency = self.top_neuron_consistency[-1] if self.top_neuron_consistency else 1.0
        structural_entropy_current = self.structural_entropy[-1] if self.structural_entropy else 0.0
        
        return {
            # Core SCBF metrics
            'symbolic_entropy_collapse': entropy_collapse,
            'activation_ancestry_stability': ancestry_stability,
            'collapse_phase_alignment': phase_alignment,
            'bifractal_lineage_strength': bifractal_strength,
            'semantic_attractor_density': attractor_density,
            'weight_drift_entropy': weight_drift,
            'entropy_gradient_alignment': entropy_gradient_alignment,
            
            # Event tracking
            'total_collapse_events': total_collapse_events,
            'recent_collapse_magnitude': recent_collapse_magnitude,
            
            # Enhanced interpretability metrics
            'entropy_variance': entropy_variance,
            'pattern_consistency': pattern_consistency,
            'recursive_activity': recursive_activity,
            'mathematical_memory_size': mathematical_memory_size,
            'top_neuron_consistency': top_neuron_consistency,
            'structural_entropy': structural_entropy_current,
            'entropy_momentum': self.entropy_momentum,
            
            # Network dynamics
            'weight_norm': self.weight_norms[-1] if self.weight_norms else 0.0,
            'adaptation_signal': self.adaptation_signals[-1] if self.adaptation_signals else 0.0
        }

class HigherOrderEntropyMonitor:
    """Monitor for higher-order mathematical patterns and complexity"""
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.complexity_metric = 0.0
        self.past_metrics = []
        self.order_history = []

    def update(self, signal):
        """Compute higher-order mathematical complexity from signal patterns"""
        # Compute multiple orders of derivatives for complexity analysis
        if signal.numel() < 3:
            complexity = 0.5
        else:
            signal_flat = signal.flatten()
            
            # First and second order differences (mathematical derivatives)
            first_diff = torch.diff(signal_flat)
            if len(first_diff) > 1:
                second_diff = torch.diff(first_diff)
                # Higher-order complexity based on curvature and variation
                complexity = (torch.var(first_diff) + torch.var(second_diff)).item()
            else:
                complexity = torch.var(signal_flat).item()
        
        self.complexity_metric = self.momentum * self.complexity_metric + (1 - self.momentum) * complexity
        self.past_metrics.append(self.complexity_metric)
        self.order_history.append(len(torch.where(torch.abs(signal.flatten()) > 0.1)[0]))
        return self.complexity_metric

    def get_variance(self):
        if len(self.past_metrics) < 2:
            return 0.0
        return torch.var(torch.tensor(self.past_metrics)).item()

def higher_order_transform(pred, target, complexity_factor):
    """Apply higher-order mathematical transformation for enhanced prediction"""
    delta = torch.abs(target - pred)
    # Use mathematical complexity to guide correction strength
    correction_strength = 0.05 * (1 + complexity_factor / 10)
    correction = correction_strength * (target - pred) / (1 + delta)
    return pred + correction

def compute_mathematical_coherence(signal):
    """Compute mathematical coherence using higher-order derivatives"""
    if signal.numel() < 3:
        return torch.tensor(0.5)
    
    signal_flat = signal.flatten()
    # Compute mathematical smoothness via higher-order derivatives
    first_grad = torch.gradient(signal_flat)[0]
    if len(first_grad) > 1:
        second_grad = torch.gradient(first_grad)[0]
        # Coherence based on smoothness of higher-order derivatives
        coherence = torch.exp(-torch.mean(torch.abs(second_grad)))
    else:
        coherence = torch.exp(-torch.mean(torch.abs(first_grad)))
    
    return coherence

class CIMMInspiredController:
    """CIMM-inspired adaptive controller for stability and performance"""
    def __init__(self, min_lr=1e-4, max_lr=0.05, entropy_window=20, damping_factor=0.9):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.entropy_window = entropy_window
        self.damping_factor = damping_factor
        self.entropy_history = []
        self.lr_history = []
        self.performance_history = []
        
    def adaptive_learning_rate(self, current_lr, entropy, performance, adaptation_signal):
        """CIMM-inspired adaptive learning rate with dynamic field-aware entropy feedback"""
        self.entropy_history.append(entropy)
        self.performance_history.append(performance)
        
        # Keep history within window
        if len(self.entropy_history) > self.entropy_window:
            self.entropy_history.pop(0)
            self.performance_history.pop(0)
        
        if len(self.entropy_history) < 3:
            return current_lr
            
        # Compute entropy variance and trends
        entropy_tensor = torch.tensor(self.entropy_history)
        performance_tensor = torch.tensor(self.performance_history)
        
        entropy_variance = torch.var(entropy_tensor).item()
        entropy_trend = entropy_tensor[-1] - entropy_tensor[0]
        performance_variance = torch.var(performance_tensor).item()
        performance_trend = performance_tensor[-1] - performance_tensor[0]
        
        # Dynamic base adjustment factors (CIMM-inspired balance) - more aggressive for faster learning
        base_adjustment = 1.0
        
        # Entropy-based adjustment with dynamic scaling - increased responsiveness
        entropy_factor = 1.0 + torch.tanh(entropy_trend * 0.5).item() * 0.4
        
        # Performance-based adjustment with balance consideration - more aggressive
        performance_factor = 1.0 + (performance_trend * 0.3) - (performance_variance * 0.1)
        
        # CIMM-inspired quantum wave learning rate adjustment with dynamic amplitude - increased amplitude
        phase_shift = torch.tanh(entropy_trend * 0.2) * 0.3
        wave_amplitude = 1.0 + 0.1 * torch.cos(torch.tensor(entropy) * torch.pi)
        quantum_adjustment = wave_amplitude * torch.exp(1j * phase_shift).real
        
        # Dynamic entropy-based damping (inspired by CIMM's superfluid dynamics) - reduced damping for faster learning
        damping_strength = 2 + entropy_variance * 3  # Reduced damping strength for faster adaptation
        damping = 1.0 / (1.0 + torch.exp(-torch.abs(torch.tensor(entropy_variance)) * damping_strength))
        
        # Apply all dynamic adjustments
        new_lr = current_lr * entropy_factor * performance_factor * quantum_adjustment.item() * damping.item()
        
        # Dynamic stability constraints based on field balance - more aggressive ranges
        min_lr_factor = 0.3 + entropy_variance * 1.0  # Higher minimum when more volatile for faster adaptation
        max_lr_factor = 3.0 - performance_variance * 0.5  # Allow higher maximum, less sensitive to variance
        
        dynamic_min_lr = self.min_lr * min_lr_factor
        dynamic_max_lr = self.max_lr * max_lr_factor
        
        new_lr = torch.clamp(torch.tensor(new_lr), dynamic_min_lr, dynamic_max_lr).item()
        
        self.lr_history.append(new_lr)
        return new_lr
        
    def compute_coherence(self, signal):
        """Compute mathematical coherence using CIMM-inspired superfluid dynamics"""
        if signal.numel() < 3:
            return torch.tensor(0.5)
        
        signal_flat = signal.flatten()
        first_grad = torch.gradient(signal_flat)[0]
        if len(first_grad) > 1:
            second_grad = torch.gradient(first_grad)[0]
            coherence = torch.exp(-torch.mean(torch.abs(second_grad)))
        else:
            coherence = torch.exp(-torch.mean(torch.abs(first_grad)))
        
        return torch.clamp(coherence, 0.0, 1.0)

class MathematicalStructureController:
    """Controller for higher-order mathematical structure adaptation with dynamic, balance-based thresholds"""
    def __init__(self, base_complexity_threshold=0.1, min_neurons=6, max_neurons=128, adaptation_window=5):
        self.base_complexity_threshold = base_complexity_threshold
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.adaptation_window = adaptation_window
        self.complexity_hist = []
        self.performance_hist = []
        self.structure_hist = []
        self.cooldown_counter = 0
        self.base_cooldown_period = 10
        
        # Dynamic threshold adaptation (CIMM-inspired)
        self.complexity_balance_history = []
        self.performance_balance_history = []
        self.structure_balance_history = []

    def _compute_dynamic_thresholds(self, complexity_metric, performance, adaptation_signal):
        """Compute dynamic, balance-based thresholds like CIMM"""
        # Update balance histories
        self.complexity_balance_history.append(complexity_metric)
        self.performance_balance_history.append(performance)
        self.structure_balance_history.append(adaptation_signal)
        
        # Keep histories manageable
        max_history = 50
        if len(self.complexity_balance_history) > max_history:
            self.complexity_balance_history.pop(0)
            self.performance_balance_history.pop(0)
            self.structure_balance_history.pop(0)
        
        if len(self.complexity_balance_history) < 5:
            # Not enough history, use base thresholds
            return {
                'complexity_threshold': self.base_complexity_threshold,
                'performance_threshold': 0.005,
                'structure_threshold': 0.005,
                'cooldown_period': self.base_cooldown_period
            }
        
        # Compute field balance metrics
        complexity_tensor = torch.tensor(self.complexity_balance_history)
        performance_tensor = torch.tensor(self.performance_balance_history)
        structure_tensor = torch.tensor(self.structure_balance_history)
        
        # Dynamic complexity threshold based on field variance
        complexity_variance = torch.var(complexity_tensor[-20:]).item()
        complexity_mean = torch.mean(complexity_tensor[-10:]).item()
        dynamic_complexity_threshold = self.base_complexity_threshold * (1 + complexity_variance) * \
                                     torch.tanh(torch.tensor(complexity_mean * 10)).item()
        
        # Dynamic performance threshold based on performance stability
        performance_variance = torch.var(performance_tensor[-20:]).item()
        performance_trend = (performance_tensor[-1] - performance_tensor[-5]).item() if len(performance_tensor) >= 5 else 0
        dynamic_performance_threshold = 0.005 * (1 + performance_variance) * (1 + abs(performance_trend))
        
        # Dynamic structure threshold based on adaptation signal variance
        structure_variance = torch.var(structure_tensor[-20:]).item()
        structure_mean = torch.mean(structure_tensor[-10:]).item()
        dynamic_structure_threshold = 0.005 * (1 + structure_variance) * \
                                    torch.sigmoid(torch.tensor(structure_mean)).item()
        
        # Dynamic cooldown period based on overall field stability
        field_stability = 1.0 / (1 + complexity_variance + performance_variance + structure_variance)
        dynamic_cooldown = int(self.base_cooldown_period * (2 - field_stability))
        
        return {
            'complexity_threshold': dynamic_complexity_threshold,
            'performance_threshold': dynamic_performance_threshold,
            'structure_threshold': dynamic_structure_threshold,
            'cooldown_period': dynamic_cooldown
        }

    def decide(self, complexity_metric, performance, adaptation_signal, num_neurons):
        """Make structure adaptation decisions based on dynamic, balance-based thresholds"""
        self.complexity_hist.append(complexity_metric)
        self.performance_hist.append(performance)
        self.structure_hist.append(adaptation_signal)
        
        # Compute dynamic thresholds
        thresholds = self._compute_dynamic_thresholds(complexity_metric, performance, adaptation_signal)
        
        # Dynamic cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return "none", 0
        
        if len(self.complexity_hist) < self.adaptation_window:
            return "none", 0
            
        complexity_arr = torch.tensor(self.complexity_hist[-self.adaptation_window:])
        performance_arr = torch.tensor(self.performance_hist[-self.adaptation_window:])
        structure_arr = torch.tensor(self.structure_hist[-self.adaptation_window:])
        
        complexity_trend = complexity_arr.mean().item()
        performance_var = performance_arr.var().item()
        structure_stability = structure_arr.std().item()
        
        action = "none"
        amount = 0
        
        # CIMM-inspired dynamic adaptation with more aggressive balance-based thresholds
        if (complexity_trend > thresholds['complexity_threshold'] and  # Restored original threshold
            performance_var > thresholds['performance_threshold'] and  # Restored original threshold
            num_neurons < self.max_neurons):
            # Increase capacity for higher-order mathematical reasoning - more aggressive
            growth_factor = min(0.2, complexity_trend / thresholds['complexity_threshold'] * 0.1)  # More aggressive growth
            amount = max(2, int(num_neurons * growth_factor))  # Minimum growth of 2
            action = "grow"
            self.cooldown_counter = thresholds['cooldown_period']  # Normal cooldown
            print(f"STRUCTURE DEBUG: GROW triggered - complexity_trend={complexity_trend:.6f} > threshold={thresholds['complexity_threshold']:.6f}, performance_var={performance_var:.6f} > threshold={thresholds['performance_threshold']:.6f}")
        elif (complexity_trend < thresholds['complexity_threshold'] * 0.2 and  # More balanced threshold for pruning
              structure_stability < thresholds['structure_threshold'] and 
              num_neurons > self.min_neurons):  # Allow pruning with normal neuron count
            # Reduce capacity when mathematical complexity is low
            prune_factor = min(0.15, thresholds['structure_threshold'] / structure_stability * 0.08)  # More balanced pruning
            amount = max(1, int(num_neurons * prune_factor))  # Minimum pruning of 1
            action = "prune"
            self.cooldown_counter = thresholds['cooldown_period']  # Normal cooldown
            print(f"STRUCTURE DEBUG: PRUNE triggered - complexity_trend={complexity_trend:.6f} < threshold={thresholds['complexity_threshold'] * 0.2:.6f}, structure_stability={structure_stability:.6f} < threshold={thresholds['structure_threshold']:.6f}")
        else:
            # Log why no action was taken
            print(f"STRUCTURE DEBUG: NO ACTION - complexity_trend={complexity_trend:.6f} (threshold={thresholds['complexity_threshold']:.6f}), performance_var={performance_var:.6f} (threshold={thresholds['performance_threshold']:.6f}), structure_stability={structure_stability:.6f}, neurons={num_neurons}, cooldown={self.cooldown_counter}")
            
        return action, amount

# --- TinyCIMM-Euler: Higher-Order Mathematical Reasoning Model ---
class TinyCIMMEuler(nn.Module):
    """
    TinyCIMM-Euler: Higher-order mathematical reasoning version of TinyCIMM.
    Designed for complex mathematical pattern recognition, sequence prediction,
    and higher-order mathematical reasoning tasks like prime number prediction.
    """
    def __init__(self, input_size, hidden_size, output_size, device, **kwargs):
        super(TinyCIMMEuler, self).__init__()
        self.device = device
        self.hidden_dim = hidden_size
        self.qbe_controller = QBEController()  # Ensure QBEController is initialized
        self.adaptation_steps = kwargs.get('adaptation_steps', 20)
        self.math_memory_size = kwargs.get('math_memory_size', 10)
        self.pattern_decay = kwargs.get('pattern_decay', 0.95)
        self.micro_memory_size = kwargs.get('micro_memory_size', 5)
        self.memory_window = kwargs.get('memory_window', 30)  # For limiting memory size
        
        # Core parameters for higher-order mathematical reasoning
        self.W = nn.Parameter(0.05 * torch.randn(hidden_size, input_size, device=device))
        self.b = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.V = nn.Parameter(0.05 * torch.randn(output_size, hidden_size, device=device))
        self.c = nn.Parameter(torch.zeros(output_size, device=device))
        
        # CIMM-inspired controllers and loss
        self.cimm_controller = CIMMInspiredController()
        self.structure_controller = MathematicalStructureController()
        self.lr_controller = self.cimm_controller  # Use CIMM controller for learning rate adaptation
        # self.field_loss = CIMMInspiredLoss(lambda_qbe=0.1, lambda_entropy=0.05, lambda_coherence=0.02)  # Commented out for SCBF testing
        
        # Higher-order mathematical components
        self.complexity_monitor = None
        self.complexity_factor = 0.0
        
        # Mathematical memory system for pattern recognition
        self.math_memory = []
        
        # Higher-order reasoning layers
        self.higher_order_processor = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),  # Better for mathematical reasoning
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        ).to(device)
        
        # Optimizer tuned for mathematical learning with CIMM-inspired parameters
        self.base_lr = 0.02  # More aggressive initial learning rate for faster mathematical learning
        self.optimizer = torch.optim.Adam(
            [self.W, self.b, self.V, self.c], 
            lr=self.base_lr,
            weight_decay=0.0005,  # Reduced weight decay for faster learning
            eps=1e-8  # Better numerical stability
        )
        
        # Mathematical SCBF interpretability tracker
        self.scbf_tracker = UnifiedSymbolicCollapseTracker()
        self.symbolic_collapse_tracker = self.scbf_tracker  # Alias for consistency
        
        # Micro-memory for TinyCIMM-Planck style SCBF analysis (matches Planck implementation)
        self.micro_memory = []
        
        # Adaptation signals for tracking model adaptation over time
        self.adaptation_signals = []
        
        # State tracking for mathematical reasoning
        self.last_h = None
        self.last_x = None
        self.last_prediction = None
        self.complexity_history = []
        self.pattern_history = []
        self.structural_entropy = []  # For tracking structural entropy over time
        
        # Adaptation state
        self.prev_loss = None
        self.prev_complexity = None
        self.current_step = 0
        self.structure_stability = 0
        self.performance_metric = 0.0

        # GPU Performance Optimizations
        if torch.cuda.is_available() and device.type == 'cuda':
            # Enable tensor cores and faster operations for CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Compile model for faster execution (PyTorch 2.0+)
            try:
                print("Attempting to compile model for faster GPU execution...")
                self = torch.compile(self, mode='reduce-overhead')
                print("Model successfully compiled for GPU optimization")
            except Exception as e:
                print(f"Model compilation failed (using standard mode): {e}")

    def set_complexity_monitor(self, monitor):
        """Set the mathematical complexity monitor"""
        self.complexity_monitor = monitor

    def forward(self, x, y_true=None):
        """Forward pass with higher-order mathematical reasoning"""
        # Standard forward pass
        h = torch.relu(x @ self.W.T + self.b)
        
        # Store activations in micro_memory for SCBF analysis (TinyCIMM-Planck style)
        # Store mean across batch to maintain consistent size
        h_mean = torch.mean(h, dim=0, keepdim=True)  # Average across batch dimension
        self.micro_memory.append(h_mean.detach().clone())
        if len(self.micro_memory) > self.micro_memory_size:
            self.micro_memory.pop(0)
        
        # Higher-order mathematical processing
        higher_order_signal = self.higher_order_processor(x)
        
        # Store in mathematical memory for pattern recognition
        self.math_memory.append(h.detach().cpu() * self.pattern_decay)
        if len(self.math_memory) > self.math_memory_size:
            self.math_memory.pop(0)
        
        # Output computation
        y = h @ self.V.T + self.c
        
        # Apply higher-order mathematical correction if target is available
        if y_true is not None:
            y = higher_order_transform(y, y_true, self.complexity_factor)
        
        # Update state
        self.last_h = h
        self.last_x = x
        self.last_prediction = y.detach()
        
        # Update complexity factor
        self.complexity_factor = (self.complexity_factor + 0.01) % 1.0
        
        return y

    def log_complexity_metric(self):
        """Compute mathematical complexity metric for current state"""
        if self.W.shape[0] > 1 and torch.isfinite(self.W).all():
            # Compute complexity based on weight distribution and higher-order patterns
            weight_var = torch.var(self.W).item()
            weight_entropy = -torch.sum(F.softmax(self.W.flatten(), dim=0) * 
                                      torch.log(F.softmax(self.W.flatten(), dim=0) + 1e-9)).item()
            complexity_metric = weight_var + 0.1 * weight_entropy
        else:
            complexity_metric = 0.5
        
        self.complexity_history.append(complexity_metric)
        return complexity_metric

    def compute_field_aware_performance(self, predictions, targets):
        """
        Compute field-aware performance based on pattern recognition and structural coherence
        rather than simple distance metrics
        """
        # Simplified metrics for SCBF testing (avoiding missing dependencies)
        # cimm_metrics = compute_cimm_error_metrics(targets, predictions)
        
        # Pattern Recognition Score (based on prediction accuracy, not just divergence)
        # Lower prediction error = better pattern recognition
        prediction_accuracy = 1.0 / (1.0 + torch.mean((predictions - targets) ** 2).item())
        # Using simplified metrics without missing dependencies
        pattern_recognition_score = prediction_accuracy
        
        # Field Coherence Score (simplified)
        field_coherence_score = 1.0 / (1.0 + torch.mean((predictions - targets) ** 2).item())
        
        # Structural Complexity Score (based on entropy and network state)
        entropy_value = 0.5  # Simplified entropy value for SCBF testing
        if hasattr(self, 'W') and self.W is not None:
            weight_complexity = torch.var(self.W).item()
            structure_stability = torch.mean(torch.abs(self.W)).item()
        else:
            weight_complexity = 0.5
            structure_stability = 0.5
            
        structural_complexity_score = entropy_value * (1 + weight_complexity) * structure_stability
        
        # Higher-Order Field Dynamics (based on prediction coherence)
        if predictions.numel() >= 3:
            pred_coherence = self.cimm_controller.compute_coherence(predictions)
            higher_order_dynamics = pred_coherence.item()
        else:
            higher_order_dynamics = 0.5
            
        # Mathematical Reasoning Consistency (based on pattern stability)
        if len(self.complexity_history) >= 3:
            complexity_stability = 1.0 / (1.0 + torch.var(torch.tensor(self.complexity_history[-5:])).item())
        else:
            complexity_stability = 0.5
            
        # Quantum Field Performance (combination of all field-aware metrics)
        quantum_field_performance = (
            0.3 * pattern_recognition_score +
            0.25 * field_coherence_score +
            0.2 * structural_complexity_score +
            0.15 * higher_order_dynamics +
            0.1 * complexity_stability
        )
        
        return {
            'pattern_recognition_score': pattern_recognition_score,
            'field_coherence_score': field_coherence_score,
            'structural_complexity_score': structural_complexity_score,
            'higher_order_dynamics': higher_order_dynamics,
            'mathematical_reasoning_consistency': complexity_stability,
            'quantum_field_performance': quantum_field_performance
        }

    def compute_mathematical_performance(self):
        """Compute mathematical reasoning performance based on current state"""
        if self.last_h is None:
            return {
                'pattern_recognition_score': 0.5,
                'field_coherence_score': 0.5,
                'structural_complexity_score': 0.5,
                'higher_order_dynamics': 0.5,
                'mathematical_reasoning_consistency': 0.5,
                'quantum_field_performance': 0.5
            }
        
        # Use current hidden state as both prediction and target for performance analysis
        h_flat = self.last_h.flatten()
        
        # Create a target based on mathematical expectations
        target = torch.tanh(h_flat)  # Mathematical transformation as expected pattern
        
        return self.compute_field_aware_performance(h_flat, target)

    def mathematical_structure_adaptation(self, complexity_metric, performance, adaptation_signal, controller):
        """Adapt network structure for higher-order mathematical reasoning using field-aware signals"""
        if self.prev_loss is None:
            self.prev_loss = adaptation_signal
        if self.prev_complexity is None:
            self.prev_complexity = complexity_metric
        
        # Mathematical structure decision making using field-aware adaptation signal
        action, amount = controller.decide(complexity_metric, performance, adaptation_signal, self.hidden_dim)
        
        min_neurons = 8
        max_neurons = 256  # Larger capacity for mathematical reasoning
        
        if action == "grow" and self.hidden_dim < max_neurons:
            # Increase capacity for complex mathematical patterns
            new_dim = min(self.hidden_dim + amount, max_neurons)
            self._grow_mathematical_network(new_dim)
            # Clear memory on growth to prevent tensor size mismatch
            self.math_memory.clear()
            self.micro_memory.clear()  # Clear micro_memory to prevent SCBF tensor size mismatch
            print(f"Mathematical network growth: {self.hidden_dim} -> {new_dim} neurons")
            
        elif action == "prune" and self.hidden_dim > min_neurons:
            # Optimize structure while maintaining mathematical capability
            new_dim = max(self.hidden_dim - amount, min_neurons)
            self._prune_mathematical_network(new_dim)
            # Clear memory on pruning to prevent tensor size mismatch
            self.math_memory.clear()
            self.micro_memory.clear()  # Clear micro_memory to prevent SCBF tensor size mismatch
            print(f"Mathematical network pruning: {self.hidden_dim} -> {new_dim} neurons")
        
        self.prev_loss = adaptation_signal
        self.prev_complexity = complexity_metric
        self.current_step += 1

    def _grow_mathematical_network(self, new_dim):
        """Grow network optimized for mathematical reasoning"""
        growth = new_dim - self.hidden_dim
        
        # Expand weight matrices with mathematical initialization
        new_W = torch.zeros(new_dim, self.W.shape[1], device=self.device)
        new_W[:self.hidden_dim] = self.W.data
        # Initialize new neurons with small random values for mathematical stability
        new_W[self.hidden_dim:] = 0.01 * torch.randn(growth, self.W.shape[1], device=self.device)
        
        new_b = torch.zeros(new_dim, device=self.device)
        new_b[:self.hidden_dim] = self.b.data
        
        new_V = torch.zeros(self.V.shape[0], new_dim, device=self.device)
        new_V[:, :self.hidden_dim] = self.V.data
        new_V[:, self.hidden_dim:] = 0.01 * torch.randn(self.V.shape[0], growth, device=self.device)
        
        # Update parameters
        self.W = nn.Parameter(new_W)
        self.b = nn.Parameter(new_b)
        self.V = nn.Parameter(new_V)
        self.hidden_dim = new_dim
        
        # Clear mathematical memory to avoid size mismatches
        self.math_memory = []
        
        # Update optimizer with all parameters - keep existing learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam([self.W, self.b, self.V, self.c] + 
                                        list(self.higher_order_processor.parameters()), 
                                        lr=current_lr, weight_decay=0.0001)

    def _prune_mathematical_network(self, new_dim):
        """Prune network while preserving mathematical reasoning capacity"""
        if new_dim >= self.hidden_dim:
            return
        
        # Compute neuron importance for mathematical reasoning
        weight_importance = torch.abs(self.W).sum(dim=1)
        output_importance = torch.abs(self.V).sum(dim=0)
        
        # Combine importance metrics
        total_importance = weight_importance + output_importance
        
        # Keep the most mathematically important neurons
        _, keep_indices = torch.topk(total_importance, new_dim)
        keep_indices = torch.sort(keep_indices)[0]
        
        # Prune parameters
        self.W = nn.Parameter(self.W.data[keep_indices])
        self.b = nn.Parameter(self.b.data[keep_indices])
        self.V = nn.Parameter(self.V.data[:, keep_indices])
        self.hidden_dim = new_dim
        
        # Clear mathematical memory to avoid size mismatches
        self.math_memory = []
        
        # Update optimizer - keep existing learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam([self.W, self.b, self.V, self.c] + 
                                        list(self.higher_order_processor.parameters()), 
                                        lr=current_lr, weight_decay=0.0001)

    def analyze_mathematical_results(self):
        """Analyze experiment results using mathematical reasoning metrics"""
        if len(self.math_memory) < 2:
            return {
                "mathematical_complexity": 0.0,
                "pattern_recognition_score": 0.0,
                "reasoning_consistency": 0.0,
                "higher_order_performance": 0.0
            }

        # Mathematical Complexity Analysis
        complexity_metric = self.log_complexity_metric()

        # Pattern Recognition Score
        if len(self.math_memory) > 1:
            pattern_correlations = []
            for i in range(len(self.math_memory)-1):
                corr = torch.corrcoef(torch.stack([self.math_memory[i].flatten(), 
                                                 self.math_memory[i+1].flatten()]))[0,1]
                if torch.isfinite(corr):
                    pattern_correlations.append(corr.item())
            pattern_score = torch.tensor(pattern_correlations).mean().item() if pattern_correlations else 0.0
        else:
            pattern_score = 0.0

        # Reasoning Consistency
        reasoning_consistency = self.compute_mathematical_performance()

        # Higher-order Performance
        if len(self.complexity_history) > 1:
            complexity_history_tensor = torch.tensor(self.complexity_history[-10:])
            complexity_stability = 1.0 / (1.0 + torch.var(complexity_history_tensor).item())
        else:
            complexity_stability = 0.5

        return {
            "mathematical_complexity": complexity_metric,
            "pattern_recognition_score": pattern_score,
            "reasoning_consistency": reasoning_consistency,
            "higher_order_performance": complexity_stability
        }

    def analyze_results(self):
        """
        Analyze experiment results using TinyCIMM-Planck style SCBF metrics.
        This method provides the same metrics format as TinyCIMM-Planck for consistency.
        """
        if len(self.micro_memory) < 2:
            print("Insufficient micro_memory for analysis.")
            return {
                "activation_ancestry": None,
                "entropy_alignment": None,
                "phase_alignment": None,
                "bifractal_consistency": None,
                "attractor_density": None
            }

        # Activation Ancestry Trace - matching TinyCIMM-Planck implementation
        activation_ancestry = torch.mean(torch.stack([
            torch.cosine_similarity(mem.flatten(), self.micro_memory[-1].flatten(), dim=0) 
            for mem in self.micro_memory[:-1]
        ]))

        # Entropy Gradient Alignment Score - matching TinyCIMM-Planck implementation  
        if hasattr(self.scbf_tracker, 'smoothed_entropies') and len(self.scbf_tracker.smoothed_entropies) > 1:
            entropy_gradients = torch.tensor(self.scbf_tracker.smoothed_entropies)
            if len(entropy_gradients) > 1:
                gradient_vals = torch.diff(entropy_gradients)
                alignment_score = torch.mean(torch.abs(gradient_vals))
            else:
                alignment_score = torch.tensor(0.0)
        else:
            alignment_score = torch.tensor(0.0)

        # Collapse Phase Alignment - matching TinyCIMM-Planck implementation
        if self.last_h is not None and self.last_x is not None:
            # Ensure same shape for comparison
            if self.last_h.shape != self.last_x.shape:
                if self.last_h.numel() >= self.last_x.numel():
                    h_flat = self.last_h.flatten()[:self.last_x.numel()]
                    x_flat = self.last_x.flatten()
                else:
                    h_flat = self.last_h.flatten()
                    x_flat = self.last_x.flatten()[:self.last_h.numel()]
            else:
                h_flat = self.last_h.flatten()
                x_flat = self.last_x.flatten()
                
            phase_alignment = torch.mean(torch.abs(h_flat - x_flat))
        else:
            phase_alignment = torch.tensor(0.0)

        # Bifractal Activation Consistency - matching TinyCIMM-Planck implementation
        bifractal_consistency = torch.mean(torch.tensor([
            torch.sum(mem) for mem in self.micro_memory
        ]))

        # Semantic Attractor Density - matching TinyCIMM-Planck implementation
        attractor_density = torch.mean(torch.tensor([
            torch.norm(mem) for mem in self.micro_memory
        ]))

        # Log the metrics
        print("Activation Ancestry Trace:", activation_ancestry.item())
        print("Entropy Gradient Alignment Score:", alignment_score.item())
        print("Collapse Phase Alignment:", phase_alignment.item())
        print("Bifractal Activation Consistency:", bifractal_consistency.item())
        print("Semantic Attractor Density:", attractor_density.item())

        return {
            "activation_ancestry": activation_ancestry.item(),
            "entropy_alignment": alignment_score.item(),
            "phase_alignment": phase_alignment.item(),
            "bifractal_consistency": bifractal_consistency.item(),
            "attractor_density": attractor_density.item()
        }
    
    def online_adaptation_step(self, x_input, y_target, recent_predictions=None):
        """
        Online adaptation step with QBE-driven dynamics matching TinyCIMM-Planck style.
        Performs forward pass, loss computation, backpropagation, and dynamic adaptation.
        """
        # Forward pass
        prediction, hidden_state, activations = self.forward_with_qbe(x_input, y_target)
        
        # Simple MSE loss since CIMMInspiredLoss is not available
        mse_loss = torch.nn.functional.mse_loss(prediction, y_target)
        
        # Create loss components dict for compatibility
        loss_components = {
            'total_loss': mse_loss,
            'mse_loss': mse_loss,
            'qbe_loss': torch.tensor(0.0),
            'entropy_loss': torch.tensor(0.0),
            'coherence_loss': torch.tensor(0.0)
        }
        
        adaptation_signal = loss_components['total_loss']
        
        # Dynamic learning rate based on QBE momentum
        lr = 0.001 + 0.005 * self.qbe_controller.momentum
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        if adaptation_signal.requires_grad:
            adaptation_signal.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        
        # --- Field-aware Performance Metrics ---
        field_performance = self.calculate_field_performance(prediction, y_target)
        
        # --- Structural Adaptation ---
        if self.complexity_monitor is None:
            # Initialize complexity monitor if not set
            self.complexity_monitor = HigherOrderEntropyMonitor()
        
        complexity_metric = self.complexity_monitor.update(activations)
        self.mathematical_structure_adaptation(complexity_metric, field_performance['quantum_field_performance'], 
                                              safe_item(adaptation_signal), self.structure_controller)
        
        # Update QBE controller with latest error and entropy
        error = torch.abs(y_target - prediction).mean().item()
        entropy = self.scbf_tracker.compute_symbolic_entropy_collapse(activations)
        self.qbe_controller.update(error, entropy)
        
        return {
            "prediction": prediction,
            "adaptation_signal": safe_item(adaptation_signal),
            "complexity_metric": complexity_metric,
            "field_performance": field_performance,
            "cimm_components": loss_components,
            "learning_rate": lr,
            "qbe_status": self.qbe_controller.get_status()
        }

    def forward_with_qbe(self, x, y_true=None):
        """
        Forward pass that returns prediction, hidden_state, and activations as expected by adaptation step.
        """
        # Standard forward pass
        prediction = self.forward(x, y_true)
        
        # Return the expected tuple format
        hidden_state = self.last_h if self.last_h is not None else torch.zeros_like(prediction)
        activations = hidden_state  # Use hidden state as activations
        
        return prediction, hidden_state, activations

    def calculate_field_performance(self, prediction, y_target):
        """
        Calculate field-aware performance metrics for the current prediction.
        """
        return self.compute_field_aware_performance(prediction, y_target)
    
    def grow_network(self, new_dim):
        """
        Grow the network to the specified dimension.
        """
        self._grow_mathematical_network(new_dim)

    def prune_network(self, new_dim):
        """
        Prune the network to the specified dimension.
        """
        self._prune_mathematical_network(new_dim)
