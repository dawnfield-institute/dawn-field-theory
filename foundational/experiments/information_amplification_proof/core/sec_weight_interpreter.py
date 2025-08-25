"""
SEC Weight Interpreter

Uses Symbolic Entropy Collapse (SEC) framework to analyze and interpret
model weights for symbolic patterns and information content.
"""

import sys
import os
import numpy as np
import json
from typing import Dict, Any, Optional, Union

# Add the models path to access SEC components
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models')
tinycimm_euler_path = os.path.join(models_path, 'TinyCIMM', 'TinyCIMM-Euler')

try:
    # Import from TinyCIMM-Euler (note the hyphen in directory name)
    if tinycimm_euler_path not in sys.path:
        sys.path.append(tinycimm_euler_path)
    from tinycimm_euler import UnifiedSymbolicCollapseTracker  # type: ignore
    SEC_AVAILABLE = True
except ImportError:
    SEC_AVAILABLE = False

from .compression_engine import CompressionEngine


class SECWeightInterpreter:
    """
    Interprets model weights using Symbolic Entropy Collapse (SEC) framework.
    
    Analyzes symbolic patterns, fractal dimensions, and information content
    encoded in neural network weight matrices.
    """
    
    def __init__(self):
        self.compression_engine = CompressionEngine()
        if SEC_AVAILABLE:
            self.sec_tracker = UnifiedSymbolicCollapseTracker()
        else:
            self.sec_tracker = None
    
    def extract_weight_tensors(self, model_object) -> np.ndarray:
        """Extract weight tensors from various model formats."""
        if isinstance(model_object, dict):
            # Handle dictionary-based models
            weights = []
            for key, value in model_object.items():
                if 'weight' in key.lower() or 'embed' in key.lower():
                    if hasattr(value, 'numpy'):
                        weights.append(value.numpy().flatten())
                    elif isinstance(value, (list, tuple)):
                        weights.extend(value)
                    elif isinstance(value, np.ndarray):
                        weights.append(value.flatten())
                    elif hasattr(value, 'flatten'):
                        weights.append(np.array(value).flatten())
            
            if weights:
                # Flatten all weights into single array
                flat_weights = []
                for w in weights:
                    if isinstance(w, (list, tuple)):
                        flat_weights.extend(w)
                    elif isinstance(w, np.ndarray):
                        flat_weights.extend(w.flatten())
                    else:
                        flat_weights.append(float(w))
                return np.array(flat_weights, dtype=np.float32)
        
        # Handle PyTorch models
        try:
            import torch
            if hasattr(model_object, 'state_dict'):
                state_dict = model_object.state_dict()
                weights = []
                for key, tensor in state_dict.items():
                    if 'weight' in key:
                        weights.append(tensor.detach().cpu().numpy().flatten())
                if weights:
                    return np.concatenate(weights)
        except ImportError:
            pass
        
        # Handle numpy arrays directly
        if isinstance(model_object, np.ndarray):
            return model_object.flatten()
        
        # Fallback: convert to numpy and flatten
        if isinstance(model_object, (list, tuple)):
            return np.array(model_object, dtype=np.float32).flatten()
        
        return np.array([0.0], dtype=np.float32)  # Minimal fallback
    
    def compute_weight_entropy_profile(self, weights: np.ndarray) -> Dict[str, float]:
        """Compute detailed entropy profile of weight distribution."""
        
        # Basic statistical measures
        weight_stats = {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'total_parameters': len(weights)
        }
        
        # Histogram-based entropy
        hist, bin_edges = np.histogram(weights, bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / np.sum(hist)  # Normalize
        histogram_entropy = -np.sum(hist * np.log2(hist))
        
        # Symbolic entropy based on weight magnitudes
        abs_weights = np.abs(weights)
        if np.sum(abs_weights) > 1e-10:
            weight_probs = abs_weights / np.sum(abs_weights)
            symbolic_entropy = -np.sum(weight_probs * np.log2(weight_probs + 1e-10))
        else:
            symbolic_entropy = 0.0
        
        # Sparsity analysis
        zero_threshold = 1e-6
        sparsity = np.sum(np.abs(weights) < zero_threshold) / len(weights)
        
        # Information content estimation
        unique_values = len(np.unique(np.round(weights, 6)))
        information_density = unique_values / len(weights)
        
        return {
            **weight_stats,
            'histogram_entropy': float(histogram_entropy),
            'symbolic_entropy': float(symbolic_entropy),
            'sparsity': float(sparsity),
            'information_density': float(information_density),
            'unique_values': unique_values
        }
    
    def analyze_weight_patterns(self, weights: np.ndarray) -> Dict[str, Any]:
        """Analyze symbolic patterns and structure in weights."""
        
        # Reshape weights to 2D for matrix analysis if possible
        if len(weights) >= 4:
            side_length = int(np.sqrt(len(weights)))
            if side_length * side_length <= len(weights):
                weight_matrix = weights[:side_length*side_length].reshape(side_length, side_length)
            else:
                # Use rectangular matrix
                rows = int(np.sqrt(len(weights)))
                cols = len(weights) // rows
                weight_matrix = weights[:rows*cols].reshape(rows, cols)
        else:
            weight_matrix = weights.reshape(-1, 1)
        
        patterns = {
            'matrix_shape': weight_matrix.shape,
            'rank_estimate': np.linalg.matrix_rank(weight_matrix) if weight_matrix.shape[0] == weight_matrix.shape[1] else None,
            'frobenius_norm': float(np.linalg.norm(weight_matrix, 'fro')),
            'condition_number': None
        }
        
        # Condition number for square matrices
        if weight_matrix.shape[0] == weight_matrix.shape[1]:
            try:
                patterns['condition_number'] = float(np.linalg.cond(weight_matrix))
            except:
                patterns['condition_number'] = None
        
        # Pattern regularity
        if weight_matrix.size > 1:
            # Measure how "structured" the weights are
            row_means = np.mean(weight_matrix, axis=1)
            col_means = np.mean(weight_matrix, axis=0)
            row_regularity = 1.0 - np.std(row_means) / (np.mean(np.abs(row_means)) + 1e-10)
            col_regularity = 1.0 - np.std(col_means) / (np.mean(np.abs(col_means)) + 1e-10)
            
            patterns['row_regularity'] = float(row_regularity)
            patterns['col_regularity'] = float(col_regularity)
        
        return patterns
    
    def compute_sec_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Compute SEC-specific metrics if available."""
        
        if not SEC_AVAILABLE or self.sec_tracker is None:
            return {
                'sec_available': False,
                'symbolic_entropy_collapse': 0.0,
                'bifractal_dimension': 0.0,
                'weight_drift_entropy': 0.0
            }
        
        try:
            # Convert to torch tensor for SEC analysis
            import torch
            weight_tensor = torch.from_numpy(weights.astype(np.float32))
            
            # Reshape to 2D matrix for SEC analysis
            if len(weight_tensor.shape) == 1:
                size = int(np.sqrt(len(weight_tensor)))
                if size * size == len(weight_tensor):
                    weight_tensor = weight_tensor.view(size, size)
                else:
                    # Make it rectangular
                    rows = int(np.sqrt(len(weight_tensor)))
                    cols = len(weight_tensor) // rows
                    weight_tensor = weight_tensor[:rows*cols].view(rows, cols)
            
            # Compute SEC metrics
            sec_metrics = self.sec_tracker.get_scbf_metrics(
                activations=weight_tensor,  # Use weights as "activations" for analysis
                weights=weight_tensor
            )
            
            return {
                'sec_available': True,
                'symbolic_entropy_collapse': sec_metrics.get('symbolic_entropy_collapse', 0.0),
                'bifractal_lineage_strength': sec_metrics.get('bifractal_lineage_strength', 0.0),
                'weight_drift_entropy': sec_metrics.get('weight_drift_entropy', 0.0),
                'semantic_attractor_density': sec_metrics.get('semantic_attractor_density', 0.0),
                'structural_entropy': sec_metrics.get('structural_entropy', 0.0)
            }
            
        except Exception as e:
            return {
                'sec_available': False,
                'sec_error': str(e),
                'symbolic_entropy_collapse': 0.0,
                'bifractal_dimension': 0.0,
                'weight_drift_entropy': 0.0
            }
    
    def interpret_model_weights(self, model_object) -> Dict[str, Any]:
        """
        Comprehensive interpretation of model weights using SEC framework.
        
        Returns detailed analysis including:
        - Weight compression metrics
        - Entropy profiles
        - Symbolic patterns
        - SEC-specific interpretations
        """
        
        # Extract weight tensors
        weights = self.extract_weight_tensors(model_object)
        
        # Compress weights to measure information content
        weight_bytes = weights.tobytes()
        weight_data_str = str(weights.tolist())  # Convert to string for compression
        weight_compression = self.compression_engine.measure_compression(weight_data_str)
        
        # Compute entropy profile
        entropy_profile = self.compute_weight_entropy_profile(weights)
        
        # Analyze patterns
        pattern_analysis = self.analyze_weight_patterns(weights)
        
        # SEC-specific metrics
        sec_metrics = self.compute_sec_metrics(weights)
        
        # Information interpretation
        total_params = len(weights)
        compressed_size = weight_compression['compressed_size']
        compression_ratio = (total_params * 4) / compressed_size if compressed_size > 0 else 0  # Assuming float32
        
        interpretation = {
            'timestamp': None,  # Will be added by the calling experiment
            'total_parameters': total_params,
            'raw_size_bytes': total_params * 4,  # float32
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'compression_algorithm': weight_compression['best_algorithm'],
            
            'entropy_profile': entropy_profile,
            'pattern_analysis': pattern_analysis,
            'sec_metrics': sec_metrics,
            
            'information_summary': {
                'effective_information_content': compressed_size,
                'redundancy_ratio': 1.0 - (compressed_size / (total_params * 4)),
                'symbolic_complexity': entropy_profile['symbolic_entropy'],
                'structural_regularity': pattern_analysis.get('row_regularity', 0.0)
            }
        }
        
        return interpretation
    
    def compare_weight_vs_output_information(self, weight_interpretation: Dict, 
                                           output_compressed_size: int) -> Dict[str, Any]:
        """
        Compare information content in weights vs output to test amplification hypothesis.
        """
        
        weight_info_content = weight_interpretation['compressed_size_bytes']
        output_info_content = output_compressed_size
        
        # Calculate if output contains more information than weights
        information_surplus = output_info_content - weight_info_content
        amplification_ratio = output_info_content / weight_info_content if weight_info_content > 0 else 0
        
        # SEC-based analysis
        weight_sec = weight_interpretation['sec_metrics']
        symbolic_emergence = weight_sec.get('symbolic_entropy_collapse', 0.0)
        
        comparison = {
            'weight_information_content': weight_info_content,
            'output_information_content': output_info_content,
            'information_surplus': information_surplus,
            'amplification_ratio': amplification_ratio,
            'amplification_detected': information_surplus > 0,
            
            'sec_analysis': {
                'weight_symbolic_entropy': symbolic_emergence,
                'emergent_complexity': amplification_ratio * symbolic_emergence,
                'information_source': 'emergent' if information_surplus > weight_info_content * 0.1 else 'encoded'
            },
            
            'interpretation': self._generate_interpretation(
                amplification_ratio, information_surplus, symbolic_emergence
            )
        }
        
        return comparison
    
    def _generate_interpretation(self, amplification_ratio: float, 
                               information_surplus: int, 
                               symbolic_entropy: float) -> str:
        """Generate human-readable interpretation of results."""
        
        if amplification_ratio > 2.0 and information_surplus > 1000:
            if symbolic_entropy > 0.5:
                return "Strong evidence of emergent information generation beyond weight encoding. High symbolic entropy suggests novel pattern creation."
            else:
                return "Significant information amplification detected, but low symbolic entropy suggests possible compression artifacts."
        elif amplification_ratio > 1.2:
            return "Moderate information amplification. Output contains more information than directly encoded in weights."
        else:
            return "Output information appears to be primarily encoded in model weights. Limited evidence of novel information generation."
