"""
Bifractal lineage and weight evolution metrics implementation.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
from scipy import stats

def compute_bifractal_lineage(weights: Union[np.ndarray, torch.Tensor],
                            previous_weights: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
    """
    Compute bifractal lineage metrics measuring the fractal structure
    of weight evolution during learning.
    
    Based on Dawn Field bifractal analysis framework.
    
    Args:
        weights: Current network weights [layers, neurons, connections]
        previous_weights: Previous network weights for comparison (optional)
        
    Returns:
        Dictionary containing:
        - fractal_dimension: Estimated fractal dimension of weight structure
        - lineage_entropy: Entropy of weight lineage patterns
        - structural_similarity: Similarity to previous weight structure
        - complexity_measure: Overall complexity of weight organization
        
    Raises:
        ValueError: If weight data is missing or invalid
    """
    if weights is None:
        raise ValueError("compute_bifractal_lineage: No weight data provided - cannot compute metrics without real experimental data")
    
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    if not isinstance(weights, np.ndarray):
        raise ValueError("compute_bifractal_lineage: Weight data must be numpy array or torch tensor")
    
    if weights.size == 0:
        raise ValueError("compute_bifractal_lineage: Empty weight data provided")
    
    # Flatten weights for analysis
    weights_flat = weights.flatten()
    
    if np.any(np.isnan(weights_flat)) or np.any(np.isinf(weights_flat)):
        raise ValueError("compute_bifractal_lineage: Invalid weight data contains NaN or Inf values")
    
    # Compute fractal dimension using box counting
    weight_min, weight_max = np.min(weights_flat), np.max(weights_flat)
    weight_range = weight_max - weight_min
    
    if weight_range < 1e-10:
        raise ValueError("compute_bifractal_lineage: Weight values are too uniform to compute fractal dimension")
    
    # Box counting for fractal dimension
    scales = np.logspace(-3, 0, 10)  # Different box sizes
    box_counts = []
    
    for scale in scales:
        box_size = weight_range * scale
        n_boxes = int(1.0 / scale)
        
        # Count non-empty boxes
        hist, _ = np.histogram(weights_flat, bins=n_boxes, range=(weight_min, weight_max))
        non_empty_boxes = np.sum(hist > 0)
        box_counts.append(non_empty_boxes)
    
    # Fit power law to estimate fractal dimension
    log_scales = np.log(scales)
    log_counts = np.log(np.array(box_counts) + 1e-10)
    
    slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
    fractal_dimension = -slope
    
    # Compute lineage entropy
    weights_normalized = (weights_flat - weight_min) / (weight_range + 1e-10)
    hist, _ = np.histogram(weights_normalized, bins=50, density=True)
    probs = hist / (np.sum(hist) + 1e-10)
    lineage_entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Compute structural similarity with previous weights
    if previous_weights is not None:
        if isinstance(previous_weights, torch.Tensor):
            previous_weights = previous_weights.detach().cpu().numpy()
        
        if previous_weights.shape != weights.shape:
            raise ValueError("compute_bifractal_lineage: Previous weights must have same shape as current weights")
        
        prev_weights_flat = previous_weights.flatten()
        structural_similarity = np.corrcoef(weights_flat, prev_weights_flat)[0, 1]
        
        if np.isnan(structural_similarity):
            structural_similarity = 0.0
    else:
        structural_similarity = 0.0
    
    # Compute complexity measure
    complexity_measure = fractal_dimension * lineage_entropy
    
    return {
        'fractal_dimension': float(fractal_dimension),
        'lineage_entropy': float(lineage_entropy),
        'structural_similarity': float(structural_similarity),
        'complexity_measure': float(complexity_measure)
    }
