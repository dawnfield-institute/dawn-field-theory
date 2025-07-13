"""
Activation ancestry and bifractal lineage metrics implementation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from scipy import stats
from .validation import validate_activation_data

def compute_activation_ancestry(activations: Union[np.ndarray, torch.Tensor],
                              layer_depths: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Compute activation ancestry tracking how activation patterns evolve
    and inherit from previous network states during learning.
    
    Based on bifractal lineage theory from Dawn Field interpretability framework.
    
    Args:
        activations: Neural activations [timesteps, neurons] or [layers, neurons]
        layer_depths: Depth indices for each layer (optional)
        
    Returns:
        Dictionary containing:
        - ancestry_strength: Correlation between successive activation states
        - lineage_stability: Stability of activation lineage over time
        - bifractal_dimension: Estimated bifractal dimension of activation evolution
        - inheritance_matrix: Activation inheritance relationships
        
    Raises:
        ValueError: If activation data is missing or invalid
    """
    activations = validate_activation_data(activations, "compute_activation_ancestry")
    
    if activations.ndim != 2:
        raise ValueError("compute_activation_ancestry: Activation data must be 2D [timesteps/layers, neurons]")
    
    steps, neurons = activations.shape
    
    if steps < 2:
        raise ValueError("compute_activation_ancestry: Need at least 2 timesteps/layers to compute ancestry")
    
    # Compute pairwise correlations between successive activation states
    ancestry_correlations = []
    
    for t in range(steps - 1):
        corr = np.corrcoef(activations[t], activations[t + 1])[0, 1]
        if not np.isnan(corr):
            ancestry_correlations.append(corr)
    
    if not ancestry_correlations:
        raise ValueError("compute_activation_ancestry: No valid correlations found - activation patterns may be constant")
    
    ancestry_strength = np.mean(ancestry_correlations)
    lineage_stability = np.std(ancestry_correlations)
    
    # Compute full inheritance matrix
    inheritance_matrix = np.corrcoef(activations)
    
    # Estimate bifractal dimension using correlation decay
    distances = []
    correlations = []
    
    for i in range(steps):
        for j in range(i + 1, min(i + 10, steps)):  # Local neighborhood
            dist = j - i
            corr = inheritance_matrix[i, j]
            if not np.isnan(corr):
                distances.append(dist)
                correlations.append(abs(corr))
    
    if len(distances) < 3:
        bifractal_dimension = 1.0  # Default dimension
    else:
        # Fit power law: correlation ~ distance^(-dimension)
        log_distances = np.log(np.array(distances) + 1e-10)
        log_correlations = np.log(np.array(correlations) + 1e-10)
        
        slope, _, _, _, _ = stats.linregress(log_distances, log_correlations)
        bifractal_dimension = -slope
    
    return {
        'ancestry_strength': float(ancestry_strength),
        'lineage_stability': float(lineage_stability),
        'bifractal_dimension': float(bifractal_dimension),
        'inheritance_matrix': inheritance_matrix.tolist(),
        'ancestry_correlations': ancestry_correlations
    }
