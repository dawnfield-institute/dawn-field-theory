"""
Symbolic entropy collapse metrics implementation.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
from .validation import validate_activation_data

def compute_symbolic_entropy_collapse(activations: Union[np.ndarray, torch.Tensor],
                                    timestamps: Optional[np.ndarray] = None,
                                    threshold: float = 0.1) -> Dict[str, float]:
    """
    Compute symbolic entropy collapse metrics measuring the transition from
    distributed to concentrated activation patterns during concept formation.
    
    Based on Dawn Field Theory equation: H_collapse = -Σ p_i * log(p_i) where
    p_i represents the probability mass of activation state i.
    
    Args:
        activations: Neural activations [timesteps, neurons] or [neurons]
        timestamps: Time indices for each activation step (optional)
        threshold: Minimum activation threshold for symbolic state detection
        
    Returns:
        Dictionary containing:
        - entropy_initial: Initial activation entropy
        - entropy_final: Final activation entropy  
        - collapse_magnitude: Total entropy reduction
        - collapse_rate: Rate of entropy reduction
        - symbolic_states: Number of detected symbolic states
        
    Raises:
        ValueError: If activation data is missing or invalid
    """
    activations = validate_activation_data(activations, "compute_symbolic_entropy_collapse")
    
    # Ensure 2D format [timesteps, neurons]
    if activations.ndim == 1:
        activations = activations.reshape(1, -1)
    elif activations.ndim > 2:
        raise ValueError("compute_symbolic_entropy_collapse: Activation data must be 1D or 2D")
    
    timesteps, neurons = activations.shape
    
    if timesteps < 2:
        raise ValueError("compute_symbolic_entropy_collapse: Need at least 2 timesteps to measure collapse")
    
    # Compute activation entropy at each timestep
    entropy_timeline = []
    symbolic_states_timeline = []
    
    for t in range(timesteps):
        activations_t = activations[t, :]
        
        # Normalize activations to probabilities
        activations_norm = np.abs(activations_t) / (np.sum(np.abs(activations_t)) + 1e-10)
        
        # Compute Shannon entropy
        entropy_t = -np.sum(activations_norm * np.log(activations_norm + 1e-10))
        entropy_timeline.append(entropy_t)
        
        # Count symbolic states (high-activation neurons)
        symbolic_states_t = np.sum(activations_norm > threshold)
        symbolic_states_timeline.append(symbolic_states_t)
    
    entropy_timeline = np.array(entropy_timeline)
    symbolic_states_timeline = np.array(symbolic_states_timeline)
    
    # Compute collapse metrics
    entropy_initial = entropy_timeline[0]
    entropy_final = entropy_timeline[-1]
    collapse_magnitude = entropy_initial - entropy_final
    
    # Compute collapse rate (entropy reduction per timestep)
    if timesteps > 1:
        collapse_rate = collapse_magnitude / (timesteps - 1)
    else:
        collapse_rate = 0.0
    
    # Average symbolic states
    symbolic_states_avg = np.mean(symbolic_states_timeline)
    
    return {
        'entropy_initial': float(entropy_initial),
        'entropy_final': float(entropy_final),
        'collapse_magnitude': float(collapse_magnitude),
        'collapse_rate': float(collapse_rate),
        'symbolic_states': float(symbolic_states_avg),
        'entropy_timeline': entropy_timeline.tolist(),
        'symbolic_states_timeline': symbolic_states_timeline.tolist()
    }
