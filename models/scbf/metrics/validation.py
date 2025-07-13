"""
Shared validation utilities for SCBF metrics.
"""

import numpy as np
import torch
from typing import Union

def validate_activation_data(activations: Union[np.ndarray, torch.Tensor], 
                           function_name: str) -> np.ndarray:
    """
    Validate activation data is real and properly formatted.
    
    Args:
        activations: Neural network activations
        function_name: Name of calling function for error messages
        
    Returns:
        Validated numpy array
        
    Raises:
        ValueError: If data is missing, empty, or invalid
    """
    if activations is None:
        raise ValueError(f"{function_name}: No activation data provided - cannot compute metrics without real experimental data")
    
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    
    if not isinstance(activations, np.ndarray):
        raise ValueError(f"{function_name}: Activation data must be numpy array or torch tensor, got {type(activations)}")
    
    if activations.size == 0:
        raise ValueError(f"{function_name}: Empty activation data provided - cannot compute metrics")
    
    if np.any(np.isnan(activations)) or np.any(np.isinf(activations)):
        raise ValueError(f"{function_name}: Invalid activation data contains NaN or Inf values")
    
    return activations
