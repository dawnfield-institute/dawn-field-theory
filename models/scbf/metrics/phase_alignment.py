"""
Phase alignment and collapse event analysis implementation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from .validation import validate_activation_data

def compute_collapse_phase_alignment(activations: Union[np.ndarray, torch.Tensor],
                                   phase_markers: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Compute phase alignment metrics measuring how activation collapses
    synchronize with learning phase transitions.
    
    Based on TinyCIMM-Euler phase transition analysis.
    
    Args:
        activations: Neural activations [timesteps, neurons]
        phase_markers: Timestep indices marking phase transitions
        
    Returns:
        Dictionary containing:
        - phase_alignment_score: Alignment between collapses and phase transitions
        - collapse_events: Detected entropy collapse events
        - phase_coherence: Coherence of collapses within phases
        - transition_strength: Strength of phase transitions
        
    Raises:
        ValueError: If activation data is missing or invalid
    """
    activations = validate_activation_data(activations, "compute_collapse_phase_alignment")
    
    if activations.ndim != 2:
        raise ValueError("compute_collapse_phase_alignment: Activation data must be 2D [timesteps, neurons]")
    
    timesteps, neurons = activations.shape
    
    if timesteps < 10:
        raise ValueError("compute_collapse_phase_alignment: Need at least 10 timesteps to detect phase alignment")
    
    # Compute entropy timeline for collapse detection
    entropy_timeline = []
    for t in range(timesteps):
        activations_t = activations[t, :]
        activations_norm = np.abs(activations_t) / (np.sum(np.abs(activations_t)) + 1e-10)
        entropy_t = -np.sum(activations_norm * np.log(activations_norm + 1e-10))
        entropy_timeline.append(entropy_t)
    
    entropy_timeline = np.array(entropy_timeline)
    
    # Detect collapse events (significant entropy drops)
    entropy_gradient = np.gradient(entropy_timeline)
    collapse_threshold = -np.std(entropy_gradient) * 2  # 2 standard deviations
    
    collapse_events = []
    for t in range(1, timesteps - 1):
        if entropy_gradient[t] < collapse_threshold:
            collapse_events.append(t)
    
    if not collapse_events:
        raise ValueError("compute_collapse_phase_alignment: No collapse events detected - activation patterns may be too stable")
    
    # If no phase markers provided, estimate them from collapse events
    if phase_markers is None:
        # Use collapse events as phase markers
        phase_markers = collapse_events
    
    if not phase_markers:
        raise ValueError("compute_collapse_phase_alignment: No phase markers provided and none could be estimated")
    
    # Compute alignment between collapse events and phase markers
    alignment_scores = []
    
    for collapse_t in collapse_events:
        # Find nearest phase marker
        distances = [abs(collapse_t - marker) for marker in phase_markers]
        min_distance = min(distances)
        
        # Alignment score: higher when collapse is closer to phase marker
        alignment_score = np.exp(-min_distance / 10.0)  # Exponential decay
        alignment_scores.append(alignment_score)
    
    phase_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0
    
    # Compute phase coherence (consistency within phases)
    phase_coherence_scores = []
    
    for i in range(len(phase_markers) - 1):
        start_phase = phase_markers[i]
        end_phase = phase_markers[i + 1]
        
        # Compute activation variance within phase
        phase_activations = activations[start_phase:end_phase, :]
        if phase_activations.shape[0] > 1:
            phase_variance = np.var(phase_activations, axis=0)
            phase_coherence = 1.0 / (1.0 + np.mean(phase_variance))
            phase_coherence_scores.append(phase_coherence)
    
    phase_coherence = np.mean(phase_coherence_scores) if phase_coherence_scores else 0.0
    
    # Compute transition strength
    transition_strengths = []
    
    for marker in phase_markers:
        if marker > 0 and marker < timesteps - 1:
            before = entropy_timeline[marker - 1]
            after = entropy_timeline[marker + 1]
            transition_strength = abs(before - after)
            transition_strengths.append(transition_strength)
    
    transition_strength = np.mean(transition_strengths) if transition_strengths else 0.0
    
    return {
        'phase_alignment_score': float(phase_alignment_score),
        'collapse_events': collapse_events,
        'phase_coherence': float(phase_coherence),
        'transition_strength': float(transition_strength),
        'entropy_timeline': entropy_timeline.tolist(),
        'phase_markers': phase_markers
    }
