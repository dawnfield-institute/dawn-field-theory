"""
Semantic attractor density analysis implementation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from .validation import validate_activation_data

def compute_semantic_attractor_density(activations: Union[np.ndarray, torch.Tensor],
                                     concept_labels: Optional[List[str]] = None,
                                     similarity_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Compute semantic attractor density measuring how activation patterns
    cluster around stable semantic concepts.
    
    Based on Dawn Field attractor dynamics theory.
    
    Args:
        activations: Neural activations [timesteps, neurons] or [samples, neurons]
        concept_labels: Semantic labels for each activation pattern (optional)
        similarity_threshold: Threshold for considering patterns as same attractor
        
    Returns:
        Dictionary containing:
        - attractor_count: Number of detected semantic attractors
        - attractor_density: Density of activation patterns around attractors
        - attractor_stability: Stability of attractor regions
        - concept_separation: Separation between different concepts
        
    Raises:
        ValueError: If activation data is missing or invalid
    """
    activations = validate_activation_data(activations, "compute_semantic_attractor_density")
    
    if activations.ndim != 2:
        raise ValueError("compute_semantic_attractor_density: Activation data must be 2D [samples, neurons]")
    
    samples, neurons = activations.shape
    
    if samples < 3:
        raise ValueError("compute_semantic_attractor_density: Need at least 3 samples to detect attractors")
    
    # Compute pairwise similarities between activation patterns
    similarity_matrix = np.corrcoef(activations)
    
    # Detect attractors using similarity clustering
    visited = np.zeros(samples, dtype=bool)
    attractors = []
    
    for i in range(samples):
        if visited[i]:
            continue
            
        # Find all samples similar to this one
        similar_indices = np.where(similarity_matrix[i, :] >= similarity_threshold)[0]
        
        if len(similar_indices) > 1:  # At least 2 similar patterns
            attractors.append(similar_indices.tolist())
            visited[similar_indices] = True
    
    attractor_count = len(attractors)
    
    if attractor_count == 0:
        raise ValueError("compute_semantic_attractor_density: No semantic attractors detected - patterns may be too diverse")
    
    # Compute attractor density (average size of attractors)
    attractor_sizes = [len(attractor) for attractor in attractors]
    attractor_density = np.mean(attractor_sizes) / samples
    
    # Compute attractor stability (internal similarity)
    stability_scores = []
    
    for attractor in attractors:
        if len(attractor) > 1:
            attractor_activations = activations[attractor, :]
            internal_similarities = []
            
            for i in range(len(attractor)):
                for j in range(i + 1, len(attractor)):
                    sim = np.corrcoef(attractor_activations[i], attractor_activations[j])[0, 1]
                    if not np.isnan(sim):
                        internal_similarities.append(sim)
            
            if internal_similarities:
                stability_scores.append(np.mean(internal_similarities))
    
    attractor_stability = np.mean(stability_scores) if stability_scores else 0.0
    
    # Compute concept separation (distance between attractor centers)
    if attractor_count > 1:
        attractor_centers = []
        for attractor in attractors:
            center = np.mean(activations[attractor, :], axis=0)
            attractor_centers.append(center)
        
        center_distances = []
        for i in range(len(attractor_centers)):
            for j in range(i + 1, len(attractor_centers)):
                dist = np.linalg.norm(attractor_centers[i] - attractor_centers[j])
                center_distances.append(dist)
        
        concept_separation = np.mean(center_distances)
    else:
        concept_separation = 0.0
    
    return {
        'attractor_count': int(attractor_count),
        'attractor_density': float(attractor_density),
        'attractor_stability': float(attractor_stability),
        'concept_separation': float(concept_separation),
        'attractors': attractors,
        'similarity_matrix': similarity_matrix.tolist()
    }
