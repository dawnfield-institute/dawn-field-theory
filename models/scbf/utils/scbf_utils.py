"""
SCBF Utilities
==============

Shared utility functions for SCBF framework.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import hashlib

def normalize_activations(activations: np.ndarray) -> np.ndarray:
    """Normalize activations to [0, 1] range."""
    min_val = np.min(activations)
    max_val = np.max(activations)
    if max_val == min_val:
        return np.zeros_like(activations)
    return (activations - min_val) / (max_val - min_val)

def compute_activation_hash(activations: np.ndarray) -> str:
    """Compute a hash of activation patterns."""
    normalized = normalize_activations(activations)
    # Convert to string representation
    activation_str = np.array2string(normalized, precision=3, separator=',')
    return hashlib.md5(activation_str.encode()).hexdigest()[:8]

def safe_entropy(probabilities: np.ndarray, base: float = 2.0) -> float:
    """Compute entropy safely, handling zero probabilities."""
    # Filter out zero probabilities
    p_filtered = probabilities[probabilities > 0]
    if len(p_filtered) == 0:
        return 0.0
    
    # Compute entropy
    entropy = -np.sum(p_filtered * np.log(p_filtered)) / np.log(base)
    return entropy

def sliding_window_analysis(data: np.ndarray, window_size: int = 5) -> Dict[str, Any]:
    """Perform sliding window analysis on time series data."""
    if len(data) < window_size:
        return {'mean': np.mean(data), 'std': np.std(data), 'windows': 0}
    
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    
    windows = np.array(windows)
    
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'windows': len(windows),
        'window_means': np.mean(windows, axis=1),
        'window_stds': np.std(windows, axis=1),
        'trend': np.polyfit(range(len(windows)), np.mean(windows, axis=1), 1)[0] if len(windows) > 1 else 0.0
    }

def detect_outliers(data: np.ndarray, threshold: float = 2.0) -> Dict[str, Any]:
    """Detect outliers using z-score method."""
    if len(data) < 2:
        return {'outliers': [], 'outlier_indices': [], 'outlier_count': 0}
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return {'outliers': [], 'outlier_indices': [], 'outlier_count': 0}
    
    z_scores = np.abs((data - mean) / std)
    outlier_indices = np.where(z_scores > threshold)[0]
    outliers = data[outlier_indices]
    
    return {
        'outliers': outliers,
        'outlier_indices': outlier_indices,
        'outlier_count': len(outliers),
        'outlier_ratio': len(outliers) / len(data)
    }

def compute_fractal_dimension(data: np.ndarray, max_scales: int = 10) -> float:
    """Compute fractal dimension using box counting method."""
    if data.ndim != 1:
        data = data.flatten()
    
    # Normalize data to [0, 1] range
    data_norm = normalize_activations(data)
    
    # Create different box sizes
    scales = np.logspace(0, np.log10(len(data_norm) // 4), max_scales, dtype=int)
    scales = np.unique(scales)
    
    box_counts = []
    
    for scale in scales:
        # Divide data into boxes of given scale
        n_boxes = len(data_norm) // scale
        if n_boxes == 0:
            continue
            
        boxes = data_norm[:n_boxes * scale].reshape(n_boxes, scale)
        
        # Count non-empty boxes (boxes with variation)
        box_variations = np.std(boxes, axis=1)
        non_empty_boxes = np.sum(box_variations > 1e-10)
        
        box_counts.append(non_empty_boxes)
    
    if len(box_counts) < 2:
        return 1.0
    
    # Fit log-log relationship
    log_scales = np.log(scales[:len(box_counts)])
    log_counts = np.log(box_counts)
    
    # Compute fractal dimension
    if len(log_scales) > 1:
        slope = np.polyfit(log_scales, log_counts, 1)[0]
        fractal_dim = -slope
    else:
        fractal_dim = 1.0
    
    # Clamp to reasonable range
    fractal_dim = np.clip(fractal_dim, 0.1, 2.0)
    
    return fractal_dim

def compute_correlation_dimension(data: np.ndarray, max_distance: float = 0.1) -> float:
    """Compute correlation dimension."""
    if data.ndim != 1:
        data = data.flatten()
    
    if len(data) < 10:
        return 1.0
    
    # Normalize data
    data_norm = normalize_activations(data)
    
    # Create distance matrix
    n_points = min(len(data_norm), 100)  # Limit for computational efficiency
    indices = np.linspace(0, len(data_norm) - 1, n_points, dtype=int)
    points = data_norm[indices]
    
    distances = np.abs(points[:, np.newaxis] - points[np.newaxis, :])
    
    # Count pairs within different distance thresholds
    distance_thresholds = np.logspace(-3, np.log10(max_distance), 10)
    pair_counts = []
    
    for threshold in distance_thresholds:
        count = np.sum(distances < threshold) - n_points  # Subtract diagonal
        pair_counts.append(count)
    
    if len(pair_counts) < 2 or pair_counts[-1] == 0:
        return 1.0
    
    # Fit log-log relationship
    log_thresholds = np.log(distance_thresholds)
    log_counts = np.log(np.maximum(pair_counts, 1))
    
    # Compute correlation dimension
    if len(log_thresholds) > 1:
        slope = np.polyfit(log_thresholds, log_counts, 1)[0]
        correlation_dim = slope
    else:
        correlation_dim = 1.0
    
    # Clamp to reasonable range
    correlation_dim = np.clip(correlation_dim, 0.1, 2.0)
    
    return correlation_dim

def validate_scbf_input(data: np.ndarray, min_samples: int = 2) -> bool:
    """Validate input data for SCBF analysis."""
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    if data.ndim < 1:
        return False
    
    if data.shape[0] < min_samples:
        return False
    
    if np.all(np.isnan(data)) or np.all(np.isinf(data)):
        return False
    
    return True

def create_mock_activations(n_timesteps: int = 50, n_neurons: int = 20, 
                          pattern: str = 'random') -> np.ndarray:
    """Create mock activation data for testing."""
    np.random.seed(42)  # For reproducibility
    
    if pattern == 'random':
        return np.random.randn(n_timesteps, n_neurons)
    
    elif pattern == 'decay':
        # Decaying pattern
        base = np.random.randn(n_timesteps, n_neurons)
        decay_factor = np.exp(-np.linspace(0, 2, n_timesteps))
        return base * decay_factor[:, np.newaxis]
    
    elif pattern == 'oscillatory':
        # Oscillatory pattern
        t = np.linspace(0, 4 * np.pi, n_timesteps)
        base = np.sin(t)[:, np.newaxis]
        noise = np.random.randn(n_timesteps, n_neurons) * 0.1
        return base + noise
    
    elif pattern == 'convergent':
        # Convergent pattern
        base = np.random.randn(n_timesteps, n_neurons)
        convergence = np.linspace(1, 0.1, n_timesteps)
        return base * convergence[:, np.newaxis]
    
    else:
        return np.random.randn(n_timesteps, n_neurons)

def format_experiment_summary(summary: Dict[str, Any]) -> str:
    """Format experiment summary for display."""
    lines = [
        "SCBF Experiment Summary",
        "=" * 40,
        f"Total Steps: {summary.get('total_steps', 'N/A')}",
        f"SCBF Analyzed Steps: {summary.get('scbf_analyzed_steps', 'N/A')}",
        f"Analysis Coverage: {summary.get('analysis_coverage', 0):.1f}%",
        f"SCBF Success Rate: {summary.get('scbf_success_rate', 0):.1f}%",
        "",
        "Metrics Breakdown:",
    ]
    
    metrics = summary.get('metrics_breakdown', {})
    for metric, count in metrics.items():
        lines.append(f"  {metric}: {count} occurrences")
    
    return "\n".join(lines)
