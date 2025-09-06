"""
GPU-Accelerated Memory Field Implementation using PyTorch and CUDA.

This module provides CUDA-accelerated versions of computationally intensive 
memory field operations for high-performance infodynamics computing.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from .memory_field import MemoryField
import math

class GPUAcceleratedMemoryField(MemoryField):
    """
    GPU-accelerated memory field with PyTorch CUDA backend.
    
    Accelerates:
    - Entropy calculations using parallel reduction
    - Pattern similarity detection using tensor operations
    - Batch memory operations
    - Large-scale field transformations
    """
    
    def __init__(self, *args, use_cuda: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.use_gpu = self.device.type == 'cuda'
        
        # Pre-allocate tensors for common operations
        if self.use_gpu:
            self._entropy_cache = torch.zeros(1, device=self.device)
            self._similarity_matrix = None
    
    def calculate_entropy_gpu(self) -> float:
        """
        GPU-accelerated entropy calculation using PyTorch.
        
        Uses parallel reduction for Shannon entropy computation.
        """
        if not self.use_gpu or len(self._content) < 100:
            # Fall back to CPU for small datasets
            return super().calculate_entropy()
        
        with self._lock:
            if not self._content:
                return self._current_entropy
            
            # Convert value counts to tensor
            value_counts = {}
            for value in self._content.values():
                value_str = str(value)
                value_counts[value_str] = value_counts.get(value_str, 0) + 1
            
            # Create probability tensor on GPU
            counts = torch.tensor(list(value_counts.values()), 
                                dtype=torch.float32, device=self.device)
            total_items = counts.sum()
            probabilities = counts / total_items
            
            # Parallel Shannon entropy calculation
            # H = -sum(p * log2(p))
            log_probs = torch.log2(probabilities + 1e-10)  # Add epsilon for numerical stability
            entropy = -(probabilities * log_probs).sum()
            
            # Normalize to 0-1 range
            max_entropy = torch.log2(total_items) if total_items > 1 else torch.tensor(1.0)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else torch.tensor(0.0)
            
            return min(normalized_entropy.item(), 0.95)
    
    def batch_similarity_detection_gpu(self, values: List[str]) -> torch.Tensor:
        """
        GPU-accelerated batch similarity detection using cosine similarity.
        
        Args:
            values: List of string values to compare
            
        Returns:
            Similarity matrix as PyTorch tensor
        """
        if not self.use_gpu or len(values) < 50:
            return None  # Fall back to CPU
        
        # Simple character-based vectorization for demo
        # In practice, you'd use more sophisticated embeddings
        max_len = max(len(v) for v in values)
        
        # Create character frequency vectors
        char_vectors = []
        for value in values:
            char_counts = torch.zeros(256, device=self.device)  # ASCII characters
            for char in value:
                char_counts[ord(char)] += 1
            char_vectors.append(char_counts / len(value))
        
        # Stack into matrix [num_values, 256]
        embedding_matrix = torch.stack(char_vectors)
        
        # Compute cosine similarity matrix
        normalized = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        return similarity_matrix
    
    def parallel_field_transform(self, transform_fn: str, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Apply transformations to field data in parallel batches.
        
        Args:
            transform_fn: Name of transformation function
            batch_size: Size of batches for parallel processing
            
        Returns:
            Dictionary of transformation results
        """
        if not self.use_gpu:
            return {}
        
        results = {}
        keys = list(self._content.keys())
        
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            batch_values = [self._content[key] for key in batch_keys]
            
            if transform_fn == 'entropy_analysis':
                # Batch entropy analysis
                batch_results = self._batch_entropy_analysis(batch_values)
                for j, key in enumerate(batch_keys):
                    results[key] = batch_results[j]
        
        return results
    
    def _batch_entropy_analysis(self, values: List[Any]) -> List[float]:
        """Batch entropy analysis for multiple values."""
        # Convert values to tensors and compute individual entropies
        entropies = []
        for value in values:
            value_str = str(value)
            char_counts = torch.zeros(256, device=self.device)
            for char in value_str:
                char_counts[ord(char)] += 1
            
            # Compute entropy of character distribution
            total_chars = len(value_str)
            if total_chars > 0:
                probs = char_counts / total_chars
                probs = probs[probs > 0]  # Remove zeros
                entropy = -(probs * torch.log2(probs)).sum()
                entropies.append(entropy.item())
            else:
                entropies.append(0.0)
        
        return entropies
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU utilization and memory statistics."""
        if not self.use_gpu:
            return {"gpu_available": False}
        
        return {
            "gpu_available": True,
            "device": str(self.device),
            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


def create_gpu_memory_field(*args, **kwargs) -> GPUAcceleratedMemoryField:
    """
    Factory function to create GPU-accelerated memory field.
    
    Automatically detects CUDA availability and falls back to CPU if needed.
    """
    return GPUAcceleratedMemoryField(*args, **kwargs)


# Utility functions for GPU acceleration setup
def check_cuda_available() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    return torch.cuda.is_available()

def get_optimal_batch_size() -> int:
    """Get optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 1000  # CPU fallback
    
    # Simple heuristic based on GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory > 8e9:  # > 8GB
        return 5000
    elif gpu_memory > 4e9:  # > 4GB
        return 2000
    else:
        return 1000

def setup_gpu_acceleration() -> Dict[str, Any]:
    """Setup GPU acceleration and return configuration."""
    config = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "recommended_batch_size": get_optimal_batch_size()
    }
    
    if config["cuda_available"]:
        config["device_properties"] = {
            "name": torch.cuda.get_device_properties(0).name,
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": torch.cuda.get_device_properties(0).major
        }
    
    return config
