"""
SEC Simulation-Based Auto-Tuning Engine

This module implements simulation-based parameter optimization rather than
relying on prediction functions. Each parameter evaluation runs a fast
simulation to get accurate metrics.
"""

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
import time
import logging
from typing import Dict, Any, Callable

@dataclass 
class SECParameters:
    """Container for SEC simulation parameters"""
    rho_thresh: float = 0.02
    dispersion_strength: float = 0.08
    clustering_strength: float = 0.12
    branching_bias: float = 0.03
    centroid_strength: float = 0.015

@dataclass
class SECTargetMetrics:
    """Target metrics for optimization"""
    fractal_dimension: float
    spatial_entropy: float
    density_variance: float

@dataclass
class SECOptimizationResult:
    """Result of simulation-based optimization"""
    optimal_params: SECParameters
    balance_improvement: float
    final_similarity: float
    optimization_time: float
    iterations: int
    success: bool
    method: str
    evaluation_history: list

class SECSimulationBasedTuning:
    """
    Simulation-based SEC parameter optimization.
    
    Runs actual simulations for each parameter evaluation to get
    accurate similarity scores rather than relying on prediction functions.
    """
    
    def __init__(self, simulation_func: Callable, device: torch.device = None):
        """
        Args:
            simulation_func: Function that takes SECParameters and returns metrics dict
            device: PyTorch device for GPU acceleration
        """
        self.simulation_func = simulation_func
        self.device = device or torch.device('cpu')
        self.evaluation_history = []
        self.best_similarity = 0.0
        self.best_params = None
        
    def objective_function(self, params_array: np.ndarray, target_metrics: SECTargetMetrics) -> float:
        """
        Objective function for optimization - returns negative similarity for minimization.
        
        Args:
            params_array: [rho_thresh, dispersion_strength, clustering_strength, branching_bias]
            target_metrics: Target metrics to optimize towards
            
        Returns:
            Negative similarity score (for minimization)
        """
        try:
            # Unpack and validate parameters
            rho_thresh, dispersion_strength, clustering_strength, branching_bias = params_array
            
            # Create SEC parameters with bounds checking
            sec_params = SECParameters(
                rho_thresh=max(0.01, min(0.08, rho_thresh)),
                dispersion_strength=max(0.08, min(0.25, dispersion_strength)), 
                clustering_strength=max(0.02, min(0.12, clustering_strength)),
                branching_bias=max(0.005, min(0.04, branching_bias)),
                centroid_strength=0.015  # Keep fixed for now
            )
            
            print(f"  Evaluating: rho={sec_params.rho_thresh:.3f}, disp={sec_params.dispersion_strength:.3f}, clust={sec_params.clustering_strength:.3f}, branch={sec_params.branching_bias:.3f}")
            
            # Run simulation with these parameters
            start_time = time.time()
            metrics = self.simulation_func(sec_params)
            sim_time = time.time() - start_time
            
            # Calculate similarity to target metrics
            similarity = self.calculate_similarity(metrics, target_metrics)
            
            print(f"    → Fractal: {metrics['fractal_dim']:.3f}, Entropy: {metrics['entropy']:.3f}, Density: {metrics['density_var']:.1f}")
            print(f"    → Similarity: {similarity:.3f} (time: {sim_time:.1f}s)")
            
            # Track best result
            if similarity > self.best_similarity:
                self.best_similarity = similarity
                self.best_params = sec_params
                print(f"    ★ NEW BEST SIMILARITY: {similarity:.3f}")
            
            # Store evaluation
            self.evaluation_history.append({
                'params': sec_params,
                'metrics': metrics,
                'similarity': similarity,
                'sim_time': sim_time
            })
            
            # Return negative for minimization
            return -similarity
            
        except Exception as e:
            logging.warning(f"Simulation evaluation failed: {e}")
            return 100.0  # Large penalty for failed evaluations
    
    def calculate_similarity(self, sim_metrics: Dict[str, float], 
                           target_metrics: SECTargetMetrics) -> float:
        """
        Calculate similarity between simulation and target metrics.
        
        Args:
            sim_metrics: Metrics from simulation
            target_metrics: Target metrics
            
        Returns:
            Combined similarity score (0-1, higher is better)
        """
        # Individual similarities with penalty for large deviations
        fractal_diff = abs(sim_metrics['fractal_dim'] - target_metrics.fractal_dimension)
        fractal_similarity = max(0, 1.0 - fractal_diff / target_metrics.fractal_dimension * 1.5)
        
        entropy_diff = abs(sim_metrics['entropy'] - target_metrics.spatial_entropy)
        entropy_similarity = max(0, 1.0 - entropy_diff / target_metrics.spatial_entropy * 1.2)
        
        density_diff = abs(sim_metrics['density_var'] - target_metrics.density_variance)
        density_similarity = max(0, 1.0 - density_diff / target_metrics.density_variance * 1.0)
        
        # Weighted combination - prioritize entropy and density variance (current weak points)
        combined_similarity = (
            fractal_similarity * 0.25 +
            entropy_similarity * 0.40 + 
            density_similarity * 0.35
        )
        
        return min(1.0, combined_similarity)
    
    def optimize_parameters(self, initial_params: SECParameters, target_metrics: SECTargetMetrics,
                          method: str = 'differential_evolution', max_evaluations: int = 15) -> SECOptimizationResult:
        """
        Optimize SEC parameters using simulation-based evaluation.
        
        Args:
            initial_params: Starting parameter values
            target_metrics: Target metrics to optimize towards  
            method: Optimization method
            max_evaluations: Maximum number of simulation evaluations
            
        Returns:
            SECOptimizationResult with optimal parameters
        """
        start_time = time.time()
        self.evaluation_history = []
        self.best_similarity = 0.0
        self.best_params = initial_params
        
        # Calculate initial similarity
        initial_metrics = self.simulation_func(initial_params)
        initial_similarity = self.calculate_similarity(initial_metrics, target_metrics)
        print(f"Initial similarity: {initial_similarity:.3f}")
        
        # Parameter bounds
        bounds = [
            (0.01, 0.08),    # rho_thresh
            (0.08, 0.25),    # dispersion_strength  
            (0.02, 0.12),    # clustering_strength
            (0.005, 0.04)    # branching_bias
        ]
        
        # Initial parameter array
        initial_array = [
            initial_params.rho_thresh,
            initial_params.dispersion_strength,
            initial_params.clustering_strength, 
            initial_params.branching_bias
        ]
        
        print(f"\nStarting {method} optimization with {max_evaluations} evaluations...")
        
        if method == 'differential_evolution':
            # Use differential evolution for global optimization
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(target_metrics,),
                maxiter=max_evaluations // 4,  # Generations, not individual evaluations
                popsize=4,  # Small population for limited evaluations
                seed=42,
                disp=True
            )
            success = result.success
            optimal_array = result.x
            iterations = result.nit
            
        else:
            # Use local optimization
            result = minimize(
                self.objective_function,
                initial_array,
                args=(target_metrics,),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_evaluations}
            )
            success = result.success
            optimal_array = result.x
            iterations = result.nit
        
        optimization_time = time.time() - start_time
        
        # Create optimal parameters
        optimal_params = SECParameters(
            rho_thresh=optimal_array[0],
            dispersion_strength=optimal_array[1],
            clustering_strength=optimal_array[2],
            branching_bias=optimal_array[3],
            centroid_strength=initial_params.centroid_strength
        )
        
        # Calculate final similarity and improvement
        final_similarity = self.best_similarity
        balance_improvement = (final_similarity - initial_similarity) / max(initial_similarity, 0.001) * 100
        
        print(f"\nOptimization completed:")
        print(f"  Initial similarity: {initial_similarity:.3f}")
        print(f"  Final similarity: {final_similarity:.3f}")
        print(f"  Improvement: {balance_improvement:.1f}%")
        print(f"  Evaluations: {len(self.evaluation_history)}")
        print(f"  Time: {optimization_time:.1f}s")
        
        return SECOptimizationResult(
            optimal_params=self.best_params,  # Use best observed parameters
            balance_improvement=balance_improvement,
            final_similarity=final_similarity,
            optimization_time=optimization_time,
            iterations=iterations,
            success=success and final_similarity > initial_similarity,
            method=method,
            evaluation_history=self.evaluation_history.copy()
        )
