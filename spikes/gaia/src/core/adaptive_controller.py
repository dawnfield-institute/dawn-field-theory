"""
GAIA Adaptive Threshold Controller
==================================

Implements RBF/QBE balance-based auto-tuning for GAIA field dynamics.
Based on proven patterns from CIMM, TinyCIMM, and dark matter SEC components.

This eliminates manual threshold tuning and provides dynamic adaptation
based on field balance metrics and entropy feedback.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
import math
from dataclasses import dataclass

# Set device for CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GAIAAdaptiveThresholds:
    """Container for dynamically computed GAIA thresholds"""
    collapse_threshold: float
    injection_energy_strength: float
    injection_info_strength: float
    diffusion_rate: float
    decay_rate: float
    field_sensitivity: float


class GAIAAdaptiveController:
    """
    Adaptive threshold controller for GAIA field dynamics using RBF/QBE balance optimization.
    
    Implements the same auto-tuning patterns used in CIMM, TinyCIMM, and dark matter SEC
    to eliminate manual threshold brittleness and provide dynamic field adaptation.
    """
    
    def __init__(self, 
                 base_collapse_threshold: float = 0.0003,
                 adaptation_window: int = 20,
                 balance_momentum: float = 0.8,
                 qbe_sensitivity: float = 0.1):
        """
        Initialize adaptive controller with base parameters
        
        Args:
            base_collapse_threshold: Initial collapse threshold
            adaptation_window: Number of timesteps for balance computation
            balance_momentum: Momentum factor for smooth adaptation
            qbe_sensitivity: Sensitivity to QBE feedback adjustments
        """
        self.base_collapse_threshold = base_collapse_threshold
        self.adaptation_window = adaptation_window
        self.balance_momentum = balance_momentum
        self.qbe_sensitivity = qbe_sensitivity
        
        # Balance history tracking (like CIMM/TinyCIMM pattern)
        self.pressure_history: List[float] = []
        self.collapse_history: List[int] = []
        self.variance_history: List[float] = []
        self.field_balance_history: List[float] = []
        
        # Current adaptive state
        self.current_thresholds = GAIAAdaptiveThresholds(
            collapse_threshold=base_collapse_threshold,
            injection_energy_strength=2.0,
            injection_info_strength=2.0,
            diffusion_rate=0.02,
            decay_rate=0.001,
            field_sensitivity=1.0
        )
        
        # QBE controller state (based on TinyCIMM pattern)
        self.qbe_momentum = 0.8
        self.qbe_error_band = 0.1
        self.energy_balance = 1.0
        
        logging.info("GAIA Adaptive Controller initialized with RBF/QBE balance tuning")
    
    def update_field_metrics(self, field_pressure: float, field_variance: float, 
                           collapse_count: int, field_balance: float) -> None:
        """Update field metrics for adaptive computation (like CIMM entropy tracking)"""
        self.pressure_history.append(field_pressure)
        self.variance_history.append(field_variance)
        self.collapse_history.append(collapse_count)
        self.field_balance_history.append(field_balance)
        
        # Maintain history window
        if len(self.pressure_history) > self.adaptation_window * 2:
            self.pressure_history = self.pressure_history[-self.adaptation_window:]
            self.variance_history = self.variance_history[-self.adaptation_window:]
            self.collapse_history = self.collapse_history[-self.adaptation_window:]
            self.field_balance_history = self.field_balance_history[-self.adaptation_window:]
    
    def compute_qbe_feedback(self) -> float:
        """Compute QBE feedback based on field balance (based on CIMM QBE pattern)"""
        if len(self.field_balance_history) < 3:
            return 0.0
        
        # Compute recent field balance trend
        recent_balance = torch.tensor(self.field_balance_history[-10:], device=device)
        balance_mean = torch.mean(recent_balance).item()
        balance_variance = torch.var(recent_balance).item()
        
        # Update QBE state (like TinyCIMM QBEController)
        self.qbe_momentum = 0.9 * self.qbe_momentum + 0.1 * abs(balance_mean - 1.0)
        self.qbe_error_band = max(0.05, min(0.2, self.qbe_error_band + 0.01 * balance_variance))
        self.energy_balance = self.qbe_momentum + self.qbe_error_band
        
        # Compute QBE feedback signal
        qbe_deviation = abs(balance_mean - 1.0)
        qbe_feedback = torch.tanh(torch.tensor(qbe_deviation * 5.0, device=device)).item()
        
        return qbe_feedback
    
    def compute_rbf_balance_score(self) -> float:
        """Compute RBF balance score (based on SEC auto-tuning engine pattern)"""
        if len(self.pressure_history) < self.adaptation_window:
            return 1.0
        
        # Get recent metrics
        recent_window = min(self.adaptation_window, len(self.pressure_history))
        pressures = torch.tensor(self.pressure_history[-recent_window:], device=device)
        variances = torch.tensor(self.variance_history[-recent_window:], device=device)
        collapses = torch.tensor(self.collapse_history[-recent_window:], dtype=torch.float32, device=device)
        
        # Target ideal state (balanced field activity)
        target_pressure = 0.001  # Ideal pressure level
        target_variance = 0.00001  # Ideal field variance
        target_collapse_rate = 0.1  # Ideal collapse frequency
        
        # RBF distance computation using Gaussian kernels (like SEC engine)
        pressure_diff = torch.mean((pressures - target_pressure) ** 2)
        variance_diff = torch.mean((variances - target_variance) ** 2)
        collapse_diff = (torch.mean(collapses) - target_collapse_rate) ** 2
        
        # Weighted RBF balance score (like SEC balance weights)
        balance_score = (0.4 * pressure_diff + 0.3 * variance_diff + 0.3 * collapse_diff).item()
        
        return balance_score
    
    def compute_dynamic_thresholds(self) -> GAIAAdaptiveThresholds:
        """
        Compute dynamic thresholds based on field balance (like TinyCIMM dynamic thresholds)
        """
        if len(self.pressure_history) < 5:
            return self.current_thresholds
        
        # Compute balance metrics
        qbe_feedback = self.compute_qbe_feedback()
        rbf_balance = self.compute_rbf_balance_score()
        
        # Field stability metrics
        pressure_tensor = torch.tensor(self.pressure_history[-self.adaptation_window:], device=device)
        variance_tensor = torch.tensor(self.variance_history[-self.adaptation_window:], device=device)
        
        pressure_variance = torch.var(pressure_tensor).item()
        variance_mean = torch.mean(variance_tensor).item()
        pressure_mean = torch.mean(pressure_tensor).item()
        
        # Dynamic collapse threshold (like CIMM adaptive scaling)
        field_stability = 1.0 / (1 + pressure_variance + variance_mean)
        
        # Adaptive collapse threshold
        if pressure_mean < 0.0001:  # Too little activity
            threshold_adjustment = 0.8  # Lower threshold to encourage collapses
        elif pressure_mean > 0.01:   # Too much activity  
            threshold_adjustment = 1.5  # Raise threshold to reduce collapses
        else:
            threshold_adjustment = 1.0 + 0.1 * qbe_feedback  # QBE-driven adjustment
        
        dynamic_collapse_threshold = self.base_collapse_threshold * threshold_adjustment * field_stability
        
        # Dynamic injection strengths (based on RBF balance)
        injection_factor = 1.0 + (rbf_balance - 1.0) * 0.2  # Scale with balance needs
        injection_factor = max(0.5, min(3.0, injection_factor))  # Safety bounds
        
        # Dynamic field evolution rates (like CIMM learning rate adaptation)
        evolution_factor = 1.0 - 0.1 * qbe_feedback  # Slower evolution when unbalanced
        evolution_factor = max(0.5, min(1.5, evolution_factor))
        
        # Apply momentum smoothing (like CIMM momentum patterns)
        momentum = self.balance_momentum
        
        new_thresholds = GAIAAdaptiveThresholds(
            collapse_threshold=momentum * self.current_thresholds.collapse_threshold + 
                             (1 - momentum) * dynamic_collapse_threshold,
            injection_energy_strength=momentum * self.current_thresholds.injection_energy_strength +
                                     (1 - momentum) * (2.0 * injection_factor),
            injection_info_strength=momentum * self.current_thresholds.injection_info_strength +
                                   (1 - momentum) * (2.0 * injection_factor),
            diffusion_rate=momentum * self.current_thresholds.diffusion_rate +
                          (1 - momentum) * (0.02 * evolution_factor),
            decay_rate=momentum * self.current_thresholds.decay_rate +
                      (1 - momentum) * (0.001 * evolution_factor),
            field_sensitivity=momentum * self.current_thresholds.field_sensitivity +
                             (1 - momentum) * (1.0 + 0.1 * qbe_feedback)
        )
        
        self.current_thresholds = new_thresholds
        return new_thresholds
    
    def get_adaptive_thresholds(self) -> GAIAAdaptiveThresholds:
        """Get current adaptive thresholds"""
        return self.current_thresholds
    
    def get_qbe_status(self) -> str:
        """Get QBE equilibrium status (like TinyCIMM QBE status)"""
        if self.energy_balance < 1.2:
            return "Near Equilibrium"
        elif self.energy_balance < 2.0:
            return "Moderate Equilibrium"
        else:
            return "Far from Equilibrium"
    
    def detect_field_pattern_type(self) -> str:
        """Detect field pattern type (like TinyCIMM pattern detection)"""
        if len(self.pressure_history) < 10:
            return "unknown"
        
        recent_pressures = torch.tensor(self.pressure_history[-10:], device=device)
        pressure_variance = torch.var(recent_pressures).item()
        
        if pressure_variance < 0.000001:
            return "convergence"
        elif pressure_variance > 0.001:
            return "chaotic"
        else:
            return "stable"
    
    def adapt_for_pattern(self, pattern_type: str) -> None:
        """Adapt thresholds based on detected pattern (like TinyCIMM pattern adaptation)"""
        if pattern_type == "convergence":
            # Field too stable - encourage much more activity (more aggressive)
            self.current_thresholds.collapse_threshold *= 0.5  # Much lower threshold
            self.current_thresholds.injection_energy_strength *= 2.0  # Much stronger injection
            self.current_thresholds.injection_info_strength *= 2.0
            logging.info(f"Convergence detected: lowered threshold to {self.current_thresholds.collapse_threshold:.6f}")
        elif pattern_type == "chaotic":
            # Field too unstable - dampen activity
            self.current_thresholds.collapse_threshold *= 1.1
            self.current_thresholds.decay_rate *= 1.05
            logging.info(f"Chaos detected: raised threshold to {self.current_thresholds.collapse_threshold:.6f}")
        # stable pattern needs no adjustment
    
    def reset_adaptation_state(self) -> None:
        """Reset adaptation state for new tests"""
        self.pressure_history.clear()
        self.collapse_history.clear()
        self.variance_history.clear()
        self.field_balance_history.clear()
        
        # Reset to base thresholds
        self.current_thresholds = GAIAAdaptiveThresholds(
            collapse_threshold=self.base_collapse_threshold,
            injection_energy_strength=2.0,
            injection_info_strength=2.0,
            diffusion_rate=0.02,
            decay_rate=0.001,
            field_sensitivity=1.0
        )
        
        self.qbe_momentum = 0.8
        self.qbe_error_band = 0.1
        self.energy_balance = 1.0
        
        logging.info("GAIA Adaptive Controller state reset")
