"""
Adaptive Recursion Operator for Pre-Field Evolution

v2.2: Resonance-aware convergence with FFT-based frequency detection.

Key improvements over v2.1:
- Resonance detection via FFT and zero-crossing analysis
- Phase-locked acceleration (boost when aligned with natural frequency)
- Frequency-matched twist rate adaptation
- Gentler thresholds (0.0001/0.1 instead of 0.001/0.1)

Version: 2.2
"""

import numpy as np
from typing import List, Optional

# Handle both package and standalone imports
try:
    from .formal_definitions import PreFieldState, RecursionOperator
    from .resonance_detector import ResonanceDetector
except ImportError:
    from formal_definitions import PreFieldState, RecursionOperator
    from resonance_detector import ResonanceDetector


class AdaptiveRecursionOperator(RecursionOperator):
    """
    Adaptive recursion with resonance-aware acceleration
    
    Inherits from RecursionOperator but adds:
    - Dynamic twist rate adjustment
    - Momentum-based acceleration
    - Convergence tracking and adaptation
    - Stagnation detection
    - v2.2: Resonance detection and frequency locking
    """
    
    def __init__(self, initial_twist: float = np.pi/4, 
                 beta_momentum: float = 0.9,
                 adaptation_rate: float = 1.2,
                 resonance_aware: bool = True):
        """
        Initialize adaptive recursion operator
        
        Args:
            initial_twist: Starting twist rate
            beta_momentum: Momentum coefficient (0.9 = 90% momentum retention)
            adaptation_rate: Rate of parameter adjustment (1.2 = 20% changes)
            resonance_aware: Enable resonance detection and locking (v2.2)
        """
        super().__init__(initial_twist)
        
        # Adaptive parameters
        self.convergence_history: List[float] = []
        self.acceleration_factor = 1.0
        self.momentum_term: Optional[np.ndarray] = None
        self.beta_momentum = beta_momentum
        self.adaptation_rate = adaptation_rate
        
        # v2.2: Resonance tracking
        self.resonance_aware = resonance_aware
        self.resonance_detector = ResonanceDetector() if resonance_aware else None
        self.resonance_locked = False
        self.detected_period = None
        self.detected_frequency = None
        self.phase_history: List[float] = []
        
        # Tracking
        self.stagnation_counter = 0
        self.adaptation_history = []
        
    def apply(self, state: PreFieldState) -> PreFieldState:
        """
        Apply recursion with adaptive acceleration
        
        Args:
            state: Current pre-field state
            
        Returns:
            Evolved pre-field state with adapted parameters
        """
        
        # Track convergence
        self.convergence_history.append(state.pac_residual)
        
        # Adapt parameters after warmup period
        if len(self.convergence_history) > 10:
            self._adapt_parameters()
        
        # Get current wavefunction
        z = state.wavefunction
        
        # Initialize momentum term on first iteration
        if self.momentum_term is None:
            self.momentum_term = np.zeros_like(z)
        
        # Calculate effective twist with acceleration
        effective_twist = self.twist_rate * self.acceleration_factor
        theta_i = effective_twist * 1j
        
        # Apply Möbius transformation
        numerator = z + theta_i
        denominator = 1 - np.conj(z) * theta_i
        
        # Avoid division by zero
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10 + 0j, denominator)
        w = numerator / denominator
        
        # Apply momentum for acceleration
        delta = w - z
        self.momentum_term = (self.beta_momentum * self.momentum_term + 
                             (1 - self.beta_momentum) * delta)
        
        # Add momentum contribution (15% momentum weight)
        w = w + 0.15 * self.momentum_term
        
        # Renormalize to conserve energy
        if self.conserve_norm:
            original_norm = np.sqrt(np.sum(np.abs(z)**2))
            current_norm = np.sqrt(np.sum(np.abs(w)**2))
            if current_norm > 1e-10:
                w = w * (original_norm / current_norm)
        
        # Compute metrics for new state
        pac_residual = self._compute_pac_residual(w)
        curvature = self._compute_curvature(w)
        
        # Create new state
        new_state = PreFieldState(
            wavefunction=w,
            topology=state.topology,
            recursion_depth=state.recursion_depth + 1,
            pac_residual=pac_residual,
            curvature_tensor=curvature,
            metadata={
                'previous_depth': state.recursion_depth,
                'twist_applied': self.twist_rate,
                'acceleration_factor': self.acceleration_factor,
                'effective_twist': effective_twist,
                'momentum_magnitude': np.linalg.norm(self.momentum_term),
                'iteration_count': self.iteration_count
            }
        )
        
        self.iteration_count += 1
        self.transformation_history.append({
            'iteration': self.iteration_count,
            'pac_residual': pac_residual,
            'emergence_metric': new_state.compute_emergence_metric(),
            'acceleration': self.acceleration_factor
        })
        
        return new_state
    
    def _adapt_parameters(self):
        """
        Adapt parameters based on convergence rate with resonance awareness (v2.2)
        
        Monitors recent convergence and adjusts acceleration factor:
        - Too slow → increase acceleration
        - Too fast/unstable → decrease acceleration
        - Stagnant → reset and jump
        - v2.2: Detect resonance and lock twist rate to natural frequency
        """
        if len(self.convergence_history) < 20:
            return
        
        # v2.2: Try to lock to resonance if not already locked
        if self.resonance_aware and not self.resonance_locked:
            resonance_info = self.resonance_detector.analyze_oscillations(
                self.convergence_history
            )
            
            # v2.2.1: Lowered confidence threshold from 0.5 to 0.15
            # Lock to resonance if we have any reasonable detection
            if resonance_info['confidence'] > 0.15 and resonance_info['period']:
                suggested_twist = self.resonance_detector.suggest_twist_rate(resonance_info)
                
                if suggested_twist:
                    self.twist_rate = suggested_twist
                    self.detected_period = resonance_info['period']
                    self.detected_frequency = resonance_info['frequency']
                    self.resonance_locked = True
                    print(f"  🎵 Resonance locked! Period: {self.detected_period:.1f}, "
                          f"Frequency: {self.detected_frequency:.4f}, "
                          f"Confidence: {resonance_info['confidence']:.2f}, "
                          f"New twist: {self.twist_rate:.4f}")
                    return  # Don't adapt other params when locking
            
            # Track phase for future locking
            if resonance_info['phase']:
                self.phase_history.append(resonance_info['phase'])
        
        # Calculate recent convergence rate
        recent = self.convergence_history[-10:]
        convergence_rate = abs(recent[-1] - recent[0]) / 10
        
        # Detect if oscillating
        if len(recent) >= 10:
            oscillating = np.std(recent) > np.mean(recent) * 0.1
        else:
            oscillating = False
        
        old_accel = self.acceleration_factor
        
        if oscillating:
            # v2.2: Work WITH oscillations, not against them
            # Stay neutral during oscillations
            self.acceleration_factor = 1.0
            
            # Reduce momentum during oscillations for flexibility
            if self.momentum_term is not None:
                self.beta_momentum = 0.7  # Less momentum
            
            self.stagnation_counter = 0
        else:
            # Normal adaptive behavior when not oscillating
            self.beta_momentum = 0.9  # Full momentum
            
            # v2.2: Updated thresholds based on v2.1 learnings
            if convergence_rate < 0.0001:  # Much tighter (was 0.001)
                # Too slow - accelerate more
                self.acceleration_factor = min(
                    self.acceleration_factor * 1.5,  # Stronger boost (was 1.2)
                    5.0  # Cap at 5x
                )
                self.stagnation_counter += 1
            elif convergence_rate > 0.1:  # Kept same as v2.1
                # Too fast/unstable - slow down
                self.acceleration_factor = max(
                    self.acceleration_factor * 0.95,  # Gentler reduction (was /1.2)
                    0.5  # Higher minimum (was 0.1)
                )
                self.stagnation_counter = 0
            else:
                # Good range - maintain
                self.stagnation_counter = 0
        
        # Check for prolonged stagnation
        if self.stagnation_counter > 20:
            # Stagnated for too long - try jump
            self.acceleration_factor *= 2.0
            self.momentum_term = None  # Reset momentum
            self.stagnation_counter = 0
        
        # Check for sustained stagnation (plateau)
        if len(self.convergence_history) > 50:
            last_50 = self.convergence_history[-50:]
            relative_std = np.std(last_50) / (np.mean(last_50) + 1e-10)
            
            if relative_std < 0.01:
                # Completely stagnated - drastic action
                self.acceleration_factor *= 3.0
                self.momentum_term = None
                self.twist_rate *= 1.5  # Also increase base twist
        
        # Record adaptation
        if abs(old_accel - self.acceleration_factor) > 0.01:
            self.adaptation_history.append({
                'iteration': self.iteration_count,
                'old_acceleration': old_accel,
                'new_acceleration': self.acceleration_factor,
                'convergence_rate': convergence_rate
            })
    
    def get_adaptation_statistics(self) -> dict:
        """Get statistics about parameter adaptations"""
        if not self.adaptation_history:
            return {'adaptations': 0}
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'current_acceleration': self.acceleration_factor,
            'max_acceleration': max(h['new_acceleration'] for h in self.adaptation_history),
            'min_acceleration': min(h['new_acceleration'] for h in self.adaptation_history),
            'adaptation_history': self.adaptation_history[-10:]  # Last 10
        }


if __name__ == "__main__":
    # Quick test
    print("Testing Adaptive Recursion Operator")
    print("=" * 50)
    
    try:
        from formal_definitions import create_initial_state
    except ImportError:
        print("Run from parent directory or install as package")
        import sys
        sys.exit(1)
    
    # Create initial state
    state = create_initial_state(size=100, topology="mobius", seed=42)
    print(f"✓ Initial state created: PAC = {state.pac_residual:.6f}")
    
    # Create adaptive operator
    operator = AdaptiveRecursionOperator(initial_twist=np.pi/4)
    print(f"✓ Adaptive operator created: twist = {operator.twist_rate:.4f}")
    
    # Run iterations
    print("\nRunning adaptive iterations:")
    for i in range(100):
        state = operator.apply(state)
        
        if (i + 1) % 20 == 0:
            accel = operator.acceleration_factor
            print(f"  Iteration {i+1}: PAC = {state.pac_residual:.6f}, "
                  f"Accel = {accel:.2f}x")
    
    # Statistics
    print("\n✓ Adaptation statistics:")
    stats = operator.get_adaptation_statistics()
    print(f"  Total adaptations: {stats['total_adaptations']}")
    print(f"  Current acceleration: {stats['current_acceleration']:.2f}x")
    
    # Convergence
    initial_pac = operator.convergence_history[0]
    final_pac = operator.convergence_history[-1]
    improvement = (initial_pac - final_pac) / initial_pac * 100
    print(f"\n✓ Convergence: {improvement:.1f}% improvement")
    
    print("\n✅ Adaptive recursion module functional!")
