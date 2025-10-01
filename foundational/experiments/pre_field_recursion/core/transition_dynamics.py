"""
Pre-Field to Field Transition Dynamics

Models the mechanism by which pre-field states crystallize into
emergent field configurations through recursive evolution.

Emergence Criteria:
1. PAC conservation achieved (residual < 10⁻¹²)
2. Curvature-residual product exceeds Ξ (1.0571)
3. Phase coherence established (variance < 0.1)

Version: 2.0
"""

import numpy as np
from typing import Optional, Tuple, List

# Handle both package and standalone imports
try:
    from .formal_definitions import PreFieldState, RecursionOperator
except ImportError:
    from formal_definitions import PreFieldState, RecursionOperator


class PreFieldTransition:
    """
    Mechanism for pre-field → field emergence
    
    Tracks the evolution of a pre-field state through recursion
    until field emergence conditions are satisfied.
    
    Critical constants from MED/SEC frameworks:
        Ξ = 1.0571 (Balance operator threshold)
        α = 1/137.036 (Fine structure constant)
        ε_PAC = 10⁻¹² (Machine precision conservation)
    """
    
    # Physical constants
    XI_CRITICAL = 1.0571      # Balance operator threshold
    ALPHA_FINE = 1/137.036     # Fine structure constant
    PAC_THRESHOLD = 1e-12      # Machine precision conservation
    PHASE_COHERENCE_THRESHOLD = 0.1  # Phase variance threshold
    
    def __init__(self, initial_state: PreFieldState, twist_rate: float = np.pi/4):
        """
        Initialize transition dynamics
        
        Args:
            initial_state: Starting pre-field configuration
            twist_rate: Möbius transformation twist rate
        """
        self.state = initial_state.copy()
        self.initial_state = initial_state.copy()
        self.history = [self.initial_state]
        self.emergence_point: Optional[int] = None
        self.recursion_op = RecursionOperator(twist_rate=twist_rate)
        
        # Tracking metrics
        self.pac_evolution = []
        self.curvature_evolution = []
        self.phase_coherence_evolution = []
        self.emergence_metric_evolution = []
    
    def evolve_until_emergence(self, max_iterations: int = 1000, 
                               verbose: bool = False) -> Tuple[bool, PreFieldState]:
        """
        Evolve pre-field state until field emergence conditions are met
        
        Args:
            max_iterations: Maximum recursion steps
            verbose: Print evolution progress
            
        Returns:
            (emerged: bool, final_state: PreFieldState)
        """
        if verbose:
            print(f"Evolving pre-field state (max {max_iterations} iterations)...")
            print(f"Target: PAC < {self.PAC_THRESHOLD:.2e}, Ξ > {self.XI_CRITICAL:.4f}, "
                  f"Phase coherence < {self.PHASE_COHERENCE_THRESHOLD}")
        
        for i in range(max_iterations):
            # Apply recursion
            self.state = self.recursion_op.apply(self.state)
            self.history.append(self.state.copy())
            
            # Track metrics
            self.pac_evolution.append(self.state.pac_residual)
            emergence_metric = self.state.compute_emergence_metric()
            self.emergence_metric_evolution.append(emergence_metric)
            
            phase_variance = self.state.compute_phase_variance()
            self.phase_coherence_evolution.append(phase_variance)
            
            # Check emergence conditions
            if self._check_emergence_criteria():
                self.emergence_point = i
                if verbose:
                    print(f"\n✓ Field emerged at iteration {i}")
                    print(f"  PAC residual: {self.state.pac_residual:.6e}")
                    print(f"  Emergence metric: {emergence_metric:.6f}")
                    print(f"  Phase variance: {phase_variance:.6f}")
                return True, self.state
            
            # Progress updates
            if verbose and (i + 1) % 100 == 0:
                print(f"  Iteration {i+1}: PAC={self.state.pac_residual:.6e}, "
                      f"Ξ={emergence_metric:.6f}, φ_var={phase_variance:.6f}")
        
        if verbose:
            print(f"\n✗ Field did not emerge within {max_iterations} iterations")
        
        return False, self.state
    
    def _check_emergence_criteria(self) -> bool:
        """
        Multi-factor emergence criteria
        
        Returns:
            True if all emergence conditions are satisfied
        """
        # Criterion 1: PAC conservation achieved
        pac_conserved = self.state.is_conserving(self.PAC_THRESHOLD)
        
        # Criterion 2: Curvature-residual product exceeds Ξ
        emergence_metric = self.state.compute_emergence_metric()
        xi_exceeded = emergence_metric > self.XI_CRITICAL
        
        # Criterion 3: Phase coherence
        phase_coherent = self._check_phase_coherence()
        
        return pac_conserved and xi_exceeded and phase_coherent
    
    def _check_phase_coherence(self) -> bool:
        """Check if phase distribution is coherent"""
        phase_variance = self.state.compute_phase_variance()
        return phase_variance < self.PHASE_COHERENCE_THRESHOLD
    
    def get_transition_metrics(self) -> dict:
        """
        Extract quantitative metrics for the transition
        
        Returns:
            Comprehensive dictionary of transition metrics
        """
        if self.emergence_point is None:
            final_pac = self.state.pac_residual
            final_emergence = self.state.compute_emergence_metric()
        else:
            final_state = self.history[self.emergence_point]
            final_pac = final_state.pac_residual
            final_emergence = final_state.compute_emergence_metric()
        
        metrics = {
            # Basic information
            'emerged': self.emergence_point is not None,
            'recursion_depth_at_emergence': self.emergence_point,
            'total_iterations': len(self.history) - 1,
            
            # Final state metrics
            'final_pac_residual': final_pac,
            'final_emergence_metric': final_emergence,
            'final_phase_variance': self.state.compute_phase_variance(),
            'final_field_energy': self.state.compute_field_energy(),
            'final_entropy': self.state.compute_information_entropy(),
            
            # Evolution metrics
            'total_twist_accumulated': self.recursion_op.twist_rate * len(self.history),
            'curvature_evolution': self.emergence_metric_evolution,
            'pac_evolution': self.pac_evolution,
            'phase_coherence_evolution': self.phase_coherence_evolution,
            
            # Convergence analysis
            'pac_convergence_rate': self._compute_convergence_rate(self.pac_evolution),
            'emergence_acceleration': self._compute_acceleration(self.emergence_metric_evolution),
            
            # Physical constants
            'xi_target': self.XI_CRITICAL,
            'alpha_fine': self.ALPHA_FINE,
            'pac_threshold': self.PAC_THRESHOLD,
        }
        
        return metrics
    
    def _compute_convergence_rate(self, values: List[float]) -> float:
        """Compute exponential convergence rate"""
        if len(values) < 10:
            return 0.0
        
        # Use first 50 values for stability
        n_points = min(50, len(values))
        x = np.arange(n_points)
        y = np.array(values[:n_points])
        
        # Avoid log of zero or negative
        y = np.maximum(y, 1e-15)
        log_y = np.log(y)
        
        # Linear fit to log values gives exponential rate
        try:
            coeffs = np.polyfit(x, log_y, 1)
            return -coeffs[0]  # Negative slope = convergence rate
        except:
            return 0.0
    
    def _compute_acceleration(self, values: List[float]) -> float:
        """Compute acceleration (second derivative)"""
        if len(values) < 3:
            return 0.0
        
        # Second derivative approximation
        second_deriv = np.gradient(np.gradient(values))
        return np.mean(np.abs(second_deriv))
    
    def compute_critical_exponents(self) -> dict:
        """
        Compute critical exponents near emergence point
        
        Analyzes scaling behavior near the transition
        """
        if self.emergence_point is None:
            return {}
        
        idx = self.emergence_point
        window = 10
        
        if idx < window or idx >= len(self.pac_evolution) - window:
            return {}
        
        # Extract values around emergence
        before_pac = self.pac_evolution[idx-window:idx]
        after_pac = self.pac_evolution[idx:idx+window]
        
        # Critical exponent for PAC (β exponent)
        try:
            if len(before_pac) > 0 and len(after_pac) > 0:
                before_rate = -np.polyfit(range(len(before_pac)), np.log(np.maximum(before_pac, 1e-15)), 1)[0]
                after_rate = -np.polyfit(range(len(after_pac)), np.log(np.maximum(after_pac, 1e-15)), 1)[0]
                
                critical_exponent = after_rate / (before_rate + 1e-10)
            else:
                critical_exponent = 1.0
        except:
            critical_exponent = 1.0
        
        return {
            'pac_critical_exponent': critical_exponent,
            'emergence_sharpness': np.mean(before_pac) / (np.mean(after_pac) + 1e-15),
            'transition_width': window * 2,
            'before_convergence': before_rate if 'before_rate' in locals() else 0.0,
            'after_convergence': after_rate if 'after_rate' in locals() else 0.0
        }
    
    def analyze_topology_role(self) -> dict:
        """
        Analyze how topology influences emergence
        
        Returns:
            Analysis of topological contributions
        """
        # Measure topological properties throughout evolution
        twist_contributions = []
        boundary_effects = []
        
        for state in self.history[::max(1, len(self.history)//20)]:  # Sample 20 points
            # Measure boundary twist
            if state.wavefunction.ndim == 1:
                boundary_twist = np.abs(state.wavefunction[-1] + state.wavefunction[0])
                twist_contributions.append(boundary_twist)
            
            # Measure global vs local structure
            global_phase = np.angle(np.sum(state.wavefunction))
            local_phases = np.angle(state.wavefunction)
            boundary_effect = np.var(local_phases - global_phase)
            boundary_effects.append(boundary_effect)
        
        return {
            'topology_type': self.state.topology,
            'twist_contribution_mean': np.mean(twist_contributions) if twist_contributions else 0.0,
            'twist_contribution_evolution': twist_contributions,
            'boundary_effect_mean': np.mean(boundary_effects) if boundary_effects else 0.0,
            'boundary_effect_evolution': boundary_effects,
            'topology_influence_score': np.std(boundary_effects) if boundary_effects else 0.0
        }
    
    def predict_emergence_iteration(self, confidence: float = 0.95) -> Tuple[Optional[int], float]:
        """
        Predict when emergence will occur based on current trajectory
        
        Args:
            confidence: Confidence level for prediction
            
        Returns:
            (predicted_iteration, confidence_score)
        """
        if len(self.pac_evolution) < 20:
            return None, 0.0
        
        # Fit exponential decay to PAC evolution
        x = np.arange(len(self.pac_evolution))
        y = np.array(self.pac_evolution)
        
        # Avoid log of zero
        y = np.maximum(y, 1e-15)
        log_y = np.log(y)
        
        try:
            # Fit: log(PAC) = a*x + b
            coeffs = np.polyfit(x, log_y, 1)
            a, b = coeffs
            
            # Predict when PAC < threshold
            # log(threshold) = a*x_pred + b
            # x_pred = (log(threshold) - b) / a
            predicted_x = (np.log(self.PAC_THRESHOLD) - b) / a
            
            # Confidence based on R² of fit
            y_pred = a * x + b
            ss_res = np.sum((log_y - y_pred)**2)
            ss_tot = np.sum((log_y - np.mean(log_y))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            return int(predicted_x), r_squared
        except:
            return None, 0.0


def run_comparative_analysis(topologies: List[str] = ['mobius', 'torus'],
                             size: int = 100,
                             max_iterations: int = 500) -> dict:
    """
    Compare emergence dynamics across different topologies
    
    Args:
        topologies: List of topology types to compare
        size: Field size
        max_iterations: Maximum iterations per topology
        
    Returns:
        Comparative analysis results
    """
    try:
        from .formal_definitions import create_initial_state
    except ImportError:
        from formal_definitions import create_initial_state
    
    results = {}
    
    for topology in topologies:
        print(f"\nAnalyzing {topology} topology...")
        
        # Create initial state
        initial_state = create_initial_state(size=size, topology=topology, seed=42)
        
        # Run transition
        transition = PreFieldTransition(initial_state)
        emerged, final_state = transition.evolve_until_emergence(
            max_iterations=max_iterations,
            verbose=True
        )
        
        # Collect metrics
        metrics = transition.get_transition_metrics()
        critical_exponents = transition.compute_critical_exponents()
        topology_analysis = transition.analyze_topology_role()
        
        results[topology] = {
            'emerged': emerged,
            'metrics': metrics,
            'critical_exponents': critical_exponents,
            'topology_analysis': topology_analysis
        }
    
    return results


if __name__ == "__main__":
    try:
        from .formal_definitions import create_initial_state
    except ImportError:
        from formal_definitions import create_initial_state
    
    print("Testing Transition Dynamics Module")
    print("=" * 60)
    
    # Create initial state
    print("\n[1] Creating initial Möbius pre-field state...")
    initial_state = create_initial_state(size=100, topology="mobius", seed=42)
    print(f"✓ Initial energy: {initial_state.compute_field_energy():.4f}")
    print(f"✓ Initial entropy: {initial_state.compute_information_entropy():.4f}")
    
    # Run transition
    print("\n[2] Running transition dynamics...")
    transition = PreFieldTransition(initial_state, twist_rate=np.pi/8)
    emerged, final_state = transition.evolve_until_emergence(
        max_iterations=500,
        verbose=True
    )
    
    # Get metrics
    print("\n[3] Analyzing transition metrics...")
    metrics = transition.get_transition_metrics()
    
    print(f"\n{'='*60}")
    print("TRANSITION ANALYSIS")
    print(f"{'='*60}")
    print(f"Emerged: {metrics['emerged']}")
    print(f"Iterations: {metrics['total_iterations']}")
    if metrics['emerged']:
        print(f"Emergence depth: {metrics['recursion_depth_at_emergence']}")
    print(f"Final PAC residual: {metrics['final_pac_residual']:.6e}")
    print(f"Final emergence metric: {metrics['final_emergence_metric']:.6f}")
    print(f"PAC convergence rate: {metrics['pac_convergence_rate']:.6f}")
    
    # Critical exponents
    if metrics['emerged']:
        print("\n[4] Computing critical exponents...")
        exponents = transition.compute_critical_exponents()
        for key, value in exponents.items():
            print(f"  {key}: {value:.4f}")
    
    # Topology analysis
    print("\n[5] Analyzing topology role...")
    topo_analysis = transition.analyze_topology_role()
    print(f"  Topology: {topo_analysis['topology_type']}")
    print(f"  Twist contribution: {topo_analysis['twist_contribution_mean']:.6f}")
    print(f"  Boundary effect: {topo_analysis['boundary_effect_mean']:.6f}")
    
    print("\n✅ Transition dynamics module functional!")
