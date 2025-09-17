"""
Structured Entropy Collapse (SEC) Dynamics using Fracton

Implements SEC theory for structure formation in infodynamic systems:
- Local entropy density calculation
- Collapse condition detection  
- Entropy redistribution and stabilization
- Integration with Fracton's entropy dispatch system
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging

try:
    from fracton.core.entropy_dispatch import EntropyDispatch
    from fracton.core.bifractal_trace import BifractalTrace
    FRACTON_AVAILABLE = True
except ImportError:
    FRACTON_AVAILABLE = False
    logging.warning("Fracton not available for SEC dynamics")

@dataclass
class SECConfig:
    """Configuration for Structured Entropy Collapse dynamics"""
    collapse_threshold: float = 0.8       # Entropy threshold for collapse
    stabilization_factor: float = 0.95    # Post-collapse entropy reduction
    memory_depth: int = 100               # History tracking depth
    force_amplification: float = 1e6      # Collapse force scaling
    entropy_floor: float = 0.01           # Minimum entropy level
    
@dataclass
class CollapseEvent:
    """Record of a collapse event for analysis"""
    time: float
    particle_indices: List[int]
    entropy_before: np.ndarray
    entropy_after: np.ndarray
    positions: np.ndarray
    force_magnitude: float
    
class SECDynamics:
    """
    Structured Entropy Collapse implementation using Fracton's architecture
    
    SEC provides structure formation mechanism through:
    1. Local entropy density calculation
    2. Collapse condition detection when entropy exceeds threshold
    3. Directed collapse forces toward entropy minima
    4. Post-collapse entropy stabilization
    """
    
    def __init__(self, config: SECConfig):
        if not FRACTON_AVAILABLE:
            logging.warning("Fracton not available - using fallback SEC implementation")
            
        self.config = config
        
        # Initialize Fracton components if available
        if FRACTON_AVAILABLE:
            self.entropy_dispatch = EntropyDispatch()
            self.bifractal_trace = BifractalTrace()
        else:
            self.entropy_dispatch = None
            self.bifractal_trace = None
        
        # State tracking
        self.collapse_history: List[CollapseEvent] = []
        self.entropy_evolution: List[float] = []
        self.structure_formation_events = 0
        
        logging.info(f"SEC initialized with threshold={config.collapse_threshold}, "
                    f"stabilization={config.stabilization_factor}")
    
    def calculate_local_entropy(self, positions: np.ndarray, masses: np.ndarray, 
                               velocities: np.ndarray) -> np.ndarray:
        """
        Calculate local entropy density at each particle location
        
        Entropy increases with:
        - Local mass density (more microstates)
        - Velocity dispersion (kinetic entropy)
        - Gravitational potential depth
        
        Args:
            positions: (N, 3) particle positions
            masses: (N,) particle masses  
            velocities: (N, 3) particle velocities
            
        Returns:
            (N,) array of local entropy densities
        """
        N = len(masses)
        entropy_density = np.zeros(N)
        
        for i in range(N):
            # Local mass density contribution
            mass_density = 0.0
            velocity_dispersion = 0.0
            gravitational_depth = 0.0
            
            for j in range(N):
                if i != j:
                    r_ij = np.linalg.norm(positions[i] - positions[j])
                    
                    # Mass density (inverse square law)
                    mass_density += masses[j] / (r_ij**2 + 1e-10)
                    
                    # Velocity dispersion contribution
                    v_rel = np.linalg.norm(velocities[i] - velocities[j])
                    velocity_dispersion += v_rel / (r_ij + 1e-10)
                    
                    # Gravitational potential depth
                    gravitational_depth += masses[j] / (r_ij + 1e-10)
            
            # Combine contributions (logarithmic scaling for stability)
            density_entropy = np.log(1 + mass_density / np.sum(masses))
            kinetic_entropy = np.log(1 + velocity_dispersion)
            potential_entropy = np.log(1 + gravitational_depth / np.sum(masses))
            
            entropy_density[i] = density_entropy + 0.1 * kinetic_entropy + 0.01 * potential_entropy
        
        # Ensure minimum entropy floor
        entropy_density = np.maximum(entropy_density, self.config.entropy_floor)
        
        # Track total entropy evolution
        self.entropy_evolution.append(np.mean(entropy_density))
        
        return entropy_density
    
    def detect_collapse_conditions(self, entropy_density: np.ndarray) -> np.ndarray:
        """
        Identify particles that meet collapse conditions
        
        Args:
            entropy_density: Local entropy at each particle
            
        Returns:
            Boolean mask indicating which particles should collapse
        """
        return entropy_density > self.config.collapse_threshold
    
    def calculate_collapse_forces(self, positions: np.ndarray, masses: np.ndarray,
                                 entropy_density: np.ndarray, 
                                 collapse_mask: np.ndarray) -> np.ndarray:
        """
        Calculate forces that drive collapse toward entropy minima
        
        Args:
            positions: Particle positions
            masses: Particle masses
            entropy_density: Local entropy field
            collapse_mask: Which particles are collapsing
            
        Returns:
            (N, 3) array of collapse forces
        """
        N = len(masses)
        collapse_forces = np.zeros((N, 3))
        
        # Find local entropy minima for each collapsing particle
        for i in np.where(collapse_mask)[0]:
            # Calculate entropy gradient around particle i
            entropy_gradient = np.zeros(3)
            
            for j in range(N):
                if i != j:
                    r_vec = positions[i] - positions[j]
                    r = np.linalg.norm(r_vec)
                    
                    if r < 1e-10:
                        continue
                        
                    r_hat = r_vec / r
                    
                    # Entropy difference drives the gradient
                    entropy_diff = entropy_density[j] - entropy_density[i]
                    
                    # Force toward lower entropy (structure formation)
                    if entropy_diff < 0:  # j has lower entropy
                        force_magnitude = abs(entropy_diff) * masses[j] / (r**2 + 1e-10)
                        entropy_gradient -= force_magnitude * r_hat
            
            # Scale by configuration parameters
            collapse_forces[i] = self.config.force_amplification * entropy_gradient
        
        return collapse_forces
    
    def execute_collapse_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one SEC collapse step
        
        Args:
            state: Current system state
            
        Returns:
            Updated state after collapse dynamics
        """
        positions = state['positions']
        velocities = state['velocities']
        masses = state['masses']
        dt = state.get('dt', 1.0)
        time = state.get('time', 0.0)
        
        # Calculate current entropy field
        entropy_density = self.calculate_local_entropy(positions, masses, velocities)
        
        # Detect collapse conditions
        collapse_mask = self.detect_collapse_conditions(entropy_density)
        
        # Initialize updated state
        new_state = state.copy()
        new_state['entropy_density'] = entropy_density
        new_state['collapse_occurred'] = np.any(collapse_mask)
        
        if np.any(collapse_mask):
            # Calculate collapse forces
            collapse_forces = self.calculate_collapse_forces(
                positions, masses, entropy_density, collapse_mask
            )
            
            # Apply forces to velocities
            new_velocities = velocities.copy()
            new_entropy = entropy_density.copy()
            
            for i in np.where(collapse_mask)[0]:
                # Apply collapse acceleration
                acceleration = collapse_forces[i] / masses[i]
                new_velocities[i] += acceleration * dt
                
                # Reduce entropy post-collapse (structure formation)
                new_entropy[i] *= self.config.stabilization_factor
            
            # Record collapse event
            collapse_event = CollapseEvent(
                time=time,
                particle_indices=np.where(collapse_mask)[0].tolist(),
                entropy_before=entropy_density[collapse_mask].copy(),
                entropy_after=new_entropy[collapse_mask].copy(),
                positions=positions[collapse_mask].copy(),
                force_magnitude=np.mean(np.linalg.norm(collapse_forces[collapse_mask], axis=1))
            )
            
            self.collapse_history.append(collapse_event)
            self.structure_formation_events += 1
            
            # Use Fracton's bifractal trace if available
            if self.bifractal_trace is not None:
                self.bifractal_trace.record_event('sec_collapse', {
                    'time': time,
                    'n_particles': len(np.where(collapse_mask)[0]),
                    'entropy_reduction': np.mean(entropy_density[collapse_mask] - new_entropy[collapse_mask]),
                    'force_magnitude': collapse_event.force_magnitude
                })
            
            # Update state
            new_state['velocities'] = new_velocities
            new_state['entropy_density'] = new_entropy
            new_state['collapse_forces'] = collapse_forces
            
            # Trim history if needed
            if len(self.collapse_history) > self.config.memory_depth:
                self.collapse_history.pop(0)
                
            logging.debug(f"SEC collapse at t={time:.2e}: {len(np.where(collapse_mask)[0])} particles")
        
        return new_state
    
    def analyze_structure_formation(self) -> Dict[str, Any]:
        """
        Analyze structure formation patterns from collapse history
        
        Returns:
            Statistical analysis of structure formation
        """
        if not self.collapse_history:
            return {'no_collapses': True}
        
        # Extract statistics
        collapse_times = [event.time for event in self.collapse_history]
        particle_counts = [len(event.particle_indices) for event in self.collapse_history]
        entropy_reductions = []
        
        for event in self.collapse_history:
            reduction = np.mean(event.entropy_before - event.entropy_after)
            entropy_reductions.append(reduction)
        
        # Use Fracton's bifractal trace analysis if available
        fracton_analysis = {}
        if self.bifractal_trace is not None:
            fracton_analysis = self.bifractal_trace.analyze_patterns('sec_collapse')
        
        return {
            'total_events': len(self.collapse_history),
            'mean_particles_per_event': np.mean(particle_counts),
            'mean_entropy_reduction': np.mean(entropy_reductions),
            'collapse_rate': len(self.collapse_history) / (collapse_times[-1] - collapse_times[0] + 1e-10),
            'structure_formation_efficiency': np.sum(entropy_reductions) / len(entropy_reductions),
            'entropy_evolution': np.array(self.entropy_evolution),
            'fracton_analysis': fracton_analysis
        }
    
    def get_current_structure_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate current structure formation metrics
        
        Args:
            state: Current system state
            
        Returns:
            Dictionary of structure metrics
        """
        if 'entropy_density' not in state:
            return {}
        
        entropy = state['entropy_density']
        positions = state['positions']
        
        # Structure concentration (how clustered are low-entropy regions)
        low_entropy_mask = entropy < np.percentile(entropy, 25)
        if np.any(low_entropy_mask):
            low_entropy_positions = positions[low_entropy_mask]
            structure_concentration = np.std(np.linalg.norm(low_entropy_positions, axis=1))
        else:
            structure_concentration = 0.0
        
        # Entropy gradient strength
        entropy_variance = np.var(entropy)
        
        # Collapse readiness (fraction near threshold)
        near_threshold = np.sum(entropy > 0.8 * self.config.collapse_threshold)
        collapse_readiness = near_threshold / len(entropy)
        
        return {
            'mean_entropy': np.mean(entropy),
            'entropy_variance': entropy_variance,
            'structure_concentration': structure_concentration,
            'collapse_readiness': collapse_readiness,
            'total_structure_events': self.structure_formation_events
        }

def test_sec_dynamics():
    """Test SEC dynamics with a simple configuration"""
    config = SECConfig(
        collapse_threshold=0.5,
        stabilization_factor=0.8,
        force_amplification=1e5
    )
    
    sec = SECDynamics(config)
    
    # Create test state with high entropy region
    N = 10
    positions = np.random.normal(0, 1e18, (N, 3))  # 0.1 kpc spread
    velocities = np.random.normal(0, 1e5, (N, 3))  # 100 km/s
    masses = np.ones(N) * 1e30  # Solar masses
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'time': 0.0,
        'dt': 3.15e13  # 1 Myr
    }
    
    print("Testing SEC dynamics...")
    
    # Run several steps
    for step in range(5):
        state = sec.execute_collapse_step(state)
        
        metrics = sec.get_current_structure_metrics(state)
        
        print(f"Step {step+1}: collapse={state.get('collapse_occurred', False)}, "
              f"mean_entropy={metrics.get('mean_entropy', 0):.3f}, "
              f"structure_events={metrics.get('total_structure_events', 0)}")
    
    # Analyze results
    analysis = sec.analyze_structure_formation()
    print(f"\nFinal analysis: {analysis.get('total_events', 0)} collapse events")
    
    return sec, state

if __name__ == "__main__":
    test_sec_dynamics()
