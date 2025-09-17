"""
Infodynamic Gravity Implementation using Fracton

This module implements the formalized infodynamic gravity arithmetic:
- I(r) = I₀ × exp(-r/λ_c) with quantum coherence floor for dark matter
- F = -k_B T ln(2) × ∇I for Landauer-based forces
- Recursive field evolution using Fracton's engine
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from fracton import RecursiveExecutor, MemoryField, EntropyDispatcher, ExecutionContext
    FRACTON_AVAILABLE = True
except ImportError:
    FRACTON_AVAILABLE = False
    logging.warning("Fracton not available. Install with: pip install fracton")

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
SOLAR_MASS = 1.989e30  # kg
KPC_TO_METERS = 3.086e19  # meters per kiloparsec

@dataclass
class InfoGravityConfig:
    """Configuration for infodynamic gravity simulation"""
    lambda_c: float = 3.086e19  # Coherence length (1 kpc in meters) - will be mass-dependent
    T_info: float = 2.7         # Information temperature (K) - CMB temperature
    alpha_info: float = 1e-4    # Information coupling constant (increased from 1e-6)
    quantum_floor: float = 0.25 # Quantum coherence floor ratio (25% for dark matter)
    kappa: float = 1e4          # Force amplification factor
    beta_floor: float = 0.25    # Quantum floor coefficient
    gamma: float = 0.2          # Power law decay exponent
    lambda_0: float = 30 * 3.086e19  # Base coherence length (30 kpc)
    c_eff: float = 1e6          # Effective information propagation speed (1000 km/s)
    xi_damping: float = 0.01    # Velocity damping coefficient
    v_max: float = 3e7          # Maximum velocity (0.1c)
    use_gpu: bool = False       # Enable GPU acceleration via Fracton
    
class InfoGravityField:
    """
    Infodynamic gravity field implementation using Fracton's recursive engine
    
    Implements the formalized arithmetic:
    - Information coherence: I(r) = I₀ × exp(-r/λ_c) + I_quantum_floor
    - Landauer forces: F = -k_B T ln(2) × ∇I
    - Recursive evolution for global information conservation
    """
    
    def __init__(self, config: InfoGravityConfig):
        if not FRACTON_AVAILABLE:
            raise ImportError("Fracton package required. Install with: pip install fracton")
            
        self.config = config
        
        # Initialize Fracton components
        self.recursive_engine = RecursiveExecutor()
        self.memory_field = MemoryField()
        self.entropy_dispatch = EntropyDispatcher()
        
        # Derived constants
        self.landauer_factor = K_B * config.T_info * np.log(2) * config.kappa
        
        # Physical constants
        self.m_proton = 1.67e-27  # kg
        
        # State tracking
        self.coherence_history = []
        self.force_history = []
        self.density_history = []
        
        logging.info(f"InfoGravityField initialized with λ_c={config.lambda_c:.2e}m, "
                    f"T_info={config.T_info}K, α={config.alpha_info}")
    
    def calculate_coherence_matrix(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Calculate information coherence matrix I(r) between all particle pairs
        
        Uses v2.0 formulation:
        I(r) = I₀ × exp(-r/λ_c) + I_quantum
        where I_quantum = β_floor × I₀ × (1 + r/λ_c)^(-γ)
        
        Args:
            positions: (N, 3) array of particle positions in meters
            masses: (N,) array of particle masses in kg
            
        Returns:
            (N, N) symmetric matrix of information coherence values
        """
        N = len(masses)
        I_matrix = np.zeros((N, N))
        
        # Calculate mass-dependent coherence length: λ_c = λ₀ × (M_total/M_sun)^0.2
        total_mass = np.sum(masses)
        lambda_c_eff = self.config.lambda_0 * (total_mass / SOLAR_MASS)**0.2
        
        # Calculate pairwise information coherence
        for i in range(N):
            for j in range(i+1, N):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                
                # Base information content I₀ = α × (m_i × m_j)/m_p²
                I_0 = self.config.alpha_info * (masses[i] * masses[j]) / (self.m_proton**2)
                
                # Exponential decay component
                I_coherent = I_0 * np.exp(-r_ij / lambda_c_eff)
                
                # Quantum coherence floor: I_quantum = β_floor × I₀ × (1 + r/λ_c)^(-γ)
                r_normalized = r_ij / lambda_c_eff
                I_quantum_floor = (self.config.beta_floor * I_0 * 
                                 (1 + r_normalized)**(-self.config.gamma))
                
                # Total information = coherent + quantum floor
                I_matrix[i, j] = I_coherent + I_quantum_floor
                I_matrix[j, i] = I_matrix[i, j]  # Symmetric
        
        # Store in Fracton memory field for history tracking
        self.memory_field.set('coherence_matrix', I_matrix)
        self.memory_field.set('lambda_c_effective', lambda_c_eff)
        self.coherence_history.append(np.sum(I_matrix))
        
        return I_matrix
    
    def _calculate_local_densities(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Calculate local mass density at each particle position"""
        N = len(masses)
        densities = np.zeros(N)
        
        for i in range(N):
            local_mass = 0.0
            for j in range(N):
                if i != j:
                    r_ij = np.linalg.norm(positions[i] - positions[j])
                    # Use kernel density estimation with characteristic scale
                    h = 1e18  # 0.1 kpc smoothing length
                    if r_ij < 3 * h:
                        local_mass += masses[j] * np.exp(-(r_ij/h)**2)
            
            # Convert to density (mass per volume)
            volume = (4/3) * np.pi * (3 * h)**3
            densities[i] = local_mass / volume
        
        self.density_history.append(np.mean(densities))
        return densities
    
    def compute_information_forces(self, positions: np.ndarray, masses: np.ndarray, 
                                   velocities: np.ndarray) -> np.ndarray:
        """
        Compute forces from information gradients using v2.0 Landauer principle
        
        F = -κ × k_B T_eff × ln(2) × ∇I(r)
        
        Args:
            positions: (N, 3) array of particle positions
            masses: (N,) array of particle masses
            velocities: (N, 3) array of particle velocities
            
        Returns:
            (N, 3) array of forces on each particle
        """
        N = len(masses)
        forces = np.zeros((N, 3))
        
        # Get effective coherence length
        total_mass = np.sum(masses)
        lambda_c_eff = self.config.lambda_0 * (total_mass / SOLAR_MASS)**0.2
        
        # Calculate local density for effective temperature
        local_densities = self._calculate_local_densities(positions, masses)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)
                
                if r < 1e-10:  # Avoid singularity
                    continue
                    
                r_hat = r_vec / r
                r_normalized = r / lambda_c_eff
                
                # Base information content I₀ = α × (m_i × m_j)/m_p²
                I_0 = self.config.alpha_info * (masses[i] * masses[j]) / (self.m_proton**2)
                
                # Calculate information gradients (piecewise)
                I_coherent = I_0 * np.exp(-r_normalized)
                I_quantum = (self.config.beta_floor * I_0 * 
                           (1 + r_normalized)**(-self.config.gamma))
                
                if I_coherent > I_quantum:
                    # Normal regime: dI/dr = -(I₀/λ_c) × exp(-r/λ_c)
                    dI_dr = -(I_0 / lambda_c_eff) * np.exp(-r_normalized)
                else:
                    # Dark matter regime: dI/dr = -β_floor × I₀ × γ × (1 + r/λ_c)^(-γ-1) / λ_c
                    dI_dr = (-(self.config.beta_floor * I_0 * self.config.gamma) * 
                           (1 + r_normalized)**(-self.config.gamma - 1) / lambda_c_eff)
                
                # Effective temperature: T_eff = T_CMB × (1 + ρ_local/ρ_crit)^0.3
                rho_crit = 9.47e-27  # kg/m³ (critical density)
                T_eff = self.config.T_info * (1 + local_densities[i] / rho_crit)**0.3
                
                # Landauer force with amplification: F = -κ × k_B T_eff ln(2) × ∇I
                landauer_factor = self.config.kappa * K_B * T_eff * np.log(2)
                F_magnitude = -landauer_factor * dI_dr
                
                forces[i] += F_magnitude * r_hat
        
        # Store force history
        total_force = np.sum(np.linalg.norm(forces, axis=1))
        self.force_history.append(total_force)
        
        return forces
    
    def recursive_evolution_step(self, state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Single recursive evolution step using Fracton's engine
        
        Args:
            state: Current system state with positions, velocities, masses
            dt: Time step in seconds
            
        Returns:
            Updated state after evolution step
        """
        positions = state['positions']
        velocities = state['velocities']
        masses = state['masses']
        
        # Calculate current information field
        I_matrix = self.calculate_coherence_matrix(positions, masses)
        
        # Compute infodynamic forces
        info_forces = self.compute_information_forces(positions, masses, velocities)
        
        # Use Fracton's recursive engine for stable integration
        def physics_step(memory, context):
            # Calculate accelerations
            accel = info_forces / masses.reshape(-1, 1)
            
            # Update velocities with damping: v += a*dt*(1 - ξ*v²/v_max²)
            v_mag_sq = np.sum(velocities**2, axis=1)
            damping_factor = 1 - self.config.xi_damping * v_mag_sq / (self.config.v_max**2)
            damping_factor = np.maximum(damping_factor, 0.1)  # Minimum 10% of acceleration
            
            new_velocities = velocities + accel * dt * damping_factor.reshape(-1, 1)
            
            # Update positions
            new_positions = positions + new_velocities * dt
            
            return {
                'positions': new_positions,
                'velocities': new_velocities,
                'masses': masses
            }
        
        # Create execution context
        exec_context = ExecutionContext(
            entropy=0.5,
            depth=0,
            metadata={'dt': dt, 'step_type': 'infodynamic_gravity'}
        )
        
        updated_state = self.recursive_engine.execute(physics_step, self.memory_field, exec_context)
        
        # Add infodynamic metadata
        updated_state.update({
            'total_information': np.sum(I_matrix),
            'coherence_matrix': I_matrix,
            'dark_matter_fraction': self._calculate_dark_matter_fraction(I_matrix, positions, masses),
            'information_erasure_rate': self._calculate_erasure_rate(),
            'landauer_energy': self._calculate_landauer_energy(info_forces),
            'lambda_c_effective': self.memory_field.get('lambda_c_effective', self.config.lambda_0),
            'time': state.get('time', 0) + dt
        })
        
        return updated_state
    
    def _calculate_dark_matter_fraction(self, I_matrix: np.ndarray, positions: np.ndarray, masses: np.ndarray) -> float:
        """Calculate fraction of information in dark matter (quantum floor) regime"""
        total_info = np.sum(I_matrix)
        if total_info == 0:
            return 0.0
        
        # Get effective coherence length
        total_mass = np.sum(masses)
        lambda_c_eff = self.config.lambda_0 * (total_mass / SOLAR_MASS)**0.2
        
        # Count information in quantum floor regime
        N = len(masses)
        quantum_info = 0.0
        total_pairs = 0
        
        for i in range(N):
            for j in range(i+1, N):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                r_normalized = r_ij / lambda_c_eff
                
                # Base information
                I_0 = self.config.alpha_info * (masses[i] * masses[j]) / (self.m_proton**2)
                
                # Components
                I_coherent = I_0 * np.exp(-r_normalized)
                I_quantum = (self.config.beta_floor * I_0 * 
                           (1 + r_normalized)**(-self.config.gamma))
                
                # If quantum floor dominates, count it as dark matter
                if I_quantum > I_coherent:
                    quantum_info += I_matrix[i, j]
                
                total_pairs += 1
        
        return quantum_info / total_info if total_info > 0 else 0.0
    
    def _calculate_erasure_rate(self) -> float:
        """Calculate current information erasure rate dI/dt"""
        if len(self.coherence_history) < 2:
            return 0.0
            
        return self.coherence_history[-1] - self.coherence_history[-2]
    
    def _calculate_landauer_energy(self, forces: np.ndarray) -> float:
        """Calculate energy associated with Landauer forces"""
        return np.sum(np.linalg.norm(forces, axis=1)) * self.landauer_factor
    
    def validate_conservation_laws(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate key conservation laws for infodynamic gravity
        
        Returns:
            Dictionary of conservation violations (should be near zero)
        """
        positions = state['positions']
        velocities = state['velocities']
        masses = state['masses']
        
        # Energy conservation (kinetic + information potential)
        kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        info_potential = state['total_information'] * self.landauer_factor
        total_energy = kinetic_energy + info_potential
        
        # Momentum conservation
        total_momentum = np.sum(masses.reshape(-1, 1) * velocities, axis=0)
        
        # Information conservation (should decrease monotonically)
        info_conservation = state.get('information_erasure_rate', 0)
        
        return {
            'total_energy': total_energy,
            'momentum_violation': np.linalg.norm(total_momentum),
            'information_increase': max(0, info_conservation),  # Should be <= 0
            'dark_matter_fraction': state['dark_matter_fraction']
        }
    
    def get_diagnostic_data(self) -> Dict[str, Any]:
        """Get diagnostic information for analysis"""
        return {
            'coherence_history': np.array(self.coherence_history),
            'force_history': np.array(self.force_history),
            'config': self.config,
            'memory_field_keys': list(self.memory_field.keys()) if hasattr(self.memory_field, 'keys') else ['coherence_matrix']
        }

def create_two_body_test() -> Tuple[InfoGravityField, Dict[str, Any]]:
    """
    Create a simple two-body test case for validation using v2.0 parameters
    
    Returns:
        Configured InfoGravityField and initial state
    """
    config = InfoGravityConfig(
        lambda_0=30 * KPC_TO_METERS,  # 30 kpc base coherence length
        T_info=2.7,
        alpha_info=1e-4,              # Increased coupling
        beta_floor=0.25,              # 25% quantum floor
        gamma=0.2,                    # Power law decay
        kappa=1e-2                    # Much smaller force amplification
    )
    
    gravity_field = InfoGravityField(config)
    
    # Two solar masses separated by 1 kpc
    initial_state = {
        'positions': np.array([
            [0, 0, 0],
            [KPC_TO_METERS, 0, 0]
        ]),
        'velocities': np.array([
            [0, 0, 0],
            [0, 0, 0]
        ]),
        'masses': np.array([SOLAR_MASS, SOLAR_MASS]),
        'time': 0.0
    }
    
    return gravity_field, initial_state

if __name__ == "__main__":
    # Basic test
    gravity, state = create_two_body_test()
    
    print("Testing infodynamic gravity...")
    print(f"Initial separation: {np.linalg.norm(state['positions'][1] - state['positions'][0])/KPC_TO_METERS:.2f} kpc")
    
    # Evolve for a few steps
    dt = 3.15e13  # 1 Myr in seconds
    
    for step in range(5):
        state = gravity.recursive_evolution_step(state, dt)
        
        separation = np.linalg.norm(state['positions'][1] - state['positions'][0])
        
        print(f"Step {step+1}: separation={separation/KPC_TO_METERS:.3f} kpc, "
              f"total_info={state['total_information']:.2e}, "
              f"dark_matter={state['dark_matter_fraction']:.1%}")
    
    # Validate conservation
    conservation = gravity.validate_conservation_laws(state)
    print(f"\nConservation check:")
    for key, value in conservation.items():
        print(f"  {key}: {value:.2e}")
