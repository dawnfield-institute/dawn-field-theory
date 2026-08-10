"""
Quantum Mechanics via PAC Conservation

Implements quantum mechanical phenomena through PAC conservation
principles, providing a unified framework where quantum effects
emerge from universal PAC dynamics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
try:
    from typing import Complex
except ImportError:
    # Complex type is not available in older Python versions
    Complex = complex
from dataclasses import dataclass
from enum import Enum
import math

class QuantumState(Enum):
    """Quantum state types"""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    COHERENT = "coherent"
    MIXED = "mixed"

@dataclass
class QuantumPACResult:
    """Result of quantum PAC operation"""
    state_vector: torch.Tensor
    probability_amplitudes: torch.Tensor
    entanglement_measure: float
    coherence_time: float
    conservation_quality: float
    quantum_phase: float

class QuantumPACModule:
    """
    Quantum mechanics through PAC conservation.
    
    Implements quantum phenomena as emergent properties of
    PAC conservation at microscopic scales.
    """
    
    def __init__(self, 
                 hbar: float = 1.0545718e-34,
                 device: str = "auto"):
        self.hbar = hbar
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Quantum PAC parameters
        self.quantum_scale = 1e-15  # Scale at which quantum effects dominate
        self.decoherence_rate = 0.01
        self.entanglement_threshold = 0.1
        
        # Conservation tracking
        self.probability_conservation_tolerance = 1e-12
        
    def evolve_quantum_pac_state(self, 
                                state_vector: torch.Tensor,
                                hamiltonian: torch.Tensor,
                                dt: float,
                                pac_constraints: Optional[Dict] = None) -> QuantumPACResult:
        """
        Evolve quantum state under PAC conservation.
        
        Args:
            state_vector: Current quantum state vector
            hamiltonian: Hamiltonian operator
            dt: Time step
            pac_constraints: PAC conservation constraints
            
        Returns:
            QuantumPACResult with evolved state
        """
        state_vector = state_vector.to(self.device).to(torch.complex128)
        hamiltonian = hamiltonian.to(self.device).to(torch.complex128)
        
        # Standard Schrödinger evolution
        unitary = torch.matrix_exp(-1j * hamiltonian * dt / self.hbar)
        evolved_state = torch.matmul(unitary, state_vector)
        
        # Apply PAC conservation constraints
        if pac_constraints:
            evolved_state = self._apply_pac_constraints(evolved_state, pac_constraints)
        
        # Ensure normalization (probability conservation)
        norm = torch.norm(evolved_state)
        if norm > 0:
            evolved_state = evolved_state / norm
        
        # Calculate quantum metrics
        prob_amplitudes = torch.abs(evolved_state) ** 2
        entanglement = self._calculate_entanglement_measure(evolved_state)
        coherence_time = self._estimate_coherence_time(evolved_state)
        conservation_quality = self._assess_probability_conservation(prob_amplitudes)
        quantum_phase = torch.angle(torch.sum(evolved_state)).item()
        
        return QuantumPACResult(
            state_vector=evolved_state,
            probability_amplitudes=prob_amplitudes,
            entanglement_measure=entanglement,
            coherence_time=coherence_time,
            conservation_quality=conservation_quality,
            quantum_phase=quantum_phase
        )
    
    def create_superposition_state(self, 
                                 basis_states: List[torch.Tensor],
                                 amplitudes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create superposition state with PAC conservation.
        
        Args:
            basis_states: List of basis state vectors
            amplitudes: Superposition amplitudes (default: equal)
            
        Returns:
            Normalized superposition state
        """
        if amplitudes is None:
            amplitudes = torch.ones(len(basis_states), dtype=torch.complex128, device=self.device)
            amplitudes = amplitudes / torch.sqrt(torch.tensor(len(basis_states), dtype=torch.float64))
        
        amplitudes = amplitudes.to(self.device).to(torch.complex128)
        
        # Create superposition
        superposition = torch.zeros_like(basis_states[0], dtype=torch.complex128)
        for i, (state, amp) in enumerate(zip(basis_states, amplitudes)):
            state = state.to(self.device).to(torch.complex128)
            superposition += amp * state
        
        # Normalize (probability conservation)
        norm = torch.norm(superposition)
        if norm > 0:
            superposition = superposition / norm
        
        # Verify PAC conservation of probability
        total_prob = torch.sum(torch.abs(superposition) ** 2)
        assert abs(total_prob.item() - 1.0) < self.probability_conservation_tolerance, \
            f"Probability not conserved: {total_prob.item()}"
        
        return superposition
    
    def create_entangled_state(self, 
                             subsystem_dims: List[int],
                             entanglement_strength: float = 1.0) -> torch.Tensor:
        """
        Create entangled state with specified entanglement strength.
        
        Args:
            subsystem_dims: Dimensions of each subsystem
            entanglement_strength: Strength of entanglement (0-1)
            
        Returns:
            Entangled state tensor
        """
        total_dim = np.prod(subsystem_dims)
        
        if entanglement_strength == 0:
            # Separable state
            state = torch.randn(total_dim, dtype=torch.complex128, device=self.device)
        else:
            # Create maximally entangled state and mix with separable
            max_entangled = self._create_maximally_entangled_state(subsystem_dims)
            separable = torch.randn(total_dim, dtype=torch.complex128, device=self.device)
            
            # Mix states
            state = (entanglement_strength * max_entangled + 
                    (1 - entanglement_strength) * separable)
        
        # Normalize
        norm = torch.norm(state)
        if norm > 0:
            state = state / norm
        
        return state
    
    def measure_observable(self, 
                         state_vector: torch.Tensor,
                         observable: torch.Tensor,
                         collapse_state: bool = True) -> Tuple[float, torch.Tensor]:
        """
        Measure quantum observable with PAC-conserved collapse.
        
        Args:
            state_vector: Quantum state vector
            observable: Observable operator (Hermitian matrix)
            collapse_state: Whether to collapse state after measurement
            
        Returns:
            Tuple of (measurement_value, collapsed_state)
        """
        state_vector = state_vector.to(self.device).to(torch.complex128)
        observable = observable.to(self.device).to(torch.complex128)
        
        # Calculate expectation value
        expectation = torch.real(torch.conj(state_vector) @ observable @ state_vector)
        measurement_value = expectation.item()
        
        if collapse_state:
            # Find eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(observable)
            eigenvalues = torch.real(eigenvalues)
            
            # Calculate measurement probabilities
            overlaps = torch.abs(torch.conj(eigenvectors.T) @ state_vector) ** 2
            
            # Sample measurement outcome
            outcome_idx = torch.multinomial(overlaps, 1).item()
            collapsed_state = eigenvectors[:, outcome_idx]
            
            # Ensure proper normalization (PAC conservation)
            norm = torch.norm(collapsed_state)
            if norm > 0:
                collapsed_state = collapsed_state / norm
            
            actual_measurement = eigenvalues[outcome_idx].item()
            
            return actual_measurement, collapsed_state
        else:
            return measurement_value, state_vector
    
    def calculate_von_neumann_entropy(self, state_vector: torch.Tensor) -> float:
        """
        Calculate von Neumann entropy of quantum state.
        
        Args:
            state_vector: Quantum state vector
            
        Returns:
            von Neumann entropy
        """
        state_vector = state_vector.to(self.device).to(torch.complex128)
        
        # Calculate density matrix
        density_matrix = torch.outer(state_vector, torch.conj(state_vector))
        
        # Calculate eigenvalues
        eigenvalues = torch.real(torch.linalg.eigvals(density_matrix))
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Remove near-zero eigenvalues
        
        # Calculate entropy
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        
        return entropy.item()
    
    def simulate_quantum_tunneling(self, 
                                 initial_state: torch.Tensor,
                                 barrier_potential: torch.Tensor,
                                 energy: float,
                                 position_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulate quantum tunneling through PAC conservation.
        
        Args:
            initial_state: Initial wave function
            barrier_potential: Potential barrier
            energy: Particle energy
            position_grid: Position coordinate grid
            
        Returns:
            Dictionary with tunneling results
        """
        initial_state = initial_state.to(self.device).to(torch.complex128)
        barrier_potential = barrier_potential.to(self.device)
        position_grid = position_grid.to(self.device)
        
        # Create Hamiltonian
        kinetic_energy = self._create_kinetic_operator(position_grid)
        hamiltonian = kinetic_energy + torch.diag(barrier_potential)
        
        # Time evolution
        dt = 0.01
        total_time = 10.0
        n_steps = int(total_time / dt)
        
        state = initial_state.clone()
        probability_density = []
        tunneling_probability = []
        
        for step in range(n_steps):
            # Evolve state
            result = self.evolve_quantum_pac_state(state, hamiltonian, dt)
            state = result.state_vector
            
            # Calculate probability density
            prob_density = torch.abs(state) ** 2
            probability_density.append(prob_density.cpu())
            
            # Calculate tunneling probability (probability beyond barrier)
            barrier_end = torch.argmax((barrier_potential < energy / 2).float()).item()
            if barrier_end < len(prob_density) - 1:
                tunnel_prob = torch.sum(prob_density[barrier_end:]).item()
                tunneling_probability.append(tunnel_prob)
            else:
                tunneling_probability.append(0.0)
        
        return {
            "final_state": state,
            "probability_density_evolution": torch.stack(probability_density),
            "tunneling_probability": torch.tensor(tunneling_probability),
            "transmission_coefficient": tunneling_probability[-1] if tunneling_probability else 0.0
        }
    
    def _apply_pac_constraints(self, 
                             state_vector: torch.Tensor,
                             constraints: Dict) -> torch.Tensor:
        """Apply PAC conservation constraints to quantum state"""
        constrained_state = state_vector.clone()
        
        # Probability conservation
        if "total_probability" in constraints:
            target_prob = constraints["total_probability"]
            current_prob = torch.sum(torch.abs(constrained_state) ** 2)
            if current_prob > 0:
                constrained_state *= torch.sqrt(target_prob / current_prob)
        
        # Phase constraints
        if "global_phase" in constraints:
            target_phase = constraints["global_phase"]
            current_phase = torch.angle(torch.sum(constrained_state))
            phase_correction = torch.exp(1j * (target_phase - current_phase))
            constrained_state *= phase_correction
        
        # Symmetry constraints
        if "symmetries" in constraints:
            for symmetry in constraints["symmetries"]:
                if symmetry == "even_parity":
                    # Enforce even parity
                    n = len(constrained_state)
                    for i in range(n // 2):
                        avg = (constrained_state[i] + constrained_state[n-1-i]) / 2
                        constrained_state[i] = avg
                        constrained_state[n-1-i] = avg
        
        return constrained_state
    
    def _calculate_entanglement_measure(self, state_vector: torch.Tensor) -> float:
        """Calculate entanglement measure (simplified)"""
        # For a bipartite system, calculate von Neumann entropy of reduced density matrix
        # This is a simplified version - assumes bipartite system
        
        n = len(state_vector)
        if n < 4:  # Need at least 2x2 system for entanglement
            return 0.0
        
        # Assume equal bipartition
        dim_a = int(np.sqrt(n))
        dim_b = n // dim_a
        
        if dim_a * dim_b != n:
            # Not a perfect bipartite system
            return 0.0
        
        # Reshape state vector into matrix
        state_matrix = state_vector.reshape(dim_a, dim_b)
        
        # Calculate reduced density matrix for subsystem A
        rho_a = torch.matmul(state_matrix, torch.conj(state_matrix).T)
        
        # Calculate eigenvalues
        eigenvalues = torch.real(torch.linalg.eigvals(rho_a))
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        # Calculate entanglement entropy
        if len(eigenvalues) > 0:
            entanglement = -torch.sum(eigenvalues * torch.log(eigenvalues))
            return entanglement.item()
        else:
            return 0.0
    
    def _estimate_coherence_time(self, state_vector: torch.Tensor) -> float:
        """Estimate quantum coherence time"""
        # Simplified coherence time based on state purity
        prob_amplitudes = torch.abs(state_vector) ** 2
        purity = torch.sum(prob_amplitudes ** 2)
        
        # Higher purity = longer coherence time
        coherence_time = 1.0 / (self.decoherence_rate * (2 - purity))
        
        return coherence_time.item()
    
    def _assess_probability_conservation(self, probabilities: torch.Tensor) -> float:
        """Assess quality of probability conservation"""
        total_prob = torch.sum(probabilities)
        deviation = abs(total_prob.item() - 1.0)
        quality = np.exp(-deviation / self.probability_conservation_tolerance)
        return quality
    
    def _create_maximally_entangled_state(self, subsystem_dims: List[int]) -> torch.Tensor:
        """Create maximally entangled state for given subsystem dimensions"""
        # Create Bell state or generalization
        if len(subsystem_dims) == 2 and subsystem_dims[0] == subsystem_dims[1]:
            # Bell state
            dim = subsystem_dims[0]
            state = torch.zeros(dim * dim, dtype=torch.complex128, device=self.device)
            for i in range(dim):
                state[i * dim + i] = 1.0 / np.sqrt(dim)
            return state
        else:
            # Generalized maximally entangled state
            total_dim = np.prod(subsystem_dims)
            state = torch.randn(total_dim, dtype=torch.complex128, device=self.device)
            # Add entanglement structure
            for i in range(min(subsystem_dims)):
                idx = sum(i * np.prod(subsystem_dims[j+1:]) for j in range(len(subsystem_dims)))
                if idx < total_dim:
                    state[idx] += 1.0
            return state
    
    def _create_kinetic_operator(self, position_grid: torch.Tensor) -> torch.Tensor:
        """Create kinetic energy operator using finite differences"""
        n = len(position_grid)
        dx = position_grid[1] - position_grid[0] if n > 1 else 1.0
        
        # Second derivative operator (kinetic energy)
        kinetic = torch.zeros(n, n, dtype=torch.complex128, device=self.device)
        
        # Finite difference approximation of second derivative
        coeff = -self.hbar**2 / (2 * dx**2)  # Assuming unit mass
        
        for i in range(n):
            kinetic[i, i] = -2 * coeff
            if i > 0:
                kinetic[i, i-1] = coeff
            if i < n - 1:
                kinetic[i, i+1] = coeff
        
        return kinetic
    
    def get_quantum_state_info(self, state_vector: torch.Tensor) -> Dict[str, Any]:
        """Get comprehensive information about quantum state"""
        state_vector = state_vector.to(self.device).to(torch.complex128)
        
        info = {
            "norm": torch.norm(state_vector).item(),
            "dimension": len(state_vector),
            "entropy": self.calculate_von_neumann_entropy(state_vector),
            "entanglement": self._calculate_entanglement_measure(state_vector),
            "coherence_time": self._estimate_coherence_time(state_vector),
            "probability_amplitudes": torch.abs(state_vector) ** 2,
            "phases": torch.angle(state_vector),
            "purity": torch.sum((torch.abs(state_vector) ** 2) ** 2).item()
        }
        
        return info

# Utility functions for quantum PAC operations
def create_pauli_operators(device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Create Pauli spin operators"""
    device = torch.device(device)
    
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    identity = torch.eye(2, dtype=torch.complex128, device=device)
    
    return {
        "X": pauli_x,
        "Y": pauli_y,
        "Z": pauli_z,
        "I": identity
    }

def create_harmonic_oscillator_hamiltonian(n_levels: int, 
                                         omega: float = 1.0,
                                         device: str = "cpu") -> torch.Tensor:
    """Create harmonic oscillator Hamiltonian"""
    device = torch.device(device)
    
    # Create ladder operators
    a_dagger = torch.zeros(n_levels, n_levels, dtype=torch.complex128, device=device)
    a = torch.zeros(n_levels, n_levels, dtype=torch.complex128, device=device)
    
    for n in range(n_levels - 1):
        a_dagger[n+1, n] = np.sqrt(n + 1)
        a[n, n+1] = np.sqrt(n + 1)
    
    # Hamiltonian: H = ħω(a†a + 1/2)
    hamiltonian = omega * (torch.matmul(a_dagger, a) + 0.5 * torch.eye(n_levels, device=device))
    
    return hamiltonian

def create_quantum_walk_operator(n_sites: int, device: str = "cpu") -> torch.Tensor:
    """Create quantum walk evolution operator"""
    device = torch.device(device)
    
    # Simple quantum walk on a line
    walk_operator = torch.zeros(n_sites, n_sites, dtype=torch.complex128, device=device)
    
    # Hopping to nearest neighbors
    for i in range(n_sites):
        if i > 0:
            walk_operator[i-1, i] = 0.5
        if i < n_sites - 1:
            walk_operator[i+1, i] = 0.5
        walk_operator[i, i] = 0.0  # No self-interaction
    
    return walk_operator
