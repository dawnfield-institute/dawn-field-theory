"""
GAIA v2.0 - Field Engine Implementation
Physics-Informed Entropy Field Dynamics for AGI

TORCH ONLY - NO NUMPY
This implementation uses PyTorch with CUDA acceleration exclusively.

Based on field_engine.md specifications:
- Simulates continuous interaction between Energy and Information fields  
- Measures entropy tension and computes field pressure
- Triggers collapse events when system deviates from balance
- Physics-informed with thermodynamic cost integration
- Auto-tuning thresholds using RBF/QBE balance optimization
"""

print("DEBUG: Starting field_engine.py execution")

import torch
print("DEBUG: torch imported")
import torch.nn.functional as F
print("DEBUG: torch.nn.functional imported")
from typing import Dict, List, Tuple, Optional, Any
print("DEBUG: typing imported")
import logging
print("DEBUG: logging imported")
import math
print("DEBUG: math imported")

# Set device for CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import shared data structures and adaptive controller
try:
    from .data_structures import FieldState, CollapseEvent
    from .adaptive_controller import GAIAAdaptiveController
except ImportError:
    # Fallback for direct execution
    from data_structures import FieldState, CollapseEvent
    from adaptive_controller import GAIAAdaptiveController


print("DEBUG: About to define FieldEngine class")

class FieldEngine:
    """
    Core Field Engine implementing physics-informed entropy dynamics with adaptive thresholds
    
    Simulates Energy/Information field interactions with entropy tension monitoring.
    Triggers collapse events based on thermodynamic principles.
    Uses RBF/QBE balance-based auto-tuning for dynamic threshold adaptation.
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...] = (32, 32),
                 collapse_threshold: float = 0.0003,  # Base threshold, will be adaptive
                 temperature: float = 1.0,
                 pi_harmonic_modulation: bool = True,
                 enable_adaptive_tuning: bool = True,  # Enable auto-tuning
                 **kwargs):  # Added kwargs to handle extra parameters
        """
        Initialize the Field Engine with adaptive threshold controller
        
        Args:
            field_shape: Shape of the entropy field tensor
            collapse_threshold: Base collapse threshold (will be adapted automatically)  
            temperature: Thermodynamic temperature for cost calculations
            pi_harmonic_modulation: Enable pi-harmonic frequency modulation
            enable_adaptive_tuning: Enable RBF/QBE auto-tuning
        """
        self.field_shape = field_shape
        self.base_collapse_threshold = collapse_threshold
        self.temperature = temperature
        self.pi_harmonic_modulation = pi_harmonic_modulation
        self.enable_adaptive_tuning = enable_adaptive_tuning
        
        # Initialize adaptive controller (like CIMM/TinyCIMM pattern)
        if enable_adaptive_tuning:
            self.adaptive_controller = GAIAAdaptiveController(
                base_collapse_threshold=collapse_threshold,
                adaptation_window=20,
                balance_momentum=0.8,
                qbe_sensitivity=0.1
            )
        else:
            self.adaptive_controller = None
        
        # Initialize fields on device
        self.energy_field = torch.zeros(field_shape, dtype=torch.float32, device=device)
        self.information_field = torch.zeros(field_shape, dtype=torch.float32, device=device)
        self.entropy_tensor = torch.zeros(field_shape, dtype=torch.float32, device=device)
        
        # State tracking
        self.field_history: List[FieldState] = []
        self.collapse_events: List[CollapseEvent] = []
        self.timestep = 0
        
        # Thermodynamic constants
        self.k_b = 1.380649e-23  # Boltzmann constant (J/K)
        
        logging.info(f"Field Engine initialized with shape {field_shape}, adaptive_tuning={enable_adaptive_tuning}")

    def inject_stimulus(self, 
                       stimulus: torch.Tensor, 
                       stimulus_type: str = "energy",
                       location: Optional[Tuple[int, ...]] = None) -> None:
        """
        Inject external stimulus into the field
        
        Args:
            stimulus: Input data to inject 
            stimulus_type: Type of stimulus ("energy" or "information")
            location: Specific field location to inject (None for center)
        """
        # Ensure stimulus is on correct device
        if not isinstance(stimulus, torch.Tensor):
            stimulus = torch.tensor(stimulus, dtype=torch.float32, device=device)
        else:
            stimulus = stimulus.to(device)
        
        # Convert stimulus to field representation
        field_stimulus = self._encode_stimulus_to_field(stimulus)
        
        if location is None:
            # Inject at field center
            center = tuple(s // 2 for s in self.field_shape)
            location = center
            
        # Inject into appropriate field
        if stimulus_type == "energy":
            self._inject_energy(field_stimulus, location)
        elif stimulus_type == "information":
            self._inject_information(field_stimulus, location)
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")
            
        logging.debug(f"Injected {stimulus_type} stimulus at {location}")
    
    def step(self) -> Optional[CollapseEvent]:
        """
        Execute one timestep of field dynamics with adaptive threshold updates
        
        Returns:
            CollapseEvent if collapse occurred, None otherwise
        """
        self.timestep += 1
        
        # 1. Update field dynamics using physics equations
        self._update_energy_field()
        self._update_information_field()
        self._compute_entropy_tensor()
        
        # 2. Calculate field pressure and collapse likelihood
        field_pressure = self._compute_field_pressure()
        collapse_likelihood = self._compute_collapse_likelihood(field_pressure)
        
        # 3. Update adaptive controller with current field metrics
        if self.adaptive_controller:
            # Compute field balance and variance for adaptation
            field_variance = (torch.var(self.energy_field) + torch.var(self.information_field)).item()
            field_balance = self._compute_field_balance()
            collapse_count = 1 if self._should_collapse(field_pressure, collapse_likelihood) else 0
            
            # Update adaptive controller
            self.adaptive_controller.update_field_metrics(
                field_pressure=field_pressure,
                field_variance=field_variance,
                collapse_count=collapse_count,
                field_balance=field_balance
            )
            
            # Compute new adaptive thresholds
            self.adaptive_controller.compute_dynamic_thresholds()
            
            # Adapt for detected patterns (like TinyCIMM)
            pattern_type = self.adaptive_controller.detect_field_pattern_type()
            if pattern_type != "unknown":
                self.adaptive_controller.adapt_for_pattern(pattern_type)
        
        # 4. Check for collapse conditions using adaptive threshold
        collapse_event = None
        if self._should_collapse(field_pressure, collapse_likelihood):
            collapse_event = self._trigger_collapse()
            
        # 5. Log current state
        current_state = FieldState(
            energy_field=self.energy_field.clone(),
            information_field=self.information_field.clone(),
            entropy_tensor=self.entropy_tensor.clone(),
            field_pressure=field_pressure,
            collapse_likelihood=collapse_likelihood,
            timestamp=self.timestep
        )
        self.field_history.append(current_state)
        
        return collapse_event
    
    def _compute_field_balance(self) -> float:
        """Compute field balance metric for adaptive controller"""
        # Balance between energy and information fields
        energy_magnitude = torch.mean(torch.abs(self.energy_field)).item()
        info_magnitude = torch.mean(torch.abs(self.information_field)).item()
        
        if energy_magnitude + info_magnitude == 0:
            return 1.0  # Perfect balance when both are zero
        
        balance = min(energy_magnitude, info_magnitude) / max(energy_magnitude, info_magnitude, 1e-8)
        return balance
    
    def _encode_stimulus_to_field(self, stimulus: torch.Tensor) -> torch.Tensor:
        """Convert arbitrary stimulus to field representation"""
        # Flatten and resize to match field shape
        flat_stimulus = stimulus.flatten()
        field_size = torch.prod(torch.tensor(self.field_shape, device=device))
        
        if len(flat_stimulus) > field_size:
            # Truncate if too large
            encoded = flat_stimulus[:field_size]
        else:
            # Pad if too small
            padding_size = field_size - len(flat_stimulus)
            padding = torch.zeros(padding_size, device=device, dtype=torch.float32)
            encoded = torch.cat([flat_stimulus, padding])
            
        return encoded.reshape(self.field_shape)
    
    def _inject_energy(self, field_stimulus: torch.Tensor, location: Tuple[int, ...]) -> None:
        """Inject energy stimulus into energy field with adaptive strength"""
        mask = self._create_injection_mask(location)
        
        # Get adaptive injection strength
        if self.adaptive_controller:
            strength = self.adaptive_controller.get_adaptive_thresholds().injection_energy_strength
        else:
            strength = 2.0  # Default fallback
            
        self.energy_field += field_stimulus * mask * strength
    
    def _inject_information(self, field_stimulus: torch.Tensor, location: Tuple[int, ...]) -> None:
        """Inject information stimulus into information field with adaptive strength"""
        mask = self._create_injection_mask(location)
        
        # Get adaptive injection strength
        if self.adaptive_controller:
            strength = self.adaptive_controller.get_adaptive_thresholds().injection_info_strength
        else:
            strength = 2.0  # Default fallback
            
        self.information_field += field_stimulus * mask * strength
    
    def _create_injection_mask(self, center: Tuple[int, ...], sigma: float = 2.0) -> torch.Tensor:
        """Create Gaussian injection mask centered at location"""
        mask = torch.zeros(self.field_shape, device=device, dtype=torch.float32)
        
        # Create coordinate grids
        if len(self.field_shape) == 2:
            h, w = self.field_shape
            y_coords = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
            x_coords = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0)
            
            # Calculate distance from center
            cy, cx = center
            distance_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
            mask = torch.exp(-distance_sq / (2 * sigma ** 2))
        else:
            # Simple 1D case
            coords = torch.arange(self.field_shape[0], device=device, dtype=torch.float32)
            distance_sq = (coords - center[0]) ** 2
            mask = torch.exp(-distance_sq / (2 * sigma ** 2))
        
        return mask
    
    def _update_energy_field(self) -> None:
        """
        Update energy field dynamics using physics-informed equations with adaptive rates
        
        From spec: Φ_E(x) = ∇·E(x) (Energy divergence)
        Energy field evolution: ∇²E - α*E (diffusion with decay)
        """
        # Compute Laplacian for diffusion
        laplacian = self._compute_laplacian(self.energy_field)
        
        # Get adaptive rates
        if self.adaptive_controller:
            thresholds = self.adaptive_controller.get_adaptive_thresholds()
            diffusion_rate = thresholds.diffusion_rate
            decay_rate = thresholds.decay_rate
        else:
            diffusion_rate = 0.02  # Default fallback
            decay_rate = 0.001
            
        self.energy_field += diffusion_rate * laplacian - decay_rate * self.energy_field
    
    def _update_information_field(self) -> None:
        """
        Update information field dynamics with adaptive rates
        
        From spec: Φ_I(x) = ∇·I(x) (Information compression gradient)
        Information field evolution: compression toward stable patterns
        """
        # Compute Laplacian for compression dynamics
        laplacian = self._compute_laplacian(self.information_field)
        
        # Get adaptive rates (scaled from energy rates)
        if self.adaptive_controller:
            thresholds = self.adaptive_controller.get_adaptive_thresholds()
            compression_rate = thresholds.diffusion_rate * 0.5  # Half of diffusion rate
            stability_rate = thresholds.decay_rate * 0.5  # Half of decay rate
        else:
            compression_rate = 0.01  # Default fallback
            stability_rate = 0.0005
            
        self.information_field += compression_rate * laplacian - stability_rate * self.information_field
    
    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian for field dynamics"""
        if len(field.shape) == 2:
            # 2D Laplacian using convolution
            field_4d = field.unsqueeze(0).unsqueeze(0)
            
            # Laplacian kernel
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                          device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Apply convolution with padding
            laplacian_4d = F.conv2d(field_4d, laplacian_kernel, padding=1)
            laplacian = laplacian_4d.squeeze(0).squeeze(0)
        else:
            # 1D finite differences
            laplacian = torch.zeros_like(field)
            laplacian[1:-1] = field[2:] - 2 * field[1:-1] + field[:-2]
        
        return laplacian
    
    def _compute_entropy_tensor(self) -> None:
        """
        Compute entropy tensor from field differences
        
        From spec: ΔS(x) = H(Φ_E(x)) - H(Φ_I(x)) (Entropic tension)
        """
        # Basic entropy: field imbalance
        self.entropy_tensor = torch.abs(self.energy_field - self.information_field)
        
        # Add local entropy based on gradients
        local_entropy = self._compute_local_entropy()
        self.entropy_tensor += local_entropy
    
    def _compute_local_entropy(self) -> torch.Tensor:
        """Compute local entropy based on field gradients"""
        if len(self.field_shape) == 2:
            # 2D gradients
            energy_grad_y, energy_grad_x = torch.gradient(self.energy_field)
            info_grad_y, info_grad_x = torch.gradient(self.information_field)
            
            # Gradient magnitudes
            energy_grad_mag = torch.sqrt(energy_grad_x**2 + energy_grad_y**2)
            info_grad_mag = torch.sqrt(info_grad_x**2 + info_grad_y**2)
        else:
            # 1D gradients
            energy_grad = torch.gradient(self.energy_field)[0]
            info_grad = torch.gradient(self.information_field)[0]
            
            energy_grad_mag = torch.abs(energy_grad)
            info_grad_mag = torch.abs(info_grad)
        
        return 0.1 * (energy_grad_mag + info_grad_mag)
    
    def _compute_field_pressure(self) -> float:
        """
        Compute overall field pressure from entropy tensor
        
        From spec: Field pressure builds when system deviates from balance
        """
        # Basic pressure: mean entropy tension
        pressure = torch.mean(self.entropy_tensor).item()
        
        # Add pi-harmonic modulation if enabled
        if self.pi_harmonic_modulation:
            # From spec: ω_collapse(x) = ω_base * (π/2)^(depth_recursion(x))
            harmonic_factor = (math.pi / 2) ** (self.timestep % 10 / 10.0)
            pressure *= harmonic_factor
            
        return pressure
    
    def _compute_collapse_likelihood(self, field_pressure: float) -> float:
        """Compute likelihood of collapse based on field pressure with adaptive threshold"""
        # Get adaptive collapse threshold
        if self.adaptive_controller:
            threshold = self.adaptive_controller.get_adaptive_thresholds().collapse_threshold
        else:
            threshold = self.base_collapse_threshold
            
        # Sigmoid mapping pressure to probability
        likelihood = 1.0 / (1.0 + torch.exp(torch.tensor(-10 * (field_pressure - threshold), device=device))).item()
        return likelihood
    
    def _should_collapse(self, field_pressure: float, collapse_likelihood: float) -> bool:
        """
        Determine if collapse should occur using adaptive threshold
        
        From spec: CollapseTrigger(x) = 1 if ΔS(x) > σ AND |∇_macro(x)| > τ
        """
        # Get adaptive collapse threshold
        if self.adaptive_controller:
            threshold = self.adaptive_controller.get_adaptive_thresholds().collapse_threshold
        else:
            threshold = self.base_collapse_threshold
        
        # Threshold check with stochastic component
        if field_pressure > threshold:
            return torch.rand(1, device=device).item() < collapse_likelihood
        return False
    
    def _trigger_collapse(self) -> CollapseEvent:
        """
        Trigger a collapse event and modify fields
        
        From spec: When triggered, system crystallizes structure into symbolic node
        """
        # Find location of maximum entropy tension
        flat_idx = torch.argmax(self.entropy_tensor)
        
        # Convert flat index to coordinates
        if len(self.entropy_tensor.shape) == 2:
            h, w = self.entropy_tensor.shape
            row = flat_idx // w
            col = flat_idx % w
            collapse_location = (row.item(), col.item())
        else:
            collapse_location = (flat_idx.item(),)
        
        # Record pre-collapse state
        pre_pressure = self._compute_field_pressure()
        pre_entropy = self.entropy_tensor[collapse_location].item()
        
        # Execute collapse with thermodynamic cost calculation
        thermodynamic_cost = self._execute_collapse_at_location(collapse_location)
        
        # Record post-collapse state
        post_pressure = self._compute_field_pressure()
        entropy_delta = pre_entropy - self.entropy_tensor[collapse_location].item()
        
        # Create collapse event
        collapse_event = CollapseEvent(
            location=collapse_location,
            entropy_delta=entropy_delta,
            field_pressure_pre=pre_pressure,
            field_pressure_post=post_pressure,
            collapse_type="entropy_resolution",
            timestamp=self.timestep,
            metadata={
                "field_size": self.field_shape,
                "thermodynamic_cost": thermodynamic_cost,
                "pi_harmonic": self.pi_harmonic_modulation
            }
        )
        
        self.collapse_events.append(collapse_event)
        logging.info(f"Collapse event at {collapse_location}, ΔS={entropy_delta:.4f}")
        
        return collapse_event
    
    def _execute_collapse_at_location(self, location: Tuple[int, ...]) -> float:
        """
        Execute collapse operation with thermodynamic cost calculation
        
        From spec: E_erasure = k_B * T * ln(2) * N_bits_erased
        """
        # Calculate thermodynamic cost (Landauer erasure)
        entropy_before = self.entropy_tensor[location].item()
        
        # Reduce entropy tension at collapse location
        self.entropy_tensor[location] *= 0.1
        
        # Stabilize local field balance
        avg_value = (self.energy_field[location] + self.information_field[location]) / 2
        self.energy_field[location] = avg_value * 1.1
        self.information_field[location] = avg_value * 0.9
        
        # Create stabilization zone around collapse point
        mask = self._create_injection_mask(location, sigma=1.0)
        stabilization = 0.1 * mask
        self.energy_field = (1 - stabilization) * self.energy_field + stabilization * avg_value
        self.information_field = (1 - stabilization) * self.information_field + stabilization * avg_value
        
        # Calculate Landauer erasure cost
        entropy_after = self.entropy_tensor[location].item()
        bits_erased = max(0, entropy_before - entropy_after)
        thermodynamic_cost = self.k_b * self.temperature * math.log(2) * bits_erased
        
        return thermodynamic_cost
    
    def get_field_state(self) -> FieldState:
        """Get current field state"""
        return FieldState(
            energy_field=self.energy_field.clone(),
            information_field=self.information_field.clone(),
            entropy_tensor=self.entropy_tensor.clone(),
            field_pressure=self._compute_field_pressure(),
            collapse_likelihood=self._compute_collapse_likelihood(self._compute_field_pressure()),
            timestamp=self.timestep
        )
    
    def reset(self) -> None:
        """Reset field engine to initial state with adaptive controller reset"""
        self.energy_field = torch.zeros(self.field_shape, dtype=torch.float32, device=device)
        self.information_field = torch.zeros(self.field_shape, dtype=torch.float32, device=device)
        self.entropy_tensor = torch.zeros(self.field_shape, dtype=torch.float32, device=device)
        self.field_history.clear()
        self.collapse_events.clear()
        self.timestep = 0
        
        # Reset adaptive controller
        if self.adaptive_controller:
            self.adaptive_controller.reset_adaptation_state()
        
        logging.info("Field Engine reset to initial state with adaptive controller reset")
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """Get current adaptive controller status"""
        if not self.adaptive_controller:
            return {"adaptive_tuning": False}
        
        thresholds = self.adaptive_controller.get_adaptive_thresholds()
        return {
            "adaptive_tuning": True,
            "qbe_status": self.adaptive_controller.get_qbe_status(),
            "pattern_type": self.adaptive_controller.detect_field_pattern_type(),
            "collapse_threshold": thresholds.collapse_threshold,
            "injection_energy_strength": thresholds.injection_energy_strength,
            "injection_info_strength": thresholds.injection_info_strength,
            "diffusion_rate": thresholds.diffusion_rate,
            "decay_rate": thresholds.decay_rate
        }


print("DEBUG: FieldEngine class definition completed")


if __name__ == "__main__":
    print("Field Engine v2.0 loaded successfully")
    print(f"FieldEngine class defined: {'FieldEngine' in globals()}")
    logging.basicConfig(level=logging.INFO)
    
    # Simple test
    engine = FieldEngine(field_shape=(16, 16), collapse_threshold=0.5)
    test_stimulus = torch.rand((4, 4), device=device)
    engine.inject_stimulus(test_stimulus, "energy")
    
    print("Running Field Engine test...")
    for step in range(10):
        collapse_event = engine.step()
        if collapse_event:
            print(f"Collapse at step {step}: location={collapse_event.location}, ΔS={collapse_event.entropy_delta:.4f}")
    
    print(f"Test complete. Total collapses: {len(engine.collapse_events)}")
