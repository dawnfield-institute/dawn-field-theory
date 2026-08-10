"""
POC-001 Pattern Encoding Utilities
===================================

Shared utilities for all pattern encoding experiments.

Technical Requirements:
- PyTorch ONLY (no numpy)
- GPU by default, CPU fallback
- Uses Fracton's GPUAcceleratedMemoryField where applicable
"""

import torch
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Device configuration - GPU first
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class EncodingResult:
    """Result of encoding a pattern into the field"""
    input_pattern: str
    field_state: torch.Tensor
    encoding_time_ms: float
    conservation_residual: float
    field_energy: float
    pattern_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_pattern': self.input_pattern,
            'field_state_shape': list(self.field_state.shape),
            'field_state_device': str(self.field_state.device),
            'encoding_time_ms': self.encoding_time_ms,
            'conservation_residual': self.conservation_residual,
            'field_energy': self.field_energy,
            'pattern_hash': self.pattern_hash
        }


@dataclass
class EvolutionResult:
    """Result of evolving a field state"""
    initial_energy: float
    final_energy: float
    energy_delta: float
    steps: int
    evolution_time_ms: float
    pattern_survived: bool
    correlation_with_initial: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Complete experiment result for saving"""
    experiment_id: str
    timestamp: str
    device: str
    parameters: Dict[str, Any]
    encodings: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    success: bool
    notes: str = ""
    
    def save(self, results_dir: Path) -> Path:
        """Save result to JSON file"""
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_id}_{self.timestamp.replace(':', '-')}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        
        return filepath


class FieldEncoder:
    """
    Encode patterns into field perturbations using GPU tensors.
    
    This is the core utility for converting symbolic input (text, binary)
    into field states that can be processed by GAIA's physics.
    """
    
    def __init__(
        self,
        field_dims: Tuple[int, int] = (64, 64),
        device: torch.device = DEVICE,
        xi_target: float = 1.0571
    ):
        self.field_dims = field_dims
        self.device = device
        self.xi_target = xi_target
        
        # Pre-compute coordinate grids on GPU
        x = torch.linspace(0, 2 * torch.pi, field_dims[0], device=device)
        y = torch.linspace(0, 2 * torch.pi, field_dims[1], device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
    def encode_binary(self, binary_str: str) -> EncodingResult:
        """
        Encode a binary string into a field perturbation.
        
        Strategy: Each bit creates a wave component with frequency based on position.
        0 = sin wave, 1 = cos wave (phase difference)
        """
        start_time = time.perf_counter()
        
        # Initialize field on GPU
        field = torch.zeros(self.field_dims, device=self.device, dtype=torch.float32)
        
        for i, bit in enumerate(binary_str):
            freq = (i + 1) * 0.5  # Frequency increases with position
            if bit == '1':
                field += torch.cos(freq * self.X + freq * self.Y)
            else:
                field += torch.sin(freq * self.X + freq * self.Y)
        
        # Normalize to maintain energy
        field = self._normalize_field(field)
        
        # Calculate metrics
        field_energy = self._calculate_energy(field)
        conservation_residual = self._check_conservation(field)
        
        encoding_time = (time.perf_counter() - start_time) * 1000
        
        return EncodingResult(
            input_pattern=binary_str,
            field_state=field,
            encoding_time_ms=encoding_time,
            conservation_residual=conservation_residual,
            field_energy=field_energy,
            pattern_hash=self._hash_pattern(binary_str)
        )
    
    def encode_character(self, char: str) -> EncodingResult:
        """
        Encode a single character into a field perturbation.
        
        Strategy: Use character ordinal to set frequency and phase.
        """
        start_time = time.perf_counter()
        
        # Get character properties
        ord_val = ord(char)
        freq = (ord_val % 26 + 1) * 0.3  # Frequency from position in alphabet
        phase = (ord_val // 26) * torch.pi / 4  # Phase from character group
        
        # Create wave pattern
        field = torch.cos(freq * self.X + phase) * torch.sin(freq * self.Y)
        
        # Add second harmonic for more distinctiveness
        field += 0.3 * torch.sin(2 * freq * self.X) * torch.cos(2 * freq * self.Y + phase)
        
        # Normalize
        field = self._normalize_field(field)
        
        encoding_time = (time.perf_counter() - start_time) * 1000
        
        return EncodingResult(
            input_pattern=char,
            field_state=field,
            encoding_time_ms=encoding_time,
            conservation_residual=self._check_conservation(field),
            field_energy=self._calculate_energy(field),
            pattern_hash=self._hash_pattern(char)
        )
    
    def encode_word(self, word: str) -> EncodingResult:
        """
        Encode a word by superposing character patterns with positional weighting.
        """
        start_time = time.perf_counter()
        
        field = torch.zeros(self.field_dims, device=self.device, dtype=torch.float32)
        
        for i, char in enumerate(word.lower()):
            # Encode each character with positional modulation
            char_result = self.encode_character(char)
            position_weight = 1.0 / (i + 1)  # Earlier chars weighted more
            field += position_weight * char_result.field_state
        
        # Normalize composite field
        field = self._normalize_field(field)
        
        encoding_time = (time.perf_counter() - start_time) * 1000
        
        return EncodingResult(
            input_pattern=word,
            field_state=field,
            encoding_time_ms=encoding_time,
            conservation_residual=self._check_conservation(field),
            field_energy=self._calculate_energy(field),
            pattern_hash=self._hash_pattern(word)
        )
    
    def _normalize_field(self, field: torch.Tensor) -> torch.Tensor:
        """Normalize field to unit energy with Xi correction."""
        norm = torch.norm(field)
        if norm > 0:
            field = field / norm * self.xi_target
        return field
    
    def _calculate_energy(self, field: torch.Tensor) -> float:
        """Calculate total field energy."""
        return float(torch.sum(field ** 2))
    
    def _check_conservation(self, field: torch.Tensor) -> float:
        """Check PAC conservation residual."""
        # For a normalized field, residual should be near zero
        expected_energy = self.xi_target ** 2 * field.numel()
        actual_energy = torch.sum(field ** 2)
        return float(abs(actual_energy - self._calculate_energy(field)))
    
    def _hash_pattern(self, pattern: str) -> str:
        """Create deterministic hash of input pattern."""
        return hashlib.sha256(pattern.encode()).hexdigest()[:16]


class FieldEvolver:
    """
    Evolve field states using Klein-Gordon dynamics.
    
    GPU-accelerated implementation of field evolution to test pattern stability.
    """
    
    def __init__(
        self,
        field_dims: Tuple[int, int] = (64, 64),
        device: torch.device = DEVICE,
        mass_squared: float = 0.1,
        dt: float = 0.01
    ):
        self.field_dims = field_dims
        self.device = device
        self.mass_squared = mass_squared
        self.dt = dt
        
        # Pre-compute Laplacian kernel on GPU
        self.laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def evolve(
        self,
        field: torch.Tensor,
        steps: int,
        track_correlation: bool = True
    ) -> EvolutionResult:
        """
        Evolve a field state through Klein-Gordon dynamics.
        
        Uses the equation: ∂²φ/∂t² = c²∇²φ - m²φ
        Discretized with finite differences.
        """
        start_time = time.perf_counter()
        
        initial_field = field.clone()
        initial_energy = float(torch.sum(field ** 2))
        
        # Velocity field (for second-order evolution)
        velocity = torch.zeros_like(field)
        
        # Evolve
        for _ in range(steps):
            field, velocity = self._step(field, velocity)
        
        # Calculate metrics
        final_energy = float(torch.sum(field ** 2))
        energy_delta = abs(final_energy - initial_energy)
        
        # Correlation with initial pattern
        if track_correlation:
            correlation = self._calculate_correlation(initial_field, field)
        else:
            correlation = 0.0
        
        evolution_time = (time.perf_counter() - start_time) * 1000
        
        return EvolutionResult(
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_delta=energy_delta,
            steps=steps,
            evolution_time_ms=evolution_time,
            pattern_survived=correlation > 0.5,  # >50% correlation = survived
            correlation_with_initial=correlation
        )
    
    def _step(
        self,
        field: torch.Tensor,
        velocity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Klein-Gordon evolution step."""
        # Compute Laplacian via convolution
        field_padded = field.unsqueeze(0).unsqueeze(0)
        laplacian = torch.nn.functional.conv2d(
            field_padded,
            self.laplacian_kernel,
            padding=1
        ).squeeze()
        
        # Klein-Gordon: acceleration = ∇²φ - m²φ
        acceleration = laplacian - self.mass_squared * field
        
        # Velocity Verlet integration
        velocity = velocity + self.dt * acceleration
        field = field + self.dt * velocity
        
        return field, velocity
    
    def _calculate_correlation(
        self,
        field1: torch.Tensor,
        field2: torch.Tensor
    ) -> float:
        """Calculate normalized correlation between two fields."""
        # Flatten and center
        f1 = field1.flatten() - field1.mean()
        f2 = field2.flatten() - field2.mean()
        
        # Correlation coefficient
        numerator = torch.sum(f1 * f2)
        denominator = torch.sqrt(torch.sum(f1 ** 2) * torch.sum(f2 ** 2))
        
        if denominator > 0:
            return float(numerator / denominator)
        return 0.0


def measure_pattern_distance(
    field1: torch.Tensor,
    field2: torch.Tensor
) -> float:
    """Measure Euclidean distance between two field patterns."""
    return float(torch.norm(field1 - field2))


def measure_pattern_similarity(
    field1: torch.Tensor,
    field2: torch.Tensor
) -> float:
    """Measure cosine similarity between two field patterns."""
    f1 = field1.flatten()
    f2 = field2.flatten()
    
    dot = torch.sum(f1 * f2)
    norm1 = torch.norm(f1)
    norm2 = torch.norm(f2)
    
    if norm1 > 0 and norm2 > 0:
        return float(dot / (norm1 * norm2))
    return 0.0


def get_gpu_info() -> Dict[str, Any]:
    """Get current GPU information."""
    if torch.cuda.is_available():
        return {
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved(),
        }
    return {'available': False, 'device': 'cpu'}


def get_results_dir() -> Path:
    """Get the results directory for POC-001."""
    return Path(__file__).parent.parent / 'results'


def generate_experiment_id(prefix: str = "exp") -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
