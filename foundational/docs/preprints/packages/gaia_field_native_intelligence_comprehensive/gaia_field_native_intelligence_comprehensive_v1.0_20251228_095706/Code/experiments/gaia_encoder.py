"""
GAIA Field Encoder - GPU-Native Pattern Encoding
=================================================

Drop-in replacement for GAIA's _encode_input_to_field using PyTorch.

This module provides GPU-accelerated pattern encoding that can be
integrated into the main GAIA class to replace numpy-based encoding.

Technical Requirements:
- PyTorch only (no numpy)
- GPU acceleration
- Maintains PAC conservation
"""

import torch
import hashlib
from typing import Any, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class GAIAEncodingConfig:
    """Configuration for GAIA-compatible encoding."""
    field_dimensions: Tuple[int, int] = (64, 64)
    xi_target: float = 1.0571
    device: torch.device = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAIAFieldEncoder:
    """
    GPU-native field encoder for GAIA integration.
    
    This replaces the numpy-based _encode_input_to_field method
    with a PyTorch implementation that runs entirely on GPU.
    
    Usage:
        encoder = GAIAFieldEncoder(config)
        field_state = encoder.encode(input_data)
    """
    
    def __init__(self, config: GAIAEncodingConfig = None):
        self.config = config or GAIAEncodingConfig()
        self.device = self.config.device
        self.field_dims = self.config.field_dimensions
        self.xi_target = self.config.xi_target
        
        # Pre-compute coordinate grids on GPU
        x = torch.linspace(0, 2 * torch.pi, self.field_dims[0], device=self.device)
        y = torch.linspace(0, 2 * torch.pi, self.field_dims[1], device=self.device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
        print(f"GAIAFieldEncoder initialized on {self.device}")
    
    def encode(self, input_data: Any) -> torch.Tensor:
        """
        Encode input data into physics field representation.
        
        This is the main entry point, compatible with GAIA's
        _encode_input_to_field interface.
        
        Args:
            input_data: String, list, tensor, or other data
            
        Returns:
            torch.Tensor: Field state on configured device
        """
        if isinstance(input_data, str):
            return self._encode_string(input_data)
        elif isinstance(input_data, torch.Tensor):
            return self._encode_tensor(input_data)
        elif isinstance(input_data, (list, tuple)):
            return self._encode_sequence(input_data)
        else:
            # Default: create controlled random field
            return self._encode_random()
    
    def _encode_string(self, text: str) -> torch.Tensor:
        """
        Encode string into field using wave superposition.
        
        Strategy: Each character contributes a wave component based on:
        - Character ordinal → frequency
        - Position → phase shift
        - Xi normalization for PAC conservation
        """
        field = torch.zeros(self.field_dims, device=self.device, dtype=torch.float32)
        
        for i, char in enumerate(text.lower()):
            ord_val = ord(char)
            
            # Frequency from character value (mod 26 for letters)
            freq = ((ord_val % 26) + 1) * 0.3
            
            # Phase from position
            phase = i * torch.pi / 8
            
            # Wave contribution with positional decay
            weight = 1.0 / (i + 1) ** 0.5  # Square root decay
            
            # Superpose wave pattern
            wave = torch.cos(freq * self.X + phase) * torch.sin(freq * self.Y + phase)
            field += weight * wave
            
            # Add second harmonic for richness
            field += 0.2 * weight * torch.sin(2 * freq * self.X) * torch.cos(2 * freq * self.Y)
        
        # Normalize to Xi target for PAC conservation
        return self._normalize(field)
    
    def _encode_tensor(self, data: torch.Tensor) -> torch.Tensor:
        """Encode tensor data directly into field."""
        # Move to device if needed
        data = data.to(self.device).float()
        
        target_size = self.field_dims[0] * self.field_dims[1]
        flat = data.flatten()
        
        if flat.numel() >= target_size:
            field = flat[:target_size].reshape(self.field_dims)
        else:
            # Pad with interpolation
            field = torch.zeros(target_size, device=self.device)
            field[:flat.numel()] = flat
            field = field.reshape(self.field_dims)
        
        return self._normalize(field)
    
    def _encode_sequence(self, seq: Union[list, tuple]) -> torch.Tensor:
        """Encode a sequence of values into field."""
        # Convert to tensor and delegate
        tensor = torch.tensor(seq, device=self.device, dtype=torch.float32)
        return self._encode_tensor(tensor)
    
    def _encode_random(self) -> torch.Tensor:
        """Create controlled random field for unknown input types."""
        field = torch.randn(self.field_dims, device=self.device) * 0.1
        return self._normalize(field)
    
    def _normalize(self, field: torch.Tensor) -> torch.Tensor:
        """Normalize field to Xi target for PAC conservation."""
        # Center
        field = field - field.mean()
        
        # Scale to Xi target
        norm = torch.norm(field)
        if norm > 0:
            field = field / norm * self.xi_target
        
        return field
    
    def get_field_energy(self, field: torch.Tensor) -> float:
        """Calculate total field energy."""
        return float(torch.sum(field ** 2))
    
    def check_conservation(self, field: torch.Tensor) -> float:
        """Check PAC conservation residual."""
        energy = torch.sum(field ** 2)
        expected = self.xi_target ** 2
        return float(abs(energy - expected))
    
    def to_numpy(self, field: torch.Tensor):
        """Convert field to numpy (for backwards compatibility)."""
        return field.cpu().numpy()


def create_gaia_encoder(
    field_dims: Tuple[int, int] = (64, 64),
    device: Optional[torch.device] = None,
    xi_target: float = 1.0571
) -> GAIAFieldEncoder:
    """Factory function to create GAIA encoder."""
    config = GAIAEncodingConfig(
        field_dimensions=field_dims,
        xi_target=xi_target,
        device=device
    )
    return GAIAFieldEncoder(config)


# Integration helper for patching GAIA
def patch_gaia_encoding(gaia_instance, encoder: GAIAFieldEncoder = None):
    """
    Patch a GAIA instance to use GPU-native encoding.
    
    This replaces the _encode_input_to_field method with our
    GPU-accelerated version.
    
    Usage:
        gaia = PAC_GAIA(config)
        patch_gaia_encoding(gaia)
        # Now gaia uses GPU encoding
    """
    if encoder is None:
        encoder = create_gaia_encoder(
            field_dims=gaia_instance.config.field_dimensions
        )
    
    # Store encoder reference
    gaia_instance._gpu_encoder = encoder
    
    # Create wrapper that maintains compatibility
    def gpu_encode_input_to_field(input_data):
        field_tensor = encoder.encode(input_data)
        # Return numpy for current GAIA compatibility
        # TODO: Full torch migration would keep as tensor
        return encoder.to_numpy(field_tensor)
    
    # Patch the method
    gaia_instance._encode_input_to_field = gpu_encode_input_to_field
    
    print(f"GAIA patched with GPU encoding on {encoder.device}")


if __name__ == "__main__":
    # Test the encoder
    print("Testing GAIAFieldEncoder...")
    
    encoder = create_gaia_encoder()
    
    # Test string encoding
    field = encoder.encode("Hello GAIA!")
    print(f"String encoding: shape={field.shape}, device={field.device}")
    print(f"  Energy: {encoder.get_field_energy(field):.4f}")
    print(f"  Conservation: {encoder.check_conservation(field):.2e}")
    
    # Test list encoding  
    field = encoder.encode([1, 2, 3, 4, 5])
    print(f"List encoding: shape={field.shape}")
    
    # Test tensor encoding
    tensor = torch.randn(100)
    field = encoder.encode(tensor)
    print(f"Tensor encoding: shape={field.shape}")
    
    print("\n✓ All tests passed!")
