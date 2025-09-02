"""
GAIA v2.0 - Core Coordinator
Main orchestration module for GAIA architecture

TORCH ONLY - NO NUMPY
This implementation uses PyTorch with CUDA acceleration exclusively.
All tensor operations are performed on GPU when available.

This module coordinates the Field Engine and Collapse Core to provide
a unified cognitive architecture interface.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Set device for CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GAIA Coordinator using device: {device}")

# Import shared data structures
try:
    from .data_structures import FieldState, CollapseEvent, SymbolicStructure, GAIAState
except ImportError:
    # Fallback for direct execution
    from data_structures import FieldState, CollapseEvent, SymbolicStructure, GAIAState

# Import modules (these should not import each other)
try:
    from .field_engine import FieldEngine
    from .collapse_core import CollapseCore
except ImportError:
    # Fallback for direct execution
    from field_engine import FieldEngine
    from collapse_core import CollapseCore


class GAIA:
    """
    Main GAIA v2.0 Coordinator
    
    Orchestrates Field Engine and Collapse Core to provide
    physics-informed AGI capabilities.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, ...] = (32, 32),
                 collapse_threshold: float = 0.7,
                 geometric_guidance: bool = True,
                 thermodynamic_optimization: bool = True,
                 scbf_logging: bool = True):
        """
        Initialize GAIA system
        
        Args:
            field_shape: Shape of entropy field
            collapse_threshold: Threshold for collapse triggering
            geometric_guidance: Enable geometric collapse guidance
            thermodynamic_optimization: Enable thermodynamic optimization
            scbf_logging: Enable SCBF integration
        """
        self.field_shape = field_shape
        
        # Initialize core modules
        self.field_engine = FieldEngine(
            field_shape=field_shape,
            collapse_threshold=collapse_threshold,
            scbf_logging=scbf_logging
        )
        
        self.collapse_core = CollapseCore(
            field_shape=field_shape,
            geometric_guidance=geometric_guidance,
            thermodynamic_optimization=thermodynamic_optimization,
            scbf_logging=scbf_logging
        )
        
        logging.info(f"GAIA v2.0 initialized with field shape {field_shape}")
    
    def process_input(self, 
                     input_data: Union[torch.Tensor, str, List],
                     input_type: str = "auto") -> GAIAState:
        """
        Process input through GAIA architecture
        
        Args:
            input_data: Input to process (can be text, numbers, images, etc.)
            input_type: Type of input ("text", "numeric", "image", "auto")
            
        Returns:
            Current GAIA state after processing
        """
        # Convert input to field representation
        field_input = self._preprocess_input(input_data, input_type)
        
        # Inject into field engine
        self.field_engine.inject_stimulus(field_input, "energy")
        
        # Process through architecture
        return self.step()
    
    def step(self) -> GAIAState:
        """Execute one step of GAIA processing"""
        # Step field engine
        collapse_event = self.field_engine.step()
        
        # Process any collapse through collapse core
        new_structure = None
        if collapse_event:
            field_state = self.field_engine.get_field_state()
            new_structure = self.collapse_core.process_collapse_event(collapse_event, field_state)
        
        # Return current state
        return self.get_state()
    
    def run_sequence(self, 
                    input_sequence: List[Any],
                    steps_per_input: int = 5) -> List[GAIAState]:
        """
        Process a sequence of inputs
        
        Args:
            input_sequence: List of inputs to process
            steps_per_input: Number of processing steps per input
            
        Returns:
            List of GAIA states after each input
        """
        states = []
        
        for input_data in input_sequence:
            # Process input
            state = self.process_input(input_data)
            
            # Additional processing steps
            for _ in range(steps_per_input - 1):
                state = self.step()
            
            states.append(state)
        
        return states
    
    def get_state(self) -> GAIAState:
        """Get current GAIA state"""
        field_state = self.field_engine.get_field_state()
        structures = self.collapse_core.get_symbolic_structures()
        
        # Calculate cognitive load
        cognitive_load = field_state.field_pressure * field_state.collapse_likelihood
        
        return GAIAState(
            field_state=field_state,
            symbolic_structures=structures,
            timestep=self.field_engine.timestep,
            total_collapses=len(self.collapse_core.collapse_history),
            cognitive_load=cognitive_load
        )
    
    def _preprocess_input(self, input_data: Any, input_type: str) -> torch.Tensor:
        """Convert input to field-compatible format"""
        if input_type == "auto":
            input_type = self._detect_input_type(input_data)
        
        if input_type == "text":
            return self._text_to_field(input_data)
        elif input_type == "numeric":
            return self._numeric_to_field(input_data)
        elif input_type == "image":
            return self._image_to_field(input_data)
        else:
            # Default: try to convert to numpy array
            if isinstance(input_data, torch.Tensor):
                return input_data.to(device)
            else:
                return torch.tensor(input_data, dtype=torch.float32, device=device)
    
    def _detect_input_type(self, input_data: Any) -> str:
        """Automatically detect input type"""
        if isinstance(input_data, str):
            return "text"
        elif isinstance(input_data, (int, float)):
            return "numeric"
        elif isinstance(input_data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in input_data):
                return "numeric"
            else:
                return "text"
        elif isinstance(input_data, torch.Tensor):
            if len(input_data.shape) > 1:
                return "image"
            else:
                return "numeric"
        else:
            return "numeric"
    
    def _text_to_field(self, text: str) -> torch.Tensor:
        """Convert text to field representation"""
        # Simple character encoding
        char_values = [ord(c) / 256.0 for c in text]
        
        # Pad or truncate to fit field
        field_size = torch.prod(torch.tensor(self.field_shape))
        if len(char_values) > field_size:
            char_values = char_values[:field_size]
        else:
            char_values.extend([0.0] * (field_size - len(char_values)))
        
        return torch.tensor(char_values, dtype=torch.float32, device=device).reshape(self.field_shape)
    
    def _numeric_to_field(self, numeric_data: Union[float, int, List, torch.Tensor]) -> torch.Tensor:
        """Convert numeric data to field representation"""
        if isinstance(numeric_data, (int, float)):
            numeric_data = [numeric_data]
        
        if isinstance(numeric_data, list):
            numeric_data = torch.tensor(numeric_data, dtype=torch.float32, device=device)
        elif isinstance(numeric_data, torch.Tensor):
            numeric_data = numeric_data.to(device)
        
        # Normalize to [0, 1] range
        if torch.max(numeric_data) != torch.min(numeric_data):
            numeric_data = (numeric_data - torch.min(numeric_data)) / (torch.max(numeric_data) - torch.min(numeric_data))
        
        # Reshape to field shape
        field_size = torch.prod(torch.tensor(self.field_shape))
        flat_data = numeric_data.flatten()
        
        if len(flat_data) > field_size:
            flat_data = flat_data[:field_size]
        else:
            padding_size = field_size - len(flat_data)
            padding = torch.zeros(padding_size, device=device, dtype=torch.float32)
            flat_data = torch.cat([flat_data, padding])
        
        return flat_data.reshape(self.field_shape)
    
    def _image_to_field(self, image_data: torch.Tensor) -> torch.Tensor:
        """Convert image data to field representation"""
        # Resize image to field shape if needed
        if image_data.shape != self.field_shape:
            # Simple resizing (would use proper image resizing in real implementation)
            # Pytorch resize - use interpolation for resizing
            import torch.nn.functional as F
            # Add batch and channel dims for interpolation
            if len(image_data.shape) == 2:
                image_4d = image_data.unsqueeze(0).unsqueeze(0)
            else:
                image_4d = image_data.unsqueeze(0)
            
            # Resize using bilinear interpolation
            resized_4d = F.interpolate(image_4d, size=self.field_shape, mode='bilinear', align_corners=False)
            resized = resized_4d.squeeze()
        else:
            resized = image_data
        
        # Normalize
        if torch.max(resized) != torch.min(resized):
            resized = (resized - torch.min(resized)) / (torch.max(resized) - torch.min(resized))
        
        return resized.to(torch.float32)
    
    def get_symbolic_summary(self) -> Dict[str, Any]:
        """Get summary of symbolic structures"""
        structures = self.collapse_core.get_symbolic_structures()
        
        if not structures:
            return {"total_structures": 0}
        
        # Analyze structures
        collapse_types = [s.structure_id.split('_')[0] for s in structures]
        type_counts = {}
        for ct in collapse_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
        
        total_entropy = sum(s.entropy_signature for s in structures)
        total_cost = sum(s.thermodynamic_cost for s in structures)
        
        return {
            "total_structures": len(structures),
            "collapse_type_counts": type_counts,
            "total_entropy_crystallized": total_entropy,
            "total_thermodynamic_cost": total_cost,
            "latest_structure": structures[-1].structure_id if structures else None
        }
    
    def reset(self) -> None:
        """Reset GAIA to initial state"""
        self.field_engine.reset()
        self.collapse_core.reset()
        logging.info("GAIA system reset to initial state")


# Initialize Python package
def __init__():
    pass


if __name__ == "__main__":
    # Simple test of GAIA system
    logging.basicConfig(level=logging.INFO)
    
    print("Testing GAIA v2.0 system...")
    
    # Create GAIA instance
    gaia = GAIA(field_shape=(16, 16), collapse_threshold=0.5)
    
    # Test with different input types
    test_inputs = [
        "hello world",
        [1, 2, 3, 4, 5],
        torch.rand((4, 4), device=device),
        42
    ]
    
    print("\nProcessing test inputs...")
    for i, input_data in enumerate(test_inputs):
        print(f"\n--- Input {i+1}: {type(input_data).__name__} ---")
        state = gaia.process_input(input_data)
        
        print(f"Timestep: {state.timestep}")
        print(f"Cognitive load: {state.cognitive_load:.3f}")
        print(f"Total collapses: {state.total_collapses}")
        
        summary = gaia.get_symbolic_summary()
        print(f"Symbolic summary: {summary}")
    
    # Test sequence processing
    print("\n--- Sequence Processing ---")
    sequence = [1, 2, 3, 5, 8, 13, 21]  # Fibonacci
    states = gaia.run_sequence(sequence, steps_per_input=3)
    
    print(f"Processed sequence of {len(sequence)} inputs")
    print(f"Final state: {states[-1].total_collapses} collapses, {len(states[-1].symbolic_structures)} structures")
    
    final_summary = gaia.get_symbolic_summary()
    print(f"Final symbolic summary: {final_summary}")
