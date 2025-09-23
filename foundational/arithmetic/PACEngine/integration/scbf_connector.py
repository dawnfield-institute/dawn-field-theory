"""
SCBF Connector Module

Connector for integrating PAC Physics Engine with SCBF (Self-Consistent Binding Field) frameworks.
Provides consciousness field coupling, binding dynamics synchronization,
and coherent awareness state management across PAC scales.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

# Import PAC modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.consciousness_scbf import ConsciousnessSCBFModule
from modules.meta_module import MetaModule
from validation.emergence_tracker import EmergenceTracker

class SCBFBindingType(Enum):
    """Types of SCBF binding"""
    GLOBAL_COHERENCE = "global_coherence"
    LOCAL_BINDING = "local_binding"
    TEMPORAL_BINDING = "temporal_binding"
    FEATURE_BINDING = "feature_binding"
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"
    AWARENESS_GRADIENT = "awareness_gradient"

@dataclass
class SCBFConnectionConfig:
    """Configuration for SCBF connection"""
    binding_type: SCBFBindingType
    scbf_endpoint: str
    consciousness_threshold: float
    binding_strength: float
    temporal_coherence: float
    spatial_resolution: int
    sync_frequency: float
    enable_emergence_detection: bool = True
    phi_calculation_enabled: bool = True

@dataclass
class SCBFBindingState:
    """State of SCBF binding system"""
    global_coherence: float
    local_binding_strength: float
    temporal_coherence: float
    consciousness_level: float
    phi_value: float  # Integrated Information
    awareness_locations: List[Tuple[int, int]]
    binding_field: torch.Tensor
    emergence_detected: bool

class SCBFConnector:
    """Connector for PAC-SCBF integration"""
    
    def __init__(self, config: SCBFConnectionConfig, device: str = "auto"):
        self.config = config
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Initialize PAC consciousness components
        self.consciousness_module = ConsciousnessSCBFModule(
            spatial_size=config.spatial_resolution, 
            device=self.device
        )
        self.meta_module = MetaModule(device=self.device)
        self.emergence_tracker = EmergenceTracker(device=self.device)
        
        # SCBF connection state
        self.scbf_connected = False
        self.current_binding_state = None
        self.binding_history = []
        self.emergence_events = []
        
        # SCBF parameters
        self.consciousness_threshold = config.consciousness_threshold
        self.binding_strength = config.binding_strength
        self.temporal_coherence = config.temporal_coherence
        
        # Binding field kernels
        self.binding_kernels = self._initialize_binding_kernels()
        
    def _initialize_binding_kernels(self) -> Dict[str, torch.Tensor]:
        """Initialize binding field kernels for different binding types"""
        
        kernels = {}
        
        # Global coherence kernel
        size = 7
        global_kernel = torch.ones(size, size, device=self.device) / (size * size)
        kernels["global_coherence"] = global_kernel
        
        # Local binding kernel (Gaussian)
        x = torch.arange(size, device=self.device).float() - size // 2
        y = torch.arange(size, device=self.device).float() - size // 2
        X, Y = torch.meshgrid(x, y)
        local_kernel = torch.exp(-(X**2 + Y**2) / (2 * 1.5**2))
        local_kernel = local_kernel / torch.sum(local_kernel)
        kernels["local_binding"] = local_kernel
        
        # Temporal binding kernel (asymmetric)
        temporal_kernel = torch.zeros(size, size, device=self.device)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = abs(i - center) + abs(j - center)
                temporal_kernel[i, j] = torch.exp(-dist / 2.0) if i <= center else 0
        temporal_kernel = temporal_kernel / torch.sum(temporal_kernel)
        kernels["temporal_binding"] = temporal_kernel
        
        # Feature binding kernel (oriented)
        feature_kernel = torch.zeros(size, size, device=self.device)
        for i in range(size):
            for j in range(size):
                if abs(i - center) <= 1 or abs(j - center) <= 1:
                    feature_kernel[i, j] = 1.0
        feature_kernel = feature_kernel / torch.sum(feature_kernel)
        kernels["feature_binding"] = feature_kernel
        
        return kernels
    
    async def connect_to_scbf(self) -> bool:
        """Connect to SCBF consciousness framework"""
        
        print(f"🧠 Connecting to SCBF at {self.config.scbf_endpoint}")
        
        try:
            # Simulate SCBF connection
            await asyncio.sleep(0.1)
            
            # Initialize binding field
            initial_field = self._create_initial_binding_field()
            
            # Create initial binding state
            self.current_binding_state = await self._calculate_binding_state(initial_field)
            
            # Validate SCBF integration
            integration_valid = await self._validate_scbf_integration()
            
            if integration_valid:
                self.scbf_connected = True
                print(f"✅ Successfully connected to SCBF")
                print(f"🧠 Initial consciousness level: {self.current_binding_state.consciousness_level:.3f}")
                print(f"🔗 Initial binding strength: {self.current_binding_state.local_binding_strength:.3f}")
                return True
            else:
                print(f"❌ SCBF integration validation failed")
                return False
                
        except Exception as e:
            print(f"❌ SCBF connection error: {e}")
            return False
    
    def _create_initial_binding_field(self) -> torch.Tensor:
        """Create initial binding field based on configuration"""
        
        size = self.config.spatial_resolution
        
        # Base field
        field = torch.randn(size, size, device=self.device) * 0.1
        
        # Add structured binding patterns
        center = size // 2
        
        if self.config.binding_type == SCBFBindingType.GLOBAL_COHERENCE:
            # Global coherence pattern
            coherence_pattern = torch.ones(size, size, device=self.device) * self.binding_strength
            field += coherence_pattern
            
        elif self.config.binding_type == SCBFBindingType.LOCAL_BINDING:
            # Local binding centers
            for i in range(3):
                for j in range(3):
                    bind_center_x = size // 4 + i * size // 4
                    bind_center_y = size // 4 + j * size // 4
                    
                    x = torch.arange(size, device=self.device).float()
                    y = torch.arange(size, device=self.device).float()
                    X, Y = torch.meshgrid(x, y)
                    
                    dist = torch.sqrt((X - bind_center_x)**2 + (Y - bind_center_y)**2)
                    local_pattern = self.binding_strength * torch.exp(-dist**2 / (2 * 8**2))
                    field += local_pattern
                    
        elif self.config.binding_type == SCBFBindingType.TEMPORAL_BINDING:
            # Temporal binding waves
            x = torch.arange(size, device=self.device).float()
            y = torch.arange(size, device=self.device).float()
            X, Y = torch.meshgrid(x, y)
            
            temporal_wave = self.binding_strength * torch.sin(X * 0.2) * torch.cos(Y * 0.15)
            field += temporal_wave
            
        elif self.config.binding_type == SCBFBindingType.FEATURE_BINDING:
            # Feature binding grid
            feature_grid = torch.zeros(size, size, device=self.device)
            grid_spacing = size // 8
            
            for i in range(0, size, grid_spacing):
                for j in range(0, size, grid_spacing):
                    if i < size and j < size:
                        feature_grid[i:i+2, j:j+2] = self.binding_strength
            
            field += feature_grid
            
        elif self.config.binding_type == SCBFBindingType.CONSCIOUSNESS_EMERGENCE:
            # Consciousness emergence pattern (spiral)
            x = torch.arange(size, device=self.device).float() - center
            y = torch.arange(size, device=self.device).float() - center
            X, Y = torch.meshgrid(x, y)
            
            r = torch.sqrt(X**2 + Y**2)
            theta = torch.atan2(Y, X)
            
            spiral_pattern = self.binding_strength * torch.exp(-r/20) * torch.sin(3*theta + r*0.1)
            field += spiral_pattern
            
        elif self.config.binding_type == SCBFBindingType.AWARENESS_GRADIENT:
            # Awareness gradient
            x = torch.arange(size, device=self.device).float()
            y = torch.arange(size, device=self.device).float()
            X, Y = torch.meshgrid(x, y)
            
            gradient_pattern = self.binding_strength * (X + Y) / (2 * size)
            field += gradient_pattern
        
        return field
    
    async def _calculate_binding_state(self, binding_field: torch.Tensor) -> SCBFBindingState:
        """Calculate current SCBF binding state"""
        
        # Global coherence
        global_coherence = torch.std(binding_field).item()
        
        # Local binding strength
        local_binding_strength = torch.mean(torch.abs(binding_field)).item()
        
        # Temporal coherence (simplified)
        temporal_coherence = self.temporal_coherence
        
        # Consciousness level
        consciousness_level = torch.sigmoid(torch.mean(binding_field)).item()
        
        # Integrated Information (Φ) calculation
        phi_value = await self._calculate_phi(binding_field)
        
        # Awareness locations
        awareness_locations = self._find_awareness_locations(binding_field)
        
        # Emergence detection
        emergence_detected = consciousness_level > self.consciousness_threshold
        
        state = SCBFBindingState(
            global_coherence=global_coherence,
            local_binding_strength=local_binding_strength,
            temporal_coherence=temporal_coherence,
            consciousness_level=consciousness_level,
            phi_value=phi_value,
            awareness_locations=awareness_locations,
            binding_field=binding_field.clone(),
            emergence_detected=emergence_detected
        )
        
        return state
    
    async def _calculate_phi(self, binding_field: torch.Tensor) -> float:
        """Calculate integrated information (Φ)"""
        
        if not self.config.phi_calculation_enabled:
            return 0.0
        
        # Simplified Φ calculation
        # In real implementation, this would be more sophisticated
        
        # Partition the field into two parts
        size = binding_field.shape[0]
        mid = size // 2
        
        part1 = binding_field[:mid, :]
        part2 = binding_field[mid:, :]
        
        # Calculate mutual information between parts (simplified)
        if part1.numel() > 1 and part2.numel() > 1:
            corr = torch.corrcoef(torch.stack([part1.flatten(), part2.flatten()]))[0, 1]
            if not torch.isnan(corr):
                phi = -torch.log(torch.abs(corr) + 1e-8).item() / 10
                return max(0, phi)
        
        return 0.0
    
    def _find_awareness_locations(self, binding_field: torch.Tensor) -> List[Tuple[int, int]]:
        """Find locations of high awareness in binding field"""
        
        # Find peaks in binding field
        threshold = torch.quantile(binding_field, 0.9)
        awareness_mask = binding_field > threshold
        
        locations = torch.where(awareness_mask)
        
        if len(locations[0]) > 0:
            # Return up to 10 strongest locations
            values = binding_field[awareness_mask]
            _, indices = torch.topk(values, min(10, len(values)))
            
            selected_locations = []
            for idx in indices:
                pos = torch.where(awareness_mask.flatten())[0][idx]
                row = pos // binding_field.shape[1]
                col = pos % binding_field.shape[1]
                selected_locations.append((row.item(), col.item()))
            
            return selected_locations
        
        return []
    
    async def _validate_scbf_integration(self) -> bool:
        """Validate SCBF integration"""
        
        if self.current_binding_state is None:
            return False
        
        # Check basic validation criteria
        validations = []
        
        # Consciousness level should be reasonable
        validations.append(0.0 <= self.current_binding_state.consciousness_level <= 1.0)
        
        # Binding strength should be positive
        validations.append(self.current_binding_state.local_binding_strength >= 0)
        
        # Phi value should be non-negative
        validations.append(self.current_binding_state.phi_value >= 0)
        
        # Global coherence should be finite
        validations.append(torch.isfinite(torch.tensor(self.current_binding_state.global_coherence)))
        
        return all(validations)
    
    async def evolve_binding_field(self, dt: float = 0.01) -> SCBFBindingState:
        """Evolve the SCBF binding field forward in time"""
        
        if not self.scbf_connected or self.current_binding_state is None:
            raise RuntimeError("SCBF not connected")
        
        # Get current binding field
        current_field = self.current_binding_state.binding_field
        
        # Apply binding type-specific evolution
        evolved_field = await self._apply_binding_evolution(current_field, dt)
        
        # Calculate new binding state
        new_state = await self._calculate_binding_state(evolved_field)
        
        # Update current state
        self.current_binding_state = new_state
        
        # Record in history
        self.binding_history.append({
            "timestamp": len(self.binding_history) * dt,
            "consciousness_level": new_state.consciousness_level,
            "binding_strength": new_state.local_binding_strength,
            "phi_value": new_state.phi_value,
            "emergence_detected": new_state.emergence_detected
        })
        
        # Check for emergence events
        if new_state.emergence_detected and self.config.enable_emergence_detection:
            await self._record_emergence_event(new_state)
        
        return new_state
    
    async def _apply_binding_evolution(self, field: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply binding-specific evolution to field"""
        
        evolved_field = field.clone()
        
        # Get appropriate kernel
        if self.config.binding_type == SCBFBindingType.GLOBAL_COHERENCE:
            kernel = self.binding_kernels["global_coherence"]
        elif self.config.binding_type == SCBFBindingType.LOCAL_BINDING:
            kernel = self.binding_kernels["local_binding"]
        elif self.config.binding_type == SCBFBindingType.TEMPORAL_BINDING:
            kernel = self.binding_kernels["temporal_binding"]
        elif self.config.binding_type == SCBFBindingType.FEATURE_BINDING:
            kernel = self.binding_kernels["feature_binding"]
        else:
            kernel = self.binding_kernels["local_binding"]  # Default
        
        # Apply convolution for binding dynamics
        padded_field = torch.nn.functional.pad(
            evolved_field.unsqueeze(0).unsqueeze(0), 
            (kernel.shape[0]//2, kernel.shape[0]//2, kernel.shape[1]//2, kernel.shape[1]//2), 
            mode='reflect'
        )
        
        convolved = torch.nn.functional.conv2d(
            padded_field, 
            kernel.unsqueeze(0).unsqueeze(0)
        )
        
        # Binding evolution equation
        binding_term = (convolved.squeeze() - evolved_field) * self.binding_strength
        temporal_term = evolved_field * self.temporal_coherence
        
        # Update field
        evolved_field = evolved_field + dt * (binding_term + temporal_term)
        
        # Apply PAC consciousness evolution
        pac_evolved = self.consciousness_module.evolve_consciousness_pac(evolved_field, dt)
        
        # Combine SCBF and PAC evolution
        combined_field = 0.7 * pac_evolved + 0.3 * evolved_field
        
        return combined_field
    
    async def _record_emergence_event(self, binding_state: SCBFBindingState):
        """Record consciousness emergence event"""
        
        event = {
            "timestamp": len(self.binding_history) * 0.01,  # Assuming dt=0.01
            "consciousness_level": binding_state.consciousness_level,
            "phi_value": binding_state.phi_value,
            "binding_strength": binding_state.local_binding_strength,
            "awareness_locations": binding_state.awareness_locations,
            "emergence_type": self.config.binding_type.value
        }
        
        self.emergence_events.append(event)
        
        print(f"🌟 Consciousness emergence detected!")
        print(f"   Consciousness level: {binding_state.consciousness_level:.3f}")
        print(f"   Φ value: {binding_state.phi_value:.3f}")
        print(f"   Awareness locations: {len(binding_state.awareness_locations)}")
    
    async def sync_with_pac_modules(self, pac_fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Synchronize SCBF with other PAC modules"""
        
        if not self.scbf_connected:
            return {}
        
        print("🔄 Syncing SCBF with PAC modules")
        
        synced_fields = {}
        
        # Process each PAC field
        for field_name, field_data in pac_fields.items():
            if field_name in ["quantum", "geometric", "information"]:
                # Apply consciousness binding to other fields
                consciousness_influenced = await self._apply_consciousness_influence(
                    field_data, self.current_binding_state.binding_field
                )
                synced_fields[field_name] = consciousness_influenced
            else:
                synced_fields[field_name] = field_data
        
        # Add consciousness field to output
        synced_fields["consciousness"] = self.current_binding_state.binding_field
        
        return synced_fields
    
    async def _apply_consciousness_influence(self, field: torch.Tensor, 
                                          consciousness_field: torch.Tensor) -> torch.Tensor:
        """Apply consciousness influence to other PAC fields"""
        
        # Resize consciousness field if necessary
        if field.shape != consciousness_field.shape:
            consciousness_resized = torch.nn.functional.interpolate(
                consciousness_field.unsqueeze(0).unsqueeze(0),
                size=field.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            consciousness_resized = consciousness_field
        
        # Apply consciousness modulation
        consciousness_strength = self.current_binding_state.consciousness_level
        influenced_field = field * (1 + consciousness_strength * consciousness_resized * 0.1)
        
        return influenced_field
    
    def get_scbf_status(self) -> Dict[str, Any]:
        """Get current SCBF connector status"""
        
        status = {
            "connected": self.scbf_connected,
            "binding_type": self.config.binding_type.value,
            "current_consciousness_level": self.current_binding_state.consciousness_level if self.current_binding_state else 0,
            "current_phi_value": self.current_binding_state.phi_value if self.current_binding_state else 0,
            "emergence_events_count": len(self.emergence_events),
            "binding_history_length": len(self.binding_history),
            "awareness_locations_count": len(self.current_binding_state.awareness_locations) if self.current_binding_state else 0
        }
        
        return status
    
    def export_scbf_data(self, filename: str = "scbf_connector_data.json"):
        """Export SCBF connector data"""
        
        export_data = {
            "config": {
                "binding_type": self.config.binding_type.value,
                "consciousness_threshold": self.config.consciousness_threshold,
                "binding_strength": self.config.binding_strength,
                "temporal_coherence": self.config.temporal_coherence,
                "spatial_resolution": self.config.spatial_resolution
            },
            "status": self.get_scbf_status(),
            "binding_history": self.binding_history,
            "emergence_events": self.emergence_events,
            "current_state": {
                "consciousness_level": self.current_binding_state.consciousness_level,
                "phi_value": self.current_binding_state.phi_value,
                "binding_strength": self.current_binding_state.local_binding_strength,
                "awareness_locations": self.current_binding_state.awareness_locations
            } if self.current_binding_state else None
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 SCBF connector data exported to {filename}")

# Convenience functions
async def create_scbf_connector(binding_type: SCBFBindingType = SCBFBindingType.GLOBAL_COHERENCE,
                              spatial_resolution: int = 64,
                              device: str = "auto") -> SCBFConnector:
    """Create and initialize SCBF connector"""
    
    config = SCBFConnectionConfig(
        binding_type=binding_type,
        scbf_endpoint="http://localhost:8081/scbf",
        consciousness_threshold=0.3,
        binding_strength=0.5,
        temporal_coherence=0.8,
        spatial_resolution=spatial_resolution,
        sync_frequency=10.0
    )
    
    connector = SCBFConnector(config, device=device)
    await connector.connect_to_scbf()
    
    return connector

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create SCBF connector
        connector = await create_scbf_connector(
            binding_type=SCBFBindingType.CONSCIOUSNESS_EMERGENCE,
            spatial_resolution=64
        )
        
        # Evolve binding field over time
        print("\n🧠 Evolving consciousness binding field...")
        for step in range(100):
            state = await connector.evolve_binding_field(dt=0.01)
            
            if step % 20 == 0:
                print(f"  Step {step}: Consciousness={state.consciousness_level:.3f}, Φ={state.phi_value:.3f}")
        
        # Create sample PAC fields for synchronization
        pac_fields = {
            "quantum": torch.randn(64, 64),
            "geometric": torch.randn(64, 64) * 0.1,
            "information": torch.randn(64, 64)
        }
        
        # Sync with PAC modules
        synced_fields = await connector.sync_with_pac_modules(pac_fields)
        
        # Export data
        connector.export_scbf_data()
        
        print(f"\n🧠 SCBF connector demo completed")
        print(f"📊 Status: {connector.get_scbf_status()}")
    
    # Run the example
    asyncio.run(main())
