"""
Dawn Models API Integration

Integration module for connecting the PAC Physics Engine with Dawn Models ecosystem.
Provides unified model interfaces, field theory integration, and consciousness modeling
capabilities across the Dawn Field Theory framework.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
import aiohttp
from pathlib import Path

# Import PAC modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.pac_kernel import PACConservationKernel
from modules.meta_module import MetaModule
from validation.cross_scale_validator import CrossScaleValidator
from modules.consciousness_scbf import ConsciousnessSCBFModule

class DawnModelType(Enum):
    """Types of Dawn models"""
    FIELD_THEORY = "field_theory"
    CONSCIOUSNESS_MODEL = "consciousness_model"
    INFORMATION_DYNAMICS = "information_dynamics"
    EMERGENCE_MODEL = "emergence_model"
    UNIFIED_FIELD = "unified_field"
    METAMODEL = "metamodel"

@dataclass
class DawnModelConfig:
    """Configuration for Dawn model integration"""
    model_type: DawnModelType
    model_name: str
    model_endpoint: str
    api_version: str
    authentication_token: Optional[str]
    field_mappings: Dict[str, str]
    sync_mode: str  # "real_time", "batch", "on_demand"
    conservation_validation: bool = True

@dataclass
class DawnModelResponse:
    """Response from Dawn model API"""
    model_name: str
    response_data: Dict[str, Any]
    field_updates: Dict[str, torch.Tensor]
    conservation_status: bool
    metadata: Dict[str, Any]
    timestamp: float

class DawnModelsAPI:
    """API integration for Dawn Models ecosystem"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # Initialize PAC components
        self.pac_kernel = PACConservationKernel(device=self.device)
        self.meta_module = MetaModule(device=self.device)
        self.validator = CrossScaleValidator(device=self.device)
        self.consciousness_module = ConsciousnessSCBFModule(64, device=self.device)
        
        # Dawn Models connection state
        self.connected_models: Dict[str, DawnModelConfig] = {}
        self.model_sessions = {}
        self.api_cache = {}
        
        # Field mapping registry
        self.field_registry = self._initialize_field_registry()
        
        # Model response history
        self.response_history = []
        
    def _initialize_field_registry(self) -> Dict[str, Dict[str, str]]:
        """Initialize field mapping registry for different Dawn models"""
        
        registry = {}
        
        # Field Theory Model mappings
        registry["field_theory"] = {
            "quantum_field": "/fields/quantum/amplitude",
            "geometric_field": "/fields/geometric/metric",
            "information_field": "/fields/information/density",
            "consciousness_field": "/fields/consciousness/binding",
            "unified_field": "/fields/unified/state"
        }
        
        # Consciousness Model mappings
        registry["consciousness_model"] = {
            "awareness_field": "/consciousness/awareness",
            "binding_field": "/consciousness/binding",
            "emergence_field": "/consciousness/emergence",
            "phi_field": "/consciousness/integrated_information",
            "coherence_field": "/consciousness/coherence"
        }
        
        # Information Dynamics mappings
        registry["information_dynamics"] = {
            "information_density": "/information/density",
            "information_flow": "/information/flow",
            "entropy_field": "/information/entropy",
            "amplification_field": "/information/amplification",
            "complexity_field": "/information/complexity"
        }
        
        # Emergence Model mappings
        registry["emergence_model"] = {
            "emergence_potential": "/emergence/potential",
            "cascade_field": "/emergence/cascade",
            "phase_transition": "/emergence/phase_transition",
            "critical_points": "/emergence/critical_points",
            "emergence_gradients": "/emergence/gradients"
        }
        
        # Unified Field mappings
        registry["unified_field"] = {
            "unified_state": "/unified/state",
            "field_tensor": "/unified/tensor",
            "conservation_field": "/unified/conservation",
            "symmetry_field": "/unified/symmetry",
            "interaction_field": "/unified/interaction"
        }
        
        return registry
    
    async def connect_to_dawn_model(self, config: DawnModelConfig) -> bool:
        """Connect to a specific Dawn model"""
        
        print(f"🌅 Connecting to Dawn model: {config.model_name}")
        print(f"🔗 Type: {config.model_type.value}")
        print(f"📡 Endpoint: {config.model_endpoint}")
        
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=10)
            session = aiohttp.ClientSession(connector=connector)
            
            # Test connection
            headers = {}
            if config.authentication_token:
                headers["Authorization"] = f"Bearer {config.authentication_token}"
            
            async with session.get(
                f"{config.model_endpoint}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    # Store connection
                    self.connected_models[config.model_name] = config
                    self.model_sessions[config.model_name] = session
                    
                    # Initialize field mappings
                    await self._initialize_model_fields(config)
                    
                    print(f"✅ Successfully connected to {config.model_name}")
                    return True
                else:
                    print(f"❌ Connection failed: HTTP {response.status}")
                    await session.close()
                    return False
                    
        except Exception as e:
            print(f"❌ Connection error: {e}")
            if 'session' in locals():
                await session.close()
            return False
    
    async def _initialize_model_fields(self, config: DawnModelConfig):
        """Initialize field mappings for the connected model"""
        
        model_type = config.model_type.value
        if model_type in self.field_registry:
            print(f"🗺️ Initialized {len(self.field_registry[model_type])} field mappings")
        else:
            print(f"⚠️ No field mappings found for model type: {model_type}")
    
    async def send_pac_fields_to_dawn(self, model_name: str, 
                                     pac_fields: Dict[str, torch.Tensor]) -> DawnModelResponse:
        """Send PAC fields to Dawn model"""
        
        if model_name not in self.connected_models:
            raise ValueError(f"Model {model_name} not connected")
        
        config = self.connected_models[model_name]
        session = self.model_sessions[model_name]
        
        print(f"📤 Sending {len(pac_fields)} PAC fields to {model_name}")
        
        # Convert PAC fields to Dawn model format
        dawn_payload = await self._convert_pac_to_dawn_format(pac_fields, config)
        
        # Prepare API request
        headers = {"Content-Type": "application/json"}
        if config.authentication_token:
            headers["Authorization"] = f"Bearer {config.authentication_token}"
        
        try:
            async with session.post(
                f"{config.model_endpoint}/api/{config.api_version}/fields/update",
                json=dawn_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Process response
                    dawn_response = await self._process_dawn_response(
                        response_data, config, model_name
                    )
                    
                    # Cache response
                    self.api_cache[f"{model_name}_latest"] = dawn_response
                    self.response_history.append(dawn_response)
                    
                    print(f"✅ Successfully sent fields to {model_name}")
                    return dawn_response
                    
                else:
                    error_text = await response.text()
                    print(f"❌ API error: HTTP {response.status} - {error_text}")
                    raise RuntimeError(f"Dawn model API error: {response.status}")
                    
        except Exception as e:
            print(f"❌ Send error: {e}")
            raise
    
    async def _convert_pac_to_dawn_format(self, pac_fields: Dict[str, torch.Tensor], 
                                        config: DawnModelConfig) -> Dict[str, Any]:
        """Convert PAC fields to Dawn model format"""
        
        dawn_payload = {
            "model_type": config.model_type.value,
            "api_version": config.api_version,
            "fields": {},
            "metadata": {
                "source": "PAC_Physics_Engine",
                "conservation_validation": config.conservation_validation,
                "device": str(self.device)
            }
        }
        
        # Get field mappings for this model type
        model_type = config.model_type.value
        if model_type in self.field_registry:
            field_mappings = self.field_registry[model_type]
        else:
            field_mappings = {}
        
        # Convert each PAC field
        for pac_field_name, pac_field_data in pac_fields.items():
            # Find corresponding Dawn field name
            dawn_field_name = None
            for dawn_name, dawn_path in field_mappings.items():
                if pac_field_name.lower() in dawn_name.lower() or dawn_name.lower() in pac_field_name.lower():
                    dawn_field_name = dawn_name
                    break
            
            if dawn_field_name is None:
                dawn_field_name = f"pac_{pac_field_name}"
            
            # Convert tensor to list for JSON serialization
            field_data = {
                "data": pac_field_data.cpu().numpy().tolist(),
                "shape": list(pac_field_data.shape),
                "dtype": str(pac_field_data.dtype),
                "device": str(pac_field_data.device),
                "pac_conservation_sum": torch.sum(pac_field_data).item(),
                "field_statistics": {
                    "mean": torch.mean(pac_field_data).item(),
                    "std": torch.std(pac_field_data).item(),
                    "min": torch.min(pac_field_data).item(),
                    "max": torch.max(pac_field_data).item()
                }
            }
            
            dawn_payload["fields"][dawn_field_name] = field_data
        
        return dawn_payload
    
    async def _process_dawn_response(self, response_data: Dict[str, Any], 
                                   config: DawnModelConfig, model_name: str) -> DawnModelResponse:
        """Process response from Dawn model"""
        
        # Extract field updates
        field_updates = {}
        if "fields" in response_data:
            for field_name, field_data in response_data["fields"].items():
                if "data" in field_data:
                    # Convert back to tensor
                    tensor_data = torch.tensor(
                        field_data["data"], 
                        device=self.device,
                        dtype=torch.float32
                    )
                    field_updates[field_name] = tensor_data
        
        # Validate conservation if enabled
        conservation_status = True
        if config.conservation_validation and field_updates:
            conservation_status = await self._validate_dawn_conservation(field_updates)
        
        # Create response object
        dawn_response = DawnModelResponse(
            model_name=model_name,
            response_data=response_data,
            field_updates=field_updates,
            conservation_status=conservation_status,
            metadata=response_data.get("metadata", {}),
            timestamp=asyncio.get_event_loop().time()
        )
        
        return dawn_response
    
    async def _validate_dawn_conservation(self, field_updates: Dict[str, torch.Tensor]) -> bool:
        """Validate PAC conservation in Dawn model response"""
        
        conservation_errors = []
        
        for field_name, field_data in field_updates.items():
            # Apply PAC conservation validation
            conserved_field = self.pac_kernel.apply_pac_conservation(field_data)
            conservation_error = torch.abs(torch.sum(field_data) - torch.sum(conserved_field))
            conservation_errors.append(conservation_error.item())
        
        # Check if all fields maintain conservation
        max_error = max(conservation_errors) if conservation_errors else 0
        conservation_valid = max_error < 1e-10
        
        if not conservation_valid:
            print(f"⚠️ Conservation violation detected: max_error={max_error:.2e}")
        
        return conservation_valid
    
    async def request_dawn_computation(self, model_name: str, 
                                     computation_type: str,
                                     parameters: Dict[str, Any]) -> DawnModelResponse:
        """Request specific computation from Dawn model"""
        
        if model_name not in self.connected_models:
            raise ValueError(f"Model {model_name} not connected")
        
        config = self.connected_models[model_name]
        session = self.model_sessions[model_name]
        
        print(f"🔬 Requesting {computation_type} computation from {model_name}")
        
        # Prepare computation request
        payload = {
            "computation_type": computation_type,
            "parameters": parameters,
            "model_type": config.model_type.value,
            "api_version": config.api_version
        }
        
        headers = {"Content-Type": "application/json"}
        if config.authentication_token:
            headers["Authorization"] = f"Bearer {config.authentication_token}"
        
        try:
            async with session.post(
                f"{config.model_endpoint}/api/{config.api_version}/compute",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Process computation response
                    dawn_response = await self._process_dawn_response(
                        response_data, config, model_name
                    )
                    
                    print(f"✅ Computation completed: {computation_type}")
                    return dawn_response
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Computation error: HTTP {response.status} - {error_text}")
                    raise RuntimeError(f"Dawn computation error: {response.status}")
                    
        except Exception as e:
            print(f"❌ Computation request error: {e}")
            raise
    
    async def sync_with_dawn_ecosystem(self, pac_fields: Dict[str, torch.Tensor]) -> Dict[str, DawnModelResponse]:
        """Synchronize with multiple Dawn models simultaneously"""
        
        print(f"🌐 Syncing with {len(self.connected_models)} Dawn models")
        
        # Create sync tasks for all connected models
        sync_tasks = []
        for model_name in self.connected_models:
            task = self.send_pac_fields_to_dawn(model_name, pac_fields)
            sync_tasks.append((model_name, task))
        
        # Execute sync tasks concurrently
        sync_results = {}
        for model_name, task in sync_tasks:
            try:
                response = await task
                sync_results[model_name] = response
                print(f"✅ {model_name}: Sync successful")
            except Exception as e:
                print(f"❌ {model_name}: Sync failed - {e}")
                continue
        
        print(f"🎯 Ecosystem sync completed: {len(sync_results)}/{len(self.connected_models)} successful")
        return sync_results
    
    async def get_dawn_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of specific Dawn model"""
        
        if model_name not in self.connected_models:
            return {"error": "Model not connected"}
        
        config = self.connected_models[model_name]
        session = self.model_sessions[model_name]
        
        headers = {}
        if config.authentication_token:
            headers["Authorization"] = f"Bearer {config.authentication_token}"
        
        try:
            async with session.get(
                f"{config.model_endpoint}/api/{config.api_version}/status",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    status_data = await response.json()
                    return status_data
                else:
                    return {"error": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def disconnect_from_dawn_model(self, model_name: str):
        """Disconnect from specific Dawn model"""
        
        if model_name in self.model_sessions:
            await self.model_sessions[model_name].close()
            del self.model_sessions[model_name]
        
        if model_name in self.connected_models:
            del self.connected_models[model_name]
        
        print(f"🔌 Disconnected from {model_name}")
    
    async def disconnect_all(self):
        """Disconnect from all Dawn models"""
        
        for model_name in list(self.connected_models.keys()):
            await self.disconnect_from_dawn_model(model_name)
        
        print("🔌 Disconnected from all Dawn models")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get overall API status"""
        
        status = {
            "connected_models": len(self.connected_models),
            "model_names": list(self.connected_models.keys()),
            "active_sessions": len(self.model_sessions),
            "response_history_length": len(self.response_history),
            "cache_size": len(self.api_cache),
            "field_registry_types": list(self.field_registry.keys())
        }
        
        return status
    
    def export_api_logs(self, filename: str = "dawn_models_api_logs.json"):
        """Export API interaction logs"""
        
        export_data = {
            "status": self.get_api_status(),
            "connected_models": {
                name: {
                    "model_type": config.model_type.value,
                    "endpoint": config.model_endpoint,
                    "api_version": config.api_version,
                    "sync_mode": config.sync_mode
                }
                for name, config in self.connected_models.items()
            },
            "response_history": [
                {
                    "model_name": response.model_name,
                    "timestamp": response.timestamp,
                    "conservation_status": response.conservation_status,
                    "field_count": len(response.field_updates),
                    "metadata": response.metadata
                }
                for response in self.response_history[-100:]  # Last 100 responses
            ],
            "field_registry": self.field_registry
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Dawn Models API logs exported to {filename}")

# Convenience functions
async def connect_to_dawn_ecosystem(device: str = "auto") -> DawnModelsAPI:
    """Connect to the complete Dawn Models ecosystem"""
    
    api = DawnModelsAPI(device=device)
    
    # Define standard Dawn model configurations
    dawn_configs = [
        DawnModelConfig(
            model_type=DawnModelType.FIELD_THEORY,
            model_name="dawn_field_theory",
            model_endpoint="http://localhost:8100/field-theory",
            api_version="v1",
            authentication_token=None,
            field_mappings={},
            sync_mode="real_time"
        ),
        DawnModelConfig(
            model_type=DawnModelType.CONSCIOUSNESS_MODEL,
            model_name="dawn_consciousness",
            model_endpoint="http://localhost:8101/consciousness",
            api_version="v1",
            authentication_token=None,
            field_mappings={},
            sync_mode="real_time"
        ),
        DawnModelConfig(
            model_type=DawnModelType.INFORMATION_DYNAMICS,
            model_name="dawn_information",
            model_endpoint="http://localhost:8102/information",
            api_version="v1",
            authentication_token=None,
            field_mappings={},
            sync_mode="batch"
        )
    ]
    
    # Connect to all models (mock connections for demo)
    for config in dawn_configs:
        try:
            # In real implementation, these would be actual API connections
            print(f"🌅 Mock connecting to {config.model_name}")
            api.connected_models[config.model_name] = config
            print(f"✅ Connected to {config.model_name}")
        except Exception as e:
            print(f"❌ Failed to connect to {config.model_name}: {e}")
    
    return api

if __name__ == "__main__":
    # Example usage
    async def main():
        # Connect to Dawn ecosystem
        api = await connect_to_dawn_ecosystem()
        
        # Create sample PAC fields
        pac_fields = {
            "quantum": torch.randn(32, 32),
            "geometric": torch.randn(32, 32) * 0.1,
            "information": torch.randn(32, 32),
            "consciousness": torch.randn(32, 32) * 0.5
        }
        
        # Mock synchronization (since we don't have real Dawn model endpoints)
        print("\n🌐 Mock ecosystem synchronization...")
        for model_name in api.connected_models:
            print(f"  📤 Mock sending fields to {model_name}")
            # In real implementation: response = await api.send_pac_fields_to_dawn(model_name, pac_fields)
            print(f"  ✅ Mock response from {model_name}")
        
        # Export logs
        api.export_api_logs()
        
        print(f"\n🌅 Dawn Models API demo completed")
        print(f"📊 Status: {api.get_api_status()}")
        
        # Cleanup
        await api.disconnect_all()
    
    # Run the example
    asyncio.run(main())
