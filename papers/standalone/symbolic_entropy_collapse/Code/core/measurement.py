"""
Information Measurement Core Module

Provides core measurement algorithms without interpretation or validation claims.
"""

from datetime import datetime
from typing import Dict, Any
from .compression_engine import CompressionEngine
from .text_generator import TextGenerator


class InformationMeasurement:
    """Core measurement system for information amplification studies."""
    
    def __init__(self):
        self.compression_engine = CompressionEngine()
        self.text_generator = TextGenerator()
    
    def measure_system_components(self, input_data: str, model_data: Any, 
                                 output_data: str, overhead_bytes: int = 100) -> Dict[str, Any]:
        """
        Measure all system components and return raw data.
        No interpretation or validation claims.
        """
        # Measure input
        input_measurement = self.compression_engine.measure_compression(input_data)
        
        # Measure model
        if isinstance(model_data, str):
            model_measurement = self.compression_engine.measure_compression(model_data)
        else:
            model_measurement = self.compression_engine.measure_object(model_data)
        
        # Measure output
        output_measurement = self.compression_engine.measure_compression(output_data)
        
        # Calculate raw metrics
        system_capacity = (input_measurement['compressed_size'] + 
                          model_measurement['compressed_size'] + 
                          overhead_bytes)
        
        surplus = output_measurement['compressed_size'] - system_capacity
        ratio = output_measurement['compressed_size'] / system_capacity if system_capacity > 0 else 0
        
        return {
            'input': input_measurement,
            'model': model_measurement,
            'output': output_measurement,
            'system_capacity': system_capacity,
            'surplus_bytes': surplus,
            'amplification_ratio': ratio,
            'overhead_bytes': overhead_bytes,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_text_generation_measurement(self, prompt: str, model_data: Any, 
                                       scale_factor: int = 1) -> Dict[str, Any]:
        """
        Run a complete text generation measurement cycle.
        Returns raw measurements only.
        """
        # Generate output text
        generated_text = self.text_generator.generate_structured_content(prompt, scale_factor)
        
        # Measure all components
        measurements = self.measure_system_components(prompt, model_data, generated_text)
        
        # Add generation metadata
        measurements['generation_metadata'] = {
            'prompt': prompt,
            'scale_factor': scale_factor,
            'output_length': len(generated_text),
            'output_lines': generated_text.count('\n')
        }
        
        return measurements
