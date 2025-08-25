"""
Information Amplification Test Framework
Core measurement and validation system for computational novelty proof.
"""

import os
import json
import hashlib
import pickle
import lzma
import gzip
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime

# Optional imports with fallbacks
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False
    print("Warning: brotli not available, using lzma and gzip only")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, limited environment profiling")


@dataclass
class CompressionResult:
    """Results from compression analysis."""
    algorithm: str
    raw_size: int
    compressed_size: int
    compression_ratio: float
    hash_sha256: str
    timestamp: str


@dataclass
class AmplificationMeasurement:
    """Complete measurement of information amplification."""
    inputs: Dict[str, CompressionResult]
    model_weights: CompressionResult
    outputs: CompressionResult
    environment: Dict[str, Any]
    amplification_ratio: float
    surplus_bytes: int
    is_amplified: bool
    timestamp: str


class OptimalCompressor:
    """Multi-algorithm compression with optimal selection."""
    
    def __init__(self, algorithms: List[str] = None):
        # Use only available algorithms
        available_algorithms = ['lzma2', 'gzip']
        if HAS_BROTLI:
            available_algorithms.append('brotli')
        
        self.algorithms = algorithms or available_algorithms
        self.results_cache = {}
    
    def compress_data(self, data: bytes, algorithm: str = 'auto') -> CompressionResult:
        """Compress data with specified or optimal algorithm."""
        if algorithm == 'auto':
            # Test all algorithms and return best compression
            best_result = None
            best_ratio = float('inf')
            
            for alg in self.algorithms:
                result = self._compress_single(data, alg)
                if result.compression_ratio < best_ratio:
                    best_ratio = result.compression_ratio
                    best_result = result
            
            return best_result
        else:
            return self._compress_single(data, algorithm)
    
    def _compress_single(self, data: bytes, algorithm: str) -> CompressionResult:
        """Compress with single algorithm."""
        raw_size = len(data)
        data_hash = hashlib.sha256(data).hexdigest()
        
        # Cache key for identical data
        cache_key = f"{algorithm}:{data_hash}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        if algorithm == 'lzma2':
            compressed = lzma.compress(data, preset=9)
        elif algorithm == 'brotli' and HAS_BROTLI:
            compressed = brotli.compress(data, quality=11)
        elif algorithm == 'gzip':
            compressed = gzip.compress(data, compresslevel=9)
        else:
            raise ValueError(f"Unknown or unavailable compression algorithm: {algorithm}")
        
        compressed_size = len(compressed)
        compression_ratio = compressed_size / raw_size if raw_size > 0 else 0
        
        result = CompressionResult(
            algorithm=algorithm,
            raw_size=raw_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            hash_sha256=data_hash,
            timestamp=datetime.now().isoformat()
        )
        
        self.results_cache[cache_key] = result
        return result


class InformationMeter:
    """Measures information content of various data types."""
    
    def __init__(self):
        self.compressor = OptimalCompressor()
    
    def measure_file(self, filepath: str) -> CompressionResult:
        """Measure information content of a file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return self.compressor.compress_data(data)
    
    def measure_object(self, obj: Any) -> CompressionResult:
        """Measure information content of Python object."""
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return self.compressor.compress_data(data)
    
    def measure_text(self, text: str, encoding: str = 'utf-8') -> CompressionResult:
        """Measure information content of text."""
        data = text.encode(encoding)
        return self.compressor.compress_data(data)
    
    def measure_numpy_array(self, array: np.ndarray) -> CompressionResult:
        """Measure information content of numpy array."""
        data = array.tobytes()
        return self.compressor.compress_data(data)


class EnvironmentProfiler:
    """Profiles computational environment for complete accounting."""
    
    def profile_environment(self) -> Dict[str, Any]:
        """Create comprehensive environment profile."""
        import platform
        import sys
        
        profile = {
            'platform': {
                'system': platform.system(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': sys.version,
                'executable': sys.executable,
                'path': sys.path[:5]  # First 5 paths only
            },
            'timestamp': datetime.now().isoformat(),
            'random_state': self._capture_random_state()
        }
        
        # Add memory info if psutil is available
        if HAS_PSUTIL:
            profile['memory'] = {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available
            }
        else:
            profile['memory'] = {'note': 'psutil not available'}
        
        return profile
    
    def _capture_random_state(self) -> Dict[str, Any]:
        """Capture current random number generator states."""
        import random
        
        return {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state()[1].tolist()[:10]  # First 10 elements
        }


class InformationAmplificationTest:
    """Main framework for testing computational information amplification."""
    
    def __init__(self, experiment_name: str, output_dir: str = None):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) if output_dir else Path('./results')
        self.output_dir.mkdir(exist_ok=True)
        
        self.meter = InformationMeter()
        self.profiler = EnvironmentProfiler()
        
        self.measurements = {
            'inputs': {},
            'model_weights': None,
            'outputs': None,
            'environment': None
        }
    
    def measure_inputs(self, inputs: Dict[str, Any]) -> Dict[str, CompressionResult]:
        """Measure all input data sources."""
        results = {}
        
        for name, data in inputs.items():
            if isinstance(data, str) and os.path.exists(data):
                # File path
                results[name] = self.meter.measure_file(data)
            elif isinstance(data, str):
                # Text data
                results[name] = self.meter.measure_text(data)
            elif isinstance(data, np.ndarray):
                # Numpy array
                results[name] = self.meter.measure_numpy_array(data)
            else:
                # Generic object
                results[name] = self.meter.measure_object(data)
        
        self.measurements['inputs'] = results
        return results
    
    def measure_model_weights(self, model_path: str = None, model_object: Any = None) -> CompressionResult:
        """Measure model weights/parameters."""
        if model_path:
            result = self.meter.measure_file(model_path)
        elif model_object:
            result = self.meter.measure_object(model_object)
        else:
            raise ValueError("Must provide either model_path or model_object")
        
        self.measurements['model_weights'] = result
        return result
    
    def measure_outputs(self, outputs: Any) -> CompressionResult:
        """Measure generated outputs."""
        if isinstance(outputs, str):
            result = self.meter.measure_text(outputs)
        elif isinstance(outputs, np.ndarray):
            result = self.meter.measure_numpy_array(outputs)
        else:
            result = self.meter.measure_object(outputs)
        
        self.measurements['outputs'] = result
        return result
    
    def calculate_amplification(self, epsilon_bytes: int = 1024) -> AmplificationMeasurement:
        """Calculate information amplification ratio."""
        if not all([self.measurements['inputs'], 
                   self.measurements['model_weights'], 
                   self.measurements['outputs']]):
            raise ValueError("Must measure inputs, model weights, and outputs first")
        
        # Sum all input sizes
        total_input_size = sum(result.compressed_size for result in self.measurements['inputs'].values())
        
        # Model weight size
        model_size = self.measurements['model_weights'].compressed_size
        
        # Output size
        output_size = self.measurements['outputs'].compressed_size
        
        # Calculate amplification
        total_input_and_model = total_input_size + model_size + epsilon_bytes
        surplus_bytes = output_size - total_input_and_model
        amplification_ratio = output_size / total_input_and_model if total_input_and_model > 0 else 0
        is_amplified = surplus_bytes > 0
        
        # Capture environment
        environment = self.profiler.profile_environment()
        self.measurements['environment'] = environment
        
        measurement = AmplificationMeasurement(
            inputs=self.measurements['inputs'],
            model_weights=self.measurements['model_weights'],
            outputs=self.measurements['outputs'],
            environment=environment,
            amplification_ratio=amplification_ratio,
            surplus_bytes=surplus_bytes,
            is_amplified=is_amplified,
            timestamp=datetime.now().isoformat()
        )
        
        return measurement
    
    def save_results(self, measurement: AmplificationMeasurement):
        """Save experimental results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        result_dict = {
            'experiment_name': self.experiment_name,
            'measurement': {
                'inputs': {k: v.__dict__ for k, v in measurement.inputs.items()},
                'model_weights': measurement.model_weights.__dict__,
                'outputs': measurement.outputs.__dict__,
                'environment': measurement.environment,
                'amplification_ratio': measurement.amplification_ratio,
                'surplus_bytes': measurement.surplus_bytes,
                'is_amplified': measurement.is_amplified,
                'timestamp': measurement.timestamp
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def print_summary(self, measurement: AmplificationMeasurement):
        """Print experimental summary."""
        print(f"\n=== Information Amplification Test: {self.experiment_name} ===")
        print(f"Timestamp: {measurement.timestamp}")
        print(f"\nINPUTS:")
        total_input_size = 0
        for name, result in measurement.inputs.items():
            print(f"  {name}: {result.raw_size:,} → {result.compressed_size:,} bytes ({result.compression_ratio:.3f})")
            total_input_size += result.compressed_size
        
        print(f"\nMODEL WEIGHTS:")
        print(f"  {measurement.model_weights.raw_size:,} → {measurement.model_weights.compressed_size:,} bytes ({measurement.model_weights.compression_ratio:.3f})")
        
        print(f"\nOUTPUTS:")
        print(f"  {measurement.outputs.raw_size:,} → {measurement.outputs.compressed_size:,} bytes ({measurement.outputs.compression_ratio:.3f})")
        
        print(f"\nAMPLIFICATION ANALYSIS:")
        print(f"  Total Input Size: {total_input_size:,} bytes")
        print(f"  Model Weight Size: {measurement.model_weights.compressed_size:,} bytes")
        print(f"  Output Size: {measurement.outputs.compressed_size:,} bytes")
        print(f"  Surplus Bytes: {measurement.surplus_bytes:,}")
        print(f"  Amplification Ratio: {measurement.amplification_ratio:.3f}")
        print(f"  Information Amplified: {'YES' if measurement.is_amplified else 'NO'}")
        
        if measurement.is_amplified:
            print(f"\n🎉 INFORMATION AMPLIFICATION DETECTED!")
            print(f"   Generated {measurement.surplus_bytes:,} bytes more information than consumed.")
        else:
            print(f"\n📊 No amplification detected. Output within expected bounds.")


if __name__ == "__main__":
    # Example usage
    test = InformationAmplificationTest("pilot_test")
    
    # Mock data for testing
    inputs = {
        'prompt': "Generate a detailed analysis of quantum computing.",
        'context': "Background information about quantum mechanics..."
    }
    
    # Mock model (in real use, load actual model weights)
    mock_model = {"weights": np.random.random((1000, 1000)), "biases": np.random.random(1000)}
    
    # Mock output (in real use, this would be model-generated)
    mock_output = "A comprehensive 10,000 word analysis of quantum computing with novel insights..."
    
    # Run test
    test.measure_inputs(inputs)
    test.measure_model_weights(model_object=mock_model)
    test.measure_outputs(mock_output)
    
    measurement = test.calculate_amplification()
    test.print_summary(measurement)
    test.save_results(measurement)
