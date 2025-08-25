"""
Basic Information Amplification Measurement

Measures information amplification using minimal input and model.
Generates raw data without interpretation.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import InformationMeasurement


def run_basic_measurement():
    """Run basic information amplification measurement."""
    
    # Initialize measurement system
    measurement_system = InformationMeasurement()
    
    # Define minimal test components
    input_prompt = "compute"
    
    model_data = {
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "vocab": ["the", "a", "is", "of", "to"]
    }
    
    # Run measurement with large scale factor
    results = measurement_system.run_text_generation_measurement(
        prompt=input_prompt,
        model_data=model_data,
        scale_factor=3  # Generate substantial output
    )
    
    return results


def save_results(results, filename="basic_measurement_results.json"):
    """Save results to file."""
    os.makedirs("../results", exist_ok=True)
    filepath = os.path.join("../results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filepath


def print_raw_data(results):
    """Print raw measurement data."""
    print("RAW MEASUREMENT DATA")
    print("=" * 50)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Input compressed size: {results['input']['compressed_size']} bytes")
    print(f"Model compressed size: {results['model']['compressed_size']} bytes") 
    print(f"Output compressed size: {results['output']['compressed_size']} bytes")
    print(f"System capacity: {results['system_capacity']} bytes")
    print(f"Surplus bytes: {results['surplus_bytes']}")
    print(f"Amplification ratio: {results['amplification_ratio']:.3f}")
    print(f"Output length: {results['generation_metadata']['output_length']} characters")


if __name__ == "__main__":
    # Run measurement
    results = run_basic_measurement()
    
    # Print raw data
    print_raw_data(results)
    
    # Save results
    filepath = save_results(results)
    print(f"\nResults saved to: {filepath}")
    
    print(f"\nMeasurement complete.")