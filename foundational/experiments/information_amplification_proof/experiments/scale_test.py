"""
Scale Test - Information Amplification Scaling Analysis

Tests information amplification across different scale factors.
Generates raw scaling data for analysis.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import InformationMeasurement


def run_scale_test():
    """Run information amplification across multiple scales."""
    
    measurement_system = InformationMeasurement()
    
    # Test parameters
    input_prompt = "analyze"
    model_data = {"params": [0.1, 0.2], "vocab": ["a", "b"]}
    
    scale_factors = [1, 2, 3, 4, 5]
    results = []
    
    for scale in scale_factors:
        result = measurement_system.run_text_generation_measurement(
            prompt=input_prompt,
            model_data=model_data,
            scale_factor=scale
        )
        
        result['scale_factor'] = scale
        results.append(result)
    
    return {
        'scale_test_results': results,
        'input_prompt': input_prompt,
        'model_data': model_data
    }


def save_results(results, filename="scale_test_results.json"):
    """Save scale test results."""
    os.makedirs("../results", exist_ok=True)
    filepath = os.path.join("../results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filepath


def print_scale_data(results):
    """Print raw scaling data."""
    print("SCALE TEST RAW DATA")
    print("=" * 50)
    print(f"Input: '{results['input_prompt']}'")
    print(f"Model: {results['model_data']}")
    print()
    print("Scale | Input | Model | Output | Capacity | Surplus | Ratio")
    print("-" * 60)
    
    for result in results['scale_test_results']:
        scale = result['scale_factor']
        input_size = result['input']['compressed_size']
        model_size = result['model']['compressed_size']
        output_size = result['output']['compressed_size']
        capacity = result['system_capacity']
        surplus = result['surplus_bytes']
        ratio = result['amplification_ratio']
        
        print(f"{scale:5d} | {input_size:5d} | {model_size:5d} | {output_size:6d} | {capacity:8d} | {surplus:7d} | {ratio:5.2f}")


if __name__ == "__main__":
    # Run scale test
    results = run_scale_test()
    
    # Print raw data
    print_scale_data(results)
    
    # Save results
    filepath = save_results(results)
    print(f"\nResults saved to: {filepath}")
    
    print(f"\nScale test complete.")
