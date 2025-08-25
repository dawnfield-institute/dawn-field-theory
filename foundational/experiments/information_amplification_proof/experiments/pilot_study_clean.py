"""
Pilot Study - Framework Validation

Validates core measurement framework with mock model components.
Generates controlled measurement data.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import InformationAmplificationTest
import numpy as np


def create_mock_language_model():
    """Create a mock language model for controlled testing."""
    vocab_size = 100
    hidden_size = 64
    
    model = {
        'embedding_weights': np.random.random((vocab_size, hidden_size)).astype(np.float32),
        'hidden_weights': np.random.random((hidden_size, hidden_size)).astype(np.float32),
        'output_weights': np.random.random((hidden_size, vocab_size)).astype(np.float32),
        'config': {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'model_type': 'mock_transformer'
        }
    }
    
    return model


def generate_mock_output(input_text: str, model: dict) -> str:
    """Generate structured output using mock generation rules."""
    
    base_concepts = [
        "information theory", "computation", "algorithms", "data structures",
        "complexity theory", "machine learning", "neural networks"
    ]
    
    output_parts = [f"# Analysis: {input_text}\n\n"]
    
    # Generate content sections
    for i, concept in enumerate(base_concepts):
        output_parts.append(f"## {concept.title()}\n")
        output_parts.append(f"The {concept} involves fundamental principles:\n\n")
        
        for j in range(3):
            output_parts.append(f"- Principle {j+1}: Core concept relationship\n")
            output_parts.append(f"- Application: Implementation in system {i+j}\n")
            output_parts.append(f"- Complexity: O(n^{i+j+1}) computational bound\n\n")
    
    # Add data table
    output_parts.append("## Computational Metrics\n\n")
    output_parts.append("| Metric | Value | Complexity |\n")
    output_parts.append("|--------|-------|------------|\n")
    
    for i in range(10):
        metric = f"M_{i+1}"
        value = f"{(i+1) * 1.5:.2f}"
        complexity = f"O(n^{i+1})"
        output_parts.append(f"| {metric} | {value} | {complexity} |\n")
    
    return "".join(output_parts)


def run_pilot_study():
    """Execute pilot study measurement."""
    
    # Initialize framework
    test = InformationAmplificationTest(
        experiment_name="pilot_validation", 
        output_dir="../results"
    )
    
    # Define inputs
    inputs = {
        'user_prompt': "Analyze computational efficiency",
        'system_context': "Technical analysis context",
        'parameters': "default_settings"
    }
    
    # Create mock model
    model = create_mock_language_model()
    
    # Measure components
    test.measure_inputs(inputs)
    test.measure_model_weights(model_object=model)
    
    # Generate and measure output
    generated_output = generate_mock_output(inputs['user_prompt'], model)
    test.measure_outputs(generated_output)
    
    # Calculate measurements
    measurement = test.calculate_amplification(epsilon_bytes=500)
    
    return measurement


def save_pilot_results(measurement, filename="pilot_study_results.json"):
    """Save pilot study results."""
    os.makedirs("../results", exist_ok=True)
    filepath = os.path.join("../results", filename)
    
    # Convert measurement to serializable format
    result_dict = {
        'inputs': {k: v.__dict__ for k, v in measurement.inputs.items()},
        'model_weights': measurement.model_weights.__dict__,
        'outputs': measurement.outputs.__dict__,
        'environment': measurement.environment,
        'amplification_ratio': measurement.amplification_ratio,
        'surplus_bytes': measurement.surplus_bytes,
        'is_amplified': measurement.is_amplified,
        'timestamp': measurement.timestamp
    }
    
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    return filepath


def print_pilot_data(measurement):
    """Print raw pilot study data."""
    print("PILOT STUDY RAW DATA")
    print("=" * 50)
    print(f"Timestamp: {measurement.timestamp}")
    print(f"Model compressed size: {measurement.model_weights.compressed_size} bytes")
    print(f"Output compressed size: {measurement.outputs.compressed_size} bytes")
    print(f"Surplus bytes: {measurement.surplus_bytes}")
    print(f"Amplification ratio: {measurement.amplification_ratio:.3f}")
    print(f"Amplification detected: {measurement.is_amplified}")


if __name__ == "__main__":
    # Run pilot study
    measurement = run_pilot_study()
    
    # Print raw data
    print_pilot_data(measurement)
    
    # Save results
    filepath = save_pilot_results(measurement)
    print(f"\nResults saved to: {filepath}")
    
    print(f"\nPilot study complete.")
