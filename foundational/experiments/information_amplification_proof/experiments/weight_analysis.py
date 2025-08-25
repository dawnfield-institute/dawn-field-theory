"""
Weight Analysis Experiment

Tests whether amplified information is encoded in model weights or emerges
during computation using SEC-based weight interpretation.
"""

import sys
import os
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import InformationAmplificationTest, SECWeightInterpreter


def run_weight_analysis_experiment():
    """
    Analyze model weights to determine if amplified information is pre-encoded
    or emerges during computation.
    """
    
    # Initialize frameworks
    amplification_test = InformationAmplificationTest(
        experiment_name="weight_analysis",
        output_dir="../results"
    )
    
    sec_interpreter = SECWeightInterpreter()
    
    # Test data
    inputs = {
        'user_prompt': "analyze",
        'system_context': "computational analysis context", 
        'parameters': "default"
    }
    
    # Create model
    model = {
        'embedding_weights': [0.1, 0.2, 0.3, 0.4, 0.5],
        'hidden_weights': [0.15, 0.25, 0.35, 0.45],
        'output_weights': [0.12, 0.22, 0.32],
        'config': {'type': 'mock_model'}
    }
    
    # Step 1: Comprehensive weight interpretation
    print("WEIGHT ANALYSIS EXPERIMENT")
    print("=" * 50)
    print("Step 1: Analyzing model weights with SEC framework...")
    
    weight_interpretation = sec_interpreter.interpret_model_weights(model)
    
    print(f"Weight compressed size: {weight_interpretation['compressed_size_bytes']} bytes")
    print(f"Weight compression ratio: {weight_interpretation['compression_ratio']:.2f}")
    print(f"Weight symbolic entropy: {weight_interpretation['entropy_profile']['symbolic_entropy']:.4f}")
    
    if weight_interpretation['sec_metrics']['sec_available']:
        print(f"SEC symbolic collapse: {weight_interpretation['sec_metrics']['symbolic_entropy_collapse']:.4f}")
        print(f"SEC bifractal strength: {weight_interpretation['sec_metrics']['bifractal_lineage_strength']:.4f}")
    else:
        print("SEC metrics not available (TinyCIMM components not found)")
    
    # Step 2: Generate output and measure amplification
    print("\nStep 2: Generating output and measuring amplification...")
    
    amplification_test.measure_inputs(inputs)
    amplification_test.measure_model_weights(model_object=model)
    
    # Generate output using the text generator
    from core import TextGenerator
    generator = TextGenerator()
    generated_output = generator.generate_structured_content(
        prompt=inputs['user_prompt'], 
        scale_factor=5
    )
    
    amplification_test.measure_outputs(generated_output)
    measurement = amplification_test.calculate_amplification(epsilon_bytes=100)
    
    print(f"Output compressed size: {measurement.outputs.compressed_size} bytes")
    print(f"Amplification ratio: {measurement.amplification_ratio:.2f}")
    print(f"Surplus bytes: {measurement.surplus_bytes}")
    
    # Step 3: Compare weight vs output information content
    print("\nStep 3: Comparing weight vs output information...")
    
    comparison = sec_interpreter.compare_weight_vs_output_information(
        weight_interpretation,
        measurement.outputs.compressed_size
    )
    
    print(f"Weight information: {comparison['weight_information_content']} bytes")
    print(f"Output information: {comparison['output_information_content']} bytes")
    print(f"Information surplus: {comparison['information_surplus']} bytes")
    print(f"Output/Weight ratio: {comparison['amplification_ratio']:.2f}")
    print(f"Amplification detected: {comparison['amplification_detected']}")
    
    # Step 4: SEC-based interpretation
    print("\nStep 4: SEC Analysis...")
    sec_analysis = comparison['sec_analysis']
    print(f"Information source: {sec_analysis['information_source']}")
    print(f"Emergent complexity score: {sec_analysis['emergent_complexity']:.4f}")
    
    print(f"\nInterpretation: {comparison['interpretation']}")
    
    # Step 5: Save comprehensive results
    print("\nStep 5: Saving results...")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    combined_results = {
        'experiment_type': 'weight_analysis',
        'timestamp': measurement.timestamp,
        'weight_interpretation': convert_for_json(weight_interpretation),
        'amplification_measurement': {
            'inputs': {k: convert_for_json(v.__dict__) for k, v in measurement.inputs.items()},
            'model_weights': convert_for_json(measurement.model_weights.__dict__),
            'outputs': convert_for_json(measurement.outputs.__dict__),
            'amplification_ratio': float(measurement.amplification_ratio),
            'surplus_bytes': int(measurement.surplus_bytes),
            'is_amplified': bool(measurement.is_amplified)
        },
        'weight_vs_output_comparison': convert_for_json(comparison),
        'conclusions': {
            'weight_encoding_hypothesis': comparison['amplification_ratio'] < 1.1,
            'emergent_generation_hypothesis': comparison['amplification_ratio'] > 2.0,
            'evidence_strength': 'strong' if abs(comparison['amplification_ratio'] - 1.0) > 1.0 else 'moderate'
        }
    }
    
    os.makedirs("../results", exist_ok=True)
    filepath = os.path.join("../results", "weight_analysis_results.json")
    
    with open(filepath, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    return combined_results, filepath


def print_weight_analysis_summary(results):
    """Print a summary of the weight analysis findings."""
    
    print("\nWEIGHT ANALYSIS SUMMARY")
    print("=" * 50)
    
    comparison = results['weight_vs_output_comparison']
    conclusions = results['conclusions']
    
    print(f"Weight Information Content: {comparison['weight_information_content']} bytes")
    print(f"Output Information Content: {comparison['output_information_content']} bytes")
    print(f"Information Amplification Ratio: {comparison['amplification_ratio']:.2f}x")
    
    print("\nHypothesis Testing:")
    if conclusions['weight_encoding_hypothesis']:
        print("✓ Weight Encoding Hypothesis: Output information is primarily encoded in weights")
    else:
        print("✗ Weight Encoding Hypothesis: Output exceeds weight information content")
    
    if conclusions['emergent_generation_hypothesis']:
        print("✓ Emergent Generation Hypothesis: Strong evidence of novel information creation")
    else:
        print("✗ Emergent Generation Hypothesis: Limited evidence of emergent information")
    
    print(f"\nEvidence Strength: {conclusions['evidence_strength'].upper()}")
    
    sec_analysis = comparison['sec_analysis']
    print(f"SEC Information Source Assessment: {sec_analysis['information_source'].upper()}")
    
    print(f"\nFinal Interpretation:")
    print(f"{comparison['interpretation']}")


if __name__ == "__main__":
    # Run the weight analysis experiment
    results, filepath = run_weight_analysis_experiment()
    
    # Print summary
    print_weight_analysis_summary(results)
    
    print(f"\nDetailed results saved to: {filepath}")
    print("\nWeight analysis experiment complete.")
