#!/usr/bin/env python3
"""
Quick test to verify all mathematical experiments can start and run successfully
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiment import run_experiment, TinyCIMMEuler

def test_all_experiments_quick():
    """Test all experiments with just 100 steps each to verify they work"""
    test_cases = [
        ("prime_deltas", {
            "hidden_size": 28,
            "math_memory_size": 35,
            "adaptation_steps": 45,
            "pattern_decay": 0.99,
            "experiment_type": "enhanced_prime_recognition"
        }),
        ("fibonacci_ratios", {
            "hidden_size": 24, 
            "math_memory_size": 25, 
            "pattern_decay": 0.96,
            "adaptation_steps": 35,
            "experiment_type": "convergence_test"
        }),
        ("polynomial_sequence", {
            "hidden_size": 22, 
            "math_memory_size": 20,
            "pattern_decay": 0.94,
            "adaptation_steps": 30,
            "experiment_type": "polynomial_analysis"
        }),
        ("recursive_sequence", {
            "hidden_size": 26,
            "math_memory_size": 30,
            "pattern_decay": 0.97,
            "adaptation_steps": 40,
            "experiment_type": "recursive_patterns"
        }),
        ("algebraic_sequence", {
            "hidden_size": 20,
            "math_memory_size": 18,
            "adaptation_steps": 25,
            "experiment_type": "algebraic_reasoning"
        })
    ]
    
    successful_experiments = []
    failed_experiments = []
    
    for test_name, model_kwargs in test_cases:
        print(f"\n=== Quick Test: {test_name} (100 steps) ===")
        
        try:
            # Run for just 100 steps to test
            run_experiment(TinyCIMMEuler, signal=test_name, steps=100, **model_kwargs)
            print(f"SUCCESS: {test_name} completed 100 steps successfully")
            successful_experiments.append(test_name)
        except Exception as e:
            print(f"FAILED: {test_name} - {str(e)}")
            failed_experiments.append((test_name, str(e)))
    
    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    print(f"Successful experiments ({len(successful_experiments)}/5):")
    for exp in successful_experiments:
        print(f"  - {exp}")
    
    if failed_experiments:
        print(f"\nFailed experiments ({len(failed_experiments)}/5):")
        for exp, error in failed_experiments:
            print(f"  - {exp}: {error}")
    else:
        print("\nAll experiments started successfully!")
    
    return len(successful_experiments) == 5

if __name__ == "__main__":
    success = test_all_experiments_quick()
    if success:
        print("\nAll experiments verified! The main run_experiment.py should work correctly.")
    else:
        print("\nSome experiments failed. Check the errors above.")
