#!/usr/bin/env python3
"""Quick test for improved prime delta prediction"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_experiment import TinyCIMMEuler, run_experiment

def test_improved_prime_deltas():
    """Test the improved prime delta prediction with normalization and better configuration"""
    print("=" * 60)
    print("Testing Improved Prime Delta Prediction")
    print("=" * 60)
    print("\nImprovements:")
    print("• Data normalization for better learning")
    print("• 8-step sequences instead of 4-step")
    print("• Larger model (40 hidden units, 50 memory)")
    print("• More careful learning rate adaptation")
    print("• Denormalized predictions for interpretability")
    print("\nRunning 2000 steps for quick validation...")
    
    try:
        run_experiment(
            TinyCIMMEuler, 
            signal="prime_deltas", 
            steps=2000,  # Shorter for quick test
            hidden_size=40,
            math_memory_size=50,
            adaptation_steps=60,
            pattern_decay=0.999,
            learning_rate=0.003,
            experiment_type="improved_test"
        )
        print("\n" + "=" * 60)
        print("Improved prime delta test completed!")
        print("Check experiment_results/prime_deltas_improved_test/ for results.")
        print("=" * 60)
    except Exception as e:
        print(f"\nError in improved prime delta test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_prime_deltas()
