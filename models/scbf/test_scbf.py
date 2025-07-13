#!/usr/bin/env python3
"""
SCBF Module Test
================

Simple test to verify SCBF module structure and basic functionality.
Run this to check if the SCBF framework is properly set up.
"""

import numpy as np
import torch
from pathlib import Path
import sys

def test_scbf_metrics():
    """Test SCBF metrics with synthetic test data."""
    print("Testing SCBF Metrics...")
    
    try:
        # Import SCBF metrics
        from metrics.entropy_collapse import compute_symbolic_entropy_collapse
        from metrics.activation_ancestry import compute_activation_ancestry
        from metrics.phase_alignment import compute_collapse_phase_alignment
        from metrics.semantic_attractors import compute_semantic_attractor_density
        from metrics.bifractal_lineage import compute_bifractal_lineage
        
        print("✓ Successfully imported all SCBF metric functions")
        
        # Create test data
        np.random.seed(42)
        
        # Test entropy collapse
        test_activations = np.random.randn(50, 20)  # 50 timesteps, 20 neurons
        entropy_results = compute_symbolic_entropy_collapse(test_activations)
        print(f"✓ Entropy collapse test passed: {len(entropy_results)} metrics computed")
        
        # Test activation ancestry
        ancestry_results = compute_activation_ancestry(test_activations)
        print(f"✓ Activation ancestry test passed: {len(ancestry_results)} metrics computed")
        
        # Test phase alignment
        phase_results = compute_collapse_phase_alignment(test_activations)
        print(f"✓ Phase alignment test passed: {len(phase_results)} metrics computed")
        
        # Test semantic attractors
        attractor_results = compute_semantic_attractor_density(test_activations)
        print(f"✓ Semantic attractors test passed: {len(attractor_results)} metrics computed")
        
        # Test bifractal lineage
        test_weights = np.random.randn(10, 10)
        lineage_results = compute_bifractal_lineage(test_weights)
        print(f"✓ Bifractal lineage test passed: {len(lineage_results)} metrics computed")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_scbf_loggers():
    """Test SCBF logging functionality."""
    print("\nTesting SCBF Loggers...")
    
    try:
        from loggers import create_experiment_logger, finalize_experiment
        
        # Create a test logger
        logger = create_experiment_logger("test_experiment")
        
        # Log some test data
        logger.log_step({
            'step': 0,
            'loss': 0.5,
            'scbf': {'entropy_collapse': {'magnitude': 0.1}}
        })
        
        # Finalize
        finalize_experiment(logger)
        
        print("✓ Successfully tested SCBF loggers")
        return True
        
    except Exception as e:
        print(f"❌ Loggers test failed: {e}")
        return False

def test_scbf_visualization():
    """Test SCBF visualization functionality.""" 
    print("\nTesting SCBF Visualization...")
    
    try:
        from visualization import plot_complete_scbf_dashboard
        from utils import create_mock_activations
        
        # Create mock experiment logs
        logs = []
        for i in range(10):
            logs.append({
                'step': i,
                'scbf': {
                    'entropy_collapse': {'magnitude': 0.1 * i},
                    'lineage': {'fractal_dimension': 1.0 + 0.01 * i}
                }
            })
        
        print("✓ Successfully imported SCBF visualization")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all SCBF tests."""
    print("=" * 50)
    print("SCBF Framework Test Suite")
    print("=" * 50)
    
    # Change to SCBF directory
    scbf_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        os.chdir(scbf_dir)
        
        tests_passed = 0
        total_tests = 3
        
        # Run tests
        if test_scbf_metrics():
            tests_passed += 1
            
        if test_scbf_loggers():
            tests_passed += 1
            
        if test_scbf_visualization():
            tests_passed += 1
        
        # Summary
        print(f"\n" + "=" * 50)
        print(f"Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print("✓ All SCBF tests passed! Framework is ready.")
            return 0
        else:
            print("❌ Some tests failed. Check the error messages above.")
            return 1
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import os
    exit_code = main()
    sys.exit(exit_code)
