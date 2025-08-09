#!/usr/bin/env python3
"""
======================================================================================
TinyCIMM-Euler Long-Term Mathematical Reasoning Experiment Suite
======================================================================================

Author: Dawn Field Theory Research Group
Date: July 10, 2025
Project: TinyCIMM-Euler - Higher-Order Mathematical Reasoning

This module provides long-term experiment runners for TinyCIMM-Euler that implement
CIMM-style learning over extended periods (100K+ to 1M+ steps). These experiments
test the model's ability to develop deep mathematical understanding through prolonged
exposure to mathematical patterns.

Key Features:
- Extreme long-term learning (100,000 steps)
- CIMM-style million-step experiments 
- Field-aware adaptation over extended periods
- Deep mathematical pattern consolidation
- Comprehensive progress tracking and logging

The experiments preserve all the highly optimized logic while providing clean,
professional interfaces for extended mathematical reasoning research.
======================================================================================
"""

import torch
import sys
import os
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinycimm_euler import TinyCIMMEuler
from run_experiment import run_experiment

# ======================================================================================
# DEBUG AND LOGGING SYSTEM
# ======================================================================================

# Global debug flag - set to False to hide debug output in production
DEBUG_MODE = True

def debug_print(message):
    """Print debug message only if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def info_print(message):
    """Print informational message (always shown)"""
    print(f"[INFO] {message}")

def run_extreme_long_term_experiment():
    """
    Run extremely long-term experiments across all mathematical patterns
    
    This function implements CIMM-style learning over 100,000 steps to test
    the model's ability to develop deep mathematical understanding through
    prolonged exposure to mathematical patterns.
    
    Experiment Details:
    - 100,000 adaptation steps per signal type
    - Individual step adaptation (CIMM-style)
    - Field-aware optimization
    - Comprehensive SCBF metrics tracking
    - Progress monitoring every 10,000 steps
    """
    info_print("=" * 80)
    info_print("TinyCIMM-Euler: EXTREME Long-term Mathematical Reasoning")
    info_print("=" * 80)
    info_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info_print("\nRunning 100,000 step experiments across all mathematical patterns (CIMM-style deep adaptation)...")
    info_print("This will take several hours - progress will be shown every 10,000 steps.")
    info_print("Using optimized GPU batch processing for maximum efficiency.")
    info_print("=" * 80)
    
    # Optimized test cases for extreme long-term mathematical learning
    # These configurations are fine-tuned for 100K+ step learning
    test_cases = [
        ("prime_deltas", {
            "hidden_size": 48,                    # Enhanced capacity for prime patterns
            "math_memory_size": 50,               # Large memory for complex sequences  
            "experiment_type": "longterm_extreme"
        }),
        ("fibonacci_ratios", {
            "hidden_size": 42,                    # Balanced for ratio convergence
            "math_memory_size": 45,               # Memory for golden ratio patterns
            "pattern_decay": 0.98,                # Slow decay for long-term learning
            "experiment_type": "longterm_extreme"
        }),
        ("polynomial_sequence", {
            "hidden_size": 44,                    # Good for algebraic patterns
            "math_memory_size": 48,               # Memory for polynomial structure
            "experiment_type": "longterm_extreme"
        }),
        ("recursive_sequence", {
            "hidden_size": 46,                    # Enhanced for recursive patterns
            "math_memory_size": 52,               # Large memory for meta-patterns
            "pattern_decay": 0.97,                # Preserve recursive structure
            "experiment_type": "longterm_extreme"
        }),
        ("mathematical_harmonic", {
            "hidden_size": 40,                    # Efficient for harmonic analysis
            "math_memory_size": 42,               # Memory for harmonic patterns
            "experiment_type": "longterm_extreme"
        }),
    ]
    
    start_time = time.time()
    successful_experiments = 0
    
    # Run experiments across all mathematical signal types
    for test_name, model_kwargs in test_cases:
        info_print(f"\n{'='*60}")
        info_print(f"=== Running Long-term Mathematical Experiment: {test_name} ===")
        
        # Determine expected difficulty level for this signal type
        challenge_level = "Extreme" if test_name == "prime_deltas" else "Very High" if "sequence" in test_name else "High"
        info_print(f"Expected challenge level: {challenge_level}")
        info_print(f"Adapting for 100,000 steps (CIMM-style long-term online learning)...")
        debug_print(f"Model configuration: {model_kwargs}")
        info_print(f"{'='*60}")
        
        try:
            # Run ultra-long experiment with individual step processing for proper CIMM adaptation
            # This preserves the fine-grained adaptation that enables breakthrough results
            debug_print(f"Starting {test_name} experiment with 100K steps...")
            
            run_experiment(
                TinyCIMMEuler, 
                signal=test_name, 
                steps=100000,     # 100k steps for deep mathematical learning
                batch_size=64,    # Small batch size for individual adaptation
                seed=42,          # Reproducible results
                **model_kwargs
            )
            
            info_print(f"✓ Completed {test_name} successfully")
            successful_experiments += 1
            debug_print(f"Successful experiments so far: {successful_experiments}")
            
        except Exception as e:
            info_print(f"✗ Error in {test_name}: {str(e)}")
            debug_print("Full traceback:")
            import traceback
            if DEBUG_MODE:
                traceback.print_exc()
            continue
        
    # Calculate experiment statistics and report results
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    
    # Comprehensive results summary
    info_print("=" * 80)
    info_print(f"✓ EXTREME long-term experiments completed!")
    info_print(f"Successfully completed: {successful_experiments}/{len(test_cases)} experiments")
    info_print(f"Total duration: {hours}h {minutes}m")
    
    if successful_experiments > 0:
        avg_steps_per_second = (successful_experiments * 100000) / duration
        info_print(f"Average steps per second: {avg_steps_per_second:.1f}")
        debug_print(f"Performance metrics - Duration: {duration:.1f}s, Steps/sec: {avg_steps_per_second:.2f}")
    
    info_print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info_print("Check experiment_results/ for detailed results organized by experiment type and date.")
    info_print("=" * 80)

def run_cimm_style_experiment():
    """Run CIMM-style experiment with 1 million steps across all mathematical patterns"""
    print("=" * 80)
    print("TinyCIMM-Euler: CIMM-Style 1 Million Step Experiments")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning 1,000,000 step experiments across all mathematical patterns (true CIMM-style)...")
    print("This will take many hours - progress will be shown every 50,000 steps.")
    print("Using maximum GPU optimization for sustained performance.")
    print("=" * 80)
    
    # Test cases for CIMM-style 1M step experiments
    test_cases = [
        ("prime_deltas", {
            "hidden_size": 64, 
            "math_memory_size": 100,
            "experiment_type": "cimm_million"
        }),
        ("fibonacci_ratios", {
            "hidden_size": 56, 
            "math_memory_size": 80, 
            "pattern_decay": 0.99,
            "experiment_type": "cimm_million"
        }),
        ("polynomial_sequence", {
            "hidden_size": 60, 
            "math_memory_size": 90,
            "experiment_type": "cimm_million"
        }),
        ("recursive_sequence", {
            "hidden_size": 62, 
            "math_memory_size": 95, 
            "pattern_decay": 0.98,
            "experiment_type": "cimm_million"
        }),
        ("mathematical_harmonic", {
            "hidden_size": 54, 
            "math_memory_size": 75,
            "experiment_type": "cimm_million"
        }),
    ]
    
    start_time = time.time()
    successful_experiments = 0
    
    for test_name, model_kwargs in test_cases:
        print(f"\n{'='*60}")
        print(f"=== Running CIMM-Style 1M Step Experiment: {test_name} ===")
        challenge_level = "Extreme" if test_name == "prime_deltas" else "Very High" if "sequence" in test_name else "High"
        print(f"Expected challenge level: {challenge_level}")
        print(f"Adapting for 1,000,000 steps (true CIMM-style online)...")
        print(f"{'='*60}")
        
        try:
            # True CIMM-style 1M step experiment with individual step processing
            run_experiment(
                TinyCIMMEuler, 
                signal=test_name, 
                steps=1000000,  # 1M steps like CIMM
                batch_size=32,  # Even smaller batches for 1M step processing
                seed=42,
                **model_kwargs
            )
            print(f"✓ Completed {test_name} successfully")
            successful_experiments += 1
            
        except Exception as e:
            print(f"✗ Error in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    
    print("=" * 80)
    print(f"✓ CIMM-style 1M step experiments completed!")
    print(f"Successfully completed: {successful_experiments}/{len(test_cases)} experiments")
    print(f"Total duration: {hours}h {minutes}m")
    print(f"Average steps per second: {(successful_experiments * 1000000) / duration:.1f}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check experiment_results/ for detailed results organized by experiment type and date.")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "extreme":
            run_extreme_long_term_experiment()
        elif sys.argv[1] == "cimm":
            run_cimm_style_experiment()
        else:
            print("Usage: python run_long_term_experiment.py [extreme|cimm]")
            print("  extreme: 100,000 steps (~few hours)")
            print("  cimm: 1,000,000 steps (~many hours)")
    else:
        print("Usage: python run_long_term_experiment.py [extreme|cimm]")
        print("  extreme: 100,000 steps (~few hours)")
        print("  cimm: 1,000,000 steps (~many hours)")
