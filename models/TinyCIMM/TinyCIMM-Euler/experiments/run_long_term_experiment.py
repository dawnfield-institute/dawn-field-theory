#!/usr/bin/env python3
"""
Long-term TinyCIMM-Euler Mathematical Reasoning Experiment
CIMM-style training over 100,000+ steps for deep mathematical pattern learning
"""

import torch
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_experiment import run_experiment, TinyCIMMEuler

def run_extreme_long_term_experiment():
    """Run extremely long-term prime delta prediction (like CIMM's 1M steps)"""
    print("=" * 80)
    print("TinyCIMM-Euler: EXTREME Long-term Mathematical Reasoning")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning 100,000 step prime delta prediction (CIMM-style deep learning)...")
    print("This will take several hours - progress will be shown every 5,000 steps.")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Ultra-long experiment with larger capacity
        run_experiment(
            TinyCIMMEuler, 
            signal="prime_deltas", 
            steps=100000,  # 100k steps for deep mathematical learning
            hidden_size=48,  # Larger capacity for complex patterns
            math_memory_size=50,  # Much more memory
            adaptation_steps=100,  # More adaptation steps
            seed=42
        )
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        print("=" * 80)
        print(f"✓ EXTREME long-term experiment completed successfully!")
        print(f"Duration: {hours}h {minutes}m")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Check experiment_images/prime_deltas/ for detailed results.")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Extreme long-term experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()

def run_cimm_style_experiment():
    """Run CIMM-style experiment with 1 million steps"""
    print("=" * 80)
    print("TinyCIMM-Euler: CIMM-Style 1 Million Step Experiment")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning 1,000,000 step prime delta prediction (true CIMM-style)...")
    print("This will take many hours - progress will be shown every 10,000 steps.")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # True CIMM-style 1M step experiment
        run_experiment(
            TinyCIMMEuler, 
            signal="prime_deltas", 
            steps=1000000,  # 1M steps like CIMM
            hidden_size=64,  # Large capacity for 1M steps
            math_memory_size=100,  # Massive memory
            adaptation_steps=200,  # Many adaptation steps
            seed=42
        )
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        print("=" * 80)
        print(f"✓ CIMM-style 1M step experiment completed successfully!")
        print(f"Duration: {hours}h {minutes}m")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Check experiment_images/prime_deltas/ for detailed results.")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ CIMM-style experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()

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
