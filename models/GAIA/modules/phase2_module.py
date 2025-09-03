"""
Phase 2 Module for GAIA Runtime
===============================

Symbolic Intelligence Testing Suite
Tests language patterns, mathematical reasoning, and visual pattern recognition.

Usage:
    python gaia_runtime.py --modules phase2_module
"""

from typing import Dict, Any, List
import torch
import time

def run_module(runtime, **kwargs) -> Dict[str, Any]:
    """
    Phase 2 testing module - Symbolic Intelligence validation
    
    Args:
        runtime: GAIARuntime instance with access to all engines
        **kwargs: Additional parameters from CLI
        
    Returns:
        Dict with comprehensive Phase 2 test results
    """
    runtime.logger.info("🚀 Starting Phase 2: Symbolic Intelligence Testing")
    
    results = {}
    total_structures = 0
    start_time = time.time()
    
    # Test 2.1: Language Pattern Emergence
    runtime.logger.info("=== Test 2.1: Language Pattern Emergence ===")
    
    language_tests = {
        "simple_words": ["the", "cat", "sat", "on", "the", "mat"],
        "repeated_patterns": ["ab", "ab", "cd", "cd", "ab", "cd", "ab", "cd"],
        "grammar_like": ["noun", "verb", "noun", "verb", "adjective", "noun"],
        "english_sample": list("hello world this is a test of symbolic emergence")
    }
    
    language_results = {}
    
    for test_name, sequence in language_tests.items():
        runtime.logger.info(f"Testing language pattern: {test_name}")
        runtime.reset()
        
        # Convert sequence to tensor
        if isinstance(sequence[0], str):
            if len(sequence[0]) == 1:  # Character sequence
                tensor_seq = torch.tensor([ord(c) for c in sequence], dtype=torch.float32, device=runtime.device)
            else:  # Word sequence
                tensor_seq = torch.tensor([len(word) + ord(word[0]) for word in sequence], dtype=torch.float32, device=runtime.device)
        else:
            tensor_seq = torch.tensor(sequence, dtype=torch.float32, device=runtime.device)
        
        # Inject with language-specific scaling
        energy_pattern = tensor_seq * 4.0
        info_pattern = tensor_seq * 2.5
        
        runtime.field_engine.inject_stimulus(energy_pattern, "energy")
        runtime.field_engine.inject_stimulus(info_pattern, "information")
        
        # Warm-up period
        for _ in range(20):
            runtime.field_engine.step()
        
        # Evolution and capture
        pressures = []
        collapses = []
        structures_count = 0
        
        for step in range(60):
            collapse_event = runtime.field_engine.step()
            state = runtime.field_engine.get_field_state()
            pressures.append(state.field_pressure)
            
            if collapse_event:
                collapses.append(collapse_event)
                structure = runtime.collapse_core.process_collapse_event(collapse_event, state)
                if structure:
                    structures_count += 1
                    total_structures += 1
        
        # Analyze patterns
        entropy_signatures = []
        if hasattr(runtime.collapse_core, 'symbolic_structures'):
            structures = runtime.collapse_core.symbolic_structures
            entropy_signatures = [s.entropy_signature for s in structures[-structures_count:]] if structures_count > 0 else []
        
        pattern_complexity = torch.var(torch.tensor(entropy_signatures, device=runtime.device)).item() if entropy_signatures else 0
        
        language_results[test_name] = {
            "total_structures": structures_count,
            "pattern_complexity": pattern_complexity,
            "mean_pressure": torch.mean(torch.tensor(pressures, device=runtime.device)).item(),
            "total_collapses": len(collapses)
        }
        
        runtime.logger.info(f"  {test_name}: {structures_count} structures, complexity: {pattern_complexity:.6f}")
    
    results["language"] = language_results
    
    # Test 2.2: Mathematical Reasoning
    runtime.logger.info("=== Test 2.2: Mathematical Reasoning ===")
    
    math_tests = {
        "fibonacci": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        "primes": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        "squares": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "powers_of_2": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "arithmetic": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        "geometric": [2, 6, 18, 54, 162, 486, 1458, 4374]
    }
    
    math_results = {}
    
    for test_name, sequence in math_tests.items():
        runtime.logger.info(f"Testing mathematical pattern: {test_name}")
        runtime.reset()
        
        # Inject with mathematical scaling
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=runtime.device)
        
        # Normalize large numbers to prevent overflow
        if torch.max(sequence_tensor) > 1000:
            sequence_tensor = torch.log(sequence_tensor + 1)
        
        # Strong mathematical scaling
        energy_pattern = sequence_tensor * 8.0
        info_pattern = sequence_tensor * 5.0
        
        runtime.field_engine.inject_stimulus(energy_pattern, "energy")
        runtime.field_engine.inject_stimulus(info_pattern, "information")
        
        # Warm-up
        for _ in range(15):
            runtime.field_engine.step()
        
        # Evolution
        pressures = []
        collapses = []
        structures_count = 0
        
        for step in range(50):
            collapse_event = runtime.field_engine.step()
            state = runtime.field_engine.get_field_state()
            pressures.append(state.field_pressure)
            
            if collapse_event:
                collapses.append(collapse_event)
                structure = runtime.collapse_core.process_collapse_event(collapse_event, state)
                if structure:
                    structures_count += 1
                    total_structures += 1
        
        # Calculate entropy efficiency
        entropy_efficiency = 0
        if hasattr(runtime.collapse_core, 'symbolic_structures') and structures_count > 0:
            recent_structures = runtime.collapse_core.symbolic_structures[-structures_count:]
            entropy_efficiency = sum(s.entropy_signature for s in recent_structures) / len(recent_structures)
        
        math_results[test_name] = {
            "total_structures": structures_count,
            "entropy_efficiency": entropy_efficiency,
            "mean_pressure": torch.mean(torch.tensor(pressures, device=runtime.device)).item(),
            "total_collapses": len(collapses)
        }
        
        runtime.logger.info(f"  {test_name}: {structures_count} structures, efficiency: {entropy_efficiency:.6f}")
    
    results["mathematics"] = math_results
    
    # Test 2.3: Visual Pattern Recognition
    runtime.logger.info("=== Test 2.3: Visual Pattern Recognition ===")
    
    def create_line_pattern(size=8):
        pattern = torch.zeros((size, size), device=runtime.device)
        pattern[size//2, :] = 1.0
        return pattern
    
    def create_cross_pattern(size=8):
        pattern = torch.zeros((size, size), device=runtime.device)
        pattern[size//2, :] = 1.0
        pattern[:, size//2] = 1.0
        return pattern
    
    def create_circle_pattern(size=8):
        pattern = torch.zeros((size, size), device=runtime.device)
        center = size // 2
        radius = size // 4
        for i in range(size):
            for j in range(size):
                if abs((i - center)**2 + (j - center)**2 - radius**2) < radius:
                    pattern[i, j] = 1.0
        return pattern
    
    def create_checkerboard_pattern(size=8):
        pattern = torch.zeros((size, size), device=runtime.device)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    pattern[i, j] = 1.0
        return pattern
    
    visual_tests = {
        "horizontal_line": create_line_pattern(),
        "cross": create_cross_pattern(),
        "circle": create_circle_pattern(),
        "checkerboard": create_checkerboard_pattern(),
        "random_noise": torch.rand((8, 8), device=runtime.device)
    }
    
    visual_results = {}
    
    for test_name, pattern in visual_tests.items():
        runtime.logger.info(f"Testing visual pattern: {test_name}")
        runtime.reset()
        
        # Convert 2D pattern to sequence for injection
        pattern_flat = pattern.flatten()
        
        # Visual scaling
        energy_pattern = pattern_flat * 6.0
        info_pattern = pattern_flat * 4.0
        
        runtime.field_engine.inject_stimulus(energy_pattern, "energy")
        runtime.field_engine.inject_stimulus(info_pattern, "information")
        
        # Evolution
        pressures = []
        collapses = []
        structures_count = 0
        
        for step in range(40):
            collapse_event = runtime.field_engine.step()
            state = runtime.field_engine.get_field_state()
            pressures.append(state.field_pressure)
            
            if collapse_event:
                collapses.append(collapse_event)
                structure = runtime.collapse_core.process_collapse_event(collapse_event, state)
                if structure:
                    structures_count += 1
                    total_structures += 1
        
        # Pattern complexity analysis
        pattern_complexity = 0
        if hasattr(runtime.collapse_core, 'symbolic_structures') and structures_count > 0:
            recent_structures = runtime.collapse_core.symbolic_structures[-structures_count:]
            entropies = [s.entropy_signature for s in recent_structures]
            pattern_complexity = torch.var(torch.tensor(entropies, device=runtime.device)).item()
        
        visual_results[test_name] = {
            "total_structures": structures_count,
            "pattern_complexity": pattern_complexity,
            "mean_pressure": torch.mean(torch.tensor(pressures, device=runtime.device)).item(),
            "total_collapses": len(collapses)
        }
        
        runtime.logger.info(f"  {test_name}: {structures_count} structures, complexity: {pattern_complexity:.6f}")
    
    results["visual"] = visual_results
    
    # Calculate overall metrics
    execution_time = time.time() - start_time
    
    # Validation checks
    max_lang_pressure = max(r["mean_pressure"] for r in language_results.values())
    total_lang_collapses = sum(r["total_collapses"] for r in language_results.values())
    
    max_math_pressure = max(r["mean_pressure"] for r in math_results.values())
    total_math_collapses = sum(r["total_collapses"] for r in math_results.values())
    
    max_visual_pressure = max(r["mean_pressure"] for r in visual_results.values())
    total_visual_collapses = sum(r["total_collapses"] for r in visual_results.values())
    
    validation_passed = (
        max_lang_pressure > 0.01 and
        total_lang_collapses > 10 and
        max_math_pressure > 0.01 and
        total_math_collapses > 20 and
        max_visual_pressure > 0.005 and
        total_visual_collapses > 5
    )
    
    runtime.logger.info(f"Phase 2 completed in {execution_time:.2f}s")
    runtime.logger.info(f"Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    return {
        'metrics': {
            'execution_time': execution_time,
            'validation_passed': validation_passed,
            'language_max_pressure': max_lang_pressure,
            'language_total_collapses': total_lang_collapses,
            'math_max_pressure': max_math_pressure,
            'math_total_collapses': total_math_collapses,
            'visual_max_pressure': max_visual_pressure,
            'visual_total_collapses': total_visual_collapses,
            'symbolic_emergence': total_structures > 50
        },
        'structures': total_structures,
        'detailed_results': results
    }
