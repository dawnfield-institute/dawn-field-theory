"""
Fracton Examples - Comprehensive examples demonstrating the Fracton language

This module provides practical examples of using Fracton for various
infodynamics research applications and recursive computation patterns.
"""

import fracton
import time
import math


@fracton.recursive
@fracton.entropy_gate(0.3, 0.8)
def fibonacci_field(memory, context):
    """
    Fibonacci computation with entropy-aware optimization.
    
    This example demonstrates basic recursive computation with entropy gates
    and memory field usage for caching and optimization.
    """
    # Base cases
    if context.depth <= 1:
        return 1
    
    # Check for cached results in low entropy conditions
    if context.entropy < 0.5:
        cache_key = f"fib_{context.depth}"
        cached = memory.get(cache_key)
        if cached is not None:
            return cached
    
    # Recursive computation
    a = fracton.recurse(fibonacci_field, memory, context.deeper(1))
    b = fracton.recurse(fibonacci_field, memory, context.deeper(2))
    result = a + b
    
    # Cache result for future use
    cache_key = f"fib_{context.depth}"
    memory.set(cache_key, result)
    
    return result


@fracton.recursive
@fracton.entropy_gate(0.5, 0.9)
def pattern_crystallizer(memory, context):
    """
    Recursive pattern detection and crystallization.
    
    This example shows how to use entropy dynamics for pattern recognition
    and structure formation in recursive data processing.
    """
    data = memory.get("input_data", [])
    
    if not data or context.depth > 20:
        return memory.get("patterns", [])
    
    # High entropy: explore new patterns
    if context.entropy > 0.7:
        # Detect patterns in current data subset
        subset_size = max(1, len(data) // (context.depth + 1))
        subset = data[:subset_size]
        
        patterns = fracton.detect_patterns(memory, min_confidence=0.6)
        
        # Recursively analyze remaining data
        remaining_data = data[subset_size:]
        memory.set("input_data", remaining_data)
        
        sub_patterns = fracton.recurse(pattern_crystallizer, memory, 
                                     context.deeper(1).with_entropy(context.entropy * 0.9))
        
        # Combine patterns
        all_patterns = patterns + sub_patterns
        memory.set("patterns", all_patterns)
        
        return all_patterns
    
    else:
        # Low entropy: crystallize existing patterns
        existing_patterns = memory.get("patterns", [])
        crystallized = fracton.crystallize(existing_patterns)
        memory.set("crystallized_patterns", crystallized)
        
        return crystallized


@fracton.recursive
@fracton.entropy_gate(0.4, 1.0)
def adaptive_search(memory, context):
    """
    Adaptive search algorithm with entropy-based exploration.
    
    Demonstrates how entropy levels can control exploration vs exploitation
    in recursive search algorithms.
    """
    search_space = memory.get("search_space", [])
    target = memory.get("target")
    current_best = memory.get("current_best", {"value": None, "score": float('inf')})
    
    if not search_space or context.depth > 50:
        return current_best
    
    # High entropy: broad exploration
    if context.entropy > 0.7:
        # Explore multiple paths
        sample_size = min(5, len(search_space))
        samples = search_space[:sample_size]
        
        best_in_samples = None
        best_score = float('inf')
        
        for i, sample in enumerate(samples):
            # Calculate fitness score
            score = abs(sample - target) if target is not None else abs(sample)
            
            if score < best_score:
                best_score = score
                best_in_samples = {"value": sample, "score": score}
            
            # Recursive exploration with decreasing entropy
            memory.set("search_space", search_space[sample_size:])
            sub_context = context.deeper(1).with_entropy(context.entropy * 0.95)
            
            sub_result = fracton.recurse(adaptive_search, memory, sub_context)
            
            if sub_result["score"] < current_best["score"]:
                current_best = sub_result
        
        # Update best if we found something better
        if best_in_samples and best_in_samples["score"] < current_best["score"]:
            current_best = best_in_samples
    
    # Medium entropy: focused search
    elif context.entropy > 0.5:
        # Binary search-like approach
        mid = len(search_space) // 2
        if mid > 0:
            left_space = search_space[:mid]
            right_space = search_space[mid:]
            
            # Search both halves
            memory.set("search_space", left_space)
            left_result = fracton.recurse(adaptive_search, memory, 
                                        context.deeper(1).with_entropy(context.entropy * 0.8))
            
            memory.set("search_space", right_space)
            right_result = fracton.recurse(adaptive_search, memory,
                                         context.deeper(1).with_entropy(context.entropy * 0.8))
            
            # Choose better result
            if left_result["score"] < right_result["score"]:
                current_best = left_result if left_result["score"] < current_best["score"] else current_best
            else:
                current_best = right_result if right_result["score"] < current_best["score"] else current_best
    
    # Low entropy: local refinement
    else:
        # Greedy local search
        if search_space:
            candidate = search_space[0]
            score = abs(candidate - target) if target is not None else abs(candidate)
            
            if score < current_best["score"]:
                current_best = {"value": candidate, "score": score}
            
            # Continue with remaining space
            memory.set("search_space", search_space[1:])
            result = fracton.recurse(adaptive_search, memory, context.deeper(1))
            
            if result["score"] < current_best["score"]:
                current_best = result
    
    memory.set("current_best", current_best)
    return current_best


@fracton.recursive
@fracton.entropy_gate(0.2, 0.8)
def entropy_dynamics_simulation(memory, context):
    """
    Simulation of entropy dynamics in complex systems.
    
    This example demonstrates modeling entropy evolution over time
    using recursive field transformations.
    """
    system_state = memory.get("system_state", {
        "particles": 100,
        "energy": 1000.0,
        "entropy": 0.5,
        "time": 0.0
    })
    
    max_time = memory.get("max_time", 100.0)
    dt = memory.get("time_step", 0.1)
    
    if system_state["time"] >= max_time or context.depth > 1000:
        return memory.get("trajectory", [])
    
    # Record current state
    trajectory = memory.get("trajectory", [])
    trajectory.append(system_state.copy())
    
    # Update system based on current entropy
    if context.entropy > 0.6:
        # High entropy: energy dispersion
        energy_loss = system_state["energy"] * 0.01
        system_state["energy"] -= energy_loss
        system_state["entropy"] = min(1.0, system_state["entropy"] + 0.02)
        
    elif context.entropy < 0.4:
        # Low entropy: energy concentration
        if system_state["energy"] < 950.0:
            system_state["energy"] += 5.0
        system_state["entropy"] = max(0.0, system_state["entropy"] - 0.01)
    
    # Time evolution
    system_state["time"] += dt
    
    # Quantum of interaction affects entropy
    entropy_fluctuation = 0.05 * math.sin(system_state["time"] * 0.5)
    new_entropy = max(0.0, min(1.0, system_state["entropy"] + entropy_fluctuation))
    
    # Update memory and context
    memory.set("system_state", system_state)
    memory.set("trajectory", trajectory)
    
    # Continue simulation with new entropy
    new_context = context.deeper(1).with_entropy(new_entropy)
    
    return fracton.recurse(entropy_dynamics_simulation, memory, new_context)


@fracton.recursive
@fracton.entropy_gate(0.3, 0.9)
def bifractal_tree_growth(memory, context):
    """
    Simulation of bifractal tree growth patterns.
    
    Demonstrates recursive structure generation with entropy-controlled
    branching patterns and self-similar growth.
    """
    tree = memory.get("tree", {
        "nodes": [{"id": 0, "x": 0, "y": 0, "level": 0}],
        "edges": [],
        "next_id": 1
    })
    
    max_levels = memory.get("max_levels", 8)
    current_level = memory.get("current_level", 0)
    
    if current_level >= max_levels or context.depth > 50:
        return tree
    
    # Get nodes at current level
    level_nodes = [n for n in tree["nodes"] if n["level"] == current_level]
    
    if not level_nodes:
        return tree
    
    new_nodes = []
    new_edges = []
    
    for node in level_nodes:
        # Entropy determines branching factor
        if context.entropy > 0.7:
            # High entropy: more branches
            branch_count = 3
        elif context.entropy > 0.4:
            # Medium entropy: binary branching
            branch_count = 2
        else:
            # Low entropy: single branch or termination
            branch_count = 1 if context.entropy > 0.2 else 0
        
        # Create branches
        for i in range(branch_count):
            angle = (i - (branch_count - 1) / 2) * (math.pi / 4)  # Spread branches
            
            # Calculate new position with some randomness
            branch_length = 10 * (0.8 ** current_level)  # Decrease with level
            dx = branch_length * math.cos(angle + node.get("angle", 0))
            dy = branch_length * math.sin(angle + node.get("angle", 0))
            
            new_node = {
                "id": tree["next_id"],
                "x": node["x"] + dx,
                "y": node["y"] + dy,
                "level": current_level + 1,
                "angle": angle + node.get("angle", 0),
                "parent_id": node["id"]
            }
            
            new_nodes.append(new_node)
            new_edges.append({"from": node["id"], "to": tree["next_id"]})
            tree["next_id"] += 1
    
    # Add new nodes and edges to tree
    tree["nodes"].extend(new_nodes)
    tree["edges"].extend(new_edges)
    
    # Update memory
    memory.set("tree", tree)
    memory.set("current_level", current_level + 1)
    
    # Continue growth with slightly modified entropy
    entropy_drift = (context.entropy - 0.5) * 0.1  # Drift toward 0.5
    new_entropy = max(0.1, min(0.9, context.entropy - entropy_drift))
    
    new_context = context.deeper(1).with_entropy(new_entropy)
    
    return fracton.recurse(bifractal_tree_growth, memory, new_context)


def run_fibonacci_example():
    """Run the Fibonacci example with tracing and analysis."""
    print("=== Fracton Fibonacci Example ===")
    
    with fracton.memory_field(capacity=1000) as field:
        # Create execution context
        context = fracton.Context(entropy=0.6, depth=10)
        
        # Create trace for analysis
        trace = fracton.BifractalTrace()
        
        # Run computation
        start_time = time.time()
        result = fibonacci_field(field, context)
        end_time = time.time()
        
        print(f"Fibonacci(10) = {result}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print(f"Memory field size: {field.size()} items")
        print(f"Final entropy: {field.get_entropy():.3f}")
        
        # Show memory contents
        print("\nCached values:")
        for key, value in field.items():
            if key.startswith("fib_"):
                print(f"  {key}: {value}")


def run_pattern_analysis_example():
    """Run the pattern analysis example."""
    print("\n=== Fracton Pattern Analysis Example ===")
    
    # Generate some test data with patterns
    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3  # Repeated pattern
    test_data.extend([100, 200, 300, 400, 500])  # Arithmetic sequence
    test_data.extend([2, 4, 8, 16, 32])  # Geometric sequence
    
    with fracton.memory_field(capacity=2000) as field:
        # Initialize data
        field.set("input_data", test_data)
        
        # Create high-entropy context for exploration
        context = fracton.Context(entropy=0.8, depth=0)
        
        # Run pattern analysis
        patterns = pattern_crystallizer(field, context)
        
        print(f"Input data size: {len(test_data)}")
        print(f"Patterns found: {len(patterns)}")
        
        for i, pattern in enumerate(patterns):
            print(f"  Pattern {i+1}: {pattern}")
        
        # Show crystallized patterns if any
        crystallized = field.get("crystallized_patterns")
        if crystallized:
            print(f"\nCrystallized patterns: {crystallized}")


def run_adaptive_search_example():
    """Run the adaptive search example."""
    print("\n=== Fracton Adaptive Search Example ===")
    
    # Create search space and target
    search_space = list(range(0, 1000, 7))  # Numbers: 0, 7, 14, 21, ...
    target = 287  # Should find 287 (close to 287)
    
    with fracton.memory_field(capacity=1000) as field:
        field.set("search_space", search_space)
        field.set("target", target)
        
        # Start with high entropy for exploration
        context = fracton.Context(entropy=0.8, depth=0)
        
        # Run search
        start_time = time.time()
        result = adaptive_search(field, context)
        end_time = time.time()
        
        print(f"Search space size: {len(search_space)}")
        print(f"Target: {target}")
        print(f"Best found: {result['value']} (score: {result['score']:.2f})")
        print(f"Search time: {end_time - start_time:.4f} seconds")


def run_entropy_simulation_example():
    """Run the entropy dynamics simulation."""
    print("\n=== Fracton Entropy Dynamics Simulation ===")
    
    with fracton.memory_field(capacity=2000) as field:
        # Initialize simulation parameters
        field.set("max_time", 20.0)
        field.set("time_step", 0.2)
        
        # Start simulation
        context = fracton.Context(entropy=0.5, depth=0)
        
        trajectory = entropy_dynamics_simulation(field, context)
        
        print(f"Simulation completed: {len(trajectory)} time steps")
        print("\nSample trajectory points:")
        for i in range(0, len(trajectory), max(1, len(trajectory) // 10)):
            state = trajectory[i]
            print(f"  t={state['time']:.1f}: energy={state['energy']:.1f}, "
                  f"entropy={state['entropy']:.3f}")


def run_tree_growth_example():
    """Run the bifractal tree growth example."""
    print("\n=== Fracton Bifractal Tree Growth Example ===")
    
    with fracton.memory_field(capacity=1000) as field:
        field.set("max_levels", 6)
        field.set("current_level", 0)
        
        # Start with medium entropy
        context = fracton.Context(entropy=0.6, depth=0)
        
        tree = bifractal_tree_growth(field, context)
        
        print(f"Tree growth completed:")
        print(f"  Total nodes: {len(tree['nodes'])}")
        print(f"  Total edges: {len(tree['edges'])}")
        print(f"  Levels: {max(n['level'] for n in tree['nodes']) + 1}")
        
        # Show level distribution
        level_counts = {}
        for node in tree["nodes"]:
            level = node["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("\nNodes per level:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} nodes")


def run_all_examples():
    """Run all Fracton examples in sequence."""
    print("Running all Fracton examples...\n")
    
    try:
        run_fibonacci_example()
        run_pattern_analysis_example()
        run_adaptive_search_example()
        run_entropy_simulation_example()
        run_tree_growth_example()
        
        print("\n" + "="*50)
        print("All examples completed successfully!")
        
        # Show runtime statistics
        stats = fracton.get_runtime_stats()
        print(f"\nRuntime Statistics:")
        print(f"  Version: {stats.get('version', 'unknown')}")
        print(f"  Memory fields created: {stats.get('memory_fields', 0)}")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
