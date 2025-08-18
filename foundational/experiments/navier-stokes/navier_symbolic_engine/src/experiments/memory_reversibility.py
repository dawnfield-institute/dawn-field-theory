"""Module 3: Memory and reversibility tests."""

import numpy as np
from core.pattern_tree import PatternTree
from core.memory_tracker import MemoryTracker

def test_symbolic_reversibility():
    """
    Module 3: Validate symbolic flow trees retain ancestral trace memory 
    and enable flow history reconstruction.
    """
    tree = PatternTree()
    tracker = MemoryTracker()
    
    # Create a navigation path
    current_node = tree.root
    navigation_steps = []
    
    for i in range(5):
        entropy_sig = np.random.rand(10)  # Random entropy for demo
        child = tree.add_pattern(
            current_node, 
            entropy_sig, 
            regime="laminar", 
            pattern_data=np.random.rand(8, 8)
        )
        tracker.record(child)
        navigation_steps.append(child)
        current_node = child
    
    # Test reversibility
    original_trace = tracker.get_trace()
    
    # Undo 3 steps
    step1 = tracker.undo()
    step2 = tracker.undo() 
    step3 = tracker.undo()
    
    # Redo 2 steps
    redo1 = tracker.redo()
    redo2 = tracker.redo()
    
    # Validate memory trace coherence
    current_trace = tracker.get_trace()
    
    # Test ancestry preservation
    final_node = navigation_steps[-1]
    ancestry = final_node.get_ancestry()
    
    assert len(ancestry) == 6, "Ancestry should include root + 5 generations"
    assert ancestry[0] == tree.root, "Ancestry should start with root"
    assert step1 is not None, "Should be able to undo"
    assert redo1 is not None, "Should be able to redo"
    
    # Calculate reversibility fidelity
    reversibility_score = len(current_trace) / len(original_trace)
    
    return {
        "original_trace_length": len(original_trace),
        "current_trace_length": len(current_trace),
        "reversibility_score": reversibility_score,
        "ancestry_depth": len(ancestry),
        "validation": "passed" if reversibility_score > 0.5 else "failed"
    }
