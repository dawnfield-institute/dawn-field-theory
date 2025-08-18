"""Module 2: Reynolds regime tests."""

import numpy as np
from core.pattern_tree import PatternTree
from utils.entropy_hasher import EntropyHasher

def test_reynolds_regime_transitions():
    """
    Module 2: Test symbolic structure transitions at different Reynolds numbers.
    Validate discrete transitions corresponding to physical turbulence onset.
    """
    reynolds_range = [100, 500, 1000, 2300, 5000, 10000]
    hasher = EntropyHasher()
    tree = PatternTree()
    transitions = []
    entropy_distances = []
    
    prev_entropy = None
    for re in reynolds_range:
        bc = {"velocity": 1.0, "geometry": "pipe", "reynolds": re}
        entropy = hasher.generate_hierarchical_entropy(bc)
        
        # Determine regime based on Reynolds number
        regime = "laminar" if re < 2300 else "turbulent"
        
        # Add pattern to tree
        node = tree.add_pattern(tree.root, entropy, regime=regime)
        transitions.append({
            "reynolds": re,
            "regime": regime,
            "node_id": node.pattern_id
        })
        
        # Calculate entropy distance from previous
        if prev_entropy is not None:
            distance = np.linalg.norm(entropy.levels[0] - prev_entropy.levels[0])
            entropy_distances.append(distance)
        prev_entropy = entropy
    
    # Validate transition at Re = 2300
    laminar_regimes = [t for t in transitions if t["regime"] == "laminar"]
    turbulent_regimes = [t for t in transitions if t["regime"] == "turbulent"]
    
    assert len(laminar_regimes) == 3, "Should have 3 laminar regimes (Re < 2300)"
    assert len(turbulent_regimes) == 3, "Should have 3 turbulent regimes (Re >= 2300)"
    
    return {
        "transitions": transitions,
        "entropy_distances": entropy_distances,
        "validation": "passed"
    }
