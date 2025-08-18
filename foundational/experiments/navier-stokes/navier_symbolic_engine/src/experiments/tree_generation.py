"""Module 1: Tree generation tests."""

import numpy as np
from core.pattern_tree import PatternTree
from utils.entropy_hasher import EntropyHasher

def test_symbolic_tree_generation():
    """
    Module 1: Validate entropy-seeded boundary conditions produce reproducible, 
    finite-depth fractal flow patterns.
    """
    # Initialize components
    hasher = EntropyHasher()
    tree = PatternTree()
    
    # Test boundary conditions
    bc1 = {"velocity": 1.0, "geometry": "pipe", "reynolds": 1000}
    bc2 = {"velocity": 1.0, "geometry": "pipe", "reynolds": 1000}  # Same as bc1
    bc3 = {"velocity": 1.0, "geometry": "channel", "reynolds": 1000}  # Different geometry
    
    # Generate entropy signatures
    entropy1 = hasher.generate_hierarchical_entropy(bc1)
    entropy2 = hasher.generate_hierarchical_entropy(bc2)
    entropy3 = hasher.generate_hierarchical_entropy(bc3)
    
    # Test reproducibility
    assert np.allclose(entropy1.levels[0], entropy2.levels[0]), "Identical BCs should produce identical entropy"
    assert not np.allclose(entropy1.levels[0], entropy3.levels[0]), "Different BCs should produce different entropy"
    
    # Add patterns to tree
    node1 = tree.add_pattern(tree.root, entropy1, regime="laminar")
    node2 = tree.add_pattern(tree.root, entropy2, regime="laminar") 
    node3 = tree.add_pattern(tree.root, entropy3, regime="laminar")
    
    # Validate tree properties
    assert tree.node_count <= 10, "Tree should remain finite depth"
    assert node1.entropy_signature is not None, "Nodes should have entropy signatures"
    
    return {
        "reproducibility_test": "passed",
        "tree_depth": max([node.depth for node in [node1, node2, node3]]),
        "node_count": tree.node_count
    }
