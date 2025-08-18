"""Entropy-driven navigation algorithms."""


from typing import Any


import numpy as np

class EntropyNavigator:
    """
    Implements entropy-guided traversal of the pattern tree.
    Uses hierarchical entropy signatures to select navigation paths.
    """
    def __init__(self, pattern_tree):
        self.pattern_tree = pattern_tree

    def _entropy_distance(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Compute distance between two entropy signatures (L2 norm).
        """
        return float(np.linalg.norm(sig1 - sig2))

    def navigate(self, hierarchical_entropy) -> list:
        """
        Navigate the pattern tree using the given hierarchical entropy signature.
        Returns the path of nodes traversed (greedy best match at each level).
        """
        node = self.pattern_tree.root
        path = [node]
        levels = getattr(hierarchical_entropy, 'levels', [hierarchical_entropy])
        for level, entropy_sig in enumerate(levels):
            if not node.children:
                break
            # Find child with closest entropy signature at this level
            best_child = min(
                node.children,
                key=lambda c: self._entropy_distance(getattr(c, 'entropy_signature', np.zeros_like(entropy_sig)), entropy_sig)
            )
            node = best_child
            path.append(node)
        return path

    def find_optimal_path(self, hierarchical_entropy) -> list:
        """
        Find the optimal navigation path for a given hierarchical entropy signature.
        Currently uses greedy navigation (can be extended for global optimization).
        """
        return self.navigate(hierarchical_entropy)
