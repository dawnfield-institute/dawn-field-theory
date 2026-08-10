"""
Entropy-driven navigation algorithms with PAC-SEC balance.

PAC Confluence Xi Integration:
============================
The entropy navigator now incorporates the attraction/repulsion duality:
    - PAC (attraction, 4/5): Navigates toward coherent structures
    - SEC (repulsion, 1/5): Navigates toward entropy maxima
    - Ξ balance: Optimal navigation at balance point

Key insight: Navigation paths should balance:
    - Structure preservation (PAC) 
    - Entropy minimization for solutions (SEC)
"""

from typing import Any, Optional
import numpy as np

# PAC Confluence Xi Constants
PHI = (1 + np.sqrt(5)) / 2           # Golden ratio: 1.618034...
XI = 1.0571                           # Balance operator
ATTRACTION_FRACTION = 4/5             # PAC contribution
REPULSION_FRACTION = 1/5              # SEC contribution


class EntropyNavigator:
    """
    Implements entropy-guided traversal of the pattern tree.
    Uses hierarchical entropy signatures to select navigation paths.
    
    PAC Confluence Xi Enhancement:
    - Navigation balances attraction (structure) vs repulsion (entropy)
    - Ξ-weighted path selection for optimal regime detection
    """
    def __init__(self, pattern_tree, pac_sec_mode: str = "balanced"):
        """
        Initialize navigator with PAC-SEC mode.
        
        Args:
            pattern_tree: The pattern tree to navigate
            pac_sec_mode: "attraction" (4/5 weight), "repulsion" (1/5 weight), 
                         or "balanced" (Ξ-optimal)
        """
        self.pattern_tree = pattern_tree
        self.pac_sec_mode = pac_sec_mode
        
        # Set navigation weights based on mode
        if pac_sec_mode == "attraction":
            self.structure_weight = ATTRACTION_FRACTION  # Favor coherent patterns
            self.entropy_weight = REPULSION_FRACTION
        elif pac_sec_mode == "repulsion":
            self.structure_weight = REPULSION_FRACTION
            self.entropy_weight = ATTRACTION_FRACTION    # Favor entropy descent
        else:  # balanced - use Ξ
            self.structure_weight = 0.5 * XI  # ~0.528
            self.entropy_weight = 0.5 / XI     # ~0.473

    def _entropy_distance(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Compute distance between two entropy signatures (L2 norm).
        """
        return float(np.linalg.norm(sig1 - sig2))
    
    def _pac_sec_distance(self, node, target_entropy: np.ndarray) -> float:
        """
        Compute PAC-SEC weighted distance.
        
        Combines:
        - Entropy distance (SEC component - minimizing disorder)
        - Structure coherence (PAC component - preserving patterns)
        """
        node_entropy = getattr(node, 'entropy_signature', np.zeros_like(target_entropy))
        
        # SEC component: entropy distance
        entropy_dist = self._entropy_distance(node_entropy, target_entropy)
        
        # PAC component: structure coherence (lower is better)
        node_complexity = getattr(node, 'depth', 0) + len(getattr(node, 'children', []))
        structure_score = 1.0 / (1.0 + node_complexity)  # Simpler = more coherent
        
        # Combine with PAC-SEC weights
        combined = (self.entropy_weight * entropy_dist + 
                   self.structure_weight * (1.0 - structure_score))
        
        return combined

    def navigate(self, hierarchical_entropy, use_pac_sec: bool = True) -> list:
        """
        Navigate the pattern tree using the given hierarchical entropy signature.
        Returns the path of nodes traversed (greedy best match at each level).
        
        Args:
            hierarchical_entropy: Target entropy signature
            use_pac_sec: If True, use PAC-SEC balanced navigation
        """
        node = self.pattern_tree.root
        path = [node]
        levels = getattr(hierarchical_entropy, 'levels', [hierarchical_entropy])
        
        for level, entropy_sig in enumerate(levels):
            if not node.children:
                break
            
            if use_pac_sec:
                # PAC-SEC balanced child selection
                best_child = min(
                    node.children,
                    key=lambda c: self._pac_sec_distance(c, entropy_sig)
                )
            else:
                # Original entropy-only selection
                best_child = min(
                    node.children,
                    key=lambda c: self._entropy_distance(
                        getattr(c, 'entropy_signature', np.zeros_like(entropy_sig)), 
                        entropy_sig
                    )
                )
            
            node = best_child
            path.append(node)
            
        return path

    def find_optimal_path(self, hierarchical_entropy, 
                         use_pac_sec: bool = True) -> list:
        """
        Find the optimal navigation path for a given hierarchical entropy signature.
        Currently uses greedy navigation (can be extended for global optimization).
        
        With PAC-SEC mode enabled, balances structure preservation with entropy descent.
        """
        return self.navigate(hierarchical_entropy, use_pac_sec=use_pac_sec)
    
    def find_xi_balanced_path(self, hierarchical_entropy) -> list:
        """
        Find path that achieves Ξ = 1.0571 balance between PAC and SEC.
        
        This is the optimal navigation for regime detection and emergence.
        """
        # Temporarily set to balanced mode
        original_mode = self.pac_sec_mode
        self.pac_sec_mode = "balanced"
        self.structure_weight = 0.5 * XI
        self.entropy_weight = 0.5 / XI
        
        path = self.navigate(hierarchical_entropy, use_pac_sec=True)
        
        # Restore original mode
        self.pac_sec_mode = original_mode
        
        return path
    
    def calculate_path_balance(self, path: list) -> dict:
        """
        Calculate the PAC-SEC balance achieved along a navigation path.
        
        Returns metrics showing how the navigation balanced:
        - Structure preservation (PAC)
        - Entropy minimization (SEC)
        - Ξ deviation (distance from optimal balance)
        """
        if not path:
            return {"pac_score": 0, "sec_score": 0, "xi_deviation": 1.0}
        
        # Calculate cumulative structure score (PAC)
        depths = [getattr(n, 'depth', 0) for n in path]
        max_depth = max(depths) if depths else 1
        pac_score = 1.0 - (sum(depths) / (len(path) * max_depth + 1e-10))
        
        # Calculate entropy reduction (SEC)
        entropies = []
        for n in path:
            sig = getattr(n, 'entropy_signature', None)
            if sig is not None:
                entropies.append(np.mean(np.abs(sig)))
        
        if len(entropies) > 1:
            sec_score = (entropies[0] - entropies[-1]) / (entropies[0] + 1e-10)
        else:
            sec_score = 0.0
        
        # Calculate Ξ deviation
        if sec_score > 0:
            actual_ratio = pac_score / (sec_score + 1e-10)
            target_ratio = ATTRACTION_FRACTION / REPULSION_FRACTION  # 4
            xi_deviation = abs(actual_ratio - target_ratio) / target_ratio
        else:
            xi_deviation = 1.0
        
        return {
            "pac_score": pac_score,
            "sec_score": sec_score,
            "xi_deviation": xi_deviation,
            "balance_quality": 1.0 / (1.0 + xi_deviation)
        }
