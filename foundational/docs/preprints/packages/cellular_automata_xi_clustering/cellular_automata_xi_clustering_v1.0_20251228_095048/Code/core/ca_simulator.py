"""
Elementary Cellular Automata Simulator
======================================

Implements 1D elementary cellular automata (256 rules) with efficient evolution
and metrics extraction for PAC attractor analysis.

Based on Wolfram's classification:
- Class I: Homogeneous (rules converge to uniform state)
- Class II: Periodic (rules produce simple periodic patterns)
- Class III: Chaotic (rules produce pseudo-random patterns)
- Class IV: Complex (rules produce complex, localized structures - edge of chaos)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class WolframClass(IntEnum):
    """Wolfram's classification of CA behavior"""
    CLASS_I = 1    # Homogeneous
    CLASS_II = 2   # Periodic
    CLASS_III = 3  # Chaotic
    CLASS_IV = 4   # Complex (edge of chaos)
    UNKNOWN = 0


# Well-known rule classifications
RULE_CLASSIFICATIONS = {
    # Class I - Homogeneous
    0: WolframClass.CLASS_I,
    8: WolframClass.CLASS_I,
    32: WolframClass.CLASS_I,
    128: WolframClass.CLASS_I,
    136: WolframClass.CLASS_I,
    160: WolframClass.CLASS_I,
    
    # Class II - Periodic
    1: WolframClass.CLASS_II,
    2: WolframClass.CLASS_II,
    4: WolframClass.CLASS_II,
    5: WolframClass.CLASS_II,
    50: WolframClass.CLASS_II,
    51: WolframClass.CLASS_II,
    76: WolframClass.CLASS_II,
    77: WolframClass.CLASS_II,
    
    # Class III - Chaotic
    18: WolframClass.CLASS_III,
    22: WolframClass.CLASS_III,
    30: WolframClass.CLASS_III,
    45: WolframClass.CLASS_III,
    60: WolframClass.CLASS_III,
    90: WolframClass.CLASS_III,
    105: WolframClass.CLASS_III,
    122: WolframClass.CLASS_III,
    126: WolframClass.CLASS_III,
    146: WolframClass.CLASS_III,
    150: WolframClass.CLASS_III,
    
    # Class IV - Complex (edge of chaos) - Most interesting for PAC
    54: WolframClass.CLASS_IV,
    106: WolframClass.CLASS_IV,
    110: WolframClass.CLASS_IV,  # Computationally universal!
    124: WolframClass.CLASS_IV,
    137: WolframClass.CLASS_IV,
    193: WolframClass.CLASS_IV,
}


@dataclass
class CAState:
    """State of a cellular automaton evolution"""
    rule: int
    width: int
    steps: int
    history: np.ndarray  # Shape: (steps, width)
    initial_condition: str  # 'single', 'random', 'custom'
    
    @property
    def wolfram_class(self) -> WolframClass:
        return RULE_CLASSIFICATIONS.get(self.rule, WolframClass.UNKNOWN)


class ElementaryCA:
    """
    Elementary 1D Cellular Automaton with 256 possible rules.
    
    Rule encoding: 8-bit binary number specifying output for each
    of the 8 possible 3-cell neighborhoods (2^3 = 8 patterns).
    """
    
    def __init__(self, rule: int, width: int = 101):
        """
        Initialize CA with specified rule and width.
        
        Args:
            rule: Integer 0-255 specifying the CA rule
            width: Number of cells in the CA (odd recommended for symmetry)
        """
        if not 0 <= rule <= 255:
            raise ValueError(f"Rule must be 0-255, got {rule}")
        
        self.rule = rule
        self.width = width
        self.rule_table = self._build_rule_table(rule)
        
    def _build_rule_table(self, rule: int) -> np.ndarray:
        """Build lookup table for rule application."""
        # Each of 8 possible neighborhoods maps to 0 or 1
        table = np.zeros(8, dtype=np.uint8)
        for i in range(8):
            table[i] = (rule >> i) & 1
        return table
    
    def _apply_rule(self, left: int, center: int, right: int) -> int:
        """Apply rule to 3-cell neighborhood."""
        index = (left << 2) | (center << 1) | right
        return self.rule_table[index]
    
    def evolve(self, steps: int, 
               initial: Optional[np.ndarray] = None,
               init_type: str = 'single') -> CAState:
        """
        Evolve CA for specified number of steps.
        
        Args:
            steps: Number of evolution steps
            initial: Optional initial state array
            init_type: 'single' (one cell on), 'random', or 'custom'
            
        Returns:
            CAState containing full evolution history
        """
        # Initialize
        if initial is not None:
            state = initial.astype(np.uint8)
            init_type = 'custom'
        elif init_type == 'single':
            state = np.zeros(self.width, dtype=np.uint8)
            state[self.width // 2] = 1
        elif init_type == 'random':
            state = np.random.randint(0, 2, self.width, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        # Evolution history
        history = np.zeros((steps, self.width), dtype=np.uint8)
        history[0] = state
        
        # Evolve with periodic boundary conditions
        for t in range(1, steps):
            new_state = np.zeros(self.width, dtype=np.uint8)
            for i in range(self.width):
                left = state[(i - 1) % self.width]
                center = state[i]
                right = state[(i + 1) % self.width]
                new_state[i] = self._apply_rule(left, center, right)
            state = new_state
            history[t] = state
            
        return CAState(
            rule=self.rule,
            width=self.width,
            steps=steps,
            history=history,
            initial_condition=init_type
        )
    
    def evolve_fast(self, steps: int, 
                    initial: Optional[np.ndarray] = None,
                    init_type: str = 'single') -> CAState:
        """
        Vectorized evolution for better performance.
        """
        # Initialize
        if initial is not None:
            state = initial.astype(np.uint8)
            init_type = 'custom'
        elif init_type == 'single':
            state = np.zeros(self.width, dtype=np.uint8)
            state[self.width // 2] = 1
        elif init_type == 'random':
            state = np.random.randint(0, 2, self.width, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        history = np.zeros((steps, self.width), dtype=np.uint8)
        history[0] = state
        
        for t in range(1, steps):
            # Vectorized neighborhood computation
            left = np.roll(state, 1)
            right = np.roll(state, -1)
            # Compute index into rule table for each cell
            indices = (left << 2) | (state << 1) | right
            state = self.rule_table[indices]
            history[t] = state
            
        return CAState(
            rule=self.rule,
            width=self.width,
            steps=steps,
            history=history,
            initial_condition=init_type
        )


def get_representative_rules() -> Dict[WolframClass, List[int]]:
    """Get representative rules for each Wolfram class."""
    result = {cls: [] for cls in WolframClass}
    for rule, cls in RULE_CLASSIFICATIONS.items():
        result[cls].append(rule)
    return result


def get_edge_of_chaos_rules() -> List[int]:
    """Get Class IV (edge of chaos) rules - most interesting for PAC."""
    return [r for r, c in RULE_CLASSIFICATIONS.items() if c == WolframClass.CLASS_IV]


# Convenience functions for batch analysis
def batch_evolve(rules: List[int], width: int = 101, steps: int = 100,
                 init_type: str = 'single') -> Dict[int, CAState]:
    """Evolve multiple rules and return results."""
    results = {}
    for rule in rules:
        ca = ElementaryCA(rule, width)
        results[rule] = ca.evolve_fast(steps, init_type=init_type)
    return results


if __name__ == "__main__":
    # Quick demo
    print("Elementary CA Simulator Demo")
    print("=" * 40)
    
    # Test Rule 110 (edge of chaos, computationally universal)
    ca = ElementaryCA(110, width=51)
    state = ca.evolve_fast(30)
    
    print(f"Rule: {state.rule}")
    print(f"Wolfram Class: {state.wolfram_class.name}")
    print(f"Shape: {state.history.shape}")
    
    # Visual representation
    print("\nEvolution (first 20 steps):")
    for row in state.history[:20]:
        print("".join("█" if c else " " for c in row))
