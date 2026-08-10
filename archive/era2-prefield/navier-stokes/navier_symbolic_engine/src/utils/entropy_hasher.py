"""SHA256-based entropy generation."""


import hashlib
import numpy as np


class HierarchicalEntropy:
    """
    Multi-level entropy signature for hierarchical navigation.
    """
    def __init__(self, levels):
        self.levels = levels  # List of np.ndarray

class EntropyHasher:
    """
    Generates hierarchical entropy signatures from boundary conditions using SHA256.
    Converts problem specification to deterministic, multi-level entropy signature.
    """
    def __init__(self, vector_length: int = 128):
        self.vector_length = vector_length

    def _hash_component(self, component: str) -> np.ndarray:
        sha = hashlib.sha256(component.encode('utf-8')).digest()
        raw = np.frombuffer(sha * ((self.vector_length // len(sha)) + 1), dtype=np.uint8)[:self.vector_length]
        vec = raw.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def generate_hierarchical_entropy(self, bc: dict) -> HierarchicalEntropy:
        """
        Generate multi-level entropy signature for hierarchical navigation.
        Levels: global, regime, local, micro.
        """
        # Level 0: Global signature
        global_str = str(sorted(bc.items()))
        global_entropy = self._hash_component(global_str)

        # Level 1: Regime signature (e.g., based on Reynolds number)
        regime_val = bc.get('reynolds', 0)
        regime_str = f"regime:{'laminar' if regime_val < 2300 else 'turbulent'}"
        regime_entropy = self._hash_component(regime_str)

        # Level 2: Local signature (geometry, boundary values)
        local_str = str(bc.get('geometry', '')) + str(bc.get('boundary_values', ''))
        local_entropy = self._hash_component(local_str)

        # Level 3: Micro signature (initial conditions, fine details)
        micro_str = str(bc.get('initial_conditions', ''))
        micro_entropy = self._hash_component(micro_str)

        return HierarchicalEntropy([global_entropy, regime_entropy, local_entropy, micro_entropy])
