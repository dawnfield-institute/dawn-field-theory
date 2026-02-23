"""
PAC Phase Space Embedding for Cellular Automata
================================================

Maps CA rule dynamics to PAC (Potential-Actualization-Conservation) phase space
coordinates. This enables testing whether CA rules cluster as discrete attractors.

PAC Interpretation for CA:
- P (Potential): Unrealized computational capacity (low entropy states)
- A (Actualization): Realized structure/pattern (information content)
- C (Conservation): P + A = constant at each scale

The hypothesis is that CA rules are not fundamental but emergent - they are
stable observational descriptions of systems at specific PAC attractor positions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Import CA simulator
sys.path.insert(0, os.path.dirname(__file__))
from ca_simulator import CAState, ElementaryCA, WolframClass


@dataclass
class PACCoordinates:
    """PAC phase space coordinates for a CA rule."""
    rule: int
    potential: float      # P: unrealized capacity
    actualization: float  # A: realized structure
    conservation: float   # C: total (should be ~constant across rules)
    xi: float             # Ξ: balance operator (deviation from symmetry)
    
    @property
    def pac_vector(self) -> np.ndarray:
        """Return [P, A, C] as numpy array."""
        return np.array([self.potential, self.actualization, self.conservation])
    
    @property
    def normalized_vector(self) -> np.ndarray:
        """Return normalized [P, A] on unit circle."""
        total = self.potential + self.actualization
        if total == 0:
            return np.array([0.5, 0.5])
        return np.array([self.potential / total, self.actualization / total])


class PACEmbedder:
    """
    Embeds CA rules into PAC phase space based on their dynamics.
    
    Three independent measurement paths:
    1. Conservation Physics: Energy flow and equilibrium analysis
    2. Geometric Topology: Structural invariants (Betti-like numbers)
    3. Information Theory: Entropy and mutual information
    """
    
    def __init__(self, width: int = 101, steps: int = 200):
        """
        Initialize embedder with CA evolution parameters.
        
        Args:
            width: CA width (cells)
            steps: Evolution steps for analysis
        """
        self.width = width
        self.steps = steps
        
    def compute_entropy(self, state: CAState) -> float:
        """
        Compute spatial entropy of CA evolution.
        
        Higher entropy = more disordered/chaotic patterns.
        """
        # Compute density over time
        densities = state.history.mean(axis=1)
        
        # Shannon entropy of density distribution
        hist, _ = np.histogram(densities, bins=20, range=(0, 1), density=True)
        hist = hist[hist > 0]  # Remove zeros
        if len(hist) == 0:
            return 0.0
        hist = hist / hist.sum()  # Normalize
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def compute_block_entropy(self, state: CAState, block_size: int = 3) -> float:
        """
        Compute block entropy - entropy of local patterns.
        
        This measures the diversity of local configurations.
        """
        history = state.history
        steps, width = history.shape
        
        # Count all block patterns
        pattern_counts = {}
        for t in range(steps):
            for i in range(width - block_size + 1):
                pattern = tuple(history[t, i:i+block_size])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Compute entropy
        total = sum(pattern_counts.values())
        probs = np.array(list(pattern_counts.values())) / total
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def compute_mutual_information(self, state: CAState, lag: int = 1) -> float:
        """
        Compute mutual information between consecutive time steps.
        
        High MI = predictable evolution (Class I/II)
        Low MI = unpredictable evolution (Class III)
        Medium MI = complex/structured (Class IV - edge of chaos)
        """
        history = state.history
        
        # Joint distribution of (state_t, state_{t+lag})
        joint_counts = {}
        for t in range(len(history) - lag):
            for i in range(self.width):
                key = (history[t, i], history[t + lag, i])
                joint_counts[key] = joint_counts.get(key, 0) + 1
        
        total = sum(joint_counts.values())
        
        # Marginal distributions
        p_x = np.zeros(2)
        p_y = np.zeros(2)
        for (x, y), count in joint_counts.items():
            p_x[x] += count
            p_y[y] += count
        p_x /= total
        p_y /= total
        
        # Mutual information
        mi = 0.0
        for (x, y), count in joint_counts.items():
            p_xy = count / total
            if p_xy > 0 and p_x[x] > 0 and p_y[y] > 0:
                mi += p_xy * np.log2(p_xy / (p_x[x] * p_y[y]))
        
        return mi
    
    def compute_lyapunov_proxy(self, state: CAState) -> float:
        """
        Compute proxy for Lyapunov exponent via damage spreading.
        
        Measures sensitivity to initial conditions:
        - Positive: Chaotic (Class III)
        - Near zero: Edge of chaos (Class IV)
        - Negative: Ordered (Class I/II)
        """
        # Evolve with slightly perturbed initial condition
        ca = ElementaryCA(state.rule, self.width)
        
        # Original evolution
        original = ca.evolve_fast(self.steps, init_type='single')
        
        # Perturbed evolution (flip one cell)
        perturbed_init = np.zeros(self.width, dtype=np.uint8)
        perturbed_init[self.width // 2] = 1
        perturbed_init[(self.width // 2) + 1] = 1  # Extra cell
        perturbed = ca.evolve_fast(self.steps, initial=perturbed_init)
        
        # Hamming distance over time
        distances = np.sum(original.history != perturbed.history, axis=1)
        
        # Fit exponential growth: d(t) ~ exp(λt)
        # Use log-linear fit for later timesteps
        t = np.arange(10, min(50, self.steps))
        d = distances[10:min(50, self.steps)]
        d = np.maximum(d, 1)  # Avoid log(0)
        
        if len(t) < 5:
            return 0.0
            
        # Linear regression on log(d) vs t
        coeffs = np.polyfit(t, np.log(d), 1)
        return coeffs[0]  # Slope = Lyapunov proxy
    
    def compute_structure_factor(self, state: CAState) -> float:
        """
        Compute structure factor - measures spatial organization.
        
        High structure = organized patterns (Class II/IV)
        Low structure = disordered (Class I/III)
        """
        # Use FFT to detect periodic structures
        history = state.history.astype(float)
        
        # Average power spectrum over time
        power_spectra = []
        for row in history[self.steps // 2:]:  # Use second half (after transient)
            fft = np.fft.fft(row - row.mean())
            power = np.abs(fft) ** 2
            power_spectra.append(power)
        
        avg_power = np.mean(power_spectra, axis=0)
        
        # Structure factor = ratio of peak to background
        # Exclude DC component
        non_dc = avg_power[1:len(avg_power)//2]
        if len(non_dc) == 0 or np.mean(non_dc) == 0:
            return 0.0
        return np.max(non_dc) / (np.mean(non_dc) + 1e-10)
    
    def embed_rule(self, rule: int, init_type: str = 'single') -> PACCoordinates:
        """
        Embed a single CA rule into PAC phase space.
        
        PAC mapping:
        - P (Potential) ∝ 1 - entropy (unused capacity)
        - A (Actualization) ∝ mutual_info + structure (realized patterns)
        - C (Conservation) = P + A (normalized to 1)
        - Ξ (Xi) = balance measure
        """
        # Evolve CA
        ca = ElementaryCA(rule, self.width)
        state = ca.evolve_fast(self.steps, init_type=init_type)
        
        # Compute metrics
        entropy = self.compute_entropy(state)
        block_entropy = self.compute_block_entropy(state)
        mutual_info = self.compute_mutual_information(state)
        structure = self.compute_structure_factor(state)
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log2(20)  # Max bins in histogram
        norm_entropy = min(entropy / max_entropy, 1.0)
        
        # PAC coordinates
        # P: Low entropy = high potential (unrealized capacity)
        potential = 1.0 - norm_entropy
        
        # A: High mutual info + structure = high actualization
        actualization = 0.5 * mutual_info + 0.3 * min(structure / 10, 1.0) + 0.2 * (block_entropy / 8)
        actualization = min(actualization, 1.0)
        
        # C: Conservation (normalize to sum = 1)
        total = potential + actualization
        if total > 0:
            potential /= total
            actualization /= total
        conservation = 1.0  # By construction
        
        # Ξ: Deviation from perfect balance (0.5, 0.5)
        xi = abs(potential - actualization) / (potential + actualization + 1e-10)
        
        return PACCoordinates(
            rule=rule,
            potential=potential,
            actualization=actualization,
            conservation=conservation,
            xi=xi
        )
    
    def embed_all_rules(self, init_type: str = 'single') -> Dict[int, PACCoordinates]:
        """Embed all 256 elementary CA rules into PAC space."""
        embeddings = {}
        for rule in range(256):
            embeddings[rule] = self.embed_rule(rule, init_type)
        return embeddings
    
    def embed_rules(self, rules: List[int], init_type: str = 'single') -> Dict[int, PACCoordinates]:
        """Embed specified rules into PAC space."""
        return {rule: self.embed_rule(rule, init_type) for rule in rules}


def compute_pac_distances(embeddings: Dict[int, PACCoordinates]) -> np.ndarray:
    """
    Compute pairwise distances between rules in PAC space.
    
    Returns distance matrix.
    """
    rules = sorted(embeddings.keys())
    n = len(rules)
    distances = np.zeros((n, n))
    
    for i, r1 in enumerate(rules):
        for j, r2 in enumerate(rules):
            v1 = embeddings[r1].normalized_vector
            v2 = embeddings[r2].normalized_vector
            distances[i, j] = np.linalg.norm(v1 - v2)
    
    return distances


if __name__ == "__main__":
    print("PAC Embedding Demo")
    print("=" * 40)
    
    embedder = PACEmbedder(width=101, steps=100)
    
    # Test key rules from each Wolfram class
    test_rules = [0, 4, 30, 90, 110, 126]
    
    print("\nRule Embeddings:")
    print("-" * 60)
    print(f"{'Rule':>6} {'Class':>10} {'P':>8} {'A':>8} {'Ξ':>8}")
    print("-" * 60)
    
    for rule in test_rules:
        coords = embedder.embed_rule(rule)
        ca = ElementaryCA(rule)
        cls = ca.rule_table  # Just for class lookup
        from ca_simulator import RULE_CLASSIFICATIONS
        wclass = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
        
        print(f"{rule:>6} {wclass.name:>10} {coords.potential:>8.4f} {coords.actualization:>8.4f} {coords.xi:>8.4f}")
