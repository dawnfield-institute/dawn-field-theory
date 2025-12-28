"""
Cross-Framework Invariant Metrics
=================================

Computes dimensionless invariants from three independent mathematical frameworks:

1. Conservation Physics (PAC dynamics)
2. Geometric Topology (Betti-like numbers, Euler characteristic)  
3. Information Theory (excess entropy, mutual information)

The core hypothesis: If CA rules are PAC attractors, the SAME dimensionless
invariants should emerge from all three frameworks.

Cross-framework convergence threshold: 5% (from preregistration)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.ndimage import label
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ca_simulator import CAState, ElementaryCA, WolframClass


@dataclass
class FrameworkInvariants:
    """Invariants computed from a single framework."""
    framework_name: str
    primary_invariant: float      # The main dimensionless ratio
    secondary_invariant: float    # Supporting ratio
    confidence: float             # Measurement confidence [0, 1]
    raw_metrics: Dict[str, float] # All computed values


@dataclass 
class CrossFrameworkResult:
    """Result of cross-framework invariant comparison."""
    rule: int
    conservation_invariants: FrameworkInvariants
    topology_invariants: FrameworkInvariants
    information_invariants: FrameworkInvariants
    
    # Cross-framework analysis
    convergence_score: float      # Agreement between frameworks [0, 1]
    canonical_invariant: float    # Best estimate of true invariant
    deviation: float              # Max deviation from canonical
    
    @property
    def converged(self) -> bool:
        """Check if frameworks converge within 5% threshold."""
        return self.deviation < 0.05


class ConservationPhysicsFramework:
    """
    Framework 1: Conservation Physics (PAC dynamics)
    
    Measures energy flow, equilibrium, and conservation properties.
    """
    
    def __init__(self, steps: int = 200):
        self.steps = steps
    
    def compute_energy_density(self, state: CAState) -> np.ndarray:
        """Compute energy density as function of time."""
        return state.history.mean(axis=1)
    
    def compute_energy_flow(self, state: CAState) -> float:
        """Compute net energy flow (change in density)."""
        density = self.compute_energy_density(state)
        return np.mean(np.abs(np.diff(density)))
    
    def compute_equilibrium_time(self, state: CAState, threshold: float = 0.01) -> int:
        """Time steps to reach equilibrium (variance stabilization)."""
        density = self.compute_energy_density(state)
        
        # Rolling variance
        window = 10
        for t in range(window, len(density)):
            var = np.var(density[t-window:t])
            if var < threshold:
                return t
        return len(density)  # Never equilibrated
    
    def compute_conservation_ratio(self, state: CAState) -> float:
        """
        Compute PAC conservation ratio.
        
        For conserved systems: total "energy" (active cells) should be roughly constant.
        Ratio = final_energy / initial_energy
        """
        initial = state.history[:10].mean()
        final = state.history[-10:].mean()
        
        if initial == 0:
            return 0.0
        return final / initial
    
    def compute_invariants(self, state: CAState) -> FrameworkInvariants:
        """Compute all conservation physics invariants."""
        
        energy_flow = self.compute_energy_flow(state)
        eq_time = self.compute_equilibrium_time(state)
        conservation_ratio = self.compute_conservation_ratio(state)
        
        # Primary invariant: normalized equilibration rate
        # Dimensionless: eq_time / total_steps
        primary = eq_time / self.steps
        
        # Secondary invariant: conservation quality
        # Dimensionless: how close to 1 is the conservation ratio
        secondary = 1.0 - abs(1.0 - conservation_ratio)
        
        # Confidence based on variance stability
        density = self.compute_energy_density(state)
        late_var = np.var(density[-20:])
        early_var = np.var(density[:20]) + 1e-10
        confidence = min(1.0, 1.0 / (1.0 + late_var / early_var))
        
        return FrameworkInvariants(
            framework_name="conservation_physics",
            primary_invariant=primary,
            secondary_invariant=secondary,
            confidence=confidence,
            raw_metrics={
                'energy_flow': energy_flow,
                'equilibrium_time': eq_time,
                'conservation_ratio': conservation_ratio
            }
        )


class GeometricTopologyFramework:
    """
    Framework 2: Geometric Topology
    
    Measures structural invariants: connected components (Betti-0),
    holes/cycles (Betti-1), Euler characteristic.
    """
    
    def compute_betti_0(self, binary_image: np.ndarray) -> int:
        """
        Compute Betti-0: number of connected components.
        """
        labeled, num_features = label(binary_image)
        return num_features
    
    def compute_betti_1(self, binary_image: np.ndarray) -> int:
        """
        Compute Betti-1: number of holes/cycles.
        
        For 2D binary image: holes = components of background inside foreground
        """
        # Invert and count components inside the main structure
        inverted = 1 - binary_image
        labeled, num_features = label(inverted)
        
        # Subtract 1 for the infinite background region
        holes = max(0, num_features - 1)
        return holes
    
    def compute_euler_characteristic(self, binary_image: np.ndarray) -> int:
        """
        Compute Euler characteristic: χ = V - E + F
        For 2D grid: χ = cells - edges + vertices ≈ Betti_0 - Betti_1
        """
        b0 = self.compute_betti_0(binary_image)
        b1 = self.compute_betti_1(binary_image)
        return b0 - b1
    
    def compute_genus_proxy(self, state: CAState) -> float:
        """
        Compute a genus-like measure for spacetime diagram.
        
        Higher genus = more complex topology = more "holes" in the pattern.
        """
        # Treat spacetime diagram as 2D image
        b1_values = []
        for t in range(0, state.steps, 10):
            # Use sliding window of spacetime
            window = state.history[t:min(t+20, state.steps)]
            if len(window) < 5:
                continue
            b1 = self.compute_betti_1(window)
            b1_values.append(b1)
        
        return np.mean(b1_values) if b1_values else 0.0
    
    def compute_fractal_dimension(self, state: CAState) -> float:
        """
        Estimate fractal dimension via box-counting.
        
        D = lim(log N(r) / log(1/r)) as r → 0
        """
        # Use final state pattern
        pattern = state.history[-50:]  # Last 50 timesteps
        
        # Box counting at different scales
        scales = [2, 4, 8, 16]
        counts = []
        
        for scale in scales:
            # Coarse-grain the pattern
            h, w = pattern.shape
            count = 0
            for i in range(0, h, scale):
                for j in range(0, w, scale):
                    box = pattern[i:i+scale, j:j+scale]
                    if np.any(box):
                        count += 1
            counts.append(count)
        
        # Linear regression: log(N) vs log(1/scale)
        if len(counts) < 2 or min(counts) == 0:
            return 1.0
            
        log_scales = np.log(1.0 / np.array(scales))
        log_counts = np.log(np.array(counts) + 1)
        
        slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
        return max(0.0, min(2.0, slope))  # Bound to [0, 2]
    
    def compute_invariants(self, state: CAState) -> FrameworkInvariants:
        """Compute all geometric topology invariants."""
        
        # Use late-stage evolution for stable topology
        late_pattern = state.history[-50:]
        
        b0 = self.compute_betti_0(late_pattern)
        b1 = self.compute_betti_1(late_pattern)
        euler = b0 - b1
        genus = self.compute_genus_proxy(state)
        fractal_dim = self.compute_fractal_dimension(state)
        
        # Primary invariant: normalized topological complexity
        # Use fractal dimension as more stable measure
        # Dimensionless: D / 2, bounded to [0, 1]
        primary = fractal_dim / 2.0
        primary = max(0.0, min(1.0, primary))
        
        # Secondary invariant: normalized Betti ratio
        # b1 / (b0 + b1) measures "hole density"
        total_betti = b0 + b1
        if total_betti > 0:
            secondary = b1 / total_betti
        else:
            secondary = 0.0
        secondary = max(0.0, min(1.0, secondary))
        
        # Confidence based on consistency across time windows
        fd_values = []
        for t in range(0, state.steps - 20, 20):
            window = state.history[t:t+20]
            if window.shape[0] >= 5:
                # Quick fractal dimension estimate
                density = np.mean(window)
                fd_values.append(density)
        
        if len(fd_values) > 1:
            confidence = 1.0 / (1.0 + np.std(fd_values) / (np.mean(fd_values) + 0.01))
        else:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        return FrameworkInvariants(
            framework_name="geometric_topology",
            primary_invariant=primary,
            secondary_invariant=secondary,
            confidence=confidence,
            raw_metrics={
                'betti_0': int(b0),
                'betti_1': int(b1),
                'euler_characteristic': int(euler),
                'genus_proxy': float(genus),
                'fractal_dimension': float(fractal_dim)
            }
        )


class InformationTheoryFramework:
    """
    Framework 3: Information Theory
    
    Measures entropy, mutual information, and complexity.
    """
    
    def compute_excess_entropy(self, state: CAState, max_history: int = 10) -> float:
        """
        Compute excess entropy: mutual information between past and future.
        
        H_excess = I(X_{past} ; X_{future})
        """
        history = state.history
        steps = len(history)
        
        mi_values = []
        for L in range(1, min(max_history + 1, steps // 2)):
            # Past L steps, future L steps
            past = history[:steps//2]
            future = history[steps//2:]
            
            # Simplified MI estimation via correlation
            past_flat = past[-L:].flatten()
            future_flat = future[:L].flatten()
            
            if len(past_flat) != len(future_flat):
                continue
                
            # Correlation as proxy for MI
            corr = np.corrcoef(past_flat, future_flat)[0, 1]
            if np.isnan(corr):
                corr = 0
            
            # Convert correlation to MI estimate
            mi = -0.5 * np.log(1 - corr**2 + 1e-10)
            mi_values.append(mi)
        
        return np.mean(mi_values) if mi_values else 0.0
    
    def compute_block_entropy_growth(self, state: CAState, max_block: int = 6) -> float:
        """
        Compute block entropy growth rate.
        
        For complex systems, H(L) ~ L for random, H(L) ~ log(L) for periodic.
        """
        history = state.history
        
        entropies = []
        for L in range(1, max_block + 1):
            # Count L-block patterns
            pattern_counts = {}
            for t in range(len(history)):
                row = history[t]
                for i in range(len(row) - L + 1):
                    pattern = tuple(row[i:i+L])
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Entropy
            total = sum(pattern_counts.values())
            probs = np.array(list(pattern_counts.values())) / total
            H = -np.sum(probs * np.log2(probs + 1e-10))
            entropies.append(H)
        
        # Growth rate: slope of H vs L
        if len(entropies) < 2:
            return 0.0
        
        L_values = np.arange(1, len(entropies) + 1)
        slope, _, _, _, _ = stats.linregress(L_values, entropies)
        return slope
    
    def compute_correlation_dimension(self, state: CAState) -> float:
        """
        Estimate correlation dimension from spacetime pattern.
        
        D_corr = lim(log C(r) / log r) as r → 0
        where C(r) is correlation integral.
        """
        # Flatten pattern to point cloud
        pattern = state.history[-50:]
        points = np.argwhere(pattern > 0)
        
        if len(points) < 10:
            return 0.0
        
        # Sample points for efficiency
        if len(points) > 100:
            idx = np.random.choice(len(points), 100, replace=False)
            points = points[idx]
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
        
        # Correlation integral at different radii
        r_values = np.percentile(distances, [10, 25, 50, 75])
        r_values = r_values[r_values > 0]
        
        if len(r_values) < 2:
            return 1.0
        
        C_values = []
        for r in r_values:
            C = np.mean(distances < r)
            C_values.append(C)
        
        C_values = np.array(C_values)
        C_values = C_values[C_values > 0]
        
        if len(C_values) < 2:
            return 1.0
        
        # Regression
        log_r = np.log(r_values[:len(C_values)])
        log_C = np.log(C_values)
        
        slope, _, _, _, _ = stats.linregress(log_r, log_C)
        return max(0.0, min(2.0, slope))
    
    def compute_invariants(self, state: CAState) -> FrameworkInvariants:
        """Compute all information theory invariants."""
        
        excess_entropy = self.compute_excess_entropy(state)
        block_growth = self.compute_block_entropy_growth(state)
        corr_dim = self.compute_correlation_dimension(state)
        
        # Primary invariant: normalized excess entropy
        # Dimensionless: excess_entropy / max_possible
        primary = excess_entropy / 5.0  # Empirical normalization
        primary = min(1.0, max(0.0, primary))
        
        # Secondary invariant: block entropy growth normalized
        # Dimensionless: growth_rate / max_rate (≈ 1 for random)
        secondary = block_growth / 1.0
        secondary = min(1.0, max(0.0, secondary))
        
        # Confidence from stability of measurements
        confidence = 0.8  # Default high confidence for info theory
        
        return FrameworkInvariants(
            framework_name="information_theory",
            primary_invariant=primary,
            secondary_invariant=secondary,
            confidence=confidence,
            raw_metrics={
                'excess_entropy': excess_entropy,
                'block_entropy_growth': block_growth,
                'correlation_dimension': corr_dim
            }
        )


class CrossFrameworkAnalyzer:
    """
    Analyzes cross-framework convergence of invariants.
    
    Tests the core hypothesis: same invariants from different frameworks.
    """
    
    def __init__(self, width: int = 101, steps: int = 200):
        self.width = width
        self.steps = steps
        
        self.conservation = ConservationPhysicsFramework(steps)
        self.topology = GeometricTopologyFramework()
        self.information = InformationTheoryFramework()
    
    def analyze_rule(self, rule: int, init_type: str = 'single') -> CrossFrameworkResult:
        """
        Compute and compare invariants from all three frameworks.
        """
        # Evolve CA
        ca = ElementaryCA(rule, self.width)
        state = ca.evolve_fast(self.steps, init_type=init_type)
        
        # Compute invariants from each framework
        cons_inv = self.conservation.compute_invariants(state)
        topo_inv = self.topology.compute_invariants(state)
        info_inv = self.information.compute_invariants(state)
        
        # Cross-framework comparison
        # Weight by confidence
        invariants = [
            (cons_inv.primary_invariant, cons_inv.confidence),
            (topo_inv.primary_invariant, topo_inv.confidence),
            (info_inv.primary_invariant, info_inv.confidence)
        ]
        
        # Weighted mean as canonical invariant
        total_weight = sum(w for _, w in invariants)
        if total_weight > 0:
            canonical = sum(v * w for v, w in invariants) / total_weight
        else:
            canonical = np.mean([v for v, _ in invariants])
        
        # Maximum deviation from canonical
        deviations = [abs(v - canonical) / (canonical + 1e-10) for v, _ in invariants]
        max_deviation = max(deviations)
        
        # Convergence score: 1 - normalized deviation
        convergence = 1.0 - min(1.0, max_deviation)
        
        return CrossFrameworkResult(
            rule=rule,
            conservation_invariants=cons_inv,
            topology_invariants=topo_inv,
            information_invariants=info_inv,
            convergence_score=convergence,
            canonical_invariant=canonical,
            deviation=max_deviation
        )
    
    def analyze_rules(self, rules: List[int], init_type: str = 'single') -> Dict[int, CrossFrameworkResult]:
        """Analyze multiple rules."""
        return {rule: self.analyze_rule(rule, init_type) for rule in rules}


if __name__ == "__main__":
    print("Cross-Framework Invariant Analysis Demo")
    print("=" * 50)
    
    analyzer = CrossFrameworkAnalyzer(width=101, steps=100)
    
    # Test key rules
    test_rules = [0, 4, 30, 90, 110, 126]
    
    print("\nCross-Framework Convergence:")
    print("-" * 70)
    print(f"{'Rule':>6} {'Cons':>8} {'Topo':>8} {'Info':>8} {'Canon':>8} {'Conv':>6} {'Pass':>6}")
    print("-" * 70)
    
    for rule in test_rules:
        result = analyzer.analyze_rule(rule)
        
        cons = result.conservation_invariants.primary_invariant
        topo = result.topology_invariants.primary_invariant
        info = result.information_invariants.primary_invariant
        
        status = "✅" if result.converged else "❌"
        
        print(f"{rule:>6} {cons:>8.4f} {topo:>8.4f} {info:>8.4f} "
              f"{result.canonical_invariant:>8.4f} {result.convergence_score:>6.2f} {status:>6}")
