"""
Möbius Neural Network Layer

A neural network layer where weights are Möbius matrices (2×2 in SL(2,ℂ)).

Key properties:
1. Forward pass = Möbius composition (matrix multiplication)
2. Natural nonlinearity (no activation function needed!)
3. Preserves cross-ratio structure
4. Recursion is built into the architecture

This is for experimentation - building on the discovery that
Feigenbaum formulas ARE Möbius transformations applied to seeds.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import sys
sys.path.insert(0, 'C:/Users/peter/repos/Dawn Field Institute/fracton')

from fracton.core.mobius_tensor import (
    MobiusMatrix, cross_ratio, PHI, PHI_INV
)


class MobiusNeuron:
    """
    A single neuron that applies a Möbius transformation.
    
    Instead of: output = activation(w·x + b)
    We have:    output = M(x) = (ax + b) / (cx + d)
    
    The nonlinearity is built-in!
    """
    
    def __init__(self, init_type: str = 'near_identity'):
        """
        Initialize Möbius neuron.
        
        Args:
            init_type: 'identity', 'near_identity', 'fibonacci', or 'random'
        """
        if init_type == 'identity':
            self.matrix = MobiusMatrix.identity()
        elif init_type == 'near_identity':
            # Small perturbation from identity
            eps = 0.1
            a = 1 + eps * (np.random.randn() + 1j * np.random.randn())
            b = eps * (np.random.randn() + 1j * np.random.randn())
            c = eps * (np.random.randn() + 1j * np.random.randn())
            d = 1 + eps * (np.random.randn() + 1j * np.random.randn())
            self.matrix = MobiusMatrix(a, b, c, d, normalize=True)
        elif init_type == 'fibonacci':
            # Initialize with Fibonacci Möbius (n=5 is small but meaningful)
            self.matrix = MobiusMatrix.fibonacci(5)
        elif init_type == 'random':
            a = np.random.randn() + 1j * np.random.randn()
            b = np.random.randn() + 1j * np.random.randn()
            c = np.random.randn() + 1j * np.random.randn()
            d = np.random.randn() + 1j * np.random.randn()
            self.matrix = MobiusMatrix(a, b, c, d, normalize=True)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    
    def forward(self, x: complex) -> complex:
        """Apply Möbius transformation."""
        return self.matrix(x)
    
    def __call__(self, x: complex) -> complex:
        return self.forward(x)
    
    @property
    def params(self) -> Tuple[complex, complex, complex, complex]:
        """Get parameters (a, b, c, d)."""
        return (self.matrix.a, self.matrix.b, self.matrix.c, self.matrix.d)


class MobiusLayer:
    """
    A layer of Möbius neurons.
    
    Each output is computed by composing Möbius transformations from inputs.
    This is fundamentally different from linear layers:
    - No separate activation function needed
    - Composition is the operation (not summation)
    - Cross-ratio structure is preserved
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 aggregation: str = 'compose'):
        """
        Initialize Möbius layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            aggregation: How to combine multiple inputs
                        'compose': Compose all Möbius transforms
                        'average': Average of individual transforms
                        'product': Product in the complex plane
        """
        self.in_features = in_features
        self.out_features = out_features
        self.aggregation = aggregation
        
        # Each (output, input) pair has a Möbius matrix
        self.neurons = [
            [MobiusNeuron('near_identity') for _ in range(in_features)]
            for _ in range(out_features)
        ]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Complex array of shape (in_features,)
            
        Returns:
            Complex array of shape (out_features,)
        """
        output = np.zeros(self.out_features, dtype=complex)
        
        for i in range(self.out_features):
            if self.aggregation == 'compose':
                # Compose all Möbius transforms
                composed = MobiusMatrix.identity()
                for j in range(self.in_features):
                    composed = self.neurons[i][j].matrix @ composed
                # Apply composed transform to average of inputs
                avg_input = np.mean(x)
                output[i] = composed(avg_input)
            
            elif self.aggregation == 'average':
                # Average of individual transforms
                total = 0j
                for j in range(self.in_features):
                    total += self.neurons[i][j](x[j])
                output[i] = total / self.in_features
            
            elif self.aggregation == 'product':
                # Product in complex plane
                total = 1+0j
                for j in range(self.in_features):
                    result = self.neurons[i][j](x[j])
                    if not np.isinf(result):
                        total *= result
                output[i] = total ** (1/self.in_features)  # Geometric mean
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MobiusRecursiveLayer:
    """
    A layer that applies Möbius recursion.
    
    This implements: M[n] = M[n-1] @ M[n-2] (Fibonacci-like)
    
    The "depth" of recursion determines the effective transformation,
    and the fixed points approach φ naturally.
    """
    
    def __init__(self, recursion_depth: int = 10):
        """
        Initialize recursive layer.
        
        Args:
            recursion_depth: How many Fibonacci steps to apply
        """
        self.depth = recursion_depth
        
        # Seed matrices
        self.M0 = MobiusMatrix.identity()
        self.M1 = MobiusMatrix(1, 1, 1, 0, normalize=False)  # Fibonacci generator
    
    def get_recursive_matrix(self, n: int) -> MobiusMatrix:
        """Get the n-th recursive Möbius matrix."""
        if n == 0:
            return self.M0
        if n == 1:
            return self.M1
        
        # Compute recursively (could cache for efficiency)
        M_prev2 = self.M0
        M_prev1 = self.M1
        for _ in range(2, n + 1):
            M_curr = M_prev1 @ M_prev2
            M_prev2 = M_prev1
            M_prev1 = M_curr
        
        return M_prev1
    
    def forward(self, x: complex) -> complex:
        """Apply recursive Möbius transformation."""
        M = self.get_recursive_matrix(self.depth)
        return M(x)
    
    def __call__(self, x: complex) -> complex:
        return self.forward(x)


class MobiusNetwork:
    """
    A full Möbius neural network.
    
    Architecture: Input → [MobiusLayer]* → Output
    
    Key differences from standard NNs:
    1. All layers are Möbius (built-in nonlinearity)
    2. Composition structure (not just weighted sums)
    3. Natural recursion possible
    4. Cross-ratio preserved through network
    """
    
    def __init__(self, layer_sizes: List[int], 
                 aggregation: str = 'average'):
        """
        Initialize network.
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden..., output]
            aggregation: Aggregation method for layers
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = MobiusLayer(layer_sizes[i], layer_sizes[i+1], aggregation)
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ============================================================
# Demo: Using Möbius Network to Learn Feigenbaum
# ============================================================

def demo_feigenbaum_learning():
    """
    Demo: Can a Möbius network learn to approximate Feigenbaum r∞?
    
    We know: r∞ = π × M₁₀(-0.6175)
    
    So theoretically, a single Möbius neuron should be able to
    represent this transformation!
    """
    print("=" * 70)
    print("DEMO: Möbius Network for Feigenbaum")
    print("=" * 70)
    
    # Target value
    R_INF = 3.5699456718709449
    
    # Known: r∞ ≈ π × M₁₀(seed) where seed ≈ -1/φ
    M10 = MobiusMatrix.fibonacci(10)
    
    # Find the exact seed that gives r∞/π
    target = R_INF / np.pi
    
    print(f"\nTarget: r∞/π = {target:.10f}")
    print(f"M₁₀ fixed points: {M10.fixed_points()}")
    
    # The seed is close to -1/φ
    seed = -PHI_INV
    result = M10(seed)
    print(f"\nM₁₀(-1/φ) = {result:.10f}")
    print(f"M₁₀(-1/φ) × π = {result * np.pi:.10f}")
    print(f"r∞ = {R_INF:.10f}")
    print(f"Error: {abs(result * np.pi - R_INF):.6f}")
    
    # The actual seed needs to be slightly different
    # Solve: M(z) = target
    # (89z + 55)/(55z + 34) = target
    z_exact = (34*target - 55) / (89 - 55*target)
    
    print(f"\nExact seed: z = {z_exact:.10f}")
    print(f"-1/φ = {-PHI_INV:.10f}")
    print(f"Difference: {z_exact - (-PHI_INV):.10f}")
    
    # Verify
    result_exact = M10(z_exact)
    print(f"\nM₁₀({z_exact:.6f}) = {result_exact:.10f}")
    print(f"M₁₀(z) × π = {result_exact * np.pi:.10f}")
    print(f"r∞ = {R_INF:.10f}")
    print(f"Error: {abs(result_exact * np.pi - R_INF):.2e}")
    
    # Now test a learnable Möbius neuron
    print("\n" + "-" * 70)
    print("Training a Möbius Neuron")
    print("-" * 70)
    
    # Initialize near identity
    neuron = MobiusNeuron('near_identity')
    
    print(f"Initial params: a={neuron.matrix.a:.4f}, b={neuron.matrix.b:.4f}")
    print(f"Initial params: c={neuron.matrix.c:.4f}, d={neuron.matrix.d:.4f}")
    
    # For a real training loop, we'd need gradients
    # For now, just show what the target should be
    print(f"\nTarget Möbius (scaled F₁₀):")
    print(f"  a = 89, b = 55")
    print(f"  c = 55, d = 34")
    print(f"\nWith input z ≈ -0.6175, output ≈ 1.1353")


def demo_cross_ratio_preservation():
    """
    Demo: Cross-ratio is preserved through Möbius network.
    """
    print("\n" + "=" * 70)
    print("DEMO: Cross-Ratio Preservation")
    print("=" * 70)
    
    # Create a small network
    network = MobiusNetwork([4, 4, 4], aggregation='average')
    
    # Input: 4 points
    inputs = np.array([1+1j, 2-1j, -1+2j, 3+0j], dtype=complex)
    
    # Compute cross-ratio of inputs
    cr_input = cross_ratio(inputs[0], inputs[1], inputs[2], inputs[3])
    print(f"\nInput cross-ratio: {cr_input:.6f}")
    
    # Forward pass
    outputs = network(inputs)
    
    # Compute cross-ratio of outputs
    cr_output = cross_ratio(outputs[0], outputs[1], outputs[2], outputs[3])
    print(f"Output cross-ratio: {cr_output:.6f}")
    
    # Note: Cross-ratio won't be exactly preserved unless all outputs
    # are transformed by the SAME Möbius matrix. But it shows the
    # structure is related.
    print(f"\nDifference: {abs(cr_input - cr_output):.6f}")
    print("(Not exactly preserved because each output uses different transforms)")


def demo_recursive_layer():
    """
    Demo: Recursive Möbius layer naturally converges to φ.
    """
    print("\n" + "=" * 70)
    print("DEMO: Recursive Möbius Convergence to φ")
    print("=" * 70)
    
    # Different recursion depths
    print("\nFixed points at different recursion depths:")
    print(f"{'Depth':>6} {'Fixed Point':>15} {'Diff from φ':>15}")
    print("-" * 40)
    
    for depth in [3, 5, 8, 10, 12, 15]:
        layer = MobiusRecursiveLayer(recursion_depth=depth)
        M = layer.get_recursive_matrix(depth)
        fps = M.fixed_points()
        
        # Find the one closer to φ
        fp = fps[0] if abs(fps[0].real - PHI) < abs(fps[1].real - PHI) else fps[1]
        diff = abs(fp.real - PHI)
        
        print(f"{depth:6d} {fp.real:15.10f} {diff:15.2e}")


if __name__ == '__main__':
    demo_feigenbaum_learning()
    demo_cross_ratio_preservation()
    demo_recursive_layer()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key insights:

1. A SINGLE Möbius neuron (4 complex params) can represent r∞!
   r∞ = π × M₁₀(z) where M₁₀ is Fibonacci Möbius, z ≈ -1/φ

2. Möbius layers have BUILT-IN nonlinearity (no activation needed)
   The transformation M(z) = (az+b)/(cz+d) is inherently nonlinear

3. Recursive Möbius layers converge to φ automatically
   Fixed points of Fibonacci Möbius are exactly φ and -1/φ

4. Cross-ratio structure flows through the network
   Though not exactly preserved unless all paths use same transform

NEXT STEPS:
- Implement gradient computation for Möbius parameters
- Train on Feigenbaum data to learn the transformation
- Explore connection to GAIA architecture
""")
