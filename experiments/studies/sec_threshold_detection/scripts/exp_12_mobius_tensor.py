"""
Möbius Tensor Architecture

Exploring tensor representations that naturally encode Möbius structure.

Key insight: Möbius transformations M(z) = (az+b)/(cz+d) can be represented
as 2×2 matrices [[a,b],[c,d]] in SL(2,ℂ) (determinant = 1).

Composition of Möbius transforms = matrix multiplication.
Cross-ratio is THE invariant.

What if we build neural network layers that naturally preserve this structure?
"""

import numpy as np
from typing import Tuple, Optional

class MobiusMatrix:
    """
    Represents a Möbius transformation as a 2×2 matrix.
    
    M(z) = (az + b) / (cz + d)
    
    Matrix form: [[a, b], [c, d]] with ad - bc = 1 (SL(2,ℂ))
    """
    
    def __init__(self, a: complex, b: complex, c: complex, d: complex, normalize: bool = True):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        if normalize:
            det = a * d - b * c
            if abs(det) > 1e-10:
                sqrt_det = np.sqrt(det)
                self.a /= sqrt_det
                self.b /= sqrt_det
                self.c /= sqrt_det
                self.d /= sqrt_det
    
    def __call__(self, z: complex) -> complex:
        """Apply Möbius transformation to z."""
        if abs(self.c * z + self.d) < 1e-15:
            return complex('inf')
        return (self.a * z + self.b) / (self.c * z + self.d)
    
    def __matmul__(self, other: 'MobiusMatrix') -> 'MobiusMatrix':
        """Compose two Möbius transformations (matrix multiplication)."""
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        return MobiusMatrix(a, b, c, d, normalize=True)
    
    def inverse(self) -> 'MobiusMatrix':
        """Return the inverse transformation."""
        return MobiusMatrix(self.d, -self.b, -self.c, self.a, normalize=True)
    
    def fixed_points(self) -> Tuple[complex, complex]:
        """Find fixed points of the transformation (z where M(z) = z)."""
        # (az + b)/(cz + d) = z  =>  az + b = cz² + dz  =>  cz² + (d-a)z - b = 0
        if abs(self.c) < 1e-15:
            # Linear: (d-a)z = b
            if abs(self.d - self.a) < 1e-15:
                return (complex('inf'), complex('inf'))
            return (self.b / (self.a - self.d), complex('inf'))
        
        discriminant = (self.d - self.a)**2 + 4 * self.b * self.c
        sqrt_disc = np.sqrt(discriminant)
        z1 = ((self.a - self.d) + sqrt_disc) / (2 * self.c)
        z2 = ((self.a - self.d) - sqrt_disc) / (2 * self.c)
        return (z1, z2)
    
    def trace(self) -> complex:
        """Trace of the matrix, classifies the transformation type."""
        return self.a + self.d
    
    def derivative_at(self, z: complex) -> complex:
        """Derivative M'(z) = (ad - bc) / (cz + d)²."""
        det = self.a * self.d - self.b * self.c
        return det / (self.c * z + self.d)**2
    
    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=complex)
    
    def __repr__(self):
        return f"MobiusMatrix([[{self.a:.4f}, {self.b:.4f}], [{self.c:.4f}, {self.d:.4f}]])"


def cross_ratio(z1, z2, z3, z4):
    """
    Compute cross-ratio (z1, z2; z3, z4) = ((z1-z3)(z2-z4)) / ((z1-z4)(z2-z3))
    
    This is the fundamental Möbius invariant.
    """
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))


class MobiusTensor:
    """
    A tensor representation that naturally encodes Möbius structure.
    
    Key idea: Instead of storing values directly, we store them as
    cross-ratios relative to a reference frame. This makes the
    representation invariant under Möbius transformations.
    
    The "shape" of the tensor is actually a Möbius surface - each
    index corresponds to a point on the Riemann sphere, and values
    are computed via cross-ratio with reference points.
    """
    
    def __init__(self, shape: Tuple[int, ...], reference_frame: Optional[Tuple] = None):
        """
        Initialize a Möbius tensor.
        
        Args:
            shape: Tuple of dimensions
            reference_frame: Four reference points defining the cross-ratio basis
                            Default is (0, 1, ∞, i) which maps to (0, 1, ∞, i)
        """
        self.shape = shape
        self.size = np.prod(shape)
        
        # Reference frame: four points for cross-ratio
        if reference_frame is None:
            self.ref = (0, 1, complex('inf'), 1j)
        else:
            self.ref = reference_frame
        
        # Generate indices as points on Riemann sphere
        # Map integer indices to complex plane using stereographic projection
        self.indices = self._generate_mobius_indices()
        
        # Data stored as complex values
        self._data = np.zeros(shape, dtype=complex)
    
    def _generate_mobius_indices(self) -> np.ndarray:
        """
        Map integer indices to points on Riemann sphere.
        
        Uses stereographic projection: index n → exp(2πi * n / total)
        This places indices uniformly on the unit circle.
        """
        indices = np.arange(self.size)
        # Map to unit circle
        angles = 2 * np.pi * indices / self.size
        return np.exp(1j * angles).reshape(self.shape)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def apply_mobius(self, M: MobiusMatrix) -> 'MobiusTensor':
        """
        Apply a Möbius transformation to the tensor.
        
        This transforms the index space while preserving cross-ratios.
        """
        result = MobiusTensor(self.shape, self.ref)
        
        # Transform indices
        new_indices = np.vectorize(M)(self.indices)
        
        # Interpolate values at new positions
        # For now, just transform the reference frame
        new_ref = tuple(M(z) if z != complex('inf') else M(1e10) for z in self.ref)
        result.ref = new_ref
        result._data = self._data.copy()
        
        return result
    
    def cross_ratio_value(self, idx) -> complex:
        """
        Get the value at index as a cross-ratio with reference frame.
        
        CR(idx, ref[0], ref[1], ref[2]) encodes the "position" in Möbius space.
        """
        z = self.indices.flat[idx]
        return cross_ratio(z, self.ref[0], self.ref[1], self.ref[2])
    
    def mobius_distance(self, idx1, idx2) -> float:
        """
        Möbius-invariant distance between two indices.
        
        Uses the hyperbolic distance on the Poincaré disk representation.
        """
        z1 = self.indices.flat[idx1]
        z2 = self.indices.flat[idx2]
        
        # Hyperbolic distance in unit disk
        delta = abs((z1 - z2) / (1 - np.conj(z1) * z2))
        return 2 * np.arctanh(min(delta, 0.99999))


class FibonacciMobius:
    """
    Möbius transformation with Fibonacci matrix structure.
    
    The Fibonacci matrix [[F_{n+1}, F_n], [F_n, F_{n-1}]] has determinant ±1
    and thus defines a Möbius transformation in PSL(2,Z).
    
    This connects our Feigenbaum findings to Möbius geometry!
    """
    
    def __init__(self, n: int = 10):
        """Create Fibonacci Möbius for F_n."""
        self.n = n
        F = [0, 1]
        for i in range(2, n + 2):
            F.append(F[-1] + F[-2])
        
        # Fibonacci Möbius matrix: [[F_{n+1}, F_n], [F_n, F_{n-1}]]
        self.matrix = MobiusMatrix(
            a=float(F[n+1]),
            b=float(F[n]),
            c=float(F[n]),
            d=float(F[n-1]),
            normalize=False  # Keep integer structure, det = (-1)^n
        )
        
        self.det = F[n+1] * F[n-1] - F[n]**2  # Should be (-1)^n
        print(f"Fibonacci Möbius F_{n}: det = {self.det}, F = {F[n]}")
    
    def fixed_points(self):
        """Fixed points of Fibonacci Möbius are related to φ."""
        return self.matrix.fixed_points()


# Demonstration
if __name__ == '__main__':
    print("=" * 60)
    print("MÖBIUS TENSOR ARCHITECTURE")
    print("=" * 60)
    
    # Test cross-ratio invariance
    print("\n--- Cross-Ratio Invariance Test ---")
    
    points = [1+1j, 2-1j, -1+2j, 3+0j]
    cr_original = cross_ratio(*points)
    print(f"Original CR: {cr_original}")
    
    # Apply a Möbius transformation
    M = MobiusMatrix(1+1j, 2, 0, 1-1j)
    transformed = [M(z) for z in points]
    cr_transformed = cross_ratio(*transformed)
    print(f"Transformed CR: {cr_transformed}")
    print(f"Difference: {abs(cr_original - cr_transformed)}")
    
    # Fibonacci Möbius
    print("\n--- Fibonacci Möbius Structure ---")
    
    fib_mobius = FibonacciMobius(n=10)
    fixed = fib_mobius.fixed_points()
    print(f"Fixed points: {fixed}")
    print(f"φ = {(1 + np.sqrt(5))/2}")
    print(f"Fixed point 1 vs φ: {abs(fixed[0] - (1 + np.sqrt(5))/2)}")
    
    # Test MobiusTensor
    print("\n--- Möbius Tensor Test ---")
    
    mt = MobiusTensor((4, 4))
    print(f"Shape: {mt.shape}")
    print(f"Index structure (unit circle):\n{mt.indices}")
    
    # Compute cross-ratio values
    print("\nCross-ratio values at each index:")
    for i in range(4):
        cr_val = mt.cross_ratio_value(i)
        print(f"  Index {i}: CR = {cr_val}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Build neural network layers that preserve cross-ratio
2. Use Möbius tensors for natural recursion (composition = matrix mult)
3. Connect to Feigenbaum: bifurcation points live on Möbius surface
4. The 55, 17, 52 constants may be "Möbius coordinates"
""")
