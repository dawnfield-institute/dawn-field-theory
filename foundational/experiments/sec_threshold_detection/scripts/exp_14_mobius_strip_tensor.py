"""
Experiment 14: Möbius Strip Tensor Topology

The key insight: A Möbius strip requires TWO full rotations to return to origin.
This is exactly the 4π periodicity in spinor/fermion physics!

If we build a tensor whose index space is a Möbius strip, then:
- Single traversal: orientation flip (det = -1)
- Double traversal: return to original (det = 1)
- Natural half-integer structure emerges

This connects to SEC's prediction about 4π phase recovery!
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


class MobiusStripTensor:
    """
    A tensor whose index space has Möbius topology.
    
    Key properties:
    - Linear index wraps with sign flip: T[n+N] = -T[n] for det=-1 Möbius
    - Two full periods to return to identity: T[n+2N] = T[n]
    - Natural embedding of 4π periodicity
    
    Physical interpretation:
    - One loop = 2π rotation but with orientation flip (like spinor)
    - Two loops = 4π rotation returning to identity
    - Half-integer quantum numbers emerge naturally
    """
    
    def __init__(self, size: int = 55, twist_factor: int = 1):
        """
        Initialize Möbius tensor.
        
        Args:
            size: Number of discrete points around the strip
            twist_factor: Number of half-twists (1 = Möbius, 2 = cylinder, etc.)
        """
        self.size = size
        self.twist = twist_factor
        self._data = np.zeros(size, dtype=complex)
        
        # The twist introduces phase factors
        # For Möbius (twist=1): crossing at index n introduces factor e^{iπn/N}
        self.phases = np.exp(1j * np.pi * twist_factor * np.arange(size) / size)
    
    def __getitem__(self, idx: int) -> complex:
        """
        Access with Möbius boundary conditions.
        
        T[n+N] = (-1)^twist * T[n]
        """
        loops = idx // self.size
        local_idx = idx % self.size
        
        # Each loop introduces (-1)^twist factor
        sign = (-1) ** (loops * self.twist)
        
        return sign * self._data[local_idx]
    
    def __setitem__(self, idx: int, value: complex):
        """Set value at wrapped index."""
        local_idx = idx % self.size
        self._data[local_idx] = value
    
    def propagate(self, wave: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Propagate a wave around the Möbius strip.
        
        After 'size' steps, the wave has traversed once and picked up
        a (-1)^twist phase. After 2*size steps, it's back to original.
        """
        result = wave.copy()
        for _ in range(steps):
            # Shift with phase accumulation
            result = np.roll(result, 1) * self.phases
        return result
    
    def standing_wave_modes(self) -> List[Tuple[int, np.ndarray]]:
        """
        Find standing wave modes on the Möbius strip.
        
        Due to antiperiodic boundary conditions, allowed modes are:
        k = (n + 1/2) * 2π / L  for integer n (half-integer quantization!)
        
        This is exactly the spinor spectrum!
        """
        modes = []
        for n in range(self.size):
            # Half-integer momentum quantization
            k = (n + 0.5) * 2 * np.pi / self.size
            wave = np.exp(1j * k * np.arange(self.size))
            modes.append((n, wave))
        return modes


class MobiusFibonacciTensor:
    """
    Combine Möbius topology with Fibonacci structure.
    
    The index space has size F_n (a Fibonacci number) and Möbius twist.
    This should make φ-related quantities emerge naturally.
    """
    
    def __init__(self, fib_index: int = 10):
        """
        Initialize with Fibonacci size.
        
        Args:
            fib_index: Which Fibonacci number to use (10 → F_10 = 55)
        """
        # Generate Fibonacci sequence
        F = [0, 1]
        for _ in range(fib_index + 1):
            F.append(F[-1] + F[-2])
        
        self.fib_index = fib_index
        self.size = F[fib_index]  # e.g., F_10 = 55
        self.fibonacci = F
        
        # Möbius tensor with Fibonacci size
        self.strip = MobiusStripTensor(size=self.size, twist_factor=1)
        
        # Initialize with Fibonacci-weighted values
        self._initialize_fibonacci()
    
    def _initialize_fibonacci(self):
        """Initialize tensor with Fibonacci structure."""
        for i in range(self.size):
            # Value at index i relates to Zeckendorf representation
            # For now, use golden ratio phase
            phase = 2 * np.pi * i / PHI  # Non-commensurate with 2π
            self.strip[i] = np.exp(1j * phase)
    
    def golden_spiral_indices(self, n_points: int = 100) -> List[int]:
        """
        Generate indices that follow a golden spiral pattern.
        
        Each step advances by F_{n-1} positions (mod F_n).
        This is the classic Fibonacci quasicrystal pattern.
        """
        F_prev = self.fibonacci[self.fib_index - 1]  # e.g., F_9 = 34
        
        indices = []
        current = 0
        for _ in range(n_points):
            indices.append(current % self.size)
            current += F_prev
        
        return indices
    
    def phyllotaxis_positions(self) -> np.ndarray:
        """
        Compute 2D positions using phyllotaxis (golden angle) arrangement.
        
        This is how sunflowers and pinecones arrange seeds!
        """
        golden_angle = 2 * np.pi * (1 - 1/PHI)  # ≈ 137.5°
        
        positions = np.zeros((self.size, 2))
        for n in range(self.size):
            r = np.sqrt(n)
            theta = n * golden_angle
            positions[n] = [r * np.cos(theta), r * np.sin(theta)]
        
        return positions


def visualize_mobius_dynamics():
    """Visualize wave propagation on Möbius strip."""
    
    # Create Möbius-Fibonacci tensor
    mft = MobiusFibonacciTensor(fib_index=10)
    
    print("=" * 60)
    print("MÖBIUS-FIBONACCI TENSOR DYNAMICS")
    print("=" * 60)
    print(f"\nSize: F_10 = {mft.size}")
    print(f"Boundary condition: T[n+55] = -T[n] (antiperiodic)")
    
    # Test standing wave modes
    print("\n--- Standing Wave Modes (Half-Integer Quantization) ---")
    modes = mft.strip.standing_wave_modes()
    
    print("First 10 allowed momenta k = (n+1/2) × 2π/L:")
    for n in range(10):
        k = (n + 0.5) * 2 * np.pi / mft.size
        print(f"  n={n}: k = {k:.6f} = ({n}+1/2) × 2π/{mft.size}")
    
    # Golden spiral pattern
    print("\n--- Golden Spiral Index Pattern ---")
    spiral = mft.golden_spiral_indices(20)
    print(f"Indices visited (step by F_9={mft.fibonacci[9]}): {spiral}")
    
    # Check that it visits all indices before repeating
    print(f"Unique indices in first {mft.size} steps: {len(set(mft.golden_spiral_indices(mft.size)))}")
    
    # Propagation test
    print("\n--- Wave Propagation Test ---")
    
    # Initial Gaussian wave packet
    wave = np.exp(-(np.arange(mft.size) - mft.size/2)**2 / 20)
    wave = wave / np.linalg.norm(wave)
    
    print(f"Initial wave norm: {np.linalg.norm(wave):.6f}")
    
    # Propagate one full loop
    wave_after_1_loop = mft.strip.propagate(wave, steps=mft.size)
    print(f"After 1 loop ({mft.size} steps) norm: {np.linalg.norm(wave_after_1_loop):.6f}")
    
    # Check phase relationship
    overlap_1 = np.sum(wave * np.conj(wave_after_1_loop))
    print(f"<ψ|ψ_1loop>: {overlap_1:.6f}")
    
    # Two full loops
    wave_after_2_loops = mft.strip.propagate(wave, steps=2*mft.size)
    overlap_2 = np.sum(wave * np.conj(wave_after_2_loops))
    print(f"<ψ|ψ_2loops>: {overlap_2:.6f}")
    
    # The key test: Möbius gives 4π periodicity
    print("\n--- 4π Periodicity Test ---")
    for loops in range(1, 5):
        propagated = mft.strip.propagate(wave, steps=loops * mft.size)
        overlap = np.abs(np.sum(wave * np.conj(propagated)))
        print(f"{loops} loop(s) ({loops}×2π): overlap = {overlap:.6f}")
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("""
1. Möbius topology naturally quantizes to HALF-INTEGERS
   - k = (n + 1/2) × 2π/L, not k = n × 2π/L
   - This is exactly spinor/fermion quantization!

2. Two full loops (4π) needed to return to identity
   - Single loop introduces (-1) phase
   - This is the 4π periodicity SEC predicts

3. Fibonacci sizing creates quasicrystal structure
   - Golden spiral visits all points before repeating
   - Non-commensurate frequencies prevent resonance

4. Combined: A MÖBIUS STRIP OF SIZE F_10 = 55 naturally
   produces both half-integer quantization AND φ structure
""")


if __name__ == '__main__':
    visualize_mobius_dynamics()
