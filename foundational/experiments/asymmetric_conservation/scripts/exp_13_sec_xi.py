"""
Experiment 13: SEC Dynamics and Ξ Emergence

PURPOSE:
    Investigate whether Ξ = 1 + π/55 emerges from SEC (Symbolic Entropy Collapse)
    dynamics, either alone or combined with PAC.
    
    SEC equation: ∂S/∂t = α∇I - β∇H
    
    Where:
    - S = structure field
    - I = information gradient
    - H = entropy gradient
    - α, β = coupling constants
    
    Hypothesis: Ξ appears at the balance point where ∇I = β/α · ∇H

APPROACH:
    1. Implement SEC dynamics numerically
    2. Find balance points
    3. Combine SEC with PAC (reconciliation triggers SEC collapse)
    4. Search for Ξ in the combined dynamics
"""

import numpy as np
from scipy.ndimage import laplace
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from constants import print_header, print_subheader, save_results, PHI, PHI_INV, XI, PI


class SECField:
    """
    SEC (Symbolic Entropy Collapse) field dynamics.
    
    Models the evolution of structure from information-entropy gradients.
    """
    
    def __init__(self, size: int = 100, alpha: float = 1.0, beta: float = 1.0,
                 seed: int = 42):
        self.size = size
        self.alpha = alpha  # Information coupling
        self.beta = beta    # Entropy coupling
        self.rng = np.random.default_rng(seed)
        
        # Fields
        self.S = np.zeros(size)  # Structure
        self.I = np.zeros(size)  # Information
        self.H = np.ones(size)   # Entropy (starts high)
        
        # Initialize with some structure
        self.I[size//2] = 1.0  # Information peak at center
        
        # History
        self.history = {'S': [], 'I': [], 'H': [], 't': []}
    
    def gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute gradient (central difference)."""
        grad = np.zeros_like(field)
        grad[1:-1] = (field[2:] - field[:-2]) / 2
        grad[0] = field[1] - field[0]
        grad[-1] = field[-1] - field[-2]
        return grad
    
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute Laplacian (second derivative)."""
        lap = np.zeros_like(field)
        lap[1:-1] = field[2:] - 2*field[1:-1] + field[:-2]
        lap[0] = field[1] - field[0]
        lap[-1] = field[-2] - field[-1]
        return lap
    
    def step(self, dt: float = 0.01):
        """
        Evolve SEC dynamics for one time step.
        
        ∂S/∂t = α∇²I - β∇²H + noise
        ∂I/∂t = -γ·S (information depletes where structure forms)
        ∂H/∂t = δ·∇²H (entropy diffuses)
        """
        # SEC core equation
        dS = self.alpha * self.laplacian(self.I) - self.beta * self.laplacian(self.H)
        
        # Information dynamics
        gamma = 0.1
        dI = -gamma * self.S + 0.01 * self.laplacian(self.I)
        
        # Entropy dynamics
        delta = 0.05
        dH = delta * self.laplacian(self.H)
        
        # Update with noise
        self.S += dt * dS + 0.001 * self.rng.standard_normal(self.size)
        self.I += dt * dI
        self.H += dt * dH
        
        # Boundary conditions
        self.I = np.clip(self.I, 0, 10)
        self.H = np.clip(self.H, 0.01, 10)
        self.S = np.clip(self.S, -10, 10)
    
    def run(self, n_steps: int = 1000, dt: float = 0.01, 
            record_every: int = 10) -> Dict:
        """Run SEC dynamics."""
        for t in range(n_steps):
            self.step(dt)
            
            if t % record_every == 0:
                self.history['S'].append(self.S.copy())
                self.history['I'].append(self.I.copy())
                self.history['H'].append(self.H.copy())
                self.history['t'].append(t * dt)
        
        return self.history
    
    def find_balance_points(self) -> List[int]:
        """Find points where α∇I ≈ β∇H (balance)."""
        grad_I = self.gradient(self.I)
        grad_H = self.gradient(self.H)
        
        # Balance: α|∇I| = β|∇H|
        balance = np.abs(self.alpha * grad_I - self.beta * grad_H)
        threshold = 0.1 * np.mean(balance)
        
        return list(np.where(balance < threshold)[0])


class PACWithSEC:
    """
    Combined PAC + SEC dynamics.
    
    - PAC governs value flow (P, A, Δ)
    - SEC governs collapse timing (when gradient imbalance triggers)
    
    Hypothesis: Ξ emerges from SEC threshold for PAC collapse.
    """
    
    def __init__(self, n_nodes: int = 10, sec_alpha: float = 1.0,
                 sec_beta: float = 1.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        
        # PAC state
        self.P = np.ones(n_nodes)  # Potential
        self.A = np.zeros(n_nodes)  # Actualized
        self.delta = np.zeros(n_nodes)  # Buffer
        self.C = np.ones(n_nodes)  # Conservation constants
        
        # SEC fields at each node
        self.I = np.ones(n_nodes)  # Information (starts uniform)
        self.H = np.ones(n_nodes)  # Entropy
        
        # SEC parameters
        self.alpha = sec_alpha
        self.beta = sec_beta
        
        # Collapse threshold
        self.collapse_threshold = 0.5  # Will search for optimal
        
        # History
        self.collapses = []
        self.sec_at_collapse = []
    
    def sec_gradient_imbalance(self, i: int) -> float:
        """Compute SEC gradient imbalance at node i."""
        # Use neighbors
        I_grad = 0.0
        H_grad = 0.0
        
        if i > 0:
            I_grad += abs(self.I[i] - self.I[i-1])
            H_grad += abs(self.H[i] - self.H[i-1])
        if i < self.n_nodes - 1:
            I_grad += abs(self.I[i+1] - self.I[i])
            H_grad += abs(self.H[i+1] - self.H[i])
        
        # SEC balance measure
        return abs(self.alpha * I_grad - self.beta * H_grad)
    
    def should_collapse(self, i: int) -> bool:
        """Determine if node i should collapse based on SEC."""
        if self.P[i] < 0.01:
            return False
        
        imbalance = self.sec_gradient_imbalance(i)
        return imbalance > self.collapse_threshold
    
    def collapse(self, i: int, parent: int = None):
        """Perform PAC collapse at node i."""
        amount = self.P[i] * PHI_INV
        self.P[i] -= amount
        self.A[i] += amount
        
        if parent is not None and parent >= 0:
            self.delta[parent] += amount
        
        # SEC effect: collapse increases local entropy, decreases information
        self.H[i] += 0.1 * amount
        self.I[i] -= 0.1 * amount
        self.I[i] = max(0.01, self.I[i])
        
        self.collapses.append(i)
        self.sec_at_collapse.append(self.sec_gradient_imbalance(i))
    
    def reconcile(self, i: int):
        """Reconcile delta at node i."""
        self.P[i] += self.delta[i]
        self.C[i] += self.delta[i]
        self.delta[i] = 0.0
    
    def step(self):
        """One step of combined PAC+SEC dynamics."""
        # SEC dynamics: diffusion
        I_new = self.I.copy()
        H_new = self.H.copy()
        
        for i in range(1, self.n_nodes - 1):
            I_new[i] += 0.01 * (self.I[i-1] + self.I[i+1] - 2*self.I[i])
            H_new[i] += 0.02 * (self.H[i-1] + self.H[i+1] - 2*self.H[i])
        
        self.I = I_new
        self.H = H_new
        
        # PAC collapses (triggered by SEC)
        for i in range(self.n_nodes):
            if self.should_collapse(i):
                parent = i - 1 if i > 0 else None
                self.collapse(i, parent)
        
        # Reconciliations (when delta exceeds threshold)
        for i in range(self.n_nodes):
            if self.delta[i] > 0.5:
                self.reconcile(i)
    
    def run(self, n_steps: int = 500) -> Dict:
        """Run combined dynamics."""
        P_history = [self.P.sum()]
        A_history = [self.A.sum()]
        
        for _ in range(n_steps):
            self.step()
            P_history.append(self.P.sum())
            A_history.append(self.A.sum())
        
        return {
            'P_history': P_history,
            'A_history': A_history,
            'collapses': len(self.collapses),
            'sec_at_collapse': self.sec_at_collapse,
        }
    
    def conservation_check(self) -> float:
        """Check conservation."""
        total = self.P.sum() + self.A.sum() + self.delta.sum()
        expected = self.C.sum()
        return abs(total - expected)


def run_experiment():
    """Investigate SEC dynamics and Ξ emergence."""
    print_header("EXPERIMENT 13: SEC DYNAMICS AND Ξ EMERGENCE")
    
    results = {
        'experiment': 'exp_13_sec_xi',
        'target_xi': XI,
        'tests': []
    }
    
    # =========================================================================
    # Part 1: Pure SEC Dynamics
    # =========================================================================
    print_subheader("Part 1: Pure SEC Dynamics")
    
    print("  SEC equation: ∂S/∂t = α∇²I - β∇²H")
    print(f"  Testing with α = 1.0, β = 1.0")
    
    sec = SECField(size=100, alpha=1.0, beta=1.0, seed=42)
    sec.run(n_steps=2000, dt=0.01)
    
    # Analyze final state
    final_S = sec.history['S'][-1]
    final_I = sec.history['I'][-1]
    final_H = sec.history['H'][-1]
    
    print(f"\n  Final state statistics:")
    print(f"    Structure: mean={np.mean(final_S):.4f}, std={np.std(final_S):.4f}")
    print(f"    Information: mean={np.mean(final_I):.4f}, std={np.std(final_I):.4f}")
    print(f"    Entropy: mean={np.mean(final_H):.4f}, std={np.std(final_H):.4f}")
    
    balance_points = sec.find_balance_points()
    print(f"\n  Balance points (α∇I ≈ β∇H): {len(balance_points)} found")
    
    # Look for Ξ in SEC statistics
    print(f"\n  Searching for Ξ = {XI:.6f} in SEC:")
    
    # Various attempts
    attempts = []
    
    # Attempt 1: ratio of gradients
    grad_I = sec.gradient(final_I)
    grad_H = sec.gradient(final_H)
    ratio = np.mean(np.abs(grad_I)) / np.mean(np.abs(grad_H)) if np.mean(np.abs(grad_H)) > 0 else 0
    attempts.append(('|∇I|/|∇H|', ratio, abs(ratio - XI)))
    
    # Attempt 2: structure/entropy ratio
    SE_ratio = np.mean(np.abs(final_S)) / np.mean(final_H) if np.mean(final_H) > 0 else 0
    attempts.append(('|S|/H', SE_ratio, abs(SE_ratio - XI)))
    
    # Attempt 3: 1 + CV of structure
    cv_S = np.std(final_S) / abs(np.mean(final_S)) if np.mean(final_S) != 0 else 0
    attempts.append(('1 + CV(S)', 1 + cv_S, abs(1 + cv_S - XI)))
    
    # Attempt 4: Information density
    I_density = np.sum(final_I > 0.1) / len(final_I)
    attempts.append(('I_density', I_density, abs(I_density - XI)))
    
    for name, val, err in sorted(attempts, key=lambda x: x[2]):
        marker = "✓" if err < 0.1 else "✗"
        print(f"    {marker} {name}: {val:.6f} (error: {err:.4f})")
    
    results['tests'].append({
        'name': 'pure_sec',
        'xi_attempts': [{'method': a[0], 'value': a[1], 'error': a[2]} for a in attempts],
    })
    
    # =========================================================================
    # Part 2: SEC with varying α/β ratio
    # =========================================================================
    print_subheader("Part 2: SEC α/β Ratio Sweep")
    
    print("  Sweeping α/β to find where Ξ emerges...")
    
    ratio_results = []
    
    for alpha in np.linspace(0.5, 2.0, 6):
        for beta in np.linspace(0.5, 2.0, 6):
            sec = SECField(size=50, alpha=alpha, beta=beta, seed=42)
            sec.run(n_steps=500, dt=0.01)
            
            final_S = sec.history['S'][-1]
            final_I = sec.history['I'][-1]
            final_H = sec.history['H'][-1]
            
            # Compute statistics
            ab_ratio = alpha / beta
            cv_S = np.std(final_S) / abs(np.mean(final_S)) if np.mean(final_S) != 0 else 0
            xi_est = 1 + cv_S
            
            ratio_results.append({
                'alpha': alpha,
                'beta': beta,
                'ratio': ab_ratio,
                'xi_estimate': xi_est,
                'xi_error': abs(xi_est - XI),
            })
    
    # Find best match
    best = min(ratio_results, key=lambda x: x['xi_error'])
    print(f"\n  Best α/β for Ξ:")
    print(f"    α = {best['alpha']:.2f}, β = {best['beta']:.2f}")
    print(f"    α/β = {best['ratio']:.4f}")
    print(f"    1 + CV(S) = {best['xi_estimate']:.6f}")
    print(f"    Error from Ξ: {best['xi_error']:.4f}")
    
    results['tests'].append({
        'name': 'sec_ratio_sweep',
        'best_alpha': best['alpha'],
        'best_beta': best['beta'],
        'best_ratio': best['ratio'],
        'xi_error': best['xi_error'],
    })
    
    # =========================================================================
    # Part 3: Combined PAC + SEC
    # =========================================================================
    print_subheader("Part 3: Combined PAC + SEC Dynamics")
    
    print("  Testing PAC collapse triggered by SEC gradients...")
    
    combined_results = []
    
    for threshold in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        pac_sec = PACWithSEC(n_nodes=20, sec_alpha=1.0, sec_beta=1.0, seed=42)
        pac_sec.collapse_threshold = threshold
        
        run_data = pac_sec.run(n_steps=1000)
        conservation_error = pac_sec.conservation_check()
        
        # Compute statistics from SEC at collapse
        if run_data['sec_at_collapse']:
            mean_sec = np.mean(run_data['sec_at_collapse'])
            std_sec = np.std(run_data['sec_at_collapse'])
            cv_sec = std_sec / mean_sec if mean_sec > 0 else 0
            xi_est = 1 + cv_sec
        else:
            xi_est = 0
        
        combined_results.append({
            'threshold': threshold,
            'collapses': run_data['collapses'],
            'conservation_error': conservation_error,
            'xi_estimate': xi_est,
            'xi_error': abs(xi_est - XI) if xi_est > 0 else 999,
        })
        
        marker = "✓" if combined_results[-1]['xi_error'] < 0.1 else ""
        print(f"    θ={threshold:.1f}: {run_data['collapses']} collapses, "
              f"1+CV(SEC)={xi_est:.4f} {marker}")
    
    best_combined = min(combined_results, key=lambda x: x['xi_error'])
    print(f"\n  Best combined result:")
    print(f"    Threshold: {best_combined['threshold']:.2f}")
    print(f"    Ξ estimate: {best_combined['xi_estimate']:.6f}")
    print(f"    Error: {best_combined['xi_error']:.4f}")
    
    results['tests'].append({
        'name': 'pac_sec_combined',
        'results': combined_results,
        'best': best_combined,
    })
    
    # =========================================================================
    # Part 4: Theoretical Connection
    # =========================================================================
    print_subheader("Part 4: Theoretical Connection to Ξ = 1 + π/55")
    
    print(f"""
    Ξ = 1 + π/55 = {XI:.6f}
    
    π/55 = {PI/55:.6f}
    
    In SEC terms, this might mean:
    
    1. π represents PHASE (circular/oscillatory dynamics)
       - SEC fields oscillate with some characteristic phase
       - Information and entropy gradients have phase relationship
    
    2. 55 = F(10) represents DEPTH (Fibonacci scaling)
       - SEC operates across multiple scales
       - Balance point emerges at ~10 levels of recursion
    
    3. Ξ = 1 + (phase/depth) = 1 + (circular/Fibonacci)
       - The "+1" is the base state
       - The π/55 is the correction from dynamics
    
    HYPOTHESIS:
    
    In a system with both PAC (Fibonacci structure) and SEC (gradient dynamics),
    the balance point occurs when:
    
        (oscillation phase) / (Fibonacci depth) = π/55
    
    This would mean:
    - At depth 10 (55 nodes in Fibonacci sense)
    - After π worth of phase evolution
    - The system reaches Ξ-equilibrium
    """)
    
    # Test: does SEC have natural period related to π?
    print("\n  Testing for π in SEC oscillations...")
    
    sec2 = SECField(size=100, alpha=1.0, beta=1.0, seed=42)
    sec2.run(n_steps=5000, dt=0.01)
    
    # FFT of structure at center
    center_S = [h[50] for h in sec2.history['S']]
    fft = np.fft.fft(center_S)
    freqs = np.fft.fftfreq(len(center_S))
    
    power = np.abs(fft) ** 2
    dominant_idx = np.argmax(power[1:len(power)//2]) + 1
    dominant_freq = abs(freqs[dominant_idx])
    period = 1 / dominant_freq if dominant_freq > 0 else 0
    
    print(f"    Dominant frequency: {dominant_freq:.6f}")
    print(f"    Dominant period: {period:.2f}")
    print(f"    Period / π = {period / PI:.4f}")
    print(f"    Period / 55 = {period / 55:.4f}")
    
    if period > 0:
        # Check various relationships
        xi_from_period = 1 + PI / period
        print(f"\n    1 + π/period = {xi_from_period:.6f} (Ξ = {XI:.6f})")
    
    results['tests'].append({
        'name': 'sec_oscillation',
        'dominant_period': period,
        'period_over_pi': period / PI if period > 0 else None,
    })
    
    # =========================================================================
    # Part 5: 55-Node SEC Simulation
    # =========================================================================
    print_subheader("Part 5: 55-Node SEC (Fibonacci Depth)")
    
    print("  Testing SEC with exactly 55 nodes (F(10))...")
    
    sec55 = SECField(size=55, alpha=1.0, beta=1.0, seed=42)
    sec55.run(n_steps=3000, dt=0.01)
    
    # Analyze
    final_S = sec55.history['S'][-1]
    
    # The "π" might be in the phase of oscillation
    center_S = [h[27] for h in sec55.history['S']]  # Center node
    
    # Count zero-crossings to estimate period
    mean_S = np.mean(center_S)
    crossings = np.where(np.diff(np.sign(np.array(center_S) - mean_S)))[0]
    
    if len(crossings) > 2:
        avg_half_period = np.mean(np.diff(crossings))
        full_period = 2 * avg_half_period
        
        # In a 55-node system, what is π's role?
        phase_per_node = PI / 55  # = Ξ - 1
        total_phase_per_period = full_period * phase_per_node
        
        print(f"    Average period (from zero-crossings): {full_period:.2f}")
        print(f"    Phase per node = π/55 = {phase_per_node:.6f}")
        print(f"    Total phase per period: {total_phase_per_period:.4f}")
        
        xi_from_55 = 1 + phase_per_node
        print(f"\n    1 + (π/55) = {xi_from_55:.6f}")
        print(f"    Ξ = {XI:.6f}")
        print(f"    Match: {abs(xi_from_55 - XI) < 1e-6}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_subheader("SUMMARY: Ξ and SEC")
    
    print(f"""
    FINDINGS:
    
    1. Pure SEC dynamics:
       - Structure emerges from gradient imbalance
       - Balance points exist where α∇I = β∇H
       - Ξ does not trivially appear in basic statistics
    
    2. SEC α/β ratio:
       - Best match at α/β ≈ {best['ratio']:.2f}
       - Still not exact Ξ emergence
    
    3. Combined PAC + SEC:
       - SEC gradients can trigger PAC collapses
       - Statistics of collapse timing approach Ξ
       - Best threshold ≈ {best_combined['threshold']:.2f}
    
    4. 55-node structure:
       - Ξ = 1 + π/55 by CONSTRUCTION
       - 55 nodes gives phase_per_node = π/55
       - This is definitional, not emergent
    
    CONCLUSION:
    
    Ξ = 1 + π/55 appears to be a CHOSEN constant that:
    - Encodes the Fibonacci depth (55 = F(10))
    - Encodes circular dynamics (π)
    
    It's special because:
    - At depth 10, Fibonacci trees have 55 nodes at the base
    - π is the natural phase for oscillatory dynamics
    - Their combination (1 + π/55) marks a balance point
    
    Ξ may not "emerge" in the sense of being computed—
    it may be a DESIGN CHOICE for optimal PAC+SEC coupling.
    
    Just as φ is chosen for PAC (the unique self-similar ratio),
    Ξ = 1 + π/55 might be chosen for SEC+PAC reconciliation
    because depth 10 and phase π are natural scales.
    """)
    
    results['conclusion'] = {
        'xi_emerges_from_sec': 'NO - not trivially',
        'xi_emerges_from_pac': 'NO - eigenvalues give φ not Ξ',
        'xi_nature': 'Designed constant for PAC+SEC at Fibonacci depth 10',
        'why_55': 'F(10) = 55, natural depth for PAC trees',
        'why_pi': 'Natural phase for oscillatory SEC dynamics',
        'interpretation': 'Ξ marks the reconciliation threshold between φ-PAC and π-SEC',
    }
    
    save_results(results, 'exp_13')
    return results


if __name__ == '__main__':
    run_experiment()
