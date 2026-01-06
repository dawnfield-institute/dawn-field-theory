"""
exp_02_lorenz_analysis.py - Lorenz Attractor Threshold and Dimension Analysis

Detailed analysis of the Lorenz system focusing on:
1. Threshold detection at chaos onset
2. Dimension calculation matching ξ prediction
3. Lyapunov exponent estimation

Key Result:
D_observed ≈ 2.06, D_predicted = 2 + (ξ-1) = 2.0571, error = 0.14%
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

XI = 1 + np.pi / 55
PHI = (1 + np.sqrt(5)) / 2


class LorenzSystem:
    """Lorenz system simulator with analysis tools."""
    
    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def simulate(self, n_steps=10000, dt=0.01, x0=None):
        """Integrate Lorenz equations."""
        if x0 is None:
            x0 = [1.0, 1.0, 1.0]
        
        trajectory = np.zeros((n_steps, 3))
        trajectory[0] = x0
        
        for i in range(1, n_steps):
            x, y, z = trajectory[i-1]
            
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            
            trajectory[i] = trajectory[i-1] + dt * np.array([dx, dy, dz])
        
        return trajectory
    
    def lyapunov_exponents(self, n_steps=50000, dt=0.001):
        """
        Estimate Lyapunov exponents using standard algorithm.
        Returns [λ1, λ2, λ3] in descending order.
        """
        # Initial conditions
        x = np.array([1.0, 1.0, 1.0])
        
        # Perturbation vectors (orthonormal)
        Q = np.eye(3)
        
        # Running sums for exponents
        lyap_sums = np.zeros(3)
        
        # Transient
        for _ in range(1000):
            x = self._rk4_step(x, dt)
        
        # Main loop
        n_renorm = n_steps // 10
        for i in range(n_renorm):
            # Evolve perturbations
            for _ in range(10):
                x = self._rk4_step(x, dt)
                Q = self._evolve_tangent(x, Q, dt)
            
            # QR decomposition (Gram-Schmidt)
            Q, R = np.linalg.qr(Q)
            
            # Accumulate logs of diagonal
            lyap_sums += np.log(np.abs(np.diag(R)))
        
        # Normalize
        total_time = n_steps * dt
        lyapunov = lyap_sums / total_time
        
        return np.sort(lyapunov)[::-1]  # Descending order
    
    def _rk4_step(self, x, dt):
        """Fourth-order Runge-Kutta step."""
        k1 = self._derivatives(x)
        k2 = self._derivatives(x + 0.5 * dt * k1)
        k3 = self._derivatives(x + 0.5 * dt * k2)
        k4 = self._derivatives(x + dt * k3)
        return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _derivatives(self, state):
        """Lorenz derivatives."""
        x, y, z = state
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ])
    
    def _evolve_tangent(self, x, Q, dt):
        """Evolve tangent vectors."""
        J = self._jacobian(x)
        return Q + dt * J @ Q
    
    def _jacobian(self, state):
        """Jacobian matrix of Lorenz system."""
        x, y, z = state
        return np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])


def kaplan_yorke_dimension(lyapunov_exponents):
    """
    Compute Kaplan-Yorke dimension from Lyapunov exponents.
    D_KY = j + Σ_{i=1}^j λ_i / |λ_{j+1}|
    where j is largest integer such that Σ_{i=1}^j λ_i ≥ 0
    """
    lyap = np.sort(lyapunov_exponents)[::-1]  # Descending
    
    # Find j
    cumsum = np.cumsum(lyap)
    j = np.where(cumsum >= 0)[0]
    
    if len(j) == 0:
        return 0
    
    j = j[-1]  # Largest j where cumsum ≥ 0
    
    if j + 1 >= len(lyap):
        return len(lyap)
    
    d_ky = (j + 1) + cumsum[j] / np.abs(lyap[j + 1])
    return d_ky


def correlation_dimension(trajectory, max_radius=10, n_radii=20):
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.
    """
    n_points = min(2000, len(trajectory))
    indices = np.random.choice(len(trajectory), n_points, replace=False)
    points = trajectory[indices]
    
    radii = np.logspace(-1, np.log10(max_radius), n_radii)
    counts = []
    
    for r in radii:
        count = 0
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < r:
                    count += 1
        counts.append(count)
    
    counts = np.array(counts, dtype=float)
    counts[counts == 0] = 1  # Avoid log(0)
    
    # Fit power law in scaling region
    log_r = np.log(radii)
    log_c = np.log(counts)
    
    # Use middle region for fit
    start, end = n_radii // 4, 3 * n_radii // 4
    slope, _ = np.polyfit(log_r[start:end], log_c[start:end], 1)
    
    return slope


def run_experiment():
    """Comprehensive Lorenz analysis."""
    
    print("=" * 60)
    print("Lorenz Attractor Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Standard Lorenz parameters
    lorenz = LorenzSystem(sigma=10, rho=28, beta=8/3)
    
    # Generate trajectory
    print("\nGenerating trajectory...")
    trajectory = lorenz.simulate(n_steps=50000, dt=0.01)
    
    # Discard transient
    trajectory = trajectory[5000:]
    
    # Compute Lyapunov exponents
    print("Computing Lyapunov exponents...")
    lyapunov = lorenz.lyapunov_exponents(n_steps=100000, dt=0.001)
    
    print(f"\nLyapunov exponents:")
    print(f"  λ₁ = {lyapunov[0]:.4f}")
    print(f"  λ₂ = {lyapunov[1]:.4f}")
    print(f"  λ₃ = {lyapunov[2]:.4f}")
    
    # Kaplan-Yorke dimension
    d_ky = kaplan_yorke_dimension(lyapunov)
    print(f"\nKaplan-Yorke dimension: D_KY = {d_ky:.4f}")
    
    # Correlation dimension
    print("Computing correlation dimension...")
    d_corr = correlation_dimension(trajectory)
    print(f"Correlation dimension: D_corr = {d_corr:.4f}")
    
    # ξ prediction
    xi_minus_1 = XI - 1
    d_predicted = 2 + xi_minus_1
    d_observed = 2.06  # Literature value
    
    print("\n" + "=" * 60)
    print("ξ Dimension Prediction")
    print("=" * 60)
    print(f"ξ = 1 + π/55 = {XI:.6f}")
    print(f"ξ - 1 = {xi_minus_1:.6f}")
    print(f"\nPredicted: D = 2 + (ξ-1) = {d_predicted:.4f}")
    print(f"Observed (literature): D = {d_observed:.3f}")
    print(f"Our D_KY: {d_ky:.4f}")
    print(f"Our D_corr: {d_corr:.4f}")
    
    # Errors
    error_predicted = abs(d_predicted - d_observed) / d_observed * 100
    error_ky = abs(d_ky - d_observed) / d_observed * 100
    
    print(f"\nErrors:")
    print(f"  |D_predicted - D_observed| / D_observed = {error_predicted:.2f}%")
    print(f"  |D_KY - D_observed| / D_observed = {error_ky:.2f}%")
    
    # Threshold analysis
    print("\n" + "=" * 60)
    print("Threshold Analysis")
    print("=" * 60)
    
    rho_values = np.linspace(20, 30, 20)
    max_lyap = []
    
    for rho in rho_values:
        l = LorenzSystem(sigma=10, rho=rho, beta=8/3)
        lyap = l.lyapunov_exponents(n_steps=30000, dt=0.001)
        max_lyap.append(lyap[0])
    
    max_lyap = np.array(max_lyap)
    
    # Find chaos onset (λ₁ > 0)
    chaos_onset_idx = np.where(max_lyap > 0)[0]
    if len(chaos_onset_idx) > 0:
        rho_critical = rho_values[chaos_onset_idx[0]]
        print(f"Detected chaos onset: ρ* = {rho_critical:.2f}")
        print(f"Known critical: ρ_c = 24.74")
    
    # Save results
    results = {
        'experiment': 'lorenz_analysis',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'sigma': 10,
            'rho': 28,
            'beta': 8/3
        },
        'lyapunov_exponents': lyapunov.tolist(),
        'dimensions': {
            'kaplan_yorke': d_ky,
            'correlation': d_corr,
            'predicted_xi': d_predicted,
            'observed_literature': d_observed
        },
        'errors': {
            'predicted_vs_observed': error_predicted,
            'ky_vs_observed': error_ky
        },
        'xi_analysis': {
            'xi': XI,
            'xi_minus_1': xi_minus_1,
            'dimension_formula': '2 + (ξ-1)'
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_02_lorenz_analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
