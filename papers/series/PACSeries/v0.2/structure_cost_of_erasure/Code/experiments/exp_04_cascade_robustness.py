"""
Experiment 04: Robustness Test for ln(φ) Emergence

Tests whether A/(A+ξ) ≈ ln(φ) is robust across many seeds
using the SAME coupling model as exp_01.

If the finding is real, it should hold across seeds.
If it's noise, variance will be high and mean will drift.
"""

import numpy as np
import json
from datetime import datetime

# Constants
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
GAMMA = 0.5772156649
XI_CONSTANT = GAMMA + LN_PHI
INV_PHI = 1 / PHI

k_B = 1.380649e-23  # Boltzmann constant


class LandauerTest:
    """
    Same coupling model as exp_01 for fair comparison.
    """
    
    def __init__(self, n_env_modes=20, n_samples=500000, seed=42):
        self.n_env = n_env_modes
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
        # Energy scale for environment modes (spread around kT)
        T = 300.0
        self.env_energies = k_B * T * (0.5 + self.rng.exponential(1.0, n_env_modes))
        self.env_probs = 1.0 / (1.0 + np.exp(self.env_energies / (k_B * T)))
    
    def sample_initial_state(self):
        """System uniform {0,1}, environment independent thermal."""
        system = self.rng.integers(0, 2, self.n_samples)
        env = np.zeros((self.n_samples, self.n_env), dtype=int)
        for j in range(self.n_env):
            env[:, j] = (self.rng.random(self.n_samples) < self.env_probs[j]).astype(int)
        return system, env
    
    def perform_erasure(self, system, env):
        """
        SAME coupling as exp_01:
        - Direct coupling 0.8 * exp(-0.3*j) to first 5 modes
        - Correlation injection: modes copy mode 0
        """
        system_post = np.zeros_like(system)
        env_post = env.copy()
        
        was_one = (system == 1)
        n_coupling_modes = min(5, self.n_env)
        
        # Direct coupling to first modes
        for j in range(n_coupling_modes):
            coupling = 0.8 * np.exp(-0.3 * j)
            flip_mask = was_one & (self.rng.random(self.n_samples) < coupling)
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
        
        # Correlation injection: subsequent modes copy mode 0
        for j in range(1, n_coupling_modes):
            correlation_mask = was_one & (self.rng.random(self.n_samples) < 0.3 * np.exp(-0.2 * j))
            env_post[correlation_mask, j] = env_post[correlation_mask, 0]
        
        return system_post, env_post
    
    def compute_entropy(self, data):
        """Shannon entropy in bits."""
        if data.ndim == 1:
            values, counts = np.unique(data, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-30))
        else:
            n_modes = min(data.shape[1], 12)
            hashes = np.zeros(data.shape[0], dtype=np.int64)
            for j in range(n_modes):
                hashes += data[:, j].astype(np.int64) * (2 ** j)
            values, counts = np.unique(hashes, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-30))
    
    def compute_total_correlation(self, data, n_modes=None):
        """TC = sum(H_i) - H_joint"""
        if n_modes is None:
            n_modes = min(data.shape[1], 12)
        
        sum_H = sum(self.compute_entropy(data[:, j]) for j in range(n_modes))
        H_joint = self.compute_entropy(data[:, :n_modes])
        return max(0, sum_H - H_joint)
    
    def compute_transfer(self, system_pre, env_post):
        """MI between pre-erasure system and post-erasure environment."""
        n_modes = min(5, self.n_env)
        
        env_hash = np.zeros(len(system_pre), dtype=np.int64)
        for j in range(n_modes):
            env_hash += env_post[:, j].astype(np.int64) * (2 ** j)
        
        H_sys = self.compute_entropy(system_pre)
        H_env = self.compute_entropy(env_hash)
        
        joint = system_pre * (2 ** 20) + env_hash
        vals, cnts = np.unique(joint, return_counts=True)
        H_joint = -np.sum((cnts / cnts.sum()) * np.log2(cnts / cnts.sum() + 1e-30))
        
        return max(0, H_sys + H_env - H_joint)
    
    def compute_pairwise_mi(self, env):
        """Sum of pairwise MI between all mode pairs."""
        n_modes = min(env.shape[1], self.n_env)
        total_mi = 0.0
        
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                # Joint distribution
                joint = env[:, i] * 2 + env[:, j]
                values, counts = np.unique(joint, return_counts=True)
                p_joint = counts / counts.sum()
                
                p_i = np.array([np.mean(env[:, i] == 0), np.mean(env[:, i] == 1)])
                p_j = np.array([np.mean(env[:, j] == 0), np.mean(env[:, j] == 1)])
                
                H_i = -np.sum(p_i * np.log2(p_i + 1e-30))
                H_j = -np.sum(p_j * np.log2(p_j + 1e-30))
                H_ij = -np.sum(p_joint * np.log2(p_joint + 1e-30))
                
                mi = max(0, H_i + H_j - H_ij)
                total_mi += mi
        
        return total_mi
    
    def run(self):
        """Execute and return PAC components - MATCHING exp_01 exactly."""
        sys_pre, env_pre = self.sample_initial_state()
        sys_post, env_post = self.perform_erasure(sys_pre, env_pre)
        
        # Pre-erasure measures
        H_sys_pre = self.compute_entropy(sys_pre)
        TC_pre = self.compute_total_correlation(env_pre)
        pairwise_pre = self.compute_pairwise_mi(env_pre)
        
        # Post-erasure measures  
        TC_post = self.compute_total_correlation(env_post)
        pairwise_post = self.compute_pairwise_mi(env_post)
        
        # Transfer: MI(S_pre; E_post)
        transfer = self.compute_transfer(sys_pre, env_post)
        
        # Deltas
        delta_TC = TC_post - TC_pre
        delta_pairwise = pairwise_post - pairwise_pre
        
        # PAC components (EXACT exp_01 formula)
        P = H_sys_pre  # ~1.0
        A = transfer   # actual: recoverable info
        xi = delta_TC + delta_pairwise  # structure: new correlations
        
        return {'P': P, 'A': A, 'xi': xi, 'delta_TC': delta_TC, 'delta_pairwise': delta_pairwise}


def main():
    print("=" * 60)
    print("ROBUSTNESS TEST: A/(A+ξ) ≈ ln(φ)?")
    print("Same coupling model as exp_01 across many seeds")
    print("=" * 60)
    print()
    
    # Fixed parameters (matching exp_01)
    n_env_modes = 20
    n_samples = 500_000
    n_seeds = 50
    
    print(f"Environment modes: {n_env_modes}")
    print(f"Samples per run: {n_samples:,}")
    print(f"Seeds: {n_seeds}")
    print()
    
    # Target values
    print("Target constants:")
    print(f"  ln(φ) = {LN_PHI:.6f}")
    print(f"  γ     = {GAMMA:.6f}")
    print(f"  Ξ     = {XI_CONSTANT:.6f}")
    print(f"  1/φ   = {INV_PHI:.6f}")
    print()
    
    all_A = []
    all_xi = []
    all_ratios = []
    
    for seed in range(n_seeds):
        exp = LandauerTest(n_env_modes, n_samples, seed)
        result = exp.run()
        
        A = result['A']
        xi = result['xi']
        
        if A + xi > 0:
            ratio = A / (A + xi)
        else:
            ratio = np.nan
        
        all_A.append(A)
        all_xi.append(xi)
        all_ratios.append(ratio)
        
        if seed < 10 or seed % 10 == 0:
            print(f"Seed {seed:3d}: A={A:.4f}, ξ={xi:.4f}, A/(A+ξ)={ratio:.4f}")
    
    all_ratios = np.array(all_ratios)
    valid = ~np.isnan(all_ratios)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    mean_ratio = np.mean(all_ratios[valid])
    std_ratio = np.std(all_ratios[valid])
    cv = std_ratio / mean_ratio if mean_ratio > 0 else np.nan
    
    print(f"A/(A+ξ) across {np.sum(valid)} seeds:")
    print(f"  Mean:   {mean_ratio:.6f}")
    print(f"  Std:    {std_ratio:.6f}")
    print(f"  CV:     {cv:.2%}")
    print(f"  Min:    {np.min(all_ratios[valid]):.6f}")
    print(f"  Max:    {np.max(all_ratios[valid]):.6f}")
    print()
    
    # Also report A and ξ separately
    print(f"A (transfer) across seeds:")
    print(f"  Mean: {np.mean(all_A):.4f}, Std: {np.std(all_A):.4f}")
    print(f"ξ (structure) across seeds:")
    print(f"  Mean: {np.mean(all_xi):.4f}, Std: {np.std(all_xi):.4f}")
    print()
    
    # Compare to constants
    print("Comparison to PAC constants:")
    print()
    
    targets = {
        'ln(φ)': LN_PHI,
        'γ': GAMMA,
        'Ξ': XI_CONSTANT,
        '1/φ': INV_PHI,
    }
    
    for name, value in targets.items():
        diff = mean_ratio - value
        pct_diff = abs(diff / value) * 100
        sigma = abs(diff / std_ratio) if std_ratio > 0 else np.nan
        print(f"  {name:6s} = {value:.6f}")
        print(f"          Mean diff: {diff:+.6f} ({pct_diff:.2f}%)")
        print(f"          {sigma:.2f}σ away from mean")
        print()
    
    # Find closest match
    best_match = min(targets.items(), key=lambda x: abs(mean_ratio - x[1]))
    print(f"Closest match: {best_match[0]} ({abs(mean_ratio - best_match[1])/best_match[1]*100:.2f}% from mean)")
    print()
    
    # Verdict
    ln_phi_diff = abs(mean_ratio - LN_PHI) / LN_PHI * 100
    if ln_phi_diff < 5 and cv < 0.20:
        verdict = "✅ ROBUST: ln(φ) match holds with acceptable variance"
    elif ln_phi_diff < 10:
        verdict = "⚠️  MARGINAL: ln(φ) close but high variance"
    else:
        verdict = "❌ NOT ROBUST: ln(φ) match does not replicate"
    
    print(f"VERDICT: {verdict}")
    print(f"  ln(φ) diff: {ln_phi_diff:.2f}%")
    print(f"  CV: {cv:.2%}")
    
    # Save results
    output = {
        'experiment': 'exp_04_robustness_test',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_env_modes': n_env_modes,
            'n_samples': n_samples,
            'n_seeds': n_seeds,
        },
        'results': {
            'A_mean': float(np.mean(all_A)),
            'A_std': float(np.std(all_A)),
            'xi_mean': float(np.mean(all_xi)),
            'xi_std': float(np.std(all_xi)),
            'ratio_mean': float(mean_ratio),
            'ratio_std': float(std_ratio),
            'ratio_cv': float(cv),
            'all_A': [float(x) for x in all_A],
            'all_xi': [float(x) for x in all_xi],
            'all_ratios': [float(x) for x in all_ratios[valid]],
        },
        'comparison': {
            name: {
                'value': float(value),
                'diff_pct': float(abs(mean_ratio - value) / value * 100),
            }
            for name, value in targets.items()
        },
        'verdict': verdict,
    }
    
    output_path = '../results/exp_04_robustness_test.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
