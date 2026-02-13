"""
Experiment 19: Θ Correction — Closing the ln(φ) Gap
=====================================================
Dawn Field Institute — PAC Exploration Series

HYPOTHESIS: The ~2% gap between measured A/(A+ξ) ≈ 0.490 and predicted
ln(φ) = 0.4812 arises from the thermal residual Θ. Specifically:

  - The derivation assumes A + ξ = P = 1 bit (perfect conservation)
  - In simulation, A + ξ ≈ 0.88, with Θ ≈ 0.12
  - Θ exists because mode-0 coupling = 0.80, not 1.0

This experiment:
  1. Sweeps coupling efficiency from 0.5 to 1.0
  2. Measures A, ξ, Θ, and A/(A+ξ) at each coupling strength
  3. Tests whether A/(A+ξ) → ln(φ) as coupling → 1.0 (Θ → 0)
  4. Tests whether A/(A+ξ) → ln(φ) as coupling → 1.0 (perfect erasure)
  5. Derives the functional relationship: A/(A+ξ) = f(Θ)
  6. Multi-seed robustness at each coupling level

If A/(A+ξ) converges to ln(φ) in the perfect-coupling limit,
the derivation is validated and the measured ~2% gap is EXPLAINED
by finite coupling efficiency — a physical effect, not a theoretical error.
"""

import numpy as np
from scipy import stats
import json
import os
from datetime import datetime

# ============================================================
# CONSTANTS
# ============================================================
k_B = 1.380649e-23
T = 300.0
LN_PHI = np.log((1 + np.sqrt(5)) / 2)  # 0.48121182505960344
PHI = (1 + np.sqrt(5)) / 2

print("=" * 70)
print("EXP 19: Θ Correction — Closing the ln(φ) Gap")  
print("=" * 70)
print(f"Target: ln(φ) = {LN_PHI:.10f}")
print()

# ============================================================
# CORE ERASURE MODEL (from exp_01, parameterized)
# ============================================================

class ParameterizedErasure:
    """
    Same model as exp_01 but with tunable base coupling strength.
    
    base_coupling controls the maximum flip probability at mode 0.
    The cascade still decays exponentially across modes.
    """
    
    def __init__(self, n_env=20, n_samples=500000, seed=42,
                 base_coupling=0.8, flip_decay=0.3, corr_base=0.3, corr_decay=0.2):
        self.n_env = n_env
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)
        self.base_coupling = base_coupling
        self.flip_decay = flip_decay
        self.corr_base = corr_base
        self.corr_decay = corr_decay
        
        # Environment setup (same as exp_01)
        self.env_energies = k_B * T * (0.5 + self.rng.exponential(1.0, n_env))
        self.env_probs = 1.0 / (1.0 + np.exp(self.env_energies / (k_B * T)))
    
    def run_single(self):
        """Run one erasure experiment, return PAC budget."""
        # Initial state
        system = self.rng.randint(0, 2, self.n_samples)
        env = np.zeros((self.n_samples, self.n_env), dtype=int)
        for j in range(self.n_env):
            env[:, j] = (self.rng.random(self.n_samples) < self.env_probs[j]).astype(int)
        
        # Pre-erasure measurements
        H_sys_pre = compute_entropy(system)
        TC_env_pre = compute_total_correlation(env)
        pairwise_pre = compute_pairwise_mi(env)
        
        # Perform erasure
        was_one = (system == 1)
        system_post = np.zeros_like(system)
        env_post = env.copy()
        
        n_coupling = min(5, self.n_env)
        
        # Transfer (flip) coupling — parameterized
        for j in range(n_coupling):
            coupling = self.base_coupling * np.exp(-self.flip_decay * j)
            flip_mask = was_one & (self.rng.random(self.n_samples) < coupling)
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
        
        # Correlation coupling — parameterized
        for j in range(1, n_coupling):
            corr_strength = self.corr_base * np.exp(-self.corr_decay * j)
            corr_mask = was_one & (self.rng.random(self.n_samples) < corr_strength)
            env_post[corr_mask, j] = env_post[corr_mask, 0]
        
        # Post-erasure measurements
        TC_env_post = compute_total_correlation(env_post)
        pairwise_post = compute_pairwise_mi(env_post)
        
        # Information transfer (how much env knows about pre-erasure system)
        transfer = compute_transfer(system, env_post)
        
        # PAC budget
        P = H_sys_pre  # potential (should be ~1.0 bit)
        A = transfer    # actual (recoverable information)
        xi = (TC_env_post - TC_env_pre) + (pairwise_post - pairwise_pre)  # new structure
        theta = P - (A + xi)  # residual
        
        ratio = A / (A + xi) if (A + xi) > 1e-10 else float('nan')
        
        return {
            'P': P, 'A': A, 'xi': xi, 'theta': theta,
            'ratio': ratio, 'A_plus_xi': A + xi
        }


def compute_entropy(data):
    """Shannon entropy in bits."""
    if data.ndim == 1:
        _, counts = np.unique(data, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-30))
    else:
        n_modes = min(data.shape[1], 12)
        hashes = np.zeros(data.shape[0], dtype=np.int64)
        for j in range(n_modes):
            hashes += data[:, j].astype(np.int64) * (2 ** j)
        _, counts = np.unique(hashes, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-30))


def compute_total_correlation(data, n_modes=None):
    """Total correlation (multi-information)."""
    if n_modes is None:
        n_modes = min(data.shape[1], 12)
    sum_H = sum(compute_entropy(data[:, j]) for j in range(n_modes))
    H_joint = compute_entropy(data[:, :n_modes])
    return max(0, sum_H - H_joint)


def compute_pairwise_mi(env):
    """Sum of pairwise mutual information across env modes."""
    n_modes = min(env.shape[1], 20)
    total = 0.0
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            joint = env[:, i] * 2 + env[:, j]
            _, counts = np.unique(joint, return_counts=True)
            p_joint = counts / counts.sum()
            p_i = np.array([np.mean(env[:, i] == 0), np.mean(env[:, i] == 1)])
            p_j = np.array([np.mean(env[:, j] == 0), np.mean(env[:, j] == 1)])
            H_i = -np.sum(p_i * np.log2(p_i + 1e-30))
            H_j = -np.sum(p_j * np.log2(p_j + 1e-30))
            H_ij = -np.sum(p_joint * np.log2(p_joint + 1e-30))
            total += max(0, H_i + H_j - H_ij)
    return total


def compute_transfer(system_pre, env_post):
    """How much env knows about pre-erasure system state."""
    n_modes = min(5, env_post.shape[1])
    env_hash = np.zeros(len(system_pre), dtype=np.int64)
    for j in range(n_modes):
        env_hash += env_post[:, j].astype(np.int64) * (2 ** j)
    
    H_s = compute_entropy(system_pre)
    H_e = compute_entropy(env_hash)
    joint = system_pre * (2 ** 20) + env_hash
    _, counts = np.unique(joint, return_counts=True)
    H_se = -np.sum((counts / counts.sum()) * np.log2(counts / counts.sum() + 1e-30))
    return max(0, H_s + H_e - H_se)


# ============================================================
# TEST 1: Coupling strength sweep
# ============================================================

print("=" * 70)
print("TEST 1: Coupling Strength Sweep (Θ → 0 limit)")
print("=" * 70)

coupling_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
n_seeds = 20
n_samples = 300000

sweep_results = []

for bc in coupling_values:
    seed_results = []
    for s in range(n_seeds):
        exp = ParameterizedErasure(
            n_env=20, n_samples=n_samples, seed=s,
            base_coupling=bc, flip_decay=0.3, corr_base=0.3, corr_decay=0.2
        )
        r = exp.run_single()
        seed_results.append(r)
    
    ratios = [r['ratio'] for r in seed_results if not np.isnan(r['ratio'])]
    thetas = [r['theta'] for r in seed_results]
    a_plus_xis = [r['A_plus_xi'] for r in seed_results]
    
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    mean_theta = np.mean(thetas)
    mean_apx = np.mean(a_plus_xis)
    ci_lo = mean_ratio - 1.96 * std_ratio / np.sqrt(len(ratios))
    ci_hi = mean_ratio + 1.96 * std_ratio / np.sqrt(len(ratios))
    pct_from_lnphi = abs(mean_ratio - LN_PHI) / LN_PHI * 100
    
    result = {
        'base_coupling': bc,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'ci_95': [ci_lo, ci_hi],
        'mean_theta': mean_theta,
        'mean_A_plus_xi': mean_apx,
        'pct_from_ln_phi': pct_from_lnphi,
        'ln_phi_in_ci': ci_lo <= LN_PHI <= ci_hi,
        'n_seeds': n_seeds
    }
    sweep_results.append(result)
    
    in_ci = "✓" if ci_lo <= LN_PHI <= ci_hi else "✗"
    print(f"  coupling={bc:.2f}  ratio={mean_ratio:.6f} ± {std_ratio:.4f}  "
          f"Θ={mean_theta:.4f}  A+ξ={mean_apx:.4f}  "
          f"vs ln(φ): {pct_from_lnphi:.2f}%  CI∋ln(φ): {in_ci}")

# ============================================================
# TEST 2: High-precision at coupling = 1.0 (perfect erasure)
# ============================================================

print("\n" + "=" * 70)
print("TEST 2: Perfect Coupling (c=1.0), 50 seeds × 500k samples")
print("=" * 70)

perfect_results = []
for s in range(50):
    exp = ParameterizedErasure(
        n_env=20, n_samples=500000, seed=s,
        base_coupling=1.0, flip_decay=0.3, corr_base=0.3, corr_decay=0.2
    )
    r = exp.run_single()
    perfect_results.append(r)

p_ratios = [r['ratio'] for r in perfect_results if not np.isnan(r['ratio'])]
p_thetas = [r['theta'] for r in perfect_results]
p_apx = [r['A_plus_xi'] for r in perfect_results]

mean_r = np.mean(p_ratios)
std_r = np.std(p_ratios)
ci_lo = mean_r - 1.96 * std_r / np.sqrt(len(p_ratios))
ci_hi = mean_r + 1.96 * std_r / np.sqrt(len(p_ratios))
pct = abs(mean_r - LN_PHI) / LN_PHI * 100

print(f"\n  PERFECT COUPLING RESULTS (50 seeds × 500k):")
print(f"  Mean A/(A+ξ):  {mean_r:.8f}")
print(f"  Std:           {std_r:.8f}")
print(f"  95% CI:        [{ci_lo:.8f}, {ci_hi:.8f}]")
print(f"  ln(φ):         {LN_PHI:.8f}")
print(f"  Deviation:     {pct:.4f}%")
print(f"  ln(φ) in CI:   {ci_lo <= LN_PHI <= ci_hi}")
print(f"  Mean Θ:        {np.mean(p_thetas):.6f}")
print(f"  Mean A+ξ:      {np.mean(p_apx):.6f}")

# ============================================================
# TEST 3: Θ vs ratio relationship
# ============================================================

print("\n" + "=" * 70)
print("TEST 3: Functional relationship A/(A+ξ) = f(Θ)")
print("=" * 70)

# Collect all data points across coupling sweep
all_thetas = []
all_ratios = []

for bc in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    for s in range(30):
        exp = ParameterizedErasure(
            n_env=20, n_samples=300000, seed=s,
            base_coupling=bc, flip_decay=0.3, corr_base=0.3, corr_decay=0.2
        )
        r = exp.run_single()
        if not np.isnan(r['ratio']):
            all_thetas.append(r['theta'])
            all_ratios.append(r['ratio'])

all_thetas = np.array(all_thetas)
all_ratios = np.array(all_ratios)

# Fit: ratio = a + b*theta
slope, intercept, r_val, p_val, std_err = stats.linregress(all_thetas, all_ratios)
print(f"\n  Linear fit: ratio = {intercept:.6f} + {slope:.6f} × Θ")
print(f"  R² = {r_val**2:.6f}")
print(f"  p = {p_val:.2e}")
print(f"  Intercept (Θ→0 prediction): {intercept:.6f}")
print(f"  ln(φ) = {LN_PHI:.6f}")
print(f"  Intercept vs ln(φ): {abs(intercept - LN_PHI)/LN_PHI*100:.4f}%")

# ============================================================
# TEST 4: Decay ratio sweep at perfect coupling
# ============================================================

print("\n" + "=" * 70)
print("TEST 4: Decay ratio sweep at coupling=1.0 (20 seeds each)")
print("=" * 70)

decay_ratios = [1.0, 1.25, 1.5, 1.618, 1.75, 2.0]
decay_results = []

for dr in decay_ratios:
    # flip_decay / corr_decay = dr
    # Keep corr_decay = 0.2, vary flip_decay
    fd = 0.2 * dr
    seed_ratios = []
    seed_thetas = []
    for s in range(20):
        exp = ParameterizedErasure(
            n_env=20, n_samples=300000, seed=s,
            base_coupling=1.0, flip_decay=fd, corr_base=0.3, corr_decay=0.2
        )
        r = exp.run_single()
        if not np.isnan(r['ratio']):
            seed_ratios.append(r['ratio'])
            seed_thetas.append(r['theta'])
    
    mr = np.mean(seed_ratios)
    sr = np.std(seed_ratios)
    mt = np.mean(seed_thetas)
    pct = abs(mr - LN_PHI) / LN_PHI * 100
    ci_lo = mr - 1.96 * sr / np.sqrt(len(seed_ratios))
    ci_hi = mr + 1.96 * sr / np.sqrt(len(seed_ratios))
    in_ci = "✓" if ci_lo <= LN_PHI <= ci_hi else "✗"
    
    decay_results.append({
        'decay_ratio': dr, 'flip_decay': fd,
        'mean_ratio': mr, 'std': sr, 'ci_95': [ci_lo, ci_hi],
        'mean_theta': mt, 'pct_from_ln_phi': pct,
        'ln_phi_in_ci': ci_lo <= LN_PHI <= ci_hi
    })
    
    print(f"  ratio={dr:.3f}  flip_decay={fd:.3f}  A/(A+ξ)={mr:.6f} ± {sr:.4f}  "
          f"Θ={mt:.4f}  vs ln(φ): {pct:.2f}%  CI∋ln(φ): {in_ci}")

# ============================================================
# SAVE RESULTS
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    'experiment': 'exp_19_theta_correction',
    'timestamp': timestamp,
    'hypothesis': 'A/(A+ξ) → ln(φ) as Θ → 0 (perfect coupling limit)',
    'test_1_coupling_sweep': sweep_results,
    'test_2_perfect_coupling': {
        'n_seeds': 50,
        'n_samples': 500000,
        'mean_ratio': mean_r,
        'std_ratio': std_r,
        'ci_95': [ci_lo, ci_hi],
        'deviation_pct': pct,
        'ln_phi_in_ci': ci_lo <= LN_PHI <= ci_hi,
        'mean_theta': float(np.mean(p_thetas)),
        'mean_A_plus_xi': float(np.mean(p_apx)),
        'ln_phi': LN_PHI
    },
    'test_3_theta_regression': {
        'intercept': intercept,
        'slope': slope,
        'r_squared': r_val**2,
        'p_value': p_val,
        'intercept_vs_ln_phi_pct': abs(intercept - LN_PHI) / LN_PHI * 100,
        'n_points': len(all_thetas)
    },
    'test_4_decay_ratio_at_perfect_coupling': decay_results
}

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
outfile = os.path.join(results_dir, f'exp_19_theta_correction_{timestamp}.json')

# Convert numpy types for JSON
def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    return obj

with open(outfile, 'w') as f:
    json.dump(results, f, indent=2, default=convert)

print(f"\n\nResults saved to {outfile}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  1. Coupling sweep: Does Θ → 0 as coupling → 1.0?")
print(f"     Θ at c=0.5: {sweep_results[0]['mean_theta']:.4f}")
print(f"     Θ at c=1.0: {sweep_results[-1]['mean_theta']:.4f}")
print(f"\n  2. Perfect coupling (50 seeds):")
print(f"     A/(A+ξ) = {mean_r:.6f} vs ln(φ) = {LN_PHI:.6f} ({pct:.4f}%)")
print(f"     ln(φ) in 95% CI: {ci_lo <= LN_PHI <= ci_hi}")
print(f"\n  3. Linear regression Θ → 0 extrapolation:")
print(f"     Intercept = {intercept:.6f} vs ln(φ) = {LN_PHI:.6f}")
print(f"     Gap: {abs(intercept - LN_PHI)/LN_PHI*100:.4f}%")
