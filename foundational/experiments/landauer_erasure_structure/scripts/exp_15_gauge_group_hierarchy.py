"""
Experiment 15: Gauge Group ξ Hierarchy
======================================
Dawn Field Institute

HYPOTHESIS (from gravity_from_maxwell_pac):
The correlational structure ξ created during information exchange should
scale with the number of modes in the gauge group:

    ξ(SU(3)) > ξ(SU(2)) > ξ(U(1))

Because:
- U(1) has 1 generator (photon) → single-mode coupling → minimal ξ
- SU(2) has 3 generators (W+, W-, Z) → 3-mode coupling → moderate ξ
- SU(3) has 8 generators (gluons) → 8-mode coupling → maximal ξ

METHOD:
Model each gauge group as a coupling topology in the Landauer framework.
Information erasure through N coupled modes should produce ξ proportional
to the correlation structure those modes can support.

PREDICTION:
If correct, this explains WHY α_s > α_W > α_EM:
- Strong coupling is strong because SU(3) creates the most structure
- EM coupling is weak because U(1) creates the least structure
- Coupling constants encode accumulated ξ at characteristic energy scales
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
k_B = 1.380649e-23
T = 300.0

print("=" * 70)
print("EXPERIMENT 15: Gauge Group ξ Hierarchy")
print("Testing: ξ(SU(3)) > ξ(SU(2)) > ξ(U(1))")
print("=" * 70)
print()


def entropy(data):
    """Shannon entropy in bits"""
    if data.ndim == 1:
        _, counts = np.unique(data, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-30))
    else:
        n = min(data.shape[1], 12)
        h = np.zeros(data.shape[0], dtype=np.int64)
        for j in range(n):
            h += data[:, j].astype(np.int64) * (2**j)
        _, counts = np.unique(h, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-30))

def total_corr(data, n=None):
    """Total correlation (multi-information)"""
    if n is None:
        n = data.shape[1]
    n = min(data.shape[1], n)
    sum_H = sum(entropy(data[:, j]) for j in range(n))
    joint_H = entropy(data[:, :n])
    return max(0, sum_H - joint_H)

def pairwise_MI(env, n=None):
    """Sum of pairwise mutual information"""
    if n is None:
        n = env.shape[1]
    n = min(env.shape[1], n)
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
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


class GaugeErasure:
    """
    Model information erasure through gauge group topologies.
    
    Each gauge group is modeled as a coupling topology:
    - U(1): Single mode (no inter-mode correlations possible)
    - SU(2): 3 modes with symmetric coupling (W+, W-, Z structure)
    - SU(3): 8 modes with color-octet structure
    
    Key differences from generic cascade:
    - Gauge symmetry constrains which modes can couple
    - Conservation laws restrict correlation patterns
    """
    
    def __init__(self, n_samples=500000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        
    def u1_erasure(self):
        """
        U(1) gauge group: Single photon mode.
        
        Information transfers to one mode only.
        No inter-mode correlations possible (only 1 mode).
        Should produce minimal ξ.
        """
        np.random.seed(self.seed)
        n_modes = 1
        
        # Initialize
        env_prob = 0.5  # Thermal
        system = np.random.randint(0, 2, self.n_samples)
        env = (np.random.random((self.n_samples, n_modes)) < env_prob).astype(int)
        
        env_pre = env.copy()
        env_post = env.copy()
        was_one = (system == 1)
        
        # Transfer: all information to single mode
        coupling = 0.7
        flip_mask = was_one & (np.random.random(self.n_samples) < coupling)
        env_post[flip_mask, 0] = 1 - env_post[flip_mask, 0]
        
        # Measure A (system-environment MI)
        env_hash = env_post[:, 0]
        joint = system * 2 + env_hash
        H_sys = entropy(system)
        H_env = entropy(env_hash)
        _, counts = np.unique(joint, return_counts=True)
        H_joint = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum() + 1e-30))
        A = max(0, H_sys + H_env - H_joint)
        
        # ξ: With only 1 mode, no inter-mode correlation is possible
        xi = 0.0
        
        return {'A': A, 'xi': xi, 'n_modes': n_modes, 'group': 'U(1)'}
    
    def su2_erasure(self):
        """
        SU(2) gauge group: 3 modes (W+, W-, Z).
        
        Symmetric coupling structure.
        W+ and W- are charge conjugates → anticorrelated.
        Z is neutral → correlates with both.
        """
        np.random.seed(self.seed)
        n_modes = 3
        
        # Initialize
        env_probs = np.array([0.5, 0.5, 0.5])  # Thermal equilibrium
        system = np.random.randint(0, 2, self.n_samples)
        env = np.zeros((self.n_samples, n_modes), dtype=int)
        for j in range(n_modes):
            env[:, j] = (np.random.random(self.n_samples) < env_probs[j]).astype(int)
        
        env_pre = env.copy()
        env_post = env.copy()
        was_one = (system == 1)
        
        tc_pre = total_corr(env_pre, n_modes)
        pmi_pre = pairwise_MI(env_pre, n_modes)
        
        # Transfer phase: distribute to all 3 modes with SU(2) symmetry
        # W+ and W- get equal coupling, Z slightly different
        base_coupling = 0.7
        couplings = [base_coupling, base_coupling, base_coupling * 0.9]  # Z slightly weaker
        
        for j in range(n_modes):
            flip_mask = was_one & (np.random.random(self.n_samples) < couplings[j])
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
        
        # Correlation phase: SU(2) structure
        # W+ and W- anticorrelate (charge conjugation)
        # Z correlates with both
        corr_coupling = 0.3
        
        # W+ → W- anticorrelation
        corr_mask = was_one & (env_post[:, 0] == 1) & (np.random.random(self.n_samples) < corr_coupling)
        env_post[corr_mask, 1] = 1 - env_post[corr_mask, 1]  # Flip W- when W+ is 1
        
        # Z correlates with W+ ⊕ W-
        corr_mask = was_one & (np.random.random(self.n_samples) < corr_coupling)
        env_post[corr_mask, 2] = env_post[corr_mask, 0] ^ env_post[corr_mask, 1]  # XOR
        
        # Measure A
        env_hash = sum(env_post[:, j].astype(np.int64) * (2**j) for j in range(n_modes))
        joint = system * (2**20) + env_hash
        H_sys = entropy(system)
        H_env = entropy(env_hash)
        _, counts = np.unique(joint, return_counts=True)
        H_joint = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum() + 1e-30))
        A = max(0, H_sys + H_env - H_joint)
        
        # ξ: New correlational structure
        tc_post = total_corr(env_post, n_modes)
        pmi_post = pairwise_MI(env_post, n_modes)
        xi = (tc_post - tc_pre) + (pmi_post - pmi_pre)
        
        return {'A': A, 'xi': xi, 'n_modes': n_modes, 'group': 'SU(2)'}
    
    def su3_erasure(self):
        """
        SU(3) gauge group: 8 gluon modes.
        
        Color-octet structure with complex inter-mode correlations.
        Gluons carry color-anticolor → rich correlation patterns.
        Should produce maximal ξ.
        """
        np.random.seed(self.seed)
        n_modes = 8
        
        # Initialize
        env_probs = np.full(n_modes, 0.5)
        system = np.random.randint(0, 2, self.n_samples)
        env = np.zeros((self.n_samples, n_modes), dtype=int)
        for j in range(n_modes):
            env[:, j] = (np.random.random(self.n_samples) < env_probs[j]).astype(int)
        
        env_pre = env.copy()
        env_post = env.copy()
        was_one = (system == 1)
        
        tc_pre = total_corr(env_pre, n_modes)
        pmi_pre = pairwise_MI(env_pre, 6)  # Limit for computational tractability
        
        # Transfer phase: cascade-like with SU(3) structure
        # Coupling decays but all 8 modes receive information
        base_coupling = 0.7
        decay = 0.15
        
        for j in range(n_modes):
            coupling = base_coupling * np.exp(-decay * j)
            flip_mask = was_one & (np.random.random(self.n_samples) < coupling)
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
        
        # Correlation phase: SU(3) color structure
        # Group modes into triplets (R, G, B) × 2 + diagonal modes
        corr_coupling = 0.3
        
        # Color triplet correlations (modes 0,1,2 and 3,4,5)
        for base in [0, 3]:
            # Cyclic color correlation: R→G→B→R
            for offset in range(3):
                source = base + offset
                target = base + (offset + 1) % 3
                corr_mask = was_one & (env_post[:, source] == 1) & \
                           (np.random.random(self.n_samples) < corr_coupling)
                env_post[corr_mask, target] = env_post[corr_mask, source]
        
        # Diagonal modes (6, 7) correlate with triplet sums
        corr_mask = was_one & (np.random.random(self.n_samples) < corr_coupling)
        env_post[corr_mask, 6] = (env_post[corr_mask, 0] + env_post[corr_mask, 1] + 
                                  env_post[corr_mask, 2]) % 2
        corr_mask = was_one & (np.random.random(self.n_samples) < corr_coupling)
        env_post[corr_mask, 7] = (env_post[corr_mask, 3] + env_post[corr_mask, 4] + 
                                  env_post[corr_mask, 5]) % 2
        
        # Cross-triplet correlations (gluon exchange)
        for i in range(3):
            corr_mask = was_one & (env_post[:, i] == 1) & \
                       (np.random.random(self.n_samples) < corr_coupling * 0.5)
            env_post[corr_mask, i + 3] = env_post[corr_mask, i]
        
        # Measure A
        env_hash = sum(env_post[:, j].astype(np.int64) * (2**j) for j in range(min(n_modes, 10)))
        joint = system * (2**20) + env_hash
        H_sys = entropy(system)
        H_env = entropy(env_hash)
        _, counts = np.unique(joint, return_counts=True)
        H_joint = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum() + 1e-30))
        A = max(0, H_sys + H_env - H_joint)
        
        # ξ: New correlational structure
        tc_post = total_corr(env_post, n_modes)
        pmi_post = pairwise_MI(env_post, 6)
        xi = (tc_post - tc_pre) + (pmi_post - pmi_pre)
        
        return {'A': A, 'xi': xi, 'n_modes': n_modes, 'group': 'SU(3)'}


# =============================================================================
# RUN THE HIERARCHY TEST
# =============================================================================

print("Testing gauge group information erasure...")
print()

gauge = GaugeErasure()

results = {
    'U(1)': gauge.u1_erasure(),
    'SU(2)': gauge.su2_erasure(),
    'SU(3)': gauge.su3_erasure(),
}

print("=" * 70)
print("RESULTS: Gauge Group ξ Hierarchy")
print("=" * 70)
print()
print(f"{'Group':>8} | {'Modes':>6} | {'A':>10} | {'ξ':>10} | {'A/(A+ξ)':>10}")
print("-" * 60)

for group in ['U(1)', 'SU(2)', 'SU(3)']:
    r = results[group]
    ratio = r['A'] / (r['A'] + r['xi']) if r['A'] + r['xi'] > 0 else float('inf')
    print(f"{group:>8} | {r['n_modes']:>6} | {r['A']:>10.4f} | {r['xi']:>10.4f} | {ratio:>10.4f}")

print()

# Check ordering
xi_u1 = results['U(1)']['xi']
xi_su2 = results['SU(2)']['xi']
xi_su3 = results['SU(3)']['xi']

print("HYPOTHESIS TEST: ξ(SU(3)) > ξ(SU(2)) > ξ(U(1))")
print()
print(f"  ξ(U(1))  = {xi_u1:.4f}")
print(f"  ξ(SU(2)) = {xi_su2:.4f}")
print(f"  ξ(SU(3)) = {xi_su3:.4f}")
print()

if xi_su3 > xi_su2 > xi_u1:
    print("✓ HIERARCHY CONFIRMED: SU(3) > SU(2) > U(1)")
    print()
    print("  This supports the claim that:")
    print("  - Gauge coupling strength correlates with ξ production")
    print("  - α_s > α_W > α_EM because SU(3) creates more structure")
else:
    print("✗ Hierarchy not confirmed in expected order")
    print(f"  Observed: {xi_su3:.4f}, {xi_su2:.4f}, {xi_u1:.4f}")


# =============================================================================
# MULTI-SEED VALIDATION
# =============================================================================

print()
print("=" * 70)
print("MULTI-SEED VALIDATION")
print("=" * 70)
print()

xi_values = {'U(1)': [], 'SU(2)': [], 'SU(3)': []}

for seed in range(30):
    g = GaugeErasure(seed=seed)
    xi_values['U(1)'].append(g.u1_erasure()['xi'])
    xi_values['SU(2)'].append(g.su2_erasure()['xi'])
    xi_values['SU(3)'].append(g.su3_erasure()['xi'])

print(f"{'Group':>8} | {'Mean ξ':>10} | {'Std ξ':>10} | {'Min':>8} | {'Max':>8}")
print("-" * 55)

for group in ['U(1)', 'SU(2)', 'SU(3)']:
    vals = xi_values[group]
    print(f"{group:>8} | {np.mean(vals):>10.4f} | {np.std(vals):>10.4f} | {np.min(vals):>8.4f} | {np.max(vals):>8.4f}")

print()

# Statistical test: is the ordering consistent?
su3_gt_su2 = sum(1 for i in range(30) if xi_values['SU(3)'][i] > xi_values['SU(2)'][i])
su2_gt_u1 = sum(1 for i in range(30) if xi_values['SU(2)'][i] > xi_values['U(1)'][i])
full_order = sum(1 for i in range(30) if xi_values['SU(3)'][i] > xi_values['SU(2)'][i] > xi_values['U(1)'][i])

print(f"Ordering consistency across 30 seeds:")
print(f"  SU(3) > SU(2): {su3_gt_su2}/30 = {100*su3_gt_su2/30:.1f}%")
print(f"  SU(2) > U(1):  {su2_gt_u1}/30 = {100*su2_gt_u1/30:.1f}%")
print(f"  Full order:    {full_order}/30 = {100*full_order/30:.1f}%")
print()

# Mann-Whitney U tests
_, p_su3_su2 = stats.mannwhitneyu(xi_values['SU(3)'], xi_values['SU(2)'], alternative='greater')
_, p_su2_u1 = stats.mannwhitneyu(xi_values['SU(2)'], xi_values['U(1)'], alternative='greater')

print(f"Mann-Whitney U tests (one-tailed):")
print(f"  SU(3) > SU(2): p = {p_su3_su2:.2e}")
print(f"  SU(2) > U(1):  p = {p_su2_u1:.2e}")
print()


# =============================================================================
# CONNECTION TO COUPLING CONSTANTS
# =============================================================================

print("=" * 70)
print("CONNECTION TO MEASURED COUPLING CONSTANTS")
print("=" * 70)
print()

# Measured coupling constants (at characteristic scales)
alpha_em = 1/137.036  # Electric (at low energy)
alpha_w = 1/30        # Weak (approximate at M_W)
alpha_s = 0.118       # Strong (at M_Z)

print("Measured coupling constants:")
print(f"  α_EM = {alpha_em:.6f}  (1/137)")
print(f"  α_W  ≈ {alpha_w:.6f}  (1/30)")
print(f"  α_s  = {alpha_s:.6f}  (0.118)")
print()

# Normalize ξ values relative to U(1)
mean_xi = {g: np.mean(xi_values[g]) for g in ['U(1)', 'SU(2)', 'SU(3)']}

# For U(1), ξ is 0, so use A instead as baseline
mean_A_u1 = np.mean([GaugeErasure(seed=s).u1_erasure()['A'] for s in range(10)])

print("ξ structure:")
print(f"  U(1):  ξ = {mean_xi['U(1)']:.4f} (baseline)")
print(f"  SU(2): ξ = {mean_xi['SU(2)']:.4f} (ratio to SU(3): {mean_xi['SU(2)']/mean_xi['SU(3)']:.3f})")
print(f"  SU(3): ξ = {mean_xi['SU(3)']:.4f} (maximum)")
print()

# Compare ξ ratios to coupling ratios
if mean_xi['SU(3)'] > 0 and mean_xi['SU(2)'] > 0:
    xi_ratio_32 = mean_xi['SU(3)'] / mean_xi['SU(2)']
    alpha_ratio_32 = alpha_s / alpha_w
    
    print("Ratio comparison (SU(3)/SU(2)):")
    print(f"  ξ ratio:     {xi_ratio_32:.3f}")
    print(f"  α ratio:     {alpha_ratio_32:.3f}")
    print(f"  Match:       {100*(1 - abs(xi_ratio_32 - alpha_ratio_32)/alpha_ratio_32):.1f}%")


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

mean_u1 = np.mean(xi_values['U(1)'])
mean_su2 = np.mean(xi_values['SU(2)'])
mean_su3 = np.mean(xi_values['SU(3)'])

if mean_su3 > mean_su2 > mean_u1:
    print("✓ HYPOTHESIS CONFIRMED")
    print()
    print("  The gauge group ξ hierarchy holds:")
    print(f"    ξ(SU(3)) = {mean_su3:.4f}")
    print(f"    ξ(SU(2)) = {mean_su2:.4f}")
    print(f"    ξ(U(1))  = {mean_u1:.4f}")
    print()
    print("  This supports the theoretical claim:")
    print("  - More gauge modes → more correlational structure")
    print("  - Coupling strength encodes accumulated ξ")
    print("  - α_s > α_W > α_EM reflects SU(3) > SU(2) > U(1)")
    print()
    print("  Statistical significance:")
    print(f"    SU(3) > SU(2): p = {p_su3_su2:.2e}")
    print(f"    SU(2) > U(1):  p = {p_su2_u1:.2e}")
else:
    print("✗ Hierarchy not confirmed")
    print("  Further investigation needed")
