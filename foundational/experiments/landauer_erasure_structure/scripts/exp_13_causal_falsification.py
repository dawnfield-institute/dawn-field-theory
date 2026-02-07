"""
Experiment 13: Causal Lag Falsification Suite
==============================================
Dawn Field Institute

GOAL: Try to BREAK the causal lag hypothesis. If we can't, it becomes stronger.

HYPOTHESIS TO FALSIFY:
  "φ emerges from the causal requirement that correlation operates
   one step behind transfer, producing A/(A+ξ) = ln(φ)"

FALSIFICATION TESTS:

1. PARAMETER ROBUSTNESS
   - Does lag=1 still win across all decay rates, base couplings, sample sizes?
   - If lag=1 only works for specific parameters, it's fragile

2. ALTERNATIVE STRUCTURES
   - Can we construct non-causal models that also produce ln(φ)?
   - If yes, causality isn't necessary

3. SHUFFLED CONTROLS
   - If we break the causal ordering by shuffling, does the signal disappear?
   - If signal persists, it wasn't causal

4. CONTINUOUS LAG
   - Test lag = 0.5, 1.5, etc. by interpolating
   - If minimum isn't at exactly 1.0, something else is at play

5. RANDOM BASELINE
   - What's the probability of hitting ln(φ) by chance?
   - Need to know the null distribution

6. FIBONACCI-SPECIFIC TEST
   - Does lag = 1/φ or φ-1 work? (These are Fibonacci-adjacent)
   - Would reveal if it's "one step" or "Fibonacci step"
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
INV_PHI = 1 / PHI
k_B = 1.380649e-23
T = 300.0

print("=" * 70)
print("EXPERIMENT 13: Causal Lag Falsification Suite")
print("Attempting to BREAK the hypothesis that lag=1 → ln(φ)")
print("=" * 70)
print()


class CausalCascade:
    """Same as exp_12 but with more options."""
    
    def __init__(self, n_env=20, n_samples=200000, seed=42):
        self.n_env = n_env
        self.n_samples = n_samples
        self.seed = seed
        
    def run(self, causal_lag=1, base_coupling=0.8, decay_rate=0.3,
            correlation_base=None, shuffle_transfer=False):
        np.random.seed(self.seed)
        
        if correlation_base is None:
            correlation_base = base_coupling * 0.5
        
        env_energies = k_B * T * (0.5 + np.random.exponential(1.0, self.n_env))
        env_probs = 1.0 / (1.0 + np.exp(env_energies / (k_B * T)))
        
        system = np.random.randint(0, 2, self.n_samples)
        env = np.zeros((self.n_samples, self.n_env), dtype=int)
        for j in range(self.n_env):
            env[:, j] = (np.random.random(self.n_samples) < env_probs[j]).astype(int)
        
        sys_pre = system.copy()
        env_pre = env.copy()
        env_post = env.copy()
        was_one = (system == 1)
        n_active = min(8, self.n_env)
        
        # Transfer phase
        transfer_state = np.zeros((self.n_samples, n_active), dtype=int)
        for j in range(n_active):
            coupling = base_coupling * np.exp(-decay_rate * j)
            flip_mask = was_one & (np.random.random(self.n_samples) < coupling)
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
            transfer_state[flip_mask, j] = 1
        
        # Optional: shuffle transfer state to break causal connection
        if shuffle_transfer:
            for j in range(n_active):
                np.random.shuffle(transfer_state[:, j])
        
        # Correlation phase with specified lag
        lag_int = int(causal_lag)
        lag_frac = causal_lag - lag_int
        
        for j in range(max(1, lag_int), n_active):
            source_mode = j - lag_int
            if source_mode >= 0 and source_mode < n_active:
                corr_coupling = correlation_base * np.exp(-decay_rate * source_mode)
                
                # Handle fractional lag by interpolation
                if lag_frac > 0 and source_mode > 0:
                    corr_coupling2 = correlation_base * np.exp(-decay_rate * (source_mode - 1))
                    corr_coupling = (1 - lag_frac) * corr_coupling + lag_frac * corr_coupling2
                
                if lag_int == 0:
                    # Simultaneous: use same mode
                    corr_mask = was_one & (transfer_state[:, j] == 1) & \
                               (np.random.random(self.n_samples) < corr_coupling)
                else:
                    corr_mask = was_one & (transfer_state[:, source_mode] == 1) & \
                               (np.random.random(self.n_samples) < corr_coupling)
                
                env_post[corr_mask, j] = env_post[corr_mask, 0]
        
        return self._compute_metrics(sys_pre, env_pre, env_post, n_active)
    
    def _compute_metrics(self, sys_pre, env_pre, env_post, n_active):
        def entropy(data):
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
        
        def total_corr(data, n=8):
            n = min(data.shape[1], n)
            sum_H = sum(entropy(data[:, j]) for j in range(n))
            joint_H = entropy(data[:, :n])
            return max(0, sum_H - joint_H)
        
        def pairwise_MI(env, n=6):
            total = 0.0
            n = min(env.shape[1], n)
            for i in range(n):
                for jj in range(i+1, n):
                    joint = env[:, i] * 2 + env[:, jj]
                    _, counts = np.unique(joint, return_counts=True)
                    p_joint = counts / counts.sum()
                    p_i = np.array([np.mean(env[:, i] == 0), np.mean(env[:, i] == 1)])
                    p_j = np.array([np.mean(env[:, jj] == 0), np.mean(env[:, jj] == 1)])
                    H_i = -np.sum(p_i * np.log2(p_i + 1e-30))
                    H_j = -np.sum(p_j * np.log2(p_j + 1e-30))
                    H_ij = -np.sum(p_joint * np.log2(p_joint + 1e-30))
                    total += max(0, H_i + H_j - H_ij)
            return total
        
        env_hash = np.zeros(len(sys_pre), dtype=np.int64)
        for j in range(min(n_active, 8)):
            env_hash += env_post[:, j].astype(np.int64) * (2**j)
        joint = sys_pre * (2**20) + env_hash
        H_sys = entropy(sys_pre)
        H_env = entropy(env_hash)
        _, counts = np.unique(joint, return_counts=True)
        H_joint = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum() + 1e-30))
        A = max(0, H_sys + H_env - H_joint)
        
        tc_post = total_corr(env_post, n_active)
        tc_pre = total_corr(env_pre, n_active)
        pmi_post = pairwise_MI(env_post, n_active)
        pmi_pre = pairwise_MI(env_pre, n_active)
        xi = (tc_post - tc_pre) + (pmi_post - pmi_pre)  # FULL xi metric
        
        ratio = A / (A + xi) if (A + xi) > 0 else 0
        deviation = 100 * abs(ratio - LN_PHI) / LN_PHI
        
        return {'A': A, 'xi': xi, 'ratio': ratio, 'deviation_pct': deviation}


# =============================================================================
# FALSIFICATION TEST 1: Parameter Robustness
# =============================================================================

print("=" * 70)
print("FALSIFICATION 1: Parameter Robustness")
print("Does lag=1 ALWAYS beat other lags, or only for specific parameters?")
print("=" * 70)
print()

param_tests = [
    {'decay_rate': 0.1, 'base_coupling': 0.8},
    {'decay_rate': 0.2, 'base_coupling': 0.8},
    {'decay_rate': 0.4, 'base_coupling': 0.8},
    {'decay_rate': 0.5, 'base_coupling': 0.8},
    {'decay_rate': 0.3, 'base_coupling': 0.5},
    {'decay_rate': 0.3, 'base_coupling': 0.9},
    {'decay_rate': 0.3, 'base_coupling': 0.6},
]

lag1_wins = 0
lag1_losses = 0

print(f"{'Parameters':>30}  {'Lag0':>8} {'Lag1':>8} {'Lag2':>8} {'Winner':>8}")
print("-" * 75)

for params in param_tests:
    c = CausalCascade()
    devs = {}
    for lag in [0, 1, 2]:
        r = c.run(causal_lag=lag, **params)
        devs[lag] = r['deviation_pct']
    
    winner = min(devs.keys(), key=lambda k: devs[k])
    param_str = f"r={params['decay_rate']}, b={params['base_coupling']}"
    
    if winner == 1:
        lag1_wins += 1
        verdict = "LAG1 ✓"
    else:
        lag1_losses += 1
        verdict = f"LAG{winner} ✗"
    
    print(f"{param_str:>30}  {devs[0]:>7.2f}% {devs[1]:>7.2f}% {devs[2]:>7.2f}% {verdict:>8}")

print()
print(f"Lag=1 wins: {lag1_wins}/{len(param_tests)}")
if lag1_losses > 0:
    print(f"⚠ FALSIFICATION SIGNAL: Lag=1 doesn't always win!")
else:
    print("✓ Lag=1 wins across all parameter combinations tested")


# =============================================================================
# FALSIFICATION TEST 2: Shuffled Control
# =============================================================================

print()
print("=" * 70)
print("FALSIFICATION 2: Shuffled Control")
print("If we break the causal link by shuffling, does signal disappear?")
print("=" * 70)
print()

c = CausalCascade(n_samples=300000)

# Normal (causal intact)
r_normal = c.run(causal_lag=1, shuffle_transfer=False)

# Shuffled (causal broken)
r_shuffled = c.run(causal_lag=1, shuffle_transfer=True)

print(f"Normal (causal intact):   A/(A+ξ) = {r_normal['ratio']:.4f}, dev = {r_normal['deviation_pct']:.2f}%")
print(f"Shuffled (causal broken): A/(A+ξ) = {r_shuffled['ratio']:.4f}, dev = {r_shuffled['deviation_pct']:.2f}%")
print()

if r_shuffled['deviation_pct'] > 2 * r_normal['deviation_pct']:
    print("✓ Shuffling breaks the signal - causality matters")
else:
    print("⚠ FALSIFICATION SIGNAL: Shuffling doesn't break it - causality may not matter!")


# =============================================================================
# FALSIFICATION TEST 3: Continuous Lag Sweep
# =============================================================================

print()
print("=" * 70)
print("FALSIFICATION 3: Continuous Lag Sweep")
print("Is the minimum at EXACTLY lag=1.0, or somewhere else?")
print("=" * 70)
print()

c = CausalCascade(n_samples=300000)

lags = [0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
results = []

for lag in lags:
    r = c.run(causal_lag=lag)
    results.append((lag, r['ratio'], r['deviation_pct']))
    marker = " ← minimum?" if r['deviation_pct'] == min(rr[2] for rr in results) else ""
    print(f"  Lag = {lag:.1f}: A/(A+ξ) = {r['ratio']:.4f}, dev = {r['deviation_pct']:.2f}%{marker}")

best_lag = min(results, key=lambda x: x[2])
print()
print(f"Best lag: {best_lag[0]:.1f} with deviation {best_lag[2]:.2f}%")

if abs(best_lag[0] - 1.0) < 0.2:
    print("✓ Minimum is near lag=1.0 as predicted")
else:
    print(f"⚠ FALSIFICATION SIGNAL: Minimum at lag={best_lag[0]}, not 1.0!")


# =============================================================================
# FALSIFICATION TEST 4: Random Baseline
# =============================================================================

print()
print("=" * 70)
print("FALSIFICATION 4: Random Baseline")
print("What's the probability of hitting ln(φ) by chance?")
print("=" * 70)
print()

# Generate random A/(A+ξ) values and see how often they're close to ln(φ)
np.random.seed(42)
random_ratios = []

for _ in range(1000):
    # Random A and ξ in plausible range
    A = np.random.uniform(0.1, 0.8)
    xi = np.random.uniform(0.1, 0.8)
    ratio = A / (A + xi)
    random_ratios.append(ratio)

random_ratios = np.array(random_ratios)
close_to_lnphi = np.abs(random_ratios - LN_PHI) / LN_PHI < 0.01  # Within 1%
p_random = np.mean(close_to_lnphi)

print(f"Random draws close to ln(φ) (within 1%): {100*p_random:.2f}%")
print(f"Our lag=1 result: {r_normal['deviation_pct']:.2f}% deviation")
print()

# What's the p-value of getting 0.39% deviation by chance?
deviation_dist = 100 * np.abs(random_ratios - LN_PHI) / LN_PHI
p_value_chance = np.mean(deviation_dist < r_normal['deviation_pct'])

print(f"P(random ≤ {r_normal['deviation_pct']:.2f}% deviation) = {p_value_chance:.4f}")

if p_value_chance < 0.05:
    print("✓ Result is statistically significant (p < 0.05)")
else:
    print("⚠ FALSIFICATION SIGNAL: Result could be chance!")


# =============================================================================
# FALSIFICATION TEST 5: Alternative Model
# =============================================================================

print()
print("=" * 70)
print("FALSIFICATION 5: Alternative Non-Causal Model")
print("Can we get ln(φ) without any causal ordering?")
print("=" * 70)
print()

# Model where correlation depends on TOTAL transfer, not causal order
class NonCausalModel:
    def __init__(self, n_samples=300000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
    
    def run(self, transfer_weight=0.8, corr_weight=0.3):
        np.random.seed(self.seed)
        
        n_env = 8
        env_pre = np.random.randint(0, 2, (self.n_samples, n_env))
        system = np.random.randint(0, 2, self.n_samples)
        sys_pre = system.copy()
        
        env_post = env_pre.copy()
        was_one = (system == 1)
        
        # All modes receive independent transfer (no cascade)
        for j in range(n_env):
            flip_mask = was_one & (np.random.random(self.n_samples) < transfer_weight * (0.9**j))
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
        
        # Correlation based on TOTAL transfer, not causal order
        total_flipped = np.sum(env_post != env_pre, axis=1)
        for j in range(1, n_env):
            corr_prob = corr_weight * (total_flipped / n_env)
            corr_mask = was_one & (np.random.random(self.n_samples) < corr_prob)
            env_post[corr_mask, j] = env_post[corr_mask, 0]
        
        # Compute metrics
        def entropy(d):
            if d.ndim == 1:
                _, c = np.unique(d, return_counts=True)
                p = c/c.sum()
                return -np.sum(p * np.log2(p + 1e-30))
            h = sum(d[:,j].astype(np.int64)*(2**j) for j in range(min(d.shape[1], 8)))
            _, c = np.unique(h, return_counts=True)
            p = c/c.sum()
            return -np.sum(p * np.log2(p + 1e-30))
        
        def tc(d):
            n = min(d.shape[1], 8)
            return max(0, sum(entropy(d[:,j]) for j in range(n)) - entropy(d[:,:n]))
        
        def pairwise_MI(env, n=6):
            total = 0.0
            n = min(env.shape[1], n)
            for i in range(n):
                for jj in range(i+1, n):
                    joint = env[:, i] * 2 + env[:, jj]
                    _, counts = np.unique(joint, return_counts=True)
                    p_joint = counts / counts.sum()
                    p_i = np.array([np.mean(env[:, i] == 0), np.mean(env[:, i] == 1)])
                    p_j = np.array([np.mean(env[:, jj] == 0), np.mean(env[:, jj] == 1)])
                    H_i = -np.sum(p_i * np.log2(p_i + 1e-30))
                    H_j = -np.sum(p_j * np.log2(p_j + 1e-30))
                    H_ij = -np.sum(p_joint * np.log2(p_joint + 1e-30))
                    total += max(0, H_i + H_j - H_ij)
            return total
        
        env_hash = sum(env_post[:,j].astype(np.int64)*(2**j) for j in range(n_env))
        joint = sys_pre * (2**20) + env_hash
        H_sys = entropy(sys_pre)
        H_env = entropy(env_hash)
        _, c = np.unique(joint, return_counts=True)
        H_joint = -np.sum((c/c.sum()) * np.log2(c/c.sum() + 1e-30))
        A = max(0, H_sys + H_env - H_joint)
        
        # FULL xi: TC + pairwise MI (consistent with CausalCascade)
        tc_post = tc(env_post)
        tc_pre = tc(env_pre)
        pmi_post = pairwise_MI(env_post, n_env)
        pmi_pre = pairwise_MI(env_pre, n_env)
        xi = (tc_post - tc_pre) + (pmi_post - pmi_pre)
        
        ratio = A / (A + xi) if (A + xi) > 0 else 0
        return ratio, 100 * abs(ratio - LN_PHI) / LN_PHI

# Grid search to see if ANY non-causal parameters hit ln(φ)
best_noncausal = (None, None, 100)

print("Searching non-causal parameter space...")
for tw in [0.3, 0.5, 0.7, 0.9]:
    for cw in [0.1, 0.3, 0.5, 0.7]:
        m = NonCausalModel()
        ratio, dev = m.run(transfer_weight=tw, corr_weight=cw)
        if dev < best_noncausal[2]:
            best_noncausal = (tw, cw, dev)

print(f"Best non-causal result: t={best_noncausal[0]}, c={best_noncausal[1]}")
print(f"  Deviation from ln(φ): {best_noncausal[2]:.2f}%")
print(f"  Causal lag=1 result:  {r_normal['deviation_pct']:.2f}%")
print()

if best_noncausal[2] < r_normal['deviation_pct']:
    print("⚠ FALSIFICATION: Non-causal model can match ln(φ) better!")
else:
    print("✓ Causal model outperforms non-causal alternatives")


# =============================================================================
# FALSIFICATION TEST 6: Multi-Seed Consistency
# =============================================================================

print()
print("=" * 70)
print("FALSIFICATION 6: Multi-Seed Consistency (50 seeds)")
print("Is lag=1 consistently optimal, or does it fluctuate?")
print("=" * 70)
print()

lag_wins = {0: 0, 1: 0, 2: 0}
lag_devs = {0: [], 1: [], 2: []}

for seed in range(50):
    c = CausalCascade(seed=seed)
    best_this_seed = (None, 100)
    for lag in [0, 1, 2]:
        r = c.run(causal_lag=lag)
        lag_devs[lag].append(r['deviation_pct'])
        if r['deviation_pct'] < best_this_seed[1]:
            best_this_seed = (lag, r['deviation_pct'])
    lag_wins[best_this_seed[0]] += 1

print(f"Wins by lag across 50 seeds:")
print(f"  Lag 0: {lag_wins[0]} wins")
print(f"  Lag 1: {lag_wins[1]} wins")
print(f"  Lag 2: {lag_wins[2]} wins")
print()
print(f"Mean deviation from ln(φ):")
print(f"  Lag 0: {np.mean(lag_devs[0]):.2f}% ± {np.std(lag_devs[0]):.2f}%")
print(f"  Lag 1: {np.mean(lag_devs[1]):.2f}% ± {np.std(lag_devs[1]):.2f}%")
print(f"  Lag 2: {np.mean(lag_devs[2]):.2f}% ± {np.std(lag_devs[2]):.2f}%")
print()

# Statistical test
t, p = stats.ttest_ind(lag_devs[1], lag_devs[0])
print(f"Lag1 vs Lag0: t={t:.2f}, p={p:.2e}")
t, p = stats.ttest_ind(lag_devs[1], lag_devs[2])
print(f"Lag1 vs Lag2: t={t:.2f}, p={p:.2e}")

if lag_wins[1] >= 40:
    print()
    print("✓ Lag=1 wins overwhelmingly (≥80%)")
elif lag_wins[1] >= 25:
    print()
    print("~ Lag=1 wins majority but not dominant")
else:
    print()
    print("⚠ FALSIFICATION: Lag=1 doesn't consistently win!")


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 70)
print("FALSIFICATION SUMMARY")
print("=" * 70)
print()

# Collect all results
tests_passed = 0
tests_total = 6

# Test 1
if lag1_losses == 0:
    tests_passed += 1
    print("1. Parameter Robustness:    ✓ PASSED")
else:
    print("1. Parameter Robustness:    ✗ FAILED")

# Test 2
if r_shuffled['deviation_pct'] > 2 * r_normal['deviation_pct']:
    tests_passed += 1
    print("2. Shuffled Control:        ✓ PASSED")
else:
    print("2. Shuffled Control:        ✗ FAILED")

# Test 3
if abs(best_lag[0] - 1.0) < 0.2:
    tests_passed += 1
    print("3. Continuous Lag Sweep:    ✓ PASSED")
else:
    print("3. Continuous Lag Sweep:    ✗ FAILED")

# Test 4
if p_value_chance < 0.05:
    tests_passed += 1
    print("4. Random Baseline:         ✓ PASSED")
else:
    print("4. Random Baseline:         ✗ FAILED")

# Test 5
if best_noncausal[2] >= r_normal['deviation_pct']:
    tests_passed += 1
    print("5. Alternative Model:       ✓ PASSED")
else:
    print("5. Alternative Model:       ✗ FAILED")

# Test 6
if lag_wins[1] >= 40:
    tests_passed += 1
    print("6. Multi-Seed Consistency:  ✓ PASSED")
else:
    print("6. Multi-Seed Consistency:  ✗ FAILED")

print()
print(f"RESULT: {tests_passed}/{tests_total} falsification tests passed")
print()

if tests_passed == tests_total:
    print("=" * 70)
    print("HYPOTHESIS SURVIVES ALL FALSIFICATION ATTEMPTS")
    print("=" * 70)
    print()
    print("The causal lag hypothesis is ROBUST:")
    print("  - φ emerges specifically from lag=1 (one Fibonacci step)")
    print("  - Breaking causality breaks the signal")
    print("  - No non-causal model matches the result")
    print("  - Result is statistically significant")
    print("  - Consistent across parameters and seeds")
    print()
    print("CONCLUSION: Evidence strongly supports that φ is REQUIRED by causality,")
    print("not fitted by parameter search.")
elif tests_passed >= 4:
    print("Hypothesis is MOSTLY robust with some caveats.")
    print("Further investigation recommended on failed tests.")
else:
    print("⚠ HYPOTHESIS IS WEAKENED OR FALSIFIED")
    print("The causal ordering may not be the correct explanation.")
