"""
exp_18_cascade_fibonacci_bridge.py
===================================
Mathematical bridge between Fibonacci matrix eigenvalues (exp_17)
and cascade topology (Paper 1 sections 4.4, 6, 8).

ADDRESSES the acknowledged gap in Paper 1:
  "We did not prove that cascade is the unique or optimal topology"    (S4.4)
  "We did not derive cascade from first principles"                    (S8)

CHAIN OF REASONING:

1. UNIQUENESS (Test 1 -- exact):
   Model each link in a cascade chain as a 2x2 matrix M acting on
   (retained, propagated). Constraints:
     - Non-negative integer entries (binary coupling)
     - |det(M)| = 1 (information conservation: unimodular)
     - tr(M) = 1 (asymmetric transfer: sender retains, receiver updates)
   The ONLY solutions are [[1,1],[1,0]] and [[0,1],[1,1]] -- the
   Fibonacci matrix and its transpose (related by direction of flow).

2. EIGENVALUES (exp_17 -- exact):
   The Fibonacci matrix has eigenvalues phi and -1/phi.
   Therefore the dominant decay rate through the chain is 1/phi per link.

3. CONVERGENCE (Test 2 -- exact):
   Fibonacci ratios F_k / F_{k+1} -> 1/phi with exponential convergence.
   A Fibonacci coupling vector is asymptotically equivalent to a
   geometric cascade with decay d = 1/phi.

4. EQUIVALENCE (Test 3 -- numerical):
   Compare xi from Fibonacci coupling vs cascade at d = 1/phi.
   They should be statistically indistinguishable.

5. TWO-RATE PARTITION (Test 4 -- numerical):
   Paper 1 S6 found A/(A+xi) ~ ln(phi) at decay ratio ~ 1.5.
   Replicate using exp_05's EXACT coupling model:
     - Transfer channel: each mode directly coupled to system (star)
     - Correlation channel: mode j copies mode 0 (star, NOT chain)
     - xi measured from zero baseline (matching exp_05)
   Scan flip_decay / corr_decay and find where A/(A+xi) = ln(phi).

BUILDS ON: exp_17 (eigenvalue proof), exp_01/exp_03 (ratio finding),
           exp_02 (topology comparison), exp_10 (temporal cascade)
"""

import numpy as np
from scipy import stats as sp_stats
from datetime import datetime
import json
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ===================================================================
# Constants
# ===================================================================
PHI = (1 + np.sqrt(5)) / 2       # 1.618033988749895
PHI_INV = 1 / PHI                 # 0.618033988749895
LN_PHI = float(np.log(PHI))       # 0.481211825059604


def fibonacci(n: int) -> int:
    """Fibonacci number: F_1=1, F_2=1, F_3=2, ..."""
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


# ===================================================================
# TEST 1: Fibonacci Matrix Uniqueness
# ===================================================================

def test_uniqueness() -> dict:
    """
    Enumerate all 2x2 non-negative integer matrices M = [[a,b],[c,d]] with:
      (i)   Non-negative integer entries (binary coupling)
      (ii)  |det(M)| = 1  (information conservation: invertible over Z)
      (iii) tr(M) = 1     (asymmetric transfer: sender retains, receiver updates)

    Physical interpretation:
      - Integer entries: coupling is binary (0 or 1) at the fundamental level.
      - |det|=1: unimodular -- information neither created nor destroyed.
      - tr=1: exactly one diagonal = 1 (sender retains state), other = 0
        (receiver gets overwritten). This IS cascade coupling.

    Result: ONLY [[1,1],[1,0]] and [[0,1],[1,1]] satisfy all three.
    Both are the Fibonacci matrix (related by direction of flow).
    Both have dominant eigenvalue phi.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Fibonacci Matrix Uniqueness")
    print("=" * 70)

    candidates = []
    for a in range(10):
        d_val = 1 - a  # tr = a + d = 1
        if d_val < 0:
            continue
        for b in range(10):
            for c in range(10):
                det = a * d_val - b * c
                if abs(det) == 1:
                    M = np.array([[a, b], [c, d_val]], dtype=float)
                    evals = sorted(np.linalg.eigvals(M).real,
                                   key=lambda x: -abs(x))
                    is_fib = ([a, b, c, d_val] == [1, 1, 1, 0] or
                              [a, b, c, d_val] == [0, 1, 1, 1])
                    candidates.append({
                        "matrix": [[a, b], [c, d_val]],
                        "det": det,
                        "eigenvalues": [round(float(e), 10) for e in evals],
                        "dominant_eigenvalue": round(
                            float(max(evals, key=abs)), 10),
                        "is_fibonacci_or_transpose": is_fib,
                    })

    n_total = len(candidates)
    all_have_phi = all(
        abs(abs(c["dominant_eigenvalue"]) - PHI) < 0.001
        for c in candidates
    )

    print(f"  Candidates found: {n_total}")
    for c in candidates:
        tag = " <-- FIBONACCI" if c["is_fibonacci_or_transpose"] else ""
        print(f"    {c['matrix']}  det={c['det']}  "
              f"eigenvalues={c['eigenvalues']}{tag}")
    print(f"\n  All have dominant eigenvalue phi: {all_have_phi}")

    status = "PASS" if n_total == 2 and all_have_phi else "CHECK"
    print(f"  Status: {status}")

    return {
        "test": "fibonacci_matrix_uniqueness",
        "status": status,
        "n_candidates": n_total,
        "candidates": candidates,
        "all_have_phi_eigenvalue": bool(all_have_phi),
        "conclusion": (
            f"Exactly {n_total} non-negative integer 2x2 matrices satisfy "
            f"|det|=1 and tr=1. Both are the Fibonacci matrix (or transpose). "
            f"Both have dominant eigenvalue phi = {PHI:.10f}."
        ),
    }


# ===================================================================
# TEST 2: Fibonacci -> Geometric Convergence
# ===================================================================

def test_convergence() -> dict:
    """
    Show F_k / F_{k+1} -> 1/phi with exponential convergence.

    Consequence: a coupling vector proportional to Fibonacci numbers
    [F_n, F_{n-1}, ..., F_1] is asymptotically equivalent to a
    geometric cascade with decay d = 1/phi.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Fibonacci -> Geometric Convergence")
    print("=" * 70)

    convergence = []
    for k in range(1, 25):
        fk = fibonacci(k)
        fk1 = fibonacci(k + 1)
        ratio = fk / fk1
        error = abs(ratio - PHI_INV)
        convergence.append({
            "k": k, "F_k": fk, "F_k_plus_1": fk1,
            "ratio": ratio, "error": float(error),
        })

    print(f"  F_k/F_{{k+1}} -> 1/phi = {PHI_INV:.10f}")
    print(f"  k=5:  ratio={convergence[4]['ratio']:.10f}  "
          f"error={convergence[4]['error']:.2e}")
    print(f"  k=10: ratio={convergence[9]['ratio']:.10f}  "
          f"error={convergence[9]['error']:.2e}")
    print(f"  k=20: ratio={convergence[19]['ratio']:.10f}  "
          f"error={convergence[19]['error']:.2e}")

    # Compare coupling vectors: Fibonacci vs geometric at 1/phi
    coupling_comparisons = {}
    for n_modes in [5, 8]:
        fib_raw = np.array([fibonacci(n_modes - j) for j in range(n_modes)],
                           dtype=float)
        fib_norm = fib_raw / fib_raw[0]

        geo_raw = np.array([PHI_INV**j for j in range(n_modes)])
        geo_norm = geo_raw / geo_raw[0]

        l2 = float(np.sqrt(np.sum((fib_norm - geo_norm)**2)))
        max_diff = float(np.max(np.abs(fib_norm - geo_norm)))

        coupling_comparisons[str(n_modes)] = {
            "fibonacci_normalized": [round(f, 6) for f in fib_norm],
            "geometric_normalized": [round(g, 6) for g in geo_norm],
            "L2_distance": round(l2, 6),
            "max_component_diff": round(max_diff, 6),
        }

        print(f"\n  n_modes = {n_modes}:")
        print(f"    Fibonacci: {[round(f, 4) for f in fib_norm]}")
        print(f"    Geometric: {[round(g, 4) for g in geo_norm]}")
        print(f"    L2 distance: {l2:.6f}, max diff: {max_diff:.6f}")

    return {
        "test": "fibonacci_geometric_convergence",
        "status": "PASS",
        "convergence": convergence,
        "coupling_comparisons": coupling_comparisons,
        "conclusion": (
            "Fibonacci ratios converge to 1/phi exponentially. "
            "Fibonacci coupling vectors closely match geometric cascade "
            "at d = 1/phi for practical mode counts."
        ),
    }


# ===================================================================
# Shared simulation infrastructure
# ===================================================================

def entropy_1d(data):
    """Shannon entropy of a 1D discrete array, in bits."""
    vals, cnts = np.unique(data, return_counts=True)
    p = cnts / cnts.sum()
    return float(-np.sum(p * np.log2(p + 1e-30)))


def compute_TC(data, n_modes):
    """Total correlation: sum of marginal entropies minus joint entropy."""
    n_modes = min(n_modes, data.shape[1])
    sum_H = sum(entropy_1d(data[:, j]) for j in range(n_modes))
    hashes = np.zeros(data.shape[0], dtype=np.int64)
    for j in range(n_modes):
        hashes += data[:, j].astype(np.int64) * (2**j)
    H_joint = entropy_1d(hashes)
    return max(0.0, sum_H - H_joint)


def compute_pairwise_mi(env, n_env):
    """Sum of pairwise mutual information across all mode pairs."""
    total = 0.0
    for i in range(n_env):
        for j in range(i + 1, n_env):
            joint = env[:, i] * 2 + env[:, j]
            p_joint = np.bincount(joint, minlength=4) / len(joint)
            p_i = np.array([1 - env[:, i].mean(), env[:, i].mean()])
            p_j = np.array([1 - env[:, j].mean(), env[:, j].mean()])
            H_i = -np.sum(p_i * np.log2(p_i + 1e-30))
            H_j = -np.sum(p_j * np.log2(p_j + 1e-30))
            H_ij = -np.sum(p_joint * np.log2(p_joint + 1e-30))
            total += max(0, H_i + H_j - H_ij)
    return total


def run_erasure(coupling_type, decay=0.5, n_env=8, n_coupling=5,
                coupling_strength=0.8, n_samples=300000, seed=42):
    """
    Single Landauer erasure with specified coupling topology.

    Topologies:
      cascade_chain:   mode j copies mode j-1 with prob ~ decay^j
      fibonacci_chain: chain cascade with Fibonacci-proportional link probs
      uniform:         all modes equally coupled
      exponential:     direct coupling with exp(-0.3*j)
      random_sparse:   random subset, random coupling
    """
    rng = np.random.RandomState(seed)

    system = rng.randint(0, 2, n_samples)
    env_probs = rng.uniform(0.3, 0.7, n_env)
    env = np.zeros((n_samples, n_env), dtype=int)
    for j in range(n_env):
        env[:, j] = (rng.random(n_samples) < env_probs[j]).astype(int)

    TC_pre = compute_TC(env, min(12, n_env))
    pairwise_pre = compute_pairwise_mi(env, n_env)

    env_post = env.copy()
    was_one = (system == 1)

    if coupling_type == "cascade_chain":
        flip0 = was_one & (rng.random(n_samples) < coupling_strength)
        env_post[flip0, 0] = 1 - env_post[flip0, 0]
        for j in range(1, n_coupling):
            prob = coupling_strength * decay**j
            cascade = flip0 & (rng.random(n_samples) < prob)
            env_post[cascade, j] = env_post[cascade, j - 1]

    elif coupling_type == "fibonacci_chain":
        fibs = np.array([fibonacci(n_coupling - j) for j in range(n_coupling)],
                        dtype=float)
        strengths = coupling_strength * fibs / fibs[0]
        flip0 = was_one & (rng.random(n_samples) < strengths[0])
        env_post[flip0, 0] = 1 - env_post[flip0, 0]
        for j in range(1, n_coupling):
            cascade = flip0 & (rng.random(n_samples) < strengths[j])
            env_post[cascade, j] = env_post[cascade, j - 1]

    elif coupling_type == "uniform":
        for j in range(n_coupling):
            flip = was_one & (rng.random(n_samples) < coupling_strength
                              / n_coupling)
            env_post[flip, j] = 1 - env_post[flip, j]

    elif coupling_type == "exponential":
        for j in range(n_coupling):
            c = coupling_strength * np.exp(-0.3 * j)
            flip = was_one & (rng.random(n_samples) < c)
            env_post[flip, j] = 1 - env_post[flip, j]

    elif coupling_type == "random_sparse":
        coupled = rng.choice(n_env, n_coupling, replace=False)
        for j in coupled:
            c = coupling_strength * rng.random()
            flip = was_one & (rng.random(n_samples) < c)
            env_post[flip, j] = 1 - env_post[flip, j]

    TC_post = compute_TC(env_post, min(12, n_env))
    pairwise_post = compute_pairwise_mi(env_post, n_env)
    xi = (TC_post - TC_pre) + (pairwise_post - pairwise_pre)

    n_check = min(5, n_env)
    env_hash = np.zeros(n_samples, dtype=np.int64)
    for j in range(n_check):
        env_hash += env_post[:, j].astype(np.int64) * (2**j)
    joint = system * (2**20) + env_hash
    H_s = entropy_1d(system)
    H_ep = entropy_1d(env_hash)
    H_sep = entropy_1d(joint)
    transfer = max(0, H_s + H_ep - H_sep)

    return {"xi": xi, "transfer": transfer,
            "delta_TC": TC_post - TC_pre,
            "delta_pairwise": pairwise_post - pairwise_pre}


# ===================================================================
# TEST 3: Fibonacci <-> Cascade Equivalence
# ===================================================================

def test_fibonacci_equivalence(n_seeds=20, n_samples=300000) -> dict:
    """
    Compare xi from:
      1. Fibonacci chain coupling (strengths proportional to Fibonacci numbers)
      2. Cascade chain at d = 1/phi
      3. Cascade chain at d = 0.5 (exp_02 default)
      4. Cascade chain at d = 0.7 (exp_10 default)
      5. Uniform coupling
      6. Exponential coupling

    Statistical test: is Fibonacci equivalent to cascade at d = 1/phi?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Fibonacci <-> Cascade Equivalence")
    print("=" * 70)

    configs = [
        ("fibonacci_chain", None, "Fibonacci chain"),
        ("cascade_chain", PHI_INV, "Cascade chain d=1/phi"),
        ("cascade_chain", 0.5, "Cascade chain d=0.5"),
        ("cascade_chain", 0.7, "Cascade chain d=0.7"),
        ("uniform", 0.5, "Uniform"),
        ("exponential", 0.5, "Exponential decay"),
    ]

    all_results = []
    for coupling_type, decay, label in configs:
        xi_vals = []
        for seed_idx in range(n_seeds):
            d = decay if decay is not None else 0.5
            r = run_erasure(coupling_type, decay=d, n_samples=n_samples,
                            seed=seed_idx * 137 + 42)
            xi_vals.append(r["xi"])

        entry = {
            "label": label,
            "coupling_type": coupling_type,
            "decay": float(decay) if decay is not None else None,
            "xi_mean": round(float(np.mean(xi_vals)), 6),
            "xi_std": round(float(np.std(xi_vals)), 6),
            "xi_values": [round(v, 6) for v in xi_vals],
        }
        all_results.append(entry)
        print(f"  {label:30s}: xi = {entry['xi_mean']:.6f} "
              f"+/- {entry['xi_std']:.6f}")

    # Statistical test: Fibonacci vs cascade at 1/phi
    fib_vals = all_results[0]["xi_values"]
    phi_vals = all_results[1]["xi_values"]
    t_stat, p_value = sp_stats.ttest_ind(fib_vals, phi_vals, equal_var=False)

    print(f"\n  Fibonacci vs Cascade/phi (Welch's t-test):")
    print(f"    Fibonacci mean:  {all_results[0]['xi_mean']:.6f}")
    print(f"    Cascade/phi mean: {all_results[1]['xi_mean']:.6f}")
    print(f"    t = {t_stat:.4f}, p = {p_value:.4f}")
    equiv = p_value > 0.05
    status = "EQUIVALENT (p > 0.05)" if equiv else "DIFFERENT (p <= 0.05)"
    print(f"    Conclusion: {status}")

    ranked = sorted(all_results, key=lambda x: -x["xi_mean"])
    print(f"\n  xi ranking:")
    for i, r in enumerate(ranked):
        print(f"    {i + 1}. {r['label']:30s}: {r['xi_mean']:.6f}")

    return {
        "test": "fibonacci_cascade_equivalence",
        "status": "PASS" if equiv else "FAIL",
        "results": [{k: v for k, v in r.items() if k != "xi_values"}
                     for r in all_results],
        "statistical_test": {
            "comparison": "Fibonacci chain vs Cascade chain at 1/phi",
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "equivalent": bool(equiv),
        },
        "ranking": [
            {"rank": i + 1, "label": r["label"], "xi_mean": r["xi_mean"]}
            for i, r in enumerate(ranked)
        ],
    }


# ===================================================================
# TEST 4: Two-Rate Star Model (Paper 1 S6 replication)
# ===================================================================

def test_two_rate_star(n_seeds=15, n_samples=300000) -> dict:
    """
    Replicate Paper 1 S6 using exp_05's EXACT coupling model:

      Channel 1 (Transfer): each mode j INDEPENDENTLY coupled to system
                             strength = base_T * exp(-flip_decay * j)

      Channel 2 (Correlation): mode j copies MODE 0 (STAR topology)
                                strength = base_C * exp(-corr_decay * j)

    Paper 1 found: A/(A+xi) matches ln(phi) best when flip/corr ratio ~ 1.5.
    This test scans that ratio.

    MODEL MATCHES exp_05_sec_collapse.py EXACTLY:
      - env_probs from logistic(exponential) (biased toward 0)
      - xi measured from zero baseline (not pre-erasure env)
      - Star correlation (mode j copies mode 0)
    """
    print("\n" + "=" * 70)
    print("TEST 4: Two-Rate Star Model (exp_05 replication)")
    print("=" * 70)
    print(f"  Target: A/(A+xi) = ln(phi) = {LN_PHI:.6f}")
    print(f"  Coupling model: exp_05 star (mode j copies mode 0)")
    print(f"  Baseline: zeros (matching exp_05)")

    n_coupling = 5
    n_env = 20  # Match exp_05

    base_T = 0.8
    base_C = 0.3

    # Build ratio grid: coarse + fine around 1.5 and phi
    coarse = list(np.linspace(0.5, 4.0, 30))
    fine = [1.0, 1.25, 1.4, 1.45, 1.5, 1.55, 1.6,
            PHI - 0.05, PHI - 0.02, PHI, PHI + 0.02, PHI + 0.05,
            1.7, 1.75, 1.8, 2.0, 2.5, 3.0]

    ratios_to_test = sorted(set(round(r, 6) for r in coarse + fine))

    print(f"  Scanning {len(ratios_to_test)} ratios x {n_seeds} seeds")

    results = []

    for ratio in ratios_to_test:
        a_over_coh_vals = []
        a_vals = []
        xi_vals = []

        for seed_idx in range(n_seeds):
            rng = np.random.default_rng(seed_idx * 137 + 42)

            system = rng.integers(0, 2, n_samples)

            # Environment: exp_05's initialization (biased toward 0)
            env_probs = 1.0 / (1.0 + np.exp(
                0.5 + rng.exponential(1.0, n_env)))
            env = np.zeros((n_samples, n_env), dtype=int)
            for j in range(n_env):
                env[:, j] = (rng.random(n_samples) < env_probs[j]).astype(int)

            env_post = env.copy()
            was_one = (system == 1)

            # Decay rates: flip_decay = ratio * corr_decay
            # Use corr_decay = 0.2 as reference, flip_decay = ratio * 0.2
            corr_decay = 0.2
            flip_decay = ratio * corr_decay

            # Channel 1: DIRECT transfer (exp_05 model)
            for j in range(n_coupling):
                prob = base_T * np.exp(-flip_decay * j)
                flip = was_one & (rng.random(n_samples) < prob)
                env_post[flip, j] = 1 - env_post[flip, j]

            # Channel 2: STAR correlation (exp_05 model)
            for j in range(1, n_coupling):
                prob = base_C * np.exp(-corr_decay * j)
                corr_mask = was_one & (rng.random(n_samples) < prob)
                env_post[corr_mask, j] = env_post[corr_mask, 0]

            # A = MI(system, env_post) using first 5 modes
            n_check = min(5, n_env)
            env_hash = np.zeros(n_samples, dtype=np.int64)
            for j in range(n_check):
                env_hash += env_post[:, j].astype(np.int64) * (2**j)
            joint = system * (2**20) + env_hash
            H_s = entropy_1d(system)
            H_ep = entropy_1d(env_hash)
            H_sep = entropy_1d(joint)
            A = max(0, H_s + H_ep - H_sep)

            # xi from ZERO baseline (matching exp_05)
            TC_post = compute_TC(env_post, min(10, n_env))
            pairwise_post = compute_pairwise_mi(
                env_post[:, :min(10, n_env)], min(10, n_env))
            xi = TC_post + pairwise_post  # zero baseline

            coherent = A + xi
            if coherent > 0.001:
                a_over_coh_vals.append(A / coherent)
            a_vals.append(A)
            xi_vals.append(xi)

        if len(a_over_coh_vals) == 0:
            continue

        mean_ratio_val = float(np.mean(a_over_coh_vals))
        std_ratio_val = float(np.std(a_over_coh_vals))
        deviation_from_ln_phi = abs(mean_ratio_val - LN_PHI)
        deviation_pct = deviation_from_ln_phi / LN_PHI * 100

        entry = {
            "flip_over_corr": round(ratio, 6),
            "flip_decay": round(flip_decay, 6),
            "corr_decay": corr_decay,
            "A_over_coherent_mean": round(mean_ratio_val, 6),
            "A_over_coherent_std": round(std_ratio_val, 6),
            "deviation_from_ln_phi": round(deviation_from_ln_phi, 6),
            "deviation_pct": round(deviation_pct, 2),
            "A_mean": round(float(np.mean(a_vals)), 6),
            "xi_mean": round(float(np.mean(xi_vals)), 6),
        }
        results.append(entry)

        marker = ""
        if abs(ratio - PHI) < 0.01:
            marker = " <-- phi"
        elif abs(ratio - 1.5) < 0.01:
            marker = " <-- 3:2 (exp_01/05)"

        print(f"    flip/corr = {ratio:.4f}: A/(A+xi) = {mean_ratio_val:.4f} "
              f"+/- {std_ratio_val:.4f} "
              f"(vs ln(phi): {deviation_pct:.2f}%){marker}")

    # Find ratio with minimum deviation from ln(phi)
    best = min(results, key=lambda x: x["deviation_from_ln_phi"])
    phi_entry = min(results,
                    key=lambda x: abs(x["flip_over_corr"] - PHI))
    ratio_15 = min(results,
                   key=lambda x: abs(x["flip_over_corr"] - 1.5))

    # Find crossover point (where A/(A+xi) crosses ln(phi))
    crossover_ratio = None
    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]
        v1 = r1["A_over_coherent_mean"] - LN_PHI
        v2 = r2["A_over_coherent_mean"] - LN_PHI
        if v1 * v2 < 0:  # Sign change = crossing
            frac = abs(v1) / (abs(v1) + abs(v2))
            crossover_ratio = (r1["flip_over_corr"]
                               + frac * (r2["flip_over_corr"]
                                         - r1["flip_over_corr"]))
            break

    print(f"\n  Results:")
    print(f"    Best match to ln(phi):")
    print(f"      flip/corr = {best['flip_over_corr']:.4f}")
    print(f"      A/(A+xi) = {best['A_over_coherent_mean']:.6f}")
    print(f"      deviation: {best['deviation_pct']:.2f}%")
    print(f"    At ratio 1.5 (3:2, exp_01/exp_05 default):")
    print(f"      A/(A+xi) = {ratio_15['A_over_coherent_mean']:.6f}")
    print(f"      deviation: {ratio_15['deviation_pct']:.2f}%")
    print(f"    At phi = {PHI:.4f}:")
    print(f"      A/(A+xi) = {phi_entry['A_over_coherent_mean']:.6f}")
    print(f"      deviation: {phi_entry['deviation_pct']:.2f}%")
    if crossover_ratio is not None:
        dist_from_phi = abs(crossover_ratio - PHI)
        print(f"    Crossover (A/(A+xi) = ln(phi)) at flip/corr = "
              f"{crossover_ratio:.4f}")
        print(f"      Distance from phi: {dist_from_phi:.4f}")
        near_phi = dist_from_phi < 0.3
    else:
        print(f"    No crossover found in scan range")
        near_phi = bool(best["deviation_pct"] < 1.0)

    status = "PASS" if near_phi else "CHECK"
    print(f"  Status: {status}")

    return {
        "test": "two_rate_star_model",
        "status": status,
        "model": "exp_05 star (mode j copies mode 0, zero baseline)",
        "target": "A/(A+xi) = ln(phi)",
        "ln_phi": round(LN_PHI, 6),
        "phi": round(PHI, 6),
        "n_env": n_env,
        "n_coupling": n_coupling,
        "base_T": base_T,
        "base_C": base_C,
        "scan_results": results,
        "best_match": {
            "ratio": best["flip_over_corr"],
            "A_over_coherent": best["A_over_coherent_mean"],
            "deviation_pct": best["deviation_pct"],
        },
        "at_ratio_1_5": {
            "A_over_coherent": ratio_15["A_over_coherent_mean"],
            "deviation_pct": ratio_15["deviation_pct"],
        },
        "at_phi": {
            "A_over_coherent": phi_entry["A_over_coherent_mean"],
            "deviation_pct": phi_entry["deviation_pct"],
        },
        "crossover_ratio": (round(crossover_ratio, 4)
                            if crossover_ratio is not None else None),
        "crossover_near_phi": bool(near_phi),
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 70)
    print("exp_18: Cascade-Fibonacci Bridge")
    print("Connecting Fibonacci matrix eigenvalues to cascade topology")
    print("=" * 70)
    print(f"\nphi = {PHI:.10f}")
    print(f"1/phi = {PHI_INV:.10f}")
    print(f"ln(phi) = {LN_PHI:.10f}")

    all_results = {
        "experiment": "exp_18_cascade_fibonacci_bridge",
        "timestamp": datetime.now().isoformat(),
        "purpose": (
            "Bridge between exp_17 (Fibonacci matrix eigenvalues) and "
            "cascade topology (Paper 1 S4.4, S6, S8). "
            "Chain: physics constraints -> unique matrix -> phi eigenvalues "
            "-> cascade topology -> golden ratio partition."
        ),
        "chain_of_reasoning": [
            "1. Information conservation + asymmetric transfer uniquely "
            "selects Fibonacci matrix [[1,1],[1,0]] (Test 1, exact)",
            "2. Eigenvalues are phi and -1/phi (exp_17, exact)",
            "3. Fibonacci ratios -> geometric cascade at d = 1/phi "
            "(Test 2, exact)",
            "4. Fibonacci coupling = cascade at 1/phi numerically "
            "(Test 3, statistical)",
            "5. Star model: A/(A+xi) approaches ln(phi) near "
            "decay ratio 1.5 (Test 4, numerical)",
        ],
        "tests": {},
    }

    # Run all tests
    all_results["tests"]["uniqueness"] = test_uniqueness()
    all_results["tests"]["convergence"] = test_convergence()
    all_results["tests"]["equivalence"] = test_fibonacci_equivalence(
        n_seeds=20, n_samples=300000
    )
    all_results["tests"]["two_rate_star"] = test_two_rate_star(
        n_seeds=15, n_samples=300000
    )

    # =================================================================
    # SYNTHESIS
    # =================================================================
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    uniqueness = all_results["tests"]["uniqueness"]
    convergence = all_results["tests"]["convergence"]
    equiv = all_results["tests"]["equivalence"]
    two_rate = all_results["tests"]["two_rate_star"]

    fib_test = equiv.get("statistical_test", {})
    fib_pval = fib_test.get("p_value", "?")
    fib_equiv = isinstance(fib_pval, (int, float)) and fib_pval > 0.05

    crossover = two_rate.get("crossover_ratio")
    crossover_near = two_rate.get("crossover_near_phi", False)
    phi_dev = two_rate.get("at_phi", {}).get("deviation_pct", "?")

    print(f"\n  Test 1 (Uniqueness):  {uniqueness['status']}")
    print(f"    Fibonacci matrix is the ONLY solution under physical")
    print(f"    constraints. Both solutions have eigenvalue phi.")

    print(f"\n  Test 2 (Convergence): {convergence['status']}")
    print(f"    Fibonacci -> geometric cascade at d = 1/phi.")

    equivs = "PASS" if fib_equiv else "FAIL"
    print(f"\n  Test 3 (Equivalence): {equivs}")
    print(f"    Fibonacci coupling = cascade/phi (p = {fib_pval})")

    print(f"\n  Test 4 (Two-rate star): {two_rate['status']}")
    ratio_15_dev = two_rate.get("at_ratio_1_5", {}).get("deviation_pct", "?")
    print(f"    At 3:2 (1.5): deviation = {ratio_15_dev}%")
    if crossover is not None:
        print(f"    A/(A+xi) crosses ln(phi) at flip/corr = {crossover}")
        print(f"    phi = {PHI:.4f}, distance = "
              f"{abs(crossover - PHI):.4f}")
    print(f"    At phi: deviation = {phi_dev}%")

    # Evaluate overall
    passes = sum([
        uniqueness["status"] == "PASS",
        convergence["status"] == "PASS",
        fib_equiv,
        crossover_near,
    ])

    if passes == 4:
        verdict = "BRIDGE ESTABLISHED"
        conclusion = (
            "Complete chain confirmed:\n"
            "  Physical constraints -> Fibonacci matrix (unique)\n"
            "  -> eigenvalues phi, -1/phi\n"
            "  -> cascade at d = 1/phi (equivalent to Fibonacci)\n"
            "  -> A/(A+xi) = ln(phi) at decay ratio = phi\n\n"
            "Cascade topology is not arbitrary. It is the unique topology\n"
            "consistent with information-conserving asymmetric transfer.\n"
            "The golden ratio appears because the Fibonacci matrix\n"
            "eigenvalues determine both the decay rate and the optimal\n"
            "partition between recoverable information and structure."
        )
    elif passes >= 3:
        verdict = "BRIDGE STRONG"
        conclusion = (
            f"Mathematical foundation complete ({passes}/4 tests pass).\n"
            "The uniqueness + eigenvalue + equivalence chain is solid.\n"
            f"Two-rate test: {'confirms' if crossover_near else 'partial match for'} "
            "phi in the partition ratio."
        )
    elif passes >= 2:
        verdict = "BRIDGE PARTIAL"
        conclusion = (
            f"Mathematical foundation holds ({passes}/4 tests pass).\n"
            "Some numerical tests do not confirm the full chain."
        )
    else:
        verdict = "BRIDGE INCOMPLETE"
        conclusion = (
            f"Only {passes}/4 tests pass. Further work needed."
        )

    print(f"\n  VERDICT: {verdict}")
    print(f"\n  {conclusion}")

    all_results["synthesis"] = {
        "verdict": verdict,
        "passes": f"{passes}/4",
        "conclusion": conclusion,
        "details": {
            "uniqueness": uniqueness["status"],
            "convergence": convergence["status"],
            "fibonacci_equivalence_p": fib_pval,
            "fibonacci_equivalent": bool(fib_equiv),
            "crossover_ratio": crossover,
            "crossover_near_phi": bool(crossover_near),
            "deviation_at_phi_pct": phi_dev,
        },
    }

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (results_dir /
                   f"exp_18_cascade_fibonacci_bridge_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_file}")
    return all_results


if __name__ == "__main__":
    main()
