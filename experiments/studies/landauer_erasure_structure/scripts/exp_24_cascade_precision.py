"""
Experiment 24: Cascade Precision — Θ Re-injection Done Right
==============================================================
Dawn Field Institute

exp_23's cascade was wrong: taking the highest-entropy env mode as the
new "system" for the next generation measures something fundamentally
different (pre-correlated system+env → inflated A → ratio ~0.94).

The CORRECT cascade (from journal §9):
  Θ from each generation re-injects as potential for the next.
  But at each generation, the ENVIRONMENT is FRESH and INDEPENDENT.
  Only the POTENTIAL changes (decreasing from Θ_{n-1}).

This tests the core hypothesis: A/(A+ξ) = ln(φ) is a STRUCTURAL INVARIANT
that holds regardless of initial potential P.

TEST 1: Clean Θ Cascade
  Gen 0: P₀ = 1 bit (fresh binary), fresh env. Measure A₀, ξ₀, Θ₀.
  Gen n: System entropy = Θ_{n-1}. Fresh env. Measure.
  → Does A/(A+ξ) = ln(φ) at EVERY generation?

TEST 2: Physical Cascade (exp_10-style, corrected)
  Highest-entropy env mode → system, remaining → env.
  With Miller-Madow correction and large N.
  → Does 53× amplification hold? What is cumulative ξ?

TEST 3: Meta-Cascade (Structure → Chaos → Structure)
  After Θ cascade dies: cumulative ξ becomes potential for NEW cascade.
  → Does A/(A+ξ) = ln(φ) at the meta-level?
  → This is the Phase I→II→III → Phase I pattern.

Pure PyTorch, GPU-accelerated, Miller-Madow corrected.
"""

import torch
import math
import json
import os
import time
from datetime import datetime

# ================================================================
# GPU
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ================================================================
# Constants
# ================================================================
k_B = 1.380649e-23
T = 300.0
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)   # 0.48121182505960344
GAMMA = 0.5772156649015329
XI_THEORY = GAMMA + LN_PHI

print(f"\nTarget: ln(φ) = {LN_PHI:.10f}")
print()

# ================================================================
# Bias-corrected entropy primitives (from exp_23)
# ================================================================

def entropy_1d_corrected(data: torch.Tensor) -> tuple:
    """
    Shannon entropy of binary data with Miller-Madow correction.
    data: (batch, n_samples) int tensor of {0,1}
    Returns: (H_raw, H_corrected) each (batch,)
    """
    n = data.shape[1]
    p1 = data.float().mean(dim=1)
    p0 = 1.0 - p1
    eps = 1e-30
    H_raw = -(p0 * torch.log2(p0 + eps) + p1 * torch.log2(p1 + eps))
    m_nz = torch.ones_like(H_raw) * 2.0
    m_nz[p0 < 1e-10] = 1.0
    m_nz[p1 < 1e-10] = 1.0
    correction = (m_nz - 1.0) / (2.0 * n * math.log(2))
    H_corrected = H_raw + correction
    return H_raw, H_corrected


def entropy_joint_corrected(data: torch.Tensor, n_modes: int, n_samples: int) -> tuple:
    """
    Joint entropy with Miller-Madow correction.
    data: (batch, n_samples, n_env)
    Returns: (H_raw, H_corrected, m_nonzero) each (batch,)
    """
    batch = data.shape[0]
    nm = min(n_modes, data.shape[2])
    n_bins = 2 ** nm
    powers = (2 ** torch.arange(nm, device=data.device)).long()
    hashes = (data[:, :, :nm].long() * powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)

    H_raw = torch.zeros(batch, device=data.device)
    H_corr = torch.zeros(batch, device=data.device)
    m_nz = torch.zeros(batch, device=data.device)

    for b in range(batch):
        counts = torch.bincount(hashes[b], minlength=n_bins).float()
        probs = counts / n_samples
        mask = probs > 0
        m_nonzero = mask.sum().float()
        H_raw[b] = -(probs[mask] * torch.log2(probs[mask])).sum()
        H_corr[b] = H_raw[b] + (m_nonzero - 1.0) / (2.0 * n_samples * math.log(2))
        m_nz[b] = m_nonzero

    return H_raw, H_corr, m_nz


def mi_system_env_corrected(sys_data: torch.Tensor, env_data: torch.Tensor,
                             n_modes: int, n_samples: int) -> tuple:
    """
    MI(system; env) with Miller-Madow correction.
    sys_data: (batch, n_samples) binary
    env_data: (batch, n_samples, n_env) binary
    Returns: (A_raw, A_corrected)
    """
    batch = sys_data.shape[0]
    nm = min(n_modes, env_data.shape[2])

    H_s_raw, H_s_corr = entropy_1d_corrected(sys_data)

    powers = (2 ** torch.arange(nm, device=env_data.device)).long()
    env_hash = (env_data[:, :, :nm].long() * powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    n_env_bins = 2 ** nm

    A_raw = torch.zeros(batch, device=sys_data.device)
    A_corr = torch.zeros(batch, device=sys_data.device)

    for b in range(batch):
        # H(env)
        ec = torch.bincount(env_hash[b], minlength=n_env_bins).float()
        ep = ec / n_samples
        mask_e = ep > 0
        m_e = mask_e.sum().float()
        H_e_raw = -(ep[mask_e] * torch.log2(ep[mask_e])).sum()
        H_e_corr = H_e_raw + (m_e - 1.0) / (2.0 * n_samples * math.log(2))

        # H(sys, env)
        joint = sys_data[b].long() * n_env_bins + env_hash[b]
        jc = torch.bincount(joint, minlength=2 * n_env_bins).float()
        jp = jc / n_samples
        mask_j = jp > 0
        m_j = mask_j.sum().float()
        H_se_raw = -(jp[mask_j] * torch.log2(jp[mask_j])).sum()
        H_se_corr = H_se_raw + (m_j - 1.0) / (2.0 * n_samples * math.log(2))

        A_raw[b] = H_s_raw[b] + H_e_raw - H_se_raw
        A_corr[b] = H_s_corr[b] + H_e_corr - H_se_corr

    return A_raw, A_corr


def total_correlation_corrected(data: torch.Tensor, n_modes: int, n_samples: int) -> tuple:
    """TC with Miller-Madow correction. No clamping."""
    nm = min(n_modes, data.shape[2])
    batch = data.shape[0]

    sum_H_raw = torch.zeros(batch, device=data.device)
    sum_H_corr = torch.zeros(batch, device=data.device)
    for j in range(nm):
        hr, hc = entropy_1d_corrected(data[:, :, j])
        sum_H_raw += hr
        sum_H_corr += hc

    Hj_raw, Hj_corr, _ = entropy_joint_corrected(data, nm, n_samples)

    TC_raw = sum_H_raw - Hj_raw
    TC_corr = sum_H_corr - Hj_corr
    return TC_raw, TC_corr


def pairwise_mi_corrected(data: torch.Tensor, n_modes: int, n_samples: int) -> tuple:
    """Sum of pairwise MI with Miller-Madow correction."""
    batch, _, n_cols = data.shape
    nm = min(n_modes, n_cols)
    total_raw = torch.zeros(batch, device=data.device)
    total_corr = torch.zeros(batch, device=data.device)

    for i in range(nm):
        for j in range(i + 1, nm):
            joint = data[:, :, i].long() * 2 + data[:, :, j].long()
            for b in range(batch):
                counts = torch.bincount(joint[b], minlength=4).float()
                p_joint = counts / n_samples
                mask = p_joint > 0
                m_nz = mask.sum().float()

                H_ij_raw = -(p_joint[mask] * torch.log2(p_joint[mask])).sum()
                H_ij_corr = H_ij_raw + (m_nz - 1.0) / (2.0 * n_samples * math.log(2))

                p_i0 = 1.0 - data[b, :, i].float().mean()
                p_i1 = 1.0 - p_i0
                p_j0 = 1.0 - data[b, :, j].float().mean()
                p_j1 = 1.0 - p_j0
                eps = 1e-30
                H_i = -(p_i0 * torch.log2(p_i0 + eps) + p_i1 * torch.log2(p_i1 + eps))
                H_j = -(p_j0 * torch.log2(p_j0 + eps) + p_j1 * torch.log2(p_j1 + eps))

                total_raw[b] += (H_i + H_j - H_ij_raw)
                H_i_c = H_i + 1.0 / (2.0 * n_samples * math.log(2))
                H_j_c = H_j + 1.0 / (2.0 * n_samples * math.log(2))
                total_corr[b] += (H_i_c + H_j_c - H_ij_corr)

    return total_raw, total_corr


# ================================================================
# Solve for bias probability: H(p) = target_entropy
# ================================================================

def solve_entropy_prob(target_H: float) -> float:
    """
    Find p such that binary entropy H(p) = target_H.
    H(p) = -p log2(p) - (1-p) log2(1-p)
    For H < 1, two solutions: p and 1-p. Returns p < 0.5.
    """
    if target_H >= 1.0:
        return 0.5
    if target_H <= 0.0:
        return 0.0

    # Bisection on [0, 0.5]
    lo, hi = 0.0, 0.5
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if mid <= 0 or mid >= 1:
            break
        H_mid = -(mid * math.log2(mid) + (1 - mid) * math.log2(1 - mid))
        if H_mid < target_H:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ================================================================
# Single erasure with tunable system entropy
# ================================================================

def run_single_erasure(n_seeds: int, n_samples: int, system_prob: float = 0.5,
                        base_coupling: float = 0.8, flip_decay: float = 0.3,
                        corr_base: float = 0.3, corr_decay: float = 0.2,
                        n_env: int = 20, n_coupled: int = 5,
                        tc_modes: int = 10, pmi_modes: int = 5,
                        base_seed: int = 0) -> dict:
    """
    Single Landauer erasure with tunable system probability.
    system_prob: probability of system=1. p=0.5 gives H=1 bit.
    Returns per-seed PAC budget.
    """
    nc = min(n_coupled, n_env)

    # Auto-reduce for large N
    if n_samples >= 2_000_000:
        tc_modes = min(tc_modes, 8)
        pmi_modes = min(pmi_modes, 4)

    all_A_corr = []
    all_xi_corr = []
    all_P_corr = []
    all_theta = []
    all_ratio = []

    # Process seeds one at a time to manage memory
    for s in range(n_seeds):
        seed = base_seed + s
        g = torch.Generator().manual_seed(seed)

        # System with tunable entropy
        system = (torch.rand(n_samples, generator=g) < system_prob).to(torch.int8)

        # Fresh independent environment
        env_pre = torch.zeros(n_samples, n_env, dtype=torch.int8)
        for j in range(n_env):
            env_pre[:, j] = torch.randint(0, 2, (n_samples,), generator=g, dtype=torch.int8)

        # Move to GPU
        system = system.to(DEVICE)
        env_pre = env_pre.to(DEVICE)

        # Erasure: system couples to environment
        env_post = env_pre.clone()
        was_one = (system == 1)
        g2 = torch.Generator().manual_seed(seed + 100000)
        for j in range(nc):
            c = base_coupling * math.exp(-flip_decay * j)
            flip_mask = was_one & (torch.rand(n_samples, generator=g2).to(DEVICE) < c)
            env_post[flip_mask, j] = 1 - env_post[flip_mask, j]
        for j in range(1, nc):
            c = corr_base * math.exp(-corr_decay * j)
            corr_mask = was_one & (torch.rand(n_samples, generator=g2).to(DEVICE) < c)
            env_post[corr_mask, j] = env_post[corr_mask, 0]

        # Measure PAC budget — batch dim = 1
        sys_2d = system.unsqueeze(0)
        env_pre_3d = env_pre.unsqueeze(0)
        env_post_3d = env_post.unsqueeze(0)

        tc_n = min(tc_modes, n_env)
        pmi_n = min(pmi_modes, n_env)

        P_raw, P_corr = entropy_1d_corrected(sys_2d)
        A_raw, A_corr = mi_system_env_corrected(sys_2d, env_post_3d, nc, n_samples)

        tc_pre_r, tc_pre_c = total_correlation_corrected(env_pre_3d, tc_n, n_samples)
        tc_post_r, tc_post_c = total_correlation_corrected(env_post_3d, tc_n, n_samples)
        pmi_pre_r, pmi_pre_c = pairwise_mi_corrected(env_pre_3d, pmi_n, n_samples)
        pmi_post_r, pmi_post_c = pairwise_mi_corrected(env_post_3d, pmi_n, n_samples)

        xi_corr = ((tc_post_c - tc_pre_c) + (pmi_post_c - pmi_pre_c)).item()
        A_cv = A_corr.item()
        P_cv = P_corr.item()
        theta = P_cv - A_cv - xi_corr

        Axi = A_cv + xi_corr
        ratio = A_cv / Axi if abs(Axi) > 1e-10 else float('nan')

        all_A_corr.append(A_cv)
        all_xi_corr.append(xi_corr)
        all_P_corr.append(P_cv)
        all_theta.append(theta)
        all_ratio.append(ratio)

        # Cleanup
        del system, env_pre, env_post, sys_2d, env_pre_3d, env_post_3d
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'A': all_A_corr,
        'xi': all_xi_corr,
        'P': all_P_corr,
        'theta': all_theta,
        'ratio': all_ratio,
    }


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    TOTAL_START = time.time()
    all_results = {}

    print("=" * 70)
    print("EXP 24: Cascade Precision — Θ Re-injection Done Right")
    print("=" * 70)

    # ============================================================
    # TEST 1: CLEAN Θ CASCADE
    # ============================================================
    print()
    print("=" * 70)
    print("TEST 1: Clean Θ Cascade")
    print("  Each generation: system entropy = Θ_{n-1}, fresh environment")
    print("  Tests: Is A/(A+ξ) = ln(φ) a structural invariant?")
    print("=" * 70)
    print()

    N_SAMPLES = 2_000_000
    N_SEEDS = 50
    MAX_GENS = 12

    cascade_data = []

    # Track per-seed cascade
    for s in range(N_SEEDS):
        t0 = time.time()
        current_theta = 1.0  # Gen 0: P = 1 bit (system_prob = 0.5)
        seed_cascade = []

        for gen in range(MAX_GENS):
            # System probability for this generation's entropy
            sys_prob = solve_entropy_prob(current_theta)

            if current_theta < 0.01 or sys_prob < 0.001:
                break  # Cascade died

            result = run_single_erasure(
                n_seeds=1, n_samples=N_SAMPLES,
                system_prob=sys_prob,
                base_coupling=0.8, flip_decay=0.3,
                corr_base=0.3, corr_decay=0.2,
                n_env=20, n_coupled=5,
                tc_modes=10, pmi_modes=5,
                base_seed=s * 1000 + gen * 100,
            )

            P = result['P'][0]
            A = result['A'][0]
            xi = result['xi'][0]
            theta = result['theta'][0]
            ratio = result['ratio'][0]

            entry = {
                'seed': s, 'gen': gen,
                'P': P, 'A': A, 'xi': xi, 'theta': theta,
                'ratio': ratio,
                'sys_prob': sys_prob,
                'target_entropy': current_theta,
            }
            seed_cascade.append(entry)
            cascade_data.append(entry)

            # Θ becomes potential for next generation
            current_theta = max(theta, 0)

        if s < 5 or s % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Seed {s:>3d}: {len(seed_cascade)} gens, "
                  f"final Θ={current_theta:.4f}, [{elapsed:.1f}s]")
            for e in seed_cascade:
                dev = (e['ratio'] - LN_PHI) / LN_PHI * 100
                print(f"    gen {e['gen']}: P={e['P']:.4f} A={e['A']:.4f} "
                      f"ξ={e['xi']:.4f} Θ={e['theta']:.4f} "
                      f"ratio={e['ratio']:.5f} ({dev:+.3f}%)")

    # Aggregate by generation
    from collections import defaultdict
    gen_ratios = defaultdict(list)
    gen_P = defaultdict(list)
    gen_xi = defaultdict(list)
    gen_theta = defaultdict(list)
    gen_A = defaultdict(list)

    for e in cascade_data:
        if not math.isnan(e['ratio']):
            gen_ratios[e['gen']].append(e['ratio'])
            gen_P[e['gen']].append(e['P'])
            gen_xi[e['gen']].append(e['xi'])
            gen_theta[e['gen']].append(e['theta'])
            gen_A[e['gen']].append(e['A'])

    print()
    print(f"  {'Gen':>4} {'N':>4} {'P_mean':>8} {'A/(A+ξ)':>10} {'Dev%':>8} "
          f"{'Std':>8} {'ξ_mean':>8} {'Θ_mean':>8}")
    print(f"  {'-'*70}")

    test1_stats = []
    for gen in sorted(gen_ratios.keys()):
        vals = gen_ratios[gen]
        if len(vals) < 5:
            continue
        m = sum(vals) / len(vals)
        s = (sum((v - m)**2 for v in vals) / len(vals)) ** 0.5
        se = s / math.sqrt(len(vals))
        dev = (m - LN_PHI) / LN_PHI * 100

        P_m = sum(gen_P[gen]) / len(gen_P[gen])
        xi_m = sum(gen_xi[gen]) / len(gen_xi[gen])
        th_m = sum(gen_theta[gen]) / len(gen_theta[gen])

        print(f"  {gen:>4} {len(vals):>4} {P_m:>8.4f} {m:>10.6f} {dev:>+7.3f}% "
              f"{s:>8.5f} {xi_m:>8.5f} {th_m:>8.5f}")

        test1_stats.append({
            'gen': gen, 'n_seeds': len(vals),
            'P_mean': P_m,
            'ratio_mean': m, 'ratio_std': s, 'ratio_se': se,
            'dev_pct': dev,
            'xi_mean': xi_m, 'theta_mean': th_m,
            'A_mean': sum(gen_A[gen]) / len(gen_A[gen]),
        })

    # Key question: is ratio invariant?
    if len(test1_stats) >= 2:
        gen0_m = test1_stats[0]['ratio_mean']
        gen0_se = test1_stats[0]['ratio_se']
        print(f"\n  Gen 0: A/(A+ξ) = {gen0_m:.6f} ({test1_stats[0]['dev_pct']:+.3f}% from ln(φ))")
        for st in test1_stats[1:]:
            diff = abs(st['ratio_mean'] - gen0_m)
            print(f"  Gen {st['gen']}: A/(A+ξ) = {st['ratio_mean']:.6f} "
                  f"({st['dev_pct']:+.3f}%) | Δ from gen0 = {diff:.6f}")

    # Cumulative ξ across cascade
    cum_xi_per_seed = defaultdict(float)
    alive_per_seed = defaultdict(int)
    for e in cascade_data:
        cum_xi_per_seed[e['seed']] += e['xi']
        alive_per_seed[e['seed']] += 1

    cum_xi_vals = list(cum_xi_per_seed.values())
    alive_vals = list(alive_per_seed.values())
    gen0_xi_vals = gen_xi.get(0, [0])

    if gen0_xi_vals and gen0_xi_vals[0] > 0:
        amplification = sum(cum_xi_vals) / len(cum_xi_vals) / (sum(gen0_xi_vals) / len(gen0_xi_vals))
    else:
        amplification = float('nan')

    print(f"\n  Cascade Statistics:")
    print(f"    Mean lifespan: {sum(alive_vals)/len(alive_vals):.1f} generations")
    print(f"    Gen 0 ξ: {sum(gen0_xi_vals)/len(gen0_xi_vals):.5f}")
    print(f"    Cumulative ξ: {sum(cum_xi_vals)/len(cum_xi_vals):.5f}")
    print(f"    Amplification: {amplification:.1f}×")

    all_results['test1_clean_cascade'] = {
        'n_samples': N_SAMPLES, 'n_seeds': N_SEEDS,
        'per_gen_stats': test1_stats,
        'mean_lifespan': sum(alive_vals) / len(alive_vals),
        'mean_cum_xi': sum(cum_xi_vals) / len(cum_xi_vals),
        'mean_gen0_xi': sum(gen0_xi_vals) / len(gen0_xi_vals),
        'amplification': amplification,
    }

    # ============================================================
    # TEST 2: PHYSICAL CASCADE (exp_10-style, corrected)
    # ============================================================
    print()
    print("=" * 70)
    print("TEST 2: Physical Cascade (highest-entropy mode → system)")
    print("  With Miller-Madow correction, N=1M, 50 seeds")
    print("=" * 70)
    print()

    N_PHYS = 1_000_000
    N_PHYS_SEEDS = 50
    N_PHYS_ENV = 8
    PHYS_DECAY = 0.7
    PHYS_GENS = 10

    phys_cascade_data = []

    for s in range(N_PHYS_SEEDS):
        seed = 5000 + s
        g = torch.Generator().manual_seed(seed)

        # Cascade coupling
        n_env = N_PHYS_ENV
        strengths = torch.tensor([PHYS_DECAY ** i for i in range(n_env)])
        strengths = strengths / strengths.sum() * 0.8

        # Gen 0: fresh binary system + fresh env
        system = torch.randint(0, 2, (N_PHYS,), generator=g, dtype=torch.int8).to(DEVICE)
        env = torch.zeros(N_PHYS, n_env, dtype=torch.int8, device=DEVICE)
        for j in range(n_env):
            env[:, j] = torch.randint(0, 2, (N_PHYS,), generator=g, dtype=torch.int8).to(DEVICE)

        for gen in range(PHYS_GENS):
            if n_env < 3:
                break

            # Measure pre-erasure TC
            env_pre_3d = env.unsqueeze(0)
            tc_n = min(n_env, 8)
            tc_pre_r, tc_pre_c = total_correlation_corrected(env_pre_3d, tc_n, N_PHYS)

            # Erasure
            env_post = env.clone()
            was_one = (system == 1)
            g_erase = torch.Generator().manual_seed(seed + gen * 50000)
            nc = n_env
            for j in range(nc):
                if strengths[j] > 0.001:
                    flip_mask = was_one & (torch.rand(N_PHYS, generator=g_erase).to(DEVICE) < strengths[j])
                    env_post[flip_mask, j] = system[flip_mask]

            # Post-erasure TC
            env_post_3d = env_post.unsqueeze(0)
            tc_post_r, tc_post_c = total_correlation_corrected(env_post_3d, tc_n, N_PHYS)

            # A = MI(system, env_post)
            sys_2d = system.unsqueeze(0)
            A_r, A_c = mi_system_env_corrected(sys_2d, env_post_3d, min(nc, 5), N_PHYS)

            xi_c = (tc_post_c - tc_pre_c).item()
            A_cv = A_c.item()
            P_r, P_c = entropy_1d_corrected(sys_2d)
            P_cv = P_c.item()
            theta = P_cv - A_cv - xi_c
            Axi = A_cv + xi_c
            ratio = A_cv / Axi if abs(Axi) > 1e-10 else float('nan')

            phys_cascade_data.append({
                'seed': s, 'gen': gen,
                'P': P_cv, 'A': A_cv, 'xi': xi_c, 'theta': theta,
                'ratio': ratio, 'n_env': n_env,
            })

            # Prepare next generation
            if theta < 0.01:
                break

            # Highest-entropy mode becomes system
            mode_H = []
            for j in range(n_env):
                p1 = env_post[:, j].float().mean()
                h = -(p1 * math.log2(max(p1.item(), 1e-30)) +
                      (1 - p1) * math.log2(max(1 - p1.item(), 1e-30)))
                mode_H.append(h.item() if isinstance(h, torch.Tensor) else h)
            best = max(range(n_env), key=lambda j: mode_H[j])
            system = env_post[:, best].clone()
            other = [j for j in range(n_env) if j != best]
            env = env_post[:, other].clone()
            n_env = len(other)
            strengths = torch.tensor([PHYS_DECAY ** i for i in range(n_env)])
            strengths = strengths / strengths.sum() * 0.8

            del env_pre_3d, env_post_3d, sys_2d
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        # Reset for next seed
        n_env = N_PHYS_ENV

        if s < 3 or s % 10 == 0:
            seed_entries = [e for e in phys_cascade_data if e['seed'] == s]
            print(f"  Seed {s}: {len(seed_entries)} gens")

    # Aggregate
    phys_gen_ratios = defaultdict(list)
    phys_gen_xi = defaultdict(list)

    for e in phys_cascade_data:
        if not math.isnan(e['ratio']):
            phys_gen_ratios[e['gen']].append(e['ratio'])
            phys_gen_xi[e['gen']].append(e['xi'])

    print(f"\n  {'Gen':>4} {'N':>4} {'A/(A+ξ)':>10} {'Dev%':>8} {'ξ_mean':>10}")
    print(f"  {'-'*45}")

    test2_stats = []
    for gen in sorted(phys_gen_ratios.keys()):
        vals = phys_gen_ratios[gen]
        xi_vals = phys_gen_xi[gen]
        if len(vals) < 3:
            continue
        m = sum(vals) / len(vals)
        s_val = (sum((v - m)**2 for v in vals) / len(vals)) ** 0.5
        dev = (m - LN_PHI) / LN_PHI * 100
        xi_m = sum(xi_vals) / len(xi_vals)

        print(f"  {gen:>4} {len(vals):>4} {m:>10.6f} {dev:>+7.3f}% {xi_m:>10.5f}")

        test2_stats.append({
            'gen': gen, 'n': len(vals),
            'ratio_mean': m, 'ratio_std': s_val, 'dev_pct': dev,
            'xi_mean': xi_m,
        })

    # Cumulative ξ
    phys_cum_xi = defaultdict(float)
    phys_gen0_xi = defaultdict(float)
    for e in phys_cascade_data:
        phys_cum_xi[e['seed']] += e['xi']
        if e['gen'] == 0:
            phys_gen0_xi[e['seed']] = e['xi']

    cum_vals = list(phys_cum_xi.values())
    g0_vals = list(phys_gen0_xi.values())
    phys_amp = (sum(cum_vals)/len(cum_vals)) / (sum(g0_vals)/len(g0_vals)) if g0_vals else 0

    print(f"\n  Physical cascade amplification: {phys_amp:.1f}×")
    print(f"  Single-event ξ: {sum(g0_vals)/len(g0_vals):.5f}")
    print(f"  Cumulative ξ:   {sum(cum_vals)/len(cum_vals):.5f}")

    all_results['test2_physical_cascade'] = {
        'n_samples': N_PHYS,
        'per_gen_stats': test2_stats,
        'amplification': phys_amp,
        'mean_gen0_xi': sum(g0_vals)/len(g0_vals),
        'mean_cum_xi': sum(cum_vals)/len(cum_vals),
    }

    # ============================================================
    # TEST 3: META-CASCADE (Structure → Chaos → Structure)
    # ============================================================
    print()
    print("=" * 70)
    print("TEST 3: Meta-Cascade")
    print("  After Θ cascade dies, cumulative ξ becomes potential for new cascade")
    print("  Tests the Phase I→II→III → Phase I recursion")
    print("=" * 70)
    print()

    N_META_SEEDS = 30
    N_META_SAMPLES = 2_000_000
    META_LEVELS = 4  # Number of meta-levels

    meta_results = []

    for s in range(N_META_SEEDS):
        t0 = time.time()
        meta_levels = []

        # Level 0: Standard cascade starting from P=1 bit
        current_potential = 1.0

        for level in range(META_LEVELS):
            if current_potential < 0.01:
                break

            # Run a cascade at this level
            level_cum_xi = 0.0
            current_theta = current_potential
            n_gens = 0

            for gen in range(10):
                sys_prob = solve_entropy_prob(current_theta)
                if current_theta < 0.01 or sys_prob < 0.001:
                    break

                result = run_single_erasure(
                    n_seeds=1, n_samples=N_META_SAMPLES,
                    system_prob=sys_prob,
                    base_coupling=0.8, flip_decay=0.3,
                    corr_base=0.3, corr_decay=0.2,
                    n_env=20, n_coupled=5,
                    tc_modes=10, pmi_modes=5,
                    base_seed=s * 100000 + level * 10000 + gen * 100,
                )

                A = result['A'][0]
                xi = result['xi'][0]
                theta = result['theta'][0]
                ratio = result['ratio'][0]

                level_cum_xi += xi
                current_theta = max(theta, 0)
                n_gens += 1

            # After this cascade dies: cumulative ξ becomes potential for NEXT level
            meta_entry = {
                'seed': s, 'level': level,
                'initial_potential': current_potential,
                'cum_xi': level_cum_xi,
                'n_gens': n_gens,
            }

            # Also measure A/(A+ξ) using cumulative ξ as potential
            # (Meta-level test: erase a system with H = cum_xi into fresh env)
            if level_cum_xi > 0.01:
                meta_prob = solve_entropy_prob(min(level_cum_xi, 1.0))
                meta_result = run_single_erasure(
                    n_seeds=1, n_samples=N_META_SAMPLES,
                    system_prob=meta_prob,
                    base_coupling=0.8, flip_decay=0.3,
                    corr_base=0.3, corr_decay=0.2,
                    n_env=20, n_coupled=5,
                    tc_modes=10, pmi_modes=5,
                    base_seed=s * 100000 + level * 10000 + 9999,
                )
                meta_entry['meta_ratio'] = meta_result['ratio'][0]
                meta_entry['meta_P'] = meta_result['P'][0]
                meta_entry['meta_A'] = meta_result['A'][0]
                meta_entry['meta_xi'] = meta_result['xi'][0]
            else:
                meta_entry['meta_ratio'] = float('nan')

            meta_levels.append(meta_entry)
            meta_results.append(meta_entry)

            # Structure becomes potential for next level
            current_potential = min(level_cum_xi, 1.0)  # Cap at 1 bit

        if s < 5 or s % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Seed {s}: {len(meta_levels)} meta-levels [{elapsed:.1f}s]")
            for ml in meta_levels:
                dev = ((ml['meta_ratio'] - LN_PHI) / LN_PHI * 100) if not math.isnan(ml['meta_ratio']) else float('nan')
                print(f"    Level {ml['level']}: P={ml['initial_potential']:.4f} → "
                      f"cum_ξ={ml['cum_xi']:.5f} ({ml['n_gens']} gens) → "
                      f"meta_ratio={ml['meta_ratio']:.5f} "
                      f"({dev:+.3f}%)" if not math.isnan(dev) else
                      f"    Level {ml['level']}: P={ml['initial_potential']:.4f} → "
                      f"cum_ξ={ml['cum_xi']:.5f} ({ml['n_gens']} gens) → "
                      f"meta_ratio=N/A")

    # Aggregate meta-level data
    meta_level_ratios = defaultdict(list)
    meta_level_xi = defaultdict(list)
    meta_level_P = defaultdict(list)
    for e in meta_results:
        if not math.isnan(e.get('meta_ratio', float('nan'))):
            meta_level_ratios[e['level']].append(e['meta_ratio'])
            meta_level_xi[e['level']].append(e['cum_xi'])
            meta_level_P[e['level']].append(e['initial_potential'])

    print(f"\n  Meta-Cascade Summary:")
    print(f"  {'Level':>6} {'N':>4} {'meta_ratio':>12} {'Dev%':>8} {'P':>8} {'cum_ξ':>10}")
    print(f"  {'-'*55}")

    test3_stats = []
    for lvl in sorted(meta_level_ratios.keys()):
        vals = meta_level_ratios[lvl]
        if len(vals) < 3:
            continue
        m = sum(vals) / len(vals)
        s_val = (sum((v - m)**2 for v in vals) / len(vals)) ** 0.5
        dev = (m - LN_PHI) / LN_PHI * 100
        P_m = sum(meta_level_P[lvl]) / len(meta_level_P[lvl])
        xi_m = sum(meta_level_xi[lvl]) / len(meta_level_xi[lvl])

        print(f"  {lvl:>6} {len(vals):>4} {m:>12.6f} {dev:>+7.3f}% "
              f"{P_m:>8.4f} {xi_m:>10.5f}")

        test3_stats.append({
            'level': lvl, 'n': len(vals),
            'ratio_mean': m, 'ratio_std': s_val, 'dev_pct': dev,
            'P_mean': P_m, 'cum_xi_mean': xi_m,
        })

    all_results['test3_meta_cascade'] = test3_stats

    # ============================================================
    # SAVE
    # ============================================================
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = os.path.join(results_dir, f'exp_24_cascade_precision_{ts}.json')

    def jd(o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return str(o)
        return str(o)

    with open(fpath, 'w') as f:
        json.dump(all_results, f, indent=2, default=jd)
    print(f"\nResults saved to {fpath}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total = time.time() - TOTAL_START
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Target: ln(φ) = {LN_PHI:.10f}")
    print()

    print("  TEST 1 — Clean Θ Cascade (ratio invariance):")
    if test1_stats:
        for st in test1_stats:
            print(f"    Gen {st['gen']}: A/(A+ξ) = {st['ratio_mean']:.6f} "
                  f"({st['dev_pct']:+.3f}%) ± {st['ratio_std']:.5f}")
        # Grand mean across all generations
        all_g = [s['ratio_mean'] for s in test1_stats]
        grand = sum(all_g) / len(all_g)
        grand_dev = (grand - LN_PHI) / LN_PHI * 100
        print(f"    → Grand mean: {grand:.6f} ({grand_dev:+.3f}%)")

    print()
    print("  TEST 2 — Physical Cascade (amplification):")
    print(f"    Amplification: {phys_amp:.1f}×")
    if test2_stats:
        print(f"    Gen 0 ratio: {test2_stats[0]['ratio_mean']:.6f} "
              f"({test2_stats[0]['dev_pct']:+.3f}%)")

    print()
    print("  TEST 3 — Meta-Cascade (structure → chaos → structure):")
    if test3_stats:
        for st in test3_stats:
            print(f"    Level {st['level']}: meta_ratio = {st['ratio_mean']:.6f} "
                  f"({st['dev_pct']:+.3f}%)")
    print()

    print(f"  Total runtime: {total:.0f}s ({total/60:.1f} min)")
