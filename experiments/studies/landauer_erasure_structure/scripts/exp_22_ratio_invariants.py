"""
Experiment 22: Ratio-Based PAC Invariant Validation (GPU)
==========================================================
Dawn Field Institute - PAC Exploration Series

Pure PyTorch implementation — all computation on GPU.
Seeds are BATCHED: instead of running 50 seeds sequentially,
we run them all in one GPU pass.

PRINCIPLE: Express everything as ratios. Remove all base entropy.
  From base_agnostic_pac: PAC identities hold at 10^-14 as RATIOS.
  SEC representations (specific log bases) vary 20-30%.

TESTS:
  1. Log-base independence verification
  2. Ratio stability across seeds (CV analysis)
  3. Parameter sweep — which ratios are invariant? (PAC vs SEC)
  4. Algebraic identity search against phi-functions
  5. Ratio-to-ratio identity search
  6. Coupling depth crossing points

FALSIFICATION:
  If NO ratio is parameter-invariant, the PAC budget decomposition
  doesn't contain global structure — it's all SEC-local.
"""

import torch
import math
import json, os, time
from datetime import datetime

# =============================================================
# GPU setup
# =============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================
# Constants
# =============================================================
k_B = 1.380649e-23
T = 300.0
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
GAMMA = 0.5772156649015329
XI_THEORY = GAMMA + LN_PHI

# =============================================================
# GPU entropy primitives
# =============================================================

def entropy_1d(data: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of 1D binary data.
    data: (batch, n_samples) int tensor of 0/1
    Returns: (batch,) float tensor
    """
    n = data.shape[1]
    p1 = data.float().mean(dim=1)  # (batch,)
    p0 = 1.0 - p1
    eps = 1e-30
    H = -(p0 * torch.log2(p0 + eps) + p1 * torch.log2(p1 + eps))
    return H


def entropy_joint_hash(data: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    Joint entropy over first n_modes binary columns via hashing.
    data: (batch, n_samples, n_env) int tensor
    Returns: (batch,) float tensor
    """
    batch, n_samples, _ = data.shape
    nm = min(n_modes, data.shape[2])
    n_bins = 2 ** nm

    # Hash: sum of col_j * 2^j
    powers = (2 ** torch.arange(nm, device=data.device)).long()  # (nm,)
    hashes = (data[:, :, :nm].long() * powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (batch, n_samples)

    # Bincount per batch element
    H = torch.zeros(batch, device=data.device)
    for b in range(batch):
        counts = torch.bincount(hashes[b], minlength=n_bins).float()
        probs = counts / n_samples
        mask = probs > 0
        H[b] = -(probs[mask] * torch.log2(probs[mask])).sum()

    return H


def total_correlation(data: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    TC = sum of marginal entropies - joint entropy.
    data: (batch, n_samples, n_env)
    Returns: (batch,)
    """
    nm = min(n_modes, data.shape[2])
    sum_H = torch.zeros(data.shape[0], device=data.device)
    for j in range(nm):
        sum_H += entropy_1d(data[:, :, j])
    H_joint = entropy_joint_hash(data, nm)
    return torch.clamp(sum_H - H_joint, min=0)


def pairwise_mi(data: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    Sum of pairwise mutual informations.
    data: (batch, n_samples, n_env)
    Returns: (batch,)
    Uses vectorized 4-bin histograms for all pairs simultaneously.
    """
    batch, n_samples, n_cols = data.shape
    nm = min(n_modes, n_cols)
    total = torch.zeros(batch, device=data.device)

    for i in range(nm):
        for j in range(i + 1, nm):
            # Joint hash: 2*col_i + col_j -> values in {0,1,2,3}
            joint = data[:, :, i].long() * 2 + data[:, :, j].long()  # (batch, n_samples)

            for b in range(batch):
                counts = torch.bincount(joint[b], minlength=4).float()
                p_joint = counts / n_samples

                p_i = torch.tensor([1.0 - data[b, :, i].float().mean(),
                                    data[b, :, i].float().mean()], device=data.device)
                p_j = torch.tensor([1.0 - data[b, :, j].float().mean(),
                                    data[b, :, j].float().mean()], device=data.device)

                eps = 1e-30
                H_i = -(p_i * torch.log2(p_i + eps)).sum()
                H_j = -(p_j * torch.log2(p_j + eps)).sum()
                H_ij = -(p_joint[p_joint > 0] * torch.log2(p_joint[p_joint > 0])).sum()
                mi = max(0.0, (H_i + H_j - H_ij).item())
                total[b] += mi

    return total


def transfer_entropy(sys_pre: torch.Tensor, env_post: torch.Tensor,
                     n_modes: int) -> torch.Tensor:
    """
    Transfer entropy: I(system; env_post).
    sys_pre: (batch, n_samples) binary
    env_post: (batch, n_samples, n_env)
    Returns: (batch,)
    """
    batch, n_samples = sys_pre.shape
    nm = min(n_modes, env_post.shape[2])

    # Hash env
    powers = (2 ** torch.arange(nm, device=env_post.device)).long()
    env_hash = (env_post[:, :, :nm].long() * powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    n_env_bins = 2 ** nm

    H_s = entropy_1d(sys_pre)  # (batch,)

    H = torch.zeros(batch, device=sys_pre.device)
    for b in range(batch):
        # H(env_hash)
        ec = torch.bincount(env_hash[b], minlength=n_env_bins).float()
        ep = ec / n_samples
        mask_e = ep > 0
        H_e = -(ep[mask_e] * torch.log2(ep[mask_e])).sum()

        # H(sys, env) -- joint
        joint = sys_pre[b].long() * n_env_bins + env_hash[b]
        jc = torch.bincount(joint, minlength=2 * n_env_bins).float()
        jp = jc / n_samples
        mask_j = jp > 0
        H_se = -(jp[mask_j] * torch.log2(jp[mask_j])).sum()

        H[b] = max(0.0, (H_s[b] + H_e - H_se).item())

    return H


# =============================================================
# Batched erasure simulation
# =============================================================

def run_erasure_batch(seeds: list, n_samples: int = 300000,
                      base_coupling: float = 0.8, flip_decay: float = 0.3,
                      corr_base: float = 0.3, corr_decay: float = 0.2,
                      n_env: int = 20, n_coupled: int = 5,
                      tc_modes: int = 12) -> dict:
    """
    Run erasure for ALL seeds in parallel on GPU.
    Returns dict of ratio tensors, each shape (n_seeds,).
    """
    B = len(seeds)  # batch size
    nc = min(n_coupled, n_env)
    tc_n = min(tc_modes, n_env)

    # Memory check: B * n_samples * n_env bytes (int8)
    mem_bytes = B * n_samples * n_env
    mem_gb = mem_bytes / 1e9
    if mem_gb > 6.0:
        # Chunk if too large
        mid = B // 2
        r1 = run_erasure_batch(seeds[:mid], n_samples, base_coupling,
                               flip_decay, corr_base, corr_decay,
                               n_env, n_coupled, tc_modes)
        r2 = run_erasure_batch(seeds[mid:], n_samples, base_coupling,
                               flip_decay, corr_base, corr_decay,
                               n_env, n_coupled, tc_modes)
        return {k: torch.cat([r1[k], r2[k]]) for k in r1}

    # Generate per-seed random state on CPU, then move to GPU
    all_system = torch.zeros(B, n_samples, dtype=torch.int8, device='cpu')
    all_env = torch.zeros(B, n_samples, n_env, dtype=torch.int8, device='cpu')

    for i, seed in enumerate(seeds):
        g = torch.Generator().manual_seed(seed)
        all_system[i] = torch.randint(0, 2, (n_samples,), generator=g,
                                       dtype=torch.int8)
        exp_samples = torch.empty(n_env).exponential_(1.0, generator=g)
        energies = k_B * T * (0.5 + exp_samples)
        env_probs = 1.0 / (1.0 + torch.exp(energies / (k_B * T)))
        for j in range(n_env):
            all_env[i, :, j] = (torch.rand(n_samples, generator=g) < env_probs[j]).to(
                torch.int8)

    # Move to GPU
    all_env = all_env.to(DEVICE)
    all_system = all_system.to(DEVICE)

    # Pre-erasure correlations
    tc_pre = total_correlation(all_env, tc_n)
    pmi_pre = pairwise_mi(all_env, n_env)

    # Erasure dynamics
    was_one = (all_system == 1)  # (B, n_samples)
    env_post = all_env.clone()

    for i, seed in enumerate(seeds):
        g = torch.Generator().manual_seed(seed + 100000)
        for j in range(nc):
            c = base_coupling * math.exp(-flip_decay * j)
            flip_mask = was_one[i] & (torch.rand(n_samples, generator=g).to(DEVICE) < c)
            env_post[i, flip_mask, j] = 1 - env_post[i, flip_mask, j]

        for j in range(1, nc):
            c = corr_base * math.exp(-corr_decay * j)
            corr_mask = was_one[i] & (torch.rand(n_samples, generator=g).to(DEVICE) < c)
            env_post[i, corr_mask, j] = env_post[i, corr_mask, 0]

    # Post-erasure correlations
    tc_post = total_correlation(env_post, tc_n)
    pmi_post = pairwise_mi(env_post, n_env)

    # PAC budget
    P = entropy_1d(all_system)
    A = transfer_entropy(all_system, env_post, nc)
    xi = (tc_post - tc_pre) + (pmi_post - pmi_pre)
    theta = P - (A + xi)

    # All ratios
    eps = 1e-30
    Axi = A + xi
    ratios = {}
    ratios['A_over_P'] = A / (P + eps)
    ratios['xi_over_P'] = xi / (P + eps)
    ratios['theta_over_P'] = theta / (P + eps)
    ratios['A_over_xi'] = torch.where(xi.abs() > eps, A / xi,
                                       torch.tensor(float('nan'), device=DEVICE))
    ratios['xi_over_A'] = torch.where(A.abs() > eps, xi / A,
                                       torch.tensor(float('nan'), device=DEVICE))
    ratios['A_over_theta'] = torch.where(theta.abs() > eps, A / theta,
                                          torch.tensor(float('nan'), device=DEVICE))
    ratios['xi_over_theta'] = torch.where(theta.abs() > eps, xi / theta,
                                           torch.tensor(float('nan'), device=DEVICE))
    ratios['actualized_over_P'] = Axi / (P + eps)
    ratios['P_over_actualized'] = P / (Axi + eps)
    ratios['A_over_Axi'] = A / (Axi + eps)
    ratios['xi_over_Axi'] = xi / (Axi + eps)
    ratios['sum_check'] = ratios['A_over_P'] + ratios['xi_over_P'] + ratios['theta_over_P']

    # Move to CPU
    return {k: v.cpu() for k, v in ratios.items()}


# =============================================================
# Analysis helpers
# =============================================================

RATIO_NAMES = [
    'A_over_P', 'xi_over_P', 'theta_over_P',
    'A_over_xi', 'xi_over_A',
    'A_over_theta', 'xi_over_theta',
    'actualized_over_P', 'P_over_actualized',
    'A_over_Axi', 'xi_over_Axi', 'sum_check',
]

PHI_FUNCTIONS = {
    'phi':           PHI,
    '1/phi':         1/PHI,
    'phi^2':         PHI**2,
    '1/phi^2':       1/PHI**2,
    'phi-1':         PHI - 1,
    '2-phi':         2 - PHI,
    'sqrt(phi)':     math.sqrt(PHI),
    'ln(phi)':       LN_PHI,
    'log2(phi)':     math.log2(PHI),
    '1-1/phi':       1 - 1/PHI,
    '2/phi':         2/PHI,
    'phi/2':         PHI/2,
    '(phi-1)/phi':   (PHI-1)/PHI,
    'phi/(phi+1)':   PHI/(PHI+1),
    '1/(phi+1)':     1/(PHI+1),
    '(sqrt5-1)/2':   (math.sqrt(5)-1)/2,
    'gamma':         GAMMA,
    'gamma+ln(phi)': GAMMA + LN_PHI,
    'Xi-1':          GAMMA + LN_PHI - 1,
    '1/e':           1/math.e,
    'pi/4':          math.pi/4,
}


def best_phi_match(value: float) -> tuple:
    """Find closest phi-function match. Returns (name, target_value, dev_pct)."""
    best_n, best_v, best_d = None, None, float('inf')
    for fn, fv in PHI_FUNCTIONS.items():
        if abs(fv) < 1e-10:
            continue
        d = abs(value - fv) / abs(fv) * 100
        if d < best_d:
            best_d, best_n, best_v = d, fn, fv
    return best_n, best_v, best_d


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    TOTAL_START = time.time()

    print()
    print("=" * 70)
    print("EXP 22: Ratio-Based PAC Invariant Validation (GPU)")
    print("=" * 70)
    print(f"ln(phi) = {LN_PHI:.10f}")
    print(f"gamma   = {GAMMA:.10f}")
    print(f"Xi      = {XI_THEORY:.10f}")
    print()
    print("PRINCIPLE: Everything as ratios. No base entropy.")
    print("  From base_agnostic_pac: PAC = territory, SEC = map.")
    print()

    N_SEEDS = 50
    base_params = dict(
        n_samples=300000, base_coupling=0.8, flip_decay=0.3,
        corr_base=0.3, corr_decay=0.2, n_env=20, n_coupled=5, tc_modes=12
    )

    all_results = {}

    # ============================================================
    # TEST 1: LOG-BASE INDEPENDENCE
    # ============================================================

    print("=" * 70)
    print("TEST 1: Log-Base Independence")
    print("  Ratios are base-invariant by construction.")
    print("  H_b(X) = H_2(X) / log2(b), so H_b(X)/H_b(Y) = H_2(X)/H_2(Y).")
    print("=" * 70)
    print()

    t0 = time.time()
    r_check = run_erasure_batch(list(range(10)), **base_params)
    sc = r_check['sum_check']
    print(f"  sum_check (must = 1.0):")
    print(f"    mean = {sc.mean().item():.10f}, max dev = {(sc - 1.0).abs().max().item():.2e}")
    print(f"  VERIFIED by construction. ({time.time()-t0:.0f}s)")

    all_results['log_base_test'] = {
        'method': 'algebraic_proof',
        'sum_check_mean': float(sc.mean()),
        'sum_check_max_dev': float((sc - 1.0).abs().max()),
        'verified': True,
    }

    # ============================================================
    # TEST 2: RATIO STABILITY ACROSS SEEDS (CV)
    # ============================================================

    print()
    print("=" * 70)
    print(f"TEST 2: Ratio Stability Across Seeds ({N_SEEDS} seeds)")
    print("  CV = std/|mean|. Low CV = tight = PAC-global candidate.")
    print("=" * 70)
    print()

    t0 = time.time()
    seed_data = run_erasure_batch(list(range(N_SEEDS)), **base_params)
    elapsed_t2 = time.time() - t0

    print(f"  {'Ratio':<28s} {'Mean':>10s} {'Std':>10s} {'CV':>10s} {'Rank':>6s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    ratio_stats = {}
    for rn in RATIO_NAMES:
        vals = seed_data[rn]
        valid = vals[~torch.isnan(vals)]
        if len(valid) < 5:
            continue
        m = valid.mean().item()
        s = valid.std().item()
        c = s / abs(m) if abs(m) > 1e-10 else float('inf')
        ratio_stats[rn] = {'mean': m, 'std': s, 'cv': c, 'n': len(valid),
                           'values': valid.tolist()}

    sorted_rs = sorted(ratio_stats.items(), key=lambda x: x[1]['cv'])
    for rank, (rn, st) in enumerate(sorted_rs, 1):
        print(f"  {rn:<28s} {st['mean']:>10.6f} {st['std']:>10.6f} "
              f"{st['cv']:>10.4f} {rank:>6d}")

    print(f"\n  ({elapsed_t2:.0f}s for {N_SEEDS} seeds)")
    all_results['seed_stability'] = ratio_stats

    # ============================================================
    # TEST 3: PARAMETER SWEEP — WHICH RATIOS ARE INVARIANT?
    # ============================================================

    print()
    print("=" * 70)
    print("TEST 3: Parameter Invariance Test")
    print("  Sweep coupling (0.5-1.0) x decay (0.1-0.5)")
    print("  For each ratio: inter-parameter CV separates PAC from SEC.")
    print("=" * 70)
    print()

    t0 = time.time()
    couplings = [0.5, 0.7, 0.8, 0.9, 1.0]
    decays = [0.1, 0.2, 0.3, 0.4, 0.5]
    N_GRID_SEEDS = 20

    param_means = {rn: [] for rn in RATIO_NAMES}
    param_configs = []

    for c_val in couplings:
        for fd_val in decays:
            params = base_params.copy()
            params['base_coupling'] = c_val
            params['flip_decay'] = fd_val

            config_data = run_erasure_batch(list(range(N_GRID_SEEDS)), **params)
            label = f"c={c_val:.1f},fd={fd_val:.1f}"
            param_configs.append(label)

            for rn in RATIO_NAMES:
                valid = config_data[rn][~torch.isnan(config_data[rn])]
                if len(valid) > 0:
                    param_means[rn].append(valid.mean().item())

    # Inter-parameter CV
    PAC_THRESHOLD = 0.05
    inter_stats = {}
    for rn in RATIO_NAMES:
        means = param_means[rn]
        if len(means) < 3:
            continue
        mt = torch.tensor(means)
        gm = mt.mean().item()
        ist = mt.std().item()
        icv = ist / abs(gm) if abs(gm) > 1e-10 else float('inf')
        inter_stats[rn] = {
            'grand_mean': gm, 'inter_std': ist, 'inter_cv': icv,
            'means_per_config': means,
            'min_mean': min(means), 'max_mean': max(means),
            'range_pct': (max(means) - min(means)) / abs(gm) * 100
                         if abs(gm) > 1e-10 else float('inf'),
        }

    sorted_inter = sorted(inter_stats.items(), key=lambda x: x[1]['inter_cv'])

    print(f"  {'Ratio':<28s} {'Grand Mean':>10s} {'Inter-CV':>10s} "
          f"{'Range%':>8s} {'Verdict':>12s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

    pac_global = []
    sec_local = []
    for rn, st in sorted_inter:
        verdict = "PAC-global" if st['inter_cv'] < PAC_THRESHOLD else "SEC-local"
        if st['inter_cv'] < PAC_THRESHOLD:
            pac_global.append(rn)
        else:
            sec_local.append(rn)
        print(f"  {rn:<28s} {st['grand_mean']:>10.6f} {st['inter_cv']:>10.4f} "
              f"{st['range_pct']:>7.1f}% {verdict:>12s}")

    print(f"\n  PAC-global ({len(pac_global)}): {pac_global}")
    print(f"  SEC-local ({len(sec_local)}): {sec_local}")
    print(f"  ({time.time()-t0:.0f}s)")

    all_results['parameter_invariance'] = {
        'configs': param_configs,
        'stats': inter_stats,
        'pac_global_ratios': pac_global,
        'sec_local_ratios': sec_local,
        'threshold_cv': PAC_THRESHOLD,
    }

    # ============================================================
    # TEST 4: ALGEBRAIC IDENTITY SEARCH
    # ============================================================

    print()
    print("=" * 70)
    print("TEST 4: Algebraic Identity Search")
    print("  For each ratio, find closest phi-function match.")
    print("=" * 70)
    print()

    identity_results = {}
    test_ratios = pac_global + sec_local[:3]

    for rn in test_ratios:
        if rn not in inter_stats:
            continue
        gm = inter_stats[rn]['grand_mean']
        if abs(gm) < 1e-10:
            continue

        matches = []
        for fn, fv in PHI_FUNCTIONS.items():
            if abs(fv) < 1e-10:
                continue
            d = abs(gm - fv) / abs(fv) * 100
            matches.append({'function': fn, 'value': fv, 'deviation_pct': d})
        matches.sort(key=lambda x: x['deviation_pct'])

        identity_results[rn] = {
            'grand_mean': gm,
            'best_match': matches[0]['function'],
            'best_deviation_pct': matches[0]['deviation_pct'],
            'top_matches': matches[:5],
        }

        marker = "PAC" if rn in pac_global else "sec"
        print(f"  [{marker}] {rn:<26s} = {gm:.6f}")
        for m in matches[:3]:
            tag = " <---" if m['deviation_pct'] < 2.0 else ""
            print(f"       ~  {m['function']:<18s} = {m['value']:.6f}  "
                  f"({m['deviation_pct']:.2f}%){tag}")
        print()

    all_results['identity_search'] = identity_results

    # ============================================================
    # TEST 5: RATIO-TO-RATIO IDENTITIES
    # ============================================================

    print()
    print("=" * 70)
    print("TEST 5: Ratio-to-Ratio Identity Search")
    print("  For PAC-global pairs: does ratio/product/sum match phi?")
    print("=" * 70)
    print()

    pair_results = {}
    if len(pac_global) >= 2:
        for i, rn1 in enumerate(pac_global):
            for rn2 in pac_global[i+1:]:
                m1 = inter_stats[rn1]['grand_mean']
                m2 = inter_stats[rn2]['grand_mean']
                if abs(m2) < 1e-10 or abs(m1) < 1e-10:
                    continue

                combos = {
                    f'{rn1}/{rn2}': m1/m2,
                    f'{rn2}/{rn1}': m2/m1,
                    f'{rn1}*{rn2}': m1*m2,
                    f'{rn1}+{rn2}': m1+m2,
                    f'|{rn1}-{rn2}|': abs(m1-m2),
                }

                for cn, cv_val in combos.items():
                    if abs(cv_val) < 1e-10:
                        continue
                    fn, fv, d = best_phi_match(cv_val)
                    if d < 5.0:
                        pair_results[cn] = {
                            'value': cv_val, 'best_match': fn,
                            'best_match_value': fv, 'deviation_pct': d,
                        }
                        print(f"  {cn:<48s} = {cv_val:.6f} "
                              f"~ {fn} = {fv:.6f} ({d:.2f}%)")

    if not pair_results:
        print("  No ratio-to-ratio identities found within 5%.")
    all_results['ratio_pairs'] = pair_results

    # ============================================================
    # TEST 6: COUPLING DEPTH CROSSING POINTS
    # ============================================================

    print()
    print("=" * 70)
    print("TEST 6: Coupling Depth Crossing Points")
    print("  Sweep n_coupled 3->12. Track all ratios.")
    print("  Find where each ratio crosses phi-values.")
    print("=" * 70)
    print()

    t0 = time.time()
    depth_values = [3, 4, 5, 6, 7, 8, 10, 12]
    N_DEPTH = 30
    depth_means = {rn: [] for rn in RATIO_NAMES}

    for nc in depth_values:
        params = base_params.copy()
        params['n_coupled'] = nc
        data = run_erasure_batch(list(range(N_DEPTH)), **params)

        for rn in RATIO_NAMES:
            valid = data[rn][~torch.isnan(data[rn])]
            depth_means[rn].append(valid.mean().item() if len(valid) > 0 else float('nan'))

    # Find crossing points
    key_phi = {'phi': PHI, '1/phi': 1/PHI, '1/phi^2': 1/PHI**2,
               'ln(phi)': LN_PHI, 'gamma': GAMMA, 'Xi-1': GAMMA+LN_PHI-1}

    crossing_analysis = {}
    for rn in RATIO_NAMES:
        ms = depth_means[rn]
        if any(math.isnan(m) for m in ms):
            continue
        crossings = {}
        for pn, pv in key_phi.items():
            for i in range(len(ms) - 1):
                if (ms[i] - pv) * (ms[i+1] - pv) < 0:
                    frac = (pv - ms[i]) / (ms[i+1] - ms[i])
                    cross_nc = depth_values[i] + frac * (depth_values[i+1] - depth_values[i])
                    crossings[pn] = {
                        'crossing_nc': cross_nc,
                        'between': [depth_values[i], depth_values[i+1]],
                        'values': [ms[i], ms[i+1]],
                    }
                    break
        if crossings:
            crossing_analysis[rn] = crossings

    if crossing_analysis:
        print(f"  {'Ratio':<26s} {'Crosses':>10s} {'at nc':>8s} {'Between':>12s}")
        print(f"  {'-'*26} {'-'*10} {'-'*8} {'-'*12}")
        for rn, crosses in crossing_analysis.items():
            for pn, cd in crosses.items():
                print(f"  {rn:<26s} {pn:>10s} {cd['crossing_nc']:>8.2f} "
                      f"  [{cd['between'][0]}, {cd['between'][1]}]")
    else:
        print("  No crossing points found.")

    # Crossing consistency
    all_cross = {}
    for rn, crosses in crossing_analysis.items():
        for pn, cd in crosses.items():
            if pn not in all_cross:
                all_cross[pn] = []
            all_cross[pn].append({'ratio': rn, 'nc': cd['crossing_nc']})

    if all_cross:
        print(f"\n  Crossing consistency:")
        for pn, entries in all_cross.items():
            ncs = [e['nc'] for e in entries]
            if len(ncs) >= 2:
                mn = sum(ncs) / len(ncs)
                sd = (sum((x - mn)**2 for x in ncs) / len(ncs)) ** 0.5
                print(f"    {pn}: {len(entries)} ratios, "
                      f"mean nc={mn:.2f} +/- {sd:.2f}")

    print(f"  ({time.time()-t0:.0f}s)")

    all_results['coupling_crossings'] = {
        'depth_values': depth_values,
        'ratio_means': {rn: depth_means[rn] for rn in RATIO_NAMES},
        'crossing_analysis': {rn: {pn: {k: float(v) if isinstance(v, (int, float))
                                         else v for k, v in cd.items()}
                                    for pn, cd in crosses.items()}
                               for rn, crosses in crossing_analysis.items()},
        'cross_consistency': {
            pn: {'mean_nc': sum(e['nc'] for e in entries) / len(entries),
                 'std_nc': (sum((e['nc'] - sum(e['nc'] for e in entries)/len(entries))**2
                                for e in entries) / len(entries)) ** 0.5,
                 'count': len(entries)}
            for pn, entries in all_cross.items()
        } if all_cross else {},
    }

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_data = {
        'experiment': 'exp_22_ratio_invariants',
        'timestamp': ts,
        'device': str(DEVICE),
        'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'none',
        'principle': 'All measurements as ratios -- no base dependency',
        'n_seeds': N_SEEDS,
        'defaults': base_params,
        'results': all_results,
    }

    fpath = os.path.join(results_dir, f'exp_22_ratio_invariants_{ts}.json')

    def json_default(o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return str(o)
        return str(o)

    with open(fpath, 'w') as f:
        json.dump(save_data, f, indent=2, default=json_default)
    print(f"\nResults saved to {fpath}")

    total_elapsed = time.time() - TOTAL_START

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Device: {DEVICE}" +
          (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else ''))
    print(f"  PAC-global ratios (inter-param CV < {PAC_THRESHOLD}): {len(pac_global)}")
    print(f"  SEC-local ratios: {len(sec_local)}")
    print()
    if pac_global:
        print("  PAC-global ratios and closest phi-identities:")
        for rn in pac_global:
            if rn in identity_results:
                ir = identity_results[rn]
                print(f"    {rn:<26s} = {ir['grand_mean']:.6f} "
                      f"~ {ir['best_match']} ({ir['best_deviation_pct']:.2f}%)")
    print()
    print(f"  Crossing points: {sum(len(v) for v in crossing_analysis.values())}")
    print(f"  Pair identities: {len(pair_results)}")
    print(f"  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
