"""
Milestone 6 -- Exp 03: Tetration Penalty Derivation

Block A: The Scope Boundary Mechanism

PURPOSE: Prove 1/phi^4 size confounding IS the tetration termination penalty.
Extend exp_37 to 100 seeds, test lattice-size independence.

Tests:
  1. 100-seed mean within 5% of 1/phi^4 -> WILL PASS
  2. Per-level confounding ~ 1/phi per level -> WILL FAIL (not cleanly decomposable)
  3. Confounding lattice-size-independent (64 vs 128 within 10%) -> WILL PASS
  4. Removing 4th hierarchy level drops confounding below 1/phi^3 -> WILL FAIL

Predicted: 2/4
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# -- paths --
SCRIPT_DIR = Path(__file__).resolve().parent
CI_SCRIPTS = SCRIPT_DIR.parents[1] / "confluent_identity" / "scripts"
M6_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CI_SCRIPTS))
sys.path.insert(0, str(M6_ROOT))

from core.scope import PHI, INV_PHI

RESULTS_DIR = M6_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Target: 1/phi^4
TARGET = INV_PHI ** 4  # 0.14590...


# ============================================================
# Lattice + hierarchy construction (self-contained for multi-seed/size)
# ============================================================

def build_lattice(N, seed=42):
    """Build NxN periodic lattice with PAC-conservative diffusion."""
    rng = np.random.RandomState(seed)

    # Initialize P and A fields
    P = rng.randn(N, N) * 0.1
    A = rng.randn(N, N) * 0.1

    # Stone mask (obstacles)
    stone_mask = np.zeros((N, N), dtype=bool)
    n_stones = max(1, N * N // 200)
    for _ in range(n_stones):
        r, c = rng.randint(0, N, 2)
        stone_mask[r, c] = True

    # Diffuse to steady state
    alpha = 0.1
    for _ in range(2000):
        P_new = P.copy()
        A_new = A.copy()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            P_new += alpha * (np.roll(P, dr, axis=0 if dr else 1) - P) / 4
            A_new += alpha * (np.roll(A, dr, axis=0 if dr else 1) - A) / 4
        P_new[stone_mask] = 0
        A_new[stone_mask] = 0
        P = P_new
        A = A_new

    C = P + A
    return C, stone_mask


def build_adjacency(C):
    """Build weighted adjacency for NxN lattice with periodic BCs."""
    N = C.shape[0]
    n = N * N
    rows, cols, vals = [], [], []

    for r in range(N):
        for c in range(N):
            idx = r * N + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = (r + dr) % N, (c + dc) % N
                nidx = nr * N + nc
                w = 1.0 / (1.0 + abs(C[r, c] - C[nr, nc]))
                rows.append(idx)
                cols.append(nidx)
                vals.append(w)

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return W


def watershed_partition(C, stone_mask, n_levels=4):
    """
    Simple watershed-style hierarchical partition.
    Returns labels_by_level list and hierarchy dict.
    """
    N = C.shape[0]
    flat = C.ravel()

    # Level 0: quantile-based regions
    n_regions_l0 = max(10, N * N // 256)
    quantiles = np.linspace(0, 100, n_regions_l0 + 1)
    thresholds = np.percentile(flat[~stone_mask.ravel()], quantiles)

    labels_l0 = np.full(N * N, -1, dtype=int)
    for i in range(n_regions_l0):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == n_regions_l0 - 1:
            mask = (flat >= lo) & (~stone_mask.ravel())
        else:
            mask = (flat >= lo) & (flat < hi) & (~stone_mask.ravel())
        labels_l0[mask] = i + 1

    labels_by_level = [labels_l0.reshape(N, N)]
    hierarchy = {}

    # Higher levels: merge adjacent regions
    for lv in range(1, n_levels):
        prev_labels = labels_by_level[lv - 1].ravel()
        unique_prev = sorted(set(prev_labels) - {-1})

        if len(unique_prev) < 4:
            break

        # Group by quantile of mean field value
        means = {}
        for rid in unique_prev:
            idx = np.where(prev_labels == rid)[0]
            means[rid] = np.mean(flat[idx])

        n_groups = max(3, len(unique_prev) // 3)
        sorted_rids = sorted(unique_prev, key=lambda r: means[r])
        group_size = max(1, len(sorted_rids) // n_groups)

        new_labels = np.full(N * N, -1, dtype=int)
        for g in range(n_groups):
            start = g * group_size
            end = len(sorted_rids) if g == n_groups - 1 else (g + 1) * group_size
            group_rids = sorted_rids[start:end]
            gid = g + 1
            for rid in group_rids:
                idx = np.where(prev_labels == rid)[0]
                new_labels[idx] = gid
                hierarchy.setdefault((lv, gid), []).append((lv - 1, rid))

        labels_by_level.append(new_labels.reshape(N, N))

    return labels_by_level, hierarchy


def compute_size_coupling_confounding(C, stone_mask, labels_by_level, hierarchy, adjacency):
    """
    Compute the size-coupling confounding (partial correlation residual).
    Returns the confounding value: rho(coupling, natural) - partial_rho(coupling, natural | size).
    """
    flat = C.ravel()

    # Collect (size, coupling, natural_identity) per region
    sizes = []
    couplings = []
    naturals = []

    for lv in range(len(labels_by_level)):
        labels = labels_by_level[lv].ravel()
        unique = sorted(set(labels) - {-1})
        for rid in unique:
            idx = np.where(labels == rid)[0]
            if len(idx) < 3:
                continue

            size = len(idx)
            # Natural identity = mean field value
            natural = np.mean(flat[idx])
            # Coupling = boundary gradient (mean absolute difference at boundary)
            boundary_grad = 0.0
            n_boundary = 0
            for i in idx:
                r, c = divmod(int(i), C.shape[1])
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = (r + dr) % C.shape[0], (c + dc) % C.shape[1]
                    nidx = nr * C.shape[1] + nc
                    if labels[nidx] != rid and labels[nidx] != -1:
                        boundary_grad += abs(flat[i] - flat[nidx])
                        n_boundary += 1
            coupling = boundary_grad / max(n_boundary, 1)

            sizes.append(size)
            couplings.append(coupling)
            naturals.append(natural)

    sizes = np.array(sizes)
    couplings = np.array(couplings)
    naturals = np.array(naturals)

    if len(sizes) < 10:
        return np.nan, np.nan, np.nan

    # Raw correlation
    rho_raw, _ = spearmanr(couplings, naturals)

    # Partial correlation (coupling, natural | size) via residualization
    from scipy.stats import rankdata
    r_coup = rankdata(couplings)
    r_nat = rankdata(naturals)
    r_size = rankdata(sizes)

    # Residualize coupling on size
    A_coup = np.column_stack([r_size, np.ones(len(r_size))])
    beta_coup = np.linalg.lstsq(A_coup, r_coup, rcond=None)[0]
    resid_coup = r_coup - A_coup @ beta_coup

    # Residualize natural on size
    beta_nat = np.linalg.lstsq(A_coup, r_nat, rcond=None)[0]
    resid_nat = r_nat - A_coup @ beta_nat

    partial_rho = np.corrcoef(resid_coup, resid_nat)[0, 1]

    confounding = abs(rho_raw) - abs(partial_rho)
    return confounding, rho_raw, partial_rho


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 70)
    print("MILESTONE 6 - EXP 03: TETRATION PENALTY DERIVATION")
    print("Block A: The Scope Boundary Mechanism")
    print("=" * 70)
    print(f"\n  Target: 1/phi^4 = {TARGET:.6f}")

    # ============================================================
    # TEST 1: 100-seed mean at 128x128
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: 100-SEED CONFOUNDING AT 128x128")
    print("=" * 60)

    N = 128
    n_seeds = 100
    confoundings_128 = []

    for seed in range(n_seeds):
        C, stone = build_lattice(N, seed=seed)
        adj = build_adjacency(C)
        labels, hier = watershed_partition(C, stone)
        conf, rho_raw, partial_rho = compute_size_coupling_confounding(
            C, stone, labels, hier, adj
        )
        if not np.isnan(conf):
            confoundings_128.append(conf)

        if (seed + 1) % 25 == 0:
            print(f"    Seed {seed + 1}/{n_seeds}: "
                  f"running mean = {np.mean(confoundings_128):.6f}")

    mean_128 = np.mean(confoundings_128)
    std_128 = np.std(confoundings_128)
    delta_pct = abs(mean_128 - TARGET) / TARGET * 100

    print(f"\n  128x128 results (n={len(confoundings_128)} seeds):")
    print(f"    Mean confounding: {mean_128:.6f}")
    print(f"    Std: {std_128:.6f}")
    print(f"    CV: {std_128 / abs(mean_128 + 1e-30):.4f}")
    print(f"    Target 1/phi^4: {TARGET:.6f}")
    print(f"    Delta: {delta_pct:.2f}%")

    # ============================================================
    # TEST 2: Per-level confounding decomposition
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: PER-LEVEL CONFOUNDING DECOMPOSITION")
    print("=" * 60)

    # Use seed=42 for detailed per-level analysis
    C42, stone42 = build_lattice(128, seed=42)
    adj42 = build_adjacency(C42)
    labels42, hier42 = watershed_partition(C42, stone42)

    per_level_conf = {}
    flat42 = C42.ravel()

    for lv in range(len(labels42)):
        lbl = labels42[lv].ravel()
        unique = sorted(set(lbl) - {-1})
        if len(unique) < 5:
            continue

        sizes = []
        couplings = []
        naturals = []
        for rid in unique:
            idx = np.where(lbl == rid)[0]
            if len(idx) < 3:
                continue
            size = len(idx)
            natural = np.mean(flat42[idx])
            # Boundary coupling
            bg = 0.0
            nb = 0
            for i in idx:
                r, c = divmod(int(i), C42.shape[1])
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = (r + dr) % C42.shape[0], (c + dc) % C42.shape[1]
                    nidx = nr * C42.shape[1] + nc
                    if lbl[nidx] != rid and lbl[nidx] != -1:
                        bg += abs(flat42[i] - flat42[nidx])
                        nb += 1
            coupling = bg / max(nb, 1)
            sizes.append(size)
            couplings.append(coupling)
            naturals.append(natural)

        if len(sizes) < 5:
            continue

        sizes = np.array(sizes)
        couplings = np.array(couplings)
        naturals = np.array(naturals)

        rho_raw, _ = spearmanr(couplings, naturals)

        from scipy.stats import rankdata
        r_c = rankdata(couplings)
        r_n = rankdata(naturals)
        r_s = rankdata(sizes)
        A_ = np.column_stack([r_s, np.ones(len(r_s))])
        bc = np.linalg.lstsq(A_, r_c, rcond=None)[0]
        bn = np.linalg.lstsq(A_, r_n, rcond=None)[0]
        rc = r_c - A_ @ bc
        rn = r_n - A_ @ bn
        partial = np.corrcoef(rc, rn)[0, 1]
        conf = abs(rho_raw) - abs(partial)
        per_level_conf[lv] = {
            'confounding': float(conf),
            'rho_raw': float(rho_raw),
            'partial_rho': float(partial),
            'n_regions': len(sizes),
        }
        print(f"    Level {lv}: confounding={conf:.4f}, rho_raw={rho_raw:.4f}, "
              f"partial={partial:.4f}, n={len(sizes)}")

    # Check if per-level ~ 1/phi
    level_confs = [v['confounding'] for v in per_level_conf.values()]
    if len(level_confs) >= 2:
        per_level_ratios = [level_confs[i + 1] / (level_confs[i] + 1e-15)
                           for i in range(len(level_confs) - 1)]
        print(f"\n  Per-level ratios: {[f'{r:.4f}' for r in per_level_ratios]}")
        print(f"  1/phi = {INV_PHI:.4f}")
    else:
        per_level_ratios = []

    # ============================================================
    # TEST 3: Lattice-size independence (64 vs 128)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: LATTICE-SIZE INDEPENDENCE")
    print("=" * 60)

    n_seeds_size = 30
    confoundings_64 = []

    for seed in range(n_seeds_size):
        C, stone = build_lattice(64, seed=seed)
        adj = build_adjacency(C)
        labels, hier = watershed_partition(C, stone, n_levels=3)  # fewer levels at 64x64
        conf, _, _ = compute_size_coupling_confounding(C, stone, labels, hier, adj)
        if not np.isnan(conf):
            confoundings_64.append(conf)

    mean_64 = np.mean(confoundings_64)
    std_64 = np.std(confoundings_64)

    size_delta = abs(mean_64 - mean_128) / abs(mean_128 + 1e-30) * 100

    print(f"\n  64x64: mean={mean_64:.6f}, std={std_64:.6f} (n={len(confoundings_64)})")
    print(f"  128x128: mean={mean_128:.6f}, std={std_128:.6f} (n={len(confoundings_128)})")
    print(f"  Size delta: {size_delta:.1f}%")

    # ============================================================
    # TEST 4: Remove 4th level -> confounding drops
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: REMOVING 4th LEVEL")
    print("=" * 60)

    confoundings_3level = []
    for seed in range(30):
        C, stone = build_lattice(128, seed=seed)
        adj = build_adjacency(C)
        labels, hier = watershed_partition(C, stone, n_levels=3)  # only 3 levels
        conf, _, _ = compute_size_coupling_confounding(C, stone, labels, hier, adj)
        if not np.isnan(conf):
            confoundings_3level.append(conf)

    mean_3lv = np.mean(confoundings_3level)
    inv_phi3 = INV_PHI ** 3

    print(f"\n  4-level confounding: {mean_128:.6f}")
    print(f"  3-level confounding: {mean_3lv:.6f}")
    print(f"  1/phi^3 = {inv_phi3:.6f}")
    print(f"  3-level below 1/phi^3: {mean_3lv < inv_phi3}")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    test1 = delta_pct < 5.0
    print(f"\n  Test 1: 100-seed mean within 5% of 1/phi^4")
    print(f"    Mean: {mean_128:.6f}, target: {TARGET:.6f}, delta: {delta_pct:.2f}%")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    # Test 2: per-level ~ 1/phi
    if per_level_ratios:
        mean_ratio = np.mean(per_level_ratios)
        ratio_delta = abs(mean_ratio - INV_PHI) / INV_PHI * 100
        test2 = ratio_delta < 20
    else:
        test2 = False
        mean_ratio = float('nan')
        ratio_delta = float('nan')
    print(f"\n  Test 2: Per-level confounding ~ 1/phi per level")
    print(f"    Mean ratio: {mean_ratio:.4f}, 1/phi: {INV_PHI:.4f}, delta: {ratio_delta:.1f}%")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    test3 = size_delta < 10.0
    print(f"\n  Test 3: Lattice-size independent (64 vs 128 within 10%)")
    print(f"    Delta: {size_delta:.1f}%")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    test4 = mean_3lv < inv_phi3
    print(f"\n  Test 4: Removing 4th level drops confounding below 1/phi^3")
    print(f"    3-level: {mean_3lv:.6f}, 1/phi^3: {inv_phi3:.6f}")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    # -- Save results --
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_03_tetration_penalty_derivation',
        'milestone': 6,
        'block': 'A',
        'target_inv_phi4': float(TARGET),
        'test1_100seed': {
            'mean': float(mean_128),
            'std': float(std_128),
            'delta_pct': float(delta_pct),
            'n_seeds': len(confoundings_128),
        },
        'test2_per_level': per_level_conf,
        'test3_size_independence': {
            'mean_64': float(mean_64),
            'mean_128': float(mean_128),
            'delta_pct': float(size_delta),
        },
        'test4_3level': {
            'mean_3level': float(mean_3lv),
            'mean_4level': float(mean_128),
            'inv_phi3': float(inv_phi3),
        },
        'verification': {
            'test1': test1,
            'test2': test2,
            'test3': test3,
            'test4': test4,
            'verified_count': verified,
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = RESULTS_DIR / f"exp_03_tetration_penalty_derivation_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
