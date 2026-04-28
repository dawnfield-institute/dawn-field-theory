"""
Diagnose: why is h1 so low?

The exploratory work found h1 = Xi ~ 1.058 at sr ~ 1.2.
But measure_entropy_rate gives h1 ~ 0.15 at sr=1.2.

Possibilities:
1. Mode sequence is too predictable with 4 PCA modes
2. Need signed projections (double alphabet)
3. Need eigenmode tracking, not PCA
4. Need different discretization (state bins, not mode identity)
5. Block entropy estimation fails with short sequences

Test multiple approaches.
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import SelfApplicator, GAMMA_EM, LN_PHI

XI = GAMMA_EM + LN_PHI


def run_sa(N=16, sr=1.2, seed=0, n_steps=5000, burn_in=1000):
    """Run SelfApplicator, return trajectory and W snapshots."""
    sa = SelfApplicator(rule_seed=seed, self_applies=True, symmetric=True, size=N)
    eigvals = np.linalg.eigvalsh(sa.W)
    current_sr = np.max(np.abs(eigvals))
    if current_sr > 1e-10:
        sa.W = sa.W * (sr / current_sr)
    sa._target_sr = sr

    for _ in range(burn_in):
        sa.step()

    traj = np.zeros((n_steps, N))
    for t in range(n_steps):
        sa.step()
        traj[t] = sa.state

    return traj, sa


def entropy_rate_blocks(sequence, max_block=6):
    """Compute block entropies and conditional entropy rate."""
    block_H = []
    for L in range(1, max_block + 1):
        counts = {}
        for i in range(len(sequence) - L + 1):
            block = tuple(sequence[i:i + L])
            counts[block] = counts.get(block, 0) + 1
        total = sum(counts.values())
        probs = np.array(list(counts.values())) / total
        H = -np.sum(probs * np.log(probs))
        block_H.append(H)

    # Conditional entropy: h1 = H(L) - H(L-1) for last values
    conditionals = [block_H[i] - block_H[i-1] for i in range(1, len(block_H))]
    h1 = np.mean(conditionals[-3:]) if len(conditionals) >= 3 else conditionals[-1]
    return h1, block_H, conditionals


# ============================================================
# Run at default sr=1.2
# ============================================================
print("=" * 70)
print("DIAGNOSING h1 MEASUREMENT")
print("=" * 70)

traj, sa = run_sa(N=16, sr=1.2, seed=0, n_steps=8000)

print(f"\nTrajectory shape: {traj.shape}")
print(f"State range: [{traj.min():.4f}, {traj.max():.4f}]")
print(f"State std per dim: {np.mean(np.std(traj, axis=0)):.4f}")


# ============================================================
# Method 1: PCA mode identity (current implementation)
# ============================================================
print("\n--- Method 1: PCA mode identity (current) ---")
cov = np.cov(traj.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]

for n_modes in [4, 8, 16]:
    top_vecs = eigvecs[:, order[:n_modes]]
    proj = traj @ top_vecs
    mode_seq = np.argmax(np.abs(proj), axis=1)

    # Mode distribution
    unique, counts = np.unique(mode_seq, return_counts=True)
    print(f"  n_modes={n_modes}: alphabet={len(unique)}, "
          f"top3 freqs = {sorted(counts/len(mode_seq), reverse=True)[:3]}")

    h1, block_H, conds = entropy_rate_blocks(mode_seq, max_block=6)
    print(f"    H(1)={block_H[0]:.4f}, h1={h1:.4f}, Xi={XI:.4f}")


# ============================================================
# Method 2: Signed PCA mode (track sign of projection)
# ============================================================
print("\n--- Method 2: Signed PCA mode ---")
for n_modes in [4, 8]:
    top_vecs = eigvecs[:, order[:n_modes]]
    proj = traj @ top_vecs
    # Mode = 2*argmax(|proj|) + (sign of that projection)
    dominant_mode = np.argmax(np.abs(proj), axis=1)
    signs = np.array([1 if proj[t, dominant_mode[t]] >= 0 else 0 for t in range(len(proj))])
    mode_seq = 2 * dominant_mode + signs

    unique, counts = np.unique(mode_seq, return_counts=True)
    print(f"  n_modes={n_modes}: alphabet={len(unique)}, "
          f"top3 freqs = {sorted(counts/len(mode_seq), reverse=True)[:3]}")

    h1, block_H, conds = entropy_rate_blocks(mode_seq, max_block=6)
    print(f"    H(1)={block_H[0]:.4f}, h1={h1:.4f}, Xi={XI:.4f}")


# ============================================================
# Method 3: W eigenmode tracking (which eigenmode of current W
# aligns most with state)
# ============================================================
print("\n--- Method 3: W eigenmode tracking ---")
# Re-run tracking W eigenmodes
sa2 = SelfApplicator(rule_seed=0, self_applies=True, symmetric=True, size=16)
eigvals_w = np.linalg.eigvalsh(sa2.W)
current_sr = np.max(np.abs(eigvals_w))
if current_sr > 1e-10:
    sa2.W = sa2.W * (1.2 / current_sr)
sa2._target_sr = 1.2

for _ in range(1000):
    sa2.step()

n_track = 5000
w_mode_seq = np.zeros(n_track, dtype=int)
for t in range(n_track):
    sa2.step()
    evals, evecs = np.linalg.eigh(sa2.W)
    proj = (evecs.T @ sa2.state) ** 2
    w_mode_seq[t] = np.argmax(proj)

unique, counts = np.unique(w_mode_seq, return_counts=True)
print(f"  alphabet={len(unique)}, "
      f"top3 freqs = {sorted(counts/len(w_mode_seq), reverse=True)[:3]}")
h1, block_H, conds = entropy_rate_blocks(w_mode_seq, max_block=6)
print(f"    H(1)={block_H[0]:.4f}, h1={h1:.4f}, Xi={XI:.4f}")


# ============================================================
# Method 4: State space binning (discretize each dimension)
# ============================================================
print("\n--- Method 4: State-space binning ---")
# Bin the sign pattern of the state vector
sign_seq = np.zeros(len(traj), dtype=int)
for t in range(len(traj)):
    # Hash the sign pattern into a single integer
    signs = (traj[t] > 0).astype(int)
    # Use first 8 dimensions to keep alphabet manageable
    sign_seq[t] = sum(s * 2**i for i, s in enumerate(signs[:8]))

unique, counts = np.unique(sign_seq, return_counts=True)
print(f"  alphabet={len(unique)}, "
      f"top3 freqs = {sorted(counts/len(sign_seq), reverse=True)[:3]}")
h1, block_H, conds = entropy_rate_blocks(sign_seq, max_block=4)
print(f"    H(1)={block_H[0]:.4f}, h1={h1:.4f}, Xi={XI:.4f}")


# ============================================================
# Method 5: Rank order (which dimensions are largest)
# ============================================================
print("\n--- Method 5: Top-k dimension ranking ---")
for k in [2, 3, 4]:
    rank_seq = np.zeros(len(traj), dtype=int)
    for t in range(len(traj)):
        top_k = tuple(np.argsort(np.abs(traj[t]))[-k:][::-1])
        # Simple hash
        rank_seq[t] = sum(d * (16 ** i) for i, d in enumerate(top_k))

    unique, counts = np.unique(rank_seq, return_counts=True)
    print(f"  k={k}: alphabet={len(unique)}, "
          f"top3 freqs = {sorted(counts/len(rank_seq), reverse=True)[:3]}")
    h1, block_H, conds = entropy_rate_blocks(rank_seq, max_block=4)
    print(f"    H(1)={block_H[0]:.4f}, h1={h1:.4f}, Xi={XI:.4f}")


# ============================================================
# Method 6: Lempel-Ziv complexity (compression-based)
# ============================================================
print("\n--- Method 6: Lempel-Ziv complexity ---")
def lz_complexity(sequence):
    """Lempel-Ziv complexity: count distinct substrings."""
    s = list(sequence)
    n = len(s)
    complexity = 1
    i = 0
    while i < n:
        l = 1
        found = True
        while found and i + l <= n:
            substr = tuple(s[i:i+l])
            # Check if this substring appeared in s[0:i+l-1]
            found = False
            for j in range(i):
                if tuple(s[j:j+l]) == substr:
                    found = True
                    break
            if found:
                l += 1
        complexity += 1
        i += l - 1
        if i >= n:
            break
        i += 1
    return complexity

# Use PCA mode sequence with 4 modes
top_vecs = eigvecs[:, order[:4]]
proj = traj @ top_vecs
mode_seq = np.argmax(np.abs(proj), axis=1)

# LZ complexity
lzc = lz_complexity(mode_seq[:2000])
# Normalized: LZ / (n / ln(n))
n = 2000
h_lz = lzc * np.log(n) / n
print(f"  LZ complexity: {lzc}")
print(f"  LZ entropy rate estimate: {h_lz:.4f}")
print(f"  Xi = {XI:.4f}")


# ============================================================
# Method 7: Transition entropy (what's the entropy of the
# next mode given the current mode?)
# ============================================================
print("\n--- Method 7: Transition matrix entropy ---")
for n_modes in [4, 8, 16]:
    top_vecs = eigvecs[:, order[:n_modes]]
    proj = traj @ top_vecs
    mode_seq = np.argmax(np.abs(proj), axis=1)

    # Build transition matrix
    trans = np.zeros((n_modes, n_modes))
    for t in range(len(mode_seq) - 1):
        trans[mode_seq[t], mode_seq[t+1]] += 1

    # Row-normalize
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    # Stationary distribution
    stat = trans.sum(axis=1) / trans.sum()

    # Transition entropy: H(next|current) = sum_i pi_i H(row_i)
    h_trans = 0
    for i in range(n_modes):
        if stat[i] > 1e-10:
            row = trans_prob[i]
            row = row[row > 1e-10]
            h_row = -np.sum(row * np.log(row))
            h_trans += stat[i] * h_row

    print(f"  n_modes={n_modes}: h_transition = {h_trans:.4f}, Xi = {XI:.4f}, "
          f"ratio = {h_trans/XI:.4f}")


# ============================================================
# Summary: which method gives h1 closest to Xi?
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Need to find the right measurement for h1 ~ Xi")
print("=" * 70)
