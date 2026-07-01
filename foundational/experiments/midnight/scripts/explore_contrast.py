"""
EXPLORATORY / post-hoc (context for the next registered prediction) — the FIXED exp_23 observable.

exp_23 (registered f1639e08, INCONCLUSIVE) tested the MEAN-CENTERED per-ion coupling for a
MONOTONE IP-ordering. The diagnostic (explore_inconclusiveness.py) showed: the structure is an
ENDPOINT CONTRAST (flat plateau + opposite-signed IP extremes), the flat middle is a
mean-centering shadow, and the real signal is the HIGH-IP vs LOW-IP coupling contrast vs local
cascade phase, measured CENTERING-FREE.

The fixed observable (principled, non-circular pivot = the Hartree reservoir quantum E_H):
  HIGH = ions with IP > E_H (27.2 eV)     -- above the reservoir quantum (active)
  LOW  = ions with IP < E_H               -- below (settled)
  per scope:  contrast = mean(logN_HIGH) - mean(logN_LOW)   [within-frame difference: centering-free]
  test: does the contrast grow with local cascade phase diseq?

Balance reading: the contrast is the high/low IP imbalance; the prediction is that cascade
transitions DRIVE that imbalance (the middle is the balance pivot, the extremes swing apart).

NOT registered. Post-hoc context to design the next pre-registered test (DESI / new data).
"""

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from exp_23_joint_coupling import load_xqr30_scopes
from exp_19_ionization_coupling import ION_IP, n_at_z

E_H = 27.2
def diseq(N): return max(0.0, 1.0 - 2.0 * abs(N - round(N)))
rng = np.random.RandomState(20260614)
scopes = load_xqr30_scopes()


def boot_slope(D, Y, nb=5000):
    s = float(np.polyfit(D, Y, 1)[0])
    bs = []
    for _ in range(nb):
        idx = rng.randint(0, len(D), len(D))
        if len(np.unique(np.round(D[idx], 3))) >= 2:
            bs.append(np.polyfit(D[idx], Y[idx], 1)[0])
    return s, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def contrast(exclude=()):
    D, Y = [], []
    for d in scopes.values():
        hi = [v for i, v in d['ions'].items() if ION_IP[i] > E_H and i not in exclude]
        lo = [v for i, v in d['ions'].items() if ION_IP[i] < E_H and i not in exclude]
        if hi and lo:
            D.append(diseq(n_at_z(d['z']))); Y.append(np.mean(hi) - np.mean(lo))
    return np.array(D), np.array(Y)


print("=" * 68)
print("FIXED observable: HIGH(IP>E_H) - LOW(IP<E_H) logN contrast vs cascade phase")
print("=" * 68)

for label, excl in [("all ions", ()), ("excl. MgII (confirmed anomaly)", ('MgII',))]:
    D, Y = contrast(exclude=excl)
    s, a, b = boot_slope(D, Y)
    rho, p = spearmanr(D, Y)
    # transition-vs-trough (avoids slope extrapolation; tapestry-style 2-sample)
    hiD, loD = Y[D > 0.3], Y[D <= 0.3]
    U, pu = mannwhitneyu(hiD, loD, alternative='greater')
    print(f"\n[{label}]  n={len(D)} scopes")
    print(f"  slope vs diseq = {s:+.3f}  CI95=[{a:+.3f},{b:+.3f}]"
          f"{'  ** excludes 0' if a > 0 or b < 0 else ''}")
    print(f"  Spearman(diseq, contrast) rho={rho:+.3f}  p={p:.4f}"
          f"{'  ** p<0.05' if p < 0.05 else ''}")
    print(f"  toward-transition (diseq>0.3, n={len(hiD)}) median={np.median(hiD):+.2f}  "
          f"vs toward-trough (n={len(loD)}) median={np.median(loD):+.2f}  "
          f"MannWhitney(greater) p={pu:.4f}{'  ** p<0.05' if pu < 0.05 else ''}")

# pairwise: which HIGH-LOW pairs carry it (centering-free)
print("\nPairwise high-low contrasts vs diseq (slope, CI95):")
HIGH = [i for i in ION_IP if ION_IP[i] > E_H]
LOW = [i for i in ION_IP if ION_IP[i] < E_H]
rows = []
for h in HIGH:
    for l in LOW:
        D, Y = [], []
        for d in scopes.values():
            if h in d['ions'] and l in d['ions']:
                D.append(diseq(n_at_z(d['z']))); Y.append(d['ions'][h] - d['ions'][l])
        D, Y = np.array(D), np.array(Y)
        if len(D) >= 8 and len(np.unique(np.round(D, 3))) >= 4:
            s, a, b = boot_slope(D, Y)
            rows.append((h, l, len(D), s, a, b, a > 0 or b < 0))
for h, l, n, s, a, b, sig in sorted(rows, key=lambda r: -r[3]):
    print(f"  {h:>4}-{l:<4} n={n:>2}  slope={s:+.3f}  CI95=[{a:+.3f},{b:+.3f}]"
          f"{'  **' if sig else ''}")

print("\n(Exploratory/post-hoc. The significant centering-free contrast is the observable to")
print(" PRE-REGISTER on new/held-out data — DESI multi-ion absorbers, ideally reaching diseq>0.7.)")
