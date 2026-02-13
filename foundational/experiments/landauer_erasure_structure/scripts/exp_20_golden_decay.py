"""
Experiment 20: Golden Ratio in Decay Structure
================================================
Dawn Field Institute â€” PAC Exploration Series

FINDING FROM EXP 19: At perfect coupling (c=1.0), the ratio A/(A+Î¾)
is closest to ln(Ï†) when flip_decay / corr_decay = Ï† = 1.618.

This experiment investigates:
  1. Fine-resolution sweep of decay ratios around Ï†
  2. Is the minimum error truly AT Ï† or just near it?
  3. Does this hold at multiple coupling strengths?
  4. 2D parameter sweep: coupling Ã— decay_ratio
  5. High-precision measurement at optimal parameters
"""

import numpy as np
from scipy import stats, optimize
import json, os, time
from datetime import datetime

k_B = 1.380649e-23; T = 300.0
LN_PHI = np.log((1 + np.sqrt(5)) / 2)  # 0.48121182505960344
PHI = (1 + np.sqrt(5)) / 2

def H1d(d):
    v, c = np.unique(d, return_counts=True); p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-30))

def Hh(d, n):
    h = np.zeros(d.shape[0], dtype=np.int64)
    for j in range(n): h += d[:, j].astype(np.int64) * (2 ** j)
    v, c = np.unique(h, return_counts=True); p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-30))

def TC(e, nm):
    return max(0, sum(H1d(e[:, j]) for j in range(nm)) - Hh(e, nm))

def PMI(e, nm):
    t = 0.0
    for i in range(nm):
        for j in range(i + 1, nm):
            jj = e[:, i] * 2 + e[:, j]
            _, c = np.unique(jj, return_counts=True); pjt = c / c.sum()
            pi = np.array([np.mean(e[:, i] == 0), np.mean(e[:, i] == 1)])
            pj = np.array([np.mean(e[:, j] == 0), np.mean(e[:, j] == 1)])
            t += max(0, -np.sum(pi * np.log2(pi + 1e-30)) - np.sum(pj * np.log2(pj + 1e-30))
                     + np.sum(pjt * np.log2(pjt + 1e-30)))
    return t

def IT(sp, ep, nm):
    h = np.zeros(len(sp), dtype=np.int64)
    for j in range(nm): h += ep[:, j].astype(np.int64) * (2 ** j)
    Hs = H1d(sp); He = H1d(h)
    jt = sp.astype(np.int64) * (2 ** 20) + h
    v, c = np.unique(jt, return_counts=True)
    Hse = -np.sum((c / c.sum()) * np.log2(c / c.sum() + 1e-30))
    return max(0, Hs + He - Hse)

def run(seed, ns=300000, bc=1.0, fd=0.3, cb=0.3, cd=0.2, ne=20):
    rng = np.random.RandomState(seed)
    ee = k_B * T * (0.5 + rng.exponential(1.0, ne))
    epr = 1.0 / (1.0 + np.exp(ee / (k_B * T)))
    sys_arr = rng.randint(0, 2, ns)
    env = np.zeros((ns, ne), dtype=int)
    for j in range(ne): env[:, j] = (rng.random(ns) < epr[j]).astype(int)
    tc_pre = TC(env, min(ne, 12)); pmi_pre = PMI(env, ne)
    epost = env.copy(); w1 = (sys_arr == 1); nc = min(5, ne)
    for j in range(nc):
        cc = bc * np.exp(-fd * j); fm = w1 & (rng.random(ns) < cc)
        epost[fm, j] = 1 - epost[fm, j]
    for j in range(1, nc):
        cm = w1 & (rng.random(ns) < cb * np.exp(-cd * j))
        epost[cm, j] = epost[cm, 0]
    tc_post = TC(epost, min(ne, 12)); pmi_post = PMI(epost, ne)
    A = IT(sys_arr, epost, nc)
    xi = (tc_post - tc_pre) + (pmi_post - pmi_pre)
    P = H1d(sys_arr)
    theta = P - (A + xi)
    ratio = A / (A + xi) if (A + xi) > 1e-10 else float('nan')
    return {'P': float(P), 'A': float(A), 'xi': float(xi), 
            'theta': float(theta), 'ratio': float(ratio)}


def multi_seed_run(n_seeds, **kwargs):
    """Run n_seeds experiments and return summary stats."""
    rats = []; thts = []
    for s in range(n_seeds):
        r = run(s, **kwargs)
        if not np.isnan(r['ratio']):
            rats.append(r['ratio'])
        thts.append(r['theta'])
    mr = float(np.mean(rats)); sr = float(np.std(rats))
    return {
        'mean_ratio': mr, 'std': sr, 'n': len(rats),
        'se': float(sr / len(rats)**0.5),
        'ci_lo': float(mr - 1.96 * sr / len(rats)**0.5),
        'ci_hi': float(mr + 1.96 * sr / len(rats)**0.5),
        'mean_theta': float(np.mean(thts)),
        'pct_from_ln_phi': float(abs(mr - LN_PHI) / LN_PHI * 100),
        'ln_phi_in_ci': bool(mr - 1.96 * sr / len(rats)**0.5 <= LN_PHI <= mr + 1.96 * sr / len(rats)**0.5),
    }


print("=" * 70)
print("EXP 20: Golden Ratio in Decay Structure")
print("=" * 70)
print(f"phi = {PHI:.10f}")
print(f"ln(phi) = {LN_PHI:.10f}")
print()

results = {'ln_phi': float(LN_PHI), 'phi': float(PHI), 'tests': {}}

# ============================================================
# TEST 1: Fine decay-ratio sweep around Ï† at coupling=1.0
# ============================================================
print("TEST 1: Fine decay-ratio sweep around Ï† (coupling=1.0)")
print("  corr_decay=0.2 fixed, varying flip_decay = ratio Ã— 0.2")
print("  30 seeds Ã— 300k each")
print()

# Coarse + fine near Ï†
decay_ratios = sorted(set([
    0.5, 0.75, 1.0, 1.25,
    1.4, 1.45, 1.5, 1.55, 1.6,
    1.61, 1.618, 1.625, 1.63, 1.65,
    1.7, 1.75, 1.8, 1.9, 2.0, 2.25, 2.5
]))

test1 = {}
for dr in decay_ratios:
    fd = 0.2 * dr
    t0 = time.time()
    r = multi_seed_run(30, bc=1.0, fd=fd, cd=0.2, ns=300000)
    r['decay_ratio'] = dr
    r['flip_decay'] = float(fd)
    test1[f"{dr:.4f}"] = r
    print(f"  ratio={dr:.4f}  fd={fd:.4f}  A/(A+Î¾)={r['mean_ratio']:.6f} Â± {r['std']:.4f}"
          f"  Î˜={r['mean_theta']:.4f}  dev={r['pct_from_ln_phi']:.3f}%  ({time.time()-t0:.0f}s)", flush=True)

results['tests']['test_1_fine_decay_sweep'] = test1

# Find minimum deviation
min_dr = min(test1.keys(), key=lambda k: test1[k]['pct_from_ln_phi'])
print(f"\n  => Minimum deviation: ratio={min_dr}, dev={test1[min_dr]['pct_from_ln_phi']:.4f}%")
print(f"  => Ï† = {PHI:.4f}, closest tested = {float(min_dr):.4f}")

# ============================================================
# TEST 2: Same sweep at coupling = 0.80 (default) 
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Decay-ratio sweep at coupling=0.80 (default)")
print("  20 seeds Ã— 300k each")
print()

# Subset of ratios
decay_ratios_2 = [0.5, 1.0, 1.25, 1.5, 1.618, 1.75, 2.0, 2.5]
test2 = {}
for dr in decay_ratios_2:
    fd = 0.2 * dr
    t0 = time.time()
    r = multi_seed_run(20, bc=0.80, fd=fd, cd=0.2, ns=300000)
    r['decay_ratio'] = dr
    r['flip_decay'] = float(fd)
    test2[f"{dr:.4f}"] = r
    print(f"  ratio={dr:.4f}  A/(A+Î¾)={r['mean_ratio']:.6f} Â± {r['std']:.4f}"
          f"  Î˜={r['mean_theta']:.4f}  dev={r['pct_from_ln_phi']:.3f}%  ({time.time()-t0:.0f}s)", flush=True)

results['tests']['test_2_default_coupling_decay_sweep'] = test2

# ============================================================
# TEST 3: 2D sweep â€” coupling Ã— decay_ratio
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: 2D sweep (coupling Ã— decay_ratio)")
print("  15 seeds Ã— 200k each â€” lighter for 2D exploration")
print()

couplings = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
dratios = [1.0, 1.25, 1.5, 1.618, 1.75, 2.0]

test3 = {}
for bc in couplings:
    for dr in dratios:
        fd = 0.2 * dr
        r = multi_seed_run(15, bc=bc, fd=fd, cd=0.2, ns=200000)
        r['coupling'] = bc; r['decay_ratio'] = dr; r['flip_decay'] = float(fd)
        key = f"c{bc:.2f}_dr{dr:.3f}"
        test3[key] = r
    # Print row
    row_data = [(dr, test3[f"c{bc:.2f}_dr{dr:.3f}"]) for dr in dratios]
    best = min(row_data, key=lambda x: x[1]['pct_from_ln_phi'])
    print(f"  c={bc:.2f}:  " + "  ".join(f"dr={dr:.3f}:{d['pct_from_ln_phi']:.2f}%" for dr, d in row_data)
          + f"  [best: dr={best[0]:.3f}]", flush=True)

results['tests']['test_3_2d_sweep'] = test3

# ============================================================
# TEST 4: High-precision at the sweet spot
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: High-precision at optimal parameters")
print()

# From Test 3, find optimal (coupling, decay_ratio) pair
best_key = min(test3.keys(), key=lambda k: test3[k]['pct_from_ln_phi'])
best_c = test3[best_key]['coupling']
best_dr = test3[best_key]['decay_ratio']
print(f"  Best from 2D sweep: coupling={best_c}, decay_ratio={best_dr}")
print(f"  Running 50 seeds Ã— 500k...")

t0 = time.time()
best_fd = 0.2 * best_dr
r_best = multi_seed_run(50, bc=best_c, fd=best_fd, cd=0.2, ns=500000)
r_best['coupling'] = best_c; r_best['decay_ratio'] = best_dr
print(f"  Result: A/(A+Î¾) = {r_best['mean_ratio']:.8f} Â± {r_best['std']:.6f}")
print(f"  95% CI: [{r_best['ci_lo']:.8f}, {r_best['ci_hi']:.8f}]")
print(f"  ln(Ï†):  {LN_PHI:.8f}")
print(f"  Dev:    {r_best['pct_from_ln_phi']:.4f}%")
print(f"  Î˜:      {r_best['mean_theta']:.6f}")
print(f"  ln(Ï†) in CI: {r_best['ln_phi_in_ci']}")
print(f"  ({time.time()-t0:.0f}s)")

results['tests']['test_4_high_precision'] = r_best

# Also test exact Ï† decay at coupling=0.90 (exp_19 crossover point)
print(f"\n  Also testing crossover coupling (c=0.90) with Ï† decay:")
t0 = time.time()
r_cross = multi_seed_run(50, bc=0.90, fd=0.2*PHI, cd=0.2, ns=500000)
r_cross['coupling'] = 0.90; r_cross['decay_ratio'] = float(PHI)
print(f"  Result: A/(A+Î¾) = {r_cross['mean_ratio']:.8f} Â± {r_cross['std']:.6f}")
print(f"  95% CI: [{r_cross['ci_lo']:.8f}, {r_cross['ci_hi']:.8f}]")
print(f"  Dev:    {r_cross['pct_from_ln_phi']:.4f}%")
print(f"  Î˜:      {r_cross['mean_theta']:.6f}")
print(f"  ({time.time()-t0:.0f}s)")

results['tests']['test_4_crossover_coupling'] = r_cross

# ============================================================
# TEST 5: Vary corr_decay instead (hold flip_decay=0.3)
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Vary corr_decay (hold flip_decay=0.3)")
print("  Does the ratio matter, or the individual decays?")
print("  20 seeds Ã— 300k")
print()

# If ratio matters, then fd=0.3, cd=0.3/Ï† should work same as fd=Ï†Ã—0.2, cd=0.2
test5 = {}
corr_decays = [0.1, 0.15, 0.185, 0.2, 0.25, 0.3, 0.4]
for cd in corr_decays:
    fd = 0.3  # original default
    ratio = fd / cd
    t0 = time.time()
    r = multi_seed_run(20, bc=1.0, fd=fd, cd=cd, ns=300000)
    r['flip_decay'] = float(fd); r['corr_decay'] = float(cd)
    r['ratio'] = float(ratio)
    test5[f"cd{cd:.3f}"] = r
    print(f"  cd={cd:.3f}  fd/cd={ratio:.4f}  A/(A+Î¾)={r['mean_ratio']:.6f}"
          f"  dev={r['pct_from_ln_phi']:.3f}%  Î˜={r['mean_theta']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

results['tests']['test_5_vary_corr_decay'] = test5

# ============================================================
# SAVE
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       '..', 'results', f'exp_20_golden_decay_{timestamp}.json')
os.makedirs(os.path.dirname(outfile), exist_ok=True)

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

print(f"\n  Test 1: Finest decay ratio = {min_dr}")
print(f"         vs Ï† = {PHI:.4f}")
print(f"         Deviation: {test1[min_dr]['pct_from_ln_phi']:.4f}%")

print(f"\n  Test 3 best 2D point: c={best_c}, dr={best_dr}")
print(f"         Deviation: {test3[best_key]['pct_from_ln_phi']:.4f}%")

print(f"\n  Test 4 high-precision:")
print(f"         A/(A+Î¾) = {r_best['mean_ratio']:.8f}")
print(f"         ln(Ï†)   = {LN_PHI:.8f}")
print(f"         Dev: {r_best['pct_from_ln_phi']:.4f}%")

