"""
EXPLORATORY diagnostic — WHY is exp_23 inconclusive?

Distinguishes three possibilities:
  (a) underpowered (real signal, too few ions/scopes/diseq-leverage)
  (b) wrong model  (structure isn't the monotone IP-ordering R1 tests)
  (c) signal carried by a few ions (endpoints), middle consistent with zero

Diagnostics:
  1. per-ion significance (which CIs exclude zero?)
  2. diseq leverage (how far from a transition do we ever get?)
  3. jackknife the two significant ions (AlII, CIV) — robust or fragile?
  4. leave-one-ion-out Spearman (which ion is the trend leaning on?)
  5. linear-vs-flat structure (is the middle a plateau?)
  6. power: how many ions would R1 need at the observed effect size?
"""

import numpy as np
from scipy.stats import spearmanr, t as tdist
from exp_23_joint_coupling import (
    load_xqr30_scopes, within_scope_table, couplings, ION_IP, LN_PHI)

scopes = load_xqr30_scopes()
table, scope_rows = within_scope_table(scopes)
rng = np.random.RandomState(20260613)   # same seed as exp_23
coup = couplings(table, rng)
usable = [i for i in sorted(coup, key=lambda x: ION_IP[x]) if not coup[i]['excluded']]

print("=" * 66)
print("DIAGNOSTIC: why is exp_23 inconclusive?")
print("=" * 66)

# ---- 1) per-ion significance ----
print("\n[1] per-ion coupling significance:")
nsig = 0
for i in usable:
    lo, hi = coup[i]['c_CI95']
    excl = lo > 0 or hi < 0
    nsig += excl
    print(f"    {i:<5} IP={ION_IP[i]:6.2f}  c={coup[i]['c']:+.2f}  "
          f"CI95=[{lo:+.2f},{hi:+.2f}]  n={coup[i]['n']:>2}  "
          f"{'** excludes 0' if excl else '   (incl 0)'}")
print(f"    -> {nsig}/{len(usable)} ions have CI excluding zero")

# ---- 2) diseq leverage ----
dq = np.array([r['diseq'] for r in scope_rows])
print("\n[2] diseq leverage across 32 scopes (transition=1, trough=0):")
print(f"    min={dq.min():.2f} median={np.median(dq):.2f} max={dq.max():.2f}; "
      f"scopes diseq>0.3: {int(np.sum(dq > 0.3))}, >0.4: {int(np.sum(dq > 0.4))}")
print(f"    (tapestry used diseq>0.7 vs <0.3 — we never reach a real transition)")

# ---- 3) jackknife the two significant ions ----
print("\n[3] jackknife AlII, CIV (drop one scope at a time):")
for ion in ('AlII', 'CIV'):
    d = table[ion]; dqi, x = d['diseq'], d['x']
    c_full = float(np.polyfit(dqi, x, 1)[0])
    jc = [float(np.polyfit(np.delete(dqi, k), np.delete(x, k), 1)[0])
          for k in range(len(dqi)) if len(np.unique(np.round(np.delete(dqi, k), 3))) >= 2]
    jc = np.array(jc)
    print(f"    {ion:<5} c={c_full:+.2f}  jackknife range=[{jc.min():+.2f},{jc.max():+.2f}]  "
          f"sign-stable={bool(np.all(np.sign(jc) == np.sign(c_full)))}")

# ---- 4) leave-one-ion-out Spearman (which ion carries the trend?) ----
print("\n[4] leave-one-ion-out Spearman(IP, c):")
ips_all = [ION_IP[i] for i in usable]; cs_all = [coup[i]['c'] for i in usable]
rho0, p20 = spearmanr(ips_all, cs_all)
print(f"    all {len(usable)} ions: rho={rho0:+.3f}, p_one={p20/2 if rho0>0 else 1-p20/2:.3f}")
for drop in usable:
    keep = [i for i in usable if i != drop]
    rho, p2 = spearmanr([ION_IP[i] for i in keep], [coup[i]['c'] for i in keep])
    p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
    print(f"    drop {drop:<5}: rho={rho:+.3f}, p_one={p1:.3f}")

# ---- 5) linear vs flat-middle structure ----
print("\n[5] structure (c vs ln IP, MgII excluded as known anomaly):")
keep = [i for i in usable if i != 'MgII']
lnip = np.log([ION_IP[i] for i in keep]); cs = np.array([coup[i]['c'] for i in keep])
sl, ic = np.polyfit(lnip, cs, 1)
resid = cs - (sl * lnip + ic)
mid = [i for i in keep if 8 <= ION_IP[i] <= 34]
print(f"    linear slope={sl:+.3f}; endpoints AlII={coup['AlII']['c']:+.2f}, CIV={coup['CIV']['c']:+.2f}")
print(f"    middle plateau (8-34 eV: {','.join(mid)}): "
      f"mean c={np.mean([coup[i]['c'] for i in mid]):+.2f}, "
      f"spread={np.std([coup[i]['c'] for i in mid]):.2f}")

# ---- 6) power: ions needed at observed effect ----
print("\n[6] power for R1 (Spearman) at observed rho:")
def spearman_p_one(rho, n):
    if n < 3 or abs(rho) >= 1:
        return np.nan
    tval = rho * np.sqrt((n - 2) / (1 - rho**2))
    return 1 - tdist.cdf(tval, n - 2)
for n in (7, 9, 11, 13, 15):
    print(f"    rho={rho0:+.2f}, n={n:>2} ions -> p_one={spearman_p_one(rho0, n):.3f}"
          f"{'  <-crosses 0.05' if spearman_p_one(rho0, n) < 0.05 else ''}")

print("\n(Exploratory — diagnostic only, nothing registered.)")
