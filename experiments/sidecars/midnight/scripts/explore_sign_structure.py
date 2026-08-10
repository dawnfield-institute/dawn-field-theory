"""
EXPLORATORY (not pre-registered, not scored) — sign structure of exp_23 couplings.

Follow-up to exp_23 to understand the AlII(-) / CIV(+) sign flip in the within-scope
cascade coupling. NOTE: single-epoch XQR-30 (all scopes N~6.5) can only LOCATE the
sign crossing -> infer N_H; it CANNOT test the phi-ladder spacing (needs multi-epoch
crossings). Results here are hypothesis-generating for a future registered test on
DESI/multi-epoch data.

Three exploratory questions:
  1. Where does signed c_i cross zero in IP, and what N_H does that imply?
  2. Is the AlII(-) vs CIV(+) contrast robust within scopes containing both?
  3. Is MgII's anomalous (+) coupling driven by a few high-leverage scopes?
"""

import numpy as np
from scipy.stats import spearmanr
from exp_23_joint_coupling import (
    load_xqr30_scopes, within_scope_table, couplings,
    ION_IP, PHI, LN_PHI, E_H)

scopes = load_xqr30_scopes()
table, scope_rows = within_scope_table(scopes)
rng = np.random.RandomState(20260614)
coup = couplings(table, rng)
usable = {i: d for i, d in coup.items() if not d['excluded']}
Nbar = float(np.mean([r['N'] for r in scope_rows]))

print("=" * 64)
print("EXPLORATORY: sign structure of exp_23 within-scope couplings")
print(f"  {len(scopes)} scopes, mean epoch Nbar={Nbar:.3f}")
print("=" * 64)

# ---- 1) sign crossing in IP -> implied N_H ----
ions = sorted(usable, key=lambda i: ION_IP[i])
ips = np.array([ION_IP[i] for i in ions])
cs = np.array([usable[i]['c'] for i in ions])
print("\n[1] signed coupling vs IP:")
for i in ions:
    print(f"    {i:<5} IP={ION_IP[i]:6.2f}  c={usable[i]['c']:+.3f}")

def crossing(ip_arr, c_arr):
    sl, ic = np.polyfit(np.log(ip_arr), c_arr, 1)
    ipc = float(np.exp(-ic / sl)) if sl != 0 else np.nan
    return sl, ipc

for label, keep in [("all usable", ions), ("excl. MgII", [i for i in ions if i != 'MgII'])]:
    ip_a = np.array([ION_IP[i] for i in keep])
    c_a = np.array([usable[i]['c'] for i in keep])
    sl, ipc = crossing(ip_a, c_a)
    if 0 < ipc < 1e4:
        N_H = Nbar - np.log(E_H / ipc) / LN_PHI
        print(f"    [{label}] slope(c vs lnIP)={sl:+.3f}, sign-cross IP={ipc:.2f} eV "
              f"-> implied N_H={N_H:.2f}")
    else:
        print(f"    [{label}] slope(c vs lnIP)={sl:+.3f}, no crossing in range")

# ---- 2) AlII vs CIV within-scope contrast ----
print("\n[2] within-scope AlII vs CIV (scopes with BOTH):")
both = []
for sysid, d in scopes.items():
    if 'AlII' in d['ions'] and 'CIV' in d['ions']:
        vals = np.array(list(d['ions'].values()), float)
        mean = float(np.mean(vals))
        x_al = d['ions']['AlII'] - mean
        x_civ = d['ions']['CIV'] - mean
        both.append((x_civ - x_al))
both = np.array(both)
if len(both) >= 3:
    # sign consistency + bootstrap CI on the mean difference
    boots = [np.mean(both[rng.randint(0, len(both), len(both))]) for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"    n={len(both)} scopes; mean (x_CIV - x_AlII)={np.mean(both):+.3f} "
          f"CI95=[{lo:+.3f},{hi:+.3f}]; frac>0={np.mean(both > 0):.2f}")
    print(f"    {'CI excludes 0 -> robust contrast' if lo > 0 or hi < 0 else 'CI includes 0'}")
else:
    print(f"    only {len(both)} scopes with both — too few")

# ---- 3) MgII leverage (jackknife over scopes) ----
print("\n[3] MgII anomaly — jackknife leverage:")
mg = table['MgII']
dq, x = mg['diseq'], mg['x']
c_full = float(np.polyfit(dq, x, 1)[0])
jack = []
for k in range(len(dq)):
    m = np.ones(len(dq), bool); m[k] = False
    if len(np.unique(np.round(dq[m], 3))) >= 2:
        jack.append((k, float(np.polyfit(dq[m], x[m], 1)[0])))
jack_c = np.array([j[1] for j in jack])
worst = sorted(jack, key=lambda j: abs(j[1] - c_full), reverse=True)[:3]
print(f"    full c={c_full:+.3f}; jackknife range=[{jack_c.min():+.3f},{jack_c.max():+.3f}]")
print(f"    3 most influential scopes shift c to: "
      f"{', '.join(f'{w[1]:+.3f}' for w in worst)}")
print(f"    sign stable under jackknife: {bool(np.all(np.sign(jack_c) == np.sign(c_full)))}")

print("\n(Exploratory — informs a future registered multi-epoch test; nothing scored.)")
