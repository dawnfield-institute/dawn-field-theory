#!/usr/bin/env python3
"""explore_r23b (instrument validation): census.one5_partners_fast (integer edge expansion) against
foldlaws.one5_diagrams (symbolic) at k <= 8: identical key sets and, per key, identical q up to
conjugation; and the k = 10 target count against exp_15's sealed 610."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
import sympy as sp
from census import one5_partners_fast, s5, t
from foldlaws import one5_diagrams
if __name__ == "__main__":
    t0 = time.time(); slow = one5_diagrams(16); print(f"symbolic map k<=8: {len(slow)} keys [{time.time()-t0:.0f}s]")
    t0 = time.time(); fast = {}
    for k in range(2, 9):
        for key, v in one5_partners_fast(k).items(): fast.setdefault(key, []).extend(v)
    print(f"fast map k<=8: {len(fast)} keys [{time.time()-t0:.0f}s]")
    skeys = {str(sp.expand(k)) for k in slow}; fkeys = set(fast)
    same_keys = skeys == fkeys
    qmatch = 0
    for key, (q, M) in slow.items():
        qs = fast[str(sp.expand(key))]
        if any(sp.expand(q - q2) == 0 or sp.expand(q.subs(s5, -s5) - q2) == 0 for q2, _, _ in qs): qmatch += 1
    print(f"same key set: {same_keys}; q matches (up to conjugation): {qmatch}/{len(slow)}")
    t0 = time.time(); k10 = one5_partners_fast(10); print(f"k=10 targets: {len(k10)} (exp_15 sealed: 610) [{time.time()-t0:.0f}s]")
    t0 = time.time(); k11 = one5_partners_fast(11); k12 = one5_partners_fast(12)
    print(f"k=11 targets: {len(k11)}; k=12 targets: {len(k12)} [{time.time()-t0:.0f}s]")
    ok = same_keys and qmatch == len(slow) and len(k10) == 610
    json.dump(dict(same_keys=same_keys, qmatch=qmatch, slow=len(slow), k10=len(k10), k11=len(k11), k12=len(k12), gate="PASS" if ok else "FAIL"),
              open(Path(__file__).parent.parent / "results" / f"explore_r23b_partner_map_gate_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)
    print("PARTNER-MAP GATE:", "PASS" if ok else "FAIL"); print("DONE")
