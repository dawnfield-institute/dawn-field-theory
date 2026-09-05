#!/usr/bin/env python3
"""explore_d0c (instrument validation, Block D draft §2.7): the sector-route certificate on complete PAC trees
must equal the full-tree Bezout-route certificate EXACTLY at d = 3, 4, 5 for the H2 pair (tr(R·D), leak_oo,
leak_total, vertex multiset). These depths are Block D objects, but this is the instrument gate the draft
requires before the seal; the values are recorded here as gate values, not scored."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
import sympy as sp
from certificate import halves, sector_projector, certificate, t, s5
from pac_sectors import pac_certificate
from ledger import cart
import networkx as nx
RES = Path(__file__).parent.parent / "results"

def pac_edges(d):
    nn = 2 ** (d + 1) - 1
    return nn, [(i, 2*i+1) for i in range(nn) if 2*i+1 < nn] + [(i, 2*i+2) for i in range(nn) if 2*i+2 < nn]

if __name__ == "__main__":
    t0 = time.time(); out = {}; allok = True
    for d in (3, 4, 5):
        n, e = pac_edges(d); C = cart(n, e); p = sp.expand(C.charpoly(t).as_expr()); q = halves(p)[0]
        G = nx.Graph(e); D = sp.diag(*[G.degree(v) for v in range(n)])
        P = sector_projector(C, q, p); full = certificate(C, D, P); full.pop("R")
        sect = pac_certificate(d, q)
        same = all(sp.simplify(sp.sympify(full[k]) - sp.sympify(sect[k])) == 0 for k in ("trRD", "leak_oo", "leak_total")) and full["vertex_sq"] == sect["vertex_sq"]
        allok &= same
        out[d] = dict(ok=same, q=str(q), bezout={k: str(full[k]) for k in ("trRD", "leak_oo", "leak_total", "vertex_sq")}, sector={k: str(sect[k]) for k in ("trRD", "leak_oo", "leak_total", "vertex_sq")})
        print(f"d={d} n={n}: sector == bezout: {same} | bezout {out[d]['bezout']} | sector {out[d]['sector']} [{time.time()-t0:.0f}s]", flush=True)
    out["ALL"] = allok
    json.dump(out, open(RES / f"explore_d0c_pac_sector_certificate_gate_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    print("PAC SECTOR-CERTIFICATE GATE:", "PASS" if allok else "FAIL"); print("DONE")
