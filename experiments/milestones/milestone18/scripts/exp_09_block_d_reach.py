#!/usr/bin/env python3
"""exp_09: Block D (SEALED by the commit carrying journals/2026-09-03_blockD_registration.md) — physical reach
of the fold certificate. O1 complete PAC trees d = 6, 7, 8 by the sector route (core/pac_sectors.py); O2 the 166
evaluable growth trees and O3 the 4 unicyclic controls by the Bezout route (certificate.sector_projector_dm),
every Galois half, some-partner carrier semantics; T2a gate (sector classes at d = 6-8); T4 (E8 KA1 + Block A's
exp_02 rerun). Results append-only, timestamped. Usage: python exp_09_block_d_reach.py [workers]. Run as a file."""
import sys, json, time, subprocess
from pathlib import Path
from multiprocessing import Pool
import numpy as np, sympy as sp, networkx as nx
HERE = Path(__file__).parent; ROOT = HERE.parent; EXP = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "core")); sys.path.insert(0, str(EXP / "studies" / "prime_growth_dynamics_v2" / "core")); sys.path.insert(0, str(EXP / "milestones" / "milestone15" / "core"))
from certificate import grade, halves, sector_projector_dm, certificate, class_sectors, golden_pairs, is_regular, t, s5, sig
from pac_sectors import pac_certificate, sectors, path_cartan
from growth_harness import replay
from ledger import cart, EDGES, H4, DegenerateFoldError
RES = ROOT / "results"
TWO_SQ5 = 2 * s5 / 5

def is_carrier(c):
    return sp.simplify(sp.sympify(c["trRD"]) - TWO_SQ5) == 0 and sp.sympify(c["leak_oo"]) == sp.Rational(2, 5)

def worker(job):
    label, n, e = job
    try:
        C = cart(n, [tuple(x) for x in e]); p = sp.expand(C.charpoly(t).as_expr())
        G = nx.Graph([tuple(x) for x in e]); G.add_nodes_from(range(n)); D = sp.diag(*[G.degree(v) for v in range(n)])
        rec = dict(label=label, n=n, grade=grade(p)[0], halves=[])
        if is_regular(D): rec["blind"] = True; return rec
        for h in halves(p):
            try:
                P = sector_projector_dm(C, h, p); c = certificate(C, D, P); c.pop("R")
                c = {k: (str(v) if not isinstance(v, list) else v) for k, v in c.items()}; c["carrier"] = is_carrier(c); c["q"] = str(h)
            except DegenerateFoldError as ex:
                c = dict(q=str(h), declared=str(ex), carrier=False)
            rec["halves"].append(c)
        rec["carrier"] = any(h.get("carrier") for h in rec["halves"]); return rec
    except Exception as ex:
        return dict(label=label, n=n, error=repr(ex), carrier=False)

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8; ts = time.strftime("%Y%m%d_%H%M%S"); t0 = time.time()
    log = open(RES / f"exp_09_block_d_reach_{ts}_log.txt", "w")
    def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    out = dict(registration="Block D (journals/2026-09-03_blockD_registration.md)", O1={}, O2=[], O3=[], T2a={}, T4={})
    # ---- O1: sector route at d = 6, 7, 8; pairs from the sector charpolys; certificate per pair and per combined half
    for d in (6, 7, 8):
        pairs = {}
        for k, m, l, degs in sectors(d):
            for g, sg, e_ in golden_pairs(sp.expand(path_cartan(k).charpoly(t).as_expr())):
                key = str(sp.expand(g)); pairs.setdefault(key, dict(g=g, sg=sg, cls=None))
                cl = [(c, str(s)) for g2, c, s in class_sectors(sp.expand(path_cartan(k).charpoly(t).as_expr())) if str(sp.expand(g2)) == key or str(sp.expand(sig(g2))) == key]
                if cl: pairs[key]["cls"] = cl[0]
        out["T2a"][d] = {k: v["cls"] for k, v in pairs.items()}
        cells = {}
        plist = list(pairs.values())
        for pr in plist:                                  # per class: the pair alone (both members)
            for q in (pr["g"], pr["sg"]):
                if q is None: continue
                c = pac_certificate(d, sp.expand(q)); c = {k: (str(v) if not isinstance(v, (list, dict)) else v) for k, v in c.items()}
                c["carrier"] = is_carrier(c); cells.setdefault(f"{pr['cls']}", []).append(dict(q=str(q), **c))
        if len(plist) > 1:                                # combined halves (some-partner semantics for T1)
            import itertools
            for choice in itertools.product([0, 1], repeat=len(plist)):
                q = sp.Integer(1)
                for cix, pr in zip(choice, plist): q *= (pr["g"] if cix == 0 else pr["sg"])
                c = pac_certificate(d, sp.expand(q)); c = {k: (str(v) if not isinstance(v, (list, dict)) else v) for k, v in c.items()}
                c["carrier"] = is_carrier(c); cells.setdefault("combined", []).append(dict(q=str(q), **c))
        out["O1"][d] = dict(n=2 ** (d + 1) - 1, cells=cells, carrier=any(c["carrier"] for v in cells.values() for c in v))
        L(f"O1 d={d}: classes {out['T2a'][d]}; cells { {k: [(c['trRD'], c['leak_oo'], c['carrier']) for c in v] for k, v in cells.items()} } [{time.time()-t0:.0f}s]")
    # ---- O2: the evaluable growth trees (replayed; n <= 100 and golden content), O3: the 4 unicyclics
    jobs = []
    triples = [(5, mc, 100, mc * 100, "exp_07") for mc in (2, 3, 4, 5, 6, 7, 8)] + [(dl, 3, 100, dl * 100, "exp_08") for dl in (1, 2, 3, 4, 5, 6, 8, 10)] + [(dl, mc, 50, mc * 100 + dl, "exp_08_grid") for mc in (2, 3, 4, 5) for dl in (1, 2, 3, 4)]
    seen = set()
    for dl, mc, it, seed, src in triples:
        _, trees = replay(1.0, dl, mc, n_iterations=it, seed=seed)
        for i, tr in enumerate(trees):
            key = (tr["n"], tuple(sorted(tr["edges"])))
            if key in seen or tr["n"] > 100 or tr["n"] < 4: continue
            seen.add(key); nodes = sorted({v for e_ in tr["edges"] for v in e_}); idx = {v: k for k, v in enumerate(nodes)}
            e = [(idx[a], idx[b]) for a, b in tr["edges"]]; n = len(nodes)
            if grade(sp.expand(cart(n, e).charpoly(t).as_expr()))[0] == "none": continue
            jobs.append((f"O2 {src} d{dl} mc{mc} s{seed} #{i}", n, e))
    L(f"O2 evaluable jobs: {len(jobs)} (registration: 166)")
    from representative import random_unicyclic, cycle_basis_single
    rng = np.random.RandomState(152); o3jobs = []
    for m in (7, 9, 11):
        for k in range(20):
            g = random_unicyclic(m, rng)
            try:
                if len(cycle_basis_single(g)) < 3: continue
            except Exception: continue
            n_ = g.shape[0]; e_ = [(i, j) for i in range(n_) for j in range(i + 1, n_) if g[i, j] != 0]
            if grade(sp.expand(cart(n_, e_).charpoly(t).as_expr()))[0] != "none": o3jobs.append((f"O3 unicyclic m={m} #{k}", n_, e_))
    L(f"O3 evaluable jobs: {len(o3jobs)} (registration: 4)")
    with Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(worker, o3jobs + sorted(jobs, key=lambda j: j[1]), chunksize=1), 1):
            (out["O3"] if rec["label"].startswith("O3") else out["O2"]).append(rec)
            L(f"  [{i}/{len(jobs)+len(o3jobs)}] {rec['label']} n={rec['n']} grade={rec.get('grade')} halves={len(rec.get('halves', []))} carrier={rec.get('carrier')} {('ERROR '+rec['error']) if 'error' in rec else ''} [{time.time()-t0:.0f}s]")
    # ---- T4: E8 KA1 rerun + Block A exp_02 rerun
    n8, e8 = EDGES["E8"]; C8 = cart(n8, e8); p8 = sp.expand(C8.charpoly(t).as_expr()); q8 = sp.expand(H4.charpoly(t).as_expr())
    G8 = nx.Graph(e8); D8 = sp.diag(*[G8.degree(v) for v in range(n8)]); c8 = certificate(C8, D8, sector_projector_dm(C8, q8, p8)); c8.pop("R")
    t4i = grade(p8)[0] == "strict" and is_carrier({k: str(v) for k, v in c8.items() if k != "vertex_sq"})
    r = subprocess.run([sys.executable, str(HERE / "exp_02_projection_carries_phi.py")], capture_output=True, text=True, timeout=1800, cwd=str(HERE))
    t4ii = (r.returncode == 0) and ("PASS" in r.stdout)
    out["T4"] = dict(i_E8_strict_carrier=t4i, ii_exp02_rerun=t4ii, exp02_tail=r.stdout[-1500:], exp02_rc=r.returncode)
    # ---- scoring to the sealed text
    live_carriers = [d for d, v in out["O1"].items() if v["carrier"]] + [x["label"] for x in out["O2"] + out["O3"] if x.get("carrier")]
    errors = [x["label"] for x in out["O2"] + out["O3"] if "error" in x]
    t2b_cells = {f"d{d}:{cls}": any(c["carrier"] for c in v) for d, o in out["O1"].items() for cls, v in o["cells"].items() if cls != "combined"}
    out["tests"] = dict(
        T1=dict(live_objects=len(out["O1"]) + len(out["O2"]) + len(out["O3"]), carriers=live_carriers, ok=len(live_carriers) >= 1),
        T2a=out["T2a"], T2b=dict(cells=t2b_cells, ok=not any(t2b_cells.values())),
        T4=dict(ok=t4i and t4ii, i=t4i, ii=t4ii), errors=errors, seconds=round(time.time() - t0, 1))
    json.dump(out, open(RES / f"exp_09_block_d_reach_{ts}.json", "w"), indent=1, default=str)
    L("TESTS: " + json.dumps(out["tests"], default=str)); L("SCORE DONE")
