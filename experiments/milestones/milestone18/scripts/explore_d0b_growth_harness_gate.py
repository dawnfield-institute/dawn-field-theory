#!/usr/bin/env python3
"""explore_d0b (instrument validation, Block D object O2): core/growth_harness.replay against the engine
`evolve_pac_tree` itself (bit-for-bit metrics for the same seed) and against the RECORDED exp_07/exp_08
results (studies/prime_growth_dynamics_v2/results). Also reports the size distribution of the grown trees,
which fixes the evaluable subset of O2 (exact charpolys are feasible to n ~ 100). No certificate is computed."""
import sys, json, time, glob
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
PGD = ROOT.parent.parent / "studies" / "prime_growth_dynamics_v2"
sys.path.insert(0, str(PGD / "core"))
from growth_harness import replay, gate_against_engine
from phase_engine import evolve_pac_tree
RES = ROOT / "results"

if __name__ == "__main__":
    t0 = time.time(); out = dict(engine_gate=[], record_gate=[], sizes={})
    rec07 = json.load(open(sorted(glob.glob(str(PGD / "results" / "exp_07*.json")))[-1]))
    rec08 = json.load(open(sorted(glob.glob(str(PGD / "results" / "exp_08*.json")))[-1]))
    triples = []
    for mc in rec07["config"]["max_children_range"]:
        triples.append(("exp_07", int(rec07["config"]["depth_limit"]), int(mc), int(rec07["config"]["n_iterations"]), int(mc) * 100, float(rec07["results_by_mc"][str(mc)]["stability_score"]) if str(mc) in rec07["results_by_mc"] else None))
    for dl in rec08["config"]["depth_range"]:
        r = rec08["results_by_depth"].get(str(dl))
        triples.append(("exp_08", int(dl), int(rec08["config"]["fixed_max_children"]), int(rec08["config"]["n_iterations"]), int(dl) * 100, float(r["stability_score"]) if r else None))
    for mc in (2, 3, 4, 5):
        for dl in (1, 2, 3, 4):
            g = rec08.get("grid_results", {}).get(f"mc{mc}_d{dl}")
            triples.append(("exp_08_grid", dl, mc, 50, mc * 100 + dl, float(g) if g is not None else None))
    all_ok = True; sizes = []
    for src, dl, mc, it, seed, recorded in triples:
        ok, cmp, trees = gate_against_engine(1.0, dl, mc, it, seed, evolve_pac_tree)
        rec_ok = None if recorded is None else (abs(cmp["stability_score"][0] - recorded) < 1e-12)
        ns = sorted(t_["n"] for t_ in trees); sizes.append((src, dl, mc, seed, min(ns), max(ns), sum(1 for x in ns if x <= 100)))
        out["engine_gate"].append(dict(src=src, depth=dl, mc=mc, seed=seed, ok=ok, cmp={k: v for k, v in cmp.items()}))
        out["record_gate"].append(dict(src=src, depth=dl, mc=mc, seed=seed, recorded=recorded, replay=cmp["stability_score"][0], ok=rec_ok))
        all_ok &= ok and (rec_ok in (True, None))
        print(f"{src} depth={dl} mc={mc} seed={seed}: engine bit-exact={ok} recorded={recorded} replay={cmp['stability_score'][0]:.12f} rec_ok={rec_ok} | trees n in [{min(ns)},{max(ns)}], n<=100: {sum(1 for x in ns if x<=100)}/{len(ns)} [{time.time()-t0:.0f}s]", flush=True)
    out["sizes"] = sizes; out["ALL"] = all_ok
    json.dump(out, open(RES / f"explore_d0b_growth_harness_gate_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    print("GROWTH HARNESS GATE:", "PASS" if all_ok else "FAIL", f"[{time.time()-t0:.0f}s]"); print("DONE")
