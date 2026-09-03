#!/usr/bin/env python3
"""explore_r23 (instrument validation): the parallel census path (core/census.py: strict_hunt_parallel)
through the sealed known-answer gate at n = 16 — 15 strict trees on 14 polynomials, and the SAME 15
edge sets as the committed census (explore_r16b_strict_n16.json). A parallel census that fails this
gate is not a census. Run as a file (macOS spawn re-imports __main__; never from stdin)."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
RES = Path(__file__).parent.parent / "results"

if __name__ == "__main__":
    from census import strict_hunt_parallel, known_answer_gate_n16
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    cnt, surv, strict = strict_hunt_parallel(n, workers=workers, block=1000, log=print, progress=5)
    out = dict(n=n, trees=cnt, survivors=len(surv), strict=[(e, str(p)) for e, p in strict], workers=workers,
               seconds=round(time.time() - t0, 1))
    if n == 16:
        ok, counts = known_answer_gate_n16(strict)
        old = json.load(open(RES / "explore_r16b_strict_n16.json"))
        key = lambda E: tuple(sorted(tuple(sorted(x)) for x in E))
        same = {key(r["edges"]) for r in old} == {key(e) for e, _ in strict}
        out.update(ka16_gate="PASS" if ok else "FAIL", ka16_counts=counts, same_edge_sets_as_committed=same)
        print(f"KA16 gate through the PARALLEL path: {out['ka16_gate']} {counts}; same 15 edge sets: {same} [{out['seconds']}s]")
    json.dump(out, open(RES / f"explore_r23_parallel_census_n{n}_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)
    print("DONE")
