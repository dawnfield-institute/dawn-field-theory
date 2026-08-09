"""
exp_01 -- R1 (diagram selectivity) + R3 (mode-count bridge).

Arms: A_6..A_9, D_6..D_8, E_6..E_8. Coupling = exp(-d_G * 0.1) on each
diagram; means vector identical across families at equal rank, so the
coupling matrix is the sole varying quantity.

R1 rule (locked): CONFIRM if at >=2 of ranks {6,7,8} at least one family
pair has non-overlapping 95% CIs AND the family ordering is identical at
every rank where separation exists; KILL if all pairs overlap at all
ranks; else INCONCLUSIVE.
R3 rule: CONFIRM if exponent strictly monotone in rank within each family.

Registration: journals/2026-07-17_ade-cascade-round1-preregistration.md
(commit c5e05712).
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from itertools import combinations

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "core"))

from coupling import dynkin_coupling            # noqa: E402
from runner import run_arm, cis_overlap         # noqa: E402

RESULTS = _HERE.parent / "results"
RANKS = (6, 7, 8)
FAMILIES = ("A", "D", "E")


def main():
    arms = {}
    for fam in FAMILIES:
        for rank in RANKS:
            name = f"{fam}_{rank}"
            print(f"running {name} ...", flush=True)
            arms[name] = run_arm(name, dynkin_coupling(fam, rank, 0.1))
    print("running A_9 ...", flush=True)
    arms["A_9"] = run_arm("A_9", dynkin_coupling("A", 9, 0.1))

    # --- R1 evaluation (locked rule) ---
    r1 = {"per_rank": {}}
    ranks_with_separation = []
    orderings = {}
    for rank in RANKS:
        trio = {f: arms[f"{f}_{rank}"] for f in FAMILIES}
        pairs = {}
        any_sep = False
        for f1, f2 in combinations(FAMILIES, 2):
            sep = not cis_overlap(trio[f1], trio[f2])
            pairs[f"{f1}-{f2}"] = {
                "delta": trio[f1]["mean"] - trio[f2]["mean"],
                "separated": sep}
            any_sep = any_sep or sep
        order = tuple(sorted(FAMILIES, key=lambda f: trio[f]["mean"]))
        orderings[rank] = order
        if any_sep:
            ranks_with_separation.append(rank)
        r1["per_rank"][rank] = {
            "means": {f: trio[f]["mean"] for f in FAMILIES},
            "cis": {f: trio[f]["ci95"] for f in FAMILIES},
            "pairs": pairs,
            "ordering_low_to_high": list(order)}
    sep_orderings = {orderings[r] for r in ranks_with_separation}
    if len(ranks_with_separation) >= 2 and len(sep_orderings) == 1:
        r1["verdict"] = "CONFIRM"
    elif len(ranks_with_separation) == 0:
        r1["verdict"] = "KILL"
    else:
        r1["verdict"] = "INCONCLUSIVE"
    r1["ranks_with_separation"] = ranks_with_separation

    # --- R3 evaluation (locked rule) ---
    r3 = {"per_family": {}}
    all_monotone = True
    fam_ranks = {"A": (6, 7, 8, 9), "D": RANKS, "E": RANKS}
    for fam, rks in fam_ranks.items():
        seq = [arms[f"{fam}_{r}"]["mean"] for r in rks]
        monotone = all(b > a for a, b in zip(seq, seq[1:])) or \
                   all(b < a for a, b in zip(seq, seq[1:]))
        r3["per_family"][fam] = {"ranks": list(rks), "means": seq,
                                 "strictly_monotone": bool(monotone)}
        all_monotone = all_monotone and monotone
    r3["verdict"] = "CONFIRM" if all_monotone else "REGISTERED DISCOVERY: non-monotone"

    out = {
        "experiment": "exp_01_diagram_selectivity",
        "registration_commit": "c5e05712",
        "R1": r1, "R3": r3,
        "t1_flags": {k: v["t1_contaminated"] for k, v in arms.items()},
        "arms": arms,
    }
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"exp_01_diagram_selectivity_{ts}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    slim = {k: v for k, v in out.items() if k != "arms"}
    print(json.dumps(slim, indent=2, default=str))
    print("saved ->", path)


if __name__ == "__main__":
    main()
