# M18 r22: replication-suite fixes from an external code audit; reruns; one false negative corrected

**Date:** 2026-09-02 (late)
**Milestone:** 18 — The Non-Crystallographic Completion
**Layer:** instruments (`core/`, `scripts/`)

## What happened
An external automated audit (Penta-Ledger "code cheating check", on the four replication scripts
shipped to Andy Farmer plus `core/ledger.py`) returned eleven findings under a "deception: HIGH"
headline. The headline is wrong (every verdict is computed); six findings are real, four cosmetic,
one not a finding. All are fixed; everything is rerun to new timestamped outputs.

## Fixed
- `core/ledger.py`: `DegenerateFoldError` replaces a bare assert; dead `H4` line removed.
- `explore_r17_construction_theorem.py`: stratified sheet-form samples (≤ 2 per k ∈ {2,4,6,7});
  no `nsimplify`; degenerate placements counted. Rerun: 117/117; sheet form 5/5 at k = 2, 4, 6
  (k = 7 cannot be strict — odd-k theorem).
- `explore_r15_matching_structure.py`: every cospectral placement, every isomorphism; no
  `nsimplify`; timestamped output. Rerun: all True at n = 12 and 16 — **the single committed False
  (det −239, "5-bond = mult-3 edge") was a false negative of the first-placement/first-isomorphism
  check**, exactly as the audit's finding 005 predicted and as the r15 journal's point 4 had
  reasoned by hand. Original file kept; forward note added to the r15 journal.
- `explore_r17b_rigidity_retrocheck.py`: twin loaded from the exhaustive r16b census, no inlined
  edge list; twin pair found by cospectrality. Rerun: 21/22 (the 22nd = the sector-strict tree,
  a non-parent by definition).
- `explore_r16b_strict_hunt.py`: docstring (15 trees / 14 polynomials); timestamped output.
  Rerun gate: PASS.
- Append-only enforced: r15 and r16b no longer write to fixed result filenames.

## Not changed
No scored result. Bundle 1's layout defect (finding 004) is fixed in bundle 2's kit (local, not in
the repo).

## Files
- `experiments/milestones/milestone18/journals/2026-09-02_r22_external_audit_fixes.md` (new)
- `experiments/milestones/milestone18/journals/2026-09-01_r15_matching_structure.md` (forward note)
- `experiments/milestones/milestone18/core/ledger.py`, `scripts/explore_r15_*.py`, `explore_r16b_*.py`, `explore_r17_*.py`, `explore_r17b_*.py`
- `experiments/milestones/milestone18/results/`: nine new rerun logs/JSONs (timestamped)
