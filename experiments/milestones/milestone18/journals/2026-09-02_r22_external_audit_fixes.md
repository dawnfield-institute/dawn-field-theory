# r22 — external audit of the replication suite: findings, verdicts, fixes, reruns (2026-09-02, late)

**Mode: instruments (→ `core/` and `scripts/`).** No claim is tested here; the scripts that test
claims are repaired and rerun, and the record is compared before and after.

An automated code audit (Andy Farmer's Penta-Ledger "code cheating check", run on the four
replication scripts we sent him in the first 2026-09-02 bundle plus `core/ledger.py`) returned
eleven findings under a headline "deception index: HIGH". The headline is wrong — every verdict in
those scripts is computed, none is asserted — and about half the findings are right. This journal
takes each one against the code, fixes what is real, and reruns everything to new timestamped
outputs (nothing committed was overwritten).

## Findings, verdicts, fixes

| # | Finding (their label) | Verdict | What changed |
|---|---|---|---|
| 001 | r17b hardcodes the twin's edge list ("simulation cheat") | **Real as provenance, not as a cheat.** The twin lives in exp_13's output, not in r15's list; its verdict (isomorphic to a construction?) was computed. | r17b now loads n ≤ 13 from r15's list and **all** n = 16 strict trees from the exhaustive r16b census (which contains the twin); the literal is gone; folds are de-duplicated. |
| 002 | r17 sampling cap saturates at k = 4; k = 6, 7 never spot-checked ("deceptive logic") | **Real bug.** The journal did not overclaim — it recorded "4 strict samples (4/4)" and the log shows k = 2, 4, 4, 4 — but the script's intent (`k in (2,4,6,7)`) was not met. | Stratified: up to 2 strict samples per k ∈ {2, 4, 6, 7}; degenerate placements counted while sampling. |
| 003 | `nsimplify` in an exact pipeline (r15, r17) | **Smell, not a flaw.** Applied to already-exact entries (radsimp + expand) it is a no-op; it does not belong in a proof pipeline. | Removed; comparisons are on exact entries. |
| 004 | Bundle scripts expect `../core` and `../results`; shipped layout crashes | **Real packaging defect** (the HOWTO said "or adjust the path"). | Bundle 2 ships the `scripts/ core/ results/` layout; imports smoke-tested from inside the bundle. Repo layout was always correct. |
| 005 | r15 tests "5-bond = mult-3 edge" under the first isomorphism only | **Real, and it had bitten.** See below. | Every isomorphism (`isomorphisms_iter`) is tried. |
| 006 | Dead `H4` assignment in `ledger.py` | Real, cosmetic. | Deleted. |
| 007 | `ledger.projector()` unused | **Not a finding** in the repo — exp_06 and exp_12 use it; it was unused only within the four scripts shipped. | None. |
| 008 | r16b docstring says "14 known strict trees"; gate checks 15 trees / 14 polynomials | Real, cosmetic. | Docstring corrected. |
| 009 | r17 skips degenerate samples without counting | Minor — they were printed, not counted. | Counted and reported. |
| 010 | r15 keeps the first cospectral placement per polynomial | **Real, and known** (documented in `foldlaws.one5_diagrams`; r17b already used lists). | r15 keeps every placement and tests the quotient against each. |
| 011 | Bare `assert` on a degenerate fold in `bezout_proj` | Real hygiene (it is the crash r21 hit). | `DegenerateFoldError(ValueError)`; no caller relied on `AssertionError` (exp_15 checks the gcd first). |

Also fixed while there: r15 and r16b wrote to **fixed** result filenames, so a rerun would have
overwritten a committed result. Both now write timestamped files (STANDARDS: results are
append-only). The committed originals are untouched and remain the inputs of r17b.

## The one verdict that changed — and what it means

Rerunning r15 at n = 16 changes exactly one field on one fold: the det −239 tree's
"5-bond = mult-3 edge" goes **False → True**. Every other verdict at n = 12 (7 folds) and n = 16
(13 folds) is identical to the committed record.

That fold is the cospectral-placement tree: its polynomial admits two inequivalent 5-bond
placements, (0,5) with halves [3, 5] and (1, 4) with halves [1, 7]. The original script (finding
010) tested against the first placement only, whose bond is *not* where the mult-3 edge sits, and
recorded False; the r15 journal's point 4 then worked out by hand that the mult-3 edge maps onto
the (0,5) placement. The repaired script — every placement, every isomorphism — returns exactly
that. So the *reasoning* in the record was right and the *raw verdict* was a false negative of the
instrument, which is what finding 005 predicted. The record now agrees with its own footnote.

## Reruns (all outputs new and timestamped; `results/*_rerun_*_log.txt`)

| script | before | after |
|---|---|---|
| r17 | charpoly 117/117; sheet form 4/4 samples at k = 2, 4, 4, 4 | charpoly **117/117**; sheet form **5/5** at k = 2, 4, 4, 6, 6 (66 degenerate placements passed over) |
| r17b | 21/21 (20 from r15 + the inlined twin) | **21/22** — the 22nd is the sector-strict n = 16 tree, which has no one-5 partner and is *not* a construction parent by definition; twin pair found by cospectrality, placements (1,4)/[1,7] and (0,5)/[3,5] as before |
| r15 n = 12 | 7 folds all True | 7 folds all True, identical |
| r15 n = 16 | 13 folds; one "5-bond = mult-3" False (det −239) | 13 folds **all True**; the False was the instrument |
| r16b n = 16 | gate PASS (15 trees, 14 polys) | gate **PASS** (15 trees, 14 polys), 19,320 trees, 706 survivors, 67 s |

**Why k = 7 has no sheet-form sample and never can:** odd-k parents carry a rational core (theorem,
`formal/theorems/` "Odd-k parents…"), so q and σ(q) share roots and no strict sample exists at
k = 7. The original `k in (2,4,6,7)` was wrong at 7 by our own theorem; the stratified check covers
every strict size ≤ 7.

## What this changes in the record

Nothing scored moves. One raw verdict in an exploration result is corrected by rerun (kept side by
side with the original), one spot-check is widened to the sizes it was meant to cover, and the
replication kit shipped to a collaborator becomes runnable as shipped. The audit's headline was
wrong; its list was useful. That is the exchange a ledger is for.
