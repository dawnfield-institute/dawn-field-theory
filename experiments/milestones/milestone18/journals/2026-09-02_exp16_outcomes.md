# exp_16 outcomes — Phase 7: does heat-kernel dynamics see the reinjection port? (seal e5263071)

**Verdict: T1 FAIL as sealed, T2 PASS.** Objects: the 21 strict folds at n ≤ 16; defect edge
recovered from R on every fold (instrument finds the branch without the construction).

| Test | Result | Count |
|---|---|---|
| T1 defect distinguished vs Poisson-binomial null (β = 1) | **FAIL** | 4 of 6 informative folds; 99% null quantile 5 |
| T2 null calibration on degree-matched random trees | **PASS** | 26/51 distinguished, quantile 34 — the null is honest |
| T4 β-robustness (recorded) | — | 4 at β = 1/2, 1, 2 — the same four |

## What happened, honestly

**The instrument had almost no power, and that was knowable before sealing.** A fold is
informative only if the defect's degree-matched class holds ≥ 3 edge orbits; 15 of 21 folds
fail that (classes of size 1 or 2). Class sizes are combinatorial — computable from the trees
alone, with no look at the observable — so a pre-seal power estimate would have shown the
test was vacuous on three quarters of its objects. It was not done. **Registration lesson 7:
compute the informative count from the objects before sealing a rank test.**

**The residual signal, labelled as a reading.** On the six informative folds the defect edge is
the *maximum* bond occupation in its class four times (ranks 1/4, 1/4, 1/4, 1/3) and rank 2/6
on the other two — never low. Under the uniform null the expected number of extremal (max or
min) outcomes is ≈ 2.7 of 6, so 4 is unremarkable; but all four being maxima and the two misses
sitting second is the direction Peter's hypothesis predicts (the port carries the most bond
occupation). Suggestive, underpowered, unscored.

**Instrument note.** The sealed script's "A4 trace gate" line was a tautology (it compared a
quantity against zero with ≥ 0); replaced after the run with a genuine edge-sum identity
(Σ_e u_e · tr K = tr(A·K)), which passes; the rerun reproduces the scored numbers exactly. The
scored run is the sealed one; this is an instrument correction, recorded.

## The next design (not sealed here)

Two routes with real power, both structural rather than degree-matched:
1. **The Π-asymmetry profile.** The two sheets of the branched cover are isomorphic except at
   the branch, so for every non-defect edge e the pair (u_e, u_{Π(e)}) should be nearly
   symmetric, with the asymmetry |u_e − u_{Π(e)}| localized near the defect and decaying with
   distance from it. That is a per-fold *profile* test (many edges per fold, no class-size
   problem), and it uses the construction's own symmetry as the null.
2. **n = 20.** The 66 strict folds at 20 have larger degree-matched classes; once exp_15
   delivers their defects, T1's own design gains power.
Phase 7b will be drafted on route 1 after Phase 6 is scored.

Kill scope honoured: this retires "heat-kernel dynamics at β = 1 distinguishes the defect
within degree-matched classes" at n ≤ 16 as an *underpowered null*; the construction theorem
and the matching laws are untouched.
