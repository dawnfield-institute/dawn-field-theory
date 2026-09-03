# exp_17 outcomes — Phase 7b: the Π-asymmetry profile (seal c683fdac)

**Verdict: T1 FAIL as sealed, T2 PASS as sealed** — and both verdicts say less than they seem,
in opposite directions. 18 informative folds (as declared pre-seal), β = 1 scored.

| Test | Result | Count |
|---|---|---|
| T1 profile: per-fold permutation p ≤ 0.10 | **FAIL** | 0/18 (quantile 5) |
| T2 peak pair at distance ≤ 1 from the defect | **PASS** | 18/18 (Poisson-binomial quantile 17) |
| T3 recorded | — | the copy sheet (s = +1) carries MORE bond occupation than its conjugate on 18/18 informative folds; β-robustness: identical counts at β = 1/2, 2 |

## Reading T1 honestly

Spearman ρ between Π-asymmetry and distance-from-defect is **negative on 18/18 informative
folds** (−0.16 to −0.49): the asymmetry does decay away from the defect, universally, in the
direction Peter's hypothesis predicts. But the sealed null asked a harder question — *is the
defect a special reference point, or would distances from any edge organize the asymmetry
just as well?* — and at n ≤ 16 the answer is "any edge would": per-fold p-values sit at
0.2–0.8. Small trees have strongly correlated distance functions; a 7-pair profile cannot tell
"organized around the branch" from "organized around the middle". T1 fails as sealed, and the
right conclusion is *not separable at this size*, not *absent*.

## Reading T2 honestly

The maximum-asymmetry pair is **adjacent to the defect (distance 0) on 21/21 folds**, informative
or not. That is the sharpest fact in this run. But the sealed statistic was d ≤ 1, and at these
sizes almost every pair is within distance 1 of the defect, so the sealed null is nearly
saturated (quantile 17 of 18): the pass is by one, and carries little evidential weight *as
sealed*. Had the seal said d = 0 the null would have been far less saturated — but that is a
post-hoc sharpening, and it stays unscored here.

**Registration lesson 8** (the sibling of lesson 7): compute the *null's* distribution from the
objects before sealing, not only the informative count. p_near was computable from the trees
alone; a saturated null is as vacuous as an uninformative object.

## What this run establishes and what it doesn't

Established at n ≤ 16, β = 1 (exploration-grade because of the nulls): the heat-kernel
asymmetry between the two sheets of a strict fold peaks at the branch and decays with distance,
on every fold. Not established: that this is the *branch's* doing rather than tree geometry's.
The separation needs larger objects — the 66 strict folds at n = 20 (9 pairs each, distances
up to ~9) — with a d = 0 peak statistic and the same permutation null, registered fresh
(Phase 7c) once exp_15 delivers their defects. Kill scope honoured: nothing here touches the
construction theorem or the matching laws.
