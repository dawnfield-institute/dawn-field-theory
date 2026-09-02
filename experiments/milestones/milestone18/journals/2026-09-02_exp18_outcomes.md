# exp_18 outcomes — Phase 7c: the branch fingerprint at n = 20 (seal 4ad88e40)

**Verdict: T1 PASS, T2 FAIL as sealed.** 47 evaluable strict folds at n = 20; defects recovered
from R (r19); null numbers computed from the trees before sealing (lesson 8).

| Test | Result | Count |
|---|---|---|
| T1 peak asymmetry pair AT the defect (distance 0) | **PASS** | **47/47** — null expectation 15.0, 99% quantile 22 |
| T2 profile organized by the branch (per-fold permutation p ≤ 0.10) | **FAIL** | 0/47 (quantile 10) |
| T3 recorded | — | copy sheet > conjugate sheet on 47/47; T1 = 47/47 at β = 1/2 and 2 |

## T1 — the first powered dynamical fingerprint

Under the heat kernel of the Cartan matrix, the asymmetry between an edge's bond occupation and
its Π-image's is maximal at the branch on every fold — with the maximum-asymmetry pair sharing an
endpoint with the defect edge 47 times out of 47, where geometry alone would place it there about
15 times. β-independent across a factor of four. This is the statement Phases 7 and 7b could not
power: **rational dynamics, which is provably blind to σ, sees the σ-fixed branch of the fold.**
Combined with r17: the construction's two sheets are dynamically distinguishable exactly where
they are glued, and the copy sheet (the one carrying the diagram's own spectrum) holds the larger
share of bond occupation everywhere (47/47).

## T2 — honest reading

Spearman ρ between asymmetry and distance-from-defect is between −0.50 and −0.56 on **all 47
folds**: the profile decays from the branch, always, steeply. The sealed null compared the
defect against every other reference edge, and the defect's rank among those 19 references is
typically third (p ≈ 0.16): reference edges *adjacent* to the defect organize the profile almost
as well, because distances from neighbouring edges are nearly the same function. The permutation
null therefore cannot resolve "the branch" from "the branch's neighbourhood" — a resolution
limit of the null, not an absence of organization. Scored FAIL to the sealed text. A future
seal could register the defect's *rank* among references (top-3 of 19 on 47/47 would be a
different, sharper claim) — not done here.

## Standing of the reinjection thread

Peter's hypothesis (the dynamics is entropy reinjection at the port) now has one scored, powered
prediction confirmed (T1) and one direction confirmed but not yet separable from geometry (T2).
Kill scope honoured throughout; the construction theorem and the matching laws are untouched.
Next: the *magnitude* law — whether the asymmetry at the branch is fixed by the degree ledger
(the same Σ(d_v − d_{Π(v)})² = 4 that fixes the leakage) — is the natural Phase 7d.

## Addendum — explore_r20: the magnitude of the branch asymmetry (exploration)

Script `explore_r20_branch_magnitude.py`, all 68 strict folds at n ≤ 20 (β = 1).

1. **The cut pair has exactly zero asymmetry — a theorem.** The two cut lifts (A,0)–(B,1) and
   (A,1)–(B,0) of the special bond are Π-images of each other, and their heat-kernel entries are
   identical: in the golden sector basis ŝ_X(γ) = ((X,0) + γ(X,1))/√(1+γ²), both entries equal
   Σ_γ γ/(1+γ²) · K^γ_{AB} (the coefficients of (A,0),(B,1) and of (A,1),(B,0) multiply to the
   same γ/(1+γ²) in each sector). Verified to 40 digits on all 47 folds at n = 20 (max
   difference below 1e−30). So the sheet asymmetry vanishes *on the bond itself*; the branch is
   seen through its neighbourhood, not through the glued edges.
2. **Corollary sharpening T1.** Since cut pairs carry zero asymmetry, the peak pair (distance 0
   on 68/68) is always a *first-neighbour* pair of the defect — an edge at one of the defect's
   endpoints other than the defect and the cut, paired with its Π-image on the other sheet.
   This is where the degree ledger bites: the defect endpoint has one more edge than its
   partner, so the two sheets first differ there.
3. **Magnitude is local.** a_max clusters by the defect's degree pair — roughly (2,3): 0.056–0.065,
   (3,3): 0.063–0.074, (3,4): 0.071–0.077, (4,4): 0.073 — and identical values recur to six
   digits across n = 12, 16, 20 (e.g. 0.065441, 0.074359, 0.060159), i.e. the branch asymmetry
   at β = 1 depends only on the branch's neighbourhood out to the kernel's range. A closed form
   in the local degree data is not yet found; the discrete recurrence says one may exist.
