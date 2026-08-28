# R1's ordering is a distance effect — the diagrams are ordinary points on a curve

**2026-08-28.** Exploratory (STANDARDS §2.8). Raised while applying §2.8's tautology and
circularity tests to existing CONFIRMs. **R1's measurement is not in question at any point.**

## The confound

R1 concluded: *"at equal rank, with identical means vectors, the Dynkin diagram alone moves the
spectral exponent."* But at equal rank **A_n is the path** — the longest-diameter tree on n
vertices — while D_n and E_n are branched. "Which Dynkin diagram" and "mean graph distance" are
therefore confounded by construction, and round 1 contains no non-Dynkin tree.

**R2's own outcome supplies the mechanism**, in the same journal: *"closing the cycle halves
graph distances… so every off-diagonal coupling strengthens… and the spectrum steepens."*
Shorter distance → stronger coupling → steeper exponent. R1's E<D<A is what that predicts.
Across R1's nine points, mean distance orders the exponents 3/3, with
partial corr(distance, exponent | rank) = **+0.966** and R² going 0.928 (rank alone) → **0.995**.

## Three passes, two of them wrong — recorded because the errors were the work

**Pass 1 — vacuous.** Generated "random non-Dynkin trees" matched on mean distance; 0/9
separated from their Dynkin partners. Then checked: **9/9 were cospectral with identical degree
sequences** — relabellings of the same graphs. Comparing sorted adjacency rows is not an
isomorphism test. The control compared each diagram to a copy of itself.

**Pass 2 — the control cannot exist.** Enumerated every non-isomorphic tree to n=14 (3159 of
them): **zero** share a Dynkin diagram's mean distance. The reason is structural, not
small-n — the ADE shapes sit in the **sparse tail of tree-space**. A_n is the path, the unique
maximiser of the Wiener index; D_n and E_n sit just below. At n=12, D₁₂'s Wiener index of 277
is held by exactly one tree (itself), while **528 of 551** trees share theirs with another. For
these diagrams the two explanations are extensionally identical and no matched control exists
at any rank.

**Pass 3 — drop the diagrams.** Go to the dense middle of tree-space and ask directly whether
the exponent is a function of distance alone. Within Wiener classes at n=12 (17, 16 and 15
distinct topologies at one mean distance each): within-class sd **0.0074** against between-class
sd **0.0301** — distance dominates 4:1, but topology contributes.

## The decisive test

All 23 non-isomorphic trees at n=8. Regress exponent on mean distance; fit using **only the 20
non-Dynkin trees**; ask where A_8/D_8/E_8 land.

Distance explains **R² = 0.98** across all 23 trees.

| fit on non-Dynkin trees | A_8 | D_8 | E_8 | residual sd |
|---|---|---|---|---|
| linear | **−5.24 σ** | −3.35 σ | −1.90 σ | 0.01133 |
| quadratic | +2.52 σ | +0.08 σ | +0.16 σ | **0.00509** |
| cubic | +0.54 σ | −0.63 σ | −0.12 σ | 0.00522 |

**The apparent 5σ Dynkin effect is curvature at the endpoint.** A_8, D_8, E_8 are ranks
**1, 2 and 3 of 23** in mean distance — the three most extreme trees — and a straight line
misfits the boundary of a curved relation. Fit the curvature the non-Dynkin trees themselves
show and all three sit on the curve within ±0.6σ, with the residual sd halved.

## Conclusion

**The exponent is a smooth nonlinear function of mean graph distance. The ADE diagrams are
ordinary points on it.** R1's separations are real and its CIs are real; what is not supported
is that *Dynkin structure* produces them. They appeared distinguished because they occupy the
extreme tail of tree-space, which is also why no matched control can be built for them.

**Consequence for the registered round 2.** The question *"does the physical exponent select a
diagram?"* cannot be answered as posed — at these ranks the diagram and its distance are the
same variable. Re-pose as **"does the physical exponent select a distance scale?"**, which is
answerable and which the n=8 curve already begins to address. Worth settling **before** round 2
runs.

R1's `[D]` note — that the legacy 3.3% miss of −5/3 is "an A-family artifact", with D_8 landing
0.26% from Kolmogorov — becomes a statement about D_8's **distance**, not about D being the
right **diagram**.

## Also fixed here

`core/coupling.py` and `core/runner.py` both resolved milestone paths three levels up, landing
on `experiments/studies/` and looking for `studies/milestone12`. The August 2026 reorg moved
milestones to `experiments/milestones/`, so **this study was unrunnable** — the third instance
of that break after milestone-r and midnight. A repo-wide scan found **16 broken
`sys.path.insert` targets across 7 experiments** (milestone1, ade_cascade, asymmetric_conservation,
dna_prime_structure, exp_31_symmetry_primitive, phi_artifact_test, spikes/infodynamic_gravity);
that is a floor, since only statically-resolvable expressions were checked.
