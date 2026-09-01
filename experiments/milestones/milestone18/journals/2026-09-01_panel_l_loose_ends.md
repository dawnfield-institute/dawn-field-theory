# 2026-09-01: Panel L — the loose ends, examined

**Mode: EXPLORING.** Each caveat and failure in the milestone's record was reopened and either
closed, downgraded, or found to contain something. Scripts: `explore_l1_accidentals.py`,
`explore_l1b_quotient_folds.py`, `explore_l2_leakage_coefficients.py`,
`explore_l4_mfpt_loose_ends.py`, plus inline probes.

## Closed

- **exp_04's E-series "march to a limit"** (E₆ 1.47 → E₇ 1.34 → E₈ 1.27) was an artifact of
  three points: E₉ 1.13, E₁₀ 1.19, E₁₁ 1.17, E₁₂ 1.49, E₁₃ 1.48, E₁₄ 1.47. No limit, no
  monotonicity; the D-series jumps likewise (D₆ 0.79 → D₇ 1.43). Frame-free MFPT s* is
  irregular topology noise. The thread is dead and exp_04's null is cleaner for it.
- **Do golden trees see the self-dual point?** No. Among all 32 √5-golden trees at n ≤ 12,
  only the two near-stars sit at s* = 1.01; every Galois fold (A₄ 1.50, E₈ 1.27, cat8 0.87,
  the 12-vertex folds 0.65–1.36) scatters like the null. s* ≈ 1 tracks hub symmetry, not
  goldenness — exp_04's declared confound, confirmed from the census side.
- **E₈ at s = ¾ (exp_07 T4 flag)**: both rational quartics have full S₄ Galois groups with
  unrelated discriminants (2¹¹·621869; 2¹³·31·14107), and path8 also splits [4,4] at s = ¾.
  A path-family rational coincidence, no field structure. Downgraded from flag to explained.
- **The two complements (the pun).** exp_06 T4 proved deletion-complement (M13 orbits) and
  orthogonal complement (σ) irreconcilable at D₆'s core and trivially compatible where P is
  functorial. Closed: they are different operations that agree only where nothing forces a
  choice.

## The accidental golden trees are quotient folds — the classification has no residue

The five non-folding golden trees (star₆; a 10-vertex spider; three 12-vertex hubs) fold by
their own **automorphism-orbit quotient**: for every one, the golden factors of the tree's
characteristic polynomial are *exactly* the golden factors of its equitable-partition
quotient matrix (5/5). Their √5 arrives through symmetric modes — star₆'s adjacency ±√5,
a 6-legged spider's ±φ² and ±φ⁻² — and the quotient is a small golden matrix, not a
Coxeter diagram. So at n ≤ 12:

    golden tree  ⟺  Galois fold (one-5 Coxeter-diagram parent)  ∪  quotient fold (orbit quotient)

and the two mechanisms are exclusive on every example: cat8 folds by Galois and not by its
Z₂ quotient; the stars fold by quotient and not by Galois. The residue of Panel G is gone.

## A new invariant: the boundary coupling is a field constant

exp_05 found ‖(I−P)BP‖² = 2/5 on A₄ and E₈ and recorded it uninterpreted. Across **all seven
strict Galois folds at n ≤ 12** — A₄, E₈, cat8, and the four 12-vertex hyperbolic parents,
linear and branched — with B = 2I − D the boundary operator and P the copy projector:

    ‖(I−P)BP‖²_F = 2/5          tr(PB) = 1 − 1/√5          (7/7)

while the diagram-dependent quantities (‖(I−P)B‖², tr(PC)) vary. Since tr(B) = Σ(2−deg) = 2
for every tree and P + σP = I, the boundary trace splits between the two golden copies as
1 ∓ 1/√5 — universally — and 1 − 1/√5 = 2/(2+φ) is Block A's inner-shell radius². The
coupling of the copies through the boundary is a constant of the fold, not of the diagram.

Reduced by hand: for any strict fold tr(PC) = n (a diagonal-2 diagram on n/2 nodes) and
tr(P) = n/2, so tr(PA) = 0 exactly and the whole invariant is one scalar,
**tr(RD) = 2/√5** with R = P − σ(P) — the degree-weighted imbalance between the copies.

And it is a property of Galois folds specifically: on the quotient folds (star₆, the
12-vertex hubs with pure (t−2)^m cores, where the core traces are gauge-independent)
tr(RD) = 0 and tr(PB) = 1 exactly — no 1/√5 shift. So tr(RD) is a **diagnostic of the fold
mechanism**: 2/√5 for a Coxeter-diagram fold, 0 for a symmetry quotient. No proof yet;
registered in the Phase 3 draft (exp_12 T4–T6), including the diagnostic as a claim that can
fail in either direction.

## Recorded, still open

- **The outer Robin point s = −1** (leaf diagonal 0): field content on paths is irregular —
  n=3: √3; n=4: √5 and √13; n=5: √2 and √17; n=9: √3; none for other n ≤ 12. Path4's
  golden content there remains unique in range. Recorded without a law.
- **Block D** stays gated; the golden-probe geometry now has a natural candidate (H₃, the
  icosahedral honeycomb), but that is a design question for Peter.
- **The reality-engine Möbius/Π question** — untouched, Peter's lane.
