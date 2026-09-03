# 2026-09-01: exp_08 (the census) outcomes — 1/2, and a second golden tree

Registration: Phase 2 seal 06073227. Prüfer-sampled trees, 500 labeled per n = 6..10,
aggregated by isomorphism class (WL hash): 6 / 10 / 22 / 41 / 86 classes seen (unlabeled
trees on 6–10 vertices number 6 / 11 / 23 / 47 / 106 — coverage is complete or near-complete
at n ≤ 8, ~85% at n = 9–10). Criterion: complete σ-pairing — every irreducible factor of
the characteristic polynomial over ℚ(√5) carries √5.

## Instrument note (transparent)

The first run reported 0 Cartan hits at n = 8, which is impossible (E₈ is a theorem-hit and
~15% of labeled 8-trees). Cause: the pairing test multiplied all golden factors into one q
instead of testing σ-closure; second fault, dedup by labeled edge-set. Fixed, and the script
now carries a known-answer self-test (E₈ Cartan positive, A₅ negative, cat8 A² positive)
that aborts the census if the criterion is wrong. No result below predates the fix.

## Results

| n | classes | Cartan complete-pairing hits | A² hits |
|---|---|---|---|
| 6 | 6 | 0 | 0 |
| 7 | 10 | 0 | 0 |
| **8** | 22 | **2** | **2** (same two classes) |
| 9 | 41 | 0 | 0 |
| 10 | 86 | 0 | 0 |

The two n = 8 classes: **E₈** (adjacency spectral radius 2cos(π/30) = 1.98904, as required)
and **the caterpillar "cat8"** — spine 0–6–1–3–4–7 with legs on 1 and 3 — spectral radius
2.09529 > 2, hence not ADE or affine ADE (Smith's theorem). cat8's Cartan-form
characteristic polynomial factors over ℚ(√5) into **two conjugate quadratic pairs**:

    (t² − (9+√5)/2·t + 4+√5)·σ(·) · (t² − (7+√5)/2·t + 2+√5)·σ(·)

so it is completely σ-paired with a *reducible* golden q = q₁q₂ — a different species from
the folding diagrams, whose q is irreducible (the H-partner's polynomial). Its "Cartan"
2I − A has determinant −11 and a negative eigenvalue: an indefinite, hyperbolic-type
diagram carrying the golden ledger. Recorded, not interpreted.

## Scoring against the seal

- (a) "Cartan complete pairing ≈ 0 outside ADE (report count)": **count = 1 class** (cat8)
  out of 165 classes at n ≤ 10, i.e. 0.6% — the registration's "≈ 0" was not a crisp
  threshold; reported as **under-specified, not scored as pass**. The existence of one
  non-ADE Cartan hit is the finding.
- (b) "A² hits > 0, characterized by spectral radius > 2; fails if hits are exclusively
  λ_max ≤ 2": hits = {E₈ (≤ 2), cat8 (> 2)} — **PASS** on the fail-condition, with the
  characterization only half-right: the A² hit set equals the Cartan hit set exactly, so at
  n ≤ 10 the squared channel adds nothing the bare channel lacks.
- Inconclusive branch not reached: the hit set is tiny but structured (n = 8 only, two
  species: irreducible-q folding diagram, reducible-q hyperbolic caterpillar).

**exp_08: 1/2.** Consequence for the collaborating framework: the "charpoly factorization
is an iff diagnostic for golden content in any graph" claim (GRA V3 §6.2 / Rule 15) is
falsified by an explicit 8-vertex tree; the theorem remains exact on ADE Cartans.
Milestone 26/31.

## Forward correction (2026-09-01, Panel G)

The sampled census missed classes. Exhaustive enumeration (`explore_g1_census_exhaustive.py`,
n ≤ 13) finds strict pairing at n = 4, 8, 12 and core-grade ledgers at n = 6, 8, 10, 12; the
"n = 8 only" statement above is withdrawn. Sampling at small n is retired in favour of
enumeration. The score is unchanged.
