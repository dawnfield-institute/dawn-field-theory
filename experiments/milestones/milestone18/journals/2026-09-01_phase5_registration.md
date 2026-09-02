# Phase 5 registration (SEALED): the off-core instrument and the selector, out of sample

**Status: SEALED 2026-09-01** (the commit carrying this heading; nothing below ran before it).
Scored to this text (§2.6). Lessons 1–5 applied: operators by construction, unreachable cases
enumerated, partners quantified, even-k mechanisms anticipated. Pre-seal self-review: fold count
corrected to 80; T5's layer-order evidence verified (the 18 clean folds at n ≤ 14 all pass trace
and structure).

## What this phase is

Phases 3–4 evaluated core folds through a conic-resolved core recipe that reached almost nothing
at n = 16 (4 of 80). Explorations r11–r14 found, at n ≤ 14, that the fold laws are properties of
the gauge-free off-core reflection R_off = P_off − σ(P_off), and that the clean regime has an
exact description. Phase 5 seals that description as predictions on objects it was not fitted
to: the partnered core-grade Galois folds at n = 16 (80 folds from exp_13's census that Phase 4
mostly had to declare).

## Instruments (by construction)

- Census and partner map: exp_13's (exhaustive n = 16; partner = any one-5 diagram whose q·σ(q)
  matches the tree's characteristic polynomial — quantified over partners per Lesson 4).
- For each fold: r = product of rational factors of p, q_off = q with rational content removed,
  P_off = (v·σq_off)(C)(I − Qc) via Bezout, Qc = orthogonal projector onto ker r(C) (all roots,
  quadratic factors included). R_off = P_off − σ(P_off). B = 2I − D.
- Checked before evaluation (theorem, Lemma 1 in the proof notes): P_off + σP_off = I − Qc,
  P_off² = P_off. A failure is an instrument fault, reported, not scored.
- **B-invariant**: B·Qc = Qc·B exactly. **Uniform mass**: Qc_vv equal at every vertex of the
  support {v : Qc_vv ≠ 0}. **Clean(F)** for a fold F: (vertex law √5·R_off,vv = ±(1 − Qc_vv) at
  every vertex) AND (off→off leak ‖σ(P_off)·B·P_off‖² = 2/5).
- Sign split: copy = {v : R_off,vv > 0}, conjugate = {v : R_off,vv < 0}; folds with any
  R_off,vv = 0 are reported and excluded from T3's structure clause with the exclusion declared.

## Tests (all PREDICTIONS, can fail; evidence is n ≤ 14 only)

- **T1 (the selector, both directions).** For every partnered core-grade Galois fold at n = 16:
  Clean(F) ⟺ (B-invariant OR uniform mass). Fails if any fold violates either direction.
  (n ≤ 14: exact, 18 = 16 ∪ 2 of 61.)
- **T2 (sufficiency for the remaining laws).** For every such fold with (B-invariant OR uniform
  mass): tr(R_off·D) = 2/√5. Fails on any violation. (n ≤ 14: 18/18.)
- **T3 (structure on the clean set).** For every such fold with (B-invariant OR uniform mass):
  under the sign split, the copy side is connected, the cut has exactly 2 edges, and the
  conjugate side has exactly two components whose sizes equal (as a multiset) the halves of
  SOME one-5 partner at its 5-bond. Fails on any violation. (n ≤ 14: 18/18 with the some-partner
  semantics.)
- **T4 (the mixed blocks, now a theorem where B-invariant).** Recorded, not scored: for
  B-invariant folds the mixed leak blocks vanish (Lemma 2); the core→core block is recorded —
  its vanishing is NOT predicted (open observation at n ≤ 14).
- **T5 (the layer order).** For every partnered core fold at 16: Clean(F) ⇒ (trace law AND
  structure law). I.e. no fold is vertex/leak-clean while breaking trace or structure — the
  peeling order observed at n ≤ 14 (laws fail from the vertex/leak end first, never the
  reverse). Fails if any fold shows Clean without trace or without structure.

Out of recipe, declared, not scored: nothing — the off-core instrument reaches every partnered
core fold. Sector-strict and quotient trees are Phase 4 objects, not re-tested here.

Kill scope: none of T1–T5 touches the milestone kill sentence. A T1 failure retires the selector
as an exact law at 16 (the sufficiency directions T2/T3 are scored separately); T2/T3 failures
retire the corresponding law's extension beyond n ≤ 14; a T5 failure retires the layer-order
claim. The strict-fold laws (13/13 at 16) are untouched by any outcome here.

## Compute estimate

80 folds at 16×16, no conic search: comparable to exp_13's stage 2, ~30–60 min.

---

**Layer (forward note, 2026-09-02, per the re-separation):** Validates an instrument (the off-core reflection) and tests the selector conjecture — feeds `core/` and `formal/`.
