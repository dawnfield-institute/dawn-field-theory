# M18 r21: polynomial integrality refuted at n = 20; denominator law filed

**Date:** 2026-09-02 (evening)
**Milestone:** 18 — The Non-Crystallographic Completion
**Mode:** exploration on held data (no seal; a kill with scope)

## Killed

- **Polynomial integrality** (`formal/conjectures/m18_open.md`): "5·b(t) ∈ ℤ[t] for the
  reflection R = √5·b(C)". Held 7/7 at n ≤ 12 (explore_r10) and 21/21 at n ≤ 16; **fails on three
  construction parents at n = 20** (den(5·b) = 3, Res(q, σq) = 3²·5⁵) with every strict law
  holding on those trees. Scope: the polynomial form only. The matrix form √5·R = S + 2Π ∈ ℤⁿˣⁿ
  (r17 theorem) is unaffected — the failure is the non-saturation of ℤ[C] in its endomorphism ring,
  which is exactly the step a proof of the polynomial form cannot skip.

## Proved in its place (eighth M18 theorem, indexed in `formal/theorems/`)

- **The denominator bound.** With q = q₀ − φ·q₁ (q₀, q₁ ∈ ℤ[t]) and v = a + √5·c, the Bézout
  identity has no √5-part and reduces to a·(2q₀ − q₁) + 5c·q₁ = 1 over ℚ[t], with
  5·b = (5c)·(2q₀ − q₁) + 5a·q₁. Sylvester then gives den(5·b) | 2^{deg q₁}·Res(q₀, q₁): the
  ramified prime never enters unless it divides the diagram resultant; 5·b ∈ ℤ[t] whenever
  Res(q₀, q₁) = ±1. Verified 208/208 halves (rational vs ℚ(√5) Bézout agree; bound holds;
  5 | den ⇔ 5 | Res). The three failures are exactly the parents with Res(q₀, q₁) = 9.

## Filed open

- **The exact denominator** on fold halves: Res = ±1 → 1 (63), Res = 4 → 1 (2), Res = 9 → 3 (3).
  Bound not sharp at 2; odd-prime half rests on three cases.
- Certificate reading (exploration): some Galois half integral ⇒ construction parent — 44/47
  parents, 0/19 non-parents at n = 20; 21/21 vs 0/1 at n ≤ 16.

## Provenance

Prompted by an external proof attempt of the unconditional statement (A. Farmer, 2026-09-02,
private communication). Its exact-odd-difference observation and the resultant factorization
Res(q, σq) = ±(√5)^k·Res(q₀, q₁) are correct and kept; its unramified-prime claim and its
matrix-to-polynomial step are the two gaps the data refutes. Its A₄/E₈/P₆/P₈ examples reproduce
ours exactly; its "det −11" example is not a Coxeter diagram (q₀ has trace 7).

## Files

- `experiments/milestones/milestone18/scripts/explore_r21_poly_integrality.py` (new)
- `experiments/milestones/milestone18/results/explore_r21_poly_integrality_20260902.json` (new, append-only)
- `experiments/milestones/milestone18/journals/2026-09-02_r21_polynomial_integrality_refuted.md` (new)
- `experiments/milestones/milestone18/journals/2026-09-01_provenance_and_proof_notes.md` (forward note appended; no sealed text edited)
- `formal/conjectures/m18_open.md` (row killed with scope; denominator-law row added)
- `experiments/milestones/milestone18/README.md` (theorems/conjectures line: killed: 1)
- `ROADMAP.md` (M18 known-open-ends line)
