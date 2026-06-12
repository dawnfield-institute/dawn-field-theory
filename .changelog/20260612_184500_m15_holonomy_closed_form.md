# M15 Phase-1 Gate: Affine Holonomy Closed Form Proven

**Date:** 2026-06-12

The first curvature invariant (M15 exp_01, round 1) is now derived in closed form and
verified. Phase 1 gate CLOSED; Phase 2 (ℤ₂ twist classification + density→holonomy
field-equation hunt) opens.

## Proven (journal `2026-06-12_holonomy_closed_form.md`, verified by exp_04)

- **θ(m) closed form**: per-edge overlap matrix M has diagonal exactly cos(πj/m) (finite-
  size corrections cancel identically); θ(m) = m·θ_T(m) reproduces all ten round-1 measured
  angles to 1e-16.
- **C₆ = −I theorem**: θ(6) = π exactly (θ_T(6) = π/6) — the ℤ₂ frame-inversion twist on the
  hexagon is structural, not a numerical accident.
- **cos θ(C₄) = −7/9** derived from first principles (matches measured exact rational).
- **Limit = 8/3 exactly** — the e candidate (from a five-point extrapolation, ~2.71) is
  RULED OUT. Derivation-first policy prevented a numerology error. 8/3 = F₆/F₄ noted [D],
  not claimed (the derivation is odd-harmonic, not Fibonacci).

## Corrected by the gate

My first-draft mechanism ("rotational covariance → H = T^m") was WRONG and exp_04 caught
it: the Procrustes edge transports carry mixed orientation (det T(0→1) = −1 reflection,
det T(1→2) = +1 rotation), so H is not a clean power. The formula θ(m) = m·θ_T still lands
exactly (per-edge rotation angle is extracted correctly; reflections telescope around the
loop), but the clean group-theoretic derivation is OPEN. The reflection structure may be
the edge-level face of the C₆ ℤ₂ twist — a Phase-2 question, registered not claimed.

## Files

- `milestone15/journals/2026-06-12_holonomy_closed_form.md` (derivation, honest §4/§8)
- `milestone15/scripts/exp_04_holonomy_closed_form.py` (verification; gate criteria explicit)
- `milestone15/results/exp_04_holonomy_closed_form_20260612_183325.json`
- README gate record added; foundational kill-sentence recorded.

## Also in this changelog batch (held round-1 outcomes, now landing)

exp_01 (3/4, first curvature invariant), exp_02 (0/3, boundary-dominated observable killed),
exp_03 (deferred — M14 null is static). See `2026-06-11_m15-round1-outcomes.md` and
`.changelog/20260611_120000_m15_founding_round1.md`.
