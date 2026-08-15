# Turbulence Re-Founding — Session Handoff (PARTIALLY FINISHED)

**Date**: 2026-08-15
**Branch**: `research/ade-cascade-round1` (second life — the original merged as PR #166 on 2026-07-17, before the Aug reorganization)
**Status**: **PARTIALLY FINISHED — work in progress, safe to build on, nothing here is a finished claim beyond what is explicitly marked CONFIRM/KILL.**

This entry is the handoff summary for the turbulence/M15 work carried out 2026-07-17
and adapted to the reorganized layout on 2026-08-15. It exists so another agent can
pick the thread up without replaying the investigation.

---

## 1. What prompted it

An audit of the turbulence/Navier-Stokes/cascade stack, checking every headline claim
against the artifacts that produced it. Findings are recorded as Entry 5 of
[`theory/corrections.md`](../theory/corrections.md) and are not repeated here. The short
version: the cascade claims were pre-milestone work resting on a coupling kernel nobody
had questioned, and the strongest signal in the corpus (exp_15's structured-vs-random
z = 52.7) had never been posed in the mature framework's language.

**The finding that reframed everything**: the legacy kernel `exp(−|i−j|·cd)` is exactly
the graph-distance kernel on the **A-family path Dynkin diagram**. The engine had always
been running one diagram without knowing it.

## 2. What was established (all pre-registered before running)

**ADE cascade round 1** — [`experiments/studies/ade_cascade/`](../experiments/studies/ade_cascade/)
Registration `c5e05712`, outcomes journal cites it; refactor-safety gate passed first.

- **R1 CONFIRM** — at equal rank, with identical means vectors, Dynkin topology *alone*
  moves the spectral exponent: E < D < A at ranks 6–8, separations 20–90× the CI
  half-width, 100 seeds/arm. The mode-count-only null is dead.
- **[D], not scored** — the legacy engine's long-unexplained 3.3% miss of −5/3 is an
  **A-family artifact**: D₈ = −1.6624, **0.26% from Kolmogorov**. Whether the physical
  exponent *selects* a diagram is the registered round-2 question, deliberately not
  claimed here (look-elsewhere: nine arms, one famous constant).
- **R2 KILL** — the affine-vertex reading of the k−1 offset died its registered death
  (median ρ = 0.91 vs the < 0.25 bar; direction reversed — closing the cycle *steepens*
  the cascade, because it halves graph distances). The k−1 offset is now cleanly
  **unexplained**; any replacement hypothesis must be registered before testing.
- **R3 CONFIRM** — mode-count monotonicity survives the kernel swap.

**Milestone 15** — [`experiments/milestones/milestone15/`](../experiments/milestones/milestone15/)

- **exp_05**: both registered candidates for the general-k holonomy limit **died**
  (measured L₃ = 5.4913, L₄ = 11.1863 vs odd-harmonic 3.0667/3.3524 and Fibonacci
  2.625/2.619). The 8/3 = F₆/F₄ coincidence is **formally retired**. The anchor gate
  caught and voided an instrument sign-convention bug before any result was read.
- **exp_06 CONFIRM**: the connection generator is the **particle-in-a-box momentum
  operator**, G[j′,j] = 4jj′/(j′² − j²) for j′−j odd (parity selection rule);
  L_k = half the nuclear norm of G_k. Registered predictions L₅ = 17.010952,
  L₆ = 25.778092 measured at 6.7×10⁻⁷ / 1.1×10⁻⁶; entrywise generator match ~10⁻⁵.
  **The general-k limit is solved.** Both prior candidates had to die because L_k is
  algebraic (L₃ = 8√106/15), rational only at k = 2 — no "nice formula" could work.
- Conceptual content: transporting the frame along the cycle shifts the deleted vertex —
  the box translates — so the generator is the translation generator. Curvature of this
  connection is momentum. **Honest grade**: the generator is *conceptually forced* once
  you notice the path graph discretizes the interval (spectral graph theory, not new
  physics). The object that is genuinely non-trivial and not recoverable from d/dx on
  [0,1] is **C₆ = −I** and the ℤ₂ twist (det H = +1 universally, k = 2…6).

**DNS instrument v0** — [`experiments/studies/dns_instrument/`](../experiments/studies/dns_instrument/)
2D pseudo-spectral incompressible NS, vorticity-streamfunction, 2/3-dealiased, RK4.
Qualified: Taylor-Green 2×10⁻¹⁵, energy-budget closure 4×10⁻⁹, resolution consistency
1.1×10⁻⁸ (the gate-design history, including two rejected wrong-test variants, is in the
script docstring). **No physics claims at v0** — instrument qualification only.
Backend-agnostic: the GPU port swaps the array backend, not the logic.

## 3. What is explicitly OPEN

- **Ξ-unification is a registered-open conjecture, NOT a result.** An earlier draft of the
  exp_06 outcomes asserted "the balance constant and the connection generator are the same
  object under two boundary twists"; adversarial review caught the declarative leak and it
  was demoted. On the record: 1 + π/55 = 3/(2N\*) exactly at the *non-integer* cutoff
  N\* = 3F₁₀/2π = 26.2606, and the integer-N spectral ratio → 1 as N → ∞, so the constant
  is not currently a scale-free invariant. The registrable question is whether a
  boundary-twist invariant of *this* connection yields π/55 with N fixed by the graph and
  no external cutoff. **Proposed next experiment (exp_07), not yet registered.**
- **M15 Phase 2 kill-sentence still stands**: if the holonomy is dynamically inert, it is
  mathematics, not physics. Nothing here shows it active. Note the Aharonov-Bohm-style
  test was considered and **rejected as currently unsound** — with exogenous flux you
  confirm Byers-Yang (zero framework content); with intrinsic holonomy C₆ = −I is a fixed
  theorem, not a knob, so you can only compare different graphs (confounded); and the
  hopping law is under-specified enough that the result is dial-able. It is blocked until
  Phase 2 produces a non-arbitrary field equation.
- **The k−1 offset** — reopened by R2's kill, no hypothesis on deck.
- **k ≥ 3 twist structure** — whether the holonomy is genuinely non-abelian there, and
  whether −I-type twists recur, is unexamined. The abelian-SO(2) reading that defuses the
  "spinor" interpretation only applies at k = 2.

## 4. Proposed next steps (none registered, none started)

1. **ade_cascade round 2** — diagram *selection* as a robustness relation across the
   (coupling_decay, nonlinear_strength) region, not a point match; look-elsewhere handled.
2. **M15 exp_07** — the cutoff-free Ξ twist test described above. Cheap (CPU, minutes),
   binary outcome. Prior expectation on record: it will *not* produce π/55.
3. **DNS GPU phase** — the 2D She-Leveque analog (k = 4 on real flow, never tested) and
   MAR's k(2+1) = 5.18. Note MAR exp_21 and exp_24 disagree with each other (5.39 vs 5.18)
   on this one testable point, which is itself informative.
4. **Up-quark mass ratios on branched diagrams** — the 40–60% failure with five dead
   rescue attempts was measured on a path-topology cascade. If the −5/3 gap was an
   A-family artifact, the same may hold here. Registered relation, not a fit.
5. **Reality-engine refurbishment** — proposed as the venue for M15 Phase 2 (a holonomy
   analyzer over emergent structures), the Ξ test on its native Möbius substrate, and a
   scorecard topology-perturbation arm mirroring the D₈ result.

## 5. Housekeeping in this change

- Session work adapted to the Aug 2026 layout: cross-directory imports repointed for
  `experiments/{milestones,studies}/`. **Verified by re-running, not assumed** —
  ade_cascade exp_00 gate (G1/G2 = 0.0, G3 baseline reproduced), M15 exp_06 (re-CONFIRMS
  at 6.7×10⁻⁷ / 1.1×10⁻⁶), DNS Taylor-Green (1.8×10⁻¹⁵). Fresh result JSONs included.
- `tags` added to the two new experiment `meta.yaml` files per schema 2.1.
- Generated indexes regenerated (`EXPERIMENTS.md`, `INVENTORY.md`, `map.yaml`).
- Peter's in-flight PACSeries v0.3 packaging state carried in: `papers/series/PACSeries/series.yaml`
  (schema 3.0, concept-DOI lineage). **Note**: his six accompanying paper `meta.yaml`
  drafts were schema 3.0 while the reorganization standardized the tree on 2.1/2.0; the
  merge resolved to the repo standard, and his drafts remain recoverable at commit
  `6b82aa6b`. **His call** whether to carry the 3.0 shape forward.

## 6. Method note

Every experiment above was pre-registered before running, with locked decision rules and
an explicit inconclusive branch, following the midnight discipline. The registration
culture killed e, K1, K2, and R2, caught an instrument bug via an anchor gate, and caught
a declarative overreach in a results section — and confirmed the momentum generator.
It cuts both ways; that is the point.
