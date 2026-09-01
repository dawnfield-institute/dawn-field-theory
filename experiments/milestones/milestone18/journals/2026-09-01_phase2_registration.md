# REGISTERED: Phase 2 — exp_05, Block E, the census

**Status: SEALED by this commit (Peter's go, 2026-09-01). Pre-seal note: exp_05 T3/T4 were
relabeled (S) before sealing — both are forced by T1's Galois symmetry and confirm rather than
predict. exp_05's only scored prediction is T2.**

## Kill-sentence status at the start of Phase 2 (honest accounting)

The founding kill sentence has two clauses. Clause B ("E₈-derived spectra do not split into
two φ-scaled families") is **not met**: the split exists as a theorem in the Cartan channel
(exp_06 T2, exp_07). Clause C ("the indefinite golden form adds no separating power at the
orbit boundary") **is met**: exp_06 T4 proved the σ-ledger and the orbit quotient
irreconcilable at D₆'s core. Per the sentence, one clause wounds, both kill. **The milestone
is wounded, not dead**, and Phase 2 must say so in the README.

## exp_05 — Galois blindness and the golden probe (replaces the Λ design)

**Source facts (read from code, not assumed).** M-R's dynamics is Laplacian diffusion
(`milestone12/core/connection_geometry.py::redistribute_on_graph`, s = 0); the stress-
barrier FPT thresholds instantaneous noise-driven edge flows and is degree-blind (M-R
exp_17); HKS is Σ e^{−λt}v² on L (one-wing). No corpus channel reads the σ-structure.

**T1 (S — theorem, verification only).** Any observable produced by rational operations on
rational operators, rational initial data, and symmetric noise is σ-invariant; therefore no
such dynamics distinguishes a golden copy from its conjugate. Verified by construction on
A₄/D₆/E₈: all trace statistics of M(s)-dynamics are rational for every s.
Consequence: Block B's original question is ill-posed for rational dynamics. Selecting a
copy requires observer-supplied golden data — M15's frame data, realized.

**T2 (registered, can fail).** With a golden probe — initial state in im(P), P the exact
Cartan-channel projector — leakage into the conjugate copy under M(s)-diffusion is
**first order** in the detuning: L(s) = ‖(I−P)·e^{−tM(s)}·P‖ ∝ |s−1|·‖(I−P)BP‖ + O((s−1)²),
with B = 2I − D the boundary operator. Prediction fixed before running: the coefficient
‖(I−P)BP‖² lies in ℚ(√5) and is nonzero for all three foldings (B does not commute with P).
Fails if leakage is second-order (B commutes with P) or the coefficient is rational.

**T3 (S — confirmation, not scored as prediction).** Under symmetric zero-mean noise at
s = 1, the σ-copy's share of injected noise power equals its dimension share (1/2 for all
three foldings, adjusted for the removed constant mode). Forced by T1; verified numerically.

**T4 (S — control, not scored as prediction).** A rational probe's leakage statistic is
σ-symmetric for every s. Forced by T1; verified numerically. Scored content of exp_05 = T2.

Frame declaration: sampled = trajectories from a declared initial state; expectation =
the exact projector algebra of the same operator; same scope. Tautology guard: T2's zero
at s = 1 is forced by [M(1), P] = 0 and is NOT scored; the scored content is the order and
coefficient of the departure.

## Block E — re-derivations as units (exp_10, 4 tests, exact)

- T1: exp_37's surviving 1/φ⁴ is a unit: N(φ⁻⁴) = +1; even power ⇒ norm +1 ⇒ "split
  side" of the mod-5 trichotomy, not the ramified core.
- T2: Baxter z_c = φ⁵ = F₅φ + F₄; N = −1; odd power ⇒ norm −1.
- T3: Lucas numbers are traces (L_n = tr Qⁿ, rational, class-level); Fibonacci numbers are
  the Δ-channel readout (F_n = (φⁿ−ψⁿ)/√5); Binet is the class + Δ decomposition —
  verified on Qⁿ for n = 1..30 exactly.
- T4: mixing-angle rationality (§2.8.1's field-membership result) restated as a ratio of
  Δ-channel amplitudes: F_m/F_n ∈ ℚ always; any expression with an unpaired φ ∉ ℚ. Verified
  on sin²θ_W = 3/13 = F₄/F₇ and the α formula's φ-bearing factor.

## The census (exp_08, Block C extension — Block D stays gated)

- Ensemble: Prüfer-sampled random trees, n = 6..10, ≥ 500 per n. No hand-drawn graphs
  (star6 and cat8 are retired as controls and recorded as findings).
- Criterion: **complete** σ-pairing — charpoly factors over ℚ(√5) as q·σ(q) with q ∉ ℚ[t]
  — tested in (a) the Cartan channel and (b) the A² channel.
- Predictions fixed before running: (a) frequency ≈ 0 outside ADE (report count);
  (b) frequency > 0, with hits characterized by spectral radius > 2 (Salem-type, per
  Smith's theorem) — if hits are exclusively λ_max ≤ 2 the prediction fails.
- Inconclusive branch: if the A² hit set has no characterizable structure at n ≤ 10, report
  and stop; no numerology over the hit list.

## Not in this milestone (queued separately, Peter's call)

Ξ-fork forward-correction in two FDOs; sec_prime_manifold and M-R exp_20/21 annotations;
the reality-engine boundary arm (Peter's lane).
