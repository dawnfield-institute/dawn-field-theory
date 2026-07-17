# Milestone 15: The Representative Problem (The DFT-Hodge Boundary)

**Status**: active (exploratory)
**Founded**: 2026-06-11
**Origin**: SEC's earliest work targeted the Hodge conjecture (see `hodge_conjecture/`,
archived; mapping honestly withdrawn from the SEC paper as unproven). Fourteen milestones
later, the framework's empirical failure boundary has reproduced the Hodge partition from
the other side. This milestone formalizes that and re-poses the continuum failures under it.
(Note: the roadmap's earlier "M15 = dynamics as orbit flow" slot was fulfilled by P13–P16
under Milestone 14; this milestone takes the number with a new subject.)

---

## The Conjecture

**DFT-Hodge Conjecture.** *Every physically measurable invariant class has an algebraic
(PAC-tree / ADE) representative. Selection of a metric representative of a class is frame
data contributed by an observer (M13 definitional parallax); it is not derivable from the
PAC/SEC axioms, and demanding it without declaring the frame is ill-posed.*

Equivalently: the framework computes cohomology; observers supply gauge. The
algebraic/continuum score split (M11 100%, M12 94%, M14 91% vs metric-layer 25–85%) is not
a competence frontier — it is the Hodge decomposition of the theory's claims. Harmonic
(class-level) content survives; representative-level content fails exactly when posed
frame-free.

## The Lemma (already proven; restated)

**Lemma (M13.5 exp_14/exp_16).** Any isomorphism-invariant metric on ADE vertex sets is
positive-semidefinite with kernel exactly the orbit equivalence: same-orbit vertices have
identical complement spectra, hence zero distance, under every candidate tested (complement
spectra, heat kernels, characteristic polynomials, spectral zeta, combinations — 0/6
diagrams PD in all cases). The framework's metric layer is therefore *canonically a metric
on the orbit quotient* — a class-level (cohomological) metric. This is not a defect to fix
(M14 proved it fundamental); it is the structural fact the conjecture is built on.

## Failure reclassification

| Failure | Class content (passed) | Representative demand (failed) | Frame data required |
|---|---|---|---|
| Rapidity composition (M13 exp_08, errors 99–292%) | Per-step deformation along paths is well-defined | Chord (pairwise spectral distance) compared to arc (path sum) — pairwise distance is not a path metric | Choice of worldline: composition is a path notion; the chord is a different object |
| Coherence limit non-universal (M13.5 exp_15, 0/4) | Per-family rates well-defined; D-family converges (CV 0.051) | One universal limit pooled across families and parity classes (A-family even/odd mixed → CV 0.45) | Scope declaration: which family/parity class |
| PSD degeneracy (M13.5 exp_14/16, 0/4) | Orbit structure exact | A positive-definite vertex metric — impossible for isomorphism-invariant constructions (the Lemma) | Symmetry breaking: distinguishing same-orbit vertices requires non-invariant data |
| Positional interference (M14 exp_06, 1/4) | Orbit-space interference exact (T1: constructive 2.0, destructive ~0) | Vertex-space fringes from an orthogonal, delocalized orbit basis (cross-terms ≡ 0) | Aut-breaking perturbation: a representative position basis is a gauge choice |

## Evidence file (prior, independent)

- M11–M14 algebraic results: 91–100% (class-level claims).
- confluent_identity (March 2026, discrete Hodge on PAC fluid): conservation is
  **within-scope** (Δ-buffer reduces variance 16% per-parent vs 3.7% pooled); per-hop
  attenuation ≈ 1/φ; 2-hop ≠ product of 1-hops (each scope boundary transforms, not
  attenuates); the coupling "ceiling" is rank-compression, not geometry.
- Midnight invariant-rule ledger: 6/6 — registered relations survive, registered
  coordinates die (adopted as registration discipline 2026-06-11).

## What would falsify the conjecture

1. A continuum-layer failure that does **not** decompose as (class content passes) +
   (representative demand fails).
2. A frame-free derivation of a metric representative from the axioms (this would be
   *good news* for the framework and fatal for the conjecture).
3. Re-posed class-level claims failing where the original representative-level claims
   failed — the failures were never about frames at all.

## Experiments

| # | Script | Re-poses | Status |
|---|--------|----------|--------|
| 01 | `exp_01_rapidity_one_form.py` | M13 exp_08 — arc vs chord; affine holonomy as first curvature invariant | **3/4** — curvature confirmed |
| 02 | `exp_02_coherence_per_scope.py` | M13.5 exp_15 — per-class limits and their ratios | **0/3 KILLED** — boundary-dominated observable |
| 03 | `exp_03_representative_gauge.py` | M14 exp_06 — visibility under Aut-breaking gauge ε | deferred (M14 null is static) |
| 04 | `exp_04_holonomy_closed_form.py` | derivation-verification of θ(m) | **GATE PASS** — formula + theorems proven |
| 05 | `exp_05_general_k_limit.py` | general-k limit: K1 (odd-harmonic) vs K2 (Fibonacci) | **both dead** (L₃=5.491, L₄=11.186); F₆/F₄ coincidence retired; ℤ₂ even-telescoping universal |
| 06 | `exp_06_k56_confirmation.py` | momentum-generator derivation, k=5,6 pre-registered test | **CONFIRM** — G = box momentum matrix 4jj′/(j′²−j²), parity rule; L₅/L₆ hit to ~10⁻⁶; general-k limit SOLVED |

Pre-registration: `journals/2026-06-11_m15-exp01-03-preregistration.md`. Round-1 outcomes:
`journals/2026-06-11_m15-round1-outcomes.md`. All claims relational per the
invariant-registration rule.

## Phase-1 gate (CLOSED — Phase 2 opens)

The affine holonomy is **proven in closed form** (`journals/2026-06-12_holonomy_closed_form.md`,
verified by exp_04):

- θ(m) = m·θ_T(m) reproduces all ten round-1 measured angles exactly.
- **C₆ = −I is a theorem** (θ(6) = π exactly): the ℤ₂ frame-inversion twist is structural,
  not numerical.
- cos θ(C₄) = −7/9 derived from first principles (matches the measured exact rational).
- **Large-rank limit = 8/3 exactly** — the e candidate from the five-point extrapolation is
  ruled out (derivation-first policy earning its keep).
- Documented open threads (NOT gate blockers): the clean group-theoretic mechanism behind
  θ(m) = m·θ_T (the edge transports carry mixed orientation — reflections, det −1 — so the
  naive H = T^m is false; this reflection structure may be the edge-level face of the same
  ℤ₂); general-k behavior; the 8/3 = F₆/F₄ coincidence (reported [D], not claimed — the
  derivation is odd-harmonic, not Fibonacci).

**Phase 2 opens** on the strength of the proven formula and the C₆ = −I twist:
(a) ℤ₂ twist classification across cycle structures; (b) the field-equation hunt — does
cascade ledger density source holonomy, and is the coupling φ-structured? Registered
relationally when it begins.

**The foundational kill-sentence (standing):** *if holonomy is dynamically inert, it is
mathematics, not physics, and M15 caps at a reclassification.* Phase 2 exists to answer
exactly this.

## Core machinery (reused, not reimplemented)

`milestone13/core/identity_complement.py`: `complement_spectrum`, `vertex_orbits`,
`complement_deformation_rate`, `max_deformation_rate`. M14 orbit construction for exp_03.

## FDO Links

- `milestone-15-representative-problem` (to be created on first outcomes)
- `pac-series`, `midnight-observational-contact`, `hodge-conjecture-symbolic-collapse`
