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
| 01 | `exp_01_rapidity_one_form.py` | M13 exp_08 — arc vs chord; affine holonomy as first curvature invariant | registered |
| 02 | `exp_02_coherence_per_scope.py` | M13.5 exp_15 — per-class limits and their ratios | registered |
| 03 | `exp_03_representative_gauge.py` | M14 exp_06 — visibility under Aut-breaking gauge ε | registered (stretch) |

Pre-registration: `journals/2026-06-11_m15-exp01-03-preregistration.md`. All claims
relational per the invariant-registration rule.

## Core machinery (reused, not reimplemented)

`milestone13/core/identity_complement.py`: `complement_spectrum`, `vertex_orbits`,
`complement_deformation_rate`, `max_deformation_rate`. M14 orbit construction for exp_03.

## FDO Links

- `milestone-15-representative-problem` (to be created on first outcomes)
- `pac-series`, `midnight-observational-contact`, `hodge-conjecture-symbolic-collapse`
