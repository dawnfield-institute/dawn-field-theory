# 2026-08-16: Founding — and a survey of what the stack left open

Two parts. Why the milestone exists, and a sweep of the loose ends across the corpus that
converge on it — including the capability holes that have to be plugged before the reality
engine is worth returning to.

---

## Part 1 — Why

Four independent routes reached the same wall inside one round:

| route | percolation | ξ |
|---|---|---|
| v3 field engine, ~40 runs, 4 grids, 3 tick counts | 0.007–0.019 | 0.63 (noise floor) |
| v4 particle substrate, attractive convention | 0.425, but spanning only 0.47 of the box | — |
| **exp_11 — the corpus's own published 3D web** | **0.0068** (noise control 0.0025) | — |
| exp_10's convention, swept 0.3 → 2.5, 5 seeds | 0.011–0.017 at **every** setting | — |

The recognition that founded this milestone: **ξ diverges at criticality.** Every one of those
runs measured ξ at the white-noise floor. That is not "no structure" — it is *maximally
sub-critical*, and all four routes were measuring distance from a critical point without naming
it.

### The identity connection is technical, not metaphorical

At a critical point ξ → L. Under M13's identity-IS-complement, a node's complement then stops
being local and becomes global — **the scale on which identity is defined changes**. In cluster
language: below p_c, finite clusters; at p_c, the largest spans the system; a thing stops being
*a clump* and becomes *the network*. Identity changes category, not degree.

Wolfram's classes say the same thing about persistence: Class II is frozen (identity too rigid
to change), Class III is chaotic (identity dissolves), **Class IV is where structures persist
and transform**. Gliders hold together while propagating and interacting.

### The corpus already found Ξ there — twice

- `cellular_automata_pac_attractors`: the four rules nearest Ξ are **all Class IV**,
  p = 8.58×10⁻⁸, 42.7× enrichment, 0/1000 random rules near Ξ. Rule 110 at 1.0579.
- `sec_threshold_detection`: *"ξ ≈ 1.0571 appears at phase transitions"*, cross-domain
  including Lorenz, combined p < 0.00001.
- CAH: Ξ as *"the maximum sustainable computational asymmetry"* — a ceiling, which is what a
  critical point is.

**Ξ is established as the critical point in computation and in dynamical systems. It has never
been checked in a structure-forming system.**

### And the reason it was never checked

exp_10 concluded *"NO discrete phase transition"* from density CV, void fraction, filament
fraction and a sampled clustering coefficient. **None of those is an order parameter.** A phase
transition is invisible to one-point statistics; it lives in the correlation length, the
cluster-size distribution and the susceptibility.

So the status is not "the framework lacks criticality in structure formation" — it is
**criticality was never looked for**. That is a capability hole, not a physics result.

---

## Part 2 — Loose ends across the stack

Surveyed from `ROADMAP.md`, milestone READMEs and study open-question sections. Grouped by
whether this milestone touches them.

### Directly addressed — candidates for closure

| where | what's open | why criticality bears on it |
|---|---|---|
| **M10 thread 1** | φ^(−1/N) converges to φ but N=8 is **3.3% off, correction underived** | Near a critical point, finite-size corrections take a *universal* form set by the critical exponents. Derivable rather than fitted. |
| **M10 thread 6 / M9** | the **8.9% slope gap** — "finite-size noise, or sub-leading physics?" | Same tool answers it. Finite-size scaling distinguishes the two directly. |
| **M10 thread 2** | all M10 experiments deterministic; *"genuine irreversibility needs stochastic self-application"* | This is the self-organized-criticality question. SOC needs noise to drive a system to its critical point. |
| **M10 thread 4** | *"Ξ in nature — found in self-referential Markov chains but not plain random walks. Where else?"* | The milestone's central question, asked from the other side. |
| **M16** | percolation floor across four routes | The empirical entry point. |
| **exp_10** | "no phase transition", concluded from one-point statistics | Re-askable with an order parameter. |

### Adjacent — informed but not closed

| where | what's open |
|---|---|
| **M13.5** | Coherence limit **not** universal (exp_15, 0/4); PSD degeneracy proven fundamental (exp_16, 0/4). M15 reclassifies these as class-level pass / representative-level fail — the identity-level statement of the same boundary. |
| **M6** | exp_03 T2 at R² = 0.67 vs a 0.75 threshold, "genuine scatter in geometric decay". Critical fluctuations produce exactly that kind of scatter — *plausible but unforced*, and not a claim. |
| **exp_31** | **Open question #1**: what bridges the local-exponential model to the Gauss/cascade model *across scales*? A critical point is a scale-free regime, which is one thing a scale bridge could be. |
| **exp_36** | *"Can cosmic web structure be quantitatively predicted from the tiling pattern?"* Percolation on the tiling is the natural form of that question. |

### Untouched by this milestone, still open

Recorded so the survey is honest about what it does *not* address:

- **M15 Phase 2** — the field-equation hunt, standing kill sentence. Nearest registrable target
  is Ξ = 1 + π/55 as a ratio of momentum-operator spectra.
- **Milestone R** — propagate the exp_24 energy-scale fix back through exp_03–09. Described in
  the roadmap as *"the highest-value unfinished work in the corpus"* and unrelated to this.
- **The orbit-flow direction** — announced by M14, renumbered past by M15, never derived. The
  two documents still disagree.
- **M5** — CP violation at 3%.
- **Observational contact** — Z′, Euclid, LISA, CTA, and the DESI w(z) tension carried as a
  possible falsification.
- **Publication repackaging** — 11 packages `needs_repackage`.

---

## Part 3 — The capability holes

What is actually missing is instrumentation, not theory:

| instrument | status |
|---|---|
| correlation length, per axis, N-D | **exists** — `reality-engine/.../structure.py` |
| percolation with occupancy, N-D | **exists**, calibrated |
| finite-size scaling — the ξ/L crossing | **missing** |
| cluster-size distribution + power-law fit | **missing** |
| susceptibility / order parameter | **missing** |
| critical exponent extraction | **missing** |
| edge-of-chaos classifier as a reusable instrument | **partial** — classification exists inside `cellular_automata_pac_attractors`, not as a shared tool |

**Every one gets calibrated against 2D site percolation before use.** p_c = 0.592746 with exact
exponents β = 5/36, γ = 43/18, ν = 4/3. Unlike most calibration targets this one has textbook
answers, so the instruments can be validated rather than trusted.

That gate is not optional. The preceding round turned up **seven** instrument faults — a
box-counting estimator returning D = 2.000 for a filament, a tautological filament fraction, a
checkerboard and a speckle field both scoring `is_web = True`, a force fitter blind in
many-body, a conservation scan certifying a dead system, and percolation swinging 18× with
binning resolution. Not one was caught by a statistic taken at face value.

## Next

exp_01: finite-size scaling on 2D site percolation — pure instrument calibration against a
system whose critical point and exponents are known exactly. No DFT system is measured until
that passes.
