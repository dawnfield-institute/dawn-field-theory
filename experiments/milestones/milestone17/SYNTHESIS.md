# M17 Synthesis — how criticality connects the stack

M17 was not opened by a new idea. It was opened by noticing that several established results
and several unresolved ones are the same statement in different vocabularies.

---

## The one claim

**A critical point is where the correlation length diverges. Under identity-as-complement, that
is where the scale on which identity is defined changes.**

Everything below is that sentence, seen from a different milestone.

---

## What already says it

### M13 — identity IS complement

A vertex's identity is the structure of the graph without it. That is a *local* statement while
ξ is finite: a node's complement is dominated by its neighbourhood. When ξ → L the complement
becomes the whole system and identity stops being local.

**M13 supplies the reason a diverging correlation length is an identity change rather than
merely a long-range correlation.** Without M13 this milestone would be ordinary critical
phenomena; with it, criticality is the identity boundary.

### `cellular_automata_pac_attractors` — Ξ sits at Class IV

The four rules nearest Ξ are all Class IV, p = 8.58×10⁻⁸, 42.7× enrichment, and 0/1000 random
rules land near Ξ.

Wolfram's classes are an identity taxonomy read as dynamics:

| class | dynamics | identity |
|---|---|---|
| II | frozen / periodic | too rigid to change |
| III | chaotic | dissolves |
| **IV** | complex, propagating structures | **persists AND transforms** |

Class IV is the edge of chaos. **The corpus has already measured Ξ at the boundary where
identity can morph without dissolving** — it simply was not phrased that way.

### `sec_threshold_detection` — Ξ at phase transitions

"ξ ≈ 1.0571 appears at phase transitions", cross-domain including Lorenz, p < 0.00001. The same
constant, in continuous dynamical systems rather than discrete computation.

### CAH — Ξ as a ceiling

*"The maximum sustainable computational asymmetry for closed recursive systems under PAC
conservation."* A maximum sustainable value is what a critical point is: past it the system
changes phase.

### M15 / M13.5 — the class/representative split

M13.5 found the coherence limit non-universal (exp_15, 0/4) and PSD degeneracy fundamental
(exp_16, 0/4). M15 reclassified these: class-level content passes, representative-level demands
fail.

That split is an identity-scale statement. A *class* is identity at one scale; a
*representative* is identity at another. **The boundary M15 found and the boundary criticality
describes may be the same boundary**, and Q4 of this milestone is where that gets tested.

---

## What it might close

### M10 thread 1 — φ^(−1/N) at N = 8, 3.3% off, correction underived

Near a critical point, finite-size corrections take a **universal form set by the critical
exponents**. A correction that is currently fitted becomes derivable — *if* the system sits near
criticality, which Block B decides.

### M10 thread 6 / M9 — the 8.9% slope gap, "finite-size noise or sub-leading physics?"

Finite-size scaling distinguishes exactly those two. This is not a reinterpretation of the gap;
it is the standard tool for the question already asked.

### M10 thread 2 — stochastic self-application

*"All M10 experiments are deterministic; genuine irreversibility needs stochastic
self-application."* Self-organized criticality requires noise to drive a system to its critical
point. Thread 2 and Q3 are one question.

### M10 thread 4 — "Ξ in nature: found in self-referential Markov chains but not plain random
walks. Where else?"

The milestone's central question, asked from the other side. The proposed answer: **at critical
points** — which is why self-referential chains show it and plain random walks do not.

---

## What it does not touch

Recorded so the synthesis is honest about scope:

- **M15 Phase 2** — the field-equation hunt and its standing kill sentence
- **Milestone R** — propagating the exp_24 energy-scale fix, the roadmap's "highest-value
  unfinished work"
- **The orbit-flow direction** — announced by M14, renumbered past by M15, never derived
- **M5** — CP violation at 3%
- **Observational contact** — Z′, Euclid, LISA, CTA, the DESI w(z) tension

---

## The methodological thread

Every result above was found with an instrument. The instruments are where the failures were.

Ten instrument faults across the two rounds preceding this milestone, all with one structure:
**the instrument reported agreement or structure where there was nothing to agree about.**
Box-counting gave D = 2.000 for a filament, a blob and scattered points alike. A checkerboard
passed a web test. A conservation scan certified a system that had stopped evolving. A crossing
finder took minimum spread and found "spread 0.000" in a saturated region.

A measurement that returns a number for any input returns a number for a meaningless input, and
*plausible* is indistinguishable from *measured* without a reference.

In experimental physics the apparatus is calibrated against nature. Here the system and the
instrument are both built, so there is no nature to calibrate against — unless a known-answer
system is constructed deliberately. That is why Block A is front-loaded, and why 2D site
percolation is the target: it is one of very few critical systems with exact results.

**And the eleventh fault was mine, in this very section.** The paragraph that stood here said
"exp_11's published web does not percolate" and proposed a corpus-wide survey on the strength
of it. It was wrong: that reading binned 4000 particles onto a 64³ grid, 0.015 particles per
cell, where a *deliberately connected* synthetic web also reads as disconnected. At exp_11's own
resolution the same run gives percolation 0.385 and `is_web=True`. See the retraction at the top
of the README.

The failure mode is worth keeping precisely because it inverts the section's own thesis. Every
fault above was *an instrument reporting structure where there was none*. This one was **an
instrument reporting no structure where there was some** — and it was more dangerous, because a
null reads as rigour. I spent hours treating a settled corpus result as falsified when the
defect was three lines from where I was looking.

**The order of suspects, when a fresh script contradicts something already settled:** my
implementation, my instrument, my reading of what the original actually claimed, a genuine
regime difference, and only then the established result. I inverted that order, and the cost was
a milestone founded on a wall that was never there.

The survey this section originally proposed — "which corpus results rest on a relational
property but were measured with a one-point statistic?" — is **not** justified by anything here.
exp_12, sitting beside exp_11, measured P(k) ~ k^−1.727, scale-free, cosmic similarity 0.849.
The corpus was not measuring the wrong things.
