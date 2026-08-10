# Maximal Coherence as the Connection Primitive

**Date:** 2026-06-13
**Status:** Working draft (confidence 0.6) — theoretical note, no result yet
**Origin:** Summary distilled from a separate Claude discussion, captured here so it does
not evaporate. Thread placement: a sibling of Thread 5 (Phase-Rate Primitive) — arguably
its own thread once it earns a result.
**Reviewed:** 2026-06-13 (see §8 Review Notes — three underweighted problems + the M14
redirection + a citation fix). The body below is the draft as received, with the single
factual citation error corrected inline and flagged.

---

## Collapsing the Connection Primitive and the Coherence Functional into One Object

## 0. Purpose and Scope

This document states a single structural move and the experiment that would convert it from an identity into a result.

The move: **a connection at full strength and two nodes at full coherence are the same state.** Coherence is not a quantity that travels across a pre-existing connection; coherence *is* the connection, and maximal coherence is the connection primitive itself. Everything else — partial connection, decoherence, relativistic offset between observers — is deformation away from that maximum.

This is intended as a restatement of the connection primitive at the level where it meets quantum mechanics. It folds the coherence functional into the primitive rather than treating them as two layers. It is self-contained: it does not depend on any external vault, codebase, or prior document to be read. Where it leans on earlier framework results, those results are restated inline so the document stands alone.

The scope is deliberately narrow. This is not a derivation of quantum mechanics. It is a claim about what *one object* is, plus a concrete discriminator that tells whether the claim is a genuine unification or a relabeling.

---

## 1. The Claim in One Paragraph

Take connection as the primitive: the substrate is geometry-of-connection, not objects-with-relations-painted-on. A node is a connection whose branches are not yet distinguished; an edge is a connection whose endpoints have been distinguished; they are the same primitive at different resolution. Now identify the *strength* of a connection with the *coherence* between its endpoints. Then "a connection exists at full strength" and "two endpoints are fully coherent" are not two facts — they are one fact. Coherence is the connection primitive read as a magnitude. Maximal coherence (value 1) is the undeformed primitive. Anything less than 1 is the primitive under deformation, and the rate of that deformation is what we elsewhere call proper time. Quantum coherence is then simply this same object **evaluated as deformation accrues moving forward** — and decoherence is the magnitude falling as differential accumulation piles up.

---

## 2. Background Restated (so the document stands alone)

Three prior results are load-bearing. They are stated here without external reference.

### 2.1 Connection as primitive

The bottom-of-stack object is connection, not the things connected. Nodes and edges are derived views of the same primitive seen at different resolution — a node is an undistinguished connection, an edge is a distinguished one. Identity is the **complement**: a node *is* everything in the connection graph that is not directly bound to it, viewed from where it sits. Thingness is the negative space — the relations that pass through a locus without terminating in it.

### 2.2 Relativity as complement-transformation; c as a coherence limit

Because identity is the complement, looking at the same locus from a different position in the graph yields a different complement-view — different deltas. This is definitional parallax, and the rule for transforming complement-views under a change of vantage is the relativistic transformation. The invariant speed `c` is not posited; it appears as a **coherence limit on the rate of definitional change** — the maximum rate at which complement-deformation can propagate. **Proper time is the complement-deformation rate** at a locus.

### 2.3 Potential is unresolved geometry; actualization is resolution, not creation

Potential is not a separate ledger from actuality. It is the *unresolved* portion of the same geometry. Actualization resolves part of it; it does not create anything. Foreclosed branches do not vanish — they **redistribute** across the available branch geometry (PAC conservation). This is the crucial property for what follows: because foreclosed structure redistributes rather than disappearing, accumulated structure is **not strictly monotone**. What was resolved one way can re-enter overlap if the geometry brings complements back into alignment.

---

## 3. The Move

### 3.1 Coherence is the magnitude of the primitive

Let coherence between two loci be a single functional over their two complements — a measure of how much of the surrounding connection-geometry the two loci share, evaluated from each one's vantage. Call it `C(i, j) ∈ [0, 1]`.

The claim is that `C` is not a property *of* a connection. It *is* the connection, expressed as a magnitude:

- `C = 1`: the undeformed connection primitive. The two loci share complement fully; there is no delta between their vantages on the shared geometry. This is the limit case where the connection is the primitive in its raw form.
- `0 < C < 1`: the primitive under deformation. The two complements have diverged by some amount; that divergence is the deficit `1 − C`.
- `C = 0`: no shared geometry; the loci are not connected in any sense the primitive recognizes. (Whether this is ever strictly reached or only approached is left open — see §6.)

Edge-existence, in this picture, is **thresholded coherence**. An "edge" is what you call a connection once `C` is high enough to resolve as a distinguished binding. This is why node and edge were the same primitive at different resolution: resolution *is* a coherence threshold.

### 3.2 Decoherence is the deformation rate applied forward

Now let deformation accrue. Each locus keeps accumulating — interacting, entangling with its surroundings, resolving branches. The complements drift. `C` falls.

This falling is decoherence. It is not a separate phenomenon layered on top of the connection; it is the connection magnitude `C` decreasing because differential accumulation is separating the two complements. The *rate* at which it falls is governed by the same complement-deformation rate that §2.2 calls proper time. *(Review note §8.2: this should be the **variance** sector of the deformation field, not proper time, which is the mean sector.)*

This is the precise content of "quantum coherence is connection applied moving forward." Static `C` is the connection. `C` evaluated as proper time accrues is quantum coherence. Decoherence is `dC/dτ < 0`. They are one object in three tenses.

### 3.3 The null-interval boundary case

Consider a photon. Along its worldline the interval is null — zero proper time accrues between emission and absorption. By §2.2, proper time *is* complement-deformation rate, so along a null interval **no complement-deformation accrues**. Nothing is added to separate the two endpoints' complements.

Therefore `C = 1` along the null interval. Emission and absorption are, informationally, the *same connection at full strength* — maximal coherence, perfect link, because no differential memory was laid down between them.

Note the asymmetry, which is the photon example doing real work rather than decoration: from the photon's own accounting `C = 1`, but a receiver who has accrued millions of years of *their own* complement-deformation reads the connection as partial. The relativistic offset between observer and photon is exactly the deficit `1 − C` as seen from the receiver's vantage. The relativistic factor and the decoherence term are the same quantity — complement-deformation — in two faces: one as the mean offset between vantages, one as the variance/erosion of shared complement. (These are distinct *properties* of the one deformation field, not statistical summaries of it.)

### 3.4 Revivals: why this survives at the quantum level

A naive "memory only grows" model would force `C` to be monotonically non-increasing: once complements diverge they stay diverged, decoherence is one-way. Quantum mechanics forbids this — interference, echoes, and revivals require coherence to be able to come *back*.

The framework permits revivals without an extra postulate, and this is precisely where §2.3 earns its place. Because foreclosed branches **redistribute rather than vanish**, accumulated structure can re-overlap: a branch that resolved one way and redistributed can re-cohere when the geometry brings the two complements back into alignment. So `C` can increase. `dC/dτ` is not sign-constrained. Revivals are re-overlap of complements driven by PAC redistribution.

This is the load-bearing check that keeps "quantum coherence is the *same* object" from collapsing into "quantum coherence is only the irreversible decoherence-dominated limit of the object." If accumulation were strictly additive, the identity would hold only in the dissipative limit. Because redistribution makes it non-monotone, the identity holds at the full quantum level.

---

## 4. What This Collapses

Stated as an inventory, because the shrinking of the primitive count is the point.

- **Connection primitive (M12)** and **the coherence functional**: one object. The primitive is coherence at its maximum; the functional is the primitive as a magnitude. *(Draft said M10; corrected — connection-as-primitive is M12. See §8.4.)*
- **Edge-existence** and **a coherence threshold**: one object. An edge is resolved coherence.
- **Decoherence** and **complement-deformation applied forward**: one object. `dC/dτ < 0`.
- **The relativistic offset between observers** and **the decoherence deficit**: two faces of one deformation field, consistent with the earlier `w(x)` phase-rate unification where relativity and decoherence are two properties of one primitive rather than two mechanisms.

The direction of all of these is the same and is the reason to take it seriously: the inventory of primitives is *shrinking* while coverage grows. The framework is absorbing distinctions other formulations have to posit separately, rather than adding structure to fit each phenomenon. A unification that pays for itself by erasing a real distinction eventually flattens something that is not flat and dies on a prediction. The discriminator in §5 is how this one is made to pay rather than merely please.

---

## 5. The Discriminator

The collapse in §3 is currently an *identity* — a clean way of saying the same thing. It becomes a *result* the moment a single functional `C` produces all three regimes without three separate hand-fitted parameterizations.

**Write `C(i, j)` once**, as one functional over two loci's complements. Then require, from that one expression:

1. **Null-interval boundary.** On a null interval (zero accrued complement-deformation), `C = 1`. The photon link is maximal with no special-casing.
2. **Decoherence curve.** Running PAC redistribution forward on two initially-coherent loci yields `C(τ)` falling with the correct functional form for decoherence — not assumed, *derived* from the redistribution dynamics. *(Review note §8.3: register this as a relational invariant — a decoherence-rate ordering/ratio across ADE types — not an absolute curve shape.)*
3. **Revival.** The *same* expression yields `C` increasing under re-overlap of complements, reproducing interference/echo/revival behavior, with no second mechanism bolted on.

If one functional does all three, then maximal-coherence-as-primitive is the correct restatement of the connection primitive, and the **photon link, the quantum coherence, and the graph edge are one object at three resolutions.** That is the claim earning its place in the stack.

If the three regimes need three different parameterizations of `C`, then "coherence is connection" is an analogy, not an identity: connection stays the primitive and coherence is one instantiation of it. That outcome is still informative — it tells you exactly where the seam is.

The test is sharp enough to fail, which is the only kind worth running.

---

## 6. Open Problems

- **Form of `C`.** The functional over two complements is not written here. It must be defined such that the three regimes in §5 fall out. This is the central piece of work. *(Review note §8.5: M14's orbit-Hilbert-space inner product is the natural candidate.)*
- **Whether `C = 0` is reached or only approached.** If complements always share *some* geometry, there is no true disconnection, only asymptotic isolation. This may matter for whether the graph is ever genuinely partitioned.
- **Phase representation.** Revivals are fundamentally about phase. The redistribution mechanism gives re-overlap, but it must be shown that the *phase* relationships in `C` track quantum phase, not merely magnitude. This is the most likely place for the identity to weaken into analogy. *(Review note §8.1: this is more central than a bullet — a real C ∈ [0,1] cannot produce interference; C must be complex.)*
- **Relating the deformation field's two faces.** The mean-offset face (relativity) and the erosion face (decoherence) are asserted to be properties of one field. The explicit relationship between them — how one constrains the other — is not yet written and would be the bridge back to the `w(x)` formulation.
- **Threshold for edge-resolution.** "Edge-existence is thresholded coherence" needs a principled threshold, or a reason the threshold is itself emergent rather than chosen.

---

## 7. Placement in the Stack

This belongs as a restatement at the M12 level (connection primitive) read forward through M13 (relativity as complement-transformation, proper time as complement-deformation rate) and into M14 (quantum mechanics from graph automorphisms). It does not introduce a new primitive; it identifies the existing primitive with maximal coherence and thereby pulls quantum coherence into the same object. The dependency reads:

```
M12  connection as primitive
  └── this doc: primitive = maximal coherence; coherence functional C is the primitive as magnitude
        └── M13  relativity as complement-transformation; proper time as complement-deformation rate
              └── M14  orbit Hilbert space L²(V/Aut(G)); interference from SEC complexification
                    └── C(τ): decoherence as dC/dτ < 0, revivals as re-overlap under PAC redistribution
                          └── next: define C (candidate: orbit overlap), run the §5 discriminator
```

The immediate next action is the only one that matters: **define `C` and run the three-regime discriminator.** Everything else in this document is staging for that one experiment.

---

## 8. Review Notes (2026-06-13)

Captured during the documenting discussion. The draft is sound and honestly framed, but it
contains **no established result yet** — all load is on "write C" — and it underweights three
problems while missing its most important neighbor (M14).

### 8.1 C cannot be a real magnitude in [0,1] — it must carry phase
Decoherence decays the magnitude of an off-diagonal element ρ_ij = |ρ_ij|·e^{iφ}, but
interference and revivals live in the **phase**. A non-negative scalar C ∈ [0,1] can give an
envelope decay; it physically cannot produce interference (which requires cancellation, hence
signed/complex amplitudes). §5 test 3 as written asks a magnitude to do what only a complex
quantity can. The phase source already exists in the framework: **M14 derives interference
from SEC complexification (A₁ → SL(2,ℂ)).** C must be complex from the start — |C| ∈ [0,1] is
connection strength, arg(C) is relative phase, C=1 means magnitude 1 *and* phase-aligned.

### 8.2 Decoherence is variance-driven, not proper-time-driven
§3.2 says dC/dτ tracks proper time; §3.3 correctly says the two faces are mean offset
(relativity) and variance/erosion (decoherence). These conflict, and Thread 5's isolated
fast atom settles it: near-c motion accrues large proper-time offset (mean) but **zero
decoherence** (no environment → zero variance). So C falls with the **variance** sector, not
proper time. Proper time (mean) sets the relativistic offset (the "1−C from the receiver");
variance sets the erosion. The null case is overdetermined — null interval kills the mean,
free propagation kills the variance → C=1. A real cosmological photon through the IGM accrues
variance (scattering) → C dips below 1, and *that dip is exactly the Thread 1 photon-archaeology
observable* (line width ↔ local SEC disequilibrium). This note's variance face is empirically
live, not just formal.

### 8.3 §5 is coordinate-flavored — violates the invariant-registration rule
Test 2's "correct functional form for decoherence" is an absolute-curve target — the exact
species of coordinate prediction that died six times in the exp_19–22 ledger. Per the
invariant-registration rule adopted 2026-06-11, register a **relation**: a decoherence-rate
ordering or ratio across ADE types, or revival-period ratios — not a curve shape.

### 8.4 Citation fix: connection-as-primitive is M12, not M10
The draft's front-matter tag, §2.1/§7, and §4 attributed connection-as-primitive to M10.
It is **M12** (M10 = symmetry self-application; M13 = identity-as-complement, cited correctly).
Corrected inline above.

### 8.5 The redirection that makes it real: C = M14 orbit overlap
The draft stops at M13 and never reaches M14 — its most important neighbor, because M14 already
built the object being hand-defined. M14 gives the orbit Hilbert space L²(V/Aut(G)); the natural
coherence is the inner product. Define **C(i,j) := ⟨i|j⟩** and most of §6 dissolves:
- **|C| ∈ [0,1] for free** (Cauchy–Schwarz) — no hand-picked threshold.
- **Phase for free** — complex inner product (resolves §8.1).
- **Revivals become a theorem, not a hope** — L²(V/Aut(G)) is finite-dimensional for a finite
  graph, so evolution is quasi-periodic and ⟨ψ(0)|ψ(τ)⟩ recurs (quantum recurrence theorem).
  Recurrence periods are set by the ADE eigenvalue spectrum (Coxeter numbers) → a **registrable
  invariant** (revival-period ratios across ADE types), exactly what §8.3 wants.
- **C=1 = state identity** — consistent with the §3.3 photon insight: a null interval literally
  identifies emitter and absorber as one locus (the Wheeler–Feynman intuition, forced).

The open theorem this surfaces is the right next problem: the draft's revival mechanism is PAC
**redistribution** (conservation); Hilbert recurrence is **unitarity**. M14 ties them
(unitary evolution = SEC arrow pre-measurement; measurement = gauge fixing). Showing
redistribution ⟺ unitary recurrence is the real result hiding in §3.4.

### 8.6 Bottom line
The move is sound (coherence-as-connection-magnitude is the natural identification once you are
in a relational quantum framework, not arbitrary). The genuinely novel content is the
**null-interval C=1 = endpoint-identity** insight. The path to a result is far less open than
§6 implies, because M14 already supplies the functional. Recommended next action if/when this
is picked up: set C = orbit overlap and test the three regimes as *relational* predictions.
