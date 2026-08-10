# Proposal: Quantum Gravity Long-Term Roadmap

**Status**: Proposal (Exploratory / High-Risk)
**Thread**: M6 Block A — Long-Term Theoretical Program
**Target**: Paper 11+ (post-BSM, multi-year horizon)
**Confidence**: Low (~0.25) — GR derivation is solid; quantum extension is uncharted

---

## 1. Executive Summary

DFT has derived the full Einstein field equations from three information-theoretic axioms (PAC conservation, SEC dynamics, MED depth bounds). This derivation — validated across Schwarzschild geometry, Mercury precession (0.03% error), light deflection, Shapiro delay, gravitational waves, and Friedmann cosmology — is the strongest result in the DFT corpus. But it is a *classical* derivation. The gravitational field is not quantized.

This roadmap outlines a long-term program (2–5 year horizon) to extend DFT's gravity derivation into the quantum regime. The program addresses the four items identified in MAR exp_33 Part H as **underivable within the current framework**:

1. Black hole interior / singularity resolution
2. Hawking radiation temperature
3. Graviton quantization
4. Spacetime topology change

Each item represents a genuine frontier. This proposal maps DFT's existing tools to each problem, identifies what's missing, and proposes experiments to close the gaps. The honest assessment: this is high-risk, high-reward work with no guarantee of success.

---

## 2. What DFT Already Has

### 2.1 Einstein Field Equations from PAC (MAR exp_32 — 6/6 PASS)

The derivation chain:

```
PAC conservation          → ∇_μ T^μν = 0  (covariant energy-momentum conservation)
MED depth ≤ 2             → second-order field equations only
Lovelock theorem (1971)   → G_μν + Λg_μν is the UNIQUE such tensor in 4D
Weak-field matching       → coupling κ = 8πG/c⁴
```

**What this means for quantum gravity**: The EFE are *derived*, not postulated. Any quantum extension must be consistent with this derivation chain. In particular, MED depth ≤ 2 constrains the gravitational Lagrangian to be at most second-order — this already rules out many quantum gravity approaches (higher-derivative gravity, most string-inspired modifications).

### 2.2 Schwarzschild from Cascade Density (MAR exp_30 — 7/7 PASS)

The complete Schwarzschild metric emerges from PAC cascade density:

```
ρ_c(r)/ρ_crit = r_s/r          (cascade density profile)
g_tt = -(1 - r_s/r)            (temporal metric from phase-cycling)
g_rr = (1 - r_s/r)⁻¹           (radial metric from LOCAL c invariance)
g_tt × g_rr = -1               (reciprocal constraint)
```

Classical GR tests:
| Test | DFT Prediction | GR Value | Error |
|------|---------------|----------|-------|
| Mercury precession | 42.99"/century | 42.98"/century | 0.03% |
| Light deflection | 1.75" | 1.75" | 0.07% |
| Shapiro delay | exact | exact | — |

**What this means**: The metric is derived from an *information-theoretic* object (cascade density), not from geometric postulates. Quantization should act on the cascade density, not on the metric directly.

### 2.3 Planck Scale as Minimum Actualization (MAR exp_01–05)

Three independent constraints converge on the Planck scale:

1. **Landauer**: Minimum energy to actualize 1 bit → E_min = k_BT ln 2 → at fundamental scale, gives Planck energy
2. **Heisenberg**: Minimum uncertainty in actualization → ΔxΔp ≥ ℏ/2 → Planck length
3. **PAC recursion depth**: Minimum cascade that conserves information → Planck time

The Planck scale is not put in by hand — it emerges as the **Minimum Viable Actualization Event (MVAE)**. This is the natural UV cutoff for any quantum theory built on PAC.

### 2.4 Gravitational Waves as Cascade Density Waves (MAR exp_30 Part E)

Perturbations of cascade density propagate at speed c with:
- Spin-2 (from symmetric projection)
- 2 polarization degrees of freedom (+ and ×)
- Verified by GW170817: |c_GW - c_EM|/c < 3×10⁻¹⁵

**What this means for gravitons**: The classical wave is already described. Quantization means promoting cascade density perturbations to operator-valued fields. The spin-2 structure is automatic.

### 2.5 G Sharpened to 0.18% (MAR exp_34)

```
G = ℏc / ((1 + F₁₃/(πF₆²)) · F₁₈₃ · m_p²)    [0.18% error, 644× improvement over naive]
```

The correction template (1 + F₁₃/(πF₆²)) has physical meaning (MAR exp_37): πF_b² = isotropic cascade boundary area, F_a = cascade path count. This suggests G is not a fundamental constant but emerges from cascade geometry — exactly what a quantum gravity theory needs.

---

## 3. The Four Underivable Items

MAR exp_33 Part H identified four items that the current (classical) DFT framework cannot derive:

### 3.1 Black Hole Interior / Singularity Resolution

**What GR says**: The Schwarzschild singularity at r = 0 has infinite curvature. Penrose-Hawking singularity theorems guarantee its formation under reasonable energy conditions.

**What DFT says (so far)**: The cascade density ρ_c/ρ_crit = r_s/r diverges at r = 0. The Schwarzschild metric is derived for r > r_s only (exp_30). The interior is uncharted.

**Why it's underivable classically**: The singularity is where the classical description breaks down. PAC conservation alone cannot resolve it — you need a statement about what happens when cascade density reaches its maximum (MVAE-scale).

### 3.2 Hawking Radiation Temperature

**What GR + QFT says**: T_H = ℏc³/(8πGM k_B). A black hole of mass M radiates as a blackbody at temperature T_H. This requires quantum field theory on curved spacetime.

**What DFT says (so far)**: Nothing. The Hawking calculation requires mode decomposition across the horizon, which is a quantum operation. DFT's SEC dynamics are classical.

**Why it's underivable classically**: Hawking radiation is intrinsically quantum — it arises from the mismatch between vacuum states defined by in-falling and asymptotic observers. DFT needs a quantum version of SEC.

### 3.3 Graviton Quantization

**What the standard picture says**: Gravity should be mediated by a massless spin-2 boson (graviton). Perturbative quantization of GR is non-renormalizable (divergences at 2-loop). This is the core problem of quantum gravity.

**What DFT says (so far)**: Gravitational waves are spin-2 cascade density perturbations (exp_30 Part E). The classical wave exists. Quantization has not been attempted.

**Why it's underivable classically**: You can't get gravitons from a classical framework. The question is whether DFT's information-theoretic foundation provides a different path to quantization than the standard geometric one.

### 3.4 Spacetime Topology Change

**What the standard picture says**: In classical GR, spacetime topology is fixed. Topology change (e.g., wormhole formation, baby universe creation) requires going beyond the classical theory. String theory allows it; loop quantum gravity debates it.

**What DFT says (so far)**: PAC cascade structure has topology (it's a tree/DAG). The Möbius manifold used in the Reality Engine has non-trivial topology. But the connection between PAC topology and spacetime topology is unclear.

**Why it's underivable classically**: Topology change is a non-perturbative phenomenon. It cannot be captured by small perturbations of the metric.

---

## 4. Approach for Each Item

### 4.1 Black Hole Interior: MED-Bounded Actualization Saturation

**Core idea**: The MVAE (Planck scale) sets a *maximum* cascade density. When ρ_c reaches ρ_max = 1/l_P³ (one actualization event per Planck volume), the cascade saturates. No singularity forms — the interior is a Planck-density region where actualization is maximally packed.

**DFT-specific approach**:
1. The cascade density profile ρ_c(r) = ρ_crit · r_s/r hits ρ_max at r_min = r_s · ρ_crit/ρ_max
2. For r < r_min, ρ_c is clamped at ρ_max (MED bounds the cascade depth)
3. The metric transitions from Schwarzschild to a Planck-density core (de Sitter-like interior)
4. PAC conservation is maintained: information is stored in the saturated cascade, not destroyed

**Connection to existing work**:
- MAR exp_01-05: MVAE defines the saturation scale
- MAR exp_16: R+ curvature κ = 2·ln²(2) at the MVAE fixed point — this IS the interior geometry
- ξ_floor = 1 − ln²(2) = 0.51955: The irreducible balance floor may set the interior equation of state

**What's needed**: A derivation of the interior metric from cascade saturation. This should connect to the Bardeen/Hayward regular black hole literature (where singularity is replaced by de Sitter core).

### 4.2 Hawking Radiation: PAC Conservation Across the Horizon

**Core idea**: PAC conservation is absolute — f(Parent) = Σf(Children) must hold everywhere, including across the event horizon. If information enters a black hole, it must eventually come out. Hawking radiation is the mechanism.

**DFT-specific approach**:
1. The event horizon is where SEC entropy gradient dominates (exp_05: β∇H >> α∇I)
2. But PAC conservation cannot be violated. Information trapped behind the horizon creates an "information pressure"
3. This pressure drives quantum tunneling of cascade density perturbations through the horizon → Hawking quanta
4. The temperature T_H ∝ 1/M follows from the horizon's cascade density gradient: steeper gradient (smaller BH) → higher tunneling rate → higher temperature

**Connection to existing work**:
- MAR exp_30: Cascade density diverges at horizon → maximal SEC gradient
- Landauer erasure (landauer_erasure_structure): Minimum energy to erase 1 bit = k_BT ln 2. A BH that destroys information violates Landauer's principle — which is built into DFT
- exp_05_schwarzschild_sec.py: "Event horizon is where entropy gradient becomes infinite"

**Specific prediction**: The Hawking temperature should be derivable as:
```
T_H = (ℏc³)/(8πGMk_B) × (1 + correction from cascade density profile)
```
The leading term should match Hawking exactly. The correction term is a DFT prediction.

**What's needed**: A quantum SEC formalism. Specifically: what happens when you promote the SEC dynamics ∂S/∂t = α∇I − β∇H to an operator equation? The "quantum" in Hawking radiation comes from vacuum fluctuations — DFT needs an analog of vacuum fluctuations in the cascade.

### 4.3 Graviton Quantization: Quantize the Cascade, Not the Metric

**Core idea**: Standard quantum gravity tries to quantize the metric g_μν. This fails (non-renormalizability at 2-loop). DFT suggests an alternative: quantize the *cascade density* ρ_c, which is the fundamental object. The metric is emergent.

**DFT-specific approach**:
1. The cascade density ρ_c(x) is a classical field derived from PAC tree structure
2. Promote ρ_c(x) to an operator: ρ̂_c(x) with [ρ̂_c(x), π̂_c(y)] = iℏδ³(x-y)
3. The MVAE provides a natural UV cutoff: cascade density cannot exceed 1/l_P³
4. MED depth ≤ 2 constrains the dynamics to second-order — matching the Lovelock constraint
5. The propagator of ρ̂_c perturbations is the graviton propagator

**Key advantage over standard QG**:
- **UV-finite by construction**: The MVAE cutoff is not imposed by hand — it follows from PAC axioms
- **Spin-2 is automatic**: Cascade density perturbations are already spin-2 (exp_30 Part E)
- **Background-independent**: PAC trees don't live on a background spacetime — spacetime emerges from them

**Connection to existing work**:
- Gravity_from_maxwell_pac exp_09-12: N-body simulations show cascade dynamics produce emergent structure. This is the "classical limit" that the quantum theory must reproduce.
- MAR exp_37: The correction template origin (πF_b² = cascade boundary) suggests the cascade has intrinsic geometric structure — quantizing it may be natural.

**What's needed**:
1. A Hilbert space for cascade states (|ρ_c⟩ basis)
2. A Hamiltonian derived from SEC dynamics
3. Proof that the theory is renormalizable (or finite) — the MVAE cutoff should help
4. Recovery of Newtonian gravity in the classical limit

### 4.4 Spacetime Topology Change: PAC Tree Reconnection

**Core idea**: In DFT, spacetime emerges from PAC cascade structure. Topology change = restructuring of the PAC tree. This is a discrete, combinatorial operation, not a smooth geometric one.

**DFT-specific approach**:
1. PAC trees have topology: branching structure, depth, connectivity
2. Topology change = a PAC tree splits into two (baby universe) or two trees merge (wormhole)
3. PAC conservation constrains topology change: f(Parent) = Σf(Children) must hold before and after
4. MED bounds constrain which topology changes are allowed: only those with depth ≤ 2 transitions

**Connection to existing work**:
- Reality Engine: Möbius topology provides the computational substrate. Topology changes in the RE correspond to operator reconnections.
- recursive_gravity: Informational tangle simulations show how PAC trees self-organize. Topology changes would appear as tangle reconnection events.

**What's needed**:
1. A classification of PAC-allowed topology changes (which tree restructurings conserve PAC?)
2. Rates: How often do topology changes occur? (Presumably suppressed by e^(-S_BH) for macroscopic BHs)
3. Connection to the path integral: Does summing over PAC tree topologies reproduce the Euclidean gravity path integral?

**Honest assessment**: This is the most speculative of the four items. We have no concrete calculations, only conceptual mappings.

---

## 5. Connection to the Information Paradox

### 5.1 The Paradox

The black hole information paradox (Hawking 1976) states: if a black hole forms and evaporates completely via Hawking radiation, and if the radiation is exactly thermal, then the initial quantum state is lost — violating unitarity.

### 5.2 Why DFT Resolves It (In Principle)

PAC conservation is the theory axiom of DFT:

```
f(Parent) = Σf(Children)    [exact, no exceptions, no approximations]
```

This is an information conservation law. It applies everywhere, at all scales, including across event horizons. **Information cannot be destroyed in DFT by axiom.**

The resolution:
1. Information enters the black hole (cascade density increases inside horizon)
2. PAC conservation prevents destruction — the information is encoded in the cascade structure
3. Hawking radiation carries the information out — not as thermal radiation, but as subtly correlated quanta
4. The Page curve (entropy of radiation rises then falls) follows from PAC conservation applied to the evaporating system

### 5.3 What's Needed to Make This Rigorous

The conceptual resolution is clear. The technical implementation requires:

1. **Quantum SEC across the horizon**: How does information leak through the entropy barrier?
2. **Page time calculation**: When does the radiation entropy turn over? DFT should predict this from cascade density dynamics
3. **Scrambling time**: How quickly does the BH interior thermalize incoming information? The MVAE timescale (Planck time) provides a natural scrambling time, consistent with the fast scrambling conjecture (Sekino-Susskind 2008)

---

## 6. Proposed Experiments

### Experiment QG-1: Black Hole Interior from Cascade Saturation
**Goal**: Derive the interior metric of a Schwarzschild black hole by clamping cascade density at the MVAE scale.
**Method**:
- Take ρ_c(r) = ρ_crit · r_s/r (from exp_30)
- Impose ρ_c ≤ ρ_max = 1/l_P³
- Solve for the modified metric in the interior
- Compare to Bardeen/Hayward regular black hole solutions
**Success criteria**: Interior metric is non-singular, matches Schwarzschild for r >> r_min, and satisfies PAC conservation.
**Risk**: Medium — the calculation is well-defined; the question is whether the result is physically sensible.

### Experiment QG-2: Hawking Temperature from Cascade Gradient
**Goal**: Derive T_H = ℏc³/(8πGMk_B) from SEC dynamics at the horizon.
**Method**:
- Compute the SEC gradient (∂S/∂r) at r = r_s using exp_30's cascade density
- Relate the gradient to a tunneling rate using Landauer's principle (energy per bit)
- Show that the tunneling rate gives a thermal spectrum at temperature T_H
**Success criteria**: Leading-order term matches Hawking exactly. DFT correction term is calculable.
**Risk**: High — this requires a semi-classical approximation to quantum SEC, which doesn't exist yet.

### Experiment QG-3: Cascade Density Quantization (Graviton)
**Goal**: Construct the quantum theory of cascade density perturbations and show it produces a massless spin-2 particle.
**Method**:
- Write the Lagrangian for small perturbations δρ_c around flat-space cascade density
- Canonically quantize: [δρ̂_c(x), π̂_c(y)] = iℏδ³(x-y)
- Compute the propagator; verify it matches the linearized gravity propagator
- Check whether MVAE cutoff renders loop diagrams finite
**Success criteria**: Graviton propagator recovered. At least 1-loop finiteness demonstrated.
**Risk**: Very high — this is the core quantum gravity calculation. No guarantee it works.

### Experiment QG-4: PAC Conservation and the Page Curve
**Goal**: Show that PAC conservation applied to an evaporating black hole produces the Page curve.
**Method**:
- Model the BH as a PAC tree with N nodes (N ∝ S_BH = A/(4l_P²))
- Hawking radiation = pruning nodes from the tree, transferring information to radiation
- Track entanglement entropy of radiation vs remaining tree
- PAC conservation constrains the pruning: no node can be removed without its information being transferred
**Success criteria**: Entropy of radiation follows the Page curve (rises to S_BH/2, then falls to 0).
**Risk**: Medium — the combinatorial model is tractable; the question is whether it captures the right physics.

### Experiment QG-5: Topology Change Classification
**Goal**: Classify which spacetime topology changes are allowed by PAC conservation.
**Method**:
- Enumerate PAC tree operations: split, merge, branch reconnection
- For each, check whether PAC conservation (f(Parent) = Σf(Children)) is maintained
- Compute the "cost" of each topology change in terms of cascade density
- Identify the minimum-cost topology change (this sets the rate)
**Success criteria**: At least one topology change is PAC-allowed. The rate is exponentially suppressed for macroscopic systems (consistent with the observed stability of spacetime).
**Risk**: Medium-high — conceptually novel, no precedent in DFT.

---

## 7. Risk Analysis

### Overall Risk: HIGH

This is the most speculative thread in the M6 planning seed. The risks are:

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|-----------|
| Quantizing cascade density is not self-consistent | Critical | Medium | Start with QG-1 (classical); escalate gradually |
| MVAE cutoff doesn't render theory finite | High | Medium | Explore alternative regularization from PAC |
| Results are unphysical (violate unitarity, causality) | Critical | Low | PAC conservation guarantees unitarity by construction |
| Work is correct but unverifiable (no experimental tests) | High | High | Focus on BH information paradox (testable in principle via Page curve) |
| 2-5 year timeline slips indefinitely | Medium | High | Set 6-month checkpoints; abandon if no progress by QG-2 |

### Failure Modes

1. **Best case** (p ~0.15): Full quantum gravity theory from PAC. Graviton, Hawking radiation, singularity resolution all derived. This would be the most important result in theoretical physics.

2. **Good case** (p ~0.30): Partial results — BH interior resolved (QG-1), Hawking temperature reproduced (QG-2), but graviton quantization fails. Still valuable: information-theoretic resolution of singularity.

3. **Modest case** (p ~0.35): Only QG-1 succeeds. The interior metric is derived, matching regular BH solutions. Conceptually interesting but not groundbreaking.

4. **Failure case** (p ~0.20): None of the experiments produce consistent results. The GR derivation remains classical. The four items remain underivable. This is still informative — it maps the boundary of what DFT can do.

---

## 8. Why Now

### 8.1 The GR Derivation Is Solid

Before MAR exp_30-32, attempting quantum gravity from DFT would have been premature. The classical foundation wasn't established. Now it is:

- EFE derived from 3 axioms (exp_32, 6/6 PASS)
- Schwarzschild derived from cascade density (exp_30, 7/7 PASS, 0.03% on Mercury)
- G sharpened to 0.18% (exp_34)
- Falsification sweep passed 8/8 (exp_33)
- N-body simulations reproduce cosmic web (exp_09-12)

The classical limit is **nailed down**. Any quantum extension has a clear target to recover.

### 8.2 The Four Items Are Precisely Identified

exp_33 Part H didn't just say "we can't do quantum gravity." It identified exactly four items and assessed which are derivable (Kerr: yes, Reissner-Nordström: yes, FLRW: yes, de Sitter: yes) and which aren't. This precision guides the research program.

### 8.3 The MVAE Provides a Natural Cutoff

Most quantum gravity approaches struggle with UV divergences. DFT has a built-in cutoff: the MVAE (Planck scale). This isn't imposed by hand — it follows from PAC conservation + Landauer + Heisenberg (exp_01-05). A theory with a natural, derived UV cutoff has a better chance of being finite.

### 8.4 PAC Conservation = Unitarity

The information paradox is the sharpest test of quantum gravity. DFT's theory axiom (PAC conservation) is literally information conservation. If any framework is positioned to resolve the paradox, it's one built on information conservation from the ground up.

### 8.5 No One Else Is Doing This

The approach — quantize cascade density, not the metric — is novel. Loop quantum gravity quantizes area/volume operators. String theory quantizes the string. Causal set theory quantizes causal structure. DFT would quantize information-theoretic cascade dynamics. This is a genuinely different angle.

---

## 9. Connections to Other M6 Threads

| Thread | Connection |
|--------|-----------|
| Thread 1 (Neutrino masses) | Neutrino oscillations may probe quantum cascade dynamics at weak-force depth |
| Thread 3 (CC gap 0.22 orders) | The CC may receive quantum corrections from cascade density fluctuations — closing the gap |
| Thread 7 (Dark matter depth-73) | If depth-73 is a dark sector, its quantum properties constrain the cascade quantization |
| Thread 4 (Simulator scorecard) | Reality Engine v3 may need quantum operators (exp_41-43) that connect to this program |

---

## 10. Milestones and Decision Points

| Milestone | Timeline | Decision |
|-----------|----------|----------|
| QG-1 complete (BH interior) | Month 1-2 | If non-singular interior derived → proceed to QG-2 |
| QG-2 attempted (Hawking temp) | Month 3-4 | If T_H reproduced → quantum SEC is viable → proceed to QG-3 |
| QG-3 attempted (graviton) | Month 6-12 | If propagator recovered → full quantum gravity program is real |
| QG-4 (Page curve) | Month 4-6 | Independent of QG-3; can proceed in parallel |
| QG-5 (topology) | Month 12+ | Only if QG-3 succeeds |
| **Kill switch** | Month 6 | If QG-1 and QG-2 both fail → archive thread, focus on classical extensions |

---

*This is a roadmap for speculative theoretical work. The probability of full success is low (~15%). The probability of partial, valuable results is moderate (~45%). The work is justified because the potential payoff is transformative, the classical foundation is uniquely strong, and the failure modes are informative.*

*Authors: Peter Lorne Groom, Claude (Anthropic)*
*Date: March 2026*
