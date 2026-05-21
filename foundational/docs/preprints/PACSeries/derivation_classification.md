# PACSeries Derivation Classification

**Purpose**: For every quantitative result in the PACSeries precision table, classify the derivation chain by how empirical input enters. This document is the honest backbone of the overview paper.

## Classification System

Each derivation step is one of:
- **(a) THEOREM**: Follows from mathematics alone. No physical input, no choice.
- **(b) UNIQUENESS**: A choice is made but justified by a uniqueness or minimality argument.
- **(c) EMPIRICAL**: A choice is made that matches observation. Justified post-hoc.

Each result is then classified:
- **Type A — Structural**: The result follows from the framework through a chain dominated by (a) and (b) steps. At most 1 empirical identification. The derivation direction runs framework -> result, not result -> framework.
- **Type B — Identified**: The result matches a clean expression, but the path involves 2+ empirical identifications or the expression was recognized in known values before being grounded in the framework.
- **Type C — Pattern-matched**: The formula was found by searching against known values, then justified post-hoc. The paper itself may acknowledge this.

The critical test: **would the derivation produce the same result if you didn't already know the target value?**

---

## Type A — Structural Results

These are the hardest to dismiss. The framework produces them through chains of theorems and uniqueness arguments. A skeptic must reject the axioms or the mathematics, not the methodology.

### A1. Lorentz Group SO(3,1) from ADE

| Step | Content | Type |
|------|---------|------|
| 1 | ADE classification is a theorem of mathematics | (a) |
| 2 | A_1 is the unique simplest ADE diagram | (b) |
| 3 | A_1 has Lie algebra su(2) | (a) |
| 4 | SEC complexification: su(2) -> su(2)_C = sl(2,C) | **(c)** — the single load-bearing physical postulate |
| 5 | sl(2,C) = so(3,1) as real Lie algebras | (a) |
| 6 | All 15 commutation relations verified to 1e-14 | (a) |

**Empirical steps: 1.** SEC complexification ("irreversibility requires complexification") is a physical assertion, not a theorem. But it is a single, clean postulate — not a parameter fit or a search result.

**Why Type A**: If you accept PAC -> ADE and SEC -> complexification, the Lorentz group is forced. You don't need to know what the Lorentz group is. The chain would produce SO(3,1) even if Minkowski had never written it down.

**Precision**: Exact (mathematical identity). Verified: 15/15 commutation relations to machine precision.

**Paper**: 10

---

### A2. Minkowski Metric Signature (1,3) from Killing Form

| Step | Content | Type |
|------|---------|------|
| 1 | Killing form B(X,Y) = Tr(ad_X ad_Y) on sl(2,C) has signature (3,3) | (a) |
| 2 | 4D vector representation of SO(3,1) selected | **(c)** — why 4D, not 6D adjoint? |
| 3 | Invariant bilinear form on 4D rep is unique by Schur's lemma | (a) |
| 4 | The unique form has signature (1,3) = Minkowski | (a) |
| 5 | 135 (vector, transform) pairs verified at 1e-5 | (a) |

**Empirical steps: 1.** The choice of 4D representation (step 2) matches known physics. Once that choice is made, Minkowski is forced.

**Why Type A**: The Killing form computation is a theorem. The 4D representation is the fundamental/defining representation — the natural first choice. A skeptic could object "why not adjoint?" but the framework answer (minimal faithful representation) is a clean uniqueness argument that borders on (b).

**Precision**: Exact (135/135 invariance checks). Selective: Euclidean, (2,2), and random metrics all fail.

**Paper**: 10

---

### A3. D_4 is the Only ADE Type with Non-Abelian Automorphisms

| Step | Content | Type |
|------|---------|------|
| 1 | Compute Aut(G) for all ADE Dynkin diagrams | (a) |
| 2 | A_n: Aut = Z_2 (reflection) for n >= 2, trivial for n=1 | (a) |
| 3 | D_n: Aut = Z_2 for n > 4 (branch swap) | (a) |
| 4 | D_4: Aut = S_3 (triality — all 3 branches equivalent) | (a) |
| 5 | E_6: Aut = Z_2, E_7: trivial, E_8: trivial | (a) |
| 6 | S_3 is the only non-abelian group that appears | (a) |

**Empirical steps: 0.** This is a pure theorem about finite graphs. Verified computationally but provable by hand.

**Why Type A**: No input from physics at all. The physical interpretation ("D_4 is the unique source of quantum uncertainty") requires an additional identification step (non-abelian Aut <-> non-commuting observables), but the mathematical fact is unconditional.

**Precision**: Exact. Holds for all ranks, not just <= 8.

**Paper**: 11

---

### A4. Bell Violation Requires Nontrivial Aut(G)

| Step | Content | Type |
|------|---------|------|
| 1 | Graphs with trivial Aut have every vertex as its own orbit | (a) |
| 2 | Orbit Hilbert space = vertex space, no quotient structure | (a) |
| 3 | Product graph orbits factorize trivially -> no entanglement | (a) |
| 4 | S <= 2 (classical bound) for trivially-automorphic graphs | (a) |
| 5 | D_4 (S_3 Aut) achieves S = 2sqrt(2) | (a) — Tsirelson's theorem on 2D orbit space |
| 6 | E_7, E_8 (trivial Aut) give S ~ 1.97-1.98 | (a) |

**Empirical steps: 0.** The topology-dependence of Bell violation is a theorem. No physical input needed.

**Caveat**: Achieving the full 2sqrt(2) requires choosing a maximally entangled state and an SU(2) rotation generator. In 2D orbit space, any traceless Hermitian generator suffices (trivially generates SU(2)), but the choice of Bell state is a modeling decision.

**Precision**: Exact algebraically. Topology-dependent: only graphs with non-abelian Aut violate.

**Paper**: 11

---

### A5. Cascade Clock Slope = 1/ln(phi)

| Step | Content | Type |
|------|---------|------|
| 1 | PAC recursion -> phi as unique stable eigenvalue | (a) |
| 2 | Phi-proportional cascade timing: level n takes phi^n time units | (b) — follows if PAC splitting is self-similar |
| 3 | N(t) = a + (1/ln(phi)) * ln(t) by inversion | (a) |

**Empirical steps: 0** (given the cascade model). The slope is determined by phi, which is determined by PAC.

**Caveat**: The cascade model itself (discrete levels with phi-proportional timing) is a framework assumption, not a theorem. But GIVEN that assumption, the slope is forced.

**Precision**: The slope 2.0781 is exact given phi. The intercept is fitted (see B5).

**Paper**: 9

---

### A6. ln(phi) = 0.4812 as PAC Decay Rate

| Step | Content | Type |
|------|---------|------|
| 1 | PAC recursion: Psi(k) = Psi(k+1) + Psi(k+2) | axiom |
| 2 | Characteristic equation x^2 = x + 1, stable root phi | (a) |
| 3 | Decay rate per level = ln(phi) | (a) |
| 4 | Only k=2 recursion gives ln(phi); all k-step recursions have effective depth 2 | (a) — proven in M3 exp_22 |

**Empirical steps: 0.** Given the PAC axiom with depth 2, ln(phi) is mathematically forced.

**Precision**: Exact (transcendental number, analytically determined).

**Paper**: 1

---

## Type B — Identified Results

These match clean expressions and have structural motivation, but the derivation involves 2+ identification steps where known physics guided the construction. A skeptic can ask: "would you have found this expression without knowing the answer?"

### B1. sin^2(theta_W) = 3/13 = F_4/F_7

| Step | Content | Type |
|------|---------|------|
| 1 | Fibonacci filter on gauge groups: N^2-1 must be Fibonacci | **(c)** — asserted, not derived |
| 2 | Only SU(2) (dim 3=F_4) and SU(3) (dim 8=F_6) pass | (a) |
| 3 | Total gauge+scalar content: 1+3+8+1 = 13 = F_7 | **(c)** — Higgs counted as 1 DOF (broken phase) |
| 4 | sin^2(theta_W) identified as F_4/F_7 = 3/13 | (b) — natural ratio given steps 1-3 |
| 5 | Running: 3/13 achieved at Q = 82.78 GeV, near M_W | (a) |

**Empirical steps: 2.** The Fibonacci filter (step 1) and the Higgs counting convention (step 3) are the empirical inputs. Given those, 3/13 is forced.

**Why Type B, not A**: The Fibonacci filter is the key bridging assumption. It is asserted ("PAC coherence requires Fibonacci generator count") but not proved from the PAC Lagrangian. If SU(5) had also passed the filter, GUT physics would have been invoked instead.

**Why Type B, not C**: 3/13 is not a search result. Given the generator counting, it is the unique answer. The identification sin^2(theta_W) = F_4/F_7 is the natural ratio of SU(2) generators to total content. You don't need to know 0.23122 to write it down.

**Precision**: 0.19% at M_Z. Exact at Q = 82.78 GeV (by construction of the RG running).

**Paper**: 4

---

### B2. Koide Formula Q = 2/3 = F_3/(F_3 + F_2)

| Step | Content | Type |
|------|---------|------|
| 1 | PAC recursion levels indexed by k | axiom |
| 2 | Charged leptons assigned to k=3 | **(c)** — "shallowest fermionic sector" |
| 3 | Q = F_3/(F_3+F_2) = 2/3 | (a) — trivial arithmetic |
| 4 | Matches measured Q = 0.666661 | verification |

**Empirical steps: 1.** The assignment of leptons to depth 3.

**Why Type B**: The identity 2/3 = F_3/(F_3+F_2) is trivially true. The nontrivial claim is the mapping. The Koide ratio Q ~ 2/3 was known before this framework. The framework provides a structural reason (recursion depth) for a previously unexplained numerical coincidence.

**Precision**: 0.5 ppm (but note: this measures how close the Koide ratio is to 2/3, which was known independently).

**Paper**: 4

---

### B3. Xi = gamma + ln(phi) = 1.0584

| Step | Content | Type |
|------|---------|------|
| 1 | ln(phi) derived from PAC | (a) — see A6 |
| 2 | gamma imported from classical mathematics | **(c)** — not derived from DFT |
| 3 | Sum Xi = gamma + ln(phi) identified as "balance constant" | **(c)** — the sum is observed, not derived from a single equation |
| 4 | Five domains converge within 0.12% | **(c)** — domains are selected, not predicted |

**Empirical steps: 3.** gamma is imported. The sum is empirical. The domains are selected.

**Why Type B, not C**: ln(phi) IS derived (Type A). The sum Xi = gamma + ln(phi) has structural motivation (discrete cost + continuous cost). But 21 other constant combinations fall within 5% of Xi, and 1/sqrt(3) performs comparably to gamma. The papers are honest about this.

**Precision**: 0.12% spread across 3 independent domains (excluding the tautological analytic value and the phi-wired Mobius simulation).

**Paper**: 2

---

### B4. H0 Ratio = phi^{1/6} = 1.0835

| Step | Content | Type |
|------|---------|------|
| 1 | phi from PAC | (a) |
| 2 | H0_local / H0_CMB = phi^{1/N} | **(c)** — functional form not derived |
| 3 | N = 6 at current epoch | **(c)** — from cascade clock (fitted) |

**Empirical steps: 2.** The paper itself classifies this as **(D) postdiction**: "phi^{-1/6} correction was derived AFTER seeing the Hubble tension data."

**Why Type B, not C**: The value phi^{1/6} is a clean, parameter-free expression (given N=6). It wasn't found by searching over expressions — it follows from the cascade model. But it was constructed after seeing the tension.

**Precision**: 0.075% (0.05 sigma from SH0ES).

**Paper**: 9

---

### B5. S8 Tension Resolution (3.22 sigma -> 0.07 sigma)

| Step | Content | Type |
|------|---------|------|
| 1 | phi from PAC | (a) |
| 2 | Clock slope = 1/ln(phi) | (a) — see A5 |
| 3 | Dissipation per level = 1/phi^2 | **(c)** — asserted, not derived |
| 4 | Clock intercept a = 1.360 | **(c)** — fitted to 3 data points |
| 5 | S8 is one of the 3 data points used for fitting | circularity concern |

**Empirical steps: 2 + circularity.** The leave-one-out test (fit to Hubble+JWST only, predict S8 blind) is the honest check. The paper reports it passes, but doesn't state the leave-one-out precision.

**Why Type B**: The cascade clock slope IS derived (Type A). The mechanism (phi-power dissipation reducing clustering) is physically motivated. But the intercept is fitted and the dissipation rate 1/phi^2 is a modeling choice.

**Key finding**: t1 = 520 Myr is NOT an independent anchor. It equals exp(-a/slope) = exp(-1.360/2.0781), derived from the fitted intercept. The coincidence with first-star formation is post-hoc.

**Precision**: 0.07 sigma (full fit, with circularity). Leave-one-out precision: not stated.

**Paper**: 9

---

### B6. Alpha_EM to 5.7 ppm

| Step | Content | Type |
|------|---------|------|
| 1 | PAC -> phi -> Fibonacci | (a) |
| 2 | Fibonacci filter -> SU(2) x SU(3) | **(c)** — see B1 |
| 3 | F_7 = 13 closure | **(c)** — Higgs counting |
| 4 | Hierarchy depth F_10 = 55 | **(c)** — not derived; "appears in Feigenbaum" |
| 5 | Formula template with correction factor | **(c)** — functional form not derived |

**Empirical steps: 4.** The template, the depth, and both gauge-structure assumptions enter.

**Why Type B, not C**: The pair (F_10, F_7) is 2870x better than all alternatives — exhaustive uniqueness test. The formula reuses the same Fibonacci indices as other results (F_4, F_7, F_10 recur across the series). The paper's Section 12.5 shows p = 0.42 for the template alone, but joint significance with sin^2(theta_W) is much stronger.

**Borderline B/C**: Honest assessment — this is the weakest Type B entry. The formula template was likely guided by the target value.

**Precision**: 5.7 ppm.

**Paper**: 4

---

## Type C — Pattern-Matched Results

These were found by searching against known values. The papers are honest about this. The significance is in the joint system (the same small set of Fibonacci numbers recurring), not in individual formulas.

### C1. Feigenbaum delta (rational formula, 8 digits)

delta = (50050 + 32*pi) / (10725 + 5*pi). Found by exhaustive search over integer triples. Post-hoc Fibonacci decomposition of 3575 = 55*65. **4 empirical steps.** The paper states: "conjectured closed forms, not theorems."

### C2. Feigenbaum r_infinity (13 digits)

Constructed by exhaustive search over 3.9M triples. The integers 55, 17, 52 selected because they give the best match. Fibonacci interpretations noted post-hoc. **5 empirical steps.** The paper states: "We make no claim about WHY Fibonacci numbers appear."

### C3. Feigenbaum alpha (6 digits)

|alpha| = (2700 + pi)/1080. The paper acknowledges this is "the weakest" result. Entirely pattern-matched. **3 empirical steps, all of them.**

### C4. Feigenbaum delta (self-closing, 13 digits) — WITH UPGRADE NOTES

delta = phi^{20/N}, N = sqrt(39 + 1/x), x = 160 + (delta-4)^2 * (1 - 1/(1371+delta-4)).

As published: **4 empirical steps** (ansatz, 39, 160, 1371). Constructed to match.

**Post-publication upgrade (exp_04-07, this week)**:
- F_10 = 55 uniqueness: PROVED by CRT + growth obstruction (THEOREM)
- 39 = phi(5)*phi(11) - 1: CRT-derived (THEOREM)
- 160 = phi(5)*phi(55): CRT-derived (THEOREM)
- 1371 = F_5^3*L_5 - phi(5): CRT-derived (THEOREM)
- The ansatz delta = phi^{20/N} motivated by Mobius eigenvalue phi^20 (THEOREM)

**Upgraded status**: The structural constants are now explained, not just fitted. But the formula TEMPLATE (the specific functional form connecting phi^20 to delta via N, sqrt, and self-reference) remains empirical. And the bridge (WHY Mobius eigenvalue = RG eigenvalue) remains open.

**Upgraded classification**: Type C -> **Type B-** (strong theorem support underneath, but the template is still empirical and the bridge is unproven).

### C5. Mass Ratios (m_mu/m_e, m_tau/m_e, m_p/m_e)

The paper explicitly states: "found by systematic search through products and ratios of Fibonacci numbers, then validated against measured values. They are pattern-matches — not predictions derived from first principles." **All steps empirical.**

### C6. Mixing Angles (Cabibbo, PMNS theta_12, theta_13, theta_23)

All index assignments guided by measured values. The paper does not derive why theta_12 uses (F_3, F_4) while theta_13 uses (F_3, F_7). **All steps empirical.**

### C7. Cosmological Constant (log10 = -122.09)

Lambda ~ phi^{-2N} with N from measured Planck/Hubble scales. Correction template with indices (3,5,4) selected from multiple options. **3/4 steps empirical.**

### C8. Dark Matter Mass (6.44 keV)

Two routes averaged. Depth 73 from cyclotomic polynomial (constrained but not unique). Exponent d/2 not derived. Anchor masses are measured inputs. **5/7 steps empirical.**

### C9. Z' Boson at 395 GeV

M_Z * F_7/F_4 = 91.19 * 13/3. Pattern extrapolation: "the next Fibonacci ratio up from Z." **2/3 steps empirical.**

### C10. JWST Floor Predictions

(1/phi) * f_PS * exp(-z/z_cascade). Model-constructed formula. **4/5 steps empirical.**

---

## Summary Table

| # | Result | Paper | Type | (a) | (b) | (c) | Precision | Would survive blind derivation? |
|---|--------|-------|------|-----|-----|-----|-----------|-------------------------------|
| A1 | Lorentz SO(3,1) | 10 | **A** | 5 | 1 | 1 | exact | Yes — SEC complexification produces it |
| A2 | Minkowski (1,3) | 10 | **A** | 3 | 0 | 1 | 135/135 | Yes — Killing form determines it |
| A3 | D_4 non-abelian | 11 | **A** | 6 | 0 | 0 | exact | Yes — pure theorem |
| A4 | Bell needs Aut | 11 | **A** | 6 | 0 | 0 | exact | Yes — pure theorem |
| A5 | Clock slope | 9 | **A** | 2 | 1 | 0 | exact | Yes — phi determines it |
| A6 | ln(phi) | 1 | **A** | 3 | 0 | 0 | exact | Yes — PAC determines it |
| B1 | sin^2(theta_W) | 4 | **B** | 2 | 1 | 2 | 0.19% | Maybe — Fibonacci filter is key assumption |
| B2 | Koide = 2/3 | 4 | **B** | 1 | 0 | 1 | 0.5 ppm | No — depth assignment needed |
| B3 | Xi = 1.0584 | 2 | **B** | 1 | 0 | 3 | 0.12% | No — gamma is imported |
| B4 | H0 ratio | 9 | **B** | 1 | 0 | 2 | 0.075% | No — postdiction (paper admits) |
| B5 | S8 resolution | 9 | **B** | 2 | 0 | 2+ | 0.07 sigma | Maybe — leave-one-out passes |
| B6 | alpha_EM | 4 | **B-** | 1 | 0 | 4 | 5.7 ppm | No — template guided by target |
| C1 | delta (rational) | 3 | **C** | 1 | 0 | 4 | 8 digits | No — exhaustive search |
| C2 | r_infinity | 3 | **C** | 2 | 0 | 5 | 13 digits | No — exhaustive search |
| C3 | alpha | 3 | **C** | 0 | 0 | 3 | 6 digits | No — pattern match |
| C4 | delta (self-closing) | 3 | **B-** | 4 | 0 | 4 | 13 digits | Partially — constants now explained (exp_04-07) |
| C5 | Mass ratios | 4 | **C** | 0 | 0 | all | 5-350 ppm | No — paper says "systematic search" |
| C6 | Mixing angles | 4 | **C** | 0 | 0 | all | 0.05-4 deg | No — index assignments from data |
| C7 | CC | 9 | **C** | 1 | 0 | 3 | 0.09 orders | No — model constructed |
| C8 | DM mass | 9 | **C** | 1 | 0 | 5 | ~0.1 orders | No — model constructed |
| C9 | Z' at 395 GeV | 9 | **C** | 1 | 0 | 2 | prediction | No — pattern extrapolation |
| C10 | JWST floor | 9 | **C** | 1 | 0 | 4 | prediction | No — model constructed |

---

## What This Classification Reveals

### The framework has three distinct strengths:

**1. Structural derivations (Type A)**: 6 results. These are genuine mathematical consequences of the axioms. The Lorentz group, Minkowski metric, D_4 uniqueness, Bell topology-dependence, and ln(phi) all follow from chains of theorems. These would survive peer review by mathematicians. They are the foundation.

**2. Identified patterns (Type B)**: 6 results. These match clean expressions with structural motivation, but the path from axioms to numbers involves identification steps guided by known physics. sin^2(theta_W) = 3/13 is the strongest — given the Fibonacci filter, it's forced. alpha_EM at 5.7 ppm is the weakest — the template was likely guided by the target. The cascade clock has a clean slope but a fitted intercept.

**3. Searched expressions (Type C)**: 10 results. These are honest pattern matches. The papers say so explicitly. The significance is in the JOINT system — the same small set {F_3, F_4, F_5, F_6, F_7, F_10} appearing across independent physical domains — not in individual formulas.

### The look-elsewhere problem by tier:

- **Type A**: Not applicable. These are theorems. There is no search.
- **Type B**: Moderate. The Fibonacci filter admits ~2 gauge groups, giving ~1 natural ratio. The look-elsewhere effect is bounded by the number of reasonable identification schemes, which is small (maybe 10-50). Joint significance across B1-B6 is real.
- **Type C**: Severe for individuals. With {phi^n, F_n, L_n, pi, gamma, sqrt(k), +, -, *, /, ^} the expression space for any 6-digit target is enormous. The Monte Carlo in Paper 4 (0/10,000 random combinations match all three mass ratios simultaneously) bounds the joint look-elsewhere, but individual Type C results have p >> 0.01.

### What the papers get right:

The papers are unusually honest. Paper 3 says "conjectured closed forms, not theorems." Paper 4 says "We do not derive the Standard Model from PAC." Paper 9 labels postdictions with (D). Paper 10's SYNTHESIS.md lists what M13 does NOT claim. This self-awareness is the strongest evidence that the framework is physics and not numerology — numerology never publishes its failure modes.

### What the overview paper should do:

1. Present the Type A results FIRST. These are the structural foundation. They don't need precision tables — they need derivation chains.

2. Present the Type B results as "structural patterns with identification gaps." Be explicit about which step is the identification. For sin^2(theta_W), it's the Fibonacci filter. For alpha_EM, it's the template.

3. Present the Type C results as "empirical patterns consistent with the framework." Do NOT present them as derivations. Present the joint significance (same indices recurring) as the evidence, not individual precisions.

4. **Never let a Type C result be the headline.** "13 digits of Feigenbaum" sounds devastating, but it's a search result. "D_4 is the only ADE type with non-abelian automorphisms" is less flashy but unassailable.

5. The self-closing Feigenbaum formula (C4) deserves a special note: it was Type C as published, but the exp_04-07 strengthening moves it toward Type B. The structural constants are now CRT-derived (theorems), F_10 uniqueness is proved, and the Mobius eigenvalue phi^20 is exact. The remaining gap (why Mobius eigenvalue = RG eigenvalue) is precisely stated and mathematically well-posed.

---

## The Honest Scorecard

| Tier | Count | Character | Peer review survival |
|------|-------|-----------|---------------------|
| Type A | 6 | Theorems with 0-1 postulates | High — math is checkable |
| Type B | 6 | Patterns with 1-4 identifications | Medium — depends on accepting bridging assumptions |
| Type C | 10 | Search results honestly labeled | Low individually, moderate jointly |

The series stands or falls on the Type A results. If the Lorentz derivation, D_4 theorem, and Minkowski metric survive scrutiny, the framework has proven it can produce physics from graph theory. The Type B results then become "interesting if the framework is taken seriously." The Type C results become "consistent patterns worth investigating."

If the Type A results are wrong or trivial, the Type B and C results are sophisticated numerology.

**The framework's actual achievement**: It produces 6 theorem-level results connecting ADE graph theory to known physics (Lorentz group, Minkowski metric, D_4 quantum uniqueness, Bell topology-dependence, phi decay rate, cascade clock slope). These are genuine mathematical contributions regardless of whether the broader framework is correct. Everything else is evidence that the framework might extend further — suggestive, but not conclusive.
