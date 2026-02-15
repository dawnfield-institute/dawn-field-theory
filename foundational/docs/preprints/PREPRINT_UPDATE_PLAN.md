# Preprint Update Plan: PACSeries Consolidation

**Date**: February 12, 2026  
**Status**: Planning Document (Updated with synthesis session results)  
**Goal**: Consolidate the PACSeries into a self-contained paper series that establishes Dawn Field Theory through clean derivation, measurement, and honest separation of established results from speculation.

---

## Context

Dawn Field Theory explores whether information and entropy are generative foundations of reality (Infodynamics), with PAC, SEC, and MED as the mathematical framework.

- **PAC** (Potential-Actualization Conservation): f(Parent) = Σf(Children)
- **SEC** (Symbolic Entropy Collapse): ∂S/∂t = α∇I - β∇H
- **MED** (Macro Emergence Dynamics): depth ≤ 2, nodes ≤ 3

The PACSeries is the primary publication vehicle. The existing standalone preprints (golden_ratio_prime_distribution, cellular_automata_xi_clustering, etc.) remain as supporting publications and will be updated to reference the PACSeries for foundational derivations.

### Why consolidate into PACSeries

The current PACSeries (5 papers, October 2025) was written before most experimental results existed. Three of five papers draw from the same GAIA run interpreted differently. Since then, the Landauer decomposition, Feigenbaum closed forms, mass ratios, cascade dynamics, and the γ + ln(φ) derivation have closed the loop. The series needs to reflect what actually exists now.

### Integration with original PACSeries

The original 5 papers (October 2025) have traction and readership. Rather than letting the old and new series exist as two disconnected narratives, we integrate them into one coherent story:

- **3 papers merge** into the new PACSeries (their content strengthens new papers that cover the same ground more rigorously)
- **2 papers get status-update appendices** (pointing readers to the new derivations while preserving the originals as intellectual provenance)

This reduces 11 papers (5 old + 6 new) to 8 papers (2 updated originals + 6 consolidated new papers). Readers of the originals get a clear path to the mature work; readers of the new papers get the fuller evidence base.

| Original Paper | Action | Target |
|----------------|--------|--------|
| xi_bounded_invariant | Status update appendix | → New Paper 2 |
| sec_med_framework | Status update appendix | → New Papers 1-6 |
| gaia_computational | **Merge** | → New Paper 6 Part A |
| relativistic_mas | **Merge** | → New Paper 6 (0.020 Hz validation) |
| mobius_confluence | **Merge** | → New Paper 5 (temporal emergence) |

### Voice and tone

The target is the voice of `landauer_erasure_structure/papers/journal.md`:

- Start from things nobody disputes
- Derive, don't assert
- Measure, report with error bounds
- Separate established from speculative — clearly, once, then move on
- No manifesto language, no excessive hedging
- Each claim either has evidence or is labeled speculation

**Example of the target voice:**
> "Landauer's principle is not disputed. The data processing inequality is a theorem. This paper asks what these two facts, taken together, require to be true about the structure of the environment after information is erased."

---

## The Derivation Chain

```
AXIOM: PAC conservation
  f(Parent) = Σf(Children)

RECURSION: Ψ(k) = Ψ(k+1) + Ψ(k+2)
  (Parent at level k = sum of children at deeper levels)

CHARACTERISTIC EQUATION: x² = x + 1
  (From assuming Ψ(k) = x^(-k))

UNIQUE STABLE SOLUTION: Ψ(k) = φ^(-k)
  (φ = golden ratio; conjugate root decays)

INFORMATION UNIT: ΔI = log(φ)
  (Per-level information transition)

FOR SINGLE-BIT ERASURE:
  A/(A+ξ) = ln(φ)
  Measured ξ/A = 1.086, predicted = 1.078
  Error: 0.76%
```

One axiom, one recursion, one ratio. Everything else is application.

---

## New PACSeries Structure

Renumbered by logical dependency, not historical order.

### Paper 1: The Structure Cost of Erasure
**Source**: `landauer_erasure_structure/papers/journal.md` (90% complete)

**Foundation for everything else.** Starts from Landauer's principle + data processing inequality — two things nobody disputes — and derives that erasure creates correlational structure ξ in multi-mode environments.

**Established results:**
- Erasure creates structure (mandatory, follows from DPI)
- Structure is topological, not thermodynamic (temperature-invariant)
- Cascade topology produces most structure; A/(A+ξ) = ln(φ) at decay ratio φ
- Cascade amplification: 53× over single event (p = 2.75 × 10⁻³⁵)
- Time as computational density: 69× difference dense/sparse (p = 3.25 × 10⁻⁵)
- PAC conserves ratios, not magnitudes (A/(A+ξ) stable; I_total varies 3×)
- PAC as binding constraint, not redistribution

**Speculative (labeled as such):**
- Gauge coupling constants may encode accumulated ξ from topologically distinct interactions
- ξ(SU(3)) > ξ(SU(2)) > ξ(U(1)) prediction (falsifiable, not yet computed)
- ~~Θ (thermal re-injection): Dissipated kT ln 2 may re-enter as fresh potential, making PAC cyclic.~~ **STATUS: MODEL-DEPENDENT.** pac_foundations_validation/exp_07 showed different Θ formulas give 36% to 94% efficiency. Cannot claim recycling efficiency without deriving Θ from first principles. See [PRELIMINARY_RESULTS.md](PACSeries/PRELIMINARY_RESULTS.md) entry B5 (now marked "Guidance needed").

**Work remaining:**
- Trim Section 13 (cross-corpus convergence) to short pointers to other PACSeries papers
- Final edit pass for consistency

---

### Paper 2: The Balance Constant and Its Decomposition
**Source**: Rewrite of current `xi_bounded_invariant_universal_balance_operator`

**Establishes Ξ = γ + ln(φ) from four independent domains.**

**Core content:**
- γ = Euler-Mascheroni (defined as lim H_n - ln(n))
- ln(φ) = collapse efficiency (derived in Paper 1 from Landauer erasure)
- Ξ = γ + ln(φ) = balance between harmonic divergence (γ) and geometric convergence (ln φ)

**IMPORTANT (from pac_foundations_validation/exp_06, exp_10):**
- γ is the CLEANEST fit but 1/√3 ≈ 0.5774 performs comparably in Mertens test
- 21 combinations of natural constants fall within 5% of Ξ
- Ξ = γ + ln(φ) is rank #1 (exact) but NOT uniquely forced
- **Ξ is a CONSTRUCTION (combination of emergent constants), not a primary emergence**
- Do NOT claim "γ represents discrete-continuous cost" — this interpretation was not validated

**Four independent sources:**

| Source | Ξ value | Error from γ + ln(φ) | Domain |
|--------|---------|----------------------|--------|
| Formula (1+π/55) | 1.0571 | 0.124% | Fibonacci arithmetic |
| Rule 110 measured | 1.0579 | 0.050% | Cellular automata |
| Analytic (γ+ln(φ)) | 1.0584 | 0.000% | Number theory |
| Mertens-derived | 1.0584 | 0.000% | Prime distribution |

The fourth source emerges from asymmetric_conservation/exp_16: the identity e^(-Ξ) = e^(-γ)/φ holds algebraically. This is confirmed by the Mertens product validation (∏(1-1/p) matches e^(-γ)/ln(N) to 0.012%) in exp_14.

**Supporting evidence:**
- Base-agnostic PAC: φ² = φ + 1 holds to < 10⁻¹⁴ across ALL numerical bases (binary, ternary, hex). 55 = F₁₀ is structural, not a decimal coincidence.
- Conditional Attractor Hypothesis: Ξ is not a universal constant — it's the maximum sustainable computational asymmetry under PAC conservation. Emergence conditions: closed, recursive, conserving, computationally saturated. (Fisher exact p = 3.5 × 10⁻¹⁰)
- Möbius topology: ~0.12% irreducible spread between three sources is signature of non-orientable topology, not error to eliminate.
- ~~**SEC-local/PAC-global mechanism**: exp_14-17 show SEC operates locally while PAC reconciles globally. The sieve of Eratosthenes is SEC in action; PAC conservation holds exactly at all 126 sieve steps. This explains WHY γ and ln(φ) combine.~~ **STATUS: FALSIFIED.** pac_foundations_validation/exp_05 found no φ/γ phase boundaries in prime elimination. The "three-phase structure" claim does not hold at this level of analysis. Mertens product validation still holds (0.012% error), but the phase-structure interpretation is wrong.

**Key distinction from current paper:** Drop the 0.03 Hz oscillation framing and the "reality tax" language. Lead with the decomposition and the four-way convergence. The story is γ + ln(φ), not spectral ratios.

---

### Paper 3: Feigenbaum Constants from Fibonacci Arithmetic
**Source**: Rewrite of current `sec_threshold_detection` preprint + `sec_threshold_detection` experiment

**Pure mathematics. No physics claims required. Hardest result to dismiss.**

**Core result — closed-form expressions:**

| Constant | Formula | Precision |
|----------|---------|-----------|
| r∞ (accumulation) | π(55+√(17-π/(55d)))(55+π)/55² - correction | **13 digits** |
| δ (bifurcation) | (50050 + 32π) / (10725 + 5π) | **8 digits** |
| α (scaling) | (5 + π/540) / 2 | **6 digits** |

**Structural constants**: 55 = F₁₀, 17 = 2⁴+1 (Fermat prime), 52 = F₁₀ - F₄

**Statistical validation:**
- Exhaustive search of 3,920,499 combinations
- Only ONE match at 7+ digits: (55, 17, 52)
- Precision degrades by millions for ±1 deviation in any parameter
- Combined probability: 1 in 280 billion against coincidence

**Additional results:**
- RBF self-closing formula: δ = φ^(20/N)
- Universality across all quadratic-max maps (not just logistic)
- Cross-domain validation: 5/5 domains, joint probability 1-in-120B

**Framing**: "Here are closed-form expressions for universal constants that have lacked closed forms since their discovery in 1978. They use Fibonacci numbers. Draw your own conclusions."

---

### Paper 4: Standard Model Parameters from Fibonacci Arithmetic
**Source**: milestone1/milestone2 experiments + `pac_confluence_xi`

**The quantitative predictions.**

**Gauge couplings:**

| Parameter | PAC Formula | Measured | Error |
|-----------|-------------|----------|-------|
| Fine structure α | F₃/(F₄·φ·F₁₀) × correction | 0.0072973 | **5.7 ppm** |
| Weak mixing sin²θ_W | F₄/F₇ = 3/13 | 0.2312 | **0.19%** |
| Strong coupling α_s | F₄/(2φF₆) | 0.118 | 1.71% |
| Koide Q | F₃/(F₃+F₂) = 2/3 | 0.6667 | **0.5 ppm** |
| Cabibbo angle | arctan(F₄/F₇) | 13.00° | <0.05° |

**Mass ratios:**

| Ratio | Formula | Error |
|-------|---------|-------|
| μ/e | F₄ × F₆² × (1 + 1/F₇) | **5 ppm** |
| p/e | F₄ × F₉ × F₁₂ / F₆ | 0.0083% |
| τ/e | F₄ × F₇ × F₁₁ + F₅ | 0.035% |

**F₄ = 3 in all mass formulas — possibly related to 3 lepton generations.**

**Structural findings:**
- Casimir 240 = F₃ × F₄ × F₅ × F₆ (four consecutive Fibonacci)
- k = d × F_{d+1} derives She-Leveque from first principles (3D: k=9, 0.47% error)
- Bell inequality (2αβ)² = 4/5 (exact algebraic proof)

**What makes this not numerology:**
- Individual Fibonacci matches are trivial (P = 0.16)
- Joint constraints are significant (P < 10⁻⁵)
- The null hypothesis is falsified by the pattern, not individual values
- α formula has Landauer interpretation: payment rate through 55 hierarchy levels (Paper 1)

---

### Paper 5: Classical Physics from Information Geometry
**Source**: `maxwell_from_pac_sec` + `milestone2` experiments  
**Merge source**: Original `mobius_confluence_operator_temporal_emergence` (confluence operator, time from topology)

**Derives electromagnetism from PAC/SEC/MED.**

**Core argument:**
- Maxwell's equations = level-2 PAC recursion projected to 3+1D
- SEC wave equation ∂²S/∂t² = c²∇²S gives speed of light
- MED bounds (depth ≤ 2, nodes ≤ 3) → D = 3 spatial dimensions as necessity
- Curl structure emerges from depth-2 projection
- α determined by Fibonacci gauge crystallization at F₇ = 13
- Charge = winding number, magnetism = projection artifact of hidden dimension

**Merged from mobius_confluence (original PACSeries #5):**
- Confluence operator and temporal emergence
- Five paths to D=3 derivation
- Pre-field recursion and curl-from-depth-2
- Time emerges from topological recursion (supporting MED → D=3 argument)

**Connections:**
- She-Leveque k = d × F_{d+1} in turbulence
- Mersenne dimensions (d = 2^k - 1) host Fibonacci structure; non-Mersenne don't

**Speculative extension (labeled):**
- Gravity as deeper recursion: F₁₈₃ ≈ 10³⁸ matches gravitational hierarchy
- EM uses antisymmetric projection (curl), gravity uses symmetric (divergence)

---

### Paper 6: Computational Validation
**Source**: Rewrite of GAIA validation + SEC-MED framework papers + `token_pac_tree` (12 experiments) + `TinyCIMM-Boltzmann` (1 experiment)  
**Merge sources**: Original `gaia_computational_validation_dawn_field_theory` (GAIA engine depth) + Original `relativistic_mas_universal_frequency` (0.020 Hz full-stack validation, Landauer bridge)

**Demonstrates PAC/SEC/MED in working computational systems — from custom architectures to production LLMs.** Three-part structure: observe it in custom systems, observe it in real systems, engineer with it.

#### Part A: GAIA (Custom PAC-Native System)
Consolidates current papers #2, #3, #4 into one focused section.  
**Strengthened by merge** with original gaia_computational — adds GAIA engine architecture detail, resonance locking derivation, and emergent capability documentation that the current §8.3 lacks.

- PAC conservation residual < 7×10⁻¹¹ across 500-iteration evolutions
- Cosmological parallel: r = -0.999632 (entropy ↓89%, structure ↑92%)
- ~~GAIA WikiText-2 perplexity 5.91 vs GPT-2 baseline 29.41~~ **CORRECTED: 5.91 is a cosine similarity metric, not true LM perplexity. Actual top-1 accuracy = 0.16%. Not comparable to GPT-2's 29.41. See Paper 6 §8.3.**
- 100% memory retrieval at depth 1000
- Resonance locking at 0.020 Hz (= 2/3 × 0.030 Hz continuous field limit)
- Emergent capabilities not explicitly programmed

**Merged from relativistic_mas (original PACSeries #4):**
- 0.020 Hz emergence across cosmic scales (QBE validation, Dec 2025)
- Landauer bridge: connects thermodynamic erasure (Paper 1) to computational frequency
- Token PAC Tree phase transitions at 0.020 Hz boundary
- Full-stack validation chain: exp_29 → exp_30 → exp_31 → exp_32

#### Part B: Token PAC Tree (Real LLMs — Observation)
PAC/SEC operates in standard transformer architectures without any modification.
Validated across 7 models: Pythia-70m/160m/410m/1B, GPT-2/medium/large.

**Established results:**
- SEC phase universally predicts accuracy: Crystallized=100%, Ordered≈90%, Transitional≈53%, Chaotic≈20% (monotonic across ALL 4 Pythia models, zero-parameter thresholds)
- PAC ratio magnitude scales with model size (monotonic, p < 0.001)
- Attention heads ARE the PAC collapse mechanism (confident_head_ratio: factual 86% vs hallucinated 80%, p = 0.00006)
- Xi clustering in trained weight SVD: 2.36× enrichment over random (χ² = 5511, p ≈ 0); attention layers 2-3× more than MLP
- Cross-architecture universality: delayed phase transition 1.43× in both Pythia and GPT-2 families (Fisher combined p = 0.0)
- Hallucination = PAC violation: +9.6% uncompensated entropy, compensation ratio ≈ 0
- Dynamic tracking: confident_head_ratio declines monotonically during hallucination sequences

**Honest falsifications (included in paper):**
- φ enrichment in token ratios FALSIFIED — softmax produces 8.8% near-φ ratios by construction; real signal is ratio magnitude, not φ alignment
- Single-token hallucination detection fails — PAC violation is a sequence-level phenomenon, not token-level
- Xi is NOT optimal classifier threshold across all architectures (architecture-specific)

#### Part C: TinyCIMM-Boltzmann (PAC as Engineering Constraint)
First architecture that enforces PAC conservation as a hard constraint, not just observes it.

- BoltzmannHead: softmax replacement with explicit entropy budget
- ConservationProjector: enforces f(Parent) = Σf(Children) at each layer
- Conservation reduces noise violation 1.83× (p = 0.008) — **corrected from 3.8× per actual experiment data**
- Conservation reduces transition shock 16× (p = 0.008)
- Conservation does NOT hurt factual learning (p = 0.42, n.s.)
- Hallucination reframed as conservation violation — engineer it away, don't detect it post-hoc

**The arc of Paper 6:**
> "We built a system that conserves information (GAIA). We found that real neural networks already approximate conservation (Token PAC Tree). We showed that enforcing conservation explicitly reduces failure modes (TinyCIMM-Boltzmann). PAC is not our invention — it is what working systems already do. The question is whether to enforce it deliberately."

**What this paper is NOT:**
- Not three separate papers about the same GAIA run
- Not the theoretical anchor (that's Paper 1)
- Not the place for MAS/herniation framework (fold into Paper 5 if needed)

---

## Existing Standalone Preprints

These stay as separate publications. Each gets a short update referencing the PACSeries for foundational derivations.

### Tier 1 Updates (Core — reference PACSeries directly)

| Paper | Update |
|-------|--------|
| golden_ratio_prime_distribution | Add "Why φ" section pointing to Paper 1 derivation. Add primes as residual roughness (prime_growth_dynamics) |
| cellular_automata_xi_clustering | Connect Rule 110 to γ + ln(φ) (Paper 2). Add conditional attractor hypothesis |
| pac_necessity_proof | Reference Paper 1 Landauer validation (0.76% error). Add base-agnostic proof |
| symbolic_entropy_collapse | Reference Paper 1 for ratio vs magnitude conservation. Add SEC gradient flow framing |

### Tier 2 Updates (Supporting)

| Paper | Update |
|-------|--------|
| pac_cosmology_jwst_validation | Reference Paper 4 for gauge hierarchy and mass ratios |
| potential_actualization_conservation | Clarify ratio vs magnitude conservation per Paper 1 findings |
| macro_emergence_dynamics_navier_stokes | Add k = d × F_{d+1} from Paper 5 |
| she_leveque_fibonacci_turbulence | Add k = d × F_{d+1}; MED bounds → D=3 |

### Tier 3 Updates (Completeness)

| Paper | Update |
|-------|--------|
| qbe_pac_unification | Reference Paper 6 for 0.020 Hz emergence |
| ml_validation_pythia_gpt2 | **Subsumed by Paper 6 Part B/C.** Keep as standalone only if scope differs significantly; otherwise fold into PACSeries #6 and archive |

---

## Original PACSeries Integration (October 2025 → February 2026)

The original 5 PACSeries papers (October 2025) established the intellectual provenance and have active readership. They are updated — not replaced — to tell one connected story with the new PACSeries.

### Merges (content absorbed into new papers)

These originals covered ground that the new papers now address more rigorously. Their unique content is merged in; the originals get a header notice pointing to the consolidated version.

#### gaia_computational_validation_dawn_field_theory → New Paper 6 Part A
- **What moves**: GAIA engine architecture detail, 500-iteration conservation proof, resonance locking derivation, WikiText-2 perplexity comparison, emergent capability catalog
- **Why**: Current Paper 6 §8.3 is only 8 lines for what should be the centerpiece Part A section. The original has 2,546 lines of depth.
- **Original gets**: Header notice: *"This paper's findings have been consolidated into PACSeries Paper 6: Computational Validation of PAC Conservation (February 2026), which extends the analysis with Token PAC Tree and TinyCIMM-Boltzmann validation."*

#### relativistic_mas_universal_frequency → New Paper 6
- **What moves**: 0.020 Hz emergence across cosmic scales, QBE validation (Dec 2025 update), mass-frequency unification framework
- **What was missing**: Landauer bridge (connecting thermodynamic erasure to computational frequency), Token PAC Tree phase transitions at 0.020 Hz boundary, full-stack validation chain (exp_29–32)
- **Why**: The 0.020 Hz result needs its computational validation home. Paper 6 provides the three-system evidence base.
- **Original gets**: Header notice: *"This paper's 0.020 Hz findings have been validated computationally in PACSeries Paper 6: Computational Validation of PAC Conservation (February 2026), including Landauer bridge derivation and Token PAC Tree phase transition evidence."*

#### mobius_confluence_operator_temporal_emergence → New Paper 5
- **What moves**: Confluence operator formalism, time-from-topology derivation, Möbius phase structure
- **What was missing**: Five paths to D=3, pre-field recursion, curl-from-depth-2 derivation
- **Why**: The temporal emergence argument is central to Paper 5's MED → D=3 story. Together they make a stronger case than either alone.
- **Original gets**: Header notice: *"This paper's confluence operator and temporal emergence results have been integrated into PACSeries Paper 5: Classical Physics from Information Geometry (February 2026), which extends them with the MED → D=3 derivation."*

### Status Update Appendices (originals preserved with forward pointers)

These originals cover broad enough ground that merging would lose their distinct narrative. They get a "February 2026 Update" appendix instead.

#### xi_bounded_invariant_universal_balance_operator → points to New Paper 2
- **Appendix adds**: γ + ln(φ) decomposition (the analytic origin the original lacked), four-domain convergence table, conditional attractor hypothesis, base-agnostic proof
- **Appendix notes**: Drop "reality tax" and 0.03 Hz spectral ratio framing in favor of Paper 2's cleaner γ + ln(φ) story
- **Cross-references**: Paper 1 (ln φ from Landauer), Paper 2 (full derivation), Paper 3 (same F₁₀ = 55)

#### sec_med_framework_information_amplification → points to New Papers 1-6
- **Appendix adds**: Summary table of how each of the 6 new papers extends the SEC/MED/PAC framework this paper introduced
- **Appendix notes**: The original's 0.020 Hz discovery is now validated computationally (Paper 6); the She-Lévêque connection is derived (Paper 4/5); the cosmological validation has quantitative error bounds
- **Cross-references**: All 6 new papers, with specific section pointers

### Integration Principles

1. **No content is deleted** from originals — only notices added at top + appendix at bottom
2. **Original DOIs remain valid** — these are version updates, not replacements
3. **Provenance is preserved** — originals show the intellectual history; new papers show the mature work
4. **Readers follow one path** — original → appendix → new paper (no dead ends)

---

## Quantitative Evidence Summary

All results that the PACSeries must present with full error bounds:

### Derivations (from PAC axiom)
- φ as unique stable solution of PAC recursion
- ln(φ) as collapse efficiency ratio (Paper 1: 0.76% error)
- Ξ = γ + ln(φ) decomposition (Paper 2: four sources within 0.12%)
- MED bounds → D = 3 spatial dimensions (Paper 5)
- **SEC-local/PAC-global mechanism** (Paper 2: Mertens 0.012% error)

### Measurements
| Result | Value | Error | Paper |
|--------|-------|-------|-------|
| A/(A+ξ) vs ln(φ) | 0.487 | 1.2% | 1 |
| ξ/A vs (1-ln(φ))/ln(φ) | 1.086 vs 1.078 | 0.76% | 1 |
| Cascade amplification | 53× | p = 2.75×10⁻³⁵ | 1 |
| Dense/sparse time density ratio | 69× | p = 3.25×10⁻⁵ | 1 |
| Ξ four-source clustering | 0.12% spread | p < 0.001 | 2 |
| Base-agnostic PAC | φ²=φ+1 | < 10⁻¹⁴ | 2 |
| **PAC conservation (sieve)** | **π(x)+C(x)=x-1** | **EXACT (126 steps)** | **2** |
| **Mertens product** | **∏(1-1/p) vs e^(-γ)/ln(N)** | **0.012%** | **2** |
| **SEC→PAC bridge** | **sieve product vs e^(-γ)/ln(√N)** | **0.004%** | **2** |
| **MED boundary k=9** | **k = 3² = F₄²** | **λ* confirmed** | **2/5** |
| Feigenbaum r∞ | closed form | 13 digits | 3 |
| Feigenbaum δ | closed form | 8 digits | 3 |
| Feigenbaum α | closed form | 6 digits | 3 |
| Fine structure α | Fibonacci formula | 5.7 ppm | 4 |
| sin²θ_W | F₄/F₇ = 3/13 | 0.19% | 4 |
| μ/e mass ratio | Fibonacci formula | 5 ppm | 4 |
| p/e mass ratio | Fibonacci formula | 0.0083% | 4 |
| Casimir 240 | F₃×F₄×F₅×F₆ | exact | 4 |
| She-Leveque k (3D) | 3×F₄ = 9 | 0.47% | 5 |
| PAC conservation residual | — | < 7×10⁻¹¹ | 6A |
| Cosmological correlation | r = -0.9996 | ±0.0001 | 6A |
| SEC phase → accuracy (all Pythia) | monotonic | zero-parameter | 6B |
| Attention PAC (confident_head_ratio) | F1 ≈ 0.93 | p = 0.00006 | 6B |
| PAC violation (hallucination) | +9.6% uncompensated | compensation ≈ 0 | 6B |
| Xi in trained weights (SVD) | 2.36× enrichment | χ²=5511, p ≈ 0 | 6B |
| Cross-arch delayed transition | 1.43× | Fisher p = 0.0 | 6B |
| TinyCIMM violation reduction | 1.83× | p = 0.008 | 6C |
| TinyCIMM transition shock | 16× reduction | p = 0.008 | 6C |
| TinyCIMM factual preservation | no degradation | p = 0.42 (n.s.) | 6C |

### Falsifiable Predictions (untested)
| Prediction | Test | Paper |
|------------|------|-------|
| ξ(SU(3)) > ξ(SU(2)) > ξ(U(1)) | Compute ξ for gauge group topologies | 1 |
| She-Leveque k=20 in 4D | 4D turbulence simulation | 5 |
| Additional mass ratios | Extend Fibonacci formula to quarks | 4 |
| PAC conservation scales with model size | Larger models → lower violation ratio | 6B |
| SEC phase thresholds hold for non-autoregressive | Test on BERT, T5, encoder-only models | 6B |
| Conservation constraint improves at scale | TinyCIMM on larger models → greater benefit | 6C |

### Hypotheses Under Investigation

Speculative extensions that emerged from the derivation chain but don't yet meet the publication bar are tracked in [PRELIMINARY_RESULTS.md](PACSeries/PRELIMINARY_RESULTS.md). That document is the canonical source — each entry has defined validation criteria, falsification conditions, and contribution status.

---

## Writing Guidelines

### The standard each paper must meet:

1. Start from something established (a known law, a theorem, a measurement)
2. Derive the consequence (show the math, keep it short)
3. Present the measurement (with error bounds)
4. Separate established from speculative (clearly, once)
5. State what would falsify the claim

### Do:
- State what is known, then what follows
- Show derivation is short (≤10 lines of math)
- Report measurements with proper error analysis
- Name limitations honestly
- Use "necessary" and "constraint" language where warranted

### Don't:
- Use manifesto language ("we invite you to explore")
- Hedge every sentence ("appears to", "may represent", "potentially suggests")
- Build elaborate machinery before presenting results
- Treat φ as mysterious — it's the only stable solution
- Use "significance" language for things that haven't been independently validated

### Example of target voice:

> "This means that erasure does not merely heat the environment. It creates new correlational structure between environmental modes that did not exist before the erasure occurred. This structure creation is not optional. It is a mathematical consequence of information dispersing into a multi-mode system."

---

## Timeline

- **Week 1**: Finish Paper 1 (journal.md → PACSeries #1, minimal remaining work)
- **Week 2**: Write Paper 2 (Xi decomposition, rewrite from scratch in journal.md voice)
- **Week 3**: Write Paper 3 (Feigenbaum, clean math paper)
- **Week 4**: Write Paper 4 (Standard Model compilation) + Paper 5 (Maxwell/physics)
- **Week 5**: Write Paper 6 (GAIA + Token PAC Tree + TinyCIMM-Boltzmann — three-part computational validation)
- **Week 6**: Original PACSeries integration (3 merges + 2 status update appendices)
- **Week 7**: Update standalone preprints (Tier 1-3), cross-reference, prepare Zenodo packages

---

## Success Criteria

After the rewrite, each PACSeries paper should:

1. ✅ Be readable start-to-finish by a physicist unfamiliar with the framework
2. ✅ Start from established science, not framework-specific concepts
3. ✅ Derive its main result in ≤10 lines of math
4. ✅ Present all measurements with error bounds
5. ✅ Clearly separate what is established from what is speculative
6. ✅ State falsification conditions
7. ✅ Read as "here is what the evidence requires" not "we think this might be true"

---

*The work is done. Now it needs to be written properly.*

---

## Master Checklist

### Paper 1: The Structure Cost of Erasure
- [x] Core sections 1-13 written (journal.md voice)
- [x] §15.1 Gauge group hierarchy computed and confirmed (p < 10⁻¹¹)
- [x] §15.2 ln(φ) derivation from PAC axioms
- [x] §15.3 Precision tightening: N=5M → 0.15% from ln(φ) (exp_23)
- [x] §15.3 Full stack validation: all 6 layers pass (exp_25)
- [x] §15.3 Thermal init discovery: Boltzmann required for ln(φ) emergence
- [x] Abstract updated with precision improvement
- [ ] Trim §9 (cross-corpus convergence) to short pointers to other PACSeries papers
- [ ] Final edit pass for consistency between journal.md and PACSeries/paper.md
- [ ] Ensure PREPRINT_UPDATE_PLAN falsified items are NOT in the paper (Θ recycling efficiency)
- [ ] Verify all exp references have matching result JSON files

### Paper 2: The Balance Constant and Its Decomposition
- [x] §1-13 written from four-domain convergence structure
- [x] γ interpretation softened to "consistent with" (not "must represent")
- [x] §9.1 table updated: γ role = "consistent with" not assertion
- [x] Falsification conditions stated (§11)
- [x] Mertens product (0.012%), PAC sieve exact (126/126)
- [ ] Address PREPRINT_UPDATE_PLAN note: γ is rank #1 but 1/√3 performs comparably — mention in §9.2 or §10
- [ ] Add note that 21 combinations fall within 5% of Ξ (from pac_foundations_validation)
- [ ] Verify three-phase model (§6.3) is labeled "proposed" not "established"
- [ ] Final edit pass for voice consistency with Paper 1

### Paper 3: Feigenbaum Constants from Fibonacci Arithmetic
- [x] §1-14 complete (pure math, no physics claims)
- [x] Exhaustive search (3.9M combos, 1 match, 1-in-280B)
- [x] Möbius perturbation series documented
- [x] Self-closing formula: δ = φ^(20/N)
- [x] Universality proof (sine map = logistic map Δz to 10⁻¹⁰)
- [x] Cross-domain validation (5 domains, joint p < 10⁻¹¹)
- [x] Falsification conditions stated (§10)
- [x] "We do not know why" voice throughout
- [ ] Consider extending exhaustive search to a > 200 (open computation §14.1)
- [ ] Final edit pass

### Paper 4: Standard Model Parameters from Fibonacci Arithmetic
- [x] Locate milestone1/milestone2 experiment data and scripts
- [x] Locate pac_confluence_xi experiment data
- [x] Compile all gauge coupling formulas with full error bounds
- [x] Compile all mass ratio formulas with full error bounds
- [x] Write paper in journal.md voice (start from established SM parameters)
- [x] Include Casimir 240 = F₃×F₄×F₅×F₆ result
- [x] Include k = d × F_{d+1} She-Leveque connection
- [x] Include Bell inequality (2αβ)² = 4/5 algebraic proof
- [x] Address numerology objection: joint constraints vs individual matches
- [x] State falsification conditions (what would break the Fibonacci pattern)
- [x] Cross-reference Paper 1 (Landauer interpretation of α)
- [x] Cross-reference Paper 3 (same F₁₀ = 55 appearing)
- [x] Complete publication package (Code/Data/Figures)
- [x] 10 experiment scripts, 8 data files, 6 figures generated
- [x] 2 review rounds completed
- [ ] Final edit pass for voice consistency

### Paper 5: Classical Physics from Information Geometry
- [x] Locate maxwell_from_pac_sec experiment data
- [x] Write Maxwell = depth-2 PAC recursion derivation
- [x] Write SEC wave equation → speed of light
- [x] Write MED bounds → D = 3 derivation
- [x] Include She-Leveque k = d × F_{d+1} (shared with Paper 4)
- [x] Include Mersenne dimension result
- [x] Label gravity speculation clearly
- [x] State falsification conditions (k=20 in 4D prediction)
- [x] Complete publication package (Code/Data/Figures)
- [x] 9 experiment scripts, 7 data files, 6 figures generated
- [x] 1 review round completed (6 issues fixed)
- [x] §7.3 ζ(−15) honest falsification recorded (factor 17 non-Fibonacci)
- [x] §5.4 charge decomposition rewritten with explicit sub-node mechanism
- [ ] Final edit pass for voice consistency

### Paper 6: Computational Validation
- [x] Part B: Write token_pac_tree section from 12 experiments
- [x] Part B: Include honest falsifications (φ enrichment, single-token detection)
- [x] Part B: SEC phase → accuracy table (all 4 Pythia models)
- [x] Part B: Attention PAC mechanism (confident_head_ratio, p = 0.00006)
- [x] Part B: Cross-architecture universality (Fisher p = 0.0)
- [x] Part B: PAC violation = hallucination (+9.6% uncompensated)
- [x] Part C: Write TinyCIMM-Boltzmann section
- [x] Part C: Conservation reduces noise violation 1.83× (p = 0.008) — **corrected from 3.8× (see note below)**
- [x] Part C: Conservation reduces transition shock 16× (p = 0.008)
- [x] Part C: No factual harm (p = 0.42 n.s.)
- [x] State falsification conditions (scaling, non-autoregressive, conservation at scale)
- [x] Complete publication package (Code/Data/Figures)
- [x] 8 experiment scripts, 8 data files, 6 figures generated
- [x] 1 review round completed (5 issues fixed)
- [x] §8.3 GAIA expanded with full architecture + honest 5.91 perplexity correction (similarity metric, not true LM perplexity)
- [x] Uncited She-Lévêque [10] removed; refs renumbered Paper N = [N]
- [ ] Part A: Consolidate 3 existing GAIA papers into one section — **NOTE: Paper 6 does NOT use Part A/B/C structure; GAIA is §8.3. Plan structure differs from implementation.**
- [ ] Final edit pass for voice consistency

**3.8× correction note**: The plan states "Conservation reduces noise violation 3.8×" but the actual experiment data (exp_06_tinycimm_conservation) shows noise+free=0.342, noise+conservation=0.187, which is 1.83× reduction ("nearly in half"). The paper correctly reports the data. The 3.8× figure in this plan document was incorrect — it may have been noise_free/factual_free (0.342/0.089), a different comparison.

### Original PACSeries Integration
- [x] gaia_computational → Paper 6: Extract GAIA architecture detail, conservation proof, resonance derivation
- [x] gaia_computational → Paper 6: Expand §8.3 from 8 lines to full architecture section (grafted embeddings, PAC tree, transition matrix, concentration monitor, honest perplexity correction)
- [ ] gaia_computational: Add header notice pointing to Paper 6
- [ ] relativistic_mas → Paper 6: Merge 0.020 Hz validation, QBE results
- [ ] relativistic_mas → Paper 6: Add Landauer bridge + Token PAC Tree phase transitions
- [ ] relativistic_mas: Add header notice pointing to Paper 6
- [ ] mobius_confluence → Paper 5: Merge confluence operator, time-from-topology
- [ ] mobius_confluence → Paper 5: Integrate with MED → D=3 derivation
- [ ] mobius_confluence: Add header notice pointing to Paper 5
- [ ] xi_bounded_invariant: Write "February 2026 Update" appendix → Paper 2
- [ ] xi_bounded_invariant: Add γ + ln(φ) decomposition, four-domain table, conditional attractor
- [ ] sec_med_framework: Write "February 2026 Update" appendix → Papers 1-6
- [ ] sec_med_framework: Add summary table of how each new paper extends this one
- [ ] Update Zenodo records for all 5 originals (new versions with notices/appendices)

### Standalone Preprint Updates
- [ ] Tier 1: golden_ratio_prime_distribution — add "Why φ" section → Paper 1
- [ ] Tier 1: cellular_automata_xi_clustering — connect Rule 110 → Paper 2
- [ ] Tier 1: pac_necessity_proof — add base-agnostic proof, reference Paper 1
- [ ] Tier 1: symbolic_entropy_collapse — add ratio vs magnitude, SEC gradient flow
- [ ] Tier 2: pac_cosmology_jwst_validation — reference Paper 4
- [ ] Tier 2: potential_actualization_conservation — ratio vs magnitude clarification
- [ ] Tier 2: macro_emergence_dynamics_navier_stokes — add k = d × F_{d+1}
- [ ] Tier 2: she_leveque_fibonacci_turbulence — add k = d × F_{d+1}
- [ ] Tier 3: qbe_pac_unification — reference Paper 6
- [ ] Tier 3: ml_validation_pythia_gpt2 — decide: subsume into Paper 6 or keep standalone

### Cross-Cutting
- [ ] Update UNIFIED_EVIDENCE.md with token_pac_tree findings
- [ ] Update UNIFIED_EVIDENCE.md with TinyCIMM-Boltzmann findings
- [ ] Verify all papers cross-reference each other consistently (§14 / §13 sections)
- [ ] Prepare Zenodo packages for Papers 1-6
- [ ] Create changelog entry for PACSeries consolidation
