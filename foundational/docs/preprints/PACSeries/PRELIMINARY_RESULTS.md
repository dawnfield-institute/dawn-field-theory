# PACSeries: Preliminary Results and Open Leads

**Status**: Working document  
**Updated**: February 2026  
**Purpose**: Catalogue results that are suggestive but not yet at PACSeries publication standard

---

## What This Document Is

The PACSeries papers present results that meet a specific bar: derived from established mathematics or physics, measured with error bounds, independently reproducible, and falsifiable. This document catalogues results that show signal but need additional work before they can meet that standard.

Each entry states: what was observed, where the evidence is, and what would be needed to tighten it to publication quality.

---

## Contributing

Results marked **Open** have well-defined next steps that don't require coordination. If you want to tackle one:

1. Open an issue referencing the entry (e.g., "A1: Wilson-Fisher null test")
2. Run the validation described in "What's needed"
3. Submit results via PR

Results marked **Guidance needed** require discussion first — open an issue to coordinate before starting work.

**3-month rule**: Any result inactive for 3+ months automatically becomes **Open** for community contribution. Check the `Last assessed` date on each entry.

| Status | Meaning |
|--------|--------|
| **Active** | Currently being worked on |
| **Open** | Well-defined task, anyone can pick it up |
| **Guidance needed** | Requires conversation before starting |

---

## Tightening Criteria

For a result to move from this document into a PACSeries paper, it must:

1. **Start from something established** — a known law, theorem, or measurement
2. **Have a derivation** — not just a numerical match, but a reason it should hold
3. **Report error bounds** — with proper statistical framework
4. **Survive a null test** — random baselines, alternative explanations checked
5. **Be independently verifiable** — reproducible from the included scripts

---

## Category A: Strong Signal, Needs Joint Statistics

These results have clear individual measurements but need joint probability analysis to rule out coincidental matching.

### ~~A1. Wilson-Fisher Critical Exponents in φ-Constants~~ → PROMOTED to Paper 4

**PROMOTED (2026-02-18)**: This result now meets all 5 tightening criteria and has been validated in milestone3/F6 (exp_07). Moved to Paper 4: Standard Model Parameters.

**Final validation results:**

| Exponent | Formula | Value | Known | Error |
|----------|---------|-------|-------|-------|
| ν | 2/(3·Ξ) | 0.6299 | 0.6300 | **0.017%** |
| η | (others) | — | 0.0362 | ~1% |
| β | (others) | — | 0.3265 | ~1% |
| γ_Ising | (others) | — | 1.2372 | ~1% |
| δ | (others) | — | 4.789 | ~1% |
| α | (others) | — | 0.110 | ~1% |

**Criteria met:**
1. ✅ Starts from established: Wilson-Fisher universality class (textbook)
2. ✅ Has a derivation: ν = (2/3) × (1/Ξ) = E-I-S cycle ratio × balance reciprocal (cascade framework)
3. ✅ Error bounds: 0.017% for ν; 6/7 exponents within 1%
4. ✅ Survives null test: MC p = 0.0000 (20 hits vs 1.89 expected from random constants); perturbation analysis shows best alternative is 1.06% (63× worse)
5. ✅ Independently verifiable: `milestone3/scripts/exp_07_wilson_fisher.py`

**Status**: ✅ **PROMOTED** — all 5 criteria satisfied  
**Last assessed**: 2026-02-18

---

### A2. Even-Odd Factorization Oscillation

**Observation**: Mean factorization depth Ω(n) oscillates by parity of distance to nearest prime. Odd-distance integers average Ω ≈ 4.34; even-distance average Ω ≈ 2.77. The oscillation amplitude ratio ≈ 1/φ at 0.03% error.

**Statistics**: t = 110.80, p ≈ 0 for the oscillation itself

**Source**: `prime_growth_dynamics/scripts/exp_04–exp_08`

**What's needed**:
- The oscillation is real (p ≈ 0). The question is whether the amplitude ratio being near 1/φ is meaningful or a single-number match.
- Need: analytic derivation of why the ratio should be 1/φ, or a parametric sweep showing it converges to 1/φ as N → ∞
- Without derivation, this is a curious observation, not a result

**Status**: Oscillation validated; φ connection unvalidated  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02

---

### A3. λ* and β from Phase Constants

**Observation**: Previously fitted constants from `sec_prime_manifold` now have candidate closed-form expressions:
- λ* ≈ 1 − Ξ/(F₁₀ + F₃) = 0.981431 (0.017% error from measured 0.9816)
- β ≈ (ln(φ) + F₃)/(π) = 0.789794 (0.026% error from measured ~0.79)

**Source**: `prime_growth_dynamics_v2/scripts/exp_01–02`

**What's needed**:
- Derivation: why should λ* = 1 − Ξ/(F₁₀ + F₃)? Currently this is a fit.
- Exhaustive search: are these the unique best expressions, or are there equally good alternatives?
- If unique and derived, this is Paper 2 material (extends the Ξ decomposition)

**Status**: Candidate formulae identified, not yet uniqueness-tested. **Not directly tested in milestone3** — no experiment specifically validates λ* or β closed forms. Would require a dedicated formula search with look-elsewhere correction (similar to exp_09 methodology).  
**Contribution status**: Open  
**Last assessed**: 2026-02-19

---

## Category B: Real Phenomenon, Interpretive Layer Needed

These show genuine signal but the connection to PAC/SEC requires interpretation that hasn't been independently validated.

### B1. Three-Phase Emergence Pipeline

**Observation**: The prime sieve decomposes into three phases with distinct boundary constants:
- Phase I (MED pruning): primes {2, 3, 5} remove 73.3% of candidates; boundary ≈ 1 − γ
- Phase II (SEC collapse): Fibonacci-structured density decay; dominant carrier p = 3 at 82.1%
- Phase III (residual smoothing): PNT rate 1/ln(x); Mertens convergence

Total: Ξ = γ + ln(φ) reconciles Phase I + II.

**Source**: `asymmetric_conservation/scripts/exp_16`, `prime_growth_dynamics_v2/PRE_STRUCTURAL_EMERGENCE.md`

**What Paper 2 already includes**: The Mertens product (0.012% error), PAC conservation at all 126 sieve steps, p = 3 carrying 82.1% of φ-clustering. These are measurements.

**What's preliminary**: The three-phase *labelling* — calling Phase I "MED pruning" and Phase II "SEC collapse" is interpretive. The sieve doesn't know about MED or SEC. Whether {2, 3, 5} being special is about MED's "nodes ≤ 3" bound, or is simply because small primes remove more multiples, hasn't been disambiguated.

**Architectural connection (from Feb 2026 synthesis)**: The three phases may map to the three-component architecture: Phase I (MED pruning) = cascade selecting topology; Phase II (SEC collapse) = dynamics driving transitions; Phase III (PNT smoothing) = PAC global reconciliation. If this mapping holds, the phases aren't just labels — they're the same three roles (topology/dynamics/accounting) operating sequentially. This would be validated if sieve-invariant.

**What's needed**:
- A prediction that distinguishes "MED bound" from "small primes remove more"
- Test: does the Phase I/II boundary shift if you use a different sieve (e.g. Sundaram)?
- If the phases are sieve-invariant, the framework gains credibility

**Status**: Measurements in Paper 2; framework characterisation as preliminary  
**Contribution status**: Open  
**Last assessed**: 2026-02

---

### B2. CA Temporal Convergence Dynamics

**Observation**: Class IV cellular automata don't start at Ξ — they *converge toward it* over time steps 0–500. The approach trajectory varies by rule but the asymptote clusters at the same value.

**Source**: `cellular_automata_pac_attractors/scripts/exp_09–11`

**What Paper 2 already includes**: The endpoint clustering (p < 10⁻⁷) from exp_07.

**What's preliminary**: The convergence *dynamics* — the trajectory shape, convergence rate, and whether it's monotonic. This would distinguish "Ξ is an attractor" from "Ξ is where the metric happens to land."

**What's needed**:
- Fit convergence curves and extract time constants
- Test whether convergence rate correlates with Wolfram class
- If Class IV has systematically different dynamics from other classes, that's Paper 2 material

**Status**: Computed, not yet analysed quantitatively  
**Contribution status**: Open  
**Last assessed**: 2026-02

---

### B3. π as Optimal Möbius Coherence Constant

**Observation**: Among transcendental constants tested (π, e, √2, ln(2), etc.), π produces 19× lower variance in Möbius coherence oscillations at σ = 1/2 than e.

**Source**: `oscillation_attractor_dynamics/scripts/exp_15–17`

**What's needed**:
- This is measured but not derived. Why should π be optimal? The connection to the Riemann zeta function is suggestive but circular (ζ is defined using primes which involve π).
- Need: independent derivation of π-optimality, or proof that the comparison is not an artefact of how coherence is measured
- 30 experiments in oscillation_attractor_dynamics, but the core claim is one ratio (19×)

**Status**: Measured; theoretical explanation missing  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02

---

### B4. Riemann Zeros as Phase Boundary

**Observation**: All 20 tested Riemann zeros detected from Möbius formula with < 0.06 error. Re(s) = 1/2 interpreted as Phase II → III boundary.

**Source**: `oscillation_attractor_dynamics`, `prime_growth_dynamics/scripts/exp_22`

**What's needed**:
- Detecting known zeros from a known formula is confirmation of existing math, not new
- The *interpretation* (critical line = phase boundary) is the new claim, and it's unfalsifiable as stated
- To be useful: must predict something about zeros that isn't already known
- Possible test: does the three-phase model predict the gap distribution of zeros?

**Status**: Computation confirmed; interpretation is framework-dependent  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02

---

### B5. Θ Thermal Re-injection (Recycling Hypothesis)

**Observation**: Paper 1 derives that erasure produces correlational structure ξ alongside dissipated heat (Landauer cost kT ln 2). The cascade amplification (53× over single event) implies that erasure products feed subsequent collapse events. The hypothesis: dissipated thermal energy Θ re-enters as fresh potential, making PAC cyclic: P → ΣA + ξ + Θ, where Θ → P' at the next level.

**Source**: `landauer_erasure_structure/papers/journal.md` (Paper 1 derivation), cascade amplification statistics (p = 2.75 × 10⁻³⁵)

**What Paper 1 already includes**: The ξ decomposition and cascade amplification measurement. The 53× ratio proves cascades create far more structure than single events.

**Milestone 3 Update (F5, exp_06)**: Cascade self-funding is partially validated:
- ✅ Monotonic ξ: 100/100 cascade steps show monotonic ξ increase
- ✅ Amplification: 29.2× cumulative ξ amplification over 100 steps
- ✅ Conservation: ΔE_total / E_input = 0.66%
- ❌ Back-pressure: r = 0.350 (FAIL — crude model, original achieved r ≈ 0.94)
- Original stub failure was a **unit mismatch** (ξ in bits vs P in energy), not physics failure
- Different Θ formulas give 36%–94% efficiency — recycling is **model-dependent**

**What's still preliminary**: The exact recycling efficiency. Self-funding is confirmed (3/4 tests pass) but the quantitative Θ budget depends on which formula is used. Cannot claim specific efficiency without deriving Θ from first principles. The honest range is **36%–94%** depending on the Θ formula — this 2.6× spread means the recycling *mechanism* is validated but the recycling *efficiency* is not a measurement, it's model-dependent.

**What's needed**:
- Derive Θ from first principles (which formula is correct?)
- Resolve back-pressure failure (crude model limitation vs real physics?)
- Energy budget accounting across cascade levels
- Narrow the 36%–94% range before claiming a specific recycling efficiency in Paper 1

**Status**: Partially validated (milestone3/F5: 3/4 PASS). Self-funding confirmed; efficiency is model-dependent (36%–94% range). Paper 1 should present the range, not a point estimate.  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02-19

---

## Category C: Cross-Domain Applications (Exploratory)

These apply PAC/SEC to new domains. The applications are interesting but each needs domain-specific validation before publication.

### C1. E = mc² in Semantic Embedding Space

**Observation**: In PAC hierarchies embedded in vector spaces, a quantity analogous to E = mc² holds. Synthetic embeddings: R² = 1.0000, c² = 1.0. Real embeddings (llama3.2): c² ≈ 416, with +330% semantic amplification (composites have *more* energy than sum of parts, unlike physical binding).

**Source**: `arithmetic/euclidean_distance_validation/` (25 experiments)

**What's needed**:
- n > 1 model. Currently c² is measured for one model (llama3.2).
- If c² varies systematically with model capability/size, strong result
- If c² is arbitrary, the "E = mc²" framing is misleading
- Exp_25 (R² equivalence resolution, p < 0.003) is solid defensive statistics

**Status**: Single-model demonstration. Needs multi-model sweep.  
**Contribution status**: Open  
**Last assessed**: 2026-02

---

### C2. PAC in JWST Black Hole Mass Distribution

**Observation**: PAC/SEC framework explains 100% of 69 JWST high-z black hole masses vs ΛCDM at 41%.

**Source**: `pac_cosmology_jwst_validation/`

**What's needed**:
- Independent astrophysics review of the comparison methodology
- The 100% vs 41% comparison may be unfair (PAC has more free parameters?)
- Need: parameter counting and information-theoretic model comparison (AIC/BIC)

**Status**: Published on Zenodo; methodology needs tightening  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02

---

### C3. DNA Prime Structure

**Observation**: 40 experiments exploring PAC patterns in DNA/genomic data.

**Source**: `experiments/dna_prime_structure/` (40 scripts)

**What's needed**: Full inventory not yet assessed for PACSeries relevance. Likely too domain-specific for the core series but could support a standalone publication.

**Status**: Unreviewed for this consolidation  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02

---

### C4. Wealth Field Dynamics

**Observation**: PAC conservation in economic systems, with φ → φ² transition at scale boundaries.

**Source**: `experiments/wealth_field_dynamics/` (16 scripts)

**What's needed**: Domain-specific validation. Economic data is noisy and model-dependent.

**Status**: Exploratory; not targeted for PACSeries  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02

---

### C5. E = mc² in Semantic Space — Engineering Utility

**Observation**: PAC hierarchies embedded in vector spaces exhibit a quantity analogous to E = mc², with model-specific c². Synthetic embeddings give R² = 1.0000, c² = 1.0. Real embeddings (llama3.2, 3072D) give c² ≈ 416 with +330% semantic amplification.

**Source**: `arithmetic/euclidean_distance_validation/` (25 experiments, 7 core)

**Why "engineering utility" not "physical law"**: The question "is E ↔ I ↔ S a physical relationship or a structural analogy?" may not be empirically resolvable. What measurement would distinguish them? Without an answer, the physics claim is philosophy. But the *engineering* claim is testable: does c² characterize models usefully?

**What's needed**:
- Multi-model c² sweep: measure c² for ≥5 LLM families (BERT, GPT-4, Mistral, Qwen, DeepSeek-R1)
- If c² correlates with model capability or architecture: useful characterization tool
- If c² predicts transfer learning compatibility: immediate engineering application
- If c² is arbitrary across models: the analogy doesn't carry engineering weight

**What this is NOT**: A claim that information *is* energy in a physical sense. It's a claim that PAC conservation produces a geometric invariant in embedding spaces that behaves like a speed-of-light constant, and that this invariant may be practically useful.

**Status**: Single-model demonstration (n=1). Needs multi-model sweep for utility claim.  
**Contribution status**: Open  
**Last assessed**: 2026-02

---

## Category D: Engineering Demonstrations

These are working implementations, not measurements. They demonstrate feasibility, not truth.

### D1. GAIA POC-019 through POC-025

**Key results**:
- POC-019: Learning without backpropagation (attractor-based)
- POC-020: Multi-model PAC with 100% transfer
- POC-024: Phi weight ablation (direct φ test)

**Source**: `dawn-models/research/GAIA/proof_of_concepts/`

**What's needed**: Engineering demos → principled comparisons vs baselines with proper ablations and datasets. POC-024 (φ ablation) comes closest to a measurement — if removing φ degrades performance, that's evidence.

**Status**: Paper 6 will include select results with appropriate framing  
**Contribution status**: Active  
**Last assessed**: 2026-02

---

### D2. Pythia/GPT-2 φ-Convergence

**Observation**: Pythia-70M delta ratios approach φ at training step 512 (p = 0.0014). GPT-2 shows consistent patterns.

**Source**: `ml_validation_pythia_gpt2/`

**What's needed**: More model families. Currently one model (Pythia-70M) at one checkpoint. If the same convergence appears in Mistral, Qwen, LLaMA at similar steps, strong. If it's Pythia-specific, weak.

**Status**: Published on Zenodo; needs multi-family replication  
**Contribution status**: Open  
**Last assessed**: 2026-02

---

## Promotion Path

When a result from this document is tightened sufficiently:

1. Run the additional validation described in "What's needed"
2. Write up with error bounds following PACSeries voice
3. Add to the appropriate paper draft
4. Remove from this document and note the promotion in the changelog

Results that fail tightening (null test shows coincidence, alternative explanation found) should be moved to a "Retired" section at the bottom with an honest note about why.

---

## Retired

*(None yet)*

---

*This document is maintained alongside the PACSeries. It is not a publication — it is a research planning tool.*
