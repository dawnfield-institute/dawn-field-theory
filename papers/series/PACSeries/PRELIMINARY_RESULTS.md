# PACSeries: Preliminary Results and Open Leads

**Status**: Working document  
**Updated**: 2026-02-19  
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
**Contribution status**: Open (3-month rule applies — inactive since creation)  
**Last assessed**: 2026-02-19  
**Milestone 3 update**: No experiment in milestone3 directly tested this. The oscillation amplitude ratio is still a single-number match without derivation. Given that phi_artifact_test showed Ξ is metric-dependent, similar caution applies here: the 1/φ match needs a mechanism, not just measurement.

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
**Milestone 3 update**: exp_27 (F25) provides indirect support — the π→φ→Fibonacci mechanism chain means φ-based closed forms are expected. prime_growth_dynamics_v2 measured λ* at 0.017% and β at 0.026%. But uniqueness remains untested: need exhaustive search over (F_a ± F_b)/(F_c ± F_d) parameter space to confirm these are the best or only solutions. If uniqueness holds, promote to Paper 2.

---

### A4. Dark Matter Density from Fibonacci Depth Map

**Observation**: Ω_c = F₇·Ξ²/F₁₀ = 0.2587 vs Planck 2018 observed 0.2589 (0.079% error). Also F₃·Ξ/F₆ at 0.148%. Cyclotomic depth F₆²+F₆+1 = 73 maps to ~15 keV (sterile neutrino range).

**Source**: `milestone3/scripts/exp_25_dark_matter_depth.py`

**Statistics**: 0.079% error; two independent Fibonacci formulas converge

**What's needed**:
- Parameter counting: how many Fibonacci expressions of form F_a·Ξ^n/F_b exist near Ω_c? If many, this is look-elsewhere. Need exhaustive search analogous to exp_09 methodology
- Physical derivation: why should dark matter live at F₇ depth specifically? The cyclotomic argument (73 = Φ₃(F₆)) is structural but not yet derived from first principles
- Comparison: does the same template predict Ω_b (baryonic)? If F_a·Ξ^n/F_b also fits Ω_b, the template is too flexible
- Universe φ-equilibrium crossing at z ≈ 0.10 is interesting but the 6.7pp deviation from 1/φ needs error analysis

**Promotion path**: If uniqueness holds (not many alternatives) AND physical derivation found → Paper 5 material

**Status**: Strong numerical match; needs uniqueness test and derivation  
**Contribution status**: Open  
**Last assessed**: 2026-02-19

---

### A5. Unified Fibonacci Correction Template F_a/(mπF_b²)

**Observation**: Both α_EM and gravity use corrections of the form F_a/(mπF_b²), with opposite signs (minus for EM screening, plus for gravity enhancement). Both anchored to F₇=13. Index gaps a−b are 3=F₄ (EM) and 7=F₇ (gravity) — both Fibonacci. 0/5000 Monte Carlo random integer sequences match both simultaneously.

**Source**: `milestone3/scripts/exp_26_correction_template.py`, `milestone3/scripts/exp_27_phase_cascade.py`

**Statistics**: 0/5000 MC; α_EM at 5.7 ppm, gravity at 0.0008 log₁₀

**What's needed**:
- Template only works for 2/5 tested constants below 100 ppm threshold (sin²θ_W at 24.1 ppm, Ω_c at 38.8 ppm). The remaining 3/5 don't fit the template well
- Why does the template fail for some constants? Is there a systematic pattern to which constants obey it?
- Physical derivation: why F_a/(mπF_b²) and not some other form? exp_27 (F25) connects this to ((φ²+1)/π, 0.62% match) via convergent error bounds — but that's a structural argument, not a derivation
- Need: predict a correction for a constant not yet tested (genuine prediction, not postdiction)

**Promotion path**: If 3+ constants fit AND physical derivation found → Paper 5 material. Currently 2/5 is below threshold.

**Status**: Pattern real for α_EM and gravity (0/5000 MC); not yet universal (2/5)  
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

**Milestone 3 update (exp_27, F25)**: π-optimality now has partial mechanistic support. exp_27 demonstrated that on a π-closed phase manifold (S¹), the golden angle α* = 1 − 1/φ minimises worst-case star discrepancy D*_N. The causal chain π (closure) → φ (non-resonance) → Fibonacci (integers) provides context for why π would be special. However, this is still one step removed: exp_27 shows π creates the stage on which φ is optimal, but doesn't explain the 19× ratio itself.

**What's still needed**:
- Derivation of the 19× ratio from the π→φ mechanism. If the ratio is predictable from the cascade framework, this is Paper 3 material
- Proof that the comparison is not an artefact of how coherence is measured
- The core claim is still one ratio (19×). Additional metrics or scales testing this ratio would strengthen it

**Status**: Measured; partial mechanistic support from exp_27, but the specific ratio is not derived  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02-19

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
- **AIC/BIC model comparison** (CRITICAL before citing): The 100% vs 41% comparison is meaningless without parameter counting. PAC may fit better simply because it has more free parameters. Compute AIC = 2k − 2ln(L) and BIC = k·ln(n) − 2ln(L) for both frameworks. If AIC/BIC still favours PAC after penalising for parameters, the claim strengthens. If not, the comparison is misleading and should not be cited.
- Independent astrophysics review of methodology
- Cross-check with non-JWST high-z observations
- **Do not cite the 100% vs 41% comparison in any paper until AIC/BIC is computed**

**Status**: Published on Zenodo; **BLOCKED on AIC/BIC analysis** before results can be used  
**Contribution status**: Open (well-defined: compute AIC/BIC for both models on the same dataset)  
**Last assessed**: 2026-02-19

---

### C3. DNA Prime Structure

**Observation**: 40 experiments exploring PAC patterns in DNA/genomic data. Key validated findings:
- Fibonacci enrichment in SEQUENCE organization: 1.28× enrichment, z = +103.4 (p < 10⁻²⁴)
- NOT in 3D geometry: −28% depleted in structural contacts
- Flexibility correlation: flexible residues 6.92× enriched vs rigid 4.01×
- Function-specific: Fibronectin 10.25×, Myosin 9.65× enrichment

**Source**: `experiments/dna_prime_structure/` (40 scripts), `experiments/studies/dna_prime_structure/SYNTHESIS.md`

**What's needed**:
- Cross-validation on non-PDB datasets (UniProt, AlphaFold DB)
- Null model: compare Fibonacci spacing against other integer sequences (Lucas, Tribonacci) in same analysis
- The z=+103.4 is strong, but is it Fibonacci-specific or just "spread-out residues are enriched"?
- Domain-expert review of the conformational signaling hypothesis
- Too domain-specific for core PACSeries; standalone biology publication path

**Status**: Strong signal (z=+103.4) but Fibonacci-specificity untested; needs null comparison  
**Contribution status**: Open  
**Last assessed**: 2026-02-19

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

## Category E: Mechanism Leads from exp_22–28 (NEW)

These results from milestone3 Blocks G/H/I passed falsification but raise new questions that need further work.

### E1. PAC-Lazy Bootstrap Fragility

**Observation**: exp_21 (F19) showed PAC-Lazy formula discrimination works (KL p=0.035, d=0.198). But exp_24 (F22) found the bootstrap CI includes zero — the signal is not robust under resampling.

**Source**: `milestone3/scripts/exp_24_pac_lazy_anatomy.py`

**What it means**: The PAC-Lazy architecture (from GAIA POCs) transfers to formula space, but raw counting is biased. Conservation normalization is required. The discrimination exists but isn't stable enough for claims.

**What's needed**:
- Identify which component of the PAC-Lazy profile carries the signal (depth? breadth? conservation ratio?)
- Test whether conservation-normalized profiles survive bootstrap
- If the signal is in conservation ratios (not raw counts), the discrimination may be rescuable

**Status**: Mechanism validated but signal fragile under bootstrap  
**Contribution status**: Open  
**Last assessed**: 2026-02-19

---

### E2. φ-Equilibrium of the Universe at z ≈ 0.10

**Observation**: The dark energy fraction 1/φ = 61.8% vs observed 68.5% differs by 6.7 percentage points. Universe crossed the φ-equilibrium at z ≈ 0.10 (recent cosmological past).

**Source**: `milestone3/scripts/exp_25_dark_matter_depth.py`

**What's needed**:
- The 6.7pp deviation is large. Is this within expected scatter, or does it falsify the idea?
- Error propagation: what's the uncertainty on the z ≈ 0.10 crossing?
- If the crossing is real, is it a coincidence (we live near z=0) or physically meaningful?
- This is the weakest finding from exp_25. The Ω_c formula at 0.079% is much stronger.

**Status**: Interesting observation; needs error analysis and theoretical motivation  
**Contribution status**: Guidance needed  
**Last assessed**: 2026-02-19

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

Results moved here either failed tightening, were falsified, or were superseded by stronger results.

### R1. Fibonacci-Specific Crystallization Order — FALSIFIED (2026-02-18)

**Original claim**: Fibonacci input produces different crystallization dynamics than other sequences in the PAC cascade.

**Falsification**: milestone3/exp_19 (F17). Crystallization order is **entirely determined by target physics**, NOT by input sequence. 0% difference across Fibonacci/Lucas/Primes/Tribonacci/Random. All produce identical order: sin²θ_W → Koide → She-Lev → ν_WF → α_s → α_em.

**Impact**: The Fibonacci *structure* of PAC is validated extensively elsewhere (Layer 1–7 in UNIFIED_EVIDENCE). What's falsified is the claim that Fibonacci *input* changes *dynamics*. Physics has a natural complexity hierarchy that is sequence-independent.

**Source**: `milestone3/FALSIFICATION_REGISTRY.md`, F17

---

### R2. Fractal Mesh Raw Pressure as Physics Discriminator — FALSIFIED (2026-02-18)

**Original claim**: Fractal recursive decomposition of formula space preferentially selects physics matches via mesh pressure.

**Falsification**: milestone3/exp_20 (F18). Raw pressure correlates with **index depth, not physics**. Physics matches have LOWER average pressure than non-matches (delta = −2703, p = 0.78, WRONG direction). 33.6× amplification over flat confirms fractal structure is real, but visit counting conflates structural depth with physical significance.

**Impact**: Led directly to exp_21's conservation-based approach (F19), which succeeded. The failure clarified that **raw counting ≠ physics discrimination** — conservation normalization is needed.

**Source**: `milestone3/FALSIFICATION_REGISTRY.md`, F18

---

### R3. Phase Ordering of Primes — FALSIFIED (2026-02-14)

**Original claim**: Primes exhibit a specific phase ordering in the SEC manifold that reflects their structural role.

**Falsification**: pac_foundations_validation/exp_05. The claimed ordering was an artefact. Removed from UNIFIED_EVIDENCE v3.4.

**Impact**: None on other results — the SEC partition at 1/φ (validated independently in phi_artifact_test) does not depend on phase ordering.

**Source**: `pac_foundations_validation/SYNTHESIS.md`, `UNIFIED_EVIDENCE.md` v3.4 changelog

---

*This document is maintained alongside the PACSeries. It is not a publication — it is a research planning tool.*

**Last substantive update**: 2026-02-19 (added A4/A5/E1/E2, updated A2/A3/B3/C2/C3, populated Retired section)
