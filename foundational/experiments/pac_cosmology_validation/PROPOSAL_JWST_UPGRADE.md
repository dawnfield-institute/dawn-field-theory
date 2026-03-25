# Proposal: JWST Cosmology Upgrade

**Date**: 2026-03-25
**Thread**: Milestone 6, Thread 5 (Medium Priority)
**FDO**: `pac-cosmology-jwst` (confidence 0.45, status: developing)
**Target Paper**: PACSeries Paper 8 — "PAC Cosmology: Fixed-Constant Predictions for JWST High-Redshift Black Holes"

---

## 1. Executive Summary

This proposal upgrades the PAC cosmology validation from an early-stage viability test (4 JWST objects, no statistical rigor, hardcoded mass limits) into a publication-ready framework tested against 50+ high-redshift objects with full statistical machinery. The core claim is powerful: PAC's fixed constants (φ, Ξ) — derived from recursion mathematics, not fitted — predict SMBH masses at high redshift without free parameters. If validated, this is a zero-parameter cosmological prediction from information-theoretic first principles.

**Why it matters for DFT**: Cosmology is the weakest link in the Dawn Field Theory chain. The SM derivations (Papers 1–7) are strong. The GR derivation (MAR exp_30–36) is converging. But cosmological predictions remain at confidence 0.45 — the lowest of any active DFT thread. Upgrading this to rigorous status either validates DFT at cosmic scales or identifies where the theory breaks down. Either outcome is scientifically valuable.

**What's at stake**: JWST is producing data faster than theory can absorb it. The "impossibly early" massive galaxies and SMBHs at z > 10 are a genuine puzzle. If PAC's SEC enhancement mechanism explains this naturally — without exotic physics — that's a significant result. If it doesn't, we need to know that too.

---

## 2. Current State: Honest Assessment

### What Exists
- **4 JWST objects tested**: UHZ-1 (z=10.07), GN-z11 (z=10.60), CEERS-1019 (z=8.68), GLASS-z12 (z=12.5)
- **12 experiment scripts** (exp_01 through exp_12): recursion tests, QBE constraints, JWST comparison, ΛCDM comparison, falsification criteria, expanded analysis
- **Core modules**: `pac_cosmology.py`, `qbe_dynamics.py`, `sec_dynamics.py`, `pac_constraints.py`, `constants.py`
- **Theoretical chain**: Infodynamics → QBE → PAC Recursion → φ-Emergence → Mass Hierarchy → JWST Predictions
- **Three unique signatures** identified for testing against future JWST data
- **φ confirmed load-bearing**: Ψ(k) = Ψ(k+1) + Ψ(k+2) has unique bounded solution Ψ(k) = φ^(−k)

### What's Wrong
| Problem | Severity | Notes |
|---------|----------|-------|
| K-level hierarchy mismatch | **Critical** | k~43 vs k~5 — the mapping between PAC hierarchy levels and physical scales is inconsistent |
| Mass limits hardcoded | **Critical** | Cap is assumed, not derived from theory. MAR exp_22 (Eddington regulator) may fix this |
| No ΛCDM baseline | **High** | Cannot claim superiority without proper comparison. exp_07 exists but is incomplete |
| Only 4 high-z objects | **High** | Statistically meaningless sample size |
| 0.024 dex RMSE | **Suspicious** | May be spuriously good — overfitting to 4 data points isn't hard |
| SEC enhancement (1.17×) not derived | **High** | Stated but not derived from first principles |
| No null hypothesis testing | **High** | No AIC/BIC, no bootstrap, no Monte Carlo |
| No error propagation | **Medium** | Observational uncertainties not propagated through predictions |

### Exploratory Cosmological Density Formulae (Unvalidated)
These Fibonacci-derived expressions are suggestive but not confirmed:
- Ω_c (dark matter): F₇·Ξ²/F₁₀ = 0.2648 (Planck: 0.265 ± 0.007, 0.079% error)
- Ω_Λ (dark energy): F₆·Ξ²/F₇ = 0.6894
- Ω_b (baryonic): F₆·Ξ⁻²/F₁₂ = 0.0496

Close numerical match does NOT confirm physical derivation. Validation requires independent methodology.

---

## 3. The 6-Phase Upgrade Plan

### Phase 1: Compile JWST Catalog (50+ High-z Objects)

**Goal**: Build a comprehensive, properly-sourced catalog of JWST-era SMBH observations with full error bars.

**Specific data sources**:
1. **Harikane et al. (2023)** — JWST AGN census at z=4–7 (~40 objects, largest single source)
2. **Maiolino et al. (2023, 2024)** — GN-z11 and spectroscopic follow-ups
3. **Goulding et al. (2023)** — UHZ-1 X-ray detection
4. **Larson et al. (2023)** — CEERS AGN sample (broad-line selected)
5. **Bogdan et al. (2024)** — X-ray detected AGN at extreme redshifts
6. **Kokorev et al. (2023, 2024)** — "Little Red Dots" population (photometric AGN candidates)
7. **Kocevski et al. (2023)** — CEERS broad-line AGN demographics
8. **Übler et al. (2023, 2024)** — High-z SMBH compilation and kinematics
9. **Greene et al. (2024)** — JWST AGN review (meta-analysis)

**Per-object data fields**: name, redshift (±error), log(M_BH/M☉) (±asymmetric errors), log(M*/M☉) (±error), detection method (broad_line/xray/photo/spectro), measurement method (virial/SED/scaling), super-Eddington evidence (bool), arxiv_id, notes.

**Deliverable**: `data/jwst_smbh_catalog_comprehensive.json` — machine-readable, version-controlled, with provenance for every entry.

**Estimated effort**: 2 days (literature compilation is the bottleneck)

### Phase 2: Derive SEC Enhancement from First Principles

**Goal**: Replace the hardcoded 1.17× SEC enhancement with a clean derivation from PAC structure.

**The derivation chain**:
1. PAC Recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2) → Ψ(k) = φ^(−k)
2. SEC Phase Dynamics: Information field oscillates I+ / I-. Run lengths L+:L- = φ:1 (measured from SEC prime manifold)
3. Equilibrium duty cycle: duty = φ/(φ+1) = 1/φ ≈ 0.618
4. Early universe modulation: At high z, unactualized potential U → 1. Run-length ratio R(z) = φ × (1 + U(z))
5. Enhancement: ε(z) = duty(z) / duty(equilibrium)

**The key open question**: What is U(z)? Three candidate mappings:
- Option A: k ∝ log(1+z) — logarithmic (simplest)
- Option B: k = arctan(z/z_eq) — bounded (physically motivated)
- Option C: k = f(matter_fraction) — physics-based (strongest if derivable)

**Critical input from MAR**: exp_22 discovered the PAC Eddington regulator (dτ/dt ≤ (1+z)·Ξ), which provides a natural mass growth cap via tanh soft regulation. This may replace BOTH the hardcoded mass limits AND provide the z-dependent enhancement function from theory. The regulator also requires PAC dilation to be LOCAL — free-streaming photons exempt, only interacting baryonic matter gets the entropic boost. This naturally explains why early galaxies appear massive while CMB matches standard cosmology.

**Deliverable**: Clean 1-page derivation with no hand-waving, implemented in `core/sec_enhancement.py`

**Estimated effort**: 1 day (if the Eddington regulator path works; 3 days if ab initio derivation needed)

### Phase 3: Statistical Rigor (AIC/BIC, Null Hypothesis Testing)

**Goal**: Proper model comparison with null hypotheses and information criteria.

**Null models to test against**:
- H0-A: Random enhancement — ε drawn from Uniform(1.0, 2.0). Does random do just as well?
- H0-B: Power-law enhancement — ε(z) = (1+z)^α for fitted α. Is PAC just a power law in disguise?
- H0-C: Constant enhancement — ε = constant (fitted). Does z-dependence matter at all?
- H0-D: Standard ΛCDM optimized — fit duty cycle and Eddington ratio freely. What parameters does ΛCDM need?

**Statistical tests**:
1. Likelihood ratio test (PAC vs each null)
2. AIC/BIC comparison (model selection — PAC has 0 free parameters, nulls have 1–3)
3. Bootstrap confidence intervals (10,000 resamples on all predictions)
4. Monte Carlo uncertainty propagation (sample within observational error bars)
5. Leave-one-out cross-validation (predictive power on held-out objects)
6. Kolmogorov-Smirnov test (distribution shape match)

**Success threshold**: PAC must outperform nulls at p < 0.01 AND ΔAIC > 10 vs ΛCDM-optimized.

**Deliverable**: `core/statistics.py` + `core/growth_models.py` implementing all tests

**Estimated effort**: 2 days

### Phase 4: Parameter Sensitivity Analysis

**Goal**: Demonstrate that PAC predictions are robust, not fine-tuned.

**Sweep variables**:
| Parameter | Values | Purpose |
|-----------|--------|---------|
| Seed mass | 10², 10³, 10⁴, 10⁵, 10⁶ M☉ | Test initial condition sensitivity |
| Base duty cycle | 0.5, 0.618, 0.7, 0.8 | Confirm φ-based duty is special |
| Eddington ratio | 0.1, 0.3, 0.5, 0.7, 1.0, 2.0 | Include super-Eddington |
| Enhancement model | PAC, power-law, constant, ΛCDM | Model comparison |
| z-mapping function | log, arctan, matter-based | Test mapping robustness |

**Key metric**: ∂(achievable_fraction)/∂(parameter). Predictions must be stable across ±20% parameter variation.

**Deliverable**: Sensitivity plots + robustness scores

**Estimated effort**: 1 day

### Phase 5: Integration with cosmo.py / Reality Engine

**Goal**: Connect predictions to the simulation infrastructure.

**cosmo.py integration**: Modify to track individual collapse centers (each → potential SMBH), track mass accumulation under QBE constraint (dI/dt + dE/dt = λ·QPL(t)).

**Reality Engine integration**: `reality-engine/cosmology/` already has `pac_cosmology.py`, `jwst_predictions.py`, `observables.py`. Merge into unified framework that can run forward simulations from initial conditions to predicted SMBH populations.

**Deliverable**: Unified simulation that predicts SMBH mass function at arbitrary redshift

**Estimated effort**: 2 days

### Phase 6: Publication Path

**Goal**: Paper-ready synthesis with predictions for future observations.

**Paper structure (PACSeries Paper 8)**:
1. Introduction: The JWST puzzle — SMBHs too massive too early
2. Theory: PAC recursion → SEC enhancement → zero-parameter predictions
3. Data: 50+ object catalog with full provenance
4. Results: Predictions vs observations, statistical tests, model comparison
5. Predictions: Specific mass bounds for z > 15, testable by JWST Cycle 3+
6. Discussion: Implications for early universe, connection to SM derivations
7. Appendix: Full derivation, sensitivity analysis, code availability

**Falsification predictions to publish**:
- Maximum SMBH mass at z > 12: log(M/M☉) < 8.5
- M_BH/M* ratio bounds at z > 10
- Specific P(k) boost ~5.8% at small scales (from MAR exp_27)
- BAO scale shift: r_s ~ 142.9 Mpc (2.8% from standard)
- H₀ shift: +2.0 km/s/Mpc toward SH0ES value

**Estimated effort**: 1 day for synthesis; actual paper writing is separate

---

## 4. Connection to MAR Results

Two MAR experiments feed directly into this upgrade:

### exp_22: PAC Eddington Regulator
- Discovered: dτ/dt ≤ (1+z)·Ξ as natural mass growth cap
- Soft regulation via tanh gives smooth high-z behavior
- **Impact**: Replaces hardcoded mass limits with theoretically derived bounds
- **CMB consistency**: PAC dilation must be LOCAL — free-streaming photons exempt, only interacting baryonic matter gets entropic boost
- **Key insight**: This naturally explains why early galaxies appear unexpectedly massive while CMB matches standard cosmology — different sectors of matter experience different effective time dilation

### exp_27: Free-Streaming Signature (5 Testable Predictions)
- P(k) boost ~5.8% at small scales
- BAO r_s ~ 142.9 Mpc (2.8% shift from standard 147.09 Mpc)
- H₀ shift +2.0 km/s/Mpc toward SH0ES value
- S₈ tension direction matches (lensing > CMB)
- JWST excess mass explained by enhanced structure formation at z > 6
- **Falsifiable by**: Euclid, Roman Space Telescope, Simons Observatory
- **Impact**: These are concrete, testable predictions that go beyond SMBH masses into large-scale structure

### exp_18: Growth Model Warning
- Unregulated entropic time dilation predicts log M ~ 380 at high z (overflow)
- The Eddington regulator fixes this but mass predictions still fall short of observations
- **Implication**: May need super-Eddington accretion periods — which several JWST objects show evidence for

---

## 5. Experiment List

The upgrade requires a new experiment sequence (replacing/extending the existing exp_01–12):

| # | Experiment | Description | Phase |
|---|-----------|-------------|-------|
| 1 | `exp_01_catalog_validation` | Validate 50+ object catalog: completeness, error bar consistency, duplicate detection, cross-reference with literature | 1 |
| 2 | `exp_02_sec_derivation` | Derive SEC enhancement from PAC recursion + Eddington regulator (exp_22). Test all three z-mapping candidates | 2 |
| 3 | `exp_03_k_level_resolution` | Attack the k~43 vs k~5 hierarchy mismatch. Map PAC levels to physical scales consistently | 2 |
| 4 | `exp_04_null_hypothesis` | Test PAC against H0-A (random), H0-B (power-law), H0-C (constant), H0-D (ΛCDM-optimized) | 3 |
| 5 | `exp_05_aic_bic_comparison` | Full AIC/BIC model selection across all growth models. PAC has 0 free parameters — this is its strength | 3 |
| 6 | `exp_06_monte_carlo` | Propagate observational uncertainties through PAC predictions. 10,000 Monte Carlo samples per object | 3 |
| 7 | `exp_07_bootstrap_cv` | Bootstrap confidence intervals + leave-one-out cross-validation on full catalog | 3 |
| 8 | `exp_08_parameter_sweep` | Sensitivity analysis: seed mass, duty cycle, Eddington ratio, z-mapping. Identify critical thresholds | 4 |
| 9 | `exp_09_super_eddington` | Separate analysis of objects with super-Eddington evidence. Does PAC naturally accommodate these? | 4 |
| 10 | `exp_10_free_streaming` | Implement exp_27's 5 predictions. Test P(k) boost, BAO shift, H₀ correction against available data | 4 |
| 11 | `exp_11_cosmo_simulation` | Forward simulation from initial conditions → SMBH population at z=6,8,10,12. Compare to observed mass function | 5 |
| 12 | `exp_12_re_integration` | Reality Engine integration: run PAC cosmology through Möbius topology + Poincaré activation | 5 |
| 13 | `exp_13_density_validation` | Independent validation of Ω_c, Ω_Λ, Ω_b Fibonacci formulae against Planck 2020 + independent measurements | 5 |
| 14 | `exp_14_falsification_bounds` | Compute and publish specific mass bounds for z > 15. These are the falsification predictions | 6 |
| 15 | `exp_15_synthesis` | Final synthesis: all results, all comparisons, paper-ready figures and tables | 6 |

---

## 6. Falsification Conditions

PAC cosmology is **FALSIFIED** if any of the following occur:

### Hard Falsification (Abandon Thread)
1. **Mass violation at extreme redshift**: SMBHs with log(M/M☉) > 8.5 confirmed at z > 12 (violates PAC hierarchy bound)
2. **φ not load-bearing**: Replacing φ with arbitrary constants gives equivalent or better predictions
3. **Null model wins**: Power-law or ΛCDM-optimized outperforms PAC at ΔAIC > 10 (PAC adds no explanatory value)
4. **QBE residuals uncorrelated**: No correlation between QBE constraint violations and observational anomalies

### Soft Falsification (Revise, Don't Abandon)
5. **K-level mismatch unresolvable**: If exp_03 cannot map hierarchy levels consistently, the theory needs restructuring but φ-necessity may still hold
6. **Super-Eddington required**: If PAC predictions only work with super-Eddington accretion, the enhancement mechanism is incomplete
7. **Free-streaming predictions wrong**: If exp_27's 5 predictions fail, the SEC-cosmology connection needs revision but SMBH-specific predictions may still hold

### What Would Strengthen the Thread
- New JWST objects falling within predicted mass bounds
- Independent confirmation of BAO shift or H₀ correction
- K-level resolution that unifies particle physics and cosmological scales

---

## 7. Target Paper: PACSeries Paper 8

**Working title**: "PAC Cosmology: Zero-Parameter Predictions for JWST High-Redshift Supermassive Black Holes"

**Key selling point**: This is a zero-free-parameter model. φ is derived from recursion mathematics. Ξ is derived from topology. The SEC enhancement is derived from phase dynamics. Unlike ΛCDM fits that optimize 2–6 parameters, PAC either works with its fixed constants or it doesn't.

**Submission target**: After all 15 experiments complete and at least 5/8 success criteria met (see UPGRADE_PLAN.md). Realistic timeline: 2–3 months from start of Phase 1.

**Pre-print strategy**: Post to arXiv (astro-ph.CO) once Phase 3 statistical tests pass, before simulation integration. This establishes priority on the zero-parameter prediction claim.

---

## 8. Timeline & Dependencies

```
Week 1:  Phase 1 (Catalog) ─────────────────────────────►
Week 1:  Phase 2 (SEC Derivation) ──────────────────────►
Week 2:  Phase 3 (Statistics) ──────────────────────────►
Week 3:  Phase 4 (Sensitivity) ─────►
Week 3:  Phase 5 (Simulation) ──────────────────────────►
Week 4:  Phase 6 (Synthesis + Paper) ───────────────────►
```

**Dependencies**:
- Phase 2 depends on MAR exp_22 results (available — complete)
- Phase 3 depends on Phase 1 (need catalog) and Phase 2 (need derived enhancement)
- Phase 5 depends on Reality Engine v3 stability (138 tests passing — ready)
- Phase 6 depends on all prior phases

**Blockers**:
- Literature access for catalog compilation (may need arXiv scraping)
- K-level hierarchy mismatch (exp_03) could stall Phase 2 if no resolution found
- If SEC enhancement cannot be derived from first principles, the proposal weakens significantly

**Cross-thread connections**:
- Thread 3 (CC gap, 0.22 orders) — shares cosmological density formulae; results feed each other
- Thread 4 (Simulator scorecard) — Reality Engine improvements benefit Phase 5
- MAR exp_35–36 — Ω_Λ correction template (0.012% error) provides the density parameter predictions tested in exp_13

---

*This proposal transforms a promising but under-developed viability test into a rigorous, falsifiable cosmological prediction framework. The honest assessment is that confidence is currently low (0.45) and several critical problems remain unsolved. But the potential payoff — a zero-parameter cosmological model from information-theoretic first principles — justifies the investment.*
