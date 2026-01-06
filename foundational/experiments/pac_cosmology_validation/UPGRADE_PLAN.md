# PAC Cosmology Validation - Rigorous Upgrade Plan

**Date**: January 6, 2026  
**Status**: Planning  
**Objective**: Transform viability test into publication-ready validation

---

## Assessment of Current State

### What We Have (Viability Test)
- 10 JWST objects (4 high-z, 6 lower-z)
- SEC duty cycle enhancement (1.17×)
- Basic ΛCDM comparison
- Four falsification criteria
- No statistical rigor

### What's Missing for Publication
1. **Comprehensive data**: Need 50+ objects with proper error bars
2. **Rigorous derivation**: SEC enhancement needs first-principles derivation
3. **Null hypothesis tests**: What does chance predict?
4. **Parameter sensitivity**: How robust are conclusions?
5. **Independent comparison models**: Not just ΛCDM
6. **Proper uncertainty propagation**: Monte Carlo on observational errors
7. **Physical simulation**: Integrate cosmo.py / reality-engine

---

## Phase 1: Data Compilation (Priority: Highest)

### Target: 50+ JWST/High-z Objects

**Sources to scrape:**
1. **Harikane et al. (2023)** - JWST AGN census at z=4-7 (~40 objects)
2. **Maiolino et al. (2023, 2024)** - GN-z11 and follow-ups
3. **Goulding et al. (2023)** - UHZ-1
4. **Larson et al. (2023)** - CEERS AGN sample
5. **Bogdan et al. (2024)** - X-ray detected AGN
6. **Kokorev et al. (2023, 2024)** - Little Red Dots population
7. **Kocevski et al. (2023)** - CEERS broad-line AGN
8. **Übler et al. (2023, 2024)** - High-z SMBH compilation
9. **Greene et al. (2024)** - JWST AGN review

**Data fields per object:**
```yaml
name: str
redshift: float
redshift_error: float
log_m_bh: float          # log10(M_BH/M_sun)
log_m_bh_error_low: float
log_m_bh_error_high: float
log_m_star: float        # log10(M*/M_sun)
log_m_star_error: float
detection_method: str    # broad_line, xray, photo, spectro
measurement_method: str  # virial, SED, scaling
super_eddington: bool    # Evidence of super-Edd accretion
arxiv_id: str
notes: str
```

### Catalog File
Create: `data/jwst_smbh_catalog_comprehensive.json`

---

## Phase 2: Theoretical Tightening

### Goal: Derive SEC Enhancement from First Principles

**Current weakness**: The 1.17× enhancement is stated but not derived cleanly.

**Derivation path:**

1. **PAC Recursion**: Ψ(k) = Ψ(k+1) + Ψ(k+2)
   - Unique bounded solution: Ψ(k) = φ^(-k)
   - This is the POTENTIAL at level k

2. **SEC Phase Dynamics**:
   - Information field oscillates between growth (I+) and contraction (I-)
   - Run lengths follow: L+ : L- = φ : 1 (from SEC prime manifold)
   - This is MEASURED, not assumed

3. **Duty Cycle Definition**:
   - duty = L+ / (L+ + L-) = φ / (φ + 1) = 1/φ ≈ 0.618
   - This is the EQUILIBRIUM duty cycle

4. **Early Universe Modulation**:
   - At high z, unactualized potential U → 1
   - Run-length ratio R(z) = φ × (1 + U(z)) where U(z) ∝ matter_fraction(z)
   - At equilibrium: R = φ, duty = 0.618
   - At z=10: U ≈ 0.9, R ≈ φ × 1.9 ≈ 3.07, duty = 0.754
   
5. **Enhancement Factor**:
   - ε(z) = duty(z) / duty(equilibrium)
   - Need to derive U(z) from PAC structure

**Key question**: What IS the PAC level k at redshift z?
- Option A: k ∝ log(1+z) (logarithmic mapping)
- Option B: k = arctan(z/z_equilibrium) (bounded)
- Option C: k = f(matter_fraction) (physics-based)

**Deliverable**: Clean 1-page derivation with no hand-waving

---

## Phase 3: Statistical Framework

### 3.1 Null Hypothesis Models

**H0-A: Random enhancement**
- Enhancement factor drawn from Uniform(1.0, 2.0)
- Does random enhancement explain data as well as φ-based?

**H0-B: Power-law enhancement**
- ε(z) = (1 + z)^α for fitted α
- Is PAC enhancement just a power law in disguise?

**H0-C: Constant enhancement**
- ε = constant (fitted)
- Does redshift-dependence matter?

**H0-D: Standard ΛCDM optimized**
- Fit duty cycle and Eddington ratio freely
- What parameters does ΛCDM need to match observations?

### 3.2 Statistical Tests

1. **Likelihood ratio test**: PAC vs null models
2. **AIC/BIC comparison**: Model selection
3. **Bootstrap confidence intervals**: On all predictions
4. **Monte Carlo uncertainty propagation**: Account for observational errors
5. **Leave-one-out cross-validation**: Predictive power

### 3.3 Metrics

- Fraction achievable (with CI)
- Mean/median residual
- RMSE
- χ² per degree of freedom
- Kolmogorov-Smirnov test for distribution match

---

## Phase 4: Parameter Sensitivity Analysis

### Variables to Sweep

1. **Seed mass**: 10², 10³, 10⁴, 10⁵, 10⁶ M☉
2. **Base duty cycle**: 0.5, 0.618, 0.7, 0.8
3. **Eddington ratio**: 0.1, 0.3, 0.5, 0.7, 1.0, 2.0 (super-Edd)
4. **Enhancement model**: PAC, power-law, constant, ΛCDM
5. **z-mapping function**: log, arctan, matter-based

### Sensitivity Metrics

- ∂(achievable_fraction) / ∂(parameter)
- Parameter robustness score
- Critical thresholds

---

## Phase 5: Simulation Integration

### cosmo.py Integration

The legacy `cosmo.py` uses:
- SHA-seeded entropy fields
- GPU-accelerated CUDA kernels
- QBE dynamics: dI/dt + dE/dt = λ·QPL(t)
- Matter generation from info-energy collapse

**Adaptation for SMBH growth:**
```python
# Modify cosmo.py to track individual collapse centers
# Each collapse center → potential SMBH
# Track mass accumulation under QBE constraint
```

### reality-engine Integration

The `reality-engine/cosmology/` already has:
- `pac_cosmology.py`: PAC state at redshift
- `jwst_predictions.py`: Object catalog + predictions
- `observables.py`: Cosmological predictions

**Merge into unified framework**

---

## Phase 6: Experiment Structure

### New Script Organization

```
pac_cosmology_validation/
├── UPGRADE_PLAN.md          # This file
├── core/
│   ├── constants.py         # DERIVED constants only
│   ├── pac_cosmology.py     # PAC state calculator
│   ├── sec_enhancement.py   # NEW: Clean SEC derivation
│   ├── growth_models.py     # NEW: ΛCDM, PAC, null models
│   └── statistics.py        # NEW: Full statistical framework
├── data/
│   ├── jwst_catalog_v2.json # Comprehensive 50+ objects
│   ├── literature_refs.bib  # All sources
│   └── validation/          # Cross-checks
├── scripts/
│   ├── exp_01_data_validation.py      # Verify catalog
│   ├── exp_02_sec_derivation.py       # Theoretical check
│   ├── exp_03_null_hypothesis.py      # H0 tests
│   ├── exp_04_parameter_sweep.py      # Sensitivity
│   ├── exp_05_monte_carlo.py          # Error propagation
│   ├── exp_06_model_comparison.py     # AIC/BIC
│   ├── exp_07_predictions.py          # Future falsification
│   ├── exp_08_simulation.py           # cosmo.py integration
│   └── exp_09_synthesis.py            # Final analysis
└── results/
```

---

## Success Criteria (Rigorous Version)

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Objects analyzed | ≥50 | ⬜ |
| PAC outperforms null | p < 0.01 | ⬜ |
| PAC outperforms ΛCDM optimized | ΔAIC > 10 | ⬜ |
| Predictions for z > 15 | Specific mass bounds | ⬜ |
| Parameter robustness | Stable across ±20% | ⬜ |
| Monte Carlo CI | 95% intervals computed | ⬜ |
| Cross-validation R² | > 0.7 | ⬜ |
| Independent code review | Passed | ⬜ |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Data compilation | 2 days | 50+ object catalog |
| 2. Theory tightening | 1 day | Clean SEC derivation |
| 3. Statistics | 2 days | Full framework |
| 4. Parameter sweeps | 1 day | Sensitivity analysis |
| 5. Simulation | 2 days | cosmo.py/reality-engine integration |
| 6. Synthesis | 1 day | Paper update |
| **Total** | **~9 days** | Publication-ready |

---

## Next Actions

1. [ ] Start with Phase 1: Build comprehensive JWST catalog
2. [ ] Read Harikane 2023 census paper
3. [ ] Create data scraping script
4. [ ] Design catalog JSON schema

---

*This plan transforms a viability test into rigorous science.*
