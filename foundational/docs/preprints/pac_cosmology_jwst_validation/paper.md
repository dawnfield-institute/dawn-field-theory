# PAC/SEC Framework Applied to JWST High-Redshift Black Holes: Falsification Criteria and Initial Validation

**Authors**: Dawn Field Institute Research Team  
**Date**: January 5, 2026  
**Version**: 1.0  
**Category**: [pac][D][v1.0][C5][I5][E]  
**Status**: Draft

---

## Abstract

The James Webb Space Telescope (JWST) has revealed supermassive black holes (SMBHs) at z > 10 with masses exceeding 10⁷ M☉, presenting significant challenges to standard ΛCDM cosmology under realistic accretion assumptions. We apply the Potential-Actualization Conservation (PAC) and Symbolic Entropy Collapse (SEC) framework to this problem using a constraint-based methodology where theoretical constants (φ = 1.618..., Ξ = 1.0571) are derived, not fitted. Our analysis of 10 JWST-detected high-z SMBHs shows that PAC/SEC with direct collapse seeds explains 90% of observations, while ΛCDM with realistic duty cycles (20%) and Eddington ratios (30%) explains 0%. We establish four quantitative falsification criteria: SEC enhancement factor (1.17 ± 0.05), maximum seed mass threshold (<10⁶ M☉), duty cycle evolution (increasing with z), and run-length ratio (φ at z=0, φ² at high-z). These criteria make PAC/SEC a testable framework for future JWST discoveries.

**Keywords**: JWST, supermassive black holes, high-redshift cosmology, PAC theory, SEC dynamics, golden ratio, falsification, ΛCDM tension

> *This work represents ongoing research applying Dawn Field Theory to observational astrophysics. Results require independent validation and peer review. We present this as a testable framework, not established science.*

---

## 1. Introduction

### 1.1 The High-Redshift Black Hole Problem

JWST observations have revealed a population of surprisingly massive black holes in the early universe [1-5]. Objects like UHZ-1 (z = 10.073, M_BH ≈ 10^7.5 M☉), GN-z11 (z = 10.603, M_BH ≈ 10^6.2 M☉), and GLASS-z12 (z = 12.5, M_BH ≈ 10^6 M☉) exhibit black hole-to-stellar mass ratios 10-1000× higher than the local Magorrian relation.

Standard ΛCDM cosmology can accommodate these objects only with assumptions that are individually plausible but collectively strained:
- Near-continuous accretion (duty cycle ~100%)
- Sustained Eddington-limited or super-Eddington rates
- Heavy seed black holes (>10^5 M☉)
- Minimal feedback effects

With more realistic assumptions (duty cycle ~20%, Eddington ratio ~30%), standard models struggle to produce the observed masses within the available cosmic time.

### 1.2 PAC/SEC Framework

Potential-Actualization Conservation (PAC) and Symbolic Entropy Collapse (SEC) provide an alternative framework based on information-energy dynamics [6-8]. Key features include:

**PAC Recursion**: The fundamental equation Ψ(k) = Ψ(k+1) + Ψ(k+2) has the unique bounded solution Ψ(k) = φ^(-k), where φ = (1+√5)/2 = 1.618... is the golden ratio. This is derived, not fitted.

**Quantum Balance Equation (QBE)**: The constraint dI/dt + dE/dt = λ·QPL(t) couples information and energy dynamics, preventing runaway growth in either direction.

**SEC Phase Dynamics**: Run-length ratios in growth vs. contraction phases follow R(k) = φ^(1 + (k_eq - k)/2), producing asymmetric duty cycles that enhance early-universe growth rates.

### 1.3 Methodology: Constraints, Not Fits

A critical distinction: we do not sweep parameters to find "optimal" values. The constants in PAC/SEC are:
- **φ = 1.618...**: Derived from PAC recursion mathematics
- **Ξ = 1.0571**: Derived from Möbius/circle topology (1 + π/55)
- **7.42**: Measured from Euclidean Distance Validation experiments

Testing whether PAC works means testing whether these fixed constants produce physically consistent predictions—not adjusting them to match data.

### 1.4 Research Questions

**Q1**: Do PAC/SEC fixed constants provide physically plausible constraints on high-z SMBH formation?

**Q2**: How does PAC/SEC compare to ΛCDM under realistic (not optimistic) assumptions?

**Q3**: What specific, quantitative predictions would falsify PAC/SEC?

---

## 2. Theoretical Framework

### 2.1 PAC Recursion and Mass Hierarchy

The PAC recursion relation:

$$\Psi(k) = \Psi(k+1) + \Psi(k+2)$$

has the unique bounded solution $\Psi(k) = \phi^{-k}$. This generates a natural mass hierarchy:

$$M(k) = M_0 \cdot \phi^{k_{\text{ref}} - k}$$

where $M_0$ is a reference mass and $k_{\text{ref}}$ is the reference level. The hierarchy is discrete, with mass ratios between adjacent levels equal to φ.

### 2.2 Quantum Balance Equation

The QBE constrains information-energy dynamics:

$$\frac{dI}{dt} + \frac{dE}{dt} = \lambda \cdot QPL(t)$$

For SMBH growth, this translates to constraints on allowed accretion states. Given observable dE/dt (accretion luminosity), QBE specifies required dI/dt (information organization rate), filtering physically impossible configurations.

### 2.3 SEC Enhancement Mechanism

SEC phase transitions exhibit asymmetric run lengths:

$$R(k) = \phi^{1 + (k_{\text{eq}} - k)/2}$$

where $k_{\text{eq}} \approx 2$ is the equilibrium level. The duty cycle (fraction of time in growth phase) is:

$$\text{duty}(k) = \frac{R(k)}{R(k) + 1}$$

**At equilibrium (k = 2, z ≈ 0)**:
- R = φ = 1.618
- duty = 61.8%

**At high redshift (k ≈ 0, z > 8)**:
- R = φ² = 2.618  
- duty = 72.3%

The SEC enhancement factor is:

$$\epsilon = \frac{\text{duty}(z)}{\text{duty}(\text{equilibrium})} = \frac{0.723}{0.618} = 1.17$$

This is a **17% improvement in effective growth time**—modest but physically significant. It is NOT a 2.6× acceleration (an earlier error that was corrected).

### 2.4 Physical Interpretation

The SEC mechanism does not invoke exotic physics. It describes:
1. Asymmetric phase transitions (more time in growth than contraction)
2. Modulated by information-energy balance constraints
3. Producing enhanced duty cycle at high z where potential exceeds actualization

The framework predicts that early-universe conditions favor slightly longer sustained growth phases, not faster accretion rates.

---

## 3. Observational Data

### 3.1 JWST SMBH Catalog

We compile 10 high-z SMBHs from JWST surveys:

| Object | z | log(M_BH/M☉) | log(M*/M☉) | M_BH/M* | Source |
|--------|---|--------------|------------|---------|--------|
| GLASS-z12 | 12.50 | 6.0 | 8.0 | 0.01 | Castellano+2024 |
| GN-z11 | 10.60 | 6.2 | 9.0 | 0.0016 | Maiolino+2023 |
| UHZ-1 | 10.07 | 7.5 | 6.85 | 4.47 | Goulding+2023 |
| CEERS-1019 | 8.68 | 7.0 | 9.5 | 0.003 | Larson+2023 |
| CEERS-746 | 8.00 | 6.8 | — | — | CEERS |
| GLASS-38108 | 6.94 | 6.5 | — | — | Harikane+2023 |
| GLASS-160133 | 6.23 | 7.8 | — | — | Harikane+2023 |
| CEERS-2782 | 5.24 | 7.2 | — | — | Harikane+2023 |
| CEERS-1670 | 4.48 | 7.5 | — | — | Harikane+2023 |
| GLASS-150029 | 4.01 | 6.3 | — | — | Harikane+2023 |

**Key anomaly**: UHZ-1 exhibits M_BH/M* ≈ 4.47, dramatically exceeding the local Magorrian value (~0.001).

### 3.2 Time Constraints

Available growth time from seed formation to observation:

| Redshift | Cosmic Age (Gyr) | Time from z=30 (Myr) |
|----------|------------------|----------------------|
| z = 12.5 | 0.35 | 250 |
| z = 10 | 0.47 | 370 |
| z = 8 | 0.65 | 550 |
| z = 6 | 0.95 | 850 |

These short timescales constrain maximum achievable masses given seed mass and accretion physics.

---

## 4. Methods

### 4.1 PAC Maximum Mass Calculation

Given seed mass $M_{\text{seed}}$, we compute maximum achievable mass:

$$M_{\text{max}} = M_{\text{seed}} \cdot e^{t \cdot \epsilon / t_{\text{Edd}}}$$

where:
- $t$ = available growth time
- $\epsilon$ = SEC enhancement factor (1.17)
- $t_{\text{Edd}}$ = Eddington timescale (45 Myr)

### 4.2 Comparison Scenarios

We test multiple assumption sets:

| Scenario | Duty Cycle | Eddington Ratio | Framework |
|----------|------------|-----------------|-----------|
| ΛCDM Optimistic | 100% | 100% | Standard |
| ΛCDM Moderate | 50% | 50% | Standard |
| ΛCDM Realistic | 20% | 30% | Standard |
| PAC/SEC | 72% (SEC) | 50% | This work |

### 4.3 Seed Mass Assumptions

- **Stellar seeds**: ~100 M☉ (Population III remnants)
- **Direct collapse (DC) seeds**: ~10^5 M☉ (primordial gas collapse)
- **Heavy seeds**: ~10^6 M☉ (invoked by Goulding+ for UHZ-1)

---

## 5. Results

### 5.1 PAC vs ΛCDM Comparison

With direct collapse seeds (10^5 M☉):

| Scenario | Objects Achievable | Fraction |
|----------|-------------------|----------|
| ΛCDM Optimistic (100%/100%) | 10/10 | 100% |
| ΛCDM Moderate (50%/50%) | 6/10 | 60% |
| ΛCDM Realistic (20%/30%) | 0/10 | 0% |
| PAC/SEC (72%/50%) | 9/10 | 90% |

**Finding**: ΛCDM with optimistic assumptions explains all objects, but these assumptions are observationally disfavored. With realistic parameters, ΛCDM explains none. PAC/SEC explains 90% with moderate assumptions.

### 5.2 The UHZ-1 Case

UHZ-1 (log M_BH = 7.5) is the most challenging object:

| Seed Type | log(M_max) | UHZ-1 Status |
|-----------|------------|--------------|
| DC (10^5 M☉) | 6.6 | ✗ Fails by 0.9 dex |
| Heavy (10^6 M☉) | 7.6 | ✓ Achievable |

**Required seed for UHZ-1**: ~10^5.9 M☉, OR 1.6× super-Eddington with DC seeds.

Note: The original Goulding+ paper explicitly invokes heavy seeds for UHZ-1. PAC explains growth; seed mass is a separate question.

### 5.3 SEC Enhancement Verification

We verify the SEC enhancement mechanism:

| Redshift | k-level | Duty Cycle | Enhancement |
|----------|---------|------------|-------------|
| z = 0 | 2.40 | 59.5% | 0.96× |
| z = 2 | 0.24 | 71.2% | 1.15× |
| z = 6 | 0.02 | 72.3% | 1.17× |
| z = 10 | 0.005 | 72.3% | 1.17× |
| z = 15 | 0.002 | 72.4% | 1.17× |

The enhancement saturates at ~1.17× for z > 6, consistent with theoretical prediction.

### 5.4 Seed Mass Predictions

PAC/SEC predicts required seed masses for each object:

| Object | z | log(M_obs) | log(M_seed) PAC | log(M_seed) ΛCDM |
|--------|---|------------|-----------------|------------------|
| UHZ-1 | 10.07 | 7.5 | 5.87 | 7.23 |
| GN-z11 | 10.60 | 6.2 | 4.68 | 5.95 |
| GLASS-z12 | 12.50 | 6.0 | 4.79 | 5.80 |
| CEERS-1019 | 8.68 | 7.0 | 5.18 | 6.54 |

PAC requires lighter seeds than ΛCDM for the same final mass due to enhanced duty cycle.

---

## 6. Falsification Criteria

We establish four quantitative criteria that would falsify PAC/SEC cosmology:

### 6.1 SEC Enhancement Factor

**Prediction**: ε = 1.17 ± 0.05

**Falsification**: If observed high-z AGN duty cycles imply enhancement outside [1.10, 1.25], the SEC mechanism is invalidated.

**Test**: Compare intrinsic AGN duty cycles at z > 8 vs z ≈ 0 using complete surveys.

### 6.2 Maximum Seed Mass

**Prediction**: PAC/SEC should produce observed masses from seeds ≤ 10^6 M☉

**Falsification**: If multiple z > 10 SMBHs require seeds exceeding 10^6 M☉, PAC/SEC enhancement is insufficient.

**Test**: Constrain seed masses from host galaxy properties and BH demographics.

### 6.3 Duty Cycle Evolution

**Prediction**: Duty cycle increases from ~60% at z ≈ 0 to ~72% at z > 8

**Falsification**: If observed duty cycle decreases or remains constant with increasing z.

**Test**: AGN luminosity function evolution and active fraction measurements.

### 6.4 Run-Length Ratio

**Prediction**: Growth/contraction phase ratio = φ at z ≈ 0, → φ² at high z

**Falsification**: If observed phase ratios ≠ φ^n for any n.

**Test**: AGN variability studies measuring active/inactive phase durations.

---

## 7. Discussion

### 7.1 What PAC/SEC Explains

1. **Enhanced growth efficiency**: 17% more effective growth time at high z
2. **Physically reasonable seeds**: DC seeds (10^5 M☉) sufficient for 9/10 objects
3. **Self-consistent framework**: Constants derived, not fitted

### 7.2 What PAC/SEC Does Not Explain

1. **Seed formation**: PAC describes growth, not seed origin
2. **Individual outliers**: UHZ-1 may require heavy seeds or super-Eddington phases
3. **Detailed accretion physics**: PAC provides constraints, not microphysics

### 7.3 Comparison to ΛCDM

PAC/SEC is not a replacement for ΛCDM but a constraint framework that may operate within it. The key distinction is that PAC provides a principled mechanism for enhanced early-universe duty cycles, rather than requiring ad-hoc assumptions about accretion efficiency.

### 7.4 Sample Limitations

The 10-object sample is small. We note:
- Statistical conclusions are preliminary
- Selection effects may bias toward anomalous objects
- Future JWST surveys will expand the sample dramatically

### 7.5 Connections to Other PAC Validations

The constants φ and Ξ appear independently in:
- Cellular automata edge-of-chaos clustering [9]
- Prime number distribution thresholds [10]
- Neural language model phase transitions [11]
- Navier-Stokes symbolic emergence [12]

This cosmological application extends PAC to astrophysical observations.

---

## 8. Conclusions

We have applied the PAC/SEC framework to JWST high-z SMBH observations using constraint-based methodology. Key findings:

1. **PAC/SEC with DC seeds explains 90% of observations**; ΛCDM realistic explains 0%
2. **SEC enhancement is modest (1.17×)** but physically significant
3. **Four falsification criteria** make this a testable framework
4. **UHZ-1 requires heavy seeds** regardless of framework (not a PAC failure)

The framework makes specific, quantitative predictions that future JWST observations can confirm or falsify. We invite the community to test these predictions and validate the methodology independently.

---

## References

[1] Goulding, A. D., et al. (2023). UHZ1: A z > 10 AGN discovered with JWST and Chandra. *ApJ Letters*, 955, L24.

[2] Maiolino, R., et al. (2023). A small and vigorous black hole in the early Universe. *Nature*, 627, 59-63.

[3] Larson, R. L., et al. (2023). A CEERS Discovery of an Accreting Supermassive Black Hole 570 Myr after the Big Bang. *ApJ*, 953, L29.

[4] Castellano, M., et al. (2024). GLASS-z12: A luminous galaxy at z ∼ 12. *ApJ Letters*, 938, L15.

[5] Harikane, Y., et al. (2023). A Comprehensive Study of Galaxies at z ∼ 9-16 Found in the Early JWST Data. *ApJS*, 265, 5.

[6] Groom, P. L. (2025). Dawn Field Theory: Infodynamics and the Information-Energy Bridge. *Dawn Field Institute Preprint*.

[7] Groom, P. L. (2025). Potential-Actualization Conservation: Mathematical Foundations. *Dawn Field Institute Preprint*.

[8] Groom, P. L. (2025). QBE-PAC Unification: The 0.02 Hz Bridge. *Dawn Field Institute Preprint*.

[9] Dawn Field Institute. (2025). Cellular Automata Xi Clustering. *Zenodo*.

[10] Dawn Field Institute. (2025). Golden Ratio Threshold in Prime Distribution. *Zenodo*.

[11] Dawn Field Institute. (2025). ML Validation: Pythia and GPT-2 Phase Transitions. *Zenodo*.

[12] Dawn Field Institute. (2025). Macro Emergence Dynamics in Navier-Stokes. *Zenodo*.

---

## Appendix A: Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| φ | 1.6180339887... | PAC recursion: φ² = φ + 1 |
| Ξ | 1.0571 | Möbius/Circle: 1 + π/55 |
| SEC duty (z=0) | 61.8% | φ/(φ+1) |
| SEC duty (high-z) | 72.3% | φ²/(φ²+1) |
| Enhancement | 1.17× | 72.3%/61.8% |
| t_Edd | 45 Myr | Eddington timescale |

## Appendix B: JWST Object Details

Full observational details for each object including uncertainty ranges, detection methods, and source references are available in `Data/jwst_catalog.json`.

---

*Preprint: Dawn Field Institute, January 2026*
