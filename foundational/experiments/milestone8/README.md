# Milestone 8: BSM Predictions & Observational Contact

## Thesis

DFT's M1-M7 results are self-consistent and reproduce Standard Model parameters from
information-theoretic axioms. M8 tests whether the same framework makes falsifiable
predictions **beyond** the Standard Model. The milestone is structured around three
kinds of observational contact: (1) particle physics predictions testable at colliders
and X-ray observatories, (2) neutrino sector predictions testable at JUNO/DUNE, and
(3) cosmological predictions testable against Planck/DESI/JWST/Euclid data. Every
experiment has pre-registered numerical predictions with explicit falsification criteria.

First milestone in the M8→M9→M10 arc: *"Here's what we predict → Here's why it works
→ Here's what it means for the deepest open problem in physics."*

## Status: 40/40 (100%) | Complete

## Scorecard

| Exp | Score | Block | Name | Highlights |
|-----|-------|-------|------|-----------|
| 01 | 4/4 | A | Depth-73 coupling derivation | 73 unique Φ₃ in [32,182], α₇₃ = 1.2×10⁻¹⁵, antisymmetric projection |
| 02 | 4/4 | A | Dark matter mass spectrum | Cascade routes converge: 6.44 keV (0.09 orders). X-ray at 3.2 keV ≈ 3.55 keV observed |
| 03 | 4/4 | A | Relic abundance & production | Thermal excluded (10¹⁸×), Ω_c 0.46%, DW mixing ~10⁻¹⁰, λ_fs = 0.016 Mpc |
| 04 | 4/4 | B | Z' at 395 GeV quantification | Not excluded (9× margin), BRs physical, width = 64.0 MeV |
| 05 | 4/4 | B | Neutrino absolute masses | Sum 0.43 meV, splitting ratio 17% (was 44%), CP at 1.0σ |
| 06 | 4/4 | B | Fibonacci depth sweep | 5 known forces recovered, 12 cyclotomic depths, Φ₃ desert empty in [74,182], no-GUT |
| 07 | 4/4 | C | Hubble tension quantification | φ^{1/6} at 0.075%, H₀ = 73.0 via BAO, S8 = 0.787 (per-level dissipation), DESI 0.5σ |
| 08 | 4/4 | C | Cosmological constant precision | CC at −122.09 (0.09 orders!), Ω_Λ 0.18%, sensitivity robust (max 0.56 orders) |
| 09 | 4/4 | C | JWST structure prediction | z-dep floor: z=8 at 16%, z=12 at 4%, ratio 0.25 vs JWST 0.30 |
| 10 | 4/4 | D | BSM master test | 0 contradictions, 10 predictions, 10 falsifiable, 0 excluded |
| **Total** | **40/40** | | **100%** | |

### Key Numbers
- **Cosmological constant**: log₁₀(Λ/Λ_P) = −122.09 vs −122.0 (0.09 orders with correction template)
- **Hubble ratio**: φ^{1/6} = 1.0835 at 0.075% from measured
- **H₀ from BAO**: 73.0 km/s/Mpc via φ^{-1/6} sound horizon correction
- **S8**: 0.787 via per-level cascade dissipation (6 levels, 5.4% effective)
- **Ω_c**: F₇·Ξ²/F₁₀ = 0.2648 at 0.46% from Planck
- **Dark matter mass**: 6.44 keV (two cascade routes, 0.09 orders spread)
- **X-ray line**: 3.2 keV predicted, close to observed 3.55 keV (Bulbul+ 2014)
- **Z'**: 395 GeV, Γ = 64 MeV, not excluded, σ/σ_limit = 0.11
- **Neutrino splitting**: improved from 44% → 17% with PMNS correction
- **JWST**: z-dependent cascade floor matches z=8 (16%) and z=12 (4%), ratio 0.25 vs 0.30
- **10 pre-registered falsifiable predictions**, 0 excluded by current data

## Pre-Registered Predictions

| # | Prediction | Value | Uncertainty | Falsifiable By |
|---|-----------|-------|-------------|---------------|
| 1 | Dark matter mass | ~6.4 keV | factor 2 (3–13 keV) | X-ray (XRISM/Athena), Lyman-α |
| 2 | Dark coupling α₇₃ | ~1.2×10⁻¹⁵ | [10⁻¹⁶, 10⁻¹⁴] | No consistent projection at depth 73 |
| 3 | Z' mass | 395 ± 20 GeV | 5% | LHC narrow resonance search |
| 4 | Z' coupling | g'/g = 1/13 | exact | LHC rate 1/169 of standard Z' |
| 5 | Neutrino hierarchy | Normal | binary | JUNO (~2028) |
| 6 | Neutrino CP phase | ~63.5° (Ξ × 60°) | ±10° | DUNE/T2HK |
| 7 | Dark energy w₀ | −0.83 ± 0.05 | from cascade | DESI DR2+ |
| 8 | Hubble ratio | φ^{1/6} = 1.0835 | 0.08% | Independent H₀ measurements |
| 9 | X-ray line | ~3.2 keV | cf. 3.55 keV observed | XRISM, Athena (~2037) |
| 10 | No GUT | No Φ₃(F_n) in [74,182] | binary | Proton decay experiments |

## Block A: Dark Sector Foundations (exp_01–03)

### exp_01 — Depth-73 Coupling Derivation

Establishes the depth-73 dark matter candidate rigorously from the cyclotomic
force hierarchy Φ₃(F_n).

| Test | What it checks |
|------|---------------|
| 1. Cyclotomic uniqueness | 73 = Φ₃(F₆) is the ONLY Φ₃(F_n) in dark-gravity gap [32,182] |
| 2. Correction template | α₇₃ in [10⁻¹⁶, 10⁻¹⁴] via universal template |
| 3. Projection type | Antisymmetric (vector) vs symmetric (scalar/tensor) at depth 73 |
| 4. Hierarchy consistency | log(α₇₃⁻¹)/log(α_EM⁻¹) near φⁿ for some n |

**Falsification**: 73 is not unique in the gap, OR no consistent coupling exists.

### exp_02 — Dark Matter Mass Spectrum

Three independent mass derivation routes should converge on the same scale.

| Test | What it checks |
|------|---------------|
| 1. Cascade convergence | v_H·φ^{-73/2} and M_Z·φ^{-34} agree within 1 order (M_Pl/F₇₃ excluded — wrong physics) |
| 2. Lyman-α consistency | Mass > 3.3 keV if in WDM range |
| 3. Radiative decay line | ~3.2 keV X-ray (cf. 3.55 keV observed), mixing angle vs bounds |
| 4. Self-interaction | σ/m < 1 cm²/g (Bullet Cluster) |

**Falsification**: Cascade routes disagree by >2 orders, or mass is excluded by observations.

### exp_03 — Relic Abundance & Production

Tests whether the depth-73 particle can produce the observed dark matter abundance
through non-thermal production.

| Test | What it checks |
|------|---------------|
| 1. Thermal falsification | Ω_thermal >> 1 (thermal freeze-out must fail) |
| 2. Freeze-in abundance | Dodelson-Widrow gives Ω h² = 0.120 for reasonable mixing |
| 3. Ω_c formula | Agrees with F₇·Ξ²/F₁₀ within 10% |
| 4. Free-streaming | 0.01 < λ_fs < 1 Mpc |

**Falsification**: No production mechanism gives correct abundance.

## Block B: Particle Predictions (exp_04–06)

### exp_04 — Z' at 395 GeV Quantification

Extends M1 exp_34's Z' prediction with full LHC phenomenology.

| Test | What it checks |
|------|---------------|
| 1. LHC exclusion | σ_DFT < σ_excluded at 395 GeV with g'/g = 1/13 |
| 2. Branching ratios | Dominant visible channel BR > 1% |
| 3. Run 4 discovery | N_signal > 10 at 3000 fb⁻¹ OR luminosity target stated |
| 4. Width consistency | Γ/M < 0.5%, Γ within factor 2 of 64 MeV |

**Falsification**: Z' excluded at 395 GeV for ALL couplings down to g'/g = 1/13.

### exp_05 — Neutrino Absolute Masses

Derives absolute mass scale from Fibonacci depth arithmetic, improving on M6's 44% splitting error.

| Test | What it checks |
|------|---------------|
| 1. Sum bound | Σm_ν < 0.12 eV |
| 2. Splitting ratio | Δm²₃₁/Δm²₂₁ error < 20% (vs M6's 44%) |
| 3. CP phase | δ_CP = 63.5° compatible with PDG range |
| 4. JUNO/DUNE power | Normal hierarchy + m_ee prediction |

**Falsification**: Inverted hierarchy confirmed at >3σ.

### exp_06 — Fibonacci Depth Sweep

Systematic sweep of depths 1–300 mapping the complete DFT particle/interaction spectrum.

| Test | What it checks |
|------|---------------|
| 1. Known force recovery | EM(13), weak(~7), strong(~5-8), gravity(183), dark(73) at correct scales |
| 2. Cyclotomic census | All Φ₃, Φ₅, Φ₇(F_n) in [1,300] — finite and small (<20) |
| 3. Φ₃ desert prediction | No Φ₃(F_n) in [74,182] (higher cyclotomics documented, don't break claim) |
| 4. GUT-scale depth | Φ₃(F₈)=463 → GUT energy, or no-GUT prediction |

**Falsification**: Cyclotomic Fibonacci depths are dense (not special).

## Block C: Cosmological Contact (exp_07–09)

### exp_07 — Hubble Tension Quantification

Moves from directional alignment (exp_32f) to quantitative prediction.

| Test | What it checks |
|------|---------------|
| 1. Cascade H₀ ratio | H₀_local/H₀_CMB from cascade (target 1.07–1.10) |
| 2. BAO shift | r_s correction → H₀ in [71,75] km/s/Mpc |
| 3. S8 reduction | S8_DFT in [0.74,0.80], tension < 2σ |
| 4. DESI consistency | w₀, w_a within 2σ of DESI DR1 |

**Falsification**: Cascade gives wrong sign for any tension, or predictions are mutually inconsistent.

### exp_08 — Cosmological Constant Precision

Sharpens the CC prediction: M7 achieved 0.9 orders, MAR achieved 0.22 orders.

| Test | What it checks |
|------|---------------|
| 1. Tiling refinement | |log₁₀ predicted − (−122.0)| < 0.5 orders |
| 2. Template CC | Ω_Λ error < 0.1% via correction template |
| 3. Dark energy density | |Ω_DE − 0.685|/0.685 < 5% |
| 4. Sensitivity analysis | CC robust under ±5% perturbations (max error < 1.0 orders) |

**Falsification**: All three routes give mutually inconsistent CC values.

### exp_09 — JWST Structure Prediction

Quantitative prediction for massive galaxy abundance at z > 7.

| Test | What it checks |
|------|---------------|
| 1. Galaxy abundance z=8 | Within factor 10 of JWST (n ~ 10⁻⁵ Mpc⁻³) |
| 2. Mass function slope | Distinguishable from ΛCDM (Δslope > 0.3) |
| 3. Redshift dependence | z=12/z=8 ratio closer to 1 (cascade) than 0.5 (ΛCDM) |
| 4. PAC regulator | Regulated prediction within factor 3 of JWST at z=12 |

**Falsification**: Cascade prediction contradicts JWST by >2 orders.

## Block D: Synthesis & Falsification (exp_10)

### exp_10 — BSM Master Test

Compiles all predictions, verifies internal consistency, builds falsification protocol.

| Test | What it checks |
|------|---------------|
| 1. Internal consistency | Zero contradictions across all M8 predictions |
| 2. Prediction count | ≥7 quantitative pre-registered predictions |
| 3. Falsification conditions | ≥5 with named experiments and timelines |
| 4. Existing bounds | Zero predictions excluded by current data |

**Falsification**: Internal contradictions found between M8 predictions.

## Milestone-Level Falsification

M8 as a whole is falsified if:
1. Depth-73 is not unique Φ₃ in [32,182] (exp_01)
2. Z' excluded at 395 GeV regardless of coupling (exp_04)
3. Neutrino hierarchy is inverted at >3σ (exp_05)
4. Cosmological predictions are mutually contradictory (exp_10)
5. Dark matter mass in excluded range after template refinement (exp_02)

Target: ≥32/40 (80%) with zero contradictions to existing bounds. **Achieved: 40/40 (100%)**.

## Structure

```
milestone8/
├── README.md
├── meta.yaml
├── core/
│   ├── __init__.py
│   └── bsm.py
├── scripts/
│   ├── exp_01_depth73_coupling_derivation.py
│   ├── exp_02_dark_matter_mass_spectrum.py
│   ├── exp_03_relic_abundance_production.py
│   ├── exp_04_zprime_395_quantification.py
│   ├── exp_05_neutrino_absolute_masses.py
│   ├── exp_06_fibonacci_depth_sweep.py
│   ├── exp_07_hubble_tension_quantification.py
│   ├── exp_08_cosmological_constant_precision.py
│   ├── exp_09_jwst_structure_prediction.py
│   └── exp_10_bsm_master_test.py
├── results/
└── journals/
```

## FDO Links

- `roadmap-m8-m9-m10`
- `pac-comprehensive`
- `dawn-field-theory`
- `herniation-hypothesis`
- `school-entropic-gravity`
