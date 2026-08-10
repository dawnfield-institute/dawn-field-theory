# PACSeries Paper 4: Standard Model Parameters from Fibonacci Arithmetic

## Overview

This paper demonstrates that a significant subset of Standard Model free parameters — coupling constants, mixing angles, and mass ratios — can be expressed as closed-form Fibonacci ratios derived from the PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2).

## Key Results

| Parameter | Precision | Category |
|-----------|-----------|----------|
| α (fine structure) | 5.7 ppm | Gauge coupling |
| sin²θ_W (weak mixing) | 0.19% | Gauge coupling |
| α_s (strong coupling) | 1.71% | Gauge coupling |
| Koide Q = 2/3 | 0.5 ppm | Mass relation |
| m_μ/m_e | 5 ppm | Mass ratio |
| m_p/m_e | 83 ppm | Mass ratio |
| m_τ/m_e | 350 ppm | Mass ratio |
| Cabibbo angle | < 0.05° | CKM mixing |
| θ₁₂ (solar neutrino) | 0.28° | PMNS mixing |
| θ₁₃ (reactor neutrino) | 0.21° | PMNS mixing |
| (2αβ)² = 4/5 | Exact | Bell entanglement |
| Casimir 240 = F₃F₄F₅F₆ | Exact | QFT regularisation |
| k = d × F_{d+1} | Formula | Turbulence She-Lévêque |

## Falsifiable Predictions

- Z' boson at 395 ± 20 GeV with coupling g'/g = 1/13
- 4D turbulence intermittency k(4) = 20

## Source Experiments

- `foundational/experiments/milestone1/` — 40 scripts (gauge couplings, mass ratios, mixing)
- `foundational/experiments/milestone2/` — 40 scripts (mass derivations, Casimir, turbulence)
- `foundational/experiments/archive/era2/pac_confluence_xi/` — 45+ scripts (α derivation, Noether, synthesis)

## Dependencies

- Paper 1: Structure Cost of Erasure (Landauer interpretation of α)
- Paper 3: Feigenbaum Constants (F₁₀ = 55 universality)

## Reproduction

```bash
cd foundational/experiments/milestone1/validated/
python exp_12_alpha_formula.py
python exp_18_weinberg_angle.py
python exp_20_koide_formula.py

cd foundational/experiments/milestone2/mass_derivation/
python exp_05_tighten_mass.py
python exp_06_validate_tight.py
```

## Status

- [x] Draft complete
- [ ] Internal review
- [ ] Final voice pass
- [ ] Figures generated
- [ ] Code package assembled
