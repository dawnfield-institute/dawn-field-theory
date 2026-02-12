# Paper 3: Feigenbaum Constants from Fibonacci Arithmetic

**PACSeries** — Dawn Field Theory  
**Status**: Draft (February 2026)

## Summary

Closed-form expressions for the three Feigenbaum constants of period-doubling universality:

| Constant | Formula | Sig. figs | Rel. error |
|----------|---------|-----------|------------|
| r∞ (accumulation) | π(55+√(17−π/(55d)))(55+π)/55² − correction | **13** | 1.16 × 10⁻¹⁴ |
| δ (bifurcation) | (50050+32π)/(10725+5π) | **8** | 1.20 × 10⁻⁹ |
| \|α\| (scaling) | (2700+π)/1080 | **6** | 4.02 × 10⁻⁷ |

Structural constants: 55 = F₁₀, 17 = 2⁴+1 (Fermat prime), 52 = F₁₀ − F₄.

Statistics: exhaustive search of 3.9M parameter triples finds (55,17,52) uniquely optimal. Combined odds against coincidence: **1 in 280 billion**.

## Reproduction

```bash
# Install dependencies
pip install -r Code/requirements.txt

# Run all experiments
python Code/reproduce.py

# Run a specific experiment
python Code/reproduce.py 1    # exp_01 only

# Generate figures
python Code/generate_figures.py
```

### Requirements

- Python 3.9+
- numpy ≥ 1.20.0
- scipy ≥ 1.7.0
- mpmath ≥ 1.3.0
- matplotlib ≥ 3.5.0

## Experiment → Section Map

| Exp | Script | Paper § | Result |
|-----|--------|---------|--------|
| 01 | `exp_01_feigenbaum_all_constants.py` | §2 | r∞ 13 digits, δ 8, α 6 |
| 02 | `exp_02_statistical_proof.py` | §4 | 1 in 280B; 16.4 surplus digits |
| 03 | `exp_03_renormalization_analysis.py` | §5 | det = −2F₇π; base+correction |
| 04 | `exp_04_crossratio_mobius.py` | §5.2 | CR → 1.16995; gaps → δ |
| 05 | `exp_05_high_precision_validation.py` | §6 | A₃/A₂ = 6050 = 2F₁₀² |
| 06 | `exp_06_theoretical_framework.py` | §6 | 1857 = F₁₀F₉−F₇; δ self-consistent to 6 digits |
| 07 | `exp_07_rbf_self_closing.py` | §7 | δ = φ^(20/N); 13 digits; 3 iterations |
| 08 | `exp_08_universality.py` | §8 | Δz universal; scale ratio = 4 |
| 09 | `exp_09_cross_domain_validation.py` | §9 | 5/5 domains significant; joint p = 8.3e-12 |

## Package Structure

```
feigenbaum_fibonacci_arithmetic/
├── paper.md                    # Main paper (§1–§14)
├── meta.yaml                   # Schema v2.0 metadata
├── README.md                   # This file
├── Code/
│   ├── reproduce.py            # Run all experiments
│   ├── generate_figures.py     # Generate publication figures
│   ├── requirements.txt        # Python dependencies
│   ├── trace.yaml              # Provenance trace to source repo
│   └── experiments/
│       ├── exp_01_feigenbaum_all_constants.py
│       ├── exp_02_statistical_proof.py
│       ├── exp_03_renormalization_analysis.py
│       ├── exp_04_crossratio_mobius.py
│       ├── exp_05_high_precision_validation.py
│       ├── exp_06_theoretical_framework.py
│       ├── exp_07_rbf_self_closing.py
│       ├── exp_08_universality.py
│       └── exp_09_cross_domain_validation.py
├── Data/
│   └── results/
│       ├── exp_07_feigenbaum_all_constants_20260106_161706.json
│       ├── exp_08_renormalization_analysis_20260106_162646.json
│       ├── exp_09_statistical_proof_20260106_164302.json
│       ├── exp_10_crossratio_mobius_20260106_185922.json
│       ├── exp_24_high_precision_20260107_141918.json
│       ├── exp_25_theoretical_framework_20260107_144129.json
│       └── exp_28_conservation_phi_fibonacci_chain_20260107_164108.json
└── Figures/
    ├── fig1_precision_hierarchy.png
    ├── fig2_cross_domain_validation.png
    ├── fig3_phi_sensitivity.png
    ├── fig4_statistical_proof.png
    ├── fig5_fibonacci_selectivity.png
    └── fig6_formula_precision.png
```

## Falsification

These results would be falsified by:
1. Other triples matching at 7+ digits in expanded search (a, b > 200)
2. Δz differing between maps at precision > 10⁻¹⁰
3. Möbius perturbation series diverging at higher orders
4. Alternative constant sets (e, √2, Lucas) matching at comparable precision

## License

AGPL-3.0 (code), CC-BY-4.0 (paper)
