# PACSeries Paper 2: The Balance Constant and Its Decomposition

**Author**: Peter Groom, Dawn Field Institute  
**Series**: PACSeries (Paper 2 of 6)  
**Status**: Draft  
**Date**: February 2026

## Overview

This paper shows that the balance constant Ξ — the value governing the boundary between ordered and disordered computation in recursive-conservation systems — decomposes as the sum of two established mathematical constants:

$$\Xi = \gamma + \ln(\varphi) \approx 0.5772 + 0.4812 = 1.0584$$

Four independent computational domains converge on this value with p < 0.004.

## Key Results

| Result | Value | Significance |
|--------|-------|-------------|
| Four-domain convergence | Ξ ≈ 1.057–1.058 | p = 0.00376 (n = 100,000) |
| Decomposition | Ξ = γ + ln(φ) | 0.124% max error |
| Class IV CA clustering | Rules 110/124 at 0.00077 | p < 10⁻⁷, enrichment 42.67× |
| Prime sieve PAC | Exact at 126/126 steps | Mertens error 0.012% |
| p = 3 φ-carrier | 82.1% of clustering impact | 2/3 = F₃/F₄ |
| Base invariance | PAC holds across 11 bases | Deviation < 10⁻¹⁴ |
| Discretisation gap | 1+π/55 vs γ+ln(φ) | 0.034%, Δk = γ/48 |

## Source Experiments

Data and scripts originate from four source experiments, traced via [Code/trace.yaml](Code/trace.yaml):

| Source | Experiments | Domain |
|--------|-------------|--------|
| `prime_growth_dynamics` | exp_01–05 | Fibonacci arithmetic, decomposition |
| `cellular_automata_pac_attractors` | exp_06 | Cellular automata |
| `asymmetric_conservation` | exp_07–09 | Prime number sieve |
| `base_agnostic_pac` | exp_10–11 | Base invariance |

### Package Contents

```
balance_constant_decomposition/
├── paper.md                    # Full paper text (§1–§13)
├── README.md                   # This file
├── meta.yaml                   # Schema 2.0 metadata
├── Code/
│   ├── reproduce.py            # Run all experiments
│   ├── generate_figures.py     # Generate all 6 publication figures
│   ├── requirements.txt        # numpy, scipy, matplotlib
│   ├── trace.yaml              # Provenance: local → source repo
│   └── experiments/
│       ├── exp_01_xi_derivation_contest.py   # §4.1: Three Ξ candidates
│       ├── exp_02_xi_exact_derivation.py     # §4.2: k = 10.0121
│       ├── exp_03_gamma_as_surplus.py        # §9.1: γ decomposition
│       ├── exp_04_gamma_falsification.py     # §9.2: γ falsification
│       ├── exp_05_universal_decomposition.py # §3.2: p = 0.00376
│       ├── exp_06_ca_xi_clustering.py        # §5: Class IV at Ξ
│       ├── exp_07_sieve_pac_conservation.py  # §6.1: PAC exact 126 steps
│       ├── exp_08_phase_decomposition.py     # §6.3: Three-phase model
│       ├── exp_09_p3_reconciliation.py       # §6.3: p=3 φ-carrier
│       ├── exp_10_base_invariance.py         # §8: 11-base test
│       └── exp_11_zeckendorf_validation.py   # §8: Zeckendorf
├── Data/
│   └── results/
│       ├── exp_23_xi_derivation_contest_*.json
│       ├── exp_25_xi_exact_derivation_*.json
│       ├── exp_29_results.json
│       ├── exp_30_results.json
│       ├── exp_31_results.json
│       ├── exp_07_definitive_*.json
│       ├── exp_14_sieve_as_local_sec_*.json
│       ├── exp_16_possibility_pruning_*.json
│       ├── exp_17_p3_reconciliation_*.json
│       └── exp_11_entropy_analysis_*.json
└── Figures/
    ├── fig1_domain_convergence.png   # §3 — four domains converge
    ├── fig2_fibonacci_depth.png      # §4 — Ξ across F₅–F₁₁
    ├── fig3_ca_clustering.png        # §5 — Class IV at Ξ
    ├── fig4_sieve_conservation.png   # §6 — PAC in prime sieve
    ├── fig5_decomposition.png        # §9 — γ + ln(φ) structure
    └── fig6_approximation_errors.png # §4.2 — discrete vs analytic
```

## Reproduction

```bash
cd Code
pip install -r requirements.txt
python reproduce.py              # Run all 11 experiments
python reproduce.py 1            # Run just exp_01
python reproduce.py --list       # List available experiments
```

All scripts are self-contained. JSON result files were captured from source experiment runs and are traced to their original locations in trace.yaml.

`generate_figures.py` loads all data from `Data/results/*.json` — no hardcoded values.

## Dependencies in the PACSeries

This paper builds on:
- **Paper 1**: Uses the ln(φ) finding from Landauer erasure to establish one component of Ξ

Papers that build on this one:
- **Paper 3**: Uses Ξ as the governing constant for Möbius/SEC dynamics
- **Paper 4**: Extends Ξ to Standard Model parameter predictions

## Paper File

- [paper.md](paper.md) — Full paper text
