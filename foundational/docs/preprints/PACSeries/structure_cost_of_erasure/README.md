# PACSeries Paper 1: The Structure Cost of Erasure

**Author**: Peter Groom, Dawn Field Institute  
**Series**: PACSeries (Paper 1 of 6)  
**Status**: Draft  
**Date**: February 2026

## Overview

This paper establishes the foundational result for the PACSeries: information erasure into multi-mode environments necessarily creates correlational structure. Starting from two undisputed facts—Landauer's principle and the data processing inequality—it derives that erasure creates new inter-mode correlations (ξ) that are topological in character, with collapse efficiency at natural parameters falling near φ-family constants (~2% proximity), consistent with a cross-domain pattern at structural boundaries.

## Key Results

| Result | Value | Significance |
|--------|-------|-------------|
| ξ emergence | Mandatory for all multi-mode topologies | Follows from DPI |
| Temperature invariance | ξ identical at 100K–5000K | Topological, not thermodynamic |
| Partition ratio A/(A+ξ) | ~0.490 at default params (100 seeds) | ~2% proximity to ln(φ) |
| Complement ratio ξ/A | 1.086 vs predicted 1.078 | 0.76% error |
| Cascade amplification | 53× over single event | p = 2.75 × 10⁻³⁵ |
| Dense/sparse time ratio | 69× | p = 3.25 × 10⁻⁵ |
| Gauge ξ hierarchy | SU(3) > SU(2) > U(1) | p = 1.51 × 10⁻¹¹ |
| ln(φ) derivation | From PAC recursion (idealized) | Assumes A+ξ=1; ~2% vs simulation |

## Source Experiment

All data and scripts originate from `foundational/experiments/landauer_erasure_structure/` and are traced via [Code/trace.yaml](Code/trace.yaml).

### Package Contents

```
structure_cost_of_erasure/
├── paper.md                    # Full paper text (§1–§15)
├── README.md                   # This file
├── meta.yaml                   # Schema 2.0 metadata
├── Code/
│   ├── reproduce.py            # Run all experiments
│   ├── generate_figures.py     # Generate all 6 publication figures
│   ├── requirements.txt        # numpy, scipy, matplotlib
│   ├── trace.yaml              # Provenance: local → source repo
│   └── experiments/
│       ├── exp_01_landauer_xi.py           # §3–4: Core erasure simulation
│       ├── exp_02_topology_analysis.py     # §4.2: Temperature independence
│       ├── exp_03_ratio_analysis.py        # §6: Ratio → ln(φ)
│       ├── exp_04_cascade_robustness.py    # §6: 30-seed robustness
│       ├── exp_05_sec_collapse.py          # §6: Decay rate sweep
│       ├── exp_06_gauge_topology.py        # §5: Gauge group topologies
│       ├── exp_07_lie_algebra_entropy.py   # §5: Lie algebra entropy
│       ├── exp_08_falsification_suite.py   # §6,8: Falsification (3.9M combos)
│       ├── exp_09_conservative_rbf.py      # §12.3: RBF binding
│       ├── exp_10_thermodynamic_cascade.py # §10: Cascade (53× amplification)
│       ├── exp_11_time_computation.py      # §11: Time density (69× ratio)
│       ├── exp_12_causal_lag_test.py       # §9.2: Causal lag
│       ├── exp_13_causal_falsification.py  # §9.2: Causal falsification
│       ├── exp_14_pac_conservation.py      # §9.2: PAC ratio vs magnitude
│       ├── exp_15_gauge_group_hierarchy.py # §5,15: Gauge ξ hierarchy
│       └── exp_16_ln_phi_derivation.py     # §6,15: ln(φ) derivation
├── Data/
│   └── results/
│       ├── exp_01_results.json             # Core results — all topologies
│       ├── exp_04_robustness_test.json     # 30-seed robustness
│       ├── exp_05_sec_collapse.json        # Decay sweep
│       ├── exp_06_gauge_topology_*.json    # Gauge topology (2 runs)
│       ├── exp_07_lie_algebra_entropy_*.json
│       ├── exp_08_falsification_*.json     # 3.9M combination search
│       ├── exp_09_conservative_rbf_*.json       # RBF binding (Ξ = 1.0571)
│       ├── exp_10_thermodynamic_cascade_*.json  # 53× cascade amplification
│       ├── exp_11_time_computation_*.json       # Dense vs sparse regimes
│       ├── exp_12_causal_lag_test_*.json        # Causal lag (0.39% deviation)
│       ├── exp_13_causal_falsification_*.json   # Falsification suite
│       ├── exp_14_pac_conservation_*.json       # PAC conservation test
│       ├── exp_15_gauge_group_hierarchy_*.json  # Gauge ξ hierarchy
│       └── exp_16_ln_phi_derivation_*.json      # ln(φ) derivation
└── Figures/
    ├── fig1_coupling_topology.png     # §4.4 — topology determines ξ
    ├── fig2_information_budget.png    # §4.5 — PAC budget P = A + ξ + Θ
    ├── fig3_decay_ratio_sweep.png     # §6 — convergence to ln(φ)
    ├── fig4_cascade_amplification.png # §10.3 — 53× amplification
    ├── fig5_dense_sparse_regimes.png  # §11.2 — 69× dense vs sparse
    └── fig6_pac_ratio_stability.png   # §9.2 — ratio stability
```

## Reproduction

```bash
cd Code
pip install -r requirements.txt
python reproduce.py              # Run all 16 experiments
python reproduce.py 1            # Run just exp_01
python reproduce.py --list       # List available experiments
```

All scripts are self-contained — no core module dependencies. Scripts exp_01–08 save JSON to `Data/results/`; scripts exp_09–16 print results to stdout. JSON files for exp_09–16 were captured and structured during the Feb 2026 packaging session.

`generate_figures.py` loads all data from `Data/results/*.json` — no hardcoded values.

## Dependencies in the PACSeries

This paper is the **foundation**. It does not depend on other PACSeries papers.

Papers that build on this one:
- **Paper 2**: Uses ln(φ) finding to establish Ξ = γ + ln(φ) decomposition
- **Paper 4**: Extends gauge topology speculation to Standard Model parameters
- **Paper 5**: Derives Maxwell's equations from the PAC/SEC framework implied here

## Paper File

- [paper.md](paper.md) — Full paper text
