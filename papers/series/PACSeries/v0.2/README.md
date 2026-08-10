# PACSeries v0.2

**Date**: February 2026  
**Status**: Current (preprint)  
**Papers**: 6  
**Previous**: [v0.1](../v0.1/) (October 2025, Zenodo [10.5281/zenodo.17295103](https://zenodo.org/records/17295103))

## Papers

| # | Paper | Version | Focus |
|---|-------|---------|-------|
| 1 | [The Structure Cost of Erasure](structure_cost_of_erasure/) | 2.1 | Landauer + DPI → ξ emergence, ln(φ) partition |
| 2 | [The Balance Constant and Its Decomposition](balance_constant_decomposition/) | 2.1 | Ξ = γ + ln(φ) from five domains (p < 0.0003) |
| 3 | [Feigenbaum Constants from Fibonacci Arithmetic](feigenbaum_fibonacci_arithmetic/) | 2.1 | Closed-form Feigenbaum constants (6–13 digits) |
| 4 | [Standard Model Parameters from Fibonacci Arithmetic](standard_model_fibonacci_arithmetic/) | 2.1 | α to 5.7 ppm, sin²θ_W = 3/13, mass ratios |
| 5 | [Classical Physics from Information Geometry](classical_physics_information_geometry/) | 2.1 | Maxwell from PAC/SEC, D=3 from MED |
| 6 | [Computational Validation](computational_validation_pac_conservation/) | 2.1 | GAIA, PAC conservation in ML systems |

## Reading Order

1. **Paper 1** — Establishes the mechanism (erasure → structure) from undisputed physics
2. **Paper 2** — Derives the balance constant Ξ and its decomposition
3. **Paper 3** — Pure mathematics (Feigenbaum), hardest result to dismiss
4. **Paper 4** — Quantitative predictions (Standard Model)
5. **Paper 5** — Physics derivations (electromagnetism from information geometry)
6. **Paper 6** — Computational validation (GAIA implementations)

## The Derivation Chain

```
AXIOM: PAC conservation — f(Parent) = Σf(Children)
  → RECURSION: Ψ(k) = Ψ(k+1) + Ψ(k+2)
  → SOLUTION: Ψ(k) = φ^(-k) (unique stable)
  → INFO UNIT: ΔI = ln(φ)
  → ERASURE: A/(A+ξ) = ln(φ) (Paper 1: 0.76% error)
  → BALANCE: Ξ = γ + ln(φ) (Paper 2: 5 domains, 0.036% best)
  → UNIVERSALITY: Feigenbaum from 55 = F₁₀ (Paper 3: 13 digits)
  → PHYSICS: sin²θ_W = 3/13, α to 5.7 ppm (Paper 4)
  → MAXWELL: Depth-2 PAC → electromagnetism (Paper 5)
  → VALIDATION: PAC conservation in ML systems (Paper 6)
```

## Each Paper Contains

```
paper_name/
├── paper.md          # Full paper text
├── README.md         # Overview and reproduction instructions
├── meta.yaml         # Schema v2.0 metadata
├── Code/
│   ├── experiments/  # Numbered experiment scripts (exp_01..exp_NN)
│   ├── generate_figures.py
│   ├── reproduce.py  # Run all experiments
│   ├── trace.yaml    # Provenance: traces code/data to source repos
│   └── requirements.txt
├── Data/
│   └── results/      # JSON outputs from experiments
└── Figures/          # Publication-quality PNGs
```

## Voice and Standard

Each paper must:
1. Start from something established (a known law, a theorem, a measurement)
2. Derive the consequence (≤10 lines of math)
3. Present measurements with error bounds
4. Separate established from speculative — clearly, once
5. State what would falsify the claim
