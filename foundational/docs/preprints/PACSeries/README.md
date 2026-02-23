# PACSeries: Dawn Field Theory

**Current Release**: v0.2 (February 2026, 6 papers)  
**Previous Release**: v0.1 (October 2025, 5 papers — Zenodo [10.5281/zenodo.17295103](https://zenodo.org/records/17295103))

## Overview

The PACSeries establishes Dawn Field Theory through clean derivation from established science, measurement with error bounds, and honest separation of established results from speculation. Papers are numbered by logical dependency, not historical order.

## Versions

### [v0.2/](v0.2/) — Current (February 2026)

Six papers, restructured by logical dependency. Complete publication packages with Code, Data, and Figures.

| # | Paper | Focus |
|---|-------|-------|
| 1 | [The Structure Cost of Erasure](v0.2/structure_cost_of_erasure/) | Landauer + DPI → ξ emergence, ln(φ) partition |
| 2 | [The Balance Constant and Its Decomposition](v0.2/balance_constant_decomposition/) | Ξ = γ + ln(φ) from five domains |
| 3 | [Feigenbaum Constants from Fibonacci Arithmetic](v0.2/feigenbaum_fibonacci_arithmetic/) | Closed-form Feigenbaum constants (6–13 digits) |
| 4 | [Standard Model Parameters from Fibonacci Arithmetic](v0.2/standard_model_fibonacci_arithmetic/) | α to 5.7 ppm, sin²θ_W = 3/13, mass ratios |
| 5 | [Classical Physics from Information Geometry](v0.2/classical_physics_information_geometry/) | Maxwell from PAC/SEC, D=3 from MED |
| 6 | [Computational Validation](v0.2/computational_validation_pac_conservation/) | GAIA, PAC conservation in ML systems |

### [v0.1/](v0.1/) — Archive (October 2025)

Original 5-paper release. Published on Zenodo as record 17295103. Superseded by v0.2 but retained for reference and citation continuity.

| # | Paper | v0.2 Disposition |
|---|-------|------------------|
| 1 | [Xi Bounded Invariant](v0.1/xi_bounded_invariant_universal_balance_operator/) | → Paper 2 (rewrite) |
| 2 | [SEC-MED Framework](v0.1/sec_med_framework_information_amplification/) | → Paper 6 (consolidate) |
| 3 | [Möbius Confluence Operator](v0.1/mobius_confluence_operator_temporal_emergence/) | → Paper 6 (consolidate) |
| 4 | [Relativistic MAS](v0.1/relativistic_mas_universal_frequency/) | → Paper 6 (consolidate) |
| 5 | [GAIA Computational Validation](v0.1/gaia_computational_validation_dawn_field_theory/) | → Paper 6 (consolidate) |

## Reading Order

1. **Start with Paper 1** — Establishes the mechanism (erasure → structure) from undisputed physics
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
  → BALANCE: Ξ = γ + ln(φ) (Paper 2: 0.12% spread)
  → UNIVERSALITY: Feigenbaum from 55 = F₁₀ (Paper 3: 13 digits)
  → PHYSICS: sin²θ_W = 3/13, α to 5.7 ppm (Paper 4)
  → MAXWELL: Depth-2 PAC → electromagnetism (Paper 5)
  → VALIDATION: PAC conservation in ML systems (Paper 6)
```

## Voice and Standard

Each paper must:
1. Start from something established (a known law, a theorem, a measurement)
2. Derive the consequence (≤10 lines of math)
3. Present measurements with error bounds
4. Separate established from speculative — clearly, once
5. State what would falsify the claim

## Contributing

Results not yet at PACSeries publication standard are catalogued in [PRELIMINARY_RESULTS.md](PRELIMINARY_RESULTS.md), with defined next steps and contribution statuses.

## Citation

```bibtex
@misc{dawnfield2026pacseries,
    title = {PACSeries: Dawn Field Theory},
    author = {Groom, Peter},
    year = {2026},
    publisher = {Zenodo},
    note = {v0.2 in preparation}
}
```

## License

AGPL-3.0 (code), CC-BY-4.0 (papers)
