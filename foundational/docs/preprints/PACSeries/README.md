# PACSeries: Dawn Field Theory

**Version**: 2.0  
**Original Release**: October 6, 2025 (v1.0, 5 papers)  
**Current**: February 2026 (v2.0, 6 papers — consolidation in progress)

## Overview

The PACSeries establishes Dawn Field Theory through clean derivation from established science, measurement with error bounds, and honest separation of established results from speculation. Papers are numbered by logical dependency, not historical order.

## Papers (v2.0)

| # | Paper | Status | Focus |
|---|-------|--------|-------|
| 1 | [The Structure Cost of Erasure](structure_cost_of_erasure/) | **Draft** | Landauer + DPI → ξ emergence, ln(φ) partition |
| 2 | The Balance Constant and Its Decomposition | Planned | Ξ = γ + ln(φ) from four domains |
| 3 | Feigenbaum Constants from Fibonacci Arithmetic | Planned | Closed-form Feigenbaum constants (6–13 digits) |
| 4 | Standard Model Parameters from Fibonacci Arithmetic | Planned | α to 5.7 ppm, sin²θ_W = 3/13, mass ratios |
| 5 | Classical Physics from Information Geometry | Planned | Maxwell from PAC/SEC, D=3 from MED |
| 6 | Computational Validation | Planned | GAIA, PAC conservation in ML systems |

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

## v1.0 Papers (October 2025 — Zenodo 17295103)

The original 5-paper release is retained in this directory for reference:

| v1# | Paper | v2 Disposition |
|-----|-------|---------------|
| 1 | [Xi Bounded Invariant](xi_bounded_invariant_universal_balance_operator/) | → Paper 2 (rewrite) |
| 2 | [SEC-MED Framework](sec_med_framework_information_amplification/) | → Paper 6 (consolidate) |
| 3 | [Möbius Confluence Operator](mobius_confluence_operator_temporal_emergence/) | → Paper 6 (consolidate) |
| 4 | [Relativistic MAS](relativistic_mas_universal_frequency/) | → Paper 6 (consolidate) |
| 5 | [GAIA Computational Validation](gaia_computational_validation_dawn_field_theory/) | → Paper 6 (consolidate) |

## Contributing

Results not yet at PACSeries publication standard are catalogued in [PRELIMINARY_RESULTS.md](PRELIMINARY_RESULTS.md), with defined next steps and contribution statuses. See that document for how to pick up open validation tasks.

## Citation

```bibtex
@misc{dawnfield2026pacseries,
    title = {PACSeries: Dawn Field Theory},
    author = {Groom, Peter},
    year = {2026},
    publisher = {Zenodo},
    note = {v2.0 in preparation}
}
```

## License

AGPL-3.0 (code), CC-BY-4.0 (papers)
