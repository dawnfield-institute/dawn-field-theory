# Prime Growth Dynamics v2: Multi-Stage Emergence Exploration

**Version**: 0.1.0  
**Status**: 🔄 In Progress  
**Date**: 2026-02-08  
**Origin**: Phase framework hypothesis from prime_growth_dynamics + PRE_STRUCTURAL_EMERGENCE.md

---

## Context

prime_growth_dynamics established that primes are **residual roughness** — memory traces left after iterative smoothing of the number line. The Mertens validation (0.9997 ratio), exact PAC conservation (π(x) + C(x) = x - 1), and Ξ = γ + ln(φ) three-source convergence were confirmed across 27 experiments.

A subsequent synthesis proposed a **three-phase emergence pipeline**:

| Phase | Name | Boundary Constant | Mechanism |
|-------|------|-------------------|-----------|
| I | Possibility Proliferation | γ = 0.577 | Combinatorial freedom under MED bounds |
| II | Symbolic Entropy Collapse | ln(φ) = 0.481 | SEC: ∂S/∂t = α∇I - β∇H |
| III | Recursive Smoothing | 1/ln(x) | PAC conservation + wave smoothing |

**Total reconciliation**: Ξ = γ + ln(φ) = 1.058

This experiment computationally tests whether the phase framework:
1. Holds up quantitatively (not just qualitatively)
2. Resolves specific open questions from other experiments
3. Makes falsifiable predictions

---

## Hypotheses

### H1: λ* Is Derivable from Phase Constants

The sec_prime_manifold critical point λ* = 0.9816 should be expressible in terms of γ and ln(φ).

**Candidates to test:**
- λ* = 1 - (1 - ln(φ))/F₁₀ → 0.991 (1% off)
- λ* = 1 - γ/F₁₀ → 0.990 (0.8% off)
- λ* = ln(φ)/γ × (something)
- Other combinations

**Success criterion**: A formula from {γ, ln(φ), F_n} that matches λ* = 0.9816 to < 0.5% without free parameters.

### H2: Forbidden k Valleys Are Wave Interference Gaps

In sec_prime_manifold, k values {5, 12-15} cannot reach φ. If these are Phase III resonance gaps, they should be predictable from the smoothing wave spectrum.

**Test**: Do forbidden k's correspond to positions where the first few sieve waves (p = 2, 3, 5) destructively interfere?

**Success criterion**: Predict ≥ 4/5 forbidden k values from wave interference calculation alone.

### H3: β ≈ 0.79 Has Phase-Ratio Origin

The sec_prime_manifold critical exponent β ≈ 0.79 may relate to ratios of phase constants.

**Candidates:**
- ln(φ)/γ = 0.834 (5% off)
- γ/Ξ = 0.546 (far)
- 1 - ln(φ) = 0.519 (far)
- 2ln(φ)/Ξ = 0.910 (far)
- Other phase-constant combinations

**Success criterion**: A formula matching β to < 2% without free parameters.

### H4: F₃ = 3 Emerges from MED Bounds in PAC Simulation

If MED (nodes ≤ 3) constrains Phase I, then PAC simulations with depth > 2 or nodes > 3 should be unstable, while depth ≤ 2, nodes ≤ 3 produces stable structures.

**Test**: Run PAC evolution at varying depth/node limits and measure stability.

**Success criterion**: Clear stability boundary at MED bounds. 4-node models demonstrably less stable.

### H5: The α Correction Term Decomposes as Cross-Phase Product

The unexplained correction [1 - F₁₀/(4πF₇²)] in the fine structure constant formula should decompose as Phase I × Phase III / Phase II² or similar cross-phase structure.

**Test**: Dimensional analysis of F₁₀/(4πF₇²) in phase-constant units.

**Success criterion**: The correction has a natural interpretation in the phase framework without ad hoc construction.

---

## Experimental Design

### Part I: Phase Constant Derivations (exp_01-03)

| Exp | Name | Tests |
|-----|------|-------|
| 01 | Lambda star derivation | Systematic search through γ, ln(φ), Ξ, F_n combinations for λ* = 0.9816 |
| 02 | Critical exponent beta | Same approach for β ≈ 0.79 |
| 03 | Alpha correction decomposition | Phase-constant decomposition of F₁₀/(4πF₇²) |

### Part II: Wave Interference Predictions (exp_04-06)

| Exp | Name | Tests |
|-----|------|-------|
| 04 | Forbidden k prediction | Compute sieve wave interference at each k, compare to forbidden valleys |
| 05 | φ-only-on-odd manifold | Model Phase III p=2 wave removing even-manifold φ emergence |
| 06 | Gap 6 as hub | Test whether 6 = 2×3 = F₃×F₄ produces maximum wave interference |

### Part III: MED Bound Tests (exp_07-09)

| Exp | Name | Tests |
|-----|------|-------|
| 07 | PAC stability vs node count | Evolve PAC trees with 2, 3, 4, 5 max children — measure stability |
| 08 | PAC stability vs depth | Evolve PAC trees with depth 1, 2, 3, 4 — measure stability |
| 09 | Three generations emergence | Do exactly 3 stable modes emerge under MED-constrained PAC? |

### Part IV: Cross-Experiment Validation (exp_10-12)

| Exp | Name | Tests |
|-----|------|-------|
| 10 | Wilson-Fisher 2% gap | Does adding γ-correction to 1/φ close the gap to ν = 0.630? |
| 11 | Alternation limit 2/3 vs 1/φ | Is 2/3 = F₃/F₄ the MED-constrained version of 1/φ? |
| 12 | Force depth mapping | Map EM (F₇), weak, strong, gravity depths to coupling strengths |

---

## Falsification Conditions

The phase framework is **falsified** if:

1. No combination of {γ, ln(φ), F_n} matches λ* to < 2% — the phase constants don't connect to measurables
2. Forbidden k values are NOT predictable from wave interference — the "Phase III resonance gap" explanation fails
3. PAC stability shows NO boundary at nodes = 3 — MED bounds don't constrain anything
4. The α correction term has no natural phase decomposition — the framework doesn't explain what exists

The framework is **supported** (not proven) if:

1. Multiple derivations land within 1% using only phase constants
2. Forbidden k prediction works for ≥ 4/5 valleys
3. PAC stability boundary is sharp at MED bounds
4. Cross-experiment discrepancies (Wilson-Fisher, alternation limit) are explained by phase corrections

---

## Dependencies

- `prime_growth_dynamics/` — smoothing model, Mertens validation, Ξ convergence
- `sec_prime_manifold/` — λ*, β, forbidden k, φ emergence data
- `landauer_erasure_structure/` — ln(φ) derivation, cascade topology
- `cellular_automata_pac_attractors/` — Ξ measurement, CAH conditions
- `milestone1/` + `milestone2/` — SM parameters, mass ratios, She-Leveque
- `maxwell_from_pac_sec/` — EM depth, MED → D=3
- `gravity_from_maxwell_pac/` — gravity depth 183
- `oscillation_attractor_dynamics/` — gap 6 hub, alternation limit, injection model
