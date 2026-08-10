# Journal: exp_35 Cosmological Constant from PAC

**Date**: 2026-03-14
**Status**: complete (4/7 PASS)

---

## Origin

After exp_34 sharpened G to 0.18% (downgrading L1 from HIGH to LOW), L5 became the last remaining HIGH-severity limitation: the cosmological constant problem (10^123 ratio between QFT prediction and observation). Peter said "lets dive into the other high."

## Structure

Seven parts attacking the CC problem from different PAC angles:

- **Part A (PASS)**: Problem statement in PAC language. QFT: ρ_Planck = 4.63e113 J/m³, observed ρ_Λ = 5.25e-10 J/m³. Ratio = 10^122.9. PAC translation: vacuum = top node, QFT counts ALL modes, PAC constrains to PAC-ALLOWED modes via MED. Omega_Lambda: observed 0.685, PAC predicts 1/φ = 0.618 (6.7pp gap), 1-1/π = 0.682 (0.33pp gap).

- **Part B (FAIL)**: Cascade cancellation. Residual per level = ln²(2) from exp_28. With N = 2×183 (gravity round-trip), get 10^-116.5 suppression vs needed 10^-122.9. **6.4 orders off.** Tantalizing — round-trip gravity depth appears naturally — but not precise.

- **Part C (PASS)**: Fibonacci suppression. Need F_N ~ 10^123, so N ~ 590. Best cyclotomic: 24²+24+1 = 601, giving F_601 ~ 10^125.25 (2.3 orders off). No clean Fibonacci depth reproduces 10^-123 exactly.

- **Part D (FAIL)**: Phase cycling cancellation. Period-4 eigenvalues {1,i,-1,-i} sum to 0 (exact cancellation). With temporal damping ln(2) per Landauer bit, get |residual| = 0.12 per cycle. After 183 phase-cycling depths: 10^-175 (52 orders overshoot). Numbers don't match.

- **Part E (FAIL)**: MED mode counting. QFT counts (L/L_P)³ ~ 10^185 modes. PAC constrains: modes need PAC parents. xi_floor^295 ~ 10^-84 (39 orders off). **DEEPEST CONCEPTUAL INSIGHT**: PAC reframes CC from "why is Λ small?" to "why would Λ be large?" Modes without PAC parents have zero weight.

- **Part F (PASS)**: **BEST QUANTITATIVE RESULT** — Omega_Lambda correction template. Extending the EM/gravity correction duality:
  - Multiplicative: Ω_Λ = (1/φ) × (1 + F₉/(4πF₅²)) = **0.684921** (0.012% error!)
  - Also good: (1/φ) × (1 + F₁₃/(4πF₇²)) = 0.685841 (0.12%)
  - Additive: 1/φ + F₈/(4πF₅²) = 0.684879 (0.018%)
  - Many Fibonacci-gap matches found, but lack the structural inevitability of G formula

- **Part G (PASS)**: Honest assessment. L5 remains HIGH. CC problem is NOT solved. But:
  1. PAC reframes the question (mode existence vs suppression)
  2. Cascade cancellation with N~2×183 gets within 6 orders
  3. Omega_Lambda correction template achieves sub-percent matches

## Key Finding: Omega_Lambda Correction

The correction template `1 ± F_a/(nπF_b²)` extends to dark energy:

| Force | Formula | Value | Error |
|-------|---------|-------|-------|
| EM (α) | 1 − F₁₀/(4πF₇²) | 0.9741 | 5.7 ppm |
| Gravity (G) | 1 + F₁₃/(πF₆²) | 2.1588 | 0.18% |
| Dark energy (Ω_Λ) | (1/φ)(1 + F₉/(4πF₅²)) | 0.6849 | 0.012% |

The dark energy formula uses n=4 like EM. Index gap a-b = 4 = F₃ + 1 (not Fibonacci — this is weaker than EM/gravity where gaps ARE Fibonacci).

## What We Can Claim

- PAC reframes CC: starting point is Λ=0 (no modes without PAC parents), not Λ=ρ_Planck
- Cascade cancellation naturally gives ~10^-117, close to but not exactly 10^-123
- Omega_Lambda can be expressed in the Fibonacci correction template at 0.012%
- BUT: the Ω_Λ formula lacks structural inevitability — too many good matches, no unique winner

## What We Cannot Claim

- No first-principles derivation of Λ
- The 10^120 problem is reframed but not solved
- Ω_Λ formulas are numerically good but physically undermotivated compared to G formula

## Limitation Update

| ID | Previous | Updated | Change |
|----|----------|---------|--------|
| L5 | HIGH: CC unsolved (10^123) | HIGH: CC unsolved but reframed | Conceptual progress only |

## Open Questions

- Can MED mode counting be made rigorous? How many vacuum modes have PAC parents in the observable universe?
- What sets the cascade cancellation residual per level EXACTLY? (Currently using ln²(2) from exp_28)
- Why does N~2×183 appear for CC suppression depth? Is the round-trip factor physical here?
- The 6.7pp gap between 1/φ and Ω_Λ: derivable from cascade dynamics or fundamentally a correction?
- Connection to Swampland conjectures (Λ > 0 constraints in string theory)?
