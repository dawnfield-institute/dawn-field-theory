# PAC Foundations Validation — Synthesis (Archived)

## Status: Archived

- **exp_01 (Möbius-Fibonacci)**: ✅ Validated → Moved to `landauer_erasure_structure/scripts/exp_17_mobius_fibonacci_derivation.py`
- **exp_02-10**: Exploratory probes, not validated, scripts removed
- **Papers**: Unchanged

---

## Context

**Important**: This experiment folder tested exploratory hypotheses from a February 2026 vision document ("The Actualization of Reality"). These are NOT claims from the published PACSeries preprints — they are speculative extensions that we tested before considering inclusion.

**Result**: Most speculative extensions were weakened or falsified. The core PACSeries papers remain unaffected.

## Cross-Connections

Each hypothesis connects to existing work:

| Hypothesis | Source | Connected Experiments |
|------------|--------|----------------------|
| H1 (Möbius→Fibonacci) | Vision doc §3.2 | oscillation_attractor_dynamics, prime_harmonic_manifold |
| H2 (Θ energy budget) | PRELIMINARY_RESULTS B5 | landauer_erasure_structure exp_10 |
| H3 (Sieve invariance) | PRELIMINARY_RESULTS B1 | asymmetric_conservation exp_14-17 |
| H4 (γ as discrete-continuous) | Vision doc §2.2 | prime_growth_dynamics, balance_constant_decomposition |

## Relationship to PACSeries

These experiments tested whether vision document ideas could extend the papers:

- **Paper 1**: H2 explored extending Θ recycling claims (result: model-dependent, needs more work)
- **Paper 2**: H4 explored γ "necessity" narrative (result: γ is convenient, not uniquely necessary)
- **Paper 3**: H1 confirmed Fibonacci-Möbius connection (this one validates)

**The papers themselves make more modest claims than the vision document proposed.**

## Key Results

*Last updated: 2026-02-12 (after deep dive)*

### Initial Validation (exp_01 — exp_04)

| Exp | Status | Key Finding |
|-----|--------|-------------|
| exp_01 | ✅ VALIDATED | Möbius-Fibonacci identity exact to 10⁻¹⁵ |
| exp_02 | ⚠️ MODEL-DEPENDENT | Θ recycling result depends on formula choice |
| exp_03 | ❌ INCONCLUSIVE | Three-phase structure doesn't match prediction |
| exp_04 | ⚠️ PARTIAL | γ + ln(φ) is best decomposition, but necessity unproven |

### Deep Dive (exp_05 — exp_07)

| Exp | Status | Key Finding |
|-----|--------|-------------|
| exp_05 | ❌ FALSIFIED | No φ/γ phase boundaries in prime elimination |
| exp_06 | ❌ WEAKENED | **1/√3 beats γ in Mertens test** (3.81% vs 3.83%) |
| exp_07 | ⚠️ UNCERTAIN | Θ formula underspecified — different formulas give 36% to 94% |

## Deep Dive Findings

### exp_05: Phase Boundary Scan
- No φ-related boundaries found within 2% tolerance
- Prime 2 closest to γ: 0.543 vs 0.577 (3.4% miss)
- **Verdict**: Three-phase model falsified at prime elimination counting level

### exp_06: γ vs 1/√3
**Critical discovery**: 1/√3 ≈ 0.5774 outperforms γ ≈ 0.5772 in Mertens theorem!
- γ and 1/√3 differ by only 0.023%
- 1/√3 is geometrically fundamental
- γ may be the closest number-theoretic constant to 1/√3
- **Verdict**: γ is convenient, not necessary

### exp_07: Θ Formula Underspecification
- exp_02 used: Θ = P - A - ξ (information budget residual) → 94.5%
- exp_07 used: Θ = ξ_k × (1 - φ^-k)/k (cascade decay) → 36%
- **Verdict**: Cannot claim efficiency without derivation from first principles

---

## Emergence Analysis (exp_08 — exp_10)

### Methodological Shift

User insight: Instead of testing "does data match our constants?", we should test "do constants emerge from optimization?" Conservation is constraint, not optimality.

### Key Results

| Exp | Question | Answer |
|-----|----------|--------|
| exp_08 | Do φ, γ, Ξ emerge from generic optimization? | **0/4** — none emerge |
| exp_09 | Do they emerge from correct mechanisms? | **φ, γ: yes; Ξ: no** |
| exp_10 | Is γ + ln(φ) uniquely forced? | **No** — 21 alternatives exist |

### The Real Picture

```
Mathematical necessities (EMERGE):
├── φ: fixed point of x = 1 + 1/x
└── γ: limit of H_n - ln(n)

Constructed combinations (DO NOT EMERGE):
└── Ξ = γ + ln(φ): cleanest combination, but not forced
```

### exp_10 Critical Finding

- **21 combinations** of natural constants fall within 5% of Ξ
- γ + ln(φ) is **rank #1** (exact match)
- BUT: PAC scale-invariance does NOT force Ξ
- Ξ is a **construction**, not an **emergence**

### Interpretation

- γ = harmonic DIVERGENCE rate
- ln(φ) = geometric CONVERGENCE rate  
- Ξ balances entry (γ) against exit (ln φ)

This explains why the combination is meaningful, but it doesn't predict it uniquely.

---

## Implications for Vision Document

**H1 VALIDATED**: Möbius-Fibonacci identity is exact — this could strengthen Paper 3.

**H2 UNCERTAIN**: Model-dependent results — don't add to papers without derivation.

**H3 NOT SUPPORTED**: Three-phase structure idea should not be added to papers.

**H4 WEAKENED**: γ "necessity" narrative too strong — keep existing paper language.

**Ξ IS CONSTRUCTED**: Don't add "emergence" language to papers.

## Recommendations

1. **H1 only**: The Möbius-Fibonacci connection could be added to Paper 3
2. **Do NOT add** H2, H3, H4 extensions to the papers — they weren't validated
3. **Keep this folder** as documentation of what was explored and why it didn't pan out

## Papers Remain Unchanged

The existing preprints make appropriately modest claims. This session tested whether bolder claims were justified — mostly they are not.
