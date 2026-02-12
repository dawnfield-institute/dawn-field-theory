# PAC Foundations Validation (Archived)

**Created**: February 2026  
**Status**: Archived — exploratory probes completed

---

## What Happened

This folder tested speculative ideas from a vision document. Most were NOT validated.

## Outcome

| Hypothesis | Result | Action |
|------------|--------|--------|
| H1 (Möbius-Fibonacci) | ✅ Validated | **Moved to landauer_erasure_structure/exp_17** |
| H2 (Θ recycling) | ⚠️ Model-dependent | Archived, not added to papers |
| H3 (Three-phase structure) | ❌ Not supported | Discarded |
| H4 (γ necessity) | ⚠️ Weakened | Archived, papers unchanged |
| H5-H10 (Emergence tests) | ℹ️ Informative | Ξ is constructed, not emergent |

## Papers Unchanged

The PACSeries preprints were not affected. This was exploratory research.

See [SYNTHESIS.md](SYNTHESIS.md) for detailed findings.

## Hypotheses Under Test

| ID | Hypothesis | Test | Pass Criterion |
|----|-----------|------|----------------|
| H1 | Möbius → Fibonacci algebraic | exp_01 | Matrix identity exact to machine precision |
| H2 | Θ recycling energy sufficient | exp_02 | Θ_k ≥ kT ln 2 at all levels k ≤ 10 |
| H3 | Three-phase sieve invariance | exp_03 | Same phase boundaries in Sundaram as Eratosthenes |
| H4 | γ as discrete-continuous cost | exp_04 | γ emerges from H_n - ln(n), not from other decompositions |

## Results Summary

| ID | Status | Result | Date |
|----|--------|--------|------|
| H1 | ✅ | Validated — Möbius-Fibonacci exact | 2026-02-12 |
| H2 | ⚠️ | Model-dependent — don't add to papers | 2026-02-12 |
| H3 | ❌ | Not supported — don't add to papers | 2026-02-12 |
| H4 | ⚠️ | Weakened — keep existing paper language | 2026-02-12 |

**Conclusion**: Only H1 validated. Vision document ideas are too speculative for papers.

## Experiment Details

### exp_01: Möbius-Fibonacci Identity

**Claim**: The iterated Möbius transformation M_n has matrix representation [[F_{n+1}, F_n], [F_n, F_{n-1}]].

**Test**: 
1. Define M(z) = (φz + 1)/(z + φ⁻¹) as the golden-ratio Möbius transformation
2. Compute M^n for n = 1..20
3. Compare matrix entries to Fibonacci numbers
4. Check identity: 89 - 55φ = 1/φ¹⁰

**Pass**: All matrix entries match Fibonacci exactly (< 10⁻¹⁴ error).

### exp_02: Θ Energy Budget

**Claim**: Cascade is self-sustaining because Θ_k (thermal output at level k) is sufficient to fuel erasure at level k+1.

**Test**:
1. Run Landauer cascade simulation (from landauer_erasure_structure)
2. Track Θ_k at each level
3. Compare to Landauer minimum (kT ln 2)
4. Check if Θ_k ≥ kT ln 2 at all levels

**Pass**: Θ_k ≥ kT ln 2 for k = 1..10 (energy budget closes).
**Fail**: Θ_k < kT ln 2 at some k (cascade requires external input).

### exp_03: Sieve Invariance

**Claim**: The three-phase structure (MED pruning → SEC collapse → PNT smoothing) is sieve-invariant.

**Test**:
1. Run Eratosthenes sieve, identify phase boundaries
2. Run Sundaram sieve on same range
3. Compare phase transition points

**Pass**: Phase boundaries match within 5%.
**Fail**: Different structure → phases are Eratosthenes-specific.

### exp_04: γ Emergence

**Claim**: γ specifically represents discrete-continuous mismatch, not other costs.

**Test**:
1. Compute γ from H_n - ln(n) definition
2. Check if γ appears in Mertens product (established)
3. Test alternative decompositions of Ξ (not involving γ)
4. Search for Ξ = f(x, y) where x, y ≠ γ with comparable precision

**Pass**: γ is necessary (no alternative decomposition works as well).
**Partial**: γ works but alternatives exist (coincidence possible).

## Dependencies

- numpy
- scipy
- mpmath (for high-precision Fibonacci/φ calculations)
