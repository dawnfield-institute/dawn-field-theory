# Falsification Experiments and Corrections Registry Update

**Date**: 2026-01-06 12:40
**Type**: research | documentation

## Summary

Ran new falsification experiments (exp_12, exp_13, exp_33) that substantially revise our understanding of φ/Ξ claims. The alternative embedding test (exp_13) revealed that Ξ ≈ 1.057 is **metric-dependent**, not universal. However, SEC prime enrichment (exp_33) proved **genuinely robust** and the 1/φ threshold emerges naturally without fitting.

Updated phi_artifact_test folder with revised conclusions and PAC framework interpretation.

## Changes

### Added
- `exp_13_alternative_embeddings.py` - Tests CA clustering across 5 different embeddings
- `exp_33_sec_robustness_no_phi.py` - Tests SEC prime enrichment without φ-targeting
- `phi_artifact_test/REVISED_CONCLUSIONS.md` - Final analysis superseding BLIND_ANALYSIS

### Changed
- `EPISTEMIC_CORRECTIONS_REGISTRY.md` - Updated with experimental results and PAC interpretation
- `phi_artifact_test/README.md` - Complete rewrite with resolved findings

## Key Findings

### Ξ is Metric-Dependent (exp_13)

| Embedding Method | Class IV Mean P/A |
|------------------|-------------------|
| Original | 12.26 |
| Pure entropy | 3.14 |
| Compression | 0.998 |
| Temporal | 11.46 |
| Frequency | 1.62 (≈ φ!) |

**Verdict**: The specific value ~1.057 depends on embedding choice. Class IV separates from other classes in all methods, but NOT at a universal value.

### SEC Prime Enrichment is Genuine (exp_33)

- Mean enrichment: 3.71× (range 2.31-5.79×)
- All 8 configurations show enrichment > 1.5×
- Mean frac(E>0) = 0.613 ≈ 1/φ = 0.618 (Δ = 0.0048)
- **Unexpected**: The 1/φ threshold emerges naturally WITHOUT fitting!

### Random Baseline (exp_12)

- 0/1000 random systems within 5% of Ξ
- 4/6 Class IV rules within 5% of Ξ
- Fisher's p < 0.0001
- **But**: Uses same embedding that defined Ξ

## Epistemic Status Changes

| Claim | Previous | New |
|-------|----------|-----|
| CA clusters at Ξ | GENUINE | METRIC-DEPENDENT |
| SEC prime enrichment | GENUINE | ✅ CONFIRMED (3.71×) |
| SEC 1/φ threshold | ARTIFACT | ✅ GENUINE (emerges naturally) |
| Ξ = 1 + π/55 formula | CURVE-FIT | CURVE-FIT |
| φ appears in CA | UNTESTED | Frequency embedding → 1.62 ≈ φ |

## Philosophical Note

This session exemplifies the Imperfection Engine philosophy. We discovered:
1. A claimed "genuine" finding (Ξ) is actually metric-dependent
2. A claimed "artifact" (1/φ threshold) is actually genuine

The original exp_03 φ-fitting was **unnecessary** - the value emerges anyway. This is a beautiful case where curve-fitting obscured a genuine result.

## Related
- `EPISTEMIC_CORRECTIONS_REGISTRY.md` (entry #2)
- Previous: `20260106_093000_comprehensive_jwst_validation.md`
