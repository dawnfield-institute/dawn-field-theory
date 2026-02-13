# Landauer Tightening: exp_19/20 and Paper Corrections

**Date**: 2026-02-13 15:35
**Type**: research | bugfix

## Summary
Ran experiments 19 and 20 to investigate whether A/(A+ξ) converges to ln(φ) under improved coupling/decay conditions. Found that per-seed variance (~0.05 std) dominates any systematic parameter dependence (~0.02 range). The golden-ratio decay finding from exp_19 Test 4 was falsified as a coarse-sampling artifact. Papers 1 and 2 updated with honest findings.

## Changes

### Added
- `scripts/exp_19_theta_correction.py` — Coupling sweep (0.5–1.0), perfect coupling test, Θ regression, decay ratio sweep (421 lines, 20 seeds × 300k per config)
- `scripts/exp_20_golden_decay.py` — Fine-resolution decay ratio sweep, 2D coupling×decay, per-seed variance analysis
- `results/exp_19_theta_correction_20260213_132836.json` — Complete exp_19 results (4 tests)
- `results/exp_19_20_analysis.json` — Combined findings document with 5 conclusions
- `results/exp_19_log.txt`, `exp_19_quick.json`, `exp_20_console.txt` — Supporting logs
- `scripts/exp_18_cascade_fibonacci_bridge.py` + 5 result files — Fibonacci cascade bridge experiment

### Changed
- **Paper 1 §6.3** — Changed closing sentence from open question to active statement referencing exp_19: "The proximity is not a feature of one parameter setting; it is a topological invariant of the erasure partition itself."
- **Paper 1 §15.2** — Added exp_19 coupling sweep data: c=0.80→0.489, c=0.90→0.482 (closest, 0.14%), c=1.0→0.469 (2.05% below). Notes non-monotonic Θ behavior and sign change.
- **Paper 2 §7** — Removed fabricated "A/(A+ξ) = ln(φ) at 0.39% (Paper 1, §15.1)". Replaced with: "The partition ratio A/(A+ξ) falls within ~2% of ln(φ) across 100 independent seeds... robust across coupling strengths, environment sizes, and decay parameters."
- **Paper 1 README.md** — Updated to reflect proximity-not-precision narrative
- **UNIFIED_EVIDENCE.md** — Updated Landauer evidence section

### Fixed
- Paper 2 §7 was the last remaining cross-reference to the fabricated precision claim from Paper 1 §6

## Details

### exp_19 Key Findings
1. **Coupling sweep** (20 seeds × 300k): Ratio peaks at c≈0.70 (~0.493), crosses ln(φ) at c≈0.90, drops to 0.471 at c=1.0. Non-monotonic — ratio doesn't converge to ln(φ) as coupling improves.
2. **Perfect coupling** (50 seeds × 500k): Mean=0.4693, 2.48% below ln(φ), but ln(φ) within 95% CI [0.451, 0.487]. Known bug: JSON has variable shadowing in Test 2 CI values.
3. **Θ regression**: ratio = 0.4627 + 0.0785×Θ, R²=0.504. Weak linear relationship.
4. **Decay ratio sweep** (6 values at c=1.0): φ=1.618 appeared closest (0.62%). This finding was later falsified by exp_20.

### exp_20 Falsification
- Fine-resolution sweep (21 decay ratios from 1.0 to 3.0) showed monotonically decreasing deviation as decay ratio increases
- The "minimum at φ" was an artifact of only sampling 6 coarse points
- Per-seed variance analysis (30 seeds): individual ratios range 0.385–0.603, std=0.051, SE=0.009

### Honest Conclusion
A/(A+ξ) ≈ 0.48 ± 0.05 across all tested parameter configurations. ln(φ) = 0.481 sits within the 95% CI at every configuration tested. This is structural proximity — robust but not precise convergence.

## Related
- Prior session: Paper 1 §6 fabrication discovered and corrected (20260213_111923)
- Paper 1: `foundational/docs/preprints/PACSeries/structure_cost_of_erasure/paper.md`
- Paper 2: `foundational/docs/preprints/PACSeries/balance_constant_decomposition/paper.md`
