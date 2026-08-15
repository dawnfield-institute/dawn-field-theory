# 2026-02-18: Null Space Predictions & Physics-Derived Matrix

## Summary
exp_16 and exp_17 close the two major gaps from the stoichiometric framework: prediction generation and the hand-built matrix problem. exp_16 fails all 4 tests (0/4) — Fibonacci null space predictions are not statistically distinguishable from random in prediction mode. exp_17 resolves exp_14's selectivity failure by deriving the stoichiometric matrix from physics constraints, achieving 1.23× selectivity (up from 0.86×).

## Timeline

### 12:00 - Experiment: exp_16 Null Space Predictions
Objective: Make the stoichiometric framework predictive — mine novel formulas from the Fibonacci null space and test whether they preferentially match known physics.

Results (0/4 PASS):
- Mining: z=0.46 vs random (not significant)
- Ratio scan: MC enrichment 0.93, p=0.79
- SEC cost: MC p=0.92
- All fail to distinguish Fibonacci from random in prediction mode

373 novel candidates found, 13 non-trivial ratio matches. Best prediction: m_τ/m_μ via F₁₁/(F₅·Ξ) at 0.002% error. But enrichment over random is not statistically significant.

**Key insight**: The null space is too large — 6 degrees of freedom in a 7×12 matrix means almost any target can be reached. The framework describes (post hoc) but doesn't predict (a priori).

### 12:05 - Experiment: exp_17 Physics-Derived Matrix
Objective: Replace the hand-built stoichiometric matrix (exp_13/14's weakness) with one derived from physics constraints.

Results (3/4 PASS):
- Physics matrix: avg formula alignment 0.676 → PASS
- Selectivity: improvement from 0.86× to 1.23× → PASS (key resolution!)
- Tightness: only 3 matches vs 5 for exp_13 → FAIL (matrix is more selective but matches fewer)
- Consensus: 160 strong consensus candidates across all 3 matrices → PASS

**Top consensus prediction**: indices [2,3,10] with avg alignment 0.91 across all three matrices (hand-built, random baseline, physics-derived).

### 12:50 - Analysis
The exp_16/17 pair reveals the framework's honest state:
- ✅ Physics can be DESCRIBED by stoichiometric Fibonacci (exp_13: 99.98th percentile)
- ✅ Selectivity is real with physics-derived constraints (exp_17: 1.23×)
- ❌ Prediction is NOT significant (exp_16: 0/4, null space too large)
- 160 consensus candidates are interesting but not validated

Gap status update:
- Prediction gap: NOT closed (exp_16 fails)
- Hand-built matrix gap: ✅ closed (exp_17 succeeds)
- "Why Fibonacci" gap: 🔄 strengthened by consensus across matrices

## Key Findings
- Fibonacci stoichiometric framework describes but does not predict
- Physics-derived matrix inverts selectivity from 0.86× to 1.23×
- 160 consensus novel predictions identified but not yet validated
- The null space has too many degrees of freedom for a priori prediction

## Next Steps
- [ ] Validate top consensus predictions against known physics
- [ ] Consider whether exp_16's failure should be added to FALSIFICATION_REGISTRY
- [ ] Investigate whether tighter constraints (e.g., conservation laws) reduce null space dimensionality
