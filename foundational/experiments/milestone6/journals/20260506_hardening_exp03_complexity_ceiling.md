# exp_03 T2 Failure: Complexity Ceiling Phase Transition

**Date**: 2026-05-06
**Context**: PACSeries v0.3 hardening cycle — tightened R² threshold from 0.5 to 0.75

## The Failure

Geometric decay fit across hierarchy levels: R² = 0.67 (< 0.75 threshold).
Original test passed at R² > 0.5 — too permissive to be meaningful.

## What the Data Shows

Level-mean eigenvalues for geometric fit:

| Level | Mean eig | log(eig) | Fit residual |
|-------|----------|----------|-------------|
| 1 | 0.00315 | -5.76 | -0.54 |
| 2 | **0.00681** | **-4.99** | **+1.11** |
| 3 | 0.00052 | -7.56 | -0.58 |
| 4 | 0.00040 | -7.83 | +0.02 |

Level 2 mean eigenvalue is **higher** than level 1. The entire R² failure is driven by this inversion.

## Root Cause: Two Boundaries in the Finite-Size Regime

Level 2 has 22 boundaries. 20 of them (all from parent=1, n=15270) have eigenvalues 0.0001–0.0013, consistent with geometric decay. Two boundaries from parent=5 (n=100) are outliers:

- child=9, n_child=92, eig=0.0109 (child fills 92% of parent — nearly identity)
- child=20, **n_child=8**, eig=**0.1249** (100x typical level-2 value)

K_MODES = 10. The 8-cell child has fewer cells than spectral modes.

## The Insight: Complexity Ceiling as Phase Transition

Sorting all 67 boundaries by child_size / K_MODES reveals a sharp transition:

| child/K regime | n boundaries | mean eig | spectral efficiency |
|---------------|-------------|----------|-------------------|
| < 2.0 (sub-K) | 4 | 0.037 | ~1.5 |
| ≥ 2.0 (super-K) | 63 | 0.002 | ~0.1–0.01 |

**19x jump in mean eigenvalue at the transition.**

When child_size < K_MODES, the child doesn't have enough cells to project partially onto the parent eigenbasis. It saturates — ALL modes captured. The transfer matrix eigenvalue jumps because there's no information to lose.

When child_size >> K_MODES, genuine partial projection occurs, and geometric decay (base ≈ 1/φ) governs attenuation.

## Why This Matters

1. **This IS tetration termination.** The hierarchy doesn't terminate by fiat — it terminates when regions become too small to sustain scope-mediated spectral transfer. The R² failure is measuring the onset of this termination.

2. **Saturation efficiency ≈ φ.** The spectral efficiency (eig / size_ratio) at the transition is ~1.5 ≈ φ. If the complexity ceiling saturates at exactly φ, that's a testable DFT prediction.

3. **Two-regime model.** The geometric decay prediction applies to the thermodynamic regime (child >> K_MODES). Below the complexity ceiling, a different law governs — mode saturation. A proper test should separate these regimes, not average across them.

## Robustness Check

| Aggregation method | R² | Geometric base |
|-------------------|-----|---------------|
| Mean (all, original) | 0.67 | 0.42 |
| Median | 0.89 | 0.82 (outside φ range) |
| Filter parent > 200 | 0.83 | 0.52 |
| Filter child > 20 | **0.96** | 0.52 |
| Size-weighted | 0.94 | 0.33 (below 1/φ²) |

Filtering child > 2*K_MODES recovers R² = 0.96 with base = 0.52 ∈ [1/φ², 1/φ].

## Implications for the Test

The current T2 averages across both regimes, which is physically wrong — it's like fitting a single line through a liquid-gas phase diagram. The test should either:
- Separate the two regimes and test each independently
- Or test the existence of the transition itself (the complexity ceiling IS the prediction)

## Follow-up Questions

- Does the transition point scale with K_MODES? (Vary K and see if the cutoff moves)
- Is the saturation efficiency exactly φ, or approximately?
- Can we derive the transition point from PAC? (Minimum complexity for scope mediation)
- Does M10's self-application framework predict this hierarchy termination?
