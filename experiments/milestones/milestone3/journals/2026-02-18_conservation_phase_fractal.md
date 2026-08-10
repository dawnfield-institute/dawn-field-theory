# 2026-02-18: Conservation, Phase Transitions, and Fractal Mesh

## Summary
exp_18 through exp_20 explore three approaches to making the formula framework discriminative: conservation-required predictions (exp_18: 2/4), phase transition dynamics (exp_19: 1/4, FALSIFIED for Fibonacci-specificity), and fractal convergence mesh (exp_20: 1/4, FALSIFIED for raw pressure). Two honest falsifications and one partial success that sets up the breakthrough in exp_21.

## Timeline

### 13:17 - Experiment: exp_18 Conservation Predictions
Objective: Test whether PAC conservation requirements discriminate physics matches from non-matches.

Results (2/4 PASS):
- Exhaustion: PASS — null dim 6, full rank exhaustion = 1.0 (6 formulas span the full null space)
- Cascade: PASS — monotonic rank increase, crystallization at α_em (step 6)
- CF vs physics: FAIL — conservation fractions show no discrimination (p=0.98)
- Sequence comparison: FAIL — Fibonacci z-score = −0.71

**Key insight**: Conservation is necessary (the null space IS fully exhausted by 6 formulas) but not sufficient for discrimination. All formulas in the null space satisfy PAC conservation equally well.

### 14:28 - Experiment: exp_19 Phase Transition Dynamics
Objective: Test whether approach to the conservation boundary differs across input sequences.

Results (1/4 PASS — **FALSIFIED** for Fibonacci-specificity):
- Cascade paths: 0% different orderings across Fibonacci/Lucas/Primes/Tribonacci/Random
- All sequences produce **identical crystallization order**: sin²θ_W → Koide → She-Lev → ν_WF → α_s → α_em
- Only test_1 passes: all paths identical (basis-independence confirmed)

**Critical falsification**: The crystallization order is entirely determined by the target physics — NOT by the input sequence. Fibonacci, Lucas, Primes, Tribonacci, and random sequences all converge to the same order. This is interesting (suggesting the physics itself has a natural hierarchy) but **falsifies the claim that Fibonacci is special for phase transitions**.

### 16:20 - Experiment: exp_20 Fractal Convergence Mesh
Objective: Replace the flat stoichiometric matrix with fractal recursive decomposition.

Results (1/4 PASS — **FALSIFIED** for discrimination):
- Mesh construction: PASS — 33.6× amplification over flat, hub structure at indices [1,2,3]
- Selection: FAIL — p=0.78, no discrimination between physics matches and non-matches
- Recursion specificity: FAIL — no sequence type discriminates
- Fractal vs flat: FAIL — fractal geometry not better than flat

**Key insight**: The fractal mesh structure is real (33.6× amplification, clear hub hierarchy) but raw pressure correlates with index depth, not physics. Physics matches have LOWER average pressure than non-matches (delta = −2703, p = 0.78, WRONG direction). The core problem: visit counting conflates structural depth with physical significance.

### 17:00 - Analysis: The Arc from exp_16 to exp_20
Five experiments, five approaches to making the framework discriminative:

| Exp | Approach | Result | Why It Failed |
|-----|----------|--------|---------------|
| 16 | Null space mining | 0/4 | Null space too large |
| 17 | Physics-derived matrix | 3/4 | Selectivity ✅ but tightness ✗ |
| 18 | Conservation predictions | 2/4 | Conservation is necessary but not sufficient |
| 19 | Phase transition dynamics | 1/4 | **FALSIFIED**: crystallization is basis-independent |
| 20 | Fractal convergence mesh | 1/4 | **FALSIFIED**: raw pressure = depth bias |

The pattern: flat approaches (16, 17, 18) describe but don't discriminate. Dynamic approaches (19) show basis-independence, not Fibonacci-specificity. Geometric approach (20) shows structure but conflates depth with significance.

What's missing: conservation-normalized profiles that capture SHAPE rather than magnitude.

## Key Findings
- Conservation is necessary but not sufficient for discrimination (exp_18)
- Crystallization order is basis-independent — **falsifies Fibonacci-specificity in phase transitions** (exp_19)
- Fractal structure is real (33.6× amplification) but raw pressure is depth-biased (exp_20)
- The common failure mode: confusing structural depth with physical significance
- Needed: PAC-normalized profiles that compare shape, not magnitude

## Next Steps
- [x] Apply PAC Lazy architecture (from GAIA POCs) to normalize away depth bias → exp_21
