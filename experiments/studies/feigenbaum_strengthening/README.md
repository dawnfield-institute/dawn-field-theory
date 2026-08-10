# Feigenbaum Strengthening

## Purpose

Two experiments that transform the Feigenbaum result (Paper 3) from "a formula that matches a number" to "a formula connected to the phenomenon and its mechanism."

## Experiments

### exp_01: Universality Across Map Families

**Hypothesis**: The Fibonacci closed-form formula for r_inf matches not just the known numerical value, but the independently computed accumulation point from multiple map families — confirming it captures the universal structure, not a coincidental number.

**Method**: For 5+ unimodal map families (logistic, quadratic, sine, Gaussian, cosine), compute superstable bifurcation points to high precision using mpmath, extract r_inf and delta from each cascade independently, and verify the formula matches all of them.

**Success criteria**: All maps converge to the same r_inf (to 10+ digits), and the formula matches all independently computed values.

### exp_02: Lanford Truncation at Fibonacci Dimensions

**Hypothesis**: The Feigenbaum RG fixed-point equation, solved by polynomial truncation at order N, shows structure at Fibonacci truncation dimensions — specifically at N = F_10 = 55.

**Method**: Solve T[g] = -alpha * g(g(-x/alpha)) by truncating g(x) = 1 + sum a_k x^{2k} to degree 2N for N = 5 to 100. At each N, solve the nonlinear system via Newton's method, recording alpha_N. Plot convergence |alpha_N - alpha_exact| and look for structure at Fibonacci values.

**Three possible outcomes** (all informative):
1. Local minimum/inflection at N=55: Fibonacci dimensionality in solution space
2. Smooth convergence through N=55: F_10 connects elsewhere (orbit combinatorics, not truncation)
3. Saturation before N=55: F_10 is about orbit structure at 2^10=1024, not polynomial degree

## Dependencies

- mpmath (arbitrary precision arithmetic)
- numpy, scipy (optimization, linear algebra)
- Existing formulas from sec_threshold_detection/exp_07

## Related Experiments

- exp_07 (sec_threshold_detection): Original formula validation
- exp_29 (milestone3): Extended parameter search
- exp_06 (sec_threshold_detection): Sensitivity analysis
