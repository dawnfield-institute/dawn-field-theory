# Note — F below x ≈ 0.085 is extrapolation, not measurement, and always will be

**Date:** 2026-09-08 · **Status:** instrument note, EXPLORING · **Applies to:** exp_05, exp_06, exp_07, exp_08.
**Not a verdict on any sealed round.** No score changes. Filed because it is a limit that appears in no
registration's threats section and should have.

## The limit

F's argument is x = φ(q)/ḡ, with ḡ = e^γ log y. The smallest reachable x needs the smallest totient and the
largest mean gap. φ_min = 2 (q = 3 or 4), and **ḡ grows only like log y**, so:

```
target x    gbar needed    y needed
   0.085         23.4         510,299    <- our deepest read (exp_07's y_eff at the 10^10 decade)
   0.050         40.0         5.7e9
   0.020        100.0         2.4e24
   0.010        200.0         5.9e48
```

**Halving x squares y.** This is not a compute limit that a better machine or a smarter sieve fixes; it is the
shape of the object. The region x → 0 — where F's asymptotics are pinned by theory and where any *derivation*
of F would make its sharpest prediction — is permanently out of measurement range.

## What that means for the sealed rounds

- **exp_05** fitted the 18-bin F over a domain starting at x = 0.2492. Cells below that did not exist in its
  training set, and its domain honestly reflected that.
- **exp_06** consequently **dropped q = 3, 4, 5, 8, 12** at every decade for falling outside the bins' domain
  (G6). That exclusion was correct behaviour, not a defect.
- **exp_07 and exp_08 used tanh, which has no domain**, so those same small-x moduli *were* scored — in a
  region where F's form has never been measured. The closed form extrapolates there smoothly and silently.
  exp_07's §0(a) criticised the bins for "having a domain at all"; that criticism was too glib. The bins were
  declaring an ignorance the closed form conceals.

**No verdict changes.** exp_07 R1 (rms 0.0402 against 0.0457) and exp_08 R1 (0.0352 against 0.0533) both pass
comfortably, and the small-x cells are a minority of each scored set. But the passes rest partly on
extrapolated predictions, and that was never registered as a threat.

## What was tried and failed today

The derivation route via pair correlations: conditioning on u being a unit, `A_m = ∏_{p ≤ y, p ∤ m} (p−2)/(p−1)`
is **exact** (it is `pair_count_formula` in the instrument), and `A_m / D → 2C₂ · S(m)` with C₂ the twin-prime
constant.

`2C₂ = 1.32032` sits 1.6 % from the fitted tanh parameter `a = 1.2998`, which is tempting and **wrong**. The
quantity a derivation predicts is F's slope **at the origin**, not a global fit parameter. Measured, δ/x by
band runs 1.02, 1.12, 1.26, 1.20, 1.12, 0.94, 0.75, 0.48 — **non-monotone**, with six cells in the lowest band.
The origin slope is not determined by this data, so 2C₂ cannot be tested against it. The near-match was between
the wrong two numbers, and taking it would have been STANDARDS §2.9 exactly.

The obstruction remains where it was: A_m gives *pairs*, and *consecutive* requires "no unit strictly between",
which is inclusion–exclusion over the intervening positions.

## Forward

Any future round that scores small-x moduli under a domain-free F should record how many of its cells sit
below x ≈ 0.085 and report its verdict with and without them.
