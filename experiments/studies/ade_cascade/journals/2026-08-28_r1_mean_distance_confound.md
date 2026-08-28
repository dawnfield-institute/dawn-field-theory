# R1's ordering is predicted by mean graph distance — a note, not a retraction

**2026-08-28.** Raised while applying STANDARDS §2.8 (recursive / tautological / circular) to
existing CONFIRMs. This does **not** challenge R1's measurement, which stands.

## The observation

R1 concluded: *"at equal rank, with identical means vectors, the Dynkin diagram alone moves the
spectral exponent."* But at equal rank **A_n is the path** — the longest-diameter tree on n
vertices — while D_n and E_n are branched. So "which Dynkin diagram" and "mean graph distance"
are **perfectly confounded by construction**, and there is no non-Dynkin tree anywhere in the
experiment.

| diagram | mean distance | R1 exponent |
|---|---|---|
| A_6 / D_6 / E_6 | 2.333 / 2.133 / 2.067 | −1.9234 / −1.9988 / −2.0170 |
| A_7 / D_7 / E_7 | 2.667 / 2.476 / 2.381 | −1.7513 / −1.8129 / −1.8376 |
| A_8 / D_8 / E_8 | 3.000 / 2.821 / 2.714 | −1.6130 / −1.6624 / −1.6880 |

Mean distance orders the exponents at **3/3 ranks**, A > D > E in both.

- partial corr(mean distance, exponent | rank) = **+0.966**
- exponent ~ rank alone R² = 0.928 → rank + mean distance **R² = 0.995**

## R2 already supplies the mechanism

From the same outcomes journal, explaining the affine result: *"closing the cycle halves graph
distances… so every off-diagonal coupling strengthens, organized fraction rises, and the
spectrum steepens."* Shorter distance → stronger coupling → steeper exponent. R1's E<D<A is
exactly what that predicts. **The two outcomes in one journal explain each other, and the
explanation is not Dynkin-specific.**

## What this changes

Nothing about the measurement: the CIs are real and the diagrams do produce distinct
exponents. What is **not established** is the interpretation that the *Dynkin structure* is
what moves the exponent, as against its distance profile — which any tree could share.

**The decisive control** (§2.8: vary the input while holding the suspected driver fixed):
**non-Dynkin trees at matched mean distance.** If a random tree with E_8's distance profile
reproduces E_8's exponent, "Dynkin" is doing no work. The harness already accepts an injected
coupling matrix, so this is a small addition rather than a new experiment.

## Consequence for the registered round 2

R1's `[D]` observation — that the legacy 3.3% miss of −5/3 is "an A-family artifact", with D_8
landing 0.26% from Kolmogorov — becomes a statement about D_8 having the right **distance**,
not about D being the right **diagram**. The registered round-2 question, *"does the physical
exponent select a diagram?"*, would be better posed as **"does it select a distance scale?"**
Worth resolving *before* round 2 is registered rather than after.

Caveat: nine points and a two-parameter model, so R² = 0.995 is not remarkable on its own. The
load-bearing facts are the 3/3 ordering match and the +0.966 partial correlation, neither of
which depends on the model.
