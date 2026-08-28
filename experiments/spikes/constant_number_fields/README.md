# Which number field does each derived constant live in? (spike)

**Exploratory. `spikes/` is exempt from the experiment standard (STANDARDS §3).** No scoring.

## How this got here — a claim that died and came back stronger

The starting claim was a **class signature**: gauge couplings carry exactly one power of φ,
ratios carry none, corrections carry π. It classified nine constants correctly, including
`λ_Higgs = φ/(4π)` which was **held out** and predicted correctly (one φ, one closure factor).

Then it died, to a one-line attack:

```
F3/(F4*phi*F10)  ==  (-1 + sqrt(5))/165        same number, ZERO phi's written
```

**The φ-count is a property of how someone wrote the formula, not of the number.** Any
"φ-power" classification is therefore an artifact of authorship and can be edited away.

## What survives — field membership

| class | field | constants |
|---|---|---|
| **counting** | **ℚ** — exactly rational | sin²θ_W = 3/13 · Koide = 2/3 · μ/e = **2688/13** · p/e = **1836** · Casimir = 240 |
| **growth** | **ℚ(√5)** | α_s = 3(√5−1)/32 |
| **closure** | transcendental (π) | α_EM · λ_Higgs · Ξ = 1 + π/55 |

**Every mixing angle and mass ratio is exactly rational. Every coupling is irrational.** You
cannot rewrite 3/13 into an irrational, or α_s into a rational — this is a fact about the
numbers, invariant under any change of notation.

This is the same three roles as the original claim (counting / growth / closure) on a footing
that cannot be edited. And it is **more** falsifiable, not less: one irrational DFT mass ratio,
or one exactly-rational coupling, kills it. The φ-count version could not die.

## Two problems it generates

**1. If "counting" quantities are exactly rational, why aren't they exact?** Deviations track
how much non-counting physics sits between the rational and the measurement:

| deviation | quantity | what is in between |
|---|---|---|
| 4.6 ppm | μ/e | lepton pole masses, pure QED |
| 8.5 ppm | Koide Q | charged-lepton pole masses, pure QED |
| 83.1 ppm | p/e | proton is a **QCD bound state** |
| 1906 ppm | sin²θ_W (MS-bar) | **runs**, scheme-dependent |
| 33033 ppm | sin²θ_W (on-shell) | *same quantity, other scheme* |

Monotonic, and the ordering is standard-physics obvious rather than fitted. Note that
**sin²θ_W was never a fair test** — the same rational is 0.19% off in MS-bar and 3.30%
on-shell, a 17× swing from scheme alone. A fixed rational cannot be tested against a running
observable.

**Caveat that matters:** "radiative corrections explain the residuals" is what standard physics
would say about *any* approximately-right formula, so it resolves the tension without providing
support. It becomes evidence only if the framework can **predict a residual's size** rather
than attribute it. An attempt to do so failed — see below.

**2. π enters three different ways** and the "closure" bucket hides it: α_EM is ℚ(√5) at
leading order *with* a π correction; λ_Higgs is ℚ(√5)/π with π at leading order; Ξ is ℚ + π.
Three structures, one label.

## Failed attempt, recorded

Tried to predict the residual magnitudes. **It does not work**, twice over:

- **Search size.** 12 candidate scales spanning four orders; a hit within 2× is near-certain by
  luck. `(α/π)² = 5.4 ppm` lands in the cluster and means nothing.
- **Selection.** The three "clean" residuals (α_EM 5.7, μ/e 4.6, Koide 8.5 ppm) cluster within
  1.86× — but these are *showcase* results, and ppm accuracy is roughly the threshold for being
  quoted at all. The cluster is plausibly the citation threshold, not a physical scale.

Also ruled out: the F_a/(nπF_b²) correction template is the **wrong scale** for this. It
produces percent-level corrections (α_EM's is −2.6%); ppm residuals are four orders below, and
no clean template instance reaches them.

## Files

`check_fields.py` — exact-arithmetic (sympy) field classification, and the rewrite that kills
the φ-count.
