# exp_02 outcomes — the position of the read: 3/4; the residual is one tenth of Buchstab's deficit, sign included

**Registration:** `ae67f522` (`2026-09-07_exp02_registration.md`), scored to the sealed text.
**Scored run:** `results/exp_02_position_of_the_read_main_20260907_133111.json` (+ `_log.txt`), depths 6..9, seed
20260908, 86 s. The first scored run, `_132740.json`, is kept beside it: identical cells (deterministic), one verdict
differs — its scorer gave R3's CONVERGED clause precedence over the KILL clause when both fired, a precedence the seal
never granted; corrected toward the seal and rerun (§R3).
**Verdicts:** R1 CONFIRM · R2 CONFIRM · R3 INCONCLUSIVE · R4 CONFIRM. **Score 3/4.**

## Scorecard

| relation | verdict | by the letter |
|---|---|---|
| R1 the residual is a scale-free property of the read's position | **CONFIRM** | (a) ε₃ at the primes' arc: 0.0101, 0.0102, 0.0087 at 10⁷, 10⁸, 10⁹ — every one within max(3·SE, 10 %) of the mean 0.0097 (10⁹ by a hair: 0.00096 against 0.00097; a registered postdiction check on round 1's numbers). The alternative r-form also converges (0.074, 0.083, 0.078). (b) ε₃ decreases at every resolved step over u = 2 → 2.1 → 2.25 (4 resolved, 4 decreasing, 0 increasing) |
| R2 the sign flip | **CONFIRM** | four of six flip cells negative and resolved at 3–6σ (m = 7: −0.00098 ± 0.00016, −0.00079 ± 0.00017 at u = 2.75, 3; m = 8: −0.00074 ± 0.00016, −0.00050 ± 0.00015); the one positive resolved cell (m = 8, u = 2.5, +0.00062 ± 0.00016) sits where the measured deficit is still positive (d = +0.0045); zero sign mismatches between ε and d in all five cells where both resolve |
| R3 Buchstab proportionality | **INCONCLUSIVE** | c = ε₃/d ≈ 0.10 on all 14 resolved cells (0.087–0.115 on the twelve with SE ≤ 0.011); the spread 0.073 lies inside the sealed tolerance 0.108 (CONVERGED clause fires) *and* two noisy flip cells, 0.139 ± 0.036 and 0.066 ± 0.020, differ by a factor 2.11 (KILL clause fires — their difference is 1.8σ). The seal states no precedence; both firing is INCONCLUSIVE |
| R4 shell independence against the ultrametric tranche | **CONFIRM** | r_q for q ∈ {3, 9, 4, 8, 5} within the sealed 15 % of their mean in every scored decade (e.g. 10⁹: 0.0778, 0.0711, 0.0803, 0.0767, 0.0766); zero shell-ordering votes at tolerance |

Kill relevance: none fires. The recorded objects of §"what the run says" are the deliverable.

## What the run says

**The residual curve is the same at every depth.** ε₃(u) — the loop's bias minus the read's, in the sealed absolute
form — across the four depths (SE ≈ 0.0002–0.0004 except the small-y low-u cells):

| u_top | 1.75 (control) | 2 (primes) | 2.1 | 2.25 | 2.5 | 2.75 | 3 | 3.5 | 4 | 5 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10⁶ | −0.015 | +0.0095 | +0.0082 | +0.0036 | +0.0008 | −0.0011 | −0.0009 | +0.0003 | +0.0001 | +0.0003 | +0.0001 |
| 10⁷ | −0.0042 | +0.0101 | +0.0096 | +0.0046 | −0.0002 | −0.0010 | −0.0008 | +0.0003 | +0.0002 | +0.0004 | −0.0000 |
| 10⁸ | −0.0049 | +0.0102 | +0.0083 | +0.0046 | +0.0006 | −0.0007 | −0.0005 | −0.0001 | +0.0004 | +0.0003 | −0.0001 |
| 10⁹ | −0.0036 | +0.0087 | +0.0083 | +0.0040 | +0.0003 | −0.0012 | −0.0010 | +0.0000 | −0.0001 | −0.0004 | +0.0002 |

and the measured density deficit d(u) = 1 − ratio on the same arcs: +0.09, +0.08, +0.04, ~0, −0.009, −0.008, +0.001,
then zero — Buchstab's curve with its overshoot at u ≈ 2.6–3.2. **ε tracks d at one tenth, sign included.** The loop
over-predicts the primes' bias by 0.009 where it over-predicts their density by 0.09; it under-predicts the bias by
0.001 where it under-predicts the density by 0.009; both vanish together from u ≈ 3.5. The residual is the position of
the read, and it is Buchstab's residual seen through a pair statistic.

**The origin is the control.** At u = 1.75 (a read inside the sieve range, where the survivors are the primes of a
smaller x) ε is negative in every depth: the bias there exceeds the loop's, as the position reading requires it to
fail — the regime is different, and the registration said so.

**The shells modulate the residual; they do not make it.** R4 confirms at the sealed 15 %, and the numbers show a
smaller, consistently-signed structure below it: the deeper shell of each prime carries a slightly *smaller* relative
residual — r₉ − r₃ = +0.002, −0.009, −0.007 and r₈ − r₄ = −0.007, −0.003, −0.004 at 10⁷, 10⁸, 10⁹; at 10⁹ the 3 → 9
difference is 3.4σ. Andy Farmer's ultrametric tranche is real at about ten percent of the residual and is not its
cause. Recorded for round 3 (a relation on the shell depth, with the profinite completion as the object if it holds).

**The q = 10 channel replicates every sign** (positive at u ≤ 2.25, negative at u = 2.75, 3, in every depth).

## Postdiction, honestly

R1(a) was a registered postdiction check (round 1 had the numbers); its live content, the decrease over u, confirmed
at every resolved step. R2's bearing was round 1's u = 3 cell; the registered cells (W = 200) were unseen and resolved
at 3–6σ. R3's c had been estimated with the asymptotic ω; the measured-d version was unseen. R4's q = 5, 8, 9 were
unseen at m ≥ 6.

## Instrument record

1. **R3 precedence.** Two sealed clauses fired at once; the script's precedence was mine, not the seal's. Corrected
   toward the seal (both firing → INCONCLUSIVE), rerun, both files kept. Round 3's rule states precedence.
2. **Gate calibration before the seal** (both on the record): G1's tolerance 1/log N failed the u = 1.75 controls by
   exactly the prime count's next term (li against x/log x); set to 2/log N with the reason. G3 read the LO–S drift
   across the 10⁹ decade as noise (3.2×); de-trended, 1.2×, bounded on both sides.
3. The m = 9 depth ran in 18 s; the m = 8 flip cells (W = 200) in 27 s. No memory issue.

## What is not claimed

The value c ≈ 0.10 (a coordinate; R3 did not converge by the seal's letter). Any statement about u < 2 beyond "a
different regime". Anything about x > 2·10⁹. No physics; nothing here touches φ, Ξ, or any milestone.

## Forward corrections (2026-09-07, late — from the round-3 design review; the verdicts above are untouched)

1. **The q = 10 "replication channel" was an identity.** Every consecutive-prime gap is even, so the transition mod
   10 is the transition mod 5: δ₁₀ ≡ δ₅ to the last digit (exp_03 G2 states it). "The q = 10 channel replicates every
   sign" replicated nothing. Withdrawn as a check; the numbers stand.
2. **R4's reading is superseded.** The deeper-shell modulation is the uniform loop's own depth profile: δ_q on the loop
   falls with depth as (log y)^{−β_q} with a modulus-dependent β_q, and the primes' arc is the loop read at the
   density-matched depth y_eff; that reproduces r_q, the shell differences and the flip cells without a free parameter
   (exp_03 registration §0, gate G7). The tranche did not modulate the residual; the loop did. Whether anything is left
   is exp_03's question.
3. **R3's "c converged"** was the same drift: c ≈ β₃δ₃(1 + d/2) = 0.117, 0.108, 0.100 at m = 7, 8, 9 — a slow
   function of y, read as constant inside a 25 % tolerance.

## Forward note

Layer: arithmetic. The sentence for the record: **the consecutive-prime residue bias of Lemke Oliver and
Soundararajan equals the primorial loop's exact rational bias at √(2x) minus one tenth of Buchstab's density deficit,
sign included — the residual is the position of the read, and it changes sign where ω(u) crosses e^{−γ}.** Round 3:
the shell modulation as a registered relation; a precedence-clean proportionality rule; the lattice shape metric.
