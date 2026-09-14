# exp_01 outcomes — the loop, the gap and the delta: 0/3 by the seal, and the residue bias is the loop's

**Registration:** `9213386c` (`2026-09-07_exp01_registration.md`), scored to the sealed text.
**Runs:** `results/exp_01_cascade_truncation_main_20260907_115156.json` (+ `_log.txt`) — decades 4..9, 212 s, seed
20260907, W = 25 × L = 2·10⁷. An earlier main run of the same day is kept beside it: it died at m = 8, u = 2 on a
display line (`str(P)` on a 6,000-digit primorial exceeds Python's 4,300-digit limit; fixed to a bit-length estimate,
no registered quantity touched) and its checkpoint through m = 7 agrees with the scored run cell for cell (same seed).
**Verdicts:** R1 INCONCLUSIVE · R2 INCONCLUSIVE · R3 INCONCLUSIVE. **Score 0/3.** Nothing was relaxed; the reasons are
instrument facts, stated below, and one of them is a finding.

## Scorecard

| relation | verdict | why, by the letter |
|---|---|---|
| R1 shape delta obeys the drift law | **INCONCLUSIVE** | at u ≥ 3 every Δ is ≤ 0.001 and below its floor in every decade (local shape = loop shape within noise — converged, which the rule cannot confirm); at u = 2.5 the differences are non-monotone; at u = 2.25 the floors are 0.09–0.43 — the metric is defective for lattice gaps (§R1) |
| R2 the residue bias is a truncation phenomenon | **INCONCLUSIVE** | (a) drift law **CONFIRM at all six live u**; (b) monotone decay at u = 2 **CONFIRM**; (c) meets the loop at u = 4.5 to 1e-4 in every decade but the half-split floor at m = 8, 9 is below 1e-4, so two of four decades fail by the letter |
| R3 termination depth scale-free | **INCONCLUSIVE** | ρ = 0.929, 0.954, 0.953, 0.959 (m = 6..9); differences 0.025, 0.001, 0.007 against floors 0.020, 0.011, 0.011 — converged within floors from m = 7 |

Kill relevance: none fires. No relation was killed; the three theorems-as-gates stand; the instrument lessons are the
deliverable of R1 and R3, and the numbers of R2 are the deliverable of the study.

## R2 — the residue bias is the loop's bias at the truncation depth, minus a scale-free residual (recorded)

The registered clauses (a) and (b) confirm. Beyond them, the run recorded something the seal did not score:

**δ depends on the depth y alone.** The consecutive-survivor diagonal deficit mod 3 in the window equals the loop's
deficit at the same y, whatever the decade, for every depth with u ≥ 2.5:

| y | window (decade, u) | δ local | δ loop | note |
|---|---|---|---|---|
| 22 | m=6 u=4.5 · m=8 u=6 · m=4 u=3 | 0.3075 · 0.3076 · 0.3084 | 0.3076 | exact 211594578367/687969884160 |
| 25 | m=7 u=5 | 0.2945 | 0.2945 | exact 32687539686737/110992602378240 |
| 32 | m=6 u=4 · m=9 u=6 | 0.2766 · 0.2761 | 0.2761 | |
| 60 | m=8 u=4.5 | 0.2447 | 0.2447 | |
| 100 | m=5 u=2.5 · m=6 u=3 · m=7 u=3.5 · m=8 u=4 · m=9 u=4.5 | 0.2217 · 0.2243 · 0.2225 · 0.2232 · 0.2231 | 0.2231 | five decades, one number |
| 464 | m=6 u=2.25 · m=8 u=3 | 0.1734 · 0.1767 | 0.1760 | |
| 1000 | m=9 u=3 | 0.1611 | 0.1604 | |
| 3981 | m=9 u=2.5 | 0.1390 | 0.1384 | |

**At the primes (u = 2, y = ⌈√(2x)⌉) the loop over-predicts by a constant.** Local vs loop at the same y:

| m | y | primes' δ | loop's δ at y | loop − primes | floor |
|---|---|---|---|---|---|
| 5 | 448 | 0.1760 | 0.1773 | 0.0013 | 0.0110 |
| 6 | 1,415 | 0.1452 | 0.1546 | **0.0093** | 0.0137 |
| 7 | 4,473 | 0.1269 | 0.1368 | **0.0098** | 0.0044 |
| 8 | 14,143 | 0.1133 | 0.1233 | **0.0099** | 0.0020 |
| 9 | 44,722 | 0.1034 | 0.1123 | **0.0088** | 0.0013 |

and at u = 2.25 the residual is 0.0026, 0.0033, 0.0028, 0.0024 (m = 6..9); at u ≥ 2.5 it is within noise. So the
Lemke Oliver–Soundararajan bias of the primes at x equals the primorial loop's bias at depth √(2x) minus a residual
ε(u) that is the same across four decades: ε ≈ 0 for u ≥ 2.5, ≈ 0.003 at u = 2.25, ≈ 0.0095 at u = 2. The loop
accounts for about 92 % of the primes' bias at 10⁹; the residual is the local/global delta of the transition
structure — the same shape as Buchstab's delta for the density (gate G5: 0.904 → e^{γ}/2 = 0.8905), appearing only in
the last quarter-unit of depth. **Recorded, not claimed:** it was not a sealed clause. It is the registrable object of
the next round: ε(u) as a scale-free function, with the loop's exact rationals as the anchor.

Also recorded: the loop's exact deficits at q = 3 fall 5/12, 223/552, 2860783/8291520, 76581569/235924480,
211594578367/687969884160, 32687539686737/110992602378240 for k = 3, 4, 6, 7, 8, 9 (0.4167 → 0.2945); the q-ratio
δ₁₀/δ₃ at the primes is 2.58, 2.53, 2.59, 2.58 (m = 6..9) against the LO–S leading term's 3 — a stable number the
next term has to explain; at q = 4 the primes' bias is 0.0963 at 10⁹ against the loop's 0.1045, the same residual.

## R1 — the shape metric is a bin-edge artifact for lattice gaps (instrument finding)

Gaps are even integers. Rescaling them by a sample mean and binning at width 0.1 puts each gap's spike at a bin edge
whenever the mean is near 15: gap 6 at mean 14.97 falls at 0.4008 and at mean 14.61 at 0.4107 — the entire spike hops
bins between the two halves of one window. That is the "floor" of 0.43 at m = 8, u = 2.25 (local halves; the loop's own
half-split floor there is 0.0005). The delta the metric reports at u ≤ 2.5 is therefore not a shape delta. What the
run does show honestly: at u ≥ 3 the local and loop shapes agree within noise in every decade (Δ ≤ 0.001), and the
crossover where Δ first exceeds 0.05 sits at u = 2.25 in all four decades — but the latter is the artifact's onset as
much as the delta's. The correct instrument compares on the lattice: match the loop's depth so that its *mean gap*
equals the window's (Mertens: y* with e^{γ}·log y* ≈ log x), then take TV between the raw integer histograms. Round 2.

## R3 — gap interiors are not random composites (recorded); the ratio has converged (INCONCLUSIVE by the rule)

ρ(m) = median(log d/log p)/median(log d_null/log p) is 0.929 at 10⁶ and 0.953–0.959 from 10⁷ to 10⁹; the medians are
0.2786 against the null's 0.2904 at 10⁹ (mean d 5,504 against 5,637). The interior of a prime gap has a smaller
largest-least-prime-factor than the same number of random composites of its decade: gap interiors are smoother than
random. The relation was INCONCLUSIVE because from m = 7 the differences sit below the floors — the quantity had already
converged, and the drift law has no verdict for that.

## Instrument lessons (added to the register)

1. **A drift law needs a CONVERGED verdict.** "Differences shrink and are resolvable" cannot confirm a quantity that has
   already converged below its floors — the best possible outcome scores INCONCLUSIVE. Next registration: CONVERGED when
   every difference is below its floor *and* the values agree within the floors across all scored decades.
2. **Never rescale a lattice variable into fixed bins.** Compare on the lattice; match scales by choosing the depth, not
   by dividing by a mean.
3. **Half-split floors on a 10⁹ window are sub-1e-4** and turn a clause "meets within the floor" into a coin flip at
   the fourth decimal. Register "meets within max(floor, 1e-3)" or a relative tolerance next time.
4. **A display line can kill a run.** `str()` on a primorial with thousands of digits; size from the bit length.

## Housekeeping found on the way (flagged, not fixed)

`asymmetric_conservation`'s "PAC conservation π(x) + C(x) = x − 1 exact at all 126 sieve steps" computes C(x) as
(x − 1) − π(x): an identity under STANDARDS §2.8 ("could any input have changed the verdict?" — no). The Mertens-product
comparison beside it is real; the "exact at every step" headline is not a measurement. That study's lane, not this one's.

## Forward note

Layer: arithmetic. What goes to Andy: the exact loop biases, the y-only table, and the residual at the primes. What goes
to round 2: ε(u) registered as a scale-free relation with a CONVERGED verdict class, the lattice shape metric, and R3
re-posed as a relation with a tolerance. Nothing here touches any milestone, Ξ, or φ.
