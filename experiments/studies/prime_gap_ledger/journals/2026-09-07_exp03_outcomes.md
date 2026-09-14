# exp_03 outcomes — the density-matched loop: 2/2; the delta is the depth shift, and nothing else

**Registration:** `ff63f8d5` (`2026-09-07_exp03_registration.md`), scored to the sealed text.
**Scored run:** `results/exp_03_density_matched_loop_main_20260907_171834.json` (+ `_log.txt`), depths 8, 9, 10, seed
20260909, 18 loops, 382 s. The first main run of the same evening computed every cell (identical, deterministic) and died
at the final serialisation on a tuple-keyed table in the scorer; it is kept beside the scored run, its checkpoint intact.
**Verdicts:** R1 CONVERGED · R2 CONVERGED. **Score 2/2.**

## Scorecard

| relation | verdict | by the letter |
|---|---|---|
| R1 the origin is a depth shift: ρ_q = r_q − r_q^eff = 0 at the primes' arc, m = 9 and 10, every unsaturated modulus on the ladders {3, 9}, {4, 8, 16}, {5}, {7} | **CONVERGED** | 14 cells counted; every \|ρ\| within its tolerance, the largest using 0.49 of it; the largest \|ρ\|/SE = 1.48; no ladder difference resolved (all within ~1σ); no KILL clause fires |
| R2 the same along the curve and through the flip: u ∈ {2.1, 2.25, 3} at m = 10 | **CONVERGED** | 21 cells counted; the largest \|ρ\| uses 0.58 of its tolerance; the largest \|ρ\|/SE = 1.73; at u = 3 the predicted sign is right for all seven moduli; no ladder difference resolved |

Kill relevance: none fires. The m = 8 cells (recorded) behave the same (7 cells, largest 2.3σ); the m = 9 curve
(recorded) CONVERGES under the same evaluation with the right signs at u = 3.

## What the run says

**At the primes' arc, ρ is zero to a thousandth at 10¹⁰.** With 427,154,205 primes and loops of 200 windows:

| q | 3 | 9 | 4 | 8 | 16 | 5 | 7 |
|---|---|---|---|---|---|---|---|
| r (primes) | 0.0814 | 0.0724 | 0.0818 | 0.0784 | 0.0667 | 0.0787 | 0.0718 |
| r^eff (density-matched loop) | 0.0814 | 0.0721 | 0.0814 | 0.0782 | 0.0665 | 0.0789 | 0.0721 |
| ρ ± SE | +0.0000 ± 0.0008 | +0.0002 ± 0.0004 | +0.0004 ± 0.0010 | +0.0002 ± 0.0005 | +0.0002 ± 0.0004 | −0.0002 ± 0.0005 | −0.0003 ± 0.0004 |

At 10⁹ the seven ρ's are −0.0014 ± 0.0011, −0.0008 ± 0.0006, −0.0009 ± 0.0016, −0.0009 ± 0.0009, +0.0003 ± 0.0005,
+0.0003 ± 0.0005, +0.0008 ± 0.0009. The shell differences that round 2 recorded at up to 3.4σ against the *uniform*
loop are gone against the *density-matched* loop: (3 → 9) +0.0002 ± 0.0009, (4 → 8) −0.0002 ± 0.0011, (8 → 16) 0.0000 ±
0.0007 at 10¹⁰. **Andy Farmer's tranche beyond position: none, down to ±0.001 in r — about one percent of the residual.**

**Through the flip with no free parameter.** At u = 3 (m = 10) the read is denser than the uniform loop, so the
density-matched depth is *shallower* (y_eff = 129,959 against y = 141,422) and the model predicts a negative residual
per modulus. Observed r against predicted r^eff: q = 3: −0.0069 / −0.0067; 9: −0.0052 / −0.0052; 4: −0.0059 / −0.0059;
8: −0.0057 / −0.0055; 16: −0.0051 / −0.0053; 5: −0.0053 / −0.0052; 7: −0.0050 / −0.0055. Round 2 measured the sign flip;
round 3 derives it.

**The depth shift is Buchstab's density ratio, exactly.** In all fifteen cells, log y_eff / log y equals 1/ratio to
three or four decimals (10¹⁰: 1.1082 / 1.1080 at u = 2; 1.1001 / 1.0998 at 2.1; 1.0471 / 1.0470 at 2.25; 0.9929 /
0.9928 at 3; 1.0000 / 1.0000 at 5) — Mertens' e^{−γ}/log y turned around. So the position curve of round 2 and the
residual curve of round 1 are one statement: *a read at depth u of the loop at y is the uniform loop at depth
y^{1/ratio(u)}*, and the transition bias follows because δ_q on the loop is a smooth function of depth.

**The loop's own shell profile** (recorded): β_q = −d log δ_q / d log log y from the two loops of each origin cell —
q = 3: 0.816, 0.825, 0.827; 9: 0.742, 0.732, 0.729; 4: 0.827, 0.828, 0.826; 8: 0.796, 0.794, 0.792; 5: 0.786, 0.795,
0.800; 7: 0.695, 0.710, 0.728 at 10⁸, 10⁹, 10¹⁰. The deeper shell of each prime has the smaller β — that, and only
that, was round 2's "modulation". Saturated moduli: 49 everywhere, 25 at 10⁸.

## The sentence, as the record now has it

**The consecutive-prime residue bias at every modulus, shells included, equals the primorial loop's bias read at the
density-matched depth y_eff = y^{1/ratio}, where ratio is Buchstab's local-to-global density ratio. The delta between
the primes and the loop is the position of the read, and nothing else, to one part in a thousand of r at 10¹⁰.**
Lemke Oliver–Soundararajan's bias, its decay, its shell structure and its Buchstab-shaped residual are one object read
at one shifted depth.

## Postdiction, honestly

The model was a review's postdiction on round 2's file (its β_q and y_eff are round-2 objects); G7 showed it
reproduces round 2 at u = 2 within round 2's SEs (14/15). What this round added that was not seen: the 10¹⁰ decade
(four hundred million primes), loops four times deeper (SE(δ₃) ≈ 1.6·10⁻⁴), the shells 16, 25, 27, 49 at m ≥ 8, and the
flip cells against a prediction with no free parameter. The registration's prediction (CONVERGED) held; the KILL
clauses were live and did not fire.

## Instrument record

1. The scorer's ladder table was keyed by tuples and the final JSON write failed after every cell had been computed;
   the key is now a string; the first main run is kept.
2. Gate G7 was recalibrated before the seal (15 comparisons at 3σ: at most one beyond 3σ, none beyond 4σ; the miss
   was m = 8, q = 5 at 3.38σ); the first gate run is kept.
3. The 10¹⁰ decade's count matched sympy's prime-counting function exactly (427,154,205), and every read's transition
   count was n − 1 exactly (the chunk carry).

## Forward corrections filed

In round 2's outcomes journal: the q = 10 channel was an identity; R4's reading is superseded; the c drift explained.

## What is not claimed

Any departure at moduli or depths not read; anything above 2·10¹⁰; any statement about u < 2; the value of any β_q
beyond "recorded". No physics; nothing here touches φ, Ξ or any milestone.

## Forward note

Layer: arithmetic. The study's three rounds close on one object. What would still be worth a registration: the
depth-shift law read at moduli with a prime factor above y (where the loop carries no shell at all), and whether the
smooth-depth profile β_q(y) has a closed form. Neither is opened here. Bundle 4 waits for Peter.
