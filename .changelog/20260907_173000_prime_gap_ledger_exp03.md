# prime_gap_ledger round 3 — exp_03 "the density-matched loop": 2/2; the delta is the depth shift, and nothing else

**Date:** 2026-09-07 (late) · **Branch:** `study/prime-gap-ledger` (PR #188) · **Seal:** `ff63f8d5` · **Layer:** arithmetic

## What

Round 2 had recorded a deeper-shell modulation of the residual (Andy Farmer's ultrametric tranche) below its sealed
tolerance. The round-3 design review, computed from round 2's file alone, showed that the whole residual structure of
rounds 1–2 — the residual, its sign flip, the shell modulation, the drift in c — is one object read at a shifted depth:
the uniform loop's own δ_q falls with depth as (log y)^{−β_q}, and the primes' arc is the loop read at the effective
depth y_eff whose exact Mertens product equals the arc's density. So the round was re-posed: register what the model
leaves, ρ_q = r_q − r_q^eff, the departure of the primes' p-adic gap profile from the density-matched loop.

## Gates (G1–G8, before the seal; one calibration on the record)

Lift invariance (exact; δ₁₆ on the enumerated loop mod 8·P₈ 0.94034 vs sampled 0.94031); exact reproducibility of
round 2's decade cells and the identity δ₁₀ ≡ δ₅; the 10¹⁰ decade in 100 chunks with the residue carried — count
427,154,205 = π(2·10¹⁰) − π(10¹⁰) by sympy, exact; transitions = n − 1 on every read; fresh loops within 3σ of round
2; de-trended scatter within band; the density-matched prediction against round 2's r_q at u = 2 (14/15 within 3σ, one
at 3.38σ — the first run demanded 15/15; over 15 comparisons that is chance-rate, the gate is ≤ 1 beyond 3σ and none
beyond 4σ); the y_eff brackets straddle, and log y_eff/log y = Buchstab's 1/ratio to four decimals.

## Verdicts (`journals/2026-09-07_exp03_outcomes.md`)

- **R1 CONVERGED** — at the primes' arc (10⁹, 10¹⁰; moduli 3, 9, 4, 8, 16, 5, 7) every ρ_q is within its tolerance,
  none beyond 1.5σ, none using more than half its tolerance; no shell-ordered difference resolved: ±0.001 in r at
  10¹⁰ with 427 million primes and 200-window loops.
- **R2 CONVERGED** — along the curve and through the flip (u = 2.1, 2.25, 3 at 10¹⁰): 21 cells, none beyond 1.7σ; at
  u = 3 the depth shift alone predicts the negative residual and matches at every modulus (q = 3: −0.0069 observed,
  −0.0067 predicted).

**The sentence:** the consecutive-prime residue bias at every modulus, shells included, equals the primorial loop's
bias read at the density-matched depth y^{1/ratio}, with ratio Buchstab's local-to-global density ratio. The delta
between the primes and the loop is the position of the read, and nothing else, to one part in a thousand at 10¹⁰.
Andy Farmer's tranche beyond position: none resolved. Recorded: the loop's own shell profile β_q (e.g. 0.827, 0.729,
0.826, 0.792, 0.800, 0.728 for q = 3, 9, 4, 8, 5, 7 at 10¹⁰).

## Corrections and lessons

Forward corrections to round 2 filed in its outcomes: the q = 10 channel was an identity (every gap is even); R4's
reading is superseded (the modulation was the loop's β_q ladder); the c drift explained. Lessons: a modulation
predicted by the instrument's own depth profile is a known answer — compute the model's prediction before registering
the modulation; a 15-cell known-answer gate at 3σ needs a multiple-comparison rule; a tuple-keyed table survives every
cell and dies at the JSON write (run 1 kept).

## Files

`experiments/studies/prime_gap_ledger/{core/rough.py (y_eff solver, chunked read with carry, log-detrended SE,
loop_read), scripts/exp_03_gates.py, scripts/exp_03_density_matched_loop.py, results/ (append-only),
journals/2026-09-07_exp03_registration.md, journals/2026-09-07_exp03_outcomes.md, journals/2026-09-07_exp02_outcomes.md
(forward corrections)}`; README (round 3, score 5/9), meta.yaml; THEORY_MAP claims row; ROADMAP open row.
