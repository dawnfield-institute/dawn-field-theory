# prime_gap_ledger round 2 — exp_02 "the position of the read": 3/4; the residual is Buchstab's, seen through a pair statistic

**Date:** 2026-09-07 (evening) · **Branch:** `study/prime-gap-ledger` (PR #188) · **Seal:** `ae67f522` · **Layer:** arithmetic

## What

Peter's reading of round 1's residual: the gaps are tranches; what the local read misses comes from the fully actualized
identity — all residues at once — so look completely locally and completely globally and *forget the delta as a thing*.
Made precise: the uniform loop is the loop read at a random position; the primes are the loop read at its **origin**
(the arc below y², where every residue is the integer's own name). Checked in exploring mode (a fixed-depth read slid
from the origin outward climbs to the loop's value within half a unit of depth and plateaus to 10⁵⁰; on the record as
`explore_r2_position_along_the_loop.py`), then registered as exp_02 with four relations and the CONVERGED verdict
class round 1 lacked. Andy Farmer's mid-round diagram (Archimedean order against the 2-adic ultrametric tree) became
R4: does the residual depend on the modulus's p-adic shell depth (his tranche) or not (the position)?

## Gates (G1–G6, before the seal; two calibrations on the record)

Arc-integral Buchstab within 2/log N (the first run's 1/log N failed the u = 1.75 controls by exactly the prime
count's next term, li against x/log x); fresh-seed uniform loop vs round 1 within 3σ; de-trended window scatter within
[0.5, 2] of the noise (the raw scatter read the LO–S drift across the 10⁹ decade as 3.2× the noise); exact
reproducibility of the decade cells; the equidistribution plateau at u ∈ {5, 7}; prime-power lifts against enumerated
loops mod lcm(q, P_k).

## Verdicts (`journals/2026-09-07_exp02_outcomes.md`)

- **R1 CONFIRM** — the residual at the primes' arc is scale-free within 10 % (0.0101, 0.0102, 0.0087 at 10⁷–10⁹) and
  decreases at every resolved step over u = 2 → 2.1 → 2.25.
- **R2 CONFIRM — the sign flip.** Where Buchstab's density ratio exceeds one (u = 2.75, 3) the loop *under*-predicts
  the bias, at 3–6σ in both depths with 200 windows; the sign of the residual tracks the sign of the measured deficit
  cell by cell, including the one positive resolved cell.
- **R3 INCONCLUSIVE** — c = ε/d ≈ 0.10 on all 14 resolved cells, but the seal's CONVERGED and KILL clauses both fire
  (two noisy flip cells differ by a factor 2.1). The scorer had given CONVERGED precedence the seal never granted;
  corrected toward the seal, rerun (deterministic), both result files kept.
- **R4 CONFIRM** — shell independence at the sealed 15 %, with a consistently-signed ~10 % deeper-shell modulation
  recorded (r₉ < r₃ at 3.4σ at 10⁹): Andy's tranche modulates the residual; it does not make it.

**The sentence:** the Lemke Oliver–Soundararajan bias of consecutive primes equals the primorial loop's exact rational
bias at √(2x) minus one tenth of Buchstab's density deficit, sign included — the residual is the position of the
read, and it changes sign where ω(u) crosses e^{−γ}.

## Lessons (register)

A leading-order formula's gate tolerance must exceed its own next term; a raw window scatter reads systematic drift as
noise (de-trend); two sealed clauses need a stated precedence; the negative control (u < 2) failed the position curve
as declared — controls that cannot fail are not controls.

## Files

`experiments/studies/prime_gap_ledger/{core/rough.py (prime-power lifts, arc-integral ω, de-trended SE),
scripts/exp_02_gates.py, scripts/exp_02_position_of_the_read.py, scripts/explore_r2_position_along_the_loop.py,
results/ (append-only), journals/2026-09-07_exp02_registration.md, journals/2026-09-07_exp02_outcomes.md,
journals/2026-09-07_exploring_position_along_the_loop.md}`; README (round 2 rows, score 3/7), meta.yaml; THEORY_MAP
claims row; ROADMAP open row. Lore page `prime-gap-ledger` rev 3.
