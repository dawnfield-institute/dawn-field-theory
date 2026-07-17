# ADE Cascade Round 1 — Outcomes

**Date:** 2026-07-17
**Registration:** commit `c5e05712` (`2026-07-17_ade-cascade-round1-preregistration.md`)
**Gate:** exp_00 PASS before any registered run (G1 kernel identity 0.0; G2 per-scale
diff 0.0; G3 baseline −1.6126 vs −1.6083 reference).
**Results:** `exp_01_diagram_selectivity_20260717_150058.json`,
`exp_02_affine_vertex_20260717_150243.json`
**Integrity:** verdicts exactly per the registered rules; no metric, threshold, or
statistic changed post-registration. T1 clean in every arm (zero PSD-shift
activations; no contamination flags).

## R1: CONFIRM — diagram topology sets the exponent, decisively

All three family pairs separated (non-overlapping 95% CIs) at **all three ranks**,
with the identical ordering **E < D < A** (E steepest) everywhere:

| rank | A | D | E | A−D | A−E | D−E |
|------|------|------|------|------|------|------|
| 6 | −1.9234 | −1.9988 | −2.0170 | 0.0754 | 0.0936 | 0.0182 |
| 7 | −1.7513 | −1.8129 | −1.8376 | 0.0616 | 0.0863 | 0.0247 |
| 8 | −1.6130 | −1.6624 | −1.6880 | 0.0494 | 0.0750 | 0.0256 |

CI half-widths ≈ 0.0008; separations run 20–90× that. The mode-count-only null is
dead: **at equal rank, with identical means vectors, the Dynkin diagram alone moves
the spectral exponent** by up to 0.09.

**Reported [D], not scored (unregistered observation):** exp_15 B.2 recorded that the
legacy engine converges to −1.6113, excluding Kolmogorov −5/3 from its CI — a 3.3%
unexplained gap. That gap is an **A-family artifact**: at rank 8, D_8 lands at
−1.6624 (0.26% from −5/3) and E_8 at −1.6880. Branched topologies close most of the
distance to the physical exponent. Whether the physical target *selects* a diagram is
exactly the round-2 question; it enters as a registered claim then, not now.

## R3: CONFIRM — the mode-count map survives the kernel swap

Strictly monotone in rank within every family (A over 4 ranks, D and E over 3). The
legacy monotonicity was not an artifact of the ad-hoc kernel.

## R2: KILL — the affine-vertex reading of the k−1 offset is dead

Registered rule: CONFIRM if median ρ < 0.25, KILL if > 0.75. Measured:

| rank | exp(A_r) | exp(Ã_r) | exp(A_{r+1}) | shift_affine | shift_path | ρ |
|------|------|------|------|------|------|-----|
| 6 | −1.9234 | −2.0409 | −1.7513 | 0.1175 | 0.1721 | 0.68 |
| 7 | −1.7513 | −1.8771 | −1.6130 | 0.1258 | 0.1383 | 0.91 |
| 8 | −1.6130 | −1.7610 | −1.4941 | 0.1480 | 0.1189 | 1.25 |

Median ρ = **0.91 → KILL**. The affine node does not act as a passive reference: it
shifts the exponent by roughly a full mode's worth. Worse for the hypothesis (and the
useful clue): the shift is in the **opposite direction** — path extension makes the
cascade shallower, cycle closure makes it *steeper* (Ã_r < A_r < A_{r+1} throughout).
Mechanically sensible in hindsight: closing the cycle halves graph distances
(diameter ⌊(r+1)/2⌋ vs r), so every off-diagonal coupling strengthens, organized
fraction rises, and the spectrum steepens. The k−1 offset therefore remains
**unexplained** — the affine reading was a wrong guess, cleanly killed by its
registered test. Its replacement hypothesis (if any) must be registered in a future
round; none is proposed post hoc here.

## Ledger (cumulative, this experiment)

- R1 diagram selectivity: **CONFIRM** (registered relation survived)
- R3 bridge monotonicity: **CONFIRM** (registered relation survived)
- R2 affine vertex: **KILL** (registered relation died; direction-reversal clue
  recorded [D])
- T2 (feedback wash-out) contingency arm: not triggered (R1 did not KILL).

Two relational registrations survived, one died with a mechanism clue — consistent
with the invariant-registration meta-claim's running tally (relations carry signal;
see midnight ledger).
