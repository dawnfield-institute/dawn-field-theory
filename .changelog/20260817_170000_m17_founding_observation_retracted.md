# M17: the founding observation is retracted

## What happened

Milestone 17 (Criticality) was opened on an empirical claim: that four independent routes had
triangulated on a percolation floor of 0.007–0.019, anchored on `gravity_from_maxwell_pac`'s
`exp_11` — the corpus's own published 3D cosmic web — at **0.0068** against a 0.0025 noise
control. The reading was that the framework is "maximally sub-critical, stated four times."

**That was a sampling artifact in my instrumentation, not a fact about exp_11.**

## The correction

exp_11's web percolates. Same run, same physics, binning alone:

| res | particles/cell | percolation | `is_web` |
|---|---|---|---|
| 16 | 0.98 | 0.472 | True |
| **32 — exp_11's own** | 0.12 | **0.385** | **True** |
| **64 — the retracted claim** | **0.015** | 0.062 | False |

At matched sampling the 3D substrate gives 0.406 ± 0.069 from exp_11's own *uncorrelated
lattice* start, 5/5 seeds passing the web gate. 4000 particles on a 64³ grid is 0.015 per cell:
the field is empty by construction, so any web — real or synthetic — reads as disconnected. A
known-connected 3D control reads 1.000 across occupancy 0.082–0.268.

The tell was printed throughout: occupancy read 0.012–0.04, and in 3D the site threshold is
0.312. Nothing there could have spanned.

## Retracted

- "Four independent routes reached one wall" — the 3D anchor is gone.
- "ξ at the white-noise floor… maximally sub-critical."
- "Criticality was never looked for, because the corpus had no instrument that could see it" —
  unfair on its own terms. `exp_12`, beside exp_11, measured P(k) ~ k^−1.727, SCALE-FREE,
  cosmic similarity 0.849.
- The proposed corpus-wide "relational claims measured with one-point statistics" survey.

## Survives

**exp_01 and the criticality instruments.** Block A was calibrated against 2D site percolation —
a system with exact known answers — and recovered them: p_c 0.5917 against an exact 0.5927460,
γ/ν 1.6233 against 1.7917, τ 1.8490 against 2.0549, R² 0.996 at p_c and 0.967 away from it. It
never depended on the retracted claim. Front-loading Block A was the right call even though the
milestone's premise was not.

## Status change

`active` → **`archived`**, confidence 0.35 → 0.15. Archived rather than falsified: the thesis
was never tested, only its motivation removed. The instruments are reusable and the question is
revivable, but that would be a new founding rather than a continuation.

## Files

- `experiments/milestones/milestone17/journals/2026-08-17_the-wall-was-my-instrument.md` — new
- `README.md`, `SYNTHESIS.md`, `meta.yaml` — retraction recorded at the top of each
- Cross-reference: `reality-engine`
  `proof_of_concepts/v4/poc_09_three_dimensions/journals/2026-08-17_the-web-percolates-the-artifact-was-mine.md`

## Note

Every measurement error in this round ran one direction: understating structure. A null reads as
rigour, so the error mode that survived review was the one that looked careful — and a milestone
got founded on it. In this work the system and the instrument are both built, so there is no
nature to calibrate against; the habit that held up was measuring the reference in the same run
rather than reasoning about thresholds.
