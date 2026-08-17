# 2026-08-17: the wall this milestone was founded on does not exist

M17 was opened on an empirical claim. That claim was a measurement artifact in my own
instrumentation. This journal records the retraction and what survives.

---

## What M17 was founded on

> **The empirical fact.** Four independent routes reached one wall, all measuring the same
> quantity without naming it… **ξ diverges at criticality.** Every one of these measured ξ at
> the white-noise floor. That is not "no structure" — it is *maximally sub-critical*, stated
> four times.

The anchor was `gravity_from_maxwell_pac/exp_11`, the corpus's own published 3D cosmic web, at
percolation **0.0068** against a 0.0025 noise control.

## What is true

**exp_11's web percolates.** Same run, same physics, same code — read at different binnings:

| res | particles/cell | percolation | `is_web` |
|---|---|---|---|
| 16 | 0.98 | 0.472 | True |
| 24 | 0.29 | 0.433 | True |
| **32 — exp_11's own** | 0.12 | **0.385** | **True** |
| 48 | 0.036 | 0.281 | False |
| **64 — where 0.0068 came from** | **0.015** | 0.062 | False |

At matched sampling the 3D substrate gives percolation **0.406 ± 0.069** from exp_11's own
*uncorrelated lattice* start, **5/5 seeds** passing the web gate. And the POC's own committed
replication script, run unmodified at exp_11's resolution, reports **0.3443 ± 0.0628**.

## The mechanism

4000 particles binned onto a 64³ grid is **0.015 particles per cell**. At that sampling the
density field is empty by construction: the overdense set shatters into singletons, so *any*
web — real or deliberately synthetic — reads as disconnected. A known-connected 3D control
reads 1.000 across occupancy 0.082–0.268, covering the whole regime, so the instrument was
sound and the sampling was not.

**The tell was in view throughout.** Occupancy read 0.012–0.04 in every one of those runs and
was printed beside every percolation value. In 3D the site-percolation threshold is 0.312;
nothing at occupancy 0.015 can span, whatever it is. I never asked whether the occupancy I was
measuring at was physically capable of producing the number I was looking for.

## What is retracted

- **"Four independent routes reached one wall."** The 3D anchor is gone. The 2D routes remain
  as measured but 2D was doing real damage on its own: the filament/sheet/node topology only
  exists in three dimensions, and site percolation sits at 0.593 in 2D against 0.312 in 3D.
- **"ξ at the white-noise floor… maximally sub-critical, stated four times."** Withdrawn.
- **"Criticality was never looked for, because the corpus had no instrument that could see
  it."** Withdrawn, and it was unfair on its own terms. `exp_12`, sitting beside exp_11,
  measured P(k) ~ k^−1.727 — "SCALE-FREE", cosmic similarity 0.849. The corpus was not
  measuring the wrong things.
- **The proposed corpus-wide survey** ("which results rest on a relational property but were
  measured with a one-point statistic?") is not justified by anything here and is withdrawn.

## What survives

**exp_01 and the instruments.** Block A's calibration stands on its own — it was validated
against 2D site percolation, a system with exact known answers, and it recovered them:
p_c = 0.5917 against an exact 0.5927460, γ/ν = 1.6233 against 1.7917, τ = 1.8490 against
2.0549, with R² 0.996 at p_c and 0.967 away from it. None of that depended on the retracted
claim. **This is the part of the milestone built the right way**, and it is why the
front-loading of Block A was correct even though the milestone's premise was not.

**The thesis is untested, not refuted.** "The limits this corpus keeps encountering are
critical points" was never actually tested — the wall it pointed at was an artifact, so there
was nothing to explain. Whether a DFT structure-forming system has a critical point, and
whether it sits at Ξ, remain open. What has changed is that the motivating observation is gone,
so the milestone needs a real reason to exist before Blocks B–E are worth running.

## The rule this earns

**Match particles-per-cell (`n/res^d ≈ 1`) before reading any connectivity statistic, and check
every threshold-based measure against a known-connected control at the same occupancy.**

Encoded in `reality-engine`'s `worldmodel.matched_res()` so the default is right without anyone
remembering it, with particles/cell printed on every run.

## The pattern worth keeping

Every measurement error in this round ran in the same direction: **understating structure.**
Percolation read low, an emergent-time transport test read null, a clock-field result read as
artifact, a forming web read as decay. An unbiased error process does not do that.

I think the cause is that skepticism was standing in for rigour. A null *feels* rigorous, so the
error mode that survived my own review was the one that looked careful — and a milestone got
founded on it.

There is a structural reason this work is exposed to it. In experimental physics the apparatus
is calibrated against nature. Here the system and the instrument are both built, so there is no
nature to calibrate against, and constructions fail toward their author's expectations. The
habit that held up was **measuring the reference in the same run** rather than reasoning about
thresholds: every result that survived had a control that could have killed it, and every
result that died lacked one.

Cross-reference: `reality-engine`
`proof_of_concepts/v4/poc_09_three_dimensions/journals/2026-08-17_the-web-percolates-the-artifact-was-mine.md`.
