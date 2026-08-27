# M17 exp_02 outcomes — the kill sentence fires, and the instrument is better than registered

**Run**: 2026-08-27. **Pre-registration**: `2026-08-27_exp02_prereg.md`, commit `996e89df`.
**Result**: `results/exp_02_correlation_length_20260827_142044.json`
**Score**: **0/3.** Four tests ran. T1 and T2 passed but were disclosed postdictive before the
run, and STANDARDS §2.7.4 is explicit that only predictions registered before measurement count
as confirmations — so they score nothing. T3 and T4, the only predictive tests, both **FAIL**.
**The kill sentence fires. Block A does not complete and Block B is not licensed by this run.**

The instrument nevertheless works, and the prose below says what it established. A score of zero
and a working instrument are not in tension: the score tracks registered predictions confirmed,
and I registered the wrong two things.

---

## The headline, stated against the exact answer

At **L = 256**, `structure.correlation_length` reads **0.6296 at the exact critical point** of
2D site percolation, against its own documented white-noise floor of **0.6321**. Its entire
dynamic range across p ∈ [0.40, 0.80] is **[0.628, 0.631]** — a span of 0.003, while the
distance from the floor at p_c is 0.0025.

And it gets *worse* with resolution. As L grows the reading converges **onto** the floor:

| L | A at exact p_c | gap to floor 0.6321 |
|---|---|---|
| 32 | 0.6111 | 0.0210 |
| 64 | 0.6199 | 0.0122 |
| 128 | 0.6257 | 0.0064 |
| 256 | **0.6296** | **0.0025** |

Meanwhile the connectivity length, on the *same lattices in the same run*, goes
**6.84 → 13.64 → 29.30 → 60.95**, scaling as **L^1.057 with R² = 1.000**.

**A system sitting exactly on its critical point reads at the white-noise floor on estimator A.**
So the inference that opened M17's fourth route —

> ξ diverges at criticality, therefore ξ at the floor means *maximally sub-critical*

— is **invalid**. The floor reading is compatible with exact criticality, and the better the
resolution the more perfectly compatible it becomes. Discrimination power confirms it
numerically: **D_A = 0.893** (needed < 1) against **D_B = 7.526** (needed ≥ 5).

This does not say the engine is critical. It says the 0.63 reading never carried the
information that was read out of it.

## Test by test

| test | standing | result | verdict |
|---|---|---|---|
| T1 discrimination | postdictive | D_A 0.893, D_B 7.526 | PASS (replication only) |
| T2 location | postdictive | B peaks 0.580 at L = 64/128/256 vs exact 0.5927; A peaks scatter to 0.675, 0.500, 0.400, 0.640 | PASS (replication only) |
| T3 collapse ν | **predictive** | ν = **1.3400** vs exact 1.33333, **+0.5%** | **FAIL** |
| T4 off-critical control | **predictive** | α(0.45) = +0.146 ✓, α(0.75) = **+0.398** ✗, ratio 2.66 < 3 | **FAIL** |

## T3 failed by being too good, and that is a real result

I registered that ν must land in [1.10, 1.60] **and** be biased **low by 5–20%**, matching the
direction of exp_01's γ/ν (9.4% low) and τ (10.0% low). The registered content was the
*relation*, not the number:

> if ν is also low by a similar margin, the instrument family has one coherent finite-size bias,
> and **that** is what licenses Block B to compare a DFT exponent against this calibration
> instead of against textbook values.

ν came out at **1.3400 — 0.5% high**, essentially exact. The number is an excellent recovery.
**The registered relation is refuted.**

That refutation has teeth, and it is the most useful thing in this run:

> **There is no single M17 finite-size bias.** exp_01's exponents are ~10% low; exp_02's ν is
> 0.5% high, at the same sizes, on the same system. Bias is a property of *each estimator*, not
> of the milestone's size range. **Block B may not apply a blanket correction, and any DFT
> exponent must be calibrated against the specific estimator that produced it.**

Recording this as a pass because "ν is close to 4/3" would have thrown away the finding. The
prediction was about the bias structure and the bias structure is not what I said it was.

## T4 failed, the kill sentence fires, and the diagnosis is a resolution floor

Registered: |α| < 0.25 in ξ ~ L^α at fixed p = 0.45 **and** p = 0.75, with ratio to α(p_c) ≥ 3.

- α(p_c) = **+1.057**, R² 1.000 — textbook ξ ~ L at criticality
- α(0.45) = **+0.146** — passes
- α(0.75) = **+0.398** — **fails**
- ratio **2.66** < 3 — fails

Kill sentence, as registered:

> If B scales with L at fixed off-critical p as strongly as it does at p_c (|α| ≥ 0.25), then B
> does not measure a critical correlation length, Block A does not complete, and Block B has no
> instrument.

**Honoured.** STANDARDS §2.7.5. exp_02 is scored as a fail and Block A stays open.

The diagnosis, which is not a rescue and does not change the score: α tracks how close ξ sits to
the lattice scale, on both sides of p_c.

| p | ξ at L=256 | α |
|---|---|---|
| 0.40 | 4.4 | +0.085 |
| 0.45 | 6.7 | +0.146 |
| 0.70 | 2.9 | +0.195 |
| **0.75** | **1.9** | **+0.398** |
| 0.80 | 1.0 | +0.565 |

Below **ξ ≈ 2 cells** the estimator cannot resolve the true length, and what it reports instead
is the small-cluster tail — which grows with L because a larger lattice samples further into
that tail. It is extreme-value sampling, not divergence.

**This is the same failure class as estimator A's white-noise floor.** An estimator near its
resolution limit returns a number that tracks the sampling rather than the physics. exp_02 set
out to document that floor for A and found that B has one too, at the other end. The control
found it, which is what the control was for.

**Limitation on this diagnosis**: the sub-critical side of the sweep never reaches ξ < 3.7 cells
at p ≥ 0.40, so the floor was exposed only on the super-critical side. Whether it is genuinely
symmetric in ξ, or something specific to finite clusters coexisting with an excluded spanning
cluster, **is not established here.**

## What B is, and is not, licensed for

Established by this run, independent of the kill sentence:

- locates p_c to within 0.013 without being told where it is (T2)
- discriminates critical from off-critical at 7.5σ against A's 0.9σ (T1)
- recovers ν to 0.5% by data collapse
- scales as L^1.057 (R² 1.000) at p_c

Not established, and blocking:

- a validated domain of use. The registered control failed at p = 0.75, and **ξ ≳ 2 cells** is a
  *post-hoc* boundary read off this run's own data. It has not been registered or tested.

A v2 registration is the correct route — the M16 precedent, where `2026-08-14_exp01_prereg_v2.md`
superseded v1 rather than v1 being re-scored. What v2 must register, before running:

1. **ξ ≳ 2 cells as a stated domain of validity**, with the control placed inside it and a
   second control placed deliberately *outside* it that is **required to fail**. A floor is only
   demonstrated if the instrument is shown breaking where the floor says it should.
2. Whether the floor is symmetric in ξ or specific to the super-critical geometry.
3. **No shared-bias assumption.** Refuted here; each exponent carries its own calibration.

## Secondary conditions

| condition | status |
|---|---|
| Occupancy reported with every percolation number | met — occupancy recorded per (L, p); it equals p by construction for site percolation |
| Spanning cluster excluded, exclusion asserted not trusted | met — 622/623/624/624 exclusions above p = 0.62 across the four L |
| Open boundaries, `periodic=False` passed explicitly to A | met |

## Instrument fault caught during the smoke test

The ν minimiser returned **2.60 — the top of its own scan range** — on the L = 16/24 smoke run.
A minimiser sitting on its boundary has not found a minimum, it has run out of room, and
reporting that as "ν = 2.60" is precisely the fault class this milestone exists to catch. Fixed
before the registered run: `nu_at_scan_boundary` is recorded and T3 fails outright on a boundary
hit. It did not trigger on the real run (ν = 1.34, interior).

That makes **twelve** instrument faults across this milestone and the two rounds preceding it.

## What this does not claim

It does not reinstate *"criticality was never looked for"* — withdrawn 2026-08-17 on exp_12's
scale-free P(k) ~ k^−1.727, and untouched here. It does not re-open M16. It does not say the
engine is critical.

It removes one piece of evidence that said "not critical", which leaves Q1 **open**. That is a
bearing, not a verdict.

## FDO Links

- Milestone node: `milestone17-criticality` (status `archived`)
- Pre-registration: `journals/2026-08-27_exp02_prereg.md` @ `996e89df`
- Estimator A: `reality-engine/proof_of_concepts/v4/structure.py`
