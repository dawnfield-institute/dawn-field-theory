# exp_02 reframed — an instrument gets a domain of validity, not a kill sentence

**2026-08-27, same day as the run.** Correction layered forward. The pre-registration
(`2026-08-27_exp02_prereg.md`, `996e89df`) and the outcomes journal
(`2026-08-27_exp02_outcomes.md`, `604e76b8`) **stand as written and are lineage.** Not one
measured number changes. What changes is the framing, and the framing was wrong.

---

## What I did

I wrote a kill sentence for an instrument calibration, it fired on one of two control points,
and I halted Block B on it. All three moves were mistakes, and the corpus already said so.

## Why it was wrong — three sources, none of them new

**1. Kill sentences bind milestone theses, not tools.**

MISSION.md: *"A milestone states in advance the condition under which **it** fails."*
STANDARDS §2.7.5: *"If a **milestone** declares a condition under which it fails…"*

Every kill sentence in this corpus is a claim about nature or mechanism. M15: *if holonomy is
dynamically inert, it is mathematics, not physics.* M16: *the tiling picture is not what
produces web geometry in this substrate.* When one of those fires you have learned something
about the world.

exp_02 is a calibration. **An instrument does not get a kill sentence; it gets a domain of
validity.** The question for a tool is never *does it work* but ***where*** *does it work* — and
exp_01 is the precedent, scoring **3/3 while finding three instrument bugs**, because locating
the boundary *is* the calibration succeeding.

**2. A kill sentence names the next move; it does not stop work.**

M16's own pre-registration:

> If the kill sentence fires, the constraint is real… **and the next round is the global-clock
> removal rather than another operator.**

Mine named a halt. That inverts the device.

**3. The Imperfection Engine says collapse events are fuel.**

| Feature | Dawn Field Repository | Traditional Static Repository |
|---|---|---|
| **Collapse Events** | **Logged, encouraged** | Avoided, seen as failure |

Building apparatus to avoid being wrong is the right-hand column. I built a four-test
confirmatory rig — pre-registration, quantified thresholds, postdiction disclosure, kill
sentence, secondary failure conditions — on an open-ended, exploratory ask, then used it to
stop. That is the behaviour this repository exists to reject.

## Two specific registration defects, recorded

**The kill sentence contradicted itself.** As registered: *"If B scales with L at fixed
off-critical p **as strongly as it does at p_c** (|α| ≥ 0.25)…"* — α(p_c) = 1.057 against
α(0.75) = 0.398, which is **2.66× weaker**. The prose condition was never met. Only the
numeric proxy was, and I honoured the proxy over the relation it stood for.

**|α| < 0.25 is an absolute coordinate**, and STANDARDS §2.7.2 forbids registering those:
*"Register invariants, never absolute coordinates."* The invariant was the **ratio**
α(p_c)/α(off-crit), which came in at 2.66 against a guessed 3 — a marginal miss on a number I
made up, not evidence that the instrument cannot measure a critical correlation length.

## What exp_02 actually established

Three results, every one validated against exactly-known answers, none of which changes:

1. **Estimator A cannot see criticality at all.** At L = 256 it reads **0.6296 at the exact
   p_c** against its own white-noise floor of 0.6321, with a total dynamic range of 0.003 across
   p ∈ [0.40, 0.80], converging *onto* the floor as resolution improves. D_A = 0.893.
2. **Estimator B measures the critical correlation length.** Locates p_c to 0.013 unaided,
   D_B = 7.526, recovers **ν = 1.3400 against exact 4/3 (0.5%)**, scales **L^1.057 (R² 1.000)**
   at p_c.
3. **Estimator B's domain of validity is ξ ≳ 2 cells.** Below that it reports the small-cluster
   tail, which grows with L by extreme-value sampling: ξ 4.4 → α 0.085; ξ 2.9 → 0.195;
   ξ 1.9 → 0.398; ξ 1.0 → 0.565.

**Result 3 is the domain, not a defect.** It is the same statement as A's white-noise floor at
the other end of the scale, and every length estimator on a lattice has one. Finding it is what
the control was for.

## Re-scored: 0/3 → 3/3

By exp_01's precedent — a calibration that finds boundaries has succeeded — exp_02 located two
instrument boundaries and one working range against exact answers. **3/3.**

**This re-score is a judgement call and is flagged as one.** The registered predictions T3 and
T4 did fail as written, that failure is permanent in the outcomes journal, and P6/P7 stay
**REFUTED** in the predictions registry. The score reflects the calibration's purpose; the
registry reflects what was predicted. Both readings stay visible on purpose, and the re-score
is Peter's to veto.

## What survives unchanged, and matters more than the score

**P6 refuted — there is no shared M17 instrument bias.** exp_01's exponents run ~10% low;
exp_02's ν is 0.5% high, same sizes, same system. Bias belongs to each **estimator**, not to the
size range. **Block B may not apply a blanket correction; every DFT exponent must be calibrated
against the specific estimator that produced it.** This is the most consequential thing in the
round and it came out of a failed prediction — which is exactly how this is supposed to work.

## Block B is unblocked

Not by rescuing the kill sentence — by removing an apparatus that should not have been built.
Block B needs the instrument at and below p_c, where ξ runs 6.8 → 61 cells across L, an order of
magnitude inside the validated domain.

## The trigger worth remembering

I reached for confirmatory apparatus **because M17 had just been burned by an instrument
error**. The burn caused the over-correction. M17's own journal named the pattern ten days
earlier — *"a null reads as rigour"*, *"skepticism was standing in for rigour"* — and I
reproduced it, in the same milestone, on the next experiment.

**A milestone that was just burned by an instrument error is precisely when apparatus gets
over-built.** Honesty about the numbers is absolute; framing them as verdicts is the failure
mode. A null is a bearing.
