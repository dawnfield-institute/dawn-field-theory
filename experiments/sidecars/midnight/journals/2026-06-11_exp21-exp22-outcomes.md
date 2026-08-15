# Outcomes: exp_21 (Coupling Law) KILLED, exp_22 (Monitoring Regulators) 2/3

**Date:** 2026-06-11
**Pre-registration:** commit `f4c3dfb1` (P17 derivation + both experiments, before execution)
**Results:** `exp_21_coupling_decoherence_20260611_101553.json`,
`exp_22_monitoring_regulators_20260611_101349.json`

---

## exp_21: KILLED (registered rule), 2/4 — and the kill localizes the model flaw

| Test | Result |
|------|--------|
| T1 knee at 27.2 eV | **FAIL** — free knee CI95 = [7.9, 8.2] eV, ΔBIC = +8.4 |
| T2 envelope vs logistic | PASS (ΔBIC = −0.4) |
| T3 per-panel knee consistency | PASS (S: [6.2, 8.5], X: [7.9, 33.5] — wide) |
| T4a FeII flip within Δz 0.5 of MgII | FAIL (MgII 2.54, FeII 3.46) |
| T4b CIV monotone | **FAIL** (flips detected both panels) — kill trigger |
| T4c SiIV monotone | PASS |

### Why it died, precisely

**The P17 model conflated two things it must not conflate.** §3 of the derivation treated
|β|(IP) as a pure monitoring-lag magnitude and §4 treated sign as a separate flux direction.
But near a species' turnover epoch, β passes through zero because the *flux reverses*, not
because monitoring is strong — so |β| measured in a window that straddles or approaches
turnover is small for flux reasons, and |β| far from turnover is inflated by transient flux,
independent of the monitoring capacity. Fitting |β| vs IP alone is therefore structurally
wrong whenever the panel's z-range interacts with a turnover, which is exactly the situation
for the X-panel low-IP ions. The fit obediently put the knee at 7.9–8.2 eV, i.e., at the
boundary between "ions in turnover transition" and "ions far from it" — a flux artifact,
not a monitoring measurement.

The kill's sharpest edge also rests on the least reliable point: SiII's β = −21.6 has
bootstrap CI95 [−69, +1.8] (consistent with zero), and the X-panel MgII β = −0.43 has
CI95 [−1.44, +0.64] (also consistent with zero). The registered fit weighted point
estimates; the geometry that excluded 27.2 eV is carried by noise-dominated values.

**Registered verdict stands: KILLED.** The registered model — magnitude and sign as
separable — is rejected. The Hartree knee was not refuted as a *location*; the model that
would have measured it was refuted as a *structure*. Honest path forward (P17 v2, not
registered here): a joint β(IP, N) = lag(IP) × flux(IP, N) formulation where the turnover
is inside the magnitude prediction, fit to windowed (not pooled) betas with CIs as weights.

### T4's brittleness (methodology lesson, also honest)

The rolling-window sign analysis flips on single-window noise: CIV's "flips" are isolated
sign alternations in sparse windows (X-panel windows hold ~160 components), and even MgII's
X-panel sequence ['+','−','+','−','−','−'] is only persistently negative from window 4.
The registration scored any flip; a persistence criterion (≥2 consecutive windows on each
side) should have been registered instead. By the registered rule CIV's noise kills the
model; the rule was brittle, and that is the registration's fault, not the data's. Recorded
for the next design.

### What survived

- **MgII turnover at z ≈ 2.5** (point estimate) — inside the predicted 2.0–2.5 window, with
  the SDSS panel monotonically positive below (no flip through z = 2.28 ✓). Suggestive, not
  established (X-panel CI includes zero).
- CaII β ≈ 0 (stable across both experiments now).
- The quadratic envelope beats the logistic alternative (T2) — shape preference, moot under
  the kill but recorded.
- Anchor N_H = 9.03 if the MgII flip is real (reported, unscored).

---

## exp_22: 2/3 — saturation regulates, the horizon is φ², no Zeno regime

| Prediction | Result |
|------------|--------|
| P22.1 saturation ceiling reduces Gini at every c | **PASS** — 0.32–0.80 vs baseline 0.930 |
| P22.2 finite monitoring horizon exists | **PASS** — peak severance only at contrast ≤ 2.62 |
| P22.3 Zeno non-monotonicity | FAIL — monotone (0.941 → 0.938 → 0.930 as monitoring rarefies) |

### The φ² horizon ([D] discovery, with derivation sketch)

Across the full θ sweep, severance reached the global peak only at peak/neighbor stress
contrast ≤ **2.62 ≈ φ² (2.618)**, while the steady-state peak sits at contrast ~34. The
post-hoc derivation is one line: the φ-split deposit gives adjacent nodes the ratio
w_k/w_{k+1} = φ², so a freshly-deposited configuration has peak contrast exactly φ² — the
maximum contrast at which the collective trigger can still see the peak's neighborhood as
co-stressed. Once accumulation drives contrast beyond φ², the peak is radiatively
unreachable, permanently. **The monitoring horizon is contrast > φ², forced by the φ-split
geometry.** This is the quantitative form of the severance/saturation division of labor
found in exp_20, and the cascade-language analog of a collapse criterion.

### Saturation as the second regulator (P22.1)

The MVAE-style ceiling regulates at every setting where severance regulated at none.
Clamp value is monotone in the ceiling; Gini = 1/φ corresponds to ceiling ≈ 2.7–2.8× mean
(φ² gives 0.596, 3.5% below 1/φ) — logged as exploratory, no claim.

### No Zeno regime (P22.3, principled failure)

Monitoring frequency vs concentration is monotone — every severance rate concentrates more
than none (the cooling-flow effect, now shown monotone). The P15 Zeno analogy fails here for
a reason worth keeping: Zeno requires coherent dynamics between measurements; this harness is
classical-dissipative — there is nothing to freeze. **The Zeno mapping belongs to the orbit
(quantum) layer, not the classical cascade layer.** The exp_19 CaII freezing should therefore
be modeled at the orbit layer in P17 v2, not by harness analogy.

---

## Ledger

- exp_21: 2/4, KILLED — registered model structurally rejected; flaw localized
  (magnitude/sign non-separability near turnover); MgII turnover and CaII zero survive
  as data points for P17 v2.
- exp_22: 2/3 — saturation regulator confirmed, **φ² monitoring horizon discovered**,
  Zeno scoped to the quantum layer.
- Registration discipline held: no metric changes, no threshold adjustments post-hoc;
  verdicts exactly per `f4c3dfb1`.
- Running tally of monitored predictions across Midnight: oscillation falsified,
  CaII-EW channel inconclusive, Hartree-knee model killed (location untested),
  severance-regulator killed, saturation-regulator confirmed, φ²-horizon discovered.
  The structure keeps winning; the point predictions keep dying young and informative.

---

## The locality correction (added after review discussion, 2026-06-11)

Nothing is universal — relativity demands it, PAC demands it (conservation is per-ledger),
SEC demands it (collapse is relative to the local gradient; M13 definitional parallax).
Re-reading this round through that principle:

**The exp_22 harness mixed frames.** The stress trigger used value/*global* mean — a
god's-eye reference no node possesses. The φ² contrast survives the critique because it is
a ratio of adjacent stresses (the global normalization cancels); the threshold θ does not.
The decomposition is exact: **the relational quantity came out φ-structured; the
coordinate quantity is arbitrary.** The φ² horizon must be stated as relative to the
severance channel's frame, not as an absolute property of the peak — the same way the GR
horizon is observer-dependent.

**The Hartree knee, restated locally.** Each absorber sees its own reservoir; the knee sits
at that scope's quantum. 27.2 eV is a candidate universal *only because the EM-depth closure
is the same closure in every local scope* — universality as a consequence of identical local
structure (like identical local light cones), never as a global backdrop. This also opens a
second derivation route: **M6 scoped mediation** — coupling β as transfer attenuated per
scope boundary between the absorber's ledger and the cosmic flow. CaII may simply sit more
boundaries away than CIV. Two independent routes (ladder occupancy, boundary attenuation)
to one law is the v2 target.

**The meta-pattern, explained.** Every casualty in the prediction ledger was a *coordinate*
(an absolute value in an implicit global frame: one oscillation phase, one pooled knee, one
global Gini bound). Every survivor is a *relation* (R² orderings, sign flips, the φ² ratio,
zero-cost constraints, CaII-vs-CIV contrast). The data has been enforcing general covariance
on the registration discipline.

**The invariant-registration rule (adopted going forward):** register invariants — ratios,
orderings, contrasts, per-scope statements. Any absolute number enters only as a derived
consequence of a relational claim, never as the claim itself. This is itself a falsifiable
meta-claim: relational registrations should keep surviving, coordinate registrations should
keep dying. Six data points so far; the next round (exp_23 topology/frame test, P17 v2
relaxation model in per-scope form) tests it knowingly.
