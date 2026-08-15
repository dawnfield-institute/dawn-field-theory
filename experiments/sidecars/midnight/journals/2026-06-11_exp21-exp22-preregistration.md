# Pre-Registration: exp_21 (Coupling Law, Decoherence-Framed) and exp_22 (Regulation as Monitoring)

**Date:** 2026-06-11
**Status:** REGISTERED BEFORE EXECUTION — registered quantities not computed as of this commit
(smoke tests of loaders/harness only). Derivation basis: `2026-06-11_p17-coupling-law-derivation.md`
(same commit).
**Experiments:** `scripts/exp_21_coupling_decoherence.py`, `scripts/exp_22_monitoring_regulators.py`

---

## exp_21: The Coupling Law (P17.1–P17.4)

Within-survey panels ONLY (the exp_19 lesson). Panel S: CaII (Sardane), MgII (DR16),
FeII (DR16), CIV (DR12). Panel X: MgII, FeII, SiII, SiIV, CIV (XQR-30). Metric per
(ion, panel): exp_19's locked β (binned-median EW vs N(z), normalized slope, 1000-draw
bootstrap), reused verbatim by import.

**Locked model (from P17 §3, parameter-free shape):**
  monitoring capacity P(IP) = 1 − (IP/E_H)²/φ² for IP ≤ E_H, else 0
  |β|(IP) = A_panel / (1 + r·P(IP)),  free: A_S, A_X, r;  E_H fixed = 27.2 eV (or free for T1)

**Registered tests:**
- **T1 (knee at the Hartree, P17.1):** fit with E_H free vs E_H = 27.2 fixed. PASS if
  CI95(E_H free) contains 27.2 AND ΔBIC(fixed − free) ≤ 2 ("the predicted knee costs nothing").
- **T2 (envelope shape, P17.2):** quadratic-envelope capacity vs a logistic capacity with the
  same parameter count, both with E_H = 27.2. PASS if BIC(envelope) ≤ BIC(logistic) + 2.
- **T3 (epoch invariance, P17.3):** E_H fit per panel separately; PASS if the two CI95s overlap.
  Report both centers' distance from the HeI alternative (24.6 eV).
- **T4 (turnover structure, P17.4):** rolling z-window sign analysis of β for each ion in each
  survey. The MgII flip location ANCHORS N_H (reported, not scored — avoids circularity).
  - T4a: FeII flips within Δz ≤ 0.5 of MgII's flip.
  - T4b: CIV shows NO flip in either survey (monotone).
  - T4c: SiIV shows no flip in XQR-30.

**Registered verdict rule:** SUPPORTED if T1, T2, T4a, T4b all pass. KILLED if (a) the free-knee
CI95 excludes 27.2 with an otherwise adequate fit, or (b) CIV flips sign (the model forbids it
unconditionally). Otherwise INCONCLUSIVE. Registered limits: 9 (ion, panel) points, 3–4
parameters — thin; SiII/SiIV CIs wide; INCONCLUSIVE is a live outcome.

## exp_22: Regulation as Monitoring (exp_20 follow-up)

Harness = exp_20's, imported verbatim (chain N=32, argmax entry, φ-split deposit, 200
generations × 50 trials). Baseline continuity gate: condition A must reproduce
Gini_A = 0.930 ± 0.005 or the run is void.

**Registered conditions and predictions:**
- **D (saturation ceiling, M11 analog):** node values capped at c·mean; overflow relaxes to
  neighbors (internally conserved, nothing leaves). Sweep c ∈ {1.5, 2, φ², 3, 5}.
  **P22.1:** Gini_steady(D) < Gini_A for every c — saturation regulates where severance could
  not. Clamp values reported; any φ-structure is exploratory [D], not scored.
- **E (monitoring horizon):** during stress-severance runs (θ sweep as exp_20), log whether
  severance events ever involve the global peak, and the peak/neighbor stress contrast.
  **P22.2:** a finite critical contrast exists above which the peak never severs
  (peak-severance probability → 0). The value is measured, not predicted [D].
- **F (Zeno sweep, P15 analog):** stress severance applied every m-th generation,
  m ∈ {1, 2, 5, 10, 20, 50} plus A (never), θ = 1.6.
  **P22.3:** Gini_steady(m) is non-monotonic in monitoring frequency, with an interior
  extremum (the anti-Zeno analog; exp_20 already showed the m=1 edge exceeding A).
  Direction/shape beyond non-monotonicity is exploratory [D].
- Conservation invariant to machine precision in all conditions (internal for D).

**Registered verdict rule:** exp_22 is characterization around one scored prediction each:
P22.1 (PASS/FAIL), P22.2 (exists/does not), P22.3 (non-monotonic/monotonic).

## Outcome commitment

All outcomes journaled and integrated whichever way they land. P17's predictions are only
cited in Paper 12 if they survive these tests.
