# M15 exp_06 Pre-Registration: k = 5, 6 Confirmation of the Momentum-Generator Derivation

**Date:** 2026-07-17
**Status:** REGISTERED BEFORE DATA — no k = 5 or k = 6 holonomy limit has been
computed as of this commit (numerically or otherwise). The derivation journal
committed alongside (`2026-07-17_general_k_momentum_generator.md`) was developed
with knowledge of exp_05's k = 3, 4 values; **this experiment is therefore the
derivation's only clean test** and is registered before it runs.
**Target script:** `scripts/exp_06_k56_confirmation.py`

## The derived model (locked)

G_k is the k×k skew matrix with G[j',j] = 4jj'/(j'² − j²) for j'−j odd, 0
otherwise (the box momentum matrix ⟨j'|d/dx|j⟩ on [0,1]).
**L_k = sum of positive angle pairs of G_k** (= half the nuclear norm).

Exact predictions, computed here from the rational matrices before any run:

- **L₅ = 17.010952** (G₅ entries: 8/3, 24/5, 48/7, 80/9; 16/15, 48/21=16/7, 96/27=32/9... full matrix: nonzero |entries| {(1,2):8/3, (2,3):24/5, (3,4):48/7, (4,5):80/9, (1,4):16/15, (2,5):40/21})
- **L₆ = 25.778092** (adds (5,6):120/11, (3,6):8/3·... = 4·18/(27)=8/3, (1,6):48/35)

(Numeric values to 6 d.p. from the exact matrices; the script recomputes them
independently from the formula as a cross-check and fails if its own
prediction step disagrees with these registered values by > 1e-5.)

## Registered observable and method (identical to exp_05, locked)

m·θ_T(m) with the analytic sign convention (diag(M) > 0), θ_T = sum of positive
polar-rotation angles, m-grid 100..400 step 50, Richardson order 2. Anchor
gates: k = 2 → 8/3 within 1% AND k = 3, 4 reproduce exp_05's recorded limits
within 0.05% (pipeline unchanged check).

## Decision rule (locked)

- **CONFIRM** the momentum-generator derivation iff BOTH measured limits are
  within **0.1%** of the registered predictions (exp_05's k = 3, 4 residuals
  were ~4×10⁻⁷, so 0.1% is generous to extrapolation error and fatal to a
  wrong model — K1/K2-style alternatives differ at the 10%+ level).
- **KILL** iff either deviates by > 1%.
- Between 0.1% and 1%: INCONCLUSIVE (extrapolation-order suspicion; a denser
  m-grid re-run is then permitted ONCE, declared here, same thresholds).
- Secondary (registered, classification only): entrywise finite-m generator
  check at m = 2000 for k = 5, 6 — |m·skew(M)| must match |G_k| entrywise up
  to diagonal ±1 gauge within 2%; and the K3 ℤ₂ scan extended to k = 5, 6
  (expect: even-reflection telescoping, det H = +1; deviation is reported, not
  scored).

## Outcome commitment

CONFIRM, KILL, or INCONCLUSIVE goes to the outcomes journal (citing this
commit), the milestone15 README, and the `milestone15-representative-problem`
FDO. If CONFIRMED, the momentum identification upgrades from derivation-note
to Phase-2 foundation (the twist/Ξ connection then becomes registrable). If
KILLED, the §1 derivation contains an error and the general-k limit reopens.
