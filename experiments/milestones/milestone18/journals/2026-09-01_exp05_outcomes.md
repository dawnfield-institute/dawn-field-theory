# 2026-09-01: exp_05 outcomes — 3/4; the sealed prediction fails on a clause T1 forbids

Registration: Phase 2 seal 06073227. Scored prediction: T2 only. T1/T3/T4 are (S).

| Test | Result |
|---|---|
| T1 (S) | For a rational probe, copy occupancy ‖Pv‖² and ‖(I−P)v‖² are Galois conjugates on A₄, D₆, E₈ — exact. A rational observer cannot label the copies. PASS |
| **T2 (prediction)** | Golden-probe leakage is **first-order** in \|s−1\| (log-log slopes 0.998 / 1.000 / 1.000) with **nonzero** exact coefficient ‖(I−P)BP‖² = 2/5 (A₄), 4/9 (D₆, gauge τ = 6−3√5), 2/5 (E₈). Both physics clauses hold. The sealed text also said "fails if the coefficient is rational" — **it is rational on all three. FAIL as sealed.** |
| T3 (S) | σ-copy noise share = tr(P·Cov)/tr(Cov): 0.3509 / 0.4155 / 0.4361 exact vs Monte Carlo within 0.001. PASS |
| T4 (S) | Rational probe under rational polynomial dynamics: copy retention and conjugate leakage are Galois conjugates at s ∈ {½, 1, 3/2}. PASS |

## Scoring note (transparent)

The first run's pass logic omitted the sealed rationality clause and reported 4/4; the
script was corrected to the sealed text and rerun — 3/4. No threshold was relaxed.

## Why the failed clause was unwinnable

‖(I−P)BP‖²_F = ‖PB(I−P)‖²_F by transposition, so the quantity equals its own σ-conjugate
and is therefore rational. The registration demanded a golden coefficient from a
σ-symmetric quantity — a clause that T1's theorem already excluded. The registration
error is recorded; the physics content of T2 (leakage exists, is first-order, is
nonzero) stands and can be re-registered forward as a postdiction-labeled claim.

Observation recorded, not interpreted: the two cyclotomically pure diagrams (A₄, E₈)
share the coefficient 2/5 exactly; the ramified D₆ gives 4/9 in the exp_06 gauge and is
gauge-dependent by construction.

Block B closes at 3/5 (exp_04 0/1, exp_05 3/4). Milestone 25/29 pending the census.
