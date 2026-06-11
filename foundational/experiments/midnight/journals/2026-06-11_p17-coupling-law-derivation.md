# P17: The Coupling Law from Orbit Decoherence and Ladder Flux

**Date:** 2026-06-11
**Status:** Derivation note — defines the registered forms for exp_21/exp_22
**Machinery:** P16 (Γ² = λ²·Var(H_env), exact), P15 (Zeno/anti-Zeno), M-R exp_24 (E_H = α²mₑc²), exp_17 (φ-split)

---

## 1. The mapping being formalized

| M14/P15/P16 object | Midnight observable |
|---|---|
| System–environment coupling λ | Equilibration rate of an ion species with its local reservoir |
| Pointer state (einselected) | Settled phase: statistics frozen (CaII, β = 0 measured in exp_19) |
| Zeno freezing (P15: P_survive → 1 at high monitoring rate) | Cheap-transaction species pinned to local equilibrium |
| Un-monitored unitary evolution | Expensive-transaction species lagging the reservoir, carrying epoch memory |
| Γ² = λ²·Var(H_env) | Epoch-coupling β as the lag of a species behind the cascade flow |

**The freezing taxonomy, corrected.** The naive reading (low-IP species decouple because the
environment "can't pay") points the wrong way: it would make CIV (47.9 eV) the most decoupled,
and CIV couples hardest. The correct reading is the **Zeno mode**: a species whose
formation/destruction transactions are *cheap* relative to the reservoir quantum is monitored
at enormous rate — it is pinned to instantaneous local equilibrium, and local equilibrium does
not know what epoch it is. Its statistics are frozen (CaII). A species whose transactions are
*expensive* is rarely monitored — it cannot re-equilibrate within a cascade tick, it lags, and
the lag tracks the clock. **β measures the lag, and the lag is the un-decohered remainder.**
Frozen-by-overmonitoring (Zeno) is the settled phase; evolving-by-undermonitoring is the
active phase. This inverts the draft taxonomy in the pre-plan discussion and is the version
the math below supports.

## 2. The reservoir: φ-split ladder (exact arithmetic)

The EM-depth reservoir quantum is E_H = α²mₑc² = 27.20 eV (one Hartree; M-R exp_24, the same
machinery that produced the Rydberg at 11.4 ppm — severance costs the full Coulomb energy,
binding holds half).

Reservoir rungs descend geometrically: E_k = E_H·φ⁻ᵏ, k = 0, 1, 2, …
PAC allocates occupancy by the φ-split (exp_17 mechanism): rung k receives share

  w_k = (1/φ)·(1 − 1/φ)ᵏ = φ⁻⁽²ᵏ⁺¹⁾   [using 1 − 1/φ = 1/φ², an identity]

Check: Σ w_k = φ⁻¹/(1 − φ⁻²) = φ⁻¹/φ⁻¹ = 1. The φ-split ladder is exactly normalized with
weight ratio 1/φ² per rung — no parameters.

## 3. Magnitude: the monitoring capacity and the knee (parameter-free shape)

A reservoir mode can drive a species' transaction (ionization channel at cost IP) iff
E_k ≥ IP. The monitoring capacity is the occupancy above the cost:

  P(IP) = Σ_{k: E_k ≥ IP} w_k = 1 − φ⁻²⁽ᵏ*⁺¹⁾,  k* = ⌊ln(E_H/IP)/ln φ⌋  (IP ≤ E_H)
  P(IP) = 0  for IP > E_H.

Continuum envelope (exact at rung boundaries):

  **P(IP) = 1 − (1/φ²)·(IP/E_H)²  for IP ≤ E_H;  P = 0 above.**

Two parameter-free features:
- **The knee is at E_H exactly**: capacity hits zero at IP = E_H — above one Hartree the
  reservoir cannot monitor at all and the species' lag saturates.
- **P(E_H) approached from below = 1/φ**: at the quantum itself the residual capacity is
  1 − 1/φ² = 1/φ. The φ-bound appears, unbidden, as the monitoring capacity at the knee.

Lag (linear response): with cascade drift rate D and equilibration rate R(IP) = R₀·P(IP),

  **|β(IP)| = β_sat · D/(D + R₀·P(IP))**

— saturating at β_sat for IP ≥ E_H (CIV, SiIV regimes), suppressed toward small IP
(CaII: P → 1, maximal monitoring, Zeno-frozen, β → β_sat·D/(D+R₀) ≈ 0 for R₀ ≫ D).
Free parameters: β_sat (amplitude, per panel) and D/R₀ (one shape parameter).
**E_H is not free.** The registered claim is the knee location, not a sigmoid width.

*Downgrade per plan:* the earlier "sigmoid with width ln φ" is NOT what the ladder forces;
the forced form is the quadratic envelope above with its hard knee. The ln φ spacing
survives only as discrete steps at IP = E_H·φ⁻ᵏ (a stretch signature, not registered).

## 4. Sign and turnover: ladder flux (one anchor parameter — declared)

The reservoir's occupancy tilt evolves with the clock: each completed cascade level shifts
characteristic weight down one rung (factor 1/φ per level — the cascade's native step). Define
the characteristic monitoring energy at clock reading N:

  E_char(N) = E_H·φ^(N − N_H)

with **one anchor parameter N_H** (the clock reading at which the characteristic energy passes
the Hartree). This is an identification, declared as such [B].

A species at cost IP is a net *beneficiary* of ladder flux while E_char > IP (its feedstock —
the stage above — is still being processed down to it) and a net *donor* after E_char < IP.
Its abundance/EW therefore peaks at the epoch where E_char(N) = IP:

  **N_turn(IP) = N_H − ln(E_H/IP)/ln φ**

Consequences (with rungs measured from E_H; positive = below):
- MgII (7.65 eV): 2.64 rungs below E_H. FeII (7.87): 2.58. SiII (8.15): 2.51.
  **The three lie within 0.13 rungs — they must turn over at nearly the same epoch.**
- CIV (47.89 eV): 1.18 rungs *above* E_H → turnover at N_H + 1.18, outside the reservoir's
  range — **CIV must be monotone over the entire observable range.** SiIV (33.49): 0.43 above —
  also monotone (or turning only at the extreme early edge).
- exp_19's "non-portability" is partly physics: SDSS MgII β = +0.22 (z < 2.28, below
  turnover: growing toward it) and XQR-30 MgII β = −0.43 (z > 2.04, above turnover: declining)
  bracket a sign flip at z ≈ 2.0–2.5. Anchoring N_H on that flip (N(z≈2.3) ≈ 6.27 →
  N_H ≈ 8.9) then *predicts* FeII and SiII flips within Δz ≲ 0.3 of MgII's, and CIV/SiIV
  monotone — all checkable in the same catalogs. Independently, strong-MgII incidence is
  known to peak near z ≈ 2–3 (literature postdiction, not used in the fit).

## 5. Registered predictions (P17.1–P17.5)

| # | Prediction | Status |
|---|-----------|--------|
| P17.1 | \|β\|(IP) saturates with knee at E_H = 27.2 eV; fixing the knee there costs no fit quality vs a free knee | [A→test] parameter-free location |
| P17.2 | Sub-knee shape is the quadratic envelope 1 − (IP/E_H)²/φ² (capacity 1/φ at the knee) | [A→test] parameter-free shape |
| P17.3 | The knee is epoch-invariant (atomic constant); the UV-background alternative pins to HeI 24.6 eV and drifts with background hardness | [P] discriminator |
| P17.4 | MgII, FeII, SiII turn over (β sign flip) within 0.13 rungs ≈ Δz ≲ 0.3 of each other near z ≈ 2.0–2.5; CIV and SiIV monotone everywhere observed | [B] one anchor (N_H) |
| P17.5 | Settledness is processing rate, not temperature: cold *active* gas (molecular/star-forming tracers) couples to the clock; cold *quiescent* gas (CaII) does not | [P] |

Honest accounting: P17.1/P17.2 are forced by the φ-split ladder with zero shape freedom;
P17.4 spends exactly one identification (N_H); the linear-response lag form in §3 is a
modeling choice (declared); the reservoir-tilt rule (one rung per level) is the cascade's
native step but is itself an assumption the turnover test probes.

## 6. What exp_21/exp_22 must do

- exp_21 tests P17.1 (knee location, zero-cost vs free knee), P17.2 (envelope shape),
  P17.3 (per-panel knee consistency), P17.4 (turnover structure: MgII/FeII flip proximity,
  CIV/SiIV monotonicity) — within-survey panels only.
- exp_22 reframes regulation as monitoring: saturation ceiling (the regulator for
  un-monitorable peaks), the critical stress contrast at which severance's collective
  trigger fails (the monitoring horizon), and the Zeno sweep (P15's non-monotonic
  frequency dependence — note exp_20 already showed the anti-Zeno edge: severance at
  every generation *raised* Gini).
- P17.5 is beyond current catalogs (needs molecular tracers); registered for the future.
