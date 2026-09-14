# exp_29 registration — with a PAC ledger, does the particle substrate hold structure? (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R; the long-horizon
plan's R1, `internal/dft/2026-09-06_long_horizon_plan.md`). Instruments — the pair energy, the
budget transfer, the conserved total — are gated in reality-engine (`.spec/v4-pac-ledger.spec.md`,
`tests/v4/test_pac_ledger.py` 7/7, POC-12 exp_01 11/11 at `c5e961e`), not scored here. Scored to
this text by `scripts/exp_29_pac_ledger.py`, thresholds as module constants byte-equal to §4
(`--selftest`). Frame declared in §3 (STANDARDS §2.7.6). Layer named (§2.7.7). Kill scope in §6.

**Thesis under test.** exp_28 (0/4, `bf833113`) showed the v4 substrate cannot hold structure and
its ledger showed why: the SEC pressure's work over a baseline run is 670,000–1,140,000 against a
binding energy of ~63,000, while gravity's net work is ~1,000. In the house's own terms — SEC is
local and free to not conserve step by step; PAC is global and must be exact at a ledger with a
bounded Δ and a reconciliation — the substrate had **no ledger**. The PAC necessity proof says a
system without one cannot hold structure. This experiment adds the ledger and asks whether it holds.

## §0 Postdiction disclosure

Everything computed before this seal, on reality-engine `feat/v4-pac-ledger` (stacked on
`feat/v4-derived-sink`, PR #10) at `c5e961e`–`91fb8af`:

1. **The design pass (proxy B0, seed 1, read-only).** At fixed entropy the pressure is already the
   gradient of the pair energy `V_ij = sec·(S_i+S_j)/2·r0·e^{−r/r0}`; only the cutoff at 2r0
   breaks it. Booked tick by tick, the pair energy **created by entropy change is 689,700 against
   667,300 of pressure work** (ratio 1.03); it peaks at 1.46 × 10⁶ at t ≈ 4.5 and converts to
   kinetic energy as the clump expands. The entropy ratchet is the whole engine.
2. **The budget calibration** (POC-12 exp_02, proxy seed 1, κ ∈ {0.25 … 16}, no structure metric by
   construction): the table in §2. Three things it changed **before** this seal, each with its reason:
   - The plan's T2 second clause, "the window mean of the conserved total is negative", is a
     **tautology**: the total is conserved, so it equals its initial value `(κ − 1)|U₀|` at every
     mark (measured: −43,620 vs −43,780 at κ = 0.25; 57,000 vs 58,370 at κ = 2). Its sign is κ < 1
     by construction. Removed. T2 is now the virial state at every mark of the window.
   - The plan's T3, "KE/|U| monotone increasing in κ", is **false on seed 1 below κ = 1**: the ratio is
     U-shaped, 0.60 / 0.51 / 0.32 / 0.43 / 0.68 / 1.10 / 2.03 at κ = 0.25 … 16, minimum at κ = 1,
     because below it the pressure does net *negative* work (a restoring force that supports the
     clump, sub-virial) and above it the engine injects. T3 is now the ordering **above** the
     threshold, κ = 1 < κ = 2 < κ = ∞, which the arithmetic predicts and seed 1 shows.
   - The plan's T4 second clause, "the budget is exhausted before t = 10", measured the **total**;
     the budget is per particle, the dense set exhausts its share while the total never falls below
     1% (the exhaustion tick is `None` at every κ). The measurable statement is that the budget
     binds on every particle that would grow: `budget_bound_frac_max = 1` at every κ tested. T4 now
     reads that.
3. **The virial prediction held pre-seal on seed 1**: bound through κ = 1 (KE/|U| 0.32–0.60), unbound
   from κ = 2 in the sense of the total, with KE/|U| 2.03 at κ = 16 against 10.7 unbounded. The
   registered expected direction (§5) is therefore *postdictive on seed 1 and predictive on seeds 2–3
   and at n = 4000*.
4. **The pipeline smoke test printed structure numbers** (seed 9, outside the registry, t_end = 6, so
   no window mark; scratchpad): at t = 6, whole-set percolation κ = 0.5: **0.721** (KE/|U| 0.41, 90%
   of the budget spent, pressure work −0.90 P₀); gravity only (κ = 0): **0.433**; today's substrate
   (κ = ∞): **0.244** (KE/|U| 12.7). The random-field floor at the same count and resolution:
   0.051 ± 0.014. Gravity alone builds a percolating web at the first collapse (peak 0.72), so the
   vacuity clause "gravity structures" **will fire**, and the frame is — as pre-declared — the
   difference over G0 as well as over B0, which T1 already demands. No threshold, arm or κ changed
   on the smoke's account; §5 leans on it and says so.
5. **Instrument corrections found by the gates** (not physics): the finite-difference gradient
   check needs float64 (a first run missed at 1.2 × 10⁻³ for precision alone); the anchor against
   exp_28's recorded baseline needs `sec_balance = XI_ANALYTIC/PHI` exactly (the rounded 0.6541
   diverges by 3 × 10⁻³ over a run) — with it the inert path reproduces exp_28's B0 seed-1 marks to
   **0.00** relative.

## §1 Objects (closed at this seal)

- **Substrate:** reality-engine `proof_of_concepts/v4/particles.py`, pipeline `CANONICAL_SINK` with
  severance and Landauer **off**; exp_11's config `n = 4000, box = 60, r0 = 10, g = 1.5, dims = 3,
  ic = lattice`; `sec_balance = XI_ANALYTIC / PHI` (`xi_variant` recorded); `memory_decay = 0.95`
  (inherited, declared, untouched); `damping = 1.0` in every arm; seeds {1, 2, 3}; `t_end = 15`;
  marks every 1.0 of simulated time. Proxy: `n = 1000, box = 37.8` (exp_11's density).
- **The ledger** (`.spec/v4-pac-ledger.spec.md`): the shifted pair energy `K_ij = sec·r0·(e^{−r/r0}
  − e^{−2})` on retained pairs within 2r0, whose gradient at fixed entropy is exactly the pressure;
  a per-particle potential budget `P_i(0) = κ·|U_grav(0)|·m_i/Σm`; entropy growth pays
  `(∂E_SEC/∂S_i)·ΔS_i` and is clipped to what the budget can pay; decay repays; the total
  `KE + U_grav + E_SEC + ΣP` is conserved to the integrator's truncation with the transfer part exact
  to 10⁻⁶. **κ is a declared ratio of the initial binding energy, swept, never fitted.**
- **Arms (κ):** **0** — gravity only, the engine removed (entropy cannot grow; bit-identical to
  `sec_balance = 0`) · **0.5, 1, 2** — the ledgered engine, the sweep · **∞** — today's substrate,
  the unbounded engine (`pac_kappa = None`, bit-identical to exp_28's code). Fifteen runs per size.
- **Instruments:** `structure.web_metrics` (percolation, xi_u, occupancy) of the **whole set** at
  `matched_res(n)`; the ledger metrics; the random-field floor at the same count and resolution
  (20 uniform draws) reported beside every percolation.
- **Derived vs declared vs inherited:** the pair energy is the force's own; the budget's price is
  the pair energy's derivative; κ declared and swept; `sec_balance`, `memory_decay`, `growth 0.1`,
  `g`, `r0` inherited and untouched; the fracton functional's `β∇²A` and `γT·A` terms excluded.

## §2 Pre-seal numbers (proxy, seed 1; POC-12 exp_02 at `c5e961e`)

| κ | P₀ | budget spent at t = 15 | mean-entropy peak | KE/\|U\| at t = 5 / 10 / 15 | conserved total at 15 | (κ−1)\|U₀\| | pressure work / P₀ |
|---|---|---|---|---|---|---|---|
| 0.25 | 14,600 | 93.5% | 0.05 | 0.45 / 0.59 / 0.60 | −43,620 | −43,780 | −1.17 |
| **0.5** | 29,200 | 91.0% | 0.09 | 0.36 / 0.47 / 0.51 | −29,620 | −29,190 | −1.21 |
| **1** | 58,400 | 80.7% | 0.16 | 0.21 / 0.26 / 0.32 | −1,146 | 0 | −0.09 |
| **2** | 116,800 | 37.2% | 0.31 | 0.23 / 0.35 / 0.43 | 57,000 | 58,370 | +0.19 |
| 4 | 233,500 | 22.2% | 0.42 | 0.33 / 0.57 / 0.68 | 173,100 | 175,100 | +0.17 |
| 8 | 467,000 | 15.4% | 0.57 | 0.55 / 0.97 / 1.10 | 405,300 | 408,600 | +0.14 |
| 16 | 934,000 | 14.0% | 0.81 | 0.91 / 1.86 / 2.03 | 869,000 | 875,600 | +0.13 |
| ∞ (exp_28 B0) | — | — | 5.7 | 10.3 / 10.7 / 10.7 | — | — | — |

The budget binds on every would-grow particle at every κ (`budget_bound_frac_max = 1`); the step
never shrank below `dt_ref` at any κ ≤ 16 (300 ticks to t = 15) where the unbounded engine takes
555. Gates: force = −∇E_SEC to 1.6 × 10⁻⁷; transfer identity 9 × 10⁻¹⁰; total-ledger drift 235
against a Courant bound of 1,420, step-halving ratio 1.92.

## §3 Frame (STANDARDS §2.7.6)

**Sampled:** the density field of the **whole set** (no severance in this round) of each arm at
`res = matched_res(n)`, occupancy reported beside every percolation. **Expectation:** the same
instrument on the two controls at the same count — the engine removed (κ = 0) and the engine
unbounded (κ = ∞) — and the random-field floor at the same count and resolution. **Same scope:**
the whole box. **Statistic:** the mean over marks t ∈ {10, …, 15}, never the t = 15 point; for T2,
every mark of the window.

## §4 Tests (M = 4; thresholds fixed; invariants only)

**T1 — holding.** At **κ = 0.5**, **3/3 seeds**: whole-set percolation over the window exceeds
**both** the κ = 0 control and the κ = ∞ control seed by seed, and each mean margin exceeds
**2× the pooled within-seed σ**. Two controls, because a substrate that holds must hold better
than gravity alone *and* better than the engine it bounds.

**T2 — bound.** At κ = 0.5 **and** at κ = 1, 3/3 seeds: **KE/|U_grav| < 1 at every mark** of the
window (today's substrate: 10–18).

**T3 — the engine's ordering above the threshold.** Per seed, 3/3: the window mean of KE/|U_grav| is
ordered **κ = 1 < κ = 2 < κ = ∞**. Predicted from the arithmetic (injected energy ≤ κ|U₀|), shown on
seed 1 pre-seal (0.32 < 0.43 < 10.7), predictive on seeds 2–3 and at n = 4000.

**T4 — the ledger did the work.** On every ledgered run: net pair energy created by entropy change
≤ P(0) exactly, and pressure work ≤ P(0) within the integrator's **10% truncation allowance**;
and at κ = 0.5, 3/3 seeds, the budget binds on every particle that would grow,
**budget_bound_frac_max = 1**.

A test passes only if it passes at every size present: the proxy decides, n = 4000 confirms.

**Kill.** *If T1 fails at κ = 0.5 — a substrate whose entropy engine is bounded at half its binding
energy still holds no more structure than gravity alone or than the unbounded engine — then
PAC-on-particles as mapped here is the wrong object. The mapping dies; the theory does not (the
necessity proof is about cascades, and the particle reading of it is what this round tests).*

## §5 Expected direction, stated honestly

**T2 pass, T3 pass, T4 pass** — each shown on seed 1 pre-seal (§2), so postdictive there and
predictive on seeds 2–3 and at full size. **T1 open, leaning pass** on the smoke (seed 9, t = 6:
0.72 against 0.43 and 0.24), with the risk stated: both the ledgered arm and gravity alone may
re-collapse into a clump by t = 10, percolation falling in both, and the difference fall inside the
within-seed spread — then T1 fails while T2–T4 pass, and the bearing is that a bounded engine
gives a *bound* system without a *held web*, which sends the next round to the pressure's functional
form (the `β∇²A` smoothing term), not to the budget.

## §6 Kill scope

The kill retires the particle mapping of P and A used here — a per-particle budget priced by the
pair energy — not PAC, not Milestone R's thesis, and not the ledger instruments, which are gated
and stay. The spec's own falsification (transfer residual > 10⁻⁶, or the total drifting past the
Courant bound with the guard unbound) invalidates the *run*, not the claim.

## §7 What would count as vacuous

The budget never binds at any κ ≤ 2 (`budget_bound_frac_max < 0.01`) — pre-seal it binds fully, so
this is not expected; gravity alone already exceeding the random-field floor by more than 2σ, which
**is** expected (§0.4) and moves nothing: the frame is the difference over G0 and it is already
in T1.

## §8 Side predictions (registered, unscored)

SP1 the pressure is a restoring force below the threshold: pressure work < 0 at κ = 0.5 in 3/3
seeds (seed 1: −1.21 P₀). SP2 spent fraction decreases with κ (93 → 14% on seed 1). SP3 the step
never shrinks below `dt_ref` at κ ≤ 2 (no detonation to resolve). SP4 the random-field floor is
below every arm's window mean at both sizes.

## §9 Excluded

Any Ξ-based amount; the functional's `β∇²A` and `γT·A` terms; severance and Landauer (R2);
`memory_decay` as a variable.

## §10 Outputs

reality-engine POC-12 `results/proxy/` and `results/full/` (one JSON + positions sidecar per run,
grid JSON with SHA256s and commit). Here: `results/exp_29_pac_ledger_grid_<size>_<ts>.json`
(copies), `results/exp_29_pac_ledger_<ts>.json` (scored), `journals/2026-09-0X_exp29_outcomes.md`
citing this seal's hash and the reality-engine hashes.
