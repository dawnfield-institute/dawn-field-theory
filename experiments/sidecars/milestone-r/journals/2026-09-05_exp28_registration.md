# exp_28 registration — does a derived severance channel let the particle substrate hold structure? (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R). The instruments —
the energy ledger and the severance bookkeeping — are validated as gates in reality-engine
(`tests/v4`, POC-11 exp_01, 13/13 PASS at `18fa2b9`), not scored here. Scored to this text by
`scripts/exp_28_dynamical_severance.py`, whose thresholds are module constants byte-equal to §4.
Frame declared in §3 (STANDARDS §2.7.6). Layer named (§2.7.7). Kills have the scopes in §6.

**Thesis under test.** Milestone R says radiation is ledger severance. Its 27 experiments establish a
derived **trigger** (exp_15/16: a vertex severs when *all* its bonds are simultaneously overstressed)
and a derived **form** (exp_01: severance is decoupling, not destruction — the severed part leaves with
its own value and never interacts again), and no amount. This is the first time either is run as a
*dynamical* step. The substrate is reality-engine's v4 particle substrate, which since 2026-09-05
integrates its force law, conserves momentum (the inherited pressure rule was self-propulsion —
reality-engine `d5fe671`) and carries an energy ledger. With no sink it is a relaxation oscillator
that forms structure at its first collapse and dissolves it.

## §0 Postdiction disclosure

Computed before this seal, on reality-engine `feat/v4-derived-sink` at `18fa2b9`–`aef135e` (the
corrected force; every number below is post-fix; the artifacts are committed at `aef135e`):

1. **The B0 signature on the proxy** (n = 1000, one seed, no sink, damping 1.0): retained-set
   percolation 0.15 → **0.37 (t ≈ 4)** → 0.17 → 0.06–0.12 over t ∈ [10, 15]; KE/|U| 0.07 (t = 3) →
   10.7 flat from t = 5; entropy → 0 by t = 10; the step recovers (dt 0.023, 555 ticks, 6 s). B0's
   "cannot hold" — structure forms and dissolves — is therefore **postdictive at proxy scale and
   predictive at n = 4000.**
2. **The trigger calibration WITH removal** (POC-11 exp_02, proxy, τ ∈ {0.5 … 50}): the table in §2.
   No structure metric of any sink arm has been computed (exp_02 does not import `structure.py`).
3. **The τ set was adjusted by the pre-declared clause.** The plan named {1, 5, 20} and said the set
   shifts pre-seal only if τ = 1 cascades above 90%. It severs 91.9% by t = 15 and leaves 131
   particles at t = 10; τ = 0.5 and 2 are no better. The registered set is **{5, 10, 20}**, fixed
   from firing rates alone.
4. **A pipeline smoke test showed sink-arm structure numbers before this seal.** Testing the
   runner and aggregator end to end (seed 9 — outside the registered seeds — to t = 12, so a window
   of two marks) exposed the D-definition bug above and, unavoidably, printed: S(τ = 10) retained-set
   percolation 0.072 against 0.088 for B0 at matched count and 0.099 for R; S(τ = 20) 0.112 against
   0.114. One seed, a truncated window, and a τ set already fixed: no threshold, arm, or τ changed on
   its account. It is the reason §5 now leans null on T2 and fail on T4.
5. **Two dependency findings disclosed, not used.** `asymmetric_conservation`'s Δ buffer is
   identically zero in its own code (every event it can emit is a pure transfer), so its
   reconciliation never fires and two of its "5/5" falsification tests pass vacuously — a
   correction candidate for a later round. And the design pass's original expected-direction
   numbers (KE/|U| ≈ 960, "decided in 30 ticks") were computed on the self-propelling pressure and
   are superseded by item 1.

## §1 Objects (closed at this seal)

- **Substrate:** reality-engine `proof_of_concepts/v4/particles.py`, pipeline `CANONICAL_SINK`;
  exp_11's config `n = 4000, box = 60, r0 = 10, g = 1.5, dims = 3, ic = lattice`; `sec_balance =
  XI_ANALYTIC / PHI` (recorded as `xi_variant`); `memory_decay = 0.95` — **the one inherited tuned
  rate**, declared; **`damping = 1.0` in every derived arm**; seeds {1, 2, 3}; `t_end = 15`; marks
  every 1.0 of simulated time. Proxy: `n = 1000, box = 37.8` (exp_11's density).
- **Severance** (`LedgerSeverance`): whole-particle decoupling — the fired particle leaves the
  interacting ledger with its mass, kinetic energy, momentum and its whole interaction energy with
  the retained set, and thereafter interacts with nothing. Trigger: `min over neighbours within
  sev_radius of |S_i − S_j| > τ` with at least one neighbour (M-R exp_15's rule on the per-edge
  gradient of the SEC entropy). **Radius = the lattice spacing** `box / ceil(n^(1/3))` (3.75 at
  n = 4000, 3.78 at n = 1000): the degree regime (1–13 neighbours) in which the derived
  all-edges barrier is live; within r0 the degree is ~80 and only outliers fire. Declared. **τ ∈
  {5, 10, 20}** entropy units, swept; a declared free scale parameter (M-R exp_16 T3: the law's form
  is τ-independent).
- **Arms.** B0 no sink · S(τ) severance · **D** matched-energy drag: damping chosen so that the
  **retained kinetic energy per particle at t = 15** matches S's (positive-definite; the signed total
  crosses zero as the set unbinds, which is why the design pass's ratio of totals was replaced before
  this seal): `ρ_K = (KE/N)_S(15) / (KE/N)_B0(15)`, `damping_D = ρ_K^(dt_ref / 30)`, one secant
  refinement if the achieved ratio falls outside [0.8, 1.25], both attempts recorded — *defined by
  matching, never a registered coordinate*; if ρ_K ≥ 1, D ≡ B0 and T3 is uninformative · **R** matched-count random
  severance: S's event log replayed by simulated time · **L** Landauer erasure only and **S+L**:
  exploratory, recorded, unscored (the form is derived — `LN2` per unit of released entropy — but the
  magnitude inherits `memory_decay`).
- **Grid.** Proxy: B0 × 3, S 3τ × 3, R 3τ × 3, D × 3 at τ*, L × 3, S+L × 3 at τ = 10, and S(τ = 10) at
  `memory_decay ∈ {0.90, 0.98}` × 3 (SP2). Full n = 4000: B0, S, D, R × 3 seeds at **τ\*** — the τ
  with the largest T2 margin on the proxy, or **τ = 10 if the proxy is null at every τ**
  (pre-declared selection step). Runs take 5–9 s on the proxy; full-size cost is measured, not budgeted.
- **Instruments.** `structure.web_metrics` (percolation, xi_u, occupancy) at `matched_res(n_ret)`;
  the ledger of `.spec/v4-derived-sink.spec.md` (`e_int`, `closure_residual`, losses, bounds).
- **Derived vs declared vs inherited:** trigger form and severance form derived; τ swept; radius
  declared; `LN2` form derived with inherited magnitude; drag tuned and confined to control D.

## §2 Pre-seal numbers (proxy, no drag, corrected force; POC-11 exp_02 at `18fa2b9`)

| τ | severed by t = 15 | retained at t = 10 | first firing t | KE/\|U\| at first firing | fired/retained KE |
|---|---|---|---|---|---|
| 0.5 | 0.949 | 85 | 2.80 | 0.05 | 0.53 |
| 1 | 0.919 | 131 | 2.80 | 0.05 | 0.53 |
| 2 | 0.835 | 266 | 2.80 | 0.05 | 0.53 |
| **5** | **0.744** | **292** | 2.85 | 0.06 | 0.46 |
| **10** | **0.416** | **584** | 2.90 | 0.06 | 0.50 |
| **20** | **0.061** | **939** | 3.15 | 0.12 | 0.46 |
| 50 | 0.000 | 1000 | — | — | — |

Severance intercepts the collapse at onset at every τ ≤ 20 (KE/\|U\| 0.05–0.12), and what fires first
is the **bound cores** (KE ratio ≈ 0.5). B0 (§0): percolation peak 0.37 at t ≈ 4, window mean over
[10, 15] ≈ 0.09 (one seed); KE/\|U\| ≈ 10.7 in the window; at_cap max 0.010; no floor ticks.

## §3 Frame (STANDARDS §2.7.6)

**Sampled:** the density field of the **retained** set of an S run, binned at `res = matched_res(n_ret)`
(~1 particle per cell), occupancy reported beside every percolation value. **Expectation:** the same
instrument on B0 and D at the **same count** — a uniform random subset of n_ret of their particles,
5 draws, mean, at each mark — and on R, whose n_ret matches by construction. **Same scope:** the whole
box; severed particles are excluded from every field (they are the radiation). **Statistic:** the mean
over marks t ∈ {10, …, 15}, never the t = 15 point (exp_04's outlier lesson).

## §4 Tests (M = 4; thresholds fixed; invariants only)

**T1 — the sink is real.** For **≥ 2 of 3 τ**, 3/3 seeds: `e_int^S(t) < e_int^B0(t)` at every mark
t ∈ [5, 15], and `Σ loss_severance_energy > 0`. *Informative if* the cumulative severed fraction
∈ [0.01, 0.9]. *Fail is informative and is the live direction:* the trigger fires first on bound
cores, whose interaction energy `u_out` is negative, so severing them can **raise** the retained
per-particle energy.

**T2 — holding.** For **≥ 2 of 3 τ** on the proxy, and at τ\* at n = 4000: `percolation_S >
percolation_B0` seed-by-seed 3/3 (B0 at matched count, §3), **and** the mean difference exceeds
**2× the pooled within-seed standard deviation**. `xi_u` against the 1 − 1/e floor is reported, not
scored. *Null:* B0 ≈ 0.09 (§2). *Vacuous at a τ if* n_ret at t = 10 is below **512** (matched_res < 8)
or the severed fraction is below 0.01. On the proxy τ = 5 leaves 292 at t = 10 and is therefore
expected vacuous *there* while informative at n = 4000 (≈ 1170); this is stated now, not after.

**T3 — the derivation does work** (conditional on T2 at τ\*): `percolation_S > percolation_D`
3/3 seeds, > 2× pooled std. If T2 fails, T3 is UNINFORMATIVE, not passed. **K1**, if T2 passes and T3
fails: *any dissipation would do — the trigger is not doing work; the radiation reading of this
sink dies; the engine keeps a sink.*

**T4 — selection matters** (conditional on T2 at τ\*): `percolation_S > percolation_R` 3/3 seeds,
> 2× pooled std. `sev_ke_ratio` is reported (R ≡ 1 by construction). **K2**, if T2 passes and T4
fails: *removing particles, not overstress selection, holds structure.*

## §5 Expected direction, stated honestly

B0 cannot hold (postdictive on the proxy; predictive at n = 4000). **T1: open, leaning fail** at
τ = 5 and 10 (bound cores leave first; `u_out < 0`), open at τ = 20. **T2: leaning null.** The design
pass's prior of "null at coarse τ" was computed on the self-propelling force and is withdrawn; on the
corrected force the retained set is unbound only ~10:1 and severance intercepts at onset, which left
holding open — until the seed-9 smoke test (§0.4) showed the overstress-selected set no more
connected than a random subset of B0, and less connected than random removal. **T3:** conditional;
if T2 passes, expected pass (drag never re-collapsed anything in exp_04). **T4: leaning fail if
reached** — the trigger fires first on bound cores, which is the structure; random removal spares
them. A T2 null at every τ is a **bearing**: the sink is real (T1) and the substrate still cannot
hold ⇒ C4.1 option 3 closes and option 2 — deriving the pressure term from the SEC functional — is
what remains; a K2 would sharpen it: *the derived trigger selects against structure*.

## §6 Kill scope

K1 kills the radiation reading of *this sink on this substrate*, not Milestone R's thesis (exp_15/16
are graph results and stand). K2 kills the selection claim only. Neither touches the ledger, the
severance bookkeeping, or the pressure fix, which are gated instruments. The spec's own falsification
(`closure_residual > 1e-5` with the guard unbound) would invalidate the *run*, not the claim.

## §7 What would count as vacuous

Trigger dead at every τ; severed fraction above 0.9 at every τ; n_ret below 512 (proxy) / 500
(n = 4000) at t = 10 at every τ; D coinciding with B0 (ρ_K ≥ 1) makes T3 uninformative.

## §8 Side predictions (registered, unscored)

SP1 cascade: at fixed τ the severed fraction with removal exceeds the no-removal estimate (ordering).
SP2 `memory_decay`: severed fraction at τ = 10 ordered 0.90 < 0.95 < 0.98, and T2's sign unchanged
across them (the entropy scale ∝ 1/(1 − md); an invariant, not a coordinate). SP3 Landauer:
`Σ loss_landauer / Σ work_pressure < 1e-2` in L. SP4 momentum closure (a gate).

## §9 Excluded

"With a sink the substrate sits below Ξ" — not measurable without a P/A mapping; not registered.

## §10 Outputs

reality-engine: one JSON per run + `exp_03_sink_arms_grid_<ts>.json` (hashes, commit), POC-11
`results/`, append-only. Here: `results/exp_28_dynamical_severance_grid_<ts>.json` (a copy),
`results/exp_28_dynamical_severance_<ts>.json` (scored), `journals/2026-09-0X_exp28_outcomes.md`
citing this seal's hash and the reality-engine hashes.
