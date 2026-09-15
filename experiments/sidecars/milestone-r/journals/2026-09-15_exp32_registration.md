# exp_32 registration — the edge as a number: where the ledgered engine's pressure stops being paid for by collapse, and what that number cannot depend on (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R). Instruments: the PAC
ledger (unchanged since exp_29), the connectivity instrument (exp_31, `5f5d690`), and the edge
instrumentation of reality-engine `feat/v4-ledger-virial` (`a2dde18`: per-particle work by force,
pair-form virial terms, gross ledger legs; `c83ab8c`: coupling overrides; `2b8c1d0`: the median
work per mark, the reservoir at the last mark, the n = 8000 size) — no physics change (spec R4, R8),
every added quantity an identity with a test (v4 suite 45). Scored to this text by
`scripts/exp_32_edge.py`, thresholds byte-equal to §4 (`--selftest`). Frame §3. Kill scope §6.

**Thesis under test.** exp_30 and exp_31 scored, on twelve seeds, that the bounded pressure's net
work over a run is negative at κ = 1 and positive at κ = 1.25, and that the web's spine is gone on
the positive side. The scoping and derivation notes of 2026-09-14 (`internal/dft/2026-09-14_edge_*`)
put that edge in the ledger's own arithmetic: W_p = T − ΔE_SEC exactly, so the edge is where the
budget spent equals the pair-energy reservoir's rise — where the entropy-weighted compression ratio
⟨K_end/K_buy⟩ crosses one — and a scaling argument says the crossing's κ cannot depend on the
couplings g and sec_balance, because the budget is defined relative to gravity and the price
relative to the pair kernel, leaving only the entropy clock as a leak. This round registers the
edge as a number with an interval, its coincidence with the *local* edge (the median particle's
work sign), and its invariances — each a relation, none a coordinate.

## §0 Postdiction disclosure

Everything computed before this seal, on seeds 1–12 at n = 4000 (exp_29–31), a κ sweep on seeds
1–3 (`results/full_explore/`), nine instrumented runs on seeds 1–3 (`results/full_virial_explore/`),
and coupling sweeps on seed 1 (`results/full_coupling_{g,sec}/`). None of it is scored.

1. **The edge, bracketed.** Net pressure work W_p/|U₀|: −2.1 (κ = 0.5), −0.9 (1), +0.12 (1.25), +0.32
   (1.5), +0.58 (2) — negative at κ = 1 in 12/12 seeds, positive at 1.25 in 9/9. Never run between.
2. **The ledger identity that makes it a crossing.** ΔE_SEC = T − W_p on all 48 runs (≤ 1 %); the
   compression ratio E_SEC(end)/T = 5.40 ± 0.05 (κ = 0.5, seeds 1–3, seed-stable to 1 %), 2.0 (1),
   0.85 (1.25), 0.67 (1.5), 0.43 (2). T saturates at ≈ |U₀| as the credit leg grows (decay credits
   back 37 %, 46 %, 62 % of what growth buys at κ = 0.5, 1, 1.25).
3. **The local edge.** Fraction of retained particles with positive cumulative pressure work over
   the window: 16 %, 37 %, 51 % at κ = 0.5, 1, 1.25 (seeds 1–3, spread ≤ 2 %); the median particle's
   cumulative pressure work −4.0, −0.6…−1.2, +0.06/−0.03/+0.05 (units P₀/n). The median crosses
   zero where the total does, in this geometry.
4. **The virial balance.** With pair-form terms, 2⟨K⟩ = |⟨V_g⟩| − ⟨V_p⟩ to ≤ 2 % at κ = 0.5 (3/3);
   not steady above (⟨dG/dt⟩ −15 to −76 % of 2⟨K⟩ at κ = 1 and 1.25).
5. **The couplings (seed 1).** g = 0.75, 1.5, 3.0 at κ = 1: W_p/|U₀| −0.28, −0.58, −0.99 (all negative);
   at κ = 1.25: +0.16, +0.17, +0.22 — the sign holds across a fourfold range of g at both κ. The
   positive-work fraction 38.4 / 38.0 / 37.8 % at κ = 1 and 50.6 / 53.3 / 51.8 % at 1.25 — invariant to
   ±1.5 %. E_SEC(end)/|U₀| at κ = 1: 1.10 / 1.45 / 1.87 — a monotone ~25–30 % shift per factor two in
   g, the entropy clock's leak, sign unchanged; at κ = 1.25 it is flat (0.66 / 0.75 / 0.78). What does
   move with g is the *consequence*: at g = 3 the body of the web at κ = 1.25 is conn_q10 = 0.45
   against 0.17, the spine still gone (q05 0.18). κ_c is a ledger number; how much web survives at it
   is gravity's. **The pair coupling is an exact identity, not an arm:** at κ = 0.5, sec = ½ Ξ/φ
   reproduces the baseline bit for bit (zero relative difference in every recorded quantity, the
   entropy exactly doubled), because every growth is clipped from its first tick and S_i = P_i/price
   with the price ∝ sec — `sec_balance` has no effect on the ledgered dynamics and is not swept.
6. **Growth is budget-limited** (`budget_bound_frac` = 1.00 at every mark from t = 3 in every
   regime), which is the scaling argument's premise.
7. **What is by construction and is struck.** "The edge is where energy flows from the SEC sector to
   motion" is the definition of positive work. It is vocabulary here, never a test.

## §1 Objects (closed at the seal)

- **Substrate:** exp_29/30/31's, unchanged; full size only; box/2r₀ = 3.0 ≥ 3 (carried).
- **The base geometry:** n = 4000, box 60, r₀ = 10, g = 1.5, sec = Ξ/φ, lattice start, clock
  (growth 0.1, decay 0.95 per reference step) — all inherited, none moved.
- **Seeds:** {13, 14, 15} — fresh, never run at any κ or coupling.
- **Arms.**
  - **The fine sweep:** κ ∈ {1.00, 1.05, 1.10, 1.15, 1.20, 1.25} at the base geometry — 18 runs.
  - **The gravity arms:** g ∈ {0.75, 3.0} at κ ∈ {1.00, 1.25} — 12 runs. (g = 3 runs are slow on
    this machine; budgeted.)
  - **The size arm:** n = 8000, box = 60 · 2^{1/3} = 75.6 (box/2r₀ = 3.78), κ ∈ {1.00, 1.25} — 6 runs.
- **Statistics (window means unless stated):** W_p over the run (total, in units of |U₀|); the
  median particle's cumulative pressure work at the last mark; the positive-work fraction; the
  compression ratio E_SEC(end)/T; the virial ratio 2⟨K⟩/(|⟨V_g⟩| − ⟨V_p⟩); `conn_q05/q10` and
  `percolation` reported, not scored.
- **κ_c, defined:** on each seed, the κ of the sign change of W_p on the fine grid, taken as the
  midpoint of the bracketing step (resolution 0.025); the seed mean is the number.

## §2 Pre-seal numbers

§0.1–0.5. The prior for κ_c: in (1.00, 1.25) on every seed so far; the fine grid has never been run.

## §3 Frame

**Sampled:** the whole retained set; every work and virial quantity summed over pairs (pair form,
origin-invariant on the torus). **Expectation:** the same statistics on the same seed's base-geometry
run at the same κ (paired on seed within an arm; the invariance arms compare to the fine sweep's
κ = 1.00 and 1.25 runs of the same seed). **Statistic:** signs, medians, bracketing steps, ratios —
relations; the one absolute number (κ_c) appears only as the location of a sign change.

## §4 Tests (M = 4; thresholds fixed)

**T1 — the edge is a number.** On the fine sweep, 3/3 seeds: W_p changes sign exactly once between
κ = 1.00 and 1.25, from negative to positive; the three seeds' bracketing steps overlap or are
adjacent (κ_c spread ≤ 0.05). The number is reported with its bracket.

**T2 — the local edge coincides with the global one.** 3/3 seeds: the median particle's cumulative
pressure work changes sign in the same bracketing step as the total (or the adjacent one); and the
positive-work fraction at the step is within [0.45, 0.55].

**T3 — the edge does not move with gravity.** For each gravity arm (g = 0.75, g = 3.0), 3/3 seeds:
W_p < 0 at κ = 1.00 and W_p > 0 at κ = 1.25 — the edge stays inside the same bracket across a
fourfold change of g. The reservoir E_SEC(end)/|U₀| at κ = 1 is reported for each arm (the entropy
clock's leak), not scored. (The pair coupling is an exact identity — §0.5 — and is not an arm.)

**T4 — the edge does not move with size.** n = 8000, 3/3 seeds: W_p < 0 at κ = 1.00 and > 0 at 1.25;
and the positive-work fraction at κ = 1.25 within [0.45, 0.60].

**Instrument gates (invalidate the run):** transfer residual ≤ 10⁻⁶, closure ≤ 0.05, at_cap ≤ 0.02,
finite; and on every run |ΔE_SEC − (T − W_p)| ≤ 0.02 |U₀| and |Σ_i W_{p,i} − W_p| ≤ 10⁻⁶ |U₀| (the
identities the derivation stands on, re-checked from the recorded marks).

**Kill.** *If T1 fails — no single sign change, or the seeds disagree by more than a step — the edge
is not a number of the substrate but of the seed, and the derivation's crossing picture is wrong at
the level of a sweep; the compression-ratio story stays descriptive and R2 chooses its operating
point by KE/|U| alone. If T3 fails — the edge moves with g or sec — the scaling argument is wrong
or the entropy clock is not a small leak; the shift is the finding and the clock becomes the object.*

## §5 Expected direction, stated honestly

T1: pass, 8 in 10 (a single crossing is the least likely thing to fail; the seed spread is the
risk). T2: pass, 7 in 10 (the median and total crossed together on three seeds, all at 1.25 — never
between). T3: 7 in 10 — all six sign tests already seen on seed 1 (the reservoir drifts 25–30 % per factor
two and the sign held); the risk is a seed whose crossing sits near 1.25 at g = 3, where the leak
pushes the reservoir up.
T4: 6 in 10 — n = 8000 has never been run with the ledger; the lattice spacing changes (3.75 → 3.75,
same density) but the largest modes double.

## §6 Kill scope

T1/T3 kills retire the derivation's *predictive* reading of the edge; they do not touch exp_29–31,
the ledger, or Milestone R's thesis. The instrument gates invalidate runs, not claims.

## §7 What would count as vacuous

- W_p within ±0.02 |U₀| of zero at every κ on the fine grid (the edge unresolved at this
  resolution) — recorded as UNSCORED, not as a pass or fail.
- The positive-work fraction pinned at 50 % by construction: it is not (a skewed distribution puts
  the median and the total at different κ; §0.3 shows 16 % → 51 %, not a fixed value).
- The budget never binding (`budget_bound_frac_max` < 0.01) in any arm — the scaling premise
  absent; that arm is UNSCORED.
- A `sec_balance` arm would pass by construction (§0.5) and is therefore not registered; the
  identity is recorded as a finding about the mapping, not scored.

## §8 Side predictions (registered, unscored)

SP1 the compression ratio E_SEC(end)/T crosses 1 in the same step as W_p, 3/3. SP2 the virial
ratio 2⟨K⟩/(|V_g| − V_p) is within 0.05 of 1 at κ = 1.00 in ≤ 1/3 seeds and never at 1.25 (above the
plateau the window is not steady). SP3 the credit/growth ratio rises monotonically across the fine
sweep, 3/3. SP4 `conn_q05` at κ = 1.25 is below its κ = 1.00 value in every arm and seed (the spine
goes past the edge, whatever the couplings).

## §9 Excluded

The proxy; seeds 1–12; the Zel'dovich start (the next round's invariance, not this one's); any
κ below 1.00 or above 1.25; the functional's β∇²A and γT·A terms; severance (R2); any named constant
for κ_c.

## §10 Outputs

reality-engine `feat/v4-ledger-virial` (runs at `2b8c1d0`) results dirs `full_r2a_fine/`,
`full_r2a_g/`, `full_r2a_n8000/`; grid JSONs with SHA256s and commit; here the copies
(`results/exp_32_edge_grid_{fine,g,double}_<ts>.json`), the scored JSON, and
`journals/2026-09-1X_exp32_outcomes.md`. Statistics read from the grids' `_summary` only.
