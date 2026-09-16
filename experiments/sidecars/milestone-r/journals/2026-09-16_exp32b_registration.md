# exp_32b registration — the edge as a number, re-sealed with every bar measured against the instrument first (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R). Instruments unchanged
from exp_32: the PAC ledger (exp_29), the connectivity instrument (exp_31, `5f5d690`), the edge
instrumentation of reality-engine `feat/v4-ledger-virial` (runs at `7be57cb`) — no physics change
(spec R4, R8), every added quantity an identity with a test. Scored to this text by
`scripts/exp_32b_edge.py`, thresholds byte-equal to §4 (`--selftest`). Frame §3. Kill scope §6.

**Why this round exists.** exp_32 (registration `091ddab2`, outcomes 2026-09-15) was **UNSCORED by its
own instrument gate**: the ledger-identity tolerance was sealed at 0.02 |U₀|, below the integrator's
documented truncation, and fired on 29 of 36 valid runs. That was my error, not the substrate's. This
round re-asks the same question on fresh seeds with every bar set from a measurement of what the
instrument resolves, recorded in §2 before the seal.

**Thesis under test.** Unchanged from exp_32. W_p = T − ΔE_SEC exactly, so the edge is where the
budget spent equals the pair-energy reservoir's rise — where the compression ratio E_SEC(end)/T
crosses one — and a scaling argument says the crossing's κ cannot depend on g or `sec_balance`,
leaving only the entropy clock as a leak. This round registers the edge as a number reported with its
seed spread, its coincidence with the local edge, and its invariances.

## §0 Postdiction disclosure

Everything below was computed before this seal and **none of it is scored**. Seeds 1–12 (exp_29–31),
and — this is the new part — the 36 runs of the **UNSCORED** exp_32 round on seeds 13–15. Those runs
were read with the gate relaxed on a scratch copy of the sealed scorer; that reading is exploring, not
a result, and it is disclosed here in full because it is what sets this round's priors.

1. **The edge, from the unscored exp_32 fine sweep (seeds 13/14/15).** W_p/|U₀| at κ = 1.00 … 1.25:
   seed 13 −1.26 −0.78 −0.31 −0.09 −0.001 +0.12; seed 14 −0.70 −0.33 −0.10 +0.08 +0.13 +0.18; seed 15
   −0.89 −0.54 −0.32 −0.08 +0.03 +0.10. One sign change each. Interpolated κ_c **1.2003, 1.1288,
   1.1851** — mean 1.171, **spread 0.0715, sd 0.0376**. Bracket spread at step 0.05 is 0.10.
2. **T1's old spread bar was 0.05 and is struck.** It was registered with no prior — the fine grid had
   never been run — at below the spread three seeds show. There is no spread bar in §4.
3. **The local edge (the old T2), unscored: 2/3.** Seeds 13 and 14 crossed in the same or adjacent
   step; seed 15's median sat at −0.02 P₀/n at κ = 1.25, one step behind. **T2 is carried at 3/3
   unchanged.** It is a claim about the substrate and it may well fail again; §5 states that honestly.
4. **The old T2 fraction band, 0.45–0.55, had +0.001 of headroom.** Crossing-step fractions measured
   0.4659/0.4924 (s13), **0.4510**/0.4766 (s14), 0.4578/0.4880 (s15). Across the whole edge region
   (κ ∈ [1.10, 1.30], 21 runs) the statistic runs 0.4246–0.5365, sd 0.0336, and a 0.45 floor fires on
   2 of 21. §4 widens it to [0.40, 0.60]. **This is a bar corrected to the instrument's measured
   spread, not a claim loosened to pass**: the claim is "near half", and 0.40–0.60 still excludes the
   16 % and 51 % values §0.6 records away from the edge.
5. **The gravity arms, unscored: 12/12 signs.** g = 0.75: W_p/|U₀| −1.02/−0.61/−0.73 at κ = 1 and
   +0.06/+0.15/+0.12 at 1.25. g = 3.0: −1.56/−0.75/−1.10 and +0.23/+0.26/+0.25. The reservoir at κ = 1
   is 1.43–1.85 (g = 0.75) and 1.63–2.47 (g = 3) |U₀| — monotone, sign unchanged.
6. **The size arm, unscored: 2/3 — and this is the honest prior for T4.** n = 8000: W_p/|U₀|
   −0.59/−2.15/−0.65 at κ = 1; at 1.25 **−0.04**/+0.07/+0.14. Seed 13's edge sits at or just above
   1.25 at the larger size. The fraction at 1.25 was 0.49/0.51/0.53.
7. **The ledger identity's measured floor, over all 36 runs.** |ΔE_SEC − (T − W_p)|/|U₀| ranges
   **0.0154 – 0.0535**, median 0.0251, sd 0.0101. The other gates have large headroom: `at_cap_max`
   max 0.0043 against 0.05 (11.8×), `closure_pac_max` max 0.0008 against 0.05 (61.6×),
   `transfer_residual_max` max 4.3 × 10⁻¹⁰. §4 sets the identity gate at **0.08 |U₀|** — ≈ median + 5 sd,
   1.50× the measured maximum, and 0/36 on the exp_32 data. 0.06 was considered and rejected: at 1.12×
   an order statistic over 36 runs it will fire on a fresh draw for reasons that are not the claim.
8. **The pair coupling is an exact identity, not an arm** (exp_32 §0.5, carried): at κ = 0.5 with
   sec = ½ Ξ/φ the run reproduces the baseline bit for bit, because growth is budget-clipped from its
   first tick and S_i = P_i/price with price ∝ sec. Not swept.
9. **Growth is budget-limited** (`budget_bound_frac` = 1.00 from t = 3 in every regime) — the scaling
   argument's premise, carried.
10. **A scorer defect found before this seal and fixed here.** exp_32's completeness check keyed on
    (κ, g, size) and omitted the seed, so a grid missing one seed at one κ was declared complete and
    then died inside scoring with a raw `KeyError`. `exp_32b_edge.py` puts the seed in the tuple. All
    seven abort paths were driven with corrupted grids before this seal (§10).

## §1 Objects (closed at the seal)

- **Substrate:** exp_29/30/31's, unchanged; full size except the size arm; box/2r₀ = 3.0 ≥ 3.
- **Base geometry:** n = 4000, box 60, r₀ = 10, g = 1.5, sec = Ξ/φ, lattice start, clock (growth 0.1,
  decay 0.95 per reference step) — all inherited, none moved.
- **Seeds:** {16, 17, 18} — fresh, never run at any κ or coupling in any POC-12 results directory
  (verified against every results dir before the seal: seeds 1–15 only).
- **Arms.**
  - **Fine sweep:** κ ∈ {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, **1.30**} at base geometry — 21 runs.
    1.30 is new: exp_32's seed 13 crossed at 1.20 on the base geometry and at or above 1.25 at
    n = 8000, so the old grid's top edge was at the phenomenon.
  - **Gravity arms:** g ∈ {0.75, 3.0} at κ ∈ {1.00, 1.25} — 12 runs.
  - **Size arm:** n = 8000, box = 75.6, κ ∈ {1.00, 1.25, **1.30**} — 9 runs. 1.30 is reported, not
    scored; it exists so that a T4 failure can be *located* rather than only recorded.
- **Statistics (window means unless stated):** W_p over the run in units of |U₀|; the median
  particle's cumulative pressure work at the last mark; the positive-work fraction; the compression
  ratio E_SEC(end)/T; the virial ratio 2⟨K⟩/(|⟨V_g⟩| − ⟨V_p⟩); `conn_q05/q10` and `percolation`
  reported, not scored.
- **κ_c, defined:** on each seed, the κ of the sign change of W_p on the fine grid, by **linear
  interpolation between the bracketing marks**; reported per seed and as a mean with its spread.
  (exp_32 took the bracket midpoint; interpolation is strictly more information from the same marks
  and is fixed here before any run.)

## §2 Pre-seal numbers

§0.1–0.9. The prior for κ_c is 1.171 ± 0.038 on three other seeds, at a grid step of 0.05.

## §3 Frame

**Sampled:** the whole retained set; every work and virial quantity summed over pairs (pair form,
origin-invariant on the torus). **Expectation:** the same statistics on the same seed's base-geometry
run at the same κ (paired on seed within an arm; the invariance arms compare to the fine sweep's
κ = 1.00 and 1.25 runs of the same seed). **Statistic:** signs, medians, bracketing steps, ratios —
relations; the one absolute number (κ_c) appears only as the location of a sign change and carries no
bar.

## §4 Tests (M = 4; thresholds fixed)

**T1 — the edge is a single crossing.** On the fine sweep, 3/3 seeds: W_p changes sign **exactly once
across the swept range κ ∈ [1.00, 1.30]**, from negative to positive. κ_c is reported per seed by
interpolation, with the three-seed mean and spread. **No spread bar** (§0.2).

**T2 — the local edge coincides with the global one.** 3/3 seeds: the median particle's cumulative
pressure work changes sign in the same bracketing step as the total, or the adjacent one; and the
positive-work fraction at the crossing step is within **[0.40, 0.60]** (§0.4).

**T3 — the edge does not move with gravity.** For each gravity arm (g = 0.75, g = 3.0), 3/3 seeds:
W_p < 0 at κ = 1.00 and W_p > 0 at κ = 1.25. The reservoir E_SEC(end)/|U₀| at κ = 1 is reported for
each arm, not scored. (The pair coupling is an exact identity — §0.8 — and is not an arm.)

**T4 — the edge does not move with size.** n = 8000, 3/3 seeds: W_p < 0 at κ = 1.00 and > 0 at 1.25;
and the positive-work fraction at κ = 1.25 within [0.45, 0.60]. W_p at κ = 1.30 is reported.

**Instrument gates (invalidate the run, not the claim):** finite; `transfer_residual_max` ≤ 10⁻⁶;
`closure_pac_max` ≤ 0.05; `at_cap_max` ≤ 0.05; and on every run **|ΔE_SEC − (T − W_p)| ≤ 0.08 |U₀|**
(§0.7) and |Σ_i W_{p,i} − W_p| ≤ 10⁻⁶ |U₀|.

**Kill.** *If T1 fails — no single sign change across the swept range — the edge is not a single
crossing of the substrate and the derivation's crossing picture is wrong at the level of a sweep; the
compression-ratio story stays descriptive and R2 chooses its operating point by KE/|U| alone. If T3
fails — the edge moves with g — the scaling argument is wrong or the entropy clock is not a small
leak; the shift is the finding and the clock becomes the object.*

## §5 Expected direction, stated honestly

- **T1: pass, 9 in 10.** Three other seeds each gave exactly one crossing inside the old narrower
  range; widening to 1.30 can only reveal a second crossing, which nothing suggests.
- **T2: pass, 4 in 10.** It was 2/3 unscored — seed 15's median trailed by a step. The fraction clause
  is now set above its measured floor, so the risk is entirely the median-crossing clause, which is a
  real claim about the substrate. I expect this one to be the round's most likely failure.
- **T3: pass, 8 in 10.** Twelve of twelve signs already seen unscored across a fourfold range of g.
- **T4: pass, 4 in 10.** It was 2/3 unscored, and the failing seed missed by 0.04 |U₀|. Registering it
  at 3/3 again with an honest 4-in-10 prior; κ = 1.30 is there to say where the edge went if it fails.

## §6 Kill scope

T1/T3 kills retire the derivation's *predictive* reading of the edge; they do not touch exp_29–31, the
ledger, or Milestone R's thesis. The instrument gates invalidate runs, not claims. Nothing here
rescores exp_32, which stays UNSCORED in the record.

## §7 What would count as vacuous

- W_p within ±0.02 |U₀| of zero at every κ on the fine grid (the edge unresolved at this resolution) —
  recorded as UNSCORED, not as a pass or fail.
- The positive-work fraction pinned at 50 % by construction: it is not — §0.1 and exp_32 §0.3 show
  16 % → 53 % across κ, not a fixed value, and the band excludes both ends.
- The budget never binding (`budget_bound_frac_max` < 0.01) in any arm — the scaling premise absent;
  that arm is UNSCORED.
- A `sec_balance` arm would pass by construction (§0.8) and is therefore not registered.
- **T1 with no spread bar is not thereby unfalsifiable:** it fails on zero crossings, on two or more,
  or on a positive-to-negative direction, in any one of three seeds.

## §8 Side predictions (registered, unscored)

SP1 the compression ratio E_SEC(end)/T crosses 1 within one step of W_p's sign change, 3/3. SP2 the
virial ratio is within 0.05 of 1 at κ = 1.00 in ≤ 1/3 seeds and never at 1.25. SP3 the credit/growth
ratio rises monotonically across the fine sweep, 3/3. SP4 `conn_q05` at κ = 1.25 is below its κ = 1.00
value in every arm and seed. SP5 κ_c's three-seed spread is ≥ 0.05 — the number is a property of the
seed's collapse geometry at this resolution, not of the substrate alone.

## §9 Excluded

The proxy; seeds 1–15; the Zel'dovich start; any κ below 1.00 or above 1.30; the functional's β∇²A and
γT·A terms; severance (R2); any named constant for κ_c.

## §10 Outputs and the pre-seal scorer exercise

reality-engine `feat/v4-ledger-virial` (runs at `7be57cb`), results dirs `full_r2b_fine/`,
`full_r2b_g/`, `full_r2b_n8000/`; grid JSONs with per-run SHA256s and commit; here the copies
(`results/exp_32b_edge_grid_{fine,g,double}_<ts>.json`), the scored JSON, and
`journals/2026-09-1X_exp32b_outcomes.md`. Statistics read from the grids' `_summary` only.

**Before this seal**, `scripts/exp_32b_edge.py` was driven down **all eight** of its non-happy paths
and each behaved as this text says: selftest mismatch, unclassifiable grid, missing grid, unregistered
seed, **incomplete grid (the defect of §0.10)**, instrument-gate failure, `--selftest` alone, and
**T1's failure branch on a seed with more than one crossing, which is the path that raises the kill**.
The last one matters most: a broken kill path is a kill that never fires.

The fixture for that exercise was the exp_32 grids with seeds relabelled 13/14/15 → 16/17/18 and a
synthetic κ = 1.30 mark added with the ledger identity preserved. It is **plumbing only, scratch only,
never committed and never a result** — it exists so that every branch of the scorer has run before any
real data reaches it. A single happy-path run reaches 90 % of statements; the eight remaining are
exactly the paths above.
