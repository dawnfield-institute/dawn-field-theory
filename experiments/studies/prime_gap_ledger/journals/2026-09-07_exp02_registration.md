# exp_02 registration — the position of the read (and Andy's ultrametric tranche)

**Date:** 2026-09-07 (evening) · **Layer:** arithmetic (a study; no physics is claimed; no φ, Ξ or Fibonacci enters).
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_02_gates_20260907_132456.json`;
the first gate run, `_132239.json`, is kept — see §2). Run after; scored to this text. Kills have the scopes in §5.
**Target script:** `scripts/exp_02_position_of_the_read.py` · **Gates (passed first):** `scripts/exp_02_gates.py`.

## §0 Postdiction disclosure

This round is more postdictive than round 1, and says so clause by clause.

Seen before this seal:
- **Round 1's decade cells** (`exp_01`, seal 9213386c): the primes' bias at u = 2 and the uniform loop's bias at the same
  depth for m = 5..9 — so **ε₃(2; y) = 0.0093, 0.0098, 0.0099, 0.0088 (m = 6..9) is known**, and R1's convergence clause
  at u = 2 is a *registered postdiction check*, not a prediction. Its live content is the approach over u. Round 1 also
  saw u = 3 at m = 8, 9 (loop under-predicting by 0.0007, 2–7σ) and δ₄'s residual at 10⁹ (r₄ = 0.078 against r₃ = 0.079).
- **The exploration** (`explore_r2_position_along_the_loop.py`, journal `2026-09-07_exploring_position_along_the_loop.md`):
  fixed-depth reads at positions 10⁴…10⁵⁰ for y ∈ {142, 448, 1415, 4473}, windows 10⁶–4·10⁶ — the climb to the plateau
  within ~½ unit of depth; its positions are not the registered arcs (N = round(y^u/2)) but they are near them.
- **An independent review** of the design (reasoning only): the parametrisation by the arc's top, ε as the sealed
  form, the power table, the gate tolerances, the sign flip as the sentence worth registering.
- **A smoke run at m = 5 (y = 448)** into the scratchpad, never scored: it saw ε₃ = +0.0017 ± 0.0022, −0.0003 ± 0.0009,
  −0.0007 ± 0.0005 at u = 2.5, 2.75, 3 (the flip's shape at the wrong depth), c = +0.094 at u = 2.25, and the q = 8, 9
  residuals at u = 2 there. It found one scoring guard (an empty scored set read as CONVERGED; fixed).
- **The gates** of §2, with their values.

Not seen: any quantity at u ∈ {2.1, 2.25} for m ≥ 6 on the registered arcs; the flip cells at m = 7, 8 with W = 200;
c with the measured deficit; the q = 5, 8, 9 residuals at m ≥ 6. Those carry the round.

## §1 Objects and instruments (closed at this seal; counting basis §6)

- **The object:** the y-rough integers at depth y (coprime to every prime ≤ y). **The uniform loop** = the read at a
  CRT-uniform point (residues drawn independently per prime; 25 windows of 2·10⁷, seed 20260908 — fresh). **A position**
  = the arc [N, 2N) of the integers sieved to depth y, N = round(y^u/2), labelled by u_top = log 2N / log y (u_bottom =
  u − log 2/log y recorded); at u = 2 the decade [10^m, 2·10^m) literally. Read as one arc when 2N ≤ 2·10⁸ (8
  sub-windows for the scatter), else W consecutive windows of 10⁷ from N (W = 32; **W = 200 at the flip cells**
  u ∈ {2.5, 2.75, 3} for m ∈ {7, 8}).
- **Depths** y = ⌈√(2·10^m)⌉ for m = 6, 7, 8, 9 (1415, 4473, 14143, 44722); **positions** u ∈ {1.75 (negative control),
  2, 2.1, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 7}.
- **Per cell:** survivors n; density ratio to Mertens and the measured deficit **d = 1 − ratio** (SE from the density
  scatter); the consecutive-survivor diagonal deficit δ_q against the product-of-marginals null for q ∈ {3, 4, 5, 8, 9,
  10}, with SE from the **de-trended** window scatter (a linear trend in window index removed — the within-arc LO–S drift
  is systematic, not noise; §2 G3); **ε_q = δ_loop,q − δ_q** (sealed primary form); r_q = ε_q/δ_loop,q (alternative,
  recorded); **c = ε₃/d** with d measured on the same arc (§2.9: same scope).
- **Prime-power moduli on the loop:** N mod pᵃ is the draw at p lifted uniformly to mod pᵃ (the loop mod lcm(q, P(y)) is
  still CRT-uniform — a squarefree sieve leaves the deeper shells free); checked exact on enumerated loops (G6).
- **Power, declared from the arithmetic** (a position at depth u holds y^u integers): evaluable cells are u = 2 at every
  m (m = 6 marginal), u ∈ {2.1, 2.25} for m ≥ 7 only, and the flip cells with W = 200 (n ≈ 10⁸, SE ≈ 7·10⁻⁵). Small-y
  cells at u ≤ 2.25 are recorded, never scored. u < 2 is a different regime (the survivors are the primes of [N, 2N) for
  every depth, so δ depends on N alone): excluded from every relation; one u = 1.75 cell per depth recorded as the
  negative control on which the position reading must fail.
- **Verdict classes: CONFIRM / CONVERGED / KILL / INCONCLUSIVE.** CONVERGED = every scored value within tolerance of the
  common value — the plateau class round 1 lacked. Tolerances are relative and floor-aware: max(3·SE, a stated fraction).

## §2 Gates (all PASS before this seal; `results/exp_02_gates_20260907_132456.json`)

| gate | claim | value |
|---|---|---|
| G1 | arc density ratio = e^{γ}[2ω(u_top) − ω(u_bottom)] within 2/log N on every position cell; exact prime counts at u = 2 | 44/44. **First run used 1/log N and failed the four u = 1.75 control cells by 0.057–0.085**: the leading-order form omits the prime count's own next term (li against x/log x ≈ 1/log N); tolerance set to 2/log N with the reason stated |
| G2 | the fresh-seed uniform loop reproduces round 1's δ₃, δ₄, δ₁₀ per depth within 3σ + 1e-4 | 12/12 (δ₃: 0.1548/0.1546, 0.1371/0.1368, 0.1235/0.1233, 0.1122/0.1123) |
| G3 | de-trended sub-window scatter of δ₃ within [0.5, 2] × 0.7/√n on the decade and plateau cells | 12/12, max ratio 1.43. **First run used the raw scatter and read 3.17× the noise on the 10⁹ decade** — the LO–S drift across the decade; de-trended, 1.2×. Under-dispersion would be an instrument fault; none |
| G4 | the decade cells reproduce round 1's δ₃ at u = 2 exactly (same objects) | 4/4, deviation < 1e-12 |
| G5 | plateau: at u ∈ {5, 7} every δ_q (q = 3, 4, 10) meets δ_loop,q within max(3·SE, 1 % δ_loop) — equidistribution, a known answer | 24/24 |
| G6 | the prime-power lift reproduces the exact δ_q on enumerated loops mod lcm(q, P_k), k = 7, 8, q = 8, 9 | 4/4 within 3·SE |

## §3 Registered relations (M = 4)

**R1 — the residual is a scale-free property of the read's position (Peter's reading).**
(a) *Registered postdiction check:* over m = 7..9 at u = 2, ε₃ is CONVERGED — |ε₃ − mean| ≤ max(3·SE, 10 % of the
mean) for every m; KILL if any |ε₃ − mean| > max(3·SE, 30 % of the mean). (b) *Live:* for m ≥ 7, ε₃ decreases
over u ∈ {2, 2.1, 2.25}; a step counts as resolved when |Δε| ≥ 2·SE of the difference; KILL if ε increases at ≥ 2
resolved steps. Verdict: KILL if either kill fires; CONFIRM if (a) converges and every resolved step decreases (≥ 1
resolved); CONVERGED if (a) converges and no step resolves; INCONCLUSIVE otherwise. *Alternative form recorded:* r₃
under the same rule (a) — which form converges is reported; ε is sealed as primary and is not changed after the fact.

**R2 — the sign flip (Buchstab for the transition).** Flip cells: u ∈ {2.5, 2.75, 3}, m ∈ {7, 8}, W = 200. A cell is
resolved when |ε₃| ≥ 3·SE_ε; d is resolved when |d| ≥ 3·SE_d. CONFIRM: ε₃ < 0 in ≥ 4 resolved cells of 6, and in every
cell where both resolve sign(ε₃) = sign(d). KILL: ε₃ ≥ 0 in ≥ 4 resolved cells (no flip), or sign(ε₃) ≠ sign(d) in ≥ 2
cells where both resolve. INCONCLUSIVE otherwise. *Bearing (disclosed):* round 1's u = 3 at m = 8, 9 gave ε₃ = −0.0007
(2–7σ) with ratio 1.002–1.004; the m = 5 smoke showed the same shape. Prediction: CONFIRM.

**R3 — Buchstab proportionality.** c = ε₃/d over the resolved cells among u ∈ {2, 2.1, 2.25} for m ≥ 7 and the resolved
flip cells (both ε and d at ≥ 3·SE). CONVERGED when the spread of c ≤ max(3 × the largest SE_c, 25 % of the mean) with
≥ 2 resolved cells; KILL when c differs by more than a factor 2 between two resolved cells (including a sign change);
INCONCLUSIVE otherwise. c is recorded (a coordinate). *Bearing:* round 1's asymptotic-ω estimates 0.087 (u = 2), 0.094
(u = 2.25); the smoke's 0.094 at u = 2.25 with the measured d, 0.10 at u = 3. No direction registered.

**R4 — shell independence (Peter) against the ultrametric tranche (Andy).** At u = 2 over m = 7..9, r_q = ε_q/δ_loop,q
for q ∈ {3, 9} (the prime 3 at shells 1, 2), {4, 8} (the prime 2 at shells 2, 3) and {5}. CONFIRM (the residual is a
position effect): every r_q within max(3·SE, 15 % of the mean over q) in every decade. KILL (the missing tranche is
the deeper shells): r₉ − r₃ and r₈ − r₄ have the same sign and each exceeds its tolerance in ≥ 2 of the 3 decades.
INCONCLUSIVE otherwise. *Bearing (disclosed):* r₄ ≈ r₃ at 10⁹ in round 1; q = 5, 8, 9 unseen at m ≥ 6. Prediction:
CONFIRM — with the caveat owed to Andy that Hardy–Littlewood weights are blind to v₂(g), so the shells were never
expected to carry the primary structure; this clause asks whether they carry the residual.

**Recorded, not scored:** the u = 1.75 negative control (ε there must not match the position curve); the q = 10 channel's
sign agreement with q = 3 wherever both resolve (a directional replication with better S/N); m = 6's ε at u = 2; every
cell's c.

## §4 What would count as vacuous

Fewer than two resolved cells for R3 (INCONCLUSIVE, said so); no resolved flip cell (R2 INCONCLUSIVE); m = 9's decade
not computable on this machine (fallback: scored decades m = 7, 8 with every SE reported — decided now).

## §5 Kill scope

- R1: "the delta is the position" as a scale-free statement about the absolute residual; the exact loop biases and
  round 1's y-only collapse are untouched.
- R2: "the transition residual is Buchstab-like" — a KILL says the origin effect on the transition structure has no
  overshoot, and the analogy to the density stops at the sign.
- R3: the proportionality only; the flip can hold without a constant c.
- R4: one of the two readings of the tranche; a KILL here is Andy's picture confirmed at LO–S scale, and is reported as
  such.
- Nothing touches any milestone, Ξ, φ or physics.

## §6 Counting basis and outputs

Per cell: n survivors, windows, SE per q; per depth: the loop's n and SE. Outputs `results/exp_02_position_of_the_read_
<tag>_<ts>.json` + `_log.txt`, append-only, checkpointed per depth; outcomes in `journals/2026-09-07_exp02_outcomes.md`
citing this seal.

## §7 Registered threats to validity

- **Postdiction in R1(a) and the bearings of R2, R3, R4** — declared above; the verdict table will mark R1(a) as a
  postdiction check.
- **The de-trended SE** removes a linear drift only; a curved drift across a long arc leaves residual over-dispersion,
  which makes tolerances wider (conservative), never narrower.
- **The flip cells are consecutive windows** from N = round(y^u/2): their u spans ≈ 0.02 at m = 8 — recorded as u_top.
- **W = 200 at six cells** is the compute budget; if a flip cell resolves at neither sign, it is INCONCLUSIVE, not
  softened.

## Outcome commitment

CONFIRM, CONVERGED, KILL or INCONCLUSIVE, in any mix, recorded in the outcomes journal citing this seal, pushed to the
same PR (#188), folded into the study README, THEORY_MAP, the Lore page and memory regardless of direction. Thresholds
and rules above are final; any post-registration edit to them voids the affected relation.

---

**Forward note (2026-09-07, before any scored quantity was read).** Layer: arithmetic. If R2 confirms, the sentence is:
the consecutive-prime residue bias equals the primorial loop's exact rational at √(2x) minus a residual that tracks
Buchstab's density deficit and changes sign where ω(u) crosses e^{−γ}. If R4 kills, Andy's shells carry the residual
and the loop must be replaced by its profinite completion in the next round. Either way, bundle 4 waits for Peter.
