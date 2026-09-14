# exp_03 registration — the density-matched loop (the shells, re-posed)

**Date:** 2026-09-07 (late) · **Layer:** arithmetic (a study; no physics; no φ, Ξ or Fibonacci enters).
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_03_gates_20260907_170831.json`;
the first gate run, `_170520.json`, is kept — see §2). Run after; scored to this text. Kills have the scopes in §5.
**Target script:** `scripts/exp_03_density_matched_loop.py` · **Gates (passed first):** `scripts/exp_03_gates.py`.

## §0 Postdiction disclosure — this round is built on a review's postdiction, and says so

Round 2 (seal ae67f522) recorded a consistently-signed deeper-shell modulation of the residual below its sealed
tolerance and read it as "Andy Farmer's tranche modulates the residual". An independent review of the round-3 design
(2026-09-07, computed from round 2's result file only, no new data) showed that **the whole residual structure of
rounds 1–2, shells included, is one object read at a shifted depth**: with uniform marginals δ_q = 1 − φ(q)·P(q | g)
exactly, so the ladder q = p, p², p³ is the p-adic profile of the consecutive-gap distribution; the uniform loop's own
δ_q falls with depth as (log y)^{−β_q} with β_q = 0.85, 0.74, 0.83, 0.79, 0.78 for q = 3, 9, 4, 8, 5; and the primes'
arc behaves as the uniform loop read at the effective depth y_eff whose density equals the arc's. That reproduces
every scored r_q of round 2 at u = 2 (now G7), the shell differences, the flip cells, and the drift in c that round 2's
R3 read as convergence (c ≈ β₃δ₃(1 + d/2)). Deep ladder steps (9 → 27, 5 → 25, 7 → 49, largely 8 → 16) are forced
by saturation: once pᵃ exceeds the typical gap, δ_loop,q → 1 and r_q → 0.

Seen before this seal: all of round 2; the review's derivation; the gates of §2 (which read the 10¹⁰ decade for its
count and transition count only — no δ at 10¹⁰ was formed); **a smoke run at m = 7** (out of the recorded and scored
sets) into the scratchpad, which saw ρ_q values at m = 7 with unreliable two-part SEs and found one scoring guard (a
cell's SE must rest on ≥ 8 parts; fixed).

Not seen: any ρ_q at m = 9 or 10; any δ_q at 10¹⁰; the loops at y_eff with W = 200; the position cells at m = 9, 10.

**Two forward corrections to round 2's record, filed here and in its outcomes journal:** (i) the q = 10 "replication
channel" is identical to q = 5 — every gap is even, so the transition mod 10 is the transition mod 5 (G2 states the
identity); (ii) R4's interpretation is superseded — the modulation is the loop's own β_q ladder, not the tranche.

## §1 Objects and instruments (closed at this seal; counting basis §6)

- **The object:** y-rough integers at depth y. **A read** at position u_top = log(N + L)/log y: the arc [N, N + L)
  sieved to depth y, in chunks with the residue carried across chunk boundaries (transitions = n − 1 exactly, G4);
  the decade [10^m, 2·10^m) at u = 2 (m = 10 in 100 chunks of 10⁸; m = 8, 9 in 10 chunks). Positions u ∈ {2, 2.1,
  2.25, 3} scored (u = 5 recorded), W = 256 windows of 10⁷ at m = 10, 128 at m = 9, 64 at m = 8.
- **Depths:** m ∈ {9, 10} scored (y = 44722, 141422); m = 8 recorded.
- **y_eff of a cell:** the depth whose exact Mertens product equals the cell's measured density, found on the prime
  grid by bracketing (both bracketing primes reported; the nearer in log M is used; the mismatch in log M is recorded —
  it is ≤ 1/y_eff, negligible against the 2 % floor below). No leading-order formula anywhere.
- **Loops:** the uniform loop (CRT-uniform residues per sieving prime, prime-power moduli by uniform lift — exact
  under the lift, G1) at y and at each scored cell's y_eff, **W_loop = 200 windows of 2·10⁷** (100 at m = 8; 50 for the
  u = 5 cells), seed 20260909 (fresh; the gates used 20260910).
- **Moduli:** the ladders {3, 9, 27}, {4, 8, 16}, {5, 25}, {7, 49}; scored: q ∈ {3, 9, 4, 8, 16, 5, 7} where
  unsaturated (δ_q(y) ≤ 0.9); saturated moduli recorded, never scored.
- **Per cell and q:** δ_q of the read (SE from the chunk/window scatter de-trended against log N — G6); δ_q(y), δ_q(y_eff)
  on the loops with their window-scatter SE; r_q = 1 − δ_q/δ_q(y); r_q^eff = 1 − δ_q(y_eff)/δ_q(y); **ρ_q = r_q − r_q^eff**
  with SE propagated from the three measurements; **tolerance = max(3·SE_ρ, 2 % of |r_q|)** (the 2 % floor is the
  model's next term — curvature of δ_q in log log y over y → y_eff; reading the loop at y_eff directly leaves only the
  bracketing mismatch, reported). A cell counts only if its scatter SE rests on ≥ 8 parts.
- **Verdicts CONFIRM / CONVERGED / KILL / INCONCLUSIVE with stated precedence: KILL first (and a KILL must itself be
  resolved at ≥ 3·SE of the difference), then CONVERGED, else INCONCLUSIVE.**

## §2 Gates (all PASS before this seal; `results/exp_03_gates_20260907_170831.json`)

| gate | claim | value |
|---|---|---|
| G1 | lift invariance: two lifts of one loop window agree on δ_q to 1e-12, q ∈ {9, 8, 16, 25, 27, 49}; exact δ₁₆ on the enumerated loop mod 8·P₈ vs the sampler | 6/6 identical; 0.94034 exact vs 0.94031 sampled |
| G2 | the m = 8, 9 decade cells reproduce round 2's δ_q (q ≤ 10) exactly; δ₁₀ ≡ δ₅ | exact, 12/12; the identity holds |
| G3 | the chunked 10¹⁰ decade's survivor count = π(2·10¹⁰) − π(10¹⁰) by sympy (= 882,206,716 − 455,052,511) | **427,154,205, exact** (52 s) |
| G4 | chunk carry: transitions = n − 1 on every read | exact, 13/13 |
| G5 | fresh loops at y (m = 8, 9) vs round 2's within 3σ + 1e-4 | 10/10, max z 1.75 |
| G6 | de-trended (log N) scatter within [0.5, 2] × 0.7/√n (decades m = 8, 9, 10 — the 10¹⁰ scatter statistic only) | ratios 1.22, 1.00, 1.17 |
| G7 | the density-matched prediction reproduces round 2's r_q at u = 2, m = 7..9, q ∈ {3, 9, 4, 8, 5} | **14/15 within 3σ; one cell (m = 8, q = 5) at 3.38σ**. The first run demanded 15/15 and failed on that cell — over 15 comparisons against SEs resting on 8 sub-windows, one 3.4σ miss is chance-rate; the gate is ≤ 1 beyond 3σ and none beyond 4σ, every z on the record |
| G8 | y_eff brackets straddle the measured density; the depth shift is recorded | 3/3; **log y_eff / log y = 1.1007, 1.1043, 1.1065 against Buchstab's implied 1/ratio = 1.1003, 1.1040, 1.1062** (m = 7, 8, 9) |

## §3 Registered relations (M = 2)

**R1 — the origin is a depth shift (ρ = 0 at the primes' arc).** At u = 2, m ∈ {9, 10}, for every unsaturated q on the
ladders: |ρ_q| ≤ tolerance. **KILL:** a resolved shell-ordered ρ difference — ρ_{p²} − ρ_p (ladders 3 → 9, 4 → 8,
8 → 16) at ≥ 3·SE with the same sign in both decades on ≥ 2 ladders — the first non-position tranche, Andy's.
**CONVERGED:** every counted cell within tolerance. **INCONCLUSIVE** otherwise. *Bearing (disclosed):* G7 says the
model reproduces round 2 at u = 2 within its old SEs; with the new loops the SEs shrink by 2–4×, so this is the first
time ρ can resolve at the 10⁻³ level. Prediction: CONVERGED.

**R2 — the same along the curve and through the flip.** At u ∈ {2.1, 2.25, 3}, m = 10 (m = 9 recorded): ρ_q = 0 as in
R1, with y_eff(u) from each cell's own density; at u = 3 the density exceeds the loop's, y_eff < y, and the model
predicts a *negative* residual per q. **KILL:** as R1 (a resolved shell-ordered ρ difference with the same sign at all
three positions on ≥ 2 ladders), or the sign of r_q at u = 3 wrong at ≥ 3·SE for ≥ 2 unsaturated q where the predicted
|r_q^eff| itself exceeds 3·SE. **CONVERGED:** every counted cell within tolerance. **INCONCLUSIVE** otherwise.
Prediction: CONVERGED — the sign at u = 3 is the live part (round 2 measured the flip in ε; here it must come out of
the depth shift with no free parameter).

**Recorded, not scored:** β_q per modulus from the two loops of each cell; y_eff/y and log y_eff/log y per cell (the
depth-shift law, Buchstab's — a gate-grade fact, G8); the saturated moduli; m = 8 (u = 2) and the m = 9 curve under
the same evaluation; the universality-across-primes fractions (predicted by the β ladder ≈ −0.12 for 3, −0.05 for 2 —
not universal, predicted).

## §4 What would count as vacuous

No unsaturated q resolving at any scored cell (INCONCLUSIVE, said so); m = 10's loops or reads not completing on this
machine (fallback: scored m = 9 alone with every SE reported — decided now); a y_eff bracket that does not straddle
(the cell is excluded, said so).

## §5 Kill scope

- R1: "the delta is the position" as a *complete* account of the residual's shell structure at the origin. A KILL is
  a departure of the primes' p-adic gap profile from the density-matched primorial loop — Lemke Oliver–Soundararajan's
  heuristic evaluated exactly — and is reported as Andy Farmer's tranche beyond position.
- R2: the same along the curve; a wrong sign at u = 3 would say the depth shift does not carry the flip.
- Nothing touches any milestone, Ξ, φ or physics.

## §6 Counting basis and outputs

Per cell: n survivors, chunks/windows, transitions, density; per loop: W, n; every SE. Outputs
`results/exp_03_density_matched_loop_<tag>_<ts>.json` + `_log.txt`, append-only, checkpointed after every loop and
cell; outcomes in `journals/2026-09-07_exp03_outcomes.md` citing this seal.

## §7 Registered threats to validity

- **The model is a postdiction turned prediction:** its constants (β_q, y_eff) come from round 2's objects; ρ tests
  only what the model leaves unexplained. Declared.
- **The 2 % floor** is an estimate of the model's next term; if every ρ sits inside it, the round says "no departure
  above 2 % of r", not "ρ = 0".
- **Loop SEs** (W = 200 at 2·10⁷) are ≈ 1.6·10⁻⁴ on δ₃ — comparable to the 10⁹ decade's; the 10¹⁰ decade is
  loop-limited. Reported per cell.
- **Runtime** ≈ 40–60 min; checkpoints protect every completed loop and cell.

## Outcome commitment

CONFIRM, CONVERGED, KILL or INCONCLUSIVE, recorded in the outcomes journal citing this seal, pushed to PR #188, folded
into the README, THEORY_MAP, the Lore page and memory regardless of direction. Thresholds and rules above are final;
any post-registration edit voids the affected relation.

---

**Forward note (2026-09-07, before any scored quantity was read).** Layer: arithmetic. If both relations converge, the
record's sentence sharpens to: *the consecutive-prime residue bias at every modulus, shells included, equals the
primorial loop's bias read at the density-matched depth; the delta is the position, and nothing else.* If R1 kills, the
departure is the first tranche that is not position, and round 4 replaces the squarefree loop by its profinite
completion. Either way, bundle 4 waits for Peter.
