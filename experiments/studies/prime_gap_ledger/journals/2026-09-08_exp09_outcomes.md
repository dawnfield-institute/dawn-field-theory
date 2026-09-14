# exp_09 outcomes — F's residual is real structure, not sampling: 2/3

**Date:** 2026-09-08 · **Layer:** arithmetic · **Seal:** `79358696`
(`journals/2026-09-08_exp09_registration.md` + `results/exp_09_gates_20260908_132250.json`, 8/8 PASS).
**Run:** `scripts/exp_09_exact_residual.py` → `results/exp_09_exact_residual_20260908_132440.json`.
Scored to the sealed text; no threshold relaxed. **87 of 105 cells scored** — 18 excluded under §7.1 for a
tail share above 10 % (q = 21…29 at the deeper y, where truncation bites).

## Verdicts

| | relation | verdict |
|---|---|---|
| R1 | the residual is real structure, not sampling noise | **CONFIRM** — rms **0.05220** against CONFIRM ≥ 0.01826, KILL ≤ 0.00913 |
| R2 | the ω(q) structure is real | **INCONCLUSIVE** — the §4 guard fired: only **two** distinct ω values on this pool |
| R3 | truncation is not driving it | **CONFIRM** — rms 0.05220 at G = 40 vs 0.05374 at G = 36, a **2.96 %** change |

**Score 2/3.** Study total **14/24**.

## R1: the question is answered, and more cleanly than the threshold framed it

The registered comparison was against exp_07's sampled loop rms of 0.03043. The direct control is better, and
it is unambiguous — measuring δ by **sampling at the identical cells**:

| | n | rms(exact) | rms(sampled) |
|---|---|---|---|
| y = 23 | 21 | 0.05383 | 0.05383 |
| y = 97 | 16 | 0.05315 | 0.05315 |
| y = 401 | 16 | 0.04950 | 0.04949 |
| **all** | **87** | **0.05220** | **0.05219** |

Mean |exact − sampled| per cell: **0.000163**. At 24 windows of 2·10⁶, δ is measured to ~1.6 × 10⁻⁴ while the
residual is 0.052 — **320× larger**. Sampling noise was never a plausible explanation for F's residual, and
this round's premise, that the two were confounded, was itself wrong.

**F_c is genuinely incomplete.** Not by a little: it leaves rms 0.052 on values with no measurement error in
them, and it **systematically over-predicts** — mean signed residual −0.00917.

Recorded, never scored (§7.3): the best-fit a on exact values is **1.26225** against the frozen **1.29980**,
2.9 % lower. Some of F_c's misfit here is that its constant was fitted on sampled cells at other depths.

**A registration flaw, stated.** R1's thresholds were multiples of a reference measured on a *different grid* —
exp_07's y = 360…92,160 with 14 moduli, against this round's y = 23…401 with 21 moduli. The ratio 1.715
therefore does **not** mean "the residual grew". It does not change the verdict: the KILL is rejected by a
factor of 5.7, and the exact-vs-sampled control above settles the question without reference to any other grid.

## R2: the guard fired, correctly, on a design limitation of mine

The effect is plainly present — corr(residual, ω) = **+0.608**, t = **+7.07** — and the sign matches exp_05
(+0.686) and exp_06 (+0.597). But §4's guard required ω to take ≥ 3 distinct values with ≥ 10 cells each, and
the pool delivers **two**: ω = 1 with 61 cells, ω = 2 with 26.

That is my fault, not the data's. I capped q ≤ 30 so the inclusion–exclusion truncation would stay affordable,
and moduli under 30 are almost all prime powers or products of two primes. The cap killed the variation the
relation needed. G7 established the *statistical* power correctly; it did not check that ω would **vary**.

Compounding it: on this pool ω correlates with q at **+0.468**, so even with more values the two would be hard
to separate — exp_06's §7.4 worry, arriving here for real.

**INCONCLUSIVE as sealed.** The ω structure is neither confirmed nor refuted by this round.

## R3: truncation is not the driver

rms 0.05220 at G = 40 against 0.05374 at G = 36 — a 2.96 % change against a 20 % bar. The 18 high-tail cells
were excluded before scoring under §7.1. The instrument is trustworthy at these depths.

## What this round establishes

**F(φ(q)/ḡ) = tanh(1.2998·x) is incomplete, and the gap is real.** Every prior round measured F against
sampled δ and could not rule out that its ~0.03–0.04 residual was noise. On exactly derived values the
residual is 0.052 and the measurement precision is 0.00016. There is a genuine object F does not capture.

That reframes rounds 5 through 8. Their verdicts stand — each was scored against thresholds fixed in advance —
but the residuals they reported were never noise, and treating them as partly noise was too generous to F.

## What is not claimed

That the residual is the ω(q) structure — R2 could not test it here. That F's *form* is wrong rather than its
constant — §7.3's 1.26225 says part of the misfit is the frozen a. Anything at the depths exp_05–exp_08
measured: this round reaches y ≤ 401 only. Anything about the primes: this is the loop.

## Forward note

1. **Re-run R2 on a pool where ω varies.** It needs q with ω = 3 (30, 42, 60, 66, 70, 78, 84, 90) and enough
   of them, which needs G large enough that their multiples are captured — the cost is 2^(g/2), so this wants
   the DP over residue patterns rather than brute-force subsets. That is the enabling piece.
2. **Fit a on exact values and re-test.** With δ exact, a can be determined without sampling error. If the
   residual stays ~0.05 with the best a, the *form* is wrong; if it drops sharply, F's shape is right and only
   its constant was mis-set. That is a one-line experiment and it separates two very different conclusions.
3. **The residual is now the object.** It is real, it is 0.052, it survives exact computation, and nothing in
   this study explains it.
