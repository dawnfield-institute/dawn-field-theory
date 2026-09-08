# The Prime Gap Ledger: the loop, the gap and the delta

**Status**: active · **Founded**: 2026-09-07 · **Score**: 12/21 (exp_01 0/3, three INCONCLUSIVE by the sealed rules — `journals/2026-09-07_exp01_outcomes.md`; exp_02 3/4 — `journals/2026-09-07_exp02_outcomes.md`; exp_03 2/2 — `journals/2026-09-07_exp03_outcomes.md`; exp_04 EXPLORING — no seal, no score, `journals/2026-09-07_exp04_exploring_the_loop_depth_profile.md`; exp_05 3/3 — `journals/2026-09-07_exp05_outcomes.md`; exp_06 2/3 — `journals/2026-09-07_exp06_outcomes.md`; exp_07 1/3 — `journals/2026-09-07_exp07_outcomes.md`; exp_08 1/3 — `journals/2026-09-08_exp08_outcomes.md`)

## The thesis

A prime gap is a cascade that truncates itself. The integers strictly between two consecutive primes are
each divisible by some prime up to √p, so the gap is the sieve by those primes running until it fails to
cover the next integer: a local instance with two ends, a cut-open loop rooted at p. It is not a Möbius
topology, and the only Möbius in the sieve is Möbius the function, a balanced signing of the divisor
lattice.

The global object is the **loop of units mod the primorial** P(y) = ∏_{p ≤ y} p: bounded, boundaryless,
naturally written in primorial (Chinese-remainder) coordinates — "the structure arithmetic forms
around". Its gaps are the *unit gaps*; their frequencies are the Hardy–Littlewood weights. The y-rough
integers of a window [x, 2x) are the loop read locally at depth u = log x / log y, and once y ≥ √(2x)
they are exactly the primes. **The delta between the local read and the loop, as a function of u, is the
object of this study** — not the gaps themselves. In the corpus's words: the loop is the global ledger,
the gap is the local cascade, and the delta is the third thing.

The known theorems that pin the words are gates, not results: Mertens (the loop's density
∏(1 − 1/p) ≈ e^{−γ}/log y), Buchstab (the local/global density ratio is e^{γ}ω(u), = e^{γ}/2 at u = 2 —
the classic 2e^{−γ} delta), Maier (the loop leaks into short intervals), Hardy–Littlewood (the loop's
pair weights), Lemke Oliver–Soundararajan (consecutive primes' residues are biased, the diagonal
repelled, the bias vanishing globally). What is registered is what none of them states.

## Registered relations (exp_01; `journals/2026-09-07_exp01_registration.md`)

| | relation | verdict |
|---|---|---|
| R1 | the mean-rescaled gap **shape** delta Δ(u; m) obeys a drift law across decades (successive-decade differences shrink at every evaluable live u) | **INCONCLUSIVE** — at u ≥ 3 the shapes agree within noise in every decade (converged, which the rule cannot confirm); at u ≤ 2.5 the metric is a bin-edge artifact for lattice gaps |
| R2 | the residue bias is a truncation phenomenon: the consecutive-survivor diagonal deficit δ(u; m) obeys the same drift law, decreases across decades at u = 2 (the primes), and meets the loop's own value at the deepest live u | **INCONCLUSIVE** — (a) CONFIRM at all six live u, (b) CONFIRM (0.176 → 0.103 from 10⁵ to 10⁹), (c) meets to 1e-4 in every decade but two floors sit below 1e-4 |
| R3 | the termination depth (max least-prime-factor over a gap's interior) against a matched null of random composites is scale-free | **INCONCLUSIVE** — ρ = 0.93, 0.95, 0.95, 0.96 (10⁶..10⁹), converged within floors from 10⁷ |

Live cells are those where the loop's period exceeds the window; where it does not, window = loop
exactly (a gate). Floors are half-splits, reported per cell.

## Round 2 — exp_02, the position of the read (`journals/2026-09-07_exp02_registration.md`, sealed ae67f522): 3/4

Peter's reading of round 1's residual: the uniform loop is the loop read at a *random* position; the primes are the
loop read at its **origin** (the arc below y², where every residue is the integer's own name and all moduli agree —
the fully actualized identity). Look completely locally and completely globally, and forget the delta as a thing:
**one object, read at two positions; the delta is the position.** Checked in exploring mode (the bias climbs from the
primes' value to the loop's within half a unit of depth and plateaus to 10⁵⁰), then registered.

| | relation | verdict |
|---|---|---|
| R1 | the residual at the primes' arc is scale-free and decreases with the read's depth | **CONFIRM** — 0.0101, 0.0102, 0.0087 at 10⁷–10⁹ within 10 %; decreasing at every resolved step over u = 2 → 2.25 |
| R2 | the sign flip: where Buchstab's density ratio exceeds one the loop must *under*-predict the bias | **CONFIRM** — negative at 3–6σ at u = 2.75 and 3 in both depths (W = 200); the sign of the residual tracks the sign of the measured deficit cell by cell, including the one positive cell |
| R3 | proportionality c = ε/d across the resolved cells | **INCONCLUSIVE** — c ≈ 0.10 on all 14 cells (0.087–0.115 where well resolved), but the seal's CONVERGED and KILL clauses both fire (two noisy flip cells differ by a factor 2.1); the seal gave no precedence |
| R4 | shell independence (Peter) against Andy Farmer's ultrametric tranche (deeper p-adic shells) | **CONFIRM** at the sealed 15 % — with a consistently-signed ~10 % deeper-shell modulation recorded (r₉ < r₃ at 3.4σ at 10⁹): the tranche modulates the residual, it does not make it |

**The sentence:** the Lemke Oliver–Soundararajan bias of consecutive primes equals the primorial loop's exact rational
bias at √(2x) minus one tenth of Buchstab's density deficit, sign included. The residual is the position of the read,
and it changes sign where ω(u) crosses e^{−γ}. Full curves in the outcomes journal.

## Round 3 — exp_03, the density-matched loop (`journals/2026-09-07_exp03_registration.md`, sealed ff63f8d5): 2/2

Round 2's shell modulation was re-posed by the design review before it was registered: the uniform loop's own δ_q
falls with depth as (log y)^{−β_q} with a modulus-dependent β_q, and the primes' arc is the loop read at the effective
depth y_eff whose exact Mertens product equals the arc's density — that reproduces every residual of rounds 1–2, shells
and flip included, with no free parameter. The registered quantity is what the model leaves: ρ_q = r_q − r_q^eff, the
departure of the primes' p-adic gap profile from the density-matched loop (Lemke Oliver–Soundararajan evaluated exactly).

| | relation | verdict |
|---|---|---|
| R1 | ρ_q = 0 at the primes' arc (10⁹, 10¹⁰; moduli 3, 9, 4, 8, 16, 5, 7) | **CONVERGED** — 14 cells, none beyond 1.5σ, none using more than half its tolerance; no shell-ordered difference resolved (±0.001 in r at 10¹⁰) |
| R2 | ρ_q = 0 along the curve and through the flip (u = 2.1, 2.25, 3 at 10¹⁰), with the flip's sign predicted from the depth shift alone | **CONVERGED** — 21 cells, none beyond 1.7σ; at u = 3 the predicted negative residual matches at every modulus (e.g. q = 3: −0.0069 observed, −0.0067 predicted) |

**The depth shift is Buchstab's density ratio exactly**: log y_eff / log y = 1/ratio to four decimals in all fifteen
cells. **The sentence:** the consecutive-prime residue bias at every modulus, shells included, equals the primorial
loop's bias read at the density-matched depth y^{1/ratio}; the delta between the primes and the loop is the position
of the read, and nothing else, to one part in a thousand at 10¹⁰. Andy Farmer's tranche beyond position: none resolved.
Forward corrections to round 2 are filed in its outcomes (the q = 10 channel was an identity; R4's reading superseded).

## Round 4 — exp_04, the loop's own depth profile (EXPLORING; no seal, no score)

Asked whether the loop's smooth-depth profile δ_q(y) ~ (log y)^{−β_q} has a closed form, which would remove
round 3's last fitted input. It does not close, and the round is filed as a null with three results that stand
(`journals/2026-09-07_exp04_exploring_the_loop_depth_profile.md`):

- **δ_{2q} = δ_q exactly, for odd q** — proved (units are odd, so CRT fixes the residue mod 2q from the residue
  mod q; the transition matrix is a padded relabelling) and verified 42/42. Round 3's "δ₁₀ ≡ δ₅" forward
  correction is the q = 5 instance of this. **q and 2q for odd q are never independent cells.**
- **δ_q → 0, not 1 − φ(q)/q.** The null is a product of marginals over the φ(q) *unit* classes, so independence
  gives exactly zero; δ₃ passes through 1/3 and keeps falling. δ_q is distance from independence.
- **The power law holds over four decades** — R² ≥ 0.994 across twelve depths, y = 60 … 141,422, no break and no
  loss of monotonicity anywhere in range.

**A candidate was raised and killed by its own control.** On the six-modulus anchor set β appeared to sort by
φ(q) rather than q (q = 5 cyclic and q = 8 Klein agreeing at 0.35σ; one common β rejected at χ²/dof = 8.55).
It is false. Holding φ *exactly* fixed and varying δ — possible because φ is non-monotone in q — β is flat in
no class (χ²/dof = 10.9 to 332.6; at φ = 24 it runs 0.332 → 0.142 across five moduli of equal totient). In the
anchor set φ and the δ-level were collinear at 0.995 and the correlation was read off the wrong variable.

δ-level does not absorb everything either: q = 11 and q = 15 sit at δ = 0.541 and 0.554 but β = 0.531 and
0.759, about 14σ apart. Pooled over 44 moduli β falls monotonically with δ (corr −0.84 / −0.97, no sign
change). **β is dominated by the δ-level with residual modulus structure, and at collinearity 0.977 neither is
isolable here.** Gate **G7 FAIL** independently: held-out classes never clear saturation at reachable depth
(matching δ = 0.40 at q = 21 needs y ~ 10^11.4).

The second candidate from round 3's forward note — the depth-shift law at moduli with a prime factor above y —
is **vacuous** and was not run: the mean gap e^γ log y is below y at every depth, so those moduli are saturated
by construction and the mechanism never runs.

## Round 5 — exp_05, the collapse (`journals/2026-09-07_exp05_registration.md`, sealed 5d6acfc6): 3/3

Round 4's failure was diagnostic: β is a local slope on a curve approaching a bound, so it is set by where on
the curve a cell sits and can never be a constant of the modulus. **β was the wrong object.** Fitting δ itself,
the whole family is one curve read at **φ(q)/ḡ** — the size of the unit group against the mean gap ḡ = e^γ log y.

| | relation | verdict |
|---|---|---|
| R1 | the collapse generalises: F fitted on 30 training moduli predicts the 14 held-out | **CONFIRM** — held-out rms 0.0346 vs training 0.0564 (ratio 0.614; CONFIRM ≤ 1.5) |
| R2 | the variable is φ/ḡ, not q/ḡ (positive control) | **CONFIRM** — 0.0346 vs 0.0986, a factor 2.85, at 9.68σ; orderings Spearman 0.852, below the 0.99 guard |
| R3 | the residual is structured by ω(q), sign registered in advance | **CONFIRM** — corr +0.686, t = +7.78, while corr with q is −0.027 |

**The sentence:** the loop's departure from residue-independence, at every modulus and depth, is one curve read
at φ(q)/ḡ; depth and modulus enter only through that ratio, and the only thing left over is **how many distinct
primes build the modulus** — the singular-series shape ∏_{p|q}(p−1)/(p−2), not the modulus itself.

Registered threat §7.2 fired and was handled as sealed: at y = 90 the measured mean gap missed e^γ log y by
2.63 %, over the 2 % bar, so those 44 cells are recorded and not scored (150 train + 70 held-out over five
depths remain). Bin sensitivity recorded, never scored: 0.0446 at 12 bins, 0.0364 at 24, against 0.0346 sealed
at 18. A postdiction confirmed out of sample — found in exp_04's data, tested on fresh depths, a fresh seed and
a mechanical split. Says nothing yet about the primes.

## Round 6 — exp_06, do the primes inherit the collapse? (`journals/2026-09-07_exp06_registration.md`, sealed 863d8081): 2/3

**The study's first a priori prediction.** F was sealed in exp_05 fitted on loop cells only — no prime involved
at any stage — and y_eff is *solved* from the arc's measured density, so δ_q(primes) = F(φ(q)/ḡ(y_eff)) has
**zero free parameters**. Gate G2 confirmed the read is genuinely the primes: counts match sympy's `primepi`
difference exactly at all three decades (606,028 / 5,317,482 / 47,374,753).

| | relation | verdict |
|---|---|---|
| R1 | the primes inherit F, nothing tuned | **CONFIRM** — rms 0.0412 against the sealed tolerance 0.0692 (KILL 0.1038), over 117 scored cells |
| R2 | the depth shift does the work (positive control) | **INCONCLUSIVE** — 0.0412 with ḡ(y_eff) against 0.0398 with ḡ(y), −0.43σ; unresolved, point estimate marginally favouring the *unshifted* depth |
| R3 | the ω(q) structure persists at the primes | **CONFIRM** — corr **+0.597**, t = +7.98, against corr with q of −0.030 |

**The sentence, with its limit:** a curve fitted entirely on the loop predicts the consecutive-prime residue
bias at 39 moduli across three decades to rms 0.041 with nothing tuned — but because ḡ(y) predicts as well as
ḡ(y_eff), this establishes that the primes lie on F *at approximately the right place*, **not** that round 3's
depth shift is what puts them there. A systematic offset of **+0.0115** (28 % of rms, 24× the median cell SE)
is present and unexplained.

**Instrument lesson filed:** R2's registered guard required the shift to *exist* (>0.02; it was 0.10). The
right guard is a *power* condition — F(x_eff) − F(x_y) must exceed the residual. It was 0.0255 against 0.0308,
so R2 was dead at the seal and its guard could not see it. Same class as exp_04's G7 saturation failure.

## Round 7 — exp_07, the closed form and the depth forecast (`journals/2026-09-07_exp07_registration.md`, sealed b43dca3f): 1/3

Replaced the 18-bin F with a **one-parameter closed form** `tanh(a·φ(q)/ḡ)`, constrained by the asymptotics
(0 at the origin, linear leaving it, → 1), `a` = 1.2998 fitted on the loop's training moduli and frozen at the
seal. Tested against a fresh decade: the 427,154,205 primes of [10¹⁰, 2·10¹⁰).

| | relation | verdict |
|---|---|---|
| R1 | the closed form predicts the 10¹⁰ primes, zero further freedom | **CONFIRM** — rms 0.0402 against the sealed 0.0457 |
| R2 | it beats the 18-bin F on those cells | **INCONCLUSIVE** — 0.0402 vs 0.0636, but 2.14σ, under the sealed 3σ |
| R3 | λ(m=10) within 0.15 of the forecast 0.652 | **INCONCLUSIVE** — measured 0.6446, but the minimum is not resolved below y_eff |

**R3 is the instructive one.** The forecast landed to 0.0074 against a tolerance of 0.15 — and it does not
count. The registered §4 guard requires the scan minimum to sit below *both* endpoints by more than the
bootstrap spread; it clears y by 0.0051 but y_eff by only 0.0015, under a spread of 0.0039. λ is unresolved to
about ±0.3, so hitting the forecast carries no information. **The apparent bullseye is a broad minimum, not a
confirmed prediction**, and the guard is the only reason it is not written up as one.

Two corrections to earlier rounds came out of it. exp_06's unexplained **+0.0115 offset halves to +0.0054**
under the better estimator — much of it was my binning, not the object. And the registration's claim that the
18-bin F was a poor estimator is **only half right**: within its own fitted domain it scores 0.0457 against the
closed form's 0.0421, a modest gap. Its real defect was having a domain at all, which cost exp_06 five moduli
per decade. Recorded, never scored: the primes prefer `a` = 1.3465, 3.6 % above the loop's.

## Round 8 — exp_08, is the coherence transition just the depth moving? (`journals/2026-09-08_exp08_registration.md`, sealed 1fd25acf): 1/3

Round 2 found in exploring mode that the bias climbs from the primes' value to the loop's "within half a unit
of depth". This round asked whether that transition is an independent phenomenon or **entirely the effective
depth moving**: δ_q(u) = F_c(φ(q)/ḡ(y_eff(u))), F_c sealed in exp_07, y_eff solved from each position's density,
**zero free parameters**. Eleven positions, N = 10⁷ to 8·10²¹.

| | relation | verdict |
|---|---|---|
| R1 | position is depth | **CONFIRM** — rms 0.0352 against the sealed 0.0533; mean signed residual ≈ 0 at **every** position |
| R2 | the transition is universal across moduli | **INCONCLUSIVE** — sd of C_q(u) = 0.1624 against a 0.15 bar (KILL 0.35) |
| R3 | positive control: tracking beats a fixed depth | **INCONCLUSIVE** — 0.0352 vs 0.0364, only 1.75σ |

**The mechanism, and its limit.** Round 2's coherence transition now has a cause: it is Buchstab's ω(u)
relaxing to e^{−γ}. The gates measured the density ratio tracking e^γω(u) to ≤ 0.4 % at every scored position,
the depth shift decaying from +0.119 at u = 2 through zero at u ≈ 2.5, then ringing slightly negative and
damping — the classical oscillation, arriving in a measurement that never assumed it. And R1 tracks it with no
bias anywhere across fifteen orders of magnitude in N. But **R3 could not show that tracking beats ignoring**,
so R1 rests on the fit being good rather than on the control separating.

**Third power-estimation failure in three rounds, each different.** G7 promised 4.07σ and delivered 1.75:
6 of 11 positions sit at y_eff ≈ y and dilute the contrast, and the bootstrap resampled 484 cells as
independent when the 44 at each position share one window. Recorded with the earlier two (per-cell vs
aggregate, exp_06 `aace9b3d`; an under-sampled grid, this round's own first gate run). The common fault is
computing power against an idealisation of the design rather than the design as built.

## What round 1 recorded (not claimed — its seal did not score it)

**The window's residue bias depends on the depth y alone.** At every depth from y = 13 to y = 4,000 the
consecutive-survivor deficit mod 3 in the window equals the primorial loop's deficit at that depth to a
thousandth, whatever the decade (y = 100: 0.2217, 0.2243, 0.2225, 0.2232, 0.2231 across five decades against
the loop's 0.2231). **At the primes themselves the loop over-predicts by a constant:** at y = ⌈√(2x)⌉ the
loop's deficit exceeds the primes' by 0.0093, 0.0098, 0.0099, 0.0088 for x = 10⁶, 10⁷, 10⁸, 10⁹. So the
Lemke Oliver–Soundararajan bias is the loop's bias at the truncation depth minus a scale-free residual
ε(u) — zero for u ≥ 2.5, about 0.003 at u = 2.25, about 0.0095 at u = 2 — the local/global delta of the
transition structure, the same shape as Buchstab's delta for the density. The loop's exact deficits at
q = 3 are 5/12, 223/552, 2860783/8291520, … (k = 3, 4, 6, …). Gap interiors are smoother than random
composites of their decade (ρ ≈ 0.95). These are the objects of round 2, with a CONVERGED verdict class,
a lattice shape metric and a tolerance on "meets" — the three instrument lessons of this round.

## Lineage, and what is deliberately not reused

The corpus's earlier prime work is cited, not built on. `sec_prime_manifold`'s 1/φ threshold is a frame
artifact (STANDARDS §2.9's worked instance: the residue class chooses the constant).
`asymmetric_conservation`'s "PAC exact at every sieve step" defines composites as N − 1 − primes, an
identity (§2.8). Milestone 3's "prime cascade reachability" is a wave-coverage model whose Fibonacci-gap
clause has no matched null. `prime_growth_dynamics_v2` exp_06 found the gap-6 hub against a Cramér
reference at N = 5·10⁵. None of φ, Ξ or Fibonacci enters this study; nothing here leans on those
statistics.

## Scripts

| Script | Purpose |
|---|---|
| `core/rough.py` | the instrument: odd sieve; one segmented sieve for the window and for the loop (CRT-uniform residues); loop enumeration k ≤ 9; gap and shape histograms; Mertens product; Buchstab ω(u); transition matrices and the diagonal deficit (exact rationals on enumerated loops); the least-prime-factor table and termination depths with the matched null |
| `scripts/exp_00_gates.py` | G1–G8, run before the seal (loop counts; CRT pair formula; champion 6; Buchstab; Mertens at u = 2; sampler vs enumeration; LO–S sign; periodic cells) |
| `scripts/exp_01_cascade_truncation.py` | round 1: R1–R3 on decades m = 6..9 (4, 5 recorded), CONFIRM / KILL / INCONCLUSIVE, append-only timestamped results |
| `scripts/explore_r2_position_along_the_loop.py` | exploring (disclosed): a fixed-depth read slid from the origin outward, 10⁴…10⁵⁰ |
| `scripts/exp_02_gates.py` | round 2 gates G1–G6 (arc-integral Buchstab; fresh-seed loop vs round 1; de-trended scatter; exact reproducibility; the equidistribution plateau; prime-power lifts vs enumerated loops) |
| `scripts/exp_02_position_of_the_read.py` | round 2: R1–R4 at positions u_top = log 2N / log y from 1.75 to 7, depths m = 6..9, W = 200 at the flip cells, verdicts CONFIRM / CONVERGED / KILL / INCONCLUSIVE |
| `scripts/exp_03_gates.py` | round 3 gates G1–G8 (lift invariance; exact reproducibility of round 2; the 10¹⁰ decade's count against sympy's π; chunk carry; fresh loops; de-trended scatter; the density-matched prediction against round 2; the y_eff solver) |
| `scripts/exp_03_density_matched_loop.py` | round 3: ρ_q at the primes' arc and along the curve, m = 8..10 (the 10¹⁰ decade chunked with residue carry), loops of 200 windows at y and at y_eff; precedence KILL → CONVERGED → INCONCLUSIVE |
| `scripts/explore_r4_loop_depth_profile.py` | exploring (disclosed): the loop's own δ_q(y) — `--mode enumerated` gives exact rationals on the enumerable loops k ≤ 9, `--mode sampled` the twelve-depth sweep to y = 141,422 |
| `scripts/explore_r4_qsweep.py` | exploring (disclosed): δ_q over q = 3..60 (2·odd excluded by the theorem) at three depths — supplies the two controlled comparisons, β at fixed φ with δ varying and β at matched δ with the modulus varying |
| `scripts/explore_r4_cyclic_convention.py` | exploring (disclosed): the enumerated loop's transition matrix should be cyclic — the linear one injects φ(P)/2 − 1 into every denominator; `2860783/8291520` is `497/1440` |
| `scripts/exp_05_gates.py` | round 5 gates G1–G8 (all PASS: the record; the δ_{2q} identity; sampler unbiasedness; seed reproducibility; a mechanical disjoint split; no extrapolation; ḡ = 1/Mertens; the fit fully specified) |
| `scripts/exp_05_collapse.py` | round 5: R1–R3 on 44 moduli × 6 fresh depths, F binned on training moduli only, paired bootstrap for R2, numpy-only |
| `scripts/exp_06_gates.py` | round 6 gates G1–G8 (all PASS; the decades read for counts, densities, y_eff and transition counts ONLY — no δ formed before the seal) |
| `scripts/exp_06_primes_inherit.py` | round 6: the sealed F evaluated at φ(q)/ḡ(y_eff) against the primes of 10⁷, 10⁸, 10⁹; zero free parameters |
| `scripts/exp_07_gates.py` | round 7 gates G1–G8 (all PASS; **G7 is a POWER gate** — the depth comparison resolves at 3.84σ by simulating the sampling spread over the actual cell count, which is exp_06's error fixed at the gate) |
| `scripts/exp_07_closed_form.py` | round 7: `tanh(a·φ/ḡ)` with `a` frozen from the loop, against the 10¹⁰ primes; the depth scan and its resolution guard |
| `scripts/exp_08_gates.py` | round 8 gates G1–G8 (all PASS; G6 **demoted to recorded** — it gated on the per-cell criterion invalidated by exp_06's correction; the u-grid was resampled after the first run under-sampled the transition) |
| `scripts/exp_08_position_is_depth.py` | round 8: the sealed F_c at y_eff(u) across eleven positions, N = 10⁷ … 8·10²¹ |
| `scripts/exp_04_gates.py` | round 4 gates G1–G8 (the record's exact rationals; the δ_{2q} = δ_q theorem; class independence; mean gap = 1/Mertens; sampler unbiasedness; reproducibility; **G7 saturation feasibility — FAILED, no seal written**) |

## Discipline

Registered before running (§2.7): relations only, thresholds fixed, gates on the record before the seal,
outcomes cite the seal's commit, results append-only, floors reported, a script that diverges from the
seal is corrected toward the seal. Postdiction disclosed in the registration's §0.
