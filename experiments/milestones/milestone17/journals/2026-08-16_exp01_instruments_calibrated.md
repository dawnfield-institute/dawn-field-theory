# 2026-08-16: exp_01 — the criticality instruments are calibrated

No DFT system is measured until this passes. It now passes.

## Result

2D site percolation, L = 32/64/128, 100 samples per point, 25 points over p ∈ [0.50, 0.70]:

| test | measured | exact | error |
|---|---|---|---|
| **T1** p_c from the finite-size crossing | **0.5917** | 0.5927460 | **0.0011** |
| **T2** χ_max ~ L^(γ/ν) | 1.6233 | 1.7917 | 0.168 (R² 0.990) |
| **T3** n_s ~ s^−τ at p_c | 1.8490 | 2.0549 | 0.206 |

**p_c recovered to 0.2%**, located purely by where the spanning-probability curves for
different L intersect — the instrument was not told where to look.

T3's second half matters as much as its first: R² is **0.996 at p_c against 0.967 at p = 0.45**.
The instrument finds a power law where there is one and not where there isn't. An instrument
that found power laws everywhere would have found nothing.

The exponents run ~10% high in the same direction, which is the expected finite-size bias at
L ≤ 128 with three sizes. Quantified rather than hidden; it means exponents measured on a DFT
system carry the same bias and should be compared like-for-like rather than against textbook
values.

## Three bugs, each caught only because the answer was known

**The crossing finder took the minimum spread across L.** Exactly wrong: deep inside either
phase every curve saturates at the same constant — spanning probability is 1.0 for all L well
above p_c — so the spread is *identically zero* over a whole region and beats the real crossing.
It returned p_c = 0.70 with "spread 0.000". Saturated agreement is not a crossing. Now located
by the sign change of (largest L − smallest L), with saturated ends excluded first.

**The cluster-size distribution included the largest cluster.** At p_c the incipient spanning
cluster is a separate object from the scaling distribution and bends the tail away from the
power law. τ read 1.46 against an exact 2.05 until it was dropped — the same reason
`susceptibility` excludes it, which I had implemented correctly there and not here.

**Per-sample fits threw away the statistics.** One L=48 lattice has too few large clusters to
constrain the tail, so per-realisation slopes are dominated by the well-populated small-s bins.
Pooling all realisations before binning moved τ from 1.61 to 1.85.

## Why this experiment exists

The preceding round produced seven instrument faults — a box-counting estimator returning
D = 2.000 for a filament, a tautological filament fraction, a checkerboard and a speckle field
both scoring `is_web = True`, a force fitter blind in many-body, a conservation scan certifying
a dead system, and percolation swinging 18× with binning resolution. **None was caught by a
statistic taken at face value.** Every one was caught by a reference whose answer was already
known.

So the first experiment of this milestone deliberately measures nothing new. Three more bugs
found here, in instruments written by someone who had just spent a day being caught out by
exactly this, is the argument for the practice rather than against it.

## The test I proposed was insufficient, and calibration showed it

exp_02 was going to ask "is the steady-state event-size distribution a power law?" — the
self-organized-criticality test, needing no tuned control parameter. Building the discriminator
first showed that question cannot answer what it was meant to.

`power_law_or_exponential` is exact on pure forms: synthetic s^-2 gives R² 1.0000 against
0.6106, synthetic exp(-s/50) the reverse. But **sub-critical percolation at p = 0.40 also reads
"power_law"** (0.9511 vs 0.8996) — and correctly, because a sub-critical cluster distribution is
a *truncated* power law, n_s ~ s^-tau exp(-s/s_c), with the s^-tau regime still present below
the cutoff.

**So "is it a power law" does not distinguish critical from sub-critical.** Only the cutoff
does, and the cutoff scale is exactly what `susceptibility` measures. `cutoff_scaling` is the
real test: chi ~ L^(gamma/nu) at criticality, chi L-independent away from it.

Calibrated on percolation at L = 32/64/128:

| system | exponent | verdict |
|---|---|---|
| at p_c = 0.5927 | **1.603** (exact γ/ν = 1.792) | scale-free |
| p = 0.50, sub-critical | 0.585 | scale-free |
| p = 0.40, sub-critical | 0.225 | characteristic-scale |

p = 0.50 reads scale-free while being below p_c, and that is right rather than wrong: at these
L its correlation length is comparable to the box, so it sits inside the critical region.
Finite systems look critical near p_c. **The exponent is the measurement; the verdict label is
crude** — it runs 0.23 → 0.58 → 1.60 approaching p_c and converges on γ/ν.

Recorded because it changes exp_02's design before any run was spent on it: the SOC test is
the **finite-size scaling of the avalanche cutoff**, not the shape of a single distribution.

## What is now available

`core/criticality.py`: `order_parameter`, `susceptibility`, `spans`,
`cluster_size_distribution`, `pooled_cluster_distribution`, `finite_size_crossing`,
`fit_power_law`, `fit_exponential`, `power_law_or_exponential`, `cutoff_scaling`,
`site_lattice` — all N-D, all exercised against a system with exact answers.

That closes four of the seven capability holes named at founding. Still missing: correlation
length as a *critical* quantity with its own finite-size scaling (the length exists in
`reality-engine/.../structure.py` but has never been scaled), and an edge-of-chaos classifier
as a reusable instrument.

## Next

exp_02 — apply the calibrated instruments to a DFT structure-forming system and ask Q1: does a
critical point exist at all?

Design settled by the calibration above: measure the **avalanche cutoff's scaling with system
size**, not the shape of one distribution. Run the engine at several L, pool avalanche sizes
over the steady state, and fit chi against L. An exponent near γ/ν says scale-free; an exponent
near zero says the system has its own characteristic scale and is not critical.

This needs no control parameter, which sidesteps the problem that `sec_balance` was a *choice*
rather than a derived analogue of occupation probability — picking it wrong would have produced
"no crossing" for a boring reason.

One open engineering question: the engines live in `reality-engine` and these instruments live
here. Cross-repo import or a shared module is a decision to make deliberately rather than by
accident.
