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

## What is now available

`core/criticality.py`: `order_parameter`, `susceptibility`, `spans`,
`cluster_size_distribution`, `pooled_cluster_distribution`, `finite_size_crossing`,
`fit_power_law`, `site_lattice` — all N-D, all exercised by the calibration.

That closes four of the seven capability holes named at founding. Still missing: correlation
length as a *critical* quantity with its own finite-size scaling (the length exists in
`reality-engine/.../structure.py` but has never been scaled), and an edge-of-chaos classifier
as a reusable instrument.

## Next

exp_02 — apply the calibrated instruments to a DFT structure-forming system and ask Q1: does a
critical point exist at all? The particle substrate is the natural first target, since it is the
one system in the corpus that produces connected structure at any setting. Registered
prediction stands: if a crossing exists, it should sit near Ξ.
