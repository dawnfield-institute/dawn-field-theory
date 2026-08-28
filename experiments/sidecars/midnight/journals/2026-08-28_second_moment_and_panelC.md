# The second moment is where the signal lives — and Panel C's interpretation isn't

**2026-08-28.** Two exploratory rounds (STANDARDS §2.8 — no pre-registration, no thresholds,
no scoring). Midnight ran for the first time since the August reorg broke it (PR #176).

## 1. exp_24 — Panel C survives its adversarial replication; its interpretation does not

Midnight's forward program named "adversarial replication of z-immune survivors". Panel C was
recorded single-pass and unreplicated.

- **Reproduces exactly**: 6437 straddling vs 9000 non-straddling pairs, medians 0.4848/0.4253,
  Mann-Whitney p = 1.8e-9.
- **The separation confound is real in the covariate but has no path to the outcome.**
  Straddling pairs are **2.6× more separated** in |ΔN| — but corr(|ΔN|, |ΔEW|) = **+0.0075**.
- **Survives |ΔN|-matched stratified permutation**: z = +4.11, p = 0.0007. Leave-one-bin-out
  keeps max p = 0.0033, so no single bin carries it.
- **But integer N is not special.** Offsets of +0.20–0.26 reproduce it with the identical
  19-bin structure; across a 50-point offset sweep integer N sits at empirical p = 0.100.

**The observation stands and is not a separation artifact. The claim that it evidences cascade
transitions at integer N does not.**

## 2. exp_25 — the second moment carries structure, in the frame-clean channel

Almost every surviving Midnight signal lives in a width, spread or shape and almost none in a
mean (exp_05 EW spread, exp_03 widths, exp_06 FWHM discrepancy, exp_07 CIV b spread, exp_08D
doublet shape). If an unmodelled primitive appears as excess variance, that is the expected
signature — and those results are the *primary* channel, not weak versions of a mean-shift.

Test: after removing a quadratic trend in z, does residual **variance** still track cascade
disequilibrium? Control: shuffle z, 200 draws.

| channel | ρ | z | p | skew | ex-kurt |
|---|---|---|---|---|---|
| **FWHM discrepancy** | **+0.585** | **+3.71** | **≤0.005** | +1.43 | +1.74 |
| FWHM ratio | +0.483 | +2.93 | ≤0.005 | +6.61 | +82.4 |
| log FWHM 2796 alone | +0.237 | +1.61 | 0.139 | | |
| log FWHM product | +0.034 | +0.19 | 0.871 | | |
| log EW 2796 | −0.050 | −0.43 | 0.816 | | |

**The signal is in the doublet *discrepancy*** — the difference between two lines of the same
doublet, in the same absorber, at the same redshift. Not either width alone, not the product.
That is a **relational within-scope quantity** (exp_23's frame-clean form), which is what the
invariant-registration rule says should survive.

**Both moments agree.** exp_06 found the *mean* discrepancy anticorrelates with disequilibrium —
lines lock together at transitions. This finds the *variance* rises away from them. Independent
moments, same physical statement.

### Limits, stated

- **p ≤ 0.005 is the permutation floor** (200 draws), not a measured value.
- **The FWHM ratio is discounted**: skew 6.6, excess kurtosis 82 — outlier-driven. The
  discrepancy is well behaved at 1.43/1.74.
- **Not blind.** Channels were chosen because exp_07 reports they survived detrending. This is
  a targeted test of an existing claim, not a discovery.
- The nulls are what make it credible: EW spread dead exactly as exp_07 found, single widths
  dead, product dead. A fishing expedition does not produce that pattern of specific nulls
  beside a specific survivor.

### A scope error, recorded

The first pass ran on **EW spread — the one channel exp_07 had already killed** — and returned
a clean null (ρ = −0.05, p = 0.82, with the shuffled null confirming it was empty rather than
underpowered). Correct scope is the channels that *survived*. Same class as probing a global
observable with a local initial condition; see `mind/feedback_scope_match_probe_to_observable`.

## Next

Replicate the discrepancy result in an independent ion (CIV is local, and exp_07's Doppler-b
spread survived there at p = 0.004). One catalogue and one channel is not enough to lean on.
