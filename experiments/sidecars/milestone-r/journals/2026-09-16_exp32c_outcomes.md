# exp_32c outcomes — 2/4, the kill fired: f carries nothing out of sample. And the edge is now measured on twelve seeds at κ_c = 1.1759 ± 0.0073.

**Registration:** `journals/2026-09-16_exp32c_registration.md`, sealed at **`ecfa0f6b`** and pushed
before any run. 42 runs at reality-engine `59f38a6`. **0 instrument gate failures. No threshold moved.**
Grid `results/exp_32c_predict_grid_fine_20260916_211*.json`; scored JSON beside it.

## What worked, first

**T4 PASS, 6/6.** W_p changes sign exactly once across κ ∈ [1.00, 1.30] on every fresh seed. That is
now **twelve seeds out of twelve** across three rounds. The edge is a single crossing, and this round
is the strongest evidence for it yet because it was collected while testing something else.

**The edge, on twelve seeds:**

| | mean | sd | SE |
|---|---|---|---|
| exp_32b, six seeds | 1.1695 | 0.0290 | 0.0119 |
| **twelve seeds** | **1.1759** | **0.0251** | **0.0073** |

The SE falls by 39 %. The mean moves up 0.0064, which is under one SE and not a shift. **SP1 held**:
the twelve-seed sd is 0.0251, inside the registered [0.020, 0.040].

## The kill: f carries no out-of-sample information

**T1 FAIL, and not narrowly.** The frozen line's RMS error is 0.04409 against the null's 0.02341 —
a skill of **−2.548**. Predicting a single fixed number beats the line by nearly a factor of two.

| seed | f | predicted | measured | error |
|---|---|---|---|---|
| 19 | 0.213757 | 1.1990 | 1.1471 | −0.0519 |
| 20 | 0.142599 | 1.1367 | 1.1963 | +0.0596 |
| 21 | 0.126312 | 1.1224 | 1.1679 | +0.0455 |
| 22 | 0.220468 | 1.2048 | 1.2049 | +0.0001 |
| 23 | 0.137870 | 1.1326 | 1.1897 | +0.0572 |
| 24 | 0.190808 | 1.1789 | 1.1878 | +0.0089 |

**T3 FAIL, 2/6** inside the ±2 residual-sd band. **T2 PASS** at ρ = +0.257 — and T2 is worth nothing
here, exactly as §5 registered: its false-positive rate under the null is 0.499. It is reported, not
believed.

The decisive number is not in the tests at all:

    r(f, kappa_c):   training six  +0.882   |   FRESH SIX  -0.064   |   all twelve  +0.333

**−0.064.** The relationship is absent on unseen seeds. The pooled +0.333 is what you get from
averaging a spurious +0.88 with a real zero, and is not evidence of a weak effect. **SP2 fails too**:
refitting on twelve gives a slope of 0.243 against the frozen 0.875, far outside the registered band.

## This failure was predictable and was predicted, four minutes before it ran

`internal/dft/2026-09-16_the_f_mechanism_map.md` — written while the sweep was running, from data
already on disk — asked whether f leaves a footprint anywhere between the initial condition and the
crossing. It does not: at κ = 1.00 over the training six, r(f, E_SEC/T) = −0.029, r(f, W_p) = −0.004,
r(f, slope) = +0.101. **f correlated with the endpoint and with none of its proximate causes**, which
is the signature of a six-point coincidence rather than a mechanism.

That map used no new runs and took four minutes. Had it been drawn before the seal, f would not have
been registered. **This is the concrete cost of pre-registering before mapping**, and it is now
written into STANDARDS §2.7.8 as "map before bit".

## The kill sentence was mis-written, and here is the correction

The seal says a T1 failure means "Milestone R stops trying to derive κ_c and treats 1.170 ± 0.012 as
the measurement it is." **That is a halt, not a next move, and it violates STANDARDS §2.7.9** (added
today, after this seal). A kill names what the programme does instead. exp_32b's did; this one
regressed. The seal stands as written — it is sealed — and this is its correction on the record.

**The next move, as it should have read:**

1. **T/|U₀|, the cumulative transfer at κ = 1**, is the surviving lead. On twelve seeds
   r(T/|U₀|, κ_c) = **+0.616**, down from +0.770 on six but not gone, and unlike f it sits between the
   initial condition and the crossing rather than only at the end. Its range is narrow (0.882–0.912),
   so the first question is whether that is resolvable at all — which is a §2.7.10 measurement, not a
   registration.
2. **Below it, the Zel'dovich deformation tensor** — the eigenvalue ordering of the initial field,
   which is the mechanism the substrate actually implements and which f was at best a crude proxy for.
   λ_max was tested and gave r = −0.655, but on a global maximum rather than a collapse-ordered
   spectrum, which is the wrong summary.
3. **Neither gets registered until it has a map.**

## What this does and does not touch

**It does not touch exp_32b's 4/4.** The edge exists, it is a single crossing, it does not move with
gravity across a fourfold range or with a doubling of particle count, and it is now measured on twelve
seeds more precisely than before. None of that depended on f.

**What it retires** is the predictability of κ_c *from this statistic*. It is not a verdict that κ_c's
spread is irreducible — that is a bearing the data cannot give. The spread of 0.0251 over twelve seeds
sits where it sits; f is inactive as a predictor of it, with its activation condition recorded as
"none found at this sample size". Two candidates remain and are named above.

## The honest ledger on my own priors

The seal registered T1 at ≈ 0.58, from a stated 55 % belief that f was real. The mechanism map moved
that belief down sharply before the result, and the note recording it is timestamped and committed
ahead of the scoring run. **The sealed prior was not revised**, because a prior adjusted after sealing
is not a prior. For the record: at scoring time I expected this to fail, and it did, and the credit
belongs to the map rather than to the registration.

Milestone R: exp_32c adds 2/4.
