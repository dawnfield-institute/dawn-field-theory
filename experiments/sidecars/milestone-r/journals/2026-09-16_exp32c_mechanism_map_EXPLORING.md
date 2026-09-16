# The map I should have drawn before sealing exp_32c (2026-09-16)

**EXPLORING.** No thresholds, no score, no kill. Six seeds (13–18), all from grids already on disk;
no new runs. This is the flashlight pass that should have preceded the registration.

## The question

exp_32c registers that κ_c is predicted by **f**, the fraction of initial power below 3 k_fundamental,
at r = +0.878 over six seeds. Before trusting that, does f leave a footprint anywhere in the physics
between the initial condition and the crossing?

## The proposed mechanism, and its death

The story I told: more large-scale power → a more coherent collapse → a larger pair-energy reservoir
→ the budget is absorbed rather than converted to motion → κ_c higher. That predicts f should raise
E_SEC at fixed κ.

At κ = 1.00, over the six seeds:

| quantity at κ = 1.00 | r with f |
|---|---|
| E_SEC(end)/T — **the reservoir ratio, the proposed middle link** | **−0.029** |
| E_SEC(end)/\|U₀\| | +0.030 |
| W_p/\|U₀\| — the work itself | **−0.004** |
| T/\|U₀\| — the transfer | +0.595 |
| KE/\|U\| | +0.626 |
| conn_q10 | +0.455 |
| occupancy | +0.128 |
| **κ_c** (for reference) | **+0.878** |

**The middle link is absent.** f does not raise the reservoir. It does not even predict the work at
κ = 1 — the correlation is four thousandths.

## So what is f riding?

If f predicts where W_p crosses zero but not its value at κ = 1, it should carry the slope. It does
not: r(f, full-sweep slope) = +0.101, r(f, local slope at the crossing) = +0.192, and for the
combination that would set the crossing, r(f, −W_p(1)/slope) = **−0.159**.

So **f correlates with the endpoint and with nothing on the way to it.** A real physical driver
should appear somewhere in the chain. Correlating with the outcome while correlating with none of its
proximate causes is the signature of a coincidence in six points, not of a mechanism.

Two further reasons not to be reassured:

- The partial correlations come out *higher* than the raw one — r(f, κ_c | W_p(1)) = +0.957,
  r(f, κ_c | slope) = +0.943. On three degrees of freedom those are not evidence of a cleaner signal,
  they are evidence that the estimates are unstable.
- κ_c is not well approximated by 1 + (−W_p(1))/slope anyway: that gives 1.233 and 1.204 for seeds 13
  and 14 against actual 1.200 and 1.129. The W_p curve is steep then flat, so no two-number summary
  captures it, and any story built on such a summary is loose.

## The one intermediate with a pulse

**T/\|U₀\|, the cumulative transfer at κ = 1**, correlates with f at +0.595 and with κ_c at +0.770 —
the only quantity measured here that sits between the two with signal on both sides. Its range is
narrow (0.882–0.912) so this may be nothing. It is the lead worth carrying, not f.

## What this costs, honestly

Had I run this ninety minutes earlier it would have cost nothing and I would not have registered f.
That is the concrete price of pre-registering before mapping: the map was available the whole time,
from data already on disk, and it says the hypothesis has no mechanistic footprint.

**exp_32c still stands and still runs.** A registered prediction I now expect to fail is a better
prediction than one I expect to pass, and its 42 runs give twelve seeds for the map either way. The
sealed prior of 58 % on T1 is not revised — a prior adjusted after sealing is not a prior — but this
note is the record that subsequent exploration moved my belief down, and the outcomes journal will
say so.

## The next move, which the kill sentence should have named

If T1 fails: not "stop trying to derive κ_c". The next move is **T/\|U₀\|** as the candidate, and
below it the Zel'dovich deformation tensor — the eigenvalue ordering of the initial field, which is
the mechanism the substrate actually implements and which f is at best a crude proxy for. λ_max was
one of the four statistics tested and came out at r = −0.655, wrong sign for the story but measured
on the wrong summary (a global max rather than the collapse-ordered spectrum).
