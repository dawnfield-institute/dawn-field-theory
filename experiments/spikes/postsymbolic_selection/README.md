# Post-symbolic selection pressure (spike)

**Exploratory. `spikes/` is exempt from the experiment standard by definition (STANDARDS §3).**
No pre-registration, no thresholds, no scoring — see STANDARDS §2.8 on exploring vs predicting.

Spirit (**not** a port) of era-1 `brain.py`: fields with history, no symbols. Collapse
reinforces a local memory; that memory suppresses future collapse; activity must move.

**What the rules contain, stated up front (§2.8):** plain decimals only — growth 0.05, couple
0.05, decay 0.90, thresholds 0.40/0.05, gain 0.05, diffusion 0.20. **No φ, Ξ, Fibonacci or π
anywhere in the dynamics.** Nothing resembling a framework constant was seeded, so nothing
resembling one could be "found".

## What it shows

**1. Reinforcement alone saturates.** era-1 QPL only grows and caps, never releases. At
`mem_decay = 1.0` the firing fraction goes to 0.0000 — memory saturates into *uniform*
suppression, and uniform suppression selects nothing. The v4 engine's `memory_decay = 0.95`
is the missing ingredient, and the lineage shows it being added.

**2. Pure conservation dies.** Diffusive redistribution conserves exactly — total excess equals
events × gain to **2.88e-16** — but with no return channel the excess accumulates monotonically
until the field stops firing.

**3. Redistribution + a return channel produces domains.** Spatial autocorrelation of the event
map:

| mode | lag1 | lag2 | lag4 | lag8 |
|---|---|---|---|---|
| local decay (potential destroyed) | 0.446 | 0.162 | 0.023 | −0.008 |
| pure redistribution (conserved, no return) | 0.418 | 0.114 | 0.002 | −0.010 |
| **redistribution + return** | 0.916 | 0.843 | 0.736 | **0.552** |

The first two carry only nearest-neighbour correlation — that is just the coupling term, gone
by lag 4. The third holds structure to lag 8+: **an intermediate scale that nobody put in.**
Relevant to M16, whose diagnosis was that the engine has "exactly two scales, the cell and the
box", with nothing between them for a web to be made of. This is a mechanism for the missing
middle scale; it is a 2D toy and not the engine, and no stronger claim is made.

**4. Structure lives in a band of non-closure rate** (`sink_scale.py`): correlation length
1.16 → 18.01 → 1.45 as non-closure runs 0.001 → 0.05 → 0.20. Too little and the excess
accumulates until firing stops; too much and nothing persists long enough to differentiate.
Not a power law — the log-log fit is poor (correlation 0.49) because the relation is
non-monotonic.

**The band's location is not interpreted.** It was measured at fixed `diff = 0.20`, so what
matters is almost certainly the ratio of return rate to diffusion rate, not either number. No
framework constant is looked for at the peak — that is exactly the circularity §2.8 forbids.

## Null results, kept

Two candidate invariants were tested and both are knob readouts, not invariants:
`gap/steps_to_cap` (CoV 0.86 across 12 settings) and `gap·damping/(cap−1)` (CoV 0.53).
Also: with release present, the memory **cap becomes inert** — mean gap is identical at
cap = 1.5, 2.0 and 3.0, because memory equilibrates below the cap and never binds.

## Files

| file | purpose |
|---|---|
| `postsymbolic_selection.py` | first pass + knob sweep; includes the T1-style live-mechanism check that caught the refractory barely biting |
| `postsym_redistribute.py` | the three modes — destroy / conserve / conserve+return — and the domain measurement |
| `sink_scale.py` | return-channel rate vs domain scale |
