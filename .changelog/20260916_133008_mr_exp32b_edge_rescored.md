# Milestone R exp_32b — the edge as a number, re-sealed and scored 4/4

exp_32 was UNSCORED by its own instrument gate. exp_32b re-asked the same question on fresh seeds
16-18 with every bar set from a measurement recorded in the seal (`3dbc0304`), and scored 4/4 on
42 runs with zero gate failures.

- **The edge reproduces.** kappa_c = 1.168 +/- 0.026 on seeds 16-18, against exp_32's unofficial
  1.171 +/- 0.038 on seeds 13-15. Two independent triples, three thousandths apart.
- **T3 12/12** across a fourfold change in gravity; **T4 3/3** at n = 8000 with its threshold untouched
  (exp_32 was 2/3 there).
- **Only the instrument amendment decided anything.** The identity gate at exp_32's 0.02 would have
  fired on 32/42; the 0.06 proposed in exp_32's bearing would have fired on 1/42 and unscored the round
  again. The struck kappa_c spread bar and the widened fraction band would both have passed at their
  OLD values, so T1 and T2 passed on their merits.
- **Instrument finding:** the ledger-identity truncation scales with gravity -- the three worst
  residuals are all g = 3.0 at kappa = 1.00. Pooling arms in the pre-seal measurement hid that tail.
- Adds `scripts/resolution_probe.py`: measure what the instrument resolves before setting any bar.
  Its own first draft merged arms and disagreed with the sealed kappa_c spread, which is what caught it.

Milestone R 71/128 -> 75/132 under the current convention (which charges an unscored round as 0/4);
75/128 if unscored rounds are excluded instead. Convention is Peter's call.
