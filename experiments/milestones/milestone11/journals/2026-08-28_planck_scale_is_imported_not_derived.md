# The Planck scale is the unit system, not a derivation — a note on one README line

**2026-08-28.** Exploratory check (STANDARDS §2.8). **M11's 52/52 is not in question**, and
none of its results change. This qualifies one sentence.

## What was checked and why

A boundary claim raised the same day — *DFT derives dimensionless quantities from structure and
imports every scale* — makes one flat falsifiable prediction: **any dimensionful quantity
derived from pure structure with no imported anchor kills it.** M11 is the corpus's strongest
apparent counterexample, since `README.md:13` reads:

> *"Planck scale derived, not assumed — response-time crossover at depth-183 reproduces l_P
> with zero free parameters (exp_01, exp_02)"*

## What the code does

```python
def crossover_energy(depth):                      # core/quantum_gravity.py:106
    return E_PLANCK_GEV * PHI ** (-depth)

def cascade_depth_response_time(depth, base_time=T_PLANCK_S):   # :88
    return base_time / (PHI ** (-depth))
```

Every dimensionful output is **[imported constant] × [dimensionless φ power]**. `HBAR`,
`C_LIGHT` and `M_PLANCK_KG` are hardcoded CODATA values (`:68`, `:69`, `:71`). exp_01 imports
`T_PLANCK_S`, `E_PLANCK_GEV` and `L_PLANCK_M` directly.

exp_02's four "routes to the Planck scale" are, in its own comments, *"length scales **in Planck
units**"* — Landauer 1/ln 2, Heisenberg 1/2, Schwarzschild 2M. Those are O(1) prefactors inside
a unit system that already contains the answer, which is also why the README's own hardening
note (`:96`) correctly downgraded T1 from "convergence" to "**bracket**".

**And M11 already says this itself**, at `core/quantum_gravity.py:113`:

> *"For gravity (depth 183): E_cross = E_Planck · φ^(−183) ~ 10⁻¹⁹ GeV. **This is NOT the
> Planck energy** — it's where classical gravity breaks."*

## Conclusion

M11 is **not** a counterexample, and the boundary survives its strongest test. M11's results are
dimensionless and legitimate; the scale is **assumed as the unit system**. "Zero free
parameters" is true — nothing is tuned — and is a *different claim* from "derived from
structure". Only the README's "derived, not assumed" overstates, and only for the scale itself.

## Why this matters beyond one line

`E_Planck · φ^(−d)` is **the same expression Milestone R uses**, where it misses nuclear and
atomic scales by 15–24 orders and is recorded as *"the most important result is a failure"*.
Same formula, opposite verdicts:

| | value | verdict |
|---|---|---|
| M11, d = 183 | 6.95e-20 GeV | sensible |
| Milestone R, d ≤ 20 | 15–24 orders off | the corpus's largest miss |

**It is not a scale law — it is a dimensionless correction to an imported anchor.** It looks
like a derivation when the anchor already sits near the target and fails when it must bridge 24
orders. Milestone R's failure is therefore not "this formula is wrong"; the formula was never
capable of setting a scale anywhere.

Detail and the standing prediction: `experiments/spikes/dimensional_boundary/`.
