# The dimensionless/dimensionful boundary (spike)

**Exploratory. `spikes/` is exempt from the experiment standard (STANDARDS §3).** No scoring.

## The claim

**DFT derives dimensionless quantities from structure, and imports every scale.**

Structure — ratios, classifications, counts — is combinatorial and needs no geometry. A length,
mass or energy needs an anchor the framework does not derive. Motivated by a structural
argument: **space is interior to a node, and scale requires space**, so scale must be set
internally rather than arriving from a space-free exterior.

**The falsifier, stated flatly:** *any dimensionful quantity derived from pure structure, with
no imported anchor, kills this.*

## The tabulation

| | error | anchor |
|---|---|---|
| **11 dimensionless** — α_EM, sin²θ_W, Koide, μ/e, p/e, Feigenbaum, Casimir, She-Leveque, α_s, PMNS angles, Ξ | exact → 1.71% | **none** |
| **dimensionful successes** — Rydberg, nuclear scale, β endpoint, Higgs mass | 11.4 ppm → 1.75× | **m_e / m_p / VEV imported** |
| **the one attempt from structure alone** — `E_Planck·φ^(−d)` | **15–24 orders off** | none — and it fails |

## The falsification test — M11, and the boundary survives

M11 claims *"Planck scale derived, not assumed — reproduces l_P with zero free parameters"*, and
scores **52/52**. It is the corpus's strongest apparent counterexample. It isn't one:

```python
def crossover_energy(depth):
    return E_PLANCK_GEV * PHI ** (-depth)

def cascade_depth_response_time(depth, base_time=T_PLANCK_S):
    return base_time / (PHI ** (-depth))
```

Every dimensionful output is **[imported constant] × [dimensionless φ power]**. `HBAR`,
`C_LIGHT` and `M_PLANCK_KG` are hardcoded CODATA values. exp_02's four "routes to the Planck
scale" are, in its own comments, *"length scales in Planck units"* — 1/ln 2, 1/2, 2M. O(1)
prefactors inside a unit system that already contains the answer.

**M11's own docstring says it plainly** (`core/quantum_gravity.py:113`):

> *"For gravity (depth 183): E_cross = E_Planck · φ^(−183) ~ 10⁻¹⁹ GeV. **This is NOT the
> Planck energy** — it's where classical gravity breaks."*

M11's results are dimensionless and legitimate. What is overstated is one README line: the
scale is **assumed as the unit system**, not derived. "Zero free parameters" is true (nothing is
tuned) and is a different claim from "derived from structure".

## The unification — one expression, two verdicts

`E_Planck · φ^(−d)` is used by **both** M11 and Milestone R.

| | | |
|---|---|---|
| M11, d = 183 | 6.95e-20 GeV | sensible, and M11 says what it is |
| Milestone R, d ≤ 20 | **15–24 orders** above nuclear/atomic scales | *"the most important result is a failure"* |

**It is not a scale law. It is a dimensionless correction to an imported anchor.** It looks like
a derivation exactly when the anchor already sits near the target, and fails visibly when it
must bridge 24 orders to reach somewhere else.

So Milestone R's central failure is not "this particular formula is wrong". **The formula was
never capable of setting a scale anywhere** — its successes and its failures are the same fact
seen from different distances to the anchor.

## What this is worth, and what it is not

It is **not** a criticism of dimensional analysis. Standard physics also cannot get a scale from
nothing — it takes masses as measured inputs and says so. The content is that **DFT tried**, in
Milestone R, and the attempt is the corpus's largest single miss; the boundary explains why, and
predicts the attempt cannot succeed by that route.

**Practical consequence:** Milestone R's remaining energy-scale propagation (exp_06/08/09) runs
against the grain. Finishing it likely means deciding what will *not* close rather than pushing.

**Standing prediction:** any future DFT derivation of a dimensionful quantity from pure
structure, with no imported anchor, falsifies this outright.

Related: `experiments/spikes/constant_number_fields/` — the same constants classified by number
field, which is the notation-independent form of the counting/growth/closure split.
