# Orrery epistemic audit — dawn-field-theory/foundational/experiments

_A mechanical audit of the claim graph. Two axes, never merged: **status** is the authored verdict (human-owned); **instrumentation** is how legibly the corpus evidences it (measured). Every figure derives from the corpus and regenerates with `orrery report`._

## Overview

- **Claims:** 31  (axiom: 2, claim: 18, constant: 3, definition: 1, prediction: 1, principle: 2, theorem: 4)
- **Experiment nodes:** 68 — mean coverage `████░░░░░░ 0.41`
- **Inconsistencies:** 1 contradiction(s) (0 open) · 0 unsupported · 3 open boundaries · 0 cycle(s)
- **Citation drift:** 31 ok of 31 — 0 problem(s)
- **Integrity:** 0 dangling edge(s) · 0 missing-evidence

## Claims — verdict × measured evidence

Ordered least-instrumented first (the evidence to-do list). `status` = verdict; `evidence` = instrumentation; `coverage` = measured.

| claim | status | evidence | coverage | statement |
|---|---|---|---|---|
| `alpha_EM` | validated | ~ sparse | `██░░░░ 0.36` | alpha = (F3/(F4 phi F10))(1 - F10/(4 pi F7^2)) — ranks #1 of 10,440 Fibonacci combos, 5… |
| `complement_identity` | validated | ~ sparse | `█████░ 0.79` | Identity IS complement: a vertex's complement spectrum uniquely determines its automorp… |
| `dimension_three` | validated | ~ sparse | `██░░░░ 0.36` | D = 3 spatial dimensions from five convergent paths (Mobius, SU(2) chirality, MED, curl… |
| `koide` | validated | ~ sparse | `██░░░░ 0.36` | Koide charge Q = F3/F4 = 2/3 exact (0.5 ppm). |
| `pac_fibonacci` | validated | ~ sparse | `██░░░░ 0.36` | PAC IS Fibonacci recursion; phi is its unique attractor. |
| `sin2_theta_W` | validated | ~ sparse | `██░░░░ 0.36` | sin^2(theta_W) = F4/F7 = 3/13 exact (gauge closure F7 = 13). |
| `ade_from_pac` | validated | ✓ well | `██████ 0.96` | ADE classification from PAC: the transfer matrix has spectral radius phi, producing Dyn… |
| `aut_gauge` | validated | ✓ well | `██████ 0.96` | Graph automorphisms Aut(G) create gauge equivalence; orbits are gauge-equivalence classes. |
| `born_rule` | validated | ✓ well | `██████ 0.96` | Born rule from the orbit measure: P(O) = \|O\|/n; probabilities sum to 1 (Gleason). |
| `cascade_clock` | validated | ✓ well | `█████░ 0.88` | Cascade clock N(t) = a + (1/ln phi) ln(t_lookback) unifies S8, Hubble, JWST. |
| `coherence_limit_universal` | falsified | ✓ well | `██████ 0.96` | The complement coherence limit is a DFT constant (it is geometric/family-dependent, not… |
| `d4_quantum` | validated | ✓ well | `██████ 0.96` | D_4 triality is the unique source of quantum uncertainty: the only ADE type with non-ab… |
| `desi_wa` | falsified | ✓ well | `█████░ 0.88` | DESI dark energy w_a ~ -0.15 (observed -0.75). |
| `lorentz` | validated | ✓ well | `██████ 0.96` | SL(2,C) double-covers SO(3,1): the Lorentz group emerges from connection + entropy. |
| `measurement_gauge` | validated | ✓ well | `██████ 0.96` | Measurement is gauge fixing: orbit projectors are idempotent, real, complete, irreversi… |
| `orbit_hilbert` | validated | ✓ well | `██████ 0.96` | Orbit Hilbert space L^2(V/Aut(G)) has identity Gram matrix (PD by construction) — resol… |
| `proper_time` | validated | ✓ well | `██████ 0.96` | Proper time tau = t / cosh(eta): boost deformation rate yields time dilation. |
| `psd_metric` | falsified | ✓ well | `██████ 0.96` | An isomorphism-invariant complement metric can be positive-definite on the full vertex … |
| `rapidity_composition` | open | ✓ well | `██████ 0.96` | Discrete complement distances compose like continuous rapidities (the discrete->continu… |
| `s8_resolution` | validated | ✓ well | `█████░ 0.88` | S8 tension resolved 3.22 sigma -> 0.07 sigma (one free parameter, t1 = 520 Myr). |
| `sec_complexification` | validated | ✓ well | `██████ 0.96` | SEC complexifies the real Weyl/SU(2) structure to non-compact SL(2,C). |
| `speed_of_light` | validated | ✓ well | `██████ 0.96` | c = the maximum complement-deformation rate (coherence limit from the rapidity bound). |
| `weyl_group` | validated | ✓ well | `██████ 0.96` | Complement transformations compose as a Weyl group (Z_2 for A_1). |
| `MED` | derived | · n/a | `░░░░░░ —` | Macro Emergence Dynamics: symbolic patterns bounded at depth <= 1, nodes <= 3 — emerges… |
| `PAC` | postulated | · n/a | `░░░░░░ —` | Potential-Actualization Conservation: at every recursive level potential equals actuali… |
| `RBF` | derived | · n/a | `░░░░░░ —` | Recursive Balance Field: a dynamic potential B(x,t) from which structure emerges — repl… |
| `SEC` | postulated | · n/a | `░░░░░░ —` | Symbolic Entropy Collapse: entropy is compressed potential that crystallizes into struc… |
| `Xi` | derived | · n/a | `░░░░░░ —` | Xi = gamma + ln(phi) = 1.0584 — the universal balance constant, zero free parameters. |
| `ln_phi` | derived | · n/a | `░░░░░░ —` | ln(phi) = 0.48121 — the Landauer erasure cost per cascade level for a phi-split. |
| `phi` | derived | · n/a | `░░░░░░ —` | phi = 1.618... — the unique stable fixed point of PAC recursion (b^2 - b - 1 = 0). |
| `self_loop` | postulated | · n/a | `░░░░░░ —` | The self-reference primitive x = 1 + 1/x, whose iteration converges to phi. |

## Inconsistencies (deterministic nominations, not verdicts)

**Contradictions**

- `orbit_hilbert` ⊥ `psd_metric` — resolved

**Unresolved boundaries**

- `coherence_limit_universal`
- `desi_wa`
- `rapidity_composition`

## Citation drift (receipts)

ok: 31

_All citations resolve._

## Coverage

Mean `████░░░░░░ 0.41`  (14 high · 4 medium · 50 low).

Lowest-coverage nodes:

- `█░░░░░ 0.20`  `DNA_repair`
- `█░░░░░ 0.20`  `entropy_information_polarity_field`
- `█░░░░░ 0.20`  `hodge_conjecture`
- `█░░░░░ 0.20`  `information_amplification`
- `█░░░░░ 0.20`  `landauer_erasure_field_cost_map`
- `█░░░░░ 0.20`  `navier-stokes`
- `█░░░░░ 0.20`  `pac_dag_fluid`
- `█░░░░░ 0.20`  `phi_artifact_test`
- `█░░░░░ 0.20`  `pi_harmonics`
- `█░░░░░ 0.20`  `pre_field_recursion`
- `█░░░░░ 0.20`  `predictive_collapse`
- `█░░░░░ 0.20`  `quantum_validation`
- `█░░░░░ 0.20`  `recursive_entropy`
- `█░░░░░ 0.20`  `recursive_gravity`
- `█░░░░░ 0.20`  `recursive_tree`

## Headline dossiers

### `ade_from_pac` — validated  (✓ well, `██████ 0.96`)

ADE classification from PAC: the transfer matrix has spectral radius phi, producing Dynkin diagrams.

_Rests on (toward the axioms):_

- `pac_fibonacci` (validated)
- `PAC` (postulated) · `phi` (derived)
- `self_loop` (postulated)

_Frontier:_ 1 unresolved boundary(ies), 0 unsupported in-neighborhood.

### `alpha_EM` — validated  (~ sparse, `██░░░░ 0.36`)

alpha = (F3/(F4 phi F10))(1 - F10/(4 pi F7^2)) — ranks #1 of 10,440 Fibonacci combos, 5.7 ppm.

_Rests on (toward the axioms):_

- `pac_fibonacci` (validated) · `phi` (derived)
- `PAC` (postulated) · `self_loop` (postulated)

### `aut_gauge` — validated  (✓ well, `██████ 0.96`)

Graph automorphisms Aut(G) create gauge equivalence; orbits are gauge-equivalence classes.

_Rests on (toward the axioms):_

- `ade_from_pac` (validated)
- `pac_fibonacci` (validated)
- `PAC` (postulated) · `phi` (derived)
- `self_loop` (postulated)

### `born_rule` — validated  (✓ well, `██████ 0.96`)

Born rule from the orbit measure: P(O) = |O|/n; probabilities sum to 1 (Gleason).

_Rests on (toward the axioms):_

- `orbit_hilbert` (validated)
- `aut_gauge` (validated) · `complement_identity` (validated)
- `ade_from_pac` (validated)
- `pac_fibonacci` (validated)
- `PAC` (postulated) · `phi` (derived)
- `self_loop` (postulated)

### `cascade_clock` — validated  (✓ well, `█████░ 0.88`)

Cascade clock N(t) = a + (1/ln phi) ln(t_lookback) unifies S8, Hubble, JWST.

_Rests on (toward the axioms):_

- `ln_phi` (derived) · `phi` (derived)
- `PAC` (postulated) · `self_loop` (postulated)

### `complement_identity` — validated  (~ sparse, `█████░ 0.79`)

Identity IS complement: a vertex's complement spectrum uniquely determines its automorphism orbit.

_Rests on (toward the axioms):_

- `ade_from_pac` (validated)
- `pac_fibonacci` (validated)
- `PAC` (postulated) · `phi` (derived)
- `self_loop` (postulated)

_Frontier:_ 1 unresolved boundary(ies), 0 unsupported in-neighborhood.

---

**How to read this.** _status_ (verdict, human-owned): `postulated · derived · validated · open · falsified · observed`. _instrumentation_ (measured evidence): `✓ well · ~ sparse · ✗ none · ? unresolved · · n/a`. The two never merge — a well-instrumented claim can be false, and a validated claim can be sparsely shown. Regenerate with `orrery report`.
