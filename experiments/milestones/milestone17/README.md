# Milestone 17: Criticality — the boundary where identity changes scale

## Score: 6/9 Block A (instrumentation) · exp_03 open · Blocks B–E unstarted

| Block | Experiments | Score | State |
|---|---|---|---|
| **A — Instrumentation** | exp_01 – exp_03 | 6/9 | exp_01 and exp_02 passed; exp_03 open |
| **B — Is there a critical point?** (Q1) | exp_04 – exp_06 | 0/12 | blocked on A |
| **C — Is it at Ξ?** (Q2) | exp_07 – exp_08 | 0/8 | blocked on B |
| **D — Self-organized or tuned?** (Q3) | exp_09 – exp_10 | 0/8 | blocked on B |
| **E — Does it derive the finite-size corrections?** (Q4) | exp_11 – exp_12 | 0/8 | blocked on C |

Block A is deliberately front-loaded and no DFT system is measured until it completes. The
preceding round produced **seven** instrument faults and exp_01 alone produced three more —
none caught by a statistic taken at face value, every one caught by a reference whose answer
was already known.

---

## Thesis

> **The limits this corpus keeps encountering are critical points: the parameter values at
> which the correlation length diverges and the scale on which identity is defined changes.**

At a critical point ξ → L. Under M13's identity-IS-complement, a node's complement then stops
being local and becomes global — **the scale at which identity is defined changes**. In cluster
language: below p_c, finite clusters; at p_c, the largest spans the system. A thing stops being
*a clump* and becomes *the network*. Identity changes category, not degree.

Wolfram's classes state the same thing about persistence. Class II is frozen (identity too
rigid to change), Class III is chaotic (identity dissolves), **Class IV is where structures
persist and transform** — gliders holding together while propagating and interacting.

This is not an analogy laid over critical phenomena. It is what the order parameter does.

## RETRACTED — the "empirical fact" this milestone was founded on

**2026-08-16. The wall does not exist. It was my measurement.**

This milestone was opened on a claim that four independent routes had triangulated on a
percolation floor of 0.007–0.019, with **exp_11's published 3D web at 0.0068** as the anchor.
That reading came from binning 4000 particles onto a 64³ grid — **0.015 particles per cell**.
At that sampling the density field is empty by construction: the overdense set shatters into
singletons and *any* web, real or deliberately synthetic, reads as disconnected.

Same run, same physics, same code — read at different binnings:

| res | particles/cell | percolation | `is_web` |
|---|---|---|---|
| 16 | 0.98 | **0.472** | True |
| 24 | 0.29 | 0.433 | True |
| **32 — exp_11's own binning** | 0.12 | **0.385** | **True** |
| 48 | 0.036 | 0.281 | False |
| **64 — where my 0.0068 came from** | **0.015** | 0.062 | False |

**exp_11's cosmic web is real and it percolates.** At matched sampling the 3D substrate gives
percolation 0.406 ± 0.069 from exp_11's own *uncorrelated lattice* start, with **5/5 seeds**
passing the web gate.

The tell was in view the whole time: **occupancy read 0.012–0.04 in every one of those runs**,
and it was printed alongside every percolation value. I never asked whether that occupancy was
physically attainable. A 3D control settles it independently — a deliberately connected web
reads 1.000 across occupancy 0.082–0.268, so the instrument was never the problem.

**Consequences for what follows:**

- "ξ at the white-noise floor, maximally sub-critical, stated four times" — **withdrawn**.
- "criticality was never looked for, because the corpus had no instrument that could see it"
  — **withdrawn**. The corpus's instrument was sound at the corpus's own resolution.
- The 2D↔3D difference is not a detail. 2D was doing much of the damage on its own: the
  filament/sheet/node topology exp_11 targets does not exist in the plane, and site
  percolation sits at 0.593 in 2D against 0.312 in 3D.

**Rule this cost enough to earn**: match particles-per-cell (`n/res^d ≈ 1`) before reading any
connectivity statistic, and check every threshold-based measure against a known-connected
control **at the same occupancy**.

## What the corpus already established

| source | result |
|---|---|
| `cellular_automata_pac_attractors` | The four rules nearest Ξ are **all Class IV**, p = 8.58×10⁻⁸, 42.7× enrichment, 0/1000 random rules near Ξ. Rule 110 at 1.0579. |
| `sec_threshold_detection` | "ξ ≈ 1.0571 **appears at phase transitions**" — cross-domain incl. Lorenz, combined p < 0.00001. |
| `sec_prime_manifold` | exp_20 *phase transition proof*, exp_29 *phase transition*. |
| CAH | Ξ as **the maximum sustainable computational asymmetry** — a ceiling, which is what a critical point is. |

**Ξ is established as the framework's critical point in computation and in dynamical systems,
and has never been checked in a structure-forming one.**

## The methodological finding

**exp_10 concluded "NO discrete phase transition" from four one-point statistics** — density CV,
void fraction, filament fraction, sampled clustering. None is an order parameter, and a
transition is invisible to all of them; it lives in the correlation length, the cluster-size
distribution and the susceptibility.

So the status is **not** "the framework lacks criticality in structure formation". It is
**criticality was never looked for**, because the corpus had no instrument that could see it.

---

## Block Structure

### Block A — Instrumentation (3/9)

No DFT system is measured until this completes.

| Exp | Question | Score | State |
|---|---|---|---|
| 01 | Do the instruments recover 2D site percolation? | **3/3** | **PASS** |
| 02 | Correlation length as a *scaled* critical quantity — ξ/L crossing | **3/3** | **PASS** |
| 03 | Edge-of-chaos classifier, calibrated against Wolfram classes | 0/3 | open |

> **exp_02 was first written up as a 0/3 failure and re-scored the same day.** It carried a kill
> sentence, which an instrument calibration should never have — a tool gets a **domain of
> validity**, not a kill sentence, and locating a boundary is what calibration *is* (exp_01
> scored 3/3 while finding three bugs). Every measured number is unchanged; the registered
> predictions P6 and P7 remain **REFUTED** below. See
> `journals/2026-08-27_exp02_reframe_domain_not_kill.md`; the original outcomes journal stands
> as lineage.

**exp_02 result (2026-08-27).** Three results against exactly-known answers: it closes the
milestone's fourth retracted route, validates the connectivity instrument, and locates that
instrument's resolution floor.

**The 0.63 inference is dead.** At L = 256, `structure.correlation_length` reads **0.6296 at the
exact p_c** against its own white-noise floor of 0.6321, and its whole dynamic range over
p ∈ [0.40, 0.80] is 0.003 wide. It converges *onto* the floor as resolution improves —
0.6111 → 0.6199 → 0.6257 → 0.6296 for L = 32 → 256. Discrimination power **D_A = 0.893**
against the connectivity estimator's **D_B = 7.526**. A system sitting exactly on its critical
point reads at the white-noise floor, so *"ξ at the floor ⇒ maximally sub-critical"* — the
fourth route into the retracted wall, and the one the 2026-08-17 retraction never examined —
**never carried the information that was read out of it.** This does not say the engine is
critical; it removes a piece of evidence that said it was not.

**The connectivity instrument works, and its domain is now located.** It finds p_c to 0.013
unaided, discriminates at 7.5σ, recovers **ν = 1.3400 against exact 4/3 (0.5%)**, and scales as
**L^1.057, R² = 1.000** at p_c.

**Its domain of validity is ξ ≳ 2 cells.** Below that it reports the small-cluster tail, which
grows with L by extreme-value sampling rather than by divergence:

| ξ at L=256 | 4.4 | 2.9 | 1.9 | 1.0 |
|---|---|---|---|---|
| **α in ξ ~ L^α** | +0.085 | +0.195 | +0.398 | +0.565 |

**This is a domain, not a defect** — the same statement as estimator A's white-noise floor at
the other end of the scale, and every length estimator on a lattice has one. Block B needs the
instrument at and below p_c, where ξ runs 6.8 → 61 cells: an order of magnitude inside the
validated range. *Limitation:* the sub-critical side of the sweep never reaches ξ < 3.7 cells,
so the floor was exposed only above p_c and its symmetry is **not** established.

**Two registered predictions were refuted, and one of them matters more than the score.**

- **P6 — there is no shared M17 instrument bias.** exp_01's exponents run ~10% low; exp_02's ν
  is 0.5% high, at the same sizes on the same system. Bias belongs to each **estimator**, not to
  the size range. **Block B may not apply a blanket correction — every DFT exponent must be
  calibrated against the specific estimator that produced it.**
- **P7** — the off-critical control at p = 0.75 gave α = +0.398 against a registered |α| < 0.25.
  That number stands; what it located is the resolution floor above, not an instrument failure.

**exp_01 result.** p_c = **0.5917** against an exact 0.5927460 (err 0.0011), located purely by
where spanning-probability curves for L = 32/64/128 intersect. χ_max ~ L^1.6233 against an exact
γ/ν = 1.7917. n_s ~ s^−1.849 against an exact τ = 2.0549, with R² 0.996 at p_c and 0.967 away
from it — the instrument finds a power law where there is one and not where there isn't.

Exponents run ~10% **low** in the same direction — γ/ν is 9.4% below exact, τ is 10.0% below —
the expected finite-size bias at L ≤ 128 over three sizes. **Any exponent measured on a DFT
system carries the same bias and must be compared against this calibration, not against
textbook values.** (The direction was written up as "high" until exp_02, which depends on it.)

### Block B — Is there a critical point at all? (Q1) (0/12)

| Exp | Question |
|---|---|
| 04 | Avalanche cutoff scaling in the v3 field engine — χ ~ L^(γ/ν) or L-independent? |
| 05 | The same in the v4 particle substrate, the one system producing connected structure |
| 06 | Order-parameter identification: what is the DFT analogue of P_∞? |

Design settled by exp_01: measure the **cutoff's scaling with system size**, not the shape of a
single distribution. Needs no control parameter, which matters because `sec_balance` is a
*choice* rather than a derived analogue of occupation probability.

### Block C — Is the critical point at Ξ? (Q2) (0/8)

| Exp | Question |
|---|---|
| 07 | If a crossing exists, where is it, and how far from Ξ in units of its own width? |
| 08 | Exponent extraction — do the DFT exponents match a known universality class? |

The registered prediction. Ξ marks transitions in Lorenz, in primes and in cellular automata at
p < 10⁻⁵; if the crossing lands near Ξ, the framework has predicted a **location** it was not
fitted to.

### Block D — Self-organized, or tuned? (Q3) (0/8)

| Exp | Question |
|---|---|
| 09 | Does the system drive itself to criticality, or must it be placed there? |
| 10 | Stochastic self-application — **M10 open thread 2** |

SOC systems reach the critical point without tuning and produce scale-free connected structure
generically. M10 thread 2 is the same question from the other side: all M10 experiments are
deterministic, and *"genuine irreversibility needs stochastic self-application."*

### Block E — Does criticality derive the finite-size corrections? (Q4) (0/8)

| Exp | Question |
|---|---|
| 11 | φ^(−1/N) at N = 8 — **M10 open thread 1**, currently 3.3% off and underived |
| 12 | The 8.9% slope gap — **M9 / M10 thread 6**, "finite-size noise or sub-leading physics?" |

Near a critical point, finite-size corrections take a *universal* form set by the critical
exponents. If these systems sit near criticality, both are derivable rather than fitted. **This
block can close existing open threads rather than adding new ones.**

---

## Predictions Registry

| # | Prediction | Registered | Status |
|---|---|---|---|
| P1 | The criticality instruments recover p_c, γ/ν and τ for 2D site percolation | 2026-08-16 | **CONFIRMED** — 0.2% on p_c |
| P2 | A DFT structure-forming system has a measurable critical point | 2026-08-16 | open (Block B) |
| P3 | That critical point sits at Ξ | 2026-08-16 | open (Block C) |
| P4 | The framework is self-organized-critical — reaches it without tuning | 2026-08-16 | open (Block D) |
| P5 | M10's φ^(−1/N) correction follows from critical exponents | 2026-08-16 | open (Block E) |
| P6 | ν shares exp_01's ~10% low finite-size bias, giving M17 one coherent instrument bias | 2026-08-27 | **REFUTED** — ν = 1.3400, 0.5% **high**; bias is per-estimator |
| P7 | The connectivity length is L-independent (\|α\| < 0.25) at fixed off-critical p | 2026-08-27 | **REFUTED** — α(0.75) = +0.398; resolution floor below ξ ≈ 2 cells |

**Registered invariants**: ξ/L and the **critical exponents**. Exponents are *universal* —
shared across every system in a universality class regardless of microscopic detail — which
makes them the strongest available registered quantity. Never ξ in cells; never a percolation
value without its occupancy.

## What would falsify the milestone

> **If ξ/L shows no crossing at any system size across the accessible parameter range, and the
> avalanche cutoff is L-independent, then the structure-forming systems are generically
> sub-critical and Ξ-at-transitions does not extend to structure formation.**

Recorded as a result either way. A weaker second failure also counts: a crossing that exists but
sits nowhere near Ξ answers Q1 yes and Q2 no, separating "the framework has criticality" from
"Ξ locates it."

## Dependencies

| Milestone | What M17 uses |
|---|---|
| **M13** | identity-IS-complement — the reason a diverging ξ *is* an identity change |
| **M13.5 / M15** | the class/representative split, the identity-level statement of the same boundary |
| **M10** | laws-as-equilibria; open threads 1, 2 and 6 are Block D and E targets |
| **M16** | the percolation floor across four routes — the empirical entry point |
| `cellular_automata_pac_attractors` | Ξ at Class IV, p = 8.58×10⁻⁸ |
| `sec_threshold_detection` | Ξ at phase transitions, cross-domain |

## Capability holes

| instrument | status |
|---|---|
| order parameter, susceptibility, spanning | **built**, calibrated (exp_01) |
| cluster-size distribution, pooled and log-binned | **built**, calibrated |
| finite-size crossing | **built**, calibrated |
| power-law vs exponential discriminator | **built** — and shown *insufficient* alone, see below |
| cutoff scaling (χ ~ L^(γ/ν)) | **built**, calibrated |
| correlation length as a *scaled* critical quantity | **built and calibrated** (exp_02: `connectivity_length`) — recovers p_c, ν to 0.5%, L^1.057 scaling; **domain of validity ξ ≳ 2 cells** |
| density-autocorrelation ξ as a criticality measure | **shown impossible** (exp_02) — reads the white-noise floor at exact p_c, D_A = 0.893 |
| edge-of-chaos classifier as a reusable instrument | open (exp_03) |

**A discriminator finding worth carrying**: `power_law_or_exponential` is exact on pure forms
but reads *sub-critical* percolation as "power_law", correctly — a sub-critical distribution is
a **truncated** power law with the scaling regime intact below the cutoff. Distribution shape
alone cannot separate critical from sub-critical. Only cutoff scaling can.

## Core machinery (reused, not reimplemented)

- `core/criticality.py` — this milestone's instruments, all N-D
- `reality-engine/proof_of_concepts/v4/structure.py` — correlation length, percolation with
  occupancy, web metrics; N-D and selftested
- `reality-engine/proof_of_concepts/v4/particles.py` — N-D particle substrate, both force
  conventions
- `experiments/studies/cellular_automata_pac_attractors/core/invariant_metrics.py` — excess
  entropy, block entropy, correlation dimension
- `experiments/studies/sec_threshold_detection/scripts/exp_01_threshold_detector.py`

## Open engineering question

The engines live in `reality-engine`; these instruments live here. Block B needs them together.
Cross-repo import versus a shared module is a structural choice to make deliberately rather
than by whichever import gets typed first.

## FDO Links

- Lore node: `milestone17-criticality`
- Related: `milestone13-identity-complement`, `milestone10-symmetry-self-application`,
  `milestone16-relational-locality`, `proj-reality-engine`
