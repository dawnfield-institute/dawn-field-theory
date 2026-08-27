# Energy-scale propagation — and the depth was in the wrong slot

**2026-08-27.** Started as the ROADMAP's "highest-value unfinished work": propagate exp_24's
energy-scale fix back through exp_03–09, which are scored against a scale now known to be
wrong. It became something else about four steps in.

Exploratory throughout. No pre-registration, no kill sentence — the leads at the end are
labelled with their strength and none of them is scored.

---

## 1. Why this had never been done: it did not run

`core/radiation_physics.py` resolved its imports as `MR_ROOT.parent / "milestone14"`. After the
August 2026 layer reorganization this sidecar lives at `experiments/sidecars/milestone-r`, so
that path points at `experiments/sidecars/milestone14`, which does not exist. **Every one of the
27 scripts imports through that module, so all 27 were unrunnable.**

`midnight/core/phase_rate.py` line 29 has the identical break and is presumably in the same
state.

Fixed by hopping two levels and across to `experiments/milestones/`. Recorded in the code
because the next reorganization will do this again.

Second blocker: **`np.trapz` was removed in NumPy 2.0.** `requirements.txt` floors numpy at
1.24, so numpy 2.x is in-spec and this is a genuine break, not a local environment problem.
Swapped for `scipy.integrate.trapezoid` in exp_03 (numerically identical — exp_03 T2's reduced
chi² is 0.3748079733738714 before and after, all sixteen digits). **Six other files still carry
it**: milestone11 ×3, milestone2, milestone4, and `minimum_actualization_resolution` ×5.

## 2. What actually propagated

exp_24 replaces `E_Planck · φ^(−d)` with `α(d)² · m_mediator`, and it reproduces its documented
numbers exactly once it can run: Rydberg to **11.41 ppm**, nuclear scale 10.4576 MeV.

Added `severance_energy_coupled(depth, mediator_mass_mev, n_boundaries)` alongside the untouched
`severance_energy`, following the pattern exp_24 already set with `coupling_boundary_count`.

| exp | test | before | after |
|---|---|---|---|
| 03 | T3 beta endpoint | FAIL — best depth **19**, the top of `range(3,20)`, 0/3 within ×10 | **PASS** — depth **7**, interior, 2/3 within ×10 |
| 05 | T1 Rydberg | FAIL — 24 orders off | **PASS** — 11.4 ppm |

exp_03 2/4 → 3/4, exp_05 1/4 → 2/4.

**The old T3 failure has the signature of a minimiser out of room**: it could not have passed at
any depth in its range, because matching a 10²² MeV scale to a ~1 MeV endpoint needs d ≈ 105.

## 3. exp_24's root-cause claim is over-broad

exp_24 states the fix resolves "8 energy-scale failures (exp_03 T3/T4, exp_04 T1, exp_05
T1/T2/T4, plus scope_boundary_count tautology)". Checked against the recorded results:

| named failure | is it the scale? |
|---|---|
| exp_03 T3 | **yes** — fixed |
| exp_05 T1 | **yes** — fixed |
| exp_03 T4 | no — settled/unsettled graph entropy, no scale term |
| exp_04 T1 | no — `equilibration_energy` is purely graph-spectral; it fails because values go **negative**, and no positive rescaling fixes a sign |
| exp_05 T2 | no — φᵏ vs k²; the script's own note says *"Expected to fail"* |
| exp_05 T4 | no — correction template magnitude; the Rydberg is now exact and it is still 486× off |

**Two of the six named test failures were the scale.** The claim should be narrowed.

## 4. The hazard in the other direction — scores may go DOWN

Several Milestone R *passes* are tautological at the Planck scale and the scorecard says so
plainly: exp_06 T1/T2/T4 "*tautological (all energies round to n=0 at Planck scale)*", exp_02 T1
"*tautological (Planck >> MeV)*", exp_08 T1 "*Boundary counts tautologically 0*", exp_09 T4
"*same energy scale problem*".

**A correct scale turns those into real tests, which can then fail.** So the net effect of
finishing this propagation is genuinely unknown and could be negative. That is the right outcome
either way — a test that passes because every input rounds to zero is not evidence — but the
ROADMAP's framing of this as converting failures into results is only half the story, and the
half it omits is the one that lowers the score.

Not yet done: exp_06, exp_08, exp_09 (9, 5 and 3 call sites of `scope_boundary_count`).

## 5. The real finding: depth is read out of the wrong slot

Propagating exp_24 through EM-depth experiments silently corrupts them, and chasing why produced
the actual result of the day.

`dft_energy_scale(depth, m) = fibonacci_depth_coupling(depth)² · m`, and
`fibonacci_depth_coupling(d) = φ^(−d)/√5`. At EM depth:

| quantity | value | vs Rydberg |
|---|---|---|
| `ALPHA_EM_DFT` | 7.2973e-3 | **11.4 ppm** |
| `fibonacci_depth_coupling(13)` | 8.5837e-4 | **72× off** |

Ratio **8.501**. exp_24's T1 quietly bypasses its own helper and uses `ALPHA_EM_DFT` directly;
the only test that *does* use `fdc` is T2, at 1.75× inside a factor-1000 acceptance window. **The
scale law is sharp only where it reduces to something already known, and degenerate everywhere
else.**

Immediate cause: `DEPTH_EM = 13` is **F₇ = 13**, a Fibonacci *value* — which is what makes
sin²θ_W = F₄/F₇ = 3/13 work — while `fdc(d)` treats d as a φ *exponent*. The same numeral is a
value in one place and an index in another.

**The (depth, mediator) pair is exactly degenerate.** Since E ~ φ^(−2d)·m, scaling m by r is
identical to shifting d by ln(r)/(2 ln φ). Measured across mediators for the beta endpoints:

| mediator | best d | E(d) MeV | within ×10 |
|---|---|---|---|
| electron | 0 | 0.108 | 3/3 |
| pion | 5 | 0.240 | 2/3 |
| proton | 7 | 0.236 | 2/3 |
| W boson | 12 | 0.164 | 3/3 |
| generic 1 GeV | 7 | 0.251 | 2/3 |

m_p/m_e = 1836 gives a predicted shift of **7.809**, which is precisely the electron→proton gap.
**So exp_03's "depth 7" above is an artifact of choosing the proton, not a result** — the fit
measures the mediator. Recorded because the pass in §2 should not be read as locating the weak
depth. (The electron would have scored 3/3; no tuning was done.)

### The cascade clock cannot be the same coordinate

Tested whether M9's cascade depth and the coupling depth are one thing. They are not:

```
M9 clock  N(t) = 1.3596 + 2.0781 ln(t_lookback_Gyr),  max physical N = 6.81 at t = 13.797 Gyr

nuclear  d=3       -> t =   2.20 Gyr   physical
beta     d=7       -> t =  15.09 Gyr   NO
alpha_EM d=8.5524  -> t =  31.86 Gyr   NO
DEPTH_EM d=13      -> t = 270.83 Gyr   NO  (20x the age of the universe)
```

Each e-fold of time buys 1/ln φ = 2.08 cascade levels, and cosmic history is ~3.3 e-folds from
the first-star anchor, so **the cascade can only ever accumulate ~6.8 levels** while the coupling
hierarchy spans 3 → 13. Whatever sets coupling depth, it is not elapsed cosmic time on this
clock. My hypothesis, killed by one line of arithmetic against existing code.

### Why a single depth cannot work at all

Every flagship constant is a **multi-index** Fibonacci expression:

| constant | indices |
|---|---|
| α_EM = F₃/(F₄·φ·F₁₀)·(1 − F₁₀/4πF₇²) | {3, 4, 7, 10} |
| sin²θ_W = F₄/F₇ | {4, 7} |
| μ/e = F₄·F₆²·(1+1/F₇) | {4, 6, 7} |
| p/e = F₄·F₉·F₁₂/F₆ | {4, 6, 9, 12} |

Forcing α_EM into the one-exponent form gives d_eff = 8.5524, which decomposes exactly:

```
F10 contributes index         10
the explicit phi             + 1   -> 11
prefactor F3/F4 and sqrt5    - 2.4474
                               -------
d_eff = 8.5526   (measured 8.5524)
```

**d_eff is not a depth.** It is a real depth (10–11) mixed with non-depth normalisation. It is
what you get forcing a four-index object through a one-index form — which is why it is
non-integer, matches no Fibonacci position, and needs a free mediator to close.

## 6. The φ/π class signature

Decomposing eleven constants as (Fibonacci multiset, φ-power, π-power). Every formula was
verified against its measured value in the same run (5.7 ppm to 2.41%).

| class | constants | φ-power | π-power |
|---|---|---|---|
| coupling | α_EM, α_s | **−1** | −1, 0 |
| mixing | sin²θ_W, Koide, θ₁₂, θ₁₃ | 0 | 0 |
| mass ratio | μ/e, p/e | 0 | 0 |
| regularisation | Casimir 240 | 0 | 0 |
| turbulence | She-Leveque | 0 | 0 |
| balance | Ξ = 1 + π/F₁₀ | 0 | **+1** |

The signature partitions **exactly** by physical class. φ⁻¹ occurs in the two gauge couplings and
nowhere else; π occurs in α_EM's correction and in Ξ and nowhere else; everything else is pure
Fibonacci ratio. Three roles: **Fibonacci indices count, φ scales, π closes.**

**α_EM's φ-power is −1 — not −13, not −8.55.** The framework encodes scale in *which* Fibonacci
numbers appear and uses φ exactly once. `fibonacci_depth_coupling` encodes scale as a *power of
φ* and drops the indices. Structurally incompatible representations of the same thing, which is
the single root cause of everything in this section.

**Consequence for the propagation**: an energy scale is a coupling-like object, so it should
carry φ⁻¹ and a Fibonacci index multiset. `α(d)²·m = φ^(−2d)·m/5` has neither.

*Caveat kept visible*: these formulas are constructions, so their form partly reflects how they
were found. The non-trivial part is that sin²θ_W = 3/13 works **exactly** without a φ — if it
needed one it would not be 3/13.

## 7. Two leads, with their strength stated

### α_s is missing a π correction

α_s is the only coupling without one, and by far the worst:

```
alpha_EM  base only  2.658%  ->  with pi correction   5.7 ppm
alpha_s   base only  1.712%  (2.24 sigma vs 0.1179 +/- 0.0009)
```

The class signature predicts it needs one of the same form. It does — with the caveat that **α_s
cannot select which**, at ±0.76% measurement against a 1.7% correction. Four candidates land
inside 0.2σ, and with ~72 in the scan a chance hit runs about one in four.

### The gaps are Lucas, and they select

exp_37 (`minimum_actualization_resolution`) asks *"why are index gaps (a−b) themselves
Fibonacci?"* — **they are not Fibonacci, they are Lucas.**

```
alpha_EM  a=10 b=7  n=4  sign=-   gap 3 = L_2
G         a=13 b=6  n=1  sign=+   gap 7 = L_4
Omega_L   a=9  b=5  n=4  sign=+   gap 4 = L_3
                                  and 3 + 4 = 7
```

4 and 7 are not Fibonacci numbers. All three are Lucas, obeying the same recursion — and Lucas
and Fibonacci are the two independent solutions of that recursion, F_n = (φⁿ−ψⁿ)/√5 against
L_n = φⁿ+ψⁿ. **Fibonacci in the terms, Lucas in the gaps**: difference mode and sum mode of one
structure.

Constraining α_s's correction to the observed gap set {3,4,7} leaves exactly one survivor:

```
F11/(4 pi F8^2)  ->  alpha_s = 0.117742   err 0.134%   0.18 sigma   gap 3   n=4
```

α_EM's correction is (a,b) = (10,7); this is **(11,8)** — both indices up one, gap and n
preserved. α_s's correction is α_EM's one step up the ladder.

**Strength**: the gap set is three data points and the {3,4,7} subset was chosen after seeing
them; the looser "any Lucas" constraint leaves three candidates. This is a lead with a mechanism,
not a result.

**What makes it testable rather than numerology**: it predicts the next member of the family sits
at **(12, 9)**, and that **no correction anywhere in the corpus has a non-Lucas gap**. Both are
checkable against constants not used to build the claim.

### TESTED SAME DAY — the selection does not survive

Two further corrections exist in code and are absent from exp_37's list:

| quantity | a | b | n | sign | gap | source |
|---|---|---|---|---|---|---|
| Lambda (cosmological constant) | 3 | 5 | 4 | − | **−2** | M8 `exp_08_cosmological_constant_precision.py:84` |
| dark coupling | 8 | 6 | 4 | − | +2 | M8 `core/bsm.py:208` |

**Negative gaps occur**, so the criterion must be `|gap|`, not signed gap — and that readmits the
candidate I excluded. `F₃/(4π F₄²)` has gap −1, and |−1| = 1 = L₁ is Lucas. It returns as the
**best** fit at 0.026%, and four candidates now survive instead of one. **The unique selection was
an artifact of excluding negative gaps and choosing the {3,4,7} subset after seeing it.**

The constraint is also far weaker than it looked: 1, 2, 3 and 4 are *all* Lucas, so
P(all five observed gaps Lucas by chance) is 0.095 on a uniform 1–8 range, 0.031 on 1–12.

**Verdict: not established.** No counterexample was found — no gap of 5 or 6 appears anywhere —
but the data has no power to distinguish "gaps are Lucas" from "gaps are small integers".

**What survives:**
- exp_37's premise is still wrong independently of any of this: the gaps are **not Fibonacci**,
  since 4 and 7 are not Fibonacci numbers.
- **|gap| ∈ {2,3,4,7}, with 5 and 6 never occurring** across five corrections. Weak, but it is
  the actual observation, and it is what a larger inventory would test.
- A **documentation discrepancy**: exp_37 lists Omega_Lambda as (a=9, b=5, n=4, +), while M8's
  code applies (a=3, b=5, n=4, −) to the cosmological constant. Either different quantities or an
  inconsistency between what is documented and what runs — worth resolving before anyone builds
  on either.

Nothing in §5 or §6 depends on this. The depth-slot diagnosis and the phi/pi class signature are
untouched.

## 8. The framing that makes this a programme

The selection problem is not a precision problem. α_s will never discriminate its own correction.
What selected it here was **consistency with the other constants** — over-determination rather
than measurement.

Stated as physics: if the terms and their gaps are the two modes of one recursion, then a
configuration that does not respect both does not close, and the corrections are not free
parameters at all — they are whatever makes the global ledger balance. Dissonance is the signal;
harmonisation is the dynamics. That is PAC as a *selection principle* rather than a bookkeeping
constraint, and it is the natural home for the question exp_37 opened and left standing.

## 9. Next

- exp_06, exp_08, exp_09 — the remaining scale call sites, expecting some tautological passes to
  become real tests and possibly fail
- Check the (12,9) prediction and the no-non-Lucas-gap claim against constants not used here
- `np.trapz` in six other files
- `midnight/core/phase_rate.py` has the same reorg path break
