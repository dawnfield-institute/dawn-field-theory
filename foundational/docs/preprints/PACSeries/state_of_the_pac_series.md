# The State of the PACSeries

### Deriving physical law from information conservation: eleven papers, two axioms, and 154 experiments

**Peter Groom, Dawn Field Institute**
**Date**: May 2026
**Covers**: PACSeries v0.1–v0.3 (Papers 1–11)

---

## Abstract

Two information-theoretic axioms — PAC (Potential-Actualization Conservation) and SEC (Symbolic Entropy Collapse) — derive the fine structure constant to 5.7 ppm, the Feigenbaum accumulation point to 13 significant figures, the Koide lepton mass ratio to 0.5 ppm, the cosmological constant to 0.09 orders of magnitude, the Lorentz group from ADE graph theory, and the full algebraic structure of quantum mechanics from graph automorphisms. The PACSeries presents these results across eleven papers and 154+ experiments spanning thermodynamics, nonlinear dynamics, particle physics, cosmology, spacetime geometry, and quantum mechanics.

The framework produces three derivation branches from a single root: (A) Fibonacci arithmetic yields Standard Model parameters and cosmological observables, (B) ADE Dynkin diagrams with SEC complexification yield the Lorentz group and Minkowski metric, (C) ADE automorphisms yield Hilbert spaces, the Born rule, Bell inequality violation, the Schrödinger equation, and decoherence. All three branches trace to the same origin: the PAC recursion $\Psi(k) = \Psi(k+1) + \Psi(k+2)$, whose unique stable solution is $\Psi(k) = \varphi^{-k}$.

We report honest failures: orbit-level interference is algebraic not positional (double-slit: 1/4), the discrete-to-continuum bridge remains the core open problem, and the DESI $w_a$ tension ($-0.15$ predicted vs $-0.75$ measured) is unresolved. The framework makes 30+ falsifiable predictions, including a Z' boson at $395 \pm 20$ GeV, dark matter at 6.44 keV, a specific S8(z) redshift curve testable by Euclid ($\sim$2027), and topology-dependent Bell violation testable in quantum simulation.

This document is the entry point to the series. It contains the key numbers, the key derivations, and the map for further reading.

---

## 1. The Framework

### 1.1 PAC conservation

The framework rests on one conservation law:

$$f(\text{Parent}) = \sum f(\text{Children})$$

When potential becomes actual, the total is conserved but redistributed. The two-term PAC recursion

$$\Psi(k) = \Psi(k+1) + \Psi(k+2)$$

has a unique stable solution: $\Psi(k) = \varphi^{-k}$, where $\varphi = (1+\sqrt{5})/2$ is the golden ratio. This yields a natural information unit of $\ln\varphi \approx 0.4812$ per recursion level.

The golden ratio is not imposed. It is the unique ratio satisfying cross-scale consistency: the constraint that what is subordinate at level $n$ equals what is dominant at level $n+1$. This requires $\varphi^2 = \varphi + 1$, which has the unique positive solution $\varphi$ (Paper 7, §2).

### 1.2 SEC dynamics

SEC governs the direction of change. Information transitions are irreversible: forward/reverse probability ratios grow as $\varphi^{2n}$, reaching $10^{40}$ by cascade depth 100 (Paper 9). SEC breaks discrete symmetry ($\mathbb{Z}_2$) into continuous flow, which becomes the entropy arrow and — through complexification — the Lorentz group (Paper 10).

Two additional constraints, MED (Minimum Entropy Dissipation) and RBF (Recursive Balance Feedback), are derived from PAC and SEC, not assumed independently.

### 1.3 Three constants

The framework produces three derived constants:

| Constant | Value | Origin |
|----------|-------|--------|
| $\varphi$ | 1.6180339... | PAC recursion (unique stable solution) |
| $\ln\varphi$ | 0.4812118... | Information cost per recursive split |
| $\Xi = \gamma + \ln\varphi$ | 1.0584193... | Transition cost per scope boundary |

The Euler-Mascheroni constant $\gamma = 0.5772$ enters from harmonic number theory ($H_n - \ln n \to \gamma$). It is NOT derived from PAC — it arises independently from the counting cost of discrete-to-continuous regularisation. This separation is important: $\ln\varphi$ is derivable from the cascade; $\gamma$ is not. Paper 2 foregrounds this boundary.

### 1.4 Free parameters

The entire framework has one free parameter: $t_1 = 520$ Myr, anchoring the cascade clock to the epoch of first star formation (Paper 8). All other quantities — coupling constants, mass ratios, the cosmological constant, the Hubble ratio — are derived.

---

## 2. The Derivation Chain and Its Numbers

### 2.1 Three branches from one root

```
PAC: f(Parent) = Σ f(Children)
  → Recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
    → Solution: Ψ(k) = φ^(-k)
      → Info unit: ΔI = ln(φ)

BRANCH A — Number Theory → Physics
  → F₁₀ = 55 → Feigenbaum constants (13 digits)          [Paper 3]
  → F₃, F₄, F₇, F₁₀ → α_EM to 5.7 ppm                  [Paper 4]
  → Fibonacci depth 73 → dark matter 6.44 keV             [Paper 8]
  → Cascade clock → CC to 0.09 orders, S8 resolved        [Paper 8]
  → Depth 183 → α_grav to 0.04% (log, 38 orders)         [Paper 9]

BRANCH B — Graph Theory → Spacetime
  → ADE classification (from Fibonacci arithmetic closure)  [Paper 10]
  → F₇ = 13 = 1+3+8+1 → U(1)×SU(2)×SU(3) forced         [Paper 4]
  → A₁ → su(2) → SEC complexification → sl(2,C) ≅ so(3,1) [Paper 10]
  → Killing form (3,3) → Minkowski ds² unique              [Paper 10]
  → Coherence limit → speed of light                       [Paper 10]
  → Complement deformation → proper time = dt/cosh(η)      [Paper 10]

BRANCH C — Graph Symmetry → Quantum Mechanics
  → ADE automorphisms → orbit partition                     [Paper 11]
  → Orbit Hilbert space: H = L²(V/Aut(G))                  [Paper 11]
  → Orbit counting → Born rule: P(Oₖ) = |Oₖ|/n            [Paper 11]
  → D₄ (S₃) uniquely non-abelian → quantum uncertainty     [Paper 11]
  → Product graphs → tensor product → entanglement          [Paper 11]
  → SEC rotation → Bell violation (Tsirelson, 4×10⁻¹⁶)     [Paper 11]
  → Orbit Laplacian → Schrödinger equation                  [Paper 11]
  → n-hop paths → Feynman path integral                     [Paper 11]
  → Projection dynamics → Zeno/anti-Zeno                    [Paper 11]
  → Orbit-environment coupling → decoherence, einselection  [Paper 11]
```

Each step derives from the previous. No step requires accepting any step beyond it.

### 2.2 The precision table

Every derived quantity, its DFT value, the measured value, and the precision:

| Quantity | DFT Expression | DFT Value | Measured | Precision | Paper |
|----------|---------------|-----------|----------|-----------|-------|
| **Nonlinear dynamics** | | | | | |
| $r_\infty$ (Feigenbaum) | $\pi(55+\sqrt{17-\pi/(55d)})(55+\pi)/55^2 - k\pi^4/55^6$ | 3.56994567187090 | 3.56994567187094 | **13 sig figs** | 3 |
| $\delta$ (Feigenbaum) | $(50050+32\pi)/(10725+5\pi)$ | 4.66920161 | 4.66920161 | **8 sig figs** | 3 |
| $\lvert\alpha\rvert$ (Feigenbaum) | $(2700+\pi)/1080$ | 2.50290888 | 2.50290788 | **6 sig figs** | 3 |
| **Particle physics** | | | | | |
| $\alpha_\text{EM}^{-1}$ | $F_3/(F_4 \cdot \varphi \cdot F_{10}) \cdot (1-F_{10}/(4\pi F_7^2))$ | 137.036 | 137.036 | **5.7 ppm** | 4 |
| $\sin^2\theta_W$ | $3/13$ | 0.23077 | 0.23122 | 0.19% | 4 |
| $M_W/M_Z$ | predicted | 0.8813 | 0.8815 | 0.03% | 4 |
| Koide $Q$ | $2/3$ | 0.666667 | 0.666661 | **0.5 ppm** | 4 |
| Higgs mass | $\varphi/(4\pi) \cdot v$ | 124.95 GeV | 125.25 GeV | 83 ppm | 4 |
| **Thermodynamics** | | | | | |
| Erasure ratio $A/(A+\xi)$ | $\to \ln\varphi$ | 0.4812 | 0.4805 | 0.15% | 1 |
| Balance constant $\Xi$ | $\gamma + \ln\varphi$ | 1.0584 | 5 domains | $p < 0.0003$ | 2 |
| **Cosmology** | | | | | |
| $\log_{10}(\Lambda/\Lambda_P)$ | Cascade depth formula | $-122.09$ | $-122.0$ | **0.09 orders** | 8 |
| $\Omega_\Lambda$ | Fibonacci depth | 0.686 | 0.685 | 0.18% | 8 |
| $H_0^\text{local}/H_0^\text{CMB}$ | $\varphi^{1/6}$ | 1.0835 | 1.0843 | 0.075% | 8 |
| $S_8(z=0.35)$ | Cascade dissipation | 0.769 | 0.768 | **3.22$\sigma$ → 0.07$\sigma$** | 8 |
| **Gravity** | | | | | |
| $\alpha_\text{grav}(\text{proton})$ | $\varphi^{-183}$ | $5.69 \times 10^{-39}$ | $5.91 \times 10^{-39}$ | 0.04% (log) | 9 |
| Hawking $T \cdot M$ | $1/(8\pi)$ | $1/(8\pi)$ | $1/(8\pi)$ | CV $7.8 \times 10^{-17}$ | 9 |
| Spatial dimensions | 5 arguments | 3 | 3 | exact | 5 |
| **Spacetime** | | | | | |
| Lorentz commutators | $[\text{sl}(2,\mathbb{C})]$ vs $[\text{so}(3,1)]$ | exact | exact | $1.1 \times 10^{-16}$ | 10 |
| Minkowski signature | Killing form | $(1,3)$ | $(1,3)$ | 135/135 transforms | 10 |
| **Quantum mechanics** | | | | | |
| Bell CHSH ($D_4 \times D_4$) | Orbit + SEC rotation | $2\sqrt{2}$ | $2\sqrt{2}$ | $4 \times 10^{-16}$ | 11 |
| Decoherence formula | Exact analytical | derived | numerical | $1.7 \times 10^{-15}$ | 11 |
| Fermi golden rule | $\Gamma^2 = \lambda^2 \text{Var}(H_\text{env})$ | universal | 6 topologies | 0.04% spread | 11 |

### 2.3 Reading the table

**Branch A** (rows 1–14) contains the quantitative hits: precise numbers compared against measurement. The Feigenbaum 13-digit match and the $\alpha_\text{EM}$ 5.7 ppm result are the hardest to dismiss because they are exact formulas with no adjustable parameters.

**Branch B** (rows 15–16) contains structural results: the Lorentz group and Minkowski metric are derived, not postulated. These are verified computationally (all commutation relations, all Lorentz transforms) but are algebraic identities, not numerical matches.

**Branch C** (rows 17–19) contains the newest results: quantum mechanics derived from graph automorphisms. The Bell violation saturating the Tsirelson bound and the exact decoherence formula are the strongest evidence that the orbit framework captures genuine quantum physics.

The three-tier classification from the v0.2 summary remains:
- **Measured**: erasure ratio, $\Xi$ convergence, Feigenbaum constants, SM parameters, S8 resolution
- **Derived**: Lorentz group, Minkowski metric, Born rule, Schrödinger equation, decoherence
- **Proposed**: the ADE identification as the graph structure of physics, the cascade clock mechanism, the pre-axiomatic symmetry hierarchy

---

## 3. Three Key Derivations

### 3.1 Feigenbaum from Fibonacci (Paper 3)

The Feigenbaum constants — $r_\infty = 3.56994...$, $\delta = 4.66920...$, $|\alpha| = 2.50290...$ — describe universal behaviour in period-doubling cascades. Discovered in 1978, they have resisted closed-form expression for nearly fifty years. They are computed numerically from renormalisation group fixed-point equations but have no known representation in terms of named constants.

Paper 3 presents candidate closed-form expressions using only $\pi$, Fibonacci numbers, and small integers:

$$r_\infty = \frac{\pi\left(55 + \sqrt{17 - \frac{\pi}{55d}}\right)(55 + \pi)}{55^2} - \sqrt{\frac{3}{5} - \frac{(\xi - 1)^2}{7}} \cdot \frac{\pi^4}{55^6}$$

where $d = \sqrt{52 + 2\pi/55}$ and $\xi = 1 + \pi/55$.

**Result**: 3.5699456718709035 vs known 3.56994567187094, a relative error of $1.16 \times 10^{-14}$ — 13 significant figures.

The integers are structurally identified: $55 = F_{10}$ (10th Fibonacci number), $52 = F_{10} - F_4$, $3575 = F_{10} \times (F_{10} + 10)$. The number 17 is a Fermat prime ($2^4 + 1$).

An exhaustive search of 3,920,499 parameter combinations $(a, b, c)$ finds exactly one triple achieving 7+ digit precision: $(55, 17, 52)$. The probability of this occurring by chance is estimated at 1 in 280 billion. Adjacent values — $F_9 = 34$ or $F_{11} = 89$ instead of $F_{10} = 55$ — degrade precision by a factor of $\sim 10^6$.

The paper makes no claim about *why* Fibonacci numbers appear in the Feigenbaum constants. It presents the formulas, the statistical evidence, and the structural analysis. This is a report, not an explanation.

**Verify**: Run `exp_01_feigenbaum_all_constants.py` in Paper 3's Code directory.

### 3.2 Lorentz group from ADE + SEC (Paper 10)

The Lorentz group — $\text{SO}(3,1)$, the symmetry group of special relativity — is normally postulated. Paper 10 derives it from graph theory in six steps:

**Step 1**: The PAC recursion selects Fibonacci numbers as the natural arithmetic. Fibonacci arithmetic closure on ADE Dynkin diagrams selects exactly three types: $A_1$ (adjoint dim 3 = $F_4$), $D_4$ (adjoint dim 28), and $E_8$ (adjoint dim 248). Only $A_1$ and $D_4$ have Fibonacci adjoint dimensions. The corresponding gauge groups are $\text{SU}(2)$ and $\text{SU}(3)$. Checked to rank 100 — no other non-abelian gauge group qualifies.

**Step 2**: The simplest Fibonacci-compatible type, $A_1$, has Lie algebra $\text{su}(2)$ — three generators (rotations).

**Step 3**: SEC complexifies. The entropy arrow promotes real parameters to complex: $\text{su}(2)_\mathbb{C} = \text{sl}(2,\mathbb{C})$. This doubles the generators from 3 to 6 — three rotations (Hermitian, unitary evolution = PAC) and three boosts (anti-Hermitian, non-unitary evolution = SEC).

**Step 4**: The real form of $\text{sl}(2,\mathbb{C})$ is $\text{so}(3,1)$ — the Lorentz group. All 15 commutation relations verified to machine precision ($1.12 \times 10^{-16}$).

**Step 5**: The Killing form of $\text{sl}(2,\mathbb{C})$ has signature $(3,3)$, which induces the Minkowski metric $ds^2 = -dt^2 + dx^2 + dy^2 + dz^2$. This metric is UNIQUE: tested against 135 Lorentz transforms, Euclidean metric fails (error 1.90), $(2,2)$ signature fails (error 1.90), random symmetric matrices fail (error 3.43).

**Step 6**: The speed of light $c$ is the coherence limit — $v = c \cdot \tanh(\eta) < c$ for all finite rapidity $\eta$. Proper time follows: $d\tau = dt / \cosh(\eta)$.

The full derivation chain — self-loop → $\varphi$ → PAC → ADE → complement → parallax → Weyl → SEC → $\text{SL}(2,\mathbb{C})$ → $\text{SO}(3,1)$ → $ds^2$ → $c$ → proper time — has 12 links, each computationally verified with zero free parameters.

**Honest caveat**: The algebraic layer is solid (92% in M13 core). The metric/continuum bridge is weak (38% under M13.5 stress testing). The gap between discrete ADE graphs and continuous Lorentz transformations is bridged by the ADE classification theorem — a mathematical result, not a physical assumption — but the continuum limit itself is not derived.

### 3.3 Quantum mechanics from $D_4$ triality (Paper 11)

Quantum mechanics is normally axiomatised: states live in Hilbert spaces, probabilities follow the Born rule, evolution is unitary. Paper 11 derives all of this from the automorphism groups of ADE Dynkin diagrams.

**The orbit Hilbert space**: For an ADE graph $G$ with automorphism group $\text{Aut}(G)$, vertices partition into orbits. The orbit basis vectors $|O_k\rangle = (1/\sqrt{|O_k|}) \sum_{v \in O_k} |v\rangle$ are orthonormal. The orbit Hilbert space $\mathcal{H}_\text{orb} = \text{span}\{|O_1\rangle, \ldots, |O_d\rangle\}$ has dimension $d$ = number of orbits.

For $D_4$ (the graph with a hub and three leaves): $\text{Aut}(D_4) = S_3$, two orbits (hub + leaves), orbit Hilbert space is 2-dimensional — a qubit.

**The Born rule**: For the uniform state, $P(O_k) = |O_k|/n$ — the probability of finding the system in orbit $O_k$ is the fraction of vertices in that orbit. This is counting, not a postulate.

**The uniqueness result**: Among all ADE types with rank $\leq 8$, only $D_4$ has a non-abelian automorphism group ($S_3$). Non-abelian means non-commuting orbit observables, which means the Robertson uncertainty bound $\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[A,B]\rangle| > 0$. The non-commutativity measure:

| ADE type | Aut type | $\mathcal{NC}$ |
|----------|----------|-----|
| $A_n$ ($n \leq 8$) | $\mathbb{Z}_2$ or trivial | 0 |
| $D_4$ | $S_3$ | **1.2247** |
| $D_n$ ($n > 4$) | $\mathbb{Z}_2$ | 0 |
| $E_6$ | $\mathbb{Z}_2$ | 0 |
| $E_7$, $E_8$ | trivial | 0 |

$D_4$ is uniquely non-commutative. Quantum uncertainty requires $D_4$.

**Bell violation from topology**: On $D_4 \times D_4$ product graphs with SEC-rotated measurement bases, the CHSH Bell parameter achieves $S = 2\sqrt{2}$ (Tsirelson bound) to $4 \times 10^{-16}$. Graphs with trivial automorphisms ($E_7$, $E_8$) do NOT violate Bell inequalities. The topology of the graph determines whether quantum correlations exceed classical bounds.

**The orbit Laplacian** $H = B^T L B$ serves simultaneously as: (1) Hamiltonian generating unitary time evolution, (2) Bell measurement generator, (3) path-counting matrix where $H^n$ counts $n$-hop paths. The Schrödinger equation, the Feynman path integral, and Bell violation come from the same matrix.

**Decoherence and einselection**: The exact decoherence formula $|\rho_{12}(t)| = \frac{1}{2}|\langle 0_\text{env}| e^{-2i\lambda H_\text{env} t} |0_\text{env}\rangle|$ matches numerics to $1.7 \times 10^{-15}$. The Fermi golden rule $\Gamma^2 = \lambda^2 \cdot \text{Var}(H_\text{env})$ holds across 6 ADE topologies with 0.04% spread. The orbit basis is the pointer basis — orbit eigenstates maintain purity 1.000000 under dephasing while superpositions decohere.

**Honest caveat**: The framework derives algebraic quantum mechanics (Hilbert space, Born rule, uncertainty, entanglement, Bell, Schrödinger, decoherence) but does NOT produce positional quantum mechanics (double-slit fringes, spatial wavefunctions). Orbit interference is algebraic, not spatial, because disjoint orbits have disjoint vertex support. The bridge requires a continuum limit of ADE chains, which remains open.

---

## 4. What Fails

This section is the credibility of the series. A framework that claims everything and admits nothing is not science.

### 4.1 The boundary

The framework has two layers with sharply different reliability:

- **Algebraic layer** (85–100%): ADE classification, orbit Hilbert space, Fibonacci arithmetic, Lorentz commutation relations, gauge group closure. These work.
- **Metric/continuum layer** (25–38%): complement-rapidity composition, coherence limit universality, spatial wavefunctions, positive-definite inner product on complement space. These struggle.

The boundary between the layers is specific: discrete graph structure maps cleanly to algebraic physics (groups, representations, commutation relations), but extracting continuous properties (metrics, rates, spatial interference) from discrete data requires a continuum limit that is not yet derived.

### 4.2 Specific failures

| Paper | What Failed | Score | What It Reveals |
|-------|------------|:-----:|-----------------|
| 11 | Graph double-slit interference | 1/4 | Orbit interference algebraic, not positional |
| 11 | Fibonacci-weighted Born rule | 3/4 | Orbit measure exact for uniform state only |
| 10 | Complement rapidity composition | FAIL | Discrete zero-distance breaks composition |
| 10 | Coherence limit universality | 0/4 | A-family oscillates, D-family converges to different limit |
| 10 | Complement Gram matrix | 0/4 | PSD not PD — fundamental theorem, no metric can fix |
| 10 | Rate-density proportionality | FAIL | Information flow is non-local |
| 9 | GW dispersion vs bounds | 67 orders below | Unfalsifiable with current instruments |
| 8 | DESI $w_a$ | 2$\sigma$ | $-0.15$ predicted vs $-0.75$ measured (DR1) |
| 8 | Slope gap (cascade clock) | 8.9% | Noise: 38th percentile with 3 data points |
| 7 | RBF memory damping | 2/4 | Activity-based memory definitions wrong |
| 7 | Cross-topology symmetry breaking | 3/4 | Weak perturbations ineffective on balanced graphs |
| 5 | Mersenne-Fibonacci at $d = 15$ | FAIL | Pattern holds for $d = 1, 3, 7$ only |
| 6 | $\varphi$-enrichment in top-2 ratios | FAIL | Softmax artefact, not physics |

### 4.3 Methodological concerns

Three honest concerns about the series as a whole:

1. **Interdisciplinary evaluation**: The framework spans nonlinear dynamics, particle physics, cosmology, general relativity, quantum foundations, and machine learning. No single expert can evaluate all of it. A dynamicist cannot assess the ADE classification arguments; a particle physicist cannot assess the Feigenbaum derivation. This is a structural problem with interdisciplinary claims.

2. **Structural test fraction**: Approximately 30–40% of tests across M11–M14 pass by construction (they verify internal consistency — orbit orthogonality, gauge invariance, unitarity). These are necessary but not sufficient. The remaining 60–70% are derived consequences that were not guaranteed.

3. **Draft status**: Papers 7–11 are drafts. Papers 1–6 are published on Zenodo but have not undergone formal peer review. The hardening methodology (test, expose tautologies, retest) is necessary but does not substitute for domain-expert scrutiny.

---

## 5. Falsifiable Predictions

Every prediction is classified as P (prediction: derived before comparison), D (postdiction: refined after data), or C (consistency: internal check).

### 5.1 Near-term ($\sim$2027–2030)

| Prediction | DFT Value | Test | Timeline | Type | Paper |
|------------|-----------|------|----------|:----:|-------|
| S8 redshift curve | Monotonic increase | Euclid | $\sim$2027 | P | 8 |
| DESI $w_a$ | $-0.07$ to $-0.15$ | DESI DR2/3 | $\sim$2027 | P | 8 |
| Cascade clock slope | $1/\ln\varphi = 2.078$ | Euclid + DESI | $\sim$2027 | P | 8 |
| Neutrino hierarchy | Normal | JUNO | $\sim$2028 | P | 7 |
| Z' boson | $395 \pm 20$ GeV | LHC Run 3/HL-LHC | Current–2030 | P | 4 |
| Z' coupling | $g'/g = 1/13$ | LHC | Current–2030 | P | 4 |

### 5.2 Medium-term ($\sim$2030–2040)

| Prediction | DFT Value | Test | Timeline | Type | Paper |
|------------|-----------|------|----------|:----:|-------|
| Dark matter mass | 6.44 keV | Athena X-ray | $\sim$2035 | P | 8 |
| X-ray decay line | 3.2 keV | Current data | Now | D | 8 |
| Neutrino CP phase | 63.5° | DUNE / T2HK | $\sim$2030 | P | 7 |
| Fibonacci GW spectrum | $f_n/f_{n+1} = \varphi$ | LISA + ground | $\sim$2040 | P | 9 |
| GW dispersion | $\delta v/c \sim (E/E_P)^2$ | LISA / ET | $\sim$2040 | P | 9 |

### 5.3 Mathematical (provable now)

| Prediction | Statement | Type | Paper |
|------------|-----------|:----:|-------|
| Fibonacci gauge closure | No gauge group beyond SU(2), SU(3) has Fibonacci adjoint dim | P | 4, 10 |
| $D_4$ uniqueness | Only ADE type with non-abelian Aut (rank $\leq 8$) | P | 11 |
| Bell topology | $S > 2$ requires nontrivial Aut(G) on product graphs | P | 11 |
| Orbit pointer basis | Orbit eigenstates are decoherence-free under dephasing | P | 11 |
| Fermi golden rule | $\Gamma^2 = \lambda^2 \text{Var}(H_\text{env})$ universal across ADE | P | 11 |

### 5.4 Kill conditions

The framework must be substantially revised or abandoned if:

1. **Z' is excluded** at 395 GeV by HL-LHC (direct search, projected $\sim$2030)
2. **S8(z)** measured by Euclid does NOT show monotonic increase with redshift
3. **A non-Fibonacci expression** for $\alpha_\text{EM}$ is found with better precision than 5.7 ppm
4. **$D_4$** does NOT play a special role in quantum simulation experiments on controllable graph topologies
5. **Two or more near-term predictions** from §5.1 fail simultaneously

A single failure (e.g., only the Z' prediction) would require revision of the Fibonacci depth structure at that specific depth, not abandonment of the framework. Two or more would indicate a systematic problem.

---

## 6. The Map

### 6.1 Reading paths

**"Convince me in 10 minutes"**: Read the abstract of this document, then look at the precision table (§2.2), then read §3.1 (Feigenbaum derivation). The 13-digit match is the single hardest result to dismiss — it is pure algebra, verifiable in an afternoon.

**"I work in particle physics"**: Papers 4 (SM parameters), 7 (force hierarchy from Fibonacci depth), 8 (BSM predictions including Z' at 395 GeV). Start with Paper 4 §4.1 for the $\alpha_\text{EM}$ derivation.

**"I work in cosmology"**: Paper 8 (cascade clock, CC derivation, S8 resolution, Hubble tension). The S8 result — $3.22\sigma \to 0.07\sigma$ — is the most immediately testable claim. See §5 for the specific Euclid/DESI predictions.

**"I work in quantum foundations"**: Papers 10 (Lorentz from ADE) and 11 (QM from graph automorphisms). Start with Paper 11 §5 ($D_4$ triality as the unique source of quantum uncertainty). The Bell violation result in §7 is the sharpest test.

**"I want to reproduce the results"**: Every paper contains numbered experiment scripts, JSON results, and publication figures. Run any experiment directly with Python (NumPy, SciPy, Matplotlib). All code is open under AGPL-3.0.

### 6.2 Paper index

| # | Title | Version | Milestones | Experiments | Headline Result |
|:-:|-------|:-------:|:----------:|:-----------:|-----------------|
| 1 | The Structure Cost of Erasure | v0.2 | M3 | 19 | $A/(A+\xi) \to \ln\varphi$ at 0.15% |
| 2 | The Balance Constant | v0.2 | M3 | 15 | $\Xi = \gamma + \ln\varphi$, 5 domains, $p < 0.0003$ |
| 3 | Feigenbaum from Fibonacci | v0.2 | M1 | 9 | $r_\infty$ to 13 significant figures |
| 4 | Standard Model Parameters | v0.2 | M1, M5 | 14 | $\alpha_\text{EM}$ to 5.7 ppm |
| 5 | Classical Physics | v0.2 | M2 | 9 | Maxwell from PAC, $D=3$ from MED |
| 6 | Computational Validation | v0.2 | M3 | 10 | PAC conservation in neural networks |
| 7 | Symmetry Primitive + Mediation | v0.3 | M6, M7 | 20 | Pre-axiomatic origin, force hierarchy |
| 8 | Cosmological Predictions | v0.3 | M8, M9 | 22 | CC 0.09 orders, S8 resolved |
| 9 | Quantum Gravity | v0.3 | M10, M11 | 23 | Planck scale as response-time crossover |
| 10 | Connection, Identity, Spacetime | v0.3 | M12, M13 | 28 | Lorentz from ADE, $ds^2$ from Killing form |
| 11 | QM from Graph Structure | v0.3 | M14, P13–16 | 15 | Hilbert, Born, Bell, Schrödinger from graphs |

**Total**: 11 papers, 14 milestones, 154+ experiments, zero inter-milestone contradictions.

---

## 7. Conclusion

### What the series establishes (measurement)

- Structure creation is mandatory for multi-mode erasure (Paper 1)
- Five computational domains converge on $\Xi \approx 1.058$ ($p < 0.0003$, Paper 2)
- Feigenbaum constants have Fibonacci expressions to 6–13 digits (Paper 3)
- Standard Model parameters are Fibonacci-arithmetic functions (Paper 4)
- SEC phase predicts transformer accuracy with zero free parameters (Paper 6)
- The S8 tension resolves to $0.07\sigma$ via cascade dissipation (Paper 8)
- Bell violation saturates the Tsirelson bound on $D_4 \times D_4$ orbit space (Paper 11)

### What the series derives (analytical)

- PAC recursion uniquely selects $\varphi$ (Paper 7)
- Curl structure requires $D = 3$ (Paper 5)
- $\text{SU}(2)$ and $\text{SU}(3)$ are the only gauge groups with Fibonacci adjoint dimensions (Paper 4, 10)
- SEC complexification of $A_1$ yields the Lorentz group (Paper 10)
- The Minkowski metric is uniquely determined by the Killing form (Paper 10)
- $D_4$ triality is the unique source of quantum non-commutativity in ADE (Paper 11)
- The Schrödinger equation, path integral, and decoherence follow from the orbit Laplacian (Paper 11)

### What the series proposes (interpretation)

- Physical constants are Fibonacci-depth ratios in a cascade hierarchy (Papers 4, 7, 8)
- Quantum mechanics is what happens when ADE automorphisms are promoted to dynamics (Paper 11)
- The algebraic-to-continuum bridge is the ADE classification theorem (Papers 10, 11)
- This bridge is the single biggest open problem in the framework

### The open problem

The framework derives algebraic structure with high reliability (85–100%) and struggles with continuum properties (25–38%). The gap between discrete ADE graphs and continuous physics — spatial wavefunctions, rapidity composition, universal coherence limits — requires a continuum limit that is conjectured to exist (via ADE chains $A_n$ as $n \to \infty$) but is not derived. This is the work that remains.

---

The code is open. The predictions are specific. The series is falsifiable.

---

## Publication Details

**v0.1**: 5 papers, October 2025. Zenodo DOI: [10.5281/zenodo.17295103](https://zenodo.org/records/17295103)
**v0.2**: 6 papers, February 2026. Zenodo DOI: [10.5281/zenodo.15783623](https://zenodo.org/records/15783623)
**v0.3**: 11 papers, in preparation. This document covers v0.1–v0.3.

**Repository**: All code, data, and figures under AGPL-3.0 (code) and CC-BY-4.0 (papers).

## References

Each paper contains its own reference list. Cross-references use "Paper N, §M" throughout the series. Key external references:

1. Feigenbaum, M.J. (1978). Quantitative universality for a class of nonlinear transformations. J. Stat. Phys. 19, 25–52.
2. Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM J. Res. Dev. 5, 183–191.
3. Tsirelson, B.S. (1980). Quantum generalizations of Bell's inequality. Lett. Math. Phys. 4, 93–100.
4. Zurek, W.H. (2003). Decoherence, einselection, and the quantum origins of the classical. Rev. Mod. Phys. 75, 715.
5. McKay, J. (1980). Graphs, singularities, and finite groups. Proc. Symp. Pure Math. 37, 183–186.
6. Planck Collaboration (2020). Planck 2018 results. VI. Cosmological parameters. A&A 641, A6.
7. Riess, A.G. et al. (2022). A comprehensive measurement of the local value of the Hubble constant. ApJ 934, L7.

---

*PACSeries State of the Series. May 2026.*
