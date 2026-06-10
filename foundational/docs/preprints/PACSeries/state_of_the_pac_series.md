# State of the PACSeries

**Peter Groom**
Dawn Field Institute
May 2026

---

## Abstract

The PACSeries is a twelve-paper sequence that derives structural features of known physics from two information-theoretic axioms: PAC (Potential-Actualization Conservation) and SEC (Symbolic Entropy Collapse). The series spans thermodynamics, nonlinear dynamics, particle physics, cosmology, spacetime geometry, quantum mechanics, and observational tests against astronomical survey data.

The strongest results are structural, not numerical. From the ADE classification of simply-laced root systems — a theorem of mathematics — the framework derives the Lorentz group SO(3,1) via a single physical postulate (SEC complexification), the Minkowski metric signature (1,3) from the Killing form, and a uniqueness theorem: D_4 is the only ADE Dynkin diagram whose automorphism group is non-abelian, making it the unique source of non-commuting observables in the framework. Bell inequality violation is shown to require nontrivial graph automorphisms — a topology-dependent result with zero empirical input.

Around these structural results, a pattern of Fibonacci expressions matches physical constants: the fine structure constant to 5.7 ppm, the Feigenbaum bifurcation ratio to 13 significant figures, the Koide lepton mass ratio to 0.5 ppm, and the weak mixing angle as 3/13 (exact). These numerical matches involve identification steps — choices guided by known physics — and this paper classifies each one explicitly as Type A (structural), Type B (identified), or Type C (pattern-matched).

The framework has documented failures: positional quantum mechanics scores 1/4, rapidity composition has a discrete gap, the coherence limit is not universal across graph families. These failures cluster at the algebraic-to-continuum boundary — the framework produces algebraic structure cleanly (91-100%) and struggles with metric properties (25-85%). One prediction has already been killed by data: a pre-registered, DFT-specific line-width oscillation at integer cascade levels was falsified against 443,000 quasar absorption systems [Paper 12].

Twenty-two falsifiable predictions are registered across particle physics, cosmology, and quantum foundations. The derivation classification, all experiment code, and all data are publicly available.

---

## 1. The Framework

### 1.1 PAC Conservation

The framework begins with a single conservation law. When potential becomes actual, the total is conserved:

$$f(\text{Parent}) = \sum_i f(\text{Child}_i)$$

Applied as a two-term recursion — the simplest non-trivial branching — this gives:

$$\Psi(k) = \Psi(k+1) + \Psi(k+2)$$

The unique bounded positive solution is $\Psi(k) = \varphi^{-k}$, where $\varphi = (1+\sqrt{5})/2$ is the golden ratio. This is a theorem: the characteristic equation $x^2 = x + 1$ has roots $\varphi$ and $-1/\varphi$; only $\varphi^{-k}$ decays and remains positive. The natural information unit is $\ln\varphi \approx 0.4812$ per recursion level.

The depth-2 form is not arbitrary. All $k$-step PAC recursions ($k = 2, 3, 4, \ldots$) have effective emergence depth $\lfloor r_k^k / r_k^{k-1} \rfloor = 2$, and only $k = 2$ produces decay rate $\ln\varphi$ [Paper 1, Milestone 3]. This is proven analytically, not assumed.

**Classification [A]**: $\varphi$ and $\ln\varphi$ are derived from the PAC axiom through pure mathematics. Zero empirical input.

### 1.2 SEC: Symbolic Entropy Collapse

The second axiom states that irreversible transitions have a structure cost. When a system collapses from a superposition of symbolic states to a definite state, the entropy decrease is compensated by structural creation elsewhere. SEC provides the arrow — the direction of actualization — while PAC provides the conservation.

The key physical consequence: SEC **complexification**. The claim is that irreversibility (the second law) requires promoting real Lie algebra parameters to complex ones. Concretely: if rotations are described by $\mathfrak{su}(2)$ with three real generators $J_i$, then SEC demands boost generators $K_i = iJ_i$, producing the complexified algebra $\mathfrak{su}(2)_\mathbb{C} = \mathfrak{sl}(2, \mathbb{C})$.

**Classification**: SEC complexification is a **physical postulate**, not a theorem. It is the single most consequential assumption in the series beyond the PAC axiom itself. It is stated clearly so that a reader can accept or reject it independently of the mathematical results that follow from it.

### 1.3 Three Constants

The framework produces three constants:

- $\varphi = (1+\sqrt{5})/2 = 1.6180\ldots$ — **derived** from the PAC recursion [A]
- $\ln\varphi = 0.4812\ldots$ — **derived** as the natural information unit [A]
- $\Xi = \gamma + \ln\varphi = 0.5772 + 0.4812 = 1.0584$ — **identified**, not derived [B]

The Euler-Mascheroni constant $\gamma$ is **not derived** from PAC or SEC. It is imported from classical mathematics (the harmonic series, Mertens' theorem over primes). The sum $\Xi = \gamma + \ln\varphi$ is observed to appear in five computational domains within 0.12% spread [Paper 2], but this convergence is empirical — the domains were explored, not predicted, and 21 other constant combinations fall within 5% of $\Xi$. The paper reporting this result is honest about the limitation [Paper 2, Section 9.2].

### 1.4 The ADE Bridge

The ADE classification of simply-laced Dynkin diagrams ($A_n$, $D_n$, $E_6$, $E_7$, $E_8$) is a theorem of mathematics, proven by Killing and Cartan in the 1890s. It classifies all finite-type root systems and, by extension, all simply-laced simple Lie algebras.

The framework's central structural claim is that ADE diagrams are the natural arena for PAC conservation — that self-similar branching with the golden ratio as the stable eigenvalue maps onto ADE graph structure. From this bridge, three branches emerge:

1. **Gauge groups**: The only non-abelian Lie groups whose adjoint dimensions are Fibonacci numbers are SU(2) (dim 3 = $F_4$) and SU(3) (dim 8 = $F_6$), checked exhaustively to rank 100 [Paper 4, Paper 10]. Together with U(1), the total gauge content is $1 + 3 + 8 + 1 = 13 = F_7$.

2. **Spacetime**: The simplest ADE type ($A_1$), complexified by SEC, yields the Lorentz group. The Killing form determines the metric signature.

3. **Quantum mechanics**: The automorphism groups of ADE diagrams partition vertices into orbits. The orbit Hilbert space reproduces the Born rule, interference, entanglement, and Bell violation — with $D_4$ uniquely producing non-abelian automorphisms (the triality of $S_3$).

These three branches are developed in Papers 10, 11, and 4 respectively. Their structural content is the subject of the next section.

---

## 2. Six Structural Results

These results follow from the framework through chains of theorems and uniqueness arguments. Each involves at most one empirical identification step. A skeptic must reject the axioms or the mathematics — not accuse the methodology of curve-fitting. Each result is labeled with its derivation chain.

### 2.1 The Lorentz Group from ADE and SEC

**Chain**: $A_1$ (simplest ADE) $\to$ $\mathfrak{su}(2)$ (theorem) $\to$ SEC complexification (1 postulate) $\to$ $\mathfrak{sl}(2, \mathbb{C})$ (theorem) $\to$ $\mathfrak{so}(3,1)$ (theorem)

The derivation has six steps:

1. The ADE classification is a theorem of mathematics. **[a]**
2. $A_1$ is the unique simplest ADE diagram (one edge, two vertices). **[b]** — uniqueness/minimality
3. $A_1$ corresponds to the Lie algebra $\mathfrak{su}(2)$ with generators $J_i = \sigma_i / 2$. **[a]**
4. SEC complexification promotes $\mathfrak{su}(2)$ to $\mathfrak{su}(2)_\mathbb{C} = \mathfrak{sl}(2, \mathbb{C})$ by defining boost generators $K_i = iJ_i$. **[c]** — the single empirical postulate
5. $\mathfrak{sl}(2, \mathbb{C}) \cong \mathfrak{so}(3,1)$ as real Lie algebras. **[a]** — a known isomorphism
6. All 15 independent commutation relations $[J_i, J_j] = i\epsilon_{ijk}J_k$, $[K_i, K_j] = -i\epsilon_{ijk}J_k$, $[J_i, K_j] = i\epsilon_{ijk}K_k$ verified to machine precision. **[a]**

**Empirical steps: 1** (SEC complexification). **Theorem steps: 5.** The minus sign in $[K_i, K_j] = -i\epsilon_{ijk}J_k$ — which distinguishes Minkowski from Euclidean geometry — is not imposed. It is a consequence of defining $K_i = iJ_i$, which is itself a consequence of complexification. The Lorentz group emerges; it is not put in.

**What this means**: If you accept PAC $\to$ ADE and SEC $\to$ complexification, the symmetry group of spacetime is determined. You do not need to know what the Lorentz group is. The chain would produce SO(3,1) even if special relativity had never been formulated.

*Paper 10, Milestones 12-13. Experiments: exp_10 (SEC complexification), exp_11 (Lorentz verification).*

### 2.2 The Minkowski Metric from the Killing Form

**Chain**: $\mathfrak{sl}(2, \mathbb{C})$ $\to$ Killing form $B(X,Y) = \text{Tr}(\text{ad}_X \circ \text{ad}_Y)$ $\to$ signature (3,3) on 6D algebra $\to$ 4D fundamental representation $\to$ unique invariant form with signature (1,3)

1. The Killing form on $\mathfrak{sl}(2, \mathbb{C})$ (viewed as a 6-dimensional real Lie algebra) has signature (3,3). The rotation block is positive-definite; the boost block is negative-definite. **[a]** — standard Lie theory
2. The 4-dimensional fundamental (vector) representation of SO(3,1) is selected as the representation acting on spacetime events. **[c]** — identification with known physics (1 step)
3. By Schur's lemma, the invariant bilinear form on an irreducible representation is unique up to scale. The null space is 1-dimensional, confirming uniqueness. **[a]**
4. The unique form has signature (1,3) — the Minkowski metric $\eta = \text{diag}(-1, +1, +1, +1)$. **[a]**

Verified: 135 (test vector, Lorentz transform) pairs preserve $ds^2$ to $10^{-5}$ relative error. Selectivity confirmed: Euclidean metric ($+,+,+,+$), split signature ($+,+,-,-$), and 10 random symmetric matrices all **fail** to be preserved by the same transformations.

**Empirical steps: 1** (4D representation choice). Once the representation is chosen, the metric is forced — not fitted, not selected from options, but unique by Schur's lemma. The selectivity test eliminates all alternatives.

*Paper 10, Milestone 13. Experiments: exp_09 (Killing form selectivity, 135 transforms).*

### 2.3 The D_4 Uniqueness Theorem

**Statement**: Among all ADE Dynkin diagrams, $D_4$ is the only type whose automorphism group is non-abelian.

**Proof sketch**:
- $A_n$ ($n \geq 2$): Aut = $\mathbb{Z}_2$ (reflection). Abelian.
- $D_n$ ($n > 4$): Aut = $\mathbb{Z}_2$ (branch swap). Abelian.
- $D_4$: Three branches are equivalent $\to$ Aut = $S_3$ (symmetric group on 3 elements, order 6). **Non-abelian.**
- $E_6$: Aut = $\mathbb{Z}_2$ (mirror symmetry). Abelian.
- $E_7$, $E_8$: Aut = trivial. Abelian.

This is the well-known **$D_4$ triality** in Lie theory. The hub vertex connects to three equivalent leaves; any permutation of the three leaves is an automorphism. No other ADE diagram has this structure.

**Empirical steps: 0.** This is a pure theorem about finite graphs. It can be verified by hand or by exhaustive computation over all ADE types at any rank.

The non-commutativity measure $\text{NC}(D_4) = 1.2247$; for all other ADE types $\text{NC} = 0$.

**Physical interpretation**: In the orbit framework of Paper 11, non-abelian automorphisms produce non-commuting observables. $D_4$ is therefore the **unique** ADE type capable of generating genuine quantum uncertainty. This interpretation requires an additional identification (automorphisms $\leftrightarrow$ observables), but the mathematical fact is unconditional.

*Paper 11, Milestone 14. Experiment: exp_07 (non-commuting observables).*

### 2.4 Bell Violation Requires Nontrivial Aut(G)

**Statement**: On ADE orbit Hilbert spaces, the CHSH parameter $S$ exceeds the classical bound of 2 if and only if Aut(G) is nontrivial.

The argument:
1. If Aut(G) is trivial, every vertex is its own orbit. The orbit Hilbert space equals the full vertex space with no quotient structure. Product graphs factorize trivially — no entanglement is possible. **[a]**
2. The CHSH parameter satisfies $S \leq 2$. **[a]**
3. If Aut(G) is non-trivial (specifically $D_4$ with $S_3$), the orbit space is 2-dimensional. The product $D_4 \times D_4$ supports a maximally entangled Bell state, and the orbit Laplacian generates SU(2) rotations in the 2D space. **[a]**
4. Tsirelson's theorem guarantees $S = 2\sqrt{2}$ is achievable for a maximally entangled 2-qubit state with SU(2) rotations. **[a]**

Verified: $D_4 \times D_4$ achieves $S = 2\sqrt{2}$ to $4 \times 10^{-16}$. $E_7$ and $E_8$ (trivial automorphisms) give $S \approx 1.97\text{--}1.98$, below the classical bound — they cannot violate Bell inequalities.

**Empirical steps: 0.** The topology-dependence of Bell violation is a mathematical result about graph products and their automorphism groups.

**Caveat**: Achieving the full Tsirelson bound requires choosing a maximally entangled state and using the orbit Laplacian as the rotation generator. In 2D, any traceless Hermitian matrix generates SU(2), so the orbit Laplacian suffices automatically. For higher-dimensional orbit spaces, this would not generically hold.

*Paper 11, Milestone 14. Experiments: exp_p13 (Bell on D_4), exp_03 (CHSH sweep across ADE).*

### 2.5 The Cascade Clock Slope

**Statement**: If PAC conservation imposes $\varphi$-proportional timing on cascade levels (each level takes $\varphi$ times the duration of the previous), the number of completed cascade levels as a function of lookback time is:

$$N(t) = a + \frac{1}{\ln\varphi} \ln(t_{\text{lookback}})$$

The slope $1/\ln\varphi = 2.0781$ is determined by $\varphi$, which is determined by PAC. **[a]** given the cascade model.

The intercept $a$ is **not** determined by the framework — it is fitted to three cosmological data points (S8 tension, Hubble ratio, JWST galaxy counts). The value $a = 1.360$ yields $t_1 = \exp(-a/\text{slope}) = 520$ Myr. This coincides with the epoch of first star formation, but the coincidence is **post-hoc**: $t_1$ is derived from the fitted intercept, not independently anchored.

**What is derived [A]**: The functional form $N \propto \ln(t)$ and the slope $1/\ln\varphi$.
**What is fitted [B]**: The intercept, and therefore $t_1$.

*Paper 9, Milestone 9. Experiments: exp_01 (phi timing), exp_07 (S8 evolution).*

### 2.6 The PAC Decay Rate ln(phi)

**Statement**: The unique bounded positive solution to the PAC two-term recursion $\Psi(k) = \Psi(k+1) + \Psi(k+2)$ is $\Psi(k) = \varphi^{-k}$, giving a decay rate of $\ln\varphi = 0.48121\ldots$ per recursion level.

This is the foundation from which all other results descend. The PAC axiom is the starting point; everything else — $\varphi$, the Fibonacci numbers, the ADE bridge — follows from it.

**Empirical steps: 0.** Given the axiom, the decay rate is mathematically forced.

The ratio $A/(A + \xi)$ in Landauer erasure simulations converges toward $\ln\varphi$ at 0.15% with $N = 5 \times 10^6$ samples [Paper 1]. This provides computational validation but is not the derivation — the derivation is the algebraic solution of the recursion.

*Paper 1, Milestone 3. Experiments: exp_22 (depth uniqueness), exp_25 (convergence).*

---

## 3. The Precision Table — Tiered

The following table classifies every quantitative result in the PACSeries by derivation type. The companion document `derivation_classification.md` provides the full step-by-step chain for each entry.

### How to Read This Table

- **Type A (Structural)**: The derivation runs framework $\to$ result. At most 1 empirical step. Would produce the same result without knowing the target value.
- **Type B (Identified)**: A clean expression matches a known value. The path involves 2+ empirical identifications. The expression was likely recognized before being structurally grounded.
- **Type C (Pattern-matched)**: The formula was found by searching against known values. The papers acknowledge this explicitly.

The critical test: *would the derivation produce the same result if you didn't already know the target?*

### Type A — Structural

| Result | Value | Verified Against | Precision | Emp. Steps | Paper |
|--------|-------|-----------------|-----------|------------|-------|
| Lorentz group SO(3,1) | $\mathfrak{sl}(2,\mathbb{C}) \cong \mathfrak{so}(3,1)$ | 15 commutation relations | exact ($< 10^{-14}$) | 1 | 10 |
| Minkowski signature (1,3) | Killing form on 4D rep | 135 invariance tests | exact ($< 10^{-5}$) | 1 | 10 |
| $D_4$ non-abelian uniqueness | Aut($D_4$) = $S_3$ | all ADE types, all ranks | exact (theorem) | 0 | 11 |
| Bell requires Aut(G) | $S > 2$ iff Aut nontrivial | $D_4$: $2\sqrt{2}$; $E_7$, $E_8$: $< 2$ | $4 \times 10^{-16}$ | 0 | 11 |
| Clock slope | $1/\ln\varphi = 2.0781$ | cascade timing model | exact (given model) | 0 | 9 |
| $\ln\varphi$ | $0.48121\ldots$ | PAC recursion | exact (analytical) | 0 | 1 |

### Type B — Identified

| Result | Formula | Measured | Precision | Emp. Steps | Paper |
|--------|---------|----------|-----------|------------|-------|
| $\sin^2\theta_W$ | $3/13 = F_4/F_7$ | 0.23122 (at $M_Z$) | 0.19% | 2 | 4 |
| Koide ratio | $2/3 = F_3/(F_3+F_2)$ | 0.666661 | 0.5 ppm | 1 | 4 |
| $\Xi$ | $\gamma + \ln\varphi$ | 5 domains | 0.12% spread | 3 | 2 |
| $H_0$ ratio | $\varphi^{1/6} = 1.0835$ | SH0ES/CMB | 0.075% (0.05$\sigma$) | 2 | 9 |
| $S_8$ resolution | cascade dissipation | $0.768 \pm 0.02$ | $3.22\sigma \to 0.07\sigma$ | 2+ | 9 |
| $\alpha_\text{EM}$ | $\frac{F_3}{F_4 \varphi F_{10}}\left(1 - \frac{F_{10}}{4\pi F_7^2}\right)$ | CODATA | 5.7 ppm | 4 | 4 |

### Type C — Pattern-Matched

| Result | Formula | Measured | Precision | Paper | Note |
|--------|---------|----------|-----------|-------|------|
| Feigenbaum $\delta$ (rational) | $(50050 + 32\pi)/(10725 + 5\pi)$ | 4.6692016... | 8 digits | 3 | exhaustive search |
| Feigenbaum $r_\infty$ | $F_{10}$-based expression | 3.5699456... | 13 digits | 3 | exhaustive search |
| Feigenbaum $\lvert\alpha\rvert$ | $(2700+\pi)/1080$ | 2.5029078... | 6 digits | 3 | weakest result |
| Feigenbaum $\delta$ (self-closing) | $\varphi^{20/N}$, $N = \sqrt{39+1/x}$ | 4.6692016... | 13 digits | 3 | **upgraded to B$^-$** (exp_04-07) |
| $m_\mu/m_e$ | $F_4 F_6^2(1+1/F_7)$ | 206.768 | 5 ppm | 4 | paper says "systematic search" |
| $m_\tau/m_e$ | $F_4 F_7 F_{11} + F_5$ | 3477.2 | 0.035% | 4 | paper says "systematic search" |
| $m_p/m_e$ | $F_4 F_9 F_{12}/F_6$ | 1836.15 | 83 ppm | 4 | paper says "systematic search" |
| CC ($\log_{10}$) | $\varphi^{-2N}$ cascade | $-122.0$ | 0.09 orders | 9 | model-constructed |
| DM mass | $6.44$ keV (geometric mean) | 3.55 keV line | $\sim 0.1$ orders | 9 | model-constructed |
| Z' boson | $M_Z \cdot 13/3 = 395$ GeV | not yet tested | prediction | 9 | pattern extrapolation |

### Joint Significance

Individual Type C results are ambiguous — with Fibonacci numbers, $\pi$, $\varphi$, and elementary arithmetic, the expression space is large enough that isolated matches at 5-digit precision are unremarkable. The evidence lies in the **joint pattern**: the same small set of Fibonacci indices $\{F_3, F_4, F_5, F_6, F_7, F_{10}\}$ appears across particle physics, cosmology, nonlinear dynamics, and turbulence. Paper 4 reports a Monte Carlo test: 0 of 10,000 random Fibonacci combinations match all three mass ratios simultaneously ($p < 10^{-4}$). The combined system has $p < 10^{-5}$.

### The Feigenbaum Strengthening

The self-closing formula $\delta = \varphi^{20/N}$ deserves special treatment. As originally published [Paper 3], it was Type C — a formula constructed to match the known value. Subsequent investigation (experiments exp_04 through exp_07) upgraded its structural foundation:

- **$F_{10} = 55$ uniqueness**: Proved that $\text{ord}(2 \bmod F_n) = 2n$ holds only for $n = 10$ among all Fibonacci numbers, by CRT analysis of $F_{10} = 5 \times 11$ combined with a growth obstruction argument ($\varphi(F_n)$ grows exponentially while $2n$ grows linearly). [exp_04]
- **Base-2 uniqueness**: Among bases 2-100, only base 2 has a unique resonance at $n = 10$, linking period-doubling specifically. [exp_05]
- **Structural constants from CRT**: All three constants in the self-closing formula derive from the Chinese Remainder Theorem factorization of $F_{10} = 5 \times 11$: $39 = \varphi(5)\varphi(11) - 1$, $160 = \varphi(5)\varphi(55)$, $1371 = F_5^3 L_5 - \varphi(5)$. These are Euler totient identities, not fitted integers. [exp_07]
- **Mobius eigenvalue**: The Fibonacci Mobius matrix $M_{10}$ has eigenvalue $\varphi^{20}$ at its unstable fixed point $-1/\varphi$. This is a theorem of Mobius arithmetic. [exp_03]

The formula template (connecting $\varphi^{20}$ to $\delta$ via $N$, a square root, and self-reference) remains empirical. And the bridge — *why* the Mobius eigenvalue equals the renormalization group leading eigenvalue — is open. The question is now precisely stated: is there a conjugacy between the RG doubling operator $\mathcal{T}$ on function space and the Mobius transformation $M_{10}$ on $\mathbb{P}^1$?

---

## 4. What Fails

This section is the credibility of the series. Numerology never fails — failures get quietly dropped. A framework with honest, documented, investigated failures is behaving like physics.

### 4.1 The Algebraic-to-Continuum Boundary

The framework produces algebraic structure cleanly and struggles at the discrete-to-continuum bridge. Milestone scores reflect this:

- **Algebraic results**: M11 100%, M12 94%, M14 91% — commutation relations, group isomorphisms, orbit decompositions
- **Metric/continuum results**: M13 85% (drops to 25% for metric-specific tests) — rapidity composition, coherence limits, continuous deformation rates

The bridge between these domains is the ADE classification itself — a theorem of mathematics. The framework's limitation is not in the bridge but in what lies beyond it: continuous geometry from discrete graphs.

### 4.2 Specific Failures

| Failure | Score | What it reveals | Paper |
|---------|-------|----------------|-------|
| Positional QM (double-slit) | 1/4 | Orbit interference is algebraic, not spatial — no position-space wavefunction | 11 |
| PSD degeneracy | 0/4 | No invariant metric distinguishes same-orbit vertices — proven fundamental | 10 |
| Rapidity composition | 0/4 | Discrete complement-deformation breaks continuous $v_1 \oplus v_2 = (v_1+v_2)/(1+v_1 v_2/c^2)$ | 10 |
| Coherence limit non-universal | 0/4 | Different ADE families converge to different speed limits; $A$-family oscillates | 10 |
| Born rule (non-uniform) | 1/4 | $P = \lvert c_k\rvert^2$ for non-uniform states requires PAC correction term | 11 |
| DESI $w(z)$ tension | partial | $w_a = -0.15$ predicted vs $-0.75$ observed (DESI 2024) | 9 |
| RBF memory damping | 2/4 | Memory damping does not decay as predicted across all topologies | 7 |
| $A$-family oscillation | partial | $A_n$ complement spectra oscillate rather than converge; no monotonic limit | 10 |
| Random graphs more constrained | 2/4 | At large rank, random graphs are MORE constrained than ADE — unexpected | 10 |
| Cascade slope gap | partial | 8.9% residual gap between predicted and fitted slope (only 3 data points) | 9 |
| Width oscillation at integer $N$ | falsified | Pre-registered, DFT-specific, killed by z-detrending across 443K systems — the clock's discrete features do not imprint on absorption lines | 12 |

### 4.3 Methodological Concerns

**Look-elsewhere effect**: Severe for Type C results. With $\{\varphi^n, F_n, L_n, \pi, \gamma, \sqrt{k}\}$ and elementary arithmetic, the expression space for any 6-digit target is enormous. Moderate for Type B (bounded by the number of reasonable identification schemes). Not applicable for Type A (theorems have no search space).

**No peer review**: The PACSeries has not undergone formal peer review. The Zenodo publication (DOI: 10.5281/zenodo.15783623) is a preprint deposit, not a journal acceptance. The derivation classification in this paper is an attempt to do the skeptic's work preemptively.

**Interdisciplinary scope**: The series spans 6 domains of physics. No single reviewer has expertise across all of them. The practical consequence: each domain's results will be evaluated in isolation, and the cross-domain pattern — which is the actual evidence — may not be seen.

**"One free parameter" framing**: Paper 9 claims the cascade clock has one free parameter ($t_1 = 520$ Myr). This is misleading. The clock has two parameters (slope and intercept). The slope is constrained by $\varphi$. The intercept $a = 1.360$ is fitted to three data points. $t_1 = \exp(-a/\text{slope})$ is derived from the fit, not independently anchored. The coincidence with first-star formation is noted post-hoc.

**"Derive" vs. "identify"**: Several papers use the word "derive" where "identify" would be more accurate. Paper 4 derives that SU(2) and SU(3) pass the Fibonacci filter — but identifying $\sin^2\theta_W$ with the ratio $F_4/F_7$ is an identification, not a derivation. Paper 11 derives the orbit Hilbert space from graph automorphisms — but identifying orbits with measurement outcomes is a modeling choice. The derivation classification (Section 3) addresses this distinction explicitly.

---

## 5. Falsifiable Predictions

### Near-Term (2025-2030)

| Prediction | Value | Instrument | Kill condition | Paper |
|------------|-------|------------|----------------|-------|
| Z' boson | $395 \pm 20$ GeV, $\Gamma = 64$ MeV | LHC Run 3+ | Excluded at 95% CL | 9 |
| $D_4$ special in quantum simulation | Non-abelian Aut $\to$ Bell violation | Trapped ions, superconducting qubits | $D_4$ topology not special | 11 |
| No Fibonacci gauge groups beyond SU(3) | $N^2 - 1 \neq F_k$ for $N > 3$ | Mathematical (checkable now) | Counterexample found | 4, 10 |
| S8(z) curve | Specific redshift dependence | Euclid, DESI | Shape rejected at $3\sigma$ | 9 |

### Medium-Term (2030-2035)

| Prediction | Value | Instrument | Kill condition | Paper |
|------------|-------|------------|----------------|-------|
| DM particle | $6.44$ keV (X-ray at $3.2$ keV) | Athena, eROSITA follow-up | No line at $3.0\text{--}3.5$ keV | 9 |
| Bell violation topology-dependent | $S$ depends on graph Aut structure | Quantum simulation at scale | Universal $S$ independent of topology | 11 |
| Anti-Zeno crossover | Rate encodes spectral gap | Controlled dephasing experiments | No topology dependence | 11 |

### Currently Testable

| Prediction | Status | Paper |
|------------|--------|-------|
| Fibonacci gauge filter: only SU(2), SU(3) pass | Confirmed to rank 100 | 4, 10 |
| $D_4$ triality: unique non-abelian ADE Aut | Theorem (proven) | 11 |
| $\sin^2\theta_W = 3/13$ at $Q \approx M_W$ | 0.19% from $M_Z$ value; RG running confirms | 4 |

**Kill condition for the framework**: If any two near-term predictions fail, the framework must be substantially revised. If the Z' is excluded at 395 GeV AND the S8(z) shape is rejected, the Fibonacci-cascade model is falsified.

---

## 6. The Map

### Reading Paths

**10 minutes**: Read this paper. The abstract, Section 2 (structural results), and the tiered precision table in Section 3 provide the complete picture.

**Structural results** (Papers 10-11): The Lorentz derivation and D_4 uniqueness theorem. These are the strongest results and the most checkable. A mathematician can verify them independently.

**Precision patterns** (Papers 3-4): The Feigenbaum formulas and Standard Model parameters. These are the most striking numerically but involve identification steps. Read with the derivation classification in mind.

**Cosmology** (Paper 9): The cascade clock and its predictions for S8, H0, JWST, and dark matter. The slope is derived; the intercept is fitted. The leave-one-out tests are the honest checks.

**Foundations** (Papers 1-2): Where $\varphi$ and $\Xi$ come from. Paper 1 is the most self-contained; Paper 2 is the most empirically grounded (5 domains, but with selection concerns).

**Falsification** (any paper): Every paper states its kill conditions. Run the code. Change the parameters. The experiments are numbered and reproducible.

### Paper Index

| # | Title | Domain | Headline Result | Type |
|---|-------|--------|-----------------|------|
| 1 | Structure Cost of Erasure | Thermodynamics | $A/(A+\xi) \to \ln\varphi$ at 0.15% | A+B |
| 2 | Balance Constant Decomposition | Mathematical physics | $\Xi = \gamma + \ln\varphi$, 5 domains | B |
| 3 | Feigenbaum from Fibonacci | Nonlinear dynamics | $\delta$ to 13 digits (self-closing formula) | C $\to$ B$^-$ |
| 4 | Standard Model Parameters | Particle physics | $\alpha_\text{EM}$ to 5.7 ppm, $\sin^2\theta_W = 3/13$ | B+C |
| 5 | Classical Physics | Field theory | Maxwell from PAC, $D=3$ from MED | B |
| 6 | Computational Validation | ML/AI | PAC conservation in neural networks | B |
| 7 | Symmetry Primitive | Foundations | Pre-axiomatic origin, force hierarchy | B |
| 8 | Quantum Gravity | Quantum gravity | Planck scale from PAC/SEC crossover | B+C |
| 9 | Cosmological Predictions | Cosmology | S8 resolved, cascade clock | A+B+C |
| 10 | Connection, Identity, Spacetime | Relativity | Lorentz from ADE, $ds^2$ from Killing form | A |
| 11 | QM from Graph Structure | Quantum foundations | $D_4$ theorem, Born rule, Bell from Aut | A+B |
| 12 | First Observational Contact | Observational cosmology | Pre-registered oscillation falsified; PAC/SEC two-channel partition | A+B+C |

### Reproducibility

Every paper in the PACSeries includes a Code/ directory with numbered experiment scripts. Results are stored as timestamped JSON files. All dependencies are standard scientific Python (NumPy, SciPy, Matplotlib). The full experiment codebase is at [github.com/dawnfield-institute/dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory).

---

## 7. Conclusion

The PACSeries contains three kinds of results, and the distinction matters.

**Six structural results (Type A)** follow from the PAC and SEC axioms through chains of theorems. The Lorentz group, the Minkowski metric, the D_4 uniqueness theorem, the topology-dependence of Bell violation, the cascade clock slope, and the PAC decay rate $\ln\varphi$ — these are mathematical consequences of the framework. A skeptic must reject the axioms or find an error in the proofs.

**Six identified patterns (Type B)** match clean Fibonacci expressions to known physical constants: $\sin^2\theta_W = 3/13$, the Koide ratio as $2/3$, the fine structure constant to 5.7 ppm, the Hubble tension as $\varphi^{1/6}$, the S8 resolution via cascade dissipation, and $\Xi = \gamma + \ln\varphi$. Each involves 1-4 identification steps where known physics guided the construction. They are precise, structurally motivated, and not fully derived.

**Ten pattern-matched results (Type C)** were found by searching against known values. The Feigenbaum formulas, mass ratios, mixing angles, and cosmological parameters are in this category. The papers say so explicitly. Their significance lies not in individual precision but in the joint pattern: the same small set of Fibonacci indices appearing across independent physical domains.

The framework's single biggest open problem is the algebraic-to-continuum bridge. It produces discrete algebraic structure (group isomorphisms, orbit decompositions, commutation relations) at 91-100% accuracy. It struggles with continuous geometry (rapidity composition, position-space wavefunctions, coherence limits) at 25-85%. The bridge is the ADE classification — a theorem — and the gap is precisely characterized.

The second open problem is specific to the strongest pillar: the Feigenbaum self-closing formula has every structural constant explained by CRT, every Fibonacci index justified by uniqueness, and a well-posed bridge question (Mobius eigenvalue $\leftrightarrow$ RG eigenvalue). This is no longer "a formula that matches a number." It is an open problem in the intersection of number theory and functional analysis.

The theorems are checkable. The predictions are specific. The failures are documented. The code is open.

---

## References

Each paper in the PACSeries contains its own reference list. Cross-references use "Paper N, Section M" notation. The derivation classification for all results is available as `derivation_classification.md` in the same directory as this paper.

**PACSeries v0.2** (Papers 1-6): Zenodo DOI [10.5281/zenodo.18743674](https://zenodo.org/records/18743674)
**PACSeries v0.3** (Papers 7-12): In preparation. (Series concept DOI: [10.5281/zenodo.15783623](https://zenodo.org/records/15783623))
**Repository**: [github.com/dawnfield-institute/dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory)

---

*PACSeries overview. All code, data, and figures are publicly available under AGPL-3.0 (code) and CC-BY-4.0 (papers).*
