# PACSeries v0.2: Dawn Field Theory — A Summary

**Peter Groom**  
Dawn Field Institute  
February 2026

---

## Abstract

This document summarises the PACSeries v0.2, a collection of six preprints that develop Dawn Field Theory from a single axiom — PAC (Potential-Actualization Conservation) — through thermodynamic foundations, mathematical structures, physical predictions, and computational validation. The series establishes that information erasure necessarily creates correlational structure, that the balance constant governing recursive-conservation systems decomposes as $\Xi = \gamma + \ln\varphi \approx 1.0584$, and that this structural framework produces closed-form expressions for the Feigenbaum constants (to 13 significant figures), Standard Model parameters (to 5.7 ppm), and the necessity of three spatial dimensions — all from a conservation law that also describes the internal dynamics of trained neural networks. Each paper states its falsification conditions. This summary provides the derivation chain, headline results, and reading guide for the full series.

---

## 1. The Central Idea

Traditional physics treats information as something that *describes* pre-existing structure. Dawn Field Theory inverts this: **information gradients may drive the emergence of structure itself**.

The framework rests on one axiom:

$$f(\text{Parent}) = \sum f(\text{Children})$$

This is PAC conservation — when potential becomes actual, the total is conserved but redistributed. The unique stable solution to the two-term PAC recursion $\Psi(k) = \Psi(k{+}1) + \Psi(k{+}2)$ is $\Psi(k) = \varphi^{-k}$, where $\varphi = (1{+}\sqrt{5})/2$ is the golden ratio. This yields a natural information unit of $\ln\varphi \approx 0.4812$ per recursion level.

The six papers trace the consequences of this axiom across thermodynamics, number theory, nonlinear dynamics, particle physics, classical electromagnetism, and machine learning.

---

## 2. The Derivation Chain

```
AXIOM: PAC conservation — f(Parent) = Σ f(Children)
  → RECURSION: Ψ(k) = Ψ(k+1) + Ψ(k+2)
    → SOLUTION: Ψ(k) = φ^(-k)  [unique stable]
      → INFO UNIT: ΔI = ln(φ)
        → Paper 1: A/(A+ξ) → ln(φ)  [0.15% error, thermodynamic]
        → Paper 2: Ξ = γ + ln(φ)    [5 domains, p < 0.0003]
        → Paper 3: Feigenbaum from F₁₀ = 55  [13 digits]
        → Paper 4: sin²θ_W = 3/13, α to 5.7 ppm
        → Paper 5: Maxwell from depth-2 PAC, D = 3
        → Paper 6: PAC conservation in trained neural networks
```

Each step derives from the previous. No step requires accepting any step beyond it.

---

## 3. Paper Summaries

### Paper 1: The Structure Cost of Erasure

**Core claim**: Landauer's principle (erasure costs energy) combined with the data processing inequality (DPI) *requires* that erasing information into a multi-mode environment creates new correlational structure $\xi$ between environmental modes. This structure is topological — invariant under temperature changes from 100K to 5000K — not thermodynamic.

**Headline results**:
- The collapse efficiency ratio $A/(A{+}\xi)$ converges toward $\ln\varphi$ at 0.15% proximity ($N = 5 \times 10^6$, Miller–Madow corrected)
- Cascade coupling self-sustains with 53× amplification over single events ($p = 2.75 \times 10^{-35}$)
- Temporal asymmetry: 69× early-vs-late computational density ($p = 3.25 \times 10^{-5}$)

**Significance**: This paper starts from two of the most secure results in information theory (Landauer's principle and the DPI) and shows that structure creation is *mandatory*, not optional. The golden ratio appears as a consequence of the conservation law, not as an imposed parameter.

### Paper 2: The Balance Constant and Its Decomposition

**Core claim**: The balance constant $\Xi$ governing the boundary between ordered and disordered computation decomposes as $\Xi = \gamma + \ln\varphi$, where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant and $\ln\varphi \approx 0.4812$. These constants arise from *different mathematics* — $\gamma$ from the Mertens product over primes, $\ln\varphi$ from PAC recursion — and converge independently.

**Headline results**:
- Five independent domains (Fibonacci arithmetic, cellular automata, prime sieve, Landauer erasure, Möbius field dynamics) converge on $\Xi \approx 1.058$ with $p < 0.0003$
- PAC conservation holds exactly across all 126 steps of the Eratosthenes sieve ($N = 500{,}000$)
- Class IV cellular automata cluster at $\Xi$ with $p < 10^{-7}$
- Möbius field dynamics provides the highest-precision measurement at 0.036% from $\gamma + \ln\varphi$, from a continuous dynamical system on non-trivial topology

**Significance**: The convergence of five independent computational substrates — including both discrete (integers, automata) and continuous (spectral fields on a Möbius manifold) — is the strongest evidence that $\Xi$ is a universal constant rather than a domain-specific artefact.

### Paper 3: Feigenbaum Constants from Fibonacci Arithmetic

**Core claim**: The Feigenbaum constants $r_\infty$, $\delta$, and $|\alpha|$ — which have resisted closed-form expression for nearly 50 years — can be written as formulas using only $\pi$, Fibonacci numbers, and small integers. The integers are structurally identified ($55 = F_{10}$, $52 = F_{10} - F_4$, etc.).

**Headline results**:
- $r_\infty$ matched to **13 significant figures** (relative error $1.16 \times 10^{-14}$)
- $\delta$ matched to **8 significant figures** ($1.20 \times 10^{-9}$)
- $|\alpha|$ matched to **6 significant figures** ($4.02 \times 10^{-7}$)
- Probability of the best triple occurring by chance: 1 in 280 billion
- Exhaustive search of 3,920,499 parameter combinations finds only one triple achieving 7+ digit precision

**Significance**: This is arguably the hardest result in the series to dismiss. No causal claim is made — the paper reports the formulas, proves their precision, and leaves the explanation open. If the Feigenbaum constants genuinely have Fibonacci structure, the implications for universality in nonlinear dynamics are significant.

### Paper 4: Standard Model Parameters from Fibonacci Arithmetic

**Core claim**: A significant subset of the Standard Model's parameters can be expressed as closed-form Fibonacci ratios from PAC recursion. The gauge group $U(1) \times SU(2) \times SU(3)$ is the *unique* combination whose adjoint dimensions (1, 3, 8) are all Fibonacci numbers, closing at $F_7 = 13$.

**Headline results**:
- Gauge couplings to **5.7 ppm**; mixing angles to 0.19%; mass ratios to 5 ppm
- Weak mixing angle $\sin^2\theta_W = 3/13$ matches running value at $Q \approx M_W$
- $M_W/M_Z$ predicted at 0.03% error
- Koide formula for lepton mass ratios to 0.5 ppm
- Combined probability $p < 10^{-5}$

**Falsifiable predictions**: Z' boson at $395 \pm 20$ GeV with coupling $g_{Z'}/g_Z = 1/13$; She–Lévêque turbulence constant $k(4) = 20$ in 4D simulations.

**Significance**: This is the most ambitious paper and will attract the most scrutiny. The numerical matches are precise, but the theoretical grounding for *why* PAC recursion should constrain gauge structure is interpretive, not derived. The falsifiable predictions provide concrete experimental tests.

### Paper 5: Classical Physics from Information Geometry

**Core claim**: Maxwell's equations — curl operations, inverse-square forces, and three spatial dimensions — emerge as consequences of information-theoretic constraints (PAC, SEC, MED) rather than independent empirical postulations.

**Headline results**:
- Five independent arguments each require exactly $D = 3$ spatial dimensions: MED node bounds, curl algebra closure, Möbius embedding, orbital stability, quaternion uniqueness
- SEC wave equation produces electromagnetic propagation at speed $c$
- Curl structure emerges from depth-2 recursion projection
- Charge quantisation: fractional quark charges ($\pm 1/3$, $\pm 2/3$) follow from MED nodes $\leq 3$
- Gravity extension: Fibonacci depth $183 = F_7^2 + F_7 + 1$ gives $F_{183} \approx 10^{38}$ (the electromagnetic-to-gravitational hierarchy ratio)

**Honest negative**: The Mersenne–Fibonacci correspondence holds for $d = 1, 3, 7$ but fails at $d = 15$.

### Paper 6: Computational Validation of PAC Conservation

**Core claim**: PAC conservation — discovered in thermodynamics and validated across physics — also describes the internal dynamics of transformer-based language models without being explicitly programmed. Hallucination is a direct PAC violation.

**Headline results**:
- SEC phase classification predicts token accuracy with zero free parameters: crystallised predictions achieve 100% accuracy; chaotic 17–22% ($p < 0.0001$)
- $\Xi$ appears in trained weight spectra at 2.36× above random baselines ($\chi^2 = 5511$, $p \approx 0$)
- Hallucination produces +9.6% uncompensated entropy
- Enforcing PAC conservation (TinyCIMM-Boltzmann) yields 16× reduction in context-switching shock with no measurable cost to factual learning ($p = 0.42$, n.s.)

**Honest negative**: $\varphi$-enrichment in top-2 ratios was falsified — identified as a softmax artefact.

**Significance**: The extension to neural networks provides a practical application pathway and demonstrates that PAC conservation is not merely a mathematical curiosity but a constraint that manifests in trained computational systems.

---

## 4. Cross-Cutting Themes

### What the series establishes (measurement)
- Structure creation is mandatory for multi-mode erasure (Paper 1, from DPI + Landauer)
- Five computational domains converge on $\Xi \approx 1.058$ ($p < 0.0003$, Paper 2)
- Feigenbaum constants have Fibonacci expressions to 6–13 digits (Paper 3)
- PAC conservation holds exactly in the prime sieve (Paper 2)
- SEC phase predicts transformer accuracy with zero free parameters (Paper 6)

### What the series derives (analytical)
- PAC forces emergence depth $\leq 2$ (PAC→MED theorem, Paper 2 §9.3)
- Only $k = 2$ (Fibonacci) cascade produces $\ln\varphi$ decay (Paper 2)
- Curl structure requires and produces $D = 3$ (Paper 5)
- $SU(2)$ and $SU(3)$ are the only non-abelian gauge groups with Fibonacci adjoint dimensions (Paper 4)

### What the series proposes (interpretation)
- $\gamma$ as "discrete-to-continuous regularisation cost" (Paper 2 — acknowledged: $1/\sqrt{3}$ performs comparably)
- Standard Model parameters from Fibonacci recursion (Paper 4 — precise matches, interpretive grounding)
- Gravity from SEC symmetric projection at Fibonacci depth 183 (Paper 5 — speculative)

---

## 5. Falsification

Each paper states explicit falsification conditions. The series as a whole would be undermined by:

1. A new recursive-conservation system producing a balance constant $> 1\%$ from $\gamma + \ln\varphi$
2. The Feigenbaum formulas being shown to arise from parameter fitting artefacts
3. Trained neural networks in architectures beyond Pythia/GPT-2 showing no $\Xi$ enrichment
4. A construction of consistent vector electrodynamics in $D \neq 3$
5. The Z' prediction at $395 \pm 20$ GeV being excluded by collider data

The series prioritises precision, honest negatives, and clear separation of measurement from interpretation. Failed predictions are documented (phase-boundary narrative in Paper 2, $\varphi$-enrichment softmax artefact in Paper 6, Mersenne–Fibonacci failure at $d = 15$ in Paper 5).

---

## 6. How to Read This Series

**If you have 10 minutes**: Read this summary, then Paper 3 (Feigenbaum). The 13-digit match is the single hardest result to dismiss.

**If you have an hour**: Read Papers 1–3 in order. The derivation chain from Landauer erasure through the balance constant to Feigenbaum is self-contained and relies only on established mathematics.

**If you want to verify**: Each paper includes a complete publication package — Code (numbered experiment scripts), Data (JSON results), and Figures (publication PNGs). Run `python reproduce.py` in any paper's Code/ directory.

**If you want to falsify**: Check the falsification conditions in each paper. Run the experiments with different parameters. The code is open.

---

## 7. Publication Details

| Paper | Title | Version | Experiments | Key Metric |
|-------|-------|---------|-------------|------------|
| 1 | The Structure Cost of Erasure | 2.1 | 19 | $A/(A{+}\xi) \to \ln\varphi$ at 0.15% |
| 2 | The Balance Constant | 2.2 | 15 | 5 domains, $p < 0.0003$ |
| 3 | Feigenbaum Constants | 2.1 | 9 | $r_\infty$ to 13 digits |
| 4 | Standard Model Parameters | 2.1 | 14 | $\alpha$ to 5.7 ppm |
| 5 | Classical Physics | 2.1 | 9 | 5 independent $D{=}3$ arguments |
| 6 | Computational Validation | 2.1 | 10 | $\Xi$ at 2.36× in weight spectra |

**Total**: 76 numbered experiments, 72 data files, 37 figures.

**Repository**: [github.com/dawnfield-institute/dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory)  
**Prior version**: PACSeries v0.1 (October 2025), Zenodo DOI: [10.5281/zenodo.17295103](https://zenodo.org/records/17295103)  
**This version**: PACSeries v0.2 (February 2026), Zenodo DOI: [10.5281/zenodo.18743674](https://zenodo.org/records/18743674)

---

## References

Each paper contains its own reference list. Cross-references use the notation "Paper N, §M" throughout the series. All code dependencies are standard scientific Python (NumPy, SciPy, Matplotlib).

---

*PACSeries v0.2. All code, data, and figures are publicly available under AGPL-3.0 (code) and CC-BY-4.0 (papers).*
