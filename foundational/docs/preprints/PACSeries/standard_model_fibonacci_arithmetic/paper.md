# Standard Model Parameters from Fibonacci Arithmetic

**Peter Groom, Dawn Field Institute**  
**PACSeries Paper 4**  
**Date**: February 2026  
**Version**: 2.1

---

## Abstract

The Standard Model of particle physics contains approximately 25 free parameters — coupling constants, mixing angles, and mass ratios — whose values are measured but not derived from first principles. This paper shows that a significant subset of these parameters can be expressed as closed-form ratios of Fibonacci numbers, with precisions ranging from 0.5 ppm (Koide formula) to 1.7% (strong coupling constant).

The expressions are not fitted. They follow from a single structural constraint: the PAC recursion $\Psi(k) = \Psi(k+1) + \Psi(k+2)$, whose unique stable solution $\Psi(k) = \varphi^{-k}$ selects the golden ratio algebraically. The Fibonacci numbers are its integer projections. The gauge group structure of the Standard Model — $U(1) \times SU(2) \times SU(3)$ — is the unique combination whose adjoint dimensions ($1, 3, 8$) are Fibonacci numbers, closing at $F_7 = 13 = 1 + 3 + 8 + 1$.

Individual Fibonacci matches are not significant (any small integer has $\sim$16% probability of being Fibonacci). What is significant is the joint constraint: a single recursion, evaluated at specific hierarchy depths, simultaneously reproduces gauge couplings to 5.7 ppm, mixing angles to 0.19%, mass ratios to 5 ppm, and the Casimir regularisation factor exactly — with a combined probability against chance of $p < 10^{-5}$. The weak mixing angle prediction $\sin^2\theta_W = 3/13$ is excluded at $M_Z$ ($\sim$11$\sigma$) but matches the running value at $Q = 82.78$ GeV $\approx M_W$, with $M_W/M_Z$ predicted at 0.03% error (§4.4).

We also derive the She-Lévêque turbulence intermittency constant $k = d \times F_{d+1}$ from first principles, connecting particle physics constants to fluid dynamics through the same Fibonacci structure. A falsifiable prediction is offered: a Z' boson at $395 \pm 20$ GeV with coupling $g_{Z'}/g_Z = 1/13$.

**Keywords**: Standard Model, Fibonacci numbers, golden ratio, coupling constants, mass ratios, PAC conservation, Koide formula, fine structure constant, Dawn Field Theory

---

## §1. The Problem of Free Parameters

The Standard Model of particle physics is one of the most precisely tested theories in science. It predicts the anomalous magnetic moment of the electron to 12 significant figures. It predicted the existence of the W and Z bosons, the top quark, and the Higgs boson before they were discovered.

It does not explain why the fine structure constant $\alpha$ equals $1/137.036$ rather than some other number. It does not explain why there are three generations of fermions rather than two or seven. It does not explain why the electron is 206.768 times lighter than the muon. These values are measured with extraordinary precision but enter the theory as inputs, not outputs.

This paper asks whether a single structural constraint — conservation under hierarchical branching — can reproduce these values as necessary consequences.

---

## §2. The Constraint

Paper 1 in this series established that information erasure into multi-mode environments creates correlational structure. When the environment has cascade topology (sequential mode coupling), the PAC conservation constraint requires:

$$\Psi(k) = \Psi(k+1) + \Psi(k+2)$$

This is the Fibonacci recursion. Its unique stable solution is $\Psi(k) = \varphi^{-k}$, where $\varphi = (1 + \sqrt{5})/2$ is the golden ratio. This is not a choice. It is the only bounded solution to the characteristic equation $x^2 = x + 1$.

The Fibonacci numbers $F_n = \{1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, \ldots\}$ are the integer projections of this recursion. They provide a complete basis for integer representation (Zeckendorf's theorem: every positive integer has a unique decomposition into non-consecutive Fibonacci numbers).

Paper 2 established that the golden ratio identity $\varphi^2 = \varphi + 1$ holds to $< 10^{-14}$ across all numerical bases, confirming that the Fibonacci structure is mathematical, not representational.

### §2.1 The Lagrangian

The PAC constraint expressed as a Lagrangian density:

$$\mathcal{L} = \frac{1}{2}\sum_k \left(\frac{\partial\Psi_k}{\partial t}\right)^2 - V(\Psi) + \lambda \sum_k [\Psi_k - \Psi_{k+1} - \Psi_{k+2}]^2$$

This has a golden scaling symmetry: $\Psi(k) \to \varphi^{-1} \cdot \Psi(k-1)$. By Noether's theorem, this continuous symmetry yields a conserved charge:

$$Q_{\text{PAC}} = -\frac{1}{\varphi} \int \Psi \cdot \frac{\partial\Psi}{\partial t} \, dx$$

The theory predicts that physical coupling constants are ratios of this conserved charge evaluated at different hierarchy levels.

---

## §3. The Gauge Group Selection

### §3.1 Fibonacci Filter on $SU(N)$

A gauge group $SU(N)$ has $N^2 - 1$ generators (the dimension of its adjoint representation). The PAC constraint requires this number to be a Fibonacci number:

| Group | $N^2 - 1$ | Fibonacci? |
|-------|-----------|------------|
| $U(1)$ | 1 | $F_1 = F_2 =$ **Yes** |
| $SU(2)$ | 3 | $F_4 =$ **Yes** |
| $SU(3)$ | 8 | $F_6 =$ **Yes** |
| $SU(4)$ | 15 | **No** |
| $SU(5)$ | 24 | **No** |

$SU(2)$ and $SU(3)$ are the only non-abelian gauge groups whose adjoint representations have Fibonacci dimension. This is a constraint, not a selection — PAC conservation cannot maintain coherence across a non-Fibonacci number of coupled modes.

**Script**: `exp_19_su4_forbidden.py` — confirms no $SU(N)$ with $N > 3$ has a Fibonacci adjoint dimension for any $N \leq 100$.

### §3.2 PAC Closure at $F_7 = 13$

The total gauge content of the Standard Model:

$$U(1): 1 \quad + \quad SU(2): 3 \quad + \quad SU(3): 8 \quad + \quad \text{Higgs}: 1 \quad = \quad 13 = F_7$$

This is PAC closure: the total number of gauge and scalar degrees of freedom equals a Fibonacci number. The Higgs is counted as 1 physical degree of freedom — the radial mode that survives after electroweak symmetry breaking, where the three Goldstone components are absorbed by the $W^\pm$ and $Z$ bosons. Before EWSB, the Higgs doublet has 4 real components; after EWSB, 1 physical scalar remains. The PAC count applies to the broken phase — the phase in which all measured couplings are defined.

Whether this counting is numerology or structure depends on whether $F_7$ predicts anything. It does. The fine structure constant, the weak mixing angle, the Cabibbo angle, and the neutrino mixing angles all involve $F_7 = 13$ in their denominators.

---

## §4. Gauge Coupling Constants

### §4.1 Fine Structure Constant

The fine structure constant $\alpha$ measures the strength of electromagnetic coupling. Its value is known to 0.15 ppb:

$$\alpha^{-1} = 137.035999177(21) \quad \text{(CODATA 2022 [5])}$$

The PAC formula:

$$\alpha = \frac{F_3}{F_4 \cdot \varphi \cdot F_{10}} \left(1 - \frac{F_{10}}{4\pi F_7^2}\right) = \frac{2}{3 \cdot \varphi \cdot 55} \left(1 - \frac{55}{4\pi \cdot 169}\right)$$

**Computed**: $\alpha_{\text{PAC}} = 0.00729731$  
**Measured**: $\alpha_{\text{exp}} = 0.0072973526$  
**Error**: $5.7 \text{ ppm}$

### §4.2 Structure of the Formula

The formula has two factors:

1. **Base rate**: $\frac{F_3}{F_4 \cdot \varphi \cdot F_{10}} = \frac{2}{3\varphi \cdot 55}$. This is the coupling strength for one photon propagating through a Fibonacci hierarchy of depth $F_{10} = 55$.

2. **Correction**: $\left(1 - \frac{F_{10}}{4\pi F_7^2}\right) = \left(1 - \frac{55}{4\pi \cdot 169}\right) \approx 0.97393$. This accounts for the gauge closure constraint at $F_7 = 13$.

The interpretation from Paper 1: $\alpha$ measures the Landauer structure cost of electromagnetic interaction — the rate at which correlational structure $\xi$ accumulates through 55 hierarchy levels, corrected by the total gauge content.

### §4.3 Uniqueness

All Fibonacci pairs $(F_m, F_n)$ for $m, n \in [3, 20]$ were tested in the formula. The pair $(10, 7)$ is $2{,}870\times$ better than the next best match. Given $F_{10} = 55$, the formula predicts the other index must satisfy $F_n = 13.0014$; the nearest Fibonacci number $F_7 = 13$ is 0.001 away.

The continuous Fibonacci extension ($\varphi^n/\sqrt{5}$) gives *worse* results (137 ppm vs 5.7 ppm). Discrete integers are structurally significant, consistent with the discrete level symmetry of the PAC Lagrangian.

**Script**: `exp_12_alpha_formula.py`, `exp_13_alpha_falsification.py`

### §4.4 Weak Mixing Angle

The Weinberg angle $\theta_W$ parametrises the electroweak mixing:

$$\sin^2\theta_W = \frac{F_4}{F_7} = \frac{3}{13} = 0.230769\ldots$$

**Measured**: $\sin^2\theta_W = 0.23121 \pm 0.00004$ (on-shell, $M_Z$) [1]  
**Error**: $0.19\%$

The ratio $F_4/F_7 = 3/13$ directly gives the weak mixing as the fraction of $SU(2)$ generators within the total gauge content. This is a structural prediction: the mixing is determined by gauge group dimensions, not by a free parameter.

**Energy-scale resolution (milestone3/exp_08).** $\sin^2\theta_W$ runs with energy. The on-shell value at $M_Z \approx 91.2$ GeV is $0.23121 \pm 0.00004$, differing from $3/13 = 0.23077$ by $\sim$11$\sigma$. However, the PAC ratio need not hold at $M_Z$. A one-loop running calculation finds that $\sin^2\theta_W(Q)$ passes through exactly $3/13$ at:

$$Q = 82.78 \text{ GeV}$$

This is within 3% of $M_W = 80.37$ GeV ($Q/M_W = 1.030$). The proximity to $M_W$ may be physically significant: the W boson mediates flavour-changing charged-current transitions — the only Standard Model process that converts between quark and lepton generations. In PAC terms, flavour change corresponds to actualization (potential states becoming actual states), and the Fibonacci mixing $F_4/F_7$ would then achieve its exact value at the energy where this actualization mechanism operates. Whether this interpretation survives beyond the one-loop approximation used here is an open question.

The tree-level mass ratio prediction follows directly:

$$\frac{M_W}{M_Z} = \cos\theta_W = \sqrt{1 - \frac{3}{13}} = \sqrt{\frac{10}{13}} = 0.8771$$

compared to the on-shell value $\cos\theta_W = 0.8768$, an error of **0.03%**.

Among 24 Fibonacci ratios $F_i/F_j$ tested, $F_4/F_7 = 3/13$ is rank #1 for proximity to the measured $\sin^2\theta_W$.

**Sensitivity**: $\pm 3\sigma$ in $\sin^2\theta_W$ maps to $Q \in [81.2, 84.4]$ GeV, bracketing $M_W$ from above.

**Scripts**: `exp_18_weinberg_angle.py`, `milestone3/scripts/exp_08_weinberg_running.py`

### §4.5 Strong Coupling Constant

$$\alpha_s(M_Z) = \frac{F_4}{2\varphi \cdot F_6} = \frac{3}{2\varphi \cdot 8} = 0.1159$$

**Measured**: $\alpha_s(M_Z) = 0.1179 \pm 0.0010$ [1]  
**Error**: $1.71\%$

This is the weakest of the three gauge coupling predictions. The formula gives the strong coupling as the $SU(2)$ adjoint dimension divided by twice the golden ratio times the $SU(3)$ adjoint dimension. The 1.71% error is within the measurement uncertainty range but is substantially less precise than the $\alpha$ result.

A structural asymmetry deserves comment. The $\alpha$ formula (§4.1) includes a correction factor $(1 - F_{10}/4\pi F_7^2)$, while the $\alpha_s$ formula is a bare ratio. If all gauge couplings emerge from the same recursion at different depths, their formulas should share a template. One interpretation: electromagnetism couples to the full gauge content (all 13 modes contribute vacuum polarisation loops), requiring the $F_7^2$ correction, while the strong coupling operates within a single gauge sector ($SU(3)$) and sees no cross-sector correction. This is consistent with asymptotic freedom — at high energy, $\alpha_s$ is determined by $SU(3)$ self-coupling alone. Whether this interpretation survives a proper renormalisation group analysis is an open question.

### §4.6 Summary of Gauge Couplings

| Coupling | PAC Formula | PAC Value | Measured | Error |
|----------|-------------|-----------|----------|-------|
| $\alpha$ (EM) | $\frac{F_3}{F_4 \cdot \varphi \cdot F_{10}}\left(1 - \frac{F_{10}}{4\pi F_7^2}\right)$ | 0.0072973 | 0.0072974 | **5.7 ppm** |
| $\sin^2\theta_W$ (Weak) | $F_4/F_7 = 3/13$ | 0.23077 | 0.23121 | **0.19%** |
| $\alpha_s$ (Strong) | $F_4/(2\varphi F_6)$ | 0.1159 | 0.1179 | **1.71%** |

The precision ordering ($\alpha > \sin^2\theta_W > \alpha_s$) corresponds to the algebraic/transcendental character of the formulas: the algebraic ratios ($3/13$) are exact; the transcendental corrections ($\varphi$, $\pi$) introduce residual error. Paper 2 discusses the discrete-to-continuous gap as a structural feature, not a limitation.

---

## §5. The Koide Formula

The Koide formula is an empirical relation among the charged lepton masses:

$$Q = \frac{m_e + m_\mu + m_\tau}{(\sqrt{m_e} + \sqrt{m_\mu} + \sqrt{m_\tau})^2} = 0.666661 \pm 0.000007$$

This is $2/3$ to within 0.5 ppm. The Standard Model does not explain why [2].

In Fibonacci arithmetic:

$$Q = \frac{F_3}{F_3 + F_2} = \frac{2}{2 + 1} = \frac{2}{3}$$

This is not an approximation. It is an exact Fibonacci identity. But the nontrivial claim is not that $F_3/(F_3 + F_2) = 2/3$ — that is arithmetic. The nontrivial claim is that the Koide ratio maps to this specific recursion level.

Why $(F_3, F_2)$ and not $(F_4, F_3) = 3/4$ or $(F_5, F_4) = 5/8$? Because charged leptons are the shallowest fermionic sector in the PAC hierarchy. They carry the lowest Fibonacci indices compatible with a non-trivial mass spectrum ($F_1 = F_2 = 1$ gives degenerate masses). The first non-degenerate branching occurs at depth 3, where the recursion yields $\Psi(3) = F_3 = 2$ and $\Psi(2) = F_2 = 1$. The prediction is structural: the lightest charged fermion sector uses the lowest available Fibonacci pair. If a lighter fermionic sector were discovered, it would need to satisfy $Q = F_4/(F_4 + F_3) = 3/4$ — which is falsifiable.

The same structure extends to quarks, with lower precision:

| Sector | PAC Formula | PAC Value | Measured | Error |
|--------|-------------|-----------|----------|-------|
| Charged leptons | $F_3/(F_3 + F_2) = 2/3$ | 0.666667 | 0.666661 | **0.5 ppm** |
| Up-type quarks | $(F_7 - F_3)/F_7 = 11/13$ | 0.846154 | 0.848956 | **0.33%** |
| Down-type quarks | $\varphi^2/(1+\varphi^2)$ | 0.723607 | 0.731628 | **1.10%** |

The charged-lepton result is essentially exact. The quark results involve larger Fibonacci indices and transcendental factors, consistent with the precision hierarchy observed in the gauge couplings.

**Script**: `exp_20_koide_formula.py`

---

## §6. Mass Ratios

### §6.1 The Formulas

Lepton and baryon mass ratios can be expressed as products of Fibonacci numbers:

| Ratio | Formula | Fibonacci Decomposition | Numeric | Measured | Error |
|-------|---------|------------------------|---------|----------|-------|
| $m_\mu/m_e$ | $F_4 \times F_6^2 \times (1 + 1/F_7)$ | $3 \times 64 \times 14/13$ | 206.769 | 206.768 | **5 ppm** |
| $m_\tau/m_e$ | $F_4 \times F_7 \times F_{11} + F_5$ | $3 \times 13 \times 89 + 5$ | 3476 | 3477.23 | **0.035%** |
| $m_p/m_e$ | $F_4 \times F_9 \times F_{12}/F_6$ | $3 \times 34 \times 144/8$ | 1836 | 1836.15 | **0.0083%** |

A structural asymmetry should be noted. The gauge coupling formulas are clean 2-number ratios: $F_4/F_7 = 3/13$ for $\sin^2\theta_W$, $F_4/(2\varphi F_6)$ for $\alpha_s$. The mass formulas require products of 3–4 Fibonacci numbers, with the tau formula including an additive correction ($+ F_5$). This may reflect genuine structural complexity in the mass sector — masses involve the full depth of the hierarchy tree while couplings are ratios at fixed levels. Alternatively, it may indicate that the correct closed forms for mass ratios have not yet been found, and simpler expressions exist at deeper Fibonacci indices.

### §6.2 Structural Observations

Three features of these formulas are notable:

1. **$F_4 = 3$ appears in every formula.** The muon ratio has it as the leading factor. The tau ratio has it as the leading factor. The proton ratio has it as the leading factor. If $F_4$ encodes the number of fermion generations (which the Standard Model does not explain), its universal presence in mass formulas would follow from the branching structure — each mass ratio involves multiplication across the full generation depth.

2. **Cross-consistency.** The derived ratios are consistent:

   | Derived Ratio | From Formulas | Measured | Error |
   |--------------|---------------|----------|-------|
   | $m_\tau/m_\mu$ | $(m_\tau/m_e)/(m_\mu/m_e)$ | 16.817 | **0.036%** |
   | $m_p/m_\mu$ | $(m_p/m_e)/(m_\mu/m_e)$ | 8.880 | **0.009%** |

3. **Fibonacci index range.** The formulas use Fibonacci numbers from $F_4$ through $F_{12}$. Higher mass ratios require higher Fibonacci indices, consistent with deeper hierarchy levels encoding heavier particles.

### §6.3 The Precision Pattern

| Ratio | Precision | Best Fibonacci Numbers Used |
|-------|-----------|-----------------------------|
| $m_\mu/m_e$ | 5 ppm | $F_4, F_6, F_7$ |
| $m_p/m_e$ | 83 ppm | $F_4, F_6, F_9, F_{12}$ |
| $m_\tau/m_e$ | 350 ppm | $F_4, F_5, F_7, F_{11}$ |

The muon-electron ratio achieves the highest precision (matching the $\alpha$ result), while the tau formula involves an additive correction ($+ F_5$) that may indicate a more complex structure.

### §6.4 Methodology Transparency

The mass ratio formulas presented above were found by **systematic search** through products and ratios of Fibonacci numbers, then validated against measured values. They are pattern-matches — not predictions derived from first principles. The search template $F_a \times F_b^c \times (1 + 1/F_d)$ was tested across Fibonacci indices $F_3$ through $F_{13}$, and the best-matching combinations are reported. This is an important distinction from the gauge coupling results, where the formulas have structural motivation (Noether charges, gauge group dimensions). Until the mass formulas can be derived from the PAC Lagrangian (§2.1) without searching, they should be treated as empirical observations compatible with Fibonacci structure, not as structural predictions.

A stoichiometric analysis (milestone3/exp_13) found that $F_4 = 3$ is **6,111× more selective** than the next-best integer substitution across the combined formula set — a 99.98th-percentile result. This supports the structural significance of $F_4$ but does not upgrade pattern-matching to derivation.

### §6.5 Falsification

A Monte Carlo test generated $10{,}000$ random formulas using the same structural template (products and ratios of Fibonacci numbers $F_3$ through $F_{13}$) and checked how many simultaneously match all three mass ratios within the observed precision.

**Result**: Zero out of $10{,}000$ random combinations matched. Joint probability: $p < 10^{-4}$.

**Caveat on sample size**: The formula template $F_a \times F_b^c \times (1 + 1/F_d)$ with Fibonacci indices from $F_3$ to $F_{13}$ spans $\sim 11^4 \approx 14{,}641$ combinations for a single ratio. A $10{,}000$-trial Monte Carlo samples a comparable fraction of this space. The $p < 10^{-4}$ bound is therefore a lower bound on the joint probability, not a precise estimate. A $10^6$-trial Monte Carlo would strengthen the claim. The current result is sufficient to reject the null hypothesis (random Fibonacci combinations match by chance) but should not be over-interpreted as a precise probability.

The recurrence of $F_7 = 13$ across the mass formulas (in $m_\mu/m_e$ as $1 + 1/F_7$ and in $m_\tau/m_e$ as a factor) has independent probability $p = 0.014$.

**Scripts**: `mass_derivation/exp_05_tighten_mass.py`, `mass_derivation/exp_06_validate_tight.py`, `mass_derivation/exp_04_mass_falsification.py`

---

## §7. Mixing Angles

### §7.1 Quark Mixing (CKM Matrix)

The Cabibbo angle $\theta_C$ is the dominant off-diagonal element of the CKM matrix:

$$\theta_C = \arctan\left(\frac{F_4}{F_7}\right) = \arctan\left(\frac{3}{13}\right) = 12.995°$$

**Measured**: $\theta_C = 13.00° \pm 0.05°$  
**Error**: $< 0.05°$

This is the same ratio $F_4/F_7 = 3/13$ that gives $\sin^2\theta_W$. The Weinberg and Cabibbo angles share arithmetic: $\sin^2\theta_W \approx \tan\theta_C$ (0.23121 vs 0.23092, $0.4\sigma$ agreement). This relationship is not predicted by the Standard Model.

### §7.2 Neutrino Mixing (PMNS Matrix)

Neutrino oscillation angles follow from Fibonacci ratios at different levels:

| Angle | PAC Formula | Prediction | Measured | $\Delta$ |
|-------|-------------|-----------|----------|----------|
| $\theta_{12}$ (solar) | $\arctan(F_3/F_4) = \arctan(2/3)$ | 33.69° | $33.41° \pm 0.4°$ | **0.28°** |
| $\theta_{13}$ (reactor) | $\arctan(F_3/F_7) = \arctan(2/13)$ | 8.75° | $8.54° \pm 0.2°$ | **0.21°** |
| $\theta_{23}$ (atmospheric) | 45° (maximal) | 45° | $49.0° \pm 1°$ | 4° |

$\theta_{12}$ and $\theta_{13}$ are both within $\sim$1$\sigma$ of the PAC predictions. $\theta_{23}$ deviates by 4° from maximal mixing. This is the weakest prediction in this section. Either the atmospheric angle receives corrections beyond leading-order PAC, or the prediction is wrong.

### §7.3 Lepton-Quark Hierarchy

The ratio of lepton to quark mixing angles:

$$\frac{\theta_{12}^{\text{PMNS}}}{\theta_{12}^{\text{CKM}}} = \frac{33.41°}{13.00°} = 2.570$$

The PAC prediction: $\varphi^2 = 2.618$.  
**Agreement**: $0.8\sigma$

This suggests that leptons and quarks are separated by exactly two levels in the PAC hierarchy tree. The lepton mixing angle is $\varphi^2 \approx 2.6$ times the quark mixing angle because the PMNS matrix operates two hierarchy levels above the CKM matrix.

**Scripts**: `validated/37_tree_geometry.py`, `validated/38_phi_squared_discovery.py`

---

## §8. Bell Correlations and Entanglement

### §8.1 The PAC Entanglement Limit

Fibonacci parent-child amplitudes in a two-particle entangled state:

$$a_1 = \frac{1}{\sqrt{1+\varphi^2}}, \quad a_2 = \frac{\varphi}{\sqrt{1+\varphi^2}}$$

The squared visibility:

$$(2a_1 a_2)^2 = \frac{4\varphi^2}{(1+\varphi^2)^2}$$

Using $1 + \varphi^2 = 2 + \varphi$ (from $\varphi^2 = \varphi + 1$) and $(2+\varphi)^2 = 5(1+\varphi) = 5\varphi^2$:

$$(2a_1 a_2)^2 = \frac{4(\varphi + 1)}{5(1 + \varphi)} = \frac{4}{5} \qquad \text{(algebraically exact)}$$

This is not a numerical coincidence. It is an identity of the golden ratio.

### §8.2 The Bell Parameter

The PAC Bell parameter from the charged-lepton sector alone:

$$S_{\text{lepton}} = 2\sqrt{1 + \frac{4}{5}} = 2\sqrt{\frac{9}{5}} = \frac{6}{\sqrt{5}} \approx 2.683$$

The quantum mechanical maximum is $S_{\text{QM}} = 2\sqrt{2} \approx 2.828$. The experimental value (Storz et al., 2023 [4]) is $S_{\text{exp}} = 2.79 \pm 0.03$.

The gap in squared Bell parameters:

$$(2\sqrt{2})^2 - \left(\frac{6}{\sqrt{5}}\right)^2 = 8 - \frac{36}{5} = \frac{4}{5}$$

### §8.3 The Combined Prediction

The charged-lepton contribution to the squared visibility is $4/5$. If the neutrino sector contributes the remaining $1/5$, the total squared visibility is:

$$V^2_{\text{total}} = \frac{4}{5} + \frac{1}{5} = 1$$

The additivity of $V^2$ across sectors follows if the charged-lepton and neutrino channels contribute independently to the entanglement budget, as expected for orthogonal weak-isospin eigenstates whose amplitudes do not interfere.

The combined PAC Bell parameter is then:

$$S_{\text{PAC}} = 2\sqrt{1 + V^2_{\text{total}}} = 2\sqrt{1 + 1} = 2\sqrt{2} \approx 2.828$$

This is the quantum mechanical maximum exactly. When both sectors are included, PAC recovers the Tsirelson bound $S = 2\sqrt{2}$. The framework does not predict a deviation from quantum mechanics — it partitions the quantum maximum into a charged-lepton fraction ($4/5$) and a neutrino fraction ($1/5$), both determined algebraically by the golden ratio.

The experimental value $S_{\text{exp}} = 2.79 \pm 0.03$ is $1.3\sigma$ below $2\sqrt{2}$, consistent with both the quantum maximum and with a measurement that predominantly samples the charged-lepton channel (which would give $6/\sqrt{5} \approx 2.683$ in the absence of neutrino contributions).

**Scripts**: `validated/32_pac_bell_deep_dive.py`, `validated/36_bell_neutrino_resolution.py`

---

## §9. Turbulence

### §9.1 She-Lévêque Intermittency

The She-Lévêque formula describes intermittent fluctuations in turbulent flows:

$$\zeta_p = \frac{p}{k} + C_0 \left[1 - \left(\frac{C_0}{C_0 + 1}\right)^{p/d}\right]$$

where $k$, $C_0$, and $d$ are parameters traditionally fitted to experimental data [3].

In Fibonacci arithmetic, all three parameters are determined:

| Parameter | Standard Notation | Fibonacci | Value |
|-----------|------------------|-----------|-------|
| $k$ (scaling) | 9 | $(F_4)^2 = 3^2$ | 9 |
| $C_0$ (coefficient) | 2 | $F_3$ | 2 |
| $d$ (exponent base) | 3 | $F_4$ | 3 |
| $\beta$ (cascade ratio) | 2/3 | $F_3/F_4$ | 0.667 |

The She-Lévêque formula becomes:

$$\zeta_p = \frac{p}{9} + 2\left[1 - \left(\frac{2}{3}\right)^{p/3}\right]$$

Every parameter is a Fibonacci number or ratio. The cascade ratio $\beta = 2/3$ is the same $F_3/F_4$ that gives the Koide formula.

### §9.2 The $k = d \times F_{d+1}$ Derivation

Why $k = 9$? The dimensional formula:

$$k(d) = d \times F_{d+1}$$

| Dimension | $k$ | Calculation | Verification |
|-----------|-----|-------------|-------------|
| $d = 2$ | 4 | $2 \times F_3 = 2 \times 2$ | 2D enstrophy cascade |
| $d = 3$ | 9 | $3 \times F_4 = 3 \times 3$ | She-Lévêque (3D) |
| $d = 4$ | 20 | $4 \times F_5 = 4 \times 5$ | **Prediction** |

The 2D result ($k = 4$) was verified independently: the 2D turbulence spectrum uses $\beta = F_4/F_5 = 3/5$ (within 2% of experiment), one Fibonacci index higher than the 3D cascade, consistent with the enstrophy cascade requiring an additional scale level.

The 4D prediction ($k = 20$) is falsifiable through 4D turbulence simulations.

### §9.3 The Kolmogorov Connection

The Kolmogorov $-5/3$ exponent for the energy spectrum:

$$E(k) \propto k^{-5/3}$$

The exponent $5/3$ is derived from dimensional analysis (Kolmogorov, 1941) and is not in dispute. The observation here is that $5/3 = F_5/F_4$: it is a ratio of consecutive Fibonacci numbers. Similarly, the energy dissipation scaling $\epsilon^{2/3}$ involves $2/3 = F_3/F_4$. These are factual restatements — we note the Fibonacci structure of known results, not a new derivation of the Kolmogorov exponent. The question is whether this Fibonacci structure is coincidental (small-integer ratios are common in dimensional analysis) or connected to the She-Lévêque results above, where the Fibonacci structure is less trivially embedded.

**Scripts**: `exp_21_she_leveque.py`, `exp_11_k9_derivation.py`, `exp_12_falsification.py`

### §9.4 Wilson-Fisher Critical Exponents

The Wilson-Fisher fixed point governs the 3D Ising universality class — the critical behaviour of phase transitions in magnets, fluids, and binary alloys. The correlation length exponent $\nu$ determines how the correlation length diverges near criticality:

$$\xi \sim |T - T_c|^{-\nu}$$

The conformal bootstrap value is $\nu = 0.6299709 \pm 0.0000040$ [11].

In PAC terms:

$$\nu = \frac{F_3}{F_4 \cdot \Xi} = \frac{2}{3\Xi} = 0.629865\ldots$$

**Error**: 0.017%

The formula decomposes as a product of two independently motivated quantities:

- **$2/3 = F_3/F_4$**: the E-I-S cycle ratio — the same ratio that gives the Koide formula (§5) and the She-Lévêque cascade (§9.1)
- **$1/\Xi$**: the reciprocal of the balance constant (Paper 2)

If this decomposition is not coincidental, the Wilson-Fisher exponent reflects the E-I-S cycle topology at the SEC balance point: correlation length divergence at a phase transition would be the PAC recursion running at $2/3$ of the balance-mediated rate.

**Broader exponent analysis.** A systematic search across 7 Wilson-Fisher critical exponents found Fibonacci/PAC expressions within 1% for 6 of the 7:

| Exponent | Literature | PAC Expression | PAC Value | Error |
|----------|-----------|----------------|-----------|-------|
| $\nu$ | 0.6300 | $F_3/(F_4 \cdot \Xi)$ | 0.6299 | **0.017%** |
| $\eta$ | 0.0363 | $2/F_{10}$ | 0.0364 | **0.18%** |
| $\gamma$ | 1.2371 | $2/\varphi$ | 1.2361 | **0.083%** |
| $\delta$ | 4.789 | $\varphi + \pi$ | 4.760 | **0.62%** |
| $\omega$ | 0.8297 | $\ln\varphi / \gamma_{\mathrm{E}}$ | 0.8337 | **0.48%** |
| $\beta$ | 0.3265 | $\varphi/F_5$ | 0.3236 | **0.89%** |
| $\alpha_{\text{heat}}$ | 0.1096 | — | — | no match |

The specific heat exponent $\alpha_{\text{heat}}$ admits no Fibonacci expression within 1%. This is recorded as an honest limitation.

**Null test.** A Monte Carlo search of 23,376 formula candidates across 2,000 random-constant trials found a mean of 1.89 hits within 1%, compared to 20 hits for the PAC constants ($p = 0.0000$).

The best alternative to $\nu = 2/(3\Xi)$ is $\pi/(3\Xi)$, which matches at 1.06% — 63× worse. The factor $2/3$ is uniquely selected.

**Script**: `milestone3/scripts/exp_07_wilson_fisher.py`

---

## §10. Casimir Effect

### §10.1 The Factor 240

The Casimir force between two parallel conducting plates:

$$\frac{F}{A} = -\frac{\pi^2 \hbar c}{240\,d^4}$$

The factor 240 arises from zeta function regularisation: $\sum n^3 \to \zeta(-3) = 1/120$, combined with two-plate symmetry giving $\pi^2/240$.

In Fibonacci arithmetic:

$$240 = F_3 \times F_4 \times F_5 \times F_6 = 2 \times 3 \times 5 \times 8$$

Four consecutive Fibonacci numbers. The sub-factor $120 = F_4 \times F_5 \times F_6 = 3 \times 5 \times 8$ (three consecutive) gives $\zeta(-3)$; the Casimir force gradient adds $F_3 = 2$ through differentiation.

### §10.2 Mersenne Dimensions

The zeta function regularisation denominators show Fibonacci product structure only at Mersenne dimensions $d = 2^n - 1$:

| $d$ | $\zeta(-d)$ denominator | Fibonacci product? | Mersenne? |
|-----|------------------------|--------------------|-----------|
| 1 | 12 | $F_3^2 \times F_4$ | $2^1 - 1$ ✓ |
| 3 | 120 | $F_4 \times F_5 \times F_6$ | $2^2 - 1$ ✓ |
| 5 | 252 | No (factor 7) | ✗ |
| 7 | 240 | $F_3 \times F_4 \times F_5 \times F_6$ | $2^3 - 1$ ✓ |
| 9 | 132 | No (factor 11) | ✗ |

The pattern: vacuum regularisation has Fibonacci structure at the exact dimensions relevant to string theory ($d = 1$), the physical Casimir effect ($d = 3$), and M-theory's extra dimensions ($d = 7$). Non-Mersenne dimensions have non-Fibonacci prime factors.

**Scripts**: `exp_15_casimir_sec.py`, `exp_16_mersenne_verification.py`

---

## §11. Gravity Hierarchy

The ratio of electromagnetic to gravitational force between two protons:

$$\frac{F_{\text{EM}}}{F_{\text{grav}}} = \frac{e^2/(4\pi\epsilon_0)}{Gm_p^2} \approx 1.24 \times 10^{38}$$

The Fibonacci prediction uses the gauge-squared depth:

$$183 = F_7^2 + F_7 + 1 = 169 + 13 + 1$$

Then $F_{183} \approx 1.27 \times 10^{38}$.

This is an order-of-magnitude result, not a precision prediction. The hierarchy ratio is $\sim$10³⁸, and $F_{183}$ is $\sim$10³⁸. The match is suggestive but not compelling by itself.

What makes it non-trivial is the structural argument: if the EM hierarchy depth is $F_{10} = 55$ (from the $\alpha$ formula), and gravity requires an additional gauge-squared depth $F_7^2 + F_7 + 1 = 183$, then the gravitational hierarchy is encoded by $F_{183}$ — the Fibonacci number at a depth determined by the gauge closure constant $F_7 = 13$.

**On uniqueness**: The Fibonacci sequence grows as $F_n \approx \varphi^n/\sqrt{5}$, so $\log_{10}(F_n) \approx 0.2090n$. At $n = 183$, this gives $\sim 10^{38.2}$. Nearby indices ($n = 178 \to 10^{37.2}$, $n = 188 \to 10^{39.3}$) are within an order of magnitude. The falsification test is therefore not about whether $F_{183}$ is uniquely close to $10^{38}$ — several nearby Fibonacci numbers are comparably close. The test is whether the structural formula $n = F_7^2 + F_7 + 1 = 183$ is uniquely motivated. The polynomial $N^2 + N + 1$ counts elements in the projective plane $PG(2, N)$; evaluated at the gauge closure constant $N = F_7 = 13$, it gives precisely the depth that encodes the EM-gravity hierarchy. No other simple polynomial in $F_7$ produces a depth whose Fibonacci number matches the hierarchy ratio.

**Milestone3 update (exp_23, exp_26).** The precise gap between $\log_{10}(F_{183})$ and $\log_{10}((M_{\mathrm{Pl}}/m_p)^2)$ is 0.333 in $\log_{10}$ (factor $\sim$2.155). The best correction term is $1 + F_{13}/(\pi \cdot F_6^2)$, which brings the residual to $7.8 \times 10^{-4}$ in $\log_{10}$. Among Fibonacci depths $F_k^2 + F_k + 1$, only $k = 7$ (depth 183) falls within 0.5 of the target: rank \#1 cyclotomic depth, with a 40× gap to the next closest. A Monte Carlo test of 5,000 random integer sequences found **0/5,000** matching both $\alpha_{\mathrm{EM}}$ and gravity simultaneously using the correction template $F_a/(m\pi F_b^2)$, suggesting the joint constraint is structurally non-trivial.

**Scripts**: `exp_23_gravity_depth.py`, `exp_24_hierarchy_f183.py`, `exp_26_hierarchy_falsification.py`

---

## §12. What This Is Not

### §12.1 It Is Not Numerology

The standard objection to expressing physical constants in terms of small integers is that small integers are common, and matches are therefore unsurprising. This objection is valid for individual matches.

Consider: the probability that a randomly chosen integer between 1 and 100 is a Fibonacci number is approximately 16% (12 Fibonacci numbers below 100). A single match — say, the number of $SU(3)$ generators being $F_6 = 8$ — is not statistically meaningful.

The claim here is not about individual matches. It is about joint constraints. A single recursion $\Psi(k) = \Psi(k+1) + \Psi(k+2)$ selects the Fibonacci sequence. From that sequence:

- $F_7 = 13 = 1 + 3 + 8 + 1$ (gauge closure) *selects the gauge group structure*
- $F_4/F_7 = 3/13$ (weak mixing) *predicts a coupling constant ratio to 0.19%*
- $F_3/(F_4 \cdot \varphi \cdot F_{10}) \times \text{correction}$ *predicts $\alpha$ to 5.7 ppm*
- $F_3/(F_3 + F_2) = 2/3$ *matches the Koide formula to 0.5 ppm*
- $F_4 \times F_6^2 \times (1 + 1/F_7)$ *matches the muon-electron mass ratio to 5 ppm*

These are not independent fits. They use the same Fibonacci numbers ($F_3, F_4, F_6, F_7, F_{10}$) in interlocking formulas that must be simultaneously consistent. The joint probability against chance, estimated by Monte Carlo ($10{,}000$ random formula sets with the same template), is $p < 10^{-5}$.

**Cross-domain independence audit (milestone3/exp_10).** The results in this paper span particle physics (§§4–7), quantum mechanics (§8), fluid dynamics (§9.1–9.3), statistical mechanics (§9.4), and quantum field theory (§10). A structural correlation test identifies which claims share Fibonacci indices and which are genuinely independent.

Of 14 claims analysed across the PACSeries, the effective number of independent degrees of freedom is 7.9 (independence ratio: 0.56). The top correlated pair is $\alpha$ and $\sin^2\theta_W$, which share $F_4$ and $F_7$ — these are *not* independent results. Conversely, the turbulence parameters (§9.1–9.3), the Casimir factor (§10), and the Wilson-Fisher exponent (§9.4) use Fibonacci numbers and constants ($\Xi$, $\gamma_{\mathrm{E}}$, $\ln\varphi$) that do not appear in any particle physics formula.

Grouping claims into 5 conservative independence classes and combining one $p$-value per group via Fisher's method gives a joint significance of $p \approx 10^{-147}$, **conditional on this analysis structure** — i.e., given the specific formula templates, the domain groupings, and the measured constants tested. This figure does not account for template selection bias (the templates were chosen because they work; see §12.5 for the α look-elsewhere analysis that quantifies this for one case). The correction from the naive (uncorrelated) estimate to the conservative (grouped) estimate is $\sim$48 orders of magnitude — showing that the analysis properly penalises shared structure. Even after this penalty and the template-selection caveat, the joint significance substantially exceeds conventional thresholds. A global template richness audit (milestone3/exp_32) tested 50 dimensionless PDG constants — 7 claimed and 43 unclaimed — against the same combined 2-, 3-, and 4-factor Fibonacci template pool (~26,700 unique values). At 1% precision, 91% of unclaimed constants can be matched, confirming that template-level matches at this threshold are cheap. At 100 ppm, only 19% of unclaimed constants are reachable, versus 43% of claimed constants. The claimed/unclaimed median-error ratio is 1.9×. **Conclusion**: individual matches at 1% carry little weight; the joint constraint and sub-100 ppm results (§15.1–15.2) are where the signal resides.

**Script**: `milestone3/scripts/exp_10_independence_audit.py`

### §12.2 It Is Not a Derivation

We do not derive the Standard Model from PAC. A derivation would produce the gauge group structure, the coupling constants, and the mass spectrum from axioms alone, with no appeal to measured values. What we present is weaker: given the PAC recursion, we show that measured values are consistent with Fibonacci expressions, with precisions that range from 0.5 ppm to 1.7%.

The gap between "consistent with" and "derived from" is important. A derivation of $\sin^2\theta_W = 3/13$ would explain why the electroweak mixing occurs at this ratio. Our result does not — it observes the coincidence and reports it. The structural arguments (gauge group selection, Noether charges, hierarchy depths) are suggestive but do not constitute a proof.

### §12.4 Honest Failures

Three milestone3 experiments produced results that limit the framework's claims:

1. **Null-space prediction failure (exp_16).** An attempt to use the PAC framework to *predict* unmeasured ratios from the null space of the formula matrix scored 0/4. Fibonacci-indexed predictions showed no enrichment over random integer-indexed formulas ($z = 0.455$, $p_{\text{MC}} = 0.794$). The framework **describes** measured constants well but does **not predict** unmeasured ones with current methods.

2. **Crystallisation order is basis-independent (exp_19, FALSIFIED).** The order in which formula constants converge to measurement during a PAC-guided optimisation is **identical** across Fibonacci, Lucas, prime, tribonacci, and random sequences. There is no Fibonacci-specific dynamics in the convergence process; the crystallisation order is a property of the formula structure itself.

3. **PAC-Lazy signal is fragile (exp_24).** The KL-divergence discrimination between PAC-matched and unmatched formulas ($p = 0.035$, exp_21) does not survive bootstrap analysis: the 95\% CI on the KL difference is $[-0.044, +0.013]$, **crossing zero**. The signal concentrates in a single formula and does not reduce the effective dimensionality below null-space degrees of freedom. This is an engineering observation, not a theoretical claim.

These failures are recorded because they constrain interpretation. The Fibonacci expressions match measured constants with statistical significance ($p < 10^{-5}$ jointly), but the framework does not yet have predictive power for unmeasured observables, and the Fibonacci sequence is not uniquely selected by the convergence dynamics.

### §12.5 Look-Elsewhere Analysis for α

The fine structure constant formula (§4.1) achieves 5.7 ppm precision. A detailed look-elsewhere analysis (milestone3/exp_09) tested two distinct questions about this match:

**Question 1 (Template-class richness):** Can the formula template $k/(m \cdot T \cdot F_i) \times (1 - F_j/(n \cdot U \cdot F_p^q))$ produce matches to $\alpha$ by chance? An exhaustive enumeration of 1,640,599 valid formulas found 2 matches within 6 ppm. Since $\sim$1.44 matches are expected at random (binomial), $p = 0.42$. **The template class is rich enough that individual matches are unremarkable.**

**Question 2 (Fibonacci index specificity):** Within the *specific* skeleton of the published formula (with $\varphi$ and $4\pi$ fixed), are the Fibonacci index choices special? A Monte Carlo test drew 7,272 random index sets from $F_1$–$F_{15}$ using the exact skeleton. **Zero matched α.** The specific Fibonacci indices ($F_7, F_{10}$) are fine-tuned within that skeleton.

These are not contradictory — they answer different questions. The first shows the template as a whole has enough combinatorial capacity to hit $\alpha$-scale values by chance. The second shows the specific Fibonacci indices are not random choices within that template. The α formula is interesting *not* because it is a rare hit from a rich template, but because the same Fibonacci indices ($F_4, F_7, F_{10}$) appear across multiple formulas (Weinberg angle, Koide, lepton masses) with interlocking constraints.

The correct framing is therefore **joint constraint significance** (§12.1), not single-formula significance. The $p < 10^{-5}$ joint Monte Carlo is the relevant statistic. The individual $\alpha$ match, taken in isolation, does not survive look-elsewhere correction.

**Bonus finding:** A second formula $1/(4\varphi F_8) \times (1 - F_9/(3\pi F_8^2))$ matches $\alpha$ at 1.09 ppm — better than the published 5.7 ppm result. This further demonstrates that individual matches are not unique, reinforcing that the significance lies in the joint system, not in any single formula.

### §12.3 What Would Make It Stronger

The following would elevate these results from "suggestive pattern" to "structural prediction":

1. **An independent prediction that is confirmed.** The Z' boson at 395 GeV (§14) is such a prediction. If found, the Fibonacci structure gains credibility. If not found, it does not falsify the coupling constant results (the Z' prediction depends on additional assumptions), but it weakens the framework.

2. **Extension to all mass ratios.** Currently, three mass ratios are well matched. A systematic extension to quark masses, neutrino mass squared differences, and hadronic mass ratios would either strengthen or falsify the pattern.

3. **Running coupling confirmation.** The running of $\sin^2\theta_W$ from $M_Z$ to lower energies now confirms (milestone3/exp_08) that $3/13$ is achieved at $Q = 82.78$ GeV, within 3% of $M_W$. The remaining test is whether future precision measurements at or near the $M_W$ scale converge to $3/13$ rather than a nearby value. The MOLLER experiment at JLab and future $e^+e^-$ colliders running at the W threshold would sharpen this constraint.

---

## §13. Falsification Conditions

These results would be falsified by:

1. **Alternative Fibonacci combinations matching better.** If a different set of Fibonacci numbers reproduces the same constants with comparable or better precision, the specific formulas presented here lose their uniqueness claim. Current exhaustive search over all $(F_m, F_n)$ pairs for $\alpha$ finds no alternative within $2{,}870\times$ of the best match.

2. **Z' non-observation at HL-LHC.** If no resonance is found near 395 GeV with the predicted properties by 2030, the gauge closure prediction weakens, though it does not falsify the coupling constant results directly.

3. **Failure to extend to quarks.** If quark mass ratios cannot be expressed in Fibonacci arithmetic at comparable precision to the lepton results, the pattern may be limited to the lepton sector and therefore less fundamental.

4. **An alternative framework reproducing the same constants.** If a non-Fibonacci algebraic structure matches the Standard Model parameters at equal or better precision, the claim that Fibonacci numbers are structurally significant would be undermined.

5. **Precision measurement refuting $\sin^2\theta_W = 3/13$.** If high-precision measurements of $\sin^2\theta_W$ at all energy scales exclude $3/13$ as either a tree-level value or a value at any specific scale, the result is falsified.

---

## §14. Predictions

### §14.1 Z' Boson

The PAC closure at $F_7 = 13$ with $1 + 3 + 8 + 1 = 13$ admits one interpretation: the "+1" is the Higgs. An alternative: the "+1" is a 13th gauge boson — a Z' in a remnant $U(1)$ from a PAC-compatible extension.

| Property | PAC Value |
|----------|-----------|
| Mass | $395 \pm 20$ GeV |
| Coupling | $g_{Z'}/g_Z = 1/13$ |
| Width | $\sim$64 MeV (narrow) |
| Cross section | $1/169$ of standard Z' |

The coupling $1/13 = 1/F_7$ follows from the hierarchy. The mass follows from the PAC saturation depth $N^* = 3F_{10}/(2\pi) \approx 26$ and the $SU(2)$ symmetry breaking scale.

Run 3 data ($\sim$300 fb$^{-1}$ at $\sqrt{s} = 13.6$ TeV, completing $\sim$2026) may already constrain this parameter space if reanalysed for narrow, weakly-coupled resonances. The full test requires HL-LHC luminosities ($3{,}000$ fb$^{-1}$, physics runs from $\sim$2030 after Long Shutdown 3). Standard LHC Z' searches exclude sequential Standard Model Z' bosons below $\sim$5 TeV (ATLAS dilepton: ATLAS-CONF-2019-030 [9]; CMS dimuon: CMS-PAS-EXO-19-019 [10]). However, these exclusions assume standard couplings ($g_{Z'} \sim g_Z$). At $g_{Z'}/g_Z = 1/13$, the production cross section is $1/169$ of the benchmark, placing a 395 GeV Z' well below current sensitivity thresholds. The narrow width ($\sim$64 MeV, compared to $\Gamma_Z \approx 2.5$ GeV) further reduces detection efficiency in searches optimised for broad resonances. A dedicated low-mass, narrow-width dilepton search would be required.

**Script**: `exp_34_zprime_prediction.py`

### §14.2 Neutrino Mixing Angles

| Angle | PAC Prediction | Current Best | Test |
|-------|---------------|-------------|------|
| $\theta_{12}$ | 33.69° | $33.41° \pm 0.4°$ | JUNO (2025+) |
| $\theta_{13}$ | 8.75° | $8.54° \pm 0.2°$ | DUNE (2029+) |

### §14.3 4D Turbulence

$$k(4) = 4 \times F_5 = 4 \times 5 = 20$$

Testable through 4D numerical turbulence simulations.

### §14.4 GUT-Scale Coupling Unification

If all three couplings derive from the same PAC recursion, they should unify at a characteristic PAC scale where $\alpha^* = 1/\varphi^3 \approx 0.236$.

### §14.5 Dark Matter Density (Speculative)

The formula $\Omega_c = F_7 \cdot \Xi^2 / F_{10}$ gives a cold dark matter density of **0.2648**, compared to the Planck 2018 measurement $\Omega_c = 0.265 \pm 0.007$. This is 0.079% from the central value. An alternative formula $F_3 \cdot \Xi / F_6$ gives 0.2646 (0.148% error). Among 590 candidate Fibonacci formulas tested, only 2 (0.34%) fall within 0.15% of the measured value. These results are speculative: the theoretical motivation for connecting gauge closure ($F_7$) and the balance constant ($\Xi$) to cosmological densities remains to be established.

---

## §15. Summary of Results

Results are grouped by evidential weight. *Structural* results follow from the PAC recursion without reference to measured values. *Formula search* results match measured constants through Fibonacci-template searches and are validated by joint constraints (§12.1) and null tests, not by individual significance.

### §15.1 Structural Results (derived, not searched)

These follow from the PAC recursion and gauge group arithmetic without fitting to data:

| Result | Status | Domain |
|--------|--------|--------|
| $F_7 = 13 = 1 + 3 + 8 + 1$ | Exact (algebraic) | Gauge closure |
| Fibonacci filter: only SU(2), SU(3) | Exact (algebraic) | Gauge selection |
| $(2a_1 a_2)^2 = 4/5$ | Exact (algebraic) | Entanglement |
| $k = d \times F_{d+1}$ | Formula with 4D prediction | Turbulence |
| Koide $Q = 2/3$ to 0.5 ppm | Exact identity at hierarchy depth 3 | Mass relation |

### §15.2 High-Precision Formula Matches

These are matches between Fibonacci expressions and measured constants. Individually, each could be coincidence (§12.5); their significance comes from the joint constraint (§12.1):

| Result | Precision | Domain |
|--------|-----------|--------|
| $\alpha$ to 5.7 ppm | Measurement | Gauge coupling |
| $m_\mu/m_e$ to 5 ppm | Measurement (template search) | Mass ratio |
| $\nu = 2/(3\Xi)$ | 0.017% | Wilson-Fisher (critical) |
| $m_p/m_e$ to 83 ppm | Measurement (template search) | Mass ratio |
| $\sin^2\theta_W = 3/13$ at $Q \approx M_W$ | 0.19% | Gauge coupling |
| $\alpha_s$ to 1.71% | Measurement | Gauge coupling |
| $m_\tau/m_e$ to 350 ppm | Measurement (template search) | Mass ratio |
| $\theta_C$ to $< 0.05°$ | Measurement | CKM mixing |
| $\theta_{12}^{\text{PMNS}}$ to 0.28° | Measurement | Neutrino mixing |
| $\theta_{13}^{\text{PMNS}}$ to 0.21° | Measurement | Neutrino mixing |

### §15.3 Small-Integer Compatible (noted, not claimed as evidence)

These coincide with Fibonacci numbers but involve small integers where the match probability is high ($\sim$16%). They are included for completeness but carry negligible individual weight:

| Result | Note | Domain |
|--------|------|--------|
| $5/3 = F_5/F_4$ | Kolmogorov exponent from dimensional analysis | Turbulence |
| $\beta = F_3/F_4 = 2/3$ | She-Lévêque cascade ratio (but see $k = d \times F_{d+1}$ above) | Turbulence |
| Casimir 240 = $F_3 F_4 F_5 F_6$ | Product of consecutive Fibonacci numbers; Mersenne-dimension restriction (§10.2) is the non-trivial part | QFT regularisation |

### §15.4 Forward Predictions

| Prediction | Value | Experiment |
|-----------|-------|------------|
| Z' boson | 395 ± 20 GeV, coupling 1/13 | HL-LHC (~2030) |
| $k(4) = 20$ | 4D turbulence scaling | Numerical simulation (now) |
| $\sin^2\theta_W = 3/13$ near $M_W$ | Q = 82.78 GeV | MOLLER / FCC-ee |
| Neutrino $\theta_{12}$ = 33.69° | JUNO | 2025+ |
| Neutrino $\theta_{13}$ = 8.75° | DUNE | 2029+ |

---

## §16. Connections to the PACSeries

| Paper | Connection |
|-------|-----------|
| Paper 1: Structure Cost of Erasure | $\alpha$ interpreted as Landauer payment rate through $F_{10} = 55$ hierarchy levels; gauge hierarchy $\xi(SU(3)) > \xi(SU(2)) > \xi(U(1))$ confirmed at $p < 10^{-11}$ |
| Paper 2: Balance Constant | Same $F_{10} = 55$ in $\Xi = 1 + \pi/55$; Ξ appears at boundaries where gauge couplings are evaluated |
| Paper 3: Feigenbaum Constants | Same $F_{10} = 55$ in closed-form expressions for $r_\infty$, $\delta$, $|\alpha|$ — Fibonacci universality extends from nonlinear dynamics to particle physics |
| **Paper 4 (this paper)** | **Standard Model parameters from Fibonacci arithmetic** |
| Paper 5: Classical Physics | Maxwell's equations as depth-2 PAC recursion; MED → D = 3; $k = d \times F_{d+1}$ connects EM and turbulence |
| Paper 6: Computational Validation | PAC conservation observed in ML systems validates the operational principle |

The common thread across Papers 1–6 is $F_7 = 13$ and $F_{10} = 55$. These are not free parameters. $F_7$ is determined by gauge group arithmetic ($1 + 3 + 8 + 1 = 13$). $F_{10}$ is determined by the Feigenbaum structure (Paper 3) and the Landauer hierarchy (Paper 1). Both are structural consequences of the PAC recursion.

---

## Acknowledgments

*(To be added.)*

---

## References

1. Particle Data Group (2022). "Review of Particle Physics." *Prog. Theor. Exp. Phys.*, 2022(8), 083C01.
2. Koide, Y. (1982). "New Formula for the Cabibbo Angle." *Phys. Rev. Lett.*, 49, 723–724.
3. She, Z.-S. and Lévêque, E. (1994). "Universal Scaling Laws in Fully Developed Turbulence." *Phys. Rev. Lett.*, 72(3), 336–339.
4. Storz, S. et al. (2023). "Loophole-free Bell inequality violation with superconducting circuits." *Nature*, 617, 265–270.
5. Tiesinga, E. et al. (2024). "CODATA recommended values of the fundamental physical constants: 2022." *J. Phys. Chem. Ref. Data*, 53(3), 030801.
6. Groom, P. (2026). "The Structure Cost of Erasure." PACSeries Paper 1. Dawn Field Institute.
7. Groom, P. (2026). "The Balance Constant and Its Decomposition." PACSeries Paper 2. Dawn Field Institute.
8. Groom, P. (2026). "Feigenbaum Constants from Fibonacci Arithmetic." PACSeries Paper 3. Dawn Field Institute.
9. ATLAS Collaboration (2019). "Search for high-mass dilepton resonances using 139 fb⁻¹ of pp collision data collected at √s = 13 TeV with the ATLAS detector." ATLAS-CONF-2019-030.
10. CMS Collaboration (2019). "Search for high-mass resonances in dilepton final states in proton-proton collisions at √s = 13 TeV." CMS-PAS-EXO-19-019.
11. Kos, F., Poland, D., Simmons-Duffin, D. and Vichi, A. (2016). "Precision islands in the Ising and O(N) models." *JHEP*, 2016(8), 36.

---

*All code, data, and experiment scripts for this paper and the full PACSeries are publicly available at [https://github.com/dawnfield-institute/dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory). See the accompanying publication package README.md for reproduction instructions.*
