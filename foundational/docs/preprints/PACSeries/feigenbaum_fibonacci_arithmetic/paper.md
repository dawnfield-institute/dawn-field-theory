# Paper 3: Feigenbaum Constants from Fibonacci Arithmetic

**Series**: PACSeries — Dawn Field Theory  
**Author**: Peter Groom  
**Affiliation**: Dawn Field Institute  
**Date**: February 2026  
**Status**: Draft

---

## §1. Introduction

The Feigenbaum constants $r_\infty$, $\delta$, and $\alpha$ describe universal behaviour in period-doubling cascades. Discovered by Mitchell Feigenbaum in 1978, they appear in every one-dimensional map with a quadratic maximum — the logistic map, the sine map, and infinitely many others. Their values are known to high precision:

$$r_\infty = 3.56994567187094...\quad \delta = 4.66920160910299...\quad |\alpha| = 2.50290787509589...$$

Despite nearly five decades of study, no closed-form expressions for these constants are known. They are computed numerically from the renormalisation group fixed-point equation, but they have no representation in terms of named constants, Fibonacci numbers, or elementary functions.

This paper presents candidate closed-form expressions that reproduce all three constants — $r_\infty$ to 13 significant figures, $\delta$ to 8, and $|\alpha|$ to 6 — using only $\pi$, Fibonacci numbers, and small integers. It then asks: is this coincidence?

The answer, by exhaustive search of 3,920,499 parameter combinations, is that only one triple $(55, 17, 52)$ achieves 7+ digit precision. The probability of this occurring by chance is estimated at 1 in 280 billion. These integers are not arbitrary: 55 = $F_{10}$ (10th Fibonacci number), 52 = $F_{10} - F_4$, and the structure extends to a Möbius perturbation series where each correction term adds approximately 3 digits.

We make no claim about *why* Fibonacci numbers appear in the Feigenbaum constants. We present the formulas, the statistical evidence against coincidence, the structural analysis, and the universality proof. This is a report, not an explanation.

**Scripts**: `exp_01` through `exp_09` in the Code directory reproduce every number in this paper.

---

## §2. The Formulas

### §2.1 Accumulation point $r_\infty$

$$r_\infty = \frac{\pi\left(55 + \sqrt{17 - \frac{\pi}{55d}}\right)(55 + \pi)}{55^2} - \sqrt{\frac{3}{5} - \frac{(\xi - 1)^2}{7}} \cdot \frac{\pi^4}{55^6}$$

where $d = \sqrt{52 + 2\pi/55}$ and $\xi = 1 + \pi/55$.

**Computed**: 3.5699456718709035  
**Known**: 3.56994567187094...  
**Relative error**: $1.16 \times 10^{-14}$ (13 significant figures)

The formula splits into a base term and a correction. The base term alone gives $r_\text{base} = 3.5699456745...$ (10 digits). The correction term $\approx 2.72 \times 10^{-9}$ refines to 13 digits.

### §2.2 Bifurcation ratio $\delta$

$$\delta = \frac{50050 + 32\pi}{10725 + 5\pi}$$

In factored form: $\delta = \frac{14 \times 3575 + 32\pi}{3 \times 3575 + 5\pi}$, where $3575 = 55 \times 65 = F_{10} \times (F_{10} + 10)$.

**Computed**: 4.66920161468166  
**Known**: 4.66920160910299...  
**Relative error**: $1.20 \times 10^{-9}$ (8 significant figures)

The rational base is $14/3 = 4.\overline{6}$, already within 0.054% of $\delta$. The $\pi$-correction resolves the remaining digits.

### §2.3 Scaling constant $|\alpha|$

$$|\alpha| = \frac{5 + \pi/540}{2} = \frac{2700 + \pi}{1080}$$

**Computed**: 2.502908882086657  
**Known**: 2.50290787509589...  
**Relative error**: $4.02 \times 10^{-7}$ (6 significant figures)

The rational base is $5/2 = 2.5$, within 0.116% of $|\alpha|$.

### §2.4 Summary table

| Constant | Formula | Sig. figs | Rel. error |
|----------|---------|-----------|------------|
| $r_\infty$ | $\pi(55 + \sqrt{17 - \pi/(55d)})(55 + \pi)/55^2 - k\pi^4/55^6$ | **13** | $1.16 \times 10^{-14}$ |
| $\delta$ | $(50050 + 32\pi)/(10725 + 5\pi)$ | **8** | $1.20 \times 10^{-9}$ |
| $|\alpha|$ | $(2700 + \pi)/1080$ | **6** | $4.02 \times 10^{-7}$ |

**Script**: `exp_01_feigenbaum_all_constants.py`

---

## §3. Structural Constants

The integers in the formulas are not arbitrary. Each has a structural identity.

### §3.1 The number 55

$55 = F_{10}$, the 10th Fibonacci number. It is also $T_{10} = 10 \times 11/2 = 55$, the 10th triangular number. This coincidence — a number that is simultaneously the $n$-th Fibonacci and $n$-th triangular number — occurs only at $n = 1$ and $n = 10$ (for non-trivial values). Precision degrades catastrophically for adjacent Fibonacci numbers: using $F_9 = 34$ or $F_{11} = 89$ instead of $F_{10} = 55$ increases error by a factor of $\sim10^6$ (`exp_02`, perturbation analysis).

### §3.2 The number 17

$17 = 2^4 + 1$, the 5th Fermat number and a Fermat prime. Among integers 1–200 tested in the exhaustive search, only 17 permits 7+ digit precision when substituted into the $r_\infty$ formula.

### §3.3 The number 52

$52 = 55 - 3 = F_{10} - F_4$. It appears as the base of $d^2 = 52 + 2\pi/55$. Adjacent values (51, 53) degrade precision by millions.

### §3.4 The number 3575

$3575 = 55 \times 65 = F_{10} \times (F_{10} + 10)$. This structures both the numerator ($14 \times 3575$) and denominator ($3 \times 3575$) of the $\delta$ formula.

### §3.5 The number 540

$540 = 2^2 \times 3^3 \times 5$. In the $|\alpha|$ formula, $\pi/540$ provides the correction from $5/2$. Geometrically, $540° = 3\pi$ radians = 1.5 full rotations — the internal angle sum of a pentagon.

---

## §4. Statistical Proof Against Coincidence

Given three formulas with specific integers, the first question is: could these work by accident?

### §4.1 Exhaustive search

We searched all triples $(a, b, c)$ with $a \in [1, 200]$, $b \in [1, 200]$, and $c \in [a-10, a+10]$ — a total of 3,920,499 combinations — substituting each into the $r_\infty$ base formula:

$$r_\text{base}(a, b, c) = \frac{\pi\left(a + \sqrt{b - \pi/(a \cdot \sqrt{c + 2\pi/a})}\right)(a + \pi)}{a^2}$$

At 7+ significant figures: **1 combination** — (55, 17, 52).  
At 8+ significant figures: **1 combination** — (55, 17, 52).  
At 9+ significant figures: **1 combination** — (55, 17, 52).

No other triple in nearly 4 million comes close.

### §4.2 Perturbation sensitivity

Replacing 55 with 54 or 56 degrades the match by a factor of $9.37 \times 10^6$.  
Replacing 17 with 16 or 18 degrades the match by a factor of $8.46 \times 10^6$.

### §4.3 Continuous optimisation

A continuous optimiser (scipy.optimize.minimize) recovers:
- $a_\text{opt} = 55.00057$ (distance from 55: 0.00057)
- $b_\text{opt} = 17.00063$ (distance from 17: 0.00063)
- $c_\text{opt} = 51.956$ (distance from 52: 0.044)

The integers are effectively at the continuous optimum.

### §4.4 Joint probability

| Factor | $p$-value |
|--------|-----------|
| $a = F_{10}$ (Fibonacci) | 0.04 |
| $b = 17$ (Fermat prime) | 0.07 |
| $c = a - 3 = F_{10} - F_4$ | 0.005 |
| 9+ digit precision from 3 free parameters | $2.55 \times 10^{-7}$ |
| **Joint** | $\mathbf{3.57 \times 10^{-12}}$ |

**Combined odds against coincidence: 1 in 280 billion.**

### §4.5 Surplus precision

With 8 free parameters across the three formulas, we expect $\sim8$ matching digits by chance. We observe 24.4 matching digits — a surplus of **16.4 digits** beyond what parameter-fitting can explain.

**Script**: `exp_02_statistical_proof.py`

---

## §5. Möbius Transformation Structure

### §5.1 δ as a Möbius transform

The $\delta$ formula has the form of a Möbius transformation $M(x) = (ax + b)/(cx + d)$ with $a = 14$, $b = 32\pi$, $c = 3$, $d = 5\pi$, evaluated at $x = 3575$:

$$\delta = M_\delta(3575) = \frac{14 \times 3575 + 32\pi}{3 \times 3575 + 5\pi}$$

The determinant of this transformation is:

$$\det M_\delta = ad - bc = 14 \times 5\pi - 32\pi \times 3 = 70\pi - 96\pi = -26\pi = -2 F_7 \pi$$

where $F_7 = 13$. The Möbius determinant is exactly $-2 \times 13 \times \pi$.

### §5.2 Cross-ratios of the bifurcation cascade

The bifurcation points $r_1, r_2, r_3, \ldots$ of the period-doubling cascade form a Möbius-invariant sequence. Their successive cross-ratios converge to $\sim1.1699$, and their gap ratios converge to $\delta$ (by definition). At 50-digit arithmetic, the cross-ratio limit is $1.16994846869906...$

### §5.3 Base-plus-correction structure

All three constants share the form: **(rational base) + $O(\pi)$ correction**.

| Constant | Rational base | Correction | Error of base alone |
|----------|--------------|------------|-------------------|
| $r_\infty$ | $\pi(55+\sqrt{17})(55+\pi)/55^2$ | $-k\pi^4/55^6$ | 0.00162% |
| $\delta$ | $14/3$ | $(32\pi - 5 \times 14\pi/3)/(10725 + 5\pi)$ | 0.054% |
| $|\alpha|$ | $5/2$ | $\pi/1080$ | 0.116% |

The universal coefficient $a_\text{univ} \approx 55/36 = F_{10}/F_9^{(2)}$ appears in the correction structure at 0.009% error.

**Scripts**: `exp_03_renormalization_analysis.py`, `exp_04_crossratio_mobius.py`

---

## §6. The Möbius Perturbation Series

### §6.1 $r_\infty$ as a Möbius fixed-point perturbation

Define the 10th Fibonacci Möbius transformation:

$$M_{10}(z) = \frac{F_{10} z + F_9}{F_9 z + F_8} = \frac{55z + 34}{34z + 21}$$

This has fixed points at $z = \varphi$ (golden ratio, stable) and $z = -1/\varphi$ (unstable). Their eigenvalues are $\varphi^{-20} \approx 6.6 \times 10^{-5}$ (stable) and $\varphi^{20} \approx 15127$ (unstable).

The accumulation point satisfies:

$$r_\infty = \pi \cdot M_{10}(-1/\varphi + \Delta z)$$

where $\Delta z \approx 5.383 \times 10^{-4}$. The inverse perturbation $1/\Delta z \approx 1857.85$, and $1857 = F_{10} \times F_9 - F_7 = 55 \times 34 - 13$.

### §6.2 Precision hierarchy

The perturbation expands as a series $\Delta z = \sum A_n / n!$, where each term adds approximately 3 digits:

| Level | Correction $C$ | $r_\infty$ approximation | Error | Digits |
|-------|----------------|--------------------------|-------|--------|
| 0 | 0 | 3.5704902 | $5.4 \times 10^{-4}$ | 3 |
| 1 | 4 | 3.5699455 | $1.8 \times 10^{-7}$ | 6 |
| 2 | 3.99868 | 3.5699456711 | $7.7 \times 10^{-10}$ | 9 |

At Level 2, the coefficient ratio $A_3/A_2 = 6050 = 55^2 \times 2 = 2F_{10}^2$ exactly. This suggests a geometric series structure after the first two terms.

### §6.3 Self-consistency: deriving $\delta$ from structure

Using only the Möbius framework (no external δ input), the self-consistency equation recovers:

$$\delta_\text{derived} = 4.669200657... \quad (\text{known: } 4.669201609...)$$

This is 6 digits of $\delta$ derived from the perturbation structure alone — not fitted, but computed from the Möbius geometry of the formula for $r_\infty$.

**Scripts**: `exp_05_high_precision_validation.py`, `exp_06_theoretical_framework.py`

---

## §7. The Self-Closing Formula

A separate route to $\delta$ bypasses the rational fraction entirely.

Define $N = \sqrt{39 + 1/x}$ where $x = 160 + (\delta - 4)^2 \cdot (1 - 1/(1371 + \delta - 4))$. Then:

$$\delta = \varphi^{20/N}$$

This is self-referential: $\delta$ appears on both sides. Starting from any initial guess and iterating, the formula converges in 3 iterations to **13 digits of $\delta$**.

### §7.1 Structural constants

- $39 = (5^4 - 1)/4^2 = 624/16$
- $160 = 4^2 \times 10 = 16 \times 10$
- $1371 = 55 \times 25 - 4 = F_{10} \times 5^2 - F_3$

The exponent 20 connects to the eigenvalue identity at $M_{10}$'s unstable fixed point: $M_{10}'(-1/\varphi) = \varphi^{20}$.

### §7.2 Eigenvalue identity

$$89 - 55\varphi = 1/\varphi^{10} \quad \text{(exact)}$$

This is equivalent to $F_{11} - F_{10}\varphi = (-1)^{10}/\varphi^{10}$, a known identity for Fibonacci Möbius matrices. The eigenvalue $\varphi^{20} = (\varphi^{10})^2$ enters through the determinant of $M_{10}^2$.

**Script**: `exp_07_rbf_self_closing.py`

---

## §8. Universality

The Feigenbaum constants are defined to be universal. But is the Fibonacci structure formula-specific (tied to the logistic map) or genuinely universal?

### §8.1 $\delta$ is universal (definition)

$\delta$ is the same across all one-dimensional maps with a quadratic maximum. The self-closing formula $\delta = \varphi^{20/N}$ therefore applies universally — it is not a property of the logistic map but of the universality class.

### §8.2 $\Delta z$ is universal (measurement)

The perturbation $\Delta z \approx 5.383 \times 10^{-4}$ is **identical** for the logistic and sine maps to within $10^{-10}$. This means the Möbius geometry $r_\infty = \pi \cdot M_{10}(-1/\varphi + \Delta z)$ is not logistic-specific.

### §8.3 Scale ratio

The system-specific accumulation points differ:
- Logistic: $r_\infty = 3.56994...$
- Sine: $a_\infty = 0.89249...$

Their ratio is:

$$r_\infty / a_\infty = 3.99999...\approx 4$$

Because $\pi / (\pi/4) = 4$, the scale factor between maps is purely geometric. The universal quantity is $U = r_\infty / S$ where $S$ is the map's characteristic scale, and $U \approx 1.1363$ for all quadratic-max maps.

### §8.4 Decomposition

$r_\infty = \delta_\text{topology} \times S_\text{geometry}$. The Fibonacci numbers live in $\delta$ (topology); the map-specific scale lives in $S$ (geometry). This explains why the formulas for $\delta$ appear cleaner than for $r_\infty$: the latter mixes universal and system-specific structure.

**Script**: `exp_08_universality.py`

---

## §9. Cross-Domain Validation

Paper 2 in this series shows that $\xi = 1 + \pi/55$ appears as a balance constant across four computational domains. The Feigenbaum formulas use the same structural constant ($F_{10} = 55$) and the same golden ratio ($\varphi$). Are these independent occurrences, or does $\varphi$ enter once and propagate?

### §9.1 Derivation chain

$\varphi$ is derived algebraically from the PAC conservation axiom:

$$f(\text{Parent}) = f(\text{Child}_1) + f(\text{Child}_2), \quad \text{with self-similarity} \implies r^2 = r + 1 \implies r = \varphi$$

This is not fitted. It is the unique positive root of the characteristic equation.

### §9.2 Five-domain test

Using $\varphi$ and Fibonacci numbers derived once, we test predictions across five independent domains:

| Domain | Prediction | Observed | Error | $p$-value |
|--------|-----------|----------|-------|-----------|
| Feigenbaum $\delta$ | 4.6692016091 | 4.6692016091 | $5.7 \times 10^{-13}$ | $\approx 0$ |
| Weak mixing angle $\sin^2\theta_W$ | $3/13 = F_4/F_7$ | 0.23121 (PDG) | 0.19% | 0.0015 |
| SEC prime partition | $1/\varphi$ | 0.613 | 0.82% | 0.050 |
| CA Class IV clustering | $\xi = 1 + \pi/55$ | 1.0566 | 0.047% | $1.1 \times 10^{-7}$ |
| $\Delta z$ universality | $5.383 \times 10^{-4}$ | $5.383 \times 10^{-4}$ | $1.7 \times 10^{-8}$ | $\approx 0$ |

All five domains individually significant at $p < 0.05$.

### §9.3 Joint probability

**Joint $p = 8.3 \times 10^{-12}$. Odds: 1 in 120 billion.**

### §9.4 Circularity check

$\varphi$ is derived once from the conservation axiom. Fibonacci numbers follow from $\varphi$'s minimal polynomial. The five domain predictions are then computed — not fitted per-domain. The derivation chain is: axiom $\to$ $\varphi$ $\to$ Fibonacci $\to$ predictions $\to$ measurements. At no point is a domain-specific value used as input to another domain's formula.

**Script**: `exp_09_cross_domain_validation.py`

---

## §10. Falsification Conditions

These results would be falsified by:

1. **Finding other triples**: If exhaustive search over a larger parameter space ($a, b > 200$) reveals other combinations matching at 7+ digits without Fibonacci structure, the uniqueness claim fails.
2. **Breaking universality**: If $\Delta z$ differs between the logistic and sine maps at higher precision than $10^{-10}$, the Möbius framework is logistic-specific.
3. **Higher-precision failure**: If the Möbius perturbation series diverges rather than converging (each term should add ~3 digits), the framework is a coincidental fit at finite precision.
4. **Formal derivation from RG theory**: If renormalisation group analysis derives the same formulas, that would *explain* the result (not falsify it). If RG analysis produces different formulas with equal or better precision, these formulas would be superseded.
5. **Alternative constant sets**: If $e$, $\sqrt{2}$, or Lucas numbers replace $\pi$ and Fibonacci numbers at comparable precision, the Fibonacci specificity is not meaningful.

The exhaustive search (§4) addresses concern 1 for $a, b \leq 200$. The universality proof (§8) addresses concern 2 to $10^{-10}$. Concerns 3–5 remain open.

---

## §11. What This Does Not Claim

1. We do not claim a *derivation* of the Feigenbaum constants from first principles. These are conjectured closed forms, not theorems.
2. We do not claim that Fibonacci numbers *cause* universality. Correlation is reported; mechanism is absent.
3. We do not claim that $\pi$ and $\varphi$ exhaust the structure. Higher-order terms may require additional constants.
4. The self-closing formula (§7) is a fixed-point equation, not a definition. It does not explain *why* $\delta$ has this value — it reveals the algebraic structure of the value it already has.
5. The connection between Feigenbaum constants and the balance constant $\xi = 1 + \pi/55$ (Paper 2) is noted but not explained.

---

## §12. Summary

We present closed-form expressions for the three Feigenbaum constants using $\pi$, Fibonacci numbers, and small integers:

- $r_\infty$ to 13 significant figures
- $\delta$ to 8 significant figures
- $|\alpha|$ to 6 significant figures

Exhaustive search of 3.9 million parameter triples finds the formula parameters uniquely optimal, with combined odds against coincidence exceeding 1 in 280 billion. The formulas embed in a Möbius perturbation series with geometric coefficient ratios, converge as a self-closing fixed-point equation, and remain invariant across all quadratic-maximum maps.

Five independent computational domains — Feigenbaum bifurcation, weak mixing angle, prime number sieve, cellular automata, and Möbius universality — produce consistent predictions from a single algebraic derivation of $\varphi$, with joint $p < 10^{-11}$.

These are the formulas. Fibonacci numbers appear in the universal constants of nonlinear dynamics, with statistical and structural evidence that this is not coincidence. We do not know why. We report what we found.

---

## §13. Connections to the PACSeries

| Paper | Connection |
|-------|-----------|
| Paper 1: Structure Cost of Erasure | $\xi = 1 + \pi/55$ first derived from Landauer erasure; same $F_{10} = 55$ |
| Paper 2: Balance Constant | $\xi = \gamma + \ln\varphi$; four-domain convergence at $p < 0.004$ |
| **Paper 3 (this paper)** | **Feigenbaum constants from Fibonacci arithmetic** |
| Paper 4: Standard Model | $\sin^2\theta_W = 3/13 = F_4/F_7$ extends the Fibonacci prediction chain |
| Paper 5: Classical Physics | Möbius transformation $M_{10}$ connects to emergence geometry |
| Paper 6: Computational Validation | Self-closing formula testable in PAC-based ML architectures |

The derivation chain established in Papers 1–2 (PAC axiom $\to$ $\varphi$ $\to$ $\ln\varphi$ $\to$ $\xi$) extends here: the same $\varphi$ and $F_{10}$ that structure the balance constant also structure the Feigenbaum constants.

---

## §14. Open Computations

1. **Extend exhaustive search** to $a, b \in [1, 1000]$ — does the uniqueness of (55, 17, 52) survive?
2. **Higher-order Möbius terms** — does $A_4/A_3$ continue the geometric ratio $2F_{10}^2$?
3. **Formal RG connection** — can the Fibonacci Möbius structure be derived from the Cvitanović renormalisation operator?
4. **F₁₀ = T₁₀ role** — is the triangular coincidence (55 is both Fibonacci and triangular) structurally necessary, or is $F_{10}$ sufficient?
5. **$|\alpha|$ refinement** — the current 6-digit formula is the weakest. Can higher Fibonacci numbers improve it?
6. **Cross-ratio limit** — the converged cross-ratio $\approx 1.16995$ has no known closed form. Does it relate to $\varphi$ or $\delta$?

---

*All code, data, and experiment scripts for this paper and the full PACSeries are publicly available at [https://github.com/dawnfield-institute/dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory). See the accompanying publication package README.md for reproduction instructions.*
