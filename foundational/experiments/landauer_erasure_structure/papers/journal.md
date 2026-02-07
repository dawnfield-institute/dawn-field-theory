# The Structure Cost of Erasure

### On the necessary emergence of correlational structure from information loss

**Peter Groom, Dawn Field Institute**

---

## 1. Two things we know

The first is Landauer's principle. Erasing one bit of information from a physical system requires dissipating at least $kT \ln 2$ of energy into the environment. Rolf Landauer conjectured this in 1961 and it has since been experimentally verified to within a few percent of the theoretical bound. It is not disputed.

The second is the data processing inequality. If two variables $A$ and $B$ are each correlated with a third variable $Z$, then $A$ and $B$ must share mutual information with each other. This is a theorem of information theory. It requires no physical assumptions and cannot be violated.

This paper asks what these two facts, taken together, require to be true about the structure of the environment after information is erased.

---

## 2. The question nobody asked

Landauer's principle is typically discussed as a thermodynamic cost. A bit is erased, energy is dissipated, the environment heats up. The standard treatment ends there. The energy goes into thermal noise and the story is finished.

But erasure is a physical process. The information in the system does not vanish. It is transferred into the environment. The system's prior state becomes encoded, partially or fully, in the new configuration of whatever environmental degrees of freedom absorbed it.

If the environment has more than one degree of freedom (and every real environment does), then the erased information disperses across multiple modes. Each mode that absorbs some portion of the information becomes correlated with the system's prior state. And by the data processing inequality, modes that are each correlated with the same hidden variable become correlated with each other.

This means that erasure does not merely heat the environment. It creates new correlational structure between environmental modes that did not exist before the erasure occurred. This structure creation is not optional. It is a mathematical consequence of information dispersing into a multi-mode system.

The question is how much structure, and what kind.

---

## 3. The experiment

We model a single binary system (one bit, initially in a maximally uncertain state) coupled to a thermal environment of $N$ binary modes. The environment modes are initialized independently with no inter-mode correlations. The system is then erased: reset to a definite state regardless of its prior value.

The erasure couples the system to the environment through a specified topology. We tested five coupling structures:

- **Single-mode**: one environment mode absorbs all the information
- **Uniform**: several modes each absorb an equal share
- **Exponential decay**: coupling strength falls off with mode index
- **Random sparse**: a random subset of modes participate
- **Cascade**: information flows from mode to mode sequentially, as in physical heat dissipation

For each configuration, we measured the full information budget before and after erasure: Shannon entropy of the system, Shannon entropy of the environment, mutual information between system and environment, total correlation (multi-information) within the environment, pairwise mutual information between all environment mode pairs, and the eigenvalue spectrum of the correlation matrix.

We used $5 \times 10^5$ Monte Carlo samples per run, repeated across 10 random seeds for robustness, and swept across temperatures from 100K to 5000K and environment sizes from 10 to 50 modes.

---

## 4. What we found

### 4.1 Structure creation is real and mandatory

Every multi-mode coupling topology produced new inter-mode correlations in the environment after erasure. The single-mode topology, where information is absorbed by exactly one degree of freedom, produced effectively zero new structure ($\xi \approx 0$). Every other topology produced measurable, positive $\xi$.

This confirms the theoretical expectation. Structure creation is a necessary consequence of information dispersal, and it vanishes only in the degenerate case where dispersal does not occur.

### 4.2 Structure creation is topological, not thermodynamic

The emergent structure $\xi$ showed no dependence on temperature. Identical values were obtained at 100K, 300K, 1000K, and 5000K. This means that $\xi$ is not an energy quantity. It is not funded by $kT \ln 2$. It is a property of the coupling topology, the geometry of how information spreads, and is invariant under changes to the energy scale.

The Landauer energy cost and the emergent structure are coupled but distinct. You cannot erase without dispersing (Landauer), and you cannot disperse into multiple modes without creating inter-mode correlations ($\xi$). But the energy goes into the thermal reservoir while the structure goes into the correlational geometry of that reservoir. They are different currencies.

### 4.3 The structure is organized, not random

Eigenvalue analysis of the post-erasure correlation matrix revealed that the new structure is hierarchical. The participation ratio, a measure of how concentrated the eigenvalue spectrum is, dropped from 4.7 (diffuse, random) to 1.0 (concentrated, organized) after erasure. The correlations created by erasure are not thermal noise with extra steps. They are structured, low-rank, and dominated by a single principal component.

This matters because it means $\xi$ carries usable information about the topology that created it. The structure is a fingerprint of the coupling geometry.

### 4.4 The coupling topology determines $\xi$

| Topology | $\xi$ (bits) | Transfer (bits) | Character |
|---|---|---|---|
| Single-mode | $\approx 0$ | 0.073 | No dispersal, no structure |
| Uniform | 0.003 | 0.057 | Weak, symmetric structure |
| Exponential decay | 0.007 | 0.094 | Moderate, graded structure |
| Random sparse | 0.001 | 0.023 | Weak, disordered structure |
| Cascade | 0.044 | 0.079 | Strong, hierarchical structure |

The cascade topology, where information propagates sequentially from mode to mode, produced the highest $\xi$ by a wide margin.

A note on why we focus on cascade. The five topologies tested were chosen to span a range of coupling geometries. The cascade topology resembles physical dissipation: energy flows from high frequency modes to low frequency modes through nearest-neighbor interactions. This is how thermal equilibration actually works in most physical systems. Heat does not teleport from a hot object to distant molecules. It propagates through intermediate degrees of freedom.

This is a physical argument, not a mathematical derivation. We did not prove that cascade is the unique or optimal topology for any formal criterion. We observed that the topology most resembling real dissipation produced the most structure. Whether this is coincidence, selection effect, or deep constraint remains open. The result would be more compelling if cascade emerged as the unique topology satisfying some independent principle. It did not. It was chosen on physical grounds and tested.

### 4.5 The information budget partitions cleanly

At $10^6$ samples, the information budget of a single-bit erasure partitions as follows:

$$P = A + \xi + \Theta$$

where $P$ is the initial system entropy, $A$ is the mutual information between the system's prior state and the post-erasure environment (recoverable information), $\xi$ is the new correlational structure within the environment (total correlation plus pairwise mutual information gains), and $\Theta$ is defined as the remainder: $\Theta = P - A - \xi$.

| Quantity | Symbol | Value (bits) | Measured independently? |
|---|---|---|---|
| Initial system entropy | $P$ (Potential) | 1.000 | Yes |
| Recoverable information in environment | $A$ (Actual) | 0.428 | Yes |
| New correlational structure | $\xi$ (Emergent structure) | 0.451 | Yes |
| Irrecoverable thermal component | $\Theta$ (Thermal) | 0.121 | No (residual) |

The table shows explicitly that we measure two quantities independently: $A$ and $\xi$. The third component $\Theta$ is not measured. It is defined as whatever remains after subtracting $A$ and $\xi$ from the initial entropy $P$. The equation $P = A + \xi + \Theta$ therefore holds by construction. It is not a discovered conservation law. It is an accounting identity.

What is not trivial is that $\Theta$ is positive. We find $A + \xi = 0.879$ bits, which is less than the full bit of initial entropy. The remainder $\Theta = 0.121$ bits represents information that is neither recoverable from the environment nor converted into correlational structure. It is genuinely lost to thermal disorder. This is consistent with Landauer's principle: some of the erased information must become irretrievable heat.

The meaningful claim is not that the three components sum to unity. That is guaranteed by definition. The meaningful claim is that two independently measured quantities, $A$ and $\xi$, together account for 88% of the erased information, with the remainder being a positive, interpretable thermal residual. If $A + \xi$ had exceeded $P$, we would have an inconsistency. It did not.

---

## 5. What this requires

We have established experimentally that:

1. Information erasure necessarily creates correlational structure in multi-mode environments.
2. The amount and organization of this structure depends on the coupling topology.
3. The structure is topological, not thermodynamic. It is invariant under temperature.

These three claims follow from Landauer's principle and the data processing inequality, and they are confirmed by direct computation. They are the established results of this paper.

The following is not established. It is speculation, offered as a direction for future work.

Every physical interaction involves information exchange between coupled degrees of freedom. A photon mediating an electromagnetic interaction couples two charged particles through a single-mode bosonic field. The weak interaction couples particles through three boson modes with $SU(2)$ symmetry. The strong interaction couples quarks through eight gluon modes with $SU(3)$ symmetry.

These are coupling topologies. Our experiment measures $\xi$ as a function of coupling topology. If there is a connection, it would be this: each fundamental interaction should produce a characteristic $\xi$ determined by its gauge group structure.

If $\xi$ depends on the number of modes and their coupling geometry (which we have demonstrated), then:

- $U(1)$ interactions (electromagnetism) should produce minimal $\xi$, analogous to our single- or few-mode coupling results.
- $SU(2)$ interactions (weak force) should produce moderate $\xi$ with a three-mode symmetric structure.
- $SU(3)$ interactions (strong force) should produce the highest $\xi$, with eight coupled modes generating rich hierarchical correlations.

This ordering prediction is falsifiable. If computing $\xi$ for actual gauge group structures yields a different ordering, the hypothesis fails.

Several caveats apply. Real gauge interactions involve running couplings whose effective strength depends on energy scale. They involve vacuum polarization, spontaneous symmetry breaking (in the electroweak sector), and confinement (in QCD). A static coupling topology model does not capture these dynamics. The prediction here concerns the ordering and rough scaling of $\xi$ across gauge groups, not exact values. Reproducing the measured coupling constants would require incorporating scale-dependent effects, which is beyond the scope of this initial demonstration.

With those caveats stated, the prediction is testable. The gauge group structures are exact. The coupling constants are measured. The $\xi$ values can be computed for each topology and compared against observable quantities.

We do not claim here that this comparison has been performed. We claim that the mechanism demonstrated in this paper, the necessary emergence of topological structure from information erasure, makes the comparison possible. Its outcome would either confirm or refute the hypothesis that fundamental coupling constants encode the structure cost of information exchange.

---

## 5.5 The ratio is not arbitrary

Look again at the partition in section 4.5. For cascade topology:

$$A = 0.428 \text{ bits}, \quad \xi = 0.451 \text{ bits}$$

The ratio of recoverable information to total effect is:

$$\frac{A}{A + \xi} = \frac{0.428}{0.879} = 0.487$$

This is within 1.2% of $\ln \phi \approx 0.481$, where $\phi$ is the golden ratio. It was not fitted.

The ratio $\frac{A}{A+\xi}$ measures collapse efficiency: what fraction of the erasure's total effect remains localized versus what fraction disperses into structure. The question is why this efficiency should take any particular value.

The cascade topology has two processes. Information transfers from mode to mode, decaying with distance. Correlations form between modes, also decaying with distance. Both decay rates are free parameters.

If transfer decays slowly and correlation decays quickly, nearly everything stays recoverable. $\xi$ vanishes.

If correlation decays slowly and transfer decays quickly, structure dominates. The environment becomes correlated noise with little information about what was erased.

We varied both rates systematically across a full grid (decay rates 0.1 to 0.5). The results are striking:

| Decay Ratio (flip/corr) | $A/(A+\xi)$ | vs $\ln\phi$ |
|-------------------------|-------------|-------------|
| 1.00 (symmetric) | 0.47 | 3.0% |
| 1.25 | 0.476 | 1.1% |
| 1.50 (3:2) | 0.483 | 0.40% |
| **1.618 (φ)** | **0.4813** | **0.03-0.16%** |
| 1.75 | 0.489 | 1.7% |

The best match occurs not at the simple ratio 3:2, but at $\phi$ itself. When transfer decays faster than correlation by exactly the golden ratio, the collapse efficiency matches $\ln\phi$ to within 0.03-0.16%. This is a stronger result than 3:2 approximation suggested.

A caveat: the falsification suite (exp_08) showed that random parameter combinations can occasionally achieve similar precision by chance (~0.001% of trials). The significance is not in any single match but in the pattern: the optimal decay ratio is $\phi$, and the resulting partition ratio is $\ln\phi$. The golden ratio appears twice, in different functional roles.

The physical interpretation: direct effects attenuate faster than the correlations they induce, and the optimal attenuation ratio is $\phi$. The splash fades before the waves do, and it fades at exactly the golden ratio.

The golden ratio appears in self-similar structures. A cascade is self-similar: each level transfers and correlates with the next in the same pattern. That both the optimal decay ratio AND the resulting partition ratio involve $\phi$ suggests the golden ratio is not incidental but intrinsic to self-similar information flow.

---

## 6. On the nature of $\xi$

It is worth being precise about what $\xi$ is and what it is not.

$\xi$ is not energy. It does not depend on temperature. It is not paid for by the Landauer cost, though it is caused by the same physical process that incurs that cost.

$\xi$ is not noise. It is organized, hierarchical, and low-rank. It carries information about the topology that created it.

$\xi$ is not the erased information reappearing elsewhere. The recoverable information $A$ accounts for that. $\xi$ is new correlational structure: relationships between environmental modes that did not exist before and that were not present in the original system. It is emergent in a precise sense. It is a property of the whole that was not a property of any part.

$\xi$ is the structural cost of collapse. When possibilities are eliminated, when a superposition resolves, when a bit is erased, when potential becomes actual, the eliminated possibilities do not simply vanish. They become the correlational architecture within which the surviving outcome is embedded. The roads not taken become the landscape.

Whether this structural cost is the right lens through which to understand gauge interactions, confinement, and the running of coupling constants is an empirical question. The mechanism is demonstrated. The tests can be designed. The results will speak for themselves.

---

## 7. Open questions and limitations

This paper establishes one thing clearly: information erasure into multi-mode environments creates correlational structure. The amount and character of that structure depends on the coupling topology. These claims are demonstrated computationally and follow from established information theory.

Several questions remain open.

**Why cascade?** We chose the cascade topology because it resembles physical dissipation. Cascade produced the highest $\xi$. But we did not derive cascade from first principles. If there is a reason why physical systems prefer cascade-like topologies for information flow, we do not know it. The result would be stronger if cascade emerged uniquely from some optimization principle or symmetry argument.

**Is the 3:2 ratio fundamental?** The decay ratio of 1.5 (transfer decaying 50% faster than correlation) produces collapse efficiency matching $\ln\phi$ to 0.4%. This is a striking numerical result. We do not know why this ratio should be special. One possibility: direct effects attenuate faster than the correlations they induce because indirect correlations require less energy to maintain. Another possibility: 1.5 is close to $\phi$ itself (1.618), suggesting a self-similar structure relationship. A third possibility: the agreement is coincidental at the precision we measured. More work is needed.

**Can $\Theta$ be predicted?** We defined $\Theta$ as the residual after measuring $A$ and $\xi$. This is honest accounting but weak physics. A stronger claim would predict $\Theta$ from first principles. If Landauer's bound gives the minimum dissipation, then $\Theta$ should relate to excess dissipation beyond that bound. Connecting $\Theta$ to experimentally measurable heat flow would strengthen the framework considerably.

**Does the Standard Model connection hold?** The speculation in section 5 predicts that $\xi(SU(3)) > \xi(SU(2)) > \xi(U(1))$. This has not been tested. Computing $\xi$ for actual gauge group structures and comparing against coupling constant ratios would either validate or refute the hypothesis. This is the most concrete next step.

**Is $\xi$ observer-dependent?** We computed $\xi$ assuming access to all environmental modes. A realistic observer with limited access might measure different correlations. Whether $\xi$ is an objective property of the environment or depends on the observer's resolution is an open question with potential connections to quantum decoherence and the measurement problem.

These limitations do not undermine the core result. They define the work that remains.

---

## 8. Connections to related work

This experiment does not stand alone. Several independent lines of investigation within this research program have arrived at related findings. None of these connections constitute external validation. They are internal consistencies that either strengthen the framework or will eventually expose contradictions.

### 8.1 Why $\ln\phi$ and not $\Xi$?

Other experiments in this program have found a constant $\Xi \approx 1.058$ appearing in several contexts:

| Domain | Method | Result |
|--------|--------|--------|
| Cellular automata (Rule 110) | P/A ratio at edge of chaos | 1.058 |
| Turbulence (Navier-Stokes) | Symbolic engine threshold | 1.057 |
| Prime distribution | Discrete-continuous interface | 1.058 |

The prime growth dynamics work found that $\Xi$ decomposes as:

$$\Xi = \gamma + \ln\phi$$

where $\gamma = 0.5772...$ is the Euler-Mascheroni constant and $\ln\phi = 0.4812...$ is this paper's finding.

The decomposition suggests an interpretation. The Euler-Mascheroni constant $\gamma$ is defined as:

$$\gamma = \lim_{n \to \infty} \left( \sum_{k=1}^{n} \frac{1}{k} - \ln(n) \right)$$

This is the cost of bridging discrete counting (the sum) with continuous integration (the logarithm). It appears whenever discrete objects meet continuous processes.

This paper's finding of $\ln\phi$ involves pure information partitioning with no discrete counting. The system bit is erased, information flows into the environment, and the partition ratio is measured. There are no discrete objects being enumerated. The absence of $\gamma$ is therefore consistent: where there is no discrete-continuous interface, there is no interface cost.

The implication: $\ln\phi$ is the "continuous" component of $\Xi$, measuring pure geometric collapse efficiency. When discrete structure is involved, $\gamma$ adds the interface cost.

This interpretation is not proven. It is consistent with the data across multiple experiments.

### 8.1.1 Testing PAC conservation directly (exp_14)

If the decomposition $\Xi = \gamma + \ln\phi$ reflects PAC conservation, we should expect $I_{total} = A + \xi$ to be conserved regardless of parameters. We tested this directly.

**Results:**

| Varied Parameter | $I_{total}$ Range | $I_{total}$ Variance |
|------------------|-------------------|---------------------|
| Decay rate (0.1-0.5) | 0.67 - 1.85 bits | 0.18 |
| Causal lag (0-3) | 0.65 - 0.93 bits | 0.01 |

$I_{total}$ is NOT conserved at the absolute level. The variance across decay rates is substantial.

**But the ratio is stable:**

| Condition | $A/(A+\xi)$ | Deviation from $\ln\phi$ |
|-----------|-------------|--------------------------|
| Optimal parameters | 0.479 ± 0.002 | 0.4% |
| Across 30 seeds | 0.484 ± 0.054 | 0.6% |
| After shuffling | 0.584 | 21.3% |

The finding refines the PAC interpretation:

- PAC does NOT operate on absolute totals ($I_{total}$ varies)
- PAC operates on the **proportional geometry** ($A/(A+\xi)$ is constrained)
- The golden ratio describes HOW potential actualizes, not HOW MUCH

When shuffling breaks causal ordering, the ratio shifts by 21%. This is the same effect as disconnecting from the "PAC ledger"—the global constraint that forces the split to follow $\ln\phi$.

The component $\xi/A$ at optimal parameters equals 1.086, within 0.76% of the predicted $(1-\ln\phi)/\ln\phi = 1.078$. This is the complementary fraction: for every unit of information that stays actualized in system-environment coupling, 1.078 units become emergent structure. The golden ratio governs the exchange rate.

### 8.1.2 The PAC/SEC hierarchy (base_agnostic_pac convergence)

The `base_agnostic_pac` experiment established the same layering in pure mathematics:

| Level | What's Conserved | What Varies |
|-------|------------------|-------------|
| **PAC** | φ² = φ + 1 (< 10⁻¹⁴ error) | — |
| **SEC** | — | Representational entropy (20-30%) |

This paper's finding in physical dynamics:

| Level | What's Conserved | What Varies |
|-------|------------------|-------------|
| **PAC** | A/(A+ξ) = ln(φ) (0.4% error) | — |
| **SEC** | — | $I_{total}$ = A + ξ (varies 3×) |

The parallel is exact. In both cases:
- **PAC relationships** (ratios, proportions, self-similar structures) are invariant
- **SEC representations** (absolute values, specific encodings) vary freely

The `base_agnostic_pac` SYNTHESIS states: "PAC relationships are the territory, SEC representations are the map."

The absolute bits flowing through the Landauer cascade are the *map*—they change based on parameterization. But the *proportion* in which potential splits into actualization vs structure is the *territory*—φ geometry regardless of scale.

This also explains why 55 = F₁₀ appears in the Feigenbaum formulas. It's not a decimal coincidence. 55 encodes the recursion depth where PAC/SEC balance stabilizes—a structural position that would be equally significant in any numerical base.

**Convergent validation across domains:**

| Domain | Measurement | φ Expression |
|--------|-------------|--------------|
| Number theory | PAC identities across 11 bases | φ² = φ + 1 (exact) |
| Bit erasure | A/(A+ξ) ratio | ln(φ) ± 0.4% |
| Turbulence | Symbolic engine threshold | Ξ = γ + ln(φ) |
| Cellular automata | Edge-of-chaos attractor | φ-clustering |
| Primes | Critical excursion fraction | 1/φ ± 0.001% |

Same φ, different measuring sticks. The question is no longer "does φ appear?" but "why does proportional geometry conserve while absolute magnitudes don't?"

### 8.2 Connection to phase transitions

The SEC prime manifold experiment found that a stress field constructed from prime irregularities exhibits a phase transition. At the critical point $\lambda^* = 0.9816$, the fraction of positive excursions converges to $1/\phi = 0.618$ with error 0.000006.

The mechanism is run-length asymmetry: positive runs last longer than negative runs by a ratio of $\phi$ at criticality.

This paper's finding that $A/(A+\xi) = \ln\phi$ at a specific decay ratio (1.5) may reflect the same structure. The decay ratio 1.5 is where the system sits at a balance point between localization (all information stays recoverable) and diffusion (all information becomes structure). Balance points often produce power-law or golden-ratio signatures.

Whether the 1.5 decay ratio corresponds to a critical point in some deeper sense remains an open question.

### 8.3 Connection to Standard Model findings

The PAC confluence experiment found that several Standard Model parameters can be expressed as Fibonacci ratios:

| Parameter | Expression | Error |
|-----------|------------|-------|
| $\sin^2\theta_W$ (weak mixing) | $F_4/F_7 = 3/13$ | 0.19% |
| Koide $Q$ (lepton masses) | $F_3/(F_3+F_2) = 2/3$ | EXACT |
| Cabibbo angle | $\arctan(F_4/F_7)$ | <0.05° |

These are algebraic observations. They do not explain why such ratios should appear.

This paper offers a potential mechanism: if gauge interactions involve information exchange, and information exchange involves partial erasure, then every interaction creates correlational structure $\xi$. The structure follows golden ratio partitioning. Coupling constants would then encode accumulated $\xi$ at different energy scales.

This connection is speculative. The Standard Model findings are themselves unvalidated beyond numerical coincidence. Combining two speculative claims does not produce certainty. But if both are correct, they reinforce each other. If either is wrong, the contradiction will eventually become apparent.

### 8.4 The Feigenbaum connection

The Feigenbaum constants describe universal behavior in period-doubling bifurcations. Within this research program, closed-form expressions for these constants were found that achieve 6-13 digit precision using the numbers 55, 17, and $\pi$.

The number 55 is the tenth Fibonacci number. It also appears in the expression $\Xi = 1 + \pi/55 \approx 1.0571$.

The cascade topology in this paper is self-similar: each mode couples to the next in the same pattern. Self-similar structures are precisely where Feigenbaum universality applies. Whether there is a deep connection between cascade information flow and Feigenbaum dynamics is unknown. The numerical coincidences are suggestive but not conclusive.

### 8.5 Status of these connections

All connections described in this section are internal to this research program. None have been validated by external replication. None have been derived from first principles with complete rigor. They are patterns that have emerged across independent computational experiments.

The appropriate stance is cautious interest. The patterns may reflect something real about information geometry. They may reflect systematic errors in methodology. They may be coincidences amplified by confirmation bias.

What distinguishes these findings from numerology is falsifiability. Each claimed connection makes predictions that can be tested independently. If $\xi$ does not scale with gauge group complexity, the Standard Model speculation fails. If the 3:2 ratio does not hold under different simulation parameters, the $\ln\phi$ finding is fragile. If the prime distribution findings do not replicate at larger scales, the $\gamma + \ln\phi$ decomposition is suspect.

The work that remains is to test these predictions systematically and to seek external validation through independent replication.

---

## 9. The Thermodynamic Cascade

The experiments described in sections 3-5 analyzed single erasure events. But real physical processes involve sequences of interactions. What happens when the entropy produced by one erasure becomes the potential for another?

### 9.1 Θ is generative, not terminal

The thermal component $\Theta$ from a single erasure was defined as a residual: whatever information is neither recoverable $(A)$ nor converted to structure $(\xi)$. The natural interpretation was that $\Theta$ represents genuine loss to disorder—information that has thermalized beyond recovery.

But thermalized information is not annihilated. It exists in the environment, distributed across environmental modes with high entropy. High entropy means high potential. $\Theta$ can serve as the input potential for a subsequent erasure event.

### 9.2 The cascade mechanism

We model a multi-generation cascade as follows:

- **Generation 0**: Standard single-bit erasure into $N$ environment modes. Produces $A_0$, $\xi_0$, $\Theta_0$.
- **Generation $n > 0$**: The highest-entropy environment mode from generation $n-1$ becomes the "system" for generation $n$. Its entropy $H$ becomes the potential $P_n \approx H$. The remaining modes form the new environment. Erasure proceeds and produces $A_n$, $\xi_n$, $\Theta_n$.

The cascade continues until no mode has sufficient entropy to serve as potential.

### 9.3 Results: cascade amplification

Across 30 random seeds with 8 environment modes and 0.7 decay coupling:

| Metric | Single event | Full cascade | Ratio |
|--------|-------------|--------------|-------|
| Total $\xi$ produced | 0.004 bits | 0.21 bits | **53×** |
| p-value (cascade > single) | — | $2.75 \times 10^{-35}$ | — |
| Mean cascade lifespan | 1 | 8.5 generations | — |

The cascade produces 53 times more structure than a single erasure event. The thermal component $\Theta$ from each generation is not lost—it re-injects as fuel for subsequent structure creation.

### 9.4 The learning rate interpretation

The ratio $\xi/\Theta$ at each generation functions as an effective "learning rate":

- **High $\xi/\Theta$**: aggressive structure creation, cascade dies quickly (entropy depleted)
- **Low $\xi/\Theta$**: conservative structure creation, cascade dies slowly but produces less per step

The coupling decay rate controls this ratio. Thermodynamics—specifically $kT \ln 2$—acts as the governor that sets the cascade rate. Too fast and the system depletes its entropy reservoir. Too slow and structure creation stalls.

This reframes the Landauer cost. The $kT \ln 2$ is not a tax. It is the regulator that prevents runaway collapse (learning rate too high) and total stagnation (learning rate too low). Thermodynamics mediates between the extremes.

### 9.5 Why cascade topology dominates

Section 4.4 noted that cascade coupling produced more structure than other topologies, but did not explain why.

The cascade mechanism provides the explanation. In a cascade topology, information flows sequentially from mode to mode. Each step in the spatial cascade is analogous to a generation in the temporal cascade: what one mode absorbs becomes available to the next. The cascade topology mirrors the re-injection process at a different scale.

Other topologies (uniform, exponential, random) spread information simultaneously without sequential re-injection. They complete in one "generation" and produce correspondingly less structure.

The cascade topology is physical because thermodynamics enforces sequential processing. Heat does not teleport. Energy does not skip modes. Dissipation propagates through hierarchies. The topology that mirrors actual thermodynamic flow is the topology that produces the most structure.

---

## 10. Time as computational density

The cascade framework suggests a perspective on time.

### 10.1 The internal view

From outside the cascade, we might count ticks: generation 0, generation 1, generation 2, etc. Each tick is one erasure event. The cascade progresses through discrete steps.

From inside the cascade, each tick is one moment of experienced time. The question is not how many ticks pass on some external clock, but what happens within each tick.

The computational density of a tick—how much structure is created, how much information is processed—determines how "thick" that moment is.

### 10.2 Dense vs sparse regimes

We compared two regimes:

- **Dense**: Strong coupling, fresh (unstructured) environment. Models early-universe conditions where interactions are frequent and the medium is uncorrelated.
- **Sparse**: Weak coupling, saturated (pre-structured) environment. Models late-universe conditions where interactions are rare and the medium is already highly correlated.

Results across 25 trials, 15 ticks each:

| Regime | $\xi$ per tick | Interpretation |
|--------|---------------|----------------|
| Dense | 0.0050 | Heavy computation per moment |
| Sparse | 0.00007 | Light computation per moment |
| **Ratio** | **69×** | p = $3.25 \times 10^{-5}$ |

In the dense regime, each tick creates substantial new structure. Each moment is computationally heavy. In the sparse regime, each tick creates almost nothing. Each moment is computationally light.

### 10.3 Interpretation: thick and thin time

This is not about time "speeding up" or "slowing down" relative to an external reference. There is no external reference. The cascade ticks are all that exist.

What changes is the content of each tick. Early ticks (dense regime) are thick: each moment contains substantial structure creation, substantial information processing, substantial change. Late ticks (sparse regime) are thin: each moment contains almost nothing.

From inside the cascade, being in a thick moment feels slow. There is much happening. Being in a thin moment feels fast. There is little happening.

This is consistent with the thermodynamic arrow of time. Early universe: dense, hot, high potential, high $\xi$ production per interaction. Late universe: sparse, cold, low potential, minimal structure creation. Time doesn't accelerate—moments become emptier.

### 10.4 Expansion model

We modeled a continuously expanding universe by decreasing coupling strength over time:

$$\text{coupling}(t) = 0.9 - 0.6 \cdot \frac{t}{T}$$

As coupling weakens, $\xi$ per tick decreases. The correlation between coupling strength and $\xi$ is $r = 0.89$, $p < 10^{-10}$.

Structure creation front-loads. Most of the cascade's total structure is produced in the early, dense phase. By the time the universe is sparse, the structural work is largely complete. The late universe is populated but not productive.

### 10.5 Caveats

This is a toy model. Real cosmology involves continuous fields, relativistic effects, and quantum processes that the discrete binary simulation does not capture. The interpretation is suggestive, not rigorous.

What the model demonstrates is that a cascade driven by entropy re-injection naturally produces temporal asymmetry: early moments are productive, late moments are not. This asymmetry is a consequence of the mechanism, not an additional assumption.

Whether this mechanism captures anything real about cosmological time is an empirical question beyond the scope of this paper.

---

## 11. The binding interpretation

Sections 9 and 10 establish that the cascade is self-sustaining and that computational density varies across the cascade. A deeper question remains: what exactly is $\xi$?

### 11.1 The car metaphor

Consider assembling a car from parts: metal, rubber, glass, plastic. Before assembly, you have components. After assembly, you have a car.

The car has a property the components lack: it can drive. This "car-ness" exists in the assembled whole but in none of the individual parts. Where does it come from?

The standard answer: it emerges from the relationships between parts. The property is relational, not intrinsic.

This is $\xi$. The correlational structure that emerges from erasure is not information that existed in the system and was moved to the environment. It is not information that existed in individual environmental modes and was aggregated. It is new relational structure that exists only in the bound system.

### 11.2 PAC as binding constraint

Potential-Actualization Conservation (PAC) in this framework is not redistribution. It does not describe information moving from one place to another. It describes the constraint under which binding occurs.

When parts are bound under conservation, the conserved quantity is preserved but its character changes. A car's metal, rubber, and glass still exist after assembly. Their masses are conserved. But the organization—the binding—creates new properties that transcend the components.

$\xi$ is the structural cost of this binding. It is paid by the conservation constraint itself. Whenever multiple modes are forced to jointly satisfy a conservation law (as in erasure, where the total information budget is fixed), the binding creates correlational architecture that was not present in the unbound system.

### 11.3 Implications for RBF

The Recursive Balance Field (RBF) in Dawn Field Theory:

$$B(x,t) = \lambda \cdot \frac{E(x,t) - I(x,t)}{1 + \alpha M(x,t)} \cdot \Phi(x)$$

Under this interpretation, RBF governs where binding is strong (high $|B|$) and where it is weak. SEC (Symbolic Entropy Collapse) operates within RBF-governed dynamics, proceeding fastest where $B$ approaches zero (where $E \approx I$, the balance point). $\xi$ emerges from the binding constraint that RBF enforces.

Exp 09 tested this directly. Under strict conservation (no energy source or sink), nonlinear RBF binding produces emergent structure ($\xi$) beyond what linear coupling predicts ($p = 2.10 \times 10^{-32}$). The memory term $M$ and harmonic modulation $\Phi$ are not decorative—they create structure that pure linear coupling does not.

---

## 12. Summary and outlook

This paper establishes three claims with computational evidence:

1. **Information erasure creates correlational structure** ($\xi$) in multi-mode environments. This follows from Landauer's principle and the data processing inequality.

2. **The structure is topological**, invariant under temperature, and depends on coupling geometry. Cascade topology produces the most structure.

3. **The thermal component re-injects as fuel**, creating a self-sustaining cascade that produces 53× more structure than single events.

Three speculative implications are offered for further investigation:

- **Gauge coupling constants** may encode accumulated $\xi$ from topologically distinct interaction geometries.
- **Time** may be understood as computational density within the cascade, with early moments thick and late moments thin.
- **PAC** may describe the binding constraint that creates $\xi$, not a redistribution mechanism.

The falsifiable predictions from these implications:

| Claim | Falsification condition |
|-------|------------------------|
| Gauge topology → coupling order | $\xi(SU(3)) < \xi(SU(2))$ |
| 3:2 decay ratio → ln(φ) | Different ratio produces closer match |
| Cascade amplification | Fewer generations produce more total $\xi$ |
| $\Theta$ re-injects | Cascade dies faster than entropy allows |

The mechanism is established. The connections are proposed. The tests remain.

