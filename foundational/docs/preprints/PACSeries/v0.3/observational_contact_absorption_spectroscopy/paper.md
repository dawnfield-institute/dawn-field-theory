# First Observational Contact: Testing the Cascade Clock Against Quasar Absorption Lines

### A probe of universality — what survived, what died, and where the clock loses its teeth

**Peter Groom, Dawn Field Institute**
**PACSeries Paper 12**
**Date**: June 2026
**Version**: 0.3 (Draft)

---

## Abstract

We make first observational contact between the DFT cascade clock and quasar absorption spectroscopy. The clock $N(z) = 1.360 + (1/\ln\varphi) \cdot \ln(t_\text{lookback})$, previously validated on three cosmological observables (S8, H0, JWST), generates one sharp, DFT-specific prediction for absorption lines: non-monotonic width oscillation at integer cascade levels, with named transition redshifts $z = 0.101, 0.171, 0.302, 0.579, 1.416$. This prediction was pre-registered on GitHub (commit `193c87d5`) before any observational data was examined.

**The prediction is falsified.** Z-detrending kills every oscillatory signal (EW spread: $p = 0.007 \to 0.87$; doublet coupling: $p = 0.006 \to 0.47$; CIV doublet ratio: $p \approx 0 \to 0.68$). The surviving smooth correlation ($b \propto \ln t_\text{lookback}$, $R^2 = 0.85$ over 443,000 CIV systems) is degenerate: $N(z)$ is a monotone reparameterization of lookback time, so any observable that evolves monotonically with cosmic time will correlate with it. A quadratic in $z$ fits better ($R^2 = 0.86$) with one more parameter.

We use the falsification to localize where the cascade clock can and cannot be tested. Interpolation within the well-sampled $z = 1.5$–$4.5$ range cannot discriminate — every smooth function fits. A cross-observable shared-normalization test (CIV $b$, MgII FWHM, Fe/Mg ratio) also fails to discriminate: the observables respond in opposite directions to the same time coordinate, consistent with both cascade redistribution and UV-background evolution. Discriminating power lives at the edges: $z > 5$ (JWST NIRSpec) and $z < 0.5$ (CaII absorbers), where the log form and polynomial mimics diverge.

The paper's genuine contribution is conceptual: DFT predicts photons carry two channels — a conserved channel (PAC: line ratios, invariant) and a historical channel (SEC: line widths/shapes, epoch-dependent). This two-channel structure survives regardless of whether the cascade clock parameterizes the historical channel better than a generic smooth function. The methodology — pre-registration, derivation classification, and systematic self-falsification — is offered as a model for how theoretical frameworks should make observational contact.

**Keywords**: cascade clock, quasar absorption lines, falsification, pre-registration, PAC conservation, Dawn Field Theory

---

## 1. The question and the test

### 1.1 What the cascade clock predicts

The DFT cascade clock (Papers 8–9) derives from the PAC conservation axiom. The temporal function $N(t) = a + (1/\ln\varphi) \cdot \ln(t_\text{lookback})$ was calibrated on three independent cosmological data points:

- $S_8(z = 0.35) = 0.769$ vs $0.768$ observed ($3.22\sigma \to 0.07\sigma$)
- $H_0$ ratio $= \varphi^{1/6}$ vs $1.0838$ observed ($0.076\%$)
- JWST galaxy counts at $z = 8$ and $z = 12$

The question: does this same clock, with no additional parameters, predict the evolution of individual ionic transitions in galaxy-halo gas clouds?

### 1.2 The discriminating prediction

We derived a DFT-specific prediction: at integer cascade levels $N$, the PAC ledger is transitioning (SEC actively redistributing entropy). Line widths should peak. At half-integer $N$, the ledger is settled. Line widths should narrow. This produces a non-monotonic oscillation in line width vs redshift, with specific transition redshifts ($z = 0.101, 0.171, 0.302, 0.579, 1.416$) determined by the clock parameters.

This prediction was registered on GitHub (commit `193c87d5`, pushed June 6, 2026) before any SDSS data was examined.

### 1.3 The degeneracy warning

$N(z)$ is a monotone, smooth function of redshift. Any observable that evolves monotonically with cosmic time — gas velocity, metallicity, ionization balance — will correlate with $N(z)$. $R^2$ against $N$ is therefore not evidence for the cascade clock specifically; it is evidence only that the observable evolves with cosmic time, which is expected on general grounds. Only a **non-monotonic** prediction can discriminate $N(z)$ from generic cosmic evolution. The oscillation was that prediction.

---

## 2. Data

**SDSS DR16 MgII**: 89,291 systems ($z = 0.35$–$2.28$) from MPA-Garching catalog.

**SDSS DR16 FeII-confirmed**: 52,872 systems with 4 transitions (MgII $\lambda\lambda 2796, 2803$ + FeII $\lambda\lambda 2586, 2600$).

**SDSS DR12 CIV**: 443,000 systems ($z = 1.4$–$5.0$) from Monadi et al. (2023), Zenodo DOI: 10.5281/zenodo.7872725.

**XQR-30**: 5,764 absorption components across 42 quasar sightlines ($z = 2.0$–$6.5$), 8 ionic species (Davies et al. 2023).

---

## 3. Result 1: The oscillation prediction is falsified

### 3.1 Initial signals

Our first experiments found cascade-correlated signals in SDSS MgII absorber statistics:

| Signal | Raw $p$ | Interpretation |
|--------|---------|---------------|
| EW spread vs disequilibrium | 0.007 | Wider distributions at cascade transitions |
| Doublet FWHM discrepancy | 0.006 | Lines lock at transitions |
| Population separation (KS) | $10^{-42}$ | Transition vs trough populations differ |

### 3.2 Z-detrending kills them

Every signal collapsed under z-detrending (removing a quadratic fit in $z$):

| Signal | Raw $p$ | Detrended $p$ | Verdict |
|--------|---------|---------------|---------|
| MgII EW spread | 0.007 | 0.87 | z-trend confound |
| Doublet FWHM discrepancy | 0.006 | 0.47 | z-trend confound |
| Inter-line FWHM correlation | 0.055 | 0.47 | z-trend confound |
| CIV doublet ratio | $\approx 0$ | 0.68 | z-trend confound |

The oscillatory component — the DFT-specific prediction — is absent from the data.

### 3.3 What this means

The cascade clock's non-monotonic feature (integer-$N$ transitions) does not imprint on absorption line statistics at the precision available in SDSS data ($\sim 10^5$ systems). This does not falsify the cascade clock itself (it remains validated on S8/H0/JWST). It falsifies the specific prediction that SEC restructuring at cascade boundaries produces detectable width oscillations in absorption lines.

---

## 4. Result 2: The smooth survivor is degenerate

### 4.1 What survived

After removing the oscillatory prediction, the smooth correlation survived: CIV Doppler $b$-parameter correlates with $\ln(t_\text{lookback})$ at $R^2 = 0.85$ (98 bins, 443K systems).

### 4.2 Why it's not evidence for the cascade clock

$N(z)$ is an affine function of $\ln(t_\text{lookback})$. Any linear fit against $N$ spans the identical function space as a linear fit against $\ln(t)$. The $\varphi$-constrained slope is absorbed by the free per-observable amplitude — a free parameter that converts between cascade units and km/s. There is no test of $\varphi$ in this fit.

Against the correct null — any smooth 2-parameter function of lookback time — the cascade clock does not win:

| Model | Parameters | $R^2$ |
|-------|-----------|-------|
| $z$ (linear) | 2 | 0.717 |
| $N(z)$ or $\ln(t)$ (cascade clock) | 2 | 0.851 |
| $z^2$ (quadratic) | 3 | 0.862 |
| $z^3$ (cubic) | 4 | 0.875 |

The cascade clock's $R^2 = 0.851$ is beaten by $z^2$ ($R^2 = 0.862$) with one more parameter. The gap is real curvature that the log-in-lookback-time form misses.

The honest statement: CIV gas velocity increases smoothly with cosmic time, which it must on general astrophysical grounds (UV background hardening, structure formation, evolving gas conditions). The cascade clock is one monotonic parameterization of this evolution. It is not uniquely selected by the data.

### 4.3 The correlations are over bins, not systems

The velocity skewness correlation ($\rho = -0.929$, $p = 0.003$) is computed over 7 binned medians, not 443K independent points. The Fe/Mg correlation ($\rho = -0.949$) is over 19 bins. These are real trends but fragile — a $p$-value over 7 data points should be held lightly.

---

## 5. Conventional explanations for observed trends

### 5.1 CIV velocity evolution

CIV absorbers trace the circumgalactic medium and outflows. Gas velocity increases with redshift due to: (a) stronger UV background driving more energetic outflows, (b) higher star formation rates at cosmic noon, (c) evolving halo gas conditions. These are standard astrophysical processes that produce monotonic $b(z)$ evolution without requiring cascade physics.

### 5.2 Fe/Mg enrichment

The Fe/Mg ratio decreasing with lookback time is textbook galactic chemical evolution. Iron is produced by Type Ia supernovae (delay time $\sim 1$ Gyr); magnesium by core-collapse supernovae (prompt). At earlier cosmic times, less time has elapsed for Type Ia enrichment. Any monotonic clock reproduces this.

### 5.3 The ionization redistribution plane

Eight ionic species in XQR-30 show low-ionization species weakening and high-ionization species strengthening with lookback time. This is the well-documented evolution of the cosmic UV background from soft (low-$z$) to hard (high-$z$). The cascade clock is consistent with this trend but does not predict it more specifically than UV-background hardening does.

---

## 6. What's genuinely new: the two-channel concept

### 6.1 PAC channel (conserved)

The novel claim is not that line ratios are invariant (the Standard Model predicts this too — atomic structure doesn't evolve) nor that $\alpha$ doesn't drift (the SM expectation is a constant $\alpha$; Webb's dipole is contested and mostly unreproduced). The novel claim is the **partition itself**: DFT tells you *a priori* which observables belong to the conserved class and which to the historical class, and the conserved class is invariant to a precision set by topology — exactly integer ADE adjacency, not approximately.

Spectral line ratios are determined by discrete graph topology (ADE classification). The adjacency matrix is integer-valued. It cannot evolve continuously. This topological exactness — conserved to machine precision because the underlying object is a graph with integer entries — is the DFT-specific prediction. The Standard Model also predicts invariant ratios, but without the topological mechanism; DFT predicts *why* they're invariant and *how precisely* (exactly, not approximately).

The derivation of $\alpha_\text{EM} = 2/(3\varphi F_{10}) \cdot (1 - F_{10}/(4\pi F_7^2))$ to 5.7 ppm is Paper 4's result (**[A]/[B]** there, not here). The non-evolution of $\alpha$ is shared with the SM and is not a DFT-discriminating test — confirming it would not favor DFT over the standard expectation.

### 6.2 SEC channel (historical)

DFT predicts that spectral line **widths and shapes** — determined by the SEC entropy state at the absorption site — are epoch-dependent. The absorbing gas's thermodynamic history is written into the line profile. Different epochs have different SEC states, so the profiles evolve.

The two-channel structure — invariant ratios, evolving profiles — is a framework prediction that survives regardless of whether the cascade clock specifically parameterizes the profile evolution better than a generic smooth function.

### 6.3 What the two-channel concept enables

If correct, the two-channel structure means every multi-line absorption system carries more information than standard analysis extracts. Current practice measures metallicity and kinematics. The PAC/SEC framing adds: which information is conserved (line ratios → atomic structure) and which records history (line shapes → entropy state). This distinction could improve absorption-line analysis by separating structural from thermodynamic information, regardless of the cascade clock's validity.

---

## 7. What would discriminate

The oscillation was the discriminating feature, and it died. What's left?

### 7.1 Extrapolation, not interpolation

Within the well-sampled $z = 1.5$–$4.5$ range, every smooth function fits. But $N(z) = a + (1/\ln\varphi) \cdot \ln(t)$ and a polynomial that agree in the middle **diverge at the edges**:

- At $z > 5$ (JWST NIRSpec): the log form flattens while polynomials continue rising/falling. If CIV velocity data at $z = 5$–$7$ follows the log flattening rather than the polynomial extrapolation, that discriminates.
- At $z < 0.5$ (CaII absorbers): the log form has a steeper slope than low-order polynomials. 435 SDSS CaII systems exist in this range.

**Prediction**: the cascade clock and its polynomial mimics diverge outside $z = 1.5$–$4.5$. That is the kill test, and it requires data we don't yet have in sufficient quantity.

### 7.2 Cross-observable shared normalization (performed)

We tested whether the three absorption observables share a common shape when plotted against $N(z)$, with only a free amplitude and offset per observable. If a single universal shape captures CIV $b$, MgII FWHM, and Fe/Mg ratio simultaneously, that would break the degeneracy — a generic polynomial refits per observable and has no reason to produce a shared shape.

**The test fails.** The observables do not share a shape. CIV $b$ increases with $N$ (slope $+92$ km/s per level), MgII FWHM decreases (slope $-2.6$), Fe/Mg decreases (slope $-0.04$). CIV and Fe/Mg are anti-correlated ($r = -0.63$, $p = 0.003$). The three responses to the same time coordinate go in different directions.

The anti-correlation is the ionization redistribution plane (§5.3) in another form: high-ionization strengthens while low-ionization weakens. The shapes are complementary, not shared. This is consistent with PAC conservation (what one gains, the other loses), but it does not discriminate from UV-background hardening, which also produces complementary ionization evolution.

The 99.9th percentile against shuffled controls means the three curves together are more structured than random in $N$-space — but this is just confirming that each observable really evolves with cosmic time, which the degeneracy warning (§1.3) already noted is expected.

**Verdict**: the cross-observable test does not break the degeneracy. **[C]**.

### 7.3 A predicted crossover energy

The ionization redistribution plane has a crossover between AlIII (18.8 eV) and SiIV (33.5 eV). DFT might predict a specific crossover ionization potential from $\varphi$-scaling of the force hierarchy. If a derived number exists there, it would be DFT-specific — UV-background hardening does not predict a specific crossover energy. This derivation has not been attempted.

---

## 8. Honest failure inventory

| # | Prediction | Result | What we learned |
|---|-----------|--------|-----------------|
| 1 | Width oscillation at integer $N$ | **Falsified** (z-detrend kills) | The cascade produces smooth evolution, not oscillation |
| 2 | EW spread correlates with disequilibrium | **Confound** ($p = 0.87$ after detrend) | z-trend dominates single-quantity statistics |
| 3 | Doublet coupling oscillates | **Confound** ($p = 0.47$ after detrend) | Relationship metrics also z-confounded when binned |
| 4 | CIV intra-doublet (KS=0.21) | **Confound** ($p = 0.68$ after detrend) | Large KS from 443K systems ≠ cascade-specific |
| 5 | N-space periodicity | **Not found** (91st percentile) | No cascade-frequency power in detrended data |
| 6 | Sharp excess at transition z | **Not above controls** | Random z-windows show equal or more excess |
| 7 | Single-tree rotation curves | **Failed** | PAC potential converges too fast for flat curves |
| 8 | Cosmic velocity ≈ lab turbulence | **Anti-correlated** ($r = -0.93$) | Different coupling regime entirely |

---

## 9. Classification

Under the PACSeries A/B/C system:

| Finding | Classification | Justification |
|---------|---------------|---------------|
| $\alpha_\text{EM}$ invariance | **[C]** | Shared with SM; non-evolution is the standard expectation; not DFT-discriminating |
| Topological exactness of PAC channel | **[B]** | Novel mechanism (integer ADE adjacency); discriminates from SM's approximate invariance |
| Integer-$N$ oscillation | **[A] falsified** | Derived prediction; killed by data |
| $b \propto \ln(t)$ correlation | **[C]** | Consistent with clock; degenerate with cosmic time |
| Fe/Mg vs $N$ | **[C]** | Standard chemical evolution; any monotonic clock fits |
| Ionization plane | **[C]** | UV-background hardening explains direction |
| Velocity skewness | **[C]** | Over 7 bins; fragile |
| Two-channel concept | **[B]** | Derived + identified with line ratios/profiles |

The paper is mostly **[C]** (pattern-consistent with a monotonic time coordinate) with one **falsified [A]** prediction as its most informative content.

---

## 10. Conclusion

### 10.1 Answering the opening question

Section 1.1 asked: does the cascade clock extend from cosmological structure to individual ion transitions? The honest answer: **absorption spectroscopy at $\sim 10^5$ systems does not have the leverage to test the clock's discriminating feature.** The oscillation prediction — the one thing only DFT predicts — is falsified. The smooth survivor is degenerate with any monotone cosmic-time coordinate. This is a statement about the instrument, not about PAC. The clock remains validated where it has leverage (S8, H0, JWST, the structural ADE results in Papers 4–11). Absorption lines are not a sharp enough probe of its non-generic features.

### 10.2 Three things that are alive

**The failure is the most informative result.** A pre-registered, DFT-specific prediction was tested and killed. The z-detrending methodology that killed it is reusable: any framework claiming cascade-specific oscillations in absorption data must survive this test. The failure localizes where the clock loses its teeth, which is a genuine contribution to understanding its scope.

**The PAC/SEC partition is the most valuable idea.** Photons carry a conserved channel (line ratios, set by discrete topology, invariant to machine precision because the adjacency matrix is integer-valued) and a historical channel (line shapes, set by the thermodynamic state at absorption, epoch-dependent). This partition tells you *a priori* which observables encode structure and which encode history. It survives regardless of the cascade clock's validity, and it could improve absorption-line analysis by separating structural from thermodynamic information.

**Extrapolation is the most promising test.** Within $z = 1.5$–$4.5$, every smooth function fits. Outside that range, the log form and polynomial mimics diverge. JWST NIRSpec data at $z > 5$ and CaII absorbers at $z < 0.5$ are the kill tests — not because the cascade clock is expected to win, but because they're the only place where it can be non-degenerately distinguished from its mimics. If a derived ionization crossover energy can be obtained from $\varphi$-scaling of the force hierarchy, that would provide a second non-degenerate test.

### 10.3 On methodology

Pre-registration, systematic self-falsification, and honest A/B/C classification are how a theoretical framework should make observational contact. This paper demonstrates the method on a case where the contact is uncomfortable. A paper that leads with its failure and classifies most of its results as **[C]** is more credible than one that buries its failures and over-labels its survivals. The PACSeries' credibility depends on grading ourselves harder than our critics would — and this paper is where that discipline was tested most severely.

---

## Data availability

All code, data references, and pre-registration are at `https://github.com/dawnfield-institute/dawn-field-theory`, directory `foundational/experiments/midnight/`. Commit `193c87d5` contains the pre-registered predictions. SDSS catalogs from MPA-Garching and Zenodo. XQR-30 from GitHub.

---

## References

- Anand, A. et al. (2021). SDSS DR16 MgII Absorber Catalog. MNRAS.
- Monadi, R. et al. (2023). CIV Absorption Lines in SDSS DR12. Zenodo.
- Davies, R. et al. (2023). XQR-30 Metal Absorber Catalog. MNRAS 521, 289.
- Groom, P. (2026). PACSeries Papers 1–11. Zenodo, DOI: 10.5281/zenodo.15783623.
- Webb, J.K. et al. (2011). Spatial variation of the fine structure constant. PRL 107, 191101.
