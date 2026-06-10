# First Observational Contact: Testing the Cascade Clock Against Quasar Absorption Lines

### A probe of universality — what survived, what died, and where to look next

**Peter Groom, Dawn Field Institute**
**PACSeries Paper 12**
**Date**: June 2026
**Version**: 0.4 (Draft)

---

## Abstract

We make first observational contact between the DFT cascade clock and quasar absorption spectroscopy. The clock $N(z) = 1.360 + (1/\ln\varphi) \cdot \ln(t_\text{lookback})$, previously validated on three cosmological observables (S8, H0, JWST), generates one sharp, DFT-specific prediction for absorption lines: non-monotonic width oscillation at integer cascade levels, with named transition redshifts $z = 0.101, 0.171, 0.302, 0.579, 1.416$. This prediction was pre-registered on GitHub (commit `193d1c8e`) before any observational data was examined.

The oscillation prediction is falsified. Z-detrending kills every oscillatory signal. The surviving smooth correlation — $b \propto \ln t_\text{lookback}$ at $R^2 = 0.85$ over 443,000 CIV systems — is consistent with the predicted logarithmic form but degenerate with generic cosmic-time evolution ($z^2$ achieves $R^2 = 0.86$ with one more parameter). Absorption spectroscopy in the well-sampled $z = 1.5$–$4.5$ range does not have the leverage to discriminate the cascade clock from its smooth mimics.

The paper contributes three things. First, a pre-registered prediction that was real enough to die — the oscillation's falsification localizes where the clock loses its teeth and establishes a reusable z-detrending methodology. Second, a two-channel concept: DFT predicts photons carry a conserved channel (PAC: line ratios, invariant to machine precision via integer ADE topology) and a historical channel (SEC: line profiles, epoch-dependent). This partition survives regardless of the cascade clock's parameterization and offers a new separation of structural from thermodynamic information in multi-line spectroscopy. Third, a forward program: the cascade clock and its polynomial mimics diverge outside the fitted range, making $z > 5$ (JWST) and $z < 0.5$ (CaII) the discriminating tests.

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

We derived a DFT-specific prediction: at integer cascade levels $N$, the PAC ledger is transitioning. Line widths should peak. At half-integer $N$, the ledger is settled. Line widths should narrow. This produces a non-monotonic oscillation with specific transition redshifts determined by the clock parameters. This prediction was registered on GitHub (commit `193d1c8e`, pushed June 6, 2026) before any SDSS data was examined.

### 1.3 Degeneracy note

$N(z)$ is a monotone function of redshift. Any observable that evolves monotonically with cosmic time will correlate with it. Smooth correlations with $N$ are therefore necessary but not sufficient evidence for the cascade clock. Discriminating power requires either non-monotonic structure (the oscillation) or extrapolation outside the fitted range where the log form and polynomial mimics diverge.

---

## 2. Data

**SDSS DR16 MgII**: 89,291 systems ($z = 0.35$–$2.28$) from MPA-Garching.
**SDSS DR16 FeII-confirmed**: 52,872 systems with 4 transitions.
**SDSS DR12 CIV**: 443,000 systems ($z = 1.4$–$5.0$) from Monadi et al. (2023).
**XQR-30**: 5,764 components, 42 sightlines ($z = 2.0$–$6.5$), 8 ionic species (Davies et al. 2023).

---

## 3. The oscillation prediction: falsified

### 3.1 Initial signals and their collapse

Our first experiments found cascade-correlated signals: EW spread ($p = 0.007$), doublet coupling ($p = 0.006$), population separation ($p = 10^{-42}$). Z-detrending — removing a quadratic fit in $z$ — killed every one:

| Signal | Raw $p$ | Detrended $p$ |
|--------|---------|---------------|
| MgII EW spread | 0.007 | 0.87 |
| Doublet FWHM discrepancy | 0.006 | 0.47 |
| CIV doublet ratio | $\approx 0$ | 0.68 |

The oscillatory component is absent from the data.

### 3.2 What the falsification means

The clock's non-monotonic feature does not imprint on absorption line statistics at this precision. This does not falsify the cascade clock itself — it remains validated on S8/H0/JWST — but it falsifies the specific prediction that SEC restructuring at cascade boundaries produces detectable oscillations. The prediction was real enough to die, which is more than most theoretical predictions achieve.

---

## 4. The smooth survivor

### 4.1 CIV velocity tracks lookback time

The CIV Doppler $b$-parameter correlates with $\ln(t_\text{lookback})$ at $R^2 = 0.85$ (98 bins, 443K systems). The data is consistent with the predicted logarithmic form.

### 4.2 Degeneracy limits

The fit does not test $\varphi$ specifically: $N(z)$ is an affine function of $\ln(t)$, so the free per-observable amplitude absorbs the $\varphi$-determined slope. Against the correct null — smooth functions of lookback time — the cascade clock ties or loses:

| Model | Parameters | $R^2$ |
|-------|-----------|-------|
| $z$ (linear) | 2 | 0.717 |
| $\ln(t)$ / cascade clock | 2 | 0.851 |
| $z^2$ (quadratic) | 3 | 0.862 |
| $z^3$ (cubic) | 4 | 0.875 |

The logarithmic form captures most of the variance — significantly more than linear $z$ — but $z^2$ beats it with one more parameter. The data confirms that CIV velocity evolves smoothly with cosmic time, consistent with the cascade clock but not uniquely selected by it.

### 4.3 Supporting trends

Three additional trends are consistent with smooth cosmic-time evolution:

**Velocity skewness** transitions from symmetric (high $N$, early) to right-skewed (low $N$, late) with $\rho = -0.929$, $p = 0.003$ over 7 bins. This is consistent with the cascade's prediction of higher information-processing rate at earlier epochs, but also with standard expectations: earlier gas is more turbulent (symmetric) and later gas is more structured (skewed). The correlation is over 7 binned medians.

**Fe/Mg ratio** decreases with $N$ ($\rho = -0.949$, $R^2 = 0.89$, 19 bins), matching the nucleosynthesis timescale. This is textbook chemical evolution — less Type Ia iron at earlier times — reproduced by any monotonic clock.

**Ionization redistribution**: 8 ions in XQR-30 show low-ionization species weakening and high-ionization strengthening with $N$ (FeII $p = 0.000$, SiII $p = 0.001$, SiIV $p = 0.011$, CIV $p = 0.000$). This is consistent with UV-background hardening, the standard explanation for cosmic ionization evolution.

All three are **[C]**: pattern-consistent with the cascade clock, also explained by standard astrophysics.

---

## 5. The two-channel concept

### 5.1 The partition

DFT predicts photons carry two channels:

**PAC channel (conserved)**: spectral line ratios, determined by discrete ADE graph topology. The adjacency matrix is integer-valued and cannot evolve continuously. The conserved class is invariant not approximately but exactly — to machine precision — because the underlying object is a graph with integer entries. This topological exactness is DFT-specific: the Standard Model also predicts invariant ratios, but DFT predicts *why* (integer topology) and *how precisely* (exactly). The derivation of $\alpha_\text{EM}$ to 5.7 ppm from Fibonacci structure (Paper 4, **[A]**) is a hard commitment: $\alpha$ cannot drift because $\varphi$ is a fixed point and Fibonacci numbers are integers. Confirming $\alpha$-invariance does not discriminate DFT from the SM (both predict it), but violation *would* falsify DFT — a genuine, asymmetric test.

**SEC channel (historical)**: spectral line widths, shapes, and profiles, determined by the entropy state at the absorption site. Different epochs have different SEC states; the profiles evolve. The cascade clock parameterizes this evolution, though not uniquely (§4.2).

### 5.2 What the partition enables

The novel claim is the partition itself: DFT tells you *a priori* which observables encode structure (conserved, topological) and which record history (evolving, thermodynamic). Current absorption-line analysis measures metallicity and kinematics without distinguishing which information is structural and which is historical. The PAC/SEC partition offers this separation. If correct, every multi-line absorption system carries more extractable information than standard analysis recovers.

---

## 6. What would discriminate

### 6.1 Extrapolation, not interpolation

Within $z = 1.5$–$4.5$, every smooth function fits. Outside that range, $\ln(t)$ and polynomial mimics diverge:

- At $z > 5$: the log form flattens while polynomials extrapolate. JWST NIRSpec data at $z = 5$–$7$ is the kill test.
- At $z < 0.5$: the log form steepens. 435 SDSS CaII systems exist in this range.

**Prediction**: the cascade clock's discriminating power lives at the edges of the current data, not in the middle.

### 6.2 Cross-observable shared normalization (performed)

We tested whether CIV $b$, MgII FWHM, and Fe/Mg ratio share a common shape against $N(z)$. They do not: CIV increases (slope $+92$), MgII decreases ($-2.6$), Fe/Mg decreases ($-0.04$). The anti-correlation ($r = -0.63$ between CIV and Fe/Mg) is the ionization redistribution plane in another form. The shapes are complementary, not shared, and do not break the degeneracy.

### 6.3 A predicted crossover energy

The ionization redistribution crossover lies between AlIII (18.8 eV) and SiIV (33.5 eV). If $\varphi$-scaling of the force hierarchy predicts a specific crossover energy, that would be DFT-specific — UV-background hardening does not predict the crossover location. This derivation has not been attempted.

---

## 7. Failure inventory

| # | Prediction | Result | Lesson |
|---|-----------|--------|--------|
| 1 | Width oscillation at integer $N$ | Falsified | Cascade produces smooth evolution, not oscillation |
| 2–4 | Binned cascade correlations | z-trend confounds | z-detrending is mandatory for cascade claims |
| 5 | N-space periodicity | Not detected | No cascade-frequency power |
| 6 | Sharp excess at transition $z$ | Not above controls | No localized features |
| 7 | Single-tree rotation curves | Failed | Network model needed |
| 8 | Cosmic velocity $\approx$ lab turbulence | Anti-correlated | Different coupling regime |

---

## 8. Classification

| Finding | Class | Justification |
|---------|-------|---------------|
| $\alpha$ invariance | **[A]** | Derived commitment; kills DFT if violated; confirmation doesn't discriminate from SM |
| Integer-$N$ oscillation | **[A] falsified** | Derived prediction; killed by z-detrending |
| Topological exactness of PAC channel | **[B]** | Novel mechanism (integer ADE adjacency) |
| Two-channel partition | **[B]** | Derived + identified with line ratios/profiles |
| $b \propto \ln(t)$ | **[C]** | Consistent; degenerate with cosmic time |
| Fe/Mg vs $N$ | **[C]** | Textbook chemical evolution |
| Ionization plane | **[C]** | UV-background hardening |
| Velocity skewness | **[C]** | 7 bins; fragile |

---

## 9. Conclusion

The cascade clock makes first observational contact with quasar absorption spectroscopy. The contact established three things:

**A real prediction that died.** The oscillation was pre-registered, DFT-specific, and falsified. This is how theory should meet data — with a prediction sharp enough to be killed. The failure scopes the clock's reach: it works at cosmological scales (S8, H0, JWST) but its non-monotonic features do not imprint on absorption line statistics at current precision. This is a statement about the instrument's leverage, not about PAC's validity.

**A partition worth keeping.** Photons carry conserved information (PAC: line ratios, topologically exact) and historical information (SEC: line shapes, epoch-dependent). This partition is DFT-specific — it predicts *which* observables are in which class and *why* (integer topology vs thermodynamic evolution). It survives regardless of the cascade clock's parameterization and could improve multi-line absorption analysis.

**A forward program.** The cascade clock and its smooth mimics agree in the middle ($z = 1.5$–$4.5$) and diverge at the edges. JWST at $z > 5$ and CaII at $z < 0.5$ are where discriminating power lives. A derived ionization crossover energy from $\varphi$-scaling would provide a second non-degenerate test. These are the next experiments.

---

## Data availability

All code, data references, and pre-registration are at `https://github.com/dawnfield-institute/dawn-field-theory`, directory `foundational/experiments/midnight/`. Commit `193d1c8e` contains the pre-registered predictions.

## References

- Anand, A. et al. (2021). SDSS DR16 MgII Absorber Catalog. MNRAS.
- Monadi, R. et al. (2023). CIV Absorption Lines in SDSS DR12. Zenodo.
- Davies, R. et al. (2023). XQR-30 Metal Absorber Catalog. MNRAS 521, 289.
- Groom, P. (2026). PACSeries Papers 1–11. Zenodo, DOI: 10.5281/zenodo.15783623.
- Webb, J.K. et al. (2011). Spatial variation of the fine structure constant. PRL 107, 191101.
