# Milestone 6: Beyond the Standard Model — Neutrino Masses, Loop Corrections & Cosmological Validation

**Version**: 0.1.0
**Status**: Planning
**Date**: 2026-03-25

---

## The Story

Milestone 5 closed the Standard Model. Every fermion mixing angle falls out of arctan(F_a/F_b). The Higgs mass lands at 83 ppm via lambda = phi/(4*pi). The strong force was already implicit in the cascade-depth tiling filter. And de-actualization — the PAC cycle's missing return leg — cut simulator coupling drift by 24%.

But M5 left three criteria unmet, and those gaps define what comes next.

**First**: the neutrino masses. Mixing angles are derived — all PMNS entries within 0.3 degrees — but absolute masses remain open. The Fibonacci arithmetic that predicts every other SM mass should predict m_nu_1, m_nu_2, m_nu_3. The four-force template from MAR exp_38 (1 ± F_a/(n*pi*F_b^2)) already distinguishes weak-force parameters as flavor=branching, parity=directionality. Neutrino masses sit at the intersection of weak-force Fibonacci structure and cascade-depth ordering. This is unfinished business, not speculation.

**Second**: the simulator scorecard stalled at 8/13 (C grade). De-actualization helped, but phi^2 spacing regressed at 40.6%, and two anti-correlated coupling groups remain locked in a structural trade-off. MAR exp_41-43 are testing emergent attractors — if they converge, the scorecard improves organically. But the phi^2 regression needs diagnosis: which operator causes it, and does it connect to the loop-level question?

**Third**: DFT predicts UV fixed points — couplings don't run. This contradicts QCD's asymptotic freedom. Either cascade dynamics generate effective running at low energies (reconciling DFT with experiment) or the UV-fixed prediction is genuinely new physics. The strong force template has two surviving candidates (n=3 fundamental vs n=8 adjoint from MAR exp_39); resolving this may require understanding what "running" means in a PAC cascade. This is the deepest theoretical question M6 must face.

Beyond these direct successors, M5 opened a door to cosmology. MAR exp_35-36 reduced the cosmological constant problem from 10^123 orders of magnitude to just 10^0.22 — tantalizingly close to zero. The gap is real, not noise, and sub-integer depth corrections (exp_40 suggested depth 182.446) or mixed tiling factors could close it. Meanwhile, the JWST cosmology validation has a documented 6-phase upgrade plan waiting for execution.

And then there's the frontier: dark matter candidates from Fibonacci depth (the depth-73 cyclotomic hierarchy maps to ~15 keV, testable against X-ray line constraints), and the PAC Knowledge Discovery tool — the first applied-DFT publication, demonstrating the framework has utility beyond fundamental physics.

Milestone 6 is where DFT stops being a theory of the Standard Model and starts being a theory of everything it claims to be.

---

## Success Criteria

| # | Criterion | Block | Measurable Target |
|---|-----------|-------|-------------------|
| 1 | Neutrino absolute masses derived | A | m_nu_1, m_nu_2, m_nu_3 within current experimental bounds (sum < 0.12 eV from Planck) |
| 2 | Mass hierarchy predicted | A | Normal or inverted ordering derived from cascade depth, not assumed |
| 3 | Dark matter candidate mass | A | At least one testable DM mass prediction with observational constraints cited |
| 4 | Loop-level resolution | B | Either derive effective running from cascade dynamics OR prove UV-fixed prediction with falsifiable consequence |
| 5 | Strong force template unique | B | n=3 vs n=8 resolved; single Fibonacci formula for alpha_s |
| 6 | CC gap < 0.1 orders | C | Improve from 10^0.22 to < 10^0.1 via derived (not fitted) correction |
| 7 | JWST validation on 50+ objects | C | AIC/BIC comparison against LCDM baseline; statistical significance reported |
| 8 | Simulator scorecard >= 11/13 | D | B- grade or better on physics scorecard with all coupling errors < 5% |
| 9 | PAC Discovery benchmarked | D | Outperform PC algorithm and NOTEARS on at least 2 of 3 benchmark datasets |
| 10 | Honest uncertainty budget | All | Every result tagged proven/suggestive/speculative with explicit falsification conditions |

---

## Block A: Neutrino Masses & BSM Physics

**Threads**: 1 (Neutrino Masses from Fibonacci Arithmetic) + 7 (Dark Matter from Fibonacci Depth)
**Goal**: Derive absolute neutrino masses; predict dark matter candidate mass
**Prior work**: M5 exp_08 (PMNS angles), MAR exp_38 (four-force template), gravity exp_03-04 (depth-73 cyclotomic)

The correction template from MAR exp_38 is the key: each force's mass spectrum follows 1 ± F_a/(n*pi*F_b^2) with force-specific (a, b, n) values. The weak force was characterized as flavor=branching, parity=directionality. Neutrino masses should be the lightest entries in the weak-force correction series, with the mass hierarchy encoded in cascade depth ordering.

### Experiments

| # | Name | Question | Approach |
|---|------|----------|----------|
| exp_01 | Weak template extraction | What are the weak-force (a, b, n) values in the correction template? | Fit to known W, Z masses using the template from exp_38; extract the Fibonacci indices |
| exp_02 | Neutrino mass series | Do the weak template's lightest entries match neutrino mass bounds? | Generate the full weak-force mass series; identify the three lightest entries; compare to cosmological bounds (sum < 0.12 eV) |
| exp_03 | Mass hierarchy from depth | Does cascade depth ordering predict normal or inverted hierarchy? | Assign cascade depths to nu_1, nu_2, nu_3; check if depth ordering matches mass ordering |
| exp_04 | PMNS consistency check | Are the derived masses consistent with the M5 mixing angles? | Reconstruct the PMNS matrix from the derived masses and mixing angles; verify unitarity and CP phase |
| exp_05 | Majorana vs Dirac test | Does the Fibonacci structure prefer Majorana or Dirac neutrinos? | Test whether the mass template admits a Majorana mass term (L-violating) or only Dirac (L-conserving); check implications for neutrinoless double beta decay |
| exp_06 | sin^2(theta_W) mass connection | Does the identity sin^2(theta_W) = 3/13 constrain neutrino masses? | The electroweak mixing angle is F4/F7; test whether this ratio appears in the neutrino mass ratios |
| exp_07 | Depth-73 dark matter mass | What mass does the cyclotomic hierarchy at depth 73 predict? | Compute m_DM from F6^2 + F6 + 1 = 73 using the correction template; compare to 15 keV sterile neutrino constraints |
| exp_08 | X-ray line constraints | Is the depth-73 prediction consistent with X-ray observations? | Compare predicted DM mass to 3.5 keV line (XMM-Newton), NuSTAR limits, and upcoming XRISM sensitivity |
| exp_09 | WIMP-range depth sweep | Do depths 74-93 produce WIMP-range masses? | Sweep cyclotomic hierarchy at higher depths; identify any resonances in the 1 GeV - 10 TeV range |
| exp_10 | Z-prime at 395 GeV | Does the BSM spectrum include the Z' predicted in M5? | Check if the correction template produces a massive neutral boson near 395 GeV; derive its couplings |
| exp_11 | Falsification bounds | What experimental results would falsify these predictions? | Define the parameter space where each prediction fails; map to upcoming experiments (KATRIN, JUNO, DUNE, XRISM) |

---

## Block B: Loop-Level & Strong Force Resolution

**Thread**: 2 (Loop-Level Coupling Running)
**Goal**: Resolve the UV-fixed vs asymptotic freedom tension; uniquely determine the strong force template
**Prior work**: M5 exp_05 (UV fixed points confirmed), MAR exp_39 (n=3 vs n=8 candidates), MAR exp_40 (tree-level only)

M5 showed DFT predicts UV fixed points — couplings don't run. The simulator confirms this: across 6x scale variation, dg/dlnk < 0.015. But QCD experiments measure running. Either DFT's prediction is wrong (and cascade dynamics produce effective running at lower energies) or it's right (and what QCD measures as "running" is actually the cascade's depth-dependent tiling — different physics, same experimental signature).

This is the deepest open question in DFT. Resolving it determines whether the theory merely reproduces known physics or makes a genuinely novel prediction about the UV structure of gauge theories.

### Experiments

| # | Name | Question | Approach |
|---|------|----------|----------|
| exp_13 | Cascade perturbation theory | Can cascade dynamics produce effective running at low energies? | Develop perturbative expansion around the UV fixed point; compute 1-loop corrections from cascade fluctuations |
| exp_14 | Tiling filter as running | Does the depth-dependent tiling filter reproduce the QCD beta function? | Compare the tiling filter's scale dependence to the 1-loop and 2-loop QCD beta function coefficients |
| exp_15 | n=3 vs n=8 discriminator | What observable distinguishes fundamental (n=3) from adjoint (n=8)? | Identify predictions where the two candidates diverge; design a computational experiment to select one |
| exp_16 | Alpha_s at M_Z | Does the winning template reproduce alpha_s(M_Z) = 0.1179? | Compute alpha_s from the selected template at the Z mass scale; compare to PDG value |
| exp_17 | Confinement from cascade depth | Does the correction template predict confinement at large distances? | Test whether the template diverges (confinement) or saturates (screening) at large cascade depth |
| exp_18 | Running fraction 0.071 | What is the PAC origin of the running fraction identified in MAR exp_40? | The number 0.071 has no known Fibonacci form; search for it in cascade statistics, tiling ratios, and spectral eigenvalues |
| exp_19 | Unification test | Do the three gauge couplings meet at a single cascade depth? | Run all three coupling templates to high cascade depth; check for convergence (GUT-like unification) |
| exp_20 | Proton decay bound | If couplings unify, what does the unification depth predict for proton lifetime? | Derive the proton decay rate from the unification depth; compare to Super-Kamiokande limit (> 10^34 years) |
| exp_21 | Falsification conditions | Under what conditions is the UV-fixed prediction falsified? | Define the experimental signatures that would require genuine running (not tiling mimicry); specify energy scales and precision thresholds |

---

## Block C: Cosmological Constant & JWST Validation

**Threads**: 3 (Cosmological Constant Gap) + 5 (JWST Cosmology Upgrade)
**Goal**: Close the 0.22-order CC gap; validate DFT cosmological predictions against JWST data
**Prior work**: MAR exp_35-36 (CC reduced to 10^0.22), MAR exp_40 (sub-integer depth), pac_cosmology_validation UPGRADE_PLAN.md, MAR exp_22 (PAC Eddington regulator), MAR exp_27 (free-streaming with 5 testable predictions)

The cosmological constant problem — the 10^123 orders-of-magnitude discrepancy between quantum field theory's prediction and observation — was reduced to 10^0.22 orders in MAR exp_35-36 via a tiling interpretation (N_eff = 2 × 183 × Xi). The remaining gap is real and structured: exp_40 suggested a sub-integer depth correction (182.446 instead of 183), and the Omega_Lambda template from exp_35 Part F achieves 0.012% error with an underived a=9 index.

The JWST thread is different in character — it's about confronting DFT's cosmological predictions with real observational data. The UPGRADE_PLAN.md in pac_cosmology_validation lays out a 6-phase programme. This is where DFT either earns its cosmological credentials or reveals where the framework breaks down.

### Experiments

| # | Name | Question | Approach |
|---|------|----------|----------|
| exp_23 | Sub-integer depth correction | Does depth 182.446 close the CC gap? | Replace integer depth 183 with the continuous value from exp_40; recompute CC; measure residual |
| exp_24 | Mixed tiling factor | Does combining Xi and xi_PAC improve the CC prediction? | Test N_eff = 2 × d × f(Xi, xi_PAC) for various combination functions; compare to integer-only tiling |
| exp_25 | Derive the a=9 index | Where does the Omega_Lambda template's a=9 come from? | Search for 9 in Fibonacci indices, cascade structure, and tiling geometry; derive rather than fit |
| exp_26 | CC from first principles | Can the full CC be derived without tiling approximation? | Attempt a direct derivation of Lambda from PAC/SEC without the N_eff intermediate step |
| exp_27 | CC sensitivity analysis | How sensitive is the CC prediction to input uncertainties? | Propagate uncertainties in Xi, phi, and cascade depth through the CC formula; establish error bars |
| exp_28 | JWST catalog compilation | Compile 50+ JWST/high-z objects with DFT-relevant observables | Catalog from published JWST data (JADES, CEERS, GLASS); include redshift, mass, SFR, morphology |
| exp_29 | SEC enhancement derivation | Derive SEC enhancement from first principles (not hardcoded) | The current SEC enhancement in cosmo.py is fitted; derive it from PAC/SEC dynamics at cosmological scales |
| exp_30 | LCDM baseline comparison | Establish null hypothesis against standard LCDM | Compute LCDM predictions for all catalog objects; define AIC/BIC comparison framework |
| exp_31 | DFT cosmological predictions | Generate DFT predictions for each catalog object | Run the PAC cosmology model on all 50+ objects; record predictions for comparison |
| exp_32 | Statistical validation | Does DFT outperform LCDM on the catalog? | Compute AIC/BIC for DFT vs LCDM; report significance levels and identify outliers |
| exp_33 | K-level mismatch diagnosis | What causes the K-level mismatch in current cosmology code? | Trace the K-level discrepancy identified in pac_cosmology_validation; determine if it's a bug or physics |
| exp_34 | Free-streaming predictions | Test the 5 testable predictions from MAR exp_27 | Each of exp_27's 5 predictions (spectral tilt, BAO feature, etc.) gets a quantitative comparison to data |
| exp_35 | Eddington regulator cosmology | Does the PAC Eddington regulator (exp_22) improve cosmological predictions? | Apply the Eddington luminosity regulator to high-z galaxy formation; compare to JWST early massive galaxies |
| exp_36 | Parameter sensitivity sweep | How robust are DFT cosmological predictions to parameter variation? | Vary all free parameters ±10%; map the sensitivity landscape; identify fragile predictions |
| exp_37 | Cosmological falsification bounds | What JWST observations would falsify DFT cosmology? | Define the regions of observable space incompatible with DFT; map to upcoming JWST Cycle 3+ programs |

---

## Block D: Simulator Scorecard & Applied DFT

**Threads**: 4 (Simulator Scorecard Push) + 6 (Knowledge Discovery Tool)
**Goal**: Push simulator scorecard to B-; validate PAC Knowledge Discovery against baselines
**Prior work**: M5 exp_06-13 (attractor dynamics), coupling-drift-physics FDO, MAR exp_41-43 (emergent attractors), pac_knowledge_discovery (5 experiments)

Two very different threads unified by pragmatism. The scorecard push continues M5's simulator work — the de-actualization fix cut error from 8.1% to 6.2%, but the phi^2 spacing regression at 40.6% and the anti-correlated coupling groups remain unsolved. MAR exp_41-43 are actively testing emergent attractors in Reality Engine v3; their results will determine whether the fix is in the operators or the architecture.

The Knowledge Discovery thread is DFT's first applied publication. If PAC residuals reliably decrease when missing features are added to a dataset, and if this outperforms established causal discovery methods, it demonstrates that the PAC framework has practical utility — a powerful argument for the theory's validity that doesn't depend on any specific physics prediction.

### Experiments

| # | Name | Question | Approach |
|---|------|----------|----------|
| exp_38 | MAR attractor integration | Do exp_41-43 emergent attractors improve the scorecard? | Integrate results from MAR exp_41-43 (actualization ratio, emergent attractor, emergent couplings) into the scorecard evaluation |
| exp_39 | Phi^2 spacing diagnosis | What operator causes the 40.6% phi^2 spacing regression? | Ablation study: disable operators one at a time; measure phi^2 spacing; identify the culprit |
| exp_40 | Anti-correlation resolution | Can the Group 1 / Group 2 trade-off be broken? | Test whether modifying PAC conservation to allow asymmetric flow resolves the anti-correlation without violating conservation |
| exp_41 | Actualization operator refinement | Does a modified ActualizationOperator improve convergence? | Test alternative actualization functions (sigmoid, tanh, stepped) against the current linear form |
| exp_42 | Entropy reduction recovery | Can entropy reduction be restored alongside de-actualization? | The entropy metric worsened with de-actualization; test entropy-aware forgetting schedules |
| exp_43 | Full scorecard evaluation | What is the scorecard after all Block D fixes? | Run the complete physics scorecard with all improvements; report all 13 metrics |
| exp_44 | PAC residual validation | Do PAC residuals decrease when missing features are added? | Test on 3 benchmark datasets (synthetic + real); measure residual reduction as features are added/removed |
| exp_45 | PC algorithm comparison | Does PAC Discovery outperform the PC algorithm? | Run both on identical datasets; compare structural Hamming distance, F1 score, and runtime |
| exp_46 | NOTEARS comparison | Does PAC Discovery outperform NOTEARS? | Same comparison framework as exp_45; include both linear and nonlinear NOTEARS variants |
| exp_47 | Scalability benchmark | How does PAC Discovery scale with variable count? | Benchmark on datasets with 10, 50, 100, 500 variables; report runtime and accuracy curves |
| exp_48 | Real-world application | Can PAC Discovery find known causal relationships in a real dataset? | Apply to a well-studied dataset (e.g., Sachs protein signaling, or climate indices); compare discovered graph to known ground truth |
| exp_49 | Tool packaging | Package PAC Discovery as installable Python tool | Create pac-discovery package with API, documentation, and example notebooks |

---

## Dependencies

Each block draws on specific prior results:

### Block A (Neutrino & BSM)
- M5 exp_08: PMNS mixing angles as arctan(F_a/F_b)
- MAR exp_38: Four-force correction template with force-specific parameters
- MAR exp_37: Complete force table with Fibonacci characterizations
- gravity exp_03-04: Cyclotomic hierarchy at depth 73 (dark matter candidate)
- M5 identity: sin^2(theta_W) = tan(theta_C) = 3/13

### Block B (Loop-Level)
- M5 exp_05: UV fixed points confirmed — dg/dlnk < 0.015 across 6x scale
- M5 exp_04: Strong force implicit in tiling filter
- MAR exp_39: Two strong force candidates (n=3 vs n=8)
- MAR exp_40: Running fraction 0.071 (no PAC form)

### Block C (Cosmology)
- MAR exp_35-36: CC reduced to 10^0.22 via tiling interpretation
- MAR exp_40: Sub-integer depth correction 182.446
- MAR exp_22: PAC Eddington regulator
- MAR exp_27: Five testable free-streaming predictions
- pac_cosmology_validation: UPGRADE_PLAN.md (6-phase programme)

### Block D (Simulator & Applied)
- M5 exp_06-13: Attractor dynamics, de-actualization, 6.2% error
- MAR exp_41-43: Emergent coupling attractors (in progress)
- coupling-drift-physics: Two anti-correlated coupling groups
- pac_knowledge_discovery: 5 seed experiments, tier-1 criteria not yet validated

---

## Risk Analysis

### Block A Risks
- **Neutrino mass sum too large**: If the Fibonacci template produces sum(m_nu) > 0.12 eV, it's immediately falsified by Planck. Mitigation: the template has correction terms that may bring masses into range.
- **Hierarchy ambiguity**: If cascade depth ordering is degenerate (admits both normal and inverted), the prediction loses power. Mitigation: check whether the PMNS angles from M5 break the degeneracy.
- **Dark matter at depth-73 already excluded**: The 15 keV region is heavily constrained. Mitigation: check all current X-ray limits before publishing; if excluded, the framework still predicts *where* DM isn't.

### Block B Risks
- **UV-fixed prediction is wrong**: If cascade dynamics produce running that matches QCD exactly, DFT loses its most distinctive prediction. This isn't a failure — it's a successful reconciliation — but it's less exciting.
- **Neither running nor fixed**: If the cascade produces something that's neither standard running nor true fixed points, the result may be hard to connect to experiment. Mitigation: focus on alpha_s(M_Z) as a concrete anchor.
- **n=3 vs n=8 underdetermined**: If no computational experiment can distinguish them, the strong force template remains ambiguous. Mitigation: look for different predictions at different energy scales.

### Block C Risks
- **CC gap is fundamental, not closable**: The 10^0.22 gap may reflect a genuine approximation in the tiling approach, not a correctable error. Mitigation: attempt the first-principles derivation (exp_26) as an alternative path.
- **JWST data insufficient**: 50+ objects may not provide enough statistical power for AIC/BIC discrimination. Mitigation: focus on the most constraining objects (highest redshift, most extreme properties).
- **K-level mismatch is a bug**: If the mismatch in cosmo.py is a coding error rather than physics, it undermines confidence in all prior cosmological results. Mitigation: diagnose early (exp_33) before building on the code.

### Block D Risks
- **Scorecard improvement is marginal**: If attractor convergence only moves the scorecard from 8/13 to 9/13, the B- target is out of reach. Mitigation: identify which specific metrics are closest to flipping and focus resources there.
- **PAC Discovery doesn't outperform baselines**: Established causal discovery methods are well-optimized. Mitigation: focus on domains where PAC's information-theoretic foundation gives a structural advantage (high-dimensional, noisy data).
- **Packaging effort exceeds value**: Building a Python tool is engineering, not physics. Mitigation: keep it minimal (API + examples); defer polished packaging to a dedicated sprint.

---

## Target Papers

| Paper | Title (working) | Blocks | Status |
|-------|-----------------|--------|--------|
| Paper 9 | Standard Model from Information — The Complete Derivation | A (neutrino masses complete the SM) | Consolidates M1-M5 + Block A |
| Paper 10 | Predictions for BSM Physics from PAC Structure | A (BSM), B (strong force resolution) | Requires Block A + B results |
| Paper 11 | Cosmological Validation of Dawn Field Theory | C (CC + JWST) | Requires Block C completion |
| PAC Discovery | Causal Discovery via Potential-Actualization Conservation | D (knowledge discovery) | First applied-DFT paper; independent of physics results |

---

## Timeline

```
Phase 1 (Parallel):  Block A (exp_01-11) + Block B (exp_13-21)
                      These are theoretically independent — neutrino masses
                      don't depend on loop-level resolution, and vice versa.

Phase 2 (Parallel):  Block C (exp_23-37) + Block D:Scorecard (exp_38-43)
                      Cosmology requires MAR results to stabilize.
                      Scorecard depends on MAR exp_41-43 completing.
                      Both can run alongside each other.

Phase 3 (Sequential): Block D:Discovery (exp_44-49)
                       Applied DFT is lowest priority and can use
                       downtime between physics experiments.

Cross-cuts:
  - Block B results (running vs fixed) may affect Block C (CC derivation
    assumes tree-level couplings; if running exists, CC needs recalculation)
  - Block A results (neutrino masses) feed Paper 9 which also needs
    Block B's strong force resolution
  - Block D scorecard improvements may reveal insights that feed back
    into Block B (simulator as theory-checking tool)
```

---

## What This Milestone Means

Milestones 1-5 established that the Standard Model's parameters — masses, mixing angles, coupling constants — can be derived from Fibonacci arithmetic operating on the PAC/SEC/MED framework. That's remarkable, but it's also retrospective: DFT reproduces known physics.

Milestone 6 is where the theory starts predicting. Neutrino masses that can be checked against KATRIN, JUNO, and DUNE. A dark matter candidate mass testable by XRISM. A cosmological constant derivation confrontable with JWST data. UV fixed points that either agree with or contradict the next generation of collider measurements.

If M6 succeeds, DFT isn't just a clever way to re-derive the Standard Model. It's a framework that tells us things we don't yet know about the universe — and tells us how to check.

---

*Dawn Field Institute, 2026*
