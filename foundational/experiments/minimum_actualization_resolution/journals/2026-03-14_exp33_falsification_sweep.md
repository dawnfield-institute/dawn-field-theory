# Journal: exp_33 Falsification Sweep

**Date**: 2026-03-14
**Status**: complete (8/8 PASS, 8 limitations documented)

---

## Origin

After exp_32 derived the Einstein field equations from PAC/MED (6/6 PASS), Peter asked to do a full falsification sweep: "yea lets do it, also if holes are poked, lets use them as fuel for our limitations." The goal was to stress-test every link in the gravity derivation chain (exp_28 through exp_32) and convert weaknesses into documented research directions.

## Structure

Eight parts, each targeting a specific vulnerability:

- **Part A (PASS)**: Circularity audit — traced all 7 links in the derivation chain. Found 1 circular (G requires measurement), 2 weak (covariance assumed; depth interpretation), 4 clean.

- **Part B (PASS)**: Lovelock smuggling test — checked whether the 3 Lovelock premises (symmetric, divergence-free, second-order) are genuinely from PAC. Result: 1 genuine (divergence-free from PAC axiom), 1 partial (symmetry from projection argument), 1 interpretive (second-order from depth mapping). No smuggling from GR.

- **Part C (PASS)**: G independence — F_183 gives G to within a factor of 2.15 (0.33 orders). In Planck units kappa = 8*pi exactly. The FORM of Einstein's equations is fully derived; the SCALE requires one measurement. Analogous to QED (form from gauge, alpha measured).

- **Part D (PASS)**: Alternative axioms — at least 6 other routes to Einstein (Hilbert, Jacobson, Padmanabhan, Verlinde, Weinberg-Witten, Deser). PAC is sufficient not necessary. Its value is in the CONNECTIONS (EM-gravity unification, Fibonacci hierarchy, dark sector predictions, turbulence link). Multiple routes = confluence, not redundancy.

- **Part E (PASS)**: Depth sensitivity — if MED allowed depth 3, field equations would have 8 DoF (including massive spin-2 ghost). Ostrogradsky instability = unphysical. MED depth <= 2 PREVENTS ghosts — a genuine falsifiable prediction. Depth-3 corrections are Planck-suppressed, so Mercury can't distinguish.

- **Part F (PASS)**: Dimensional robustness — derivation requires d=4 and breaks cleanly otherwise. d<4: no GWs. d>4: extra Lovelock terms. d=5 gives 1/r^2 potential which kills Mercury precession (exp_31 Part D). PAC predicts d=4 via exp_17 (confluence period-4) and Gauss's law (1/r only in d=3).

- **Part G (PASS, 2 weaknesses)**: Dark sector robustness — Omega_c = F3*Xi/F6 is linearly sensitive to Xi and NOT unique (5 Fibonacci formulas match within 1%). 1/phi is NOT the best simple fit for Omega_Lambda (1-1/pi is closer at 0.3pp vs 6.7pp). Both predictions need theoretical grounding beyond numerics.

- **Part H (PASS)**: Beyond Schwarzschild — Reissner-Nordstrom derivable from EM stress-energy. Kerr derivable from no-hair theorem (G_muv = 0 + axisymmetry) but PAC physical picture (angular cascade density) not yet formulated. FLRW/de Sitter derived. Nonlinear GWs follow from G_muv nonlinearity. Underivable: BH interior, Hawking temperature, graviton quantization, topology change.

## Limitation Registry

| ID | Severity | Limitation | Path Forward |
|----|----------|-----------|--------------|
| L1 | HIGH | G not derived from PAC | Improve F_183 precision beyond OoM |
| L2 | MEDIUM | General covariance assumed | Derive from PAC phase-cycling on discrete lattice |
| L3 | MEDIUM | MED depth -> derivative order interpretive | Find testable setup for depth interpretation |
| L4 | LOW | PAC sufficient not necessary | Show PAC unifies other routes (confluence) |
| L5 | HIGH | Cosmological constant unsolved | Cascade cancellation / Fibonacci suppression / phase cycling |
| L6 | MEDIUM | Dark sector may be coincidence | Derive Omega_c from cascade dynamics not Fibonacci fitting |
| L7 | LOW | Frame dragging picture missing | Formulate angular cascade density |
| L8 | LOW | Local-Gauss bridge open | Find scale-dependent transition |

## Key Insight

The gravity derivation chain is structurally sound. The two HIGH-severity issues (G not derived, CC unsolved) are shared by ALL approaches to gravity. The MEDIUM-severity issues are genuinely PAC-specific and define clear research directions. Every limitation is fuel.
