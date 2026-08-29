"""Does DFT derive DIMENSIONLESS quantities from structure and IMPORT every scale?

Exploratory spike. Two parts: a tabulation, and a falsification test against the corpus's
strongest apparent counterexample.

THE CLAIM. Structure (ratios, classifications, counts) can be produced from the framework's
combinatorics. SCALE cannot -- a length, mass or energy requires an anchor the framework does
not derive. Motivated by the observation that space is INTERIOR to a node, so a scale, which
needs space, must be set internally rather than imposed from outside.

THE FALSIFIER, stated flatly: any dimensionful quantity derived from pure structure with NO
imported anchor kills this.
"""
import math

ALPHA = 7.2973525693e-3

DIMENSIONLESS = [
    ("alpha_EM",              "5.7 ppm",   "none"),
    ("sin^2 theta_W = F4/F7", "0.19%",     "none"),
    ("Koide Q = 2/3",         "0.5 ppm",   "none"),
    ("mu/e mass RATIO",       "5 ppm",     "none"),
    ("p/e mass RATIO",        "0.0083%",   "none"),
    ("Feigenbaum delta",      "~13 digits","none"),
    ("Casimir 240",           "exact",     "none"),
    ("She-Leveque beta=2/3",  "exact",     "none"),
    ("alpha_s = F4/(F3 phi F6)", "1.71%",  "none"),
    ("theta_12, theta_13",    "0.28/0.21 deg", "none"),
    ("Xi = 1 + pi/F10",       "0.12%",     "none"),
]
DIMENSIONFUL = [
    ("Rydberg (energy)",       "11.4 ppm",         "m_e  IMPORTED"),
    ("nuclear scale (energy)", "1.75x",            "m_p  IMPORTED"),
    ("beta endpoint (energy)", "factor ~10, 2/3",  "m_p  IMPORTED (and degenerate)"),
    ("Higgs mass 125.26 GeV",  "83 ppm",           "Higgs VEV IMPORTED"),
    ("E_Planck * phi^-d",      "15-24 ORDERS OFF", "none -- and it FAILS"),
    ("M11 Planck-scale work",  "O(1) prefactors",  "E_PLANCK/T_PLANCK/HBAR/C IMPORTED"),
]


def main():
    print("DIMENSIONLESS -- derived with no imported scale")
    for n, e, a in DIMENSIONLESS:
        print(f"   {n:<26}{e:>18}   anchor: {a}")
    print("\nDIMENSIONFUL -- every success borrows a scale the framework does not derive")
    for n, e, a in DIMENSIONFUL:
        print(f"   {n:<26}{e:>18}   anchor: {a}")

    print("\n" + "=" * 78)
    print("  FALSIFICATION TEST vs M11 -- 'Planck scale derived, not assumed', 52/52")
    print("=" * 78)
    print("""
  M11 core/quantum_gravity.py:

      def crossover_energy(depth):
          return E_PLANCK_GEV * PHI ** (-depth)

      def cascade_depth_response_time(depth, base_time=T_PLANCK_S):
          return base_time / (PHI ** (-depth))

  Every dimensionful output is [imported constant] x [dimensionless phi power].
  HBAR, C_LIGHT and M_PLANCK_KG are hardcoded CODATA values. exp_02's four
  'routes to the Planck scale' are, in its own words, "length scales in Planck
  units" -- 1/ln(2), 1/2, 2M. O(1) prefactors inside a unit system that already
  contains the answer.

  M11's own docstring, line 113:
      "For gravity (depth 183): E_cross = E_Planck * phi^(-183) ~ 10^-19 GeV.
       This is NOT the Planck energy -- it's where classical gravity breaks."

  VERDICT: not a counterexample. M11 imports the anchor and derives dimensionless
  factors within it. The boundary SURVIVES its strongest test.""")

    print("\n" + "=" * 78)
    print("  THE UNIFICATION: one expression, two verdicts")
    print("=" * 78)
    e_cross = 1.220890e19 * (((1 + 5 ** 0.5) / 2) ** -183)
    print(f"""
  E_Planck * phi^(-d) is used by BOTH M11 and Milestone R.

    M11,          d=183 : {e_cross:.2e} GeV -- sensible, and M11 says so
    Milestone R,  d<=20 : 15-24 ORDERS above nuclear/atomic scales -- the corpus
                          calls it "the most important result is a failure"

  It is NOT a scale law. It is a DIMENSIONLESS CORRECTION TO AN IMPORTED ANCHOR.
  It looks like a derivation exactly when the anchor already sits near the target
  (M11, working at the Planck scale) and fails visibly when it must bridge 24
  orders to reach somewhere else (Milestone R, reaching for MeV).

  So Milestone R's failure is not "this formula is wrong". The formula was never
  capable of setting a scale anywhere; its successes and failures are the same
  fact seen from different distances to the anchor.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
