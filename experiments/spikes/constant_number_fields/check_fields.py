"""Which NUMBER FIELD does each derived DFT constant live in?

Exploratory spike. The motivating claim was that gauge couplings carry exactly one power of
phi while ratios carry none -- a "class signature". That claim DIED under attack, because the
phi-count is notation, not physics:

    F3/(F4*phi*F10)  ==  (-1 + sqrt(5))/165        <- same number, zero phi's written

What survives the rewrite is FIELD MEMBERSHIP, which is a property of the number and cannot be
edited away. Run in exact arithmetic with sympy.
"""
import sympy as sp

phi = (1 + sp.sqrt(5)) / 2
F = {3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 9: 34, 10: 55, 12: 144}

CASES = [
    ("alpha_EM",      "gauge coupling", sp.Integer(F[3])/(F[4]*phi*F[10])
                                        * (1 - sp.Rational(F[10], 4*F[7]**2)/sp.pi)),
    ("alpha_s",       "gauge coupling", sp.Integer(F[4])/(F[3]*phi*F[6])),
    ("lambda_Higgs",  "SELF-coupling",  phi/(4*sp.pi)),
    ("sin^2 theta_W", "mixing",         sp.Rational(F[4], F[7])),
    ("Koide Q",       "ratio",          sp.Rational(F[3], F[4])),
    ("mu/e",          "mass ratio",     sp.Integer(F[4]*F[6]**2)*(1 + sp.Rational(1, F[7]))),
    ("p/e",           "mass ratio",     sp.Rational(F[4]*F[9]*F[12], F[6])),
    ("Casimir 240",   "count",          sp.Integer(F[3]*F[4]*F[5]*F[6])),
    ("Xi",            "balance",        1 + sp.pi/F[10]),
]


def field_of(e):
    e = sp.simplify(e)
    if e.is_rational:
        return "Q            counting"
    if e.has(sp.pi):
        return "transcendental  closure"
    return "Q(sqrt5)     growth"


def main():
    print("phi-COUNTING IS NOTATION -- demonstrated exactly:")
    a = sp.Integer(F[3])/(F[4]*phi*F[10])
    print(f"   F3/(F4*phi*F10) = {sp.radsimp(a)}   (zero phi's; same number)\n")
    print(f"{'constant':<16}{'class':<16}{'exact value':<26}{'field'}")
    for name, cls, v in CASES:
        e = sp.simplify(v)
        val = str(e) if len(str(e)) < 25 else f"{float(e):.9f}..."
        print(f"{name:<16}{cls:<16}{val:<26}{field_of(e)}")
    print("\nEvery mixing angle and mass ratio is EXACTLY RATIONAL.")
    print("Every coupling is irrational. Neither can be changed by rewriting.")
    print("\nmu/e  = 2688/13 exactly;  p/e = 1836 exactly (an integer).")


if __name__ == "__main__":
    raise SystemExit(main())
