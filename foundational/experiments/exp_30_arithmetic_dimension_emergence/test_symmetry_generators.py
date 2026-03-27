"""
Test 1: Symmetry Generator Independence
Verify that translation, dilation, and rotation are algebraically independent
generators of the conformal group (Möbius transformations) on the complex plane.
"""

import numpy as np
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}

# --- Möbius transformation matrices (SL(2,C) representatives) ---
def translation(b):
    """T_b(z) = z + b"""
    return np.array([[1, b], [0, 1]], dtype=complex)

def dilation(lam):
    """D_λ(z) = λz"""
    return np.array([[lam, 0], [0, 1]], dtype=complex)

def rotation(theta):
    """R_θ(z) = e^{iθ}z"""
    return np.array([[np.exp(1j * theta), 0], [0, 1]], dtype=complex)

def inversion():
    """I(z) = 1/z"""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def mobius_apply(M, z):
    """Apply Möbius transformation M to z."""
    a, b, c, d = M[0,0], M[0,1], M[1,0], M[1,1]
    if np.abs(c * z + d) < 1e-14:
        return np.inf
    return (a * z + b) / (c * z + d)

def matrices_close(A, B, tol=1e-10):
    """Check if two SL(2,C) matrices are equal up to scalar (projective equivalence)."""
    # Normalize both
    A_n = A / A.flat[np.argmax(np.abs(A.flat))]
    B_n = B / B.flat[np.argmax(np.abs(B.flat))]
    return np.allclose(A_n, B_n, atol=tol)

# ============================================================
# (a) Decomposition: Any Möbius matrix from generators
# ============================================================
print("=" * 60)
print("TEST 1a: Decomposition of arbitrary Möbius transformations")
print("=" * 60)

def decompose_mobius(M):
    """
    Decompose M = [[a,b],[c,d]] into products of T, D, R, and I.

    Standard decomposition:
    If c != 0:
        f(z) = (az+b)/(cz+d) = a/c - (ad-bc)/(c(cz+d))
        = T(a/c) . D(-(ad-bc)/c^2) . I . T(d/c) applied to z
    If c == 0:
        f(z) = (a/d)z + b/d = T(b/d) . D(a/d) applied to z
    """
    a, b, c, d = M[0,0], M[0,1], M[1,0], M[1,1]
    det = a*d - b*c

    if np.abs(c) < 1e-14:
        # f(z) = (a/d)z + b/d
        lam = a / d
        shift = b / d
        factors = []
        # Decompose λ into |λ| * e^{iθ}
        r = np.abs(lam)
        theta = np.angle(lam)
        if np.abs(shift) > 1e-14:
            factors.append(("T", shift))
        if abs(r - 1.0) > 1e-14:
            factors.append(("D", r))
        if abs(theta) > 1e-14:
            factors.append(("R", theta))
        return factors, False  # No inversion needed
    else:
        # Need inversion
        shift1 = d / c
        scale = -(det) / (c**2)
        shift2 = a / c
        factors = []
        if np.abs(shift2) > 1e-14:
            factors.append(("T", shift2))
        # Decompose scale
        r = np.abs(scale)
        theta = np.angle(scale)
        if abs(r - 1.0) > 1e-14:
            factors.append(("D", r))
        if abs(theta) > 1e-14:
            factors.append(("R", theta))
        factors.append(("I", None))
        if np.abs(shift1) > 1e-14:
            factors.append(("T", shift1))
        return factors, True  # Inversion needed

# Test decomposition on random Möbius transformations
np.random.seed(42)
decomp_results = []
n_tests = 100
n_success = 0

for i in range(n_tests):
    # Random SL(2,C) matrix
    M = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2))
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    M = M / np.sqrt(det)  # Normalize to det=1

    factors, needs_inv = decompose_mobius(M)

    # Reconstruct: factors are listed left-to-right as matrix product
    R = np.eye(2, dtype=complex)
    for ftype, val in factors:
        if ftype == "T":
            R = R @ translation(val)
        elif ftype == "D":
            R = R @ dilation(val)
        elif ftype == "R":
            R = R @ rotation(val)
        elif ftype == "I":
            R = R @ inversion()

    # Check on test points
    test_z = [0.5+0.3j, 1.0+1.0j, -0.7+0.2j, 2.0-1.5j]
    match = True
    for z in test_z:
        w1 = mobius_apply(M, z)
        w2 = mobius_apply(R, z)
        if w1 != np.inf and w2 != np.inf:
            if np.abs(w1 - w2) > 1e-8:
                match = False
                break
    if match:
        n_success += 1

print(f"Decomposition test: {n_success}/{n_tests} random Möbius transforms correctly decomposed")
results["decomposition_success_rate"] = n_success / n_tests

# ============================================================
# (b) No proper subset generates the full group
# ============================================================
print("\n" + "=" * 60)
print("TEST 1b: No proper subset generates the full group")
print("=" * 60)

# Test: translations alone generate only translations (upper triangular, c=0, a=d=1)
# Dilations alone generate only dilations (diagonal, c=0, b=0)
# Rotations alone generate only rotations (diagonal, |a|=1)

subsets = {
    "translations_only": {
        "generators": lambda: [translation(np.random.randn() + 1j*np.random.randn()) for _ in range(5)],
        "description": "Translations form abelian subgroup; cannot produce c≠0 or scaling",
        "test": lambda M: np.abs(M[1,0]) < 1e-10 and np.abs(M[0,0] - 1) < 1e-10
    },
    "dilations_only": {
        "generators": lambda: [dilation(np.random.rand()*5 + 0.1) for _ in range(5)],
        "description": "Dilations form abelian subgroup; cannot produce b≠0 or c≠0",
        "test": lambda M: np.abs(M[0,1]) < 1e-10 and np.abs(M[1,0]) < 1e-10
    },
    "rotations_only": {
        "generators": lambda: [rotation(np.random.rand()*2*np.pi) for _ in range(5)],
        "description": "Rotations form abelian subgroup SO(2); cannot produce |a|≠1, b≠0, c≠0",
        "test": lambda M: np.abs(M[0,1]) < 1e-10 and np.abs(M[1,0]) < 1e-10 and np.abs(np.abs(M[0,0]) - 1) < 1e-10
    },
    "T_and_D": {
        "generators": lambda: [translation(np.random.randn() + 1j*np.random.randn()) for _ in range(3)] +
                               [dilation(np.random.rand()*5 + 0.1) for _ in range(3)],
        "description": "T+D generate affine group az+b; cannot produce c≠0 (no inversion)",
        "test": lambda M: np.abs(M[1,0]) < 1e-10
    },
    "T_and_R": {
        "generators": lambda: [translation(np.random.randn() + 1j*np.random.randn()) for _ in range(3)] +
                               [rotation(np.random.rand()*2*np.pi) for _ in range(3)],
        "description": "T+R generate rigid motions (Euclidean group); cannot produce scaling or c≠0",
        "test": lambda M: np.abs(M[1,0]) < 1e-10 and np.abs(np.abs(M[0,0]) - 1) < 1e-10
    },
    "D_and_R": {
        "generators": lambda: [dilation(np.random.rand()*5 + 0.1) for _ in range(3)] +
                               [rotation(np.random.rand()*2*np.pi) for _ in range(3)],
        "description": "D+R generate diagonal matrices; cannot produce b≠0 or c≠0",
        "test": lambda M: np.abs(M[0,1]) < 1e-10 and np.abs(M[1,0]) < 1e-10
    },
}

np.random.seed(123)
subset_results = {}
for name, info in subsets.items():
    gens = info["generators"]()
    # Generate products of up to 20 random generators
    all_in_subgroup = True
    for _ in range(200):
        M = np.eye(2, dtype=complex)
        n_prod = np.random.randint(1, 8)
        for _ in range(n_prod):
            idx = np.random.randint(len(gens))
            if np.random.rand() < 0.5:
                M = M @ gens[idx]
            else:
                M = M @ np.linalg.inv(gens[idx])
        if not info["test"](M):
            all_in_subgroup = False
            break

    subset_results[name] = {
        "stays_in_subgroup": all_in_subgroup,
        "description": info["description"]
    }
    status = "CONFIRMED (proper subgroup)" if all_in_subgroup else "BROKEN OUT"
    print(f"  {name}: {status}")

results["subset_tests"] = subset_results

# ============================================================
# (c) Commutator structure (algebraic independence)
# ============================================================
print("\n" + "=" * 60)
print("TEST 1c: Commutator structure")
print("=" * 60)

def commutator(A, B):
    """[A,B] = ABA^{-1}B^{-1}"""
    return A @ B @ np.linalg.inv(A) @ np.linalg.inv(B)

def is_identity(M, tol=1e-10):
    """Check if M is proportional to identity."""
    M_n = M / M[0,0] if np.abs(M[0,0]) > 1e-14 else M
    return np.allclose(M_n, np.eye(2), atol=tol)

# Test commutators between generators
b_val = 1.0 + 0.5j
lam_val = 2.0
theta_val = np.pi / 4

T = translation(b_val)
D = dilation(lam_val)
R = rotation(theta_val)
I_mat = inversion()

pairs = [
    ("T,D", T, D),
    ("T,R", T, R),
    ("D,R", D, R),
    ("T,I", T, I_mat),
    ("D,I", D, I_mat),
    ("R,I", R, I_mat),
]

commutator_results = {}
for name, A, B in pairs:
    C = commutator(A, B)
    is_id = is_identity(C)
    # Frobenius norm of C - I (normalized)
    C_n = C / C[0,0] if np.abs(C[0,0]) > 1e-14 else C
    dist = np.linalg.norm(C_n - np.eye(2))
    commutator_results[name] = {
        "commutes": bool(is_id),
        "distance_from_identity": float(dist)
    }
    print(f"  [{name}] commutes: {is_id}, ||[A,B]-I|| = {dist:.6f}")

results["commutators"] = commutator_results

# D and R commute (both diagonal), but neither commutes with T, and none with I
# This shows algebraic independence of the generator types

# ============================================================
# Inversion: can it be composed from T, D, R?
# ============================================================
print("\n" + "=" * 60)
print("TEST 1d: Is inversion needed as a 4th generator?")
print("=" * 60)

# T, D, R all preserve c=0 (affine maps). Inversion has c≠0.
# Any finite product of T, D, R has c=0.
# Therefore inversion CANNOT be composed from T, D, R.

# Verify numerically: products of T, D, R always have M[1,0]=0
np.random.seed(77)
c_always_zero = True
for _ in range(1000):
    M = np.eye(2, dtype=complex)
    for _ in range(np.random.randint(1, 15)):
        choice = np.random.randint(3)
        if choice == 0:
            M = M @ translation(np.random.randn() + 1j*np.random.randn())
        elif choice == 1:
            M = M @ dilation(np.random.rand()*5 + 0.1)
        else:
            M = M @ rotation(np.random.rand()*2*np.pi)
    if np.abs(M[1,0]) > 1e-10:
        c_always_zero = False
        break

print(f"  Products of T,D,R always have c=0: {c_always_zero}")
print(f"  Inversion has c=1 (non-zero)")
print(f"  => Inversion CANNOT be composed from T,D,R")
print(f"  => Inversion is a necessary 4th generator for the full Möbius group")

results["inversion_independence"] = {
    "TDR_products_always_affine": c_always_zero,
    "inversion_needed": True,
    "explanation": "T, D, R generate only affine maps (c=0). Inversion introduces c!=0, required for full Mobius group."
}

# ============================================================
# Summary: Generator structure
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

summary = {
    "generators": {
        "translation": "T_b: z -> z+b, matrix [[1,b],[0,1]], arithmetic: addition",
        "dilation": "D_lambda: z -> lambda*z, matrix [[lambda,0],[0,1]], arithmetic: multiplication",
        "rotation": "R_theta: z -> e^{i*theta}*z, matrix [[e^{itheta},0],[0,1]], arithmetic: exponentiation (Euler)",
        "inversion": "I: z -> 1/z, matrix [[0,1],[1,0]], arithmetic: reciprocal/division"
    },
    "group_structure": {
        "TDR_generate": "Affine group (az+b), proper subgroup of Mobius",
        "TDRI_generate": "Full Mobius group PSL(2,C)",
        "minimal_generators": "T, D (or R), and I suffice; T+D+R is redundant for affine but all 4 types needed for full group"
    },
    "algebraic_independence": {
        "T_commutes_with_T": True,
        "D_commutes_with_D": True,
        "R_commutes_with_R": True,
        "D_commutes_with_R": True,
        "T_commutes_with_D": False,
        "T_commutes_with_R": False,
        "conclusion": "D and R commute (both diagonal/multiplicative), but T is algebraically independent from D,R. I is independent from all three."
    }
}
results["summary"] = summary

for k, v in summary.items():
    print(f"\n{k}:")
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")

# Save results
def make_serializable(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj

outpath = os.path.join(OUTPUT_DIR, "test1_symmetry_generators.json")
with open(outpath, "w") as f:
    json.dump(make_serializable(results), f, indent=2)

print(f"\nResults saved to {outpath}")
