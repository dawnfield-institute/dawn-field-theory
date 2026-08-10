"""
Test 3: Recursive Closure Verification
Starting from 1 and "compress repeated application," verify the hierarchy:
  successor -> addition -> multiplication -> exponentiation -> tetration
"""

import numpy as np
import json
import os
import math
from decimal import Decimal, getcontext

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

getcontext().prec = 50

results = {}

# ============================================================
# Level 0: Successor S(n) = n + 1
# Level 1: Addition a + b = apply S b times to a
# Level 2: Multiplication a * b = add a, b times
# Level 3: Exponentiation a^b = multiply by a, b times
# Level 4: Tetration ^b a = exponentiate a, b times
# ============================================================

print("=" * 60)
print("RECURSIVE CLOSURE HIERARCHY VERIFICATION")
print("=" * 60)

# --- Verify compression identity ---
print("\n--- Verification: Each level compresses the one below ---")

def successor(n):
    return n + 1

def addition_via_successor(a, b):
    """a + b via repeated successor"""
    result = a
    for _ in range(b):
        result = successor(result)
    return result

def multiplication_via_addition(a, b):
    """a * b via repeated addition"""
    result = 0
    for _ in range(b):
        result = addition_via_successor(result, a)
    return result

def multiplication_via_addition_fast(a, b):
    """a * b via repeated addition (uses built-in addition, not successor)"""
    result = 0
    for _ in range(b):
        result += a
    return result

def exponentiation_via_multiplication(a, b):
    """a^b via repeated multiplication (uses built-in multiply for speed)"""
    result = 1
    for _ in range(b):
        result *= a
    return result

def exponentiation_via_multiplication_slow(a, b):
    """a^b via repeated mul_via_add (only for small values)"""
    result = 1
    for _ in range(b):
        result = multiplication_via_addition_fast(result, a)
    return result

def tetration(a, b, max_digits=1000):
    """^b a via repeated exponentiation. Returns int or 'OVERFLOW' string."""
    if b == 0:
        return 1
    result = a
    for _ in range(b - 1):
        # Guard against astronomical values
        if isinstance(result, str) or (isinstance(result, int) and result > max_digits):
            return "OVERFLOW"
        try:
            result = a ** result
            if isinstance(result, int) and result.bit_length() > 3000:
                return "OVERFLOW"
        except (OverflowError, MemoryError, ValueError):
            return "OVERFLOW"
    return result

# Verify small cases
print("\nCompression verification (small values):")
compression_checks = []
for a in range(1, 6):
    for b in range(1, 6):
        add_ok = addition_via_successor(a, b) == a + b
        mul_ok = multiplication_via_addition_fast(a, b) == a * b
        exp_ok = exponentiation_via_multiplication_slow(a, b) == a ** b
        compression_checks.append((a, b, add_ok, mul_ok, exp_ok))

all_add = all(c[2] for c in compression_checks)
all_mul = all(c[3] for c in compression_checks)
all_exp = all(c[4] for c in compression_checks)

print(f"  Addition = repeated successor:       ALL MATCH = {all_add}")
print(f"  Multiplication = repeated addition:  ALL MATCH = {all_mul}")
print(f"  Exponentiation = repeated multiply:  ALL MATCH = {all_exp}")

results["compression_verification"] = {
    "addition_from_successor": all_add,
    "multiplication_from_addition": all_mul,
    "exponentiation_from_multiplication": all_exp,
    "test_range": "a,b in 1..5"
}

# ============================================================
# Closure on naturals
# ============================================================
print("\n--- Closure Properties ---")

closure_results = {}

# Addition: N x N -> N (closed on naturals)
print("\nAddition: closed on N? YES (sum of naturals is natural)")
closure_results["addition"] = {
    "closed_on_naturals": True,
    "closed_on_integers": True,
    "closed_on_rationals": True,
    "closed_on_reals": True,
    "closed_on_complex": True
}

# Multiplication: N x N -> N (closed on naturals)
print("Multiplication: closed on N? YES")
closure_results["multiplication"] = {
    "closed_on_naturals": True,
    "closed_on_integers": True,
    "closed_on_rationals": True,
    "closed_on_reals": True,
    "closed_on_complex": True
}

# Exponentiation: N x N -> N (closed on naturals, for a>=1, b>=0)
print("Exponentiation: closed on N? YES (for a>=1, b>=0)")
closure_results["exponentiation"] = {
    "closed_on_naturals": True,
    "note": "0^0 conventionally 1; closed for positive bases and natural exponents",
    "closed_on_positive_reals": True,
    "closed_on_reals": "partial (negative base with non-integer exponent problematic)",
    "closed_on_complex": True
}

# Tetration: grows too fast
print("Tetration: closed on N? YES (but grows hyper-exponentially)")
closure_results["tetration"] = {
    "closed_on_naturals": True,
    "note": "Values grow so fast they quickly exceed computational limits",
    "closed_on_reals": "problematic (non-integer heights ill-defined in general)",
    "closed_on_complex": "open research problem"
}

results["closure"] = closure_results

# ============================================================
# Commutativity
# ============================================================
print("\n--- Commutativity Tests ---")

comm_results = {}

# Addition: a + b = b + a
add_comm = all(a + b == b + a for a in range(1, 20) for b in range(1, 20))
print(f"  Addition commutative:       {add_comm}")
comm_results["addition"] = {"commutative": add_comm}

# Multiplication: a * b = b * a
mul_comm = all(a * b == b * a for a in range(1, 20) for b in range(1, 20))
print(f"  Multiplication commutative: {mul_comm}")
comm_results["multiplication"] = {"commutative": mul_comm}

# Exponentiation: a^b != b^a in general
exp_counter = []
for a in range(2, 10):
    for b in range(2, 10):
        if a != b and a**b != b**a:
            exp_counter.append((a, b, a**b, b**a))
exp_comm = len(exp_counter) == 0
print(f"  Exponentiation commutative: {exp_comm}")
if exp_counter:
    ex = exp_counter[0]
    print(f"    Counterexample: {ex[0]}^{ex[1]}={ex[2]} != {ex[1]}^{ex[0]}={ex[3]}")
comm_results["exponentiation"] = {
    "commutative": exp_comm,
    "counterexample": f"{exp_counter[0][0]}^{exp_counter[0][1]}={exp_counter[0][2]} vs {exp_counter[0][1]}^{exp_counter[0][0]}={exp_counter[0][3]}" if exp_counter else None
}

# Tetration: ^b a != ^a b in general
tet_counter = []
for a in range(2, 4):
    for b in range(2, 4):
        if a != b:
            ta = tetration(a, b)
            tb = tetration(b, a)
            if ta != tb:
                tet_counter.append((a, b, ta, tb))
tet_comm = len(tet_counter) == 0
print(f"  Tetration commutative:      {tet_comm}")
if tet_counter:
    ex = tet_counter[0]
    print(f"    Counterexample: ^{ex[1]}{ex[0]}={ex[2]} != ^{ex[0]}{ex[1]}={ex[3]}")
comm_results["tetration"] = {
    "commutative": tet_comm,
    "counterexample": f"tet({tet_counter[0][0]},{tet_counter[0][1]})={tet_counter[0][2]} vs tet({tet_counter[0][1]},{tet_counter[0][0]})={tet_counter[0][3]}" if tet_counter else None
}

results["commutativity"] = comm_results

# ============================================================
# Invertibility
# ============================================================
print("\n--- Invertibility Tests ---")

inv_results = {}

# Addition -> Subtraction: works on Z
print("  Addition inverse (subtraction): works on integers")
inv_results["addition"] = {
    "inverse_operation": "subtraction",
    "domain_for_closure": "integers (Z)",
    "universal_on_naturals": False,
    "universal_on_integers": True,
    "note": "3 - 5 = -2, not a natural number"
}

# Multiplication -> Division: works on Q\{0}
print("  Multiplication inverse (division): works on rationals \\ {0}")
inv_results["multiplication"] = {
    "inverse_operation": "division",
    "domain_for_closure": "rationals (Q \\ {0})",
    "universal_on_naturals": False,
    "universal_on_integers": False,
    "universal_on_rationals_nonzero": True,
    "note": "5 / 3 not an integer; division by 0 undefined"
}

# Exponentiation -> Logarithm: works on R+
# Also roots: b-th root of a^b = a
print("  Exponentiation inverse (logarithm/root): works on positive reals")
# Check: log_a(a^b) = b
log_checks = []
for a in [2, 3, 5, 7, 10]:
    for b in [1, 2, 3, 4, 5]:
        val = math.log(a**b) / math.log(a)
        log_checks.append(abs(val - b) < 1e-10)
print(f"    log_a(a^b) = b verified: {all(log_checks)}")
inv_results["exponentiation"] = {
    "inverse_operations": ["logarithm (inverts exponent)", "root (inverts base)"],
    "domain_for_closure": "positive reals (R+)",
    "universal_on_naturals": False,
    "universal_on_positive_reals": True,
    "note": "log not defined for negative/zero; even roots of negatives problematic in R"
}

# Tetration -> Super-logarithm: problematic
print("  Tetration inverse (super-logarithm): problematic")
# slog_a(tetration(a, b)) should = b
# Only defined for integer heights; extension to reals is non-unique
inv_results["tetration"] = {
    "inverse_operation": "super-logarithm (slog)",
    "domain_for_closure": "ill-defined for non-integer heights",
    "universal": False,
    "note": "No standard unique extension of tetration to real/complex heights; multiple competing proposals exist"
}

results["invertibility"] = inv_results

# ============================================================
# Growth rates: f(n, n) for n = 1..10
# ============================================================
print("\n--- Growth Rates: f(n, n) ---")

growth = {"n": list(range(1, 11))}

# Addition: n + n = 2n
growth["addition_n_plus_n"] = [n + n for n in range(1, 11)]
print(f"  n + n:   {growth['addition_n_plus_n']}")

# Multiplication: n * n = n^2
growth["multiplication_n_times_n"] = [n * n for n in range(1, 11)]
print(f"  n * n:   {growth['multiplication_n_times_n']}")

# Exponentiation: n^n
growth["exponentiation_n_to_n"] = [n**n for n in range(1, 11)]
print(f"  n ^ n:   {growth['exponentiation_n_to_n']}")

# Tetration: ^n n (only small values computable)
tet_values = []
for n in range(1, 11):
    try:
        if n <= 3:
            val = tetration(n, n)
            tet_values.append(val)
        elif n == 4:
            # 4^^4 = 4^(4^(4^4)) = 4^(4^256) which is astronomical
            tet_values.append("4^(4^256) ~ 10^(10^153)")
        else:
            tet_values.append("OVERFLOW (hyper-exponential)")
    except (OverflowError, RecursionError):
        tet_values.append("OVERFLOW")

growth["tetration_n_tet_n"] = [str(v) for v in tet_values]
print(f"  n ^^ n:  {growth['tetration_n_tet_n']}")

results["growth_rates"] = growth

# ============================================================
# Tetration: qualitative difference
# ============================================================
print("\n--- Tetration: Qualitative Differences ---")

tet_analysis = {}

# Loss of commutativity (already shown above)
tet_analysis["commutativity_lost"] = True
tet_analysis["commutativity_example"] = "2^^3 = 2^(2^2) = 16, but 3^^2 = 3^3 = 27"

# Loss of associativity
# (2^^2)^^2 vs 2^^(2^^2) = 2^^4
t1 = tetration(tetration(2, 2), 2)  # (2^^2)^^2 = 4^^2 = 4^4 = 256
t2_inner = tetration(2, 2)  # 2^^2 = 4
t2 = tetration(2, t2_inner)  # 2^^4 = 2^(2^(2^2)) = 2^(2^4) = 2^16 = 65536
tet_analysis["associativity_lost"] = True
tet_analysis["associativity_example"] = f"(2^^2)^^2 = {t1}, 2^^(2^^2) = 2^^4 = {t2}"
print(f"  Associativity: (2^^2)^^2 = {t1}, 2^^(2^^2) = {t2}")

# Loss of general invertibility
tet_analysis["invertibility_problematic"] = True
tet_analysis["reason"] = "No unique extension to non-integer heights; super-logarithm and super-root are not uniquely defined"

# Tetration values for small n
tet_small = {}
for b in range(0, 6):
    for a in [2, 3]:
        try:
            val = tetration(a, b)
            if isinstance(val, str):
                tet_small[f"{a}^^{b}"] = val
            elif isinstance(val, int) and val < 10**15:
                tet_small[f"{a}^^{b}"] = val
            elif isinstance(val, int) and val > 0:
                tet_small[f"{a}^^{b}"] = f"~10^{math.log10(float(val)):.1f}"
            else:
                tet_small[f"{a}^^{b}"] = str(val)
        except (OverflowError, ValueError):
            tet_small[f"{a}^^{b}"] = "OVERFLOW"

tet_analysis["small_values"] = {k: str(v) for k, v in tet_small.items()}
print(f"  Small tetration values: {tet_small}")

results["tetration_analysis"] = tet_analysis

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: Recursive Closure Hierarchy")
print("=" * 60)

summary_table = {
    "Level 0 - Successor": {"commutative": "N/A (unary)", "invertible": "Yes (predecessor)", "growth": "O(1)"},
    "Level 1 - Addition": {"commutative": "Yes", "invertible": "Yes (on Z)", "growth": "O(n)"},
    "Level 2 - Multiplication": {"commutative": "Yes", "invertible": "Yes (on Q\\{0})", "growth": "O(n^2)"},
    "Level 3 - Exponentiation": {"commutative": "No", "invertible": "Yes (on R+)", "growth": "O(n^n)"},
    "Level 4 - Tetration": {"commutative": "No", "invertible": "Problematic", "growth": "hyper-exponential"},
}

for level, props in summary_table.items():
    print(f"  {level}: {props}")

results["summary"] = summary_table

# Save
outpath = os.path.join(OUTPUT_DIR, "test3_recursive_closure.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {outpath}")
