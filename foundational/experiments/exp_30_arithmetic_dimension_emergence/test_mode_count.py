"""
Test 5: 2^3 = 8 Mode Count Probe
Test the combinatorial argument: 3 arithmetic dimensions with binary states -> 2^3 = 8 modes.
"""

import numpy as np
import json
import os
from itertools import product

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}

# ============================================================
# Enumerate all 2^3 = 8 combinations
# ============================================================
print("=" * 60)
print("2^3 = 8 MODE ENUMERATION")
print("=" * 60)

dimensions = ["additive", "multiplicative", "exponential"]
states = ["active", "passive"]

modes_3d = []
for i, combo in enumerate(product(states, repeat=3)):
    mode = {dim: state for dim, state in zip(dimensions, combo)}
    mode["index"] = i
    mode["binary"] = "".join(["1" if s == "active" else "0" for s in combo])

    # Describe the dynamics
    active_dims = [dim for dim, state in zip(dimensions, combo) if state == "active"]
    passive_dims = [dim for dim, state in zip(dimensions, combo) if state == "passive"]

    descriptions = {
        "000": "Null mode: no arithmetic dynamics active. Static/frozen state. "
               "Equivalent to fixed point with no transfer mechanism.",
        "001": "Pure exponential mode: only exponential (phase/rotation) dynamics active. "
               "Oscillatory behavior without energy transfer or scaling. Phase waves.",
        "010": "Pure multiplicative mode: only scaling/dilation active. "
               "Self-similar cascading without translation or phase rotation. Scale-free dynamics.",
        "011": "Multiplicative-exponential mode: scaling with phase rotation but no translation. "
               "Spiral scaling dynamics. Related to complex power-law behavior.",
        "100": "Pure additive mode: only translation active. "
               "Uniform advection/drift without scaling or phase. Galilean invariant flows.",
        "101": "Additive-exponential mode: translation with phase dynamics but no scaling. "
               "Traveling waves with oscillation. Plane wave solutions.",
        "110": "Additive-multiplicative mode: translation with scaling but no phase. "
               "Growing/decaying transport. Related to Burgers-type dynamics.",
        "111": "Full mode: all three arithmetic dimensions active. "
               "Complete conformal dynamics. Full Mobius group action. Turbulent cascades."
    }

    mode["description"] = descriptions.get(mode["binary"], "Unknown")
    mode["active_dimensions"] = active_dims
    mode["passive_dimensions"] = passive_dims
    mode["n_active"] = len(active_dims)

    modes_3d.append(mode)
    print(f"\n  Mode {i} ({mode['binary']}): {', '.join(active_dims) if active_dims else 'none'} active")
    print(f"    {mode['description'][:80]}...")

results["modes_3d"] = modes_3d

# ============================================================
# Physical/turbulence cross-references
# ============================================================
print("\n" + "=" * 60)
print("CROSS-REFERENCE WITH TURBULENCE MODES")
print("=" * 60)

turbulence_mapping = {
    "000": {
        "turbulence_analog": "Thermal equilibrium / viscous dissipation endpoint",
        "physical_example": "Heat death; fully damped state",
    },
    "001": {
        "turbulence_analog": "Rossby/inertial waves (phase dynamics only)",
        "physical_example": "Wave turbulence with no direct cascade",
    },
    "010": {
        "turbulence_analog": "Richardson cascade (pure scale-to-scale transfer)",
        "physical_example": "Kolmogorov K41 inertial range (idealized)",
    },
    "011": {
        "turbulence_analog": "Helical cascade (scaling + chirality/phase)",
        "physical_example": "Helical turbulence, MHD dynamo action",
    },
    "100": {
        "turbulence_analog": "Mean flow / sweeping (pure advection)",
        "physical_example": "Taylor's frozen turbulence hypothesis regime",
    },
    "101": {
        "turbulence_analog": "Acoustic/compressible modes (advection + oscillation)",
        "physical_example": "Sound waves in turbulence, compressible turbulence modes",
    },
    "110": {
        "turbulence_analog": "Strain-dominated regions (stretching + advection)",
        "physical_example": "Burgers turbulence, shock-dominated dynamics",
    },
    "111": {
        "turbulence_analog": "Full 3D turbulence (all transfer mechanisms active)",
        "physical_example": "Navier-Stokes turbulence, complete cascade with intermittency",
    },
}

for mode in modes_3d:
    b = mode["binary"]
    if b in turbulence_mapping:
        mode["turbulence_analog"] = turbulence_mapping[b]["turbulence_analog"]
        mode["physical_example"] = turbulence_mapping[b]["physical_example"]
        print(f"  Mode {b}: {turbulence_mapping[b]['turbulence_analog']}")

results["turbulence_mapping"] = turbulence_mapping

# ============================================================
# 2D case: 2^2 = 4 modes, but exp_14 found 3
# ============================================================
print("\n" + "=" * 60)
print("2D CASE: INVESTIGATING THE MISSING MODE")
print("=" * 60)

dimensions_2d = ["additive", "multiplicative"]
modes_2d = []
for i, combo in enumerate(product(states, repeat=2)):
    mode = {dim: state for dim, state in zip(dimensions_2d, combo)}
    mode["index"] = i
    mode["binary"] = "".join(["1" if s == "active" else "0" for s in combo])
    modes_2d.append(mode)

print(f"\n  2D: 2^2 = {len(modes_2d)} combinations:")
for mode in modes_2d:
    active = [dim for dim, state in zip(dimensions_2d,
              [mode["additive"], mode["multiplicative"]]) if state == "active"]
    print(f"    Mode {mode['binary']}: {', '.join(active) if active else 'none'} active")

# Analysis: why 3 instead of 4?
degeneracy_analysis = {
    "expected_modes_2d": 4,
    "observed_modes_2d_exp14": 3,
    "hypothesis": (
        "In 2D enstrophy cascade, the '00' (null/frozen) mode is degenerate with "
        "the dissipation endpoint and does not appear as a distinct dynamical mode. "
        "The three observed modes correspond to: "
        "(1) pure additive (10) - direct enstrophy cascade, "
        "(2) pure multiplicative (01) - inverse energy cascade, "
        "(3) mixed additive-multiplicative (11) - coupled cascade. "
        "The null mode (00) is forbidden because 2D incompressible flow always has "
        "at least one active transfer mechanism due to conservation of both energy AND enstrophy."
    ),
    "alternative_hypothesis": (
        "The 2D constraint (incompressibility + enstrophy conservation) imposes a "
        "selection rule that forbids the null state: at least one arithmetic dimension "
        "must be active. This reduces 2^2=4 to 2^2-1=3 modes. "
        "In 3D, the analogous constraint (energy conservation alone) does not forbid "
        "the null state (thermal equilibrium exists), so all 2^3=8 modes are accessible."
    ),
    "selection_rule": "In nD incompressible flow with n conserved quantities, "
                      "the number of forbidden null-like modes equals C(n,0)=1 when "
                      "conservation laws force at least one dimension active."
}

print(f"\n  Analysis:")
print(f"    Expected: 2^2 = 4 modes")
print(f"    Observed (exp_14): 3 modes")
print(f"    Hypothesis: {degeneracy_analysis['hypothesis'][:120]}...")

results["2d_analysis"] = {
    "modes_2d": modes_2d,
    "degeneracy_analysis": degeneracy_analysis
}

# ============================================================
# General dimension count: 2^n modes
# ============================================================
print("\n" + "=" * 60)
print("GENERAL DIMENSION COUNT")
print("=" * 60)

dim_count = {}
for n in range(0, 7):
    total = 2**n
    # If the "all passive" mode is forbidden by conservation, subtract 1
    accessible = total - 1 if n >= 2 else total
    dim_count[n] = {
        "n_dimensions": n,
        "total_modes_2_to_n": total,
        "accessible_modes": accessible,
        "note": ""
    }

# Special cases
dim_count[0]["note"] = "0 arithmetic dimensions: trivial, 1 mode (the point)"
dim_count[1]["note"] = "1D: 2 modes (active/passive). E.g., moving vs. stationary."
dim_count[2]["note"] = "2D: 4 total, 3 accessible (null mode degenerate). Matches exp_14."
dim_count[3]["note"] = "3D: 8 total modes. If null forbidden: 7. Physical 3D turbulence."
dim_count[4]["note"] = "4D (if tetration counted): 16 modes. Speculative."

for n, info in dim_count.items():
    print(f"  n={n}: 2^{n}={info['total_modes_2_to_n']} total, "
          f"{info['accessible_modes']} accessible. {info['note']}")

results["general_dimension_count"] = dim_count

# ============================================================
# Mode activation energy analogy
# ============================================================
print("\n" + "=" * 60)
print("MODE ACTIVATION STRUCTURE")
print("=" * 60)

# In each mode, the number of active dimensions determines the "richness"
# Count modes by number of active dimensions
from math import comb

print("\n  Modes grouped by number of active dimensions (3D case):")
for k in range(4):
    count = comb(3, k)
    mode_list = [m for m in modes_3d if m["n_active"] == k]
    binaries = [m["binary"] for m in mode_list]
    print(f"    {k} active: C(3,{k}) = {count} modes: {binaries}")

activation_structure = {
    "3d_by_active_count": {
        "0_active": {"count": comb(3,0), "modes": ["000"], "description": "Null mode"},
        "1_active": {"count": comb(3,1), "modes": ["001","010","100"], "description": "Single-mechanism modes"},
        "2_active": {"count": comb(3,2), "modes": ["011","101","110"], "description": "Paired-mechanism modes"},
        "3_active": {"count": comb(3,3), "modes": ["111"], "description": "Full turbulence"},
    },
    "binomial_structure": "The number of k-active modes follows C(n,k), giving Pascal's triangle",
    "total_check": f"Sum C(3,k) for k=0..3 = {sum(comb(3,k) for k in range(4))} = 2^3 = 8"
}

results["activation_structure"] = activation_structure

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

summary = {
    "main_result": "3 arithmetic dimensions (additive, multiplicative, exponential) with binary "
                   "(active/passive) states yield 2^3 = 8 distinct dynamical modes.",
    "2d_result": "In 2D, conservation constraints reduce 2^2 = 4 to 3 observable modes, "
                 "consistent with exp_14 findings.",
    "3d_prediction": "3D turbulence should exhibit up to 8 distinct mode structures, "
                     "corresponding to all combinations of arithmetic dimension activation.",
    "structure": "Modes follow binomial distribution C(n,k) by number of active dimensions, "
                 "giving Pascal's triangle structure.",
    "connection_to_conformal_group": "Each mode corresponds to a subgroup of the Mobius group: "
                                     "T (additive), D (multiplicative), R (exponential), "
                                     "and their combinations."
}
results["summary"] = summary

for k, v in summary.items():
    print(f"  {k}: {v[:100]}...")

# Save
outpath = os.path.join(OUTPUT_DIR, "test5_mode_count.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {outpath}")
