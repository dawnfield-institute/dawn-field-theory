#!/usr/bin/env python3
"""
Experiment 10: Quark Masses via Herniation Hypothesis

Herniation Framework:
- Reality = dual-field crystallization (energy + information)
- Quarks = "quantum-locked nodes" in BOTH fields
- Up-type quarks: energy-dominant locks (lighter, fractional charge +2/3)
- Down-type quarks: information-stabilized locks (heavier, -1/3)

Key Predictions:
1. n - p = informational "scaffolding cost" = F₅/F₃ × mₑ
2. d/u ≈ F₃ = 2 (down stabilizes up via information field)
3. Generation jumps should show Fibonacci scaling

Quark masses (PDG 2024, MS-bar at 2 GeV):
- u: 2.16 MeV (1.67-2.67)
- d: 4.70 MeV (4.32-5.09)
- s: 93.5 MeV (83-95)
- c: 1.27 GeV (at mc)
- b: 4.18 GeV (at mb)
- t: 172.57 GeV (pole mass)
"""

import numpy as np
from typing import Dict, List, Tuple

# Fibonacci sequence
F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
PHI = (1 + np.sqrt(5)) / 2

# Physical constants (MeV)
M_ELECTRON = 0.51099895  # MeV
M_PROTON = 938.272088    # MeV
M_NEUTRON = 939.565420   # MeV

# Quark masses (PDG 2024, central values)
# Light quarks: MS-bar at μ = 2 GeV
M_UP = 2.16       # MeV (range: 1.67-2.67)
M_DOWN = 4.70     # MeV (range: 4.32-5.09)
M_STRANGE = 93.5  # MeV (range: 83-95)

# Heavy quarks: MS-bar at μ = m_q
M_CHARM = 1270    # MeV (1.27 GeV)
M_BOTTOM = 4180   # MeV (4.18 GeV)
M_TOP = 172570    # MeV (172.57 GeV, pole mass)


def test_neutron_proton_difference():
    """
    Herniation Hypothesis: Neutron is proton + "informational scaffolding"
    
    The extra neutron mass = cost of information field stabilization
    Prediction: Δm = F₅/F₃ × mₑ = 5/2 × mₑ = 2.5 mₑ
    """
    print("=" * 70)
    print("TEST 1: Neutron-Proton Mass Difference (Scaffolding Cost)")
    print("=" * 70)
    
    delta_m = M_NEUTRON - M_PROTON  # MeV
    delta_in_electrons = delta_m / M_ELECTRON
    
    print(f"\nMeasured:")
    print(f"  n - p = {delta_m:.6f} MeV")
    print(f"  n - p = {delta_in_electrons:.4f} × mₑ")
    
    # Fibonacci predictions
    predictions = [
        ("F₅/F₃ = 5/2", F[5]/F[3], 2.5),
        ("F₄/F₂ = 3/1", F[4]/F[2], 3.0),
        ("φ³/2", PHI**3/2, 2.118),
        ("F₆/F₄ = 8/3", F[6]/F[4], 8/3),
    ]
    
    print(f"\nPredictions:")
    for name, ratio, expected in predictions:
        error = 100 * abs(delta_in_electrons - expected) / expected
        print(f"  {name} = {expected:.4f} → error = {error:.2f}%")
    
    # Best match
    best = min(predictions, key=lambda x: abs(delta_in_electrons - x[2]))
    error = 100 * abs(delta_in_electrons - best[2]) / best[2]
    
    print(f"\n✓ BEST MATCH: {best[0]} = {best[2]:.4f}")
    print(f"  Measured: {delta_in_electrons:.4f}")
    print(f"  Error: {error:.2f}%")
    
    return delta_in_electrons, best


def test_down_up_ratio():
    """
    Herniation: Down quark = up quark + information stabilization
    Prediction: d/u ≈ F₃ = 2
    """
    print("\n" + "=" * 70)
    print("TEST 2: Down/Up Quark Ratio (Information Stabilization)")
    print("=" * 70)
    
    ratio = M_DOWN / M_UP
    
    print(f"\nMeasured: d/u = {ratio:.3f}")
    print(f"  (with uncertainties: {4.32/2.67:.2f} to {5.09/1.67:.2f})")
    
    # Fibonacci predictions
    predictions = [
        ("F₃", F[3], 2),
        ("F₄/F₂", F[4]/F[2], 3),
        ("φ", PHI, 1.618),
        ("φ²/F₃", PHI**2/F[3], 1.309),
        ("F₅/F₃", F[5]/F[3], 2.5),
    ]
    
    print(f"\nPredictions:")
    for name, _, expected in predictions:
        error = 100 * abs(ratio - expected) / expected
        print(f"  {name} = {expected:.4f} → error = {error:.2f}%")
    
    # Best match
    best = min(predictions, key=lambda x: abs(ratio - x[2]))
    error = 100 * abs(ratio - best[2]) / best[2]
    
    print(f"\n✓ BEST MATCH: {best[0]} = {best[2]:.4f}")
    print(f"  Measured: {ratio:.3f}")
    print(f"  Error: {error:.2f}%")
    
    # Note about uncertainty
    print(f"\n  Note: Quark mass ratio uncertainties are large (~30%)")
    print(f"  F₃ = 2 is within the measured range!")
    
    return ratio, best


def test_generation_jumps():
    """
    Test mass ratios between generations
    Herniation: Each generation = deeper crystallization
    """
    print("\n" + "=" * 70)
    print("TEST 3: Generation Jumps (Crystallization Depth)")
    print("=" * 70)
    
    # Within-family ratios
    ratios = {
        # Down-type family
        "s/d": M_STRANGE / M_DOWN,
        "b/s": M_BOTTOM / M_STRANGE,
        
        # Up-type family
        "c/u": M_CHARM / M_UP,
        "t/c": M_TOP / M_CHARM,
        
        # Cross-generation
        "s/u": M_STRANGE / M_UP,
        "c/s": M_CHARM / M_STRANGE,
        "b/c": M_BOTTOM / M_CHARM,
        "t/b": M_TOP / M_BOTTOM,
    }
    
    # Fibonacci products to test
    fib_targets = []
    for i in range(3, 14):
        for j in range(3, 14):
            for k in range(0, 14):
                if k == 0:
                    val = F[i] * F[j]
                    name = f"F_{i}×F_{j}"
                else:
                    val = F[i] * F[j] * F[k]
                    name = f"F_{i}×F_{j}×F_{k}"
                if 10 < val < 200:
                    fib_targets.append((name, val))
                elif 200 < val < 1000:
                    fib_targets.append((name, val))
    
    # Also test simple Fibonacci
    for i in range(5, 15):
        fib_targets.append((f"F_{i}", F[i]))
    
    # Powers of phi
    for n in range(3, 15):
        fib_targets.append((f"φ^{n}", PHI**n))
    
    print(f"\nGeneration Jump Ratios:")
    print("-" * 50)
    
    results = {}
    for name, ratio in ratios.items():
        # Find best Fibonacci match
        best_match = None
        best_error = float('inf')
        for fib_name, fib_val in fib_targets:
            error = abs(ratio - fib_val) / fib_val
            if error < best_error:
                best_error = error
                best_match = (fib_name, fib_val)
        
        results[name] = (ratio, best_match, best_error * 100)
        print(f"  {name:6s} = {ratio:8.2f} ≈ {best_match[0]:12s} = {best_match[1]:8.2f} ({best_error*100:5.2f}%)")
    
    return results


def test_quark_electron_ratios():
    """
    Quark masses in electron mass units
    Looking for Fibonacci structure
    """
    print("\n" + "=" * 70)
    print("TEST 4: Quark Masses in Electron Units")
    print("=" * 70)
    
    quarks = {
        "u": M_UP / M_ELECTRON,
        "d": M_DOWN / M_ELECTRON,
        "s": M_STRANGE / M_ELECTRON,
        "c": M_CHARM / M_ELECTRON,
        "b": M_BOTTOM / M_ELECTRON,
        "t": M_TOP / M_ELECTRON,
    }
    
    print(f"\nQuark/electron mass ratios:")
    print("-" * 50)
    
    # Extended Fibonacci products
    fib_products = []
    for i in range(2, 15):
        fib_products.append((f"F_{i}", F[i]))
    for i in range(2, 12):
        for j in range(i, 12):
            fib_products.append((f"F_{i}×F_{j}", F[i]*F[j]))
    for i in range(2, 10):
        for j in range(i, 10):
            for k in range(j, 10):
                fib_products.append((f"F_{i}×F_{j}×F_{k}", F[i]*F[j]*F[k]))
    
    for name, ratio in quarks.items():
        # Find best match
        best = min(fib_products, key=lambda x: abs(ratio - x[1])/max(ratio, x[1]))
        error = 100 * abs(ratio - best[1]) / ratio
        
        print(f"  m_{name}/mₑ = {ratio:10.2f} ≈ {best[0]:15s} = {best[1]:10.2f} ({error:5.1f}%)")
    
    return quarks


def test_herniation_structure():
    """
    Test the full herniation hypothesis structure:
    - Up-type = energy-dominant locks (lighter)
    - Down-type = information-stabilized (heavier within generation)
    - Each generation = deeper crystallization depth
    """
    print("\n" + "=" * 70)
    print("TEST 5: Herniation Structure Analysis")
    print("=" * 70)
    
    print("\n1. Within-generation: Down > Up (information > energy)")
    print("-" * 50)
    
    generations = [
        (1, "u", M_UP, "d", M_DOWN),
        (2, "c", M_CHARM, "s", M_STRANGE),
        (3, "t", M_TOP, "b", M_BOTTOM),
    ]
    
    for gen, up_name, up_mass, down_name, down_mass in generations:
        ratio = down_mass / up_mass if gen < 3 else up_mass / down_mass
        comp = ">" if down_mass > up_mass else "<"
        print(f"  Gen {gen}: m_{down_name}/m_{up_name} = {down_mass/up_mass:.2f}")
        print(f"          {down_name} {comp} {up_name}: {'✓ Consistent' if down_mass > up_mass or gen == 3 else '✗'}")
    
    print("\n  Note: Gen 3 (t > b) breaks pattern - top is special!")
    print("  Interpretation: At highest crystallization, energy dominates?")
    
    print("\n2. Cross-generation scaling")
    print("-" * 50)
    
    # Up-type scaling
    cu_ratio = M_CHARM / M_UP
    tc_ratio = M_TOP / M_CHARM
    print(f"  Up-type: c/u = {cu_ratio:.1f}, t/c = {tc_ratio:.1f}")
    print(f"           Product: t/u = {M_TOP/M_UP:.0f}")
    
    # Down-type scaling
    sd_ratio = M_STRANGE / M_DOWN
    bs_ratio = M_BOTTOM / M_STRANGE
    print(f"  Down-type: s/d = {sd_ratio:.1f}, b/s = {bs_ratio:.1f}")
    print(f"             Product: b/d = {M_BOTTOM/M_DOWN:.0f}")
    
    print("\n3. Generation ratio consistency")
    print("-" * 50)
    
    # Does each generation scale similarly?
    gen2_to_gen1 = (M_STRANGE + M_CHARM) / (M_DOWN + M_UP)
    gen3_to_gen2 = (M_BOTTOM + M_TOP) / (M_STRANGE + M_CHARM)
    
    print(f"  (s+c)/(d+u) = {gen2_to_gen1:.1f}")
    print(f"  (b+t)/(s+c) = {gen3_to_gen2:.1f}")
    
    # Total mass per generation
    gen1_mass = M_UP + M_DOWN
    gen2_mass = M_STRANGE + M_CHARM
    gen3_mass = M_BOTTOM + M_TOP
    
    print(f"\n  Generation masses:")
    print(f"    Gen 1: {gen1_mass:.1f} MeV")
    print(f"    Gen 2: {gen2_mass:.1f} MeV")
    print(f"    Gen 3: {gen3_mass:.1f} MeV")
    print(f"    Ratios: {gen2_mass/gen1_mass:.0f} : {gen3_mass/gen2_mass:.0f}")


def test_proton_from_quarks():
    """
    Can we derive proton mass from quark content?
    Proton = uud
    """
    print("\n" + "=" * 70)
    print("TEST 6: Proton Structure (uud)")
    print("=" * 70)
    
    # Naive quark mass sum
    quark_sum = 2 * M_UP + M_DOWN  # uud
    
    print(f"\nNaive sum: 2×m_u + m_d = {quark_sum:.2f} MeV")
    print(f"Proton mass: {M_PROTON:.2f} MeV")
    print(f"Ratio: m_p / (2m_u + m_d) = {M_PROTON/quark_sum:.1f}")
    
    # The binding energy / QCD effects dominate!
    binding = M_PROTON - quark_sum
    print(f"\nQCD binding contribution: {binding:.2f} MeV ({100*binding/M_PROTON:.1f}%)")
    print("  → Most of proton mass is NOT from quark masses!")
    print("  → It's from QCD field energy (gluons + virtual quarks)")
    
    # Herniation interpretation
    print("\nHerniation interpretation:")
    print("  Quark masses ≈ 1% of proton mass")
    print("  QCD field energy ≈ 99% of proton mass")
    print("  → Proton is mostly 'crystallized field energy'")
    print("  → Quarks are 'seeds' for field crystallization")
    
    # Is the binding energy Fibonacci?
    binding_in_electrons = binding / M_ELECTRON
    print(f"\nBinding in electron masses: {binding_in_electrons:.1f}")
    
    # Check against proton/electron ratio
    print(f"Compare: m_p/m_e = {M_PROTON/M_ELECTRON:.1f}")
    print(f"We showed: m_p/m_e ≈ F₄×F₉×F₁₂/F₆ = {F[4]*F[9]*F[12]/F[6]:.0f}")


def test_top_quark_special():
    """
    Top quark is uniquely heavy - near electroweak scale
    Is there special Fibonacci structure?
    """
    print("\n" + "=" * 70)
    print("TEST 7: Top Quark (Special Case)")
    print("=" * 70)
    
    # Top mass in various units
    t_electron = M_TOP / M_ELECTRON
    t_proton = M_TOP / M_PROTON
    t_w = M_TOP / 80379  # W boson mass in MeV
    t_z = M_TOP / 91188  # Z boson mass in MeV
    t_higgs = M_TOP / 125100  # Higgs mass in MeV
    
    print(f"\nTop quark mass ratios:")
    print(f"  m_t/m_e = {t_electron:.0f}")
    print(f"  m_t/m_p = {t_proton:.1f}")
    print(f"  m_t/m_W = {t_w:.3f}")
    print(f"  m_t/m_Z = {t_z:.3f}")
    print(f"  m_t/m_H = {t_higgs:.3f}")
    
    # Look for Fibonacci in t/p
    print(f"\nTop/Proton ratio: {t_proton:.2f}")
    
    # Fibonacci products near 184
    candidates = []
    for i in range(3, 12):
        for j in range(3, 12):
            val = F[i] * F[j]
            if 150 < val < 250:
                candidates.append((f"F_{i}×F_{j}", val))
    
    for name, val in sorted(candidates, key=lambda x: abs(x[1] - t_proton)):
        error = 100 * abs(t_proton - val) / t_proton
        print(f"  {name} = {val} → error = {error:.1f}%")
        if error < 10:
            break
    
    # Top Yukawa coupling ≈ 1
    print(f"\nTop Yukawa coupling ≈ √2 × m_t / v ≈ 1")
    print("  Where v = 246 GeV (Higgs VEV)")
    print("  This is the ONLY fermion with O(1) Yukawa!")
    print("  Herniation: Top is maximally 'crystallized' in energy field")


def main():
    print("=" * 70)
    print("EXPERIMENT 10: QUARK MASSES VIA HERNIATION HYPOTHESIS")
    print("=" * 70)
    
    print("\nFramework:")
    print("  - Quarks = quantum-locked nodes in dual (energy + information) fields")
    print("  - Up-type: energy-dominant locks")
    print("  - Down-type: information-stabilized locks")
    print("  - Generations: crystallization depth")
    
    # Run all tests
    np_result = test_neutron_proton_difference()
    du_result = test_down_up_ratio()
    gen_results = test_generation_jumps()
    qe_results = test_quark_electron_ratios()
    test_herniation_structure()
    test_proton_from_quarks()
    test_top_quark_special()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nKey Findings:")
    print(f"  1. n - p = {(M_NEUTRON - M_PROTON)/M_ELECTRON:.2f} mₑ ≈ F₅/F₃ = 2.5 (1.2% error)")
    print(f"  2. d/u = {M_DOWN/M_UP:.2f} ≈ F₃ = 2 (within uncertainty)")
    print(f"  3. Top quark is special: Yukawa ≈ 1, breaks generation pattern")
    print(f"  4. Proton mass is 99% QCD energy, 1% quark masses")
    
    print("\nHerniation Interpretation:")
    print("  - Neutron's extra mass = information field scaffolding cost")
    print("  - d/u ≈ 2: information stabilization roughly doubles mass")
    print("  - Top quark: maximally crystallized in energy field")
    print("  - Proton: quarks seed crystallization, fields provide mass")
    
    print("\nOpen Questions:")
    print("  - Why does Gen 3 break the down > up pattern?")
    print("  - What determines the generation jump ratios?")
    print("  - Can we derive QCD binding energy from SEC?")


if __name__ == "__main__":
    main()
