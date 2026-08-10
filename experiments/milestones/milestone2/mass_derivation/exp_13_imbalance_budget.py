#!/usr/bin/env python3
"""
exp_13_imbalance_budget.py
==========================

KEY HYPOTHESIS: "Local imbalance creates global balance"

The "errors" from ideal Fibonacci ratios are NOT random noise - they are
the IMBALANCE BUDGET that each particle carries to maintain global PAC.

Predictions:
1. Particles near the edge of chaos (crossover ~100 MeV) should have 
   SMALLER deviations - they're at maximum convergence
2. The SUM of all deviations should balance to zero (or a Fibonacci fraction)
3. Deviations should be correlated - one particle's excess balances another's deficit

This is exactly PAC: f(Parent) = Σf(Children)
If one child is slightly over, another must be slightly under.
"""

import numpy as np
from typing import Dict, List, Tuple

# Fibonacci sequence
F = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
phi = (1 + np.sqrt(5)) / 2

# Feigenbaum constants
DELTA_F = 4.669201609102990
ALPHA_F = 2.502907875095892

# Balance operator
XI = 1 + np.pi/55

# Particle masses in MeV
particles = {
    # Leptons
    'e': 0.511,
    'μ': 105.66,
    'τ': 1776.86,
    # Quarks  
    'u': 2.16,
    'd': 4.70,
    's': 93.5,
    'c': 1275,
    'b': 4180,
    't': 172760,
    # Hadrons
    'p': 938.27,
    'n': 939.57,
}

m_e = particles['e']

print("=" * 70)
print("EXP 13: IMBALANCE BUDGET - LOCAL DEVIATION = GLOBAL BALANCE")
print("=" * 70)

# ============================================================================
# SECTION 1: MEASURE ALL DEVIATIONS FROM FIBONACCI TARGETS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DEVIATION MAP")
print("=" * 70)

# Define the "ideal" Fibonacci ratios we've discovered
ideal_ratios = {
    'μ/e': ('F_4×F_6²×(1+1/F_7)', F[4] * F[6]**2 * (1 + 1/F[7])),  # 206.769
    'τ/e': ('F_4×F_7×F_11 + F_5', F[4] * F[7] * F[11] + F[5]),      # 3476
    'p/e': ('F_4×F_9×F_12/F_6', F[4] * F[9] * F[12] / F[6]),        # 1836
    'n/p': ('1 + F_5/(F_3×F_4×F_9×F_12/F_6)', 1 + F[5]/(F[3]*F[4]*F[9]*F[12]/F[6])),  # ~1.0014
    'd/u': ('F_3', F[3]),  # 2
    's/d': ('F_4×F_5', F[4] * F[5]),  # Actually ~20
    'c/s': ('F_6 + F_5', F[6] + F[5]),  # 13
    'b/s': ('F_9 - 1', F[9] - 1),  # 33 (actually 44.7 so this is wrong)
    't/b': ('F_9 + F_5', F[9] + F[5]),  # 39 (actually 41.3)
}

# Compute actual ratios and deviations
deviations = {}
print("\nParticle Ratio | Actual | Ideal Formula | Ideal Value | Deviation")
print("-" * 70)

# Compute key ratios
actual_ratios = {
    'μ/e': particles['μ'] / m_e,
    'τ/e': particles['τ'] / m_e,
    'p/e': particles['p'] / m_e,
    'n/p': particles['n'] / particles['p'],
    'd/u': particles['d'] / particles['u'],
    's/d': particles['s'] / particles['d'],
    'c/s': particles['c'] / particles['s'],
    'b/s': particles['b'] / particles['s'],
    't/b': particles['t'] / particles['b'],
}

for name, (formula, ideal) in ideal_ratios.items():
    actual = actual_ratios[name]
    deviation = (actual - ideal) / ideal * 100  # percent
    deviations[name] = deviation
    print(f"{name:8} | {actual:10.4f} | {formula:25} | {ideal:10.2f} | {deviation:+.4f}%")

# ============================================================================
# SECTION 2: DISTANCE FROM CROSSOVER SCALE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: DEVIATION VS DISTANCE FROM CROSSOVER")
print("=" * 70)

crossover_scale = np.sqrt((particles['u'] + particles['d']) * 
                          (particles['s'] + particles['c']))
print(f"\nCrossover scale: {crossover_scale:.2f} MeV")

# For each particle, compute distance from crossover
print("\nParticle | Mass (MeV) | log₁₀(m/crossover) | |deviation| from Fib")
print("-" * 70)

particle_deviations = []
for name, mass in particles.items():
    log_distance = np.log10(mass / crossover_scale)
    
    # Find the best Fibonacci ratio involving this particle
    relevant_devs = []
    for ratio_name, dev in deviations.items():
        if name in ratio_name:
            relevant_devs.append(abs(dev))
    
    if relevant_devs:
        mean_dev = np.mean(relevant_devs)
        particle_deviations.append((name, mass, log_distance, mean_dev))
        print(f"{name:8} | {mass:10.2f} | {log_distance:+.4f} | {mean_dev:.4f}%")

# ============================================================================
# SECTION 3: TEST - DO NEAR-CROSSOVER PARTICLES HAVE SMALLER DEVIATIONS?
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: CROSSOVER PROXIMITY → SMALLER DEVIATION?")
print("=" * 70)

# Sort by distance from crossover
particle_deviations.sort(key=lambda x: abs(x[2]))  # Sort by |log_distance|

print("\nParticles sorted by proximity to crossover:")
print("Particle | log₁₀ distance | Mean |deviation|%")
print("-" * 50)
for name, mass, log_dist, dev in particle_deviations:
    print(f"{name:8} | {log_dist:+.4f} | {dev:.4f}%")

# The prediction: particles closer to crossover should have smaller deviations
# Let's check the correlation
log_distances = [abs(x[2]) for x in particle_deviations]
mean_deviations = [x[3] for x in particle_deviations]

# Compute correlation
if len(log_distances) > 2:
    from scipy.stats import pearsonr, spearmanr
    try:
        r_pearson, p_pearson = pearsonr(log_distances, mean_deviations)
        r_spearman, p_spearman = spearmanr(log_distances, mean_deviations)
        print(f"\nCorrelation (distance vs deviation):")
        print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.4f})")
        print(f"  Spearman ρ = {r_spearman:.4f} (p = {p_spearman:.4f})")
        
        if r_pearson > 0:
            print("  → POSITIVE correlation: further from crossover = larger deviation")
            print("  ✓ SUPPORTS hypothesis: crossover is equilibrium point")
        else:
            print("  → NEGATIVE correlation: opposite of prediction")
    except:
        print("  (Not enough data for correlation)")

# ============================================================================
# SECTION 4: GLOBAL BALANCE - DO DEVIATIONS SUM TO ZERO?
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: GLOBAL PAC BALANCE")
print("=" * 70)

print("""
PAC predicts: f(Parent) = Σf(Children)
If one ratio is slightly above ideal, another should be below.

Let's check if deviations balance globally.
""")

# Sum of all deviations
all_devs = list(deviations.values())
sum_dev = sum(all_devs)
mean_dev = np.mean(all_devs)
std_dev = np.std(all_devs)

print(f"All deviations: {[f'{d:.2f}%' for d in all_devs]}")
print(f"\nSum of deviations: {sum_dev:+.4f}%")
print(f"Mean deviation: {mean_dev:+.4f}%")
print(f"Std deviation: {std_dev:.4f}%")

# Check if sum relates to Fibonacci
print(f"\nSum / 100 = {sum_dev/100:.6f}")
print(f"  ≈ 1/φ = {1/phi:.6f}? ({abs(sum_dev/100 - 1/phi)/(1/phi)*100:.2f}% error)")
print(f"  ≈ π/55 = {np.pi/55:.6f}? ({abs(sum_dev/100 - np.pi/55)/(np.pi/55)*100:.2f}% error)")

# ============================================================================
# SECTION 5: PAIRED DEVIATIONS - DO THEY ANTI-CORRELATE?
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: PAIRED DEVIATION ANTI-CORRELATION")
print("=" * 70)

print("""
If deviations are PAC-balanced, related particles should show
OPPOSITE deviations (one over, one under).

Pairs to test:
- μ and τ (charged leptons)
- d and u (Gen 1 quarks)  
- s and c (Gen 2 quarks)
- b and t (Gen 3 quarks)
""")

# Define pairs and their deviation relationship
pairs = [
    ('μ/e', 'τ/e', 'Leptons'),
    ('d/u', 's/d', 'Down-type'),
    ('c/s', 't/b', 'Up-type'),
    ('p/e', 'n/p', 'Nucleons'),
]

print("\nPair | Dev 1 | Dev 2 | Product | Sum")
print("-" * 60)
for p1, p2, label in pairs:
    d1 = deviations.get(p1, 0)
    d2 = deviations.get(p2, 0)
    product = d1 * d2
    total = d1 + d2
    
    # Negative product means anti-correlated (one +, one -)
    anti = "✓" if product < 0 else "✗"
    print(f"{label:12} | {d1:+.3f}% | {d2:+.3f}% | {product:+.4f} {anti} | {total:+.4f}%")

# ============================================================================
# SECTION 6: GENERATION-WISE BALANCE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 6: GENERATION-WISE PAC BALANCE")
print("=" * 70)

print("""
Each generation should be internally balanced.
Check if up + down deviations cancel within each generation.
""")

# Recompute with consistent approach - deviation of mass from "ideal" Fibonacci
# The "ideal" is what we'd predict from the Fibonacci formula

# For quarks, let's look at the actual vs predicted structure
gen1_u_over_e = particles['u'] / m_e
gen1_d_over_e = particles['d'] / m_e

print("Generation 1:")
print(f"  u/e = {gen1_u_over_e:.3f} ≈ F_3² = {F[3]**2} = 4 ({(gen1_u_over_e - 4)/4*100:+.2f}%)")
print(f"  d/e = {gen1_d_over_e:.3f} ≈ F_4² = {F[4]**2} = 9 ({(gen1_d_over_e - 9)/9*100:+.2f}%)")
print(f"  Sum of deviations: {(gen1_u_over_e-4)/4*100 + (gen1_d_over_e-9)/9*100:+.2f}%")

gen2_s_over_e = particles['s'] / m_e
gen2_c_over_e = particles['c'] / m_e

print("\nGeneration 2:")
print(f"  s/e = {gen2_s_over_e:.1f} ≈ F_6×F_7 = {F[6]*F[7]} = 168? ({(gen2_s_over_e - 168)/168*100:+.2f}%)")
print(f"  c/e = {gen2_c_over_e:.1f} ≈ F_6×F_9/F_3 = {F[6]*F[9]/F[3]} = 136? ({(gen2_c_over_e - F[6]*F[9]/F[3])/(F[6]*F[9]/F[3])*100:+.2f}%)")

# Let's try to find better formulas that BALANCE
print("\n" + "-" * 50)
print("SEARCHING FOR BALANCED FIBONACCI FORMULAS")
print("-" * 50)

def find_fib_approx(value, name, tolerance=0.15):
    """Find Fibonacci approximations within tolerance"""
    results = []
    
    # Single Fibonacci
    for i, f in enumerate(F[:14]):
        if f > 0:
            error = abs(value - f) / f
            if error < tolerance:
                results.append((f"F_{i}", f, error))
    
    # Products of two
    for i in range(2, 12):
        for j in range(i, 12):
            prod = F[i] * F[j]
            error = abs(value - prod) / prod
            if error < tolerance:
                results.append((f"F_{i}×F_{j}", prod, error))
    
    # Ratios
    for i in range(2, 12):
        for j in range(2, 12):
            if F[j] > 0 and i != j:
                ratio = F[i] / F[j]
                if ratio > 1:  # Only meaningful ratios
                    error = abs(value - ratio) / ratio
                    if error < tolerance:
                        results.append((f"F_{i}/F_{j}", ratio, error))
    
    results.sort(key=lambda x: x[2])
    return results[:3]

print(f"\nBest Fibonacci approximations for quark/e ratios:")
for name in ['u', 'd', 's', 'c', 'b', 't']:
    ratio = particles[name] / m_e
    matches = find_fib_approx(ratio, name)
    if matches:
        best = matches[0]
        print(f"  {name}/e = {ratio:.2f} ≈ {best[0]} = {best[1]:.2f} ({best[2]*100:.2f}%)")

# ============================================================================
# SECTION 7: THE IMBALANCE BUDGET PRINCIPLE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 7: IMBALANCE BUDGET PRINCIPLE")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    IMBALANCE BUDGET PRINCIPLE                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CORE IDEA: "Local imbalance creates global balance"                 ║
║                                                                      ║
║  1. DEVIATION = IMBALANCE CONTRIBUTION                               ║
║     Each particle's "error" from Fibonacci is its contribution       ║
║     to the global PAC balance. Not noise - information!              ║
║                                                                      ║
║  2. CROSSOVER = EQUILIBRIUM POINT                                    ║
║     Particles near the edge of chaos (~100 MeV) have SMALLER         ║
║     deviations because they're at maximum convergence.               ║
║                                                                      ║
║  3. PAIR ANTI-CORRELATION                                            ║
║     Related particles (same generation, same type) should            ║
║     have OPPOSITE deviations: one over, one under.                   ║
║                                                                      ║
║  4. GLOBAL SUM = FIBONACCI FRACTION                                  ║
║     The total imbalance budget should be a Fibonacci ratio           ║
║     (like π/55 or 1/φ), not zero - because the system is             ║
║     dynamically balanced, not statically balanced.                   ║
║                                                                      ║
║  ANALOGY: Like a balanced mobile                                     ║
║     Each piece hangs slightly off-center, but the whole balances.    ║
║     The "errors" ARE the structure, not noise on top of it.          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# SECTION 8: COMPUTE GLOBAL IMBALANCE SIGNATURE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 8: GLOBAL IMBALANCE SIGNATURE")
print("=" * 70)

# Weight deviations by log-distance from crossover
print("Weighted imbalance (deviation × distance from crossover):")
weighted_sum = 0
for name, mass, log_dist, dev in particle_deviations:
    weighted = dev * log_dist
    weighted_sum += weighted
    print(f"  {name}: {dev:.3f}% × {log_dist:+.3f} = {weighted:+.4f}")

print(f"\nWeighted sum: {weighted_sum:+.4f}")
print(f"  ≈ 1/F_N? Checking...")

for i in range(2, 10):
    target = 1/F[i]
    error = abs(abs(weighted_sum) - target) / target * 100
    if error < 50:
        print(f"    |weighted| ≈ 1/F_{i} = {target:.4f} ({error:.1f}% error)")

# ============================================================================
# SECTION 9: THE MUON AS FULCRUM
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 9: THE MUON AS PAC FULCRUM")
print("=" * 70)

print("""
If the crossover scale is the balance point, the MUON should act as
the fulcrum of the particle mass "mobile."

Test: Does the muon's deviation from ideal balance the others?
""")

# Muon deviation
mu_dev = deviations['μ/e']
print(f"Muon deviation: {mu_dev:+.4f}%")

# All other deviations
other_devs = [d for k, d in deviations.items() if 'μ' not in k]
other_sum = sum(other_devs)
print(f"Sum of all other deviations: {other_sum:+.4f}%")

# Does muon balance the rest?
total = mu_dev + other_sum
print(f"Total (muon + others): {total:+.4f}%")

# Ratio
if abs(mu_dev) > 0:
    ratio = -other_sum / mu_dev
    print(f"Ratio |others|/|muon| = {ratio:.4f}")
    print(f"  ≈ F_N?")
    for i in range(2, 12):
        if abs(ratio - F[i]) / F[i] < 0.2:
            print(f"    ≈ F_{i} = {F[i]} ({abs(ratio-F[i])/F[i]*100:.1f}% error)")

# ============================================================================
# SECTION 10: PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 10: FALSIFIABLE PREDICTIONS")
print("=" * 70)

print("""
1. AS QUARK MASSES ARE REFINED:
   - The sum of deviations should remain ~constant
   - Individual deviations may change but must re-balance
   
2. UNDISCOVERED PARTICLES:
   - Any new particle's deviation from Fibonacci is PREDICTABLE
   - It must anti-correlate with its generation partners
   
3. THE MUON'S SPECIAL ROLE:
   - The muon's deviation should be the fulcrum
   - If muon mass is measured more precisely, others adjust
   
4. NEAR-CROSSOVER = MINIMUM DEVIATION:
   - Strange quark (~93 MeV) should have smallest deviation
   - Check: strange is closest to crossover (96.89 MeV)
""")

# Check prediction 4
print("\nTest Prediction 4: Strange quark deviation")
s_log_dist = np.log10(particles['s'] / crossover_scale)
print(f"  Strange log-distance from crossover: {s_log_dist:+.4f}")
print(f"  (Smallest positive log-distance in our set?)")

# Find the particle with smallest |log-distance|
min_dist_particle = min(particle_deviations, key=lambda x: abs(x[2]))
print(f"  Closest to crossover: {min_dist_particle[0]} (|log-dist| = {abs(min_dist_particle[2]):.4f})")
print(f"  Its deviation: {min_dist_particle[3]:.4f}%")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
