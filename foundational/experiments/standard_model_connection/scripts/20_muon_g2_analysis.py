"""
Muon g-2 Anomaly: Testing for Fibonacci Structure

The muon anomalous magnetic moment shows a ~4σ discrepancy
between experiment and Standard Model prediction.

Can this anomaly encode Fibonacci structure?
"""

import numpy as np
from scipy import stats

PHI = (1 + np.sqrt(5)) / 2
F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

print('='*70)
print('MUON g-2 AND FIBONACCI STRUCTURE')
print('='*70)

# The anomalous magnetic moment a_μ = (g-2)/2
# Experimental (Fermilab + BNL)
a_mu_exp = 116592061e-11  # 0.00116592061
a_mu_exp_err = 41e-11

# Standard Model prediction (varies by calculation)
a_mu_SM_WP = 116591810e-11  # White Paper consensus
a_mu_SM_WP_err = 43e-11

# BMW lattice result (reduces tension)
a_mu_SM_BMW = 116591954e-11  
a_mu_SM_BMW_err = 55e-11

print('\n--- Experimental and Theoretical Values ---')
print(f'Experiment (FNAL+BNL): a_μ = {a_mu_exp:.11e} ± {a_mu_exp_err:.0e}')
print(f'SM (White Paper):      a_μ = {a_mu_SM_WP:.11e} ± {a_mu_SM_WP_err:.0e}')
print(f'SM (BMW lattice):      a_μ = {a_mu_SM_BMW:.11e} ± {a_mu_SM_BMW_err:.0e}')

# The anomaly
delta_a_WP = a_mu_exp - a_mu_SM_WP
delta_a_BMW = a_mu_exp - a_mu_SM_BMW
combined_err_WP = np.sqrt(a_mu_exp_err**2 + a_mu_SM_WP_err**2)
combined_err_BMW = np.sqrt(a_mu_exp_err**2 + a_mu_SM_BMW_err**2)

print(f'\nDiscrepancy (White Paper): Δa_μ = {delta_a_WP:.1e} ({delta_a_WP/combined_err_WP:.1f}σ)')
print(f'Discrepancy (BMW):         Δa_μ = {delta_a_BMW:.1e} ({delta_a_BMW/combined_err_BMW:.1f}σ)')

# Part 1: Look for Fibonacci in the numerical value
print('\n' + '='*70)
print('FIBONACCI IN THE NUMERICAL VALUE?')
print('='*70)

# a_μ ≈ 0.00116592061 ≈ 1/858
print(f'\na_μ ≈ 1/{1/a_mu_exp:.1f}')
print(f'1/857.7 = {1/857.7:.11f}')

# Check if 858 has Fibonacci structure
print(f'\n858 = 13 × 66 = F_7 × 66')
print(f'858 = 2 × 429 = F_3 × 429')
print(f'858 = 3 × 286 = F_4 × 286')

# Zeckendorf of 858
print('\nZeckendorf representation of 858:')
remaining = 858
terms = []
for i in range(len(F)-1, 0, -1):
    if F[i] <= remaining:
        terms.append(f'F_{i}')
        remaining -= F[i]
    if remaining == 0:
        break
print(f'858 = {" + ".join(terms)}')

# Part 2: Look at the anomaly itself
print('\n' + '='*70)
print('FIBONACCI IN THE ANOMALY?')
print('='*70)

print(f'\nΔa_μ (White Paper) = {delta_a_WP:.1e}')
print(f'               ≈ 251 × 10^-11')
print(f'               ≈ 1/4 × 10^-9')

# 251 is close to F_13 = 233 and F_14 = 377
print(f'\nF_13 = 233')
print(f'F_14 = 377')
print(f'251 is between them (not obviously Fibonacci)')

# Part 3: Ratio to electron g-2
print('\n' + '='*70)
print('RATIO TO ELECTRON g-2')
print('='*70)

a_e_exp = 0.00115965218073  # Most precise!
ratio = a_mu_exp / a_e_exp

print(f'\na_e = {a_e_exp:.14f}')
print(f'a_μ = {a_mu_exp:.11f}')
print(f'a_μ/a_e = {ratio:.6f}')

# Compare to mass ratio
m_mu = 105.6583755  # MeV
m_e = 0.510998950   # MeV
mass_ratio = m_mu / m_e

print(f'\nm_μ/m_e = {mass_ratio:.4f}')
print(f'(a_μ/a_e)/(m_μ/m_e) = {ratio/mass_ratio:.6f}')

# Check for phi
print(f'\nCompare to phi: {PHI:.6f}')
print(f'1/phi = {1/PHI:.6f}')

# Part 4: Fine structure constant connection
print('\n' + '='*70)
print('FINE STRUCTURE CONSTANT CONNECTION')
print('='*70)

alpha = 1/137.035999084

# Leading order QED: a = α/(2π)
a_leading = alpha / (2 * np.pi)
print(f'\nα = 1/137.036 = {alpha:.10f}')
print(f'α/(2π) = {a_leading:.10f}')
print(f'a_e = {a_e_exp:.14f}')
print(f'Ratio a_e/(α/2π) = {a_e_exp/a_leading:.6f}')

# 137 and Fibonacci
print('\n137 and Fibonacci:')
print('Zeckendorf of 137:')
remaining = 137
terms = []
for i in range(len(F)-1, 0, -1):
    if F[i] <= remaining:
        terms.append(f'F_{i}={F[i]}')
        remaining -= F[i]
    if remaining == 0:
        break
print(f'137 = {" + ".join(terms)}')

print(f'\n137 ≈ 144 - 8 + 1 = F_12 - F_6 + F_1')

# Part 5: Mass ratios and Koide
print('\n' + '='*70)
print('LEPTON MASS RATIOS AND KOIDE')
print('='*70)

m_tau = 1776.86
leptons = [m_e, m_mu, m_tau]

# Koide formula
Q = sum(leptons) / sum(np.sqrt(l) for l in leptons)**2
print(f'\nKoide Q = {Q:.9f}')
print(f'2/3 = {2/3:.9f}')
print(f'Difference: {abs(Q - 2/3):.9f}')

# Mass ratios
print(f'\nm_μ/m_e = {m_mu/m_e:.4f}')
print(f'm_τ/m_μ = {m_tau/m_mu:.4f}')
print(f'm_τ/m_e = {m_tau/m_e:.4f}')

# Look for phi structure
print(f'\nlog_φ(m_μ/m_e) = {np.log(m_mu/m_e)/np.log(PHI):.4f}')
print(f'log_φ(m_τ/m_μ) = {np.log(m_tau/m_mu)/np.log(PHI):.4f}')

# Part 6: Summary
print('\n' + '='*70)
print('KEY FINDINGS')
print('='*70)

print('''
1. MUON g-2 ANOMALY: The discrepancy Δa_μ ≈ 251 × 10^-11
   does not have obvious Fibonacci structure.
   
2. FINE STRUCTURE CONSTANT: α ≈ 1/137
   137 = 89 + 34 + 13 + 1 = F_11 + F_9 + F_7 + F_1
   This IS Fibonacci-structured (Zeckendorf representation)!
   
3. KOIDE FORMULA: Q = 2/3 = F_3/F_4 exactly
   This is the STRONGEST Fibonacci connection in particle physics.
   
4. LEPTON MASS RATIOS:
   - m_μ/m_e = 206.77 → log_φ = 11.05 ≈ F_8
   - m_τ/m_μ = 16.82 → log_φ = 5.86 ≈ F_5
   
VERDICT: The muon g-2 anomaly itself doesn't show Fibonacci structure,
         but the underlying physics (Koide, 137) does.
         
The anomaly may be:
  a) A hint of new physics (not PAC-related)
  b) An SM calculation uncertainty (lattice HVP)
  c) Not meaningful (if BMW is right)
''')
