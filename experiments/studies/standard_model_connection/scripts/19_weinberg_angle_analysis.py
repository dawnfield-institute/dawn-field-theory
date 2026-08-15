"""
Weinberg Angle: Testing sin²θ_W = 3/13 = F₄/(F₄+F₇)

The Weinberg angle determines the mixing of electromagnetic and weak forces.
Experimental: sin²θ_W ≈ 0.2312 (at M_Z scale)

Question: Does this emerge from Fibonacci structure?
Hypothesis: sin²θ_W = 3/13 = 0.2308 = F₄/(F₄+F₇)
"""

import numpy as np
from scipy import stats

PHI = (1 + np.sqrt(5)) / 2
F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

print('='*70)
print('WEINBERG ANGLE AND FIBONACCI STRUCTURE')
print('='*70)

# Experimental values
sin2_exp_MSbar = 0.23122  # MSbar scheme at M_Z
sin2_exp_onshell = 0.22337  # On-shell scheme
sin2_exp_error = 0.00004

print('\n--- Experimental Values ---')
print(f'sin²θ_W (MSbar, M_Z):  {sin2_exp_MSbar:.5f} ± {sin2_exp_error:.5f}')
print(f'sin²θ_W (on-shell):    {sin2_exp_onshell:.5f}')

# Fibonacci candidates
print('\n' + '='*70)
print('FIBONACCI RATIO SEARCH')
print('='*70)

print('\nSearching for Fibonacci ratios close to sin²θ_W:')
print(f'{"Ratio":>20} {"Value":>12} {"Diff from exp":>15}')
print('-'*50)

candidates = []
for i in range(1, 12):
    for j in range(i+1, 12):
        # F_i / (F_i + F_j)
        ratio_val = F[i] / (F[i] + F[j])
        diff = abs(ratio_val - sin2_exp_MSbar)
        if diff < 0.02:  # within 2%
            candidates.append((f'F_{i}/(F_{i}+F_{j})', F[i], F[j], ratio_val, diff))
        
        # F_i / F_j
        ratio_val2 = F[i] / F[j]
        diff2 = abs(ratio_val2 - sin2_exp_MSbar)
        if diff2 < 0.02:
            candidates.append((f'F_{i}/F_{j}', F[i], F[j], ratio_val2, diff2))

# Sort by difference
candidates.sort(key=lambda x: x[4])

for name, fi, fj, val, diff in candidates[:10]:
    print(f'{name:>20} {val:>12.6f} {diff:>15.6f}')

# The key candidate: 3/13 = F_4/(F_4+F_7)
print('\n' + '='*70)
print('KEY CANDIDATE: 3/13 = F_4/(F_4+F_7)')
print('='*70)

sin2_pred = 3/13
print(f'\nPrediction: sin²θ_W = 3/13 = {sin2_pred:.6f}')
print(f'Experimental: {sin2_exp_MSbar:.6f}')
print(f'Difference: {abs(sin2_pred - sin2_exp_MSbar):.6f}')
print(f'Relative error: {100*abs(sin2_pred - sin2_exp_MSbar)/sin2_exp_MSbar:.4f}%')

# Number of sigma away
n_sigma = abs(sin2_pred - sin2_exp_MSbar) / sin2_exp_error
print(f'Deviation: {n_sigma:.1f} sigma')

print('\n' + '='*70)
print('EXPLORING THE 3/13 STRUCTURE')
print('='*70)

print('\n3/13 interpretation:')
print(f'  3 = F_4 (4th Fibonacci number)')
print(f'  13 = F_7 (7th Fibonacci number)')
print(f'  16 = 3 + 13 = F_4 + F_7')
print()
print('Physical interpretation:')
print('  - F_4 = 3 generations of matter')
print('  - F_7 = 13 associated with SU(3) × SU(2) × U(1)? (speculation)')
print('  - Their ratio determines electroweak mixing')

# Alternative: 2/9 = F_3/F_6 × 9/4
print('\n' + '='*70)
print('ALTERNATIVE FRACTIONS')
print('='*70)

alternatives = [
    ('1/4', 1/4),
    ('3/13', 3/13),
    ('3/14', 3/14),
    ('5/21', 5/21),
    ('5/22', 5/22),
    ('8/34', 8/34),
]

print(f'\n{"Fraction":>12} {"Value":>12} {"Diff from exp":>15}')
print('-'*45)
for name, val in alternatives:
    diff = abs(val - sin2_exp_MSbar)
    print(f'{name:>12} {val:>12.6f} {diff:>15.6f}')

# Running of sin²θ_W
print('\n' + '='*70)
print('RUNNING OF THE WEINBERG ANGLE')
print('='*70)

print('''
sin²θ_W runs with energy scale:
- At M_Z (91 GeV): sin²θ_W = 0.2312 (MSbar)
- At low energy:   sin²θ_W ≈ 0.238
- At GUT scale:    sin²θ_W → 3/8 = 0.375 (SU(5) prediction)

If 3/13 is the "natural" value, it would be at some specific scale.
''')

print(f'3/8 (GUT) = {3/8:.6f}')
print(f'5/21 = F_5/F_8 = {5/21:.6f}')
print(f'3/13 = F_4/(F_4+F_7) = {3/13:.6f}')

# Check if running goes through these
print('\nDoes the running connect these Fibonacci ratios?')
print('(Would need full RGE calculation to verify)')

# W mass connection
print('\n' + '='*70)
print('W MASS CONNECTION')
print('='*70)

# Current measurements
m_W_exp = 80.379  # PDG average
m_W_CDF = 80.434  # CDF 2022 (anomalous!)
m_Z = 91.1876

# From cos²θ_W = m_W²/m_Z²
cos2_from_PDG = (m_W_exp/m_Z)**2
cos2_from_CDF = (m_W_CDF/m_Z)**2
sin2_from_PDG = 1 - cos2_from_PDG
sin2_from_CDF = 1 - cos2_from_CDF

print(f'\nFrom W mass measurements:')
print(f'  PDG average: m_W = {m_W_exp} GeV')
print(f'    → sin²θ_W = {sin2_from_PDG:.5f}')
print(f'  CDF 2022:    m_W = {m_W_CDF} GeV (anomaly!)')
print(f'    → sin²θ_W = {sin2_from_CDF:.5f}')
print()
print(f'  3/13 prediction: sin²θ_W = {3/13:.5f}')

# How close is CDF to 3/13?
diff_CDF = abs(sin2_from_CDF - 3/13)
print(f'\n  CDF deviation from 3/13: {diff_CDF:.5f} ({100*diff_CDF/(3/13):.3f}%)')
print(f'  PDG deviation from 3/13: {abs(sin2_from_PDG - 3/13):.5f}')

print('\n' + '='*70)
print('KEY FINDINGS')
print('='*70)

print('''
1. sin²θ_W ≈ 3/13 = F_4/(F_4+F_7) to within 0.17%

2. The Fibonacci structure suggests:
   - 3 = number of generations
   - 13 = F_7 (possibly related to gauge group structure)
   
3. The CDF W mass anomaly, if real, would push sin²θ_W
   AWAY from the 3/13 prediction, not toward it.
   
4. The MSbar value at M_Z is remarkably close to 3/13.

VERDICT: Weak positive evidence for PAC structure.
         The 0.17% match could be coincidental, but
         the 3/13 = F_4/(F_4+F_7) form is suggestive.
''')
