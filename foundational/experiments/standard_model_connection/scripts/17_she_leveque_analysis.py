"""
She-Leveque Turbulence: Testing the 2/3 = F3/F4 Connection

The She-Leveque model (1994) describes intermittency in turbulence.
The model has a free parameter beta = 2/3 that fits experimental data.
WHY 2/3? This script explores the Fibonacci connection.
"""

import numpy as np
from scipy import stats

PHI = (1 + np.sqrt(5)) / 2
F = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def she_leveque(p, beta=2/3):
    """Standard She-Leveque scaling exponent."""
    return p/9 + 2*(1 - beta**(p/3))

def k41(p):
    """Kolmogorov 1941 (non-intermittent) prediction."""
    return p/3

print('='*70)
print('SHE-LEVEQUE TURBULENCE: THE 2/3 CONNECTION')
print('='*70)

# Part 1: The She-Leveque model
print('\n--- She-Leveque Model ---\n')
print('Structure function exponents: zeta_p = p/9 + 2[1 - (2/3)^(p/3)]')
print()
print('Comparison with K41 (Kolmogorov 1941):')
print('-'*50)
print(f'{"p":>4} {"zeta_p (SL)":>12} {"zeta_p (K41)":>12} {"Deviation":>10}')
print('-'*50)

for p in [1, 2, 3, 4, 5, 6, 8, 10]:
    z_sl = she_leveque(p)
    z_k41 = k41(p)
    dev = 100 * (z_sl - z_k41) / z_k41
    print(f'{p:>4} {z_sl:>12.4f} {z_k41:>12.4f} {dev:>9.2f}%')

# Part 2: Why 2/3?
print('\n' + '='*70)
print('WHY 2/3? THE FIBONACCI CONNECTION')
print('='*70)

print('\nKey observation: 2/3 = F_3/F_4 = 2/3')
print()
print('Fibonacci ratios around 2/3:')
for i in range(2, 8):
    ratio = F[i] / F[i+1]
    print(f'  F_{i}/F_{i+1} = {F[i]}/{F[i+1]} = {ratio:.6f}')

print()
print('The Koide formula also uses 2/3:')
print('  Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2 = 2/3')
print()
print('This suggests 2/3 is not arbitrary - it encodes F_3/F_4.')

# Part 3: Testing alternative beta values
print('\n' + '='*70)
print('WHAT IF BETA = OTHER FIBONACCI RATIOS?')
print('='*70)

print('\nComparing She-Leveque with different beta values:')
print()

# Experimental data (approximate from DNS/experiments)
# These are typical measured values for zeta_p
experimental = {
    2: 0.70,  # zeta_2 ~ 0.70 (vs K41 = 0.67)
    3: 1.00,  # zeta_3 = 1.00 exactly (energy conservation)
    4: 1.28,  # zeta_4 ~ 1.28 (vs K41 = 1.33)
    6: 1.78,  # zeta_6 ~ 1.78 (vs K41 = 2.00)
}

beta_candidates = [
    ('2/3 = F3/F4', 2/3),
    ('1/phi', 1/PHI),
    ('3/5 = F4/F5', 3/5),
    ('5/8 = F5/F6', 5/8),
    ('phi-1', PHI-1),
]

print(f'{"Beta":>15} {"zeta_2":>8} {"zeta_3":>8} {"zeta_4":>8} {"zeta_6":>8} {"RMS Error":>10}')
print('-'*60)

# Experimental reference
print(f'{"Experimental":>15} {experimental[2]:>8.3f} {experimental[3]:>8.3f} {experimental[4]:>8.3f} {experimental[6]:>8.3f}')
print('-'*60)

for name, beta in beta_candidates:
    z2 = she_leveque(2, beta)
    z3 = she_leveque(3, beta)
    z4 = she_leveque(4, beta)
    z6 = she_leveque(6, beta)
    
    # RMS error vs experimental
    errors = [(z2 - experimental[2])**2, 
              (z3 - experimental[3])**2,
              (z4 - experimental[4])**2,
              (z6 - experimental[6])**2]
    rms = np.sqrt(np.mean(errors))
    
    print(f'{name:>15} {z2:>8.3f} {z3:>8.3f} {z4:>8.3f} {z6:>8.3f} {rms:>10.4f}')

# Part 4: Physical interpretation
print('\n' + '='*70)
print('PHYSICAL INTERPRETATION')
print('='*70)

print('''
She-Leveque derived beta from log-Poisson statistics, but the VALUE
2/3 was determined empirically to fit turbulence data.

Fibonacci interpretation:
  - Turbulent cascade is a branching process (eddies split)
  - At each scale, energy partitions into smaller eddies
  - The cascade has FINITE depth (viscous cutoff)
  
  - 2/3 = F_3/F_4 encodes the ratio at depth 3-4
  - This suggests turbulent cascades have effective depth ~3-4
    scales before viscous damping dominates
    
  - In PAC terms: the cascade "tree" has depth ~4
  - The 2/3 coefficient is the F ratio at this truncation point
''')

# Part 5: Testable prediction
print('='*70)
print('TESTABLE PREDICTION')
print('='*70)

print('''
If 2/3 = F_3/F_4 encodes cascade depth, then:

1. Systems with DEEPER cascades (higher Reynolds number) should show
   beta closer to phi^(-1) = 0.618 (the infinite-depth limit)
   
2. Systems with SHALLOWER cascades should show beta closer to 
   F_2/F_3 = 1/2 = 0.5

Let's compute what beta should be at different "depths":
''')

print(f'{"Depth n":>10} {"F_n/F_{n+1}":>15} {"Beta":>10}')
print('-'*40)
for n in range(1, 10):
    beta = F[n] / F[n+1]
    print(f'{n:>10} {f"F_{n}/F_{n+1}":>15} {beta:>10.6f}')

print(f'{"∞":>10} {"1/phi":>15} {1/PHI:>10.6f}')

print('''
Prediction: High-Reynolds turbulence should show beta → 0.618
            Low-Reynolds turbulence should show beta < 0.667
''')

# Part 6: Literature check
print('='*70)
print('CHECKING AGAINST LITERATURE')
print('='*70)

print('''
Known variations in measured beta:

- DNS at moderate Re: beta ~ 0.67 (matches 2/3)
- High Re experiments: some report beta ~ 0.64-0.66
- Very high Re (atmospheric): some suggest lower intermittency

This is CONSISTENT with the prediction that higher Re → beta → 1/phi!

However: Need more systematic data to confirm.
The variation could also be measurement uncertainty.
''')
