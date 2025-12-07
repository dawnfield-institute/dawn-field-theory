"""
Nuclear Magic Numbers: Testing the Fibonacci×2π Pattern

Nuclear shell model magic numbers: 2, 8, 20, 28, 50, 82, 126
These give exceptional nuclear stability.

Question: Do these emerge from Fibonacci × 2π structure?
"""

import numpy as np
from scipy import stats

PHI = (1 + np.sqrt(5)) / 2
TWO_PI = 2 * np.pi
F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Nuclear magic numbers
MAGIC = [2, 8, 20, 28, 50, 82, 126]

print('='*70)
print('NUCLEAR MAGIC NUMBERS AND FIBONACCI STRUCTURE')
print('='*70)

print('\nMagic numbers: 2, 8, 20, 28, 50, 82, 126')
print('These arise from strong spin-orbit coupling in nuclear shell model.')
print()

# Part 1: Basic patterns
print('='*70)
print('BASIC FIBONACCI PATTERNS')
print('='*70)

print('\nFibonacci × 2π:')
print(f'{"n":>4} {"F_n":>6} {"F_n × 2π":>12} {"Nearest Magic":>14} {"Distance":>10}')
print('-'*52)

for i in range(1, 13):
    val = F[i] * TWO_PI
    # Find nearest magic number
    nearest = min(MAGIC, key=lambda x: abs(x - val))
    dist = abs(val - nearest)
    print(f'{i:>4} {F[i]:>6} {val:>12.2f} {nearest:>14} {dist:>10.2f}')

# Part 2: Look for F_n × integer patterns
print('\n' + '='*70)
print('SEARCHING FOR F_n × k PATTERNS')
print('='*70)

print('\nTrying magic = F_n × k for various k:')
print()

for magic in MAGIC:
    print(f'Magic number {magic}:')
    found = False
    for n in range(1, 10):
        if magic % F[n] == 0 and F[n] > 0:
            k = magic // F[n]
            if 1 <= k <= 20:
                print(f'  {magic} = F_{n} × {k} = {F[n]} × {k}')
                found = True
    if not found:
        print(f'  No simple F_n × k factorization found')
    print()

# Part 3: Direct Fibonacci analysis
print('='*70)
print('DIRECT FIBONACCI ANALYSIS')
print('='*70)

print('\nMagic numbers as Fibonacci sums:')
for magic in MAGIC:
    # Zeckendorf representation (sum of non-consecutive Fibonacci numbers)
    remaining = magic
    terms = []
    for i in range(len(F)-1, 0, -1):
        if F[i] <= remaining:
            terms.append(f'F_{i}')
            remaining -= F[i]
        if remaining == 0:
            break
    print(f'{magic:>4} = {" + ".join(terms)}')

# Part 4: Ratios between consecutive magic numbers
print('\n' + '='*70)
print('RATIOS BETWEEN CONSECUTIVE MAGIC NUMBERS')
print('='*70)

print(f'\n{"i":>4} {"M_i":>6} {"M_i/M_{i-1}":>14} {"Closest φ power":>16}')
print('-'*46)

for i in range(1, len(MAGIC)):
    ratio = MAGIC[i] / MAGIC[i-1]
    # Find closest phi power
    best_power = None
    best_dist = float('inf')
    for p in range(-5, 5):
        phi_p = PHI ** p
        if abs(phi_p - ratio) < best_dist:
            best_dist = abs(phi_p - ratio)
            best_power = p
    print(f'{i:>4} {MAGIC[i]:>6} {ratio:>14.4f} φ^{best_power:>2} = {PHI**best_power:>7.4f}')

# Part 5: Spin-orbit pattern
print('\n' + '='*70)
print('SPIN-ORBIT STRUCTURE')
print('='*70)

print('''
The magic numbers arise from filled nuclear shells with spin-orbit splitting:

Shell:   1s   1p    1d2s   1f2p   1g2d3s    1h2f3p   1i2g3d4s
Fill:     2    6     12      8      22        32        44
Total:    2    8     20     28      50        82       126

Key observation: The shell capacities follow a pattern related to
2n^2 (harmonic oscillator) modified by spin-orbit coupling.
''')

# Part 6: Testing against random baseline
print('='*70)
print('STATISTICAL TEST: FIBONACCI PROXIMITY')
print('='*70)

# How close are magic numbers to Fibonacci numbers?
distances = []
for magic in MAGIC:
    # Find distance to nearest Fibonacci
    min_dist = float('inf')
    nearest_f = None
    for f in F[1:]:
        if abs(magic - f) < min_dist:
            min_dist = abs(magic - f)
            nearest_f = f
    distances.append(min_dist)
    print(f'Magic {magic:>3} → nearest F is {nearest_f:>3}, distance = {min_dist:>3}')

avg_dist = np.mean(distances)
print(f'\nAverage distance to nearest Fibonacci: {avg_dist:.2f}')

# Random baseline: what distance would random numbers have?
np.random.seed(42)
n_trials = 10000
random_avg_dists = []
for _ in range(n_trials):
    random_nums = np.random.randint(2, 150, size=len(MAGIC))
    r_dists = []
    for r in random_nums:
        min_d = min(abs(r - f) for f in F[1:])
        r_dists.append(min_d)
    random_avg_dists.append(np.mean(r_dists))

random_avg = np.mean(random_avg_dists)
percentile = 100 * np.sum(np.array(random_avg_dists) <= avg_dist) / n_trials

print(f'Random baseline average distance: {random_avg:.2f}')
print(f'Percentile rank of magic numbers: {percentile:.1f}%')

# Part 7: Key findings
print('\n' + '='*70)
print('KEY FINDINGS')
print('='*70)

print('''
1. Magic numbers are NOT directly F_n × 2π
   (The values don't align well)

2. Some magic numbers have Fibonacci factorizations:
   - 8 = F_6 = 8 (is itself Fibonacci!)
   - 2 = F_3 = 2 (is itself Fibonacci!)
   - 126 = F_5 × 25 + 1 (weak)

3. The ratios between consecutive magic numbers are NOT
   close to phi powers (ranges from 1.40 to 4.00)

4. Fibonacci proximity: magic numbers are slightly closer
   to Fibonacci numbers than random, but not dramatically so.

CONCLUSION: The nuclear magic numbers likely do NOT have a 
            simple Fibonacci/PAC explanation.
            
            They arise from quantum mechanics of the nuclear
            shell model with spin-orbit coupling, which is
            fundamentally different from Fibonacci branching.
''')

# Part 8: What WOULD show PAC structure?
print('='*70)
print('WHAT WOULD SHOW PAC STRUCTURE?')
print('='*70)

print('''
If nuclear magic numbers had PAC origin, we would expect:

1. Magic numbers to be F_n × k for small k
2. Ratios close to phi or phi powers
3. Zeckendorf representations with few terms

What we actually see:
1. Only 2 and 8 are Fibonacci numbers themselves
2. Ratios don't cluster around phi
3. Zeckendorf representations are complex

This is a NEGATIVE RESULT for PAC in nuclear physics.
Not all physics exhibits Fibonacci structure.

The distinction may be:
- PAC applies to: hierarchical systems, cascades, generations
- PAC does NOT apply to: quantum bound states in confining potentials
''')
