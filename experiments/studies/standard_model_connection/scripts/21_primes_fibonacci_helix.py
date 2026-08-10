"""
Primes on the Fibonacci Helix: Testing for Geometric Structure

Framework:
- Build a Fibonacci helix: r(θ) = a * φ^(θ/(π/2))
- Place integers along arclength
- Attach a ribbon, measure local twist
- Mark primes vs composites
- Look for structure: do primes cluster at specific twist phases?

The hypothesis: Primes are "irreducible SEC threads" - they should
appear at geometrically special locations on the Fibonacci manifold.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

def is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def sieve_primes(n_max):
    """Generate primes up to n_max using Sieve of Eratosthenes."""
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n_max)) + 1):
        if sieve[i]:
            for j in range(i*i, n_max + 1, i):
                sieve[j] = False
    return set(i for i, is_p in enumerate(sieve) if is_p)

print('='*70)
print('PRIMES ON THE FIBONACCI HELIX')
print('='*70)

# Parameters
N_MAX = 10000
A = 1.0  # Initial radius
P = 0.1  # Pitch of helix (z-growth per radian)

# Generate primes
primes = sieve_primes(N_MAX)
print(f'\nGenerated {len(primes)} primes up to {N_MAX}')

# Step 1: Build the Fibonacci helix
# r(θ) = a * φ^(θ/(π/2))
# x = r*cos(θ), y = r*sin(θ), z = p*θ

def fibonacci_helix(theta, a=A, p=P):
    """Compute (x, y, z) on Fibonacci helix at angle theta."""
    r = a * PHI ** (theta / (np.pi/2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = p * theta
    return np.array([x, y, z]), r

# Step 2: Map integers to arclength along helix
# We'll use theta proportional to n for simplicity
# (A more rigorous approach would use actual arclength)

def get_helix_data(n, theta_scale=0.1):
    """Get helix position and local properties for integer n."""
    theta = n * theta_scale
    pos, r = fibonacci_helix(theta)
    
    # Tangent vector (derivative of position w.r.t. theta)
    eps = 1e-6
    pos_plus, _ = fibonacci_helix(theta + eps)
    pos_minus, _ = fibonacci_helix(theta - eps)
    tangent = (pos_plus - pos_minus) / (2 * eps)
    tangent = tangent / np.linalg.norm(tangent)
    
    # Radial direction (from z-axis to point, projected to xy-plane)
    radial = np.array([pos[0], pos[1], 0])
    if np.linalg.norm(radial) > 0:
        radial = radial / np.linalg.norm(radial)
    
    # Normal to ribbon (perpendicular to tangent in plane containing radial)
    normal = np.cross(tangent, np.array([0, 0, 1]))
    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)
    
    return {
        'n': n,
        'theta': theta,
        'pos': pos,
        'r': r,
        'tangent': tangent,
        'radial': radial,
        'normal': normal
    }

# Step 3: Compute local twist for each integer
print('\nComputing helix data and local twist...')

# Note: Initial analysis showed no geometric signal on helix.
# The more interesting finding is the DUALITY between Fibonacci and Primes:
# - Fibonacci: deterministic, autocorrelated, ratio -> phi
# - Primes: stochastic, uncorrelated, ratio -> 1
# - They are ORTHOGONAL structures (cross-correlation ~ 0)
# - See 22_fibonacci_prime_duality.py for this analysis

helix_data = []
for n in range(1, N_MAX + 1):
    data = get_helix_data(n)
    data['is_prime'] = n in primes
    helix_data.append(data)

# Compute incremental twist between consecutive positions
# Twist = angle between ribbon normals projected onto plane perpendicular to tangent
twists = []
for i in range(1, len(helix_data)):
    prev = helix_data[i-1]
    curr = helix_data[i]
    
    # Project previous normal onto plane perpendicular to current tangent
    n_prev = prev['normal']
    t_curr = curr['tangent']
    n_curr = curr['normal']
    
    # Remove component along tangent
    n_prev_proj = n_prev - np.dot(n_prev, t_curr) * t_curr
    if np.linalg.norm(n_prev_proj) > 1e-10:
        n_prev_proj = n_prev_proj / np.linalg.norm(n_prev_proj)
        
        # Angle between projected previous normal and current normal
        dot = np.clip(np.dot(n_prev_proj, n_curr), -1, 1)
        twist_angle = np.arccos(dot)
        
        # Sign from cross product
        cross = np.cross(n_prev_proj, n_curr)
        if np.dot(cross, t_curr) < 0:
            twist_angle = -twist_angle
    else:
        twist_angle = 0
    
    helix_data[i]['twist'] = twist_angle
    twists.append(twist_angle)

helix_data[0]['twist'] = 0  # First point has no previous

print(f'Computed twist for {len(twists)} transitions')

# Step 4: Analyze prime vs composite distributions
print('\n' + '='*70)
print('PRIME VS COMPOSITE TWIST DISTRIBUTION')
print('='*70)

prime_twists = [helix_data[n-1]['twist'] for n in range(2, N_MAX+1) if n in primes]
composite_twists = [helix_data[n-1]['twist'] for n in range(2, N_MAX+1) if n not in primes and n > 1]

print(f'\nPrimes: {len(prime_twists)} samples')
print(f'Composites: {len(composite_twists)} samples')

print(f'\nMean twist:')
print(f'  Primes:     {np.mean(prime_twists):.6f} ± {np.std(prime_twists)/np.sqrt(len(prime_twists)):.6f}')
print(f'  Composites: {np.mean(composite_twists):.6f} ± {np.std(composite_twists)/np.sqrt(len(composite_twists)):.6f}')

# Statistical test
t_stat, p_value = stats.ttest_ind(prime_twists, composite_twists)
print(f'\nT-test: t={t_stat:.4f}, p={p_value:.4f}')

ks_stat, ks_p = stats.ks_2samp(prime_twists, composite_twists)
print(f'KS test: stat={ks_stat:.4f}, p={ks_p:.4f}')

# Step 5: Look at twist PHASE (mod 2π)
print('\n' + '='*70)
print('TWIST PHASE ANALYSIS (mod 2π)')
print('='*70)

# Cumulative twist
cumulative_twist = np.cumsum([0] + twists)

# Phase = cumulative twist mod 2π
phases = cumulative_twist % (2 * np.pi)

prime_phases = [phases[n-1] for n in range(2, N_MAX+1) if n in primes]
composite_phases = [phases[n-1] for n in range(2, N_MAX+1) if n not in primes]

print(f'\nPhase distribution (0 to 2π):')
print(f'  Prime mean phase:     {np.mean(prime_phases):.4f} rad ({np.mean(prime_phases)*180/np.pi:.1f}°)')
print(f'  Composite mean phase: {np.mean(composite_phases):.4f} rad ({np.mean(composite_phases)*180/np.pi:.1f}°)')

# Circular statistics (Rayleigh test for uniformity)
def rayleigh_test(angles):
    """Test for uniformity of circular distribution."""
    n = len(angles)
    cos_sum = np.sum(np.cos(angles))
    sin_sum = np.sum(np.sin(angles))
    R = np.sqrt(cos_sum**2 + sin_sum**2) / n
    # Rayleigh test statistic
    z = n * R**2
    # p-value (approximate)
    p = np.exp(-z)
    return R, z, p

R_prime, z_prime, p_prime = rayleigh_test(prime_phases)
R_comp, z_comp, p_comp = rayleigh_test(composite_phases)

print(f'\nRayleigh test for non-uniformity:')
print(f'  Primes: R={R_prime:.4f}, z={z_prime:.2f}, p={p_prime:.4e}')
print(f'  Composites: R={R_comp:.4f}, z={z_comp:.2f}, p={p_comp:.4e}')

# Step 6: Look at theta phase (position on helix)
print('\n' + '='*70)
print('HELIX ANGLE PHASE ANALYSIS (θ mod 2π)')
print('='*70)

theta_scale = 0.1
prime_theta_phases = [(n * theta_scale) % (2 * np.pi) for n in primes if n <= N_MAX]
composite_theta_phases = [(n * theta_scale) % (2 * np.pi) for n in range(2, N_MAX+1) if n not in primes]

R_prime_th, z_prime_th, p_prime_th = rayleigh_test(prime_theta_phases)
R_comp_th, z_comp_th, p_comp_th = rayleigh_test(composite_theta_phases)

print(f'Rayleigh test for θ-phase non-uniformity:')
print(f'  Primes: R={R_prime_th:.4f}, z={z_prime_th:.2f}, p={p_prime_th:.4e}')
print(f'  Composites: R={R_comp_th:.4f}, z={z_comp_th:.2f}, p={p_comp_th:.4e}')

# Step 7: Golden angle analysis
print('\n' + '='*70)
print('GOLDEN ANGLE ANALYSIS')
print('='*70)

GOLDEN_ANGLE = 2 * np.pi * (1 - 1/PHI)  # ~137.5°

print(f'Golden angle: {GOLDEN_ANGLE:.4f} rad ({GOLDEN_ANGLE*180/np.pi:.2f}°)')

# Place integers at golden angle increments
golden_phases = [(n * GOLDEN_ANGLE) % (2 * np.pi) for n in range(1, N_MAX+1)]

prime_golden = [golden_phases[n-1] for n in primes if n <= N_MAX]
composite_golden = [golden_phases[n-1] for n in range(2, N_MAX+1) if n not in primes]

R_prime_g, z_prime_g, p_prime_g = rayleigh_test(prime_golden)
R_comp_g, z_comp_g, p_comp_g = rayleigh_test(composite_golden)

print(f'\nRayleigh test for golden-angle phase:')
print(f'  Primes: R={R_prime_g:.4f}, z={z_prime_g:.2f}, p={p_prime_g:.4e}')
print(f'  Composites: R={R_comp_g:.4f}, z={z_comp_g:.2f}, p={p_comp_g:.4e}')

# Step 8: Bin analysis - do primes cluster in certain phase bins?
print('\n' + '='*70)
print('PHASE BIN ANALYSIS')
print('='*70)

n_bins = 12
bin_edges = np.linspace(0, 2*np.pi, n_bins + 1)

prime_hist, _ = np.histogram(prime_golden, bins=bin_edges)
composite_hist, _ = np.histogram(composite_golden, bins=bin_edges)

# Expected if uniform
expected_prime = len(prime_golden) / n_bins
expected_composite = len(composite_golden) / n_bins

# Chi-square test
chi2_prime = np.sum((prime_hist - expected_prime)**2 / expected_prime)
chi2_composite = np.sum((composite_hist - expected_composite)**2 / expected_composite)

p_chi2_prime = 1 - stats.chi2.cdf(chi2_prime, n_bins - 1)
p_chi2_composite = 1 - stats.chi2.cdf(chi2_composite, n_bins - 1)

print(f'\nChi-square test for uniformity across {n_bins} phase bins:')
print(f'  Primes: χ²={chi2_prime:.2f}, p={p_chi2_prime:.4f}')
print(f'  Composites: χ²={chi2_composite:.2f}, p={p_chi2_composite:.4f}')

print('\nPrime distribution across golden-angle phase bins:')
for i in range(n_bins):
    start = bin_edges[i] * 180 / np.pi
    end = bin_edges[i+1] * 180 / np.pi
    expected = expected_prime
    observed = prime_hist[i]
    ratio = observed / expected
    bar = '█' * int(ratio * 20)
    print(f'  {start:5.0f}°-{end:5.0f}°: {observed:4d} (exp {expected:.0f}) {bar}')

# Step 9: Summary
print('\n' + '='*70)
print('KEY FINDINGS')
print('='*70)

print('''
TWIST ANALYSIS:
- Primes and composites have similar mean twist (no significant difference)
- T-test and KS-test show no significant distribution difference

PHASE UNIFORMITY:
- Both primes and composites appear uniformly distributed in phase
- Rayleigh tests show no significant clustering for either group

GOLDEN ANGLE PLACEMENT:
- Primes distributed nearly uniformly across golden-angle phases
- No evidence of phase preference

INTERPRETATION:
At this scale (N up to 10,000) and with this simple helix mapping,
primes do NOT show obvious geometric clustering on the Fibonacci helix.

This could mean:
1. The mapping is too simple - need more sophisticated SEC density measure
2. The effect is subtle - need larger N or finer statistical tools
3. Primes aren't geometrically special in this particular sense

NEXT STEPS:
- Try Xi-bounded twist windows
- Look at prime GAPS rather than prime positions
- Analyze Fibonacci-index primes specifically (F_n where n is prime)
- Use actual arclength instead of angle parameterization
''')
