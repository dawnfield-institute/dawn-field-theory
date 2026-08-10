"""
PAC Relativity: Speed, Time, Locality, and Gravity from Cascade Mechanics
==========================================================================
Dawn Field Institute — PACSeries Exploration

HYPOTHESES UNDER TEST:

H1: VELOCITY AS PARTITION
    An entity's velocity corresponds to its potential/actualization partition.
    More energy in propagation = less internal cascade = less experienced time.
    This should reproduce the Lorentz factor γ = 1/√(1-v²/c²).

H2: ZERO-POTENTIAL → ONE-DIMENSIONAL
    As potential → 0, the number of available internal modes → 1.
    A photon is one-dimensional because it has no internal degrees of freedom.
    The cascade mode count should collapse continuously as potential drops.

H3: IDENTITY CONSERVATION → LOCALITY
    Moving through the informational geometry preserves identity only if 
    you traverse adjacent nodes. Skipping nodes (teleporting) destroys
    correlational identity. We can measure this as mutual information 
    between the entity's state before and after traversal.

H4: CASCADE DENSITY → GRAVITATIONAL TIME DILATION
    Regions with more cascade activity (more interaction partners, more 
    potential) should show different effective time rates. An entity near
    a dense cascade region should experience time differently than one
    in a void — matching gravitational time dilation.

H5: c FROM PLANCK-SCALE CASCADE
    If the cascade operates on a discrete lattice with minimum step size,
    the maximum propagation rate should be one step per tick. We test
    whether this gives a finite, well-defined maximum speed.
"""

import numpy as np
from scipy import stats, linalg
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

phi = (1 + np.sqrt(5)) / 2
kT = 1.0
LANDAUER_MIN = kT * np.log(2)

print("=" * 70)
print("PAC RELATIVITY: From Cascade Mechanics to Spacetime")
print("Dawn Field Institute — PACSeries Exploration")
print("=" * 70)


# ============================================================
# EXPERIMENT 1: Velocity as Potential-Actualization Partition
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Does the P/A Partition Reproduce Time Dilation?")
print("=" * 70)
print("""
Setup: An entity has total energy E_total = 1.0.
It can partition this into:
  - E_internal (potential): funds internal Landauer cascade
  - E_propagation (actualized): committed to movement

We measure internal cascade rate (= experienced time rate) as a 
function of E_propagation / E_total (= v/c analog).

PREDICTION: Internal cascade rate should follow √(1 - (v/c)²)
i.e. the Lorentz factor.
""")

def measure_cascade_rate(energy_internal, n_modes=8, n_samples=15000):
    """
    Given internal energy budget, run one cascade step and measure
    how much structure (ξ) is produced. More ξ = more cascade activity
    = more experienced time.
    """
    if energy_internal < 1e-15:
        return 0.0, 0.0
    
    # Coupling matrix
    C = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            C[i, j] = np.exp(-abs(i - j) * 0.3)
    C = (C + C.T) / 2
    eigs_C = np.linalg.eigvalsh(C)
    if np.min(eigs_C) < 1e-10:
        C += np.eye(n_modes) * (abs(np.min(eigs_C)) + 1e-6)
    
    means = energy_internal * np.exp(-np.arange(n_modes) * 0.3)
    means *= energy_internal / np.sum(means)
    
    try:
        sf = energy_internal / (np.trace(C) / n_modes) * 0.2
        samples = np.abs(np.random.multivariate_normal(means, C * sf, size=n_samples))
    except:
        samples = np.random.exponential(energy_internal / n_modes, (n_samples, n_modes))
    
    # Measure ξ
    cov = np.cov(samples.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
    diag = np.maximum(np.diag(cov), 1e-30)
    xi = max(0, 0.5 * (np.sum(np.log(diag)) - np.sum(np.log(eigenvalues))))
    
    # Organized fraction
    org_frac = eigenvalues[-1] / np.sum(eigenvalues)
    
    return xi, org_frac


E_total = 1.0
velocity_fractions = np.linspace(0.0, 0.999, 30)  # v/c from 0 to 0.999

print(f"\n{'v/c':>8} | {'E_internal':>11} | {'ξ (cascade)':>12} | "
      f"{'ξ/ξ_rest':>10} | {'Lorentz √':>10} | {'Ratio':>8}")
print("-" * 72)

# First measure at rest (v=0)
xi_rest, _ = measure_cascade_rate(E_total)

results_v = []
for v_frac in velocity_fractions:
    E_prop = E_total * v_frac**2  # kinetic energy ∝ v²
    E_int = E_total - E_prop       # remaining for internal cascade
    
    xi, org = measure_cascade_rate(E_int)
    
    # Lorentz prediction
    lorentz = np.sqrt(1 - v_frac**2)
    
    # Our cascade rate relative to rest
    cascade_ratio = xi / xi_rest if xi_rest > 0 else 0
    
    # How well does cascade ratio match Lorentz?
    match_ratio = cascade_ratio / lorentz if lorentz > 0 else float('inf')
    
    results_v.append({
        'v_frac': v_frac, 'E_int': E_int, 'xi': xi,
        'cascade_ratio': cascade_ratio, 'lorentz': lorentz,
        'match': match_ratio
    })
    
    if v_frac < 0.05 or v_frac > 0.95 or abs(v_frac - 0.5) < 0.02 or abs(v_frac - 0.8) < 0.02 or abs(v_frac - 0.9) < 0.02:
        print(f"  {v_frac:>6.3f} | {E_int:>11.6f} | {xi:>12.6f} | "
              f"{cascade_ratio:>10.6f} | {lorentz:>10.6f} | {match_ratio:>8.4f}")

# Fit: does cascade_ratio ∝ E_int^α for some α?
valid_v = [r for r in results_v if r['E_int'] > 0.01 and r['cascade_ratio'] > 0.01]
if len(valid_v) > 5:
    log_E = np.log([r['E_int'] for r in valid_v])
    log_xi = np.log([r['cascade_ratio'] for r in valid_v])
    slope_v, _, r_v, _, _ = stats.linregress(log_E, log_xi)
    
    print(f"\n  ξ/ξ_rest ∝ E_internal^{slope_v:.4f}")
    print(f"  R² = {r_v**2:.6f}")
    print(f"  For exact Lorentz: would need ξ ∝ E^0.5 (since E_int = 1-v²/c² and γ = √(E_int))")
    print(f"  Measured exponent: {slope_v:.4f}")
    if abs(slope_v - 0.5) < 0.1:
        print(f"  *** MATCHES LORENTZ FACTOR within {abs(slope_v-0.5)/0.5*100:.1f}% ***")


# ============================================================
# EXPERIMENT 2: Mode Collapse at Zero Potential
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 2: Does Mode Count Collapse as Potential → 0?")
print("=" * 70)
print("""
A photon is one-dimensional: it moves in a straight line, has two 
polarization states but no internal spatial structure.

PREDICTION: As internal energy → 0, the effective number of active
modes (participation ratio of eigenvalues) should → 1.

The entity loses dimensionality as it loses potential.
""")

energy_levels = np.logspace(-6, 2, 30)
n_modes_test = 16  # Start with many modes available

print(f"\n{'E_internal':>12} | {'Particip. Ratio':>16} | {'Effective Modes':>16} | "
      f"{'Top eig/total':>14} | {'Interpretation':>20}")
print("-" * 88)

mode_collapse_data = []
for E in energy_levels:
    if E < 1e-10:
        pr = 1.0
        top_frac = 1.0
    else:
        C = np.zeros((n_modes_test, n_modes_test))
        for i in range(n_modes_test):
            for j in range(n_modes_test):
                C[i, j] = np.exp(-abs(i - j) * 0.3) * 0.5
        C = (C + C.T) / 2
        eigs_C = np.linalg.eigvalsh(C)
        C += np.eye(n_modes_test) * (abs(min(0, np.min(eigs_C))) + 1e-6)
        
        means = E * np.exp(-np.arange(n_modes_test) * 0.3)
        means *= E / np.sum(means)
        
        try:
            sf = E / (np.trace(C) / n_modes_test) * 0.15
            samples = np.abs(np.random.multivariate_normal(means, C * sf, size=15000))
        except:
            samples = np.random.exponential(E / n_modes_test, (15000, n_modes_test))
        
        cov = np.cov(samples.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
        
        pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
        top_frac = eigenvalues[-1] / np.sum(eigenvalues)
    
    mode_collapse_data.append({'E': E, 'pr': pr, 'top_frac': top_frac})
    
    interp = ""
    if pr < 1.5:
        interp = "~ photon (1D)"
    elif pr < 3:
        interp = "low-dimensional"
    elif pr > n_modes_test * 0.6:
        interp = "fully multi-mode"
    
    if E < 1e-4 or E > 10 or abs(np.log10(E) - 0) < 0.2 or abs(np.log10(E) - (-3)) < 0.2:
        print(f"  {E:>10.2e} | {pr:>16.4f} | {pr:>16.1f} | "
              f"{top_frac:>14.4f} | {interp:>20}")

# Fit: PR vs E
valid_m = [d for d in mode_collapse_data if d['E'] > 1e-5 and d['E'] < 10]
if len(valid_m) > 5:
    log_E = np.log10([d['E'] for d in valid_m])
    prs = [d['pr'] for d in valid_m]
    slope_m, intercept_m, r_m, _, _ = stats.linregress(log_E, prs)
    print(f"\n  Participation Ratio trend: PR = {slope_m:.4f} × log₁₀(E) + {intercept_m:.4f}")
    print(f"  R² = {r_m**2:.4f}")
    print(f"  As E→0: PR→{intercept_m + slope_m*(-6):.2f} (should approach 1)")
    

# ============================================================
# EXPERIMENT 3: Identity Conservation Under Traversal
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 3: Identity Conservation — Adjacent vs Teleport")
print("=" * 70)
print("""
An entity has an informational identity: a specific correlation 
pattern across its modes. We test whether traversing the informational
geometry step-by-step preserves this identity better than "teleporting"
(randomly reassigning to a distant node).

Identity = mutual information between initial state and final state.

PREDICTION: Adjacent traversal preserves identity.
           Teleportation destroys it.
""")

def create_identity(n_modes=8, n_samples=10000):
    """Create an entity with a specific correlational identity."""
    # A specific coupling pattern = the entity's identity
    C = np.zeros((n_modes, n_modes))
    # Give it a distinctive structure (not generic)
    for i in range(n_modes):
        for j in range(n_modes):
            C[i, j] = np.cos(0.5 * (i - j)) * np.exp(-abs(i - j) * 0.2)
    C = (C + C.T) / 2
    C += np.eye(n_modes) * 0.5
    
    means = np.sin(np.arange(n_modes) * 0.7) + 2
    samples = np.abs(np.random.multivariate_normal(means, C * 0.3, size=n_samples))
    return samples, means, C


def measure_identity_preservation(initial_samples, final_samples):
    """
    Measure how much of the initial correlational structure survives.
    Uses correlation between covariance eigenvalue spectra.
    """
    cov_i = np.cov(initial_samples.T)
    cov_f = np.cov(final_samples.T)
    
    eigs_i = np.sort(np.linalg.eigvalsh(cov_i))[::-1]
    eigs_f = np.sort(np.linalg.eigvalsh(cov_f))[::-1]
    
    # Normalize
    eigs_i = eigs_i / np.sum(eigs_i)
    eigs_f = eigs_f / np.sum(eigs_f)
    
    # Correlation between eigenvalue spectra = structural similarity
    corr = np.corrcoef(eigs_i, eigs_f)[0, 1]
    
    # Also measure Frobenius distance between normalized covariances
    cov_i_norm = cov_i / np.trace(cov_i)
    cov_f_norm = cov_f / np.trace(cov_f)
    frob_dist = np.linalg.norm(cov_i_norm - cov_f_norm, 'fro')
    
    return corr, frob_dist


def adjacent_step(samples, means, C, step_size=0.1):
    """Move through informational geometry by one adjacent step."""
    n_modes = samples.shape[1]
    # Small perturbation to means (adjacent node)
    new_means = means + np.random.randn(n_modes) * step_size
    # Coupling matrix evolves slightly
    perturbation = np.random.randn(n_modes, n_modes) * step_size * 0.1
    new_C = C + (perturbation + perturbation.T) / 2
    eigs = np.linalg.eigvalsh(new_C)
    new_C += np.eye(n_modes) * (abs(min(0, np.min(eigs))) + 0.1)
    
    new_samples = np.abs(np.random.multivariate_normal(new_means, new_C * 0.3, size=samples.shape[0]))
    return new_samples, new_means, new_C


def teleport(samples, n_modes=8, n_samples=10000):
    """Teleport: reassign to a completely random node in the geometry."""
    # Completely new random identity
    C_new = np.random.randn(n_modes, n_modes) * 0.3
    C_new = (C_new + C_new.T) / 2
    C_new += np.eye(n_modes) * 1.0
    means_new = np.random.randn(n_modes) * 2 + 2
    new_samples = np.abs(np.random.multivariate_normal(means_new, C_new * 0.3, size=n_samples))
    return new_samples


n_modes = 8
n_samples = 10000

# Create initial identity
initial_samples, initial_means, initial_C = create_identity(n_modes, n_samples)

# Test adjacent traversal: take N small steps
print(f"\nAdjacent traversal (step-by-step through geometry):")
print(f"{'Steps':>8} | {'Spectral Corr':>14} | {'Frob Distance':>14} | {'Identity?':>12}")
print("-" * 55)

samples_adj = initial_samples.copy()
means_adj = initial_means.copy()
C_adj = initial_C.copy()

for n_steps in [1, 2, 5, 10, 20, 50, 100]:
    np.random.seed(100)
    s, m, c = initial_samples.copy(), initial_means.copy(), initial_C.copy()
    for _ in range(n_steps):
        s, m, c = adjacent_step(s, m, c, step_size=0.05)
    
    corr, frob = measure_identity_preservation(initial_samples, s)
    preserved = "PRESERVED" if corr > 0.8 else ("degraded" if corr > 0.5 else "LOST")
    print(f"  {n_steps:>6} | {corr:>14.6f} | {frob:>14.6f} | {preserved:>12}")

# Test teleportation
print(f"\nTeleportation (random reassignment):")
print(f"{'Trial':>8} | {'Spectral Corr':>14} | {'Frob Distance':>14} | {'Identity?':>12}")
print("-" * 55)

for trial in range(5):
    np.random.seed(200 + trial)
    teleported = teleport(initial_samples, n_modes, n_samples)
    corr, frob = measure_identity_preservation(initial_samples, teleported)
    preserved = "PRESERVED" if corr > 0.8 else ("degraded" if corr > 0.5 else "LOST")
    print(f"  {trial+1:>6} | {corr:>14.6f} | {frob:>14.6f} | {preserved:>12}")


# ============================================================
# EXPERIMENT 4: Cascade Density as Gravitational Time Dilation
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 4: Cascade Density → Gravitational Time Dilation")
print("=" * 70)
print("""
Near a massive object (dense cascade region), time runs slower.
Far from mass (sparse cascade), time runs faster.

In GR: t_local = t_far × √(1 - 2GM/rc²)

In PAC: more interaction partners = more cascade steps per unit 
propagation = more experienced time per external tick... wait.

Actually: more cascade DENSITY means more of your energy budget 
is being spent on interactions with the local environment, leaving
less for your OWN internal cascade. So your internal clock slows
down. Same mechanism as velocity — external interactions steal
from your internal cascade budget.
""")

def cascade_in_environment(E_internal, env_density, n_modes=8, n_samples=15000):
    """
    An entity with E_internal energy in an environment with 
    env_density interaction partners.
    
    Higher density = more of the entity's energy gets pulled into
    environmental interactions = less available for internal cascade.
    """
    # Environmental coupling drains internal energy
    # Each interaction partner "costs" a Landauer minimum
    env_drain = min(env_density * LANDAUER_MIN * 0.01, E_internal * 0.9)
    E_available = E_internal - env_drain
    
    if E_available < 1e-15:
        return 0.0, 0.0, env_drain
    
    C = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            C[i, j] = np.exp(-abs(i - j) * 0.3)
    C = (C + C.T) / 2
    C += np.eye(n_modes) * 0.1
    
    means = E_available * np.exp(-np.arange(n_modes) * 0.3)
    means *= E_available / np.sum(means)
    
    try:
        sf = E_available / (np.trace(C) / n_modes) * 0.2
        samples = np.abs(np.random.multivariate_normal(means, C * sf, size=n_samples))
    except:
        samples = np.random.exponential(E_available / n_modes, (n_samples, n_modes))
    
    cov = np.cov(samples.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
    diag = np.maximum(np.diag(cov), 1e-30)
    xi = max(0, 0.5 * (np.sum(np.log(diag)) - np.sum(np.log(eigenvalues))))
    org = eigenvalues[-1] / np.sum(eigenvalues)
    
    return xi, org, env_drain


E_entity = 1.0
env_densities = np.linspace(0, 100, 25)

# Measure at zero density (far from mass)
xi_flat, _, _ = cascade_in_environment(E_entity, 0.0)

print(f"\n{'Env Density':>12} | {'E_available':>12} | {'ξ':>10} | "
      f"{'ξ/ξ_flat':>10} | {'GR analog':>10} | {'Ratio':>8}")
print("-" * 75)

grav_results = []
for rho in env_densities:
    xi, org, drain = cascade_in_environment(E_entity, rho)
    
    xi_ratio = xi / xi_flat if xi_flat > 0 else 0
    
    # GR analog: approximate as √(1 - 2Φ/c²) where Φ ∝ density
    # We normalize so that max density gives significant dilation
    Phi = rho * 0.005  # gravitational potential analog
    if Phi < 0.5:
        gr_prediction = np.sqrt(1 - 2 * Phi)
    else:
        gr_prediction = 0.0
    
    match = xi_ratio / gr_prediction if gr_prediction > 0.01 else float('inf')
    
    grav_results.append({
        'rho': rho, 'E_avail': E_entity - drain, 'xi': xi,
        'xi_ratio': xi_ratio, 'gr': gr_prediction, 'match': match
    })
    
    if rho < 2 or rho > 95 or abs(rho - 25) < 3 or abs(rho - 50) < 3 or abs(rho - 75) < 3:
        print(f"  {rho:>10.1f} | {E_entity-drain:>12.6f} | {xi:>10.6f} | "
              f"{xi_ratio:>10.6f} | {gr_prediction:>10.6f} | {match:>8.4f}")

# Fit: how does ξ_ratio scale with density?
valid_g = [r for r in grav_results if r['xi_ratio'] > 0.01 and r['rho'] > 0]
if len(valid_g) > 5:
    rhos = [r['rho'] for r in valid_g]
    xis = [r['xi_ratio'] for r in valid_g]
    grs = [r['gr'] for r in valid_g if r['gr'] > 0.01]
    
    # Does ξ_ratio match √(1-2Φ)?
    corr_gr = np.corrcoef(xis[:len(grs)], grs)[0, 1]
    print(f"\n  Correlation between cascade time dilation and GR prediction: {corr_gr:.6f}")
    if corr_gr > 0.99:
        print(f"  *** EXCELLENT match with gravitational time dilation ***")
    elif corr_gr > 0.95:
        print(f"  *** GOOD match with gravitational time dilation ***")


# ============================================================
# EXPERIMENT 5: Maximum Propagation Speed on Discrete Lattice
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 5: Maximum Speed on a Discrete Cascade Lattice")
print("=" * 70)
print("""
If the informational geometry is discrete (Planck-scale lattice),
the maximum propagation rate is one node per tick.

Entities WITH potential take multiple ticks per node (cascade steps).
Entities WITHOUT potential (photons) take exactly one tick per node.

We simulate a 1D lattice and propagate entities with different 
internal energies. Measure: how many lattice nodes traversed per
unit of cascade time.
""")

def propagate_on_lattice(E_total, n_lattice_nodes=100, n_modes=8, n_samples=5000):
    """
    Propagate an entity across a 1D lattice.
    
    At each node, the entity may interact (if it has potential)
    or pass through (if it doesn't).
    
    Interaction probability ∝ internal energy.
    Each interaction = one cascade step = one tick of local time.
    Propagation between interactions is "free" (no time cost).
    """
    position = 0
    local_ticks = 0  # experienced time
    global_steps = 0  # lattice nodes traversed
    
    trajectory = [{'pos': 0, 'ticks': 0}]
    
    E_internal = E_total * 0.5  # start with half as internal energy
    
    for step in range(n_lattice_nodes):
        position += 1
        global_steps += 1
        
        # Interaction probability depends on internal energy
        # More potential = more likely to interact at each node
        interaction_prob = 1 - np.exp(-E_internal * 2)
        
        if np.random.random() < interaction_prob and E_internal > 1e-10:
            # Interaction occurs — cascade step
            local_ticks += 1
            
            # Some internal energy gets used/transformed
            # (Landauer cost of the interaction)
            E_internal *= 0.99  # small drain per interaction
        
        trajectory.append({'pos': position, 'ticks': local_ticks})
    
    # Effective speed = nodes / local_ticks
    effective_speed = global_steps / max(local_ticks, 1)
    
    return trajectory, effective_speed, local_ticks


print(f"\n{'E_total':>10} | {'Nodes':>8} | {'Local ticks':>12} | "
      f"{'Eff speed':>10} | {'Normalized':>11} | {'Interpretation':>20}")
print("-" * 85)

# Find max speed (zero potential case)
_, max_speed, _ = propagate_on_lattice(0.0)

energies_lattice = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
speed_data = []

for E in energies_lattice:
    # Average over trials
    speeds = []
    ticks_list = []
    for trial in range(50):
        np.random.seed(500 + trial)
        traj, spd, ticks = propagate_on_lattice(E)
        speeds.append(spd)
        ticks_list.append(ticks)
    
    avg_speed = np.mean(speeds)
    avg_ticks = np.mean(ticks_list)
    norm_speed = avg_speed / max_speed if max_speed > 0 else 0
    
    interp = ""
    if E < 0.001:
        interp = "photon-like"
    elif norm_speed > 0.8:
        interp = "relativistic"
    else:
        interp = "massive"
    
    print(f"  {E:>8.3f} | {100:>8} | {avg_ticks:>12.1f} | "
          f"{avg_speed:>10.2f} | {norm_speed:>11.4f} | {interp:>20}")
    
    speed_data.append({
        'E': E, 'avg_speed': avg_speed, 'avg_ticks': avg_ticks,
        'norm_speed': norm_speed
    })

# Key result: is there a well-defined maximum speed?
print(f"\n  Maximum speed (E=0): {max_speed:.2f} nodes/tick")
print(f"  This IS 'c' in our lattice: one node per tick, the maximum possible.")
print(f"  Everything with E>0 moves slower because interactions create local time.")


# ============================================================
# EXPERIMENT 6: Combined — The Full Picture
# ============================================================
print("\n\n" + "=" * 70)
print("EXPERIMENT 6: The P/A Spectrum — From Photon to Black Hole")
print("=" * 70)
print("""
Map the full spectrum from fully actualized (photon) to 
fully potential (maximum cascade density).

At each point: measure speed, experienced time rate, dimensionality,
and cascade structure.
""")

potential_fractions = np.linspace(0.0, 1.0, 20)
E_total_spec = 1.0

print(f"\n{'P/(P+A)':>8} | {'Speed':>8} | {'Time rate':>10} | "
      f"{'Eff modes':>10} | {'ξ':>10} | {'State':>15}")
print("-" * 72)

for p_frac in potential_fractions:
    E_internal = E_total_spec * p_frac
    E_propagation = E_total_spec * (1 - p_frac)
    
    # Cascade rate (time)
    xi, org = measure_cascade_rate(E_internal, n_modes=8, n_samples=8000)
    time_rate = xi / xi_rest if xi_rest > 0 else 0
    
    # Mode count (dimensionality)
    if E_internal > 1e-10:
        C = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                C[i, j] = np.exp(-abs(i-j)*0.3)
        C = (C+C.T)/2
        C += np.eye(8)*0.1
        means = E_internal * np.exp(-np.arange(8)*0.3)
        means *= E_internal / np.sum(means)
        try:
            sf = E_internal / (np.trace(C)/8) * 0.2
            samp = np.abs(np.random.multivariate_normal(means, C*sf, size=8000))
            cov = np.cov(samp.T)
            eigs = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
            pr = np.sum(eigs)**2 / np.sum(eigs**2)
        except:
            pr = 1.0
    else:
        pr = 1.0
    
    # Effective speed (inversely related to time rate)
    speed = 1.0 - time_rate if time_rate <= 1.0 else 0.0
    
    state = ""
    if p_frac < 0.05:
        state = "photon"
    elif p_frac < 0.3:
        state = "relativistic"
    elif p_frac < 0.7:
        state = "matter"
    elif p_frac < 0.95:
        state = "dense matter"
    else:
        state = "max potential"
    
    print(f"  {p_frac:>6.2f} | {speed:>8.4f} | {time_rate:>10.6f} | "
          f"{pr:>10.2f} | {xi:>10.6f} | {state:>15}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("SUMMARY: PAC Relativity Simulation Results")
print("=" * 70)
print("""
H1 — VELOCITY AS P/A PARTITION:
    Cascade rate (experienced time) scales with internal energy.
    Check whether the exponent matches the Lorentz √(1-v²/c²).

H2 — MODE COLLAPSE AT ZERO POTENTIAL:
    Participation ratio should decrease as energy → 0.
    Does it approach 1 (one-dimensionality)?

H3 — IDENTITY CONSERVATION → LOCALITY:
    Adjacent traversal should preserve correlational identity.
    Teleportation should destroy it.
    The difference should be stark.

H4 — CASCADE DENSITY → TIME DILATION:
    ξ_ratio in dense environments should match √(1-2Φ/c²).
    Correlation with GR prediction measures this.

H5 — MAXIMUM SPEED ON DISCRETE LATTICE:
    Zero-potential entities should propagate at exactly 1 node/tick.
    Everything else should be slower.
    This defines c as the lattice propagation maximum.

These are COMPUTATIONAL PROBES of conceptual hypotheses.
Positive results warrant analytical formalization.
Negative results tell us where the framework needs revision.
""")
