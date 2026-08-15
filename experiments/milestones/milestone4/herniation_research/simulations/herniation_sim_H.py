"""
Herniation Simulation — Experiment H
=====================================
Connecting the bound state spectrum to the photon.

Hypothesis: If particles are quantum locks at the order-chaos boundary,
the photon is the SHALLOWEST lock — barely bound, almost fully resolved,
one interaction from actualization.

This means:
- Deep bound states = quarks, heavy particles (lots of unresolved potential)
- Middle bound states = electrons, lighter particles
- Shallowest bound states = photon-like (minimal unresolved potential)

Predictions for photon-like (shallowest) states:
1. Nearly zero binding energy (mass → 0)
2. Maximum spatial extent (delocalized)
3. Balance between order and chaos approaches 1.0 (perfect mediator)
4. The gap between the last bound state and the continuum should
   relate to the interaction rate (c-like constraint)

Peter McNally / Dawn Field Institute, 2026
"""

import numpy as np
from scipy.linalg import eigh

def build_cascade_system(N=1000, coupling=5.0, n_levels=5, base_width=3):
    """Build the cascade potential and solve for all states."""
    dx = 0.15
    x = np.arange(N) * dx - N * dx / 2
    
    # Cascade potential
    V = np.zeros(N)
    for level in range(n_levels):
        depth = coupling / (level + 1)
        width = base_width * (level + 1)
        V -= depth * np.exp(-x**2 / (2 * width**2))
    
    # Kinetic
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = -2.0
        if i + 1 < N: T[i, i+1] = 1.0
        if i - 1 >= 0: T[i, i-1] = 1.0
    T *= -0.5 / dx**2
    
    H = T + np.diag(V)
    eigenvalues, eigenvectors = eigh(H)
    
    bound_mask = eigenvalues < 0
    bound_E = eigenvalues[bound_mask]
    bound_psi = eigenvectors[:, bound_mask]
    
    # Also get first few continuum states
    continuum_E = eigenvalues[~bound_mask][:10]
    continuum_psi = eigenvectors[:, ~bound_mask][:, :10]
    
    return {
        'x': x, 'dx': dx, 'V': V,
        'bound_E': bound_E, 'bound_psi': bound_psi,
        'continuum_E': continuum_E, 'continuum_psi': continuum_psi,
        'all_E': eigenvalues, 'all_psi': eigenvectors,
        'n_bound': len(bound_E)
    }


def analyze_state_properties(system):
    """
    Analyze each bound state's properties along the 
    deep → shallow spectrum.
    """
    x = system['x']
    dx = system['dx']
    bound_E = system['bound_E']
    bound_psi = system['bound_psi']
    n = system['n_bound']
    
    results = []
    
    for i in range(n):
        psi = bound_psi[:, i]
        prob = psi**2
        
        # Spatial extent (standard deviation of position)
        mean_x = np.sum(x * prob) * dx
        var_x = np.sum((x - mean_x)**2 * prob) * dx
        extent = np.sqrt(var_x)
        
        # Localization (inverse participation ratio)
        ipr = np.sum(prob**2) * dx
        delocalization = 1.0 / (ipr * len(x) * dx)  # normalized
        
        # Order-chaos balance: left half vs right half
        left = np.sum(prob[x < 0]) * dx
        right = np.sum(prob[x > 0]) * dx
        balance = min(left, right) / max(left, right) if max(left, right) > 0 else 0
        
        # "Mass" proxy: binding energy (deeper = more massive)
        mass_proxy = abs(bound_E[i])
        
        # Momentum spread (from wavefunction curvature)
        # Related to kinetic energy
        dpsi = np.gradient(psi, dx)
        kinetic = 0.5 * np.sum(dpsi**2) * dx
        
        # Potential energy contribution
        potential = np.sum(system['V'] * prob) * dx
        
        # Ratio of kinetic to potential (photons should be mostly kinetic)
        ke_ratio = kinetic / abs(potential) if abs(potential) > 0 else float('inf')
        
        results.append({
            'level': i,
            'energy': bound_E[i],
            'mass_proxy': mass_proxy,
            'spatial_extent': extent,
            'delocalization': delocalization,
            'balance': balance,
            'kinetic': kinetic,
            'potential': potential,
            'ke_ratio': ke_ratio
        })
    
    return results


def analyze_cascade_as_particle_spectrum(system, state_props):
    """
    Map the bound state spectrum onto a particle-like hierarchy.
    
    The Landauer cascade tells us:
    - Each level of the cascade resolves some potential
    - The remainder cascades forward
    - Deeper states = more unresolved potential = more "mass"
    - Shallower states = less unresolved potential = less "mass"
    - The shallowest = photon-like (barely any unresolved potential)
    """
    n = len(state_props)
    
    # Divide into generations based on energy
    energies = np.array([s['energy'] for s in state_props])
    extents = np.array([s['spatial_extent'] for s in state_props])
    ke_ratios = np.array([s['ke_ratio'] for s in state_props])
    balances = np.array([s['balance'] for s in state_props])
    masses = np.array([s['mass_proxy'] for s in state_props])
    
    # The spectrum should show distinct regimes
    # Heavy (deep), medium, light, photon-like (shallow)
    
    # Normalize mass to ground state
    mass_ratios = masses / masses[0]
    
    # Look at how properties change along the spectrum
    print("=" * 80)
    print("BOUND STATE SPECTRUM: DEEP (HEAVY) → SHALLOW (PHOTON-LIKE)")
    print("=" * 80)
    print(f"\n{'Level':>5} {'Energy':>10} {'Mass%':>8} {'Extent':>8} {'Balance':>8} {'KE/PE':>8}")
    print("-" * 55)
    
    for s in state_props:
        i = s['level']
        mr = mass_ratios[i]
        print(f"{i:5d} {s['energy']:10.4f} {mr*100:7.1f}% {s['spatial_extent']:8.2f} "
              f"{s['balance']:8.3f} {s['ke_ratio']:8.3f}")
    
    # The transition from particle-like to photon-like
    print("\n" + "=" * 80)
    print("TRANSITION ANALYSIS")
    print("=" * 80)
    
    # Where does KE/PE ratio cross 1.0? (photon = all kinetic)
    crossover = None
    for i in range(len(ke_ratios) - 1):
        if ke_ratios[i] < 1.0 and ke_ratios[i+1] >= 1.0:
            crossover = i
            break
    
    if crossover is not None:
        print(f"\n  KE/PE crossover at level {crossover}")
        print(f"  Below: potential-dominated (massive particle-like)")
        print(f"  Above: kinetic-dominated (photon-like)")
        print(f"  States below crossover: {crossover + 1} ('massive' particles)")
        print(f"  States above crossover: {n - crossover - 1} ('massless' or near-massless)")
    
    # Spatial extent scaling
    print(f"\n  Spatial extent scaling:")
    print(f"  Ground state (heaviest): {extents[0]:.2f}")
    print(f"  Last bound state (lightest): {extents[-1]:.2f}")
    print(f"  Ratio: {extents[-1]/extents[0]:.2f}x")
    
    # The gap to the continuum — this is the "unbound" threshold
    last_bound = energies[-1]
    first_continuum = system['continuum_E'][0] if len(system['continuum_E']) > 0 else 0
    gap_to_continuum = first_continuum - last_bound
    
    print(f"\n  Last bound state energy: {last_bound:.6f}")
    print(f"  First continuum state: {first_continuum:.6f}")
    print(f"  Gap to continuum: {gap_to_continuum:.6f}")
    print(f"  (This gap = minimum energy to 'unbind' = Landauer cost of dissolution)")
    
    # Mass hierarchy
    print(f"\n  MASS HIERARCHY (binding energy ratios):")
    if n >= 6:
        print(f"  Ground / 1st excited: {masses[0]/masses[1]:.4f}")
        print(f"  1st / 2nd excited: {masses[1]/masses[2]:.4f}")
        print(f"  2nd / 3rd excited: {masses[2]/masses[3]:.4f}")
        
        # Compare to known particle mass ratios
        print(f"\n  For reference — known particle mass ratios:")
        print(f"  proton/electron = 1836.15")
        print(f"  tau/muon = 16.82")
        print(f"  muon/electron = 206.77")
        print(f"  bottom/charm quark ≈ 3.0")
        print(f"  charm/strange quark ≈ 13.5")
        print(f"  W boson/proton ≈ 85.7")
    
    return {
        'mass_ratios': mass_ratios.tolist(),
        'crossover_level': crossover,
        'extent_ratio': float(extents[-1] / extents[0]),
        'gap_to_continuum': float(gap_to_continuum)
    }


def test_photon_properties(system, state_props):
    """
    Test whether the shallowest bound states have photon-like properties:
    
    1. Zero effective mass (binding energy → 0)
    2. Maximum delocalization (extended across entire boundary)
    3. Perfect order-chaos balance (photon mediates between both)
    4. Kinetic >> Potential (photon carries momentum, not structure)
    5. The shallowest state should be an almost-free wave
    """
    print("\n" + "=" * 80)
    print("PHOTON-LIKENESS TEST")
    print("=" * 80)
    
    n = len(state_props)
    
    # Last 5 states (most photon-like)
    print(f"\n  5 SHALLOWEST BOUND STATES (most photon-like):")
    print(f"  {'Level':>5} {'E':>10} {'|E|':>10} {'Extent':>8} {'Balance':>8} {'KE/PE':>8}")
    
    for s in state_props[-5:]:
        print(f"  {s['level']:5d} {s['energy']:10.6f} {s['mass_proxy']:10.6f} "
              f"{s['spatial_extent']:8.2f} {s['balance']:8.3f} {s['ke_ratio']:8.3f}")
    
    # 5 DEEPEST states (most particle-like)
    print(f"\n  5 DEEPEST BOUND STATES (most particle-like):")
    print(f"  {'Level':>5} {'E':>10} {'|E|':>10} {'Extent':>8} {'Balance':>8} {'KE/PE':>8}")
    
    for s in state_props[:5]:
        print(f"  {s['level']:5d} {s['energy']:10.6f} {s['mass_proxy']:10.6f} "
              f"{s['spatial_extent']:8.2f} {s['balance']:8.3f} {s['ke_ratio']:8.3f}")
    
    # Trends
    shallow = state_props[-1]
    deep = state_props[0]
    
    print(f"\n  PHOTON-LIKENESS INDICATORS:")
    print(f"  Mass ratio (shallow/deep): {shallow['mass_proxy']/deep['mass_proxy']:.6f}")
    print(f"  Extent ratio (shallow/deep): {shallow['spatial_extent']/deep['spatial_extent']:.2f}")
    print(f"  KE/PE deep: {deep['ke_ratio']:.3f} (potential-dominated = massive)")
    print(f"  KE/PE shallow: {shallow['ke_ratio']:.3f} (kinetic-dominated = massless)")
    
    # Does the shallowest state look like a free wave?
    # A free particle in a box has ψ ∝ sin(nπx/L), KE/PE → ∞
    print(f"\n  FREE WAVE TEST:")
    print(f"  Shallowest KE/PE = {shallow['ke_ratio']:.3f}")
    print(f"  Deepest KE/PE = {deep['ke_ratio']:.3f}")
    if shallow['ke_ratio'] > 2 * deep['ke_ratio']:
        print(f"  ✓ Shallowest is significantly more kinetic (more wave-like)")
    
    # The photon prediction: as binding energy → 0,
    # the state should become maximally delocalized and balanced
    print(f"\n  DELOCALIZATION TREND:")
    extents = [s['spatial_extent'] for s in state_props]
    # Is it monotonically increasing?
    monotonic = all(extents[i] <= extents[i+1] for i in range(len(extents)-1))
    print(f"  Monotonically increasing: {monotonic}")
    if not monotonic:
        # Find where it breaks
        breaks = [i for i in range(len(extents)-1) if extents[i] > extents[i+1]]
        print(f"  Breaks at levels: {breaks[:5]}")
    
    # The continuum = fully unbound = the photon has LEFT the boundary
    # When a bound state falls off the spectrum (E → 0), it becomes
    # a free wave. That's emission. The photon IS the state that
    # just barely didn't make it as a bound state.
    print(f"\n  EMISSION INTERPRETATION:")
    print(f"  The last bound state has E = {shallow['energy']:.6f}")
    print(f"  If perturbed slightly, it would become unbound → free wave")
    print(f"  This is photon emission: a quantum lock dissolving into a free wave")
    print(f"  The energy released = |E| = {shallow['mass_proxy']:.6f}")
    print(f"  This IS the photon's energy (frequency × h)")


def test_landauer_connection(system, state_props):
    """
    Test whether the energy gaps map onto Landauer cascade levels.
    
    In the cascade: each level's Landauer cost (kT ln 2) funds the next.
    In the spectrum: each energy gap should relate to kT ln 2 somehow.
    """
    print("\n" + "=" * 80)
    print("LANDAUER CASCADE CONNECTION")
    print("=" * 80)
    
    energies = np.array([s['energy'] for s in state_props])
    gaps = np.diff(energies)
    
    # The Landauer cost is kT ln(2)
    # In our units, if we set kT = 1 (natural units), Landauer = ln(2) = 0.6931
    ln2 = np.log(2)
    
    # Check if gaps are multiples or fractions of ln(2)
    gap_over_ln2 = gaps / ln2
    
    print(f"\n  Energy gaps in units of ln(2):")
    for i, (g, r) in enumerate(zip(gaps[:15], gap_over_ln2[:15])):
        nearest_frac = round(r * 4) / 4  # nearest quarter
        print(f"  Gap {i:2d}: {g:.6f} = {r:.4f} × ln(2) ≈ {nearest_frac:.2f} × ln(2)")
    
    # Sum of all gaps = total cascade energy
    total_gap = energies[-1] - energies[0]
    total_ln2 = total_gap / ln2
    print(f"\n  Total energy span: {total_gap:.4f} = {total_ln2:.2f} × ln(2)")
    print(f"  Number of Landauer units: {abs(total_ln2):.1f}")
    print(f"  Number of bound states: {len(energies)}")
    print(f"  States per Landauer unit: {len(energies)/abs(total_ln2):.2f}")
    
    # The cascade predicts: each level's remainder funds the next
    # So gap_n should relate to gap_{n+1} by the cascade ratio
    if len(gaps) > 2:
        cascade_ratios = gaps[:-1] / gaps[1:]
        print(f"\n  Cascade ratios (gap_n / gap_n+1):")
        for i, r in enumerate(cascade_ratios[:10]):
            print(f"  Level {i}: {r:.4f}")
        
        # The cascade amplification factor from Exp 6 was ~53x
        # That was over the full cascade. Per step it should be ~ξ_PAC
        mean_ratio = np.mean(cascade_ratios[:5])
        print(f"\n  Mean cascade ratio (first 5): {mean_ratio:.4f}")
        print(f"  ξ_PAC = 1.0571")
        print(f"  Δ = {abs(mean_ratio - 1.0571):.4f}")


def test_c_as_gap(system):
    """
    Test: is there a natural 'speed' implied by the spectrum?
    
    If c = the rate of one interaction, then the spacing between
    energy levels should encode something about the maximum update rate.
    
    In a quantum system, the time to transition between states is
    related to the energy gap by: Δt ~ ℏ/ΔE
    
    The smallest gap (between the most photon-like states) gives
    the FASTEST transition time — this might be the c-analog.
    """
    print("\n" + "=" * 80)
    print("SPEED OF LIGHT AS MINIMUM TRANSITION TIME")
    print("=" * 80)
    
    bound_E = system['bound_E']
    gaps = np.diff(bound_E)
    
    # In natural units (ℏ = 1), transition time ~ 1/ΔE
    # Smallest gap → fastest transition → maximum speed
    
    min_gap = gaps[-1]  # between two shallowest states
    max_gap = gaps[0]   # between two deepest states
    
    print(f"\n  Largest gap (deep states): {max_gap:.6f}")
    print(f"  Smallest gap (shallow states): {min_gap:.6f}")
    print(f"  Ratio: {max_gap/min_gap:.2f}")
    
    transition_fast = 1.0 / max_gap  # fastest (most energetic)
    transition_slow = 1.0 / min_gap  # slowest (least energetic)
    
    print(f"\n  Fastest transition time (ℏ/ΔE_max): {transition_fast:.4f}")
    print(f"  Slowest transition time (ℏ/ΔE_min): {transition_slow:.4f}")
    
    # The gap to the continuum
    gap_continuum = system['continuum_E'][0] - bound_E[-1]
    t_escape = 1.0 / gap_continuum if gap_continuum > 0 else float('inf')
    
    print(f"\n  Gap to continuum: {gap_continuum:.6f}")
    print(f"  Escape time (time to emit photon): {t_escape:.4f}")
    
    print(f"\n  INTERPRETATION:")
    print(f"  The minimum gap between consecutive states sets the")
    print(f"  fastest possible oscillation. But the gap to the continuum")
    print(f"  sets how fast a photon can be emitted — that's the maximum")
    print(f"  rate of information transfer between bound systems.")
    print(f"  This is the c-analog in the spectrum.")
    
    # All gaps as "speeds"
    speeds = 1.0 / gaps
    speed_ratios = speeds / speeds[0]
    print(f"\n  Speed spectrum (normalized to ground transition):")
    for i, (g, s, r) in enumerate(zip(gaps[:10], speeds[:10], speed_ratios[:10])):
        print(f"  Level {i}: gap={g:.6f}, speed={s:.4f}, ratio={r:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    
    print("=" * 80)
    print("EXPERIMENT H: THE PHOTON AS SHALLOWEST QUANTUM LOCK")
    print("=" * 80)
    
    # Build system with cascade potential
    print("\nBuilding cascade system (g=5.0, 5 levels)...")
    system = build_cascade_system(coupling=5.0, n_levels=5, base_width=3)
    print(f"Found {system['n_bound']} bound states")
    
    # Analyze all state properties
    print("\nAnalyzing state properties...")
    state_props = analyze_state_properties(system)
    
    # Map onto particle spectrum
    spectrum = analyze_cascade_as_particle_spectrum(system, state_props)
    
    # Test photon-likeness
    test_photon_properties(system, state_props)
    
    # Landauer connection
    test_landauer_connection(system, state_props)
    
    # Speed of light analog
    test_c_as_gap(system)
    
    # ================================================================
    # EXPERIMENT H2: Double-well version (order vs chaos explicitly)
    # ================================================================
    print("\n\n" + "=" * 80)
    print("EXPERIMENT H2: DOUBLE-WELL — PHOTON AS SPANNING LOCK")
    print("=" * 80)
    
    # Build double-well with cascade structure in each well
    N = 1000
    dx = 0.15
    x = np.arange(N) * dx - N * dx / 2
    
    # Order well (left): structured, cascade-shaped
    V_order = np.zeros(N)
    for level in range(5):
        depth = 5.0 / (level + 1)
        width = 3 * (level + 1)
        V_order -= depth * np.exp(-((x + 15)**2) / (2 * width**2))
    
    # Chaos well (right): smoother, wider
    V_chaos = -4.0 * np.exp(-((x - 15)**2) / (2 * 12**2))
    
    V = V_order + V_chaos
    
    # Solve
    T_mat = np.zeros((N, N))
    for i in range(N):
        T_mat[i, i] = -2.0
        if i + 1 < N: T_mat[i, i+1] = 1.0
        if i - 1 >= 0: T_mat[i, i-1] = 1.0
    T_mat *= -0.5 / dx**2
    
    H = T_mat + np.diag(V)
    eigenvalues, eigenvectors = eigh(H)
    
    bound_mask = eigenvalues < 0
    bound_E = eigenvalues[bound_mask]
    bound_psi = eigenvectors[:, bound_mask]
    
    print(f"\n  Total bound states: {len(bound_E)}")
    
    # Classify each state
    print(f"\n  {'Level':>5} {'Energy':>10} {'Left%':>8} {'Right%':>8} {'Type':>12} {'Extent':>8}")
    print("  " + "-" * 60)
    
    n_spanning = 0
    n_order = 0
    n_chaos = 0
    spanning_energies = []
    
    for i in range(min(len(bound_E), 50)):
        psi = bound_psi[:, i]
        prob = psi**2
        
        left = np.sum(prob[x < 0]) * dx
        right = np.sum(prob[x > 0]) * dx
        total = left + right
        left_pct = left / total * 100
        right_pct = right / total * 100
        
        balance = min(left, right) / max(left, right)
        
        mean_x = np.sum(x * prob) * dx
        var_x = np.sum((x - mean_x)**2 * prob) * dx
        extent = np.sqrt(var_x)
        
        if balance > 0.3:
            state_type = "SPANNING"
            n_spanning += 1
            spanning_energies.append(bound_E[i])
        elif left > right:
            state_type = "Order"
            n_order += 1
        else:
            state_type = "Chaos"
            n_chaos += 1
        
        if i < 10 or i >= len(bound_E) - 5 or state_type == "SPANNING":
            print(f"  {i:5d} {bound_E[i]:10.4f} {left_pct:7.1f}% {right_pct:7.1f}% "
                  f"{state_type:>12} {extent:8.2f}")
    
    print(f"\n  Summary: {n_spanning} SPANNING, {n_order} Order, {n_chaos} Chaos")
    
    if len(spanning_energies) > 1:
        span_E = np.array(spanning_energies)
        span_gaps = np.diff(span_E)
        
        print(f"\n  SPANNING STATE (quantum lock) spectrum:")
        print(f"  Energies: {[f'{e:.4f}' for e in span_E[:8]]}")
        print(f"  Gaps: {[f'{g:.4f}' for g in span_gaps[:7]]}")
        
        if len(span_gaps) > 1:
            span_ratios = span_gaps[:-1] / span_gaps[1:]
            print(f"  Gap ratios: {[f'{r:.4f}' for r in span_ratios[:6]]}")
        
        # The shallowest spanning state is the most photon-like
        print(f"\n  Shallowest quantum lock: E = {span_E[-1]:.6f}")
        print(f"  Deepest quantum lock: E = {span_E[0]:.6f}")
        print(f"  Lock mass ratio (deep/shallow): {abs(span_E[0]/span_E[-1]):.2f}")
    
    print("\n" + "=" * 80)
    print("SYNTHESIS")  
    print("=" * 80)
    print("""
    THE PARTICLE-PHOTON SPECTRUM:
    
    Deep bound states (high |E|):
    - Strongly bound quantum locks
    - Small spatial extent (localized)
    - High "mass" (lots of unresolved potential)
    - Potential-dominated (KE/PE < 1)
    → These are the heavy particles (quarks, W/Z bosons)
    
    Middle bound states:
    - Moderately bound
    - Medium extent
    - Medium mass
    → These are lighter particles (electrons, muons)
    
    Shallow bound states (low |E|):
    - Barely bound quantum locks
    - Large spatial extent (delocalized)
    - Very low "mass" (almost no unresolved potential)
    - Kinetic-dominated (KE/PE > 1)
    → These are photon-like (neutrinos? nearly massless particles)
    
    Just beyond the bound spectrum (E = 0+):
    - The lock dissolves entirely
    - The state becomes a free wave
    - Zero mass, maximum delocalization
    - Pure kinetic, no potential binding
    → THIS IS THE PHOTON
    
    The photon isn't a special object. It's what happens when a 
    quantum lock at the order-chaos boundary becomes too shallow
    to hold. It's the last stage of the Landauer cascade — the
    bound state that has resolved almost all its potential and
    is one interaction away from fully actualizing.
    
    When a deep state transitions to a shallower one, the energy
    difference is emitted as a photon — which IS a dissolved lock,
    a free wave carrying the energy difference. The photon doesn't
    have mass because it has no binding energy left. It doesn't
    experience time because it's not bound to the boundary anymore.
    It travels at c because c is the natural propagation rate of
    unbound waves in the field.
    
    c is not special. It's just what unbound states do.
    """)
